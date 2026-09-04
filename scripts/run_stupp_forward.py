#!/usr/bin/env python
"""Forward-solve one treated tumor growth with fisher_kpp_jax.StuppFKPPSolver.

Reads a JSON run manifest (see scripts/stupp_manifest_example.json and
read_manifest / params_from_manifest in fisher_kpp_jax.solvers: the
StuppFKPPSolver parameters by name, with the tissue probability maps, the
labelled segmentation of the resection cavity and the TOTAL-dose map as
NIfTI paths and the time step as steps_per_day, dt or n_steps), runs the
solver and writes, into its own run directory

  <output-dir>/<run-name>/
    final_cell_density.nii.gz  the final tumor cell density, affine and header
                               of the wm pbmap (float32 voxels)
    run_config.json            scalar params, CLI args, manifest path, final
                               time / stopping criterion / stopping quantity,
                               wall time
    manifest.json              verbatim copy of the manifest (provenance)
    overview.png (+ .pdf)      unless --no-plot: the 3x3 montage of
                               scripts/run_stupp_example.py (seed, before and
                               after resection; three frames inside the
                               radiotherapy block; three from its end to the
                               end of the run) on the axial slice
                               through the dose map's center of mass, over
                               --background-image or the tissue maps, plus total mass vs time
                               with the treatment events marked

The run directory is created with exist_ok=False and nothing is written
outside it, so many evaluations can run in parallel against the same
output-dir (the default run name carries a UTC timestamp and the pid).
The eight montage frames are the only snapshots recorded, and only for the
plot; with --no-plot nothing is recorded (memory-light for parallel sweeps).

The manifest carries every parameter; explicit CLI arguments (--wm, --gm
and the solver knobs) override the manifest entries of the same name.
--n-steps or --steps-per-day replaces the manifest's time step entry
altogether. The horizon is resection_time + time_after_resection (the run
ends that many days after the resection).

Run from the project root, e.g.:
  JAX_PLATFORMS=cpu python scripts/run_stupp_forward.py --output-dir runs/
  python scripts/run_stupp_forward.py --output-dir runs/ --rho 0.08 --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver  # noqa: E402
from fisher_kpp_jax.solvers import params_from_manifest, read_manifest  # noqa: E402
from fisher_kpp_jax.util import (  # noqa: E402
    jsonable,
    montage_days,
    render,
    scalar_params,
    select_panels,
)

DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")
DEFAULT_THRESHOLD = 0.01  # overlay transparency threshold (cell density)

# CLI option -> manifest entry (solver parameter name) it overrides.
CLI_OVERRIDES: dict[str, str] = {
    "wm": "white_matter_pbmap",
    "gm": "gray_matter_pbmap",
    "rho": "rho",
    "white_matter_diffusivity": "white_matter_diffusivity",
    "resolution_factor": "resolution_factor",
    "time_after_resection": "time_after_resection",
    "n_steps": "n_steps",
    "steps_per_day": "steps_per_day",
    "precision": "precision",
    "seed_x": "gaussian_seed_x_fraction",
    "seed_y": "gaussian_seed_y_fraction",
    "seed_z": "gaussian_seed_z_fraction",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wm", default=None, help="white matter pbmap NIfTI (overrides the manifest)"
    )
    parser.add_argument(
        "--gm", default=None, help="gray matter pbmap NIfTI (overrides the manifest)"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="run manifest JSON")
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name",
        default=None,
        help="run directory name (default: stupp_<UTC timestamp>_<pid>)",
    )
    # Solver knobs default to None: unset ones leave the manifest entry.
    knobs = parser.add_argument_group(
        "solver knobs", "override the manifest entries of the same name"
    )
    knobs.add_argument("--rho", type=float, default=None, help="proliferation rate, 1/day")
    knobs.add_argument(
        "--white-matter-diffusivity", type=float, default=None, help="mm^2/day"
    )
    knobs.add_argument("--resolution-factor", type=float, default=None)
    knobs.add_argument(
        "--time-after-resection",
        type=float,
        default=None,
        help="horizon: the run ends this many days after the resection",
    )
    knobs.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="explicit step count (overrides the manifest's time step)",
    )
    knobs.add_argument(
        "--steps-per-day",
        type=float,
        default=None,
        help="time step as steps per day (overrides the manifest's time step)",
    )
    knobs.add_argument("--precision", choices=("f32", "f64"), default=None)
    knobs.add_argument("--seed-x", type=float, default=None, help="seed x fraction")
    knobs.add_argument("--seed-y", type=float, default=None, help="seed y fraction")
    knobs.add_argument("--seed-z", type=float, default=None, help="seed z fraction")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="cell density below which the overlay is transparent",
    )
    parser.add_argument(
        "--background-image",
        default=None,
        help="background image NIfTI for overview.png (default: wm + gm pbmaps)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip overview.png and request no time series",
    )
    return parser.parse_args(argv)


def default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"stupp_{stamp}_{os.getpid()}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or default_run_name()
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = read_manifest(args.manifest)
    shutil.copyfile(args.manifest, run_dir / "manifest.json")
    overrides = {
        entry: getattr(args, option)
        for option, entry in CLI_OVERRIDES.items()
        if getattr(args, option) is not None
    }
    # A CLI time step replaces the manifest's, whichever form that has.
    if "n_steps" in overrides or "steps_per_day" in overrides:
        for key in ("n_steps", "dt", "steps_per_day"):
            manifest.pop(key, None)
    entries = {**manifest, **overrides}
    for key in ("white_matter_pbmap", "gray_matter_pbmap"):
        if key not in entries:
            raise ValueError(f"{key} is set neither in the manifest nor on the command line.")
    wm_path, gm_path = entries["white_matter_pbmap"], entries["gray_matter_pbmap"]
    wm_img = nib.load(wm_path)  # affine and header of the output volume
    voxel_size = tuple(float(v) for v in wm_img.header.get_zooms()[:3])

    params = params_from_manifest({"voxel_size_mm": voxel_size, "verbose": True, **entries})
    wm, gm = params["white_matter_pbmap"], params["gray_matter_pbmap"]
    n_steps, dt = StuppFKPPSolver(params)._resolve_time_stepping()
    if not args.no_plot:
        # The montage frames are the only snapshots.
        resection_time = float(params["resection_time"])
        panel_days = montage_days(
            resection_time,
            dt,
            params["rt_times"],
            resection_time + float(params["time_after_resection"]),
        )
        params["snapshot_times"] = panel_days
    solver = StuppFKPPSolver(params)

    print(f"run directory: {run_dir}")
    print(f"tissue: wm={wm_path} gm={gm_path}")
    print(f"grid: {wm.shape}, voxel size {voxel_size} mm")
    wall_start = time.perf_counter()
    result = solver.solve()
    wall = time.perf_counter() - wall_start

    config: dict[str, Any] = {
        "run_name": run_name,
        "cli_args": {key: jsonable(value) for key, value in vars(args).items()},
        "manifest": str(Path(args.manifest).resolve()),
        "tissue": {"wm": wm_path, "gm": gm_path},
        "params": scalar_params(solver.params),
        "grid_shape": list(wm.shape),
        "success": result.success,
        "error": result.error,
        "final_time": result.final_time,
        "stopping_criterion": result.stopping_criterion,
        "final_stopping_quantity": result.final_stopping_quantity,
        "wall_time_s": wall,
    }
    if result.success and not np.isfinite(result.final_state["cell_density"]).all():
        # The isotropic model has no device guard: an explicit-Euler blow-up
        # surfaces as NaN/inf in the final state. Report it as a failure.
        result.success = False
        result.error = "non-finite final cell density (time step too large?)"
        config["success"] = False
        config["error"] = result.error
    if not result.success:
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"FAILED after {wall:.1f} s: {result.error}", file=sys.stderr)
        return 1

    config["n_steps"] = n_steps
    config["dt"] = dt
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    cell_density = np.asarray(result.final_state["cell_density"], dtype=np.float32)
    image = nib.Nifti1Image(cell_density, wm_img.affine, header=wm_img.header)
    image.set_data_dtype(np.float32)
    nib.save(image, str(run_dir / "final_cell_density.nii.gz"))

    if not args.no_plot:
        frames = result.time_series["cell_density"]
        times = result.snapshot_times
        masses = frames.sum(axis=(1, 2, 3))
        seed_voxel = tuple(
            int(solver.params[f"gaussian_seed_{axis}_fraction"] * n)
            for axis, n in zip("xyz", wm.shape)
        )
        if args.background_image is not None:
            background = np.asarray(nib.load(args.background_image).get_fdata(), dtype=np.float64)
            if background.shape != wm.shape:
                raise ValueError(
                    f"--background-image shape {background.shape} differs from the tissue maps {wm.shape}."
                )
        else:
            background = wm + gm
        z = int(round(center_of_mass(params["rt_dose"])[2]))
        panels = select_panels(result.initial_state["cell_density"], frames, times, panel_days)
        header = (
            f"{run_name}: {Path(args.manifest).name}, axial slice z={z}, "
            f"densities >= {args.threshold:g}\n"
            f"rho {solver.params['rho']:g}, D {solver.params['white_matter_diffusivity']:g}, "
            f"ratio {solver.params['diffusivity_ratio']:g}, resolution "
            f"{solver.params['resolution_factor']:g}, {n_steps} steps"
        )
        render(
            run_dir / "overview",
            header,
            panels,
            background,
            params["resection_cavity"],
            seed_voxel,
            z,
            args.threshold,
            times,
            masses,
            params,
        )

    print(
        f"done in {wall:.1f} s: final_time={result.final_time:g}, "
        f"criterion={result.stopping_criterion}, "
        f"final mass={result.final_stopping_quantity:.6g}, "
        f"max density={float(cell_density.max()):.4g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
