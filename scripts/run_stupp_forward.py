#!/usr/bin/env python
"""Forward-solve one treated tumor growth with fisher_kpp_jax.StuppFKPPSolver.

Loads the patient's WM/GM tissue probability maps, reads a JSON run manifest
(see scripts/stupp_manifest_example.json and the loaders in
fisher_kpp_jax.solvers: 'tissue' pbmap paths, 'solver' scalar parameters
incl. dt, and the treatment sections -- resection cavity from a labelled
segmentation, chemotherapy session times/rates, radiotherapy fraction times
and TOTAL-dose map with the linear-quadratic alpha/beta), runs the solver and
writes, into its own run directory

  <output-dir>/<run-name>/
    final_cell_density.nii.gz  the final tumor cell density, affine and header
                               of the wm pbmap (float32 voxels)
    run_config.json            scalar params, CLI args, manifest path, final
                               time / stopping criterion / stopping quantity,
                               wall time
    manifest.json              verbatim copy of the manifest (provenance)
    overview.png (+ .pdf)      unless --no-plot: the montage of
                               scripts/run_stupp_example.py (seed, before and
                               after resection, snapshots across the
                               radiotherapy block, end) on the axial slice
                               through the dose map's center of mass, over
                               --t1c or the tissue maps, plus total mass vs time
                               with the treatment events marked

The run directory is created with exist_ok=False and nothing is written
outside it, so many evaluations can run in parallel against the same
output-dir (the default run name carries a UTC timestamp and the pid).
Snapshot volumes are only kept in memory for the plot; with --no-plot no
time series is requested at all (memory-light for parallel sweeps) and the
extra short solve up to the resection (the "before resection" panel) is
skipped.

Parameter precedence, lowest to highest: the script defaults below, the
manifest's 'tissue' / 'solver' sections, explicit CLI arguments. The horizon
and the time step are resolved after the merge, each from the
highest-precedence layer that sets any key of its group: horizon =
--stopping-time / --time-after-resection (stopping_time = resection time +
time after resection; default 100 days after the resection), time step =
--n-steps / --steps-per-day (default 12 steps per day), with the manifest's
solver keys of the same names in between. The script defaults: rho, white
matter diffusivity and diffusivity ratio are the means of
scripts/parameter_range.txt (previous inverse runs); seed position (image
center) and pbmap paths are PLACEHOLDERS for the SAILOR subject.

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

from fisher_kpp_jax import FKPPSolver, StuppFKPPSolver  # noqa: E402
from fisher_kpp_jax.solvers import (  # noqa: E402
    resolve_horizon,
    resolve_time_step,
    solver_params_from_manifest,
    tissue_paths_from_manifest,
    treatment_params_from_manifest,
)
from run_stupp_example import render, select_panels, snapshot_steps  # noqa: E402

_SAILOR_TISSUE = "/mnt/Drive4/lucas/SAILOR/processed/sub-01/ses-01/tissue_segmentation"
DEFAULT_WM = f"{_SAILOR_TISSUE}/wm_pbmap.nii.gz"
DEFAULT_GM = f"{_SAILOR_TISSUE}/gm_pbmap.nii.gz"
DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")

# Solver defaults -- see the module docstring. Keyed by the solver
# parameter names; the manifest 'solver' section and explicit CLI arguments
# override them in that order.
DEFAULT_SOLVER_PARAMS: dict[str, Any] = {
    "rho": 0.075777,  # 1/day, mean of scripts/parameter_range.txt
    "white_matter_diffusivity": 0.80021,  # mm^2/day, mean of parameter_range.txt
    "diffusivity_ratio": 227.35,  # 10^mean(log10 ratio) of parameter_range.txt
    "resolution_factor": 1.0,
    "time_after_resection": 100.0,  # days; stopping_time = resection time + this
    "steps_per_day": 12,  # dt = 1/12 day; n_steps = ceil(stopping_time * 12)
    "precision": "f32",
    "gaussian_seed_x_fraction": 0.5,  # of the grid extent, per axis
    "gaussian_seed_y_fraction": 0.5,
    "gaussian_seed_z_fraction": 0.5,
}
DEFAULT_N_SNAPSHOTS = 25
DEFAULT_N_TREATMENT_PANELS = 5
DEFAULT_THRESHOLD = 0.01  # overlay transparency threshold (cell density)

# CLI option -> solver parameter name of the pass-through knobs.
CLI_SOLVER_KNOBS: dict[str, str] = {
    "rho": "rho",
    "diffusivity": "white_matter_diffusivity",
    "resolution_factor": "resolution_factor",
    "stopping_time": "stopping_time",
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
        "--wm",
        default=None,
        help=f"white matter pbmap NIfTI (default: manifest 'tissue', else {DEFAULT_WM})",
    )
    parser.add_argument(
        "--gm",
        default=None,
        help=f"gray matter pbmap NIfTI (default: manifest 'tissue', else {DEFAULT_GM})",
    )
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST, help="treatment manifest JSON"
    )
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name",
        default=None,
        help="run directory name (default: stupp_<UTC timestamp>_<pid>)",
    )
    # Solver knobs default to None: unset ones fall back to the manifest
    # 'solver' section, then to DEFAULT_SOLVER_PARAMS.
    knobs = parser.add_argument_group(
        "solver knobs",
        "override the manifest 'solver' section; defaults: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_SOLVER_PARAMS.items()),
    )
    knobs.add_argument("--rho", type=float, default=None, help="proliferation rate, 1/day")
    knobs.add_argument(
        "--diffusivity", type=float, default=None, help="white matter diffusivity, mm^2/day"
    )
    knobs.add_argument("--resolution-factor", type=float, default=None)
    knobs.add_argument(
        "--stopping-time", type=float, default=None, help="total horizon in days from the seed"
    )
    knobs.add_argument(
        "--time-after-resection",
        type=float,
        default=None,
        help="horizon as days after the resection (overrides the manifest's horizon)",
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
        "--n-snapshots",
        type=int,
        default=DEFAULT_N_SNAPSHOTS,
        help="evenly spaced time-series snapshots recorded for overview.png",
    )
    parser.add_argument(
        "--n-treatment-panels",
        type=int,
        default=DEFAULT_N_TREATMENT_PANELS,
        help="overview panels across the radiotherapy block",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="cell density below which the overlay is transparent",
    )
    parser.add_argument(
        "--t1c",
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


def jsonable(value: Any) -> Any:
    """Convert numpy scalars/1-D arrays and paths to JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)  # np.inf stopping_threshold
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    """The params without the volumes (pbmaps, cavity, dose)."""
    return {
        key: jsonable(value)
        for key, value in params.items()
        if not (isinstance(value, np.ndarray) and value.ndim > 1)
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or default_run_name()
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    tissue_paths = tissue_paths_from_manifest(args.manifest)
    wm_path = args.wm or str(tissue_paths.get("wm", DEFAULT_WM))
    gm_path = args.gm or str(tissue_paths.get("gm", DEFAULT_GM))
    wm_img = nib.load(wm_path)
    gm_img = nib.load(gm_path)
    wm = np.asarray(wm_img.get_fdata(), dtype=np.float64)
    gm = np.asarray(gm_img.get_fdata(), dtype=np.float64)
    voxel_size = tuple(float(v) for v in wm_img.header.get_zooms()[:3])

    treatment = treatment_params_from_manifest(args.manifest)
    shutil.copyfile(args.manifest, run_dir / "manifest.json")

    # Precedence: script defaults < manifest 'solver' < explicit CLI knobs.
    # The time step is kept unresolved (n_steps / dt / steps_per_day) until
    # the merge is complete: the highest-precedence layer that sets any of
    # the three wins as a whole, and only then is it translated to n_steps
    # with the resolved stopping_time.
    manifest_solver = solver_params_from_manifest(args.manifest, resolve=False)
    cli_solver = {
        param: getattr(args, option)
        for option, param in CLI_SOLVER_KNOBS.items()
        if getattr(args, option) is not None
    }
    key_groups = (
        ("stopping_time", "time_after_resection"),
        ("n_steps", "dt", "steps_per_day"),
    )
    merged: dict[str, Any] = {}
    for layer in (DEFAULT_SOLVER_PARAMS, manifest_solver, cli_solver):
        for group in key_groups:
            if any(key in layer for key in group):
                for key in group:
                    merged.pop(key, None)
        merged.update(layer)
    merged = resolve_time_step(resolve_horizon(merged, treatment["resection_time"]))
    n_snapshots = None if args.no_plot else int(args.n_snapshots)
    params: dict[str, Any] = {
        "voxel_size_mm": voxel_size,
        "verbose": True,
        **merged,
        "gray_matter_pbmap": gm,
        "white_matter_pbmap": wm,
        "n_time_series_snapshots": n_snapshots,
        **treatment,
    }
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

    n_steps, dt = solver._resolve_time_stepping()
    config["n_steps"] = n_steps
    config["dt"] = dt
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    cell_density = np.asarray(result.final_state["cell_density"], dtype=np.float32)
    image = nib.Nifti1Image(cell_density, wm_img.affine, header=wm_img.header)
    image.set_data_dtype(np.float32)
    nib.save(image, str(run_dir / "final_cell_density.nii.gz"))

    if not args.no_plot:
        frames = result.time_series["cell_density"]
        times = ((snapshot_steps(n_steps, n_snapshots) + 1) * dt)[: frames.shape[0]]
        masses = frames.sum(axis=(1, 2, 3))
        seed_voxel = tuple(
            int(solver.params[f"gaussian_seed_{axis}_fraction"] * n)
            for axis, n in zip("xyz", wm.shape)
        )
        # The state just before the resection: a second, shorter, untreated
        # solve with FKPPSolver, whose dynamics StuppFKPPSolver reproduces
        # up to that step.
        pre_resection = None
        pre_points = [(0.0, float(result.initial_state["cell_density"].sum()))]
        n_pre = int(round(float(treatment["resection_time"]) / dt))
        if n_pre >= 1:
            untreated = {key: value for key, value in params.items() if key not in treatment}
            pre = FKPPSolver(
                {
                    **untreated,
                    "stopping_time": n_pre * dt,
                    "n_steps": n_pre,
                    "n_time_series_snapshots": None,
                }
            ).solve()
            if not pre.success:
                raise RuntimeError(f"pre-resection solve failed: {pre.error}")
            pre_state = pre.final_state["cell_density"]
            pre_resection = (n_pre * dt, pre_state)
            pre_points.append((n_pre * dt, float(pre_state.sum())))
        if args.t1c is not None:
            background = np.asarray(nib.load(args.t1c).get_fdata(), dtype=np.float64)
            if background.shape != wm.shape:
                raise ValueError(
                    f"--t1c shape {background.shape} differs from the tissue maps {wm.shape}."
                )
        else:
            background = wm + gm
        z = int(round(center_of_mass(treatment["rt_dose"])[2]))
        panels = select_panels(
            result.initial_state["cell_density"],
            frames,
            times,
            pre_resection,
            float(treatment["resection_time"]),
            np.asarray(treatment["rt_times"], dtype=np.float64),
            args.n_treatment_panels,
        )
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
            treatment["resection_cavity"],
            seed_voxel,
            z,
            args.threshold,
            times,
            masses,
            pre_points,
            treatment,
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
