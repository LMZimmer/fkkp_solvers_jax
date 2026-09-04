#!/usr/bin/env python
"""Run the example Stupp-protocol manifest on the SAILOR subject and render
the treatment course as a figure.

One solve of fisher_kpp_jax.StuppFKPPSolver with the manifest's parameters
(scripts/stupp_manifest_example.json by default; it must name the tissue
maps, the cavity segmentation, the dose map and the time step), recording
the state at the eight montage days below plus --n-snapshots evenly spaced
days that sample the mass curve.

The seed is placed at the center of mass of one label of the manifest's
cavity segmentation (--seed-label, default 3 = enhancing core), replacing the
manifest's seed fractions. The figure shows, on the axial slice through the
dose map's center of mass (or --slice-z), the T1c image with the cell
density overlaid in the style of PredictGBM's multislice plots
(np.rot90 orientation, inferno overlay, densities below --threshold
transparent), three by three: the seed, one step before and one day after
the resection; three evenly spaced days inside the radiotherapy block;
three evenly spaced days from its end to the end of the run (each recorded
at the nearest time step). The cavity outline and the seed voxel are
marked. A total-mass-vs-time panel with the treatment events marked sits
below. Written into <output-dir>/<run-name>/ (exist_ok=False, nothing
outside it): overview.png, overview.pdf and run_summary.json (parameters,
seed voxel, slice, snapshot times and masses, wall times).

Run from the project root, e.g.:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/run_stupp_example.py --output-dir runs/
  JAX_PLATFORMS=cpu python scripts/run_stupp_example.py --output-dir runs/ --n-snapshots 13
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
# Background image: <session dir of the wm pbmap>/skull_stripped/t1c_skullstripped.nii.gz
# unless --background-image is given (the SAILOR layout, so a patient change in the
# manifest's tissue maps carries over).
T1C_RELATIVE = Path("skull_stripped") / "t1c_skullstripped.nii.gz"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="run manifest JSON")
    parser.add_argument(
        "--background-image",
        default=None,
        help=f"background image NIfTI (default: <wm pbmap session dir>/{T1C_RELATIVE})",
    )
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name", default=None, help="run directory name (default: stupp_example_<UTC>_<pid>)"
    )
    parser.add_argument(
        "--seed-label",
        type=int,
        default=3,
        help="segmentation label whose center of mass seeds the tumor (default 3)",
    )
    parser.add_argument(
        "--slice-z", type=int, default=None, help="axial slice (default: dose-map center of mass)"
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=25,
        help="evenly spaced snapshot days sampling the mass curve (default 25)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="cell density below which the overlay is transparent"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or (
        f"stupp_example_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = read_manifest(args.manifest)
    wm_path, gm_path = manifest["white_matter_pbmap"], manifest["gray_matter_pbmap"]
    wm_img = nib.load(wm_path)
    voxel_size = tuple(float(v) for v in wm_img.header.get_zooms()[:3])
    segmentation = np.rint(
        nib.load(manifest["resection_cavity"]["segmentation"]).get_fdata()
    ).astype(np.int64)
    if not (segmentation == args.seed_label).any():
        raise ValueError(f"seed label {args.seed_label} is absent from the cavity segmentation.")
    seed_voxel = tuple(int(v) for v in np.rint(center_of_mass(segmentation == args.seed_label)))
    # +0.5 so the solver's int(fraction * N) lands on the voxel.
    seed_fractions = {
        f"gaussian_seed_{axis}_fraction": (seed_voxel[i] + 0.5) / segmentation.shape[i]
        for i, axis in enumerate("xyz")
    }
    params = params_from_manifest(
        {"voxel_size_mm": voxel_size, "verbose": False, **manifest, **seed_fractions}
    )
    wm = params["white_matter_pbmap"]
    background_path = (
        Path(args.background_image)
        if args.background_image
        else Path(wm_path).parent.parent / T1C_RELATIVE
    )
    if not background_path.is_file():
        raise FileNotFoundError(
            f"background image not found: {background_path} (pass --background-image)"
        )
    background_volume = np.asarray(nib.load(str(background_path)).get_fdata(), dtype=np.float64)
    if background_volume.shape != wm.shape:
        raise ValueError(
            f"--background-image shape {background_volume.shape} differs from the tissue "
            f"maps {wm.shape}."
        )
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(params["rt_dose"])[2]))

    resection_time = float(params["resection_time"])
    stopping_time = resection_time + float(params["time_after_resection"])
    n_steps = int(params["n_steps"])
    dt = stopping_time / n_steps
    panel_days = montage_days(resection_time, dt, params["rt_times"], stopping_time)
    # The montage frames plus evenly spaced days sampling the mass curve.
    params["snapshot_times"] = [*panel_days, *np.linspace(0.0, stopping_time, args.n_snapshots)]
    print(f"run directory: {run_dir}")
    print(f"seed voxel {seed_voxel} (label {args.seed_label} CoM), slice z={z}, {n_steps} steps (dt={dt:.4g} d)")

    wall = time.perf_counter()
    result = StuppFKPPSolver(params).solve()
    if not result.success:
        raise RuntimeError(f"solve failed: {result.error}")
    wall = time.perf_counter() - wall
    frames = result.time_series["cell_density"]
    times = result.snapshot_times
    masses = frames.sum(axis=(1, 2, 3))
    print(
        f"solve ({n_steps} steps, {times.size} snapshots): {wall:.1f} s, "
        f"final mass {result.final_stopping_quantity:.1f}"
    )
    panels = select_panels(result.initial_state["cell_density"], frames, times, panel_days)

    header = (
        f"D {params['white_matter_diffusivity']:g}, rho {params['rho']:g}, "
        f"ratio {params['diffusivity_ratio']:g}, "
        f"alpha {params['rt_alpha']:g}, beta {params['rt_beta']:g}, "
        f"kill {params['chemo_kill_rate']:g} /(mg/m^2), decay {params['chemo_decay_rate']:g}, "
        f"TMZ {np.min(params['chemo_doses']):g}-{np.max(params['chemo_doses']):g} mg/m^2"
    )
    render(
        run_dir / "overview",
        header,
        panels,
        background_volume,
        params["resection_cavity"],
        seed_voxel,
        z,
        args.threshold,
        times,
        masses,
        params,
    )
    summary = {
        "run_name": run_name,
        "cli_args": jsonable(vars(args)),
        "manifest": str(Path(args.manifest).resolve()),
        "tissue": {"wm": wm_path, "gm": gm_path},
        "background_image": str(background_path),
        "params": scalar_params(params),
        "seed_voxel": list(seed_voxel),
        "slice_z": z,
        "n_steps": n_steps,
        "dt": dt,
        "final_time": result.final_time,
        "stopping_criterion": result.stopping_criterion,
        "final_mass": result.final_stopping_quantity,
        "wall_time_s": wall,
        "snapshot_times": jsonable(times),
        "snapshot_masses": jsonable(masses),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {run_dir / 'overview.png'} (+ .pdf, run_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
