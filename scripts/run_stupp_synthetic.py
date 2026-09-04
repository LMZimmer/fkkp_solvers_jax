#!/usr/bin/env python
"""Run the example Stupp-protocol manifest on the reference_solves tissue maps
with fully synthetic treatment inputs and render the course as a figure.

The counterpart of scripts/run_stupp_example.py without patient data: the
tissue maps default to reference_solves/{wm,gm}_pbmap.nii.gz and the cavity
and dose map are derived from the simulation itself instead of segmentations
and a planning dose. Two solves of fisher_kpp_jax.StuppFKPPSolver with the
manifest's parameters (scripts/stupp_manifest_example.json by default; its
volume entries white_matter_pbmap, gray_matter_pbmap, resection_cavity and
rt_dose are ignored and may be omitted):

  1. seed -> resection time with no event firing. The state just BEFORE the
     resection defines the synthetic treatment inputs: the resection cavity
     is the region with cell density > --tumor-threshold (default 0.4), and
     the radiotherapy dose map is --total-dose-gy (default 60 Gy, TOTAL over
     all fractions) on that region dilated by --margin-mm (default 15 mm,
     Euclidean in world units), zero elsewhere.
  2. the full treated horizon with those inputs, recording the eight
     montage frames.

The seed is placed at --seed-voxel, replacing the manifest's seed fractions
(default (132, 103, 90), right-hemisphere deep white matter around
mid-height of the grid). The figure shows, on the axial slice through the
dose map's center of mass (or --slice-z), the gm pbmap as the background
with the cell density overlaid (np.rot90 orientation, inferno
overlay, densities below --threshold transparent), three by three: the
seed, one step before and one day after the resection; three evenly spaced
days inside the radiotherapy block; three evenly spaced days from its end
to the end of the run (each recorded at the nearest time step). The cavity
outline and the seed voxel are marked. The figure helpers are those of
scripts/run_stupp_example.py. A total-mass-vs-time panel
with the treatment events marked sits below. Written into
<output-dir>/<run-name>/ (exist_ok=False, nothing outside it): overview.png,
overview.pdf and run_summary.json (parameters, seed voxel, slice, synthetic
cavity/dose stats, snapshot times and masses, wall times).

Run from the project root, e.g.:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/run_stupp_synthetic.py --output-dir runs/
  JAX_PLATFORMS=cpu python scripts/run_stupp_synthetic.py --output-dir runs/ --n-snapshots 13
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
from scipy.ndimage import center_of_mass, distance_transform_edt  # noqa: E402

from fisher_kpp_jax import FKPPSolver, StuppFKPPSolver  # noqa: E402
from fisher_kpp_jax.solvers import params_from_manifest, read_manifest  # noqa: E402
from run_stupp_example import (  # noqa: E402
    jsonable,
    montage_days,
    render,
    scalar_params,
    select_panels,
)

DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")
DEFAULT_WM = str(_ROOT / "reference_solves" / "wm_pbmap.nii.gz")
DEFAULT_GM = str(_ROOT / "reference_solves" / "gm_pbmap.nii.gz")
# Right-hemisphere deep white matter around mid-height of the
# reference_solves grid (the reference solves' seed (140, 116, 55) sits low,
# near the skull base).
DEFAULT_SEED_VOXEL = (132, 103, 90)
# Manifest entries replaced by --wm/--gm and the synthetic cavity and dose map.
SYNTHETIC_VOLUME_KEYS = ("white_matter_pbmap", "gray_matter_pbmap", "resection_cavity", "rt_dose")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="run manifest JSON")
    parser.add_argument("--wm", default=DEFAULT_WM, help="white matter pbmap NIfTI")
    parser.add_argument("--gm", default=DEFAULT_GM, help="gray matter pbmap NIfTI")
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name", default=None, help="run directory name (default: stupp_synthetic_<UTC>_<pid>)"
    )
    parser.add_argument(
        "--seed-voxel",
        type=int,
        nargs=3,
        default=list(DEFAULT_SEED_VOXEL),
        metavar=("X", "Y", "Z"),
        help=f"voxel that seeds the tumor (default {DEFAULT_SEED_VOXEL})",
    )
    parser.add_argument(
        "--tumor-threshold",
        type=float,
        default=0.4,
        help="cell density above which the pre-resection tumor is resected and targeted (default 0.4)",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=15.0,
        help="radiotherapy target margin around the resected region in mm (default 15)",
    )
    parser.add_argument(
        "--total-dose-gy",
        type=float,
        default=60.0,
        help="TOTAL radiotherapy dose over all fractions inside the target in Gy (default 60)",
    )
    parser.add_argument(
        "--slice-z", type=int, default=None, help="axial slice (default: dose-map center of mass)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="cell density below which the overlay is transparent"
    )
    return parser.parse_args(argv)


def synthetic_treatment_volumes(
    pre_resection_density: np.ndarray,
    tumor_threshold: float,
    margin_mm: float,
    total_dose_gy: float,
    voxel_size: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    The synthetic cavity and dose map from the state just before the
    resection: the cavity is the region with cell density above
    tumor_threshold, the dose map is total_dose_gy on the cavity dilated
    by margin_mm (Euclidean distance in mm, anisotropic voxels honored)
    and zero elsewhere.

    Returns:
        (cavity, dose): a bool array and a float64 array in Gy.
    """
    cavity = pre_resection_density > tumor_threshold
    if not cavity.any():
        raise ValueError(
            f"no voxel exceeds the tumor threshold {tumor_threshold:g} before the "
            "resection; lower --tumor-threshold or grow longer."
        )
    distance_to_cavity = distance_transform_edt(~cavity, sampling=voxel_size)
    target = distance_to_cavity <= margin_mm
    dose = np.where(target, float(total_dose_gy), 0.0)
    return cavity, dose


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or (
        f"stupp_synthetic_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # The manifest's volumes are replaced: the tissue maps by --wm/--gm, the
    # cavity and dose map by the synthetic ones below.
    manifest = {
        key: value
        for key, value in read_manifest(args.manifest).items()
        if key not in SYNTHETIC_VOLUME_KEYS
    }
    wm_img = nib.load(args.wm)
    wm = np.asarray(wm_img.get_fdata(), dtype=np.float64)
    gm = np.asarray(nib.load(args.gm).get_fdata(), dtype=np.float64)
    if gm.shape != wm.shape:
        raise ValueError(f"--gm shape {gm.shape} differs from --wm {wm.shape}.")
    voxel_size = tuple(float(v) for v in wm_img.header.get_zooms()[:3])
    background_volume = gm

    seed_voxel = tuple(int(v) for v in args.seed_voxel)
    if not all(0 <= seed_voxel[i] < wm.shape[i] for i in range(3)):
        raise ValueError(f"--seed-voxel {seed_voxel} lies outside the grid {wm.shape}.")
    # +0.5 so the solver's int(fraction * N) lands on the voxel.
    seed_fractions = {
        f"gaussian_seed_{axis}_fraction": (seed_voxel[i] + 0.5) / wm.shape[i]
        for i, axis in enumerate("xyz")
    }

    params = params_from_manifest(
        {
            "voxel_size_mm": voxel_size,
            "verbose": False,
            **manifest,
            "gray_matter_pbmap": gm,
            "white_matter_pbmap": wm,
            **seed_fractions,
        }
    )
    resection_time = float(params["resection_time"])
    stopping_time = resection_time + float(params["time_after_resection"])
    n_steps = int(params["n_steps"])
    dt = stopping_time / n_steps
    print(f"run directory: {run_dir}")
    print(f"seed voxel {seed_voxel}, {n_steps} steps (dt={dt:.4g} d)")

    # Solve 1: untreated up to the resection (FKPPSolver, whose dynamics
    # StuppFKPPSolver reproduces up to that step); its final state defines
    # the synthetic cavity and dose map, inputs of the treated solve.
    n_pre = int(round(resection_time / dt))
    wall = time.perf_counter()
    untreated = {
        key: value
        for key, value in params.items()
        if key not in StuppFKPPSolver.TREATMENT_KEYS and key != "time_after_resection"
    }
    pre = FKPPSolver({**untreated, "stopping_time": n_pre * dt, "n_steps": n_pre}).solve()
    if not pre.success:
        raise RuntimeError(f"pre-resection solve failed: {pre.error}")
    wall_pre = time.perf_counter() - wall
    print(f"pre-resection solve ({n_pre} steps): {wall_pre:.1f} s, mass {pre.final_stopping_quantity:.1f}")

    # The synthetic cavity and dose map from the state just before the
    # resection.
    cavity, dose = synthetic_treatment_volumes(
        pre.final_state["cell_density"],
        args.tumor_threshold,
        args.margin_mm,
        args.total_dose_gy,
        voxel_size,
    )
    params["resection_cavity"] = cavity
    params["rt_dose"] = dose
    print(
        f"synthetic cavity (density > {args.tumor_threshold:g}): {int(cavity.sum())} voxels; "
        f"dose target (+{args.margin_mm:g} mm): {int((dose > 0).sum())} voxels "
        f"at {args.total_dose_gy:g} Gy total"
    )
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(dose)[2]))
    print(f"slice z={z}")

    # Solve 2: the treated horizon, recording the montage frames.
    panel_days = montage_days(resection_time, dt, params["rt_times"], stopping_time)
    params["snapshot_times"] = panel_days
    wall = time.perf_counter()
    treated = StuppFKPPSolver(params).solve()
    if not treated.success:
        raise RuntimeError(f"treated solve failed: {treated.error}")
    wall_treated = time.perf_counter() - wall
    print(f"treated solve ({n_steps} steps): {wall_treated:.1f} s, final mass {treated.final_stopping_quantity:.1f}")

    frames = treated.time_series["cell_density"]
    times = treated.snapshot_times
    masses = frames.sum(axis=(1, 2, 3))
    panels = select_panels(pre.initial_state["cell_density"], frames, times, panel_days)

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
        cavity,
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
        "tissue": {"wm": str(Path(args.wm).resolve()), "gm": str(Path(args.gm).resolve())},
        "params": scalar_params(params),
        "seed_voxel": list(seed_voxel),
        "slice_z": z,
        "n_steps": n_steps,
        "dt": dt,
        "synthetic_treatment": {
            "tumor_threshold": args.tumor_threshold,
            "margin_mm": args.margin_mm,
            "total_dose_gy": args.total_dose_gy,
            "cavity_voxels": int(cavity.sum()),
            "dose_target_voxels": int((dose > 0).sum()),
        },
        "pre_resection": {
            "final_time": pre.final_time,
            "final_mass": pre.final_stopping_quantity,
            "wall_time_s": wall_pre,
        },
        "treated": {
            "final_time": treated.final_time,
            "stopping_criterion": treated.stopping_criterion,
            "final_mass": treated.final_stopping_quantity,
            "wall_time_s": wall_treated,
            "snapshot_times": jsonable(times),
            "snapshot_masses": jsonable(masses),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {run_dir / 'overview.png'} (+ .pdf, run_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
