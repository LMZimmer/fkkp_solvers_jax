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
  2. the full treated horizon with those inputs and evenly spaced snapshots
     (--n-snapshots).

The seed is placed at --seed-voxel, replacing the manifest's seed fractions
(default (132, 103, 90), right-hemisphere deep white matter around
mid-height of the grid). The figure shows, on the axial slice through the
dose map's center of mass (or --slice-z), the gm pbmap as the background
with the cell density overlaid (np.rot90 orientation, inferno
overlay, densities below --threshold transparent), three by three: the
seed, the state before and one day after the resection; three evenly
spaced times inside the radiotherapy block; three evenly spaced times from
its end to the end of the run. Each is the recorded snapshot nearest to its
time. The cavity outline and the seed voxel are marked. A total-mass-vs-time panel
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
from typing import Any

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

DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")
DEFAULT_WM = str(_ROOT / "reference_solves" / "wm_pbmap.nii.gz")
DEFAULT_GM = str(_ROOT / "reference_solves" / "gm_pbmap.nii.gz")
# Right-hemisphere deep white matter around mid-height of the
# reference_solves grid (the reference solves' seed (140, 116, 55) sits low,
# near the skull base).
DEFAULT_SEED_VOXEL = (132, 103, 90)
# Manifest entries replaced by --wm/--gm and the synthetic cavity and dose map.
SYNTHETIC_VOLUME_KEYS = ("white_matter_pbmap", "gray_matter_pbmap", "resection_cavity", "rt_dose")

CAVITY_COLOR = (210 / 255.0, 43 / 255.0, 43 / 255.0, 1)
SEED_COLOR = (34 / 255.0, 139 / 255.0, 34 / 255.0, 1)


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
        "--n-snapshots", type=int, default=25, help="evenly spaced snapshots of the treated run"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="cell density below which the overlay is transparent"
    )
    return parser.parse_args(argv)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: jsonable(value)
        for key, value in params.items()
        if not (isinstance(value, np.ndarray) and value.ndim > 1)
    }


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


def snapshot_steps(n_steps: int, n_snapshots: int) -> np.ndarray:
    """Recording schedule of operators._run_time_loop."""
    return np.unique(np.linspace(0, n_steps - 1, n_snapshots, dtype=np.int64))


def select_panels(
    initial: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    pre_resection: tuple[float, np.ndarray],
    resection_time: float,
    rt_times: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """
    The nine montage panels, three per row. Apart from the first two, each
    is the recorded frame nearest to its target time (frames repeat when
    the snapshots are coarser than the targets):

      row 1: the seed (t = 0), the state just before the resection, one day
             after the resection;
      row 2: three evenly spaced times inside the radiotherapy block (first
             to last fraction, endpoints excluded);
      row 3: three evenly spaced times from the last fraction to the end of
             the run, the end included.

    Args:
        initial: Initial state (t = 0).
        frames: Recorded frames of the treated run, (n_frames, ...).
        times: Simulation time of each frame.
        pre_resection: (time, state) just before the resection.
        resection_time: Resection time.
        rt_times: Radiotherapy fraction times.

    Returns:
        (title, volume) pairs.
    """

    def nearest(label: str, t: float) -> tuple[str, np.ndarray]:
        k = int(np.argmin(np.abs(times - t)))
        return f"t = {times[k]:.0f} d{label}", frames[k]

    rt_start, rt_end = float(np.min(rt_times)), float(np.max(rt_times))
    panels = [
        ("t = 0 d (seed)", initial),
        (f"t = {pre_resection[0]:.0f} d (before resection)", pre_resection[1]),
        nearest(" (after resection)", resection_time + 1.0),
    ]
    panels += [nearest(" (RT/TMZ)", t) for t in np.linspace(rt_start, rt_end, 5)[1:-1]]
    panels += [nearest("", t) for t in np.linspace(rt_end, float(times[-1]), 4)[1:]]
    return panels


def session_blocks(days: np.ndarray, max_gap: float = 1.0) -> list[tuple[float, float]]:
    """(first, last) day of each run of sessions that are at most max_gap
    days apart, for sorted session days."""
    blocks: list[tuple[float, float]] = []
    for day in days:
        if blocks and day - blocks[-1][1] <= max_gap:
            blocks[-1] = (blocks[-1][0], float(day))
        else:
            blocks.append((float(day), float(day)))
    return blocks


def render(
    outfile_stem: Path,
    header: str,
    panels: list[tuple[str, np.ndarray]],
    background_volume: np.ndarray,
    cavity: np.ndarray | None,
    seed_voxel: tuple[int, int, int],
    z: int,
    threshold: float,
    times: np.ndarray,
    masses: np.ndarray,
    pre_points: list[tuple[float, float]],
    params: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_col = 3
    n_row = int(np.ceil(len(panels) / n_col))
    fig = plt.figure(figsize=(5 * n_col + 0.6, 4 * n_row + 3.6), constrained_layout=True)
    grid = fig.add_gridspec(
        n_row + 1, n_col + 1, height_ratios=[1.0] * n_row + [0.75],
        width_ratios=[1.0] * n_col + [0.05],
    )
    background = np.rot90(background_volume[:, :, z])
    cavity_slice = np.rot90(cavity[:, :, z]) if cavity is not None else None
    # np.rot90 maps array (i, j) to image row (ny - 1 - j), column i.
    seed_col, seed_row = seed_voxel[0], background_volume.shape[1] - 1 - seed_voxel[1]
    image = None
    for index, (title, volume) in enumerate(panels):
        ax = fig.add_subplot(grid[index // n_col, index % n_col])
        ax.imshow(background, cmap="gray", interpolation="none")
        overlay = np.ma.masked_less(np.rot90(volume[:, :, z]), threshold)
        image = ax.imshow(
            overlay, cmap="inferno", alpha=0.90, vmin=0.0, vmax=1.0, interpolation="none"
        )
        if cavity_slice is not None and cavity_slice.any():
            ax.contour(cavity_slice.astype(float), levels=[0.5], colors=[CAVITY_COLOR], linewidths=1.2)
        if seed_voxel[2] == z:
            ax.plot(seed_col, seed_row, "+", color=SEED_COLOR, markersize=10, markeredgewidth=1.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(
            0.02, 0.02, f"mass {volume.sum():.0f}", transform=ax.transAxes,
            color="white", fontsize=10, va="bottom",
        )
        ax.axis("off")
    if image is not None:
        cax = fig.add_subplot(grid[:n_row, n_col])
        fig.colorbar(image, cax=cax, label="cell density")

    ax = fig.add_subplot(grid[n_row, :n_col])
    ax.plot(times, masses, "ko-", markersize=3, label="mass")
    if pre_points:
        ax.plot(*zip(*pre_points), "s", color="gray", label="before resection")
    ax.axvline(
        float(params["resection_time"]), color=CAVITY_COLOR, linewidth=1.5, label="resection"
    )
    # Radiotherapy days as lines, split by whether chemotherapy is given the
    # same day; chemotherapy-only days (daily sessions, many of them) as one
    # shaded band per contiguous block of sessions.
    rt_days = np.atleast_1d(np.asarray(params["rt_times"], dtype=np.float64))
    ct_days = np.atleast_1d(np.asarray(params["chemo_times"], dtype=np.float64))
    with_ct = np.isin(rt_days, ct_days)
    for days, label, color in (
        (rt_days[with_ct], "radio- + chemotherapy", "blue"),
        (rt_days[~with_ct], "radiotherapy only", "purple"),
    ):
        for index, t in enumerate(days):
            ax.axvline(
                float(t), color=color, linewidth=0.6, alpha=0.4, label=label if index == 0 else None
            )
    ct_only = np.sort(ct_days[~np.isin(ct_days, rt_days)])
    for index, (start, end) in enumerate(session_blocks(ct_only)):
        ax.axvspan(
            start - 0.5, end + 0.5, color="green", alpha=0.25, linewidth=0,
            label="chemotherapy only" if index == 0 else None,
        )
    ax.set_yscale("log")
    ax.set_xlabel("time [days]", fontsize=12)
    ax.set_ylabel("total mass", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    fig.suptitle(header, horizontalalignment="left", x=0.02, fontsize=15, fontweight="bold")
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)


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
    # StuppFKPPSolver reproduces up to that step).
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

    # Solve 2: the treated horizon with snapshots.
    wall = time.perf_counter()
    treated = StuppFKPPSolver({**params, "n_time_series_snapshots": args.n_snapshots}).solve()
    if not treated.success:
        raise RuntimeError(f"treated solve failed: {treated.error}")
    wall_treated = time.perf_counter() - wall
    print(f"treated solve ({n_steps} steps): {wall_treated:.1f} s, final mass {treated.final_stopping_quantity:.1f}")

    frames = treated.time_series["cell_density"]
    times = ((snapshot_steps(n_steps, args.n_snapshots) + 1) * dt)[: frames.shape[0]]
    masses = frames.sum(axis=(1, 2, 3))
    panels = select_panels(
        pre.initial_state["cell_density"],
        frames,
        times,
        (n_pre * dt, pre.final_state["cell_density"]),
        resection_time,
        np.asarray(params["rt_times"], dtype=np.float64),
    )

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
        [(n_pre * dt, float(pre.final_state["cell_density"].sum()))],
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
