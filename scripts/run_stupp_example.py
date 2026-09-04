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
from typing import Any

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

DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")
# Background image: <session dir of the wm pbmap>/skull_stripped/t1c_skullstripped.nii.gz
# unless --background-image is given (the SAILOR layout, so a patient change in the
# manifest's tissue maps carries over).
T1C_RELATIVE = Path("skull_stripped") / "t1c_skullstripped.nii.gz"

CAVITY_COLOR = (210 / 255.0, 43 / 255.0, 43 / 255.0, 1)
SEED_COLOR = (34 / 255.0, 139 / 255.0, 34 / 255.0, 1)


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


def montage_days(
    resection_time: float, dt: float, rt_times: np.ndarray, stopping_time: float
) -> list[float]:
    """
    Days of the eight recorded montage panels (the seed is the initial
    state): one step before and one day after the resection; three evenly
    spaced days inside the radiotherapy block (first to last fraction,
    endpoints excluded); three evenly spaced days from the last fraction to
    the end of the run, the end included. Passed as the solver's
    snapshot_times, so each frame is recorded at the nearest time step.
    """
    rt_start, rt_end = float(np.min(rt_times)), float(np.max(rt_times))
    return [
        resection_time - dt,
        resection_time + 1.0,
        *np.linspace(rt_start, rt_end, 5)[1:-1],
        *np.linspace(rt_end, stopping_time, 4)[1:],
    ]


def select_panels(
    initial: np.ndarray, frames: np.ndarray, times: np.ndarray, days: list[float]
) -> list[tuple[str, np.ndarray]]:
    """
    The nine montage panels, three per row: the seed, then for each of
    the ``montage_days`` the recorded frame nearest to it.

    Args:
        initial: Initial state (t = 0).
        frames: Recorded frames of the treated run, (n_frames, ...).
        times: Simulation day of each frame (Result.snapshot_times).
        days: The eight montage days the frames were recorded for.

    Returns:
        (title, volume) pairs.
    """
    labels = [" (before resection)", " (after resection)", *[" (RT/TMZ)"] * 3, *[""] * 3]
    panels = [("t = 0 d (seed)", initial)]
    for day, label in zip(days, labels, strict=True):
        k = int(np.argmin(np.abs(times - day)))
        panels.append((f"t = {times[k]:.0f} d{label}", frames[k]))
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
