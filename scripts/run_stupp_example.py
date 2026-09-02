#!/usr/bin/env python
"""Run the example Stupp-protocol manifest on the SAILOR subject and render
the treatment course as a figure.

Two solves of fisher_kpp_jax.StuppFKPPSolver with the manifest's solver and
treatment parameters (scripts/stupp_manifest_example.json by default):

  1. seed -> resection time with no event firing (the state just BEFORE
     the resection),
  2. the full treated horizon with evenly spaced snapshots (--n-snapshots).

The seed is placed at the center of mass of one label of the manifest's
tumor segmentation (--seed-label, default 3 = enhancing core), replacing the
manifest's seed fractions. The figure shows, on the axial slice through the
dose map's center of mass (or --slice-z), the T1c image with the cell
density overlaid in the style of PredictGBM's multislice plots
(np.rot90 orientation, inferno overlay, densities below --threshold
transparent): the seed, the state before and after resection, the snapshots
nearest to --n-treatment-panels evenly spaced times across the radiotherapy
block, and the end of the run; the cavity outline and the seed voxel are
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

from fisher_kpp_jax import FKPPSolver, StuppFKPPSolver  # noqa: E402
from fisher_kpp_jax.solvers import (  # noqa: E402
    solver_params_from_manifest,
    tissue_paths_from_manifest,
    treatment_params_from_manifest,
)

DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")
# Background image: <session dir of the wm pbmap>/skull_stripped/t1c_skullstripped.nii.gz
# unless --t1c is given (the SAILOR layout, so a patient change in the
# manifest's 'tissue' section carries over).
T1C_RELATIVE = Path("skull_stripped") / "t1c_skullstripped.nii.gz"
DEFAULT_WM = "/mnt/Drive4/lucas/SAILOR/processed/sub-01/ses-01/tissue_segmentation/wm_pbmap.nii.gz"
DEFAULT_GM = "/mnt/Drive4/lucas/SAILOR/processed/sub-01/ses-01/tissue_segmentation/gm_pbmap.nii.gz"

CAVITY_COLOR = (210 / 255.0, 43 / 255.0, 43 / 255.0, 1)
SEED_COLOR = (34 / 255.0, 139 / 255.0, 34 / 255.0, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="run manifest JSON")
    parser.add_argument(
        "--t1c",
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
        "--n-snapshots", type=int, default=25, help="evenly spaced snapshots of the treated run"
    )
    parser.add_argument(
        "--n-treatment-panels", type=int, default=5, help="panels across the radiotherapy block"
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


def snapshot_steps(n_steps: int, n_snapshots: int) -> np.ndarray:
    """Recording schedule of operators._run_time_loop."""
    return np.unique(np.linspace(0, n_steps - 1, n_snapshots, dtype=np.int64))


def select_panels(
    initial: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    pre_resection: tuple[float, np.ndarray] | None,
    resection_time: float | None,
    rt_times: np.ndarray | None,
    n_treatment_panels: int,
) -> list[tuple[str, np.ndarray]]:
    """
    Pick the montage panels: the seed, the state before and after the
    resection (if any), the recorded frames nearest to n_treatment_panels
    evenly spaced times across the radiotherapy block (or across the whole
    horizon without radiotherapy), and the last frame.

    Args:
        initial: Initial state (t = 0).
        frames: Recorded frames of the treated run, (n_frames, ...).
        times: Simulation time of each frame.
        pre_resection: (time, state) just before the resection, or None.
        resection_time: Resection time, or None.
        rt_times: Radiotherapy fraction times, or None.
        n_treatment_panels: Number of panels between the resection and the
            end.

    Returns:
        (title, volume) pairs.
    """

    def nearest(t: float) -> int:
        return int(np.argmin(np.abs(times - t)))

    panels = [("t = 0 d (seed)", initial)]
    if pre_resection is not None:
        panels.append((f"t = {pre_resection[0]:.0f} d, before resection", pre_resection[1]))
    if resection_time is not None:
        k = nearest(resection_time)
        panels.append((f"t = {times[k]:.1f} d, after resection", frames[k]))
    if rt_times is not None and rt_times.size:
        span, label = (float(rt_times.min()), float(rt_times.max())), " (RT/TMZ)"
    else:
        span, label = (float(times[0]), float(times[-1])), ""
    used = {nearest(resection_time)} if resection_time is not None else set()
    for t in np.linspace(span[0], span[1], n_treatment_panels + 2)[1:-1]:
        k = nearest(t)
        if k in used or k == frames.shape[0] - 1:
            continue
        used.add(k)
        panels.append((f"t = {times[k]:.1f} d{label}", frames[k]))
    panels.append((f"t = {times[-1]:.0f} d (end)", frames[-1]))
    return panels


def render(
    outfile_stem: Path,
    header: str,
    panels: list[tuple[str, np.ndarray]],
    t1c: np.ndarray,
    cavity: np.ndarray | None,
    seed_voxel: tuple[int, int, int],
    z: int,
    threshold: float,
    times: np.ndarray,
    masses: np.ndarray,
    pre_points: list[tuple[float, float]],
    treatment: dict[str, Any],
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
    background = np.rot90(t1c[:, :, z])
    cavity_slice = np.rot90(cavity[:, :, z]) if cavity is not None else None
    # np.rot90 maps array (i, j) to image row (ny - 1 - j), column i.
    seed_col, seed_row = seed_voxel[0], t1c.shape[1] - 1 - seed_voxel[1]
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
    events = [("resection_time", "resection", CAVITY_COLOR, 1.5, 1.0)]
    if np.array_equal(
        np.atleast_1d(treatment["rt_times"]), np.atleast_1d(treatment["chemo_times"])
    ):
        # One event class: the fractions and the sessions coincide.
        events.append(("rt_times", "radio-/chemotherapy", "blue", 0.6, 0.4))
    else:
        events.append(("rt_times", "radiotherapy", "blue", 0.6, 0.4))
        events.append(("chemo_times", "chemotherapy", "green", 0.6, 0.4))
    for key, label, color, width, alpha in events:
        for index, t in enumerate(np.atleast_1d(treatment[key])):
            ax.axvline(float(t), color=color, linewidth=width, alpha=alpha, label=label if index == 0 else None)
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

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    tissue_paths = tissue_paths_from_manifest(args.manifest)
    wm_path = str(tissue_paths.get("wm", DEFAULT_WM))
    gm_path = str(tissue_paths.get("gm", DEFAULT_GM))
    wm_img = nib.load(wm_path)
    wm = np.asarray(wm_img.get_fdata(), dtype=np.float64)
    gm = np.asarray(nib.load(gm_path).get_fdata(), dtype=np.float64)
    t1c_path = Path(args.t1c) if args.t1c else Path(wm_path).parent.parent / T1C_RELATIVE
    if not t1c_path.is_file():
        raise FileNotFoundError(f"background image not found: {t1c_path} (pass --t1c)")
    t1c = np.asarray(nib.load(str(t1c_path)).get_fdata(), dtype=np.float64)
    if t1c.shape != wm.shape:
        raise ValueError(f"--t1c shape {t1c.shape} differs from the tissue maps {wm.shape}.")
    voxel_size = tuple(float(v) for v in wm_img.header.get_zooms()[:3])

    solver_params = solver_params_from_manifest(args.manifest)
    treatment = treatment_params_from_manifest(args.manifest)
    segmentation = np.rint(
        nib.load(str(Path(args.manifest).parent / manifest["resection"]["tumor_segmentation"]))
        .get_fdata()
    ).astype(np.int64)
    if not (segmentation == args.seed_label).any():
        raise ValueError(f"seed label {args.seed_label} is absent from the tumor segmentation.")
    seed_voxel = tuple(int(v) for v in np.rint(center_of_mass(segmentation == args.seed_label)))
    # +0.5 so the solver's int(fraction * N) lands on the voxel.
    seed_fractions = {
        f"gaussian_seed_{axis}_fraction": (seed_voxel[i] + 0.5) / wm.shape[i]
        for i, axis in enumerate("xyz")
    }
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(treatment["rt_dose"])[2]))

    base: dict[str, Any] = {
        "gray_matter_pbmap": gm,
        "white_matter_pbmap": wm,
        "voxel_size_mm": voxel_size,
        "verbose": False,
        **solver_params,
        **seed_fractions,
    }
    resection_time = float(treatment["resection_time"])
    stopping_time = resection_time + float(base["time_after_resection"])
    n_steps = int(base["n_steps"])
    dt = stopping_time / n_steps
    print(f"run directory: {run_dir}")
    print(f"seed voxel {seed_voxel} (label {args.seed_label} CoM), slice z={z}, {n_steps} steps (dt={dt:.4g} d)")

    # Solve 1: untreated up to the resection (FKPPSolver, whose dynamics
    # StuppFKPPSolver reproduces up to that step).
    n_pre = int(round(resection_time / dt))
    wall = time.perf_counter()
    untreated = {key: value for key, value in base.items() if key != "time_after_resection"}
    pre = FKPPSolver({**untreated, "stopping_time": n_pre * dt, "n_steps": n_pre}).solve()
    if not pre.success:
        raise RuntimeError(f"pre-resection solve failed: {pre.error}")
    wall_pre = time.perf_counter() - wall
    print(f"pre-resection solve ({n_pre} steps): {wall_pre:.1f} s, mass {pre.final_stopping_quantity:.1f}")

    # Solve 2: the treated horizon with snapshots.
    wall = time.perf_counter()
    treated = StuppFKPPSolver(
        {**base, **treatment, "n_time_series_snapshots": args.n_snapshots}
    ).solve()
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
        np.asarray(treatment["rt_times"], dtype=np.float64),
        args.n_treatment_panels,
    )

    header = (
        f"D {base['white_matter_diffusivity']:g}, rho {base['rho']:g}, "
        f"ratio {base['diffusivity_ratio']:g}, "
        f"alpha {treatment['rt_alpha']:g}, beta {treatment['rt_beta']:g}, "
        f"kill {treatment['chemo_kill_rate']:g}, decay {treatment['chemo_decay_rate']:g}"
    )
    render(
        run_dir / "overview",
        header,
        panels,
        t1c,
        treatment["resection_cavity"],
        seed_voxel,
        z,
        args.threshold,
        times,
        masses,
        [(n_pre * dt, float(pre.final_state["cell_density"].sum()))],
        treatment,
    )
    summary = {
        "run_name": run_name,
        "cli_args": jsonable(vars(args)),
        "manifest": str(Path(args.manifest).resolve()),
        "tissue": {"wm": wm_path, "gm": gm_path},
        "t1c": str(t1c_path),
        "params": scalar_params({**base, **treatment}),
        "seed_voxel": list(seed_voxel),
        "slice_z": z,
        "n_steps": n_steps,
        "dt": dt,
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
