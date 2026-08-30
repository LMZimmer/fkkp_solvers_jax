#!/usr/bin/env python
"""Forward-solve one treated tumor growth with fisher_kpp_jax.StuppFKPPSolver.

Loads the patient's WM/GM tissue probability maps, merges the treatment
parameters of a JSON manifest (see scripts/stupp_manifest_example.json and
fisher_kpp_jax.solvers.treatment_params_from_manifest: resection cavity from a
labelled segmentation, chemotherapy session times/rates, radiotherapy fraction
times and TOTAL-dose map with the linear-quadratic alpha/beta), runs the
solver and writes, into its own run directory

  <output-dir>/<run-name>/
    final_cell_density.nii.gz  the final tumor cell density, affine and header
                               of the wm pbmap (float32 voxels)
    run_config.json            scalar params, CLI args, manifest path, final
                               time / stopping criterion / stopping quantity,
                               wall time
    manifest.json              verbatim copy of the manifest (provenance)
    overview.png               unless --no-plot: axial slices through the seed
                               voxel at each snapshot plus total mass vs time
                               with the treatment events marked

The run directory is created with exist_ok=False and nothing is written
outside it, so many evaluations can run in parallel against the same
output-dir (the default run name carries a UTC timestamp and the pid).
Snapshot volumes are only kept in memory for the plot; with --no-plot no
time series is requested at all (memory-light for parallel sweeps).

All solver defaults below (rho, diffusivity, resolution, horizon, seed
position) are PLACEHOLDERS for the SAILOR subject -- review before use.

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

from fisher_kpp_jax import StuppFKPPSolver  # noqa: E402
from fisher_kpp_jax.solvers import treatment_params_from_manifest  # noqa: E402

_SAILOR_TISSUE = "/mnt/Drive4/lucas/SAILOR/processed/sub-01/ses-01/tissue_segmentation"
DEFAULT_WM = f"{_SAILOR_TISSUE}/wm_pbmap.nii.gz"
DEFAULT_GM = f"{_SAILOR_TISSUE}/gm_pbmap.nii.gz"
DEFAULT_MANIFEST = str(_ROOT / "scripts" / "stupp_manifest_example.json")

# Placeholder solver defaults -- see the module docstring.
DEFAULT_RHO = 0.1  # 1/day
DEFAULT_DIFFUSIVITY = 0.5  # white matter diffusivity, mm^2/day
DEFAULT_RESOLUTION_FACTOR = 0.5
DEFAULT_STOPPING_TIME = 100.0  # days
DEFAULT_SEED_FRACTION = 0.5  # of the grid extent, per axis
DEFAULT_N_SNAPSHOTS = 6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wm", default=DEFAULT_WM, help="white matter pbmap NIfTI")
    parser.add_argument("--gm", default=DEFAULT_GM, help="gray matter pbmap NIfTI")
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST, help="treatment manifest JSON"
    )
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name",
        default=None,
        help="run directory name (default: stupp_<UTC timestamp>_<pid>)",
    )
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="proliferation rate, 1/day")
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=DEFAULT_DIFFUSIVITY,
        help="white matter diffusivity, mm^2/day",
    )
    parser.add_argument(
        "--resolution-factor", type=float, default=DEFAULT_RESOLUTION_FACTOR
    )
    parser.add_argument(
        "--stopping-time", type=float, default=DEFAULT_STOPPING_TIME, help="days"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="explicit step count (default: the solver's stability formula)",
    )
    parser.add_argument("--precision", choices=("f32", "f64"), default="f32")
    parser.add_argument("--seed-x", type=float, default=DEFAULT_SEED_FRACTION, help="seed x fraction")
    parser.add_argument("--seed-y", type=float, default=DEFAULT_SEED_FRACTION, help="seed y fraction")
    parser.add_argument("--seed-z", type=float, default=DEFAULT_SEED_FRACTION, help="seed z fraction")
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=DEFAULT_N_SNAPSHOTS,
        help="time-series snapshots for overview.png",
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


def snapshot_times(solver: StuppFKPPSolver, n_snapshots: int) -> np.ndarray:
    """Simulation times of the recorded frames (each holds the state after
    its scheduled step, at (step + 1) * dt); see operators._run_time_loop
    for the schedule."""
    n_steps, dt = solver._resolve_time_stepping()
    steps = np.unique(np.linspace(0, n_steps - 1, n_snapshots, dtype=np.int64))
    return (steps + 1) * dt


def render_overview(
    path: Path,
    frames: np.ndarray,
    times: np.ndarray,
    seed_voxel: tuple[int, int, int],
    voxel_volume: float,
    treatment: dict[str, Any],
    resolution_factor: float,
) -> None:
    """Axial-slice montage through the seed voxel at each snapshot (shared
    color scale) plus total mass vs time with the treatment events marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = frames.shape[0]
    k = seed_voxel[2]
    vmax = max(float(frames.max()), 1e-12)
    n_cols = n_frames
    fig, axes = plt.subplots(
        2, n_cols, figsize=(2.6 * n_cols, 6.0), gridspec_kw={"height_ratios": [1.0, 0.8]}
    )
    axes = np.atleast_2d(axes)
    for column in range(n_cols):
        ax = axes[0, column]
        image = ax.imshow(
            frames[column, :, :, k].T, origin="lower", vmin=0.0, vmax=vmax, cmap="magma"
        )
        ax.set_title(f"t = {times[column]:.1f} d", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label="cell density")

    # Mass panel spans the bottom row.
    for ax in axes[1, :]:
        ax.remove()
    ax = fig.add_subplot(2, 1, 2)
    masses = frames.sum(axis=(1, 2, 3)) * voxel_volume
    ax.plot(times, masses, marker="o", color="black", label="total mass (snapshots)")
    events = [
        ("resection_time", "resection", "tab:red"),
        ("chemo_times", "chemotherapy", "tab:green"),
        ("rt_times", "radiotherapy", "tab:blue"),
    ]
    for key, label, color in events:
        if key not in treatment:
            continue
        for index, t in enumerate(np.atleast_1d(treatment[key])):
            ax.axvline(
                float(t), color=color, alpha=0.5, linewidth=0.8,
                label=label if index == 0 else None,
            )
    ax.set_xlabel("time [days]")
    ax.set_ylabel("total mass [density * mm^3]")
    ax.set_title(
        f"axial slice z={k} through the seed voxel; resolution factor "
        f"{resolution_factor:g}",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or default_run_name()
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    wm_img = nib.load(args.wm)
    gm_img = nib.load(args.gm)
    wm = np.asarray(wm_img.get_fdata(), dtype=np.float64)
    gm = np.asarray(gm_img.get_fdata(), dtype=np.float64)
    voxel_size = tuple(float(v) for v in wm_img.header.get_zooms()[:3])

    treatment = treatment_params_from_manifest(args.manifest)
    shutil.copyfile(args.manifest, run_dir / "manifest.json")

    n_snapshots = None if args.no_plot else int(args.n_snapshots)
    params: dict[str, Any] = {
        "white_matter_diffusivity": args.diffusivity,
        "rho": args.rho,
        "gray_matter_pbmap": gm,
        "white_matter_pbmap": wm,
        "gaussian_seed_x_fraction": args.seed_x,
        "gaussian_seed_y_fraction": args.seed_y,
        "gaussian_seed_z_fraction": args.seed_z,
        "resolution_factor": args.resolution_factor,
        "stopping_time": args.stopping_time,
        "n_steps": args.n_steps,
        "precision": args.precision,
        "voxel_size_mm": voxel_size,
        "n_time_series_snapshots": n_snapshots,
        "verbose": True,
        **treatment,
    }
    solver = StuppFKPPSolver(params)

    print(f"run directory: {run_dir}")
    print(f"grid: {wm.shape}, voxel size {voxel_size} mm, treatments: {sorted(treatment)}")
    wall_start = time.perf_counter()
    result = solver.solve()
    wall = time.perf_counter() - wall_start

    config: dict[str, Any] = {
        "run_name": run_name,
        "cli_args": {key: jsonable(value) for key, value in vars(args).items()},
        "manifest": str(Path(args.manifest).resolve()),
        "params": scalar_params(solver.params),
        "grid_shape": list(wm.shape),
        "success": result.success,
        "error": result.error,
        "final_time": result.final_time,
        "stopping_criterion": result.stopping_criterion,
        "final_stopping_quantity": result.final_stopping_quantity,
        "wall_time_s": wall,
    }
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
        times = snapshot_times(solver, n_snapshots)[: frames.shape[0]]
        seed_voxel = tuple(
            int(fraction * n)
            for fraction, n in zip((args.seed_x, args.seed_y, args.seed_z), wm.shape)
        )
        render_overview(
            run_dir / "overview.png",
            frames,
            times,
            seed_voxel,
            float(np.prod(voxel_size)),
            treatment,
            args.resolution_factor,
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
