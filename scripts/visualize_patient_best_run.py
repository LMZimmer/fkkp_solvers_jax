#!/usr/bin/env python
"""Re-solve the best run of a scripts/patient_sensitivity_analysis.py sweep
and render it next to the patient's longitudinal tumour segmentations.

The best run is the qoi.csv row with the largest --criterion (default
dice_mean_core, the sweep's per-run metric: the mean core Dice over the
post-op sessions at the row's sampled thresholds). Its run config
(<run-root>/configs/<run>.json, the sweep's own; --run-root defaults to
the sweep's default output root joined with spec.json's name) is solved
once more with fisher_kpp_jax.StuppFKPPSolver, recording the state at the
patient's session moments exactly as the sweep did (the last step end at
least half a step before t_pre + offset, ``session_snapshot_days``) plus a
frame every --snapshot-interval-days days (default 2) that samples the
core volume curve; the sweep's saved run is not read.

Core threshold. --core-threshold logged (the default) takes the best
row's core_threshold column (sampled mode records it per row; in
profiled mode the column is absent and the script falls back to infer).
--core-threshold infer maximises the mean over all sessions of the Dice
between the segmented core and the thresholded field over the grid
0.05..0.95 step 0.01. A number fixes it. Masks follow the sweep: the
reference core is labels 1 (necrotic) and 3 (enhancing), the pre-op
segmentation relabelled 4 -> 3; the model core is field >= threshold;
the label-4 (cavity) voxels of a post-op session's own segmentation are
removed from both masks (``reference_masks``). The session frames are
rounded for storage as the sweep rounds its fields before the masks are
taken, so the per-session Dice and volumes equal the sweep's. The
volume curve between sessions has no segmentation to exclude with, so
the curve is the plain thresholded volume; the session markers on it
carry the exclusion, as the Dice does.

Two figures, the same layout: one panel per session on the axial slice
through the centre of mass of the pre-op reference core (or --slice-z),
the session's own T1c as the background (pre-op:
skull_stripped/t1c_skullstripped.nii.gz; later sessions:
longitudinal/t1c_warped_longitudinal.nii.gz, registered to the pre-op
space; all on the sweep's grid, checked), np.rot90 orientation, panels
in time order in rows of --columns (default 4), and a volume-vs-time
panel below with the treatment events marked (fisher_kpp_jax.util.
mark_treatment_events) and the session moments as ticks:
  segmentations   the session's labels overlaid (necrotic, edema,
                  enhancing, cavity, one colour each); below, the
                  reference core volume per session
  model           the recorded field overlaid (inferno, densities below
                  --display-threshold transparent) with the core
                  threshold as a white contour and the session's cavity
                  outlined; below, the thresholded core volume along the
                  run, the session values marked, the reference core
                  volumes as hollow markers, and the per-session Dice in
                  each panel
Written into <output-dir>/<run-name>/ (exist_ok=False, nothing outside
it): config.json, result.json, initial_cell_density.nii.gz and
final_cell_density.nii.gz (fisher_kpp_jax.Result.save), the session
frames <session>_cell_density.nii.gz, segmentations.pdf/.png,
model.pdf/.png and run_summary.json (the best row and its criterion
value, the threshold and how it was chosen, the slice, per session the
moment, recorded day, Dice, model and reference core volume, and the
snapshot days and volumes of the curve).

Run from the project root, e.g.:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/visualize_patient_best_run.py \\
      --sweep-dir results/sa_2026-09-14_SAILOR_sub-01 --output-dir runs/
  JAX_PLATFORMS=cpu python scripts/visualize_patient_best_run.py --sweep-dir <dir> \\
      --output-dir runs/ --core-threshold infer
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver, read_config  # noqa: E402
from fisher_kpp_jax.config import jsonable  # noqa: E402
from fisher_kpp_jax.util import CAVITY_COLOR, mark_treatment_events  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    LABEL_CAVITY,
    LABEL_EDEMA,
    LABEL_ENHANCING,
    LABEL_NECROTIC,
    LABEL_PREOP,
    PREOP_TIME_FACTOR,
    Reference,
    dice,
    load_segmentation,
    reference_masks,
    relabel_preop,
    run_snapshots,
    session_snapshot_days,
    snapshot_file,
)
from sensitivity_analysis import as_float, read_csv, read_json, round_field, write_json  # noqa: E402

DEFAULT_CRITERION = "dice_mean_core"
THRESHOLD_COLUMN = "core_threshold"
INFER_GRID: NDArray = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)
# Background image per session below <root>/<patient>/<session>/.
PREOP_T1C = Path("skull_stripped") / "t1c_skullstripped.nii.gz"
LATER_T1C = Path("longitudinal") / "t1c_warped_longitudinal.nii.gz"
LABEL_NAMES: dict[int, str] = {
    LABEL_NECROTIC: "necrotic",
    LABEL_EDEMA: "edema",
    LABEL_ENHANCING: "enhancing",
    LABEL_CAVITY: "cavity",
}
LABEL_COLORS: dict[int, tuple[float, float, float, float]] = {
    LABEL_NECROTIC: (0.95, 0.85, 0.20, 0.85),
    LABEL_EDEMA: (0.25, 0.65, 0.95, 0.65),
    LABEL_ENHANCING: (0.95, 0.35, 0.15, 0.85),
    LABEL_CAVITY: (*CAVITY_COLOR[:3], 0.75),
}
CORE_CONTOUR_COLOR = "white"
MODEL_COLOR = "black"
REFERENCE_COLOR = (0.85, 0.25, 0.10, 1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweep-dir", required=True, help="sweep directory holding spec.json, design.csv and qoi.csv"
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="sweep directory holding configs/<run>.json (default: the patient script's "
        "default output root joined with spec.json's name)",
    )
    parser.add_argument(
        "--criterion", default=DEFAULT_CRITERION, help=f"qoi.csv column maximised (default {DEFAULT_CRITERION})"
    )
    parser.add_argument(
        "--core-threshold",
        default="logged",
        help="'logged' (the best row's core_threshold; default), 'infer' (mean Dice over the sessions "
        "maximised on a grid) or a number",
    )
    parser.add_argument(
        "--snapshot-interval-days", type=float, default=2.0, help="spacing of the curve frames (default 2)"
    )
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name", default=None, help="run directory name (default: best_<run>_<UTC>_<pid>)"
    )
    parser.add_argument(
        "--patient-root", default=None, help="patient data root (default: spec.json's)"
    )
    parser.add_argument(
        "--slice-z", type=int, default=None, help="axial slice (default: pre-op core centre of mass)"
    )
    parser.add_argument(
        "--display-threshold", type=float, default=0.01, help="field below which the overlay is transparent"
    )
    parser.add_argument("--columns", type=int, default=4, help="panels per row (default 4)")
    return parser.parse_args(argv)


def best_row(qoi_rows: list[dict[str, str]], criterion: str) -> dict[str, str]:
    """The qoi.csv row with the largest finite criterion."""
    if criterion not in qoi_rows[0]:
        raise KeyError(f"qoi.csv has no column {criterion!r}.")
    values = np.asarray([as_float(row[criterion]) for row in qoi_rows], dtype=np.float64)
    if not np.isfinite(values).any():
        raise ValueError(f"no row has a finite {criterion}.")
    return qoi_rows[int(np.nanargmax(values))]


def logged_threshold(row: dict[str, str]) -> float | None:
    """The row's core threshold, None when the sweep did not record one."""
    if THRESHOLD_COLUMN not in row:
        return None
    value = as_float(row[THRESHOLD_COLUMN])
    return float(value) if np.isfinite(value) else None


def infer_threshold(fields: list[NDArray], references: list[Reference], grid: NDArray = INFER_GRID) -> float:
    """The grid threshold maximising the mean over the sessions of the
    core Dice (empty pairs, NaN, are left out of the mean)."""
    scores = []
    for threshold in grid:
        values = [
            dice((field >= threshold) & reference.valid, reference.core)
            for field, reference in zip(fields, references, strict=True)
        ]
        scores.append(np.nanmean(values) if np.isfinite(values).any() else -np.inf)
    return float(grid[int(np.argmax(scores))])


def session_background(root: Path, patient: str, session: dict[str, Any]) -> Path:
    """The session's T1c in the pre-op space."""
    relative = PREOP_T1C if session["label"] == LABEL_PREOP else LATER_T1C
    return root / patient / session["id"] / relative


def load_volume(path: Path, shape: tuple[int, ...]) -> NDArray:
    """A NIfTI volume as float64, checked against the run's grid."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    volume = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64)
    if volume.shape != shape:
        raise ValueError(f"{path}: shape {volume.shape} differs from the run's grid {shape}.")
    return volume


def recorded_frame(times: NDArray, frames: NDArray, day: float) -> tuple[float, NDArray]:
    """The frame recorded for a requested snapshot day (exact match, as the
    solver records the requested days)."""
    matches = np.flatnonzero(np.isclose(times, day, rtol=1e-9, atol=1e-9))
    if matches.size == 0:
        raise ValueError(f"the solver did not record the requested day {day!r}.")
    return float(times[int(matches[0])]), frames[int(matches[0])]


def session_title(session: dict[str, Any], moment: float) -> str:
    approx = "*" if session.get("approximate") else ""
    return f"{session['id']} {session['label']} {session['date']}{approx} (day {moment:.0f})"


def _figure(n_panels: int, n_col: int) -> tuple[Any, list[Any], Any]:
    """The shared layout: n_panels axes in rows of n_col plus one wide axis
    below for the volume curve."""
    n_col = max(1, min(n_col, n_panels))
    n_row = int(np.ceil(n_panels / n_col))
    fig = plt.figure(figsize=(4.4 * n_col + 0.6, 4.6 * n_row + 4.0), constrained_layout=True)
    grid = fig.add_gridspec(
        n_row + 1, n_col + 1, height_ratios=[1.0] * n_row + [0.9], width_ratios=[1.0] * n_col + [0.05]
    )
    axes = [fig.add_subplot(grid[k // n_col, k % n_col]) for k in range(n_panels)]
    bottom = fig.add_subplot(grid[n_row, :n_col])
    colorbar_ax = fig.add_subplot(grid[:n_row, n_col])
    return fig, axes, (bottom, colorbar_ax)


def _finish_curve(ax: Any, params: dict[str, Any], moments: list[float], stopping_time: float) -> None:
    mark_treatment_events(ax, params)
    for k, t in enumerate(moments):
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.9, label="scan" if k == 0 else None)
    ax.set_xlim(-0.01 * stopping_time, 1.01 * stopping_time)
    ax.set_xlabel("time [days]", fontsize=12)
    ax.set_ylabel("core volume [mL]", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=3)


def render_segmentations(
    outfile_stem: Path,
    header: str,
    sessions: list[dict[str, Any]],
    labels: list[NDArray],
    backgrounds: list[NDArray],
    z: int,
    moments: list[float],
    reference_volumes: list[float],
    params: dict[str, Any],
    stopping_time: float,
    n_col: int,
) -> None:
    fig, axes, (bottom, colorbar_ax) = _figure(len(sessions), n_col)
    colors = [(0, 0, 0, 0)] + [LABEL_COLORS[k] for k in sorted(LABEL_COLORS)]
    cmap = ListedColormap(colors)
    for ax, session, label_volume, background, moment in zip(
        axes, sessions, labels, backgrounds, moments, strict=True
    ):
        ax.imshow(np.rot90(background[:, :, z]), cmap="gray", interpolation="none")
        ax.imshow(
            np.rot90(label_volume[:, :, z]), cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5, interpolation="none"
        )
        ax.set_title(session_title(session, moment), fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")
    colorbar_ax.axis("off")
    colorbar_ax.legend(
        handles=[Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k]) for k in sorted(LABEL_COLORS)],
        loc="upper left", fontsize=10, frameon=False,
    )
    bottom.plot(
        moments, reference_volumes, "o-", color=REFERENCE_COLOR, markersize=6, label="segmented core"
    )
    _finish_curve(bottom, params, moments, stopping_time)
    fig.suptitle(header, horizontalalignment="left", x=0.02, fontsize=14, fontweight="bold")
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)


def render_model(
    outfile_stem: Path,
    header: str,
    sessions: list[dict[str, Any]],
    fields: list[NDArray],
    references: list[Reference],
    backgrounds: list[NDArray],
    z: int,
    threshold: float,
    display_threshold: float,
    moments: list[float],
    dices: list[float],
    model_volumes: list[float],
    reference_volumes: list[float],
    curve_times: NDArray,
    curve_volumes: NDArray,
    params: dict[str, Any],
    stopping_time: float,
    n_col: int,
) -> None:
    fig, axes, (bottom, colorbar_ax) = _figure(len(sessions), n_col)
    image = None
    for ax, session, field, reference, background, moment, value in zip(
        axes, sessions, fields, references, backgrounds, moments, dices, strict=True
    ):
        ax.imshow(np.rot90(background[:, :, z]), cmap="gray", interpolation="none")
        field_slice = np.rot90(field[:, :, z])
        image = ax.imshow(
            np.ma.masked_less(field_slice, display_threshold), cmap="inferno", alpha=0.85,
            vmin=0.0, vmax=1.0, interpolation="none",
        )
        core = np.rot90(((field >= threshold) & reference.valid)[:, :, z])
        if core.any():
            ax.contour(core.astype(float), levels=[0.5], colors=[CORE_CONTOUR_COLOR], linewidths=1.0)
        cavity = np.rot90((~reference.valid)[:, :, z])
        if cavity.any():
            ax.contour(cavity.astype(float), levels=[0.5], colors=[CAVITY_COLOR], linewidths=1.2)
        ax.set_title(session_title(session, moment), fontsize=12, fontweight="bold", pad=8)
        ax.text(
            0.02, 0.02, f"Dice core {value:.2f}", transform=ax.transAxes, color="white", fontsize=10, va="bottom"
        )
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, cax=colorbar_ax, label="cell density")
    bottom.plot(curve_times, curve_volumes, "-", color=MODEL_COLOR, linewidth=1.2, label=f"model core (u >= {threshold:g})")
    bottom.plot(moments, model_volumes, "s", color=MODEL_COLOR, markersize=6, label="model core at scan (cavity excluded)")
    bottom.plot(
        moments, reference_volumes, "o", markerfacecolor="none", markeredgecolor=REFERENCE_COLOR,
        markersize=7, markeredgewidth=1.5, label="segmented core",
    )
    _finish_curve(bottom, params, moments, stopping_time)
    fig.suptitle(header, horizontalalignment="left", x=0.02, fontsize=14, fontweight="bold")
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_dir = Path(args.sweep_dir)
    spec = read_json(sweep_dir / "spec.json")
    run_root = Path(args.run_root) if args.run_root else DEFAULT_OUTPUT_DIR / spec["name"]
    row = best_row(read_csv(sweep_dir / "qoi.csv"), args.criterion)
    run_name_sweep = row["run_name"]
    design_record = next(r for r in read_csv(sweep_dir / "design.csv") if r["row_name"] == row["row_name"])
    config_path = run_root / "configs" / f"{run_name_sweep}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"run config {config_path} not found (pass --run-root).")
    criterion_value = as_float(row[args.criterion])
    print(f"best row {row['row_name']} (run {run_name_sweep}): {args.criterion} = {criterion_value:.4f}")

    run_name = args.run_name or (
        f"best_{run_name_sweep}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # The sweep's snapshot rule for the session moments, plus the curve frames.
    config = read_config(config_path, solver=StuppFKPPSolver)
    probe = StuppFKPPSolver(config)
    n_steps, dt = probe.resolve_time_stepping()
    del probe
    moments = run_snapshots(spec, design_record)
    session_days = session_snapshot_days(moments, dt)
    stopping_time = float(config["resection_time"]) + float(config["time_after_resection"])
    curve_days = np.arange(0.0, stopping_time, args.snapshot_interval_days)
    config["snapshot_times"] = sorted({*session_days.values(), *curve_days.tolist(), stopping_time})
    solver = StuppFKPPSolver(config)
    params = solver.params
    print(f"run directory: {run_dir}")
    print(f"{n_steps} steps (dt={dt:.4g} d), {len(config['snapshot_times'])} snapshots requested")
    result = solver.solve(store_result=True, outdir=run_dir)
    if not result.success:
        raise RuntimeError(f"solve failed: {result.error}")
    times = np.asarray(result.snapshot_times, dtype=np.float64)
    frames = result.time_series["cell_density"]
    affine = np.eye(4) if result.affine is None else np.asarray(result.affine, dtype=np.float64)
    shape = tuple(int(n) for n in frames.shape[1:])
    voxel_volume_ml = float(np.prod(np.asarray(params["voxel_size_mm"], dtype=np.float64))) / 1000.0
    print(f"solve ({result.n_steps} steps, {times.size} frames): {result.wall_time_s:.1f} s")

    # Per session: the recorded frame (saved as the sweep saves it), the
    # reference masks, the display labels and the background.
    sessions = spec["patient"]["sessions"]
    patient_root = Path(args.patient_root) if args.patient_root else Path(spec["patient"]["root"])
    patient = spec["patient"]["patient"]
    fields: list[NDArray] = []
    recorded_days: list[float] = []
    references: list[Reference] = []
    labels: list[NDArray] = []
    backgrounds: list[NDArray] = []
    for session in sessions:
        sid = session["id"]
        day, frame = recorded_frame(times, frames, session_days[sid])
        # Rounded for storage as the sweep stores its fields, so the Dice
        # and the volumes below equal the sweep's, which it read off the
        # saved fields.
        frame = round_field(frame)
        image = nib.Nifti1Image(frame, affine)
        image.set_data_dtype(np.float32)
        nib.save(image, str(run_dir / snapshot_file(sid)))
        fields.append(np.asarray(frame, dtype=np.float64))
        recorded_days.append(day)
        segmentation, _ = load_segmentation(session["segmentation"])
        if segmentation.shape != shape:
            raise ValueError(f"{session['segmentation']}: shape {segmentation.shape} differs from the run's grid {shape}.")
        preop = session["label"] == LABEL_PREOP
        references.append(reference_masks(segmentation, preop, sid))
        labels.append(relabel_preop(segmentation) if preop else segmentation)
        backgrounds.append(load_volume(session_background(patient_root, patient, session), shape))

    if args.core_threshold == "logged":
        threshold = logged_threshold(row)
        threshold_source = "logged"
        if threshold is None:
            print("qoi.csv records no core_threshold for the row; inferring it.")
            threshold, threshold_source = infer_threshold(fields, references), "inferred"
    elif args.core_threshold == "infer":
        threshold, threshold_source = infer_threshold(fields, references), "inferred"
    else:
        threshold, threshold_source = float(args.core_threshold), "given"
    print(f"core threshold {threshold:g} ({threshold_source})")

    dices = [dice((f >= threshold) & r.valid, r.core) for f, r in zip(fields, references, strict=True)]
    model_volumes = [
        float(((f >= threshold) & r.valid).sum()) * voxel_volume_ml for f, r in zip(fields, references, strict=True)
    ]
    reference_volumes = [float(r.core.sum()) * voxel_volume_ml for r in references]
    curve_volumes = np.asarray([(frame >= threshold).sum() for frame in frames], dtype=np.float64) * voxel_volume_ml
    preop_core = references[0].core
    if not preop_core.any():
        raise ValueError("the pre-op reference core is empty; pass --slice-z.")
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(preop_core)[2]))
    session_moments = [float(moments[s["id"]]) for s in sessions]
    for session, value, mv, rv in zip(sessions, dices, model_volumes, reference_volumes, strict=True):
        print(f"  {session['id']}: Dice core {value:.3f}, model {mv:.2f} mL, segmented {rv:.2f} mL")

    header_run = (
        f"{patient}, {row['row_name']} ({args.criterion} {criterion_value:.3f}), core threshold {threshold:g} "
        f"({threshold_source}), slice z={z}"
    )
    header_params = (
        f"D {params['white_matter_diffusivity']:.3g} mm^2/d, rho {params['rho']:.3g} /d, "
        f"ratio {params['diffusivity_ratio']:.3g}, t_pre {as_float(design_record[PREOP_TIME_FACTOR]):.1f} d, "
        f"alpha {params['rt_alpha']:.3g} /Gy, a/b {params['rt_alpha_beta_ratio']:.3g} Gy, "
        f"kill {params['chemo_kill_rate']:.3g} /(mg/m^2)"
    )
    header = f"{header_run}\n{header_params}"
    render_segmentations(
        run_dir / "segmentations", header, sessions, labels, backgrounds, z, session_moments,
        reference_volumes, params, stopping_time, args.columns,
    )
    render_model(
        run_dir / "model", header, sessions, fields, references, backgrounds, z, threshold,
        args.display_threshold, session_moments, dices, model_volumes, reference_volumes, times,
        curve_volumes, params, stopping_time, args.columns,
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "run_name": run_name,
            "cli_args": jsonable(vars(args)),
            "sweep_dir": str(sweep_dir.resolve()),
            "run_config": str(config_path),
            "best_row": row["row_name"],
            "best_run": run_name_sweep,
            "criterion": args.criterion,
            "criterion_value": criterion_value,
            "core_threshold": threshold,
            "core_threshold_source": threshold_source,
            "slice_z": z,
            "voxel_volume_ml": voxel_volume_ml,
            "dt": result.dt,
            "n_steps": result.n_steps,
            "stopping_time": stopping_time,
            "sessions": [
                {
                    "id": s["id"],
                    "label": s["label"],
                    "date": s["date"],
                    "moment": m,
                    "recorded_day": d,
                    "dice_core": v,
                    "model_core_volume_ml": mv,
                    "reference_core_volume_ml": rv,
                    "background": str(session_background(patient_root, patient, s)),
                    "file": snapshot_file(s["id"]),
                }
                for s, m, d, v, mv, rv in zip(
                    sessions, session_moments, recorded_days, dices, model_volumes, reference_volumes, strict=True
                )
            ],
            "curve_days": times.tolist(),
            "curve_core_volume_ml": curve_volumes.tolist(),
        },
    )
    print(f"saved {run_dir / 'segmentations.pdf'} and {run_dir / 'model.pdf'} (+ .png, run_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
