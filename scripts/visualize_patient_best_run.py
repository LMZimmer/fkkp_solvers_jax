#!/usr/bin/env python
"""Re-solve the best run of a scripts/patient_sensitivity_analysis.py sweep
or of a scripts/patient_cmaes_fit.py fit and render it next to the
patient's longitudinal tumour segmentations.

Sweep (--sweep-dir). The best run is the qoi.csv row with the largest
--criterion (default dice_mean_core, the sweep's per-run metric: the mean
core Dice over the post-op sessions at the row's sampled thresholds). Its
run config (<run-root>/configs/<run>.json, the sweep's own; --run-root
defaults to the sweep's default output root joined with spec.json's name)
is solved once more with fisher_kpp_jax.StuppFKPPSolver, recording the
state at the patient's session moments exactly as the sweep did (the last
step end at least half a step before t_pre + offset,
``session_snapshot_days``), with --volume-curve plus a frame every
--snapshot-interval-days days (default 2) that samples the core volume
curve; the sweep's saved run is not read.

Fit (--fit-dir). The best run is the fit's re-solved best evaluation
(best/config.json, written by the fit's resolve). It is solved once more
as the fit saved it: its schedule, its fixed n_steps and the session
moments and snapshot days of the config's "_fit" record, over the
sessions the fit simulated (plus the curve frames with --volume-curve); the patient and the
sessions come from the fit's spec.json. The criterion is the re-solve's
J (best/objective.json). --run-root, --criterion and --schedule are
sweep options.

Core threshold. --core-threshold logged (the default) takes the best
row's core_threshold column (sampled mode records it per row; in
profiled mode the column is absent and the script falls back to infer);
for a fit it is the re-solve's profiled core threshold, so the
per-session Dice equals the fit's dice_core_star (printed next to it and
recorded as fit_dice_core) when the fit was scored with the current
masks (its spec.json's label_conventions equal FIT_LABEL_CONVENTIONS;
a note is printed otherwise). A fit's masks are
patient_cmaes_fit.fit_session_references: the ones below with the
resection cavity (the post-op session's label 4) also removed in every
later session.
--core-threshold infer maximises the mean over all sessions of the Dice
between the segmented core and the thresholded field over the grid
0.05..0.95 step 0.01. A number fixes it. Masks follow the sweep
(``session_references``): the reference core is labels 1 (necrotic) and
3 (enhancing) in the pre-op session (a pre-op segmentation with label 4
is refused) and label 3 (enhancing) alone in every post-op session
(since 2026-10-01); the model
core is field >= threshold; in a post-op session a necrotic voxel that
an earlier post-op session labelled cavity counts as cavity, and the
session's label-4 (cavity) voxels are then removed from both masks. The
segmentation overlay of the panels shows the raw labels; the Dice, the
volumes, the cavity outline and the threshold inference use the
corrected masks. The session frames are
rounded for storage as the sweep rounds its fields before the masks are
taken, so the per-session Dice and volumes equal the sweep's. The
volume curve (only with --volume-curve) masks every frame with the valid mask (the corrected
cavity excluded) of the most recent session recorded at or before it,
nothing before the first session, so at each session it equals the
session's model core volume and in between it carries the last scan's
mask forward.

Schedule. --schedule current (the default) rebuilds the clinical
timeline from spec.json's sessions and protocol doses with the patient
script's schedule constants as they are now (``build_timeline``) and
puts its model days at the row's preop_time into the config
(resection_time, time_after_resection, rt_times, chemo_times,
chemo_doses), so a run of an older sweep is re-solved under the current
protocol definition (sweeps before 2026-09-17 used 14-day adjuvant
cycles, the script 28-day ones since); the differences to the sweep's
config are printed and recorded. --schedule sweep solves the config as
the sweep saved it. Under the current schedule the per-session Dice and
volumes are those of the new solve, not the sweep's logged values.

Two figures, the same layout: one panel per session on the axial slice
through the centre of mass of the pre-op reference core (or --slice-z),
the session's own T1c as the background (pre-op:
skull_stripped/t1c_skullstripped.nii.gz; later sessions:
longitudinal/t1c_warped_longitudinal.nii.gz, registered to the pre-op
space; all on the sweep's grid, checked), np.rot90 orientation, panels
in time order in rows of --columns (default 4), each titled
"<session> <label> (day <n>)" with the clinical day relative to the
post-op scan (day 0; the pre-op scan negative; from spec.json's dates),
and a volume-vs-time panel below in model days (the seed is day 0, the
pre-op scan at the row's preop_time) with the treatment events marked
(fisher_kpp_jax.util.mark_treatment_events); no figure title (the run,
its parameters and the threshold are in run_summary.json and
config.json):
  segmentations   the session's labels overlaid in the palette of
                  PredictGBM's scripts/visualize_respond_10.py (necrosis
                  orange, edema blue, enhancing violet, cavity green;
                  alpha 0.6 as there); below, the reference core volume
                  per session, each point labelled with its session id
  model           the recorded field overlaid (inferno at alpha 0.75 as
                  PredictGBM's prediction overlay, densities below
                  --display-threshold transparent) with the session's
                  segmented tumour core (labels 1 + 3 of the display
                  labels) as a red contour and, in every post-op and
                  follow-up panel, the resection cavity (the post-op
                  session's label 4, the one the solver removed) outlined
                  in the palette's green, both named in a legend above
                  the panels ("observed core", "resection cavity"); the
                  Dice's full exclusion (the session's own cavity and the
                  earlier sessions' cavities as well) is not drawn;
                  below, the model core
                  volume per session
                  (cavity excluded) as hollow red circles, with --volume-curve
                  the thresholded core volume along the run (latest
                  scan's mask) as a dashed red line through them (the
                  resection line is solid red), the reference
                  core volumes as filled black squares, and the
                  per-session Dice in each panel
Written into <output-dir>/<run-name>/ (exist_ok=False, nothing outside
it): config.json, result.json, initial_cell_density.nii.gz and
final_cell_density.nii.gz (fisher_kpp_jax.Result.save), the session
frames <session>_cell_density.nii.gz, segmentations.pdf/.png,
model.pdf/.png and run_summary.json (the best row, or the fit's best
restart and evaluation, and its criterion
value, the threshold and how it was chosen, the slice, per session the
model moment, post-op day, recorded day, Dice, model and reference core
volume, excluded cavity and relabelled voxel counts, and with
--volume-curve the snapshot days and volumes of the curve).

Run from the project root, e.g.:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/visualize_patient_best_run.py \\
      --sweep-dir results/sa_2026-09-14_SAILOR_sub-01 --output-dir runs/
  JAX_PLATFORMS=cpu python scripts/visualize_patient_best_run.py --sweep-dir <dir> \\
      --output-dir runs/ --core-threshold infer
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/visualize_patient_best_run.py \\
      --fit-dir /mnt/Drive4/lucas/stupp_patient_fit/fit_sub01_2026-09-27 --output-dir runs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
# numpy asks for transparent hugepages on large arrays; on a host with
# fragmented memory every page fault of the multi-GB frame stack then stalls
# in direct compaction (2026-09-28: np.stack of 37 frames ran > 4 min in
# kernel time, 12 s without). Read by numpy at import.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver, read_config  # noqa: E402
from fisher_kpp_jax.config import jsonable  # noqa: E402
from fisher_kpp_jax.util import mark_treatment_events  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    LABEL_CAVITY,
    LABEL_EDEMA,
    LABEL_ENHANCING,
    LABEL_NECROTIC,
    LABEL_POSTOP,
    LABEL_PREOP,
    PREOP_TIME_FACTOR,
    Protocol,
    Reference,
    Session,
    build_timeline,
    dice,
    check_preop_labels,
    load_segmentation,
    run_snapshots,
    session_references,
    session_snapshot_days,
    snapshot_file,
)
from patient_cmaes_fit import fit_session_references, same_label_conventions  # noqa: E402
from sensitivity_analysis import as_float, read_csv, read_json, round_field, write_json  # noqa: E402

DEFAULT_CRITERION = "dice_mean_core"
# The legend entry of the segmented core volumes (the SA's core rule).
REFERENCE_CORE_LABEL = "segmented core (pre-op: necrotic + enhancing; post-op: enhancing)"
THRESHOLD_COLUMN = "core_threshold"
# A fit directory's re-solved best run (scripts/patient_cmaes_fit.py resolve).
FIT_BEST_DIR = "best"
FIT_OBJECTIVE_FILE = "objective.json"
FIT_CRITERION = "J"
INFER_GRID: NDArray = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)
# Background image per session below <root>/<patient>/<session>/.
PREOP_T1C = Path("skull_stripped") / "t1c_skullstripped.nii.gz"
LATER_T1C = Path("longitudinal") / "t1c_warped_longitudinal.nii.gz"
# Segmentation palette, legend and overlay alphas of PredictGBM's
# scripts/visualize_respond_10.py (SEG_COLORS, SEG_LABELS and its imshow
# calls), so the figures read like that project's patient plots.
LABEL_NAMES: dict[int, str] = {
    LABEL_NECROTIC: "Necrosis / non-enhancing",
    LABEL_EDEMA: "Edema",
    LABEL_ENHANCING: "Enhancing tumor",
    LABEL_CAVITY: "Resection cavity",
}
LABEL_COLORS: dict[int, tuple[float, float, float, float]] = {
    LABEL_NECROTIC: (1.0, 127 / 255, 0.0, 1.0),
    LABEL_EDEMA: (30 / 255, 144 / 255, 1.0, 1.0),
    LABEL_ENHANCING: (138 / 255, 43 / 255, 226 / 255, 1.0),
    LABEL_CAVITY: (34 / 255, 139 / 255, 34 / 255, 1.0),
}
SEGMENTATION_ALPHA = 0.6
FIELD_ALPHA = 0.75
CAVITY_OUTLINE_COLOR = LABEL_COLORS[LABEL_CAVITY]
# model.pdf: the session's segmented tumour core (necrotic + enhancing,
# raw display labels) outlined in red, the corrected cavity in green.
OBSERVED_CORE_LABELS = (LABEL_NECROTIC, LABEL_ENHANCING)
OBSERVED_CORE_CONTOUR_COLOR = "red"
CONTOUR_LEGEND: dict[str, tuple[Any, float]] = {
    "observed core": (OBSERVED_CORE_CONTOUR_COLOR, 1.2),
    "resection cavity": (CAVITY_OUTLINE_COLOR, 1.2),
}
# segmentations.pdf: the segmented core volumes in red.  model.pdf: the
# model in red (hollow circles, the curve), the segmented core volumes as
# filled black squares.
REFERENCE_COLOR = (0.85, 0.25, 0.10, 1.0)
MODEL_COLOR = REFERENCE_COLOR
OBSERVED_COLOR = "black"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sweep-dir", default=None, help="sweep directory holding spec.json, design.csv and qoi.csv"
    )
    source.add_argument(
        "--fit-dir",
        default=None,
        help="scripts/patient_cmaes_fit.py fit directory holding spec.json and the re-solved best/",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="sweep directory holding configs/<run>.json (default: the patient script's "
        "default output root joined with spec.json's name; --sweep-dir only)",
    )
    parser.add_argument(
        "--criterion",
        default=DEFAULT_CRITERION,
        help=f"qoi.csv column maximised (default {DEFAULT_CRITERION}; --sweep-dir only)",
    )
    parser.add_argument(
        "--core-threshold",
        default="logged",
        help="'logged' (the best row's core_threshold, or the fit's profiled core threshold; default), "
        "'infer' (mean Dice over the sessions maximised on a grid) or a number",
    )
    parser.add_argument(
        "--schedule",
        choices=("current", "sweep"),
        default=None,
        help="'current' (default) rebuilds the treatment schedule with the patient script's present "
        "protocol constants; 'sweep' solves the config as the sweep saved it (--sweep-dir only; a fit's "
        "config is solved with its own schedule)",
    )
    parser.add_argument(
        "--volume-curve",
        action="store_true",
        help="draw the model core volume along the run as a line on the model figure's volume panel",
    )
    parser.add_argument(
        "--snapshot-interval-days",
        type=float,
        default=2.0,
        help="spacing of the curve frames (default 2; --volume-curve only)",
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
    args = parser.parse_args(argv)
    if args.fit_dir is not None:
        if args.run_root is not None or args.schedule is not None or args.criterion != DEFAULT_CRITERION:
            parser.error("--run-root, --criterion and --schedule apply to --sweep-dir only.")
    elif args.schedule is None:
        args.schedule = "current"
    return args


SCHEDULE_KEYS = ("resection_time", "time_after_resection", "rt_times", "chemo_times", "chemo_doses")


def current_schedule(config: dict[str, Any], spec: dict[str, Any], preop_time: float) -> dict[str, Any]:
    """
    Replace the config's schedule (SCHEDULE_KEYS) with the timeline of
    spec.json's sessions and protocol doses under the patient script's
    present schedule constants, at preop_time.

    Returns:
        The record of the change: the protocol used, the sweep's own
        cycle length, the keys whose values changed, and the chemo
        session count and total dose before and after.
    """
    sessions = [
        Session(s["id"], s["label"], date.fromisoformat(s["date"]), bool(s.get("approximate", False)))
        for s in spec["patient"]["sessions"]
    ]
    doses = spec["protocol"]
    protocol = Protocol(
        concomitant_dose=float(doses["concomitant_dose_mg_m2"]),
        adjuvant_first_dose=float(doses["adjuvant_first_cycle_dose_mg_m2"]),
        adjuvant_later_dose=float(doses["adjuvant_later_cycles_dose_mg_m2"]),
        base_cycle_days=doses.get("base_config_cycle_days"),
    )
    days = build_timeline(sessions, protocol).model_days(preop_time)
    before = {key: config[key] for key in SCHEDULE_KEYS}
    changed = [key for key in SCHEDULE_KEYS if not np.array_equal(np.asarray(before[key], dtype=np.float64), np.asarray(days[key], dtype=np.float64))]
    for key in SCHEDULE_KEYS:
        config[key] = days[key]
    return {
        "source": "current",
        "protocol": protocol.record(),
        "sweep_adjuvant_cycle_days": doses.get("adjuvant_cycle_days"),
        "changed_keys": changed,
        "chemo_sessions": {"sweep": len(before["chemo_times"]), "current": len(days["chemo_times"])},
        "chemo_total_dose_mg_m2": {"sweep": float(np.sum(before["chemo_doses"])), "current": float(np.sum(days["chemo_doses"]))},
        "chemo_times": {"sweep": list(before["chemo_times"]), "current": list(days["chemo_times"])},
    }


def best_row(qoi_rows: list[dict[str, str]], criterion: str) -> dict[str, str]:
    """The qoi.csv row with the largest finite criterion."""
    if criterion not in qoi_rows[0]:
        raise KeyError(f"qoi.csv has no column {criterion!r}.")
    values = np.asarray([as_float(row[criterion]) for row in qoi_rows], dtype=np.float64)
    if not np.isfinite(values).any():
        raise ValueError(f"no row has a finite {criterion}.")
    return qoi_rows[int(np.nanargmax(values))]


def fit_best_run(fit_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """
    The re-solved best run of a fit directory: the path of its config, the
    config's "_fit" record (schedule-independent moments, snapshot days,
    n_steps, dt) and best/objective.json (restart, eval_id and the
    "resolved" objective).
    """
    config_path = fit_dir / FIT_BEST_DIR / "config.json"
    objective_path = fit_dir / FIT_BEST_DIR / FIT_OBJECTIVE_FILE
    for path in (config_path, objective_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found (the fit's resolve writes it).")
    fit = read_json(config_path).get("_fit")
    if fit is None:
        raise ValueError(f"{config_path} has no \"_fit\" record.")
    objective = read_json(objective_path)
    if not objective["resolved"].get("success"):
        raise ValueError(f"{objective_path}: the re-solve failed ({objective['resolved'].get('error')}).")
    return config_path, fit, objective


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


def postop_days(sessions: list[dict[str, Any]]) -> list[int]:
    """The clinical day of every session relative to the post-op scan
    (label postop, day 0; the pre-op scan negative), from spec.json's dates."""
    postops = [s for s in sessions if s["label"] == LABEL_POSTOP]
    if len(postops) != 1:
        raise ValueError(f"expected one session labelled {LABEL_POSTOP}, got {[s['id'] for s in postops]}.")
    origin = date.fromisoformat(postops[0]["date"])
    return [(date.fromisoformat(s["date"]) - origin).days for s in sessions]


def session_title(session: dict[str, Any], postop_day: int) -> str:
    return f"{session['id']} {session['label']} (day {postop_day})"


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


def _finish_curve(ax: Any, params: dict[str, Any], stopping_time: float) -> None:
    mark_treatment_events(ax, params)
    ax.set_xlim(-0.01 * stopping_time, 1.01 * stopping_time)
    ax.set_xlabel("time [days]", fontsize=12)
    ax.set_ylabel("core volume [mL]", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=3)


def render_segmentations(
    outfile_stem: Path,
    sessions: list[dict[str, Any]],
    labels: list[NDArray],
    backgrounds: list[NDArray],
    z: int,
    moments: list[float],
    days: list[int],
    reference_volumes: list[float],
    params: dict[str, Any],
    stopping_time: float,
    n_col: int,
) -> None:
    fig, axes, (bottom, colorbar_ax) = _figure(len(sessions), n_col)
    colors = [(0, 0, 0, 0)] + [LABEL_COLORS[k] for k in sorted(LABEL_COLORS)]
    cmap = ListedColormap(colors)
    for ax, session, label_volume, background, day in zip(
        axes, sessions, labels, backgrounds, days, strict=True
    ):
        ax.imshow(np.rot90(background[:, :, z]), cmap="gray", interpolation="none")
        ax.imshow(
            np.rot90(label_volume[:, :, z]), cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5,
            alpha=SEGMENTATION_ALPHA, interpolation="none",
        )
        ax.set_title(session_title(session, day), fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")
    colorbar_ax.axis("off")
    colorbar_ax.legend(
        handles=[Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k]) for k in sorted(LABEL_COLORS)],
        loc="upper left", fontsize=10, frameon=False,
    )
    bottom.plot(
        moments, reference_volumes, "o-", color=REFERENCE_COLOR, markersize=6, label=REFERENCE_CORE_LABEL
    )
    for session, moment, volume in zip(sessions, moments, reference_volumes, strict=True):
        bottom.annotate(
            session["id"], (moment, volume), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=REFERENCE_COLOR,
        )
    low, high = bottom.get_ylim()
    bottom.set_ylim(low, high + 0.08 * (high - low))  # room for the label above the highest point
    _finish_curve(bottom, params, stopping_time)
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)


def render_model(
    outfile_stem: Path,
    sessions: list[dict[str, Any]],
    fields: list[NDArray],
    resection_cavity: NDArray,
    labels: list[NDArray],
    backgrounds: list[NDArray],
    z: int,
    threshold: float,
    display_threshold: float,
    moments: list[float],
    days: list[int],
    dices: list[float],
    model_volumes: list[float],
    reference_volumes: list[float],
    curve_times: NDArray | None,
    curve_volumes: NDArray | None,
    params: dict[str, Any],
    stopping_time: float,
    n_col: int,
) -> None:
    """The model figure: per session the segmented tumour core of the
    display labels (necrotic + enhancing, red) and, in every session but
    the pre-op one, the resection cavity (the post-op session's label-4
    mask, green) over the field, the two contours named in a frameless
    legend above the panels; the volume curve is drawn only when given."""
    fig, axes, (bottom, colorbar_ax) = _figure(len(sessions), n_col)
    fig.legend(
        handles=[
            Line2D([], [], color=color, linewidth=width, label=label)
            for label, (color, width) in CONTOUR_LEGEND.items()
        ],
        loc="outside upper center", ncol=len(CONTOUR_LEGEND), fontsize=10, frameon=False,
    )
    image = None
    cavity = np.rot90(np.asarray(resection_cavity, dtype=bool)[:, :, z])
    for ax, session, field, label_volume, background, day, value in zip(
        axes, sessions, fields, labels, backgrounds, days, dices, strict=True
    ):
        ax.imshow(np.rot90(background[:, :, z]), cmap="gray", interpolation="none")
        field_slice = np.rot90(field[:, :, z])
        image = ax.imshow(
            np.ma.masked_less(field_slice, display_threshold), cmap="inferno", alpha=FIELD_ALPHA,
            vmin=0.0, vmax=1.0, interpolation="none",
        )
        observed_core = np.rot90(np.isin(label_volume, OBSERVED_CORE_LABELS)[:, :, z])
        if observed_core.any():
            ax.contour(
                observed_core.astype(float), levels=[0.5], colors=[OBSERVED_CORE_CONTOUR_COLOR], linewidths=1.2
            )
        if session["label"] != LABEL_PREOP and cavity.any():
            ax.contour(cavity.astype(float), levels=[0.5], colors=[CAVITY_OUTLINE_COLOR], linewidths=1.2)
        ax.set_title(session_title(session, day), fontsize=12, fontweight="bold", pad=8)
        ax.text(
            0.02, 0.02, f"Dice core {value:.2f}", transform=ax.transAxes, color="white", fontsize=10, va="bottom"
        )
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, cax=colorbar_ax, label="cell density")
    if curve_times is not None and curve_volumes is not None:
        bottom.plot(
            curve_times, curve_volumes, "--", color=MODEL_COLOR, linewidth=1.2,  # dashed: the resection line is solid red
            label=f"model core (u >= {threshold:g}, latest scan's cavity excluded)",
        )
    bottom.plot(
        moments, model_volumes, "o", markerfacecolor="none", markeredgecolor=MODEL_COLOR,
        markersize=7, markeredgewidth=1.5, label="model core at scan (cavity excluded)",
    )
    bottom.plot(moments, reference_volumes, "s", color=OBSERVED_COLOR, markersize=6, label=REFERENCE_CORE_LABEL)
    _finish_curve(bottom, params, stopping_time)
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fit_logged_dice: dict[str, float] = {}
    if args.fit_dir is not None:
        fit_dir = Path(args.fit_dir)
        spec = read_json(fit_dir / "spec.json")
        solver_mode = spec.get("solver", {}).get("mode", "standard")
        if solver_mode != "standard":
            raise NotImplementedError(
                f"the fit {fit_dir} was run with --solver {solver_mode}; this script renders the cell_density frames "
                "of the isotropic solver only."
            )
        scope = spec.get("objective", {}).get("scope", "full")
        if scope != "full":
            raise NotImplementedError(
                f"the fit {fit_dir} was run with --objective {scope}; this script renders the treated timeline "
                "(resection, fractions, chemotherapy) of a full-series fit only."
            )
        config_path, fit, objective = fit_best_run(fit_dir)
        resolved = objective["resolved"]
        criterion, criterion_value = FIT_CRITERION, float(resolved[FIT_CRITERION])
        recorded_threshold = resolved.get("core_threshold_star")
        recorded_threshold = None if recorded_threshold is None else float(recorded_threshold)
        fit_logged_dice = {sid: float(v["dice_core_star"]) for sid, v in resolved["per_session"].items()}
        default_name = f"best_{spec['name']}_restart{objective['restart']}_eval{objective['eval_id']}"
        origin: dict[str, Any] = {
            "fit_dir": str(fit_dir.resolve()),
            "best_restart": objective["restart"],
            "best_eval_id": objective["eval_id"],
            "fit_loss": float(resolved["loss"]),
        }
        print(
            f"best of fit {spec['name']}: restart {objective['restart']}, evaluation {objective['eval_id']}: "
            f"{criterion} = {criterion_value:.4f}"
        )
    else:
        sweep_dir = Path(args.sweep_dir)
        spec = read_json(sweep_dir / "spec.json")
        run_root = Path(args.run_root) if args.run_root else DEFAULT_OUTPUT_DIR / spec["name"]
        row = best_row(read_csv(sweep_dir / "qoi.csv"), args.criterion)
        run_name_sweep = row["run_name"]
        design_record = next(r for r in read_csv(sweep_dir / "design.csv") if r["row_name"] == row["row_name"])
        config_path = run_root / "configs" / f"{run_name_sweep}.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"run config {config_path} not found (pass --run-root).")
        criterion, criterion_value = args.criterion, as_float(row[args.criterion])
        recorded_threshold = logged_threshold(row)
        default_name = f"best_{run_name_sweep}"
        origin = {"sweep_dir": str(sweep_dir.resolve()), "best_row": row["row_name"], "best_run": run_name_sweep}
        print(f"best row {row['row_name']} (run {run_name_sweep}): {criterion} = {criterion_value:.4f}")

    run_name = args.run_name or (
        f"{default_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # The sweep's snapshot rule for the session moments (a fit records its
    # own), plus the curve frames.
    config = read_config(config_path, solver=StuppFKPPSolver)
    if args.fit_dir is not None:
        schedule: dict[str, Any] = {"source": "fit"}
        n_steps, dt = int(fit["n_steps"]), float(fit["dt"])
        moments = {sid: float(moment) for sid, moment in fit["snapshot_moments"].items()}
        session_days = {sid: float(day) for sid, day in fit["snapshot_days"].items()}
        sessions = [s for s in spec["patient"]["sessions"] if s["id"] in session_days]
        if len(sessions) != len(session_days):
            raise ValueError(f"spec.json lacks sessions of the fit's snapshot days {sorted(session_days)}.")
    elif args.schedule == "current":
        schedule = current_schedule(config, spec, as_float(design_record[PREOP_TIME_FACTOR]))
        print(
            f"schedule rebuilt with the current protocol (adjuvant cycle {schedule['protocol']['adjuvant_cycle_days']} d; "
            f"the sweep used {schedule['sweep_adjuvant_cycle_days']} d): changed {schedule['changed_keys'] or 'nothing'}; "
            f"chemo sessions {schedule['chemo_sessions']['sweep']} -> {schedule['chemo_sessions']['current']}, "
            f"total dose {schedule['chemo_total_dose_mg_m2']['sweep']:g} -> {schedule['chemo_total_dose_mg_m2']['current']:g} mg/m^2"
        )
    else:
        schedule = {"source": "sweep", "sweep_adjuvant_cycle_days": spec["protocol"].get("adjuvant_cycle_days")}
    if args.fit_dir is None:
        probe = StuppFKPPSolver(config)
        n_steps, dt = probe.resolve_time_stepping()
        del probe
        moments = run_snapshots(spec, design_record)
        session_days = session_snapshot_days(moments, dt)
        sessions = spec["patient"]["sessions"]
    stopping_time = float(config["resection_time"]) + float(config["time_after_resection"])
    if args.volume_curve:
        curve_days = np.arange(0.0, stopping_time, args.snapshot_interval_days)
        config["snapshot_times"] = sorted({*session_days.values(), *curve_days.tolist(), stopping_time})
    else:
        config["snapshot_times"] = sorted(set(session_days.values()))
    solver = StuppFKPPSolver(config)
    params = solver.params
    print(f"run directory: {run_dir}")
    print(f"{n_steps} steps (dt={dt:.4g} d), {len(config['snapshot_times'])} snapshots requested")
    result = solver.solve(store_result=True, outdir=run_dir)
    if not result.success:
        raise RuntimeError(f"solve failed: {result.error}")
    if result.n_steps != n_steps:
        raise RuntimeError(f"the solver used n_steps={result.n_steps}, not the planned {n_steps}.")
    times = np.asarray(result.snapshot_times, dtype=np.float64)
    frames = result.time_series["cell_density"]
    affine = np.eye(4) if result.affine is None else np.asarray(result.affine, dtype=np.float64)
    shape = tuple(int(n) for n in frames.shape[1:])
    voxel_volume_ml = float(np.prod(np.asarray(params["voxel_size_mm"], dtype=np.float64))) / 1000.0
    print(f"solve ({result.n_steps} steps, {times.size} frames): {result.wall_time_s:.1f} s")

    # Per session: the recorded frame (saved as the sweep saves it), the
    # reference masks, the display labels and the background.
    patient_root = Path(args.patient_root) if args.patient_root else Path(spec["patient"]["root"])
    patient = spec["patient"]["patient"]
    fields: list[NDArray] = []
    recorded_days: list[float] = []
    segmentations: list[NDArray] = []
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
        segmentations.append(segmentation)
        labels.append(check_preop_labels(segmentation, session["segmentation"]) if preop else segmentation)
        backgrounds.append(load_volume(session_background(patient_root, patient, session), shape))
    # The sweep's reference masks: the later sessions corrected with the
    # earlier sessions' cavities before their own cavity is excluded; a
    # fit's also exclude the resection cavity in every later session. The
    # overlay (labels) stays the raw segmentation.
    # The resection cavity drawn in the model figure: the post-op
    # session's label-4 voxels (what the solver removed).
    postop_index = next(i for i, s in enumerate(sessions) if s["label"] == LABEL_POSTOP)
    resection_cavity = segmentations[postop_index] == LABEL_CAVITY
    if args.fit_dir is not None:
        cavity_spec = config["resection_cavity"]
        fit_cavity = load_segmentation(cavity_spec["segmentation"])[0] == int(cavity_spec["label"])
        if not np.array_equal(fit_cavity, resection_cavity):
            print(
                f"note: the fit's resection cavity ({cavity_spec['segmentation']}, label {cavity_spec['label']}) differs "
                f"from the post-op session's label {LABEL_CAVITY}; the masks use the fit's, the figure outlines the latter"
            )
        references: list[Reference] = fit_session_references(segmentations, sessions, fit_cavity)
        if not same_label_conventions(spec.get("label_conventions")):
            print(
                "note: the fit was scored with older reference masks (its spec.json's label_conventions differ from "
                "FIT_LABEL_CONVENTIONS), so the Dice below differ from the fit's"
            )
    else:
        references = session_references(segmentations, sessions)
    for reference in references:
        if reference.n_relabelled:
            print(f"  {reference.session}: {reference.n_relabelled} necrotic voxels counted as cavity (earlier cavities)")

    if args.core_threshold == "logged":
        threshold = recorded_threshold
        threshold_source = "logged"
        if threshold is None:
            print("the best run records no core threshold; inferring it.")
            threshold, threshold_source = infer_threshold(fields, references), "inferred"
    elif args.core_threshold == "infer":
        threshold, threshold_source = infer_threshold(fields, references), "inferred"
    else:
        threshold, threshold_source = float(args.core_threshold), "given"
    print(f"core threshold {threshold:g} ({threshold_source}); reference core: necrotic + enhancing pre-op, enhancing post-op")

    dices = [dice((f >= threshold) & r.valid, r.core) for f, r in zip(fields, references, strict=True)]
    model_volumes = [
        float(((f >= threshold) & r.valid).sum()) * voxel_volume_ml for f, r in zip(fields, references, strict=True)
    ]
    reference_volumes = [float(r.core.sum()) * voxel_volume_ml for r in references]
    curve_times: NDArray | None = None
    curve_volumes: NDArray | None = None
    if args.volume_curve:
        # Each curve frame is masked as the most recent session at or before
        # it masks its own frame (nothing before the first session), so the
        # curve passes through the session markers.
        session_frame_days = np.asarray(recorded_days, dtype=np.float64)
        curve_times = times
        curve_volumes = np.empty(times.size, dtype=np.float64)
        for i, (day, frame) in enumerate(zip(times, frames, strict=True)):
            core = frame >= threshold
            earlier = np.flatnonzero(session_frame_days <= day)
            if earlier.size:
                core &= references[int(earlier[np.argmax(session_frame_days[earlier])])].valid
            curve_volumes[i] = float(core.sum()) * voxel_volume_ml
    preop_core = references[0].core
    if not preop_core.any():
        raise ValueError("the pre-op reference core is empty; pass --slice-z.")
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(preop_core)[2]))
    session_moments = [float(moments[s["id"]]) for s in sessions]
    session_postop_days = postop_days(sessions)
    for session, value, mv, rv in zip(sessions, dices, model_volumes, reference_volumes, strict=True):
        logged = fit_logged_dice.get(session["id"])
        fit_note = "" if logged is None else f" (fit {logged:.3f})"
        print(f"  {session['id']}: Dice core {value:.3f}{fit_note}, model {mv:.2f} mL, segmented {rv:.2f} mL")

    render_segmentations(
        run_dir / "segmentations", sessions, labels, backgrounds, z, session_moments, session_postop_days,
        reference_volumes, params, stopping_time, args.columns,
    )
    render_model(
        run_dir / "model", sessions, fields, resection_cavity, labels, backgrounds, z, threshold,
        args.display_threshold, session_moments, session_postop_days, dices, model_volumes, reference_volumes,
        curve_times, curve_volumes, params, stopping_time, args.columns,
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "run_name": run_name,
            "cli_args": jsonable(vars(args)),
            **origin,
            "run_config": str(config_path),
            "criterion": criterion,
            "criterion_value": criterion_value,
            "core_threshold": threshold,
            "core_threshold_source": threshold_source,
            "schedule": schedule,
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
                    "postop_day": pd,
                    "recorded_day": d,
                    "dice_core": v,
                    "model_core_volume_ml": mv,
                    "reference_core_volume_ml": rv,
                    "n_cavity_excluded": r.n_cavity,
                    "n_necrotic_relabelled_cavity": r.n_relabelled,
                    "background": str(session_background(patient_root, patient, s)),
                    "file": snapshot_file(s["id"]),
                    **({"fit_dice_core": fit_logged_dice[s["id"]]} if s["id"] in fit_logged_dice else {}),
                }
                for s, m, pd, d, v, mv, rv, r in zip(
                    sessions, session_moments, session_postop_days, recorded_days, dices, model_volumes,
                    reference_volumes, references, strict=True,
                )
            ],
            **(
                {"curve_days": curve_times.tolist(), "curve_core_volume_ml": curve_volumes.tolist()}
                if curve_times is not None and curve_volumes is not None
                else {}
            ),
        },
    )
    print(f"saved {run_dir / 'segmentations.pdf'} and {run_dir / 'model.pdf'} (+ .png, run_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
