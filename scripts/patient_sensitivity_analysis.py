#!/usr/bin/env python
"""Variance-based (Sobol') sensitivity analysis of the Stupp-protocol forward
model, fisher_kpp_jax.StuppFKPPSolver, on ONE real patient of the SAILOR
cohort, with quantities of interest (QoIs) measuring the agreement of the
forward run with the patient's longitudinal tumour segmentations.

The machinery is that of scripts/sensitivity_analysis.py (the atlas
analysis) and is imported from it: the Saltelli design on a scrambled
Sobol' sequence, the search-space files with their derived factor groups,
the nested-range seed projection, the run bookkeeping, SALib's Sobol'
estimators with bootstrap confidence intervals and the half-sample check,
the run/block accounting, the figures. What differs is below.

Patient data (--patient, --patient-root, --session-labels, --sessions;
defaults sub-01 of /mnt/Drive4/lucas/SAILOR/processed and
/mnt/Drive4/lucas/SAILOR/session_labels.tsv, sessions ses-02 to ses-08, DEFAULT_SESSIONS).
Everything is in the patient's pre-op space at 1 mm, preprocessed:
  tissue:   <root>/<patient>/<pre-op session>/tissue_segmentation/{gm,wm}_pbmap.nii.gz
  pre-op:   <root>/<patient>/<pre-op session>/tumor_segmentation/tumor_seg.nii.gz
  later:    <root>/<patient>/<session>/longitudinal/recurrence_preop.nii.gz
  dose:     <root>/<patient>/<CRT start session>/longitudinal/dose_warped_longitudinal.nii.gz
  sessions: the tsv (columns subject, session, label, date; a date with a
            trailing '*' is an approximate one, recorded as such)
The pre-op session is the one labelled preop, the post-op one the one
labelled postop, the CRT start session the first follow-up after it; the
later sessions analysed are --sessions (which must hold the post-op
session, the source of the resection cavity) and must all be after the
pre-op scan. Labels of every tumour segmentation: 1 necrotic, 2 edema,
3 enhancing, 4 cavity; the pre-op segmentation is relabelled 4 -> 3
before use. The design step reads the tsv, loads every segmentation,
prints the label counts per session, the dose map's range and nonzero
volume and checks the shapes and affines against the pre-op
segmentation's (nothing is resampled; a mismatch is an error) before
sampling anything (``check_patient_data``; the record goes into
spec.json under data_checks).

Clinical schedule (fixed): resection RESECTION_DAYS_BEFORE_POSTOP (3)
days before the post-op scan date; chemoradiotherapy (CRT) from the CRT
start date, N_FRACTIONS (30) fractions on calendar weekdays (Mon-Fri)
and CONCOMITANT_DAYS (42) daily temozolomide (TMZ) days from that date,
with the concomitant dose of the base config's protocol
(``protocol_from_config``: 75 mg/m^2); adjuvant TMZ from
ADJUVANT_DELAY_DAYS (28) days after the last CRT day (the later of the
last fraction and the last concomitant day), ADJUVANT_DAYS_ON (5) days
on then 9 off in ADJUVANT_CYCLE_DAYS (14) cycles with the base config's
per-cycle doses (150 mg/m^2 in cycle 1, 200 afterwards), as many cycles
as start before the last session date, a cycle truncated at the end of
the run. NOTE: the base config's own protocol uses 28-day adjuvant
cycles; the 14-day cycle is this analysis's definition, and the
concomitant and adjuvant dose values are the only things taken from
the config's schedule. The design step prints the calendar and the
cycle table (``build_timeline``, ``print_timeline``).

Timeline (model days). Day 0 is the seed; the pre-op scan is at
t_pre = preop_time, a sampled factor (30-200 days) in the role of the
atlas's resection_time. Every later event is at t_pre + (calendar date
- pre-op date) in days: the resection, the fractions, the TMZ days, the
adjuvant days and one snapshot per session; the run ends at the last
session date. The run configs hold resection_time = t_pre + resection
offset, time_after_resection = last session offset - resection offset,
the shifted rt_times, chemo_times (with chemo_doses) and
  resection_cavity = {"segmentation": <post-op recurrence_preop.nii.gz>, "label": 4}
  rt_dose = <dose map>  (the solver's TOTAL dose over all fractions: the
            per-fraction dose is rt_dose / len(rt_times); the design step
            prints the map's maximum and warns when it is not a plausible
            total of about 60 Gy, RT_TOTAL_DOSE_PLAUSIBLE_GY)
  white_matter_pbmap / gray_matter_pbmap = the patient's maps
and everything else from the base config (--config, default
fisher_kpp_jax/configs/StuppFKPPSolver.json: steps_per_day 12 among
it; the design step refuses a config without a time-step entry as the
atlas script does). Snapshots: the session snapshot is the state at the
start of the scan day, before any event of that day fires: run-one
resolves the run's time step dt (``StuppFKPPSolver.resolve_time_stepping``
of the run config) and requests the last step end at least half a step
before the moment t_pre + offset, m dt = floor((t_pre + offset) / dt - 1/2) dt
(``session_snapshot_days``, the atlas's ``snapshot_days`` rule applied to
the moment itself instead of the end of the day), which the solver's
snapshot_times mechanism records exactly; both the moment and the
recorded day go into runs/<run>/timeline.json and the run's config.json
holds the requested days as snapshot_times, so a re-solve reproduces
the frames. The frames are saved as runs/<run>/<session>_cell_density.nii.gz
(e.g. ses03_cell_density.nii.gz; float32, rounded for storage as the
atlas fields are, ``round_field``) next to Result.save's
final_cell_density.nii.gz (the state at the horizon). One solve per run,
no growth stage: the cavity and the dose map are the patient's.

Seeding: the seedable voxels are the pre-op core (labels 1 and 3 after
the relabelling) carrying tissue (wm + gm >= min_tissue_fraction, the
solver's flux threshold as resolved from the base config), and the three
gaussian_seed_{x,y,z}_fraction factors are mapped onto them in nested
ranges exactly as the atlas script maps them onto the atlas tissue
(``patient_seed_geometry`` builds the SeedGeometry of that mask,
``project_seeds`` does the mapping).

Search space (--search-space, default
fisher_kpp_jax/search_spaces/sailor_patient_search_space.json), read as
the atlas script reads its files (``load_search_space``) with three
script factors that are not solver parameters: preop_time and the two
threshold factors core_threshold and edema_threshold_ratio, which carry
"cheap": true (the solve does not depend on them; the flag is stripped
before the file is handed to ``load_search_space``). Its ranges follow
fisher_kpp_jax/search_spaces/stupp_fkpp_sigma_v2_search_space.json
(2026-09-15): with front_width_mm up to 4 mm and seed_sigma_mm from
5 mm the atlas loader warns that the seed width floor is below twice the
front width cap, which is expected (a seed diffusion flattens shows as
an empty model mask, counted per session in qoi_summary.json). The
default design is --log2-n DEFAULT_LOG2_N (10, N = 1024). The timeline
entries (resection_time, time_after_resection, chemo_times,
chemo_doses, rt_times, snapshot_times), the tissue maps,
resection_cavity and rt_dose may not appear in it. The derived groups'
design-time checks run against the resolved parameters of a
StuppFKPPSolver built from the base config with the patient's volumes
and the timeline at the midpoint of preop_time (no solve).

Threshold modes (--threshold-mode, THRESHOLD_MODES, default sampled,
DEFAULT_THRESHOLD_MODE):
  profiled  the cheap factors are dropped from the design (k = 12 with
            the shipped search space). Per session and region the Dice
            is evaluated on the fixed grid THRESHOLD_GRID_CORE
            (0.30..0.85 step 0.05) x THRESHOLD_GRID_EDEMA (0.10..0.60
            step 0.05), pairs with edema < core only
            (``threshold_pairs``), and the QoIs report Dice* and the
            thresholds of the best pair (below) as well as the Dice at
            the fixed pair TAU_CORE / TAU_EDEMA = 0.6 / 0.3 (the row
            thresholds of this mode).
  sampled   (the default) the two cheap factors are in the design
            (k = 14 with the shipped search space), with
            edema_threshold = edema_threshold_ratio * core_threshold
            (a deterministic map, no rejection; edema < core on every
            row). The run list is DEDUPLICATED on the non-cheap columns
            (``dedup_runs``): rows sharing every solver parameter (the
            A_B^(i) rows of the cheap columns, whose dynamics equal
            their block's A row) share one run directory. design.csv
            keeps one line per Saltelli row (row_name) with a run_name
            column pointing at the shared run; spec.json records the
            mapping counts (n_rows = N (k + 2), n_runs = N (k_dyn + 2))
            and the design step prints them. The row thresholds of this
            mode are the sampled pair, and the QoIs of the fixed pair
            0.6 / 0.3 are computed as well under the suffix _fixed.
Both modes analyse their QoIs through the atlas's ``analyze_response``
(bootstrap, half-sample bookkeeping); the analysed list is explicit
(``analysed_qois``).

QoIs, per session s (prefixed by the session id, e.g. ses03_) and region
r in {core, whole}, with u the session's snapshot field, dV the voxel
volume (mm^3) and, in the post-op sessions, every label-4 voxel of THAT
session's segmentation excluded from the model and the reference masks
(``load_reference``; the pre-op session excludes nothing):
  model core  = u >= core_threshold;  reference core  = labels {1, 3}
  model whole = u >= edema_threshold; reference whole = labels {1, 2, 3}
  dice_<r>           Dice of the row thresholds; NaN when both masks are
                     empty (counted), 0 when one is (``dice``)
  log10_V_<r>        log10(V_model + dV), the atlas's volume convention
  log_vol_ratio_<r>  log10_V_<r> - log10(V_reference + dV) (NaN when the
                     reference is empty, counted)
  msd_<r>, hd95_<r>  the mean and the 95th percentile of the symmetric
                     surface distances in mm (``surface_distances``:
                     distances of the surface voxels of each mask to the
                     other mask's surface, scipy distance transforms,
                     both directions pooled); NaN when either mask is
                     empty
  dice_star_<r>, core_threshold_star, edema_threshold_star
                     the profiled thresholds: the grid pair maximising
                     dice_core + dice_whole (``profiled_qois``; where
                     the constraint edema < core does not bind these are
                     the per-region maxima and their argmax thresholds;
                     a region whose Dice is NaN on the whole grid, the
                     model and the reference both empty at every
                     threshold, is left out of the objective and its
                     Dice* is NaN)
  <name>_fixed       in sampled mode, the five threshold QoIs per
                     region at the fixed pair 0.6 / 0.3
  mass, log10_mass, centroid_drift, R_g, anisotropy, log10_anisotropy,
  wm_fraction        the atlas's threshold-free QoIs of the field
                     (``compute_qois`` with the patient's white-matter
                     map and the projected seed voxel)
and per run dice_mean_core, the mean of the finite dice_core over the
post-op sessions. Every Dice / volume / distance is computed on the
bounding box of the reference whole mask and the field at or above the
smallest threshold in use (exact: every mask lies inside it).

Output layout (--output-dir, default DEFAULT_OUTPUT_DIR; --name required):
  <output-dir>/<name>/
    search_space.json  copy of the search-space file
    base_config.json   the base config with the patient's volumes and the
                       timeline at the midpoint of preop_time (the
                       config the design-time checks ran on)
    spec.json          patient, sessions (ids, labels, dates, files), the
                       label conventions, data_checks, the protocol
                       doses, the timeline (calendar dates and offsets of
                       the resection, the fractions, the TMZ days, the
                       adjuvant cycles, the snapshots, the horizon),
                       threshold_mode, thresholds (fixed pair, grid,
                       factors), cheap_factors, dedup counts, the search
                       space record of the atlas script (factors,
                       overrides, derived groups, ...), N, k, k_dyn,
                       factor order, seedable voxels, chemo budget, the
                       dose map's maximum, analysed_qois
    design.csv         one line per Saltelli row: row_name, run_name (the
                       shared run), index, row, matrix, u_<factor> and
                       <factor> per factor, the derived parameters and
                       extras of the groups, seed_voxel_i/j/k and, in
                       sampled mode, edema_threshold
    configs/<run>.json one config per distinct run
    logs/<run>.log     stdout/stderr of the runs
    runs/<run>/        Result.save's output without the initial state
                       (config.json, result.json, final_cell_density.nii.gz),
                       the session snapshots <session>_cell_density.nii.gz
                       and timeline.json (dt, the snapshot moments and the
                       recorded days)
    run_status.csv     appended as runs finish (STATUS_COLUMNS)
    qoi.csv            one line per Saltelli row (QOI columns)
    qoi_summary.json   run counts, NaN counts per session, the per-QoI
                       run/block accounting
    sobol.csv, sobol_summary.json, figures/   as the atlas script writes them

Run from the project root, e.g.:
  python scripts/patient_sensitivity_analysis.py design --name sa_sub01 --log2-n 3 --threshold-mode profiled
  python scripts/patient_sensitivity_analysis.py run --sweep-dir <output-dir>/sa_sub01 --gpus 2,3
  python scripts/patient_sensitivity_analysis.py qoi --sweep-dir <output-dir>/sa_sub01
  python scripts/patient_sensitivity_analysis.py analyze --sweep-dir <output-dir>/sa_sub01
  python scripts/patient_sensitivity_analysis.py all --name sa_sub01 --gpus 2,3
The run step is resumable (runs whose result.json reports success are
skipped, partial run directories are redone). Nothing is written outside
<output-dir>/<name>/.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver, read_config, write_config  # noqa: E402
from sensitivity_analysis import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIG,
    DEFAULT_DESIGN_SEED,
    DEFAULT_GPUS,
    DEFAULT_JOBS_PER_GPU,
    DEFAULT_N_BOOTSTRAP,
    DERIVED_VOLUME_KEYS,
    MASS_FLOOR_VOXELS,
    SOBOL_COLUMNS,
    SOLVER_KEY,
    TAU_CORE,
    TAU_EDEMA,
    TIME_STEP_KEYS,
    SearchSpace,
    SeedGeometry,
    _csv_cell,
    _is_success,
    _load_field,
    _load_wm,
    accounting_line,
    analyze_response,
    as_float,
    block_size,
    compute_qois,
    design_table,
    load_search_space,
    make_figures,
    matrix_labels,
    mean_run_wall_time,
    qoi_summary_entry,
    qoi_values,
    read_csv,
    read_json,
    response_accounting,
    round_field,
    run_is_done,
    run_records,
    saltelli_design,
    seed_geometry,
    snapshot_days,
    sobol_rows,
    write_csv,
    write_json,
)

SOLVER_NAME = "StuppFKPPSolver"
DEFAULT_SEARCH_SPACE = _ROOT / "fisher_kpp_jax" / "search_spaces" / "sailor_patient_search_space.json"
DEFAULT_OUTPUT_DIR = Path("/mnt/Drive4/lucas/stupp_sensitivity_analysis_patient")
DEFAULT_PATIENT = "sub-01"
DEFAULT_PATIENT_ROOT = Path("/mnt/Drive4/lucas/SAILOR/processed")
DEFAULT_SESSION_LABELS = Path("/mnt/Drive4/lucas/SAILOR/session_labels.tsv")
DEFAULT_SESSIONS = "02-08"  # the later sessions analysed (``parse_sessions``)
# N = 1024 base points: 16 384 rows and 14 336 distinct solves in sampled mode
# with the shipped search space (the atlas script defaults to 11).
DEFAULT_LOG2_N = 10
DEFAULT_QOI_WORKERS = min(16, os.cpu_count() or 1)

# Patient file layout below <root>/<patient>/<session>/.
TISSUE_FILES: dict[str, str] = {
    "white_matter_pbmap": "tissue_segmentation/wm_pbmap.nii.gz",
    "gray_matter_pbmap": "tissue_segmentation/gm_pbmap.nii.gz",
}
PREOP_SEGMENTATION_FILE = "tumor_segmentation/tumor_seg.nii.gz"
LATER_SEGMENTATION_FILE = "longitudinal/recurrence_preop.nii.gz"
DOSE_FILE = "longitudinal/dose_warped_longitudinal.nii.gz"
SESSION_COLUMNS: tuple[str, ...] = ("subject", "session", "label", "date")
LABEL_PREOP, LABEL_POSTOP, LABEL_FOLLOWUP = "preop", "postop", "followup"

# Segmentation labels (all tumour segmentations) and the compartments.
LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING, LABEL_CAVITY = 1, 2, 3, 4
KNOWN_LABELS: tuple[int, ...] = (0, LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING, LABEL_CAVITY)
CORE_LABELS: tuple[int, ...] = (LABEL_NECROTIC, LABEL_ENHANCING)
WHOLE_LABELS: tuple[int, ...] = (LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING)
PREOP_RELABEL: dict[int, int] = {LABEL_CAVITY: LABEL_ENHANCING}
LABEL_CONVENTIONS: dict[str, Any] = {
    "labels": {"1": "necrotic", "2": "edema", "3": "enhancing", "4": "cavity"},
    "reference_core": list(CORE_LABELS),
    "reference_whole": list(WHOLE_LABELS),
    "preop_relabel": {str(k): v for k, v in PREOP_RELABEL.items()},
    "postop_exclusion": f"label {LABEL_CAVITY} voxels of the session's own segmentation are removed from the model and the reference masks",
}

# The clinical schedule.
RESECTION_DAYS_BEFORE_POSTOP = 3
CRT_WEEKS = 6
FRACTIONS_PER_WEEK = 5
N_FRACTIONS = CRT_WEEKS * FRACTIONS_PER_WEEK  # 30, on calendar weekdays
CONCOMITANT_DAYS = 7 * CRT_WEEKS  # 42 daily TMZ days
ADJUVANT_DELAY_DAYS = 28  # after the last CRT day
ADJUVANT_DAYS_ON = 5
ADJUVANT_CYCLE_DAYS = 14  # 5 on / 9 off (the base config's protocol has 28)
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# The dose map's maximum should be the total prescribed dose, about 60 Gy.
RT_TOTAL_DOSE_PLAUSIBLE_GY: tuple[float, float] = (50.0, 70.0)

# Script factors: not solver parameters.
PREOP_TIME_FACTOR = "preop_time"
CORE_THRESHOLD_FACTOR = "core_threshold"
EDEMA_RATIO_FACTOR = "edema_threshold_ratio"
EDEMA_THRESHOLD_COLUMN = "edema_threshold"
THRESHOLD_FACTORS: tuple[str, ...] = (CORE_THRESHOLD_FACTOR, EDEMA_RATIO_FACTOR)
SCRIPT_FACTORS: tuple[str, ...] = (PREOP_TIME_FACTOR, *THRESHOLD_FACTORS)
CHEAP_KEY = "cheap"
# Config entries the script sets per run; a search space may not hold them.
TIMELINE_KEYS: tuple[str, ...] = (
    "resection_time",
    "time_after_resection",
    "chemo_times",
    "chemo_doses",
    "rt_times",
    "snapshot_times",
)
PATIENT_VOLUME_KEYS: tuple[str, ...] = (*TISSUE_FILES, *DERIVED_VOLUME_KEYS)

THRESHOLD_MODES: tuple[str, ...] = ("profiled", "sampled")
DEFAULT_THRESHOLD_MODE = "sampled"
FIXED_THRESHOLDS: tuple[float, float] = (TAU_CORE, TAU_EDEMA)  # 0.6 / 0.3
THRESHOLD_GRID_CORE: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(12))  # 0.30..0.85
THRESHOLD_GRID_EDEMA: tuple[float, ...] = tuple(round(0.10 + 0.05 * i, 2) for i in range(11))  # 0.10..0.60

# Run directory files besides Result.save's.
SNAPSHOT_SUFFIX = "_cell_density.nii.gz"
TIMELINE_FILE = "timeline.json"

REGIONS: tuple[str, ...] = ("core", "whole")
THRESHOLD_QUANTITIES: tuple[str, ...] = ("dice", "log10_V", "log_vol_ratio", "msd", "hd95")
THRESHOLD_QOIS: list[str] = [f"{quantity}_{region}" for quantity in THRESHOLD_QUANTITIES for region in REGIONS]
STAR_QOIS: list[str] = [*(f"dice_star_{region}" for region in REGIONS), "core_threshold_star", "edema_threshold_star"]
FIXED_SUFFIX = "_fixed"
FIXED_QOIS: list[str] = [f"{name}{FIXED_SUFFIX}" for name in THRESHOLD_QOIS]
# The atlas's threshold-free QoIs kept per session (``compute_qois``).
FIELD_QOIS: list[str] = ["mass", "log10_mass", "centroid_drift", "R_g", "anisotropy", "log10_anisotropy", "wm_fraction"]
FIELD_ANALYSED_QOIS: list[str] = ["log10_mass", "centroid_drift", "R_g", "log10_anisotropy", "wm_fraction"]
RUN_QOIS: list[str] = ["dice_mean_core"]
ROW_THRESHOLD_COLUMNS: list[str] = ["core_threshold", "edema_threshold"]
CARRIED_COLUMNS: list[str] = ["voxel_volume", "final_time", "n_steps", "dt", "wall_time_s"]
STATUS_COLUMNS: list[str] = ["run_name", "success", "exit_code", "wall_time_s", "error", "final_time", "n_steps", "dt"]

ACCOUNTING_NOTE = (
    "Counts prefixed n_runs_ are Saltelli rows (one qoi.csv line each; in sampled "
    "mode several rows share one solve); counts prefixed n_blocks_ are Saltelli "
    "blocks of block_size rows. A block is dropped from a QoI's Sobol' analysis "
    "when any of its rows is failed or non-finite for that QoI. A Dice is NaN "
    "when the model and the reference masks are both empty, a log_vol_ratio when "
    "the reference is empty, msd / hd95 when either mask is empty, the "
    "mass-weighted field QoIs when the field's mass is at or below the mass floor."
)


# --- sessions and patient files ---


@dataclass(frozen=True)
class Session:
    """
    One row of the session-labels tsv for the patient.

    Attributes:
        id: The session id (e.g. "ses-03").
        label: The normalised type: preop, postop or followup
            (``normalise_label``).
        date: The scan date.
        approximate: Whether the tsv marks the date with a trailing '*'.
    """

    id: str
    label: str
    date: date
    approximate: bool = False

    @property
    def prefix(self) -> str:
        """The QoI column prefix of the session: "ses-03" -> "ses03_"."""
        return self.id.replace("-", "") + "_"

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "date": self.date.isoformat(), "approximate": self.approximate}


def normalise_label(text: str) -> str:
    """The session type without case, hyphens, underscores and spaces
    ("follow-up" -> "followup")."""
    return "".join(ch for ch in str(text).strip().lower() if ch not in "-_ ")


def parse_tsv_date(text: str) -> tuple[date, bool]:
    """The date of a tsv cell and whether it carried the '*' mark."""
    text = str(text).strip()
    approximate = text.endswith("*")
    return date.fromisoformat(text.rstrip("*")), approximate


def read_session_labels(path: str | Path, patient: str) -> list[Session]:
    """
    The patient's rows of the session-labels tsv, in file order.

    Raises:
        ValueError: The tsv lacks one of SESSION_COLUMNS, or has no row
            for the patient.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"session labels not found: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in SESSION_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: lacks the column(s) {missing}; found {reader.fieldnames}.")
        rows = [row for row in reader if row["subject"].strip() == patient]
    if not rows:
        raise ValueError(f"{path}: no row for subject {patient!r}.")
    sessions = []
    for row in rows:
        scan_date, approximate = parse_tsv_date(row["date"])
        sessions.append(Session(row["session"].strip(), normalise_label(row["label"]), scan_date, approximate))
    return sessions


def parse_sessions(text: str) -> list[str]:
    """
    The session ids of a --sessions argument: a range "02-08" (ses-02 to
    ses-08) or a comma-separated list of ids ("ses-02,ses-04") or numbers
    ("2,4").
    """
    text = text.strip()
    if not text:
        raise ValueError("--sessions must name at least one session.")
    if "," not in text and "-" in text and not text.startswith("ses"):
        first, _, last = text.partition("-")
        lo, hi = int(first), int(last)
        if lo > hi:
            raise ValueError(f"--sessions: the range {text!r} is empty.")
        return [f"ses-{n:02d}" for n in range(lo, hi + 1)]
    ids = []
    for item in (part.strip() for part in text.split(",") if part.strip()):
        ids.append(item if item.startswith("ses") else f"ses-{int(item):02d}")
    return ids


def select_sessions(sessions: Sequence[Session], later_ids: Sequence[str]) -> list[Session]:
    """
    The sessions of a design: the pre-op session (the one row labelled
    preop) first, then the requested later sessions sorted by date.

    Raises:
        ValueError: Not exactly one pre-op row; a requested id not in the
            tsv; a later session not after the pre-op scan; the post-op
            session (the cavity's source) not requested, or not exactly
            one row labelled postop.
    """
    preops = [s for s in sessions if s.label == LABEL_PREOP]
    if len(preops) != 1:
        raise ValueError(f"expected one session labelled {LABEL_PREOP}, got {[s.id for s in preops]}.")
    preop = preops[0]
    by_id = {s.id: s for s in sessions}
    unknown = [sid for sid in later_ids if sid not in by_id]
    if unknown:
        raise ValueError(f"session(s) {unknown} are not in the tsv for the patient; known: {sorted(by_id)}.")
    later = sorted((by_id[sid] for sid in dict.fromkeys(later_ids)), key=lambda s: s.date)
    early = [s.id for s in later if s.date <= preop.date]
    if early:
        raise ValueError(f"session(s) {early} are not after the pre-op scan {preop.id} ({preop.date}).")
    postops = [s for s in later if s.label == LABEL_POSTOP]
    if len(postops) != 1:
        raise ValueError(
            f"the later sessions must hold exactly one session labelled {LABEL_POSTOP} (the resection "
            f"cavity's source), got {[s.id for s in postops]} among {[s.id for s in later]}."
        )
    return [preop, *later]


@dataclass(frozen=True)
class PatientData:
    """
    The patient's files of a design.

    Attributes:
        patient: The subject id.
        root: The processed data root.
        sessions: The pre-op session first, then the later sessions by date.
        tissue: The pre-op tissue maps by config key.
        preop_segmentation: The pre-op tumour segmentation.
        segmentations: The later sessions' recurrence_preop segmentations
            by session id.
        cavity_session: The post-op session (label 4 is the cavity).
        dose_session: The CRT start session (holds the dose map).
        dose: The dose map.
    """

    patient: str
    root: Path
    sessions: tuple[Session, ...]
    tissue: dict[str, Path]
    preop_segmentation: Path
    segmentations: dict[str, Path]
    cavity_session: str
    dose_session: str
    dose: Path

    @property
    def preop(self) -> Session:
        return self.sessions[0]

    @property
    def later(self) -> tuple[Session, ...]:
        return self.sessions[1:]

    def segmentation(self, session: Session) -> Path:
        """The segmentation file of a session (pre-op or later)."""
        return self.preop_segmentation if session.id == self.preop.id else self.segmentations[session.id]

    def record(self) -> dict[str, Any]:
        return {
            "patient": self.patient,
            "root": str(self.root),
            "sessions": [
                {**s.record(), "segmentation": str(self.segmentation(s))} for s in self.sessions
            ],
            "tissue": {key: str(path) for key, path in self.tissue.items()},
            "cavity_session": self.cavity_session,
            "cavity_label": LABEL_CAVITY,
            "dose_session": self.dose_session,
            "dose": str(self.dose),
        }


def crt_start_session(sessions: Sequence[Session]) -> Session:
    """The CRT start session: the first follow-up after the post-op
    session (the sessions given by date, the pre-op one first)."""
    postop = next(s for s in sessions if s.label == LABEL_POSTOP)
    for session in sessions:
        if session.label == LABEL_FOLLOWUP and session.date > postop.date:
            return session
    raise ValueError(f"no {LABEL_FOLLOWUP} session after the {LABEL_POSTOP} session {postop.id}.")


def patient_files(root: str | Path, patient: str, sessions: Sequence[Session]) -> PatientData:
    """
    The patient's file paths (the layout of the module docstring), every
    one checked to exist.

    Raises:
        FileNotFoundError: Naming every missing file.
    """
    root = Path(root).resolve()
    preop = sessions[0]
    base = root / patient
    tissue = {key: base / preop.id / file for key, file in TISSUE_FILES.items()}
    later = list(sessions[1:])
    segmentations = {s.id: base / s.id / LATER_SEGMENTATION_FILE for s in later}
    cavity = next(s for s in later if s.label == LABEL_POSTOP)
    crt = crt_start_session(sessions)
    data = PatientData(
        patient=patient,
        root=root,
        sessions=tuple(sessions),
        tissue=tissue,
        preop_segmentation=base / preop.id / PREOP_SEGMENTATION_FILE,
        segmentations=segmentations,
        cavity_session=cavity.id,
        dose_session=crt.id,
        dose=base / crt.id / DOSE_FILE,
    )
    paths = [*tissue.values(), data.preop_segmentation, *segmentations.values(), data.dose]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("patient file(s) not found:\n  " + "\n  ".join(missing))
    return data


def _label_counts(segmentation: NDArray) -> dict[str, int]:
    labels, counts = np.unique(segmentation, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def load_segmentation(path: str | Path) -> tuple[NDArray, nib.Nifti1Image]:
    """A tumour segmentation as int64 labels (values rounded) and its
    image; labels outside KNOWN_LABELS are an error."""
    image = nib.load(str(path))
    segmentation = np.rint(np.asarray(image.get_fdata(), dtype=np.float64)).astype(np.int64)
    unknown = sorted(set(np.unique(segmentation).tolist()) - set(KNOWN_LABELS))
    if unknown:
        raise ValueError(f"{path}: label(s) {unknown} are not among {list(KNOWN_LABELS)}.")
    return segmentation, image


def check_patient_data(data: PatientData) -> dict[str, Any]:
    """
    The data checks of the design step (STEP 0): every segmentation and
    the dose map are loaded, the label counts per session, the dose map's
    range and nonzero volume are printed, and every volume's shape and
    affine must equal the pre-op segmentation's (nothing is resampled).

    Returns:
        The spec.json record (data_checks).
    """
    preop_seg, preop_image = load_segmentation(data.preop_segmentation)
    affine = np.asarray(preop_image.affine, dtype=np.float64)
    zooms = tuple(float(z) for z in preop_image.header.get_zooms()[:3])
    shape = preop_seg.shape
    record: dict[str, Any] = {
        "grid_shape": list(shape),
        "voxel_size_mm": list(zooms),
        "affine": affine.tolist(),
        "sessions": {},
        "tissue": {},
        "dose": {},
    }

    def check_geometry(name: str, image: Any) -> None:
        image_shape = tuple(int(s) for s in image.shape[:3])
        if image_shape != tuple(shape):
            raise ValueError(f"{name}: shape {image_shape} differs from the pre-op segmentation's {tuple(shape)}.")
        if not np.allclose(np.asarray(image.affine, dtype=np.float64), affine, rtol=0, atol=1e-3):
            raise ValueError(
                f"{name}: affine\n{np.asarray(image.affine)}\ndiffers from the pre-op segmentation's\n{affine}\n"
                "(the data must be in the pre-op space; nothing is resampled)."
            )

    print(f"patient {data.patient} ({data.root}); grid {list(shape)}, voxel {zooms} mm; pre-op affine:\n{affine}")
    for session in data.sessions:
        if session.id == data.preop.id:
            segmentation, image = preop_seg, preop_image
        else:
            segmentation, image = load_segmentation(data.segmentation(session))
            check_geometry(str(data.segmentation(session)), image)
        counts = _label_counts(segmentation)
        entry: dict[str, Any] = {
            "file": str(data.segmentation(session)),
            "label_counts": counts,
            "dtype": str(image.get_data_dtype()),
        }
        note = ""
        if session.id == data.preop.id:
            relabelled = relabel_preop(segmentation)
            entry["label_counts_after_relabel"] = _label_counts(relabelled)
            note = f"; after 4 -> 3: {entry['label_counts_after_relabel']}"
        elif session.id == data.cavity_session:
            note = f"; cavity (label {LABEL_CAVITY}): {counts.get(str(LABEL_CAVITY), 0)} voxels"
        record["sessions"][session.id] = entry
        flag = "~" if session.approximate else ""
        print(f"  {session.id} ({session.label}, {flag}{session.date}): labels {counts}{note}")
    for key, path in data.tissue.items():
        image = nib.load(str(path))
        check_geometry(str(path), image)
        values = np.asarray(image.dataobj, dtype=np.float64)
        record["tissue"][key] = {"file": str(path), "min": float(values.min()), "max": float(values.max())}
        print(f"  {key}: {path} range [{values.min():.4g}, {values.max():.4g}]")
    dose_image = nib.load(str(data.dose))
    check_geometry(str(data.dose), dose_image)
    dose = np.asarray(dose_image.dataobj, dtype=np.float64)
    if not np.all(np.isfinite(dose)) or dose.min() < 0:
        raise ValueError(f"{data.dose}: the dose map must be finite and nonnegative.")
    n_nonzero = int(np.count_nonzero(dose))
    record["dose"] = {
        "file": str(data.dose),
        "session": data.dose_session,
        "min_gy": float(dose.min()),
        "max_gy": float(dose.max()),
        "n_nonzero": n_nonzero,
        "nonzero_volume_mm3": float(n_nonzero * np.prod(zooms)),
        "affine_matches_preop": True,
    }
    print(
        f"  dose map ({data.dose_session}): {data.dose}; min {dose.min():.3g} Gy, max {dose.max():.4g} Gy, "
        f"{n_nonzero} nonzero voxels ({n_nonzero * np.prod(zooms) / 1000:.1f} cm^3); affine equals the pre-op segmentation's"
    )
    lo, hi = RT_TOTAL_DOSE_PLAUSIBLE_GY
    if not lo <= dose.max() <= hi:
        warnings.warn(
            f"the dose map's maximum {dose.max():.4g} Gy is outside [{lo:g}, {hi:g}] Gy; the solver takes the map "
            "as the TOTAL dose over all fractions (about 60 Gy for a Stupp course).",
            stacklevel=2,
        )
    return record


def relabel_preop(segmentation: NDArray) -> NDArray:
    """The pre-op segmentation with PREOP_RELABEL applied (4 -> 3)."""
    out = np.array(segmentation, dtype=np.int64, copy=True)
    for source, target in PREOP_RELABEL.items():
        out[segmentation == source] = target
    return out


# --- the clinical schedule ---


@dataclass(frozen=True)
class Protocol:
    """
    The dose values of the Stupp protocol taken from the base config
    (``protocol_from_config``) and the schedule constants of this analysis.

    Attributes:
        concomitant_dose: TMZ dose of the concomitant days, mg/m^2.
        adjuvant_first_dose: TMZ dose of adjuvant cycle 1.
        adjuvant_later_dose: TMZ dose of the later cycles.
        base_cycle_days: The base config's own cycle length (informational).
    """

    concomitant_dose: float
    adjuvant_first_dose: float
    adjuvant_later_dose: float
    base_cycle_days: float | None = None
    n_fractions: int = N_FRACTIONS
    concomitant_days: int = CONCOMITANT_DAYS
    adjuvant_delay_days: int = ADJUVANT_DELAY_DAYS
    adjuvant_days_on: int = ADJUVANT_DAYS_ON
    adjuvant_cycle_days: int = ADJUVANT_CYCLE_DAYS

    def record(self) -> dict[str, Any]:
        return {
            "concomitant_dose_mg_m2": self.concomitant_dose,
            "adjuvant_first_cycle_dose_mg_m2": self.adjuvant_first_dose,
            "adjuvant_later_cycles_dose_mg_m2": self.adjuvant_later_dose,
            "n_fractions": self.n_fractions,
            "fractions_on": "calendar weekdays (Mon-Fri) from the CRT start date",
            "concomitant_days": self.concomitant_days,
            "adjuvant_delay_days": self.adjuvant_delay_days,
            "adjuvant_days_on": self.adjuvant_days_on,
            "adjuvant_cycle_days": self.adjuvant_cycle_days,
            "base_config_cycle_days": self.base_cycle_days,
            "resection_days_before_postop": RESECTION_DAYS_BEFORE_POSTOP,
        }


def protocol_from_config(base: Mapping[str, Any]) -> Protocol:
    """
    The dose values of the base config's schedule: the concomitant dose
    (the chemo sessions within CONCOMITANT_DAYS of the first fraction,
    all one dose), the adjuvant doses (the later sessions grouped into
    cycles at gaps of more than a day: cycle 1's dose and the later
    cycles' dose, each cycle ADJUVANT_DAYS_ON sessions of one dose).

    Raises:
        ValueError: The base schedule does not have that structure.
    """
    rt_times = np.asarray(base["rt_times"], dtype=np.float64)
    chemo_times = np.asarray(base["chemo_times"], dtype=np.float64)
    chemo_doses = np.asarray(base["chemo_doses"], dtype=np.float64)
    if rt_times.size != N_FRACTIONS:
        raise ValueError(f"the base config has {rt_times.size} fractions, the protocol {N_FRACTIONS}.")
    if chemo_times.shape != chemo_doses.shape:
        raise ValueError("the base config's chemo_times and chemo_doses differ in length.")
    order = np.argsort(chemo_times)
    chemo_times, chemo_doses = chemo_times[order], chemo_doses[order]
    start = float(rt_times.min())
    concomitant = (chemo_times >= start) & (chemo_times < start + CONCOMITANT_DAYS)
    doses = np.unique(chemo_doses[concomitant])
    if concomitant.sum() != CONCOMITANT_DAYS or doses.size != 1:
        raise ValueError(
            f"the base config must give {CONCOMITANT_DAYS} concomitant TMZ sessions of one dose within "
            f"{CONCOMITANT_DAYS} days of the first fraction, got {int(concomitant.sum())} sessions with doses {doses.tolist()}."
        )
    concomitant_dose = float(doses[0])
    later_times = chemo_times[chemo_times >= start + CONCOMITANT_DAYS]
    later_doses = chemo_doses[chemo_times >= start + CONCOMITANT_DAYS]
    if later_times.size == 0:
        raise ValueError("the base config has no adjuvant TMZ sessions after the concomitant phase.")
    breaks = np.flatnonzero(np.diff(later_times) > 1.0) + 1
    cycles_times = np.split(later_times, breaks)
    cycles_doses = np.split(later_doses, breaks)
    cycle_doses = []
    for times, doses_ in zip(cycles_times, cycles_doses):
        unique = np.unique(doses_)
        if times.size != ADJUVANT_DAYS_ON or unique.size != 1:
            raise ValueError(
                f"every adjuvant cycle of the base config must be {ADJUVANT_DAYS_ON} consecutive sessions of one "
                f"dose, got days {times.tolist()} with doses {doses_.tolist()}."
            )
        cycle_doses.append(float(unique[0]))
    later_unique = sorted(set(cycle_doses[1:]))
    if len(later_unique) > 1:
        raise ValueError(f"the adjuvant cycles after the first have several doses: {later_unique}.")
    starts = [float(times[0]) for times in cycles_times]
    base_cycle = float(np.unique(np.diff(starts))[0]) if len(starts) > 1 and np.unique(np.diff(starts)).size == 1 else None
    return Protocol(
        concomitant_dose=concomitant_dose,
        adjuvant_first_dose=cycle_doses[0],
        adjuvant_later_dose=later_unique[0] if later_unique else cycle_doses[0],
        base_cycle_days=base_cycle,
    )


@dataclass(frozen=True)
class AdjuvantCycle:
    """One adjuvant TMZ cycle: its number, start offset, the day offsets
    within the run, the dose and the days dropped at the horizon."""

    number: int
    start_offset: int
    offsets: tuple[int, ...]
    dose: float
    n_dropped: int


@dataclass(frozen=True)
class Timeline:
    """
    The patient's calendar as day offsets after the pre-op scan
    (``build_timeline``); ``model_days`` shifts it by preop_time.

    Attributes:
        sessions: The sessions, the pre-op one first.
        resection_offset: Resection day.
        crt_start_offset: The first fraction and the first TMZ day.
        rt_offsets: The fractions within the run.
        concomitant_offsets: The concomitant TMZ days within the run.
        adjuvant_cycles: The cycles starting within the run.
        horizon_offset: The last session's day, the end of the run.
        n_rt_dropped, n_concomitant_dropped: Events after the horizon.
    """

    sessions: tuple[Session, ...]
    resection_offset: int
    crt_start_offset: int
    rt_offsets: tuple[int, ...]
    concomitant_offsets: tuple[int, ...]
    adjuvant_cycles: tuple[AdjuvantCycle, ...]
    horizon_offset: int
    n_rt_dropped: int
    n_concomitant_dropped: int
    concomitant_dose: float

    @property
    def preop_date(self) -> date:
        return self.sessions[0].date

    def date_of(self, offset: int) -> date:
        return self.preop_date + timedelta(days=int(offset))

    @property
    def last_crt_offset(self) -> int:
        """The last CRT day: the later of the last fraction and the last
        concomitant TMZ day (of the full course, dropped days included)."""
        return max(self.crt_start_offset + self.concomitant_days_total - 1, self.rt_offsets_total[-1])

    @property
    def concomitant_days_total(self) -> int:
        return CONCOMITANT_DAYS

    @property
    def rt_offsets_total(self) -> tuple[int, ...]:
        """All N_FRACTIONS fraction offsets (weekdays from the CRT start),
        the dropped ones included."""
        return weekday_offsets(self.preop_date, self.crt_start_offset, N_FRACTIONS)

    @property
    def snapshot_offsets(self) -> dict[str, int]:
        return {s.id: (s.date - self.preop_date).days for s in self.sessions}

    @property
    def chemo_schedule(self) -> tuple[list[int], list[float]]:
        """(offsets, doses) of every TMZ day within the run, ascending."""
        events = [(offset, self.concomitant_dose) for offset in self.concomitant_offsets]
        for cycle in self.adjuvant_cycles:
            events.extend((offset, cycle.dose) for offset in cycle.offsets)
        events.sort()
        return [offset for offset, _ in events], [dose for _, dose in events]

    def model_days(self, preop_time: float) -> dict[str, Any]:
        """
        The run's event days for preop_time = t_pre: resection_time,
        time_after_resection, rt_times, chemo_times, chemo_doses and the
        snapshot moments by session id (t_pre + offset each).
        """
        t_pre = float(preop_time)
        chemo_offsets, chemo_doses = self.chemo_schedule
        return {
            "resection_time": t_pre + self.resection_offset,
            "time_after_resection": float(self.horizon_offset - self.resection_offset),
            "rt_times": [t_pre + offset for offset in self.rt_offsets],
            "chemo_times": [t_pre + offset for offset in chemo_offsets],
            "chemo_doses": list(chemo_doses),
            "snapshots": {sid: t_pre + offset for sid, offset in self.snapshot_offsets.items()},
            "stopping_time": t_pre + self.horizon_offset,
        }

    def record(self) -> dict[str, Any]:
        """The spec.json record: calendar dates and offsets of everything."""
        chemo_offsets, chemo_doses = self.chemo_schedule

        def dated(offsets: Sequence[int]) -> list[dict[str, Any]]:
            return [{"date": self.date_of(o).isoformat(), "offset_days": int(o)} for o in offsets]

        return {
            "preop_date": self.preop_date.isoformat(),
            "model_day": "t_pre + offset_days, t_pre = the row's preop_time (the seed is day 0)",
            "sessions": [{**s.record(), "offset_days": self.snapshot_offsets[s.id]} for s in self.sessions],
            "resection": {"date": self.date_of(self.resection_offset).isoformat(), "offset_days": self.resection_offset},
            "crt_start": {"date": self.date_of(self.crt_start_offset).isoformat(), "offset_days": self.crt_start_offset},
            "last_crt_day": {"date": self.date_of(self.last_crt_offset).isoformat(), "offset_days": self.last_crt_offset},
            "rt_fractions": dated(self.rt_offsets),
            "n_rt_fractions": len(self.rt_offsets),
            "n_rt_fractions_dropped": self.n_rt_dropped,
            "concomitant_tmz": dated(self.concomitant_offsets),
            "concomitant_dose_mg_m2": self.concomitant_dose,
            "n_concomitant_days": len(self.concomitant_offsets),
            "n_concomitant_days_dropped": self.n_concomitant_dropped,
            "adjuvant_start": {
                "date": self.date_of(self.last_crt_offset + ADJUVANT_DELAY_DAYS).isoformat(),
                "offset_days": self.last_crt_offset + ADJUVANT_DELAY_DAYS,
            },
            "adjuvant_cycles": [
                {
                    "cycle": c.number,
                    "start_date": self.date_of(c.start_offset).isoformat(),
                    "start_offset_days": c.start_offset,
                    "days": dated(c.offsets),
                    "n_days": len(c.offsets),
                    "n_days_dropped": c.n_dropped,
                    "dose_mg_m2": c.dose,
                }
                for c in self.adjuvant_cycles
            ],
            "chemo_offsets": list(chemo_offsets),
            "chemo_doses": list(chemo_doses),
            "chemo_total_dose_mg_m2": float(sum(chemo_doses)),
            "snapshot_offsets": self.snapshot_offsets,
            "snapshot_moment": "the state at the start of the scan day, before any event of that day (see the module docstring)",
            "horizon": {"date": self.date_of(self.horizon_offset).isoformat(), "offset_days": self.horizon_offset},
        }


def weekday_offsets(preop_date: date, start_offset: int, n: int) -> tuple[int, ...]:
    """The first n calendar weekdays (Mon-Fri) from the start offset, as
    offsets after the pre-op date."""
    offsets: list[int] = []
    offset = int(start_offset)
    while len(offsets) < n:
        if (preop_date + timedelta(days=offset)).weekday() < 5:
            offsets.append(offset)
        offset += 1
    return tuple(offsets)


def build_timeline(sessions: Sequence[Session], protocol: Protocol) -> Timeline:
    """
    The clinical timeline of the module docstring on the patient's
    calendar: the sessions (the pre-op one first, then by date), the
    resection RESECTION_DAYS_BEFORE_POSTOP days before the post-op scan,
    the CRT from the CRT start session's date (N_FRACTIONS fractions on
    weekdays, CONCOMITANT_DAYS daily TMZ days), the adjuvant cycles from
    ADJUVANT_DELAY_DAYS after the last CRT day in ADJUVANT_CYCLE_DAYS
    cycles of ADJUVANT_DAYS_ON days, as many as start by the last session
    date, and every event after the last session dropped (counted).

    Raises:
        ValueError: The resection falls before the pre-op scan or after
            the CRT start.
    """
    sessions = tuple(sessions)
    preop = sessions[0]
    if preop.label != LABEL_PREOP:
        raise ValueError(f"the first session must be the pre-op one, got {preop.id} ({preop.label}).")
    postop = next((s for s in sessions if s.label == LABEL_POSTOP), None)
    if postop is None:
        raise ValueError(f"no session labelled {LABEL_POSTOP}.")
    crt = crt_start_session(sessions)
    offset = {s.id: (s.date - preop.date).days for s in sessions}
    horizon = max(offset.values())
    resection = offset[postop.id] - RESECTION_DAYS_BEFORE_POSTOP
    if resection < 0:
        raise ValueError(f"the resection ({resection} days after the pre-op scan) precedes the pre-op scan.")
    if resection >= offset[crt.id]:
        raise ValueError(f"the resection (day {resection}) is not before the CRT start (day {offset[crt.id]}).")
    rt_all = weekday_offsets(preop.date, offset[crt.id], protocol.n_fractions)
    concomitant_all = tuple(offset[crt.id] + i for i in range(protocol.concomitant_days))
    last_crt = max(rt_all[-1], concomitant_all[-1])
    adjuvant_start = last_crt + protocol.adjuvant_delay_days
    cycles: list[AdjuvantCycle] = []
    number, start = 1, adjuvant_start
    while start <= horizon:
        days = [start + i for i in range(protocol.adjuvant_days_on)]
        kept = tuple(d for d in days if d <= horizon)
        dose = protocol.adjuvant_first_dose if number == 1 else protocol.adjuvant_later_dose
        cycles.append(AdjuvantCycle(number, start, kept, dose, len(days) - len(kept)))
        number += 1
        start += protocol.adjuvant_cycle_days
    rt_kept = tuple(o for o in rt_all if o <= horizon)
    concomitant_kept = tuple(o for o in concomitant_all if o <= horizon)
    return Timeline(
        sessions=sessions,
        resection_offset=resection,
        crt_start_offset=offset[crt.id],
        rt_offsets=rt_kept,
        concomitant_offsets=concomitant_kept,
        adjuvant_cycles=tuple(cycles),
        horizon_offset=horizon,
        n_rt_dropped=len(rt_all) - len(rt_kept),
        n_concomitant_dropped=len(concomitant_all) - len(concomitant_kept),
        concomitant_dose=protocol.concomitant_dose,
    )


def format_timeline(timeline: Timeline) -> str:
    """The printed calendar and cycle table of a timeline."""
    lines = ["timeline (model day = t_pre + offset; t_pre = the row's preop_time):"]
    lines.append(f"  {'date':<12}{'day':<5}{'offset':>7}  event")

    def row(offset: int, event: str) -> str:
        d = timeline.date_of(offset)
        return f"  {d.isoformat():<12}{WEEKDAY_NAMES[d.weekday()]:<5}{offset:>+7d}  {event}"

    events: list[tuple[int, int, str]] = []
    for s in timeline.sessions:
        flag = " (date approximate)" if s.approximate else ""
        events.append((timeline.snapshot_offsets[s.id], 0, f"scan {s.id} ({s.label}){flag}: snapshot"))
    events.append((timeline.resection_offset, 1, f"resection ({RESECTION_DAYS_BEFORE_POSTOP} days before the post-op scan)"))
    rt, conc = timeline.rt_offsets, timeline.concomitant_offsets
    events.append((timeline.crt_start_offset, 2, f"CRT start: RT fraction 1 of {N_FRACTIONS}, TMZ {timeline.concomitant_dose:g} mg/m^2 day 1 of {CONCOMITANT_DAYS}"))
    if rt:
        events.append((rt[-1], 3, f"RT fraction {len(rt)} (last within the run; {timeline.n_rt_dropped} dropped after the horizon)"))
    if conc:
        events.append((conc[-1], 4, f"concomitant TMZ day {len(conc)} (last within the run; {timeline.n_concomitant_dropped} dropped)"))
    adjuvant_start = timeline.last_crt_offset + ADJUVANT_DELAY_DAYS
    events.append((adjuvant_start, 5, f"adjuvant TMZ start ({ADJUVANT_DELAY_DAYS} days after the last CRT day {timeline.date_of(timeline.last_crt_offset)})" + (" - after the horizon, no cycle runs" if adjuvant_start > timeline.horizon_offset else "")))
    for c in timeline.adjuvant_cycles:
        events.append((c.start_offset, 6, f"adjuvant cycle {c.number}: {len(c.offsets)} day(s) at {c.dose:g} mg/m^2" + (f", {c.n_dropped} dropped" if c.n_dropped else "")))
    events.append((timeline.horizon_offset, 9, "end of the run (last session)"))
    for offset, _, text in sorted(events):
        lines.append(row(offset, text))
    lines.append(
        f"  fractions: {len(rt)} on weekdays {timeline.date_of(rt[0]) if rt else '-'} .. {timeline.date_of(rt[-1]) if rt else '-'}; "
        f"concomitant TMZ: {len(conc)} days; adjuvant: {sum(len(c.offsets) for c in timeline.adjuvant_cycles)} days in "
        f"{len(timeline.adjuvant_cycles)} cycle(s) ({ADJUVANT_DAYS_ON} on / {ADJUVANT_CYCLE_DAYS - ADJUVANT_DAYS_ON} off)"
    )
    lines.append("adjuvant cycle table:")
    lines.append(f"  {'cycle':<6}{'start':<12}{'end':<12}{'days on':>8}{'dropped':>8}{'dose':>8}")
    if not timeline.adjuvant_cycles:
        lines.append("  (none: the first cycle would start after the last session)")
    for c in timeline.adjuvant_cycles:
        end = timeline.date_of(c.offsets[-1]) if c.offsets else timeline.date_of(c.start_offset)
        lines.append(f"  {c.number:<6}{timeline.date_of(c.start_offset).isoformat():<12}{end.isoformat():<12}{len(c.offsets):>8}{c.n_dropped:>8}{c.dose:>8g}")
    return "\n".join(lines)


# --- search space ---


def read_patient_search_space(path: str | Path, threshold_mode: str) -> tuple[SearchSpace, dict[str, Any]]:
    """
    The search space of a design: the file's entries with the "cheap"
    flags stripped (recorded), the cheap factors dropped in profiled mode,
    handed to the atlas's ``load_search_space`` with the script factors
    among the admissible keys.

    Returns:
        (space, meta) with meta holding threshold_mode, cheap_factors (in
        factor order; empty in profiled mode), threshold_factors (the two
        threshold ranges as read, also in profiled mode) and source.

    Raises:
        ValueError: The mode is unknown; the file holds a timeline entry
            or a patient volume; preop_time is not a factor; a cheap
            factor is not one of THRESHOLD_FACTORS or one of them is not
            cheap; in sampled mode a threshold factor is not linear
            within (0, 1] or the ratio's max is not below 1.
    """
    if threshold_mode not in THRESHOLD_MODES:
        raise ValueError(f"threshold mode must be one of {THRESHOLD_MODES}, got {threshold_mode!r}.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"search space not found: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, Mapping):
        raise ValueError(f"search space {path}: must be a JSON object.")
    forbidden = [key for key in entries if key in TIMELINE_KEYS or key in PATIENT_VOLUME_KEYS]
    if forbidden:
        raise ValueError(f"search space {path}: {forbidden} are set per run from the patient's data and may not appear.")
    cheap = [key for key, value in entries.items() if isinstance(value, Mapping) and value.get(CHEAP_KEY)]
    if sorted(cheap) != sorted(THRESHOLD_FACTORS):
        raise ValueError(
            f"search space {path}: the factors marked {CHEAP_KEY!r} must be exactly {list(THRESHOLD_FACTORS)}, got {cheap}."
        )
    stripped: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for key, value in entries.items():
        if key in cheap:
            entry = {k: v for k, v in value.items() if k != CHEAP_KEY}
            thresholds[key] = entry
            if threshold_mode == "sampled":
                stripped[key] = entry
        else:
            stripped[key] = value
    keys = StuppFKPPSolver.config_keys() | set(SCRIPT_FACTORS)
    space = load_search_space(stripped, keys)
    if PREOP_TIME_FACTOR not in space.factors:
        raise ValueError(f"search space {path}: {PREOP_TIME_FACTOR} must be a factor, got {entries.get(PREOP_TIME_FACTOR)!r}.")
    if space.factors[PREOP_TIME_FACTOR].low <= 0:
        raise ValueError(f"search space {path}: {PREOP_TIME_FACTOR} must be positive.")
    if threshold_mode == "sampled":
        for key in THRESHOLD_FACTORS:
            factor = space.factors[key]
            if factor.scale != "linear" or factor.low <= 0 or factor.high > 1:
                raise ValueError(f"search space {path}: {key} must be a linear factor within (0, 1], got {factor}.")
        if space.factors[EDEMA_RATIO_FACTOR].high >= 1:
            raise ValueError(f"search space {path}: {EDEMA_RATIO_FACTOR} max must be below 1 so that edema < core.")
    cheap_in_order = [name for name in space.names if name in cheap]
    meta = {
        "threshold_mode": threshold_mode,
        "cheap_factors": cheap_in_order,
        "threshold_factors": thresholds,
        "source": dict(entries),
    }
    return space, meta


def solver_values(space: SearchSpace, record: Mapping[str, Any]) -> dict[str, float]:
    """The solver parameter values of a design record: the atlas's
    ``SearchSpace.solver_values`` without the script factors."""
    return {key: value for key, value in space.solver_values(record).items() if key not in SCRIPT_FACTORS}


# --- seeding and design ---


def patient_seed_geometry(wm: NDArray, gm: NDArray, core: NDArray, min_tissue_fraction: float) -> SeedGeometry:
    """
    The seedable voxels of the patient: the pre-op core mask (labels 1
    and 3 after the relabelling) intersected with the tissue
    (wm + gm >= min_tissue_fraction), built with the atlas's
    ``seed_geometry`` on the tissue maps zeroed outside the core (so
    that its threshold test selects exactly core & tissue).
    """
    core = np.asarray(core, dtype=bool)
    if core.shape != np.shape(wm):
        raise ValueError(f"the core mask {core.shape} and the tissue maps {np.shape(wm)} differ in shape.")
    return seed_geometry(np.where(core, wm, 0.0), np.where(core, gm, 0.0), min_tissue_fraction)


def dedup_runs(samples: NDArray, cheap_columns: Sequence[int]) -> NDArray:
    """
    The representative row of every design row: the first row (lowest
    index) with the same unit-cube coordinates in every non-cheap column
    (exact equality, as SALib copies the coordinates between the A, B
    and A_B rows). Rows sharing a representative share one solve.

    Args:
        samples: The Saltelli sample, (n_rows, k).
        cheap_columns: The columns the solve does not depend on.

    Returns:
        The representative index per row, int64 (n_rows,).
    """
    samples = np.asarray(samples, dtype=np.float64)
    keep = [column for column in range(samples.shape[1]) if column not in set(cheap_columns)]
    first: dict[tuple[float, ...], int] = {}
    representative = np.empty(samples.shape[0], dtype=np.int64)
    for index in range(samples.shape[0]):
        key = tuple(samples[index, keep].tolist())
        representative[index] = first.setdefault(key, index)
    return representative


def patient_design_table(
    samples: NDArray, space: SearchSpace, geometry: SeedGeometry, cheap_columns: Sequence[int]
) -> list[dict[str, Any]]:
    """
    The design records of the atlas's ``design_table`` with the row /
    run distinction: row_name (the atlas's r{row:04d}_{matrix} name of
    the Saltelli row), run_name (the row_name of the row's representative,
    ``dedup_runs``) and, with the threshold factors sampled,
    edema_threshold = edema_threshold_ratio * core_threshold.
    """
    records = design_table(samples, space, geometry)
    representative = dedup_runs(samples, cheap_columns)
    sampled = CORE_THRESHOLD_FACTOR in space.factors
    out = []
    for record in records:
        row_name = record["run_name"]
        entry: dict[str, Any] = {"row_name": row_name, "run_name": records[int(representative[record["index"]])]["run_name"]}
        entry.update({key: value for key, value in record.items() if key != "run_name"})
        if sampled:
            entry[EDEMA_THRESHOLD_COLUMN] = float(record[EDEMA_RATIO_FACTOR]) * float(record[CORE_THRESHOLD_FACTOR])
        out.append(entry)
    return out


def run_config(
    base: Mapping[str, Any],
    space: SearchSpace,
    values: Mapping[str, Any],
    data: PatientData,
    timeline: Timeline,
    preop_time: float,
) -> dict[str, Any]:
    """
    The config of one run: the base config with the search space's
    overrides and the sampled solver values, the patient's tissue maps,
    cavity (the post-op segmentation's label 4) and dose map, and the
    timeline at preop_time (``Timeline.model_days``); snapshot_times stays
    None (run-one sets it once it knows the time step).
    """
    days = timeline.model_days(preop_time)
    config: dict[str, Any] = {
        SOLVER_KEY: base.get(SOLVER_KEY, SOLVER_NAME),
        "_design": (
            "scripts/patient_sensitivity_analysis.py: the base config with the search space's overrides and the "
            "sampled factor values substituted, the patient's tissue maps, resection cavity and dose map, and "
            "the clinical timeline shifted by the row's preop_time (resection_time, time_after_resection, "
            "chemo_times, chemo_doses, rt_times); snapshot_times is set by the run; see design.csv and "
            "spec.json in the parent directory."
        ),
    }
    config.update({key: value for key, value in base.items() if key != SOLVER_KEY})
    config.update(space.overrides)
    config.update(values)
    config.update({key: str(path) for key, path in data.tissue.items()})
    config["resection_cavity"] = {"segmentation": str(data.segmentations[data.cavity_session]), "label": LABEL_CAVITY}
    config["rt_dose"] = str(data.dose)
    for key in ("resection_time", "time_after_resection", "rt_times", "chemo_times", "chemo_doses"):
        config[key] = days[key]
    config["snapshot_times"] = None
    return config


def chemo_budget(space: SearchSpace, base: Mapping[str, Any], total_dose: float) -> tuple[float, float]:
    """The range of the total log kill L = chemo_kill_rate * D_tot /
    chemo_decay_rate over the factor ranges (a fixed value stands in for
    a range where the parameter is not a factor)."""

    def bounds(key: str) -> tuple[float, float]:
        factor = space.factors.get(key)
        if factor is not None:
            return factor.low, factor.high
        value = float(space.overrides.get(key, base[key]))
        return value, value

    kill_lo, kill_hi = bounds("chemo_kill_rate")
    decay_lo, decay_hi = bounds("chemo_decay_rate")
    return kill_lo * total_dose / decay_hi, kill_hi * total_dose / decay_lo


def make_design(
    search_space_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    name: str,
    log2_n: int = DEFAULT_LOG2_N,
    seed: int = DEFAULT_DESIGN_SEED,
    threshold_mode: str = DEFAULT_THRESHOLD_MODE,
    patient: str = DEFAULT_PATIENT,
    patient_root: str | Path = DEFAULT_PATIENT_ROOT,
    session_labels: str | Path = DEFAULT_SESSION_LABELS,
    sessions: str = DEFAULT_SESSIONS,
) -> Path:
    """
    Check the patient's data (STEP 0), build the timeline, sample the
    Saltelli design and write the sweep directory (everything but the
    runs). Refuses to overwrite an existing directory.

    Returns:
        The sweep directory <output_dir>/<name>.
    """
    sweep_dir = Path(output_dir) / name
    if sweep_dir.exists():
        raise FileExistsError(f"{sweep_dir} exists; a design is never overwritten.")
    config_path = Path(config_path).resolve()
    search_space_path = Path(search_space_path).resolve()
    session_labels = Path(session_labels).resolve()
    base = read_config(config_path, solver=StuppFKPPSolver)
    all_sessions = read_session_labels(session_labels, patient)
    print(f"session labels {session_labels}, rows of {patient}:")
    for s in all_sessions:
        print(f"  {s.id:<8}{s.label:<10}{s.date}{'*' if s.approximate else ''}")
    selected = select_sessions(all_sessions, parse_sessions(sessions))
    data = patient_files(patient_root, patient, selected)
    checks = check_patient_data(data)
    protocol = protocol_from_config(base)
    timeline = build_timeline(data.sessions, protocol)
    print(
        f"protocol doses from {config_path.name}: concomitant {protocol.concomitant_dose:g}, adjuvant cycle 1 "
        f"{protocol.adjuvant_first_dose:g}, later cycles {protocol.adjuvant_later_dose:g} mg/m^2 (the config's own "
        f"cycle length is {protocol.base_cycle_days} days; this analysis uses {ADJUVANT_CYCLE_DAYS}-day cycles)"
    )
    print(format_timeline(timeline))
    space, meta = read_patient_search_space(search_space_path, threshold_mode)
    cheap_columns = [space.names.index(f) for f in meta["cheap_factors"]]
    # The config every run shares, at the midpoint of preop_time: its
    # StuppFKPPSolver instance loads the patient's volumes, resolves the
    # defaults and validates the schedule and the maps; no solve is run.
    preop = space.factors[PREOP_TIME_FACTOR]
    nominal_preop_time = 0.5 * (preop.low + preop.high)
    effective = run_config(base, space, {}, data, timeline, nominal_preop_time)
    effective["_design"] = (
        "the base config with the search space's overrides, the patient's volumes and the timeline at the "
        f"midpoint of {PREOP_TIME_FACTOR} ({nominal_preop_time:g} days): the config the design-time checks ran on"
    )
    solver = StuppFKPPSolver(read_config_from_mapping(effective))
    wm, gm = solver.params["white_matter_pbmap"], solver.params["gray_matter_pbmap"]
    min_tissue_fraction = float(solver.params["min_tissue_fraction"])
    time_step = {key: solver.params[key] for key in TIME_STEP_KEYS}
    if all(value is None for value in time_step.values()):
        raise ValueError(
            f"the base config {config_path} (with the search space's overrides) sets none of {list(TIME_STEP_KEYS)}: "
            'every run would fall back to the solver\'s horizon-dependent stability estimate; set one (e.g. "steps_per_day": 12).'
        )
    derived_groups = {
        group.name: {
            "derives": list(group.derivation.derives),
            "factors": list(group.factors),
            "requires": list(group.derivation.requires),
            "optional": list(group.derivation.optional),
            "extras": list(group.derivation.extras),
            "formula": group.derivation.formula,
            **group.derivation.validate(group.factors, solver.params, space.group_inputs(group)),
        }
        for group in space.groups.values()
    }
    preop_seg, _ = load_segmentation(data.preop_segmentation)
    core = np.isin(relabel_preop(preop_seg), CORE_LABELS)
    geometry = patient_seed_geometry(wm, gm, core, min_tissue_fraction)
    print(
        f"seedable voxels: {geometry.n_voxels} (pre-op core {int(core.sum())} voxels of labels {list(CORE_LABELS)} "
        f"after the relabelling, with wm + gm >= {min_tissue_fraction:g}); box "
        f"{[round(float(v), 4) for v in geometry.bbox_lo]} .. {[round(float(v), 4) for v in geometry.bbox_hi]}"
    )
    samples = saltelli_design(space.names, log2_n, seed)
    table = patient_design_table(samples, space, geometry, cheap_columns)
    n_rows = len(table)
    run_names = list(dict.fromkeys(record["run_name"] for record in table))
    k, k_dyn = len(space.names), len(space.names) - len(cheap_columns)
    n_blocks = 2 ** int(log2_n)
    expected_runs = n_blocks * (k_dyn + 2)
    if len(run_names) != expected_runs:
        raise RuntimeError(f"dedup gave {len(run_names)} distinct runs, expected N (k_dyn + 2) = {expected_runs}.")
    chemo_total = float(sum(timeline.chemo_schedule[1]))
    log_kill = chemo_budget(space, base, chemo_total)
    analysed = analysed_qois([s.prefix for s in data.sessions], threshold_mode)
    spec = {
        "name": name,
        "patient": data.record(),
        "session_labels": str(session_labels),
        "sessions_requested": sessions,
        "label_conventions": LABEL_CONVENTIONS,
        "data_checks": checks,
        "protocol": protocol.record(),
        "timeline": timeline.record(),
        "threshold_mode": threshold_mode,
        "thresholds": {
            "fixed": {"core": FIXED_THRESHOLDS[0], "edema": FIXED_THRESHOLDS[1]},
            "grid_core": list(THRESHOLD_GRID_CORE),
            "grid_edema": list(THRESHOLD_GRID_EDEMA),
            "grid_rule": "pairs with edema < core; the best pair maximises dice_core + dice_whole",
            "factors": meta["threshold_factors"],
            "edema_threshold": f"{EDEMA_RATIO_FACTOR} * {CORE_THRESHOLD_FACTOR} (sampled mode)",
        },
        "cheap_factors": meta["cheap_factors"],
        "dedup": {
            "rule": "rows with equal unit-cube coordinates in every non-cheap column share one run (the row_name of the first)",
            "n_rows": n_rows,
            "n_runs": len(run_names),
            "n_rows_expected": n_blocks * (k + 2),
            "n_runs_expected": expected_runs,
        },
        "search_space_path": str(search_space_path),
        "search_space": meta["source"],
        "factors": {f.name: {"min": f.low, "max": f.high, "scale": f.scale} for f in space.factors.values()},
        "overrides": space.overrides,
        "derived_groups": derived_groups,
        "derived_keys": space.derived_keys,
        "extra_keys": space.extra_keys,
        "script_factors": [f for f in SCRIPT_FACTORS if f in space.factors],
        "base_config": str(config_path),
        "time_step": time_step,
        "seed": int(seed),
        "log2_n": int(log2_n),
        "N": n_blocks,
        "k": k,
        "k_dyn": k_dyn,
        "factor_names": space.names,
        "second_order": False,
        "block_size": block_size(k),
        "matrix_labels": matrix_labels(space.names),
        "n_rows": n_rows,
        "n_runs": len(run_names),
        "salib_version": importlib.metadata.version("SALib"),
        "grid_shape": list(geometry.shape),
        "seed_bbox_lo": geometry.bbox_lo.tolist(),
        "seed_bbox_hi": geometry.bbox_hi.tolist(),
        "n_seedable_voxels": geometry.n_voxels,
        "n_preop_core_voxels": int(core.sum()),
        "seed_min_tissue_fraction": min_tissue_fraction,
        "chemo_total_dose": chemo_total,
        "chemo_log_kill_range": list(log_kill),
        "rt_dose_max_gy": checks["dose"]["max_gy"],
        "rt_dose_per_fraction_max_gy": checks["dose"]["max_gy"] / max(len(timeline.rt_offsets), 1),
        "analysed_qois": analysed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    sweep_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("configs", "logs", "runs"):
        (sweep_dir / sub).mkdir()
    shutil.copyfile(search_space_path, sweep_dir / "search_space.json")
    write_config(effective, sweep_dir / "base_config.json")
    write_json(sweep_dir / "spec.json", spec)
    write_csv(sweep_dir / "design.csv", table)
    by_row = {record["row_name"]: record for record in table}
    for run in run_names:
        record = by_row[run]
        config = run_config(base, space, solver_values(space, record), data, timeline, float(record[PREOP_TIME_FACTOR]))
        write_config(config, sweep_dir / "configs" / f"{run}.json")
    return sweep_dir


def read_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """A config mapping without its '_' comment entries (what
    ``read_config`` returns for a file; the volume paths are absolute
    already)."""
    return {key: value for key, value in config.items() if not key.startswith("_")}


def analysed_qois(prefixes: Sequence[str], threshold_mode: str) -> list[str]:
    """The analysed QoI columns: per session the threshold QoIs, the
    profiled ones, in sampled mode the fixed-pair ones, and the analysed
    field QoIs; then the per-run QoIs."""
    per_session = [*THRESHOLD_QOIS, *STAR_QOIS]
    if threshold_mode == "sampled":
        per_session += FIXED_QOIS
    per_session += FIELD_ANALYSED_QOIS
    return [f"{prefix}{name}" for prefix in prefixes for name in per_session] + list(RUN_QOIS)


def session_qoi_names(threshold_mode: str) -> list[str]:
    """Every QoI column computed per session, in column order."""
    names = [*THRESHOLD_QOIS, *STAR_QOIS]
    if threshold_mode == "sampled":
        names += FIXED_QOIS
    return names + FIELD_QOIS


def qoi_columns(prefixes: Sequence[str], threshold_mode: str) -> list[str]:
    """The qoi.csv columns."""
    return [
        "row_name",
        "run_name",
        "index",
        "row",
        "matrix",
        "success",
        *ROW_THRESHOLD_COLUMNS,
        *RUN_QOIS,
        *(f"{prefix}{name}" for prefix in prefixes for name in session_qoi_names(threshold_mode)),
        *CARRIED_COLUMNS,
    ]


# --- runs ---


def session_snapshot_days(moments: Mapping[str, float], dt: float) -> dict[str, float]:
    """
    The days the run records its session snapshots on: for every session
    the last step end at least half a step before the scan moment t,
    m dt = floor(t / dt - 1/2) dt, so that the frame holds the state
    before any event of the scan day (an event at t fires in the step
    containing it, which ends at or after t) and the state dtype's
    rounding of the step ends cannot move it. This is the atlas's
    ``snapshot_days`` rule applied to the moment itself instead of the
    end of the day (its t_end = offset + 1), hence the shift by one day.
    """
    return snapshot_days(0.0, {name: float(t) - 1.0 for name, t in moments.items()}, dt)


def snapshot_file(session_id: str) -> str:
    """runs/<run>/<session>_cell_density.nii.gz ("ses-03" -> "ses03_cell_density.nii.gz")."""
    return session_id.replace("-", "") + SNAPSHOT_SUFFIX


def format_snapshots(moments: Mapping[str, float]) -> str:
    """The --snapshots argument of run-one: "ses-01=137.3,ses-02=152.3"."""
    return ",".join(f"{name}={float(t)!r}" for name, t in moments.items())


def parse_snapshots(text: str) -> dict[str, float]:
    """The inverse of ``format_snapshots``; the moments must be finite and
    nonnegative, the names unique."""
    moments: dict[str, float] = {}
    for item in (part.strip() for part in text.split(",") if part.strip()):
        name, _, value = item.partition("=")
        name = name.strip()
        if not name or name in moments:
            raise ValueError(f"--snapshots: bad or repeated session name in {text!r}.")
        moment = float(value)
        if not (np.isfinite(moment) and moment >= 0):
            raise ValueError(f"--snapshots: {name}: the moment must be finite and nonnegative, got {value!r}.")
        moments[name] = moment
    return moments


def _save_session_snapshots(run_dir: Path, result: Any, days: Mapping[str, float], affine: NDArray) -> dict[str, float]:
    """
    Save the recorded session frames (the atlas's ``_save_snapshots``
    with the session file names): for every session whose requested day
    the solver recorded, the frame as run_dir/<session>_cell_density.nii.gz
    (float32, rounded for storage, the given affine).

    Returns:
        The recorded day by session id, for the ones saved.
    """
    recorded_days = np.asarray(result.snapshot_times, dtype=np.float64)
    frames = result.time_series["cell_density"]
    recorded: dict[str, float] = {}
    for name, day in days.items():
        matches = np.flatnonzero(np.isclose(recorded_days, day, rtol=1e-9, atol=1e-9))
        if matches.size == 0:
            continue
        image = nib.Nifti1Image(round_field(frames[int(matches[0])]), affine)
        image.set_data_dtype(np.float32)
        nib.save(image, str(run_dir / snapshot_file(name)))
        recorded[name] = float(recorded_days[int(matches[0])])
    return recorded


def run_one(config_path: str | Path, run_dir: str | Path, snapshots: Mapping[str, float]) -> int:
    """
    Solve one run config into run_dir: the time step is resolved first
    (``StuppFKPPSolver.resolve_time_stepping`` of the config), the
    session snapshot days follow (``session_snapshot_days``), the solver
    is built with them as snapshot_times and solved, the frames are saved
    per session, then ``Result.save`` writes config.json, result.json and
    the final field (without the initial state; every field rounded for
    storage) and timeline.json records dt, the moments and the recorded
    days.

    Returns:
        0 on success, 1 if the solve reports a failure.
    """
    run_dir = Path(run_dir).resolve()
    config = read_config(config_path, solver=StuppFKPPSolver)
    missing = [key for key in DERIVED_VOLUME_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"run config {config_path} lacks {missing}: the patient's cavity and dose map must be set.")
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = StuppFKPPSolver(config)
    n_steps, dt = probe.resolve_time_stepping()
    del probe
    days = session_snapshot_days(snapshots, dt)
    horizon = n_steps * dt
    late = {name: day for name, day in days.items() if day > horizon + 1e-9}
    if late:
        raise ValueError(f"the snapshots {late} lie beyond the horizon {horizon:g}.")
    config["snapshot_times"] = sorted(set(days.values())) if days else None
    solver = StuppFKPPSolver(config)
    result = solver.solve()
    result.initial_state = {}  # the seed is not kept (design.csv holds its voxel)
    result.final_state = {key: round_field(value) for key, value in result.final_state.items()}
    affine = np.eye(4) if result.affine is None else np.asarray(result.affine, dtype=np.float64)
    recorded = {} if result.snapshot_times is None else _save_session_snapshots(run_dir, result, days, affine)
    result.time_series = None  # written above, not by Result.save
    result.save(run_dir)
    write_json(
        run_dir / TIMELINE_FILE,
        {
            "dt": result.dt,
            "n_steps": result.n_steps,
            "dt_resolved_before_solve": dt,
            "resection_time": float(config["resection_time"]),
            "stopping_time": float(config["resection_time"]) + float(config["time_after_resection"]),
            "snapshots": {
                name: {
                    "moment": float(snapshots[name]),
                    "requested_day": days[name],
                    "recorded_day": recorded.get(name),
                    "file": snapshot_file(name) if name in recorded else None,
                }
                for name in snapshots
            },
        },
    )
    status = "ok" if result.success else f"FAILED: {result.error}"
    print(
        f"solve: {status} (final_time={result.final_time:g}, n_steps={result.n_steps}, dt={result.dt}, "
        f"wall={result.wall_time_s:.1f} s); snapshots recorded: {len(recorded)}/{len(snapshots)}",
        flush=True,
    )
    if result.dt is not None and abs(result.dt - dt) > 1e-9 * dt:
        print(f"WARNING: the solve's step {result.dt:g} differs from the resolved {dt:g}.", flush=True)
    return 0 if result.success else 1


def run_subprocess(sweep_dir: Path, name: str, gpu: str | None, snapshots: Mapping[str, float]) -> dict[str, Any]:
    """
    Run one design point in its own process (``run-one``) and read its
    result record back (STATUS_COLUMNS; wall_time_s is the subprocess's).
    """
    run_dir = sweep_dir / "runs" / name
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-one",
        "--config",
        str(sweep_dir / "configs" / f"{name}.json"),
        "--run-dir",
        str(run_dir),
        "--snapshots",
        format_snapshots(snapshots),
    ]
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if gpu is None:
        env["JAX_PLATFORMS"] = "cpu"
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.perf_counter()
    with open(sweep_dir / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    own = {"run_name": name, "exit_code": code, "wall_time_s": round(time.perf_counter() - start, 3)}
    saved = run_records(run_dir)
    return {key: own[key] if key in own else saved.get(key) for key in STATUS_COLUMNS}


def run_snapshots(spec: Mapping[str, Any], design_record: Mapping[str, Any]) -> dict[str, float]:
    """The snapshot moments of a run: t_pre + offset per session, from the
    run's preop_time and spec.json's timeline."""
    t_pre = float(design_record[PREOP_TIME_FACTOR])
    return {sid: t_pre + float(offset) for sid, offset in spec["timeline"]["snapshot_offsets"].items()}


def distinct_runs(design: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The distinct runs of a design table, by run name, with the design
    record of the row the run is named after (the representative)."""
    runs: dict[str, Mapping[str, Any]] = {}
    for record in design:
        if record["run_name"] == record["row_name"]:
            runs[record["run_name"]] = record
    missing = sorted({record["run_name"] for record in design} - set(runs))
    if missing:
        raise ValueError(f"design.csv: the runs {missing[:5]}... have no representative row.")
    return runs


def run_sweep(sweep_dir: str | Path, gpus: Sequence[str], jobs_per_gpu: int = 1) -> dict[str, int]:
    """
    Run every distinct run that is not done yet, one subprocess per run,
    on the given GPU slots (a thread per slot pulling from a queue; the
    atlas's run loop with the patient's runs and snapshots).

    Returns:
        Counts: skipped (already successful), ok, failed.
    """
    sweep_dir = Path(sweep_dir)
    spec = read_json(sweep_dir / "spec.json")
    runs = distinct_runs(read_csv(sweep_dir / "design.csv"))
    pending: list[str] = []
    counts = {"skipped": 0, "ok": 0, "failed": 0}
    for name in runs:
        run_dir = sweep_dir / "runs" / name
        if run_is_done(run_dir):
            counts["skipped"] += 1
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)  # a partial run directory is redone
            pending.append(name)
    slots: list[str | None] = [gpu for gpu in gpus for _ in range(jobs_per_gpu)] or [None] * max(1, jobs_per_gpu)
    queue: Queue[str] = Queue()
    for name in pending:
        queue.put(name)
    print(
        f"{len(pending)} runs to do ({counts['skipped']} done already; {len(runs)} distinct runs of "
        f"{spec['n_rows']} rows) on {len(slots)} slot(s): {'CPU' if not gpus else 'GPU ' + ','.join(gpus)}",
        flush=True,
    )
    status_path = sweep_dir / "run_status.csv"
    columns = list(STATUS_COLUMNS)
    if status_path.is_file() and status_path.stat().st_size:
        with open(status_path, newline="", encoding="utf-8") as existing:
            columns = next(csv.reader(existing))
    lock = threading.Lock()
    start = time.perf_counter()
    with open(status_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()

        def worker(gpu: str | None) -> None:
            while True:
                try:
                    name = queue.get_nowait()
                except Empty:
                    return
                record = run_subprocess(sweep_dir, name, gpu, run_snapshots(spec, runs[name]))
                with lock:
                    writer.writerow({key: _csv_cell(record.get(key)) for key in columns})
                    handle.flush()
                    counts["ok" if record["success"] else "failed"] += 1
                    done = counts["ok"] + counts["failed"]
                    status = "ok" if record["success"] else f"FAILED ({record['error']})"
                    print(f"  {name}: {status} in {record['wall_time_s']:.0f} s ({done}/{len(pending)})", flush=True)

        threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    print(
        f"runs: {counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped, "
        f"{(time.perf_counter() - start) / 60:.1f} min",
        flush=True,
    )
    return counts


# --- quantities of interest ---


@dataclass(frozen=True)
class Reference:
    """
    One session's reference masks.

    Attributes:
        session: The session id.
        core: Labels CORE_LABELS, the cavity excluded.
        whole: Labels WHOLE_LABELS, the cavity excluded.
        valid: The voxels kept: all but the session's label-4 voxels
            (all voxels for the pre-op session).
        n_cavity: The excluded voxels.
    """

    session: str
    core: NDArray
    whole: NDArray
    valid: NDArray
    n_cavity: int


def reference_masks(segmentation: NDArray, preop: bool, session: str = "") -> Reference:
    """
    The reference masks of a segmentation: the pre-op one relabelled
    4 -> 3 and nothing excluded; a post-op one with its label-4 voxels
    excluded from the masks (and, through ``valid``, from the model).
    """
    segmentation = np.asarray(segmentation)
    if preop:
        segmentation = relabel_preop(segmentation)
        valid = np.ones(segmentation.shape, dtype=bool)
    else:
        valid = segmentation != LABEL_CAVITY
    return Reference(
        session=session,
        core=np.isin(segmentation, CORE_LABELS) & valid,
        whole=np.isin(segmentation, WHOLE_LABELS) & valid,
        valid=valid,
        n_cavity=int((~valid).sum()),
    )


def load_reference(path: str | Path, preop: bool, session: str = "") -> Reference:
    """``reference_masks`` of a segmentation file."""
    segmentation, _ = load_segmentation(path)
    return reference_masks(segmentation, preop, session)


def dice(a: NDArray, b: NDArray) -> float:
    """The Dice coefficient of two boolean masks: NaN when both are
    empty, 0 when one is."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n_a, n_b = int(a.sum()), int(b.sum())
    if n_a == 0 and n_b == 0:
        return float("nan")
    if n_a == 0 or n_b == 0:
        return 0.0
    return 2.0 * int((a & b).sum()) / (n_a + n_b)


def surface(mask: NDArray) -> NDArray:
    """The surface voxels of a mask: those with a 6-neighbour outside it
    (the array border counts as outside)."""
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)


def surface_distances(a: NDArray, b: NDArray, zooms: Sequence[float]) -> NDArray:
    """The distances in mm from the surface voxels of a to the nearest
    surface voxel of b (a Euclidean distance transform of b's surface);
    empty for an empty mask."""
    surface_a, surface_b = surface(a), surface(b)
    if not surface_a.any() or not surface_b.any():
        return np.empty(0, dtype=np.float64)
    distance = distance_transform_edt(~surface_b, sampling=np.asarray(zooms, dtype=np.float64))
    return np.asarray(distance[surface_a], dtype=np.float64)


def symmetric_surface_distances(a: NDArray, b: NDArray, zooms: Sequence[float]) -> tuple[float, float]:
    """
    (msd, hd95) of two masks: the mean and the 95th percentile of the
    surface distances of both directions pooled (``surface_distances``);
    both NaN when either mask is empty.
    """
    if not np.any(a) or not np.any(b):
        return float("nan"), float("nan")
    distances = np.concatenate([surface_distances(a, b, zooms), surface_distances(b, a, zooms)])
    return float(distances.mean()), float(np.percentile(distances, 95))


def threshold_pairs() -> list[tuple[float, float]]:
    """The profiled threshold grid: every (core, edema) pair of
    THRESHOLD_GRID_CORE x THRESHOLD_GRID_EDEMA with edema < core, in
    grid order."""
    return [(core, edema) for core in THRESHOLD_GRID_CORE for edema in THRESHOLD_GRID_EDEMA if edema < core]


def crop_box(density: NDArray, reference: Reference, min_threshold: float) -> tuple[slice, slice, slice]:
    """The bounding box of the field at or above the smallest threshold in
    use and the reference whole mask: every mask of the session's QoIs
    lies inside it, so the QoIs computed within it are exact (a mask on a
    box face has background beyond it). The whole grid when both are
    empty."""
    inside = (np.asarray(density) >= float(min_threshold)) | reference.whole
    if not inside.any():
        return tuple(slice(0, n) for n in inside.shape)  # type: ignore[return-value]
    box = []
    for axis in range(3):
        along = np.flatnonzero(inside.any(axis=tuple(a for a in range(3) if a != axis)))
        box.append(slice(int(along[0]), int(along[-1]) + 1))
    return tuple(box)  # type: ignore[return-value]


def _region_masks(
    density: NDArray, reference: Reference, core_threshold: float, edema_threshold: float
) -> dict[str, tuple[NDArray, NDArray]]:
    """(model, reference) masks per region at the given thresholds, the
    reference's excluded voxels removed from the model."""
    return {
        "core": ((density >= float(core_threshold)) & reference.valid, reference.core),
        "whole": ((density >= float(edema_threshold)) & reference.valid, reference.whole),
    }


def threshold_qois(
    density: NDArray,
    reference: Reference,
    zooms: Sequence[float],
    core_threshold: float,
    edema_threshold: float,
    distances: bool = True,
) -> dict[str, float]:
    """
    The threshold-dependent QoIs of one session at one threshold pair
    (THRESHOLD_QOIS): per region the Dice, log10(V_model + dV),
    log10(V_model + dV) - log10(V_reference + dV) (NaN for an empty
    reference), msd and hd95 (``symmetric_surface_distances``; skipped,
    NaN, with distances=False).

    Args:
        density: The session's field (any box holding every mask).
        reference: The session's masks on the same box.
        zooms: Voxel size per axis in mm.
        core_threshold: Model core = density >= it.
        edema_threshold: Model whole = density >= it.
        distances: Whether the surface distances are computed.
    """
    voxel_volume = float(np.prod(np.asarray(zooms, dtype=np.float64)))
    out: dict[str, float] = {}
    for region, (model, ref) in _region_masks(density, reference, core_threshold, edema_threshold).items():
        n_model, n_ref = int(model.sum()), int(ref.sum())
        out[f"dice_{region}"] = dice(model, ref)
        out[f"log10_V_{region}"] = float(np.log10(voxel_volume * (n_model + 1)))
        out[f"log_vol_ratio_{region}"] = (
            out[f"log10_V_{region}"] - float(np.log10(voxel_volume * (n_ref + 1))) if n_ref else float("nan")
        )
        msd, hd95 = symmetric_surface_distances(model, ref, zooms) if distances else (float("nan"), float("nan"))
        out[f"msd_{region}"] = msd
        out[f"hd95_{region}"] = hd95
    return out


def profiled_qois(density: NDArray, reference: Reference) -> dict[str, float]:
    """
    The profiled-threshold QoIs of one session (STAR_QOIS): the Dice of
    both regions on the grid of ``threshold_pairs`` (dice_core depends
    on the core threshold only, dice_whole on the edema threshold only,
    so each is evaluated once per threshold), the pair maximising
    dice_core + dice_whole (the first in grid order on a tie; a region
    whose Dice is NaN on the whole grid, the model and the reference both
    empty at every threshold, is left out of the objective; an empty
    reference against a nonempty model is a Dice of 0, not NaN) and the
    two Dice at that pair.
    """
    dice_core = {t: dice((density >= t) & reference.valid, reference.core) for t in THRESHOLD_GRID_CORE}
    dice_whole = {t: dice((density >= t) & reference.valid, reference.whole) for t in THRESHOLD_GRID_EDEMA}
    use_core = any(np.isfinite(v) for v in dice_core.values())
    use_whole = any(np.isfinite(v) for v in dice_whole.values())
    best_pair: tuple[float, float] | None = None
    best = -np.inf
    if use_core or use_whole:
        for core, edema in threshold_pairs():
            objective = (dice_core[core] if use_core else 0.0) + (dice_whole[edema] if use_whole else 0.0)
            if objective > best:
                best, best_pair = objective, (core, edema)
    if best_pair is None:
        return {name: float("nan") for name in STAR_QOIS}
    core, edema = best_pair
    return {
        "dice_star_core": dice_core[core] if use_core else float("nan"),
        "dice_star_whole": dice_whole[edema] if use_whole else float("nan"),
        "core_threshold_star": core,
        "edema_threshold_star": edema,
    }


def field_qois(density: NDArray, zooms: Sequence[float], seed_voxel: Sequence[int], wm: NDArray) -> dict[str, float]:
    """The atlas's threshold-free QoIs of a field (FIELD_QOIS and
    voxel_volume, from ``compute_qois``)."""
    qois = compute_qois(density, zooms, seed_voxel, wm)
    return {name: qois[name] for name in (*FIELD_QOIS, "voxel_volume")}


_REFERENCE_CACHE: dict[str, Reference] = {}


def _cached_reference(path: str, preop: bool, session: str) -> Reference:
    """A session's reference masks, cached per process."""
    if path not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[path] = load_reference(path, preop, session)
    return _REFERENCE_CACHE[path]


def session_records(
    density: NDArray,
    reference: Reference,
    zooms: Sequence[float],
    seed_voxel: Sequence[int],
    wm: NDArray,
    row_thresholds: Sequence[tuple[float, float]],
    threshold_mode: str,
    min_threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    The QoIs of one session field for the rows sharing its run: the
    run-level QoIs (STAR_QOIS, in sampled mode the fixed-pair FIXED_QOIS,
    FIELD_QOIS and voxel_volume) and, per row, the THRESHOLD_QOIS at that
    row's (core, edema) pair; everything but the field QoIs on the crop
    box (``crop_box``).

    Returns:
        (shared, per_row), unprefixed names.
    """
    box = crop_box(density, reference, min_threshold)
    cropped = density[box]
    cropped_reference = Reference(
        reference.session, reference.core[box], reference.whole[box], reference.valid[box], reference.n_cavity
    )
    shared = profiled_qois(cropped, cropped_reference)
    if threshold_mode == "sampled":
        fixed = threshold_qois(cropped, cropped_reference, zooms, *FIXED_THRESHOLDS)
        shared.update({f"{name}{FIXED_SUFFIX}": value for name, value in fixed.items()})
    shared.update(field_qois(density, zooms, seed_voxel, wm))
    per_row = [threshold_qois(cropped, cropped_reference, zooms, core, edema) for core, edema in row_thresholds]
    return shared, per_row


def run_qoi_records(
    sweep_dir: Path,
    run_name: str,
    rows: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    wm: NDArray,
    wm_zooms: Sequence[float],
    threshold_mode: str,
    min_threshold: float,
) -> list[dict[str, Any]]:
    """
    The qoi.csv records of the rows sharing one run: the design
    bookkeeping, the row thresholds (the fixed pair in profiled mode, the
    sampled pair in sampled mode), the carried columns of result.json
    and, for a successful run with every session field, the session QoIs
    under the session prefixes and dice_mean_core. A run without a
    session's field is recorded as failed for every row.

    Args:
        sessions: spec.json's patient sessions (id, label, segmentation)
            in order.
    """
    run_dir = sweep_dir / "runs" / run_name
    saved = run_records(run_dir)
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "row_name": row["row_name"],
            "run_name": run_name,
            "index": int(row["index"]),
            "row": int(row["row"]),
            "matrix": row["matrix"],
            "success": False,
        }
        if threshold_mode == "sampled":
            record["core_threshold"] = float(row[CORE_THRESHOLD_FACTOR])
            record["edema_threshold"] = float(row[EDEMA_THRESHOLD_COLUMN])
        else:
            record["core_threshold"], record["edema_threshold"] = FIXED_THRESHOLDS
        record.update({key: saved.get(key) for key in ("final_time", "n_steps", "dt", "wall_time_s")})
        records.append(record)
    if not saved["success"]:
        return records
    fields = {s["id"]: run_dir / snapshot_file(s["id"]) for s in sessions}
    missing = [sid for sid, path in fields.items() if not path.is_file()]
    if missing:
        for record in records:
            record["error"] = f"missing snapshot field(s) {missing}"
        return records
    seed_voxel = tuple(int(rows[0][f"seed_voxel_{ijk}"]) for ijk in "ijk")
    row_thresholds = [(float(r["core_threshold"]), float(r["edema_threshold"])) for r in records]
    dice_core_postop: list[list[float]] = [[] for _ in records]
    for session in sessions:
        prefix = session["prefix"]
        reference = _cached_reference(session["segmentation"], session["label"] == LABEL_PREOP, session["id"])
        density, zooms = _load_field(fields[session["id"]], wm.shape, wm_zooms)
        shared, per_row = session_records(
            density, reference, zooms, seed_voxel, wm, row_thresholds, threshold_mode, min_threshold
        )
        for record, own, collected in zip(records, per_row, dice_core_postop):
            record.update({f"{prefix}{name}": value for name, value in shared.items() if name != "voxel_volume"})
            record.update({f"{prefix}{name}": value for name, value in own.items()})
            record["voxel_volume"] = shared["voxel_volume"]
            if session["label"] != LABEL_PREOP and np.isfinite(own["dice_core"]):
                collected.append(own["dice_core"])
    for record, collected in zip(records, dice_core_postop):
        record["dice_mean_core"] = float(np.mean(collected)) if collected else float("nan")
        record["success"] = True
    return records


def _qoi_job(job: tuple[str, str, list[dict[str, Any]], list[dict[str, Any]], str, str, float]) -> list[dict[str, Any]]:
    """Worker of ``qoi_table``."""
    sweep_dir, run_name, rows, sessions, wm_path, threshold_mode, min_threshold = job
    wm, zooms = _load_wm(wm_path)
    return run_qoi_records(Path(sweep_dir), run_name, rows, sessions, wm, zooms, threshold_mode, min_threshold)


def spec_sessions(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sessions of a spec.json record with their QoI prefixes."""
    return [{**s, "prefix": s["id"].replace("-", "") + "_"} for s in spec["patient"]["sessions"]]


def min_threshold_in_use(spec: Mapping[str, Any]) -> float:
    """The smallest threshold any QoI of the design applies (the crop
    box's threshold): the grid's and the fixed pair's minima and, in
    sampled mode, the smallest sampled edema threshold."""
    thresholds = spec["thresholds"]
    values = [min(thresholds["grid_edema"]), min(thresholds["grid_core"]), thresholds["fixed"]["edema"], thresholds["fixed"]["core"]]
    if spec["threshold_mode"] == "sampled":
        factors = thresholds["factors"]
        values.append(float(factors[CORE_THRESHOLD_FACTOR]["min"]) * float(factors[EDEMA_RATIO_FACTOR]["min"]))
    return float(min(values))


def qoi_table(sweep_dir: str | Path, workers: int = 1) -> list[dict[str, Any]]:
    """
    The QoIs of every design row of a sweep directory, in design order
    (one job per distinct run, the rows sharing it evaluated together).
    """
    sweep_dir = Path(sweep_dir)
    spec = read_json(sweep_dir / "spec.json")
    design = read_csv(sweep_dir / "design.csv")
    wm_path = str(read_json(sweep_dir / "base_config.json")["white_matter_pbmap"])
    sessions = spec_sessions(spec)
    min_threshold = min_threshold_in_use(spec)
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for record in design:
        rows_by_run.setdefault(record["run_name"], []).append(record)
    jobs = [
        (str(sweep_dir), run, rows, sessions, wm_path, spec["threshold_mode"], min_threshold)
        for run, rows in rows_by_run.items()
    ]
    records: list[dict[str, Any]] = []
    done = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(_qoi_job, jobs, chunksize=2):
                records.extend(batch)
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(jobs)} runs read", flush=True)
    else:
        for job in jobs:
            records.extend(_qoi_job(job))
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} runs read", flush=True)
    records.sort(key=lambda r: r["index"])
    return records


def qoi_summary(
    records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    status_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """
    The qoi_summary.json record: the row and run counts, the mean wall
    time of a run-one subprocess, per session the NaN counts (Dice with
    both masks empty, empty model masks at the row thresholds, empty
    references, NaN log_vol_ratio, extinct fields) and per analysed QoI
    ``response_accounting``.
    """
    successes = [r for r in records if _is_success(r)]
    n_blocks, size = int(spec["N"]), int(spec["block_size"])
    per_session: dict[str, Any] = {}
    for session in spec_sessions(spec):
        prefix = session["prefix"]
        entry: dict[str, Any] = {}
        for region in REGIONS:
            entry[f"n_nan_dice_{region}"] = sum(1 for r in successes if not np.isfinite(as_float(r.get(f"{prefix}dice_{region}"))))
            entry[f"n_zero_dice_{region}"] = sum(1 for r in successes if as_float(r.get(f"{prefix}dice_{region}")) == 0)
            entry[f"n_nan_log_vol_ratio_{region}"] = sum(
                1 for r in successes if not np.isfinite(as_float(r.get(f"{prefix}log_vol_ratio_{region}")))
            )
            entry[f"n_nan_msd_{region}"] = sum(1 for r in successes if not np.isfinite(as_float(r.get(f"{prefix}msd_{region}"))))
            entry[f"n_nan_dice_star_{region}"] = sum(
                1 for r in successes if not np.isfinite(as_float(r.get(f"{prefix}dice_star_{region}")))
            )
        entry["n_extinct"] = sum(
            1 for r in successes if as_float(r.get(f"{prefix}mass")) <= MASS_FLOOR_VOXELS * as_float(r.get("voxel_volume"))
        )
        entry["n_nan_mass_weighted"] = sum(1 for r in successes if not np.isfinite(as_float(r.get(f"{prefix}centroid_drift"))))
        per_session[session["id"]] = entry
    return {
        "_note": ACCOUNTING_NOTE,
        "n_rows": len(records),
        "n_runs": spec["n_runs"],
        "n_success": len(successes),
        "n_failed": len(records) - len(successes),
        "mean_run_wall_time_s": mean_run_wall_time(status_records),
        "block_size": size,
        "n_blocks_total": n_blocks,
        "threshold_mode": spec["threshold_mode"],
        "thresholds": spec["thresholds"],
        "mass_floor_voxels": MASS_FLOOR_VOXELS,
        "n_nan_dice_mean_core": sum(1 for r in successes if not np.isfinite(as_float(r.get("dice_mean_core")))),
        "per_session": per_session,
        "per_qoi": {qoi: response_accounting(records, qoi, n_blocks, size) for qoi in spec["analysed_qois"]},
    }


# --- Sobol' analysis ---


def analyze_sweep(
    sweep_dir: str | Path, n_bootstrap: int = DEFAULT_N_BOOTSTRAP, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> dict[str, dict[str, Any]]:
    """
    Analyse every analysed QoI of a sweep directory with the atlas's
    ``analyze_response``: sobol.csv, sobol_summary.json and the figures
    (the atlas's ``analyze_sweep`` over spec.json's analysed_qois).
    """
    sweep_dir = Path(sweep_dir)
    spec = read_json(sweep_dir / "spec.json")
    names: list[str] = list(spec["factor_names"])
    qoi_records = read_csv(sweep_dir / "qoi.csv")
    analysed: list[str] = list(spec["analysed_qois"])
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    n_blocks, size = int(spec["N"]), int(spec["block_size"])
    summary: dict[str, Any] = {"_note": ACCOUNTING_NOTE}
    summary.update({key: spec[key] for key in ("N", "k", "k_dyn", "factor_names", "cheap_factors", "threshold_mode", "salib_version")})
    summary.update(
        second_order=False, block_size=size, n_blocks_total=n_blocks, n_bootstrap=int(n_bootstrap), seed=int(seed),
        analysed_qois=analysed, qois={},
    )
    for qoi in analysed:
        accounting = response_accounting(qoi_records, qoi, n_blocks, size)
        print(accounting_line(qoi, accounting), flush=True)
        values = qoi_values(qoi_records, qoi)
        result = analyze_response(values, names, n_blocks, False, n_bootstrap, seed)
        if result is None:
            print(f"  {qoi}: fewer than two complete blocks, skipped", flush=True)
            summary["qois"][qoi] = {**accounting, "skipped": True}
            continue
        results[qoi] = result
        rows.extend(sobol_rows(qoi, names, result))
        summary["qois"][qoi] = qoi_summary_entry(names, result, accounting)
        ranking = ", ".join(summary["qois"][qoi]["ranking_ST"][:3])
        print(f"  {qoi}: {result['n_blocks_used']} blocks, sum S1 = {summary['qois'][qoi]['sum_S1']:.2f}, top ST: {ranking}", flush=True)
    write_csv(sweep_dir / "sobol.csv", rows, SOBOL_COLUMNS)
    write_json(sweep_dir / "sobol_summary.json", summary)
    make_figures(sweep_dir / "figures", results, names, spec, read_csv(sweep_dir / "design.csv"), qoi_records)
    return results


# --- command line ---


def _add_design_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE), help="search-space JSON")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="base config JSON")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="parent of the sweep directory")
    parser.add_argument("--name", required=True, help="sweep directory name")
    parser.add_argument("--log2-n", type=int, default=DEFAULT_LOG2_N, help=f"N = 2 ** log2_n base points (default {DEFAULT_LOG2_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_DESIGN_SEED, help="seed of the Sobol' sequence")
    parser.add_argument(
        "--threshold-mode", choices=THRESHOLD_MODES, default=DEFAULT_THRESHOLD_MODE,
        help=f"profiled: thresholds on a fixed grid, k = 12; sampled (default): the two cheap threshold factors in the design, k = 14, runs deduplicated (default {DEFAULT_THRESHOLD_MODE})",
    )
    parser.add_argument("--patient", default=DEFAULT_PATIENT, help="subject id")
    parser.add_argument("--patient-root", default=str(DEFAULT_PATIENT_ROOT), help="processed data root")
    parser.add_argument("--session-labels", default=str(DEFAULT_SESSION_LABELS), help="session labels tsv")
    parser.add_argument(
        "--sessions", default=DEFAULT_SESSIONS,
        help=f"later sessions analysed: a range '02-08' or a list 'ses-02,ses-03'; must hold the post-op session (default {DEFAULT_SESSIONS})",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="comma-separated CUDA device ids; '' = CPU only")
    parser.add_argument("--jobs-per-gpu", type=int, default=DEFAULT_JOBS_PER_GPU, help="slots per GPU (CPU: parallel workers)")


def _add_qoi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=DEFAULT_QOI_WORKERS, help="processes reading the fields")


def _add_analyze_args(parser: argparse.ArgumentParser, seed_flag: str) -> None:
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="bootstrap resamples")
    parser.add_argument(seed_flag, dest="bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="bootstrap seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_design_args(commands.add_parser("design", help="check the patient data, build the timeline, sample the design"))
    run = commands.add_parser("run", help="run the distinct runs (resumable)")
    run.add_argument("--sweep-dir", required=True)
    _add_run_args(run)
    one = commands.add_parser("run-one", help="solve one run config into a run directory")
    one.add_argument("--config", required=True)
    one.add_argument("--run-dir", required=True)
    one.add_argument("--snapshots", default="", help="session snapshot moments as ses-01=<day>,...; the run pass passes spec.json's")
    qoi = commands.add_parser("qoi", help="compute the QoIs of the finished runs")
    qoi.add_argument("--sweep-dir", required=True)
    _add_qoi_args(qoi)
    analyze = commands.add_parser("analyze", help="Sobol' indices and figures")
    analyze.add_argument("--sweep-dir", required=True)
    _add_analyze_args(analyze, "--seed")
    everything = commands.add_parser("all", help="design, run, qoi and analyze in one go")
    _add_design_args(everything)
    _add_run_args(everything)
    _add_qoi_args(everything)
    _add_analyze_args(everything, "--bootstrap-seed")
    return parser


def design_command(args: argparse.Namespace) -> Path:
    """The design subcommand: on the CPU (JAX_PLATFORMS=cpu while the
    solver is built; the previous value is restored for the runs)."""
    previous = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        sweep_dir = make_design(
            args.search_space,
            args.config,
            args.output_dir,
            args.name,
            args.log2_n,
            args.seed,
            args.threshold_mode,
            args.patient,
            args.patient_root,
            args.session_labels,
            args.sessions,
        )
    finally:
        if previous is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = previous
    spec = read_json(sweep_dir / "spec.json")
    print(f"design directory: {sweep_dir}")
    print(
        f"threshold mode {spec['threshold_mode']}: N = {spec['N']} (2^{spec['log2_n']}), k = {spec['k']} "
        f"(k_dyn = {spec['k_dyn']}, cheap: {spec['cheap_factors'] or 'none'}), {spec['n_rows']} rows -> "
        f"{spec['n_runs']} distinct solves (expected N (k_dyn + 2) = {spec['dedup']['n_runs_expected']}); "
        f"seed {spec['seed']}, SALib {spec['salib_version']}"
    )
    print(f"factors: {', '.join(spec['factor_names'])}")
    for name, group in spec["derived_groups"].items():
        print(f"derived group {name}: {', '.join(group['factors'])} -> {', '.join(group['derives'])}")
        for key, value in group.items():
            if key.endswith("_range"):
                print(f"  implied {key[: -len('_range')]} in [{value[0]:.4g}, {value[1]:.4g}]")
    time_step = {key: value for key, value in spec["time_step"].items() if value is not None}
    print(f"time step: {time_step} (raised to the solver's stability estimate where that is stricter)")
    low, high = spec["chemo_log_kill_range"]
    print(
        f"chemotherapy within the run: {len(spec['timeline']['chemo_offsets'])} TMZ days, total dose "
        f"{spec['chemo_total_dose']:g} mg/m^2, total log kill kill * dose / decay in [{low:.3g}, {high:.3g}]"
    )
    print(
        f"radiotherapy: {spec['timeline']['n_rt_fractions']} fractions within the run, dose map maximum "
        f"{spec['rt_dose_max_gy']:.4g} Gy total ({spec['rt_dose_per_fraction_max_gy']:.3g} Gy per fraction)"
    )
    print(f"analysed QoIs: {len(spec['analysed_qois'])}")
    return sweep_dir


def qoi_command(sweep_dir: Path, workers: int) -> None:
    """The qoi subcommand: qoi.csv and qoi_summary.json."""
    spec = read_json(sweep_dir / "spec.json")
    records = qoi_table(sweep_dir, workers)
    columns = qoi_columns([s["prefix"] for s in spec_sessions(spec)], spec["threshold_mode"])
    write_csv(sweep_dir / "qoi.csv", records, columns)
    status_path = sweep_dir / "run_status.csv"
    status_records = read_csv(status_path) if status_path.is_file() else []
    summary = qoi_summary(records, spec, status_records)
    write_json(sweep_dir / "qoi_summary.json", summary)
    print(
        f"qoi.csv: {summary['n_success']}/{summary['n_rows']} successful rows ({summary['n_runs']} distinct runs); "
        f"mean wall time {summary['mean_run_wall_time_s']} s per run",
        flush=True,
    )
    for session, entry in summary["per_session"].items():
        print(f"  {session}: " + ", ".join(f"{key} {value}" for key, value in entry.items() if value), flush=True)
    for qoi, accounting in summary["per_qoi"].items():
        if accounting["n_runs_nan"] or accounting["n_runs_failed"]:
            print(accounting_line(qoi, accounting), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-one":
        try:
            return run_one(args.config, args.run_dir, parse_snapshots(args.snapshots))
        except Exception:  # noqa: BLE001 - the log gets the traceback, the parent the exit code
            traceback.print_exc()
            return 1
    if args.command == "design":
        design_command(args)
        return 0
    if args.command == "run":
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        counts = run_sweep(args.sweep_dir, gpus, args.jobs_per_gpu)
        return 1 if counts["failed"] else 0
    if args.command == "qoi":
        qoi_command(Path(args.sweep_dir), args.workers)
        return 0
    if args.command == "analyze":
        analyze_sweep(args.sweep_dir, args.n_bootstrap, args.bootstrap_seed)
        return 0
    sweep_dir = design_command(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    counts = run_sweep(sweep_dir, gpus, args.jobs_per_gpu)
    qoi_command(sweep_dir, args.workers)
    analyze_sweep(sweep_dir, args.n_bootstrap, args.bootstrap_seed)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
