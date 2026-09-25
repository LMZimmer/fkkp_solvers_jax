#!/usr/bin/env python
"""CMA-ES fit of the Stupp-protocol forward model, fisher_kpp_jax.StuppFKPPSolver,
to ONE real patient of the SAILOR cohort: nine parameters are adjusted so that
the forward run agrees with the patient's longitudinal tumour segmentations,
with the covariance matrix adaptation evolution strategy (CMA-ES) of the
glioma inverse fitting tool (GIFT,
https://github.com/jonasw247/glioma-inverse-fitting-tool, branch dtiStuff,
file cmaes.py, "derived from implementation by Sergey Litvinov"), ported
into this script (``CMAES``) with one deliberate change, the bound handling
(below).

The patient data, the sessions, the clinical timeline, the reference masks,
the snapshot moments, the seedable voxels and the Dice machinery are those
of scripts/patient_sensitivity_analysis.py (the sensitivity analysis, SA)
and are imported from it, together with the atlas script's derivations
(scripts/sensitivity_analysis.py); nothing of the two is copied. What the
fit adds is below.

Objective. loss = 1 - J on the unit cube [0, 1]^9. One evaluation is one
StuppFKPPSolver run recording one field u_s per session s (the pre-op
session and the later sessions of the design, ``select_sessions``), and J
is the agreement of the fields with the sessions' references
(``session_references``: cavity-corrected masks, the session's own cavity
voxels excluded from the model and the reference as in the SA). Per
session and core threshold tau_c of THRESHOLD_GRID_CORE (0.30..0.85):
dice_core_s(tau_c) = dice((u_s >= tau_c) & valid_s, core_s); per edema
threshold tau_e of THRESHOLD_GRID_EDEMA (0.10..0.60): dice_whole_s(tau_e)
against the whole-tumour reference; both on the session's ``crop_box``
(exact). The thresholds are PROFILED, one pair for all sessions:
  --loss core (default)  J = max over tau_c of the weighted mean over the
                         objective's sessions of dice_core_s(tau_c). The
                         edema threshold is profiled separately for
                         reporting only, over the grid values below the
                         profiled core threshold (the pairs rule of the
                         SA); it does not enter the loss.
  --loss core+whole      J = max over the pairs (tau_c, tau_e) of
                         ``threshold_pairs`` (tau_e < tau_c) of the mean of
                         the two regions' weighted means, dice_core at
                         tau_c and dice_whole at tau_e (the SA's
                         ``profiled_qois`` rule with weights).
The objective's sessions are --sessions (default 01-08: the pre-op session
ses-01 and the later sessions; the pre-op session and every requested
later session are simulated and recorded, and only the requested ones
enter the sums), weighted by --session-weights (ses-01=1,ses-02=1,...;
default 1 each). A session whose Dice is NaN at a threshold (the model and
the reference both empty, the SA's ``dice`` convention) is dropped from
that threshold's mean with its weight (the mean is renormalised over the
sessions with a finite Dice); a region whose mean is NaN at every
threshold is left out of the core+whole objective; the number of
objective sessions with a NaN Dice at the profiled pair is recorded
(n_nan_sessions). Ties in the argmax go to the first grid value (grid
order: core ascending, then edema ascending). A failed solve
(Result.success False, an exception, a non-finite field, a missing
snapshot) gets loss = 1 and its error string, GIFT's convention; an
objective that is NaN (every session's Dice NaN everywhere) gets loss = 1
as well and an error string. Per evaluation the record holds the profiled
pair, J, loss, per session the Dice of both regions at the pair
(<ses>_dice_core_star, <ses>_dice_whole_star), log_vol_ratio_core at the
pair (the SA's ``threshold_qois`` without distances) and the Dice at the
SA's fixed pair 0.6 / 0.3 (<ses>_dice_core_fixed, <ses>_dice_whole_fixed)
for comparability with the sweeps.

Parametrisation (--search-space, default
fisher_kpp_jax/search_spaces/sailor_patient_fit_search_space.json, read by
this script's own ``read_fit_search_space``: keys starting with '_' are
comments, an entry {"min", "max", "scale"} ("log" or "linear", mapped with
the atlas's ``transform_factor``) is a fitted factor, any other entry is a
fixed value: a solver parameter is written into every run config,
seed_peak_density is consumed by the seed derivation; "solver" must be
StuppFKPPSolver; the factor order in the file is the coordinate order of
the unit cube). The nine factors and the derivations, per evaluation
(``FitProblem.derive``):
  growth        white_matter_diffusivity = v lambda / 2 and rho = v / (2 lambda)
                from v = front_speed_mm_per_day and lambda = front_width_mm
                (``growth_parameters``).
  seed shape    gaussian_seed_diffusion_time = sigma^2 / 2 and
                gaussian_seed_mass = c_peak (4 pi tau)^(3/2) from
                sigma = seed_sigma_mm and the fixed c_peak = seed_peak_density
                (``seed_parameters``); the peak is checked against the base
                config's gaussian_seed_floor and the clip at 1 at start-up,
                and gaussian_seed_scale must be 1.
  clock         preop_time = growth_length_mm / front_speed_mm_per_day
                (days), clamped to [preop_time_min, preop_time_max] of
                --preop-time-range (default 7,200); preop_time_clamped
                records whether the clamp was active. The timeline is the
                SA's at that preop_time (``Timeline.model_days``:
                resection_time, time_after_resection, rt_times,
                chemo_times, chemo_doses and the session moments
                t_pre + offset_s).
  seed position seed_bbox_x/y/z in [0, 1] map onto the seedable voxels of
                ``patient_seed_geometry`` (pre-op core, labels 1 and 3 after
                4 -> 3, with wm + gm >= min_tissue_fraction): with lo, hi
                the inclusive index bounds of the seedable voxels per axis,
                the point p = lo + u (hi - lo) is rounded to the nearest
                voxel (numpy rounding, half to even); a voxel that is not
                seedable is replaced by the nearest seedable voxel in
                Euclidean voxel distance (scipy's cKDTree over the seedable
                indices, built once); seed_voxel_i/j/k and seed_snapped are
                recorded. The config receives
                gaussian_seed_{x,y,z}_fraction = (v + 0.5) / n as
                ``project_seeds`` writes them (int(fraction n) recovers v;
                checked for every seedable voxel at start-up).
  direct        rt_alpha and chemo_kill_rate are written as they are.
Fixed values (the search-space file's, overriding the base config's):
seed_peak_density 0.6, rt_alpha_beta_ratio 8 Gy and diffusivity_ratio 10,
which the SA found inert on the agreement QoIs; chemo_decay_rate 9.24
(confounded with chemo_kill_rate); gaussian_seed_scale 1. Everything else
comes from the base config (--config, default
fisher_kpp_jax/configs/StuppFKPPSolver.json) with the patient's tissue
maps, resection_cavity (the post-op session's label 4) and rt_dose set as
the SA's ``run_config`` sets them (its signature takes a SearchSpace, so
the assembly is repeated here with the same keys, ``FitProblem.config_of``).
The unit-cube <-> physical mapping is ``FitSpace.to_physical`` /
``FitSpace.to_unit`` (pure; to_unit is the inverse of transform_factor) and
the config assembly ``FitProblem.evaluation_config(u, resolution_factor)``,
so that starts are taken from sweep rows and the best configuration is
rebuilt from its logged unit vector.

Forward evaluation (in a worker, ``evaluate_point``). One solve per
evaluation, no growth stage: the cavity and the dose map are the
patient's. The time step: n_steps is FIXED per resolution factor,
n_steps = ceil(steps_per_day (preop_time_max + last session offset)) with
--steps-per-day (default 12, the base config's value), and dt and
steps_per_day are None in the config, so the solver uses
dt = stopping_time / n_steps <= 1/12 day (the horizon is
preop_time + last session offset). n_steps and the number of snapshot
slots are jit-static arguments of the solver's time scan
(fisher_kpp_jax.operators._run_time_scan), so a fixed n_steps means one
compilation per worker and resolution factor instead of one per distinct
horizon; the price is that dt differs slightly between evaluations and
from the sweeps' 1/12 day. The design-time check builds a StuppFKPPSolver
at the stiffest corner (largest diffusivity, longest horizon) per factor
and confirms with ``resolve_time_stepping`` that the solver keeps n_steps
(it raises the count only where its stability estimate is stricter); if
it raises it, the raised count is used and logged (spec.json
time_stepping). An evaluation whose solve reports another n_steps is
recorded as such (its snapshots then miss their days and the evaluation
fails). Snapshots: the session moments t_pre + offset_s mapped with
``session_snapshot_days`` to the last step end at least half a step before
the moment (the SA's run-one rule), written into the config as
snapshot_times so that the config alone reproduces the frames; the
sessions' days are distinct (checked), so the slot count equals the
number of sessions. Resolution: resolution_factor from the schedule
(below), 1.0 by default; the recorded frames come back on the
full-resolution grid (``_assemble_result`` upsamples them with
``_upsample_to``), so the Dice is always on the 1 mm segmentation grid.
Fields: every session frame is rounded for storage (``round_field``)
before the masks are taken, as the SA rounds its fields, the metrics are
computed, and the fields are discarded. NOTHING is written per
evaluation; the main process appends one row to evaluations.csv per
result.

Execution (``WorkerPool``): multiprocessing "spawn", one persistent worker
process per entry of --gpus (comma-separated CUDA device ids;
--jobs-per-gpu, default 1, starts several per device; '' starts CPU
workers, for plumbing checks only). A worker's environment is set before
it starts (it inherits it, so the variables are in place before jax is
imported, which jax 0.11 needs: JAX_PLATFORMS and JAX_COMPILATION_CACHE_DIR
are read at import): CUDA_VISIBLE_DEVICES=<id>,
XLA_PYTHON_CLIENT_PREALLOCATE=false and
JAX_COMPILATION_CACHE_DIR=<output-dir>/<name>/jax_cache. Each worker loads
the references, reports its JAX backend and device (a GPU worker on the
CPU backend is reported as a WARNING: a shell whose LD_LIBRARY_PATH breaks
the CUDA plugin makes jax fall back to the CPU silently), then loops on
the job queue: (eval_id, u, resolution_factor, save_dir) -> a flat result
record on the result queue. An exception in an evaluation becomes a
failed record (loss 1, the error string), never a hang; a worker process
that dies is detected by the main loop (it polls is_alive while waiting
with a timeout) and aborts the run with a message naming the worker; None
on the job queue stops a worker. The main process runs no JAX operation
and never initialises a JAX backend (it imports jax through
fisher_kpp_jax, as the SA scripts do; the design-time checks run in a
separate CPU-only subprocess, and the final re-solve runs in a worker on
--gpu), so it never takes GPU memory. Population size: --popsize, default
4 x the number of workers (16 on 4 GPUs) so that every generation fills
the workers; "gift" gives GIFT's default 4 + floor(3 ln N) = 10; an
integer sets it. The wall time of the first evaluation of every worker
(it includes the compilation) is reported separately (first_on_worker in
evaluations.csv, workers in summary.json). --dry-run evaluates the starts
once (one generation of K points, rows in dry_run/evaluations.csv, tag
dry_run), prints the per-evaluation wall times and exits: the plumbing
and timing check to run before a fit (it creates the directory; the fit
is then started with --resume, which finds no restart and begins at
restart 0).

CMA-ES (``CMAES``, ask/tell): GIFT's update equations and constants.
lambda = popsize, mu = lambda // 2, weights log((lambda + 1) / 2) - log(i + 1)
normalised, mueff = 1 / sum w^2, cc = (4 + mueff / N) / (N + 4 + 2 mueff / N),
cs = (mueff + 2) / (N + mueff + 5), c1 = 2 / ((N + 1.3)^2 + mueff),
cmu = min(1 - c1, 2 (mueff - 2 + 1 / mueff) / ((N + 2)^2 + mueff)),
damps = 1 + 2 max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs,
chiN = sqrt(2) Gamma((N + 1) / 2) / Gamma(N / 2). Sampling
x = m + sigma sqrtC z, z ~ N(0, I); the members sorted by loss;
m = sum w_i x_i over the mu best; ps = (1 - cs) ps + sqrt(cs (2 - cs) mueff) sum w_i z_i;
sigma *= exp(cs / damps (|ps| / chiN - 1)); Cmu = sum w_i y_i y_i^T with
y = sqrtC z; the stall test (N + 1) |ps|^2 < 2 N (N + 3) (1 - (1 - cs)^(2 gen))
(gen the 1-based generation number) with the two branches as written in
GIFT: pc = (1 - cc) pc + sqrt(cc (2 - cc) mueff) sum w_i y_i and
C = (1 - c1 - cmu) C + c1 pc pc^T + cmu Cmu when it holds, else
pc = (1 - cc) pc and C = (1 - c1 - cmu) C + c1 (pc pc^T + cc (2 - cc) C) + cmu Cmu
(sigma is updated before C, as in GIFT). Differences to GIFT's cmaes.py:
(1) bound handling, the ONE deliberate deviation: after sampling, x is
clipped to [0, 1], the clipped point is evaluated, and the step is
REPAIRED so that every update sees the same point, y = (x_clip - m) / sigma
and z = invsqrtC y, used in the mean, path and covariance updates (GIFT
evaluates the clipped point but updates the paths and the covariance with
the unclipped z and y, which degenerates the distribution on a boundary
optimum); (2) numpy.random.Generator (seeded, --seed, one stream per
restart) instead of random.gauss; (3) the symmetric eigendecomposition
C = B diag(d^2) B^T (numpy.linalg.eigh of the symmetrised C) gives
sqrtC = B diag(d) B^T and invsqrtC = B diag(1 / d) B^T instead of
scipy.linalg.sqrtm, eigenvalues below EIGENVALUE_FLOOR floored so that
invsqrtC exists; (4) float64 numpy sums instead of math.fsum; (5) ties in
the loss are broken by member order (a stable sort) rather than by GIFT's
tuple comparison; (6) the start x0 itself is evaluated first (generation
0, tag start) so that the incumbent exists from the first generation on
and a restart is never worse than its start. Restarts run sequentially,
each a fresh CMAES from its own start with --sigma0 (default 0.1; a
comma-separated list gives one value per start) and --generations
(default 100; a list likewise); --min-sigma stops a restart when
sigma max(d) < min_sigma (default 0 = off). The optimizer state (m, sigma,
C, ps, pc, gen, the rng state, the incumbent and its loss) is
checkpointed to restart_<k>/state.npz after every generation; --resume
continues an interrupted run from the last complete generation of the
last incomplete restart (the evaluations.csv rows of the incomplete
generation are discarded and redone; completed restarts are kept), with
the starts of starts.json and the settings checked against spec.json.
Resolution schedule (--resolution-schedule, default 0:1.0 = full
resolution throughout): GIFT's format by generation fraction, e.g.
0:0.5,0.8:1.0 = factor 0.5 from the start and 1.0 from 80 % of the
restart's generations on (generation g runs at the factor of the last
entry whose fraction is at most (g - 1) / generations). At every switch
sigma is reset to the restart's sigma0, the incumbent is re-evaluated at
the new factor (one extra evaluation, tag incumbent_reeval) and the
incumbent is reset to it, so bests are never compared across factors;
every row carries its factor, and a restart's best is the incumbent of
its final factor.

Starts. --init-from-sweep <sweep-dir> (default DEFAULT_SWEEP_DIR, a
patient_sensitivity_analysis.py sweep of sub-01) with --top K (default 5):
the sweep's design.csv, qoi.csv and spec.json are read, its patient must
be --patient and the objective's sessions must be among its sessions.
Runs are ranked by a PROXY of the objective: the qoi rows are grouped by
run_name, per run the maximum over its rows of the weighted mean (the
fit's weights, NaN dropped) over the objective's sessions of
<ses>_dice_core is taken, the runs are sorted descending and the top K
runs with DISTINCT starts are kept (runs of one Saltelli block that
differ only in parameters the fit fixes, or in the cheap thresholds,
convert to the same start; the later ones are skipped and recorded). The proxy is the closest quantity qoi.csv offers:
in a sampled-mode sweep every row of a run carries its own sampled core
threshold, so the maximum over the rows profiles the threshold over the
sampled values (not the fit's grid), and the Dice are the sweep's (14-day
adjuvant cycles before 2026-09-17, recorded in the sweep's spec.json; the
fit's timeline uses the SA's current constants), so the proxy is not the
fit's objective at that point; the start is evaluated with the objective
at generation 0. Each run's design row is converted: v, lambda, sigma
from the row, growth_length_mm = v preop_time, the seed voxel
seed_voxel_i/j/k mapped back into box coordinates ((v - lo) / (hi - lo)),
rt_alpha and chemo_kill_rate; the values are clipped into the fit's
ranges (recorded) and mapped with to_unit. The row's seed_peak_density,
diffusivity_ratio and rt_alpha_beta_ratio are not carried (fixed in the
fit; recorded). If the sweep directory does not exist and no
--init-values is given the script exits with an error; with
--init-values it warns and uses the manual start. --no-sweep-init
disables the sweep starts. --init-values name=value,... (physical units,
every fitted factor named) adds one manual start after the sweep starts.
--no-sweep-init without --init-values gives one start at the unit-cube
centre (the seed at the box centre). starts.json records every start
(source, physical and unit values, proxy). The subcommand starts prints
the resolved starts and exits (no GPU).

Output layout (--output-dir, default DEFAULT_OUTPUT_DIR; --name required;
nothing is written outside <output-dir>/<name>/):
  spec.json            patient, sessions, timeline (the SA's record), label
                       conventions, data_checks, protocol, the search space
                       (source, factors, fixed values), the objective
                       settings, the parametrisation (clamp range, seed box,
                       formulas), the CMA-ES settings, time_stepping
                       (n_steps, dt range and the design-time check per
                       resolution factor), workers, argv, created
  search_space.json    copy of the file used
  base_config.json     the config of the unit-cube centre with the
                       patient's volumes and its timeline (the design-time
                       checks ran on its corner variants; no solve)
  starts.json          the resolved starts
  restart_<k>/
    evaluations.csv    one row per evaluation, appended as results arrive
                       (``evaluation_columns``: eval_id, restart,
                       generation, member, worker, resolution_factor,
                       u_<factor> x 9, <factor> x 9,
                       white_matter_diffusivity, rho, gaussian_seed_mass,
                       gaussian_seed_diffusion_time, preop_time,
                       preop_time_clamped, seed_voxel_i/j/k, seed_snapped,
                       success, error, loss, J, core_threshold_star,
                       edema_threshold_star, n_nan_sessions, per session
                       <ses>_dice_core_star, <ses>_dice_whole_star,
                       <ses>_log_vol_ratio_core, <ses>_dice_core_fixed,
                       <ses>_dice_whole_fixed, n_steps, dt,
                       solve_wall_time_s, wall_time_s, first_on_worker,
                       tag; tag is empty, start, incumbent_reeval or
                       dry_run; member is -1 for the start and the
                       re-evaluations)
    trace.npz          per generation: gen, xmean, sigma, C, ps, pc,
                       best_loss, mean_loss, n_failed, resolution_factor
                       (GIFT's trace without wandb)
    state.npz          the resume checkpoint
  dry_run/evaluations.csv   the --dry-run rows
  summary.json         restarts ranked by best loss (best unit and physical
                       values, derived values, thresholds, per-session
                       Dice, n_evaluations, wall time), the overall best
                       (restart, eval_id), the workers, the resolve record
  best/                written by resolve only: config.json (with the
                       evaluation's _fit record), result.json,
                       <ses>_cell_density.nii.gz, final_cell_density.nii.gz
                       as the SA's run-one writes them (``Result.save`` and
                       ``_save_session_snapshots``), timeline.json, and
                       objective.json with the objective recomputed at full
                       resolution next to the logged values
  jax_cache/           the workers' persistent compilation cache

Subcommands: starts (the resolved starts, no GPU); fit (everything:
design-time checks -> starts -> restarts -> summary -> resolve, unless
--no-resolve); resolve --fit-dir <dir> [--restart k] --gpu <id> (rebuild
the config of the overall best, or of restart k's best, from its logged
unit vector with evaluation_config(u, 1.0), re-solve it once at
resolution_factor 1.0 in one worker on --gpu, snapshot the sessions and
write best/; the resolve is recorded in summary.json). Run from the
project root, e.g.:
  python scripts/patient_cmaes_fit.py starts --top 5
  python scripts/patient_cmaes_fit.py fit --name fit_sub01 --gpus 0,1,2,3 --top 5 --dry-run
  python scripts/patient_cmaes_fit.py fit --name fit_sub01 --gpus 0,1,2,3 --top 5 --resume
  python scripts/patient_cmaes_fit.py fit --name fit_sub01 --gpus 0,1,2,3 --top 5
  python scripts/patient_cmaes_fit.py resolve --fit-dir /mnt/Drive4/lucas/stupp_patient_fit/fit_sub01 --gpu 0

Adaptations of the imported code, without changing it: ``parse_sessions``
does not know the pre-op session, so the pre-op id is removed from the
requested ids before ``select_sessions`` (which puts it first) and kept
for the objective; ``run_config`` takes a SearchSpace, so the config is
assembled here with the same keys; the frames of ``Result.time_series``
are matched to the sessions by their recorded day (nearest within half a
step); the final re-solve runs in a worker process rather than in the
main process, so that the main process never initialises JAX; the
protocol doses of a fit directory are taken from its spec.json (the
copied base_config.json holds the patient's shifted schedule, which
``protocol_from_config`` is not meant to parse).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import jax  # noqa: E402  (the workers report their backend; the main process runs no JAX operation)
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver, read_config, write_config  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    ADJUVANT_CYCLE_DAYS,
    CORE_LABELS,
    DEFAULT_PATIENT,
    DEFAULT_PATIENT_ROOT,
    DEFAULT_SESSION_LABELS,
    FIXED_THRESHOLDS,
    LABEL_CAVITY,
    LABEL_CONVENTIONS,
    LABEL_PREOP,
    PATIENT_VOLUME_KEYS,
    SOLVER_NAME,
    THRESHOLD_GRID_CORE,
    THRESHOLD_GRID_EDEMA,
    TIMELINE_FILE,
    TIMELINE_KEYS,
    PatientData,
    Protocol,
    Reference,
    Session,
    Timeline,
    _save_session_snapshots,
    build_timeline,
    check_patient_data,
    crop_box,
    dice,
    distinct_runs,
    format_timeline,
    load_segmentation,
    load_session_references,
    parse_sessions,
    patient_files,
    patient_seed_geometry,
    protocol_from_config,
    read_config_from_mapping,
    read_session_labels,
    relabel_preop,
    select_sessions,
    session_snapshot_days,
    snapshot_file,
    threshold_pairs,
    threshold_qois,
)
from sensitivity_analysis import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_GPUS,
    GROWTH_SPEED_FACTOR,
    GROWTH_WIDTH_FACTOR,
    SEED_PEAK_FACTOR,
    SEED_SIGMA_FACTOR,
    SOLVER_KEY,
    TIME_STEP_KEYS,
    SearchSpaceParameter,
    SeedGeometry,
    _csv_cell,
    _is_success,
    _parse_parameter,
    as_float,
    growth_parameters,
    read_csv,
    read_json,
    round_field,
    seed_parameters,
    transform_factor,
    write_csv,
    write_json,
)

DEFAULT_SEARCH_SPACE = _ROOT / "fisher_kpp_jax" / "search_spaces" / "sailor_patient_fit_search_space.json"
DEFAULT_OUTPUT_DIR = Path("/mnt/Drive4/lucas/stupp_patient_fit")
DEFAULT_SWEEP_DIR = Path("/mnt/Drive4/lucas/stupp_sensitivity_analysis_patient/sa_sub01_2026-09-16")
DEFAULT_SESSIONS = "01-08"  # the objective's sessions, the pre-op one included (``split_sessions``)
DEFAULT_JOBS_PER_GPU = 1
DEFAULT_TOP = 5
DEFAULT_PREOP_TIME_RANGE = "7,200"  # days
DEFAULT_STEPS_PER_DAY = 12.0
DEFAULT_SIGMA0 = "0.1"
DEFAULT_GENERATIONS = "100"
DEFAULT_MIN_SIGMA = 0.0
DEFAULT_SEED = 0
DEFAULT_RESOLUTION_SCHEDULE = "0:1.0"
DEFAULT_POPSIZE = "auto"
POPSIZE_PER_WORKER = 4  # "auto": this many members per worker
LOSS_MODES: tuple[str, ...] = ("core", "core+whole")
DEFAULT_LOSS = "core"
FULL_RESOLUTION = 1.0

# The script factors (not solver parameters) and the script constant.
GROWTH_LENGTH_FACTOR = "growth_length_mm"
SEED_BBOX_FACTORS: tuple[str, ...] = ("seed_bbox_x", "seed_bbox_y", "seed_bbox_z")
SCRIPT_FACTORS: tuple[str, ...] = (
    GROWTH_SPEED_FACTOR,
    GROWTH_WIDTH_FACTOR,
    SEED_SIGMA_FACTOR,
    GROWTH_LENGTH_FACTOR,
    *SEED_BBOX_FACTORS,
)
SCRIPT_CONSTANTS: tuple[str, ...] = (SEED_PEAK_FACTOR,)
DERIVED_SOLVER_KEYS: tuple[str, ...] = (
    "white_matter_diffusivity",
    "rho",
    "gaussian_seed_mass",
    "gaussian_seed_diffusion_time",
)
SEED_FRACTION_KEYS: tuple[str, ...] = tuple(f"gaussian_seed_{axis}_fraction" for axis in "xyz")
# Config entries the script sets per evaluation; a search space may not hold them.
FORBIDDEN_KEYS: tuple[str, ...] = (
    *TIMELINE_KEYS,
    *PATIENT_VOLUME_KEYS,
    *TIME_STEP_KEYS,
    "resolution_factor",
    *DERIVED_SOLVER_KEYS,
    *SEED_FRACTION_KEYS,
)
# The resolved base-config values the parametrisation depends on.
CONSTANT_KEYS: tuple[str, ...] = ("min_tissue_fraction", "gaussian_seed_floor", "gaussian_seed_scale")

# The smallest threshold any metric applies (the crop box's threshold).
MIN_THRESHOLD = float(min(*THRESHOLD_GRID_CORE, *THRESHOLD_GRID_EDEMA, *FIXED_THRESHOLDS))

# evaluations.csv columns besides the factor and session columns.
BOOKKEEPING_COLUMNS: list[str] = ["eval_id", "restart", "generation", "member", "worker", "resolution_factor"]
DERIVED_COLUMNS: list[str] = [
    *DERIVED_SOLVER_KEYS,
    "preop_time",
    "preop_time_clamped",
    "seed_voxel_i",
    "seed_voxel_j",
    "seed_voxel_k",
    "seed_snapped",
]
OUTCOME_COLUMNS: list[str] = [
    "success",
    "error",
    "loss",
    "J",
    "core_threshold_star",
    "edema_threshold_star",
    "n_nan_sessions",
]
SESSION_COLUMNS: list[str] = [
    "dice_core_star",
    "dice_whole_star",
    "log_vol_ratio_core",
    "dice_core_fixed",
    "dice_whole_fixed",
]
TAIL_COLUMNS: list[str] = ["n_steps", "dt", "solve_wall_time_s", "wall_time_s", "first_on_worker", "tag"]
INT_COLUMNS: frozenset[str] = frozenset(
    {"eval_id", "restart", "generation", "member", "worker", "seed_voxel_i", "seed_voxel_j", "seed_voxel_k", "n_nan_sessions", "n_steps"}
)
BOOL_COLUMNS: frozenset[str] = frozenset({"preop_time_clamped", "seed_snapped", "success", "first_on_worker"})
STR_COLUMNS: frozenset[str] = frozenset({"error", "tag"})
TAG_START = "start"
TAG_REEVAL = "incumbent_reeval"
TAG_DRY_RUN = "dry_run"
MEMBER_NONE = -1  # the member column of the start and the re-evaluations

# Files of a restart directory and of the fit directory.
EVALUATIONS_FILE = "evaluations.csv"
TRACE_FILE = "trace.npz"
STATE_FILE = "state.npz"
SPEC_FILE = "spec.json"
STARTS_FILE = "starts.json"
SUMMARY_FILE = "summary.json"
BEST_DIR = "best"
DRY_RUN_DIR = "dry_run"
CACHE_DIR = "jax_cache"
OBJECTIVE_FILE = "objective.json"

# Worker pool.
POLL_SECONDS = 5.0
WORKER_STOP_TIMEOUT_S = 60.0
READY_NOTE_SECONDS = 60.0
# The CPU-only environment of the design-time check subprocess.
DESIGN_CHECK_ENV: dict[str, str | None] = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}

# CMA-ES numerics.
EIGENVALUE_FLOOR = 1e-20


# --- command-line value parsing ---


def parse_preop_time_range(text: str) -> tuple[float, float]:
    """The clamp range of preop_time from "lo,hi" (days): 0 < lo < hi."""
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--preop-time-range must be 'lo,hi' in days, got {text!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError(f"--preop-time-range needs 0 < lo < hi, got {lo!r}, {hi!r}.")
    return lo, hi


def parse_session_weights(text: str | Mapping[str, Any], session_ids: Sequence[str]) -> dict[str, float]:
    """
    The objective's session weights from "ses-01=1,ses-02=1,..." (or a
    mapping): every named session must be one of the objective's sessions,
    an unnamed one weighs 1, no weight is negative and at least one is
    positive.
    """
    given: dict[str, float] = {}
    if isinstance(text, Mapping):
        given = {str(key): float(value) for key, value in text.items()}
    else:
        for item in (part.strip() for part in str(text).split(",") if part.strip()):
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"--session-weights: expected name=value, got {item!r}.")
            given[name.strip()] = float(value)
    unknown = sorted(set(given) - set(session_ids))
    if unknown:
        raise ValueError(f"--session-weights: session(s) {unknown} are not among the objective's {list(session_ids)}.")
    weights = {sid: float(given.get(sid, 1.0)) for sid in session_ids}
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError(f"--session-weights must be finite and nonnegative, got {weights}.")
    if not any(w > 0 for w in weights.values()):
        raise ValueError("--session-weights: at least one weight must be positive.")
    return weights


def parse_float_list(text: str, name: str, n: int) -> list[float]:
    """A comma-separated list of positive floats broadcast to n values
    (one value applies to every start; else one per start)."""
    values = [float(part) for part in str(text).split(",") if part.strip()]
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"{name} must be positive number(s), got {text!r}.")
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name}: {len(values)} values for {n} start(s); give one or one per start.")
    return values


def parse_int_list(text: str, name: str, n: int) -> list[int]:
    """As ``parse_float_list`` for positive integers."""
    values = [int(part) for part in str(text).split(",") if part.strip()]
    if not values or any(v < 1 for v in values):
        raise ValueError(f"{name} must be positive integer(s), got {text!r}.")
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name}: {len(values)} values for {n} start(s); give one or one per start.")
    return values


def parse_gpus(text: str) -> list[str | None]:
    """The GPU ids of --gpus; '' gives one CPU slot (None)."""
    gpus: list[str | None] = [g.strip() for g in str(text).split(",") if g.strip()]
    return gpus or [None]


@dataclass(frozen=True)
class ResolutionSchedule:
    """
    GIFT's resolution schedule by generation fraction: entries
    (fraction, factor), the first at fraction 0; generation g (1-based) of
    a restart with G generations runs at the factor of the last entry
    whose fraction is at most (g - 1) / G.
    """

    entries: tuple[tuple[float, float], ...]

    @classmethod
    def parse(cls, text: str) -> ResolutionSchedule:
        entries: list[tuple[float, float]] = []
        for item in (part.strip() for part in str(text).split(",") if part.strip()):
            fraction, sep, factor = item.partition(":")
            if not sep:
                raise ValueError(f"--resolution-schedule: expected fraction:factor, got {item!r}.")
            entries.append((float(fraction), float(factor)))
        if not entries:
            raise ValueError("--resolution-schedule must name at least one entry.")
        entries.sort()
        fractions = [f for f, _ in entries]
        if fractions[0] != 0.0 or any(not 0 <= f < 1 for f in fractions) or len(set(fractions)) != len(fractions):
            raise ValueError(f"--resolution-schedule: the fractions must start at 0, lie in [0, 1) and be distinct, got {fractions}.")
        if any(not (np.isfinite(r) and 0 < r <= 1) for _, r in entries):
            raise ValueError(f"--resolution-schedule: the factors must lie in (0, 1], got {[r for _, r in entries]}.")
        return cls(tuple(entries))

    @property
    def factors(self) -> list[float]:
        """The distinct factors, ascending."""
        return sorted({factor for _, factor in self.entries})

    def factor_at(self, generation: int, n_generations: int) -> float:
        fraction = (int(generation) - 1) / float(n_generations)
        factor = self.entries[0][1]
        for start, value in self.entries:
            if start <= fraction:
                factor = value
        return factor

    def record(self) -> dict[str, Any]:
        return {
            "entries": [{"fraction": f, "factor": r} for f, r in self.entries],
            "rule": "generation g (1-based) runs at the factor of the last entry whose fraction is at most (g - 1) / generations",
        }


# --- search space and parametrisation ---


@dataclass(frozen=True)
class FitSpace:
    """
    The fit's search-space file (``read_fit_search_space``).

    Attributes:
        factors: The fitted factors by name, in file order (the coordinate
            order of the unit cube).
        overrides: Fixed solver parameters written into every run config.
        constants: Fixed script constants (seed_peak_density).
        source: The file's entries as read, comments included.
        path: The file.
    """

    factors: dict[str, SearchSpaceParameter]
    overrides: dict[str, Any]
    constants: dict[str, float]
    source: dict[str, Any]
    path: Path

    @property
    def names(self) -> list[str]:
        return list(self.factors)

    @property
    def dimension(self) -> int:
        return len(self.factors)

    @property
    def solver_factor_names(self) -> list[str]:
        """The factors that are solver parameters, written directly."""
        return [name for name in self.factors if name not in SCRIPT_FACTORS]

    def check_unit(self, u: NDArray | Sequence[float]) -> NDArray:
        """u as a float64 vector of the space's dimension within [0, 1]."""
        vector = np.asarray(u, dtype=np.float64).ravel()
        if vector.shape != (self.dimension,):
            raise ValueError(f"a unit vector must have {self.dimension} coordinates, got shape {vector.shape}.")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0) or np.any(vector > 1):
            raise ValueError(f"a unit vector must lie in [0, 1]^{self.dimension}, got {vector.tolist()}.")
        return vector

    def to_physical(self, u: NDArray | Sequence[float]) -> dict[str, float]:
        """The physical factor values of a unit-cube point (``transform_factor``)."""
        vector = self.check_unit(u)
        return {
            name: float(transform_factor(vector[i], factor.low, factor.high, factor.scale))
            for i, (name, factor) in enumerate(self.factors.items())
        }

    def to_unit(self, values: Mapping[str, float]) -> NDArray:
        """
        The unit-cube point of physical factor values, the inverse of
        ``transform_factor`` per factor: (x - min) / (max - min) for a
        linear factor, (log10 x - log10 min) / (log10 max - log10 min) for
        a log one. Values outside a range give coordinates outside [0, 1]
        (``clip`` first).
        """
        u = np.empty(self.dimension, dtype=np.float64)
        for i, (name, factor) in enumerate(self.factors.items()):
            if name not in values:
                raise KeyError(f"no value for the factor {name!r}.")
            x = float(values[name])
            if factor.scale == "log":
                if x <= 0:
                    raise ValueError(f"{name}: a log-scaled factor needs a positive value, got {x!r}.")
                u[i] = (np.log10(x) - np.log10(factor.low)) / (np.log10(factor.high) - np.log10(factor.low))
            else:
                u[i] = (x - factor.low) / (factor.high - factor.low)
        return u

    def clip(self, values: Mapping[str, float]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """The values clipped into the ranges and, by factor, the ones that
        were ({"given", "clipped"})."""
        clipped: dict[str, float] = {}
        changed: dict[str, dict[str, float]] = {}
        for name, factor in self.factors.items():
            x = float(values[name])
            y = float(min(max(x, factor.low), factor.high))
            clipped[name] = y
            if y != x:
                changed[name] = {"given": x, "clipped": y}
        return clipped, changed

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "factor_names": self.names,
            "factors": {name: {"min": f.low, "max": f.high, "scale": f.scale} for name, f in self.factors.items()},
            "overrides": dict(self.overrides),
            "constants": dict(self.constants),
            "source": dict(self.source),
        }


def read_fit_search_space(path: str | Path) -> FitSpace:
    """
    Read the fit's search-space file (the module docstring's format).

    Raises:
        ValueError: The file is not a JSON object or names another solver;
            an entry is set per evaluation (FORBIDDEN_KEYS); a script factor
            is not a range or a script constant not a number; an unknown
            key; a missing script factor or constant; a seed box factor not
            linear on [0, 1].
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"search space not found: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    where = f"search space {path}"
    if not isinstance(entries, Mapping):
        raise ValueError(f"{where}: must be a JSON object.")
    if entries.get(SOLVER_KEY) != SOLVER_NAME:
        raise ValueError(f"{where}: {SOLVER_KEY!r} must be {SOLVER_NAME!r}, got {entries.get(SOLVER_KEY)!r}.")
    solver_keys = StuppFKPPSolver.config_keys()
    factors: dict[str, SearchSpaceParameter] = {}
    overrides: dict[str, Any] = {}
    constants: dict[str, float] = {}
    for key, value in entries.items():
        if key.startswith("_") or key == SOLVER_KEY:
            continue
        if key in FORBIDDEN_KEYS:
            raise ValueError(f"{where}: {key} is set per evaluation by the script and may not appear.")
        if isinstance(value, Mapping):
            if key in SCRIPT_CONSTANTS:
                raise ValueError(f"{where}: {key} must be a fixed number, not a range.")
            if key not in SCRIPT_FACTORS and key not in solver_keys:
                raise ValueError(f"{where}: unknown factor {key!r}; a factor is a script factor {list(SCRIPT_FACTORS)} or a StuppFKPPSolver parameter.")
            factors[key] = _parse_parameter(key, value, where)
        elif key in SCRIPT_CONSTANTS:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{where}: {key} must be a finite number, got {value!r}.")
            constants[key] = float(value)
        elif key in SCRIPT_FACTORS:
            raise ValueError(f"{where}: the script factor {key} must be a range {{\"min\", \"max\", \"scale\"}}, got {value!r}.")
        elif key in solver_keys:
            overrides[key] = value
        else:
            raise ValueError(f"{where}: unknown key {key!r}.")
    missing = [name for name in SCRIPT_FACTORS if name not in factors]
    if missing:
        raise ValueError(f"{where}: the script factor(s) {missing} must be ranges.")
    missing_constants = [name for name in SCRIPT_CONSTANTS if name not in constants]
    if missing_constants:
        raise ValueError(f"{where}: the constant(s) {missing_constants} must be given as fixed numbers.")
    for name in SEED_BBOX_FACTORS:
        factor = factors[name]
        if factor.scale != "linear" or factor.low != 0.0 or factor.high != 1.0:
            raise ValueError(f"{where}: {name} must be linear on [0, 1], got {factor}.")
    return FitSpace(factors=factors, overrides=overrides, constants=constants, source=dict(entries), path=path)


@dataclass(frozen=True)
class SeedMap:
    """
    The seed position map of the module docstring: box coordinates in
    [0, 1]^3 -> a seedable voxel.

    Attributes:
        geometry: The seedable voxels (``patient_seed_geometry``).
        lo: The inclusive lower index bound of the seedable voxels per
            axis, int64 (3,).
        hi: The inclusive upper bound.
        tree: A cKDTree over the seedable voxel indices.
    """

    geometry: SeedGeometry
    lo: NDArray
    hi: NDArray
    tree: cKDTree

    @classmethod
    def build(cls, geometry: SeedGeometry) -> SeedMap:
        voxels = np.asarray(geometry.voxels, dtype=np.int64)
        return cls(
            geometry=geometry,
            lo=voxels.min(axis=0),
            hi=voxels.max(axis=0),
            tree=cKDTree(voxels.astype(np.float64)),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.geometry.shape

    def voxel(self, u_xyz: Sequence[float]) -> tuple[tuple[int, int, int], bool]:
        """
        The seed voxel of box coordinates: p = lo + u (hi - lo) rounded to
        the nearest voxel (half to even), replaced by the nearest seedable
        voxel in Euclidean voxel distance when it is not seedable.

        Returns:
            (voxel, snapped).
        """
        u = np.asarray(u_xyz, dtype=np.float64).ravel()
        if u.shape != (3,) or not np.all(np.isfinite(u)) or np.any(u < 0) or np.any(u > 1):
            raise ValueError(f"box coordinates must be three values in [0, 1], got {u.tolist()}.")
        point = self.lo + u * (self.hi - self.lo)
        rounded = np.rint(point).astype(np.int64)
        if self.geometry.mask[tuple(rounded)]:
            return (int(rounded[0]), int(rounded[1]), int(rounded[2])), False
        _, index = self.tree.query(rounded.astype(np.float64))
        nearest = self.geometry.voxels[int(index)]
        return (int(nearest[0]), int(nearest[1]), int(nearest[2])), True

    def fractions(self, voxel: Sequence[int]) -> tuple[float, float, float]:
        """The config fractions (v + 0.5) / n of a voxel, checked to map
        back onto it through the solver's int(fraction n)."""
        out = []
        for v, n in zip(voxel, self.shape, strict=True):
            fraction = (int(v) + 0.5) / n
            if int(fraction * n) != int(v):
                raise RuntimeError(f"the seed fraction {fraction!r} of voxel index {v} on {n} voxels does not map back onto it.")
            out.append(float(fraction))
        return out[0], out[1], out[2]

    def to_bbox(self, voxel: Sequence[int]) -> tuple[float, float, float]:
        """The box coordinates of a voxel, (v - lo) / (hi - lo) per axis
        (0.5 on an axis of one slab), clipped into [0, 1]."""
        out = []
        for v, lo, hi in zip(voxel, self.lo, self.hi, strict=True):
            out.append(float(np.clip((int(v) - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.5)
        return out[0], out[1], out[2]

    def check_fractions(self) -> None:
        """Every seedable voxel's fractions map back onto it."""
        voxels = np.asarray(self.geometry.voxels, dtype=np.int64)
        n = np.asarray(self.shape, dtype=np.float64)
        back = ((voxels + 0.5) / n * n).astype(np.int64)
        if not np.array_equal(back, voxels):
            raise RuntimeError("the (v + 0.5) / n seed fractions do not map back onto every seedable voxel.")

    def record(self) -> dict[str, Any]:
        return {
            "grid_shape": list(self.shape),
            "n_seedable_voxels": self.geometry.n_voxels,
            "bbox_lo_voxel": self.lo.tolist(),
            "bbox_hi_voxel": self.hi.tolist(),
            "bbox_lo_fraction": np.asarray(self.geometry.bbox_lo).tolist(),
            "bbox_hi_fraction": np.asarray(self.geometry.bbox_hi).tolist(),
            "rule": (
                "p = lo + u (hi - lo) in voxel coordinates rounded to the nearest voxel (half to even); a voxel that is "
                "not seedable is replaced by the nearest seedable voxel in Euclidean voxel distance (cKDTree); the "
                "config fractions are (v + 0.5) / n"
            ),
        }


@dataclass(frozen=True)
class Point:
    """
    One evaluated parameter point: the unit vector, the physical factor
    values and everything derived from them (``FitProblem.derive``).
    """

    unit: NDArray
    physical: dict[str, float]
    white_matter_diffusivity: float
    rho: float
    gaussian_seed_mass: float
    gaussian_seed_diffusion_time: float
    preop_time_raw: float
    preop_time: float
    preop_time_clamped: bool
    seed_voxel: tuple[int, int, int]
    seed_snapped: bool
    seed_fractions: tuple[float, float, float]

    def record(self, names: Sequence[str]) -> dict[str, Any]:
        """The evaluations.csv fields of the point (u_<factor>, <factor>,
        DERIVED_COLUMNS)."""
        out: dict[str, Any] = {f"u_{name}": float(self.unit[i]) for i, name in enumerate(names)}
        out.update({name: float(self.physical[name]) for name in names})
        out.update(
            white_matter_diffusivity=self.white_matter_diffusivity,
            rho=self.rho,
            gaussian_seed_mass=self.gaussian_seed_mass,
            gaussian_seed_diffusion_time=self.gaussian_seed_diffusion_time,
            preop_time=self.preop_time,
            preop_time_clamped=self.preop_time_clamped,
            seed_voxel_i=self.seed_voxel[0],
            seed_voxel_j=self.seed_voxel[1],
            seed_voxel_k=self.seed_voxel[2],
            seed_snapped=self.seed_snapped,
        )
        return out


# --- objective ---


def _grid_index(grid: Sequence[float], value: float) -> int:
    """The index of a threshold in a grid (exact up to rounding)."""
    matches = [i for i, t in enumerate(grid) if abs(float(t) - float(value)) < 1e-9]
    if not matches:
        raise ValueError(f"the threshold {value} is not on the grid {list(grid)}.")
    return matches[0]


FIXED_CORE_INDEX = _grid_index(THRESHOLD_GRID_CORE, FIXED_THRESHOLDS[0])
FIXED_EDEMA_INDEX = _grid_index(THRESHOLD_GRID_EDEMA, FIXED_THRESHOLDS[1])


def _first_nanargmax(values: NDArray) -> int | None:
    """The first index of the largest finite value, None without one."""
    finite = np.isfinite(values)
    if not finite.any():
        return None
    best = np.nanmax(values)
    return int(np.flatnonzero(finite & (values >= best))[0])


@dataclass(frozen=True)
class ProfiledObjective:
    """
    The profiled objective of one evaluation.

    Attributes:
        J: The objective (NaN when undefined).
        loss: 1 - J, 1 when J is NaN.
        core_index, edema_index: The grid indices of the profiled pair,
            None where undefined.
        n_nan_sessions: The objective sessions with a NaN Dice at the pair.
        error: Why the objective is undefined, else "".
    """

    J: float
    loss: float
    core_index: int | None
    edema_index: int | None
    n_nan_sessions: int
    error: str = ""

    @property
    def core_threshold(self) -> float:
        return float("nan") if self.core_index is None else float(THRESHOLD_GRID_CORE[self.core_index])

    @property
    def edema_threshold(self) -> float:
        return float("nan") if self.edema_index is None else float(THRESHOLD_GRID_EDEMA[self.edema_index])


@dataclass(frozen=True)
class Objective:
    """
    The loss of the module docstring.

    Attributes:
        mode: "core" or "core+whole" (LOSS_MODES).
        session_ids: The sessions of the sums, in simulation order.
        weights: Their weights.
    """

    mode: str
    session_ids: tuple[str, ...]
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if self.mode not in LOSS_MODES:
            raise ValueError(f"the loss must be one of {LOSS_MODES}, got {self.mode!r}.")
        if not self.session_ids:
            raise ValueError("the objective needs at least one session.")

    def weighted_means(self, dice_by_session: Mapping[str, NDArray]) -> NDArray:
        """
        The weighted mean over the objective's sessions per grid value, a
        session with a NaN Dice dropped from that value's mean with its
        weight; NaN where no session has a finite Dice.
        """
        stack = np.asarray([np.asarray(dice_by_session[sid], dtype=np.float64) for sid in self.session_ids])
        weights = np.asarray([self.weights[sid] for sid in self.session_ids], dtype=np.float64)[:, None]
        finite = np.isfinite(stack)
        numerator = np.where(finite, stack * weights, 0.0).sum(axis=0)
        denominator = (finite * weights).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)

    def profile(self, dice_core: Mapping[str, NDArray], dice_whole: Mapping[str, NDArray]) -> ProfiledObjective:
        """
        The profiled thresholds and J of one evaluation.

        Args:
            dice_core: Per session (at least the objective's) the Dice of
                the core over THRESHOLD_GRID_CORE.
            dice_whole: Likewise over THRESHOLD_GRID_EDEMA.
        """
        mean_core = self.weighted_means(dice_core)
        mean_whole = self.weighted_means(dice_whole)
        core_index: int | None = None
        edema_index: int | None = None
        J = float("nan")
        if self.mode == "core":
            core_index = _first_nanargmax(mean_core)
            if core_index is not None:
                J = float(mean_core[core_index])
                tau_c = THRESHOLD_GRID_CORE[core_index]
                below = np.array([np.isfinite(m) and t < tau_c for t, m in zip(THRESHOLD_GRID_EDEMA, mean_whole)])
                candidates = np.where(below, mean_whole, np.nan)
                edema_index = _first_nanargmax(candidates)
        else:
            best = -np.inf
            for tau_c, tau_e in threshold_pairs():
                i = _grid_index(THRESHOLD_GRID_CORE, tau_c)
                j = _grid_index(THRESHOLD_GRID_EDEMA, tau_e)
                finite = [m for m in (mean_core[i], mean_whole[j]) if np.isfinite(m)]
                if not finite:
                    continue
                value = float(sum(finite) / len(finite))
                if value > best:
                    best, core_index, edema_index = value, i, j
            if core_index is not None:
                J = best
        n_nan = 0
        for sid in self.session_ids:
            nan_core = core_index is not None and not np.isfinite(dice_core[sid][core_index])
            nan_whole = self.mode != "core" and edema_index is not None and not np.isfinite(dice_whole[sid][edema_index])
            n_nan += int(nan_core or nan_whole)
        if not np.isfinite(J):
            return ProfiledObjective(
                float("nan"), 1.0, None, None, len(self.session_ids),
                "objective undefined: the Dice of every objective session is NaN at every threshold (model and reference empty)",
            )
        return ProfiledObjective(J, 1.0 - J, core_index, edema_index, n_nan)

    def record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "loss": "1 - J",
            "sessions": list(self.session_ids),
            "weights": dict(self.weights),
            "grid_core": list(THRESHOLD_GRID_CORE),
            "grid_edema": list(THRESHOLD_GRID_EDEMA),
            "fixed_pair": {"core": FIXED_THRESHOLDS[0], "edema": FIXED_THRESHOLDS[1]},
            "min_threshold": MIN_THRESHOLD,
            "rule": (
                "core: J = max over the core grid of the weighted mean over the objective sessions of dice_core; the edema "
                "threshold is profiled for reporting over the grid values below the profiled core threshold. core+whole: "
                "J = max over the pairs (edema < core) of the mean of the two regions' weighted means. A session whose "
                "Dice is NaN at a threshold (model and reference both empty) is dropped from that threshold's mean with "
                "its weight; a region with a NaN mean at every threshold is left out of core+whole; ties go to the first "
                "grid value; a failed solve or an undefined objective gets loss 1"
            ),
        }


# --- the fit problem: patient, timeline, parametrisation, configs ---


def split_sessions(sessions: Sequence[Session], requested_ids: Sequence[str]) -> tuple[str, list[str]]:
    """
    The pre-op session id (the one row labelled preop) and the requested
    ids without it, the later sessions ``select_sessions`` takes (the SA's
    ``parse_sessions`` does not know the pre-op session).
    """
    preops = [s for s in sessions if s.label == LABEL_PREOP]
    if len(preops) != 1:
        raise ValueError(f"expected one session labelled {LABEL_PREOP}, got {[s.id for s in preops]}.")
    later = [sid for sid in requested_ids if sid != preops[0].id]
    if not later:
        raise ValueError("--sessions must name at least one later session (the post-op one among them).")
    return preops[0].id, later


def n_steps_count(steps_per_day: float, preop_time_max: float, horizon_offset: float) -> int:
    """The fixed step count: ceil(steps_per_day (preop_time_max + horizon_offset))."""
    if not (np.isfinite(steps_per_day) and steps_per_day > 0):
        raise ValueError(f"--steps-per-day must be positive, got {steps_per_day!r}.")
    return int(math.ceil(float(steps_per_day) * (float(preop_time_max) + float(horizon_offset)) - 1e-9))


def resolved_constants(base: Mapping[str, Any], space: FitSpace) -> dict[str, float]:
    """The CONSTANT_KEYS as the solver resolves them: the class default
    config, the base config, the search space's overrides (no solver is
    built; the design-time check confirms the values)."""
    merged = {**StuppFKPPSolver.get_default_config(), **base, **space.overrides}
    missing = [key for key in CONSTANT_KEYS if merged.get(key) is None]
    if missing:
        raise ValueError(f"the base config resolves no value for {missing}.")
    return {key: float(merged[key]) for key in CONSTANT_KEYS}


def build_seed_map(data: PatientData, min_tissue_fraction: float) -> tuple[SeedMap, tuple[float, float, float], NDArray]:
    """
    The seed map of the patient (``patient_seed_geometry`` of the pre-op
    core and the tissue maps), the voxel size of the white-matter map and
    its affine.
    """
    wm_image = nib.load(str(data.tissue["white_matter_pbmap"]))
    wm = np.asarray(wm_image.get_fdata(), dtype=np.float64)
    gm = np.asarray(nib.load(str(data.tissue["gray_matter_pbmap"])).get_fdata(), dtype=np.float64)
    segmentation, _ = load_segmentation(data.preop_segmentation)
    core = np.isin(relabel_preop(segmentation), CORE_LABELS)
    geometry = patient_seed_geometry(wm, gm, core, min_tissue_fraction)
    seed_map = SeedMap.build(geometry)
    seed_map.check_fractions()
    zooms = tuple(float(z) for z in wm_image.header.get_zooms()[:3])
    return seed_map, (zooms[0], zooms[1], zooms[2]), np.asarray(wm_image.affine, dtype=np.float64)


@dataclass(frozen=True)
class FitProblem:
    """
    Everything one evaluation needs besides the point: the base config,
    the search space, the patient, the timeline, the seed map, the
    objective and the time-stepping rule. Picklable (it is sent to the
    workers once).

    Attributes:
        base: The base config (``read_config``).
        space: The search space.
        data: The patient's files and sessions (the pre-op one first).
        timeline: The clinical timeline (``build_timeline``).
        seed_map: The seed position map.
        objective: The loss.
        preop_time_range: The clamp range of preop_time (days).
        steps_per_day: The steps per day the fixed n_steps is built from.
        n_steps: The fixed step count per resolution factor.
        constants: The resolved CONSTANT_KEYS.
        voxel_size_mm: The segmentation grid's voxel size.
        protocol: The protocol doses the timeline was built with.
    """

    base: dict[str, Any]
    space: FitSpace
    data: PatientData
    timeline: Timeline
    seed_map: SeedMap
    objective: Objective
    preop_time_range: tuple[float, float]
    steps_per_day: float
    n_steps: dict[float, int]
    constants: dict[str, float]
    voxel_size_mm: tuple[float, float, float]
    protocol: Protocol

    @property
    def sessions(self) -> tuple[Session, ...]:
        """The simulated sessions, the pre-op one first."""
        return self.data.sessions

    @property
    def prefixes(self) -> list[str]:
        return [s.prefix for s in self.sessions]

    def session_records(self) -> list[dict[str, Any]]:
        """The records ``load_session_references`` takes, with the prefixes."""
        return [
            {**s.record(), "segmentation": str(self.data.segmentation(s)), "prefix": s.prefix}
            for s in self.sessions
        ]

    @property
    def horizon_offset(self) -> int:
        return int(self.timeline.horizon_offset)

    def to_physical(self, u: NDArray | Sequence[float]) -> dict[str, float]:
        return self.space.to_physical(u)

    def to_unit(self, values: Mapping[str, float]) -> NDArray:
        return self.space.to_unit(values)

    def derive(self, values: Mapping[str, float], unit: NDArray | None = None) -> Point:
        """The derived quantities of physical factor values (the module
        docstring's derivations)."""
        physical = {name: float(values[name]) for name in self.space.names}
        growth = growth_parameters(physical[GROWTH_SPEED_FACTOR], physical[GROWTH_WIDTH_FACTOR])
        seed = seed_parameters(self.space.constants[SEED_PEAK_FACTOR], physical[SEED_SIGMA_FACTOR])
        preop_raw = physical[GROWTH_LENGTH_FACTOR] / physical[GROWTH_SPEED_FACTOR]
        lo, hi = self.preop_time_range
        preop = float(min(max(preop_raw, lo), hi))
        voxel, snapped = self.seed_map.voxel([physical[name] for name in SEED_BBOX_FACTORS])
        return Point(
            unit=np.asarray(self.space.to_unit(physical) if unit is None else unit, dtype=np.float64),
            physical=physical,
            white_matter_diffusivity=float(growth["white_matter_diffusivity"]),
            rho=float(growth["rho"]),
            gaussian_seed_mass=float(seed["gaussian_seed_mass"]),
            gaussian_seed_diffusion_time=float(seed["gaussian_seed_diffusion_time"]),
            preop_time_raw=float(preop_raw),
            preop_time=preop,
            preop_time_clamped=preop != preop_raw,
            seed_voxel=voxel,
            seed_snapped=snapped,
            seed_fractions=self.seed_map.fractions(voxel),
        )

    def stopping_time(self, preop_time: float) -> float:
        return float(preop_time) + float(self.horizon_offset)

    def dt_range(self, resolution_factor: float) -> tuple[float, float]:
        """The range of dt = stopping_time / n_steps over the clamp range."""
        n = self.n_steps[float(resolution_factor)]
        return self.stopping_time(self.preop_time_range[0]) / n, self.stopping_time(self.preop_time_range[1]) / n

    def config_of(self, point: Point, resolution_factor: float) -> dict[str, Any]:
        """
        The run config of a point: the base config with the search
        space's fixed values, the derived and direct factor values, the
        seed fractions, the patient's tissue maps, cavity and dose map, the
        timeline at the point's preop_time, the fixed n_steps of the
        resolution factor and the session snapshot days; the "_fit" entry
        records the point (stripped before the solver sees the config,
        ``read_config_from_mapping``).
        """
        factor = float(resolution_factor)
        if factor not in self.n_steps:
            raise ValueError(f"no fixed n_steps for the resolution factor {factor!r}; known: {sorted(self.n_steps)}.")
        n_steps = int(self.n_steps[factor])
        days = self.timeline.model_days(point.preop_time)
        stopping_time = float(days["stopping_time"])
        dt = stopping_time / n_steps
        snapshot_days = session_snapshot_days(days["snapshots"], dt)
        if len(set(snapshot_days.values())) != len(snapshot_days):
            raise RuntimeError(f"two sessions share a snapshot step at dt={dt:g}: {snapshot_days}.")
        config: dict[str, Any] = {
            SOLVER_KEY: self.base.get(SOLVER_KEY, SOLVER_NAME),
            "_fit": {
                "script": "scripts/patient_cmaes_fit.py",
                "unit": {name: float(point.unit[i]) for i, name in enumerate(self.space.names)},
                "physical": dict(point.physical),
                "constants": dict(self.space.constants),
                "preop_time_raw": point.preop_time_raw,
                "preop_time": point.preop_time,
                "preop_time_clamped": point.preop_time_clamped,
                "preop_time_range": list(self.preop_time_range),
                "seed_voxel": list(point.seed_voxel),
                "seed_snapped": point.seed_snapped,
                "resolution_factor": factor,
                "n_steps": n_steps,
                "dt": dt,
                "stopping_time": stopping_time,
                "snapshot_moments": dict(days["snapshots"]),
                "snapshot_days": dict(snapshot_days),
            },
        }
        config.update({key: value for key, value in self.base.items() if key != SOLVER_KEY})
        config.update(self.space.overrides)
        config.update(
            white_matter_diffusivity=point.white_matter_diffusivity,
            rho=point.rho,
            gaussian_seed_mass=point.gaussian_seed_mass,
            gaussian_seed_diffusion_time=point.gaussian_seed_diffusion_time,
        )
        config.update({name: point.physical[name] for name in self.space.solver_factor_names})
        config.update(dict(zip(SEED_FRACTION_KEYS, point.seed_fractions, strict=True)))
        config.update({key: str(path) for key, path in self.data.tissue.items()})
        config["resection_cavity"] = {
            "segmentation": str(self.data.segmentations[self.data.cavity_session]),
            "label": LABEL_CAVITY,
        }
        config["rt_dose"] = str(self.data.dose)
        for key in ("resection_time", "time_after_resection", "rt_times", "chemo_times", "chemo_doses"):
            config[key] = days[key]
        config["resolution_factor"] = factor
        config["n_steps"] = n_steps
        config["dt"] = None
        config["steps_per_day"] = None
        config["snapshot_times"] = sorted(set(snapshot_days.values()))
        return config

    def evaluation_config(self, u: NDArray | Sequence[float], resolution_factor: float) -> dict[str, Any]:
        """The run config of a unit-cube point (``derive`` and ``config_of``)."""
        vector = self.space.check_unit(u)
        return self.config_of(self.derive(self.space.to_physical(vector), vector), resolution_factor)

    def record(self) -> dict[str, Any]:
        """The spec.json parametrisation record."""
        return {
            "factor_names": self.space.names,
            "preop_time_range": list(self.preop_time_range),
            "preop_time": f"{GROWTH_LENGTH_FACTOR} / {GROWTH_SPEED_FACTOR}, clamped to preop_time_range (preop_time_clamped records the clamp)",
            "growth": (
                f"white_matter_diffusivity = {GROWTH_SPEED_FACTOR} * {GROWTH_WIDTH_FACTOR} / 2, "
                f"rho = {GROWTH_SPEED_FACTOR} / (2 {GROWTH_WIDTH_FACTOR})"
            ),
            "seed_shape": (
                f"gaussian_seed_diffusion_time = {SEED_SIGMA_FACTOR}^2 / 2, gaussian_seed_mass = {SEED_PEAK_FACTOR} "
                "(4 pi gaussian_seed_diffusion_time)^(3/2)"
            ),
            "seed_position": self.seed_map.record(),
            "direct": self.space.solver_factor_names,
            "constants": dict(self.constants),
            "seed_peak_density": self.space.constants[SEED_PEAK_FACTOR],
            "steps_per_day": self.steps_per_day,
            "horizon_offset_days": self.horizon_offset,
            "n_steps": {str(factor): n for factor, n in sorted(self.n_steps.items())},
            "n_steps_rule": "ceil(steps_per_day (preop_time_max + horizon_offset)), fixed per resolution factor; dt = stopping_time / n_steps",
            "dt_range": {str(factor): list(self.dt_range(factor)) for factor in sorted(self.n_steps)},
            "voxel_size_mm": list(self.voxel_size_mm),
        }


def build_problem(
    config_path: str | Path,
    search_space_path: str | Path,
    patient: str,
    patient_root: str | Path,
    session_labels: str | Path,
    sessions: str,
    session_weights: str | Mapping[str, Any],
    loss: str,
    preop_time_range: tuple[float, float],
    steps_per_day: float,
    resolution_factors: Sequence[float],
    n_steps: Mapping[float, int] | None = None,
    protocol: Protocol | None = None,
) -> FitProblem:
    """
    Assemble the fit problem: the base config, the search space, the
    patient's sessions and files (``read_session_labels``,
    ``split_sessions``, ``select_sessions``, ``patient_files``), the
    objective, the timeline (the base config's protocol doses unless a
    Protocol is given), the seed map and the fixed step counts (the
    formula unless given). No solve, no JAX operation.
    """
    base = read_config(config_path, solver=StuppFKPPSolver)
    space = read_fit_search_space(search_space_path)
    all_sessions = read_session_labels(session_labels, patient)
    requested = parse_sessions(sessions)
    _, later_ids = split_sessions(all_sessions, requested)
    selected = select_sessions(all_sessions, later_ids)
    data = patient_files(patient_root, patient, selected)
    objective_ids = tuple(s.id for s in selected if s.id in set(requested))
    weights = parse_session_weights(session_weights, objective_ids)
    objective = Objective(loss, objective_ids, weights)
    doses = protocol if protocol is not None else protocol_from_config(base)
    timeline = build_timeline(data.sessions, doses)
    constants = resolved_constants(base, space)
    seed_map, zooms, _ = build_seed_map(data, constants["min_tissue_fraction"])
    factors = sorted({float(f) for f in resolution_factors} | {FULL_RESOLUTION})
    if n_steps is None:
        counts = {factor: n_steps_count(steps_per_day, preop_time_range[1], timeline.horizon_offset) for factor in factors}
    else:
        counts = {float(k): int(v) for k, v in n_steps.items()}
        missing = [factor for factor in factors if factor not in counts]
        if missing:
            raise ValueError(f"no n_steps for the resolution factor(s) {missing}.")
    return FitProblem(
        base=base,
        space=space,
        data=data,
        timeline=timeline,
        seed_map=seed_map,
        objective=objective,
        preop_time_range=(float(preop_time_range[0]), float(preop_time_range[1])),
        steps_per_day=float(steps_per_day),
        n_steps=counts,
        constants=constants,
        voxel_size_mm=zooms,
        protocol=doses,
    )


def protocol_from_spec(spec: Mapping[str, Any]) -> Protocol:
    """The protocol doses of a fit's spec.json (``Protocol.record``)."""
    record = spec["protocol"]
    return Protocol(
        concomitant_dose=float(record["concomitant_dose_mg_m2"]),
        adjuvant_first_dose=float(record["adjuvant_first_cycle_dose_mg_m2"]),
        adjuvant_later_dose=float(record["adjuvant_later_cycles_dose_mg_m2"]),
        base_cycle_days=record.get("base_config_cycle_days"),
    )


def problem_from_fit_dir(fit_dir: Path, spec: Mapping[str, Any]) -> FitProblem:
    """
    Rebuild the problem of a fit directory from its spec.json and its
    copies (search_space.json; base_config.json as the base config: the
    evaluation overwrites every entry it sets), the protocol doses from
    the spec, the step counts from the spec; the rebuilt timeline must
    equal the spec's.
    """
    parametrisation = spec["parametrisation"]
    problem = build_problem(
        fit_dir / "base_config.json",
        fit_dir / "search_space.json",
        spec["patient"]["patient"],
        spec["patient"]["root"],
        spec["session_labels"],
        spec["sessions_requested"],
        spec["objective"]["weights"],
        spec["objective"]["mode"],
        tuple(parametrisation["preop_time_range"]),
        float(parametrisation["steps_per_day"]),
        [float(f) for f in parametrisation["n_steps"]],
        {float(f): int(n) for f, n in parametrisation["n_steps"].items()},
        protocol_from_spec(spec),
    )
    rebuilt = problem.timeline.record()
    for key in ("snapshot_offsets", "chemo_offsets", "chemo_doses", "resection", "rt_fractions"):
        if json.dumps(rebuilt[key], sort_keys=True) != json.dumps(spec["timeline"][key], sort_keys=True):
            raise RuntimeError(f"the rebuilt timeline differs from {fit_dir / SPEC_FILE} in {key}; the schedule constants changed?")
    if problem.space.names != list(spec["parametrisation"]["factor_names"]):
        raise RuntimeError(f"the factor order of {fit_dir / 'search_space.json'} differs from the spec's.")
    return problem


# --- evaluation records ---


def evaluation_columns(names: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    """The evaluations.csv columns (the module docstring)."""
    return [
        *BOOKKEEPING_COLUMNS,
        *(f"u_{name}" for name in names),
        *names,
        *DERIVED_COLUMNS,
        *OUTCOME_COLUMNS,
        *(f"{prefix}{name}" for prefix in prefixes for name in SESSION_COLUMNS),
        *TAIL_COLUMNS,
    ]


def typed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """An evaluation record with typed values (a CSV row's strings turned
    into int / bool / float, None for an empty field; a typed record is
    returned as it is)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            if key in STR_COLUMNS:
                out[key] = value
            elif value == "":
                out[key] = None if key in INT_COLUMNS or key in BOOL_COLUMNS else float("nan")
            elif key in INT_COLUMNS:
                out[key] = int(float(value))
            elif key in BOOL_COLUMNS:
                out[key] = value == "True"
            else:
                out[key] = float(value)
        else:
            out[key] = value
    return out


def read_evaluations(path: Path) -> list[dict[str, Any]]:
    """The typed rows of an evaluations.csv (empty without the file)."""
    if not path.is_file():
        return []
    return [typed_row(row) for row in read_csv(path)]


def empty_record(problem: FitProblem, u: NDArray | Sequence[float]) -> dict[str, Any]:
    """A failed evaluation's record before anything is known: the unit
    vector, loss 1, everything else empty."""
    vector = np.asarray(u, dtype=np.float64).ravel()
    record: dict[str, Any] = {f"u_{name}": float(vector[i]) for i, name in enumerate(problem.space.names)}
    record.update(success=False, error="", loss=1.0, J=float("nan"))
    record.update(core_threshold_star=float("nan"), edema_threshold_star=float("nan"), n_nan_sessions=None)
    for prefix in problem.prefixes:
        record.update({f"{prefix}{name}": float("nan") for name in SESSION_COLUMNS})
    record.update(n_steps=None, dt=None, solve_wall_time_s=None)
    return record


def session_frames(result: Any, snapshot_days: Mapping[str, float], dt: float) -> dict[str, NDArray]:
    """
    The recorded frame of every session (``Result.time_series`` matched to
    the session's requested day, the nearest recorded day within half a
    step), rounded for storage (``round_field``).

    Raises:
        RuntimeError: No frames, a session without a frame, a non-finite
            frame.
    """
    if result.time_series is None or result.snapshot_times is None:
        raise RuntimeError("the solve recorded no snapshots.")
    recorded = np.asarray(result.snapshot_times, dtype=np.float64)
    frames = result.time_series["cell_density"]
    out: dict[str, NDArray] = {}
    for session_id, day in snapshot_days.items():
        if recorded.size == 0:
            raise RuntimeError(f"no frame recorded for {session_id} (day {day:g}).")
        index = int(np.argmin(np.abs(recorded - float(day))))
        if abs(float(recorded[index]) - float(day)) > 0.5 * float(dt) + 1e-9:
            raise RuntimeError(
                f"no frame recorded for {session_id} at day {day:g} (nearest recorded day {recorded[index]:g}, dt {dt:g})."
            )
        frame = np.asarray(frames[index])
        if not np.all(np.isfinite(frame)):
            raise RuntimeError(f"the frame of {session_id} is not finite.")
        out[session_id] = round_field(frame)
    return out


def objective_metrics(
    problem: FitProblem,
    references: Mapping[str, Reference],
    zooms: Sequence[float],
    frames: Mapping[str, NDArray],
) -> dict[str, Any]:
    """
    The objective and the per-session metrics of the session fields (the
    module docstring's Objective): per session the Dice of both regions
    over the grids on the crop box, the profiled pair, J, loss, and per
    session the Dice at the pair, log_vol_ratio_core at the pair and the
    Dice at the fixed pair.
    """
    dice_core: dict[str, NDArray] = {}
    dice_whole: dict[str, NDArray] = {}
    cropped: dict[str, tuple[NDArray, Reference]] = {}
    for session in problem.sessions:
        reference = references[session.id]
        density = np.asarray(frames[session.id])
        box = crop_box(density, reference, MIN_THRESHOLD)
        field = density[box]
        local = Reference(
            reference.session, reference.core[box], reference.whole[box], reference.valid[box],
            reference.n_cavity, reference.n_relabelled,
        )
        dice_core[session.id] = np.array([dice((field >= t) & local.valid, local.core) for t in THRESHOLD_GRID_CORE])
        dice_whole[session.id] = np.array([dice((field >= t) & local.valid, local.whole) for t in THRESHOLD_GRID_EDEMA])
        cropped[session.id] = (field, local)
    profiled = problem.objective.profile(dice_core, dice_whole)
    record: dict[str, Any] = {
        "success": True,
        "error": profiled.error,
        "loss": profiled.loss,
        "J": profiled.J,
        "core_threshold_star": profiled.core_threshold,
        "edema_threshold_star": profiled.edema_threshold,
        "n_nan_sessions": profiled.n_nan_sessions,
    }
    for session in problem.sessions:
        prefix = session.prefix
        core, whole = dice_core[session.id], dice_whole[session.id]
        record[f"{prefix}dice_core_star"] = float(core[profiled.core_index]) if profiled.core_index is not None else float("nan")
        record[f"{prefix}dice_whole_star"] = float(whole[profiled.edema_index]) if profiled.edema_index is not None else float("nan")
        if profiled.core_index is not None and profiled.edema_index is not None:
            field, local = cropped[session.id]
            qois = threshold_qois(field, local, zooms, profiled.core_threshold, profiled.edema_threshold, distances=False)
            record[f"{prefix}log_vol_ratio_core"] = float(qois["log_vol_ratio_core"])
        else:
            record[f"{prefix}log_vol_ratio_core"] = float("nan")
        record[f"{prefix}dice_core_fixed"] = float(core[FIXED_CORE_INDEX])
        record[f"{prefix}dice_whole_fixed"] = float(whole[FIXED_EDEMA_INDEX])
    return record


def save_resolve(run_dir: Path, result: Any, config: Mapping[str, Any], fit: Mapping[str, Any]) -> dict[str, float]:
    """
    Write the re-solved best run as the SA's run-one does: the session
    frames (``_save_session_snapshots``), ``Result.save`` without the
    initial state and with the final field rounded (config.json,
    result.json, final_cell_density.nii.gz), the config with its "_fit"
    record over Result.save's config.json, and timeline.json.

    Returns:
        The recorded day by session id.
    """
    run_dir.mkdir(parents=True, exist_ok=False)
    days = {sid: float(day) for sid, day in fit["snapshot_days"].items()}
    result.initial_state = {}
    result.final_state = {key: round_field(value) for key, value in result.final_state.items()}
    affine = np.eye(4) if result.affine is None else np.asarray(result.affine, dtype=np.float64)
    recorded = {} if result.snapshot_times is None else _save_session_snapshots(run_dir, result, days, affine)
    result.time_series = None
    result.save(run_dir)
    write_config(config, run_dir / "config.json")
    write_json(
        run_dir / TIMELINE_FILE,
        {
            "dt": result.dt,
            "n_steps": result.n_steps,
            "dt_planned": fit["dt"],
            "resection_time": float(config["resection_time"]),
            "stopping_time": float(config["resection_time"]) + float(config["time_after_resection"]),
            "snapshots": {
                sid: {
                    "moment": float(fit["snapshot_moments"][sid]),
                    "requested_day": days[sid],
                    "recorded_day": recorded.get(sid),
                    "file": snapshot_file(sid) if sid in recorded else None,
                }
                for sid in days
            },
        },
    )
    return recorded


def evaluate_point(
    problem: FitProblem,
    references: Mapping[str, Reference],
    zooms: Sequence[float],
    u: NDArray,
    resolution_factor: float,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """
    One forward evaluation (the module docstring's Forward evaluation):
    the config of the point, one StuppFKPPSolver solve, the session
    frames, the metrics; with save_dir the run is written there as well
    (the resolve). Every failure is a record with success False, loss 1
    and the error string.
    """
    record = empty_record(problem, u)
    try:
        vector = problem.space.check_unit(u)
        point = problem.derive(problem.to_physical(vector), vector)
        record.update(point.record(problem.space.names))
        config = problem.config_of(point, resolution_factor)
        fit = config["_fit"]
        solver = StuppFKPPSolver(read_config_from_mapping(config))
        result = solver.solve()
        record.update(n_steps=result.n_steps, dt=result.dt, solve_wall_time_s=result.wall_time_s)
        if not result.success:
            record["error"] = result.error or "solver failure"
            return record
        if result.n_steps != fit["n_steps"]:
            record["error"] = f"the solver used n_steps={result.n_steps}, not the fixed {fit['n_steps']}"
            return record
        frames = session_frames(result, fit["snapshot_days"], float(result.dt))
        record.update(objective_metrics(problem, references, zooms, frames))
        if save_dir is not None:
            recorded = save_resolve(save_dir, result, config, fit)
            if len(recorded) != len(frames):
                record["error"] = f"{len(recorded)} of {len(frames)} session frames saved"
    except Exception as exc:  # noqa: BLE001 - every failure is a failed evaluation, never a dead worker
        record["success"] = False
        record["loss"] = 1.0
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


# --- persistent workers ---


def start_process(ctx: BaseContext, target: Any, args: tuple[Any, ...], env: Mapping[str, str | None]) -> BaseProcess:
    """
    Start a spawned process with the given environment entries set (None
    removes one) in the parent's environment while it starts, so that the
    child inherits them before it imports anything; the parent's
    environment is restored afterwards.
    """
    saved = {key: os.environ.get(key) for key in env}
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        process = ctx.Process(target=target, args=args, daemon=True)
        process.start()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return process


def worker_environment(gpu: str | None, cache_dir: Path) -> dict[str, str | None]:
    """The environment of a worker: its CUDA device (CPU workers hide the
    devices and force the CPU backend), no preallocation, the
    compilation cache directory."""
    return {
        "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
        "JAX_PLATFORMS": "cpu" if gpu is None else None,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
    }


def worker_main(worker_id: int, gpu: str | None, problem: FitProblem, jobs: Any, results: Any) -> None:
    """
    The worker loop: load the references once, report readiness (the
    JAX backend and device), then evaluate jobs (eval_id, u,
    resolution_factor, save_dir) until None arrives. Every message on the
    result queue is (kind, worker_id, payload) with kind "ready", "result"
    or "error" (a failure of the setup; the process then exits).
    """
    label = f"[worker {worker_id}]"
    start = time.perf_counter()
    try:
        references = {r.session: r for r in load_session_references(problem.session_records())}
        zooms = problem.voxel_size_mm
        backend = jax.default_backend()
        device_kind = jax.devices()[0].device_kind
        info = {
            "worker": worker_id,
            "gpu": gpu,
            "backend": backend,
            "device_kind": device_kind,
            "setup_wall_time_s": round(time.perf_counter() - start, 3),
            "pid": os.getpid(),
        }
        results.put(("ready", worker_id, info))
    except Exception:  # noqa: BLE001 - reported to the main process, which aborts
        results.put(("error", worker_id, traceback.format_exc()))
        return
    n_done = 0
    while True:
        job = jobs.get()
        if job is None:
            return
        eval_id, u, factor, save_dir = job
        started = time.perf_counter()
        try:
            record = evaluate_point(
                problem, references, zooms, np.asarray(u, dtype=np.float64), float(factor),
                None if save_dir is None else Path(save_dir),
            )
        except Exception as exc:  # noqa: BLE001 - never a dead worker
            record = empty_record(problem, u)
            record["error"] = f"{type(exc).__name__}: {exc}"
        record.update(
            eval_id=int(eval_id),
            worker=worker_id,
            resolution_factor=float(factor),
            wall_time_s=round(time.perf_counter() - started, 3),
            first_on_worker=n_done == 0,
        )
        n_done += 1
        print(
            f"{label} eval {eval_id}: {'ok' if record['success'] else 'FAILED'} loss={record['loss']:.4f} "
            f"in {record['wall_time_s']:.1f} s" + (f" ({record['error']})" if record["error"] else ""),
            flush=True,
        )
        results.put(("result", worker_id, record))


class WorkerPool:
    """
    The persistent workers of a fit (the module docstring's Execution):
    one spawned process per slot, a job queue and a result queue.

    Args:
        problem: The fit problem, sent to every worker once.
        gpus: One entry per GPU (None for a CPU slot).
        jobs_per_gpu: Slots per entry.
        cache_dir: The workers' JAX compilation cache directory.
    """

    def __init__(self, problem: FitProblem, gpus: Sequence[str | None], jobs_per_gpu: int, cache_dir: Path) -> None:
        if jobs_per_gpu < 1:
            raise ValueError(f"--jobs-per-gpu must be at least 1, got {jobs_per_gpu}.")
        self.ctx = get_context("spawn")
        self.jobs = self.ctx.Queue()
        self.results = self.ctx.Queue()
        self.slots: list[str | None] = [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
        self.info: dict[int, dict[str, Any]] = {}
        self.first_evaluation: dict[int, dict[str, Any]] = {}
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.processes: list[BaseProcess] = [
            start_process(self.ctx, worker_main, (wid, gpu, problem, self.jobs, self.results), worker_environment(gpu, cache_dir))
            for wid, gpu in enumerate(self.slots)
        ]

    @property
    def n_workers(self) -> int:
        return len(self.slots)

    def _check_alive(self, pending: int) -> None:
        for wid, process in enumerate(self.processes):
            if not process.is_alive():
                raise RuntimeError(
                    f"worker {wid} ({'CPU' if self.slots[wid] is None else 'GPU ' + str(self.slots[wid])}) died with exit "
                    f"code {process.exitcode} while {pending} evaluation(s) were pending; the run is aborted (its "
                    "checkpoints allow --resume)."
                )

    def _next_message(self, pending: int, waiting_for: str) -> tuple[str, int, Any]:
        """The next result-queue message, polling the workers' liveness."""
        last_note = time.perf_counter()
        while True:
            try:
                kind, wid, payload = self.results.get(timeout=POLL_SECONDS)
            except Empty:
                self._check_alive(pending)
                if time.perf_counter() - last_note >= READY_NOTE_SECONDS:
                    print(f"  waiting for {waiting_for} ({pending} pending)", flush=True)
                    last_note = time.perf_counter()
                continue
            if kind == "error":
                raise RuntimeError(f"worker {wid} failed during its setup:\n{payload}")
            return kind, wid, payload

    def _note_ready(self, wid: int, info: Mapping[str, Any]) -> None:
        self.info[wid] = dict(info)
        where = "CPU" if info["gpu"] is None else f"GPU {info['gpu']}"
        print(
            f"worker {wid}: {where}, JAX backend {info['backend']} ({info['device_kind']}), setup {info['setup_wall_time_s']:.1f} s",
            flush=True,
        )
        if info["gpu"] is not None and info["backend"] != "gpu":
            print(
                f"WARNING: worker {wid} was given GPU {info['gpu']} but runs on the {info['backend']} backend "
                "(a broken CUDA plugin, e.g. through LD_LIBRARY_PATH, makes JAX fall back to the CPU silently).",
                flush=True,
            )

    def wait_ready(self) -> None:
        """Block until every worker reported readiness."""
        pending = {wid for wid in range(self.n_workers) if wid not in self.info}
        while pending:
            kind, wid, payload = self._next_message(len(pending), "the workers' setup")
            if kind == "ready":
                self._note_ready(wid, payload)
                pending.discard(wid)
            else:
                raise RuntimeError(f"unexpected {kind!r} message from worker {wid} before it was ready.")

    def evaluate(self, jobs: Sequence[tuple[int, NDArray, float, str | None]]) -> Iterator[dict[str, Any]]:
        """
        Dispatch jobs (eval_id, u, resolution_factor, save_dir) and yield
        the result records as they arrive (any order). A worker's first
        record is kept in ``first_evaluation`` (its wall time includes the
        compilation).
        """
        for job in jobs:
            eval_id, u, factor, save_dir = job
            self.jobs.put((int(eval_id), np.asarray(u, dtype=np.float64).tolist(), float(factor), save_dir))
        pending = {int(job[0]) for job in jobs}
        while pending:
            kind, wid, payload = self._next_message(len(pending), "evaluations")
            if kind == "ready":
                self._note_ready(wid, payload)
                continue
            record = dict(payload)
            if record.get("first_on_worker") and wid not in self.first_evaluation:
                self.first_evaluation[wid] = {
                    "eval_id": record["eval_id"],
                    "wall_time_s": record["wall_time_s"],
                    "solve_wall_time_s": record["solve_wall_time_s"],
                    "resolution_factor": record["resolution_factor"],
                }
                print(
                    f"worker {wid}: first evaluation {record['wall_time_s']:.1f} s wall "
                    f"(solve {record['solve_wall_time_s'] if record['solve_wall_time_s'] is not None else float('nan'):.1f} s, "
                    "includes the compilation)",
                    flush=True,
                )
            pending.discard(int(record["eval_id"]))
            yield record

    def record(self) -> list[dict[str, Any]]:
        """The workers' records for summary.json."""
        return [
            {**self.info.get(wid, {"worker": wid, "gpu": gpu}), "first_evaluation": self.first_evaluation.get(wid)}
            for wid, gpu in enumerate(self.slots)
        ]

    def close(self) -> None:
        """Stop the workers (None per worker), terminate the ones that do
        not stop in time."""
        for process in self.processes:
            if process.is_alive():
                try:
                    self.jobs.put(None)
                except (ValueError, OSError):
                    pass
        deadline = time.perf_counter() + WORKER_STOP_TIMEOUT_S
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.perf_counter()))
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for queue in (self.jobs, self.results):
            try:
                queue.close()
            except (ValueError, OSError):
                pass


# --- design-time checks (a CPU-only subprocess) ---


def design_checks(problem: FitProblem, resolution_factors: Sequence[float]) -> dict[str, Any]:
    """
    The design-time checks (no solve): a StuppFKPPSolver of the unit-cube
    centre config confirms the resolved constants (CONSTANT_KEYS) and the
    voxel size; the fixed seed peak is checked against gaussian_seed_floor
    and the clip at 1 and gaussian_seed_scale must be 1; per resolution
    factor a solver at the stiffest corner (the largest diffusivity of
    the space, the longest horizon of the clamp range) resolves its time
    stepping (``resolve_time_stepping``) so that the fixed n_steps is
    confirmed or the raised count reported.

    Returns:
        The spec.json record: centre, seed_peak, time_stepping per factor
        (n_steps_requested, n_steps_resolved, dt_resolved, raised, the
        corner's D, rho and stopping time, the low-resolution grid).
    """
    space = problem.space
    centre_config = problem.evaluation_config(np.full(space.dimension, 0.5), FULL_RESOLUTION)
    solver = StuppFKPPSolver(read_config_from_mapping(centre_config))
    params = solver.params
    for key in CONSTANT_KEYS:
        if float(params[key]) != problem.constants[key]:
            raise RuntimeError(f"the solver resolves {key} = {params[key]!r}, the script assumed {problem.constants[key]!r}.")
    voxel_size = tuple(float(v) for v in params["voxel_size_mm"])
    if not np.allclose(voxel_size, problem.voxel_size_mm, rtol=1e-6, atol=0):
        raise RuntimeError(f"the solver's voxel size {voxel_size} differs from the white-matter map's {problem.voxel_size_mm}.")
    peak = space.constants[SEED_PEAK_FACTOR]
    floor = float(params["gaussian_seed_floor"])
    if not peak > floor:
        raise ValueError(f"{SEED_PEAK_FACTOR} = {peak!r} must exceed the base config's gaussian_seed_floor {floor:g}.")
    if peak > 1:
        raise ValueError(f"{SEED_PEAK_FACTOR} = {peak!r} must be at most 1 (the solver clips above 1).")
    if float(params["gaussian_seed_scale"]) != 1.0:
        raise ValueError(f"{SEED_SIGMA_FACTOR} is in mm, which needs gaussian_seed_scale = 1, got {params['gaussian_seed_scale']!r}.")
    record: dict[str, Any] = {
        "centre": {
            **{key: float(params[key]) for key in CONSTANT_KEYS},
            "voxel_size_mm": list(voxel_size),
            "grid_shape": list(params["white_matter_pbmap"].shape),
            "stopping_time": float(params["stopping_time"]),
            "preop_time": centre_config["_fit"]["preop_time"],
        },
        "seed_peak": {"value": peak, "gaussian_seed_floor": floor, "clip": 1.0},
        "time_stepping": {},
    }
    corner_values = {name: factor.high for name, factor in space.factors.items()}
    corner_values.update({name: 0.5 for name in SEED_BBOX_FACTORS})
    corner = replace(problem.derive(corner_values), preop_time=problem.preop_time_range[1], preop_time_clamped=True)
    for factor in sorted({float(f) for f in resolution_factors} | {FULL_RESOLUTION}):
        config = problem.config_of(corner, factor)
        probe = StuppFKPPSolver(read_config_from_mapping(config))
        n_resolved, dt_resolved = probe.resolve_time_stepping()
        requested = int(config["n_steps"])
        record["time_stepping"][str(factor)] = {
            "n_steps_requested": requested,
            "n_steps_resolved": int(n_resolved),
            "dt_resolved": float(dt_resolved),
            "raised": int(n_resolved) > requested,
            "corner": {
                "white_matter_diffusivity": corner.white_matter_diffusivity,
                "rho": corner.rho,
                "stopping_time": config["_fit"]["stopping_time"],
            },
            "lowres_grid_shape": list(probe.grid_shape),
            "grid_spacing_mm": list(probe.grid_spacing),
        }
    return record


def _design_check_process(problem: FitProblem, resolution_factors: Sequence[float], queue: Any) -> None:
    """The subprocess side of ``run_design_checks``."""
    try:
        queue.put(("ok", design_checks(problem, resolution_factors)))
    except Exception:  # noqa: BLE001 - reported to the main process
        queue.put(("error", traceback.format_exc()))


def run_design_checks(problem: FitProblem, resolution_factors: Sequence[float]) -> dict[str, Any]:
    """``design_checks`` in a spawned CPU-only subprocess (JAX_PLATFORMS=cpu,
    no visible CUDA device), so that the main process never initialises a
    JAX backend."""
    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = start_process(ctx, _design_check_process, (problem, list(resolution_factors), queue), DESIGN_CHECK_ENV)
    try:
        while True:
            try:
                kind, payload = queue.get(timeout=POLL_SECONDS)
                break
            except Empty:
                if not process.is_alive():
                    raise RuntimeError(f"the design-check subprocess died with exit code {process.exitcode}.") from None
    finally:
        process.join(timeout=WORKER_STOP_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
    if kind != "ok":
        raise RuntimeError(f"the design-time checks failed:\n{payload}")
    return payload


# --- CMA-ES (a port of GIFT's cmaes.py) ---


class CMAES:
    """
    The CMA-ES of GIFT's cmaes.py (the module docstring's CMA-ES
    section) as an ask/tell optimizer on [0, 1]^N with repaired bounds.

    Args:
        x0: The start, in [0, 1]^N.
        sigma0: The initial step size.
        popsize: lambda, the number of members per generation (at least 2).
        rng: The random generator of the sampling.

    Attributes:
        mean: m, the distribution mean.
        sigma: The step size.
        C: The covariance matrix.
        ps, pc: The evolution paths.
        generation: The number of completed generations (GIFT's gen after
            the update).
    """

    def __init__(self, x0: NDArray | Sequence[float], sigma0: float, popsize: int, rng: np.random.Generator) -> None:
        mean = np.array(x0, dtype=np.float64).ravel()
        n = int(mean.size)
        if n < 1:
            raise ValueError("CMA-ES needs at least one coordinate.")
        if int(popsize) < 2:
            raise ValueError(f"the population size must be at least 2, got {popsize}.")
        if not (np.isfinite(sigma0) and sigma0 > 0):
            raise ValueError(f"sigma0 must be positive, got {sigma0!r}.")
        self.n = n
        self.popsize = int(popsize)
        self.mu = self.popsize // 2
        raw = np.array([math.log((self.popsize + 1) / 2) - math.log(i + 1) for i in range(self.mu)], dtype=np.float64)
        self.weights = raw / raw.sum()
        self.mueff = 1.0 / float(np.sum(self.weights**2))
        mueff = self.mueff
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(2) * math.gamma((n + 1) / 2) / math.gamma(n / 2)
        self.mean = mean
        self.sigma = float(sigma0)
        self.C = np.identity(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.generation = 0
        self.rng = rng
        self._d: NDArray | None = None
        self._sqrt_c: NDArray | None = None
        self._inv_sqrt_c: NDArray | None = None

    def _decompose(self) -> None:
        """C = B diag(d^2) B^T of the symmetrised C (eigenvalues floored
        at EIGENVALUE_FLOOR): sqrtC = B diag(d) B^T, invsqrtC = B diag(1/d) B^T."""
        self.C = 0.5 * (self.C + self.C.T)
        eigenvalues, basis = np.linalg.eigh(self.C)
        d = np.sqrt(np.maximum(eigenvalues, EIGENVALUE_FLOOR))
        self._d = d
        self._sqrt_c = (basis * d) @ basis.T
        self._inv_sqrt_c = (basis / d) @ basis.T

    @property
    def max_std(self) -> float:
        """sigma max(d), the largest standard deviation of the distribution."""
        if self._d is None:
            self._decompose()
        assert self._d is not None
        return float(self.sigma * self._d.max())

    def ask(self) -> NDArray:
        """The members of the next generation, (popsize, N), each
        x = m + sigma sqrtC z clipped into [0, 1]."""
        self._decompose()
        assert self._sqrt_c is not None
        z = self.rng.standard_normal((self.popsize, self.n))
        y = z @ self._sqrt_c  # sqrtC is symmetric: row i is sqrtC z_i
        return np.clip(self.mean + self.sigma * y, 0.0, 1.0)

    def tell(self, xs: NDArray, losses: Sequence[float]) -> None:
        """
        GIFT's update with the evaluated (clipped) members: the steps are
        repaired, y = (x - m) / sigma and z = invsqrtC y, the members
        sorted by loss (stable), then the mean, the paths, sigma and C.
        """
        if self._inv_sqrt_c is None:
            raise RuntimeError("tell() needs the decomposition of ask().")
        xs = np.asarray(xs, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64)
        if xs.shape != (self.popsize, self.n) or losses.shape != (self.popsize,):
            raise ValueError(f"tell() expects {self.popsize} members of {self.n} coordinates and one loss each.")
        if not np.all(np.isfinite(losses)):
            raise ValueError("every loss must be finite (a failed evaluation is loss 1).")
        y_all = (xs - self.mean) / self.sigma
        z_all = y_all @ self._inv_sqrt_c
        order = np.argsort(losses, kind="stable")[: self.mu]
        x_sel, y_sel, z_sel = xs[order], y_all[order], z_all[order]
        w = self.weights
        n, cs, cc, c1, cmu = self.n, self.cs, self.cc, self.c1, self.cmu
        self.mean = w @ x_sel
        self.ps = (1 - cs) * self.ps + math.sqrt(cs * (2 - cs) * self.mueff) * (w @ z_sel)
        pssq = float(self.ps @ self.ps)
        self.sigma *= math.exp(cs / self.damps * (math.sqrt(pssq) / self.chiN - 1))
        c_mu = (y_sel.T * w) @ y_sel
        self.generation += 1
        if (n + 1) * pssq < 2 * n * (n + 3) * (1 - (1 - cs) ** (2 * self.generation)):
            self.pc = (1 - cc) * self.pc + math.sqrt(cc * (2 - cc) * self.mueff) * (w @ y_sel)
            self.C = (1 - c1 - cmu) * self.C + c1 * np.outer(self.pc, self.pc) + cmu * c_mu
        else:
            self.pc = (1 - cc) * self.pc
            self.C = (1 - c1 - cmu) * self.C + c1 * (np.outer(self.pc, self.pc) + cc * (2 - cc) * self.C) + cmu * c_mu

    def state(self) -> dict[str, Any]:
        """The checkpoint of the optimizer (``from_state``)."""
        return {
            "mean": self.mean.copy(),
            "sigma": self.sigma,
            "C": self.C.copy(),
            "ps": self.ps.copy(),
            "pc": self.pc.copy(),
            "generation": self.generation,
            "rng_state": json.dumps(self.rng.bit_generator.state),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], popsize: int) -> CMAES:
        rng = np.random.default_rng(0)
        rng.bit_generator.state = json.loads(str(state["rng_state"]))
        optimizer = cls(np.asarray(state["mean"]), float(state["sigma"]), popsize, rng)
        optimizer.C = np.array(state["C"], dtype=np.float64)
        optimizer.ps = np.array(state["ps"], dtype=np.float64)
        optimizer.pc = np.array(state["pc"], dtype=np.float64)
        optimizer.generation = int(state["generation"])
        return optimizer

    def constants(self) -> dict[str, float]:
        return {
            "N": self.n,
            "popsize": self.popsize,
            "mu": self.mu,
            "weights": self.weights.tolist(),
            "mueff": self.mueff,
            "cc": self.cc,
            "cs": self.cs,
            "c1": self.c1,
            "cmu": self.cmu,
            "damps": self.damps,
            "chiN": self.chiN,
        }


def gift_popsize(n: int) -> int:
    """GIFT's default population size 4 + floor(3 ln N)."""
    return 4 + int(3 * math.log(n))


def resolve_popsize(text: str, n_workers: int, dimension: int) -> int:
    """--popsize: "auto" = POPSIZE_PER_WORKER x workers, "gift" = GIFT's
    default, or an integer of at least 2."""
    text = str(text).strip().lower()
    if text == "auto":
        return POPSIZE_PER_WORKER * max(1, int(n_workers))
    if text == "gift":
        return gift_popsize(dimension)
    popsize = int(text)
    if popsize < 2:
        raise ValueError(f"--popsize must be at least 2, got {popsize}.")
    return popsize


# --- starts ---


@dataclass(frozen=True)
class Start:
    """
    One restart's start.

    Attributes:
        index: The restart index.
        label: "sweep:<run>", "manual" or "centre".
        unit: The unit-cube point.
        physical: The physical factor values.
        source: Where it came from (the sweep run, the proxy, clipping).
    """

    index: int
    label: str
    unit: NDArray
    physical: dict[str, float]
    source: dict[str, Any]

    def record(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "unit": {name: float(self.unit[i]) for i, name in enumerate(names)},
            "physical": dict(self.physical),
            "source": dict(self.source),
        }


def sweep_proxy(row: Mapping[str, str], objective: Objective) -> float:
    """The weighted mean over the objective's sessions of a qoi row's
    <ses>_dice_core, NaN entries dropped with their weight (NaN when none
    is finite)."""
    numerator = denominator = 0.0
    for sid in objective.session_ids:
        value = as_float(row.get(sid.replace("-", "") + "_dice_core"))
        if np.isfinite(value):
            numerator += objective.weights[sid] * value
            denominator += objective.weights[sid]
    return numerator / denominator if denominator > 0 else float("nan")


def convert_sweep_row(record: Mapping[str, str], problem: FitProblem) -> tuple[dict[str, float], dict[str, Any]]:
    """
    The fit's physical factor values of a sweep design row (the module
    docstring's Starts) and a record of the conversion: the row's
    preop_time and seed voxel, the not-carried parameters and the
    clipping.

    Returns:
        (values clipped into the fit's ranges, conversion record).
    """
    space = problem.space
    speed = as_float(record[GROWTH_SPEED_FACTOR])
    preop_time = as_float(record["preop_time"])
    voxel = tuple(int(float(record[f"seed_voxel_{axis}"])) for axis in "ijk")
    values: dict[str, float] = {
        GROWTH_SPEED_FACTOR: speed,
        GROWTH_WIDTH_FACTOR: as_float(record[GROWTH_WIDTH_FACTOR]),
        SEED_SIGMA_FACTOR: as_float(record[SEED_SIGMA_FACTOR]),
        GROWTH_LENGTH_FACTOR: speed * preop_time,
    }
    values.update(dict(zip(SEED_BBOX_FACTORS, problem.seed_map.to_bbox(voxel), strict=True)))
    for name in space.solver_factor_names:
        values[name] = as_float(record[name])
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError(f"the sweep row {record.get('row_name')!r} lacks a value the fit needs: {values}.")
    clipped, changed = space.clip(values)
    not_carried = {
        key: as_float(record[key])
        for key in (SEED_PEAK_FACTOR, "diffusivity_ratio", "rt_alpha_beta_ratio", "core_threshold", "edema_threshold")
        if key in record and record[key] != ""
    }
    conversion = {
        "sweep_preop_time": preop_time,
        "fit_preop_time": clipped[GROWTH_LENGTH_FACTOR] / clipped[GROWTH_SPEED_FACTOR],
        "sweep_seed_voxel": list(voxel),
        "seed_voxel_seedable_in_fit": bool(problem.seed_map.geometry.mask[voxel]),
        "fit_seed_voxel": list(problem.seed_map.voxel([clipped[name] for name in SEED_BBOX_FACTORS])[0]),
        "not_carried": not_carried,
        "clipped": changed,
    }
    return clipped, conversion


def sweep_starts(sweep_dir: Path, problem: FitProblem, top: int) -> tuple[list[Start], dict[str, Any]]:
    """
    The top-K starts of a patient_sensitivity_analysis.py sweep (the
    module docstring's Starts): the runs ranked by ``sweep_proxy`` (the
    maximum over the rows sharing the run), converted with
    ``convert_sweep_row``.

    Returns:
        (starts, record of the ranking for starts.json).

    Raises:
        FileNotFoundError: The sweep lacks design.csv, qoi.csv or spec.json.
        ValueError: Another patient; an objective session the sweep does
            not have; no successful run.
    """
    for name in ("design.csv", "qoi.csv", "spec.json"):
        if not (sweep_dir / name).is_file():
            raise FileNotFoundError(f"sweep file not found: {sweep_dir / name}")
    spec = read_json(sweep_dir / "spec.json")
    if spec["patient"]["patient"] != problem.data.patient:
        raise ValueError(f"the sweep {sweep_dir} is of {spec['patient']['patient']!r}, the fit of {problem.data.patient!r}.")
    sweep_ids = [s["id"] for s in spec["patient"]["sessions"]]
    missing = [sid for sid in problem.objective.session_ids if sid not in sweep_ids]
    if missing:
        raise ValueError(f"the objective's session(s) {missing} are not among the sweep's {sweep_ids}.")
    design = read_csv(sweep_dir / "design.csv")
    runs = distinct_runs(design)
    needed = [GROWTH_SPEED_FACTOR, GROWTH_WIDTH_FACTOR, SEED_SIGMA_FACTOR, "preop_time", "seed_voxel_i", "seed_voxel_j", "seed_voxel_k"]
    needed += problem.space.solver_factor_names
    columns = set(design[0]) if design else set()
    lacking = [column for column in needed if column not in columns]
    if lacking:
        raise ValueError(f"{sweep_dir / 'design.csv'} lacks the column(s) {lacking} the fit's parametrisation needs.")
    best: dict[str, float] = {}
    n_rows = n_success = 0
    for row in read_csv(sweep_dir / "qoi.csv"):
        n_rows += 1
        if not _is_success(row):
            continue
        n_success += 1
        proxy = sweep_proxy(row, problem.objective)
        if np.isfinite(proxy) and (row["run_name"] not in best or proxy > best[row["run_name"]]):
            best[row["run_name"]] = proxy
    if not best:
        raise ValueError(f"no successful run with a finite proxy in {sweep_dir / 'qoi.csv'}.")
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    starts: list[Start] = []
    duplicates: list[dict[str, Any]] = []
    taken: list[NDArray] = []
    for rank, (run, proxy) in enumerate(ranked):
        if len(starts) >= int(top):
            break
        record = runs[run]
        values, conversion = convert_sweep_row(record, problem)
        unit = problem.to_unit(values)
        same = next((i for i, other in enumerate(taken) if np.allclose(unit, other, rtol=0, atol=1e-12)), None)
        if same is not None:
            # Runs of one Saltelli block that differ only in parameters the
            # fit fixes (or in the cheap thresholds) give the same start.
            duplicates.append({"run_name": run, "rank": rank, "proxy": proxy, "same_as": starts[same].label})
            continue
        taken.append(unit)
        starts.append(
            Start(
                index=len(starts),
                label=f"sweep:{run}",
                unit=unit,
                physical=values,
                source={"sweep_dir": str(sweep_dir), "run_name": run, "row_name": record["row_name"], "rank": rank, "proxy": proxy, **conversion},
            )
        )
    if len(starts) < top:
        warnings.warn(f"the sweep offers {len(starts)} distinct starts, fewer than --top {top}.", stacklevel=2)
    ranking = {
        "sweep_dir": str(sweep_dir),
        "sweep_name": spec.get("name"),
        "sweep_threshold_mode": spec.get("threshold_mode"),
        "sweep_adjuvant_cycle_days": spec.get("protocol", {}).get("adjuvant_cycle_days"),
        "fit_adjuvant_cycle_days": ADJUVANT_CYCLE_DAYS,
        "proxy": (
            "per run the maximum over its qoi rows of the weighted mean over the objective's sessions of <ses>_dice_core "
            "(NaN dropped with its weight); in a sampled-mode sweep the rows of a run differ in their sampled thresholds"
        ),
        "n_qoi_rows": n_rows,
        "n_successful_rows": n_success,
        "n_ranked_runs": len(best),
        "top": int(top),
        "distinct": "runs whose converted unit vector equals an earlier start's (they differ only in parameters the fit fixes) are skipped",
        "taken": [{"run_name": s.source["run_name"], "rank": s.source["rank"], "proxy": s.source["proxy"]} for s in starts],
        "skipped_duplicates": duplicates,
    }
    return starts, ranking


def manual_start(text: str, problem: FitProblem, index: int) -> Start:
    """The start of --init-values name=value,... (physical units; every
    fitted factor named; clipped into the ranges, recorded)."""
    given: dict[str, float] = {}
    for item in (part.strip() for part in str(text).split(",") if part.strip()):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--init-values: expected name=value, got {item!r}.")
        given[name.strip()] = float(value)
    names = problem.space.names
    unknown = sorted(set(given) - set(names))
    missing = [name for name in names if name not in given]
    if unknown or missing:
        raise ValueError(f"--init-values: unknown factor(s) {unknown}, missing {missing}; the factors are {names}.")
    values, changed = problem.space.clip(given)
    return Start(index, "manual", problem.to_unit(values), values, {"given": given, "clipped": changed})


def centre_start(problem: FitProblem, index: int) -> Start:
    """The start at the unit-cube centre."""
    unit = np.full(problem.space.dimension, 0.5)
    return Start(index, "centre", unit, problem.to_physical(unit), {"rule": "the unit-cube centre"})


def resolve_starts(
    problem: FitProblem, sweep_dir: Path | None, top: int, init_values: str | None
) -> tuple[list[Start], dict[str, Any]]:
    """
    The starts of a fit (the module docstring's Starts): the sweep's
    top-K (sweep_dir None = --no-sweep-init), then the manual start, else
    the centre.

    Raises:
        FileNotFoundError: The sweep directory does not exist and no
            manual start is given.
    """
    starts: list[Start] = []
    record: dict[str, Any] = {"sweep": None, "manual": init_values, "centre": False}
    if sweep_dir is not None:
        if not sweep_dir.is_dir():
            if not init_values:
                raise FileNotFoundError(
                    f"sweep directory not found: {sweep_dir} (give --init-from-sweep, --init-values or --no-sweep-init)."
                )
            print(f"WARNING: sweep directory not found: {sweep_dir}; only the manual start is used.", flush=True)
            record["sweep"] = {"sweep_dir": str(sweep_dir), "missing": True}
        else:
            starts, ranking = sweep_starts(sweep_dir, problem, top)
            record["sweep"] = ranking
    if init_values:
        starts.append(manual_start(init_values, problem, len(starts)))
    if not starts:
        starts.append(centre_start(problem, 0))
        record["centre"] = True
    return starts, record


def load_starts(path: Path, problem: FitProblem) -> list[Start]:
    """The starts of a starts.json."""
    entries = read_json(path)
    names = problem.space.names
    starts = []
    for entry in entries["starts"]:
        unit = np.array([float(entry["unit"][name]) for name in names])
        starts.append(Start(int(entry["index"]), entry["label"], unit, {n: float(entry["physical"][n]) for n in names}, dict(entry["source"])))
    return starts


def format_starts(starts: Sequence[Start], problem: FitProblem) -> str:
    """The printed table of the starts."""
    names = problem.space.names
    lines = [f"{len(starts)} start(s):"]
    for start in starts:
        point = problem.derive(start.physical, start.unit)
        proxy = start.source.get("proxy")
        lines.append(
            f"  start {start.index} ({start.label}"
            + (f", proxy {proxy:.4f}" if isinstance(proxy, float) else "")
            + f"): preop_time {point.preop_time:.1f} d"
            + (" (clamped)" if point.preop_time_clamped else "")
            + f", seed voxel {list(point.seed_voxel)}" + (" (snapped)" if point.seed_snapped else "")
            + (f", clipped {sorted(start.source['clipped'])}" if start.source.get("clipped") else "")
        )
        for name in names:
            lines.append(f"    {name:<24} u = {start.unit[names.index(name)]:.4f}   {start.physical[name]:.6g}")
        lines.append(
            f"    {'derived':<24} D = {point.white_matter_diffusivity:.4g} mm^2/day, rho = {point.rho:.4g} /day, "
            f"mass = {point.gaussian_seed_mass:.4g}, tau = {point.gaussian_seed_diffusion_time:.4g} mm^2"
        )
    return "\n".join(lines)


# --- the fit: settings, checkpoints, restarts, summary ---


@dataclass(frozen=True)
class FitSettings:
    """
    The CMA-ES settings of a fit.

    Attributes:
        popsize: lambda.
        sigma0s: The initial step size per start.
        generations: The generations per start.
        min_sigma: The termination threshold on sigma max(d) (0 = off).
        seed: The base seed (restart k draws from default_rng([seed, k])).
        schedule: The resolution schedule.
    """

    popsize: int
    sigma0s: tuple[float, ...]
    generations: tuple[int, ...]
    min_sigma: float
    seed: int
    schedule: ResolutionSchedule

    @classmethod
    def from_args(cls, args: argparse.Namespace, n_workers: int, n_starts: int, dimension: int) -> FitSettings:
        return cls(
            popsize=resolve_popsize(args.popsize, n_workers, dimension),
            sigma0s=tuple(parse_float_list(args.sigma0, "--sigma0", n_starts)),
            generations=tuple(parse_int_list(args.generations, "--generations", n_starts)),
            min_sigma=float(args.min_sigma),
            seed=int(args.seed),
            schedule=ResolutionSchedule.parse(args.resolution_schedule),
        )

    def record(self) -> dict[str, Any]:
        return {
            "popsize": self.popsize,
            "mu": self.popsize // 2,
            "sigma0": list(self.sigma0s),
            "generations": list(self.generations),
            "min_sigma": self.min_sigma,
            "seed": self.seed,
            "rng": "numpy default_rng([seed, restart]) (PCG64)",
            "resolution_schedule": self.schedule.record(),
            "bound_handling": (
                "members are clipped into [0, 1] and the step repaired: y = (x_clip - m) / sigma, z = invsqrtC y in "
                "every update (GIFT updates with the unclipped z, y)"
            ),
            "source": "GIFT cmaes.py (github.com/jonasw247/glioma-inverse-fitting-tool, branch dtiStuff), ported; see the module docstring",
        }

    def check_against(self, record: Mapping[str, Any]) -> None:
        """Refuse a resume whose settings differ from the fit's spec.json."""
        mine = self.record()
        differences = [
            key for key in ("popsize", "sigma0", "generations", "min_sigma", "seed", "resolution_schedule")
            if json.dumps(mine[key], sort_keys=True) != json.dumps(record.get(key), sort_keys=True)
        ]
        if differences:
            raise ValueError(f"--resume: the setting(s) {differences} differ from the fit's spec.json; give the fit's values.")


@dataclass(frozen=True)
class Incumbent:
    """The best evaluated point of a restart at its current resolution
    factor: its unit vector, loss, eval_id, generation and record."""

    unit: NDArray
    loss: float
    eval_id: int
    generation: int
    record: dict[str, Any]


def _savez_atomic(path: Path, **arrays: Any) -> None:
    """np.savez into path through a temporary file (a crash mid-write
    leaves the previous file)."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def save_state(
    path: Path,
    optimizer: CMAES,
    incumbent: Incumbent | None,
    next_eval_id: int,
    resolution_factor: float,
    complete: bool,
    wall_time_s: float,
    termination: str,
) -> None:
    """The resume checkpoint of a restart (state.npz)."""
    state = optimizer.state()
    _savez_atomic(
        path,
        mean=state["mean"],
        sigma=np.float64(state["sigma"]),
        C=state["C"],
        ps=state["ps"],
        pc=state["pc"],
        generation=np.int64(state["generation"]),
        rng_state=np.array(state["rng_state"]),
        incumbent_unit=np.zeros(0) if incumbent is None else np.asarray(incumbent.unit, dtype=np.float64),
        incumbent_loss=np.float64(np.nan if incumbent is None else incumbent.loss),
        incumbent_eval_id=np.int64(-1 if incumbent is None else incumbent.eval_id),
        incumbent_generation=np.int64(-1 if incumbent is None else incumbent.generation),
        next_eval_id=np.int64(next_eval_id),
        resolution_factor=np.float64(resolution_factor),
        complete=np.bool_(complete),
        wall_time_s=np.float64(wall_time_s),
        termination=np.array(termination),
    )


def load_state(path: Path) -> dict[str, Any]:
    """The checkpoint of ``save_state`` as plain values."""
    with np.load(path) as archive:
        state = {key: archive[key] for key in archive.files}
    out: dict[str, Any] = {
        "mean": np.asarray(state["mean"], dtype=np.float64),
        "sigma": float(state["sigma"]),
        "C": np.asarray(state["C"], dtype=np.float64),
        "ps": np.asarray(state["ps"], dtype=np.float64),
        "pc": np.asarray(state["pc"], dtype=np.float64),
        "generation": int(state["generation"]),
        "rng_state": str(state["rng_state"]),
        "incumbent_unit": np.asarray(state["incumbent_unit"], dtype=np.float64),
        "incumbent_loss": float(state["incumbent_loss"]),
        "incumbent_eval_id": int(state["incumbent_eval_id"]),
        "incumbent_generation": int(state["incumbent_generation"]),
        "next_eval_id": int(state["next_eval_id"]),
        "resolution_factor": float(state["resolution_factor"]),
        "complete": bool(state["complete"]),
        "wall_time_s": float(state["wall_time_s"]),
        "termination": str(state["termination"]),
    }
    return out


TRACE_KEYS: tuple[str, ...] = ("gen", "xmean", "sigma", "C", "ps", "pc", "best_loss", "mean_loss", "n_failed", "resolution_factor")


def save_trace(path: Path, trace: Sequence[Mapping[str, Any]]) -> None:
    """trace.npz: one array per TRACE_KEYS with the generations along the
    first axis."""
    if not trace:
        return
    _savez_atomic(path, **{key: np.asarray([entry[key] for entry in trace]) for key in TRACE_KEYS})


def load_trace(path: Path, up_to_generation: int) -> list[dict[str, Any]]:
    """The trace entries of the generations up to and including the given one."""
    if not path.is_file():
        return []
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in TRACE_KEYS if key in archive.files}
    entries = []
    for i in range(len(arrays["gen"])):
        if int(arrays["gen"][i]) <= up_to_generation:
            entries.append({key: arrays[key][i] for key in arrays})
    return entries


def best_summary(record: Mapping[str, Any], problem: FitProblem) -> dict[str, Any]:
    """The summary.json entry of one evaluation record (a live record or
    a typed CSV row)."""
    row = typed_row(record)
    names = problem.space.names
    return {
        "eval_id": row["eval_id"],
        "generation": row.get("generation"),
        "member": row.get("member"),
        "resolution_factor": row["resolution_factor"],
        "tag": row.get("tag", ""),
        "success": row["success"],
        "error": row.get("error", ""),
        "loss": row["loss"],
        "J": row["J"],
        "core_threshold_star": row["core_threshold_star"],
        "edema_threshold_star": row["edema_threshold_star"],
        "n_nan_sessions": row["n_nan_sessions"],
        "unit": {name: row[f"u_{name}"] for name in names},
        "physical": {name: row.get(name) for name in names},
        "derived": {column: row.get(column) for column in DERIVED_COLUMNS},
        "per_session": {
            session.id: {name: row.get(f"{session.prefix}{name}") for name in SESSION_COLUMNS} for session in problem.sessions
        },
        "n_steps": row.get("n_steps"),
        "dt": row.get("dt"),
        "solve_wall_time_s": row.get("solve_wall_time_s"),
    }


def write_summary(
    fit_dir: Path,
    problem: FitProblem,
    settings: FitSettings,
    restarts: Sequence[Mapping[str, Any]],
    workers: Sequence[Mapping[str, Any]],
    wall_time_s: float,
    complete: bool,
) -> dict[str, Any]:
    """summary.json: the restarts ranked by best loss, the overall best,
    the workers; an existing resolve record is kept."""
    path = fit_dir / SUMMARY_FILE
    previous = read_json(path) if path.is_file() else {}
    ranked = sorted(restarts, key=lambda r: (r["best"]["loss"] if r.get("best") else np.inf, r["restart"]))
    best = next((r for r in ranked if r.get("best")), None)
    summary: dict[str, Any] = {
        "name": fit_dir.name,
        "fit_dir": str(fit_dir),
        "complete": complete,
        "n_restarts_done": len(restarts),
        "n_restarts": len(settings.sigma0s),
        "n_evaluations": int(sum(r["n_evaluations"] for r in restarts)),
        "wall_time_s": round(float(wall_time_s), 3),
        "objective": problem.objective.record(),
        "restarts": list(ranked),
        "best": None if best is None else {"restart": best["restart"], "eval_id": best["best"]["eval_id"], "loss": best["best"]["loss"]},
        "workers": list(workers),
        "resolve": previous.get("resolve"),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(path, summary)
    return summary


class EvaluationLog:
    """An evaluations.csv opened for appending: one row per record as it
    arrives, flushed at once."""

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self.handle = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.columns, extrasaction="ignore")
        if self.handle.tell() == 0:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, record: Mapping[str, Any]) -> None:
        self.writer.writerow({key: _csv_cell(record.get(key)) for key in self.columns})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def truncate_evaluations(path: Path, columns: Sequence[str], up_to_generation: int) -> tuple[list[dict[str, Any]], int]:
    """Keep the rows of the generations up to the given one (the rows of
    an interrupted generation are dropped and the file rewritten)."""
    rows = read_evaluations(path)
    kept = [row for row in rows if row["generation"] is not None and row["generation"] <= up_to_generation]
    if len(kept) != len(rows):
        write_csv(path, kept, columns)
    return kept, len(rows) - len(kept)


def submit(
    pool: WorkerPool,
    log: EvaluationLog,
    restart: int,
    generation: int,
    jobs: Sequence[tuple[int, NDArray, float, str | None]],
    members: Mapping[int, int],
    tag: str,
) -> list[dict[str, Any]]:
    """Evaluate jobs on the pool, log every record as it arrives with its
    bookkeeping, and return the records in eval_id order."""
    records: dict[int, dict[str, Any]] = {}
    for record in pool.evaluate(jobs):
        eval_id = int(record["eval_id"])
        record.update(restart=restart, generation=generation, member=members[eval_id], tag=tag)
        log.write(record)
        records[eval_id] = record
    return [records[int(job[0])] for job in jobs]


def run_restart(
    k: int,
    start: Start,
    problem: FitProblem,
    settings: FitSettings,
    pool: WorkerPool,
    fit_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """
    One restart (the module docstring's CMA-ES section): the start
    evaluated as generation 0, then the generations with the resolution
    schedule, the incumbent, the trace and the checkpoint after every
    generation; resumed from state.npz when it exists.

    Returns:
        The restart's summary.json entry.
    """
    restart_dir = fit_dir / f"restart_{k}"
    state_path = restart_dir / STATE_FILE
    csv_path = restart_dir / EVALUATIONS_FILE
    trace_path = restart_dir / TRACE_FILE
    columns = evaluation_columns(problem.space.names, problem.prefixes)
    sigma0 = settings.sigma0s[k]
    n_generations = settings.generations[k]
    schedule = settings.schedule
    incumbent: Incumbent | None = None
    trace: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"{state_path} exists; a restart is never overwritten (use --resume).")
        state = load_state(state_path)
        rows, dropped = truncate_evaluations(csv_path, columns, state["generation"])
        optimizer = CMAES.from_state(state, settings.popsize)
        if state["incumbent_eval_id"] >= 0:
            row = next(r for r in rows if r["eval_id"] == state["incumbent_eval_id"])
            incumbent = Incumbent(state["incumbent_unit"], state["incumbent_loss"], state["incumbent_eval_id"], state["incumbent_generation"], row)
        next_eval_id = max((r["eval_id"] for r in rows), default=-1) + 1
        current_factor: float | None = state["resolution_factor"]
        trace = load_trace(trace_path, state["generation"])
        elapsed_before = state["wall_time_s"]
        if state["complete"]:
            print(f"restart {k}: complete ({state['termination']}), {len(rows)} evaluations; skipped", flush=True)
            return restart_summary(k, start, problem, settings, optimizer, incumbent, rows, state["wall_time_s"], state["termination"], current_factor)
        print(
            f"restart {k}: resuming after generation {state['generation']} of {n_generations} "
            f"({len(rows)} evaluations kept, {dropped} of the interrupted generation dropped)",
            flush=True,
        )
    else:
        if restart_dir.exists() and not resume:
            raise FileExistsError(f"{restart_dir} exists; a restart is never overwritten (use --resume).")
        restart_dir.mkdir(parents=True, exist_ok=True)
        if csv_path.is_file():  # an interrupted generation 0 (no checkpoint yet): redone
            print(f"restart {k}: {csv_path} has rows but no checkpoint; they are discarded", flush=True)
            csv_path.unlink()
        optimizer = CMAES(start.unit, sigma0, settings.popsize, np.random.default_rng([settings.seed, k]))
        next_eval_id = 0
        current_factor = None
        print(f"restart {k}: start {start.label}, sigma0 {sigma0:g}, {n_generations} generations, popsize {settings.popsize}", flush=True)
    started = time.perf_counter()
    termination = "interrupted"
    log = EvaluationLog(csv_path, columns)
    try:
        if incumbent is None:
            factor = schedule.factor_at(1, n_generations)
            current_factor = factor
            eval_id = next_eval_id
            next_eval_id += 1
            record = submit(pool, log, k, 0, [(eval_id, start.unit, factor, None)], {eval_id: MEMBER_NONE}, TAG_START)[0]
            incumbent = Incumbent(np.asarray(start.unit, dtype=np.float64), float(record["loss"]), eval_id, 0, record)
            print(f"restart {k} generation 0: start loss {incumbent.loss:.4f} at factor {factor:g}", flush=True)
            save_state(state_path, optimizer, incumbent, next_eval_id, factor, False, elapsed_before + time.perf_counter() - started, termination)
        assert current_factor is not None
        for generation in range(optimizer.generation + 1, n_generations + 1):
            factor = schedule.factor_at(generation, n_generations)
            if factor != current_factor:
                optimizer.sigma = sigma0
                eval_id = next_eval_id
                next_eval_id += 1
                record = submit(pool, log, k, generation, [(eval_id, incumbent.unit, factor, None)], {eval_id: MEMBER_NONE}, TAG_REEVAL)[0]
                previous_loss = incumbent.loss
                incumbent = Incumbent(incumbent.unit, float(record["loss"]), eval_id, generation, record)
                current_factor = factor
                print(
                    f"restart {k} generation {generation}: resolution factor {factor:g}, sigma reset to {sigma0:g}, "
                    f"incumbent re-evaluated: loss {previous_loss:.4f} -> {incumbent.loss:.4f}",
                    flush=True,
                )
            members = optimizer.ask()
            jobs = [(next_eval_id + i, members[i], factor, None) for i in range(settings.popsize)]
            member_of = {next_eval_id + i: i for i in range(settings.popsize)}
            next_eval_id += settings.popsize
            gen_started = time.perf_counter()
            records = submit(pool, log, k, generation, jobs, member_of, "")
            losses = np.array([float(r["loss"]) for r in records])
            optimizer.tell(members, losses)
            best_member = int(np.argmin(losses))
            if losses[best_member] < incumbent.loss:
                incumbent = Incumbent(members[best_member].copy(), float(losses[best_member]), int(records[best_member]["eval_id"]), generation, records[best_member])
            n_failed = int(sum(1 for r in records if not r["success"]))
            trace.append(
                {
                    "gen": generation,
                    "xmean": optimizer.mean.copy(),
                    "sigma": optimizer.sigma,
                    "C": optimizer.C.copy(),
                    "ps": optimizer.ps.copy(),
                    "pc": optimizer.pc.copy(),
                    "best_loss": float(losses[best_member]),
                    "mean_loss": float(losses.mean()),
                    "n_failed": n_failed,
                    "resolution_factor": factor,
                }
            )
            save_trace(trace_path, trace)
            wall = elapsed_before + time.perf_counter() - started
            save_state(state_path, optimizer, incumbent, next_eval_id, factor, False, wall, termination)
            print(
                f"restart {k} generation {generation}/{n_generations}: best {losses[best_member]:.4f}, mean {losses.mean():.4f}, "
                f"incumbent {incumbent.loss:.4f} (eval {incumbent.eval_id}), sigma {optimizer.sigma:.4g}, "
                f"sigma max(d) {optimizer.max_std:.4g}, factor {factor:g}, {n_failed} failed, "
                f"{time.perf_counter() - gen_started:.0f} s",
                flush=True,
            )
            if settings.min_sigma > 0 and optimizer.max_std < settings.min_sigma:
                termination = "min_sigma"
                print(f"restart {k}: sigma max(d) {optimizer.max_std:.4g} < {settings.min_sigma:g}, stopped", flush=True)
                break
        else:
            termination = "generations"
    finally:
        log.close()
    wall = elapsed_before + time.perf_counter() - started
    save_state(state_path, optimizer, incumbent, next_eval_id, current_factor, True, wall, termination)
    rows = read_evaluations(csv_path)
    print(f"restart {k}: done ({termination}), best loss {incumbent.loss:.4f} (eval {incumbent.eval_id}), {len(rows)} evaluations, {wall / 60:.1f} min", flush=True)
    return restart_summary(k, start, problem, settings, optimizer, incumbent, rows, wall, termination, current_factor)


def restart_summary(
    k: int,
    start: Start,
    problem: FitProblem,
    settings: FitSettings,
    optimizer: CMAES,
    incumbent: Incumbent | None,
    rows: Sequence[Mapping[str, Any]],
    wall_time_s: float,
    termination: str,
    resolution_factor: float | None,
) -> dict[str, Any]:
    """The summary.json entry of a restart."""
    return {
        "restart": k,
        "start": start.record(problem.space.names),
        "sigma0": settings.sigma0s[k],
        "generations_requested": settings.generations[k],
        "generations_done": optimizer.generation,
        "popsize": settings.popsize,
        "n_evaluations": len(rows),
        "n_failed": int(sum(1 for r in rows if not r["success"])),
        "wall_time_s": round(float(wall_time_s), 3),
        "termination": termination,
        "final_sigma": optimizer.sigma,
        "final_max_std": optimizer.max_std,
        "final_resolution_factor": resolution_factor,
        "best": None if incumbent is None else best_summary(incumbent.record, problem),
    }


def dry_run(pool: WorkerPool, problem: FitProblem, starts: Sequence[Start], fit_dir: Path) -> None:
    """Evaluate every start once at full resolution (dry_run/evaluations.csv)
    and print the wall times."""
    directory = fit_dir / DRY_RUN_DIR
    directory.mkdir(parents=True, exist_ok=True)
    log = EvaluationLog(directory / EVALUATIONS_FILE, evaluation_columns(problem.space.names, problem.prefixes))
    try:
        jobs = [(start.index, start.unit, FULL_RESOLUTION, None) for start in starts]
        started = time.perf_counter()
        records = []
        for record in pool.evaluate(jobs):
            record.update(restart=record["eval_id"], generation=0, member=MEMBER_NONE, tag=TAG_DRY_RUN)
            log.write(record)
            records.append(record)
    finally:
        log.close()
    total = time.perf_counter() - started
    print(f"dry run: {len(records)} evaluations in {total:.1f} s on {pool.n_workers} worker(s)", flush=True)
    for record in sorted(records, key=lambda r: r["eval_id"]):
        status = "ok" if record["success"] else f"FAILED ({record['error']})"
        solve = record["solve_wall_time_s"]
        print(
            f"  start {record['eval_id']} on worker {record['worker']}: {status}, loss {record['loss']:.4f}, J {record['J']:.4f}, "
            f"thresholds {record['core_threshold_star']:.2f}/{record['edema_threshold_star']:.2f}, "
            f"wall {record['wall_time_s']:.1f} s, solve {solve if solve is None else round(solve, 1)} s"
            + (" (first on its worker: includes the compilation)" if record["first_on_worker"] else ""),
            flush=True,
        )
    later = [r["wall_time_s"] for r in records if not r["first_on_worker"]]
    if later:
        print(f"  wall time per evaluation after the compilation: mean {np.mean(later):.1f} s (n = {len(later)})", flush=True)


def resolve_best(pool: WorkerPool, problem: FitProblem, fit_dir: Path, restart: int | None) -> dict[str, Any]:
    """
    Re-solve the overall best evaluation (or restart k's best) at full
    resolution in one worker, writing best/ (``save_resolve``) and
    best/objective.json (the objective recomputed next to the logged
    values); the record goes into summary.json.
    """
    summary_path = fit_dir / SUMMARY_FILE
    summary = read_json(summary_path)
    if restart is None:
        if not summary.get("best"):
            raise ValueError(f"{summary_path} has no best evaluation to resolve.")
        restart = int(summary["best"]["restart"])
    entry = next((r for r in summary["restarts"] if int(r["restart"]) == restart), None)
    if entry is None or not entry.get("best"):
        raise ValueError(f"restart {restart} has no best evaluation in {summary_path}.")
    eval_id = int(entry["best"]["eval_id"])
    rows = read_evaluations(fit_dir / f"restart_{restart}" / EVALUATIONS_FILE)
    row = next((r for r in rows if r["eval_id"] == eval_id), None)
    if row is None:
        raise ValueError(f"evaluation {eval_id} is not in restart_{restart}/{EVALUATIONS_FILE}.")
    unit = np.array([float(row[f"u_{name}"]) for name in problem.space.names])
    best_dir = fit_dir / BEST_DIR
    if best_dir.exists():
        raise FileExistsError(f"{best_dir} exists; move it away before resolving again.")
    print(f"resolve: restart {restart}, evaluation {eval_id} (logged loss {row['loss']:.4f} at factor {row['resolution_factor']:g}) at factor 1.0", flush=True)
    started = time.perf_counter()
    record = next(iter(pool.evaluate([(eval_id, unit, FULL_RESOLUTION, str(best_dir))])))
    record.update(restart=restart, generation=row.get("generation"), member=row.get("member"), tag="resolve")
    logged = best_summary(row, problem)
    resolved = best_summary(record, problem)
    objective = {
        "restart": restart,
        "eval_id": eval_id,
        "logged": logged,
        "resolved": resolved,
        "loss_difference": None if not record["success"] else float(record["loss"]) - float(row["loss"]),
        "note": (
            "resolved: the objective of the re-solve at resolution_factor 1.0 in best/; logged: the evaluation's row "
            "(at its own resolution factor); the two agree up to floating-point differences when the factor was 1.0"
        ),
        "wall_time_s": round(time.perf_counter() - started, 3),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(best_dir / OBJECTIVE_FILE, objective)
    summary["resolve"] = {
        "restart": restart,
        "eval_id": eval_id,
        "directory": str(best_dir),
        "success": record["success"],
        "error": record["error"],
        "logged_loss": row["loss"],
        "resolved_loss": record["loss"],
        "created": objective["created"],
    }
    write_json(summary_path, summary)
    status = "ok" if record["success"] else f"FAILED ({record['error']})"
    print(f"resolve: {status}, loss {record['loss']:.4f} (logged {row['loss']:.4f}), written to {best_dir}", flush=True)
    return objective


# --- the fit directory ---


def write_fit_directory(
    fit_dir: Path,
    problem: FitProblem,
    settings: FitSettings,
    starts: Sequence[Start],
    starts_record: Mapping[str, Any],
    checks: Mapping[str, Any],
    design: Mapping[str, Any],
    session_labels: Path,
    sessions: str,
    gpus: Sequence[str | None],
    jobs_per_gpu: int,
) -> None:
    """spec.json, search_space.json, base_config.json and starts.json of a
    new fit directory (created here, never overwritten)."""
    fit_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(problem.space.path, fit_dir / "search_space.json")
    centre = problem.evaluation_config(np.full(problem.space.dimension, 0.5), FULL_RESOLUTION)
    centre["_design"] = (
        "the config of the unit-cube centre with the patient's volumes and its timeline: the design-time checks ran "
        "on its corner variants (no solve); see spec.json"
    )
    write_config(centre, fit_dir / "base_config.json")
    write_json(fit_dir / STARTS_FILE, {"starts": [s.record(problem.space.names) for s in starts], **starts_record})
    spec = {
        "name": fit_dir.name,
        "patient": problem.data.record(),
        "session_labels": str(session_labels),
        "sessions_requested": sessions,
        "sessions_simulated": [s.id for s in problem.sessions],
        "label_conventions": LABEL_CONVENTIONS,
        "data_checks": checks,
        "protocol": problem.protocol.record(),
        "timeline": problem.timeline.record(),
        "search_space": problem.space.record(),
        "objective": problem.objective.record(),
        "parametrisation": problem.record(),
        "cmaes": settings.record(),
        "time_stepping": design["time_stepping"],
        "design_checks": {key: value for key, value in design.items() if key != "time_stepping"},
        "n_starts": len(starts),
        "gpus": ["cpu" if gpu is None else gpu for gpu in gpus],
        "jobs_per_gpu": jobs_per_gpu,
        "n_workers": len(gpus) * jobs_per_gpu,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    write_json(fit_dir / SPEC_FILE, spec)


def prepare_fit(args: argparse.Namespace, fit_dir: Path, gpus: Sequence[str | None]) -> tuple[FitProblem, FitSettings, list[Start]]:
    """
    A new fit directory (or the reloaded one on --resume): the problem,
    the settings and the starts.
    """
    n_workers = len(gpus) * int(args.jobs_per_gpu)
    schedule = ResolutionSchedule.parse(args.resolution_schedule)
    if args.resume:
        spec_path = fit_dir / SPEC_FILE
        if not spec_path.is_file():
            raise FileNotFoundError(f"--resume: {spec_path} not found.")
        spec = read_json(spec_path)
        problem = problem_from_fit_dir(fit_dir, spec)
        starts = load_starts(fit_dir / STARTS_FILE, problem)
        settings = FitSettings.from_args(args, n_workers, len(starts), problem.space.dimension)
        settings.check_against(spec["cmaes"])
        print(f"resuming {fit_dir}: {len(starts)} start(s), popsize {settings.popsize}", flush=True)
        return problem, settings, starts
    if fit_dir.exists():
        raise FileExistsError(f"{fit_dir} exists; a fit directory is never overwritten (continue it with --resume).")
    problem = build_problem(
        args.config,
        args.search_space,
        args.patient,
        args.patient_root,
        args.session_labels,
        args.sessions,
        args.session_weights,
        args.loss,
        parse_preop_time_range(args.preop_time_range),
        float(args.steps_per_day),
        schedule.factors,
    )
    checks = check_patient_data(problem.data)
    print(format_timeline(problem.timeline), flush=True)
    print(
        f"objective: loss {problem.objective.mode} over {list(problem.objective.session_ids)} with weights "
        f"{problem.objective.weights}; simulated sessions {[s.id for s in problem.sessions]}",
        flush=True,
    )
    print(
        f"seedable voxels: {problem.seed_map.geometry.n_voxels}, box {problem.seed_map.lo.tolist()} .. {problem.seed_map.hi.tolist()}; "
        f"preop_time clamp {list(problem.preop_time_range)} days; horizon offset {problem.horizon_offset} days",
        flush=True,
    )
    design = run_design_checks(problem, schedule.factors)
    raised = {float(f): entry["n_steps_resolved"] for f, entry in design["time_stepping"].items() if entry["raised"]}
    if raised:
        print(f"WARNING: the solver's stability estimate raises n_steps at the corner: {raised}; using the raised counts.", flush=True)
        problem = replace(problem, n_steps={**problem.n_steps, **raised})
    for factor_text, entry in design["time_stepping"].items():
        lo, hi = problem.dt_range(float(factor_text))
        print(
            f"resolution factor {factor_text}: n_steps {problem.n_steps[float(factor_text)]} (solver resolves "
            f"{entry['n_steps_resolved']} at the corner D {entry['corner']['white_matter_diffusivity']:.3g}, "
            f"T {entry['corner']['stopping_time']:.0f}), dt in [{lo:.4g}, {hi:.4g}] day, low-resolution grid {entry['lowres_grid_shape']}",
            flush=True,
        )
    sweep_dir = None if args.no_sweep_init else Path(args.init_from_sweep)
    starts, starts_record = resolve_starts(problem, sweep_dir, int(args.top), args.init_values)
    print(format_starts(starts, problem), flush=True)
    settings = FitSettings.from_args(args, n_workers, len(starts), problem.space.dimension)
    write_fit_directory(
        fit_dir, problem, settings, starts, starts_record, checks, design, Path(args.session_labels).resolve(),
        args.sessions, gpus, int(args.jobs_per_gpu),
    )
    print(f"fit directory: {fit_dir}; popsize {settings.popsize} on {n_workers} worker(s)", flush=True)
    return problem, settings, starts


def run_fit(args: argparse.Namespace) -> int:
    """The fit subcommand (the module docstring)."""
    fit_dir = Path(args.output_dir).resolve() / args.name
    gpus = parse_gpus(args.gpus)
    problem, settings, starts = prepare_fit(args, fit_dir, gpus)
    started = time.perf_counter()
    pool = WorkerPool(problem, gpus, int(args.jobs_per_gpu), fit_dir / CACHE_DIR)
    try:
        pool.wait_ready()
        if args.dry_run:
            dry_run(pool, problem, starts, fit_dir)
            return 0
        restarts: list[dict[str, Any]] = []
        for k, start in enumerate(starts):
            restarts.append(run_restart(k, start, problem, settings, pool, fit_dir, bool(args.resume)))
            write_summary(fit_dir, problem, settings, restarts, pool.record(), time.perf_counter() - started, False)
        summary = write_summary(fit_dir, problem, settings, restarts, pool.record(), time.perf_counter() - started, True)
        best = summary["best"]
        print(
            f"fit done: {summary['n_evaluations']} evaluations in {summary['wall_time_s'] / 60:.1f} min; best restart "
            f"{best['restart']}, evaluation {best['eval_id']}, loss {best['loss']:.4f}" if best else "fit done: no successful evaluation",
            flush=True,
        )
        if not args.no_resolve and best:
            resolve_best(pool, problem, fit_dir, None)
    finally:
        pool.close()
    return 0


def run_resolve(args: argparse.Namespace) -> int:
    """The resolve subcommand: one worker on --gpu re-solves the best."""
    fit_dir = Path(args.fit_dir).resolve()
    spec = read_json(fit_dir / SPEC_FILE)
    problem = problem_from_fit_dir(fit_dir, spec)
    pool = WorkerPool(problem, [args.gpu], 1, fit_dir / CACHE_DIR)
    try:
        pool.wait_ready()
        resolve_best(pool, problem, fit_dir, None if args.restart is None else int(args.restart))
    finally:
        pool.close()
    return 0


def run_starts(args: argparse.Namespace) -> int:
    """The starts subcommand: the resolved starts, printed (no GPU)."""
    problem = build_problem(
        args.config,
        args.search_space,
        args.patient,
        args.patient_root,
        args.session_labels,
        args.sessions,
        args.session_weights,
        args.loss,
        parse_preop_time_range(args.preop_time_range),
        float(args.steps_per_day),
        [FULL_RESOLUTION],
    )
    sweep_dir = None if args.no_sweep_init else Path(args.init_from_sweep)
    starts, record = resolve_starts(problem, sweep_dir, int(args.top), args.init_values)
    if record.get("sweep") and not record["sweep"].get("missing"):
        ranking = record["sweep"]
        print(
            f"sweep {ranking['sweep_dir']}: {ranking['n_successful_rows']}/{ranking['n_qoi_rows']} successful rows, "
            f"{ranking['n_ranked_runs']} ranked runs (adjuvant cycle {ranking['sweep_adjuvant_cycle_days']} days in the sweep, "
            f"{ranking['fit_adjuvant_cycle_days']} in the fit)",
            flush=True,
        )
    print(format_starts(starts, problem), flush=True)
    return 0


# --- command line ---


def _add_problem_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE), help=f"search-space JSON (default {DEFAULT_SEARCH_SPACE.name})")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="base config JSON")
    parser.add_argument("--patient", default=DEFAULT_PATIENT, help="subject id")
    parser.add_argument("--patient-root", default=str(DEFAULT_PATIENT_ROOT), help="processed data root")
    parser.add_argument("--session-labels", default=str(DEFAULT_SESSION_LABELS), help="session labels tsv")
    parser.add_argument(
        "--sessions", default=DEFAULT_SESSIONS,
        help=f"the objective's sessions: a range '01-08' or a list 'ses-01,ses-03'; the later ones must hold the post-op session (default {DEFAULT_SESSIONS})",
    )
    parser.add_argument("--session-weights", default="", help="weights 'ses-01=1,ses-02=1,...' (default 1 each)")
    parser.add_argument("--loss", choices=LOSS_MODES, default=DEFAULT_LOSS, help=f"the profiled objective (default {DEFAULT_LOSS})")
    parser.add_argument("--preop-time-range", default=DEFAULT_PREOP_TIME_RANGE, help=f"clamp of preop_time in days 'lo,hi' (default {DEFAULT_PREOP_TIME_RANGE})")
    parser.add_argument("--steps-per-day", type=float, default=DEFAULT_STEPS_PER_DAY, help=f"the fixed n_steps is ceil(steps_per_day (preop_time_max + horizon)) (default {DEFAULT_STEPS_PER_DAY:g})")


def _add_start_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--init-from-sweep", default=str(DEFAULT_SWEEP_DIR), help="a patient_sensitivity_analysis.py sweep directory")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"starts taken from the sweep (default {DEFAULT_TOP})")
    parser.add_argument("--no-sweep-init", action="store_true", help="no sweep starts")
    parser.add_argument("--init-values", default="", help="one manual start 'name=value,...' in physical units, every factor named")


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="parent of the fit directory")
    parser.add_argument("--name", required=True, help="fit directory name")
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="comma-separated CUDA device ids, one worker each; '' = one CPU worker")
    parser.add_argument("--jobs-per-gpu", type=int, default=DEFAULT_JOBS_PER_GPU, help=f"workers per device (default {DEFAULT_JOBS_PER_GPU})")
    parser.add_argument("--popsize", default=DEFAULT_POPSIZE, help=f"'auto' = {POPSIZE_PER_WORKER} x workers, 'gift' = 4 + floor(3 ln N), or an integer (default {DEFAULT_POPSIZE})")
    parser.add_argument("--sigma0", default=DEFAULT_SIGMA0, help=f"initial step size, one value or one per start (default {DEFAULT_SIGMA0})")
    parser.add_argument("--generations", default=DEFAULT_GENERATIONS, help=f"generations per restart, one value or one per start (default {DEFAULT_GENERATIONS})")
    parser.add_argument("--min-sigma", type=float, default=DEFAULT_MIN_SIGMA, help="stop a restart when sigma max(d) falls below it (default 0 = off)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--resolution-schedule", default=DEFAULT_RESOLUTION_SCHEDULE, help=f"'fraction:factor,...' by generation fraction (default {DEFAULT_RESOLUTION_SCHEDULE})")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted fit from its checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="evaluate the starts once and exit")
    parser.add_argument("--no-resolve", action="store_true", help="do not re-solve the best at the end")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    starts = commands.add_parser("starts", help="print the resolved starts (no GPU)")
    _add_problem_args(starts)
    _add_start_args(starts)
    fit = commands.add_parser("fit", help="design-time checks, starts, restarts, summary, resolve")
    _add_problem_args(fit)
    _add_start_args(fit)
    _add_fit_args(fit)
    resolve = commands.add_parser("resolve", help="re-solve the best evaluation of a fit directory into best/")
    resolve.add_argument("--fit-dir", required=True, help="the fit directory")
    resolve.add_argument("--restart", type=int, default=None, help="resolve this restart's best instead of the overall best")
    resolve.add_argument("--gpu", required=True, help="the CUDA device id of the worker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "starts":
        return run_starts(args)
    if args.command == "fit":
        return run_fit(args)
    return run_resolve(args)


if __name__ == "__main__":
    sys.exit(main())
