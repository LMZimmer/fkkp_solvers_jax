#!/usr/bin/env python
"""Variance-based (Sobol') sensitivity analysis of the Stupp-protocol forward
model, fisher_kpp_jax.StuppFKPPSolver.

A Saltelli design over the factors of a search-space file is run through
the solver (one process per run, one run directory each), quantities of
interest (QoIs) are read off the saved final cell densities afterwards,
and SALib's Sobol' estimators give the first-order (S1) and total-order
(ST) indices of every QoI with bootstrap confidence intervals.

Search-space file (JSON, --search-space; default
fisher_kpp_jax/search_spaces/stupp_fkpp_search_space.json). It reads like
a config: "solver" names StuppFKPPSolver (checked against the class the
script uses), keys starting with '_' are comments and are dropped, every
other key must be a StuppFKPPSolver parameter and is either
  - a plain JSON value: a fixed override written into every run config, or
  - {"min": a, "max": b, "scale": "linear" | "log"}: a factor, sampled on
    the unit cube (u in [0, 1)) and transformed to a + u (b - a) for
    "linear" or 10 ** (log10 a + u (log10 b - log10 a)) for "log".
Everything not listed comes from the base config (--config, default
scripts/stupp_config_example.json, read with fisher_kpp_jax.read_config so
every run config carries absolute volume paths). chemo_times and rt_times
are shifted by resection_time - base resection_time, so the treatment
block keeps its offset after surgery; the base config's time step is
copied through (the solver raises a coarse step to its stability estimate
itself).

Seed reinterpretation. The three gaussian_seed_{x,y,z}_fraction entries
must be linear factors within [0, 1], but their sampled value u is NOT the
fraction of the image axis the solver takes: per axis it is a coordinate
in the bounding box of the seedable voxels, the voxels carrying the base
config's resection_cavity label with wm + gm > 0, fraction =
lo + u (hi - lo) with lo/hi the box bounds in the solver's (v + 0.5) / n
fraction convention. The point is then projected onto the nearest
seedable voxel (Euclidean distance in voxel units) and that voxel's
fractions go into the config, so the Sobol' indices of the three seed
factors are those of the seed position within the cavity box. design.csv
records u, the projected fractions and the voxel; spec.json the box and
the seedable voxel count.

Design (SALib.sample.sobol: Saltelli scheme on a scrambled Sobol'
sequence). k factors, N = 2 ** log2_n base points, N (k + 2) runs
(N (2k + 2) with --second-order). SALib's row order, on which the analysis
depends and which tests/test_sensitivity_analysis.py verifies against the
installed version: N blocks of k + 2 rows; in block j, position 0 is A_j,
positions 1..k are A_B^(i) (column i of A_j replaced from B_j) in factor
order and position k + 1 is B_j (with --second-order, positions
k + 1..2k are B_A^(i) and B_j comes last). Run names are
r{row:04d}_{A | B | AB-<factor> | BA-<factor>}, row being the block j.

Output layout:
  <output-dir>/<name>/
    search_space.json    copy of the search-space file used
    base_config.json     the base config as read (absolute volume paths)
    spec.json            search space, base config path, seed, N, k, factor
                         order, SALib version, seed box, seedable voxel count
    design.csv           one line per run: run_name, index (position in the
                         SALib array), row (block), matrix (A, B, AB:<factor>),
                         u_<factor> and <factor> (the transformed value; seed
                         factors: the projected fractions) for every factor,
                         seed_voxel_i/j/k
    configs/<run>.json   the run configs
    logs/<run>.log       stdout/stderr of the runs
    runs/<run>/          Result.save output (config.json, result.json,
                         initial_/final_cell_density.nii.gz)
    run_status.csv       appended as runs finish: run_name, success,
                         exit_code, wall_time_s, error, final_time, n_steps
    qoi.csv              one line per run: run_name, index, row, matrix,
                         success, the QoIs, final_time, n_steps, wall_time_s
    qoi_summary.json     counts: runs, successes, empty compartments, NaNs
    sobol.csv            one line per (qoi, factor): S1, S1_conf, ST, ST_conf,
                         S1_half, ST_half, n_blocks_used
    sobol_summary.json   N, k, factor order, SALib version and per QoI the
                         dropped blocks, the sum of S1, the ranking by ST and
                         the factors whose ST is not distinguishable from zero
    figures/             heatmaps of ST and S1 (factors x QoIs), per-QoI bar
                         charts of S1 and ST with confidence whiskers, per-QoI
                         scatter grids over the A and B rows (PNG + PDF)

QoIs, computed from runs/<run>/final_cell_density.nii.gz with c_v the
final density, dV the voxel volume in mm^3 (NIfTI zooms), x_v the voxel
centres in mm (index times zooms), x_seed the projected seed voxel in mm,
Omega_tau = {v: c_v >= tau} (tau_core = 0.6, tau_edema = 0.3) and
M = dV sum_v c_v:
  mass, log10_mass             M and log10 M (NaN when M = 0)
  V_core, V_edema              dV |Omega_tau| in mm^3
  log10_V_core, log10_V_edema  log10(V + dV)
  r95_core, r95_edema          95th percentile of |x_v - x_seed| over
                               Omega_tau; 0 when Omega_tau is empty (the
                               treatment eliminated that compartment, a
                               valid outcome, not a failure)
  centroid_drift               |xbar - x_seed|, xbar the mass centroid
  R_g                          sqrt(sum_v c_v |x_v - xbar|^2 / sum_v c_v)
  anisotropy, log10_anisotropy lambda_max / lambda_min of the mass-weighted
                               covariance of the voxel centres
  wm_fraction                  sum_v c_v p_v / sum_v c_v with p the base
                               config's white-matter probability map
  n_core, n_edema              |Omega_tau| (voxel counts)
The four mass-weighted QoIs (centroid_drift, R_g, anisotropy, wm_fraction)
are NaN when M < 1e-3 dV (no mass left to locate). final_time, n_steps and
wall_time_s are carried over from result.json.

Sobol' analysis (SALib.analyze.sobol). Analysed columns: log10_mass,
V_core, V_edema, log10_V_core, log10_V_edema, r95_core, r95_edema,
centroid_drift, R_g, log10_anisotropy, wm_fraction. The response is
assembled in SALib's row order from qoi.csv (joined on index); SALib needs
complete blocks and infers N from the length, so a block is dropped
entirely when any of its runs failed or has a NaN value (reported per
QoI). SALib uses the Saltelli et al. 2010 (Comput. Phys. Commun. 181:259)
first-order and the Jansen 1999 (CPC 117:35) total-order estimators; the
*_conf values are normal-approximation 95 % half-widths from the bootstrap
standard deviation (resampling blocks), not percentile intervals. As a
convergence check the point estimates are repeated on the first half of
the blocks (S1_half, ST_half). Whether the design holds the second-order
rows is taken from spec.json, so the --second-order flag of design/all
reaches the analysis (which then writes sobol_S2.csv as well). SALib
treats seed 0 as "unseeded", so the bootstrap is seeded with --seed + 1.

Run from the project root, e.g.:
  python scripts/sensitivity_analysis.py design --output-dir /mnt/Drive4/lucas/SAILOR/sa --name sa_sub15
  python scripts/sensitivity_analysis.py run --sweep-dir /mnt/Drive4/lucas/SAILOR/sa/sa_sub15 --gpus 1,2,3,5
  python scripts/sensitivity_analysis.py qoi --sweep-dir /mnt/Drive4/lucas/SAILOR/sa/sa_sub15
  python scripts/sensitivity_analysis.py analyze --sweep-dir /mnt/Drive4/lucas/SAILOR/sa/sa_sub15
  python scripts/sensitivity_analysis.py all --output-dir /mnt/Drive4/lucas/SAILOR/sa --name sa_sub15 --gpus 1,2,3,5
  python scripts/sensitivity_analysis.py all --output-dir runs/ --name check --log2-n 2 --gpus ""   (CPU only)
The run step is resumable: runs whose result.json reports success are
skipped, partial run directories are deleted and redone; nothing is ever
written outside <output-dir>/<name>/.

Budget: N = 1024 and k = 13 give 15 360 runs; at about 12 s per run on
one GPU (1 mm SAILOR grid, day-360 horizon) that is about 12.8 h on 4 GPUs
and about 85 GB of run directories (5.6 MB each). One design point of the
example config measured 7.4 s of process wall time (5.6 s solve, 3975
steps) on a Quadro RTX 8000 and 2.3 MB on disk, so the figures above are
conservative.
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
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend. fisher_kpp_jax (and with it jax) is imported
# lazily, inside the subcommands that need the solver: design and run-one.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

SOLVER_NAME = "StuppFKPPSolver"
SOLVER_KEY = "solver"
DEFAULT_SEARCH_SPACE = _ROOT / "fisher_kpp_jax" / "search_spaces" / "stupp_fkpp_search_space.json"
DEFAULT_CONFIG = _ROOT / "scripts" / "stupp_config_example.json"
DEFAULT_GPUS = "1,2,3,5"
DEFAULT_LOG2_N = 10
DEFAULT_DESIGN_SEED = 1
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_SEED = 0
TAU_CORE = 0.6
TAU_EDEMA = 0.3
# Below this many voxel volumes of mass the mass-weighted QoIs are NaN.
MASS_FLOOR_VOXELS = 1e-3

SEED_KEYS: tuple[str, ...] = tuple(f"gaussian_seed_{axis}_fraction" for axis in "xyz")
# Event times shifted with the sampled resection time.
SHIFTED_TIME_KEYS: tuple[str, ...] = ("chemo_times", "rt_times")
# Base config entries the design needs (the volumes for the seed geometry,
# the times for the shift).
BASE_KEYS_NEEDED: tuple[str, ...] = (
    "white_matter_pbmap",
    "gray_matter_pbmap",
    "resection_cavity",
    "resection_time",
    *SHIFTED_TIME_KEYS,
)

STATUS_COLUMNS = ["run_name", "success", "exit_code", "wall_time_s", "error", "final_time", "n_steps"]
QOI_NAMES = [
    "mass",
    "log10_mass",
    "V_core",
    "V_edema",
    "log10_V_core",
    "log10_V_edema",
    "r95_core",
    "r95_edema",
    "centroid_drift",
    "R_g",
    "anisotropy",
    "log10_anisotropy",
    "wm_fraction",
    "n_core",
    "n_edema",
]
MASS_WEIGHTED_QOIS: tuple[str, ...] = ("centroid_drift", "R_g", "anisotropy", "wm_fraction")
QOI_COLUMNS = ["run_name", "index", "row", "matrix", "success", *QOI_NAMES, "final_time", "n_steps", "wall_time_s"]
ANALYSED_QOIS = [
    "log10_mass",
    "V_core",
    "V_edema",
    "log10_V_core",
    "log10_V_edema",
    "r95_core",
    "r95_edema",
    "centroid_drift",
    "R_g",
    "log10_anisotropy",
    "wm_fraction",
]
SOBOL_COLUMNS = ["qoi", "factor", "S1", "S1_conf", "ST", "ST_conf", "S1_half", "ST_half", "n_blocks_used"]

# Figure colours (a validated categorical pair and neutral inks).
COLOR_ST = "#2a78d6"
COLOR_S1 = "#eb6834"
COLOR_POINTS = "#52514e"
COLOR_TEXT = "#0b0b0b"


# --- search space ---


@dataclass(frozen=True)
class Factor:
    """
    A swept parameter: its physical range and the sampling scale.

    Attributes:
        name: The StuppFKPPSolver parameter name.
        low: Lower bound of the range.
        high: Upper bound of the range.
        scale: "linear" (uniform on [low, high]) or "log" (log10-uniform).
    """

    name: str
    low: float
    high: float
    scale: Literal["linear", "log"]

    def transform(self, u: NDArray | float) -> NDArray:
        """The physical value(s) of unit-cube coordinate(s) u."""
        return transform_factor(u, self.low, self.high, self.scale)


@dataclass(frozen=True)
class SearchSpace:
    """
    A loaded search-space file.

    Attributes:
        factors: The factors by name, in file order (the factor order of
            the design).
        overrides: Fixed values written into every run config, by name.
        source: The file's entries as read, comments included.
    """

    factors: dict[str, Factor]
    overrides: dict[str, Any]
    source: dict[str, Any]

    @property
    def names(self) -> list[str]:
        """The factor order."""
        return list(self.factors)


def transform_factor(
    u: NDArray | float, low: float, high: float, scale: str
) -> NDArray:
    """
    Map unit-cube coordinates to a factor's physical range.

    Args:
        u: Coordinates in [0, 1], any shape.
        low: Lower bound of the range (> 0 for "log").
        high: Upper bound of the range.
        scale: "linear": low + u (high - low); "log":
            10 ** (log10 low + u (log10 high - log10 low)).

    Returns:
        The transformed values, a float64 array of u's shape.
    """
    u = np.asarray(u, dtype=np.float64)
    if scale == "log":
        lo, hi = np.log10(low), np.log10(high)
        return 10.0 ** (lo + u * (hi - lo))
    if scale == "linear":
        return low + u * (high - low)
    raise ValueError(f"scale must be 'linear' or 'log', got {scale!r}.")


def _parse_factor(name: str, entry: Mapping[str, Any], where: str) -> Factor:
    """A factor from a {"min", "max", "scale"} search-space entry."""
    if set(entry) != {"min", "max", "scale"}:
        raise ValueError(
            f'{where}: {name} must be {{"min": <float>, "max": <float>, "scale": '
            f'"linear" | "log"}}, got {dict(entry)!r}.'
        )
    scale = entry["scale"]
    if scale not in ("linear", "log"):
        raise ValueError(f"{where}: {name}: scale must be 'linear' or 'log', got {scale!r}.")
    low, high = float(entry["min"]), float(entry["max"])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(f"{where}: {name}: min < max must be finite, got {low!r}, {high!r}.")
    if scale == "log" and low <= 0:
        raise ValueError(f"{where}: {name}: a log-scaled range needs min > 0, got {low!r}.")
    return Factor(name, low, high, scale)


def load_search_space(
    source: str | Path | Mapping[str, Any],
    config_keys: Iterable[str],
    solver_name: str = SOLVER_NAME,
) -> SearchSpace:
    """
    Load and check a search space (see the module docstring).

    Args:
        source: Path of the JSON file, or its entries as a mapping.
        config_keys: The solver's parameter names (``config_keys()``);
            every non-comment key must be one of them or "solver".
        solver_name: The class name a "solver" entry must equal.

    Returns:
        The search space: factors in file order, fixed overrides, the
        source entries. The three seed fractions must be linear factors
        within [0, 1].
    """
    if isinstance(source, Mapping):
        entries: Any = dict(source)
        where = "search space"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"search space not found: {path}")
        entries = json.loads(path.read_text(encoding="utf-8"))
        where = f"search space {path}"
    if not isinstance(entries, Mapping):
        raise ValueError(f"{where}: must be a JSON object.")
    known = frozenset(config_keys)
    factors: dict[str, Factor] = {}
    overrides: dict[str, Any] = {}
    for key, value in entries.items():
        if key.startswith("_"):
            continue
        if key == SOLVER_KEY:
            if value != solver_name:
                raise ValueError(f"{where}: names solver {value!r}, not {solver_name}.")
            continue
        if key not in known:
            raise ValueError(
                f"{where}: unknown key {key!r}; every key must be a {solver_name} "
                f"parameter or {SOLVER_KEY!r}."
            )
        if isinstance(value, Mapping) and set(value) & {"min", "max", "scale"}:
            factors[key] = _parse_factor(key, value, where)
        else:
            overrides[key] = value
    for key in SEED_KEYS:
        factor = factors.get(key)
        if factor is None:
            raise ValueError(
                f"{where}: {key} must be a linear factor within [0, 1] (its value is "
                f"reinterpreted inside the cavity box), got {entries.get(key)!r}."
            )
        if factor.scale != "linear" or factor.low < 0 or factor.high > 1:
            raise ValueError(
                f"{where}: {key} must be a linear factor within [0, 1], got "
                f"[{factor.low}, {factor.high}] {factor.scale}."
            )
    return SearchSpace(factors, overrides, dict(entries))


# --- seed geometry ---


@dataclass(frozen=True)
class SeedGeometry:
    """
    The seedable voxels of a base config and their bounding box.

    Attributes:
        shape: The full-resolution grid shape.
        voxels: The seedable voxel indices, (n, 3) int64.
        bbox_lo: Lower box bound per axis, in the (v + 0.5) / n fraction
            convention.
        bbox_hi: Upper box bound per axis, same convention.
    """

    shape: tuple[int, int, int]
    voxels: NDArray
    bbox_lo: NDArray
    bbox_hi: NDArray

    @property
    def n_voxels(self) -> int:
        """The number of seedable voxels."""
        return int(len(self.voxels))


def seed_geometry(cavity: NDArray, wm: NDArray, gm: NDArray) -> SeedGeometry:
    """
    The seedable voxels: cavity voxels carrying tissue (wm + gm > 0), the
    voxels the solver accepts a seed in.

    Args:
        cavity: Boolean resection-cavity mask (the solver's loaded
            ``params["resection_cavity"]``).
        wm: White-matter probability map, same shape.
        gm: Gray-matter probability map, same shape.

    Returns:
        The seedable voxels and their bounding box.
    """
    seedable = np.asarray(cavity, dtype=bool) & (
        (np.asarray(wm, dtype=np.float64) + np.asarray(gm, dtype=np.float64)) > 0
    )
    voxels = np.argwhere(seedable).astype(np.int64)
    if not len(voxels):
        raise ValueError("no seedable voxel: no cavity voxel carries tissue (wm + gm > 0).")
    n = np.asarray(seedable.shape, dtype=np.float64)
    return SeedGeometry(
        shape=tuple(int(s) for s in seedable.shape),
        voxels=voxels,
        bbox_lo=(voxels.min(axis=0) + 0.5) / n,
        bbox_hi=(voxels.max(axis=0) + 0.5) / n,
    )


def project_seeds(u: NDArray, geometry: SeedGeometry) -> tuple[NDArray, NDArray]:
    """
    Reinterpret unit-cube seed coordinates inside the seedable box and
    project them onto the nearest seedable voxel.

    Args:
        u: Coordinates in [0, 1], (n, 3) (or (3,) for one point).
        geometry: The seedable voxels and their box.

    Returns:
        (fractions, voxels): the projected voxels' fractions in the
        (v + 0.5) / n convention, (n, 3) float64, and the voxels, (n, 3)
        int64. A point that lies on a seedable voxel projects onto itself.
    """
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    if u.shape[1:] != (3,):
        raise ValueError(f"seed coordinates must be (n, 3), got {u.shape}.")
    n = np.asarray(geometry.shape, dtype=np.float64)
    fractions = geometry.bbox_lo + u * (geometry.bbox_hi - geometry.bbox_lo)
    points = fractions * n - 0.5  # continuous voxel coordinates
    _, nearest = cKDTree(geometry.voxels).query(points)
    voxels = geometry.voxels[np.asarray(nearest, dtype=np.int64)]
    return (voxels + 0.5) / n, voxels


# --- Saltelli design ---


def salib_problem(names: Sequence[str]) -> dict[str, Any]:
    """The SALib problem on the unit cube for the given factor order."""
    return {"num_vars": len(names), "names": list(names), "bounds": [[0.0, 1.0]] * len(names)}


def saltelli_design(
    names: Sequence[str], log2_n: int, seed: int, second_order: bool = False
) -> NDArray:
    """
    The Saltelli sample of SALib on the unit cube.

    Args:
        names: Factor order (k names).
        log2_n: N = 2 ** log2_n base points.
        seed: Seed of the scrambled Sobol' sequence.
        second_order: Whether the B_A^(i) rows are included.

    Returns:
        The N (k + 2) x k (or N (2k + 2) x k) array in [0, 1), in SALib's
        row order (see the module docstring).
    """
    from SALib.sample import sobol as sobol_sample

    samples = sobol_sample.sample(
        salib_problem(names),
        2 ** int(log2_n),
        calc_second_order=second_order,
        scramble=True,
        seed=int(seed),
    )
    return np.asarray(samples, dtype=np.float64)


def block_size(k: int, second_order: bool = False) -> int:
    """Rows per block of the Saltelli design: k + 2, or 2k + 2 with the
    second-order rows."""
    return 2 * k + 2 if second_order else k + 2


def matrix_labels(names: Sequence[str], second_order: bool = False) -> list[str]:
    """The matrix label of every position within a block, in SALib's row
    order: A, AB:<factor> per factor, (BA:<factor> per factor,) B."""
    labels = ["A", *(f"AB:{name}" for name in names)]
    if second_order:
        labels += [f"BA:{name}" for name in names]
    return [*labels, "B"]


def run_name(row: int, label: str) -> str:
    """The run name of block row and matrix label: r{row:04d}_{A|B|AB-<factor>}."""
    return f"r{row:04d}_{label.replace(':', '-')}"


def design_table(
    samples: NDArray, space: SearchSpace, geometry: SeedGeometry, second_order: bool = False
) -> list[dict[str, Any]]:
    """
    The design bookkeeping of a Saltelli sample: one record per run.

    Args:
        samples: The SALib sample, (n_runs, k) in [0, 1) in factor order.
        space: The search space (its factor order is the column order).
        geometry: The seedable voxels the seed coordinates are projected in.
        second_order: Whether the sample holds the B_A rows.

    Returns:
        Records with run_name, index (position in the sample), row (the
        block), matrix, u_<factor> for every factor, <factor> transformed
        (seed factors: the projected fractions) and seed_voxel_i/j/k.
    """
    names = space.names
    size = block_size(len(names), second_order)
    if samples.ndim != 2 or samples.shape[1] != len(names) or samples.shape[0] % size:
        raise ValueError(
            f"sample shape {samples.shape} does not match {len(names)} factors in "
            f"blocks of {size} rows."
        )
    labels = matrix_labels(names, second_order)
    seed_columns = [names.index(key) for key in SEED_KEYS]
    fractions, voxels = project_seeds(samples[:, seed_columns], geometry)
    values = {
        name: space.factors[name].transform(samples[:, column])
        for column, name in enumerate(names)
        if name not in SEED_KEYS
    }
    for axis, key in enumerate(SEED_KEYS):
        values[key] = fractions[:, axis]
    records = []
    for index in range(samples.shape[0]):
        block, position = divmod(index, size)
        record: dict[str, Any] = {
            "run_name": run_name(block, labels[position]),
            "index": index,
            "row": block,
            "matrix": labels[position],
        }
        record.update({f"u_{name}": float(samples[index, column]) for column, name in enumerate(names)})
        record.update({name: float(values[name][index]) for name in names})
        record.update({f"seed_voxel_{ijk}": int(voxels[index, axis]) for axis, ijk in enumerate("ijk")})
        records.append(record)
    return records


def run_config(
    base: Mapping[str, Any], space: SearchSpace, values: Mapping[str, Any]
) -> dict[str, Any]:
    """
    The config of one run: the base config with the fixed overrides and
    the sampled values substituted, chemo_times and rt_times shifted by
    resection_time - base resection_time.

    Args:
        base: The base config (``read_config``).
        space: The search space (its overrides are applied first).
        values: The factor values of the run (seed factors: the projected
            fractions).

    Returns:
        The run config, with a '_design' comment entry.
    """
    config: dict[str, Any] = {
        SOLVER_KEY: base.get(SOLVER_KEY, SOLVER_NAME),
        "_design": (
            "scripts/sensitivity_analysis.py: the base config with the search space's "
            "overrides and the sampled factor values substituted, chemo_times and "
            "rt_times shifted with resection_time; see design.csv and spec.json in "
            "the parent directory."
        ),
    }
    config.update({key: value for key, value in base.items() if key != SOLVER_KEY})
    config.update(space.overrides)
    config.update(values)
    shift = float(config["resection_time"]) - float(base["resection_time"])
    for key in SHIFTED_TIME_KEYS:
        times = space.overrides.get(key, base[key])
        config[key] = [float(t) + shift for t in times]
    return config


# --- CSV / JSON helpers ---


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    """Write records as CSV (columns: the given order, else the first
    record's keys); NaN is written as an empty field."""
    columns = list(columns) if columns is not None else list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})


def _csv_cell(value: Any) -> Any:
    """A CSV field: NaN/None empty, numpy scalars unwrapped."""
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into records of strings."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(text: Any) -> float:
    """A CSV field as float, NaN when empty."""
    if text is None or text == "":
        return float("nan")
    return float(text)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    """Write a JSON object file (indented, numpy values converted, non-finite
    floats as null)."""
    path.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")


def _jsonable(record: Any) -> Any:
    """JSON conversion of numpy values without importing fisher_kpp_jax
    (inf/nan become null)."""
    if isinstance(record, Mapping):
        return {str(k): _jsonable(v) for k, v in record.items()}
    if isinstance(record, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in record]
    if isinstance(record, np.generic):
        record = record.item()
    if isinstance(record, float) and not np.isfinite(record):
        return None
    return record


# --- design ---


def make_design(
    search_space_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    name: str,
    log2_n: int = DEFAULT_LOG2_N,
    seed: int = DEFAULT_DESIGN_SEED,
    second_order: bool = False,
) -> Path:
    """
    Sample the Saltelli design and write the sweep directory (everything
    but the runs). Refuses to overwrite an existing directory.

    Args:
        search_space_path: The search-space JSON.
        config_path: The base config JSON.
        output_dir: Parent of the sweep directory.
        name: The sweep directory name.
        log2_n: N = 2 ** log2_n base points.
        seed: Seed of the Sobol' sequence.
        second_order: Whether the B_A rows are sampled.

    Returns:
        The sweep directory <output_dir>/<name>.
    """
    from fisher_kpp_jax import StuppFKPPSolver, read_config

    sweep_dir = Path(output_dir) / name
    if sweep_dir.exists():
        raise FileExistsError(f"{sweep_dir} exists; a design is never overwritten.")
    config_path = Path(config_path).resolve()
    search_space_path = Path(search_space_path).resolve()
    base = read_config(config_path, solver=StuppFKPPSolver)
    missing = [key for key in BASE_KEYS_NEEDED if base.get(key) is None]
    if missing:
        raise ValueError(f"the base config {config_path} lacks {missing}, which the design needs.")
    space = load_search_space(search_space_path, StuppFKPPSolver.config_keys())
    # The instance holds the loaded full-resolution volumes (and validates
    # the base config); no solve is run.
    solver = StuppFKPPSolver(base)
    geometry = seed_geometry(
        solver.params["resection_cavity"],
        solver.params["white_matter_pbmap"],
        solver.params["gray_matter_pbmap"],
    )
    samples = saltelli_design(space.names, log2_n, seed, second_order)
    table = design_table(samples, space, geometry, second_order)
    spec = {
        "name": name,
        "search_space_path": str(search_space_path),
        "search_space": space.source,
        "factors": {f.name: {"min": f.low, "max": f.high, "scale": f.scale} for f in space.factors.values()},
        "overrides": space.overrides,
        "base_config": str(config_path),
        "seed": int(seed),
        "log2_n": int(log2_n),
        "N": int(2 ** int(log2_n)),
        "k": len(space.names),
        "factor_names": space.names,
        "second_order": bool(second_order),
        "block_size": block_size(len(space.names), second_order),
        "matrix_labels": matrix_labels(space.names, second_order),
        "n_runs": len(table),
        "salib_version": importlib.metadata.version("SALib"),
        "grid_shape": list(geometry.shape),
        "seed_bbox_lo": geometry.bbox_lo.tolist(),
        "seed_bbox_hi": geometry.bbox_hi.tolist(),
        "n_seedable_voxels": geometry.n_voxels,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    write_design_outputs(sweep_dir, search_space_path, base, space, table, spec)
    return sweep_dir


def write_design_outputs(
    sweep_dir: Path,
    search_space_path: Path,
    base: Mapping[str, Any],
    space: SearchSpace,
    table: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> None:
    """Create the sweep directory and write search_space.json,
    base_config.json, spec.json, design.csv and configs/<run>.json."""
    from fisher_kpp_jax import write_config

    sweep_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("configs", "logs", "runs"):
        (sweep_dir / sub).mkdir()
    shutil.copyfile(search_space_path, sweep_dir / "search_space.json")
    write_config(base, sweep_dir / "base_config.json")
    write_json(sweep_dir / "spec.json", spec)
    write_csv(sweep_dir / "design.csv", table)
    for record in table:
        values = {name: record[name] for name in space.names}
        write_config(run_config(base, space, values), sweep_dir / "configs" / f"{record['run_name']}.json")


# --- runs ---


def run_is_done(run_dir: Path) -> bool:
    """Whether runs/<run>/result.json exists and reports success."""
    record_path = run_dir / "result.json"
    if not record_path.is_file():
        return False
    try:
        return bool(read_json(record_path).get("success"))
    except (OSError, ValueError):
        return False


def run_subprocess(sweep_dir: Path, name: str, gpu: str | None) -> dict[str, Any]:
    """
    Run one design point in its own process (``run-one``) and read its
    result record back.

    Args:
        sweep_dir: The sweep directory.
        name: The run name.
        gpu: The CUDA device id of the slot, None for the CPU.

    Returns:
        A run_status.csv record.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-one",
        "--config",
        str(sweep_dir / "configs" / f"{name}.json"),
        "--run-dir",
        str(sweep_dir / "runs" / name),
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
    record: dict[str, Any] = {
        "run_name": name,
        "success": False,
        "exit_code": code,
        "wall_time_s": round(time.perf_counter() - start, 3),
        "error": "no result.json (see log)",
        "final_time": None,
        "n_steps": None,
    }
    record_path = sweep_dir / "runs" / name / "result.json"
    if record_path.is_file():
        result = read_json(record_path)
        record.update(
            success=bool(result.get("success")),
            error=result.get("error") or "",
            final_time=result.get("final_time"),
            n_steps=result.get("n_steps"),
        )
    return record


def run_sweep(sweep_dir: str | Path, gpus: Sequence[str], jobs_per_gpu: int = 1) -> dict[str, int]:
    """
    Run every design point that is not done yet, one subprocess per run,
    on the given GPU slots (a thread per slot pulling from a queue).

    Args:
        sweep_dir: The sweep directory of ``make_design``.
        gpus: CUDA device ids; empty for the CPU (JAX_PLATFORMS=cpu), then
            with jobs_per_gpu parallel workers.
        jobs_per_gpu: Slots per GPU.

    Returns:
        Counts: skipped (already successful), ok, failed.
    """
    sweep_dir = Path(sweep_dir)
    pending: list[str] = []
    counts = {"skipped": 0, "ok": 0, "failed": 0}
    for record in read_csv(sweep_dir / "design.csv"):
        run_dir = sweep_dir / "runs" / record["run_name"]
        if run_is_done(run_dir):
            counts["skipped"] += 1
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)  # a partial run directory is redone
            pending.append(record["run_name"])
    slots: list[str | None] = [gpu for gpu in gpus for _ in range(jobs_per_gpu)] or [None] * max(1, jobs_per_gpu)
    queue: Queue[str] = Queue()
    for name in pending:
        queue.put(name)
    print(
        f"{len(pending)} runs to do ({counts['skipped']} done already) on "
        f"{len(slots)} slot(s): {'CPU' if not gpus else 'GPU ' + ','.join(gpus)}",
        flush=True,
    )
    status_path = sweep_dir / "run_status.csv"
    lock = threading.Lock()
    start = time.perf_counter()
    with open(status_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()

        def worker(gpu: str | None) -> None:
            while True:
                try:
                    name = queue.get_nowait()
                except Empty:
                    return
                record = run_subprocess(sweep_dir, name, gpu)
                with lock:
                    writer.writerow({key: _csv_cell(record.get(key)) for key in STATUS_COLUMNS})
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


def run_one(config_path: str | Path, run_dir: str | Path) -> int:
    """
    Solve one run config and save the Result into run_dir
    (``solve(store_result=True, outdir=run_dir)``).

    Returns:
        0 on success, 1 if the solver reports a failure.
    """
    from fisher_kpp_jax import StuppFKPPSolver, read_config

    solver = StuppFKPPSolver(read_config(config_path, solver=StuppFKPPSolver))
    result = solver.solve(store_result=True, outdir=run_dir)
    status = "ok" if result.success else f"FAILED: {result.error}"
    print(
        f"{status} (final_time={result.final_time:g}, n_steps={result.n_steps}, "
        f"wall={result.wall_time_s:.1f} s)",
        flush=True,
    )
    return 0 if result.success else 1


# --- quantities of interest ---


def compute_qois(
    density: NDArray,
    zooms: Sequence[float],
    seed_voxel: Sequence[int],
    wm: NDArray,
    tau_core: float = TAU_CORE,
    tau_edema: float = TAU_EDEMA,
) -> dict[str, float]:
    """
    The QoIs of one final cell-density field (see the module docstring).

    Args:
        density: The final cell density on the full-resolution grid.
        zooms: Voxel size per axis in mm.
        seed_voxel: The (projected) seed voxel index.
        wm: The white-matter probability map, same shape as density.
        tau_core: Density threshold of the core compartment.
        tau_edema: Density threshold of the edema compartment.

    Returns:
        The QoIs by name (``QOI_NAMES``); the mass-weighted ones are NaN
        below ``MASS_FLOOR_VOXELS`` voxel volumes of mass. Moments run
        over the nonzero voxels only, in one pass.
    """
    density = np.asarray(density, dtype=np.float64)
    wm = np.asarray(wm, dtype=np.float64)
    if wm.shape != density.shape:
        raise ValueError(f"white-matter map shape {wm.shape} differs from the field's {density.shape}.")
    zooms = np.asarray(zooms, dtype=np.float64)
    voxel_volume = float(np.prod(zooms))
    nonzero = np.nonzero(density > 0)
    c = density[nonzero]
    x = np.stack(nonzero, axis=1).astype(np.float64) * zooms  # voxel centres, mm
    x_seed = np.asarray(seed_voxel, dtype=np.float64) * zooms
    total = float(c.sum())
    mass = voxel_volume * total
    out: dict[str, float] = {"mass": mass, "log10_mass": float(np.log10(mass)) if mass > 0 else np.nan}
    for label, tau in (("core", tau_core), ("edema", tau_edema)):
        selected = c >= tau
        n_selected = int(selected.sum())
        out[f"V_{label}"] = voxel_volume * n_selected
        out[f"log10_V_{label}"] = float(np.log10(voxel_volume * (n_selected + 1)))
        distances = np.linalg.norm(x[selected] - x_seed, axis=1)
        out[f"r95_{label}"] = float(np.percentile(distances, 95)) if n_selected else 0.0
        out[f"n_{label}"] = n_selected
    if total < MASS_FLOOR_VOXELS:
        out.update({key: np.nan for key in (*MASS_WEIGHTED_QOIS, "log10_anisotropy")})
        return out
    weights = c / total
    centroid = weights @ x
    offsets = x - centroid
    out["centroid_drift"] = float(np.linalg.norm(centroid - x_seed))
    out["R_g"] = float(np.sqrt(weights @ np.einsum("ij,ij->i", offsets, offsets)))
    covariance = (offsets * weights[:, None]).T @ offsets
    eigenvalues = np.linalg.eigvalsh(covariance)
    anisotropy = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else np.nan
    out["anisotropy"] = anisotropy
    out["log10_anisotropy"] = float(np.log10(anisotropy)) if np.isfinite(anisotropy) else np.nan
    out["wm_fraction"] = float(weights @ wm[nonzero])
    return out


_WM_CACHE: dict[str, tuple[NDArray, tuple[float, float, float]]] = {}


def _load_wm(path: str) -> tuple[NDArray, tuple[float, float, float]]:
    """The white-matter map and its zooms, cached per process."""
    if path not in _WM_CACHE:
        image = nib.load(path)
        zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
        _WM_CACHE[path] = (np.asarray(image.get_fdata(), dtype=np.float64), zooms)
    return _WM_CACHE[path]


def qoi_record(
    sweep_dir: Path,
    design_record: Mapping[str, Any],
    wm: NDArray,
    wm_zooms: Sequence[float],
    tau_core: float,
    tau_edema: float,
) -> dict[str, Any]:
    """
    The qoi.csv record of one run: the design bookkeeping, the result
    record's final_time / n_steps / wall_time_s and, for a successful run,
    the QoIs of its final field. The field's shape and zooms must match
    the base config's white-matter map.
    """
    name = design_record["run_name"]
    run_dir = sweep_dir / "runs" / name
    record: dict[str, Any] = {
        "run_name": name,
        "index": int(design_record["index"]),
        "row": int(design_record["row"]),
        "matrix": design_record["matrix"],
        "success": False,
    }
    record_path = run_dir / "result.json"
    if not record_path.is_file():
        return record
    result = read_json(record_path)
    record.update(
        final_time=result.get("final_time"), n_steps=result.get("n_steps"), wall_time_s=result.get("wall_time_s")
    )
    field_path = run_dir / "final_cell_density.nii.gz"
    if not result.get("success") or not field_path.is_file():
        return record
    image = nib.load(str(field_path))
    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    density = np.asarray(image.get_fdata(), dtype=np.float64)
    if density.shape != wm.shape or not np.allclose(zooms, wm_zooms, rtol=1e-4, atol=0):
        raise ValueError(
            f"{field_path}: shape {density.shape} / zooms {zooms} differ from the base "
            f"white-matter map's {wm.shape} / {tuple(wm_zooms)}."
        )
    seed_voxel = tuple(int(design_record[f"seed_voxel_{ijk}"]) for ijk in "ijk")
    record.update(compute_qois(density, zooms, seed_voxel, wm, tau_core, tau_edema))
    record["success"] = True
    return record


def _qoi_job(job: tuple[str, dict[str, Any], str, float, float]) -> dict[str, Any]:
    """Worker of ``qoi_table``: (sweep_dir, design record, wm path, taus)."""
    sweep_dir, design_record, wm_path, tau_core, tau_edema = job
    wm, zooms = _load_wm(wm_path)
    return qoi_record(Path(sweep_dir), design_record, wm, zooms, tau_core, tau_edema)


def qoi_table(
    sweep_dir: str | Path, tau_core: float = TAU_CORE, tau_edema: float = TAU_EDEMA, workers: int = 1
) -> list[dict[str, Any]]:
    """
    The QoIs of every run of a sweep directory, in design order.

    Args:
        sweep_dir: The sweep directory.
        tau_core: Density threshold of the core compartment.
        tau_edema: Density threshold of the edema compartment.
        workers: Processes reading the fields (1: in this process).

    Returns:
        The qoi.csv records.
    """
    sweep_dir = Path(sweep_dir)
    design = read_csv(sweep_dir / "design.csv")
    wm_path = str(read_json(sweep_dir / "base_config.json")["white_matter_pbmap"])
    jobs = [(str(sweep_dir), record, wm_path, tau_core, tau_edema) for record in design]
    records: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_qoi_job, jobs, chunksize=4)
            for record in iterator:
                records.append(record)
                if len(records) % 500 == 0:
                    print(f"  {len(records)}/{len(jobs)} fields read", flush=True)
    else:
        for job in jobs:
            records.append(_qoi_job(job))
            if len(records) % 500 == 0:
                print(f"  {len(records)}/{len(jobs)} fields read", flush=True)
    return records


def qoi_summary(records: Sequence[Mapping[str, Any]], tau_core: float, tau_edema: float) -> dict[str, Any]:
    """Counts over the qoi records: runs, successes, empty compartments per
    threshold, NaN mass-weighted QoIs, NaN log10_mass."""
    successes = [r for r in records if r.get("success")]
    return {
        "n_runs": len(records),
        "n_success": len(successes),
        "n_failed": len(records) - len(successes),
        "tau_core": tau_core,
        "tau_edema": tau_edema,
        "n_empty_core": sum(1 for r in successes if r["n_core"] == 0),
        "n_empty_edema": sum(1 for r in successes if r["n_edema"] == 0),
        "n_nan_mass_weighted": sum(1 for r in successes if not np.isfinite(r["centroid_drift"])),
        "n_nan_anisotropy": sum(1 for r in successes if not np.isfinite(r["anisotropy"])),
        "n_nan_log10_mass": sum(1 for r in successes if not np.isfinite(r["log10_mass"])),
    }


# --- Sobol' analysis ---


def assemble_response(
    values: Mapping[int, float], n_blocks: int, size: int
) -> tuple[NDArray, list[int], list[int]]:
    """
    Assemble a response in SALib's row order from values keyed by design
    index, dropping every block with a missing or non-finite value.

    Args:
        values: QoI value by design index (missing indices count as NaN).
        n_blocks: N, the number of blocks of the design.
        size: Rows per block (``block_size``).

    Returns:
        (Y, kept, dropped): the response of the kept blocks, concatenated
        in order, and the kept and dropped block indices.
    """
    kept: list[int] = []
    dropped: list[int] = []
    chunks: list[NDArray] = []
    for block in range(n_blocks):
        y = np.array(
            [values.get(index, np.nan) for index in range(block * size, (block + 1) * size)],
            dtype=np.float64,
        )
        if np.all(np.isfinite(y)):
            kept.append(block)
            chunks.append(y)
        else:
            dropped.append(block)
    response = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return response, kept, dropped


def sobol_indices(
    response: NDArray,
    names: Sequence[str],
    second_order: bool = False,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, NDArray]:
    """
    The Sobol' indices of a response in SALib's row order.

    SALib.analyze.sobol.analyze: the Saltelli et al. 2010 first-order and
    the Jansen 1999 total-order estimators; S1_conf / ST_conf are
    normal-approximation 95 % half-widths from the bootstrap standard
    deviation over resampled blocks. SALib treats seed 0 as "unseeded",
    so seed + 1 is passed.

    Args:
        response: Y, length N' (k + 2) (or N' (2k + 2)).
        names: Factor order.
        second_order: Whether the design holds the B_A rows (S2 is then
            returned as well).
        n_bootstrap: Bootstrap resamples of the confidence intervals.
        seed: Bootstrap seed.

    Returns:
        S1, S1_conf, ST, ST_conf (and S2, S2_conf), float64 arrays.
    """
    from SALib.analyze import sobol as sobol_analyze

    result = sobol_analyze.analyze(
        salib_problem(names),
        np.asarray(response, dtype=np.float64),
        calc_second_order=second_order,
        num_resamples=int(n_bootstrap),
        conf_level=0.95,
        seed=int(seed) + 1,
        print_to_console=False,
    )
    keys = ["S1", "S1_conf", "ST", "ST_conf"] + (["S2", "S2_conf"] if second_order else [])
    return {key: np.asarray(result[key], dtype=np.float64) for key in keys}


def analyze_response(
    values: Mapping[int, float],
    names: Sequence[str],
    n_blocks: int,
    second_order: bool = False,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """
    Sobol' indices of one QoI with the block bookkeeping and the
    half-design convergence check.

    Returns:
        A record with the index arrays (S1, S1_conf, ST, ST_conf, S1_half,
        ST_half, S2/S2_conf with second_order), kept/dropped blocks and
        n_blocks_used; None if fewer than two blocks survive.
    """
    size = block_size(len(names), second_order)
    response, kept, dropped = assemble_response(values, n_blocks, size)
    if len(kept) < 2:
        return None
    indices: dict[str, Any] = sobol_indices(response, names, second_order, n_bootstrap, seed)
    n_half = len(kept) // 2
    if n_half >= 2:
        half = sobol_indices(response[: n_half * size], names, second_order, n_bootstrap, seed)
        indices["S1_half"], indices["ST_half"] = half["S1"], half["ST"]
    else:
        indices["S1_half"] = indices["ST_half"] = np.full(len(names), np.nan)
    indices.update(kept_blocks=kept, dropped_blocks=dropped, n_blocks_used=len(kept), n_blocks_half=n_half)
    return indices


def analyze_sweep(
    sweep_dir: str | Path, n_bootstrap: int = DEFAULT_N_BOOTSTRAP, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> dict[str, dict[str, Any]]:
    """
    Analyse every QoI of a sweep directory: sobol.csv, sobol_summary.json
    and the figures.

    Args:
        sweep_dir: The sweep directory with qoi.csv.
        n_bootstrap: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        The analysis records by QoI (``analyze_response``).
    """
    sweep_dir = Path(sweep_dir)
    spec = read_json(sweep_dir / "spec.json")
    names: list[str] = list(spec["factor_names"])
    qoi_records = read_csv(sweep_dir / "qoi.csv")
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {key: spec[key] for key in ("N", "k", "factor_names", "second_order", "salib_version")}
    summary.update(n_bootstrap=int(n_bootstrap), seed=int(seed), analysed_qois=list(ANALYSED_QOIS), qois={})
    for qoi in ANALYSED_QOIS:
        values = {int(r["index"]): as_float(r.get(qoi)) for r in qoi_records if r.get("success") == "True"}
        result = analyze_response(values, names, int(spec["N"]), bool(spec["second_order"]), n_bootstrap, seed)
        if result is None:
            print(f"  {qoi}: fewer than two complete blocks, skipped", flush=True)
            summary["qois"][qoi] = {"n_blocks_used": 0, "skipped": True}
            continue
        results[qoi] = result
        rows.extend(sobol_rows(qoi, names, result))
        summary["qois"][qoi] = qoi_summary_entry(names, result)
        ranking = ", ".join(summary["qois"][qoi]["ranking_ST"][:3])
        print(f"  {qoi}: {result['n_blocks_used']} blocks, sum S1 = {summary['qois'][qoi]['sum_S1']:.2f}, top ST: {ranking}", flush=True)
    write_csv(sweep_dir / "sobol.csv", rows, SOBOL_COLUMNS)
    if spec["second_order"]:
        write_csv(sweep_dir / "sobol_S2.csv", second_order_rows(names, results))
    write_json(sweep_dir / "sobol_summary.json", summary)
    make_figures(sweep_dir / "figures", results, names, spec, read_csv(sweep_dir / "design.csv"), qoi_records)
    return results


def sobol_rows(qoi: str, names: Sequence[str], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sobol.csv records of one QoI, one per factor."""
    return [
        {
            "qoi": qoi,
            "factor": name,
            "S1": float(result["S1"][i]),
            "S1_conf": float(result["S1_conf"][i]),
            "ST": float(result["ST"][i]),
            "ST_conf": float(result["ST_conf"][i]),
            "S1_half": float(result["S1_half"][i]),
            "ST_half": float(result["ST_half"][i]),
            "n_blocks_used": int(result["n_blocks_used"]),
        }
        for i, name in enumerate(names)
    ]


def second_order_rows(names: Sequence[str], results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The sobol_S2.csv records: one per (qoi, factor pair)."""
    rows = []
    for qoi, result in results.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rows.append(
                    {
                        "qoi": qoi,
                        "factor_i": names[i],
                        "factor_j": names[j],
                        "S2": float(result["S2"][i, j]),
                        "S2_conf": float(result["S2_conf"][i, j]),
                    }
                )
    return rows


def qoi_summary_entry(names: Sequence[str], result: Mapping[str, Any]) -> dict[str, Any]:
    """The sobol_summary.json entry of one QoI."""
    st, st_conf = result["ST"], result["ST_conf"]
    order = np.argsort(-st)
    finite_half = np.isfinite(result["ST_half"])
    return {
        "n_blocks_used": int(result["n_blocks_used"]),
        "n_blocks_dropped": len(result["dropped_blocks"]),
        "dropped_blocks": list(result["dropped_blocks"]),
        "n_blocks_half": int(result["n_blocks_half"]),
        "sum_S1": float(np.sum(result["S1"])),
        "S1": {name: float(result["S1"][i]) for i, name in enumerate(names)},
        "ST": {name: float(st[i]) for i, name in enumerate(names)},
        "ranking_ST": [names[i] for i in order],
        "ST_not_distinguishable_from_zero": [names[i] for i in range(len(names)) if st[i] - st_conf[i] <= 0],
        "max_abs_change_S1_half": float(np.max(np.abs(result["S1"] - result["S1_half"])[finite_half], initial=np.nan)) if finite_half.any() else None,
        "max_abs_change_ST_half": float(np.max(np.abs(st - result["ST_half"])[finite_half], initial=np.nan)) if finite_half.any() else None,
    }


# --- figures ---


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    """Save a figure as PNG and PDF and close it."""
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=150)
    figure.savefig(paths[1])
    import matplotlib.pyplot as plt

    plt.close(figure)
    return paths


def make_figures(
    figure_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    spec: Mapping[str, Any],
    design: Sequence[Mapping[str, str]],
    qoi_records: Sequence[Mapping[str, str]],
) -> list[Path]:
    """
    Write the figures: heatmaps of ST and S1 (factors x QoIs), a bar chart
    of S1 and ST with confidence whiskers per QoI, and a scatter grid of
    every QoI against every factor over the A and B rows.

    Returns:
        The files written.
    """
    import matplotlib

    matplotlib.use("Agg")
    figure_dir.mkdir(exist_ok=True)
    written: list[Path] = []
    if not results:
        return written
    for key in ("ST", "S1"):
        written += _save_figure(_heatmap(results, names, key), figure_dir / f"heatmap_{key}")
    for qoi, result in results.items():
        written += _save_figure(_bar_chart(qoi, names, result), figure_dir / f"bars_{qoi}")
    ab_indices = [int(r["index"]) for r in design if r["matrix"] in ("A", "B")]
    x_by_name = {name: {int(r["index"]): as_float(r[name]) for r in design} for name in names}
    factor_specs = spec["factors"]
    for qoi in results:
        y_by_index = {int(r["index"]): as_float(r.get(qoi)) for r in qoi_records if r.get("success") == "True"}
        figure = _scatter_grid(qoi, names, factor_specs, x_by_name, y_by_index, ab_indices)
        written += _save_figure(figure, figure_dir / f"scatter_{qoi}")
    return written


def _heatmap(results: Mapping[str, Mapping[str, Any]], names: Sequence[str], key: str) -> Any:
    """Factors x QoIs heatmap of one index (single-hue sequential map,
    values annotated)."""
    import matplotlib.pyplot as plt

    qois = list(results)
    matrix = np.array([[float(results[qoi][key][i]) for qoi in qois] for i in range(len(names))])
    figure, axis = plt.subplots(figsize=(1.0 + 0.75 * len(qois), 0.9 + 0.42 * len(names)))
    image = axis.imshow(np.clip(matrix, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(qois)), qois, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(qois)):
            value = matrix[i, j]
            axis.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.55 else COLOR_TEXT,
            )
    axis.set_title(f"Sobol' {key} index", fontsize=10)
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label=key)
    figure.tight_layout()
    return figure


def _bar_chart(qoi: str, names: Sequence[str], result: Mapping[str, Any]) -> Any:
    """Grouped bars of S1 and ST per factor with 95 % whiskers."""
    import matplotlib.pyplot as plt

    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(1.5 + 0.55 * len(names), 3.6))
    axis.bar(
        positions - width / 2, result["S1"], width, yerr=result["S1_conf"], color=COLOR_S1,
        label="S1 (first order)", error_kw={"elinewidth": 1, "capsize": 2, "ecolor": COLOR_TEXT},
    )
    axis.bar(
        positions + width / 2, result["ST"], width, yerr=result["ST_conf"], color=COLOR_ST,
        label="ST (total order)", error_kw={"elinewidth": 1, "capsize": 2, "ecolor": COLOR_TEXT},
    )
    axis.axhline(0, color=COLOR_POINTS, linewidth=0.8)
    axis.set_xticks(positions, names, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Sobol' index")
    axis.set_title(f"{qoi} ({result['n_blocks_used']} blocks, 95 % bootstrap whiskers)", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _binned_median(x: NDArray, y: NDArray, n_bins: int = 20) -> tuple[NDArray, NDArray]:
    """Median of y in equal-count bins along x (x-position: the bin's
    median x)."""
    order = np.argsort(x)
    bins = np.array_split(order, max(1, min(n_bins, len(order) // 10)))
    x_med = np.array([np.median(x[b]) for b in bins if len(b)])
    y_med = np.array([np.median(y[b]) for b in bins if len(b)])
    return x_med, y_med


def _scatter_grid(
    qoi: str,
    names: Sequence[str],
    factor_specs: Mapping[str, Mapping[str, Any]],
    x_by_name: Mapping[str, Mapping[int, float]],
    y_by_index: Mapping[int, float],
    indices: Sequence[int],
) -> Any:
    """One QoI against every factor over the given (A and B) rows,
    rasterised points with a binned-median line; log-x for log factors."""
    import matplotlib.pyplot as plt

    n_cols = min(4, len(names))
    n_rows = int(np.ceil(len(names) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)
    y = np.array([y_by_index.get(i, np.nan) for i in indices])
    for axis, name in zip(axes.flat, names):
        x = np.array([x_by_name[name].get(i, np.nan) for i in indices])
        valid = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[valid], y[valid], s=6, alpha=0.25, color=COLOR_POINTS, linewidths=0, rasterized=True)
        if valid.sum() >= 10:
            x_med, y_med = _binned_median(x[valid], y[valid])
            axis.plot(x_med, y_med, color=COLOR_ST, linewidth=2)
        if factor_specs[name]["scale"] == "log":
            axis.set_xscale("log")
        axis.set_xlabel(name, fontsize=8)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(qoi, fontsize=8)
    figure.suptitle(f"{qoi} over the A and B rows ({int(np.isfinite(y).sum())} runs); line: binned median", fontsize=10)
    figure.tight_layout()
    return figure


# --- command line ---


def _add_design_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE), help="search-space JSON")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="base config JSON")
    parser.add_argument("--output-dir", required=True, help="parent of the sweep directory")
    parser.add_argument("--name", required=True, help="sweep directory name")
    parser.add_argument("--log2-n", type=int, default=DEFAULT_LOG2_N, help="N = 2 ** log2_n base points")
    parser.add_argument("--seed", type=int, default=DEFAULT_DESIGN_SEED, help="seed of the Sobol' sequence")
    parser.add_argument("--second-order", action="store_true", help="sample the B_A rows too (N (2k + 2) runs)")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="comma-separated CUDA device ids; '' = CPU only")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="slots per GPU (CPU: parallel workers)")


def _add_qoi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tau-core", type=float, default=TAU_CORE, help="core density threshold")
    parser.add_argument("--tau-edema", type=float, default=TAU_EDEMA, help="edema density threshold")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="processes reading the fields")


def _add_analyze_args(parser: argparse.ArgumentParser, seed_flag: str) -> None:
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="bootstrap resamples")
    parser.add_argument(seed_flag, dest="bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="bootstrap seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_design_args(commands.add_parser("design", help="sample the design and write the run configs"))
    run = commands.add_parser("run", help="run the design points (resumable)")
    run.add_argument("--sweep-dir", required=True)
    _add_run_args(run)
    one = commands.add_parser("run-one", help="solve one run config into a run directory")
    one.add_argument("--config", required=True)
    one.add_argument("--run-dir", required=True)
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
    solver is imported; the previous value is restored for the runs)."""
    previous = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        sweep_dir = make_design(
            args.search_space, args.config, args.output_dir, args.name, args.log2_n, args.seed, args.second_order
        )
    finally:
        if previous is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = previous
    spec = read_json(sweep_dir / "spec.json")
    print(f"design directory: {sweep_dir}")
    print(
        f"N = {spec['N']} (2^{spec['log2_n']}), k = {spec['k']}, {spec['n_runs']} runs "
        f"(seed {spec['seed']}, SALib {spec['salib_version']}"
        f"{', second order' if spec['second_order'] else ''})"
    )
    print(f"factors: {', '.join(spec['factor_names'])}")
    print(
        f"seedable voxels: {spec['n_seedable_voxels']} in the box "
        f"{[round(v, 4) for v in spec['seed_bbox_lo']]} .. {[round(v, 4) for v in spec['seed_bbox_hi']]} "
        f"(fractions of the grid {spec['grid_shape']})"
    )
    return sweep_dir


def qoi_command(sweep_dir: Path, tau_core: float, tau_edema: float, workers: int) -> None:
    """The qoi subcommand: qoi.csv and qoi_summary.json."""
    records = qoi_table(sweep_dir, tau_core, tau_edema, workers)
    write_csv(sweep_dir / "qoi.csv", records, QOI_COLUMNS)
    summary = qoi_summary(records, tau_core, tau_edema)
    write_json(sweep_dir / "qoi_summary.json", summary)
    print(
        f"qoi.csv: {summary['n_success']}/{summary['n_runs']} successful runs; empty core "
        f"{summary['n_empty_core']}, empty edema {summary['n_empty_edema']}, NaN mass-weighted "
        f"{summary['n_nan_mass_weighted']}, NaN anisotropy {summary['n_nan_anisotropy']}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-one":
        try:
            return run_one(args.config, args.run_dir)
        except Exception:  # noqa: BLE001 - the log gets the traceback, the parent the exit code
            traceback.print_exc()
            return 1
    if args.command == "design":
        design_command(args)
        return 0
    if args.command == "run":
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        return 1 if run_sweep(args.sweep_dir, gpus, args.jobs_per_gpu)["failed"] else 0
    if args.command == "qoi":
        qoi_command(Path(args.sweep_dir), args.tau_core, args.tau_edema, args.workers)
        return 0
    if args.command == "analyze":
        analyze_sweep(args.sweep_dir, args.n_bootstrap, args.bootstrap_seed)
        return 0
    # all
    sweep_dir = design_command(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    counts = run_sweep(sweep_dir, gpus, args.jobs_per_gpu)
    qoi_command(sweep_dir, args.tau_core, args.tau_edema, args.workers)
    analyze_sweep(sweep_dir, args.n_bootstrap, args.bootstrap_seed)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
