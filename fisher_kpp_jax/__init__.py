"""Fisher-KPP tumor growth forward solvers.

Four solvers share one parameter interface and return a ``Result``
(``StuppFKPPSolver`` extends the isotropic model with resection,
chemotherapy and radiotherapy). A solver is built from its parameters as
one mapping or as keyword arguments, ``Solver(params)`` or
``Solver(**config)``; volumes (tissue maps, tensors, treatment maps) may be
arrays or NIfTI paths, in which case the voxel size and affine come from
the header. Every solver holds its ``config`` (the parameters as given,
defaults filled in, volumes as paths), ``read_config`` / ``write_config``
move configs to and from JSON and ``Solver.get_default_config()`` returns
the pre-written default config of a class. ``solve(store_result=True,
outdir=...)`` (or ``solver.save(outdir)``, ``Result.save``) writes the
config, a result record and the result volumes into a directory, and
``SolverClass(read_config(path))`` reproduces the saved run: the caller
names the class, and a ``"solver"`` entry in the config is checked against
it, never used to pick one.

Common solver options: ``precision: "f32" | "f64"`` (default "f32")
selects the device state dtype; the time step is given as at most one of
``n_steps``, ``dt`` (days) or ``steps_per_day``, raised to the solver's
stability estimate with a warning when coarser (none given: the
estimate); ``snapshot_times`` (a list of days) records the state at the
nearest steps into ``Result.time_series`` with the recorded days in
``Result.snapshot_times``. The explicit-Euler time loop runs as a jitted
``jax.lax.scan`` on GPU when one is available, with automatic CPU
fallback. ``scripts/run_reference_solves.py`` checks that the reference
results are matched.
"""

from __future__ import annotations

import os

import jax

from .base import Result
from .config import (
    SOLVER_KEY,
    VOLUME_IN_MEMORY,
    read_config,
    solver_class,
    write_config,
)
from .solvers import (
    AnisotropicFKPPSolver,
    FKPPSolver,
    StuppFKPPSolver,
    TwoCompartmentWithNutrientFKPPSolver,
)

# Persistent compilation cache: shapes are static per crop box, so identical
# repeat solves (and re-runs of the same script) skip XLA recompilation.
# Best-effort only — never fatal, and deliberately NOT touching x64 config.
try:  # pragma: no cover - cache availability depends on the platform
    _cache_dir = os.environ.get(
        "FISHER_KPP_JAX_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "fisher_kpp_jax"),
    )
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
except Exception:  # noqa: BLE001
    pass

__all__ = [
    "SOLVER_KEY",
    "VOLUME_IN_MEMORY",
    "AnisotropicFKPPSolver",
    "FKPPSolver",
    "Result",
    "StuppFKPPSolver",
    "TwoCompartmentWithNutrientFKPPSolver",
    "read_config",
    "solver_class",
    "write_config",
]
