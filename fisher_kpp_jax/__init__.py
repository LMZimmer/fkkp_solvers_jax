"""Fisher-KPP tumor growth forward solvers.

Three solvers share one params-dict interface and return a ``Result``.
Common solver options: ``precision: "f32" | "f64"`` (default "f32") selects
the device state dtype, and ``n_steps`` pins an explicit step count
(dt = stopping_time / n_steps) in place of the stability formula. The
explicit-Euler time loop runs as a jitted ``jax.lax.scan`` on GPU when one
is available, with automatic CPU fallback.
``scripts/run_reference_solves.py`` checks that the reference results are
matched.
"""

from __future__ import annotations

import os

import jax

from .base import Result
from .solvers import (
    AnisotropicFKPPSolver,
    FKPPSolver,
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
    "AnisotropicFKPPSolver",
    "FKPPSolver",
    "Result",
    "TwoCompartmentWithNutrientFKPPSolver",
]
