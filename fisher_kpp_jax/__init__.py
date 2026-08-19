"""Fisher-KPP tumor growth forward solvers, JAX port of ``fisher_kpp``.

Public API mirrors the NumPy reference: same class names, same params dicts
(plus two optional solver options: ``precision: "f32" | "f64"``, default
"f32", and ``n_steps``, an explicit step count pinning
dt = stopping_time / n_steps in place of the stability formula), same
``Result`` dataclass semantics. The explicit-Euler time loop runs as a
jitted ``jax.lax.scan`` on GPU when one is available, with automatic CPU
fallback.
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
