"""Shared fixtures: path setup and small synthetic phantoms.

Tests cover ``fisher_kpp_jax`` only; comparison against the frozen NumPy
reference package lives in ``scripts/run_reference_solves.py``.

Runnable on CPU: ``JAX_PLATFORMS=cpu pytest tests`` must pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The fisher_kpp_jax package lives at the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PHANTOM_N = 24


def _radius_grid(n: int) -> np.ndarray:
    """Distance of each voxel from the grid center, shape (n, n, n)."""
    idx = np.indices((n, n, n))
    return np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))


@pytest.fixture(scope="session")
def tissue_phantom() -> tuple[np.ndarray, np.ndarray]:
    """(gray_matter, white_matter): spherical WM core with a GM shell, 24^3."""
    r = _radius_grid(PHANTOM_N)
    wm = (r < 6).astype(np.float64)
    gm = ((r >= 6) & (r < 9)).astype(np.float64)
    return gm, wm


@pytest.fixture(scope="session")
def tensor_phantom() -> np.ndarray:
    """Random SPD tensors inside a sphere, zero outside, 24^3."""
    n = PHANTOM_N
    rng = np.random.default_rng(42)
    b = rng.normal(size=(n, n, n, 3, 3))
    tensors = b @ b.transpose(0, 1, 2, 4, 3) * 0.05 + 0.3 * np.eye(3)
    tensors[_radius_grid(n) >= 9] = 0.0
    return tensors
