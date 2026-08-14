"""Shared fixtures: path setup and small synthetic phantoms.

Runnable on CPU: ``JAX_PLATFORMS=cpu pytest tests`` must pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
# The JAX port lives at the repo root; the NumPy reference package lives in
# the hyphenated ./fisher-kpp project directory.
for entry in (str(_ROOT), str(_ROOT / "fisher-kpp")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

PHANTOM_N = 24


@pytest.fixture(scope="session")
def tissue_phantom() -> tuple[np.ndarray, np.ndarray]:
    """(gray_matter, white_matter): spherical WM core with a GM shell, 24^3."""
    n = PHANTOM_N
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
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
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    tensors[r >= 9] = 0.0
    return tensors
