"""Each ported device operator vs. its NumPy original (tight tolerance, f64).

The jnp ports are run under a local ``jax.enable_x64()`` scope so the f64
comparison is meaningful; expected differences are at most a few ULP from
XLA's transcendental implementations and reduction order.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from fisher_kpp import operators as ref_ops
from fisher_kpp_jax import operators as jax_ops

RNG = np.random.default_rng(7)
SHAPE = (12, 13, 14)
SPACING = (1.3, 0.9, 1.1)


def _x64(fn, *args, **kwargs) -> np.ndarray:
    with jax.enable_x64():
        return np.asarray(fn(*args, **kwargs))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shift", [1, -1])
def test_edge_roll(axis: int, shift: int) -> None:
    field = RNG.random(SHAPE)
    np.testing.assert_array_equal(
        _x64(jax_ops.edge_roll, field, shift, axis),
        ref_ops.edge_roll(field, shift, axis),
    )


def test_edge_roll_rejects_large_shift() -> None:
    with pytest.raises(ValueError):
        with jax.enable_x64():
            jax_ops.edge_roll(np.zeros(SHAPE), 2, 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_face_average(axis: int) -> None:
    field = RNG.random(SHAPE)
    np.testing.assert_allclose(
        _x64(jax_ops.face_average, field, axis),
        ref_ops.face_average(field, axis),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_masked_face_average(axis: int) -> None:
    field = RNG.random(SHAPE)
    mask = RNG.random(SHAPE) > 0.3
    np.testing.assert_allclose(
        _x64(jax_ops.masked_face_average, field, mask, axis),
        ref_ops.masked_face_average(field, mask, axis),
        rtol=0,
        atol=0,
    )


def test_diffusion_term() -> None:
    u = RNG.random(SHAPE)
    minus = {name: RNG.random(SHAPE) for name in ("x", "y", "z")}
    faces = {}
    for axis, name in enumerate(("x", "y", "z")):
        faces[f"minus_{name}"] = minus[name]
        faces[f"plus_{name}"] = ref_ops.edge_roll(minus[name], 1, axis=axis)
    ours = _x64(jax_ops.diffusion_term, u, faces, SPACING)
    theirs = ref_ops.diffusion_term(u, faces, SPACING)
    np.testing.assert_allclose(ours, theirs, rtol=1e-14, atol=1e-14)


def test_logistic_growth() -> None:
    u = RNG.random(SHAPE)
    np.testing.assert_allclose(
        _x64(jax_ops.logistic_growth, u, 0.17),
        ref_ops.logistic_growth(u, 0.17),
        rtol=1e-15,
        atol=0,
    )


def test_logistic_sigmoid() -> None:
    x = RNG.normal(scale=10.0, size=SHAPE)
    np.testing.assert_allclose(
        _x64(jax_ops.logistic_sigmoid, x),
        ref_ops.logistic_sigmoid(x),
        rtol=1e-14,
        atol=0,
    )


def test_clipped_gaussian() -> None:
    shape = (20, 21, 22)
    center = (10, 9, 11)
    spacing = (1.4, 1.1, 0.8)
    with jax.enable_x64():
        ours = np.asarray(
            jax_ops.clipped_gaussian(
                shape, center, spacing, scale=1.3, dtype=np.float64
            )
        )
    theirs = ref_ops.clipped_gaussian(shape, center, spacing, scale=1.3)
    # exp() may differ by ~1 ULP between XLA and libm; the floor/cap clipping
    # thresholds are only crossed well away from these values here.
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, atol=1e-16)


def test_tissue_bounding_box_crop_embed() -> None:
    mask = np.zeros((16, 17, 18), dtype=bool)
    mask[4:9, 5:7, 10:15] = True
    ours = jax_ops.tissue_bounding_box(mask, margin=2)
    theirs = ref_ops.tissue_bounding_box(mask, margin=2)
    assert ours == theirs
    field = RNG.random(mask.shape)
    np.testing.assert_array_equal(
        jax_ops.crop(field, ours), ref_ops.crop(field, theirs)
    )
    np.testing.assert_array_equal(
        jax_ops.embed(jax_ops.crop(field, ours), ours, mask.shape),
        ref_ops.embed(ref_ops.crop(field, theirs), theirs, mask.shape),
    )


def test_elongate_tensor_along_principal_axis(tensor_phantom: np.ndarray) -> None:
    """jnp.linalg.eigh port vs. the torch original.

    Both implementations cast to float32 before the eigendecomposition, so
    LAPACK-backend differences bound the achievable agreement: the comparison
    tolerance is float32-level (rel L2 ~1e-6 observed; 1e-5 asserted), not
    the f64 tolerance used for the other operators.
    """
    ours = jax_ops.elongate_tensor_along_principal_axis(tensor_phantom, 1.5)
    theirs = ref_ops.elongate_tensor_along_principal_axis(tensor_phantom, 1.5)
    assert ours.dtype == np.float32
    assert theirs.dtype == np.float32
    rel_l2 = np.linalg.norm((ours - theirs).ravel()) / np.linalg.norm(
        theirs.ravel()
    )
    assert rel_l2 < 1e-5, rel_l2
    # NOTE: despite the original's comments, its adjustment sequence does NOT
    # keep the eigenvalue sum constant (net trace change is +2/3 of the
    # scaling difference); the port reproduces that quirk, so only the traces
    # of the two implementations are compared against each other.
    np.testing.assert_allclose(
        np.trace(ours, axis1=-2, axis2=-1),
        np.trace(theirs, axis1=-2, axis2=-1),
        rtol=1e-4,
        atol=1e-5,
    )
