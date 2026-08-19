"""Device-operator tests: ports vs. their NumPy originals, plus properties.

Most operators are compared against ``fisher_kpp.operators`` at tight f64
tolerance; the jnp ports are run under a local ``jax.enable_x64()`` scope so
the comparison is meaningful, with expected differences at most a few ULP
from XLA's transcendental implementations and reduction order. The tensor
elongation operator is instead checked against its documented properties at
float32 tolerance.
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
    """Trace preserved, max eigenvalue scaled by factor, eigenvectors kept.

    The operator works in float32, so all assertions use float32-level
    tolerances (per-voxel errors ~2e-6 observed on the phantom).
    """
    factor = 1.5
    out = jax_ops.elongate_tensor_along_principal_axis(tensor_phantom, factor)
    assert out.dtype == np.float32

    tensors_f32 = tensor_phantom.astype(np.float32)
    np.testing.assert_allclose(
        np.trace(out, axis1=-2, axis2=-1),
        np.trace(tensors_f32, axis1=-2, axis2=-1),
        rtol=0,
        atol=1e-5,
    )

    e_in, v_in = np.linalg.eigh(tensors_f32)
    e_out = np.linalg.eigvalsh(out)
    np.testing.assert_allclose(
        e_out[..., -1], factor * e_in[..., -1], rtol=0, atol=1e-5
    )

    # In the input eigenbasis the output must be diagonal (eigenvectors
    # preserved), with the max eigenvalue scaled and half of its increase
    # subtracted from each of the other two.
    difference = (factor - 1) * e_in[..., -1:]
    e_expected = np.concatenate(
        [e_in[..., :-1] - difference / 2, factor * e_in[..., -1:]], axis=-1
    )
    rotated = np.swapaxes(v_in, -2, -1) @ out @ v_in
    expected = e_expected[..., None, :] * np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(rotated, expected, rtol=0, atol=1e-5)
