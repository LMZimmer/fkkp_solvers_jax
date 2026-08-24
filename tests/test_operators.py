"""Device-operator tests against independent NumPy specs.

Each operator is compared with a small NumPy implementation of its documented
semantics, written here (``_edge_shift``, ``_diffusion_term_numpy``, the
analytic Gaussian) — no dependency on the frozen reference package. The jnp
ports run under a local ``jax.enable_x64()`` scope so comparisons are at
float64; expected differences are at most a few ULP from XLA transcendentals
and scaling-order. The tensor elongation operator is checked against its
documented properties at float32 tolerance.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from fisher_kpp_jax import operators as jax_ops

SHAPE = (12, 13, 14)
SPACING = (1.3, 0.9, 1.1)


def _x64(fn, *args, **kwargs) -> np.ndarray:
    with jax.enable_x64():
        return np.asarray(fn(*args, **kwargs))


def _edge_shift(field: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """Independent spec of edge_shift: unit shift with edge replication."""
    v = np.moveaxis(field, axis, 0)
    if shift == 1:
        shifted = np.concatenate([v[:1], v[:-1]])
    else:
        shifted = np.concatenate([v[1:], v[-1:]])
    return np.moveaxis(shifted, 0, axis)


def _diffusion_term_numpy(u: np.ndarray, faces: dict, spacing: tuple) -> np.ndarray:
    """Independent spec of diffusion_term: divergence of forward-face fluxes
    with zero flux through the boundary faces."""
    out = np.zeros_like(u)
    for axis, (name, h) in enumerate(zip("xyz", spacing)):
        d = faces[f"fwd_{name}"]  # forward face, between cells i and i+1
        flux = d * (_edge_shift(u, -1, axis) - u)  # zero at the last face
        div = np.moveaxis(flux.copy(), axis, 0)
        div[1:] -= np.moveaxis(flux, axis, 0)[:-1]  # backward flux; zero at i=0
        out += np.moveaxis(div, 0, axis) / (h * h)
    return out


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shift", [1, -1])
def test_edge_shift(axis: int, shift: int) -> None:
    field = np.random.default_rng(7).random(SHAPE)
    np.testing.assert_array_equal(
        _x64(jax_ops.edge_shift, field, shift, axis),
        _edge_shift(field, shift, axis),
    )


def test_edge_shift_rejects_large_shift() -> None:
    with pytest.raises(ValueError):
        with jax.enable_x64():
            jax_ops.edge_shift(np.zeros(SHAPE), 2, 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_face_average(axis: int) -> None:
    field = np.random.default_rng(8).random(SHAPE)
    np.testing.assert_array_equal(
        _x64(jax_ops.face_average, field, axis),
        (field + _edge_shift(field, -1, axis)) / 2,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_masked_face_average(axis: int) -> None:
    rng = np.random.default_rng(9)
    field = rng.random(SHAPE)
    mask = rng.random(SHAPE) > 0.3
    both_valid = mask & _edge_shift(mask, -1, axis)
    expected = np.where(both_valid, (field + _edge_shift(field, -1, axis)) / 2, 0.0)
    np.testing.assert_array_equal(
        _x64(jax_ops.masked_face_average, field, mask, axis), expected
    )


def test_diffusion_term() -> None:
    rng = np.random.default_rng(10)
    u = rng.random(SHAPE)
    faces = {}
    for axis, name in enumerate("xyz"):
        fwd = rng.random(SHAPE)
        faces[f"fwd_{name}"] = fwd
        faces[f"bwd_{name}"] = _edge_shift(fwd, 1, axis)
    ours = _x64(jax_ops.diffusion_term, u, faces, SPACING)
    expected = _diffusion_term_numpy(u, faces, SPACING)
    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=1e-13)


def test_diffusion_term_conserves_mass() -> None:
    """Zero-flux boundaries: the divergence sums to zero over the grid."""
    rng = np.random.default_rng(11)
    u = rng.random(SHAPE)
    faces = {}
    for axis, name in enumerate("xyz"):
        fwd = rng.random(SHAPE)
        faces[f"fwd_{name}"] = fwd
        faces[f"bwd_{name}"] = _edge_shift(fwd, 1, axis)
    term = _x64(jax_ops.diffusion_term, u, faces, SPACING)
    assert abs(term.sum()) < 1e-11


def test_diffusion_term_zero_for_uniform_field() -> None:
    faces = {}
    rng = np.random.default_rng(12)
    for axis, name in enumerate("xyz"):
        fwd = rng.random(SHAPE)
        faces[f"fwd_{name}"] = fwd
        faces[f"bwd_{name}"] = _edge_shift(fwd, 1, axis)
    term = _x64(jax_ops.diffusion_term, np.full(SHAPE, 0.7), faces, SPACING)
    np.testing.assert_array_equal(term, np.zeros(SHAPE))


def test_logistic_growth() -> None:
    u = np.random.default_rng(13).random(SHAPE)
    np.testing.assert_allclose(
        _x64(jax_ops.logistic_growth, u, 0.17), 0.17 * u * (1 - u), rtol=1e-15
    )


def test_logistic_sigmoid() -> None:
    x = np.random.default_rng(14).normal(scale=10.0, size=SHAPE)
    np.testing.assert_allclose(
        _x64(jax_ops.logistic_sigmoid, x), 1 / (1 + np.exp(-x)), rtol=1e-14
    )


@pytest.mark.parametrize("mass", [jax_ops.GAUSSIAN_SEED_MASS, 2000.0])
def test_clipped_gaussian(mass: float) -> None:
    """Analytic heat-kernel profile, floored at 0.1 and capped at 1.

    mass=2000 raises the amplitude above 1 so the cap is exercised too.
    exp() may differ by ~1 ULP between XLA and libm; the floor/cap clipping
    thresholds are only crossed well away from these values here.
    """
    shape = (20, 21, 22)
    center = (10, 9, 11)
    spacing = (1.4, 1.1, 0.8)
    scale = 1.3
    with jax.enable_x64():
        ours = np.asarray(
            jax_ops.clipped_gaussian(
                shape, center, spacing, scale=scale, dtype=np.float64, mass=mass
            )
        )

    idx = np.indices(shape, dtype=np.float64)
    sq = sum(
        ((idx[a] - center[a]) * spacing[a] / scale) ** 2 for a in range(3)
    )
    diffusion_time = jax_ops.GAUSSIAN_SEED_DIFFUSION_TIME
    amplitude = mass / (4 * np.pi * diffusion_time) ** 1.5
    expected = amplitude * np.exp(-sq / (4 * diffusion_time))
    expected[expected <= jax_ops.GAUSSIAN_SEED_FLOOR] = 0.0
    expected = np.minimum(expected, 1.0)
    assert (ours == 1.0).any() == (mass == 2000.0)  # cap active only at high mass
    assert (ours == 0.0).any()  # floor active
    np.testing.assert_allclose(ours, expected, rtol=1e-13, atol=1e-16)


def test_tissue_bounding_box() -> None:
    mask = np.zeros((16, 17, 18), dtype=bool)
    mask[4:9, 5:7, 10:15] = True
    box = jax_ops.tissue_bounding_box(mask, margin=2)
    assert box == (slice(2, 11), slice(3, 9), slice(8, 17))


def test_tissue_bounding_box_clips_margin_to_bounds() -> None:
    mask = np.zeros((16, 17, 18), dtype=bool)
    mask[0:3, 15:17, 0:2] = True
    box = jax_ops.tissue_bounding_box(mask, margin=2)
    assert box == (slice(0, 5), slice(13, 17), slice(0, 4))


def test_embed_roundtrip() -> None:
    box = (slice(2, 11), slice(3, 9), slice(8, 17))
    shape = (16, 17, 18)
    field = np.random.default_rng(15).random(shape)
    cropped = field[box]
    assert cropped.shape == (9, 6, 9)
    restored = jax_ops.embed(cropped, box, shape)
    np.testing.assert_array_equal(restored[box], field[box])
    outside = np.ones(shape, dtype=bool)
    outside[box] = False
    assert (restored[outside] == 0).all()


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
