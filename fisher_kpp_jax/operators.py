"""Stateless numerical operators for the JAX Fisher-KPP solvers.

Device operators (``jax.numpy``) implement the numerics of
``fisher_kpp.operators``; host operators (bounding box, crop/embed) stay in
NumPy because they only run once per solve, outside the jitted time loop.

Boundary convention: all stencils apply zero-flux (homogeneous Neumann)
boundaries via edge replication — the ghost cell outside the array equals the
boundary cell, so boundary faces carry zero net flux. Interior values are
bitwise identical to the original ``np.roll``-based (periodic-wrap) stencils;
only the 1-voxel boundary shell differs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

FaceFields = dict[str, jax.Array]
"""Keys: 'fwd_x', 'fwd_y', 'fwd_z', 'bwd_x', 'bwd_y', 'bwd_z'.

'fwd' along an axis is the forward face value (between cells i and i+1);
'bwd' is its edge-replicated shift by +1 (see ``edge_shift``), the backward
face (between cells i-1 and i).
"""

# Magic constants of the original `gauss_sol3d` seed profile
# ("experimentally chosen" per the original comment).
GAUSSIAN_SEED_DIFFUSION_TIME: float = 5.0  # "Dt" of the analytic heat kernel
GAUSSIAN_SEED_MASS: float = 250.0  # "M", total mass of the kernel
GAUSSIAN_SEED_FLOOR: float = 0.1  # values at or below this are zeroed


def edge_shift(field: jax.Array, shift: int, axis: int) -> jax.Array:
    """Unit shift with edge replication.

    Shifts ``field`` by one cell along ``axis``; the entries vacated by the
    shift are filled with the array edge value (ghost cell equals boundary
    cell), which realizes zero-flux boundaries in the stencils below. Only
    unit shifts (+1 / -1) are supported; anything else raises ValueError.
    """
    pad_width = [(0, 0)] * field.ndim
    index: list[slice] = [slice(None)] * field.ndim
    n = field.shape[axis]
    if shift == 1:
        pad_width[axis] = (1, 0)
        index[axis] = slice(0, n)
    elif shift == -1:
        pad_width[axis] = (0, 1)
        index[axis] = slice(1, n + 1)
    else:
        raise ValueError("shift must be +1 or -1")
    return jnp.pad(field, pad_width, mode="edge")[tuple(index)]


def face_average(field: jax.Array, axis: int) -> jax.Array:
    """Arithmetic average of a cell-centered field onto forward faces.

    For axis=0: out[i, j, k] = (field[i, j, k] + field[i+1, j, k]) / 2, the
    value on the face between cells i and i+1 (zero-flux edge replication at
    the boundary, see module docstring).
    """
    return (edge_shift(field, -1, axis=axis) + field) / 2


def masked_face_average(
    field: jax.Array, valid_mask: jax.Array, axis: int
) -> jax.Array:
    """Like ``face_average``, but a face is zeroed unless both adjacent cells
    satisfy ``valid_mask`` — blocks flux across faces touching invalid cells
    (e.g. CSF, background, fully necrotic tissue).
    """
    condition = jnp.logical_and(edge_shift(valid_mask, -1, axis=axis), valid_mask)
    return jnp.where(condition, (edge_shift(field, -1, axis=axis) + field) / 2, 0)


def diffusion_term(
    u: jax.Array,
    diffusivity: FaceFields,
    spacing: tuple[float, float, float],
) -> jax.Array:
    """Conservative finite-volume discretization of div(D grad u).

    Uses per-axis face diffusivities (see ``FaceFields``) with zero-flux
    boundaries: the edge-replicated ghost cell equals the boundary cell, so
    the flux through a boundary face is exactly zero. ``spacing`` is
    (dx, dy, dz) in mm.
    """
    dx, dy, dz = spacing
    d = diffusivity
    div_x = 1 / (dx * dx) * (
        d["bwd_x"] * (edge_shift(u, 1, axis=0) - u)
        - d["fwd_x"] * (u - edge_shift(u, -1, axis=0))
    )
    div_y = 1 / (dy * dy) * (
        d["bwd_y"] * (edge_shift(u, 1, axis=1) - u)
        - d["fwd_y"] * (u - edge_shift(u, -1, axis=1))
    )
    div_z = 1 / (dz * dz) * (
        d["bwd_z"] * (edge_shift(u, 1, axis=2) - u)
        - d["fwd_z"] * (u - edge_shift(u, -1, axis=2))
    )
    return div_x + div_y + div_z


def logistic_growth(u: jax.Array, rho: float) -> jax.Array:
    """rho * u * (1 - u)."""
    return rho * jnp.multiply(u, 1 - u)


def logistic_sigmoid(x: jax.Array) -> jax.Array:
    """1 / (1 + exp(-x))."""
    return 1 / (1 + jnp.exp(-x))


def clipped_gaussian(
    shape: tuple[int, int, int],
    center_voxel: tuple[int, int, int],
    spacing: tuple[float, float, float],
    scale: float = 1.0,
    dtype: jnp.dtype = jnp.float64,
    diffusion_time: float = GAUSSIAN_SEED_DIFFUSION_TIME,
    mass: float = GAUSSIAN_SEED_MASS,
    floor: float = GAUSSIAN_SEED_FLOOR,
) -> jax.Array:
    """Isotropic Gaussian profile centered at ``center_voxel``; the initial
    tumor cell density.

    Reproduces the original ``gauss_sol3d`` exactly, including its clipping
    order: values at or below ``floor`` are zeroed first (strictly-greater
    keeps the value), then the profile is capped at 1. ``diffusion_time``
    ("Dt") and ``mass`` ("M") are the analytic heat kernel's width and total
    mass, ``scale`` widens the seed, and the whole profile is evaluated at
    ``dtype``.
    """
    xv, yv, zv = jnp.meshgrid(
        jnp.arange(0, shape[0], dtype=dtype),
        jnp.arange(0, shape[1], dtype=dtype),
        jnp.arange(0, shape[2], dtype=dtype),
        indexing="ij",
    )
    x_scaled = (xv - center_voxel[0]) * spacing[0] / scale
    y_scaled = (yv - center_voxel[1]) * spacing[1] / scale
    z_scaled = (zv - center_voxel[2]) * spacing[2] / scale

    # The scalar amplitude is computed in float64 on the host and cast, so a
    # NumPy scalar never promotes the device array dtype.
    amplitude = jnp.asarray(
        mass / np.power(4 * np.pi * diffusion_time, 3 / 2),
        dtype=dtype,
    )
    gauss = amplitude * jnp.exp(
        -(jnp.power(x_scaled, 2) + jnp.power(y_scaled, 2) + jnp.power(z_scaled, 2))
        / (4 * diffusion_time)
    )
    gauss = jnp.where(gauss > floor, gauss, 0)
    gauss = jnp.where(gauss > 1, jnp.asarray(1.0, dtype=dtype), gauss)
    return gauss


def tissue_bounding_box(
    mask: NDArray, margin: int = 2
) -> tuple[slice, slice, slice]:
    """Axis-aligned bounding box of the True region of ``mask``.

    Expanded by ``margin`` voxels and clipped to the array bounds. Host-side
    (NumPy): runs once per solve and its output (static slices) determines
    the jitted shapes.
    """
    indices = np.argwhere(mask)
    min_coords = np.maximum(indices.min(axis=0) - margin, 0)
    max_coords = np.minimum(indices.max(axis=0) + margin + 1, mask.shape)
    return (
        slice(int(min_coords[0]), int(max_coords[0])),
        slice(int(min_coords[1]), int(max_coords[1])),
        slice(int(min_coords[2]), int(max_coords[2])),
    )


def embed(
    field: NDArray, box: tuple[slice, ...], full_shape: tuple[int, ...]
) -> NDArray:
    """Place a cropped field into a float64 zero array of full_shape at box."""
    full = np.zeros(full_shape)
    full[box] = field
    return full


def elongate_tensor_along_principal_axis(
    tensors: NDArray, factor: float
) -> NDArray:
    """Scale each (..., 3, 3) tensor along its principal eigenvector by factor.

    Each tensor is eigendecomposed (batched ``jnp.linalg.eigh`` in float32),
    its largest eigenvalue is multiplied by ``factor``, and half of the
    resulting increase is subtracted from each of the two remaining
    eigenvalues, so the eigenvalue sum — the tensor trace — is preserved up
    to float32 round-off. Eigenvectors are unchanged; returns float32 NumPy.
    """
    tensor_array = jnp.asarray(np.asarray(tensors), dtype=jnp.float32)

    e, v = jnp.linalg.eigh(tensor_array)
    max_eigenvalue_indices = jnp.argmax(e, axis=-1, keepdims=True)
    max_eigenvalues = jnp.take_along_axis(e, max_eigenvalue_indices, axis=-1)
    scaled_max_eigenvalues = max_eigenvalues * factor
    difference = scaled_max_eigenvalues - max_eigenvalues

    mask = jnp.put_along_axis(
        jnp.ones_like(e, dtype=bool),
        max_eigenvalue_indices,
        False,
        axis=-1,
        inplace=False,
    )

    # Trace-preserving adjustment: the two non-max eigenvalues each absorb
    # half of the max eigenvalue's increase.
    e_final = jnp.where(mask, e - difference / 2, e)
    e_final = jnp.put_along_axis(
        e_final, max_eigenvalue_indices, scaled_max_eigenvalues, axis=-1, inplace=False
    )

    # Reconstruct the tensor
    diagonal = e_final[..., None, :] * jnp.eye(3, dtype=e_final.dtype)
    tensor_array_prime = v @ diagonal @ jnp.swapaxes(v, -2, -1)
    return np.asarray(tensor_array_prime, dtype=np.float32)
