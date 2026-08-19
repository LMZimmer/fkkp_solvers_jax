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
"""Keys: 'minus_x', 'minus_y', 'minus_z', 'plus_x', 'plus_y', 'plus_z'.

Naming is inherited from the original solvers: the 'minus' field along an
axis holds the *forward* face value (between cells i and i+1) and 'plus' is
its shift by +1 (edge-replicated, see ``edge_roll``), i.e. the backward face
(between cells i-1 and i).
"""

# Magic constants of the original `gauss_sol3d` seed profile
# ("experimentally chosen" per the original comment).
GAUSSIAN_SEED_DIFFUSION_TIME: float = 5.0  # "Dt" of the analytic heat kernel
GAUSSIAN_SEED_MASS: float = 250.0  # "M", total mass of the kernel
GAUSSIAN_SEED_FLOOR: float = 0.1  # values at or below this are zeroed


def edge_roll(field: jax.Array, shift: int, axis: int) -> jax.Array:
    """Roll with edge replication instead of periodic wrap.

    ``jnp.roll(field, shift, axis)`` where the entries vacated by the shift
    are filled with the array edge value (ghost cell equals boundary cell),
    which realizes zero-flux boundaries in the stencils below.

    Args:
        field: Array to shift.
        shift: +1 or -1 (only unit shifts are supported).
        axis: Axis along which to shift.

    Returns:
        The shifted array, same shape and dtype as ``field``.

    Raises:
        ValueError: If ``shift`` is not +1 or -1.
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

    Example: for axis=0, out[i, j, k] = (field[i, j, k] + field[i+1, j, k]) / 2,
    i.e. the value on the face between cells i and i+1 (zero-flux edge
    replication at the boundary, see module docstring).

    Args:
        field: Cell-centered field.
        axis: Axis along which to average.

    Returns:
        Face-averaged field, same shape as ``field``.
    """
    return (edge_roll(field, -1, axis=axis) + field) / 2


def masked_face_average(
    field: jax.Array, valid_mask: jax.Array, axis: int
) -> jax.Array:
    """Like ``face_average``, but faces touching invalid cells are zeroed.

    A face value is set to 0 unless both adjacent cells satisfy
    ``valid_mask``. Used to block flux across faces that touch invalid cells
    (e.g. CSF, background, fully necrotic tissue).

    Args:
        field: Cell-centered field.
        valid_mask: Boolean validity mask, same shape as ``field``.
        axis: Axis along which to average.

    Returns:
        Masked face-averaged field, same shape as ``field``.
    """
    condition = jnp.logical_and(edge_roll(valid_mask, -1, axis=axis), valid_mask)
    return jnp.where(condition, (edge_roll(field, -1, axis=axis) + field) / 2, 0)


def diffusion_term(
    u: jax.Array,
    diffusivity: FaceFields,
    spacing: tuple[float, float, float],
) -> jax.Array:
    """Conservative finite-volume discretization of div(D grad u).

    Uses per-axis face diffusivities and zero-flux boundaries: the
    edge-replicated ghost cell equals the boundary cell, so the boundary-face
    difference (and hence the flux through it) is exactly zero.

    Args:
        u: Cell-centered density field.
        diffusivity: Face diffusivity fields (see ``FaceFields``).
        spacing: Grid spacing (dx, dy, dz) in mm.

    Returns:
        The discretized divergence term, same shape as ``u``.
    """
    dx, dy, dz = spacing
    d = diffusivity
    sp_x = 1 / (dx * dx) * (
        d["plus_x"] * (edge_roll(u, 1, axis=0) - u)
        - d["minus_x"] * (u - edge_roll(u, -1, axis=0))
    )
    sp_y = 1 / (dy * dy) * (
        d["plus_y"] * (edge_roll(u, 1, axis=1) - u)
        - d["minus_y"] * (u - edge_roll(u, -1, axis=1))
    )
    sp_z = 1 / (dz * dz) * (
        d["plus_z"] * (edge_roll(u, 1, axis=2) - u)
        - d["minus_z"] * (u - edge_roll(u, -1, axis=2))
    )
    return sp_x + sp_y + sp_z


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
    """Isotropic Gaussian profile centered at ``center_voxel``.

    Zeroed below a floor value and capped at 1; used as the initial tumor
    cell density. Reproduces the original ``gauss_sol3d`` exactly, including
    its clipping order: floor first (strictly-greater keeps the value), then
    cap at 1.

    Args:
        shape: Grid shape.
        center_voxel: Seed voxel indices.
        spacing: Grid spacing (dx, dy, dz) in mm.
        scale: Seed width scale factor.
        dtype: Device dtype of the returned field (the whole profile is
            evaluated at this precision).
        diffusion_time: "Dt" of the analytic heat kernel (kernel width).
        mass: "M", total mass of the kernel (with diffusion_time, sets the
            amplitude).
        floor: Values at or below this are zeroed.

    Returns:
        The clipped Gaussian seed profile on the device.
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


def crop(field: NDArray, box: tuple[slice, ...]) -> NDArray:
    """Crop field to box (a view; leading axes only for >3D fields)."""
    return field[box]


def embed(
    field: NDArray, box: tuple[slice, ...], full_shape: tuple[int, ...]
) -> NDArray:
    """Inverse of crop: place field into a zero array of full_shape at box."""
    full = np.zeros(full_shape)
    full[box] = field
    return full


def elongate_tensor_along_principal_axis(
    tensors: NDArray, factor: float
) -> NDArray:
    """Scale each voxel's tensor along its principal eigenvector by factor.

    Each tensor is eigendecomposed (batched ``jnp.linalg.eigh`` in float32),
    its largest eigenvalue is multiplied by ``factor``, and half of the
    resulting increase is subtracted from each of the two remaining
    eigenvalues, so the eigenvalue sum — the tensor trace — is preserved up
    to float32 round-off. The eigenvectors are unchanged.

    Args:
        tensors: Tensor field of shape (..., 3, 3).
        factor: Scaling factor for the principal eigenvalue.

    Returns:
        The elongated tensor field as a float32 NumPy array.
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
