"""Stateless numerical operators for the Fisher-KPP solvers.

Boundary convention: all stencils apply zero-flux (homogeneous Neumann)
boundaries via edge replication — the ghost cell outside the array equals the
boundary cell, so boundary faces carry zero net flux. Interior values are
bitwise identical to the original ``np.roll``-based (periodic-wrap) stencils;
only the 1-voxel boundary shell differs.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

FaceFields = dict[str, NDArray]
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


def edge_roll(field: NDArray, shift: int, axis: int) -> NDArray:
    """``np.roll(field, shift, axis)`` with edge replication instead of
    periodic wrap: the entries vacated by the shift are filled with the array
    edge value (ghost cell equals boundary cell), which realizes zero-flux
    boundaries in the stencils below. Only shifts of +/-1 are supported.
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
    return np.pad(field, pad_width, mode="edge")[tuple(index)]


def face_average(field: NDArray, axis: int) -> NDArray:
    """Arithmetic average of a cell-centered field onto forward faces along axis.

    Example: for axis=0, out[i, j, k] = (field[i, j, k] + field[i+1, j, k]) / 2,
    i.e. the value on the face between cells i and i+1 (zero-flux edge
    replication at the boundary, see module docstring).
    """
    return (edge_roll(field, -1, axis=axis) + field) / 2


def masked_face_average(field: NDArray, valid_mask: NDArray, axis: int) -> NDArray:
    """Like face_average, but a face value is set to 0 unless both adjacent
    cells satisfy valid_mask. Used to block flux across faces that touch
    invalid cells (e.g. CSF, background, fully necrotic tissue).
    """
    condition = np.logical_and(edge_roll(valid_mask, -1, axis=axis), valid_mask)
    return np.where(condition, (edge_roll(field, -1, axis=axis) + field) / 2, 0)


def diffusion_term(
    u: NDArray,
    diffusivity: FaceFields,
    spacing: tuple[float, float, float],
) -> NDArray:
    """Conservative finite-volume discretization of div(D grad u) with
    per-axis face diffusivities and zero-flux boundaries: the edge-replicated
    ghost cell equals the boundary cell, so the boundary-face difference (and
    hence the flux through it) is exactly zero."""
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


def logistic_growth(u: NDArray, rho: float) -> NDArray:
    """rho * u * (1 - u)."""
    return rho * np.multiply(u, 1 - u)


def logistic_sigmoid(x: NDArray) -> NDArray:
    """1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


def clipped_gaussian(
    shape: tuple[int, int, int],
    center_voxel: tuple[int, int, int],
    spacing: tuple[float, float, float],
    scale: float = 1.0,
    diffusion_time: float = GAUSSIAN_SEED_DIFFUSION_TIME,
    mass: float = GAUSSIAN_SEED_MASS,
    floor: float = GAUSSIAN_SEED_FLOOR,
) -> NDArray:
    """Isotropic Gaussian profile centered at center_voxel, zeroed below a
    floor value and capped at 1. Used as the initial tumor cell density.

    Reproduces the original `gauss_sol3d` exactly, including its clipping
    order: floor first (strictly-greater keeps the value), then cap at 1.
    diffusion_time ("Dt") sets the kernel width, mass ("M") its total mass
    (together they set the amplitude), and floor the zeroing threshold; the
    defaults are the original's hardcoded constants.
    """
    xv, yv, zv = np.meshgrid(
        np.arange(0, shape[0]),
        np.arange(0, shape[1]),
        np.arange(0, shape[2]),
        indexing="ij",
    )
    x_scaled = (xv - center_voxel[0]) * spacing[0] / scale
    y_scaled = (yv - center_voxel[1]) * spacing[1] / scale
    z_scaled = (zv - center_voxel[2]) * spacing[2] / scale

    gauss = mass / np.power(
        4 * np.pi * diffusion_time, 3 / 2
    ) * np.exp(
        -(np.power(x_scaled, 2) + np.power(y_scaled, 2) + np.power(z_scaled, 2))
        / (4 * diffusion_time)
    )
    gauss = np.where(gauss > floor, gauss, 0)
    gauss = np.where(gauss > 1, np.float64(1), gauss)
    return gauss


def tissue_bounding_box(mask: NDArray, margin: int = 2) -> tuple[slice, slice, slice]:
    """Axis-aligned bounding box of the True region of mask, expanded by
    margin voxels and clipped to the array bounds."""
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


def embed(field: NDArray, box: tuple[slice, ...], full_shape: tuple[int, ...]) -> NDArray:
    """Inverse of crop: place field into a zero array of full_shape at box."""
    full = np.zeros(full_shape)
    full[box] = field
    return full


def elongate_tensor_along_principal_axis(
    tensors: NDArray | torch.Tensor, factor: float
) -> NDArray:
    """Scale each voxel's tensor along its principal eigenvector by factor.

    Port of `FK_DTI.tools.elongate_tensor_along_main_axis_torch` with the
    numerics reproduced exactly: same eigendecomposition path and the same
    float32 cast, so the returned array is float32. The other two eigenvalues
    are adjusted to keep the eigenvalue sum constant.
    """
    if isinstance(tensors, np.ndarray):
        tensor_array = torch.from_numpy(tensors)
    else:
        tensor_array = tensors
    tensor_array = tensor_array.float()

    e, v = torch.linalg.eigh(tensor_array)
    # Original sum of eigenvalues
    original_sum = torch.sum(e, dim=-1, keepdim=True)
    # Identify and scale the maximum eigenvalue
    max_eigenvalue_indices = torch.argmax(e, dim=-1, keepdim=True)
    max_eigenvalues = torch.gather(e, -1, max_eigenvalue_indices)
    scaled_max_eigenvalues = max_eigenvalues * factor

    # Calculate the difference introduced by scaling
    difference = scaled_max_eigenvalues - max_eigenvalues

    # Prepare to adjust the other eigenvalues to keep the sum constant
    adjustment = difference / 2
    mask = torch.ones_like(e, dtype=torch.bool)
    mask.scatter_(-1, max_eigenvalue_indices, 0)  # Mask out the max eigenvalue

    # Adjust the other two eigenvalues
    e_adjusted = torch.where(mask, e - adjustment, e)
    e_adjusted_sum = torch.sum(e_adjusted, dim=-1, keepdim=True)

    # Calculate final adjustments due to precision errors
    final_adjustment = (original_sum - e_adjusted_sum) / 3
    e_final = e_adjusted + torch.where(
        mask, final_adjustment, torch.zeros_like(final_adjustment)
    )

    # Ensure the scaled max eigenvalue is set correctly
    e_final.scatter_(-1, max_eigenvalue_indices, scaled_max_eigenvalues)

    # Reconstruct the tensor
    tensor_array_prime = v @ torch.diag_embed(e_final) @ v.transpose(-2, -1)
    return tensor_array_prime.detach().numpy()
