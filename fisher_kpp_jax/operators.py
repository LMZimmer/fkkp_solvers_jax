"""Stateless numerical operators and the jitted time loop for the JAX
Fisher-KPP solvers.

Device operators use ``jax.numpy``; host operators (bounding box, embed) stay
in NumPy because they only run once per solve, outside the jitted time loop.
All stencils apply zero-flux (homogeneous Neumann) boundaries via edge
replication: the ghost cell outside the array equals the boundary cell, so
boundary faces carry zero net flux. Which arguments of the jitted time loop
are dynamic and which are static is documented at ``StepSpec``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


# Empirically chosen constants of the Gaussian seed profile.
GAUSSIAN_SEED_DIFFUSION_TIME: float = 5.0  # width of the analytic heat kernel
GAUSSIAN_SEED_MASS: float = 250.0  # total mass of the kernel
GAUSSIAN_SEED_FLOOR: float = 0.1  # values at or below this are zeroed


class StepSpec(TypedDict):
    """Step specification for the jitted time loop.

    func must be defined at module level so that every solve passes the 
    identical function object and the jit cache is reused.
    Its required signature::

        def step_func(
            state: dict[str, jax.Array],
            constants: dict[str, Any],
            dynamic_scalars: dict[str, jax.Array],
            *static_args: Any,
        ) -> dict[str, jax.Array]: ...

    dynamic_scalars holds 0-d device arrays of individual parameters
    (changing a value never recompiles); static_args holds static
    values (changing recompiles).
    """

    func: Callable[..., dict[str, jax.Array]]
    dynamic_scalars: dict[str, jax.Array]
    static_args: tuple[Any, ...]


class QuantitySpec(TypedDict):
    """Stopping-quantity specification, same slots as ``StepSpec``.

    func is called as func(state, constants, dynamic_scalars, *static_args)
    and returns the float64 stopping quantity.
    """

    func: Callable[..., jax.Array]
    dynamic_scalars: dict[str, jax.Array]
    static_args: tuple[Any, ...]


class GuardSpec(TypedDict):
    """Sanity check evaluated on the device after every step.

    The guard compares the stepped state against the previous one to
    detect a solve gone wrong (e.g. an explicit-Euler instability); if it
    fires, the time loop stops and the run is reported as a failure. The
    default guard (``_no_guard``) never fires. Same slots as ``StepSpec``.

    func is called as
    func(new_state, previous_state, constants, dynamic_scalars,
    *static_args) and returns (code, mass change, integrated density);
    code 0 = no guard fired, 1 = shrinkage, 2 = vanishing volume.
    """

    func: Callable[..., tuple[jax.Array, jax.Array, jax.Array]]
    dynamic_scalars: dict[str, jax.Array]
    static_args: tuple[Any, ...]


def shift_grid_by_one(field: jax.Array, shift: int, axis: int) -> jax.Array:
    """
    Shift a field by one cell along an axis with edge replication.

    The entries vacated by the shift are filled with the array edge value
    (ghost cell equals boundary cell), which realizes zero-flux boundaries
    in the stencils below.

    Args:
        field: Field to shift.
        shift: Shift direction; only unit shifts (+1 / -1) are supported,
            anything else raises ValueError.
        axis: Axis to shift along.

    Returns:
        The shifted field, same shape as field.
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
        raise ValueError("shift must be +1 or -1.")
    return jnp.pad(field, pad_width, mode="edge")[tuple(index)]


def face_average(field: jax.Array, axis: int) -> jax.Array:
    """
    Average a cell-centered field onto forward faces.

    For axis=0: out[i, j, k] = (field[i, j, k] + field[i+1, j, k]) / 2, the
    value on the face between cells i and i+1, with zero-flux edge
    replication at the boundary.

    Args:
        field: Cell-centered field.
        axis: Axis along which to build the faces.

    Returns:
        The face-averaged field, same shape as field.
    """
    return (shift_grid_by_one(field, -1, axis=axis) + field) / 2


def masked_face_average(
    field: jax.Array, valid_mask: jax.Array, axis: int
) -> jax.Array:
    """
    Average a cell-centered field onto forward faces, zeroing invalid faces.

    Like ``face_average``, but a face is zeroed unless both adjacent cells
    satisfy valid_mask -- this blocks flux across faces touching invalid
    cells (e.g. CSF, background, fully necrotic tissue).

    Args:
        field: Cell-centered field.
        valid_mask: Boolean mask of valid cells.
        axis: Axis along which to build the faces.

    Returns:
        The masked face-averaged field, same shape as field.
    """
    condition = jnp.logical_and(shift_grid_by_one(valid_mask, -1, axis=axis), valid_mask)
    return jnp.where(condition, (shift_grid_by_one(field, -1, axis=axis) + field) / 2, 0)


def diffusion_term(
    u: jax.Array,
    diffusivity: dict[str, jax.Array],
    spacing: tuple[float, float, float],
) -> jax.Array:
    """
    Compute the conservative finite-volume discretization of div(D grad u).

    Args:
        u: Cell-centered density field.
        diffusivity: Per-axis face diffusivities with keys 'fwd_x', 'fwd_y',
            'fwd_z', 'bwd_x', 'bwd_y', 'bwd_z', each the shape of u. 'fwd'
            along an axis is the forward face value (between cells i and
            i+1); 'bwd' is its edge-replicated shift by +1 (see
            ``shift_grid_by_one``), the backward face (between cells i-1
            and i).
        spacing: Grid spacing (dx, dy, dz) in mm.

    Returns:
        The diffusion term, same shape as u. Boundary faces carry zero flux:
        the edge-replicated ghost cell equals the boundary cell, so the flux
        through a boundary face is exactly zero.
    """
    dx, dy, dz = spacing
    d = diffusivity
    div_x = 1 / (dx * dx) * (
        d["bwd_x"] * (shift_grid_by_one(u, 1, axis=0) - u)
        - d["fwd_x"] * (u - shift_grid_by_one(u, -1, axis=0))
    )
    div_y = 1 / (dy * dy) * (
        d["bwd_y"] * (shift_grid_by_one(u, 1, axis=1) - u)
        - d["fwd_y"] * (u - shift_grid_by_one(u, -1, axis=1))
    )
    div_z = 1 / (dz * dz) * (
        d["bwd_z"] * (shift_grid_by_one(u, 1, axis=2) - u)
        - d["fwd_z"] * (u - shift_grid_by_one(u, -1, axis=2))
    )
    return div_x + div_y + div_z


def logistic_growth(u: jax.Array, rho: float) -> jax.Array:
    """Compute the logistic growth term rho * u * (1 - u)."""
    return rho * (u * (1 - u))


def logistic_sigmoid(x: jax.Array) -> jax.Array:
    """Compute the logistic sigmoid 1 / (1 + exp(-x))."""
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
    """
    Create the initial tumor cell density: an isotropic Gaussian profile
    centered at center_voxel.

    The clipping order is deliberate and results depend on it: values at or
    below floor are
    zeroed first (strictly-greater keeps the value), then the profile is
    capped at 1.

    Args:
        shape: Grid shape of the output field.
        center_voxel: Voxel index of the profile center.
        spacing: Grid spacing (dx, dy, dz) in mm.
        scale: Widens the seed by scaling the voxel coordinates.
        dtype: Dtype at which the whole profile is evaluated.
        diffusion_time: Width of the analytic heat kernel.
        mass: Total mass of the analytic heat kernel.
        floor: Values at or below this are zeroed.

    Returns:
        The clipped Gaussian profile of the given shape and dtype.
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
        -(x_scaled**2 + y_scaled**2 + z_scaled**2) / (4 * diffusion_time)
    )
    gauss = jnp.where(gauss > floor, gauss, 0)
    gauss = jnp.where(gauss > 1, jnp.asarray(1.0, dtype=dtype), gauss)
    return gauss


def tissue_bounding_box(
    mask: NDArray, margin: int = 2
) -> tuple[slice, slice, slice]:
    """
    Compute the axis-aligned bounding box of the True region of mask.

    Host-side (NumPy): runs once per solve and its output (static slices)
    determines the jitted shapes.

    Args:
        mask: Boolean 3D array.
        margin: Number of voxels the box is expanded by (clipped to the
            array bounds).

    Returns:
        A tuple of three slices selecting the bounding box.
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
    """
    Place a cropped field into a float64 zero array of the full shape.

    Args:
        field: Cropped field.
        box: Slices at which the field is placed.
        full_shape: Shape of the output array.

    Returns:
        A float64 array of shape full_shape with field placed at box.
    """
    full = np.zeros(full_shape)
    full[box] = field
    return full


def elongate_tensor_along_principal_axis(
    tensors: NDArray, factor: float
) -> NDArray:
    """
    Scale each (..., 3, 3) tensor along its principal eigenvector.

    Each tensor is eigendecomposed (batched ``jnp.linalg.eigh`` in float32),
    its largest eigenvalue is multiplied by factor, and half of the
    resulting increase is subtracted from each of the two remaining
    eigenvalues, so the eigenvalue sum -- the tensor trace -- is preserved
    up to float32 round-off. Eigenvectors are unchanged.

    Args:
        tensors: Array of tensors, shape (..., 3, 3).
        factor: Scaling factor applied to the largest eigenvalue.

    Returns:
        The elongated tensors as a float32 NumPy array of the same shape.
    """
    tensor_array = jnp.asarray(np.asarray(tensors), dtype=jnp.float32)

    eigenvalues, eigenvectors = jnp.linalg.eigh(tensor_array)
    max_eigenvalue_indices = jnp.argmax(eigenvalues, axis=-1, keepdims=True)
    max_eigenvalues = jnp.take_along_axis(
        eigenvalues, max_eigenvalue_indices, axis=-1
    )
    scaled_max_eigenvalues = max_eigenvalues * factor
    difference = scaled_max_eigenvalues - max_eigenvalues

    mask = jnp.put_along_axis(
        jnp.ones_like(eigenvalues, dtype=bool),
        max_eigenvalue_indices,
        False,
        axis=-1,
        inplace=False,
    )

    # Trace-preserving adjustment: the two non-max eigenvalues each absorb
    # half of the max eigenvalue's increase.
    final_eigenvalues = jnp.where(mask, eigenvalues - difference / 2, eigenvalues)
    final_eigenvalues = jnp.put_along_axis(
        final_eigenvalues,
        max_eigenvalue_indices,
        scaled_max_eigenvalues,
        axis=-1,
        inplace=False,
    )

    # Reconstruct the tensor from the adjusted eigendecomposition
    diagonal = final_eigenvalues[..., None, :] * jnp.eye(
        3, dtype=final_eigenvalues.dtype
    )
    elongated_tensors = (
        eigenvectors @ diagonal @ jnp.swapaxes(eigenvectors, -2, -1)
    )
    return np.asarray(elongated_tensors, dtype=np.float32)


# --- jitted time loop ---

# DTI guard thresholds (see solvers._dti_guard): referenced by the device
# guard and by base's error messages, so they can never disagree.
SHRINKAGE_LIMIT: float = 10.0
VANISHING_DENSITY_LIMIT: float = 1e-6

# Stop-kind codes carried through the scan.
_RUNNING: int = 0
_STOP_THRESHOLD: int = 1
_STOP_SHRINKAGE: int = 2
_STOP_VANISHING: int = 3

SCAN_TRACE_COUNT: int = 0
"""Number of times the time scan has been traced in this process
(diagnostic: identical consecutive solves must not increase it)."""


def _no_guard(
    new_state: dict[str, jax.Array],
    previous_state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Default guard: never fires (pruned by XLA)."""
    del new_state, previous_state, constants, dynamic_scalars
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    zero_f = jnp.asarray(0.0, dtype=jnp.float64)
    return zero_i, zero_f, zero_f


@partial(
    jax.jit,
    static_argnames=(
        "step_func",
        "step_static",
        "quantity_func",
        "quantity_static",
        "guard_func",
        "guard_static",
        "n_steps",
        "n_slots",
    ),
)
def _run_time_scan(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamics: dict[str, Any],
    slot_ids: jax.Array,
    *,
    step_func: Callable[..., dict[str, jax.Array]],
    step_static: tuple[Any, ...],
    quantity_func: Callable[..., jax.Array],
    quantity_static: tuple[Any, ...],
    guard_func: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    guard_static: tuple[Any, ...],
    n_steps: int,
    n_slots: int,
) -> dict[str, Any]:
    """
    Run the explicit-Euler time loop as one jitted ``lax.scan``.

    The loop body order is fixed -- results depend on it: step -> per-step
    quantities -> threshold-stop check (priority) -> guard check -> snapshot
    recording (skipped at a break step). After a stop, every remaining scan
    iteration is a masked no-op.

    This is a single module-level jitted function: the step, quantity and
    guard functions and their static argument tuples are stable across
    solves, while device arrays and physical scalars are dynamic arguments
    (see ``StepSpec``), so re-solves retrace
    only on a shape/dtype/step-count change (see SCAN_TRACE_COUNT).

    Args:
        state: Initial device state on the cropped grid.
        constants: Constant device arrays consumed by the functions.
        dynamics: Each function's 0-d device scalars under the keys "step",
            "quantity" and "guard", plus the float64 "stopping_threshold".
        slot_ids: Snapshot slot of each step (-1: no snapshot).
        step_func: Step function, see ``StepSpec``.
        step_static: Static arguments of the step function.
        quantity_func: Stopping-quantity function, see ``QuantitySpec``.
        quantity_static: Static arguments of the quantity function.
        guard_func: Post-step guard function, see ``GuardSpec``.
        guard_static: Static arguments of the guard function.
        n_steps: Number of scan iterations.
        n_slots: Number of snapshot slots.

    Returns:
        The final scan carry: final state, stop bookkeeping scalars,
        snapshot buffers and the recorded-frame count (device values).
    """
    global SCAN_TRACE_COUNT
    SCAN_TRACE_COUNT += 1  # incremented once per trace (diagnostic)

    xs = (jnp.arange(n_steps, dtype=jnp.int32), slot_ids)

    def body(carry: dict[str, Any], x: tuple[jax.Array, jax.Array]):
        t, slot = x
        prev = carry["state"]
        active = carry["active"]
        new_state = step_func(prev, constants, dynamics["step"], *step_static)
        quantity = quantity_func(
            new_state, constants, dynamics["quantity"], *quantity_static
        )
        threshold_hit = quantity >= dynamics["stopping_threshold"]
        guard_code, guard_mass_change, guard_density = guard_func(
            new_state, prev, constants, dynamics["guard"], *guard_static
        )

        stop_threshold = active & threshold_hit
        stop_guard = active & jnp.logical_not(threshold_hit) & (guard_code > 0)
        stopped_now = stop_threshold | stop_guard
        # Recording happens after both stop checks, so a break step is never
        # snapshotted.
        do_record = active & jnp.logical_not(stopped_now) & (slot >= 0)

        buffers = carry["buffers"]
        if n_slots > 0:
            slot_clipped = jnp.clip(slot, 0, n_slots - 1)
            buffers = {
                k: buf.at[slot_clipped].set(
                    jnp.where(do_record, new_state[k], buf[slot_clipped])
                )
                for k, buf in buffers.items()
            }

        stop_kind = jnp.where(
            stop_threshold,
            _STOP_THRESHOLD,
            jnp.where(
                stop_guard,
                jnp.where(guard_code == 1, _STOP_SHRINKAGE, _STOP_VANISHING),
                carry["stop_kind"],
            ),
        )
        next_carry = {
            "state": {k: jnp.where(active, new_state[k], prev[k]) for k in new_state},
            "active": active & jnp.logical_not(stopped_now),
            "stop_kind": stop_kind,
            "stop_step": jnp.where(stopped_now, t, carry["stop_step"]),
            # The loop's last computed stopping quantity: frozen at the value
            # of the stopping step once the loop is done.
            "quantity": jnp.where(active, quantity, carry["quantity"]),
            "guard_mass_change": jnp.where(
                stop_guard, guard_mass_change, carry["guard_mass_change"]
            ),
            "guard_density": jnp.where(
                stop_guard, guard_density, carry["guard_density"]
            ),
            "n_recorded": carry["n_recorded"] + do_record.astype(jnp.int32),
            "buffers": buffers,
        }
        return next_carry, None

    initial_carry = {
        "state": state,
        "active": jnp.asarray(True),
        "stop_kind": jnp.asarray(_RUNNING, dtype=jnp.int32),
        "stop_step": jnp.asarray(0, dtype=jnp.int32),
        "quantity": jnp.asarray(0.0, dtype=jnp.float64),
        "guard_mass_change": jnp.asarray(0.0, dtype=jnp.float64),
        "guard_density": jnp.asarray(0.0, dtype=jnp.float64),
        "n_recorded": jnp.asarray(0, dtype=jnp.int32),
        "buffers": {
            k: jnp.zeros((n_slots,) + v.shape, dtype=v.dtype)
            for k, v in state.items()
        },
    }
    final_carry, _ = jax.lax.scan(body, initial_carry, xs)
    return final_carry


def _run_time_loop(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    step_spec: StepSpec,
    quantity_spec: QuantitySpec,
    guard_spec: GuardSpec,
    n_steps: int,
    stopping_threshold: float,
    record_steps: NDArray,
) -> dict[str, Any]:
    """
    Prepare the host-side inputs and call the jitted time scan.

    Maps the (possibly duplicated) record_steps onto unique snapshot slots --
    each step is recorded at most once.

    Args:
        state: Initial device state on the cropped grid.
        constants: Constant device arrays consumed by the functions.
        step_spec: Step specification, see ``StepSpec``.
        quantity_spec: Stopping-quantity specification, see ``QuantitySpec``.
        guard_spec: Post-step guard specification, see ``GuardSpec``.
        n_steps: Number of time steps.
        stopping_threshold: Threshold on the stopping quantity.
        record_steps: Step indices at which snapshots are recorded.

    Returns:
        The final scan carry, see ``_run_time_scan``.
    """
    slot_steps = np.unique(np.asarray(record_steps, dtype=np.int64))
    n_slots = int(slot_steps.size)
    slot_ids = np.full(n_steps, -1, dtype=np.int32)
    if n_slots:
        slot_ids[slot_steps] = np.arange(n_slots, dtype=np.int32)
    dynamics = {
        "step": step_spec["dynamic_scalars"],
        "quantity": quantity_spec["dynamic_scalars"],
        "guard": guard_spec["dynamic_scalars"],
        # f64 to match the f64 stopping-quantity reduction it is compared to.
        "stopping_threshold": jnp.asarray(stopping_threshold, dtype=jnp.float64),
    }
    return _run_time_scan(
        state,
        constants,
        dynamics,
        jnp.asarray(slot_ids),
        step_func=step_spec["func"],
        step_static=step_spec["static_args"],
        quantity_func=quantity_spec["func"],
        quantity_static=quantity_spec["static_args"],
        guard_func=guard_spec["func"],
        guard_static=guard_spec["static_args"],
        n_steps=n_steps,
        n_slots=n_slots,
    )
