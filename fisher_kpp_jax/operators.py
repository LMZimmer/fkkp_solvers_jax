"""Stateless numerical operators and the jitted time loop for the JAX
Fisher-KPP solvers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


GAUSSIAN_SEED_DIFFUSION_TIME: float = 5.0  # width of the analytic heat kernel
GAUSSIAN_SEED_MASS: float = 250.0  # total mass of the kernel
GAUSSIAN_SEED_FLOOR: float = 0.1  # values at or below this are zeroed

# DTI guard thresholds (see solvers._dti_guard)
SHRINKAGE_LIMIT: float = 10.0
VANISHING_DENSITY_LIMIT: float = 1e-6

# Stop-kind codes
_RUNNING: int = 0
_STOP_THRESHOLD: int = 1
_STOP_SHRINKAGE: int = 2
_STOP_VANISHING: int = 3

SCAN_TRACE_COUNT: int = 0


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
    grid_spacing: tuple[jax.Array, jax.Array, jax.Array],
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
        grid_spacing: Grid spacing (dx, dy, dz) in mm, 0-d device scalars.

    Returns:
        The diffusion term, same shape as u. Boundary faces carry zero flux:
        the edge-replicated ghost cell equals the boundary cell, so the flux
        through a boundary face is exactly zero.
    """
    dx, dy, dz = grid_spacing
    d = diffusivity
    div_x = 1 / (dx * dx) * (
        d["fwd_x"] * (shift_grid_by_one(u, -1, axis=0) - u)
        - d["bwd_x"] * (u - shift_grid_by_one(u, 1, axis=0))
    )
    div_y = 1 / (dy * dy) * (
        d["fwd_y"] * (shift_grid_by_one(u, -1, axis=1) - u)
        - d["bwd_y"] * (u - shift_grid_by_one(u, 1, axis=1))
    )
    div_z = 1 / (dz * dz) * (
        d["fwd_z"] * (shift_grid_by_one(u, -1, axis=2) - u)
        - d["bwd_z"] * (u - shift_grid_by_one(u, 1, axis=2))
    )
    return div_x + div_y + div_z


def logistic_growth(u: jax.Array, rho: jax.Array | float) -> jax.Array:
    """Compute the logistic growth term rho * u * (1 - u)."""
    return rho * (u * (1 - u))


def logistic_sigmoid(x: jax.Array) -> jax.Array:
    """Compute the logistic sigmoid 1 / (1 + exp(-x))."""
    return 1 / (1 + jnp.exp(-x))


def clipped_gaussian(
    grid_shape: tuple[int, int, int],
    center_voxel: tuple[int, int, int],
    grid_spacing: tuple[float, float, float],
    scale: float = 1.0,
    dtype: jnp.dtype = jnp.float64,
    diffusion_time: float = GAUSSIAN_SEED_DIFFUSION_TIME,
    mass: float = GAUSSIAN_SEED_MASS,
    floor: float = GAUSSIAN_SEED_FLOOR,
) -> jax.Array:
    """
    Create the initial tumor cell density: an isotropic Gaussian profile
    centered at center_voxel. Values below floor are set to 0, values above
    1 are clipped to 1.

    Args:
        grid_shape: Grid shape of the output field.
        center_voxel: Voxel index of the profile center.
        grid_spacing: Grid spacing (dx, dy, dz) in mm.
        scale: Widens the seed by scaling the voxel coordinates.
        dtype: Dtype at which the whole profile is evaluated.
        diffusion_time: Width of the analytic heat kernel.
        mass: Total mass of the analytic heat kernel.
        floor: Values at or below this are zeroed.

    Returns:
        The clipped Gaussian profile of the given shape and dtype.
    """
    xv, yv, zv = jnp.meshgrid(
        jnp.arange(0, grid_shape[0], dtype=dtype),
        jnp.arange(0, grid_shape[1], dtype=dtype),
        jnp.arange(0, grid_shape[2], dtype=dtype),
        indexing="ij",
    )
    x_scaled = (xv - center_voxel[0]) * grid_spacing[0] / scale
    y_scaled = (yv - center_voxel[1]) * grid_spacing[1] / scale
    z_scaled = (zv - center_voxel[2]) * grid_spacing[2] / scale

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


def chemo_concentration(
    t: jax.Array,
    session_times: jax.Array,
    session_doses: jax.Array,
    decay_rate: jax.Array,
) -> jax.Array:
    """
    Evaluate the chemotherapy drug concentration at time t.

    Each session j deposits a concentration equal to its dose d_j that
    decays exponentially::

        C(t) = sum_j d_j [t >= t_j] exp(-decay_rate (t - t_j))

    Sessions in the future of t contribute zero. The solver integrates C
    over each step with ``chemo_exposure`` instead of sampling it; this
    point evaluation is kept for tests and plots.

    Args:
        t: Evaluation time, 0-d scalar.
        session_times: Chemotherapy session times t_j, 1-D array.
        session_doses: Session doses d_j (mg/m^2), 1-D array of the same
            length as session_times.
        decay_rate: Exponential decay rate of the concentration, 0-d
            scalar.

    Returns:
        The concentration C(t), 0-d, at the dtype of the inputs.
    """
    elapsed = t - session_times
    contributions = jnp.where(
        elapsed >= 0, session_doses * jnp.exp(-decay_rate * elapsed), 0
    )
    return jnp.sum(contributions)


def chemo_exposure(
    t0: jax.Array,
    t1: jax.Array,
    session_times: jax.Array,
    session_doses: jax.Array,
    decay_rate: jax.Array,
) -> jax.Array:
    """
    Integrate the chemotherapy drug concentration over the interval
    [t0, t1] in closed form.

    With C(t) of ``chemo_concentration`` and a_j = (t0 - t_j)^+,
    b_j = (t1 - t_j)^+ the exposure is::

        int_{t0}^{t1} C dt = (1 / decay_rate) sum_j d_j (exp(-decay_rate a_j)
                                                          - exp(-decay_rate b_j))

    which is exact for any interval length and for session times inside
    the interval (then a_j = 0). Sessions after t1 contribute exactly zero
    (a_j = b_j = 0). The difference is evaluated as
    exp(-decay_rate a_j) * (-expm1(-decay_rate (b_j - a_j))) so that short
    intervals keep their precision in f32.

    Args:
        t0: Interval start, 0-d scalar.
        t1: Interval end, 0-d scalar, >= t0.
        session_times: Chemotherapy session times t_j, 1-D array.
        session_doses: Session doses d_j (mg/m^2), 1-D array of the same
            length as session_times.
        decay_rate: Exponential decay rate of the concentration, 0-d
            scalar, > 0.

    Returns:
        The exposure, 0-d, at the dtype of the inputs.
    """
    a = jnp.maximum(t0 - session_times, 0)
    b = jnp.maximum(t1 - session_times, 0)
    per_session = jnp.exp(-decay_rate * a) * (-jnp.expm1(-decay_rate * (b - a)))
    return jnp.sum(session_doses * per_session) / decay_rate


def lq_log_kill(
    dose_per_fraction: jax.Array,
    alpha: jax.Array | float,
    beta: jax.Array | float,
) -> jax.Array:
    """
    Evaluate the linear-quadratic log cell kill of one radiotherapy
    fraction, E(x) = alpha * d(x) + beta * d(x)^2.

    The surviving fraction of a single fraction is exp(-E(x)); n
    fractions of the same dose survive with exp(-n * E(x)).

    Args:
        dose_per_fraction: Dose per fraction d(x) in Gy, any shape.
        alpha: Linear radiosensitivity in 1/Gy.
        beta: Quadratic radiosensitivity in 1/Gy^2.

    Returns:
        E(x), same shape as dose_per_fraction.
    """
    return alpha * dose_per_fraction + beta * dose_per_fraction**2


# --- jitted time loop ---


def _no_guard(
    new_state: dict[str, jax.Array],
    previous_state: dict[str, jax.Array],
    constants: Mapping[str, Any],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Default guard: never fires (pruned by XLA)."""
    del new_state, previous_state, constants
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    zero_f = jnp.asarray(0.0, dtype=jnp.float64)
    return zero_i, zero_f, zero_f


def _time_step(
    carry: dict[str, Any],
    step_input: tuple[jax.Array, jax.Array],
    *,
    step_func: Callable[..., dict[str, jax.Array]],
    quantity_func: Callable[..., jax.Array],
    guard_func: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    constants: Mapping[str, Any],
    n_slots: int,
) -> tuple[dict[str, Any], None]:
    """
    Perform one time step. Keyword arguments (after *) are bound via
    functools.partial such that lax.scan runs with
    _time_step(carry, step_input).

    Args:
        carry: The loop state, see ``_run_time_scan``'s Returns.
        step_input: Per-step scan inputs (step index, snapshot slot).

    Returns:
        (next carry, None): scan carries the loop state; nothing is
        emitted per step.
    """
    step_index, snapshot_slot = step_input
    previous_state = carry["state"]
    is_active = carry["active"]
    new_state = step_func(previous_state, constants, step_index)
    stopping_quantity = quantity_func(new_state, constants)
    threshold_hit = stopping_quantity >= constants["stopping_threshold"]
    guard_code, guard_mass_change, guard_density = guard_func(
        new_state, previous_state, constants
    )

    stop_threshold = is_active & threshold_hit
    stop_guard = is_active & jnp.logical_not(threshold_hit) & (guard_code > 0)
    stopped_now = stop_threshold | stop_guard
    
    do_record = is_active & jnp.logical_not(stopped_now) & (snapshot_slot >= 0)

    buffers = carry["buffers"]
    if n_slots > 0:
        # set buffers at the correct slot to new_state; non-recording steps
        # target the out-of-bounds slot n_slots, which mode="drop" discards.
        write_slot = jnp.where(do_record, snapshot_slot, n_slots)
        buffers = {
            k: buf.at[write_slot].set(new_state[k], mode="drop")
            for k, buf in buffers.items()
        }

    # First true condition wins; once the loop is inactive all conditions
    # are False and the default latches the recorded value.
    stop_kind = jnp.select(
        [stop_threshold, stop_guard & (guard_code == 1), stop_guard],
        [_STOP_THRESHOLD, _STOP_SHRINKAGE, _STOP_VANISHING],
        default=carry["stop_kind"],
    )
    next_carry = {
        "state": {
            k: jnp.where(is_active, new_state[k], previous_state[k])
            for k in new_state
        },
        "active": is_active & jnp.logical_not(stopped_now),
        "stop_kind": stop_kind,
        # Update or use old values decided by jnp.where
        "stop_step": jnp.where(stopped_now, step_index, carry["stop_step"]),
        "stopping_quantity": jnp.where(
            is_active, stopping_quantity, carry["stopping_quantity"]
        ),
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


@partial(
    jax.jit,
    static_argnames=(
        "step_func",
        "quantity_func",
        "guard_func",
        "n_steps",
        "n_slots",
    ),
)
def _run_time_scan(
    state: dict[str, jax.Array],
    constants: Mapping[str, Any],
    slot_ids: jax.Array,
    *,
    step_func: Callable[..., dict[str, jax.Array]],
    quantity_func: Callable[..., jax.Array],
    guard_func: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    n_steps: int,
    n_slots: int,
) -> dict[str, Any]:
    """
    Run the explicit-Euler time loop as one jitted ``lax.scan``.
    Jitting is done via the decorator.

    The loop length is fixed. After a stop, every remaining scan
    iteration is a masked no-op.

    All keyword-only arguments (after *) are jit-static, passing a new value
    triggers recompile. The positional arguments (state, constants,
    slot_ids) are traced device values, so solves that differ only in them
    reuse the compiled scan.

    Args:
        state: Initial device state on the cropped grid.
        constants: Device arrays and 0-d scalars consumed by the functions
            and the stop logic, see ``_run_time_loop``.
        slot_ids: Snapshot slot of each step (-1: no snapshot).
        step_func: Step function, see ``_run_time_loop``.
        quantity_func: Stopping-quantity function, see ``_run_time_loop``.
        guard_func: Post-step guard function, see ``_run_time_loop``.
        n_steps: Number of scan iterations.
        n_slots: Number of snapshot slots.

    Returns:
        The final scan carry: a dict of device values with keys
        'state' (the fields after the last active step),
        'active' (False if a stop fired),
        'stop_kind' (_RUNNING, _STOP_THRESHOLD, _STOP_SHRINKAGE or
        _STOP_VANISHING),
        'stop_step' (step index at which the loop stopped, 0 if it never
        did),
        'stopping_quantity' (stopping quantity of the last active step,
        float64),
        'guard_mass_change' and 'guard_density' (guard diagnostics of the
        stopping step, 0.0 unless a guard fired) and
        'buffers' / 'n_recorded' (per-field snapshot arrays of shape
        (n_slots, *field_shape) and the number of frames written).
    """
    global SCAN_TRACE_COUNT  # diagnostic, incremented once per compile
    SCAN_TRACE_COUNT += 1

    # first axis dimension sets the number of iterations in jax.lax.scan
    step_inputs = (jnp.arange(n_steps, dtype=jnp.int32), slot_ids)

    time_step = partial(
        _time_step,
        step_func=step_func,
        quantity_func=quantity_func,
        guard_func=guard_func,
        constants=constants,
        n_slots=n_slots,
    )

    initial_carry = {
        "state": state,
        "active": jnp.asarray(True),
        "stop_kind": jnp.asarray(_RUNNING, dtype=jnp.int32),
        "stop_step": jnp.asarray(0, dtype=jnp.int32),
        "stopping_quantity": jnp.asarray(0.0, dtype=jnp.float64),
        "guard_mass_change": jnp.asarray(0.0, dtype=jnp.float64),
        "guard_density": jnp.asarray(0.0, dtype=jnp.float64),
        "n_recorded": jnp.asarray(0, dtype=jnp.int32),
        "buffers": {
            k: jnp.zeros((n_slots,) + v.shape, dtype=v.dtype)
            for k, v in state.items()
        },
    }
    final_carry, _ = jax.lax.scan(time_step, initial_carry, step_inputs)
    return final_carry


def _run_time_loop(
    state: dict[str, jax.Array],
    constants: Mapping[str, Any],
    step_func: Callable[..., dict[str, jax.Array]],
    quantity_func: Callable[..., jax.Array],
    guard_func: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    n_steps: int,
    record_steps: NDArray,
) -> dict[str, Any]:
    """
    Prepare the host-side inputs and call the time scan (jitted loop).

    The three functions must be defined at module level so that every solve
    passes the identical function object and the jit cache is reused. Their
    required signatures::

        def step_func(
            state: dict[str, jax.Array],
            constants: Mapping[str, Any],
            step_index: jax.Array,
        ) -> dict[str, jax.Array]: ...  # step_index: 0-d int32 scan index

        def quantity_func(
            state: dict[str, jax.Array], constants: Mapping[str, Any]
        ) -> jax.Array: ...  # the float64 stopping quantity

        def guard_func(
            new_state: dict[str, jax.Array],
            previous_state: dict[str, jax.Array],
            constants: Mapping[str, Any],
        ) -> tuple[jax.Array, jax.Array, jax.Array]: ...

    The guard is a post-step sanity check: it compares the stepped state
    against the previous one to detect a solve gone wrong (e.g. an
    explicit-Euler instability) and returns (code, mass change, integrated
    density); code 0 = no guard fired, 1 = shrinkage, 2 = vanishing volume.
    If it fires, the time loop stops and the run is reported as a failure.
    The default guard (``_no_guard``) never fires.

    Args:
        state: Initial device state on the cropped grid.
        constants: Device arrays and 0-d device scalars consumed by the
            functions and the stop logic: the solver's field arrays and
            physical parameters, plus the shared keys built by the base
            solver ('dt', 'grid_spacing', 'voxel_volume',
            'stopping_threshold' and, in volume mode, 'volume_threshold').
        step_func: Step function, see above.
        quantity_func: Stopping-quantity function, see above.
        guard_func: Post-step guard function, see above.
        n_steps: Number of time steps.
        record_steps: Step indices in [0, n_steps) after which a snapshot
            of the state is recorded, in ascending order; empty records
            nothing. The frame of step s holds the state after that step.

    Returns:
        The final scan carry, see ``_run_time_scan``.
    """
    slot_steps = np.unique(np.asarray(record_steps, dtype=np.int64))
    n_slots = int(slot_steps.size)
    slot_ids = np.full(n_steps, -1, dtype=np.int32)
    if n_slots:
        slot_ids[slot_steps] = np.arange(n_slots, dtype=np.int32)
    return _run_time_scan(
        state,
        constants,
        jnp.asarray(slot_ids),
        step_func=step_func,
        quantity_func=quantity_func,
        guard_func=guard_func,
        n_steps=n_steps,
        n_slots=n_slots,
    )
