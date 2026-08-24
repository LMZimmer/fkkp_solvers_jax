"""Stateless numerical operators and the jitted time loop for the JAX
Fisher-KPP solvers.

Device operators use ``jax.numpy``; host operators (bounding box, embed)
stay in NumPy because they only run once per solve, outside the jitted time
loop.

Boundary convention: all stencils apply zero-flux (homogeneous Neumann)
boundaries via edge replication — the ghost cell outside the array equals
the boundary cell, so boundary faces carry zero net flux.

The time loop is a single module-level jitted scan driver
(``_scan_driver``): statics are stable across solves, while device arrays
and physical scalars are dynamic arguments, so re-solves retrace only on a
shape/dtype/step-count change. The dynamic/static contract each solver must
follow is documented at ``StepSpec``; ``_run_time_loop`` is the host-side
entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

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

# Empirically chosen constants of the Gaussian seed profile.
GAUSSIAN_SEED_DIFFUSION_TIME: float = 5.0  # width of the analytic heat kernel
GAUSSIAN_SEED_MASS: float = 250.0  # total mass of the kernel
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
    return rho * (u * (1 - u))


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

    The clipping order is part of the contract: values at or below ``floor``
    are zeroed first (strictly-greater keeps the value), then the profile is
    capped at 1. ``diffusion_time`` and ``mass`` are the analytic heat
    kernel's width and total mass, ``scale`` widens the seed, and the whole
    profile is evaluated at ``dtype``.
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

State = dict[str, jax.Array]
Consts = dict[str, Any]
#: (impl, dynamic scalars, static args) triples. The impl is a module-level
#: function (stable identity, so the jit cache persists across solves). The
#: dynamic scalars are a dict of named 0-d device arrays (a jit pytree with
#: a stable treedef) — every PHYSICAL parameter a sweep or optimizer would
#: vary (dt, rates, thresholds) goes here, already cast on the host to its
#: use dtype, so changing its value never recompiles. The static args tuple
#: is hashable and holds only structural values (grid spacing, voxel volume
#: — geometry that cannot change without a shape change). The step impl is
#: called step_impl(state, consts, dyn, *static).
StepSpec = tuple[Callable[..., State], dict[str, jax.Array], tuple[Any, ...]]
#: quantity_impl(state, consts, dyn, *static) -> f64 stopping quantity.
QuantitySpec = tuple[Callable[..., jax.Array], dict[str, jax.Array], tuple[Any, ...]]
#: guard_impl(new_state, prev_state, consts, dyn, *static) ->
#: (code, shrinkage change, integrated density); code 0 = no guard fired,
#: 1 = shrinkage, 2 = vanishing volume.
GuardSpec = tuple[
    Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    dict[str, jax.Array],
    tuple[Any, ...],
]

#: Number of times the scan driver has been traced in this process
#: (diagnostic: identical consecutive solves must not increase it).
SCAN_TRACE_COUNT: int = 0


def _no_guard(
    new_state: State,
    previous_state: State,
    consts: Consts,
    dyn: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Default guard: never fires (pruned by XLA)."""
    del new_state, previous_state, consts, dyn
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    zero_f = jnp.asarray(0.0, dtype=jnp.float64)
    return zero_i, zero_f, zero_f


@partial(
    jax.jit,
    static_argnames=(
        "step_impl",
        "step_static",
        "quantity_impl",
        "quantity_static",
        "guard_impl",
        "guard_static",
        "n_steps",
        "n_slots",
    ),
)
def _scan_driver(
    state: State,
    consts: Consts,
    dynamics: dict[str, Any],
    slot_ids: jax.Array,
    *,
    step_impl: Callable[..., State],
    step_static: tuple[Any, ...],
    quantity_impl: Callable[..., jax.Array],
    quantity_static: tuple[Any, ...],
    guard_impl: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    guard_static: tuple[Any, ...],
    n_steps: int,
    n_slots: int,
) -> dict[str, Any]:
    """Jitted explicit-Euler loop as one ``lax.scan`` with a done flag.

    Loop body order (fixed — results depend on it): step -> per-step
    quantities -> threshold-stop check (priority) -> guard check -> snapshot
    recording (skipped at a break step). After a stop, every remaining scan
    iteration is a masked no-op.

    A single module-level jitted function: statics are stable across solves,
    while device arrays and physical scalars are dynamic arguments (the
    ``StepSpec`` contract), so re-solves retrace only on a shape/dtype/
    step-count change (see SCAN_TRACE_COUNT).

    Calling conventions for the impls and their dynamic/static splits are
    documented at ``StepSpec``/``QuantitySpec``/``GuardSpec``. ``dynamics``
    holds each impl's 0-d device scalars under "step"/"quantity"/"guard",
    plus the f64 "stopping_threshold"; ``slot_ids`` maps each step to its
    snapshot slot (-1: none). Returns the final scan carry: final state,
    stop bookkeeping scalars, snapshot buffers and the recorded-frame count
    (device values).
    """
    global SCAN_TRACE_COUNT
    SCAN_TRACE_COUNT += 1  # trace-time side effect only, by design

    xs = (jnp.arange(n_steps, dtype=jnp.int32), slot_ids)

    def body(carry: dict[str, Any], x: tuple[jax.Array, jax.Array]):
        t, slot = x
        prev = carry["state"]
        active = carry["active"]
        new_state = step_impl(prev, consts, dynamics["step"], *step_static)
        quantity = quantity_impl(
            new_state, consts, dynamics["quantity"], *quantity_static
        )
        threshold_hit = quantity >= dynamics["stopping_threshold"]
        guard_code, guard_change, guard_density = guard_impl(
            new_state, prev, consts, dynamics["guard"], *guard_static
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
            "guard_change": jnp.where(stop_guard, guard_change, carry["guard_change"]),
            "guard_density": jnp.where(
                stop_guard, guard_density, carry["guard_density"]
            ),
            "n_recorded": carry["n_recorded"] + do_record.astype(jnp.int32),
            "buffers": buffers,
        }
        return next_carry, None

    carry0 = {
        "state": state,
        "active": jnp.asarray(True),
        "stop_kind": jnp.asarray(_RUNNING, dtype=jnp.int32),
        "stop_step": jnp.asarray(0, dtype=jnp.int32),
        "quantity": jnp.asarray(0.0, dtype=jnp.float64),
        "guard_change": jnp.asarray(0.0, dtype=jnp.float64),
        "guard_density": jnp.asarray(0.0, dtype=jnp.float64),
        "n_recorded": jnp.asarray(0, dtype=jnp.int32),
        "buffers": {
            k: jnp.zeros((n_slots,) + v.shape, dtype=v.dtype)
            for k, v in state.items()
        },
    }
    final_carry, _ = jax.lax.scan(body, carry0, xs)
    return final_carry


def _run_time_loop(
    state: State,
    consts: Consts,
    step_spec: StepSpec,
    quantity_spec: QuantitySpec,
    guard_spec: GuardSpec,
    n_steps: int,
    stopping_threshold: float,
    record_steps: NDArray,
) -> dict[str, Any]:
    """Host wrapper around the jitted scan driver.

    Maps the (possibly duplicated) ``record_steps`` onto unique snapshot
    slots — each step is recorded at most once — and splits each spec into
    its dynamic device scalars and its structural static arguments.
    """
    slot_steps = np.unique(np.asarray(record_steps, dtype=np.int64))
    n_slots = int(slot_steps.size)
    slot_ids = np.full(n_steps, -1, dtype=np.int32)
    if n_slots:
        slot_ids[slot_steps] = np.arange(n_slots, dtype=np.int32)
    step_impl, step_dyn, step_static = step_spec
    quantity_impl, quantity_dyn, quantity_static = quantity_spec
    guard_impl, guard_dyn, guard_static = guard_spec
    dynamics = {
        "step": step_dyn,
        "quantity": quantity_dyn,
        "guard": guard_dyn,
        # f64 to match the f64 stopping-quantity reduction it is compared to.
        "stopping_threshold": jnp.asarray(stopping_threshold, dtype=jnp.float64),
    }
    return _scan_driver(
        state,
        consts,
        dynamics,
        jnp.asarray(slot_ids),
        step_impl=step_impl,
        step_static=step_static,
        quantity_impl=quantity_impl,
        quantity_static=quantity_static,
        guard_impl=guard_impl,
        guard_static=guard_static,
        n_steps=n_steps,
        n_slots=n_slots,
    )
