"""Jitted explicit-Euler time loop shared by all solvers.

A single module-level jitted scan driver (``_scan_driver``) runs the whole
loop: statics are stable across solves, while device arrays and physical
scalars are dynamic arguments, so re-solves retrace only on a shape/dtype/
step-count change. The dynamic/static contract each solver must follow is
documented at ``StepSpec``; ``_run_time_loop`` is the host-side entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

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
