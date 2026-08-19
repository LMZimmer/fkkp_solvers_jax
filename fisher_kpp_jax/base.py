"""Shared solve() pipeline for the JAX Fisher-KPP forward solvers.

The pipeline mirrors ``fisher_kpp.base`` step by step: unpack/validate params
-> downsample tissue (host, ``scipy.ndimage.zoom``) -> grid geometry -> ad-hoc
stability-derived dt (or the explicit ``n_steps`` override) -> Gaussian seed
(device) -> bounding-box crop -> face diffusivities (device) -> jitted
``jax.lax.scan`` explicit-Euler loop with stopping checks and optional
time-series recording -> uncrop -> upsample (host) -> Result.

Host/device split: parameter validation, ``zoom`` down/upsampling, crop-box
computation and the final embed/upsample stay on the host in NumPy (``zoom``
is deliberately not ported — ``jax.image.resize`` is not numerically
equivalent). State initialization, face-diffusivity construction and the time
loop run on the device; the loop is compiled once per cropped shape (see
``_scan_driver``), and x64 is enabled locally around the device portion —
never globally at import.

Stopping semantics are documented on ``Result`` and the ``_quantity_spec``
hook; the dynamic/static argument contract that keeps re-solves from
recompiling is documented at ``StepSpec``.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from .operators import clipped_gaussian, embed, tissue_bounding_box

logger = logging.getLogger(__name__)

CROP_MARGIN: int = 2  # voxels of margin left around the crop mask

# DTI guard thresholds (see solvers._dti_guard): referenced by the device
# guard and by the error messages below, so they can never disagree.
SHRINKAGE_LIMIT: float = 10.0
VANISHING_DENSITY_LIMIT: float = 1e-6

# Stop-kind codes carried through the scan.
_RUNNING: int = 0
_STOP_VOLUME: int = 1
_STOP_SHRINKAGE: int = 2
_STOP_VANISHING: int = 3

State = dict[str, jax.Array]
Consts = dict[str, Any]
#: (impl, dynamic scalars, static args) triples. The impl is a module-level
#: function (stable identity, so the jit cache persists across solves). The
#: dynamic scalars are 0-d device arrays — every PHYSICAL parameter a sweep
#: or optimizer would vary (dt, rates, thresholds) goes here, already cast
#: on the host to its use dtype, so changing its value never recompiles. The
#: static args tuple is hashable and holds only structural values (grid
#: spacing, voxel volume — geometry that cannot change without a shape
#: change). The step impl is called step_impl(state, consts, dyn, *static).
StepSpec = tuple[Callable[..., State], tuple[jax.Array, ...], tuple[Any, ...]]
#: quantity_impl(state, consts, dyn, *static) -> f64 stopping quantity.
QuantitySpec = tuple[Callable[..., jax.Array], tuple[jax.Array, ...], tuple[Any, ...]]
#: guard_impl(new_state, prev_state, consts, dyn, *static) ->
#: (code, shrinkage change, integrated density); code 0 = no guard fired,
#: 1 = shrinkage, 2 = vanishing volume.
GuardSpec = tuple[
    Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    tuple[jax.Array, ...],
    tuple[Any, ...],
]

#: Number of times the scan driver has been traced in this process
#: (diagnostic: identical consecutive solves must not increase it).
SCAN_TRACE_COUNT: int = 0


def _no_guard(
    new_state: State,
    previous_state: State,
    consts: Consts,
    dyn: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Default guard: never fires (pruned by XLA)."""
    del new_state, previous_state, consts, dyn
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    zero_f = jnp.asarray(0.0, dtype=jnp.float64)
    return zero_i, zero_f, zero_f


@dataclass(slots=True)
class Result:
    """Solver outcome.

    ``final_stopping_quantity`` is the last value of the quantity compared
    against ``stopping_threshold``: total cell mass for ``stopping_mode="mass"``
    (an integrated density, not a physical volume), thresholded volume for
    ``stopping_mode="volume"``.
    """

    success: bool
    initial_state: dict[str, NDArray]
    final_state: dict[str, NDArray]
    final_time: float
    final_stopping_quantity: float
    stopping_criterion: Literal["time", "volume", "error"]
    time_series: dict[str, NDArray] | None = None
    error: str | None = None


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

    Reproduces the original host loop body order exactly: step -> per-step
    quantities -> volume-stop check (priority) -> guard check -> snapshot
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
        volume_hit = quantity >= dynamics["stopping_threshold"]
        guard_code, guard_change, guard_density = guard_impl(
            new_state, prev, consts, dynamics["guard"], *guard_static
        )

        stop_volume = active & volume_hit
        stop_guard = active & jnp.logical_not(volume_hit) & (guard_code > 0)
        stopped_now = stop_volume | stop_guard
        # The original records after both checks, so a break step is never
        # snapshotted.
        do_record = active & jnp.logical_not(stopped_now) & (slot >= 0)

        buffers = carry["buffers"]
        if n_slots > 0:
            slot_clipped = jnp.clip(slot, 0, n_slots - 1)

            def write(buffer: jax.Array, field: jax.Array) -> jax.Array:
                current = jax.lax.dynamic_index_in_dim(
                    buffer, slot_clipped, 0, keepdims=False
                )
                value = jnp.where(do_record, field, current)
                return jax.lax.dynamic_update_index_in_dim(
                    buffer, value, slot_clipped, 0
                )

            buffers = {k: write(buffers[k], new_state[k]) for k in buffers}

        stop_kind = jnp.where(
            stop_volume,
            _STOP_VOLUME,
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


_scan_driver = jax.jit(
    _scan_driver,
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
    slots — each step is recorded at most once, matching the original
    ``t in record_steps`` semantics — and splits each spec into its dynamic
    device scalars and its structural static arguments.
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


DEFAULT_DENSITY_THRESHOLD: float = 0.5


def _merge_params(
    params: Mapping[str, Any],
    required: frozenset[str],
    defaults: Mapping[str, Any],
    solver_name: str,
) -> dict[str, Any]:
    """Strict parameter merge: unknown keys and missing required keys raise.

    ``stopping_volume`` is accepted as a deprecated alias of
    ``stopping_threshold`` (the quantity it thresholds is only a physical
    volume in "volume" mode); supplying both raises.
    """
    params = dict(params)
    if "stopping_volume" in params:
        if "stopping_threshold" in params:
            raise ValueError(
                f"{solver_name}: pass only one of 'stopping_threshold' and its "
                "deprecated alias 'stopping_volume'"
            )
        warnings.warn(
            f"{solver_name}: 'stopping_volume' is deprecated, use "
            "'stopping_threshold' (identical semantics)",
            DeprecationWarning,
            stacklevel=3,
        )
        params["stopping_threshold"] = params.pop("stopping_volume")
    unknown = sorted(set(params) - required - set(defaults))
    if unknown:
        raise ValueError(f"{solver_name}: unknown parameter(s): {unknown}")
    missing = sorted(required - set(params))
    if missing:
        raise KeyError(f"{solver_name}: missing required parameter(s): {missing}")
    merged = dict(defaults)
    merged.update(params)
    _validate_stopping_params(merged, solver_name)
    if merged["precision"] not in ("f32", "f64"):
        raise ValueError(
            f"{solver_name}: precision must be 'f32' or 'f64', "
            f"got {merged['precision']!r}"
        )
    n_steps = merged["n_steps"]
    if n_steps is not None and not (
        isinstance(n_steps, (int, np.integer))
        and not isinstance(n_steps, bool)
        and n_steps >= 1
    ):
        raise ValueError(
            f"{solver_name}: n_steps must be a positive integer or None, "
            f"got {n_steps!r}"
        )
    return merged


def _validate_stopping_params(merged: dict[str, Any], solver_name: str) -> None:
    """stopping_mode / density_threshold validation, shared by all solvers.

    density_threshold is only meaningful for "volume" mode and is rejected
    otherwise (no silent unused parameters).
    """
    mode = merged["stopping_mode"]
    if mode not in ("mass", "volume"):
        raise ValueError(
            f"{solver_name}: stopping_mode must be 'mass' or 'volume', got {mode!r}"
        )
    if mode == "mass":
        if merged["density_threshold"] is not None:
            raise ValueError(
                f"{solver_name}: density_threshold is only valid with "
                "stopping_mode='volume'"
            )
    elif merged["density_threshold"] is None:
        merged["density_threshold"] = DEFAULT_DENSITY_THRESHOLD


def _validate_seed_fractions(params: Mapping[str, Any]) -> None:
    assert 0 <= params["gaussian_seed_x_fraction"] <= 1, "gaussian_seed_x_fraction must be between 0 and 1"
    assert 0 <= params["gaussian_seed_y_fraction"] <= 1, "gaussian_seed_y_fraction must be between 0 and 1"
    assert 0 <= params["gaussian_seed_z_fraction"] <= 1, "gaussian_seed_z_fraction must be between 0 and 1"


class BaseFKPPSolver(ABC):
    """Template method: solve() owns the shared pipeline; subclasses fill hooks.

    __init__ merges and validates params against the class's _REQUIRED /
    _DEFAULTS schema. Grid attributes (grid_shape, grid_spacing, seed_voxel)
    are populated by solve() after downsampling, before any hook that uses
    them is called.
    """

    grid_shape: tuple[int, int, int]
    grid_spacing: tuple[float, float, float]  # mm, post-downsampling
    seed_voxel: tuple[int, int, int]

    #: Parameter schema, defined per solver class.
    _REQUIRED: ClassVar[frozenset[str]]
    _DEFAULTS: ClassVar[dict[str, Any]]
    #: Stopping-quantity device impls consumed by _mass_spec/_volume_spec:
    #: module-level functions wrapped in ``staticmethod`` (stable identity
    #: for the jit cache), set per solver class.
    _mass_impl: ClassVar[Callable[..., jax.Array]]
    _volume_impl: ClassVar[Callable[..., jax.Array]]

    def __init__(self, params: Mapping[str, Any]) -> None:
        merged = _merge_params(
            params, self._REQUIRED, self._DEFAULTS, type(self).__name__
        )
        _validate_seed_fractions(merged)
        self._validate_extra(merged)
        self.params = merged
        self._crop_box: tuple[slice, slice, slice] | None = None

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        """Solver-specific validation beyond the shared schema merge."""

    @property
    def voxel_volume(self) -> float:
        """Product of grid_spacing components (mm^3)."""
        dx, dy, dz = self.grid_spacing
        return dx * dy * dz

    @property
    def _dtype(self) -> jnp.dtype:
        """Device state dtype selected by the ``precision`` parameter."""
        return jnp.float64 if self.params["precision"] == "f64" else jnp.float32

    def _dyn_scalar(self, value: Any) -> jax.Array:
        """Physical scalar as a dynamic 0-d device array at the state dtype.

        Casting on the host keeps the value out of the jit cache key while
        matching closure-literal numerics exactly: a weak f64 Python float
        combined with an f32 array is likewise computed at f32 after
        rounding the scalar.
        """
        return jnp.asarray(float(value), dtype=self._dtype)

    def _gaussian_seed(self) -> jax.Array:
        """Clipped-Gaussian initial tumor density on the full low-res grid,
        from the gaussian_seed_* params, at the ``precision`` dtype."""
        return clipped_gaussian(
            self.grid_shape,
            self.seed_voxel,
            self.grid_spacing,
            scale=self.params["gaussian_seed_scale"],
            dtype=self._dtype,
            diffusion_time=self.params["gaussian_seed_diffusion_time"],
            mass=self.params["gaussian_seed_mass"],
            floor=self.params["gaussian_seed_floor"],
        )

    # --- shared pipeline ---

    def solve(self) -> Result:
        """Run the full pipeline; never raises — any failure is returned as
        ``Result(success=False, stopping_criterion="error")``."""
        try:
            return self._run_pipeline()
        except Exception as exc:  # noqa: BLE001 - originals funnel errors into the result
            if self.params.get("verbose", False):
                logger.debug("solver failed: %s", exc)
            return Result(
                success=False,
                initial_state={},
                final_state={},
                final_time=0.0,
                final_stopping_quantity=0.0,
                stopping_criterion="error",
                error=str(exc),
            )

    def _run_pipeline(self) -> Result:
        params = self.params
        stopping_time = float(params["stopping_time"])
        stopping_threshold = float(params["stopping_threshold"])
        verbose = bool(params["verbose"])
        resolution_factor = params["resolution_factor"]

        low_shape, full_shape = self._prepare_fields()
        self.grid_shape = low_shape
        vx, vy, vz = params["voxel_size_mm"]
        self.grid_spacing = (
            vx / resolution_factor,
            vy / resolution_factor,
            vz / resolution_factor,
        )
        nx, ny, nz = low_shape
        self.seed_voxel = (
            int(params["gaussian_seed_x_fraction"] * nx),
            int(params["gaussian_seed_y_fraction"] * ny),
            int(params["gaussian_seed_z_fraction"] * nz),
        )
        self._check_seed_early()

        if params["n_steps"] is None:
            n_steps, dt = self._time_step_count()
            dt = float(dt)
        else:
            # Explicit step count: bypasses the solver's stability formula.
            n_steps = int(params["n_steps"])
            dt = stopping_time / n_steps
        if verbose:
            logger.debug("number of simulation timesteps: %d", n_steps)

        n_snapshots: int | None = params["n_time_series_snapshots"]
        record_steps = self._record_steps(n_steps, n_snapshots)

        # x64 is enabled locally (config-safe, never globally on import): the
        # state keeps its explicit f32/f64 dtype either way, while the
        # stopping-quantity and guard reductions always run in float64.
        with jax.enable_x64():
            state_full = self._initialize_state()
            initial_snapshot = {
                k: np.asarray(v, dtype=np.float64) for k, v in state_full.items()
            }

            box = tissue_bounding_box(self._crop_mask(), margin=CROP_MARGIN)
            self._crop_box = box
            state = {k: v[box] for k, v in state_full.items()}

            consts = self._device_constants(dt)
            self._check_seed_late()

            loop = _run_time_loop(
                state,
                consts,
                self._step_spec(dt),
                self._quantity_spec(),
                self._guard_spec(),
                n_steps,
                stopping_threshold,
                record_steps,
            )
            final_state_low = {
                k: np.asarray(v, dtype=np.float64) for k, v in loop["state"].items()
            }
            stop_kind = int(loop["stop_kind"])
            stop_step = int(loop["stop_step"])
            stopping_quantity = float(loop["quantity"])
            guard_change = float(loop["guard_change"])
            guard_density = float(loop["guard_density"])
            n_recorded = int(loop["n_recorded"])
            buffers = (
                {
                    k: np.asarray(v[:n_recorded], dtype=np.float64)
                    for k, v in loop["buffers"].items()
                }
                if n_snapshots is not None
                else None
            )

        if stop_kind == _RUNNING:
            final_time = stopping_time
        else:
            final_time = stop_step * dt

        guard_error: str | None = None
        if stop_kind == _STOP_SHRINKAGE:
            guard_error = (
                "shrinkage guard fired: step-to-step cell-density sum "
                f"decreased by {-guard_change} (> {SHRINKAGE_LIMIT:g}) "
                f"(at simulation time {stop_step * dt})"
            )
        elif stop_kind == _STOP_VANISHING:
            guard_error = (
                "vanishing-volume guard fired: integrated cell density "
                f"{guard_density} < {VANISHING_DENSITY_LIMIT:g} "
                f"(at simulation time {stop_step * dt})"
            )
        if guard_error is not None and verbose:
            logger.debug("early loop exit at t=%s: %s", stop_step * dt, guard_error)

        final_low = {
            k: embed(v, box, low_shape) for k, v in final_state_low.items()
        }

        result_time_series: dict[str, NDArray] | None = None
        if buffers is not None:
            result_time_series = {
                key: np.array(
                    [
                        self._upsample_to(embed(frame, box, low_shape), full_shape)
                        for frame in frames
                    ]
                )
                for key, frames in buffers.items()
            }

        if guard_error is not None:
            stopping_criterion: Literal["time", "volume", "error"] = "error"
        elif stop_kind == _STOP_VOLUME:
            stopping_criterion = "volume"
        else:
            stopping_criterion = "time"

        return Result(
            success=guard_error is None,
            initial_state={
                k: self._upsample_to(v, full_shape) for k, v in initial_snapshot.items()
            },
            final_state={
                k: self._upsample_to(v, full_shape) for k, v in final_low.items()
            },
            final_time=final_time,
            final_stopping_quantity=stopping_quantity,
            stopping_criterion=stopping_criterion,
            time_series=result_time_series,
            error=guard_error,
        )

    def _downsample(
        self, field: NDArray, factor: float | Sequence[float], order: int = 1
    ) -> NDArray:
        return zoom(field, factor, order=order)

    def _upsample_to(self, field: NDArray, shape: tuple[int, ...]) -> NDArray:
        # Reproduces the originals' extrapolate_factor: new size / old size
        # per axis, then a linear zoom back up.
        factor = tuple(
            new_sz / float(orig_sz) for new_sz, orig_sz in zip(shape, field.shape)
        )
        return np.array(zoom(field, factor, order=1))

    def _record_steps(self, n_steps: int, n_records: int | None) -> NDArray:
        if n_records is None:
            return np.empty(0, dtype=int)
        return np.linspace(0, n_steps - 1, n_records, dtype=int)

    # --- hooks ---

    @abstractmethod
    def _prepare_fields(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Downsample the solver's tissue/diffusivity input fields (host) and
        store them on self. Returns (low-res 3D grid shape, full-res 3D
        shape)."""

    @abstractmethod
    def _crop_mask(self) -> NDArray:
        """Boolean host mask whose bounding box defines the simulation
        subdomain. Tissue occupancy for the tissue-based solvers, brainmask
        for the DTI solver."""

    @abstractmethod
    def _initialize_state(self) -> State:
        """Device state on the full low-res grid, at the ``precision`` dtype."""

    @abstractmethod
    def _device_constants(self, dt: float) -> Consts:
        """Constant device arrays for the scan (face diffusivities, tissue
        masks), built once per solve on the cropped grid. Passed to the scan
        driver as dynamic arguments, so re-solves with the same shapes and
        dtype reuse the compiled driver."""

    @abstractmethod
    def _step_spec(self, dt: float) -> StepSpec:
        """``StepSpec`` for one explicit-Euler step on the cropped grid:
        step_impl(state, consts, dyn, *static). See the ``StepSpec`` contract
        for the dynamic/static split. A solver that rebuilds its diffusivity
        every step does so inside step_impl, from the carried state."""

    def _quantity_spec(self) -> QuantitySpec:
        """Stopping-quantity spec dispatched on stopping_mode: total cell
        mass ("mass", the default) or a thresholded volume ("volume") — see
        ``Result.final_stopping_quantity`` for the semantics. The reduction
        runs in float64 regardless of the state dtype."""
        if self.params["stopping_mode"] == "volume":
            return self._volume_spec()
        return self._mass_spec()

    def _mass_spec(self) -> QuantitySpec:
        """Total-cell-mass spec (``_mass_impl``): voxel_volume * sum of all
        cell density fields, summed in float64 (nutrient fields are not cell
        densities and are excluded)."""
        return self._mass_impl, (), (self.voxel_volume,)

    def _volume_spec(self) -> QuantitySpec:
        """Thresholded-volume spec (``_volume_impl``): voxel_volume *
        count(summed cell density > density_threshold); P + N for the
        necrotic solver, the nutrient field never included."""
        dyn = (self._dyn_scalar(self.params["density_threshold"]),)
        return self._volume_impl, dyn, (self.voxel_volume,)

    @abstractmethod
    def _time_step_count(self) -> tuple[int, float]:
        """(N_simulation_steps, dt) from each solver's own ad-hoc stability
        formula; bypassed entirely when the ``n_steps`` param is set. The
        three formulas remain mutually inconsistent by heritage and may be
        unified later."""

    def _guard_spec(self) -> GuardSpec:
        """Post-step guard spec, evaluated on the device inside the scan.

        guard_impl(new_state, prev_state, consts, dyn, *static) returns
        (code, shrinkage value, integrated density) with code 0 when no
        guard fires. The AnisotropicFKPPSolver overrides; the default never
        fires (and is pruned by XLA)."""
        return _no_guard, (), ()

    def _check_seed_early(self) -> None:
        """Seed-position guard right after grid geometry is known (the
        original FK solver checks here, before the time-step count)."""

    def _check_seed_late(self) -> None:
        """Seed-position guard after cropping and diffusivity construction
        (the original DTI solver raises at this point in its pipeline)."""
