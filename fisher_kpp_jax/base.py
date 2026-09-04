"""
``BaseFKPPSolver`` implements the pipeline (validate parameters,
downsample the tissue fields (on host), crop to the bounding box,
time stepping (on device), embed and upsample the results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import zoom

from .operators import (
    SHRINKAGE_LIMIT,
    VANISHING_DENSITY_LIMIT,
    _RUNNING,
    _STOP_SHRINKAGE,
    _STOP_THRESHOLD,
    _STOP_VANISHING,
    _no_guard,
    _run_time_loop,
    clipped_gaussian,
    embed,
    tissue_bounding_box,
)

CROP_MARGIN: int = 2

DEFAULT_VOLUME_THRESHOLD: float = 0.5

GAUSSIAN_SEED_POSITION_FRACTION: tuple[str, ...] = tuple(
    f"gaussian_seed_{axis}_fraction" for axis in "xyz"
)


class _SharedConstants(TypedDict):
    """
    Device constants shared by all solvers, built by
    ``BaseFKPPSolver._run_device_loop`` and merged flat with the
    solver-specific keys returned by ``_build_device_constants``.

    Attributes:
        dt: Time step size, 0-d scalar at the state dtype.
        grid_spacing: Grid spacing (dx, dy, dz) in mm, 0-d scalars at the
            state dtype.
        voxel_volume: Voxel volume in mm^3, 0-d float64 scalar (feeds the
            f64 stopping-quantity reduction).
        stopping_threshold: Stopping threshold, 0-d float64 scalar.
        volume_threshold: Density threshold of the volume stopping
            quantity, 0-d scalar at the state dtype; only present with
            stopping_mode='volume'.
    """

    dt: jax.Array
    grid_spacing: tuple[jax.Array, jax.Array, jax.Array]
    voxel_volume: jax.Array
    stopping_threshold: jax.Array
    volume_threshold: NotRequired[jax.Array]


@dataclass(slots=True)
class Result:
    """
    Outcome of a solver run.

    Attributes:
        success: Whether the solve completed without an error.
        initial_state: Initial fields on the full-resolution grid.
        final_state: Final fields on the full-resolution grid.
        final_time: Simulation time at which the loop ended.
        final_stopping_quantity: Last value of the quantity compared against
            stopping_threshold: total cell mass for stopping_mode="mass", 
            volume for stopping_mode="volume".
        stopping_criterion: What ended the run: "time", "threshold" or
            "error".
        time_series: Recorded snapshots per field, or None if none were
            requested.
        error: Description of the failure, else None.
    """

    success: bool
    initial_state: dict[str, NDArray]
    final_state: dict[str, NDArray]
    final_time: float
    final_stopping_quantity: float
    stopping_criterion: Literal["time", "threshold", "error"]
    time_series: dict[str, NDArray] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _TimeLoopOutputs:
    """
    Host-side outputs of ``BaseFKPPSolver._run_device_loop``, consumed by
    ``_assemble_result``.

    Attributes:
        initial_state: Initial fields on the full low-resolution grid.
        final_state_cropped: Final fields on the cropped grid.
        crop_box: Slices of the tissue bounding box the loop ran on.
        stop_kind: Stop-kind code (_RUNNING, _STOP_THRESHOLD,
            _STOP_SHRINKAGE or _STOP_VANISHING).
        stop_step: Step index at which the loop stopped, 0 if it never did.
        stopping_quantity: Stopping quantity of the last active step.
        guard_mass_change: Guard diagnostic of the stopping step, 0.0
            unless a guard fired.
        guard_density: Guard diagnostic of the stopping step, 0.0 unless a
            guard fired.
        buffers: Recorded snapshot frames per field on the cropped grid, or
            None if none were requested.
    """

    initial_state: dict[str, NDArray]
    final_state_cropped: dict[str, NDArray]
    crop_box: tuple[slice, slice, slice]
    stop_kind: int
    stop_step: int
    stopping_quantity: float
    guard_mass_change: float
    guard_density: float
    buffers: dict[str, NDArray] | None


def _merge_parameters(
    params: Mapping[str, Any],
    required: frozenset[str],
    defaults: Mapping[str, Any],
    solver_name: str,
) -> dict[str, Any]:
    """
    Merge user parameters with the solver defaults, strictly: unknown keys
    and missing required keys raise.

    Args:
        params: User-supplied parameters.
        required: Names of the required parameters.
        defaults: Default values of the optional parameters.
        solver_name: Solver class name, used in error messages.

    Returns:
        The merged parameter dict.
    """
    params = dict(params)
    unknown = sorted(set(params) - required - set(defaults))
    if unknown:
        raise ValueError(f"{solver_name}: unknown parameter(s): {unknown}.")
    missing = sorted(required - set(params))
    if missing:
        raise KeyError(f"{solver_name}: missing required parameter(s): {missing}.")
    merged = dict(defaults)
    merged.update(params)
    return merged


def _validate_parameters(parameters: dict[str, Any], solver_name: str) -> None:
    """
    Check the parameter values shared by all solvers.

    volume_threshold is only meaningful for "volume" mode and is rejected
    otherwise.

    Args:
        parameters: Merged parameter dict, modified in place.
        solver_name: Solver class name, used in error messages.
    """
    mode = parameters["stopping_mode"]
    if mode not in ("mass", "volume"):
        raise ValueError(
            f"{solver_name}: stopping_mode must be 'mass' or 'volume', got {mode!r}."
        )
    if mode == "mass":
        if parameters["volume_threshold"] is not None:
            raise ValueError(
                f"{solver_name}: volume_threshold is only valid with "
                "stopping_mode='volume'."
            )
    elif parameters["volume_threshold"] is None:
        parameters["volume_threshold"] = DEFAULT_VOLUME_THRESHOLD
        logger.info(
            f"{solver_name}: volume_threshold not set, defaulting to "
            f"{DEFAULT_VOLUME_THRESHOLD}."
        )
    if parameters["precision"] not in ("f32", "f64"):
        raise ValueError(
            f"{solver_name}: precision must be 'f32' or 'f64', "
            f"got {parameters['precision']!r}."
        )
    n_steps = parameters["n_steps"]
    if n_steps is not None and not (
        isinstance(n_steps, (int, np.integer))
        and not isinstance(n_steps, bool)
        and n_steps >= 1
    ):
        raise ValueError(
            f"{solver_name}: n_steps must be a positive integer or None, "
            f"got {n_steps!r}."
        )


def _validate_unit_interval(
    parameters: Mapping[str, Any], key: str, solver_name: str
) -> None:
    """Check that the parameter named key lies in [0, 1]."""
    value = parameters[key]
    if not 0 <= value <= 1:
        raise ValueError(f"{solver_name}: {key} must be between 0 and 1, got {value!r}.")


def _validate_tissue_arrays(parameters: Mapping[str, Any], solver_name: str) -> None:
    """Check that gray_matter_pbmap and white_matter_pbmap are 3D numpy
    arrays of one shape."""
    gm = parameters["gray_matter_pbmap"]
    wm = parameters["white_matter_pbmap"]
    if not (isinstance(gm, np.ndarray) and isinstance(wm, np.ndarray)):
        raise ValueError(
            f"{solver_name}: gray_matter_pbmap and white_matter_pbmap must be numpy arrays."
        )
    if not (gm.ndim == 3 and wm.ndim == 3):
        raise ValueError(
            f"{solver_name}: gray_matter_pbmap and white_matter_pbmap must be 3D arrays."
        )
    if gm.shape != wm.shape:
        raise ValueError(
            f"{solver_name}: gray_matter_pbmap and white_matter_pbmap shapes differ: "
            f"{gm.shape} vs {wm.shape}."
        )


def _validate_nonnegative_scalar(
    parameters: Mapping[str, Any], key: str, solver_name: str
) -> float:
    """Check that the parameter named key is a finite scalar >= 0; return it
    as a float."""
    value = parameters[key]
    if not (np.isscalar(value) and np.isfinite(value) and value >= 0):
        raise ValueError(
            f"{solver_name}: {key} must be a finite nonnegative scalar, got {value!r}."
        )
    return float(value)


def _validate_positive_scalar(
    parameters: Mapping[str, Any], key: str, solver_name: str
) -> float:
    """Check that the parameter named key is a finite scalar > 0; return it
    as a float."""
    value = parameters[key]
    if not (np.isscalar(value) and np.isfinite(value) and value > 0):
        raise ValueError(
            f"{solver_name}: {key} must be a finite positive scalar, got {value!r}."
        )
    return float(value)


def _validate_nonnegative_sequence(
    parameters: Mapping[str, Any], key: str, solver_name: str
) -> NDArray:
    """Check that the parameter named key is a 1-D sequence of finite
    nonnegative numbers; return it as a float64 array."""
    values = np.asarray(parameters[key], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{solver_name}: {key} must be a 1-D sequence.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{solver_name}: {key} must be finite and nonnegative.")
    return values


def _validate_event_times(
    parameters: Mapping[str, Any], key: str, solver_name: str
) -> NDArray:
    """
    Check that the parameter named key is a 1-D sequence of finite
    nonnegative times; return it as a float64 array. Times beyond the
    parameters' stopping_time are legal but never fire; a warning names
    them.
    """
    times = _validate_nonnegative_sequence(parameters, key, solver_name)
    stopping_time = float(parameters["stopping_time"])
    late = times[times > stopping_time]
    if late.size:
        logger.warning(
            f"{solver_name}: {key} contains {late.size} time(s) beyond "
            f"stopping_time={stopping_time:g} that will never fire: {late.tolist()}."
        )
    return times


def _validate_volume(
    parameters: Mapping[str, Any], key: str, shape: tuple[int, ...], solver_name: str
) -> NDArray:
    """Check that the parameter named key is a 3D numpy array of the given
    (tissue map) shape; return it."""
    value = parameters[key]
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{solver_name}: {key} must be a numpy array.")
    if value.ndim != 3:
        raise ValueError(f"{solver_name}: {key} must be a 3D array.")
    if value.shape != shape:
        raise ValueError(
            f"{solver_name}: {key} shape {value.shape} differs from the tissue map "
            f"shape {shape}."
        )
    return value


class BaseFKPPSolver(ABC):
    """
    Base class implementing the shared solve() pipeline as a template method.

    __init__ merges and validates params against the class's _REQUIRED /
    _DEFAULTS schema; subclasses implement the solver-specific hooks.

    Attributes:
        params: Merged and validated solver parameters.
        grid_shape: Low-resolution 3D grid shape, populated by solve() after
            downsampling, before any hook that uses it is called.
        grid_spacing: Grid spacing in mm after downsampling, populated by
            solve() alongside grid_shape.
        seed_voxel: Voxel index of the Gaussian seed center, populated by
            solve() alongside grid_shape.
    """

    grid_shape: tuple[int, int, int]
    grid_spacing: tuple[float, float, float]
    seed_voxel: tuple[int, int, int]

    # Required and default parameters, implemented by each solver
    _REQUIRED: ClassVar[frozenset[str]]
    _DEFAULTS: ClassVar[dict[str, Any]]

    # Device functions of the time loop, set per solver class. They must be
    # module-level functions (see ``operators._run_time_loop`` for the
    # required signatures). _step_func performs one time step.
    # _mass_func/_volume_func are the stopping quantities dispatched on
    # stopping_mode; _guard_func is the post-step sanity check (the default
    # never fires).
    _step_func: ClassVar[Callable[..., dict[str, jax.Array]]]
    _mass_func: ClassVar[Callable[..., jax.Array]]
    _volume_func: ClassVar[Callable[..., jax.Array]]
    _guard_func: ClassVar[
        Callable[..., tuple[jax.Array, jax.Array, jax.Array]]
    ] = staticmethod(_no_guard)

    def __init__(self, params: Mapping[str, Any]) -> None:
        merged = _merge_parameters(
            params, self._REQUIRED, self._DEFAULTS, type(self).__name__
        )
        _validate_parameters(merged, type(self).__name__)
        for key in GAUSSIAN_SEED_POSITION_FRACTION:
            _validate_unit_interval(merged, key, type(self).__name__)
        self._validate_extra(merged)
        self.params = merged

    def _validate_extra(self, params: dict[str, Any]) -> None:
        """Solver-specific validation beyond the shared parameters. The
        merged dict is passed and may be completed with derived parameters
        (as ``_validate_parameters`` does for volume_threshold)."""

    @property
    def voxel_volume(self) -> float:
        """Product of the grid_spacing components (mm^3)."""
        dx, dy, dz = self.grid_spacing
        return dx * dy * dz

    @property
    def _dtype(self) -> jnp.dtype:
        """Device state dtype selected by the ``precision`` parameter."""
        return jnp.float64 if self.params["precision"] == "f64" else jnp.float32

    def _dynamic_scalar(self, value: Any) -> jax.Array:
        """
        Cast a scalar to a 0-d device array of dtype.

        As an array it is a dynamic jit argument, so changing its value
        never triggers recompilation.
        """
        return jnp.asarray(float(value), dtype=self._dtype)

    def _gaussian_seed(self) -> jax.Array:
        """
        Create the clipped-Gaussian initial tumor density on the full
        low-resolution grid.
        """
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

    def solve(self) -> Result:
        """
        Run the full pipeline.

        Returns:
            The Result of the run.
        """
        try:
            return self._run_pipeline()
        except Exception as exc:  # noqa: BLE001 - all failures become an error Result
            if self.params["verbose"]:
                logger.error(f"Solver failed: {exc}")
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
        original_shape, _ = self._setup_grid()
        n_steps, dt = self._resolve_time_stepping()
        loop_results = self._run_device_loop(n_steps, dt)
        return self._assemble_result(loop_results, dt, original_shape)

    def _setup_grid(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        Downsample the input fields and set up the grid geometry.

        Populates grid_shape, grid_spacing and seed_voxel, then runs the
        seed check.

        Returns:
            (full-resolution 3D grid shape, low-resolution 3D grid shape).
            The full-resolution shape is used to upsample the results.
        """
        params = self.params
        resolution_factor = params["resolution_factor"]

        lowres_shape, original_shape = self._prepare_input_fields()
        self.grid_shape = lowres_shape
        vx, vy, vz = params["voxel_size_mm"]
        self.grid_spacing = (
            vx / resolution_factor,
            vy / resolution_factor,
            vz / resolution_factor,
        )
        nx, ny, nz = lowres_shape
        self.seed_voxel = (
            int(params["gaussian_seed_x_fraction"] * nx),
            int(params["gaussian_seed_y_fraction"] * ny),
            int(params["gaussian_seed_z_fraction"] * nz),
        )
        self._check_seed()
        return original_shape, lowres_shape

    def _resolve_time_stepping(self) -> tuple[int, float]:
        """
        Choose the number of simulation time steps and the step size dt.

        Returns:
            (n_steps, dt): the solver's own stability formula by default, or
            the explicit n_steps override (dt = stopping_time / n_steps).
        """
        if self.params["n_steps"] is None:
            n_steps, dt = self._time_step_count()
            dt = float(dt)
        else:
            n_steps = int(self.params["n_steps"])
            dt = float(self.params["stopping_time"]) / n_steps
        if self.params["verbose"]:
            logger.info(f"Number of simulation timesteps: {n_steps}")
        return n_steps, dt

    def _run_device_loop(self, n_steps: int, dt: float) -> _TimeLoopOutputs:
        """
        Initialize the device state, crop it to the tissue bounding box and
        run the jitted time loop.

        Args:
            n_steps: Number of time steps.
            dt: Time step size.

        Returns:
            The host-side loop outputs, see ``_TimeLoopOutputs``.
        """
        params = self.params
        n_snapshots: int | None = params["n_time_series_snapshots"]

        # x64 is enabled locally (never globally on import): the state keeps
        # its explicit f32/f64 dtype either way, while the stopping-quantity
        # and guard reductions always run in float64.
        with jax.enable_x64():
            state_lowres = self._initialize_state()
            initial_state = {
                k: np.asarray(v, dtype=np.float64) for k, v in state_lowres.items()
            }

            box = tissue_bounding_box(self._crop_mask(), margin=CROP_MARGIN)
            state_cropped = {k: v[box] for k, v in state_lowres.items()}

            dx, dy, dz = self.grid_spacing
            shared: _SharedConstants = {
                "dt": self._dynamic_scalar(dt),
                "grid_spacing": (
                    self._dynamic_scalar(dx),
                    self._dynamic_scalar(dy),
                    self._dynamic_scalar(dz),
                ),
                # f64 factors and threshold to match the f64
                # stopping-quantity reduction they feed.
                "voxel_volume": jnp.asarray(self.voxel_volume, dtype=jnp.float64),
                "stopping_threshold": jnp.asarray(
                    float(params["stopping_threshold"]), dtype=jnp.float64
                ),
            }
            if params["stopping_mode"] == "volume":
                # Compared against the state fields, so at the state dtype.
                shared["volume_threshold"] = self._dynamic_scalar(
                    params["volume_threshold"]
                )
            # shared is spread last so a stray solver key can never
            # overwrite a shared entry.
            constants: dict[str, Any] = {
                **self._build_device_constants(box),
                **shared,
            }

            device_outputs = _run_time_loop(
                state_cropped,
                constants,
                self._step_func,
                self._quantity_func(),
                self._guard_func,
                n_steps,
                n_snapshots,
            )
            final_state_cropped = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in device_outputs["state"].items()
            }
            n_recorded = int(device_outputs["n_recorded"])
            buffers = (
                {
                    k: np.asarray(v[:n_recorded], dtype=np.float64)
                    for k, v in device_outputs["buffers"].items()
                }
                if n_snapshots is not None
                else None
            )

        return _TimeLoopOutputs(
            initial_state=initial_state,
            final_state_cropped=final_state_cropped,
            crop_box=box,
            stop_kind=int(device_outputs["stop_kind"]),
            stop_step=int(device_outputs["stop_step"]),
            stopping_quantity=float(device_outputs["stopping_quantity"]),
            guard_mass_change=float(device_outputs["guard_mass_change"]),
            guard_density=float(device_outputs["guard_density"]),
            buffers=buffers,
        )

    def _guard_error_message(
        self,
        stop_kind: int,
        stop_step: int,
        dt: float,
        guard_mass_change: float,
        guard_density: float,
    ) -> str | None:
        """Build the error message of a fired guard, or None if none fired."""
        if stop_kind == _STOP_SHRINKAGE:
            return (
                "shrinkage guard fired: step-to-step cell-density sum "
                f"decreased by {-guard_mass_change} (> {SHRINKAGE_LIMIT:g}) "
                f"(at simulation time {stop_step * dt})"
            )
        if stop_kind == _STOP_VANISHING:
            return (
                "vanishing-volume guard fired: integrated cell density "
                f"{guard_density} < {VANISHING_DENSITY_LIMIT:g} "
                f"(at simulation time {stop_step * dt})"
            )
        return None

    def _assemble_result(
        self,
        loop_results: _TimeLoopOutputs,
        dt: float,
        original_shape: tuple[int, int, int],
    ) -> Result:
        """
        Embed the cropped loop outputs into the full low-resolution grid,
        upsample them to the original resolution and build the Result.

        Args:
            loop_results: Host-side loop outputs of _run_device_loop.
            dt: Time step size.
            original_shape: Full-resolution 3D grid shape.

        Returns:
            The assembled Result.
        """
        stop_kind = loop_results.stop_kind
        stop_step = loop_results.stop_step
        lowres_shape = self.grid_shape
        box = loop_results.crop_box

        if stop_kind == _RUNNING:
            final_time = float(self.params["stopping_time"])
        else:
            final_time = stop_step * dt

        guard_error = self._guard_error_message(
            stop_kind,
            stop_step,
            dt,
            loop_results.guard_mass_change,
            loop_results.guard_density,
        )
        if guard_error is not None and self.params["verbose"]:
            logger.info(f"Early loop exit at t={stop_step * dt}: {guard_error}")

        final_state = {
            k: embed(v, box, lowres_shape)
            for k, v in loop_results.final_state_cropped.items()
        }

        time_series: dict[str, NDArray] | None = None
        if loop_results.buffers is not None:
            time_series = {
                key: np.array(
                    [
                        self._upsample_to(
                            embed(frame_cropped, box, lowres_shape), original_shape
                        )
                        for frame_cropped in frames
                    ]
                )
                for key, frames in loop_results.buffers.items()
            }

        if guard_error is not None:
            stopping_criterion: Literal["time", "threshold", "error"] = "error"
        elif stop_kind == _STOP_THRESHOLD:
            stopping_criterion = "threshold"
        else:
            stopping_criterion = "time"

        return Result(
            success=guard_error is None,
            initial_state={
                k: self._upsample_to(v, original_shape)
                for k, v in loop_results.initial_state.items()
            },
            final_state={
                k: self._upsample_to(v, original_shape)
                for k, v in final_state.items()
            },
            final_time=final_time,
            final_stopping_quantity=loop_results.stopping_quantity,
            stopping_criterion=stopping_criterion,
            time_series=time_series,
            error=guard_error,
        )

    def _downsample(
        self, field: NDArray, factor: float | Sequence[float], order: int = 1
    ) -> NDArray:
        """
        Downsample a host field with ``scipy.ndimage.zoom``.
        """
        return zoom(field, factor, order=order)

    def _upsample_to(self, field: NDArray, shape: tuple[int, ...]) -> NDArray:
        """Upsample a host field to shape with a linear ``scipy.ndimage.zoom``."""
        # Per-axis zoom factor new size / old size, then a linear zoom up.
        factor = tuple(
            new_sz / float(orig_sz) for new_sz, orig_sz in zip(shape, field.shape)
        )
        return np.array(zoom(field, factor, order=1))

    # --- hooks ---

    @abstractmethod
    def _prepare_input_fields(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        Downsample the solver's tissue/diffusivity input fields on the host,
        preserving the field of view, and store them on self.

        Returns:
            (low-resolution 3D grid shape, full-resolution 3D grid shape).
        """

    @abstractmethod
    def _crop_mask(self) -> NDArray:
        """
        Return the boolean host mask whose bounding box defines the
        simulation subdomain: tissue occupancy for the tissue-based solvers,
        brain mask for the DTI solver.
        """

    @abstractmethod
    def _initialize_state(self) -> dict[str, jax.Array]:
        """
        Return the device state on the full low-resolution grid, at the
        ``precision`` dtype.
        """

    @abstractmethod
    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> Mapping[str, Any]:
        """
        Build the solver-specific device inputs of the scan, once per
        solve on the cropped grid: the field arrays (face diffusivities,
        tissue masks) and the solver's physical parameters as 0-d device
        scalars.

        Each solver types the returned dict as its own TypedDict, so its
        key set is visible and checkable in one place. The base solver
        merges it flat with the ``_SharedConstants`` it builds itself;
        solver keys must not collide with the shared ones (the flat
        per-solver TypedDicts enforce this statically).

        Args:
            box: Slices of the tissue bounding box; the solver's host
                fields are cropped to it before moving to the device.
        """

    def _quantity_func(self) -> Callable[..., jax.Array]:
        """
        Return the stopping-quantity function dispatched on stopping_mode.

        See ``Result.final_stopping_quantity``: total cell mass
        (``_mass_func``, the default) or
        voxel_volume * count(cell density > volume_threshold)
        (``_volume_func``). The reduction runs in float64 regardless of the
        state dtype; which fields count as cell density is documented on the
        per-solver functions.
        """
        if self.params["stopping_mode"] == "volume":
            return self._volume_func
        return self._mass_func

    @abstractmethod
    def _time_step_count(self) -> tuple[int, float]:
        """
        Return (N_simulation_steps, dt) from the solver's own ad-hoc
        stability formula.

        Bypassed entirely when the ``n_steps`` param is set. The three
        formulas are deliberately not unified -- do not merge or "fix" them.
        """

    def _check_seed(self) -> None:
        """
        Raise if the seed voxel lies outside the solver's simulated tissue.

        Runs once grid geometry and the seed voxel are known; the default
        accepts any seed (the two-compartment solver never checks).
        """
