"""
``BaseFKPPSolver`` implements the pipeline (validate parameters,
downsample the tissue fields (on host), crop to the bounding box,
time stepping (on device), embed and upsample the results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import zoom

from .operators import (
    Constants,
    GuardSpec,
    QuantitySpec,
    SHRINKAGE_LIMIT,
    State,
    StepSpec,
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
    parameters: Mapping[str, Any], keys: Sequence[str], solver_name: str
) -> None:
    """Check that each named parameter lies in [0, 1]."""
    for key in keys:
        value = parameters[key]
        if not 0 <= value <= 1:
            raise ValueError(
                f"{solver_name}: {key} must be between 0 and 1, got {value!r}."
            )


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
    
    # Stopping-quantity device functions consumed by _quantity_spec:
    # module-level functions wrapped in staticmethod (stable identity for
    # the jit cache), set per solver class.
    _mass_func: ClassVar[Callable[..., jax.Array]]
    _volume_func: ClassVar[Callable[..., jax.Array]]

    def __init__(self, params: Mapping[str, Any]) -> None:
        merged = _merge_parameters(
            params, self._REQUIRED, self._DEFAULTS, type(self).__name__
        )
        _validate_parameters(merged, type(self).__name__)
        _validate_unit_interval(
            merged, GAUSSIAN_SEED_POSITION_FRACTION, type(self).__name__
        )
        self._validate_extra(merged)
        self.params = merged
        self._crop_box: tuple[slice, slice, slice] | None = None

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        """Solver-specific validation beyond the shared schema merge."""

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
        Convert a physical scalar to a dynamic 0-d device array at the state
        dtype.

        Casting on the host keeps the value out of the jit cache key while
        matching closure-literal numerics exactly: a weak f64 Python float
        combined with an f32 array is likewise computed at f32 after
        rounding the scalar.
        """
        return jnp.asarray(float(value), dtype=self._dtype)

    def _gaussian_seed(self) -> jax.Array:
        """
        Create the clipped-Gaussian initial tumor density on the full
        low-resolution grid, from the gaussian_seed_* params, at the
        ``precision`` dtype.
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

    # --- shared pipeline ---

    def solve(self) -> Result:
        """
        Run the full pipeline.

        Never raises: any failure is returned as Result(success=False,
        stopping_criterion="error").

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
        original_shape = self._setup_grid()
        n_steps, dt = self._resolve_time_stepping()
        loop = self._run_device_loop(n_steps, dt)
        return self._assemble_result(loop, dt, original_shape)

    def _setup_grid(self) -> tuple[int, int, int]:
        """
        Downsample the input fields and set up the grid geometry.

        Populates grid_shape, grid_spacing and seed_voxel, then runs the
        seed check.

        Returns:
            The full-resolution 3D grid shape, used to upsample the results.
        """
        params = self.params
        resolution_factor = params["resolution_factor"]

        lowres_shape, original_shape = self._prepare_fields()
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
        return original_shape

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
            # Explicit step count: bypasses the solver's stability formula.
            n_steps = int(self.params["n_steps"])
            dt = float(self.params["stopping_time"]) / n_steps
        if self.params["verbose"]:
            logger.info(f"Number of simulation timesteps: {n_steps}")
        return n_steps, dt

    def _run_device_loop(self, n_steps: int, dt: float) -> dict[str, Any]:
        """
        Initialize the device state, crop it to the tissue bounding box and
        run the jitted time loop.

        Args:
            n_steps: Number of time steps.
            dt: Time step size.

        Returns:
            Host-side loop outputs: the initial low-resolution state, the
            final cropped state, the stop bookkeeping values and the
            recorded snapshot buffers (None unless snapshots were requested).
        """
        params = self.params
        n_snapshots: int | None = params["n_time_series_snapshots"]
        record_steps = self._record_steps(n_steps, n_snapshots)

        # x64 is enabled locally (never globally on import): the state keeps
        # its explicit f32/f64 dtype either way, while the stopping-quantity
        # and guard reductions always run in float64.
        with jax.enable_x64():
            state_lowres = self._initialize_state()
            initial_state = {
                k: np.asarray(v, dtype=np.float64) for k, v in state_lowres.items()
            }

            box = tissue_bounding_box(self._crop_mask(), margin=CROP_MARGIN)
            self._crop_box = box
            state_cropped = {k: v[box] for k, v in state_lowres.items()}

            constants = self._device_constants(dt)

            loop = _run_time_loop(
                state_cropped,
                constants,
                self._step_spec(dt),
                self._quantity_spec(),
                self._guard_spec(),
                n_steps,
                float(params["stopping_threshold"]),
                record_steps,
            )
            final_state_cropped = {
                k: np.asarray(v, dtype=np.float64) for k, v in loop["state"].items()
            }
            n_recorded = int(loop["n_recorded"])
            buffers = (
                {
                    k: np.asarray(v[:n_recorded], dtype=np.float64)
                    for k, v in loop["buffers"].items()
                }
                if n_snapshots is not None
                else None
            )

        return {
            "initial_state": initial_state,
            "final_state_cropped": final_state_cropped,
            "stop_kind": int(loop["stop_kind"]),
            "stop_step": int(loop["stop_step"]),
            "quantity": float(loop["quantity"]),
            "guard_change": float(loop["guard_change"]),
            "guard_density": float(loop["guard_density"]),
            "buffers": buffers,
        }

    def _guard_error_message(
        self,
        stop_kind: int,
        stop_step: int,
        dt: float,
        guard_change: float,
        guard_density: float,
    ) -> str | None:
        """Build the error message of a fired guard, or None if none fired."""
        if stop_kind == _STOP_SHRINKAGE:
            return (
                "shrinkage guard fired: step-to-step cell-density sum "
                f"decreased by {-guard_change} (> {SHRINKAGE_LIMIT:g}) "
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
        loop: dict[str, Any],
        dt: float,
        original_shape: tuple[int, int, int],
    ) -> Result:
        """
        Embed the cropped loop outputs into the full low-resolution grid,
        upsample them to the original resolution and build the Result.

        Args:
            loop: Host-side loop outputs of _run_device_loop.
            dt: Time step size.
            original_shape: Full-resolution 3D grid shape.

        Returns:
            The assembled Result.
        """
        stop_kind = loop["stop_kind"]
        stop_step = loop["stop_step"]
        lowres_shape = self.grid_shape
        box = self._crop_box

        if stop_kind == _RUNNING:
            final_time = float(self.params["stopping_time"])
        else:
            final_time = stop_step * dt

        guard_error = self._guard_error_message(
            stop_kind, stop_step, dt, loop["guard_change"], loop["guard_density"]
        )
        if guard_error is not None and self.params["verbose"]:
            logger.info(f"Early loop exit at t={stop_step * dt}: {guard_error}")

        final_state = {
            k: embed(v, box, lowres_shape)
            for k, v in loop["final_state_cropped"].items()
        }

        time_series: dict[str, NDArray] | None = None
        if loop["buffers"] is not None:
            time_series = {
                key: np.array(
                    [
                        self._upsample_to(
                            embed(frame_cropped, box, lowres_shape), original_shape
                        )
                        for frame_cropped in frames
                    ]
                )
                for key, frames in loop["buffers"].items()
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
                for k, v in loop["initial_state"].items()
            },
            final_state={
                k: self._upsample_to(v, original_shape)
                for k, v in final_state.items()
            },
            final_time=final_time,
            final_stopping_quantity=loop["quantity"],
            stopping_criterion=stopping_criterion,
            time_series=time_series,
            error=guard_error,
        )

    def _downsample(
        self, field: NDArray, factor: float | Sequence[float], order: int = 1
    ) -> NDArray:
        """
        Downsample a host field with ``scipy.ndimage.zoom``.

        ``scipy.ndimage.zoom`` is deliberate -- ``jax.image.resize`` is not
        numerically equivalent and must not replace it.
        """
        return zoom(field, factor, order=order)

    def _upsample_to(self, field: NDArray, shape: tuple[int, ...]) -> NDArray:
        """Upsample a host field to shape with a linear ``scipy.ndimage.zoom``."""
        # Per-axis zoom factor new size / old size, then a linear zoom up.
        factor = tuple(
            new_sz / float(orig_sz) for new_sz, orig_sz in zip(shape, field.shape)
        )
        return np.array(zoom(field, factor, order=1))

    def _record_steps(self, n_steps: int, n_records: int | None) -> NDArray:
        """Step indices at which snapshots are recorded (empty for None)."""
        if n_records is None:
            return np.empty(0, dtype=int)
        return np.linspace(0, n_steps - 1, n_records, dtype=int)

    # --- hooks ---

    @abstractmethod
    def _prepare_fields(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        Downsample the solver's tissue/diffusivity input fields (host) and
        store them on self.

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
    def _initialize_state(self) -> State:
        """
        Return the device state on the full low-resolution grid, at the
        ``precision`` dtype.
        """

    @abstractmethod
    def _device_constants(self, dt: float) -> Constants:
        """
        Build the constant device arrays for the scan (face diffusivities,
        tissue masks), once per solve on the cropped grid.

        They are passed to the scan driver as dynamic arguments, so
        re-solves with the same shapes and dtype reuse the compiled driver.
        """

    @abstractmethod
    def _step_spec(self, dt: float) -> StepSpec:
        """
        Return the ``StepSpec`` for one explicit-Euler step on the cropped
        grid.

        See ``StepSpec`` for the function signature and the dynamic/static
        split. A solver that rebuilds its diffusivity every step does so
        inside the step function, from the carried state.
        """

    def _quantity_spec(self) -> QuantitySpec:
        """
        Return the stopping-quantity spec dispatched on stopping_mode.

        See ``Result.final_stopping_quantity``: total cell mass
        (``_mass_func``, the default) or
        voxel_volume * count(cell density > volume_threshold)
        (``_volume_func``). The reduction runs in float64 regardless of the
        state dtype; which fields count as cell density is documented on the
        per-solver functions.
        """
        if self.params["stopping_mode"] == "volume":
            dynamic_scalars = {
                "volume_threshold": self._dynamic_scalar(
                    self.params["volume_threshold"]
                )
            }
            return {
                "func": self._volume_func,
                "dynamic_scalars": dynamic_scalars,
                "static_args": (self.voxel_volume,),
            }
        return {
            "func": self._mass_func,
            "dynamic_scalars": {},
            "static_args": (self.voxel_volume,),
        }

    @abstractmethod
    def _time_step_count(self) -> tuple[int, float]:
        """
        Return (N_simulation_steps, dt) from the solver's own ad-hoc
        stability formula.

        Bypassed entirely when the ``n_steps`` param is set. The three
        formulas are deliberately not unified -- do not merge or "fix" them.
        """

    def _guard_spec(self) -> GuardSpec:
        """
        Return the post-step guard spec, evaluated on the device inside the
        scan.

        See ``GuardSpec`` for the function signature. The AnisotropicFKPPSolver
        overrides; the default never fires (and is pruned by XLA).
        """
        return {"func": _no_guard, "dynamic_scalars": {}, "static_args": ()}

    def _check_seed(self) -> None:
        """
        Raise if the seed voxel lies outside the solver's simulated tissue.

        Runs once grid geometry and the seed voxel are known; the default
        accepts any seed (the two-compartment solver never checks).
        """
