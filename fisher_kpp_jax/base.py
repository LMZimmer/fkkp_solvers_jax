"""
``BaseFKPPSolver`` implements the pipeline (validate parameters, load the
volumes, downsample the tissue fields (on host), crop to the bounding box,
time stepping (on device), embed and upsample the results), holds the
config of the run and saves results with their config.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import zoom

from .config import (
    SOLVER_KEY,
    VOLUME_IN_MEMORY,
    jsonable,
    read_config,
    register_solver,
    resolve_config_path,
    write_config,
)
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

# Voxel size used when the volumes are in-memory arrays and voxel_size_mm
# is not given.
DEFAULT_VOXEL_SIZE_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)

# The three ways to give the time step; at most one of them may be set.
TIME_STEP_KEYS: tuple[str, ...] = ("n_steps", "dt", "steps_per_day")

# Parameters the solver may derive when the config does not give them;
# their derived values are reported in Result.derived.
DERIVED_KEYS: tuple[str, ...] = ("stopping_time", "volume_threshold", "voxel_size_mm")

# Directory of the pre-written default configs, one per solver class.
DEFAULT_CONFIG_DIR: Path = Path(__file__).resolve().parent / "configs"

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
        time_series: Recorded snapshots per field, (n_frames, *grid), or
            None if none were requested (``snapshot_times`` param None).
        snapshot_times: Simulation day of each recorded frame, or None if
            none were requested. Frames are recorded after the step whose
            end time is nearest to each requested day (a day below one
            step: after the first step); days sharing a step give one
            frame, days beyond the horizon are dropped with a warning, and
            frames after an early stop are not recorded, so this holds the
            days actually recorded, ascending.
        error: Description of the failure, else None.
        config: The solver's config (see ``BaseFKPPSolver.config``), None
            until ``solve`` fills it in.
        n_steps: Number of time steps the run used, None if the run failed
            before the time step was resolved.
        dt: Time step size in days, None like n_steps.
        wall_time_s: Wall time of ``solve`` in seconds.
        derived: Parameter values the solver derived because the config did
            not give them (``DERIVED_KEYS``): stopping_time, a defaulted
            volume_threshold, a voxel_size_mm read from a NIfTI header.
        affine: Voxel-to-world affine of the volumes, from the NIfTI header
            of the reference volume when it was given as a path, else a
            diagonal of the voxel size; used to write the result volumes.
    """

    success: bool
    initial_state: dict[str, NDArray]
    final_state: dict[str, NDArray]
    final_time: float
    final_stopping_quantity: float
    stopping_criterion: Literal["time", "threshold", "error"]
    time_series: dict[str, NDArray] | None = None
    snapshot_times: NDArray | None = None
    error: str | None = None
    config: dict[str, Any] | None = None
    n_steps: int | None = None
    dt: float | None = None
    wall_time_s: float | None = None
    derived: dict[str, Any] = field(default_factory=dict)
    affine: NDArray | None = None

    def save(
        self, outdir: str | Path, time_series: bool = False, overwrite: bool = False
    ) -> Path:
        """
        Write the result into a directory.

        Files written (float32 NIfTIs with the Result's affine):

        - ``config.json``: the config (``write_config``), loadable as is,
          if the Result has one;
        - ``result.json``: solver name, success, error, stopping criterion,
          final time, final stopping quantity, n_steps, dt, wall time, grid
          shape, the recorded snapshot days, the derived parameter values
          and the list of files written;
        - ``initial_<field>.nii.gz`` and ``final_<field>.nii.gz`` per state
          field, for every field the Result holds (a failed Result without
          states writes the two JSON files only);
        - ``time_series_<field>.nii.gz`` (4D, frames last) per recorded
          field, only with time_series=True.

        Args:
            outdir: Directory to write into, created if needed.
            time_series: Whether to write the recorded snapshots.
            overwrite: Whether an existing ``result.json`` in outdir may be
                overwritten; else it raises.

        Returns:
            The directory written.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        record_path = outdir / "result.json"
        if record_path.exists() and not overwrite:
            raise FileExistsError(f"{record_path} exists (pass overwrite=True).")
        files: list[str] = []
        if self.config is not None:
            write_config(self.config, outdir / "config.json")
            files.append("config.json")
        affine = np.eye(4) if self.affine is None else self.affine
        for prefix, states in (("initial", self.initial_state), ("final", self.final_state)):
            for key, volume in states.items():
                name = f"{prefix}_{key}.nii.gz"
                _save_nifti(outdir / name, volume, affine)
                files.append(name)
        if time_series and self.time_series is not None:
            for key, frames in self.time_series.items():
                name = f"time_series_{key}.nii.gz"
                _save_nifti(outdir / name, np.moveaxis(frames, 0, -1), affine)
                files.append(name)
        grid_shape = None
        for volume in self.final_state.values():
            grid_shape = list(volume.shape)
            break
        record = {
            SOLVER_KEY: None if self.config is None else self.config.get(SOLVER_KEY),
            "success": self.success,
            "error": self.error,
            "stopping_criterion": self.stopping_criterion,
            "final_time": self.final_time,
            "final_stopping_quantity": self.final_stopping_quantity,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "wall_time_s": self.wall_time_s,
            "grid_shape": grid_shape,
            "snapshot_times": jsonable(self.snapshot_times),
            "derived": jsonable(self.derived),
            "files": files + ["result.json"],
        }
        record_path.write_text(json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8")
        return outdir


def _save_nifti(path: Path, volume: NDArray, affine: NDArray) -> None:
    """Write a float32 NIfTI with the given affine."""
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), np.asarray(affine))
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))


def n_steps_from_dt(stopping_time: float, dt: float) -> int:
    """
    Translate a requested time step into a step count.

    n_steps = ceil(stopping_time / dt), so the effective step
    stopping_time / n_steps is the largest step <= dt that divides the
    horizon exactly; a warning reports it when it differs from dt.

    Args:
        stopping_time: Simulation horizon in days.
        dt: Requested time step in days, > 0.

    Returns:
        The number of steps.
    """
    stopping_time = float(stopping_time)
    dt = float(dt)
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be a positive finite number, got {dt!r}.")
    if not (np.isfinite(stopping_time) and stopping_time > 0):
        raise ValueError(
            f"stopping_time must be a positive finite number, got {stopping_time!r}."
        )
    n_steps = int(np.ceil(stopping_time / dt - 1e-9))
    effective = stopping_time / n_steps
    if abs(effective - dt) > 1e-9 * dt:
        logger.warning(
            f"dt={dt:g} does not divide stopping_time={stopping_time:g}; using "
            f"n_steps={n_steps} (dt={effective:g})."
        )
    return n_steps


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
        snapshot_times: Simulation day of each recorded frame, or None if
            none were requested.
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
    snapshot_times: NDArray | None


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
    for key in ("dt", "steps_per_day"):
        value = parameters[key]
        if value is not None and not (
            np.isscalar(value) and np.isfinite(value) and value > 0
        ):
            raise ValueError(
                f"{solver_name}: {key} must be a positive finite number or None, "
                f"got {value!r}."
            )
    given = [key for key in TIME_STEP_KEYS if parameters[key] is not None]
    if len(given) > 1:
        raise ValueError(
            f"{solver_name}: set at most one of {list(TIME_STEP_KEYS)}, got {given}."
        )
    snapshot_times = parameters["snapshot_times"]
    if snapshot_times is not None:
        days = np.asarray(snapshot_times, dtype=np.float64)
        if days.ndim != 1:
            raise ValueError(
                f"{solver_name}: snapshot_times must be a 1-D sequence of days or None."
            )
        if not np.all(np.isfinite(days)) or np.any(days < 0):
            raise ValueError(f"{solver_name}: snapshot_times must be finite and nonnegative.")
        parameters["snapshot_times"] = np.sort(days)


def _snapshot_steps(
    days: NDArray, n_steps: int, dt: float, solver_name: str
) -> tuple[NDArray, NDArray]:
    """
    Map requested snapshot days to the steps to record.

    A frame recorded at step s holds the state after that step, at
    (s + 1) dt, so each day is mapped to the step whose end time is nearest
    (a day below one step to the first step). Days beyond the horizon are
    dropped with a warning; days sharing a step collapse into one frame.

    Args:
        days: Requested days, finite and nonnegative.
        n_steps: Number of time steps.
        dt: Time step size in days.
        solver_name: Solver class name, used in the warning.

    Returns:
        (steps, times): the sorted unique steps to record and the
        simulation day of each.
    """
    steps = np.clip(np.rint(days / dt).astype(np.int64) - 1, 0, None)
    late = days[steps >= n_steps]
    if late.size:
        logger.warning(
            f"{solver_name}: {late.size} snapshot day(s) beyond the horizon "
            f"{n_steps * dt:g} are dropped: {late.tolist()}."
        )
    steps = np.unique(steps[steps < n_steps])
    return steps, (steps + 1) * dt


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

    A solver is constructed from its parameters, given as one mapping
    (``Solver(params)``) or as keyword arguments (``Solver(**config)``); a
    ``"solver"`` entry naming the class is accepted and checked, so a
    config written by ``Result.save`` or read by ``read_config`` can be
    passed as is. __init__ merges and validates the parameters against the
    class's _REQUIRED / _DEFAULTS key sets and loads the volumes given as
    NIfTI paths; subclasses implement the solver-specific hooks.

    Attributes:
        params: Merged and validated solver parameters, volumes as arrays.
        config: The run's config: the "solver" entry plus every parameter
            as given, the class defaults filled in for the ones not given
            and volumes recorded as their path (or the entry they were
            given as) or as ``VOLUME_IN_MEMORY`` when given as arrays.
            Derived values are not in it (``DERIVED_KEYS``), so it
            reproduces the run through ``solver_from_config``.
        affine: Voxel-to-world affine, from the NIfTI header of the
            reference volume (``_REFERENCE_VOLUME_KEY``) if that was given
            as a path, else of any volume given as a path, else a diagonal
            of the voxel size.
        result: The Result of the last ``solve``, None before the first.
        n_steps: Number of time steps of the last (or running) solve, None
            before the time step was resolved.
        dt: Time step of the last (or running) solve in days, None like
            n_steps.
        grid_shape: Low-resolution 3D grid shape, populated by solve() after
            downsampling, before any hook that uses it is called.
        grid_spacing: Grid spacing in mm after downsampling, populated by
            solve() alongside grid_shape.
        seed_voxel: Voxel index of the Gaussian seed center, populated by
            solve() alongside grid_shape.
    """

    params: dict[str, Any]
    config: dict[str, Any]
    affine: NDArray
    result: Result | None
    n_steps: int | None
    dt: float | None
    grid_shape: tuple[int, int, int]
    grid_spacing: tuple[float, float, float]
    seed_voxel: tuple[int, int, int]

    # Required and default parameters, implemented by each solver
    _REQUIRED: ClassVar[frozenset[str]]
    _DEFAULTS: ClassVar[dict[str, Any]]

    # Parameters holding volumes (arrays in params, NIfTI paths in a
    # config), and the one whose NIfTI header provides the voxel size and
    # affine when it is given as a path.
    _VOLUME_KEYS: ClassVar[frozenset[str]]
    _REFERENCE_VOLUME_KEY: ClassVar[str]

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        register_solver(cls)

    def __init__(self, params: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
        name = type(self).__name__
        if params is not None and kwargs:
            raise TypeError(
                f"{name}: give the parameters either as one mapping or as keyword "
                "arguments, not both."
            )
        given = dict(params) if params is not None else dict(kwargs)
        named = given.pop(SOLVER_KEY, None)
        if named is not None and named != name:
            raise ValueError(f"{name}: the config names solver {named!r}.")
        merged = _merge_parameters(given, self._REQUIRED, self._DEFAULTS, name)
        # The config records the parameters as given, before the volumes
        # are loaded and before validation derives anything.
        self.config = self._build_config(given)
        self.result = None
        self.n_steps = None
        self.dt = None
        header_voxel_size = self._load_volumes(merged)
        self._resolve_voxel_size(merged, header_voxel_size)
        _validate_parameters(merged, name)
        for key in GAUSSIAN_SEED_POSITION_FRACTION:
            _validate_unit_interval(merged, key, name)
        self._validate_extra(merged)
        self.params = merged

    @classmethod
    def config_keys(cls) -> frozenset[str]:
        """The parameter names of the class: required ones and defaults."""
        return frozenset(cls._REQUIRED) | frozenset(cls._DEFAULTS)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        The class's default config, read from
        ``fisher_kpp_jax/configs/<ClassName>.json``.

        The file defines every parameter, required ones included, with the
        example volumes shipped with the repository as paths (an entry is
        null where no example volume is available yet); the optional
        entries equal the class defaults.

        Returns:
            The config entries with absolute volume paths (``read_config``).
        """
        return read_config(DEFAULT_CONFIG_DIR / f"{cls.__name__}.json", solver=cls)

    def _build_config(self, given: Mapping[str, Any]) -> dict[str, Any]:
        """
        The config of the given parameters: the "solver" entry, the given
        entries in their order, then the defaults not given. A volume given
        as an array is recorded as ``VOLUME_IN_MEMORY``; a path or a
        solver-specific entry is kept.
        """
        config: dict[str, Any] = {SOLVER_KEY: type(self).__name__}
        for key, value in given.items():
            config[key] = self._config_volume_entry(value) if key in self._VOLUME_KEYS else value
        for key, value in self._DEFAULTS.items():
            if key not in config:
                config[key] = value
        return config

    @staticmethod
    def _config_volume_entry(value: Any) -> Any:
        """The config record of a volume parameter value."""
        if value is None or isinstance(value, (str, Path)):
            return None if value is None else str(value)
        if isinstance(value, Mapping):
            return dict(value)
        return VOLUME_IN_MEMORY

    @classmethod
    def _resolve_config_volume(cls, key: str, value: Any, base_dir: Path, where: str) -> Any:
        """
        Resolve a volume entry of a config file for ``read_config``: a
        NIfTI path is made absolute (relative to base_dir, the config's
        directory), None stays None. Solvers with other entry formats
        override this.
        """
        if value is None:
            return None
        return resolve_config_path(value, base_dir, f"{where}: {key}")

    def _load_volume_entry(self, key: str, value: Any) -> tuple[NDArray, Any] | None:
        """
        Load one volume parameter given as a NIfTI path.

        Args:
            key: Parameter name.
            value: The parameter value as given.

        Returns:
            (array, image) with the float64 array and the nibabel image the
            path was loaded from, or None if the value is not a path (an
            array passes through unchanged). Solvers with other entry
            formats (a labelled segmentation) override this.
        """
        if isinstance(value, (str, Path)):
            if str(value) == VOLUME_IN_MEMORY:
                raise ValueError(
                    f"{type(self).__name__}: {key} was an in-memory array when the config "
                    "was written and cannot be reloaded; give a NIfTI path or an array."
                )
            image = nib.load(str(value))
            return np.asarray(image.get_fdata(), dtype=np.float64), image
        return None

    def _load_volumes(self, merged: dict[str, Any]) -> tuple[float, float, float] | None:
        """
        Replace the volume parameters given as paths by their arrays (in
        place) and set the affine.

        The reference volume is loaded first, so its header wins when it
        is a path; else the first other path's header; else the affine is
        a diagonal of the voxel size (set by ``_resolve_voxel_size``).

        Returns:
            The voxel size of the header the affine came from, or None
            when no volume was given as a path.
        """
        name = type(self).__name__
        header_voxel_size: tuple[float, float, float] | None = None
        affine: NDArray | None = None
        others = sorted(self._VOLUME_KEYS - {self._REFERENCE_VOLUME_KEY})
        for key in (self._REFERENCE_VOLUME_KEY, *others):
            value = merged.get(key)
            if value is None:
                if key in self._REQUIRED:
                    raise ValueError(f"{name}: {key} is None; give a NIfTI path or an array.")
                continue
            loaded = self._load_volume_entry(key, value)
            if loaded is None:
                continue
            merged[key], image = loaded
            if affine is None and image is not None:
                affine = np.asarray(image.affine, dtype=np.float64)
                header_voxel_size = tuple(float(v) for v in image.header.get_zooms()[:3])
        if affine is not None:
            self.affine = affine
        return header_voxel_size

    def _resolve_voxel_size(
        self, merged: dict[str, Any], header_voxel_size: tuple[float, float, float] | None
    ) -> None:
        """
        Set merged['voxel_size_mm'] (in place): the given value if any (a
        warning names a differing header value), else the reference
        volume's header value, else ``DEFAULT_VOXEL_SIZE_MM``; and the
        affine if no header provided one.
        """
        name = type(self).__name__
        given = merged["voxel_size_mm"]
        if given is None:
            voxel_size = header_voxel_size or DEFAULT_VOXEL_SIZE_MM
        else:
            values = np.asarray(given, dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(
                    f"{name}: voxel_size_mm must be three positive finite numbers, got {given!r}."
                )
            voxel_size = tuple(float(v) for v in values)
            if header_voxel_size is not None and not np.allclose(
                voxel_size, header_voxel_size, rtol=1e-4, atol=0
            ):
                logger.warning(
                    f"{name}: voxel_size_mm={voxel_size} differs from the NIfTI header's "
                    f"{header_voxel_size}; using the given value (the affine stays the "
                    "header's)."
                )
        merged["voxel_size_mm"] = voxel_size
        if not hasattr(self, "affine"):
            self.affine = np.diag((*voxel_size, 1.0))

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

    def solve(
        self,
        store_result: bool = False,
        outdir: str | Path | None = None,
        save_time_series: bool = False,
    ) -> Result:
        """
        Run the full pipeline.

        The Result carries the config, the resolved n_steps and dt, the
        wall time, the derived parameter values and the affine, and is
        kept as ``self.result``.

        Args:
            store_result: Whether to save the Result into outdir afterwards
                (``save``), a failed one included.
            outdir: Directory to save into; required with store_result.
            save_time_series: Whether the saved Result includes the
                recorded snapshots.

        Returns:
            The Result of the run.
        """
        if store_result and outdir is None:
            raise ValueError("store_result=True needs an outdir.")
        self.result = None
        self.n_steps = None
        self.dt = None
        start = time.perf_counter()
        try:
            result = self._run_pipeline()
        except Exception as exc:  # noqa: BLE001 - all failures become an error Result
            if self.params["verbose"]:
                logger.error(f"Solver failed: {exc}")
            result = Result(
                success=False,
                initial_state={},
                final_state={},
                final_time=0.0,
                final_stopping_quantity=0.0,
                stopping_criterion="error",
                error=str(exc),
            )
        result.wall_time_s = time.perf_counter() - start
        result.config = dict(self.config)
        result.n_steps = self.n_steps
        result.dt = self.dt
        result.derived = self._derived_parameters()
        result.affine = self.affine
        self.result = result
        if store_result:
            self.save(outdir, time_series=save_time_series)
        return result

    def save(
        self, outdir: str | Path, time_series: bool = False, overwrite: bool = False
    ) -> Path:
        """
        Save the Result of the last ``solve`` (``Result.save``).

        Args:
            outdir: Directory to write into, created if needed.
            time_series: Whether to write the recorded snapshots.
            overwrite: Whether an existing result in outdir may be
                overwritten.

        Returns:
            The directory written.
        """
        if self.result is None:
            raise RuntimeError(f"{type(self).__name__}: nothing to save, call solve() first.")
        return self.result.save(outdir, time_series=time_series, overwrite=overwrite)

    def _derived_parameters(self) -> dict[str, Any]:
        """The ``DERIVED_KEYS`` values the solver derived because the config
        does not give them."""
        return {
            key: self.params[key]
            for key in DERIVED_KEYS
            if key in self.params
            and self.params[key] is not None
            and self.config.get(key) is None
        }

    def resolve_time_stepping(self) -> tuple[int, float]:
        """
        The (n_steps, dt) a solve will use, before running one.

        Sets up the grid (downsampling the input fields, checking the
        seed) as ``solve`` does, since the stability estimate needs the
        grid spacing.

        Returns:
            (n_steps, dt), see ``_resolve_time_stepping``.
        """
        self._setup_grid()
        return self._resolve_time_stepping()

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
        Choose the number of simulation time steps and the step size dt,
        and store them as ``n_steps`` / ``dt``.

        Without a time-step parameter the solver's own stability formula
        (``_time_step_count``) decides. A given ``n_steps``, ``dt`` or
        ``steps_per_day`` (at most one; dt = 1 / steps_per_day; dt is
        translated by ``n_steps_from_dt``) is used as long as it gives at
        least the formula's step count; a coarser request is raised to the
        formula's (n_steps, dt) with a warning.

        Returns:
            (n_steps, dt).
        """
        params = self.params
        name = type(self).__name__
        stopping_time = float(params["stopping_time"])
        n_formula, dt_formula = self._time_step_count()
        n_formula, dt_formula = int(n_formula), float(dt_formula)
        requested: str | None = None
        if params["n_steps"] is not None:
            n_steps = int(params["n_steps"])
            requested = f"n_steps={n_steps}"
        elif params["dt"] is not None:
            n_steps = n_steps_from_dt(stopping_time, float(params["dt"]))
            requested = f"dt={float(params['dt']):g}"
        elif params["steps_per_day"] is not None:
            n_steps = n_steps_from_dt(stopping_time, 1.0 / float(params["steps_per_day"]))
            requested = f"steps_per_day={float(params['steps_per_day']):g}"
        if requested is None:
            n_steps, dt = n_formula, dt_formula
        elif n_steps < n_formula:
            logger.warning(
                f"{name}: {requested} gives {n_steps} steps (dt={stopping_time / n_steps:g}), "
                f"fewer than the stability estimate of {n_formula} (dt={dt_formula:g}); "
                "using the estimate."
            )
            n_steps, dt = n_formula, dt_formula
        else:
            dt = stopping_time / n_steps
        if params["verbose"]:
            logger.info(f"Number of simulation timesteps: {n_steps}")
        self.n_steps = n_steps
        self.dt = dt
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
        record_steps = np.empty(0, dtype=np.int64)
        snapshot_times: NDArray | None = None
        if params["snapshot_times"] is not None:
            record_steps, snapshot_times = _snapshot_steps(
                params["snapshot_times"], n_steps, dt, type(self).__name__
            )

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
                record_steps,
            )
            final_state_cropped = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in device_outputs["state"].items()
            }
            # An early stop leaves the frames scheduled after it unrecorded.
            n_recorded = int(device_outputs["n_recorded"])
            buffers = None
            if snapshot_times is not None:
                buffers = {
                    k: np.asarray(v[:n_recorded], dtype=np.float64)
                    for k, v in device_outputs["buffers"].items()
                }
                snapshot_times = snapshot_times[:n_recorded]

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
            snapshot_times=snapshot_times,
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
        # Solvers without a device guard: an explicit-Euler blow-up surfaces
        # as NaN/inf in the final state.
        error = guard_error
        if error is None and not all(np.isfinite(v).all() for v in final_state.values()):
            error = "non-finite final state (time step too large?)"
            if self.params["verbose"]:
                logger.error(f"Solver failed: {error}")

        time_series: dict[str, NDArray] | None = None
        if loop_results.buffers is not None:
            time_series = {}
            for key, frames in loop_results.buffers.items():
                upsampled = [
                    self._upsample_to(embed(frame, box, lowres_shape), original_shape)
                    for frame in frames
                ]
                time_series[key] = (
                    np.stack(upsampled) if upsampled else np.zeros((0, *original_shape))
                )

        if error is not None:
            stopping_criterion: Literal["time", "threshold", "error"] = "error"
        elif stop_kind == _STOP_THRESHOLD:
            stopping_criterion = "threshold"
        else:
            stopping_criterion = "time"

        return Result(
            success=error is None,
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
            snapshot_times=loop_results.snapshot_times,
            error=error,
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

        A given time step (``n_steps``, ``dt`` or ``steps_per_day``)
        replaces it unless it is coarser, see ``_resolve_time_stepping``.
        The three formulas are deliberately not unified -- do not merge or
        "fix" them.
        """

    def _check_seed(self) -> None:
        """
        Raise if the seed voxel lies outside the solver's simulated tissue.

        Runs once grid geometry and the seed voxel are known; the default
        accepts any seed (the two-compartment solver never checks).
        """
