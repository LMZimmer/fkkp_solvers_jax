"""Shared solve() pipeline for the Fisher-KPP forward solvers.

The pipeline reproduces the original TumorGrowthToolkit solvers step by step:
unpack/validate params -> downsample tissue -> grid geometry -> ad-hoc
stability-derived dt -> Gaussian seed -> bounding-box crop -> face
diffusivities -> explicit-Euler loop with mass/volume stopping and optional
time-series recording -> uncrop -> upsample -> Result.

Stopping quantity: ``stopping_mode="mass"`` (the default) compares
``stopping_volume`` against the total cell mass — voxel volume times the sum
of all cell density fields. This reproduces the original solvers' behavior
(modulo the FK_2c voxel-volume fix, see TwoCompartmentWithNutrientFKPPSolver):
it is an integrated density, not a physical volume, despite the threshold
parameter's name. ``stopping_mode="volume"`` compares against a true thresholded volume,
voxel_volume * count(summed cell density > density_threshold).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from .operators import FaceFields, crop, embed, tissue_bounding_box

logger = logging.getLogger(__name__)

CROP_MARGIN: int = 2  # voxels of margin left around the crop mask (original)


@dataclass(slots=True)
class Result:
    """Solver outcome.

    ``final_stopping_quantity`` is the last value of the quantity compared
    against ``stopping_volume``: total cell mass for ``stopping_mode="mass"``
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


class BaseFKPPSolver(ABC):
    """Template method: solve() owns the shared pipeline; subclasses fill hooks.

    Grid attributes (grid_shape, grid_spacing, seed_voxel) are populated by
    solve() after downsampling, before any hook that uses them is called.
    """

    grid_shape: tuple[int, int, int]
    grid_spacing: tuple[float, float, float]  # mm, post-downsampling
    seed_voxel: tuple[int, int, int]

    #: When True, solve() snapshots the state before each step and passes it
    #: to _post_step_checks (the original DTI solver copies its state every
    #: step for its shrinkage guard; the other solvers do not pay that cost).
    _requires_previous_state: ClassVar[bool] = False

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = self._validate_params(params)
        self._crop_box: tuple[slice, slice, slice] | None = None

    @property
    def voxel_volume(self) -> float:
        """Product of grid_spacing components (mm^3)."""
        dx, dy, dz = self.grid_spacing
        return dx * dy * dz

    # --- shared pipeline ---

    def solve(self) -> Result:
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
        stopping_volume = float(params["stopping_volume"])
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
            int(params["seed_x_fraction"] * nx),
            int(params["seed_y_fraction"] * ny),
            int(params["seed_z_fraction"] * nz),
        )
        self._check_seed_early()

        n_steps, dt = self._time_step_count()
        if verbose:
            logger.debug("number of simulation timesteps: %d", n_steps)

        state_full = self._initialize_state()
        initial_snapshot = {k: np.copy(v) for k, v in state_full.items()}

        box = tissue_bounding_box(self._crop_mask(), margin=CROP_MARGIN)
        self._crop_box = box
        state = {k: crop(v, box) for k, v in state_full.items()}

        diffusivity = self._build_diffusivity(state)
        self._check_seed_late()

        n_snapshots: int | None = params["n_time_series_snapshots"]
        record_steps = self._record_steps(n_steps, n_snapshots)
        time_series: dict[str, list[NDArray]] | None = (
            {k: [] for k in state} if n_snapshots is not None else None
        )

        final_time: float | None = None
        guard_error: str | None = None
        stopping_quantity = 0.0
        for t in range(n_steps):
            previous = (
                {k: np.copy(v) for k, v in state.items()}
                if self._requires_previous_state
                else state
            )
            state = self._step(state, diffusivity, dt)
            if self._diffusivity_needs_update():
                diffusivity = self._build_diffusivity(state)

            stopping_quantity = self._stopping_quantity(state)
            if stopping_quantity >= stopping_volume:
                final_time = t * dt
                break

            message = self._post_step_checks(state, previous)
            if message is not None:
                if verbose:
                    logger.debug("early loop exit at t=%s: %s", t * dt, message)
                guard_error = f"{message} (at simulation time {t * dt})"
                final_time = t * dt
                break

            if time_series is not None and t in record_steps:
                for key, field in state.items():
                    time_series[key].append(np.copy(field))

        if final_time is None:
            final_time = stopping_time

        final_low = {k: embed(v, box, low_shape) for k, v in state.items()}

        result_time_series: dict[str, NDArray] | None = None
        if time_series is not None:
            result_time_series = {
                key: np.array(
                    [
                        self._upsample_to(embed(frame, box, low_shape), full_shape)
                        for frame in frames
                    ]
                )
                for key, frames in time_series.items()
            }

        if guard_error is not None:
            stopping_criterion: Literal["time", "volume", "error"] = "error"
        elif stopping_quantity >= stopping_volume:
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
            final_stopping_quantity=float(stopping_quantity),
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
    def _validate_params(self, params: Mapping[str, Any]) -> Mapping[str, Any]: ...

    @abstractmethod
    def _prepare_fields(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Downsample the solver's tissue/diffusivity input fields and store
        them on self. Returns (low-res 3D grid shape, full-res 3D shape)."""

    @abstractmethod
    def _crop_mask(self) -> NDArray:
        """Boolean mask whose bounding box defines the simulation subdomain.
        Tissue occupancy for the tissue-based solvers, brainmask for the
        DTI solver."""

    @abstractmethod
    def _initialize_state(self) -> dict[str, NDArray]: ...

    @abstractmethod
    def _build_diffusivity(self, state: dict[str, NDArray]) -> dict[str, FaceFields]: ...

    @abstractmethod
    def _step(
        self,
        state: dict[str, NDArray],
        diffusivity: dict[str, FaceFields],
        dt: float,
    ) -> dict[str, NDArray]: ...

    def _stopping_quantity(self, state: dict[str, NDArray]) -> float:
        """Quantity compared against stopping_volume, dispatched on
        stopping_mode: total cell mass ("mass", the default — reproduces the
        original solvers' behavior modulo the FK_2c voxel-volume fix; an
        integrated density, not a physical volume) or a thresholded volume
        ("volume")."""
        if self.params["stopping_mode"] == "volume":
            return self._thresholded_volume(state)
        return self._cell_mass(state)

    def _thresholded_volume(self, state: dict[str, NDArray]) -> float:
        """voxel_volume * count(summed cell density > density_threshold) —
        a true physical volume, unlike the "mass" mode quantity."""
        density = self._cell_density_sum(state)
        return self.voxel_volume * float(
            np.count_nonzero(density > self.params["density_threshold"])
        )

    @abstractmethod
    def _cell_mass(self, state: dict[str, NDArray]) -> float:
        """Total cell mass: voxel_volume * sum of all cell density fields
        (nutrient fields are not cell densities and are excluded)."""

    @abstractmethod
    def _cell_density_sum(self, state: dict[str, NDArray]) -> NDArray:
        """Sum of the cell density fields, voxelwise (P + N for the necrotic
        solver; the nutrient field is never included)."""

    @abstractmethod
    def _time_step_count(self) -> tuple[int, float]:
        """(N_simulation_steps, dt) from each solver's own ad-hoc stability
        formula. The three formulas remain mutually inconsistent by heritage
        (see the notes at each implementation) and may be unified later."""

    def _diffusivity_needs_update(self) -> bool:
        return False  # TwoCompartmentWithNutrientFKPPSolver returns True

    def _post_step_checks(
        self, state: dict[str, NDArray], previous_state: dict[str, NDArray]
    ) -> str | None:
        return None  # AnisotropicFKPPSolver overrides

    def _check_seed_early(self) -> None:
        """Seed-position guard right after grid geometry is known (the
        original FK solver checks here, before the time-step count)."""

    def _check_seed_late(self) -> None:
        """Seed-position guard after cropping and diffusivity construction
        (the original DTI solver raises at this point in its pipeline)."""
