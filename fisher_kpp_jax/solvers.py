"""The JAX Fisher-KPP forward solvers: ``FKPPSolver``,
``TwoCompartmentWithNutrientFKPPSolver``, ``AnisotropicFKPPSolver`` and the
treatment-extended ``StuppFKPPSolver``.

Each solver's time step is a module-level function with a stable identity,
so the jitted time scan's cache persists across solves (see ``operators._run_time_loop``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from .base import (
    BaseFKPPSolver,
    _SharedConstants,
    _validate_event_times,
    _validate_nonnegative_scalar,
    _validate_nonnegative_sequence,
    _validate_positive_scalar,
    _validate_tissue_arrays,
    _validate_volume,
    n_steps_from_dt,  # noqa: F401 - re-exported
)
from .config import resolve_config_path
from .operators import (
    GAUSSIAN_SEED_DIFFUSION_TIME,
    GAUSSIAN_SEED_FLOOR,
    GAUSSIAN_SEED_MASS,
    SHRINKAGE_LIMIT,
    VANISHING_DENSITY_LIMIT,
    chemo_exposure,
    diffusion_term,
    elongate_tensor_along_principal_axis,
    face_average,
    logistic_growth,
    logistic_sigmoid,
    lq_log_kill,
    masked_face_average,
    shift_grid_by_one,
)

# Tissue-occupancy threshold defining the crop mask.
CROP_TISSUE_THRESHOLD: float = 0.5

# Steepness of the smooth descending switch on the nutrient level.
NECROSIS_SWITCH_STEEPNESS: float = 50.0

_AXES = ("x", "y", "z")

_COMMON_DEFAULTS: dict[str, Any] = {
    "diffusivity_ratio": 10.0,
    # None: from the NIfTI header of the reference volume when it is given
    # as a path, else 1 mm isotropic.
    "voxel_size_mm": None,
    "gaussian_seed_scale": 1.0,
    "gaussian_seed_diffusion_time": GAUSSIAN_SEED_DIFFUSION_TIME,
    "gaussian_seed_mass": GAUSSIAN_SEED_MASS,
    "gaussian_seed_floor": GAUSSIAN_SEED_FLOOR,
    "stopping_time": 100,
    "stopping_threshold": np.inf,
    "stopping_mode": "mass",
    "volume_threshold": None,  # only valid with stopping_mode="volume"
    "snapshot_times": None,  # days at which to record the state, see Result
    # The time step, at most one of the three (see
    # BaseFKPPSolver._resolve_time_stepping); None: the stability formula.
    "n_steps": None,
    "dt": None,  # days
    "steps_per_day": None,
    "verbose": False,
    "precision": "f32",
}

# The tissue probability maps of the WM/GM mixture solvers; the white
# matter map's NIfTI header provides the voxel size and affine.
_TISSUE_VOLUME_KEYS: frozenset[str] = frozenset({"gray_matter_pbmap", "white_matter_pbmap"})


class _SingleFieldSpecificConstants(TypedDict):
    """
    Solver-specific device constants of the single-field solvers
    (FKPPSolver and AnisotropicFKPPSolver), returned by their
    ``_build_device_constants``.

    Attributes:
        face_diffusivities: Face diffusivity arrays, keys 'fwd_x/y/z' and
            'bwd_x/y/z' (see ``diffusion_term``); constant in time.
        rho: Proliferation rate, 0-d scalar at the state dtype.
    """

    face_diffusivities: dict[str, jax.Array]
    rho: jax.Array


class _SingleFieldConstants(_SharedConstants, _SingleFieldSpecificConstants):
    """
    Flat device constants of the single-field solvers, merged by the base
    solver: the ``_SharedConstants`` keys plus the
    ``_SingleFieldSpecificConstants`` keys in one dict.
    """


class _TwoCompartmentSpecificConstants(TypedDict):
    """
    Solver-specific device constants of
    TwoCompartmentWithNutrientFKPPSolver, returned by its
    ``_build_device_constants``.

    Attributes:
        wm: White matter fraction field on the cropped grid.
        gm: Gray matter fraction field on the cropped grid.
        tissue_mask: Boolean mask of cells with enough tissue to carry
            flux; the tumor faces are rebuilt from it every step.
        nutrient_faces: Nutrient face diffusivities, keys 'fwd_x/y/z' and
            'bwd_x/y/z'; constant in time.
        white_matter_diffusivity: 0-d scalar at the state dtype, like all
            scalars below.
        diffusivity_ratio: White-to-gray-matter diffusivity ratio.
        rho: Proliferation rate.
        necrosis_rate: Proliferative-to-necrotic conversion rate.
        nutrient_consumption_rate: Nutrient consumption rate.
        nutrient_threshold: Nutrient level of the necrosis switch.
        max_tumor_occupancy: Occupancy above which faces carry no flux.
    """

    wm: jax.Array
    gm: jax.Array
    tissue_mask: jax.Array
    nutrient_faces: dict[str, jax.Array]
    white_matter_diffusivity: jax.Array
    diffusivity_ratio: jax.Array
    rho: jax.Array
    necrosis_rate: jax.Array
    nutrient_consumption_rate: jax.Array
    nutrient_threshold: jax.Array
    max_tumor_occupancy: jax.Array


class _TwoCompartmentConstants(_SharedConstants, _TwoCompartmentSpecificConstants):
    """
    Flat device constants of TwoCompartmentWithNutrientFKPPSolver, merged
    by the base solver: the ``_SharedConstants`` keys plus the
    ``_TwoCompartmentSpecificConstants`` keys in one dict.
    """


class _StuppSpecificConstants(_SingleFieldSpecificConstants):
    """
    Solver-specific device constants of StuppFKPPSolver, returned by its
    ``_build_device_constants``: the single-field keys plus the inputs of
    the three treatments, all always present. ``_stupp_step`` applies every
    treatment in every step; a treatment is switched off by its values
    (see the StuppFKPPSolver docstring), never by a missing key.

    Attributes:
        resection_time: Resection time in days, 0-d scalar at the state
            dtype.
        cavity: Boolean resection-cavity mask on the cropped grid.
        face_diffusivities_post: Post-resection face diffusivities, same
            keys as face_diffusivities; every face touching a cavity voxel
            is zero.
        chemo_times: Chemotherapy session times in days, 1-D at the state
            dtype.
        chemo_doses: Chemotherapy session doses in mg/m^2, 1-D at the
            state dtype, one per entry of chemo_times.
        chemo_kill_rate: Chemotherapy kill rate per unit dose (1/day per
            mg/m^2), 0-d scalar at the state dtype.
        chemo_decay_rate: Exponential decay rate of the drug
            concentration, 0-d scalar at the state dtype.
        rt_times: Radiotherapy fraction times in days, 1-D at the state
            dtype.
        rt_log_kill: Per-fraction linear-quadratic log kill E(x) on the
            cropped grid, at the state dtype.
    """

    resection_time: jax.Array
    cavity: jax.Array
    face_diffusivities_post: dict[str, jax.Array]
    chemo_times: jax.Array
    chemo_doses: jax.Array
    chemo_kill_rate: jax.Array
    chemo_decay_rate: jax.Array
    rt_times: jax.Array
    rt_log_kill: jax.Array


class _StuppConstants(_SharedConstants, _StuppSpecificConstants):
    """
    Flat device constants of StuppFKPPSolver, merged by the base solver:
    the ``_SharedConstants`` keys plus the ``_StuppSpecificConstants`` keys
    in one dict.
    """


def _mixture_face_fields(
    wm: jax.Array,
    gm: jax.Array,
    valid_mask: jax.Array,
    diffusivity: float | jax.Array,
    wm_to_gm_ratio: float | jax.Array,
) -> dict[str, jax.Array]:
    """
    Build white/gray-matter mixture face diffusivities (device).

    D = diffusivity * (wm_face + gm_face / wm_to_gm_ratio), faces masked by
    valid_mask. The 'bwd' fields are the edge-replicated shift of the 'fwd'
    fields (zero-flux boundary convention).

    Args:
        wm: White matter fraction field.
        gm: Gray matter fraction field.
        valid_mask: Boolean mask of valid cells; faces touching invalid
            cells carry zero diffusivity.
        diffusivity: White matter diffusivity.
        wm_to_gm_ratio: White-to-gray-matter diffusivity ratio.

    Returns:
        The face diffusivities: keys 'fwd_x/y/z' and 'bwd_x/y/z', each the
        shape of the input grid; see ``diffusion_term``.
    """
    faces: dict[str, jax.Array] = {}
    for axis, name in enumerate(_AXES):
        wm_face = masked_face_average(wm, valid_mask, axis)
        gm_face = masked_face_average(gm, valid_mask, axis)
        fwd = diffusivity * (wm_face + gm_face / wm_to_gm_ratio)
        faces[f"fwd_{name}"] = fwd
        faces[f"bwd_{name}"] = shift_grid_by_one(fwd, 1, axis=axis)
    return faces


# --- module-level device functions (stable identity so the jitted time
# --- loop's cache persists across solves; otherwise new parameters would
# --- trigger recompilation. see operators._run_time_scan) ---


def _single_field_step(
    state: dict[str, jax.Array],
    constants: _SingleFieldConstants,
    step_index: jax.Array,
) -> dict[str, jax.Array]:
    """
    Perform one explicit Euler step of the single-field solvers
    (FKPPSolver and AnisotropicFKPPSolver).

    The face diffusivities are constant in time and come from constants.

    Args:
        state: State dict with key 'cell_density'.
        constants: Device inputs, see ``_SingleFieldConstants``.
        step_index: Scan step index, unused (the step is autonomous).

    Returns:
        The stepped state.
    """
    del step_index
    dt = constants["dt"]
    rho = constants["rho"]
    u = state["cell_density"]
    diffusion = diffusion_term(
        u, constants["face_diffusivities"], constants["grid_spacing"]
    )
    delta_u = (diffusion + logistic_growth(u, rho)) * dt
    return {"cell_density": u + delta_u}


def _two_compartment_step(
    state: dict[str, jax.Array],
    constants: _TwoCompartmentConstants,
    step_index: jax.Array,
) -> dict[str, jax.Array]:
    """
    Perform one explicit Euler step of the proliferative/necrotic/nutrient
    system.

    The update order is deliberately sequential: the necrotic and nutrient
    updates see the already-updated proliferative field.

    Args:
        state: State dict with keys 'proliferative', 'necrotic' and
            'nutrient'.
        constants: Device inputs, see ``_TwoCompartmentConstants``.
        step_index: Scan step index, unused (the step is autonomous).

    Returns:
        The stepped state.
    """
    del step_index
    dt = constants["dt"]
    grid_spacing = constants["grid_spacing"]
    necrosis_rate = constants["necrosis_rate"]
    proliferative = state["proliferative"]
    necrotic = state["necrotic"]
    nutrient = state["nutrient"]

    # Per-step tumor-diffusivity rebuild from the carried state: on step t
    # the mask deliberately uses the post-step P and N of step t-1 (the
    # initial state on step 0).
    occupancy_valid = (proliferative + necrotic) <= constants[
        "max_tumor_occupancy"
    ]
    tumor_faces = _mixture_face_fields(
        constants["wm"],
        constants["gm"],
        jnp.logical_and(constants["tissue_mask"], occupancy_valid),
        constants["white_matter_diffusivity"],
        constants["diffusivity_ratio"],
    )

    # Smooth descending switch on the nutrient level.
    switch = logistic_sigmoid(
        -NECROSIS_SWITCH_STEEPNESS * (nutrient - constants["nutrient_threshold"])
    )

    tumor_diffusion = diffusion_term(proliferative, tumor_faces, grid_spacing)
    delta_proliferative = (
        tumor_diffusion
        + constants["rho"]
        * (nutrient * proliferative)
        * (1 - proliferative - necrotic)
        - necrosis_rate * proliferative * switch
    ) * dt
    proliferative = proliferative + delta_proliferative

    delta_necrotic = necrosis_rate * proliferative * switch * dt
    necrotic = necrotic + delta_necrotic

    nutrient_diffusion = diffusion_term(
        nutrient, constants["nutrient_faces"], grid_spacing
    )
    delta_nutrient = (
        nutrient_diffusion
        - constants["nutrient_consumption_rate"] * nutrient * proliferative
    ) * dt
    nutrient = nutrient + delta_nutrient

    return {
        "proliferative": proliferative,
        "necrotic": necrotic,
        "nutrient": nutrient,
    }


def _stupp_step(
    state: dict[str, jax.Array],
    constants: _StuppConstants,
    step_index: jax.Array,
) -> dict[str, jax.Array]:
    """
    Perform one explicit Euler step of the treatment-extended isotropic
    model (StuppFKPPSolver), followed by the discrete treatment events
    of the step.

    Both are computed as products (never accumulated), so the step
    intervals (t0, t1] partition the horizon exactly at the state dtype.
    In-step operation order:

      1. Euler update at the pre-step state,
         du/dt = div(D grad u) + rho u (1 - u),
         with D = the post-resection faces once t1 >= resection_time,
         else the pre-resection faces;
      2. chemotherapy impulse u <- u exp(-chemo_kill_rate E_ct), with
         E_ct = int_{t0}^{t1} C dt the exact drug exposure of the step
         (``chemo_exposure``), so the chemotherapy kill is independent of
         the step size;
      3. radiotherapy impulse u <- u exp(-E(x) n_hits), n_hits = number of
         rt_times in (t0, t1] (exact impulse map, not part of the Euler
         right-hand side);
      4. resection projection u <- 0 inside the cavity, for every step
         with t1 >= resection_time (idempotent).

    The two impulses are pointwise multiplications and commute; the
    projection is applied last so that the cavity is empty at the end of
    every post-resection step.

    Every treatment term is evaluated in every step. With neutral
    treatment values (an all-False cavity, a zero chemotherapy kill rate
    or zero doses or no session, a zero radiotherapy log kill) the update
    equals ``_single_field_step`` up to floating-point rounding.

    Args:
        state: State dict with key 'cell_density'.
        constants: Device inputs, see ``_StuppConstants``.
        step_index: Scan step index, 0-d int32.

    Returns:
        The stepped state.
    """
    dt = constants["dt"]
    rho = constants["rho"]
    u = state["cell_density"]
    t0 = step_index.astype(u.dtype) * dt
    t1 = (step_index + 1).astype(u.dtype) * dt

    post = t1 >= constants["resection_time"]
    faces_pre = constants["face_diffusivities"]
    faces_post = constants["face_diffusivities_post"]
    faces = {key: jnp.where(post, faces_post[key], faces_pre[key]) for key in faces_pre}

    reaction = logistic_growth(u, rho)
    diffusion = diffusion_term(u, faces, constants["grid_spacing"])
    delta_u = (diffusion + reaction) * dt
    u = u + delta_u

    exposure = chemo_exposure(
        t0,
        t1,
        constants["chemo_times"],
        constants["chemo_doses"],
        constants["chemo_decay_rate"],
    )
    u = u * jnp.exp(-constants["chemo_kill_rate"] * exposure)
    rt_times = constants["rt_times"]
    n_hits = jnp.sum(jnp.logical_and(rt_times > t0, rt_times <= t1)).astype(u.dtype)
    u = u * jnp.exp(-constants["rt_log_kill"] * n_hits)
    u = jnp.where(jnp.logical_and(post, constants["cavity"]), 0, u)
    return {"cell_density": u}


def _mass_single(
    state: dict[str, jax.Array],
    constants: _SharedConstants,
) -> jax.Array:
    """Integrated cell density of the single-field solvers, summed in f64."""
    return constants["voxel_volume"] * jnp.sum(
        state["cell_density"], dtype=jnp.float64
    )


def _mass_two_compartment(
    state: dict[str, jax.Array],
    constants: _SharedConstants,
) -> jax.Array:
    """
    Two-compartment integrated cell density, f64:
    voxel_volume * (sum(P) + sum(N)); the voxel-volume factor deliberately
    multiplies BOTH terms.
    """
    return constants["voxel_volume"] * (
        jnp.sum(state["proliferative"], dtype=jnp.float64)
        + jnp.sum(state["necrotic"], dtype=jnp.float64)
    )


def _volume_single(
    state: dict[str, jax.Array],
    constants: _SharedConstants,
) -> jax.Array:
    """
    Thresholded volume of the single-field solvers; the volume threshold
    is a 0-d scalar at the state dtype.
    """
    threshold = constants["volume_threshold"]
    count = jnp.count_nonzero(state["cell_density"] > threshold)
    return constants["voxel_volume"] * count.astype(jnp.float64)


def _volume_two_compartment(
    state: dict[str, jax.Array],
    constants: _SharedConstants,
) -> jax.Array:
    """Thresholded volume of P + N (the nutrient field is never included)."""
    threshold = constants["volume_threshold"]
    density = state["proliferative"] + state["necrotic"]
    count = jnp.count_nonzero(density > threshold)
    return constants["voxel_volume"] * count.astype(jnp.float64)


def _dti_guard(
    new_state: dict[str, jax.Array],
    previous_state: dict[str, jax.Array],
    constants: _SharedConstants,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Evaluate the shrinkage/vanishing guards of the anisotropic solver.

    Shrinkage takes precedence; the thresholds are SHRINKAGE_LIMIT and
    VANISHING_DENSITY_LIMIT, and the sums run in float64 regardless of the
    state dtype. A firing guard is reported as Result(success=False,
    stopping_criterion="error") with final_time at the actual exit step.
    """
    voxel_volume = constants["voxel_volume"]
    new_sum = jnp.sum(new_state["cell_density"], dtype=jnp.float64)
    prev_sum = jnp.sum(previous_state["cell_density"], dtype=jnp.float64)
    total_change = new_sum - prev_sum
    integrated_density = voxel_volume * new_sum
    code = jnp.select(
        [
            total_change < -SHRINKAGE_LIMIT,
            integrated_density < VANISHING_DENSITY_LIMIT,
        ],
        [1, 2],
        default=0,
    ).astype(jnp.int32)
    return code, total_change, integrated_density


class FKPPSolver(BaseFKPPSolver):
    """
    Isotropic Fisher-KPP solver on WM/GM tissue maps.

    State key: 'cell_density'. Diffusivity is a WM/GM mixture,
    D = white_matter_diffusivity * (wm_face + gm_face / diffusivity_ratio),
    with faces masked by min_tissue_fraction, built once on the device.
    """

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "white_matter_diffusivity",
            "rho",
            "gray_matter_pbmap",
            "white_matter_pbmap",
            "gaussian_seed_x_fraction",
            "gaussian_seed_y_fraction",
            "gaussian_seed_z_fraction",
            "resolution_factor",
        }
    )
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        **_COMMON_DEFAULTS,
        # Cells with wm + gm below this carry no flux (CSF/background).
        "min_tissue_fraction": 0.1,
    }
    _VOLUME_KEYS: ClassVar[frozenset[str]] = _TISSUE_VOLUME_KEYS
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "white_matter_pbmap"

    # static methods allows passing of stable module level functions as attributes
    _step_func = staticmethod(_single_field_step)
    _mass_func = staticmethod(_mass_single)
    _volume_func = staticmethod(_volume_single)

    _gm_lowres: NDArray
    _wm_lowres: NDArray

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        _validate_tissue_arrays(params, type(self).__name__)

    def _prepare_input_fields(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        factor = self.params["resolution_factor"]
        self._gm_lowres = self._downsample(self.params["gray_matter_pbmap"], factor)
        self._wm_lowres = self._downsample(self.params["white_matter_pbmap"], factor)
        return self._gm_lowres.shape, self.params["gray_matter_pbmap"].shape

    def _check_seed(self) -> None:
        i, j, k = self.seed_voxel
        if self._gm_lowres[i, j, k] == 0 and self._wm_lowres[i, j, k] == 0:
            raise ValueError("Initial tumor position is outside the brain matter.")

    def _crop_mask(self) -> NDArray:
        return (self._gm_lowres + self._wm_lowres) >= CROP_TISSUE_THRESHOLD

    def _initialize_state(self) -> dict[str, jax.Array]:
        return {"cell_density": self._gaussian_seed()}

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> _SingleFieldSpecificConstants:
        gm_host = self._gm_lowres[box]
        wm_host = self._wm_lowres[box]

        tissue_mask_host = (wm_host + gm_host) >= float(
            self.params["min_tissue_fraction"]
        )
        gm = jnp.asarray(gm_host, dtype=self._dtype)
        wm = jnp.asarray(wm_host, dtype=self._dtype)
        faces = _mixture_face_fields(
            wm,
            gm,
            jnp.asarray(tissue_mask_host),
            float(self.params["white_matter_diffusivity"]),
            float(self.params["diffusivity_ratio"]),
        )
        return {
            "face_diffusivities": faces,
            "rho": self._dynamic_scalar(self.params["rho"]),
        }

    def _time_step_count(self) -> tuple[int, float]:
        stopping_time = self.params["stopping_time"]
        diffusivity_wm = self.params["white_matter_diffusivity"]
        rho = self.params["rho"]
        dx, dy, dz = self.grid_spacing
        # np.power kept deliberately: CPython's ** is not bit-identical to it.
        n_timesteps = max(
            stopping_time * diffusivity_wm / np.power(min(dx, dy, dz), 2) * 8 + 100,
            stopping_time * rho * 1.1,
        )
        dt = stopping_time / n_timesteps
        return int(np.ceil(n_timesteps)), dt


class TwoCompartmentWithNutrientFKPPSolver(BaseFKPPSolver):
    """
    Two-compartment solver for the proliferative/necrotic/nutrient system.

    State keys: 'proliferative', 'necrotic', 'nutrient'. Tumor diffusivity
    faces are additionally masked where proliferative + necrotic exceeds
    max_tumor_occupancy and are rebuilt every step from the carried state
    (see ``_two_compartment_step`` for the update-order semantics). The
    nutrient diffuses with nutrient_diffusivity, masked by tissue only,
    built once.

    The "mass" stopping quantity deliberately applies the voxel-volume
    factor to the necrotic term as well -- see ``_mass_two_compartment``.
    """

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "white_matter_diffusivity",
            "rho",
            "necrosis_rate",
            "nutrient_threshold",
            "nutrient_diffusivity",
            "nutrient_consumption_rate",
            "gray_matter_pbmap",
            "white_matter_pbmap",
            "gaussian_seed_x_fraction",
            "gaussian_seed_y_fraction",
            "gaussian_seed_z_fraction",
            "resolution_factor",
        }
    )
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        **_COMMON_DEFAULTS,
        "min_tissue_fraction": 0.1,
        "max_tumor_occupancy": 0.9,
        "nt_multiplier": 8,
    }
    _VOLUME_KEYS: ClassVar[frozenset[str]] = _TISSUE_VOLUME_KEYS
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "white_matter_pbmap"

    _step_func = staticmethod(_two_compartment_step)
    _mass_func = staticmethod(_mass_two_compartment)
    _volume_func = staticmethod(_volume_two_compartment)

    _gm_lowres: NDArray
    _wm_lowres: NDArray

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        _validate_tissue_arrays(params, type(self).__name__)

    def _prepare_input_fields(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        factor = self.params["resolution_factor"]
        self._gm_lowres = self._downsample(self.params["gray_matter_pbmap"], factor)
        self._wm_lowres = self._downsample(self.params["white_matter_pbmap"], factor)
        return self._gm_lowres.shape, self.params["gray_matter_pbmap"].shape

    def _crop_mask(self) -> NDArray:
        return (self._gm_lowres + self._wm_lowres) >= CROP_TISSUE_THRESHOLD

    def _initialize_state(self) -> dict[str, jax.Array]:
        proliferative = self._gaussian_seed()
        necrotic = jnp.zeros(proliferative.shape, dtype=self._dtype)
        nutrient = jnp.ones(proliferative.shape, dtype=self._dtype)
        # Remove CSF from the nutrient field (host-side float64 tissue mask,
        # identical for both precisions).
        tissue_host = (self._wm_lowres + self._gm_lowres) >= float(
            self.params["min_tissue_fraction"]
        )
        nutrient = jnp.where(jnp.asarray(tissue_host), nutrient, 0)
        return {
            "proliferative": proliferative,
            "necrotic": necrotic,
            "nutrient": nutrient,
        }

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> _TwoCompartmentSpecificConstants:
        gm_host = self._gm_lowres[box]
        wm_host = self._wm_lowres[box]
        # Time-constant validity mask, computed host-side in float64 so it
        # is identical for both precisions.
        tissue_mask_host = (wm_host + gm_host) >= float(
            self.params["min_tissue_fraction"]
        )
        gm = jnp.asarray(gm_host, dtype=self._dtype)
        wm = jnp.asarray(wm_host, dtype=self._dtype)
        tissue_mask = jnp.asarray(tissue_mask_host)

        # Nutrient faces are built once (constant in time; ratio 1 means
        # gray matter conducts nutrient like white matter).
        # The tumor faces are rebuilt every step inside the step function.
        nutrient_faces = _mixture_face_fields(
            wm, gm, tissue_mask, float(self.params["nutrient_diffusivity"]), 1
        )
        scalar = self._dynamic_scalar
        params = self.params
        return {
            "wm": wm,
            "gm": gm,
            "tissue_mask": tissue_mask,
            "nutrient_faces": nutrient_faces,
            "white_matter_diffusivity": scalar(params["white_matter_diffusivity"]),
            "diffusivity_ratio": scalar(params["diffusivity_ratio"]),
            "rho": scalar(params["rho"]),
            "necrosis_rate": scalar(params["necrosis_rate"]),
            "nutrient_consumption_rate": scalar(params["nutrient_consumption_rate"]),
            "nutrient_threshold": scalar(params["nutrient_threshold"]),
            "max_tumor_occupancy": scalar(params["max_tumor_occupancy"]),
        }

    def _time_step_count(self) -> tuple[int, float]:
        stopping_time = self.params["stopping_time"]
        diffusivity_wm = self.params["white_matter_diffusivity"]
        diffusivity_nutrient = self.params["nutrient_diffusivity"]
        rho = self.params["rho"]
        dx, dy, dz = self.grid_spacing
        n_timesteps = max(
            stopping_time
            * max(diffusivity_wm, diffusivity_nutrient)
            / np.power(min(dx, dy, dz), 2)
            * self.params["nt_multiplier"]
            + 300,
            # Reaction-rate guard: without it, dt can violate the ~1/rho
            # explicit-Euler reaction bound for large rho.
            stopping_time * rho * 1.1,
        )
        dt = stopping_time / n_timesteps
        return int(np.ceil(n_timesteps)), dt


class AnisotropicFKPPSolver(BaseFKPPSolver):
    """
    Anisotropic solver with axis-wise diffusivity from the DTI tensor
    diagonal.

    State key: 'cell_density'. The per-axis diffusivity field (shape
    (Nx, Ny, Nz, 3)) is derived from the tensor diagonals on the host; the
    crop mask and the seed guard come from a brain mask thresholded on that
    field. The shrinkage/vanishing guards run on the device inside the scan,
    reading the previous state from the carry (no explicit per-step copies).
    """

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "diffusivity",
            "rho",
            "diffusion_tensors",
            "gaussian_seed_x_fraction",
            "gaussian_seed_y_fraction",
            "gaussian_seed_z_fraction",
            "resolution_factor",
        }
    )
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        **_COMMON_DEFAULTS,
        "ellipsoid_scaling": 1.0,
        "normalization_std": None,
        "tensor_exponent": 1,
        "tensor_linear_term": 0,
        "uniform_gray_matter": False,
        "gray_matter_pbmap": None,
        "white_matter_pbmap": None,
        "diffusivity_upper_limit": 2,
        "diffusivity_lower_limit": 0,
    }
    # The tensor field is a 5D NIfTI, (Nx, Ny, Nz, 3, 3); the tissue maps
    # are only needed with uniform_gray_matter.
    _VOLUME_KEYS: ClassVar[frozenset[str]] = _TISSUE_VOLUME_KEYS | {"diffusion_tensors"}
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "diffusion_tensors"
    _step_func = staticmethod(_single_field_step)
    _mass_func = staticmethod(_mass_single)
    _volume_func = staticmethod(_volume_single)
    # DTI shrinkage/vanishing guards -- semantics at ``_dti_guard``.
    _guard_func = staticmethod(_dti_guard)

    _axial_lowres: NDArray
    _axial_original_max: float
    _brainmask_lowres: NDArray

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        tensors = params["diffusion_tensors"]
        if not isinstance(tensors, np.ndarray):
            raise ValueError(
                "AnisotropicFKPPSolver: diffusion_tensors must be a numpy array."
            )
        if tensors.ndim != 5 or tensors.shape[-2:] != (3, 3):
            raise ValueError(
                "AnisotropicFKPPSolver: diffusion_tensors must have shape "
                f"(Nx, Ny, Nz, 3, 3), got {tensors.shape}."
            )
        if params["uniform_gray_matter"] and (
            params["gray_matter_pbmap"] is None or params["white_matter_pbmap"] is None
        ):
            raise KeyError(
                "AnisotropicFKPPSolver: uniform_gray_matter=True requires "
                "gray_matter_pbmap and white_matter_pbmap."
            )

    def _axial_diffusivity_from_tensor(
        self,
        tensor: NDArray,
        wm: NDArray | None,
        gm: NDArray | None,
        diffusivity_ratio: float | None,
    ) -> NDArray:
        """
        Compute the per-axis diffusivity field from the tensor diagonals
        (host-side NumPy).

        The operation order is protected numerics -- in particular the
        sequential in-place mean/std normalization, where the std is
        computed on the already mean-shifted field.

        Args:
            tensor: Diffusion tensors, shape (Nx, Ny, Nz, 3, 3).
            wm: White matter fraction field; None unless
                uniform_gray_matter is set.
            gm: Gray matter fraction field; None unless uniform_gray_matter
                is set.
            diffusivity_ratio: White-to-gray-matter diffusivity ratio; None
                unless uniform_gray_matter is set.

        Returns:
            The per-axis diffusivity field, shape (Nx, Ny, Nz, 3). All other
            inputs come from self.params.
        """
        exponent = self.params["tensor_exponent"]
        linear_term = self.params["tensor_linear_term"]
        normalization_std = self.params["normalization_std"]
        upper_limit = self.params["diffusivity_upper_limit"]
        lower_limit = self.params["diffusivity_lower_limit"]
        axial = np.zeros(tensor.shape[:4])

        axial[:, :, :, 0] = tensor[:, :, :, 0, 0]
        axial[:, :, :, 1] = tensor[:, :, :, 1, 1]
        axial[:, :, :, 2] = tensor[:, :, :, 2, 2]

        axial[axial < 0] = 0

        brainmask_original = np.max(axial, axis=-1) > 0

        if wm is not None:
            normalization_mask = wm > 0
        else:
            normalization_mask = brainmask_original

        if normalization_std is not None:
            axial[brainmask_original] -= np.mean(axial[normalization_mask])
            axial[brainmask_original] /= np.std(axial[normalization_mask])
            axial[brainmask_original] *= normalization_std
            axial[brainmask_original] += 1
        else:
            axial[brainmask_original] /= np.mean(axial[normalization_mask])

        if not (wm is None or gm is None or diffusivity_ratio is None):
            if self.params["verbose"]:
                logger.info("Setting gm to uniform diffusivity and wm to DTI.")
            csf_mask = np.logical_and(wm <= 0, gm <= 0)
            axial[csf_mask] = 0
            gm_threshold = 1.0 / diffusivity_ratio
            axial[gm > 0] = gm_threshold  # fix gray matter
            border_mask = binary_dilation(csf_mask, iterations=1)
            axial[border_mask] = 0
            # clip wm to lowest gm
            axial[
                np.logical_and(
                    np.repeat((wm > 0)[..., np.newaxis], repeats=3, axis=-1),
                    axial < gm_threshold,
                )
            ] = gm_threshold

        axial[axial < 0] = 0
        axial = axial**exponent + linear_term * axial

        axial[axial > upper_limit] = upper_limit
        axial[axial < 0] = 0
        axial[
            np.logical_and(
                np.repeat((brainmask_original > 0)[..., np.newaxis], repeats=3, axis=-1),
                axial < lower_limit,
            )
        ] = lower_limit

        return axial

    def _prepare_input_fields(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        params = self.params
        scaling = params["ellipsoid_scaling"]
        if params["verbose"]:
            logger.info(f"Ellipsoid scaling: {scaling}")
        if scaling == 1:
            tensors = params["diffusion_tensors"]
        else:
            tensors = elongate_tensor_along_principal_axis(
                params["diffusion_tensors"], scaling
            )

        uniform = params["uniform_gray_matter"]
        axial_original = self._axial_diffusivity_from_tensor(
            tensors,
            wm=params["white_matter_pbmap"] if uniform else None,
            gm=params["gray_matter_pbmap"] if uniform else None,
            diffusivity_ratio=params["diffusivity_ratio"] if uniform else None,
        )

        factor = params["resolution_factor"]
        axial_lowres = self._downsample(axial_original, [factor, factor, factor, 1])
        axial_lowres[axial_lowres <= 0] = 0
        self._axial_lowres = axial_lowres
        # The stability formula deliberately uses the max of the
        # original-resolution field, before downsampling.
        self._axial_original_max = np.max(axial_original)
        self._brainmask_lowres = np.max(axial_lowres, axis=-1) > 0.00001
        return axial_lowres.shape[:3], axial_original.shape[:3]

    def _crop_mask(self) -> NDArray:
        return self._brainmask_lowres

    def _check_seed(self) -> None:
        if not self._brainmask_lowres[self.seed_voxel]:
            raise ValueError("Initial tumor position is outside the brain mask.")

    def _initialize_state(self) -> dict[str, jax.Array]:
        cell_density = self._gaussian_seed()
        if self.params["verbose"]:
            logger.info(
                f"Initial state shape: {cell_density.shape}, volume of initial "
                f"tumor: {float(jnp.sum(cell_density, dtype=jnp.float64))}"
            )
        return {"cell_density": cell_density}

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> _SingleFieldSpecificConstants:
        axial = jnp.asarray(self._axial_lowres[box], dtype=self._dtype)
        diffusivity = float(self.params["diffusivity"])
        faces: dict[str, jax.Array] = {}
        for axis, name in enumerate(_AXES):
            face = face_average(axial[:, :, :, axis], axis)
            faces[f"fwd_{name}"] = diffusivity * face
            faces[f"bwd_{name}"] = diffusivity * shift_grid_by_one(face, 1, axis=axis)
        return {
            "face_diffusivities": faces,
            "rho": self._dynamic_scalar(self.params["rho"]),
        }

    def _time_step_count(self) -> tuple[int, float]:
        stopping_time = self.params["stopping_time"]
        diffusivity = self.params["diffusivity"]
        rho = self.params["rho"]
        dx, dy, dz = self.grid_spacing
        # Scales with the max of the original-resolution axial diffusivity
        # field, which is over-conservative (the downsampled field's max is
        # <= it); deliberate -- do not change.
        n_timesteps = max(
            stopping_time
            * diffusivity
            * self._axial_original_max
            / np.power(min(dx, dy, dz), 2)
            * 8
            + 100,
            stopping_time * rho * 1.1,
        )
        dt = stopping_time / n_timesteps
        return int(np.ceil(n_timesteps)), dt


# Treatment parameters of StuppFKPPSolver, all required but
# rt_alpha_beta_ratio, which has a default (see the class docstring for the
# values that switch a treatment off).
_STUPP_TREATMENT_KEYS: frozenset[str] = frozenset(
    {
        "resection_time",
        "resection_cavity",
        "chemo_times",
        "chemo_doses",  # mg/m^2 per session, one per chemo_times entry
        "chemo_kill_rate",  # 1/day per mg/m^2
        "chemo_decay_rate",
        "rt_times",
        "rt_dose",  # TOTAL dose over all fractions, 3D array in Gy
        "rt_alpha",  # 1/Gy
        "rt_alpha_beta_ratio",  # Gy, the linear-quadratic alpha/beta ratio
    }
)

class StuppFKPPSolver(BaseFKPPSolver):
    """
    Isotropic Fisher-KPP solver on WM/GM tissue maps, extended by the
    treatment effects of a Stupp protocol: surgical resection,
    chemotherapy (CT) and radiotherapy (RT).

    State key: 'cell_density' (u in [0, 1]); grid in mm, time in days
    with the seed at t = 0. The horizon is given relative to the surgery:
    the run ends at resection_time + time_after_resection. The shared
    ``stopping_time`` parameter is not accepted; ``params['stopping_time']``
    holds the derived sum after construction.

    In a config the treatment volumes are NIfTI paths like the tissue
    maps: ``rt_dose`` (Gy, TOTAL over all fractions) directly, and
    ``resection_cavity`` as ``{"segmentation": <NIfTI path>, "label":
    <int>}``, the cavity being the voxels carrying that label (values
    rounded to the nearest integer first). Both may also be given as
    arrays.

    Every treatment parameter is required (rt_alpha_beta_ratio excepted,
    which defaults to 10 Gy); a treatment is switched off by its values,
    not by omitting it: an all-False resection_cavity leaves the dynamics
    untouched (the post-resection faces then equal the pre-resection
    ones), an empty chemo_times, chemo_kill_rate = 0 or all-zero
    chemo_doses removes the chemotherapy kill (chemo_decay_rate must stay
    > 0), and a zero rt_dose (or rt_alpha = 0, which zeroes the derived
    rt_beta with it) makes the radiotherapy impulse the identity. With all
    three neutral the solver reproduces ``FKPPSolver`` up to
    floating-point rounding (the treatment terms are still evaluated, so
    the compiled arithmetic is not identical).

    Continuous model (explicit Euler at the pre-step state; growth and
    diffusion only)::

        du/dt = div(D grad u) + rho u (1 - u)

    Chemotherapy acts through the drug concentration C(t), in which each
    session j at chemo_times[j] deposits its dose chemo_doses[j] (mg/m^2)
    that decays exponentially::

        C(t) = sum_j chemo_doses[j] [t >= chemo_times[j]]
                     exp(-chemo_decay_rate (t - chemo_times[j]))

    Discrete events, applied after the Euler update of the step whose
    interval (t0, t1] contains them, in this order (see ``_stupp_step``):

      1. CT impulse: u <- u exp(-chemo_kill_rate E_ct) with
         E_ct = int_{t0}^{t1} C dt the exact exposure of the step
         (``chemo_exposure``). chemo_kill_rate is the kill rate per unit
         dose, in 1/day per mg/m^2, so the log kill of one session of
         dose d over its whole decay is chemo_kill_rate d / chemo_decay_rate.
         The exposure is exact for any step size, so a fitted kill rate
         is transferable across time steps.
      2. RT impulse: u <- u exp(-E(x) n_hits) with the linear-quadratic
         log kill E(x) = rt_alpha d(x) + rt_beta d(x)^2 and n_hits the
         number of rt_times in (t0, t1]. The parameters are rt_alpha
         (1/Gy) and the alpha/beta ratio rt_alpha_beta_ratio (Gy);
         rt_beta = rt_alpha / rt_alpha_beta_ratio (1/Gy^2) is computed on
         the host where E(x) is built and is neither a parameter nor a
         config entry (as diffusivity_ratio stands in for a gray-matter
         diffusivity). Per-fraction dose convention: rt_dose holds the
         TOTAL dose over all fractions, so d(x) = rt_dose / len(rt_times)
         (computed once on the host).
      3. Resection: u <- 0 inside resection_cavity for every step with
         t1 >= resection_time, and from the same step on the face
         diffusivities switch to a post-resection set in which every face
         touching a cavity voxel is zero (zero-flux Neumann on the cavity
         boundary).
    """

    # The treatment parameters, which FKPPSolver does not have (a params
    # dict without them and time_after_resection, plus stopping_time, is
    # the untreated FKPPSolver run).
    TREATMENT_KEYS: ClassVar[frozenset[str]] = _STUPP_TREATMENT_KEYS
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        **{key: value for key, value in _COMMON_DEFAULTS.items() if key != "stopping_time"},
        # Cells with wm + gm below this carry no flux (CSF/background).
        "min_tissue_fraction": 0.1,
        # Linear-quadratic alpha/beta ratio in Gy; rt_beta = rt_alpha / it
        # is derived on the host. The one treatment parameter with a default.
        "rt_alpha_beta_ratio": 10.0,
    }
    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "white_matter_diffusivity",
            "rho",
            "gray_matter_pbmap",
            "white_matter_pbmap",
            "gaussian_seed_x_fraction",
            "gaussian_seed_y_fraction",
            "gaussian_seed_z_fraction",
            "resolution_factor",
            "time_after_resection",  # days; the horizon is resection_time + it
        }
        | (TREATMENT_KEYS - frozenset(_DEFAULTS))
    )
    _VOLUME_KEYS: ClassVar[frozenset[str]] = _TISSUE_VOLUME_KEYS | {"rt_dose", "resection_cavity"}
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "white_matter_pbmap"

    # static methods allows passing of stable module level functions as attributes
    _step_func = staticmethod(_stupp_step)
    _mass_func = staticmethod(_mass_single)
    _volume_func = staticmethod(_volume_single)

    _gm_lowres: NDArray
    _wm_lowres: NDArray
    _cavity_lowres: NDArray
    _rt_dose_lowres: NDArray

    @classmethod
    def _resolve_config_volume(cls, key: str, value: Any, base_dir: Path, where: str) -> Any:
        """The cavity entry is ``{"segmentation": <NIfTI path>, "label":
        <int>}`` with the path made absolute; the other volumes are paths."""
        if key != "resection_cavity" or value is None:
            return super()._resolve_config_volume(key, value, base_dir, where)
        if not isinstance(value, Mapping) or set(value) != {"segmentation", "label"}:
            raise ValueError(
                f"{where}: resection_cavity must be an object "
                '{"segmentation": <NIfTI path>, "label": <int>}.'
            )
        return {
            "segmentation": resolve_config_path(
                value["segmentation"], base_dir, f"{where}: resection_cavity segmentation"
            ),
            "label": int(value["label"]),
        }

    def _load_volume_entry(self, key: str, value: Any) -> tuple[NDArray, Any] | None:
        """The cavity given as ``{"segmentation": <NIfTI path>, "label":
        <int>}`` is the boolean mask of the label in the segmentation."""
        if key != "resection_cavity" or not isinstance(value, Mapping):
            return super()._load_volume_entry(key, value)
        if set(value) != {"segmentation", "label"}:
            raise ValueError(
                f"{type(self).__name__}: resection_cavity must be an array or "
                '{"segmentation": <NIfTI path>, "label": <int>}.'
            )
        image = nib.load(str(value["segmentation"]))
        segmentation = np.rint(np.asarray(image.get_fdata(), dtype=np.float64)).astype(np.int64)
        return segmentation == int(value["label"]), image

    def _validate_extra(self, params: dict[str, Any]) -> None:
        name = type(self).__name__
        _validate_tissue_arrays(params, name)
        shape = params["gray_matter_pbmap"].shape

        resection_time = _validate_nonnegative_scalar(params, "resection_time", name)
        time_after = _validate_nonnegative_scalar(params, "time_after_resection", name)
        # The horizon of the shared pipeline, derived; not an input.
        params["stopping_time"] = resection_time + time_after
        cavity = _validate_volume(params, "resection_cavity", shape, name)
        if cavity.dtype != bool and not np.isin(cavity, (0, 1)).all():
            raise ValueError(
                f"{name}: resection_cavity must be a binary (bool or 0/1) array."
            )

        chemo_times = _validate_event_times(params, "chemo_times", name)
        chemo_doses = _validate_nonnegative_sequence(params, "chemo_doses", name)
        if chemo_doses.size != chemo_times.size:
            raise ValueError(
                f"{name}: chemo_doses has {chemo_doses.size} entries but chemo_times "
                f"has {chemo_times.size}; one dose per session is required."
            )
        _validate_nonnegative_scalar(params, "chemo_kill_rate", name)
        _validate_positive_scalar(params, "chemo_decay_rate", name)

        rt_times = _validate_event_times(params, "rt_times", name)
        if rt_times.size < 1:
            raise ValueError(f"{name}: rt_times must contain at least one time.")
        if np.any(rt_times == 0):
            logger.warning(
                f"{name}: rt_times contains 0, which lies in no step interval "
                "(t0, t1] and will never fire."
            )
        dose = _validate_volume(params, "rt_dose", shape, name)
        if not np.all(np.isfinite(dose)) or np.any(dose < 0):
            raise ValueError(f"{name}: rt_dose must be finite and nonnegative.")
        _validate_nonnegative_scalar(params, "rt_alpha", name)
        _validate_positive_scalar(params, "rt_alpha_beta_ratio", name)

    def _prepare_input_fields(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        params = self.params
        factor = params["resolution_factor"]
        self._gm_lowres = self._downsample(params["gray_matter_pbmap"], factor)
        self._wm_lowres = self._downsample(params["white_matter_pbmap"], factor)
        # Linear downsampling of the treatment volumes with the tissue
        # maps' factor, so the low-resolution grids coincide.
        cavity = np.asarray(params["resection_cavity"], dtype=np.float64)
        self._cavity_lowres = self._downsample(cavity, factor) >= 0.5
        dose = np.asarray(params["rt_dose"], dtype=np.float64)
        self._rt_dose_lowres = np.clip(self._downsample(dose, factor), 0, None)
        return self._gm_lowres.shape, params["gray_matter_pbmap"].shape

    def _check_seed(self) -> None:
        i, j, k = self.seed_voxel
        if self._gm_lowres[i, j, k] == 0 and self._wm_lowres[i, j, k] == 0:
            raise ValueError("Initial tumor position is outside the brain matter.")

    def _crop_mask(self) -> NDArray:
        return (self._gm_lowres + self._wm_lowres) >= CROP_TISSUE_THRESHOLD

    def _initialize_state(self) -> dict[str, jax.Array]:
        return {"cell_density": self._gaussian_seed()}

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> _StuppSpecificConstants:
        params = self.params
        gm_host = self._gm_lowres[box]
        wm_host = self._wm_lowres[box]

        # Host-side float64 tissue mask, identical for both precisions.
        tissue_mask_host = (wm_host + gm_host) >= float(params["min_tissue_fraction"])
        gm = jnp.asarray(gm_host, dtype=self._dtype)
        wm = jnp.asarray(wm_host, dtype=self._dtype)
        diffusivity = float(params["white_matter_diffusivity"])
        ratio = float(params["diffusivity_ratio"])
        faces = _mixture_face_fields(
            wm, gm, jnp.asarray(tissue_mask_host), diffusivity, ratio
        )
        # Post-resection faces: the cavity is removed from the valid mask,
        # so every face touching a cavity voxel carries no flux.
        cavity_host = self._cavity_lowres[box]
        faces_post = _mixture_face_fields(
            wm,
            gm,
            jnp.asarray(np.logical_and(tissue_mask_host, ~cavity_host)),
            diffusivity,
            ratio,
        )
        rt_times = np.asarray(params["rt_times"], dtype=np.float64)
        # Per-fraction dose: rt_dose is the TOTAL dose over all fractions.
        dose_per_fraction = jnp.asarray(
            self._rt_dose_lowres[box] / rt_times.size, dtype=self._dtype
        )
        # The quadratic coefficient from the alpha/beta ratio, on the host;
        # rt_beta is derived here and is not a parameter.
        rt_alpha = float(params["rt_alpha"])
        rt_beta = rt_alpha / float(params["rt_alpha_beta_ratio"])
        return {
            "face_diffusivities": faces,
            "rho": self._dynamic_scalar(params["rho"]),
            "resection_time": self._dynamic_scalar(params["resection_time"]),
            "cavity": jnp.asarray(cavity_host),
            "face_diffusivities_post": faces_post,
            "chemo_times": jnp.asarray(
                np.asarray(params["chemo_times"], dtype=np.float64), dtype=self._dtype
            ),
            "chemo_doses": jnp.asarray(
                np.asarray(params["chemo_doses"], dtype=np.float64), dtype=self._dtype
            ),
            "chemo_kill_rate": self._dynamic_scalar(params["chemo_kill_rate"]),
            "chemo_decay_rate": self._dynamic_scalar(params["chemo_decay_rate"]),
            "rt_times": jnp.asarray(rt_times, dtype=self._dtype),
            "rt_log_kill": lq_log_kill(dose_per_fraction, rt_alpha, rt_beta),
        }

    def _time_step_count(self) -> tuple[int, float]:
        # NOTE: preliminary stability heuristic -- the isotropic formula of
        # FKPPSolver; the CT, RT and resection events are impulse maps and
        # do not constrain dt.
        stopping_time = self.params["stopping_time"]
        diffusivity_wm = self.params["white_matter_diffusivity"]
        rho = self.params["rho"]
        reaction_rate = rho
        dx, dy, dz = self.grid_spacing
        # np.power kept deliberately: CPython's ** is not bit-identical to it.
        n_timesteps = max(
            stopping_time * diffusivity_wm / np.power(min(dx, dy, dz), 2) * 8 + 100,
            stopping_time * reaction_rate * 1.1,
        )
        dt = stopping_time / n_timesteps
        return int(np.ceil(n_timesteps)), dt
