"""The JAX Fisher-KPP forward solvers: ``FKPPSolver``,
``TwoCompartmentWithNutrientFKPPSolver``, ``AnisotropicFKPPSolver`` and the
treatment-extended ``StuppFKPPSolver`` (with its manifest loader
``treatment_params_from_manifest``).

Each solver's time step is a module-level function with a stable identity,
so the jitted time scan's cache persists across solves (see ``operators._run_time_loop``).
"""

from __future__ import annotations

import json
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

from .base import BaseFKPPSolver, _SharedConstants
from .operators import (
    GAUSSIAN_SEED_DIFFUSION_TIME,
    GAUSSIAN_SEED_FLOOR,
    GAUSSIAN_SEED_MASS,
    SHRINKAGE_LIMIT,
    VANISHING_DENSITY_LIMIT,
    chemo_concentration,
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
    "voxel_size_mm": (1.0, 1.0, 1.0),
    "gaussian_seed_scale": 1.0,
    "gaussian_seed_diffusion_time": GAUSSIAN_SEED_DIFFUSION_TIME,
    "gaussian_seed_mass": GAUSSIAN_SEED_MASS,
    "gaussian_seed_floor": GAUSSIAN_SEED_FLOOR,
    "stopping_time": 100,
    "stopping_threshold": np.inf,
    "stopping_mode": "mass",
    "volume_threshold": None,  # only valid with stopping_mode="volume"
    "n_time_series_snapshots": None,
    "n_steps": None,
    "verbose": False,
    "precision": "f32",
}


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
        chemo_kill_rate: Chemotherapy kill rate per unit concentration,
            0-d scalar at the state dtype.
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


def _validate_tissue_arrays(gm: NDArray, wm: NDArray) -> None:
    if not (isinstance(gm, np.ndarray) and isinstance(wm, np.ndarray)):
        raise ValueError("gray_matter_pbmap and white_matter_pbmap must be numpy arrays.")
    if not (gm.ndim == 3 and wm.ndim == 3):
        raise ValueError("gray_matter_pbmap and white_matter_pbmap must be 3D arrays.")
    if gm.shape != wm.shape:
        raise ValueError(
            "gray_matter_pbmap and white_matter_pbmap shapes differ: "
            f"{gm.shape} vs {wm.shape}."
        )


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
         du/dt = div(D grad u) + rho u (1 - u) - chemo_kill_rate C(t0) u,
         with D = the post-resection faces once t1 >= resection_time,
         else the pre-resection faces;
      2. radiotherapy impulse u <- u exp(-E(x) n_hits), n_hits = number of
         rt_times in (t0, t1] (exact impulse map, not part of the Euler
         right-hand side);
      3. resection projection u <- 0 inside the cavity, for every step
         with t1 >= resection_time (idempotent).

    Every treatment term is evaluated in every step. With neutral
    treatment values (an all-False cavity, a zero chemotherapy kill rate
    or no session, a zero radiotherapy log kill) the update equals
    ``_single_field_step`` up to floating-point rounding.

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

    concentration = chemo_concentration(
        t0, constants["chemo_times"], constants["chemo_decay_rate"]
    )
    reaction = logistic_growth(u, rho) - constants["chemo_kill_rate"] * concentration * u
    diffusion = diffusion_term(u, faces, constants["grid_spacing"])
    delta_u = (diffusion + reaction) * dt
    u = u + delta_u

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

    # static methods allows passing of stable module level functions as attributes
    _step_func = staticmethod(_single_field_step)
    _mass_func = staticmethod(_mass_single)
    _volume_func = staticmethod(_volume_single)

    _gm_lowres: NDArray
    _wm_lowres: NDArray

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        _validate_tissue_arrays(params["gray_matter_pbmap"], params["white_matter_pbmap"])

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

    _step_func = staticmethod(_two_compartment_step)
    _mass_func = staticmethod(_mass_two_compartment)
    _volume_func = staticmethod(_volume_two_compartment)

    _gm_lowres: NDArray
    _wm_lowres: NDArray

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        _validate_tissue_arrays(params["gray_matter_pbmap"], params["white_matter_pbmap"])

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


# Treatment parameters of StuppFKPPSolver, all required (see the class
# docstring for the values that switch a treatment off).
_STUPP_TREATMENT_KEYS: frozenset[str] = frozenset(
    {
        "resection_time",
        "resection_cavity",
        "chemo_times",
        "chemo_kill_rate",
        "chemo_decay_rate",
        "rt_times",
        "rt_dose",  # TOTAL dose over all fractions, 3D array in Gy
        "rt_alpha",
        "rt_beta",
    }
)

# Number of host-side sample points of the chemotherapy concentration used
# by StuppFKPPSolver's time-step formula.
_CHEMO_CONCENTRATION_SAMPLES: int = 10001


class StuppFKPPSolver(BaseFKPPSolver):
    """
    Isotropic Fisher-KPP solver on WM/GM tissue maps, extended by the
    treatment effects of a Stupp protocol: surgical resection,
    chemotherapy (CT) and radiotherapy (RT).

    State key: 'cell_density' (u in [0, 1]); grid in mm, time in days on
    the ``stopping_time`` clock with the seed at t = 0.

    Every treatment parameter is required; a treatment is switched off by
    its values, not by omitting it: an all-False resection_cavity leaves
    the dynamics untouched (the post-resection faces then equal the
    pre-resection ones), an empty chemo_times or chemo_kill_rate = 0
    removes the chemotherapy term, and a zero rt_dose (or
    rt_alpha = rt_beta = 0) makes the radiotherapy impulse the identity.
    With all three neutral the solver reproduces ``FKPPSolver`` up to
    floating-point rounding (the treatment terms are still evaluated, so
    the compiled arithmetic is not identical).

    Continuous model (explicit Euler at the pre-step state, drug
    concentration evaluated at the step start t0)::

        du/dt = div(D grad u) + rho u (1 - u) - chemo_kill_rate C(t) u
        C(t)  = sum_j [t >= chemo_times[j]] exp(-chemo_decay_rate (t - chemo_times[j]))

    Discrete events, applied after the Euler update of the step whose
    interval (t0, t1] contains them, in this order (see ``_stupp_step``):

      1. RT impulse: u <- u exp(-E(x) n_hits) with the linear-quadratic
         log kill E(x) = rt_alpha d(x) + rt_beta d(x)^2 and n_hits the
         number of rt_times in (t0, t1]. Per-fraction dose convention:
         rt_dose holds the TOTAL dose over all fractions, so
         d(x) = rt_dose / len(rt_times) (computed once on the host).
      2. Resection: u <- 0 inside resection_cavity for every step with
         t1 >= resection_time, and from the same step on the face
         diffusivities switch to a post-resection set in which every face
         touching a cavity voxel is zero (zero-flux Neumann on the cavity
         boundary).
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
        | _STUPP_TREATMENT_KEYS
    )
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        **_COMMON_DEFAULTS,
        # Cells with wm + gm below this carry no flux (CSF/background).
        "min_tissue_fraction": 0.1,
    }

    # static methods allows passing of stable module level functions as attributes
    _step_func = staticmethod(_stupp_step)
    _mass_func = staticmethod(_mass_single)
    _volume_func = staticmethod(_volume_single)

    _gm_lowres: NDArray
    _wm_lowres: NDArray
    _cavity_lowres: NDArray
    _rt_dose_lowres: NDArray

    def _validate_extra(self, params: Mapping[str, Any]) -> None:
        name = type(self).__name__
        gm = params["gray_matter_pbmap"]
        wm = params["white_matter_pbmap"]
        _validate_tissue_arrays(gm, wm)
        stopping_time = float(params["stopping_time"])

        def check_times(key: str) -> NDArray:
            times = np.asarray(params[key], dtype=np.float64)
            if times.ndim != 1:
                raise ValueError(f"{name}: {key} must be a 1-D sequence of times.")
            if not np.all(np.isfinite(times)) or np.any(times < 0):
                raise ValueError(f"{name}: {key} must be finite and nonnegative.")
            late = times[times > stopping_time]
            if late.size:
                logger.warning(
                    f"{name}: {key} contains {late.size} time(s) beyond "
                    f"stopping_time={stopping_time:g} that will never fire: "
                    f"{late.tolist()}."
                )
            return times

        def check_rate(key: str) -> None:
            value = params[key]
            if not (np.isscalar(value) and np.isfinite(value) and value >= 0):
                raise ValueError(
                    f"{name}: {key} must be a finite nonnegative scalar, got "
                    f"{value!r}."
                )

        def check_volume(key: str) -> NDArray:
            value = params[key]
            if not isinstance(value, np.ndarray):
                raise ValueError(f"{name}: {key} must be a numpy array.")
            if value.ndim != 3:
                raise ValueError(f"{name}: {key} must be a 3D array.")
            if value.shape != gm.shape:
                raise ValueError(
                    f"{name}: {key} shape {value.shape} differs from the "
                    f"tissue map shape {gm.shape}."
                )
            return value

        check_rate("resection_time")
        if float(params["resection_time"]) > stopping_time:
            logger.warning(
                f"{name}: resection_time={float(params['resection_time']):g} "
                f"lies beyond stopping_time={stopping_time:g} and will never "
                "fire."
            )
        cavity = check_volume("resection_cavity")
        if cavity.dtype != bool and not np.isin(cavity, (0, 1)).all():
            raise ValueError(
                f"{name}: resection_cavity must be a binary (bool or 0/1) array."
            )
        check_times("chemo_times")
        check_rate("chemo_kill_rate")
        check_rate("chemo_decay_rate")
        rt_times = check_times("rt_times")
        if rt_times.size < 1:
            raise ValueError(f"{name}: rt_times must contain at least one time.")
        if np.any(rt_times == 0):
            logger.warning(
                f"{name}: rt_times contains 0, which lies in no step interval "
                "(t0, t1] and will never fire."
            )
        dose = check_volume("rt_dose")
        if not np.all(np.isfinite(dose)) or np.any(dose < 0):
            raise ValueError(f"{name}: rt_dose must be finite and nonnegative.")
        check_rate("rt_alpha")
        check_rate("rt_beta")

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
        return {
            "face_diffusivities": faces,
            "rho": self._dynamic_scalar(params["rho"]),
            "resection_time": self._dynamic_scalar(params["resection_time"]),
            "cavity": jnp.asarray(cavity_host),
            "face_diffusivities_post": faces_post,
            "chemo_times": jnp.asarray(
                np.asarray(params["chemo_times"], dtype=np.float64), dtype=self._dtype
            ),
            "chemo_kill_rate": self._dynamic_scalar(params["chemo_kill_rate"]),
            "chemo_decay_rate": self._dynamic_scalar(params["chemo_decay_rate"]),
            "rt_times": jnp.asarray(rt_times, dtype=self._dtype),
            "rt_log_kill": lq_log_kill(
                dose_per_fraction, float(params["rt_alpha"]), float(params["rt_beta"])
            ),
        }

    def _max_chemo_concentration(self) -> float:
        """
        Host-side upper bound of the drug concentration C(t) over
        [0, stopping_time]: C is evaluated with numpy on a dense grid plus
        the session times themselves (where its local maxima lie). Zero
        without a session.
        """
        params = self.params
        stopping_time = float(params["stopping_time"])
        session_times = np.asarray(params["chemo_times"], dtype=np.float64)
        decay_rate = float(params["chemo_decay_rate"])
        grid = np.union1d(
            np.linspace(0.0, stopping_time, _CHEMO_CONCENTRATION_SAMPLES),
            session_times[session_times <= stopping_time],
        )
        elapsed = grid[:, None] - session_times[None, :]
        concentration = np.where(elapsed >= 0, np.exp(-decay_rate * elapsed), 0.0)
        return float(concentration.sum(axis=1).max(initial=0.0))

    def _time_step_count(self) -> tuple[int, float]:
        # NOTE: preliminary stability heuristic -- the isotropic formula of
        # FKPPSolver with the reaction bound extended from rho to
        # rho + chemo_kill_rate * max_t C(t); the RT and resection events
        # are impulse maps and do not constrain dt.
        stopping_time = self.params["stopping_time"]
        diffusivity_wm = self.params["white_matter_diffusivity"]
        rho = self.params["rho"]
        reaction_rate = rho + self.params["chemo_kill_rate"] * self._max_chemo_concentration()
        dx, dy, dz = self.grid_spacing
        # np.power kept deliberately: CPython's ** is not bit-identical to it.
        n_timesteps = max(
            stopping_time * diffusivity_wm / np.power(min(dx, dy, dz), 2) * 8 + 100,
            stopping_time * reaction_rate * 1.1,
        )
        dt = stopping_time / n_timesteps
        return int(np.ceil(n_timesteps)), dt


# --- manifest loaders ---

# Manifest schema: section name -> required keys (None: the free-form
# 'solver' section, validated against _MANIFEST_SOLVER_KEYS). Keys starting
# with '_' (comments) are ignored everywhere.
_MANIFEST_SECTIONS: dict[str, tuple[str, ...] | None] = {
    "tissue": ("wm", "gm"),
    "solver": None,
    "resection": ("time", "tumor_segmentation", "cavity_label"),
    "chemotherapy": ("times", "kill_rate", "decay_rate"),
    "radiotherapy": ("times", "dose", "alpha", "beta"),
}
_MANIFEST_TREATMENT_SECTIONS = ("resection", "chemotherapy", "radiotherapy")

# Keys accepted in the 'solver' section: every scalar StuppFKPPSolver
# parameter (the volumes come from the 'tissue' section, the treatment
# parameters from their own sections) plus the horizon alternative
# 'time_after_resection' (stopping_time = resection time + it) and the
# time-step alternatives 'dt' and 'steps_per_day', translated to n_steps.
_MANIFEST_HORIZON_KEYS: tuple[str, ...] = ("stopping_time", "time_after_resection")
_MANIFEST_TIME_STEP_KEYS: tuple[str, ...] = ("n_steps", "dt", "steps_per_day")
_MANIFEST_SOLVER_KEYS: frozenset[str] = frozenset(
    (StuppFKPPSolver._REQUIRED | set(StuppFKPPSolver._DEFAULTS))
    - {"gray_matter_pbmap", "white_matter_pbmap"}
    - _STUPP_TREATMENT_KEYS
) | set(_MANIFEST_HORIZON_KEYS) | set(_MANIFEST_TIME_STEP_KEYS)


def _read_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read and structurally check a manifest: a JSON object whose non-'_'
    top-level keys are known sections."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"manifest {manifest_path}: must be a JSON object.")
    unknown = sorted(
        key
        for key in manifest
        if not key.startswith("_") and key not in _MANIFEST_SECTIONS
    )
    if unknown:
        raise ValueError(
            f"manifest {manifest_path}: unknown top-level key(s) {unknown}; "
            f"expected a subset of {list(_MANIFEST_SECTIONS)}."
        )
    return manifest_path, dict(manifest)


def _manifest_section(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return the checked manifest section name without its '_' keys: for
    the fixed-schema sections all keys present and none unknown, for the
    'solver' section only known solver keys."""
    section = manifest[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"manifest section {name!r} must be a JSON object.")
    present = {key for key in section if not key.startswith("_")}
    keys = _MANIFEST_SECTIONS[name]
    if keys is None:
        unknown = sorted(present - _MANIFEST_SOLVER_KEYS)
        if unknown:
            raise ValueError(
                f"manifest section {name!r}: unknown key(s) {unknown}; expected "
                f"a subset of {sorted(_MANIFEST_SOLVER_KEYS)}."
            )
        return {key: section[key] for key in sorted(present)}
    unknown = sorted(present - set(keys))
    if unknown:
        raise ValueError(
            f"manifest section {name!r}: unknown key(s) {unknown}; expected "
            f"{list(keys)}."
        )
    missing = [key for key in keys if key not in present]
    if missing:
        raise ValueError(f"manifest section {name!r} is missing key(s) {missing}.")
    return {key: section[key] for key in keys}


def _manifest_file(path_value: Any, manifest_path: Path, what: str) -> Path:
    """Resolve a file named in the manifest (relative to the manifest's
    directory) and check that it exists."""
    path = Path(str(path_value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"manifest {manifest_path}: {what} not found: {path}")
    return path


def _load_manifest_volume(path_value: Any, manifest_path: Path, what: str) -> NDArray:
    """Load a NIfTI volume named in the manifest as a float64 array."""
    path = _manifest_file(path_value, manifest_path, f"{what} volume")
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64)


def n_steps_from_dt(stopping_time: float, dt: float) -> int:
    """
    Translate a requested time step into the solver's n_steps parameter.

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


def tissue_paths_from_manifest(path: str | Path) -> dict[str, Path]:
    """
    Load the tissue-map paths of a manifest's optional 'tissue' section.

    ::

        "tissue": {"wm": <white matter pbmap NIfTI>, "gm": <gray matter pbmap NIfTI>}

    Paths are absolute or relative to the manifest's directory and must
    exist. See ``treatment_params_from_manifest`` for the manifest layout.

    Args:
        path: Path of the JSON manifest.

    Returns:
        {'wm': path, 'gm': path} if the section is present, else {}.
    """
    manifest_path, manifest = _read_manifest(path)
    if "tissue" not in manifest:
        return {}
    section = _manifest_section(manifest, "tissue")
    return {
        key: _manifest_file(section[key], manifest_path, f"{key} pbmap")
        for key in ("wm", "gm")
    }


def resolve_horizon(
    params: dict[str, Any], resection_time: float | None
) -> dict[str, Any]:
    """
    Translate the horizon alternative 'time_after_resection' of a params
    dict into the solver's stopping_time parameter,
    stopping_time = resection_time + time_after_resection.

    At most one of 'stopping_time' and 'time_after_resection' may be set;
    the latter needs a resection time.

    Args:
        params: Parameter dict, possibly holding 'time_after_resection'.
        resection_time: The resection time in days, or None if the run has
            no resection.

    Returns:
        A copy without 'time_after_resection', with 'stopping_time' set if
        it was present.
    """
    params = dict(params)
    present = [key for key in _MANIFEST_HORIZON_KEYS if params.get(key) is not None]
    if len(present) > 1:
        raise ValueError(f"set at most one of {list(_MANIFEST_HORIZON_KEYS)}, got {present}.")
    time_after = params.pop("time_after_resection", None)
    if time_after is not None:
        if resection_time is None:
            raise ValueError("'time_after_resection' requires a resection time.")
        if not (np.isfinite(time_after) and time_after >= 0):
            raise ValueError(
                f"time_after_resection must be a finite nonnegative number, got {time_after!r}."
            )
        params["stopping_time"] = float(resection_time) + float(time_after)
    return params


def resolve_time_step(params: dict[str, Any]) -> dict[str, Any]:
    """
    Translate the time-step alternatives 'dt' and 'steps_per_day' of a
    params dict into the solver's n_steps parameter.

    At most one of 'n_steps', 'dt' and 'steps_per_day' may be set; 'dt'
    and 'steps_per_day' (dt = 1 / steps_per_day) need 'stopping_time' in
    the same dict and go through ``n_steps_from_dt``.

    Args:
        params: Parameter dict, possibly holding 'dt' or 'steps_per_day'.

    Returns:
        A copy without 'dt' / 'steps_per_day', with 'n_steps' set if one
        of them was present.
    """
    params = dict(params)
    present = [key for key in _MANIFEST_TIME_STEP_KEYS if params.get(key) is not None]
    if len(present) > 1:
        raise ValueError(f"set at most one of {list(_MANIFEST_TIME_STEP_KEYS)}, got {present}.")
    dt = params.pop("dt", None)
    steps_per_day = params.pop("steps_per_day", None)
    if steps_per_day is not None:
        if not (np.isfinite(steps_per_day) and steps_per_day > 0):
            raise ValueError(
                f"steps_per_day must be a positive finite number, got {steps_per_day!r}."
            )
        dt = 1.0 / float(steps_per_day)
    if dt is not None:
        if params.get("stopping_time") is None:
            raise ValueError(f"{present[0]!r} requires 'stopping_time' alongside it.")
        params["n_steps"] = n_steps_from_dt(params["stopping_time"], dt)
    return params


def solver_params_from_manifest(
    path: str | Path, resolve: bool = True
) -> dict[str, Any]:
    """
    Load the scalar solver parameters of a manifest's optional 'solver'
    section.

    The section may hold any scalar StuppFKPPSolver parameter under its
    solver name (rho, white_matter_diffusivity, diffusivity_ratio,
    resolution_factor, stopping_time, n_steps, precision,
    gaussian_seed_x/y/z_fraction, gaussian_seed_scale/diffusion_time/
    mass/floor, voxel_size_mm, stopping_threshold, stopping_mode,
    volume_threshold, min_tissue_fraction, n_time_series_snapshots,
    verbose), plus the horizon alternative 'time_after_resection'
    (stopping_time = the manifest's resection time + it, see
    ``resolve_horizon``) and the time-step alternatives 'dt' (days) and
    'steps_per_day', translated to n_steps by ``resolve_time_step`` with
    the resolved stopping_time. A 'stopping_threshold' of "inf" is
    accepted (JSON has no infinity); voxel_size_mm lists become tuples.
    Values are passed through otherwise; the solver validates them.

    The treatment parameters live in their own sections, see
    ``treatment_params_from_manifest``; the tissue maps in 'tissue', see
    ``tissue_paths_from_manifest``.

    Args:
        path: Path of the JSON manifest.
        resolve: Translate 'time_after_resection' into 'stopping_time'
            and 'dt' / 'steps_per_day' into 'n_steps' (the default). With
            False they are returned as given (checked for mutual
            exclusivity only), for callers that merge further overrides
            before calling ``resolve_horizon`` / ``resolve_time_step``
            themselves.

    Returns:
        The solver parameters of the section (an empty dict if absent);
        with resolve=True ready to be merged into the StuppFKPPSolver
        params dict.
    """
    _, manifest = _read_manifest(path)
    if "solver" not in manifest:
        return {}
    params = _manifest_section(manifest, "solver")
    for group in (_MANIFEST_HORIZON_KEYS, _MANIFEST_TIME_STEP_KEYS):
        present = [key for key in group if key in params]
        if len(present) > 1:
            raise ValueError(
                f"manifest section 'solver': set at most one of {list(group)}, "
                f"not {present}."
            )
    if resolve:
        resection_time = None
        if "resection" in manifest:
            resection_time = float(_manifest_section(manifest, "resection")["time"])
        try:
            params = resolve_time_step(resolve_horizon(params, resection_time))
        except ValueError as exc:
            raise ValueError(f"manifest section 'solver': {exc}") from exc
    if "voxel_size_mm" in params:
        params["voxel_size_mm"] = tuple(float(v) for v in params["voxel_size_mm"])
    if isinstance(params.get("stopping_threshold"), str):
        params["stopping_threshold"] = float(params["stopping_threshold"])
    return params


def treatment_params_from_manifest(path: str | Path) -> dict[str, Any]:
    """
    Load the StuppFKPPSolver treatment parameters from a JSON manifest.

    The manifest is a JSON object of sections. This function reads the
    three treatment sections, all of which must be present (the solver
    requires every treatment parameter; see ``StuppFKPPSolver`` for the
    values that switch a treatment off)::

        "resection":     {"time": <day>, "tumor_segmentation": <NIfTI path>,
                          "cavity_label": <int>}
        "chemotherapy":  {"times": [<days>], "kill_rate": <1/day>,
                          "decay_rate": <1/day>}
        "radiotherapy":  {"times": [<days>], "dose": <NIfTI path, TOTAL Gy>,
                          "alpha": <1/Gy>, "beta": <1/Gy^2>}

    The companion sections 'solver' (scalar solver parameters, see
    ``solver_params_from_manifest``) and 'tissue' (pbmap paths, see
    ``tissue_paths_from_manifest``) are validated here but not returned.
    All times are days on the solver clock, whose origin t = 0 is the
    seed: the simulated time before resection is resection 'time' itself.

    Top-level and per-section keys starting with '_' are comments and are
    ignored. NIfTI paths are absolute or relative to the manifest's
    directory; the volumes are loaded with nibabel. The resection cavity
    is ``segmentation == cavity_label`` (values rounded to the nearest
    integer first).

    Args:
        path: Path of the JSON manifest.

    Returns:
        The solver parameters of the treatment sections, ready to be merged
        into the StuppFKPPSolver params dict: 'resection_time',
        'resection_cavity' (bool array), 'chemo_times', 'chemo_kill_rate',
        'chemo_decay_rate', 'rt_times', 'rt_dose' (float64 array, Gy),
        'rt_alpha', 'rt_beta'.
    """
    manifest_path, manifest = _read_manifest(path)
    missing = [name for name in _MANIFEST_TREATMENT_SECTIONS if name not in manifest]
    if missing:
        raise ValueError(
            f"manifest {manifest_path}: missing treatment section(s) {missing}; "
            f"StuppFKPPSolver requires all of {list(_MANIFEST_TREATMENT_SECTIONS)}."
        )
    params: dict[str, Any] = {}
    section = _manifest_section(manifest, "resection")
    segmentation = _load_manifest_volume(
        section["tumor_segmentation"], manifest_path, "tumor_segmentation"
    )
    label = int(section["cavity_label"])
    params["resection_time"] = float(section["time"])
    params["resection_cavity"] = np.rint(segmentation).astype(np.int64) == label
    section = _manifest_section(manifest, "chemotherapy")
    params["chemo_times"] = np.asarray(section["times"], dtype=np.float64)
    params["chemo_kill_rate"] = float(section["kill_rate"])
    params["chemo_decay_rate"] = float(section["decay_rate"])
    section = _manifest_section(manifest, "radiotherapy")
    params["rt_times"] = np.asarray(section["times"], dtype=np.float64)
    params["rt_dose"] = _load_manifest_volume(section["dose"], manifest_path, "dose")
    params["rt_alpha"] = float(section["alpha"])
    params["rt_beta"] = float(section["beta"])
    return params
