"""The three JAX Fisher-KPP forward solvers: ``FKPPSolver``,
``TwoCompartmentWithNutrientFKPPSolver`` and ``AnisotropicFKPPSolver``.

The device code is purely functional: each solver's step is a module-level
function with a stable identity, so the jitted scan driver's cache persists
across solves (see ``operators._run_time_loop``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from .base import BaseFKPPSolver
from .operators import (
    GAUSSIAN_SEED_DIFFUSION_TIME,
    GAUSSIAN_SEED_FLOOR,
    GAUSSIAN_SEED_MASS,
    SHRINKAGE_LIMIT,
    VANISHING_DENSITY_LIMIT,
    diffusion_term,
    elongate_tensor_along_principal_axis,
    face_average,
    logistic_growth,
    logistic_sigmoid,
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
    "n_steps": None,  # explicit step count (dt = stopping_time / n_steps);
    #                   None -> the solver's own stability formula
    "verbose": False,
    "precision": "f32",  # device state dtype: "f32" (default) or "f64"
}


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
    ratio: float | jax.Array,
) -> dict[str, jax.Array]:
    """
    Build white/gray-matter mixture face diffusivities (device).

    D = diffusivity * (wm_face + gm_face / ratio), faces masked by
    valid_mask. The 'bwd' fields are the edge-replicated shift of the 'fwd'
    fields (zero-flux boundary convention).

    Args:
        wm: White matter fraction field.
        gm: Gray matter fraction field.
        valid_mask: Boolean mask of valid cells; faces touching invalid
            cells carry zero diffusivity.
        diffusivity: White matter diffusivity.
        ratio: White-to-gray-matter diffusivity ratio.

    Returns:
        The face diffusivities: keys 'fwd_x/y/z' and 'bwd_x/y/z', each the
        shape of the input grid; see ``diffusion_term``.
    """
    faces: dict[str, jax.Array] = {}
    for axis, name in enumerate(_AXES):
        wm_face = masked_face_average(wm, valid_mask, axis)
        gm_face = masked_face_average(gm, valid_mask, axis)
        fwd = diffusivity * (wm_face + gm_face / ratio)
        faces[f"fwd_{name}"] = fwd
        faces[f"bwd_{name}"] = shift_grid_by_one(fwd, 1, axis=axis)
    return faces


# --- module-level device functions (stable identity so the jitted scan
# --- driver's cache persists across solves; see operators._scan_driver) ---


def _single_field_step(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    spacing: tuple[float, float, float],
) -> dict[str, jax.Array]:
    """
    Perform one Euler step of the single-field solvers (FKPPSolver and
    AnisotropicFKPPSolver).

    The face diffusivities are constant in time and come from constants.

    Args:
        state: State dict with key 'cell_density'.
        constants: Constant device arrays with key 'face_diffusivities'.
        dynamic_scalars: 0-d device scalars 'dt' and 'rho'.
        spacing: Grid spacing (dx, dy, dz) in mm.

    Returns:
        The stepped state.
    """
    dt = dynamic_scalars["dt"]
    rho = dynamic_scalars["rho"]
    u = state["cell_density"]
    diffusion = diffusion_term(u, constants["face_diffusivities"], spacing)
    delta_u = (diffusion + logistic_growth(u, rho)) * dt
    return {"cell_density": u + delta_u}


def _two_compartment_step(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    spacing: tuple[float, float, float],
) -> dict[str, jax.Array]:
    """
    Perform one Euler step of the proliferative/necrotic/nutrient system.

    The update order is deliberately sequential: the necrotic and nutrient
    updates see the already-updated proliferative field.

    Args:
        state: State dict with keys 'proliferative', 'necrotic' and
            'nutrient'.
        constants: Constant device arrays with keys 'wm', 'gm',
            'tissue_valid' and 'nutrient_faces'.
        dynamic_scalars: 0-d device scalars, see
            TwoCompartmentWithNutrientFKPPSolver._step_spec.
        spacing: Grid spacing (dx, dy, dz) in mm.

    Returns:
        The stepped state.
    """
    dt = dynamic_scalars["dt"]
    necrosis_rate = dynamic_scalars["necrosis_rate"]
    proliferative = state["proliferative"]
    necrotic = state["necrotic"]
    nutrient = state["nutrient"]

    # Per-step tumor-diffusivity rebuild from the carried state: on step t
    # the mask deliberately uses the post-step P and N of step t-1 (the
    # initial state on step 0).
    occupancy_valid = (proliferative + necrotic) <= dynamic_scalars[
        "max_tumor_occupancy"
    ]
    tumor_faces = _mixture_face_fields(
        constants["wm"],
        constants["gm"],
        jnp.logical_and(constants["tissue_valid"], occupancy_valid),
        dynamic_scalars["white_matter_diffusivity"],
        dynamic_scalars["diffusivity_ratio"],
    )

    # Smooth descending switch on the nutrient level.
    switch = logistic_sigmoid(
        -NECROSIS_SWITCH_STEEPNESS * (nutrient - dynamic_scalars["nutrient_threshold"])
    )

    tumor_diffusion = diffusion_term(proliferative, tumor_faces, spacing)
    delta_proliferative = (
        tumor_diffusion
        + dynamic_scalars["rho"]
        * (nutrient * proliferative)
        * (1 - proliferative - necrotic)
        - necrosis_rate * proliferative * switch
    ) * dt
    proliferative = proliferative + delta_proliferative

    delta_necrotic = necrosis_rate * proliferative * switch * dt
    necrotic = necrotic + delta_necrotic

    nutrient_diffusion = diffusion_term(
        nutrient, constants["nutrient_faces"], spacing
    )
    delta_nutrient = (
        nutrient_diffusion
        - dynamic_scalars["nutrient_consumption_rate"] * nutrient * proliferative
    ) * dt
    nutrient = nutrient + delta_nutrient

    return {
        "proliferative": proliferative,
        "necrotic": necrotic,
        "nutrient": nutrient,
    }


def _mass_single(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    voxel_volume: float,
) -> jax.Array:
    """Integrated cell density of the single-field solvers, summed in f64."""
    del constants, dynamic_scalars
    return voxel_volume * jnp.sum(state["cell_density"], dtype=jnp.float64)


def _mass_two_compartment(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    voxel_volume: float,
) -> jax.Array:
    """
    Two-compartment integrated cell density, f64:
    voxel_volume * (sum(P) + sum(N)); the voxel-volume factor deliberately
    multiplies BOTH terms.
    """
    del constants, dynamic_scalars
    return voxel_volume * (
        jnp.sum(state["proliferative"], dtype=jnp.float64)
        + jnp.sum(state["necrotic"], dtype=jnp.float64)
    )


def _volume_single(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    voxel_volume: float,
) -> jax.Array:
    """
    Thresholded volume of the single-field solvers; the volume threshold
    is a dynamic 0-d scalar at the state dtype.
    """
    del constants
    threshold = dynamic_scalars["volume_threshold"]
    count = jnp.count_nonzero(state["cell_density"] > threshold)
    return voxel_volume * count.astype(jnp.float64)


def _volume_two_compartment(
    state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    voxel_volume: float,
) -> jax.Array:
    """Thresholded volume of P + N (the nutrient field is never included)."""
    del constants
    threshold = dynamic_scalars["volume_threshold"]
    density = state["proliferative"] + state["necrotic"]
    count = jnp.count_nonzero(density > threshold)
    return voxel_volume * count.astype(jnp.float64)


def _dti_guard(
    new_state: dict[str, jax.Array],
    previous_state: dict[str, jax.Array],
    constants: dict[str, Any],
    dynamic_scalars: dict[str, jax.Array],
    voxel_volume: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Evaluate the shrinkage/vanishing guards of the anisotropic solver.

    Shrinkage takes precedence; the thresholds are SHRINKAGE_LIMIT and
    VANISHING_DENSITY_LIMIT, and the sums run in float64 regardless of the
    state dtype. A firing guard is reported as Result(success=False,
    stopping_criterion="error") with final_time at the actual exit step.
    """
    del constants, dynamic_scalars
    new_sum = jnp.sum(new_state["cell_density"], dtype=jnp.float64)
    prev_sum = jnp.sum(previous_state["cell_density"], dtype=jnp.float64)
    total_change = new_sum - prev_sum
    integrated_density = voxel_volume * new_sum
    code = jnp.where(
        total_change < -SHRINKAGE_LIMIT,
        1,
        jnp.where(integrated_density < VANISHING_DENSITY_LIMIT, 2, 0),
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
        "min_tissue_fraction": 0.1,
    }
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

    def _build_device_constants(self, dt: float) -> dict[str, Any]:
        del dt
        assert self._crop_box is not None
        gm_host = self._gm_lowres[self._crop_box]
        wm_host = self._wm_lowres[self._crop_box]
        # Time-constant validity mask, computed host-side in float64 so it
        # is identical for both precisions.
        valid_host = (wm_host + gm_host) >= float(self.params["min_tissue_fraction"])
        gm = jnp.asarray(gm_host, dtype=self._dtype)
        wm = jnp.asarray(wm_host, dtype=self._dtype)
        faces = _mixture_face_fields(
            wm,
            gm,
            jnp.asarray(valid_host),
            float(self.params["white_matter_diffusivity"]),
            float(self.params["diffusivity_ratio"]),
        )
        return {"face_diffusivities": faces}

    def _step_spec(self, dt: float) -> dict[str, Any]:
        dynamic_scalars = {
            "dt": self._dynamic_scalar(dt),
            "rho": self._dynamic_scalar(self.params["rho"]),
        }
        return {
            "func": _single_field_step,
            "dynamic_scalars": dynamic_scalars,
            "static_args": (self.grid_spacing,),
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

    def _build_device_constants(self, dt: float) -> dict[str, Any]:
        del dt
        assert self._crop_box is not None
        gm_host = self._gm_lowres[self._crop_box]
        wm_host = self._wm_lowres[self._crop_box]
        # Time-constant validity mask, computed host-side in float64 so it
        # is identical for both precisions.
        tissue_valid_host = (wm_host + gm_host) >= float(
            self.params["min_tissue_fraction"]
        )
        gm = jnp.asarray(gm_host, dtype=self._dtype)
        wm = jnp.asarray(wm_host, dtype=self._dtype)
        tissue_valid = jnp.asarray(tissue_valid_host)

        # Nutrient faces are built once (constant in time; ratio 1 means
        # gray matter conducts nutrient like white matter).
        # The tumor faces are rebuilt every step inside the step function.
        nutrient_faces = _mixture_face_fields(
            wm, gm, tissue_valid, float(self.params["nutrient_diffusivity"]), 1
        )
        return {
            "wm": wm,
            "gm": gm,
            "tissue_valid": tissue_valid,
            "nutrient_faces": nutrient_faces,
        }

    def _step_spec(self, dt: float) -> dict[str, Any]:
        param_keys = (
            "white_matter_diffusivity",
            "diffusivity_ratio",
            "rho",
            "necrosis_rate",
            "nutrient_consumption_rate",
            "nutrient_threshold",
            "max_tumor_occupancy",
        )
        dynamic_scalars = {
            k: self._dynamic_scalar(self.params[k]) for k in param_keys
        }
        dynamic_scalars["dt"] = self._dynamic_scalar(dt)
        return {
            "func": _two_compartment_step,
            "dynamic_scalars": dynamic_scalars,
            "static_args": (self.grid_spacing,),
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
    _mass_func = staticmethod(_mass_single)
    _volume_func = staticmethod(_volume_single)

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

    def _build_device_constants(self, dt: float) -> dict[str, Any]:
        del dt
        assert self._crop_box is not None
        axial = jnp.asarray(self._axial_lowres[self._crop_box], dtype=self._dtype)
        diffusivity = float(self.params["diffusivity"])
        faces: dict[str, jax.Array] = {}
        for axis, name in enumerate(_AXES):
            face = face_average(axial[:, :, :, axis], axis)
            faces[f"fwd_{name}"] = face * diffusivity
            faces[f"bwd_{name}"] = diffusivity * shift_grid_by_one(face, 1, axis=axis)
        return {"face_diffusivities": faces}

    def _step_spec(self, dt: float) -> dict[str, Any]:
        dynamic_scalars = {
            "dt": self._dynamic_scalar(dt),
            "rho": self._dynamic_scalar(self.params["rho"]),
        }
        return {
            "func": _single_field_step,
            "dynamic_scalars": dynamic_scalars,
            "static_args": (self.grid_spacing,),
        }

    def _guard_spec(self) -> dict[str, Any]:
        """DTI shrinkage/vanishing guards -- semantics at ``_dti_guard``."""
        return {
            "func": _dti_guard,
            "dynamic_scalars": {},
            "static_args": (self.voxel_volume,),
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
