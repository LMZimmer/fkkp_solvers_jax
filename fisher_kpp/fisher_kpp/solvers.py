"""The three Fisher-KPP forward solvers, ported from TumorGrowthToolkit.

Numerical behavior follows the originals closely. Deliberate deviations —
zero-flux boundaries instead of periodic wrap, the FK_2c stopping-mass
voxel-volume fix and reaction-rate dt guard, and the DTI guard-exit failure
semantics — are documented at their sites. The per-solver time-step formulas
remain mutually inconsistent by heritage and may be unified later.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from .base import BaseFKPPSolver
from .operators import (
    GAUSSIAN_SEED_DIFFUSION_TIME,
    GAUSSIAN_SEED_FLOOR,
    GAUSSIAN_SEED_MASS,
    FaceFields,
    clipped_gaussian,
    crop,
    diffusion_term,
    edge_roll,
    elongate_tensor_along_principal_axis,
    face_average,
    logistic_growth,
    logistic_sigmoid,
    masked_face_average,
)

logger = logging.getLogger(__name__)

# Crop-mask threshold hard-coded at the originals' call sites (their crop
# helpers advertise different defaults that are never used).
CROP_TISSUE_THRESHOLD: float = 0.5

# Steepness of the descending nutrient switch in the original FK_2c
# smooth_heaviside (k=50).
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
    "stopping_volume": np.inf,
    "stopping_mode": "mass",
    "density_threshold": None,  # only valid with stopping_mode="volume"
    "n_time_series_snapshots": None,
    "verbose": False,
}

DEFAULT_DENSITY_THRESHOLD: float = 0.5


def _merge_params(
    params: Mapping[str, Any],
    required: frozenset[str],
    defaults: Mapping[str, Any],
    solver_name: str,
) -> dict[str, Any]:
    """Strict parameter merge: unknown keys and missing required keys raise."""
    unknown = sorted(set(params) - required - set(defaults))
    if unknown:
        raise ValueError(f"{solver_name}: unknown parameter(s): {unknown}")
    missing = sorted(required - set(params))
    if missing:
        raise KeyError(f"{solver_name}: missing required parameter(s): {missing}")
    merged = dict(defaults)
    merged.update(params)
    _validate_stopping_params(merged, solver_name)
    return merged


def _validate_stopping_params(merged: dict[str, Any], solver_name: str) -> None:
    """stopping_mode / density_threshold validation, shared by all solvers.

    "mass" (the default) reproduces the original solvers' stopping behavior
    (modulo the FK_2c voxel-volume fix): an integrated cell density, not a
    physical volume. density_threshold is only meaningful for "volume" mode
    and is rejected otherwise (no silent unused parameters).
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


def _validate_tissue_arrays(gm: NDArray, wm: NDArray) -> None:
    assert isinstance(gm, np.ndarray), "gray_matter must be a numpy array"
    assert isinstance(wm, np.ndarray), "white_matter must be a numpy array"
    assert gm.ndim == 3, "gray_matter must be a 3D numpy array"
    assert wm.ndim == 3, "white_matter must be a 3D numpy array"
    assert gm.shape == wm.shape


def _validate_seed_fractions(params: Mapping[str, Any]) -> None:
    assert 0 <= params["gaussian_seed_x_fraction"] <= 1, "gaussian_seed_x_fraction must be between 0 and 1"
    assert 0 <= params["gaussian_seed_y_fraction"] <= 1, "gaussian_seed_y_fraction must be between 0 and 1"
    assert 0 <= params["gaussian_seed_z_fraction"] <= 1, "gaussian_seed_z_fraction must be between 0 and 1"


def _mixture_face_fields(
    wm: NDArray,
    gm: NDArray,
    valid_mask: NDArray,
    diffusivity: float,
    ratio: float,
) -> FaceFields:
    """White/gray-matter mixture face diffusivities:
    D = diffusivity * (wm_face + gm_face / ratio), faces masked by valid_mask.
    The 'plus' fields are the edge-replicated shift of the 'minus' fields
    (zero-flux boundary convention; interior bitwise identical to the original
    roll-based construction)."""
    faces: FaceFields = {}
    for axis, name in enumerate(_AXES):
        wm_face = masked_face_average(wm, valid_mask, axis)
        gm_face = masked_face_average(gm, valid_mask, axis)
        minus = diffusivity * (wm_face + gm_face / ratio)
        faces[f"minus_{name}"] = minus
        faces[f"plus_{name}"] = edge_roll(minus, 1, axis=axis)
    return faces


class FKPPSolver(BaseFKPPSolver):
    """Isotropic Fisher-KPP on WM/GM tissue maps (source: FK.FK.Solver).

    State key: 'cell_density'. Diffusivity is a WM/GM mixture,
    D = white_matter_diffusivity * (wm_face + gm_face / diffusivity_ratio),
    with faces masked by min_tissue_fraction.
    """

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "white_matter_diffusivity",
            "rho",
            "gray_matter",
            "white_matter",
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

    _gm_low: NDArray
    _wm_low: NDArray
    _gm_cropped: NDArray | None = None
    _wm_cropped: NDArray | None = None

    def _validate_params(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        merged = _merge_params(params, self._REQUIRED, self._DEFAULTS, type(self).__name__)
        _validate_tissue_arrays(merged["gray_matter"], merged["white_matter"])
        _validate_seed_fractions(merged)
        return merged

    def _prepare_fields(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        factor = self.params["resolution_factor"]
        self._gm_low = self._downsample(self.params["gray_matter"], factor)
        self._wm_low = self._downsample(self.params["white_matter"], factor)
        self._gm_cropped = None
        self._wm_cropped = None
        return self._gm_low.shape, self.params["gray_matter"].shape

    def _check_seed_early(self) -> None:
        i, j, k = self.seed_voxel
        if self._gm_low[i, j, k] == 0 and self._wm_low[i, j, k] == 0:
            raise ValueError("Initial tumor position is outside the brain matter")

    def _crop_mask(self) -> NDArray:
        return (self._gm_low + self._wm_low) >= CROP_TISSUE_THRESHOLD

    def _initialize_state(self) -> dict[str, NDArray]:
        return {
            "cell_density": clipped_gaussian(
                self.grid_shape,
                self.seed_voxel,
                self.grid_spacing,
                scale=self.params["gaussian_seed_scale"],
                diffusion_time=self.params["gaussian_seed_diffusion_time"],
                mass=self.params["gaussian_seed_mass"],
                floor=self.params["gaussian_seed_floor"],
            )
        }

    def _build_diffusivity(self, state: dict[str, NDArray]) -> dict[str, FaceFields]:
        if self._gm_cropped is None:
            assert self._crop_box is not None
            self._gm_cropped = crop(self._gm_low, self._crop_box)
            self._wm_cropped = crop(self._wm_low, self._crop_box)
        gm, wm = self._gm_cropped, self._wm_cropped
        valid = (wm + gm) >= self.params["min_tissue_fraction"]
        return {
            "cell_density": _mixture_face_fields(
                wm,
                gm,
                valid,
                self.params["white_matter_diffusivity"],
                self.params["diffusivity_ratio"],
            )
        }

    def _step(
        self,
        state: dict[str, NDArray],
        diffusivity: dict[str, FaceFields],
        dt: float,
    ) -> dict[str, NDArray]:
        u = state["cell_density"]
        sp = diffusion_term(u, diffusivity["cell_density"], self.grid_spacing)
        diff_u = (sp + logistic_growth(u, self.params["rho"])) * dt
        u += diff_u
        return state

    def _cell_mass(self, state: dict[str, NDArray]) -> float:
        # "mass" stopping mode: integrated cell density, reproducing the
        # original FK stopping quantity exactly. Not a physical volume.
        return self.voxel_volume * np.sum(state["cell_density"])

    def _cell_density_sum(self, state: dict[str, NDArray]) -> NDArray:
        return state["cell_density"]

    def _time_step_count(self) -> tuple[int, float]:
        stopping_time = self.params["stopping_time"]
        dw = self.params["white_matter_diffusivity"]
        rho = self.params["rho"]
        dx, dy, dz = self.grid_spacing
        # NOTE: ad-hoc stability formula; the three solvers' formulas remain
        # mutually inconsistent by heritage and may be unified later.
        nt = np.max(
            [
                stopping_time * dw / np.power(np.min([dx, dy, dz]), 2) * 8 + 100,
                stopping_time * rho * 1.1,
            ]
        )
        dt = stopping_time / nt
        return int(np.ceil(nt)), dt


class TwoCompartmentWithNutrientFKPPSolver(BaseFKPPSolver):
    """Proliferative/necrotic/nutrient system (source: FK_2c.Solver).

    State keys: 'proliferative', 'necrotic', 'nutrient'. Tumor diffusivity
    faces are additionally masked where proliferative + necrotic exceeds
    max_tumor_occupancy and are rebuilt every step. The nutrient diffuses
    with nutrient_diffusivity, masked by tissue only, built once.

    Stopping quantity in "mass" mode: voxel_volume * (sum(P) + sum(N)).
    This fixes the original FK_2c formula, which dropped the voxel-volume
    factor on the necrotic term (mixing units), so "mass"-mode stopping
    behavior intentionally differs from FK_2c whenever a finite
    stopping_volume is set. As in the other solvers, "mass" is an integrated
    cell density (the original behavior modulo this fix), not a physical
    volume; the nutrient field S is never included.

    Aliasing note: the per-step tumor diffusivity rebuild uses the POST-step
    P and N fields. The original code appears to pass the pre-step P, but its
    in-place updates meant the argument already aliased the updated field, so
    it effectively used post-step values; this refactor reproduces that
    effective behavior. Original author intent is unclear.
    """

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "white_matter_diffusivity",
            "rho",
            "necrosis_rate",
            "nutrient_threshold",
            "nutrient_diffusivity",
            "nutrient_consumption_rate",
            "gray_matter",
            "white_matter",
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

    _gm_low: NDArray
    _wm_low: NDArray
    _gm_cropped: NDArray | None = None
    _wm_cropped: NDArray | None = None
    _nutrient_faces: FaceFields | None = None

    def _validate_params(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        merged = _merge_params(params, self._REQUIRED, self._DEFAULTS, type(self).__name__)
        _validate_tissue_arrays(merged["gray_matter"], merged["white_matter"])
        _validate_seed_fractions(merged)
        return merged

    def _prepare_fields(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        factor = self.params["resolution_factor"]
        self._gm_low = self._downsample(self.params["gray_matter"], factor)
        self._wm_low = self._downsample(self.params["white_matter"], factor)
        self._gm_cropped = None
        self._wm_cropped = None
        self._nutrient_faces = None
        return self._gm_low.shape, self.params["gray_matter"].shape

    def _crop_mask(self) -> NDArray:
        return (self._gm_low + self._wm_low) >= CROP_TISSUE_THRESHOLD

    def _initialize_state(self) -> dict[str, NDArray]:
        proliferative = clipped_gaussian(
            self.grid_shape,
            self.seed_voxel,
            self.grid_spacing,
            scale=self.params["gaussian_seed_scale"],
            diffusion_time=self.params["gaussian_seed_diffusion_time"],
            mass=self.params["gaussian_seed_mass"],
            floor=self.params["gaussian_seed_floor"],
        )
        necrotic = np.zeros(proliferative.shape)
        nutrient = np.ones(proliferative.shape)
        # remove CSF from the nutrient field
        nutrient = np.where(
            self._wm_low + self._gm_low >= self.params["min_tissue_fraction"],
            nutrient,
            0,
        )
        return {
            "proliferative": proliferative,
            "necrotic": necrotic,
            "nutrient": nutrient,
        }

    def _build_diffusivity(self, state: dict[str, NDArray]) -> dict[str, FaceFields]:
        if self._gm_cropped is None:
            assert self._crop_box is not None
            self._gm_cropped = crop(self._gm_low, self._crop_box)
            self._wm_cropped = crop(self._wm_low, self._crop_box)
        gm, wm = self._gm_cropped, self._wm_cropped

        tissue_valid = (wm + gm) >= self.params["min_tissue_fraction"]

        # NOTE(ported aliasing): the per-step rebuild (triggered via
        # _diffusivity_needs_update) receives the POST-step P and N. The
        # original passes what looks like the pre-step P, but its in-place
        # mutation meant that argument already aliased the updated P, so the
        # effective behavior (post-step P and N) is reproduced here. Original
        # author intent is unclear. See the class docstring.
        occupancy_valid = (
            state["proliferative"] + state["necrotic"]
        ) <= self.params["max_tumor_occupancy"]

        tumor_faces = _mixture_face_fields(
            wm,
            gm,
            np.logical_and(tissue_valid, occupancy_valid),
            self.params["white_matter_diffusivity"],
            self.params["diffusivity_ratio"],
        )

        if self._nutrient_faces is None:
            # Built once, as in the original (constant in time; ratio 1 means
            # gray matter conducts nutrient like white matter).
            self._nutrient_faces = _mixture_face_fields(
                wm, gm, tissue_valid, self.params["nutrient_diffusivity"], 1
            )

        return {"tumor": tumor_faces, "nutrient": self._nutrient_faces}

    def _diffusivity_needs_update(self) -> bool:
        # Rebuilt every step, from the post-step fields (see the aliasing
        # note in _build_diffusivity and the class docstring).
        return True

    def _step(
        self,
        state: dict[str, NDArray],
        diffusivity: dict[str, FaceFields],
        dt: float,
    ) -> dict[str, NDArray]:
        proliferative = state["proliferative"]
        necrotic = state["necrotic"]
        nutrient = state["nutrient"]
        rho = self.params["rho"]
        necrosis_rate = self.params["necrosis_rate"]
        consumption_rate = self.params["nutrient_consumption_rate"]

        # Descending switch on the nutrient level (original smooth_heaviside).
        switch = logistic_sigmoid(
            -NECROSIS_SWITCH_STEEPNESS * (nutrient - self.params["nutrient_threshold"])
        )

        # Sequential in-place updates as in the original: the necrotic and
        # nutrient updates below see the already-updated proliferative field.
        sp = diffusion_term(proliferative, diffusivity["tumor"], self.grid_spacing)
        diff_p = (
            sp
            + rho * np.multiply(nutrient, proliferative) * (1 - proliferative - necrotic)
            - necrosis_rate * proliferative * switch
        ) * dt
        proliferative += diff_p

        diff_n = necrosis_rate * proliferative * switch * dt
        necrotic += diff_n

        ss = diffusion_term(nutrient, diffusivity["nutrient"], self.grid_spacing)
        diff_s = (ss - consumption_rate * nutrient * proliferative) * dt
        nutrient += diff_s

        return state

    def _cell_mass(self, state: dict[str, NDArray]) -> float:
        # "mass" stopping mode. The voxel-volume factor multiplies BOTH
        # terms; the original FK_2c dropped it on the necrotic term (mixed
        # units), so this intentionally deviates. See the class docstring.
        return self.voxel_volume * (
            np.sum(state["proliferative"]) + np.sum(state["necrotic"])
        )

    def _cell_density_sum(self, state: dict[str, NDArray]) -> NDArray:
        return state["proliferative"] + state["necrotic"]

    def _time_step_count(self) -> tuple[int, float]:
        stopping_time = self.params["stopping_time"]
        dw = self.params["white_matter_diffusivity"]
        d_s = self.params["nutrient_diffusivity"]
        rho = self.params["rho"]
        dx, dy, dz = self.grid_spacing
        # NOTE: ad-hoc stability formula; the three solvers' formulas remain
        # mutually inconsistent by heritage and may be unified later.
        nt = np.max(
            [
                stopping_time
                * np.max([dw, d_s])
                / np.power(np.min([dx, dy, dz]), 2)
                * self.params["nt_multiplier"]
                + 300,
                # Reaction-rate guard in the same form the FK solver uses.
                # The original FK_2c omitted it; without it, dt can violate
                # the ~1/rho explicit-Euler reaction bound for large rho.
                stopping_time * rho * 1.1,
            ]
        )
        dt = stopping_time / nt
        return int(np.ceil(nt)), dt


class AnisotropicFKPPSolver(BaseFKPPSolver):
    """Axis-wise diffusivity from the DTI tensor diagonal (source: FK_DTI).

    State key: 'cell_density'. The per-axis diffusivity field (shape
    (Nx, Ny, Nz, 3)) is derived from the tensor diagonals; the crop mask and
    the seed guard come from a brainmask thresholded on that field.
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
        "gray_matter": None,
        "white_matter": None,
        "diffusivity_upper_limit": 2,
        "diffusivity_lower_limit": 0,
    }

    _requires_previous_state: ClassVar[bool] = True

    _axial_low: NDArray
    _axial_cropped: NDArray | None = None
    _axial_full_max: float
    _brainmask: NDArray

    def _validate_params(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        merged = _merge_params(params, self._REQUIRED, self._DEFAULTS, type(self).__name__)
        _validate_seed_fractions(merged)
        if merged["uniform_gray_matter"] and (
            merged["gray_matter"] is None or merged["white_matter"] is None
        ):
            raise KeyError(
                "AnisotropicFKPPSolver: uniform_gray_matter=True requires "
                "gray_matter and white_matter"
            )
        return merged

    def _axial_diffusivity_from_tensor(
        self,
        tensor: NDArray,
        exponent: float,
        linear_term: float,
        wm: NDArray | None,
        gm: NDArray | None,
        diffusivity_ratio: float | None,
        normalization_std: float | None,
    ) -> NDArray:
        """Per-axis diffusivity (Nx, Ny, Nz, 3) from the tensor diagonals.

        Replicates the original makeXYZ_rgb_from_tensor normalization and
        clipping logic exactly, in its original operation order (including
        the sequential in-place mean/std normalization, where the std is
        computed on the already mean-shifted field).
        """
        upper_limit = self.params["diffusivity_upper_limit"]
        lower_limit = self.params["diffusivity_lower_limit"]
        output = np.zeros(tensor.shape[:4])

        # use diagonal elements
        output[:, :, :, 0] = tensor[:, :, :, 0, 0]
        output[:, :, :, 1] = tensor[:, :, :, 1, 1]
        output[:, :, :, 2] = tensor[:, :, :, 2, 2]

        output[output < 0] = 0

        brain_mask = np.max(output, axis=-1) > 0

        if wm is not None:
            normalization_mask = wm > 0
        else:
            normalization_mask = brain_mask

        if normalization_std is not None:
            output[brain_mask] -= np.mean(output[normalization_mask])
            output[brain_mask] /= np.std(output[normalization_mask])
            output[brain_mask] *= normalization_std
            output[brain_mask] += 1
        else:
            output[brain_mask] /= np.mean(output[normalization_mask])

        if not (wm is None or gm is None or diffusivity_ratio is None):
            if self.params["verbose"]:
                logger.debug("set gm to uniform and wm to DTI")
            csf_mask = np.logical_and(wm <= 0, gm <= 0)
            output[csf_mask] = 0
            gm_threshold = 1.0 / diffusivity_ratio
            output[gm > 0] = gm_threshold  # fix gray matter
            border_mask = binary_dilation(csf_mask, iterations=1)
            output[border_mask] = 0
            # clip wm to lowest gm
            output[
                np.logical_and(
                    np.repeat((wm > 0)[..., np.newaxis], repeats=3, axis=-1),
                    output < gm_threshold,
                )
            ] = gm_threshold

        output[output < 0] = 0
        output = output**exponent + linear_term * output

        output[output > upper_limit] = upper_limit
        output[output < 0] = 0
        output[
            np.logical_and(
                np.repeat((brain_mask > 0)[..., np.newaxis], repeats=3, axis=-1),
                output < lower_limit,
            )
        ] = lower_limit

        return output

    def _prepare_fields(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        params = self.params
        scaling = params["ellipsoid_scaling"]
        if params["verbose"]:
            logger.debug("ellipsoid_scaling: %s", scaling)
        if scaling == 1:
            tensors = params["diffusion_tensors"]
        else:
            tensors = elongate_tensor_along_principal_axis(
                params["diffusion_tensors"], scaling
            )

        if params["uniform_gray_matter"]:
            axial = self._axial_diffusivity_from_tensor(
                tensors,
                exponent=params["tensor_exponent"],
                linear_term=params["tensor_linear_term"],
                wm=params["white_matter"],
                gm=params["gray_matter"],
                diffusivity_ratio=params["diffusivity_ratio"],
                normalization_std=params["normalization_std"],
            )
        else:
            axial = self._axial_diffusivity_from_tensor(
                tensors,
                exponent=params["tensor_exponent"],
                linear_term=params["tensor_linear_term"],
                wm=None,
                gm=None,
                diffusivity_ratio=None,
                normalization_std=params["normalization_std"],
            )

        assert isinstance(axial, np.ndarray), "sRGB must be a numpy array"
        assert axial.ndim == 4, (
            "sRGB must be a 4D numpy array, with the last dimension being 3 (RGB)"
        )

        factor = params["resolution_factor"]
        low = self._downsample(axial, [factor, factor, factor, 1])
        # NOTE(ported): the original also computed a second, double-zoomed
        # brainmask here at threshold 1e-6 (`brainmask_low_res`) that is never
        # used; that dead computation is dropped.
        low[low <= 0] = 0
        self._axial_low = low
        self._axial_cropped = None
        # The Nt stability formula uses the max of the FULL-resolution field,
        # before downsampling (original behavior).
        self._axial_full_max = np.max(axial)
        self._brainmask = np.max(low, axis=-1) > 0.00001
        return low.shape[:3], axial.shape[:3]

    def _crop_mask(self) -> NDArray:
        return self._brainmask

    def _check_seed_late(self) -> None:
        # The original checks the (uncropped) brainmask inside its try block,
        # after cropping and diffusivity construction.
        if not self._brainmask[self.seed_voxel]:
            raise ValueError("Origin not within brainmask")

    def _initialize_state(self) -> dict[str, NDArray]:
        cell_density = clipped_gaussian(
            self.grid_shape,
            self.seed_voxel,
            self.grid_spacing,
            scale=self.params["gaussian_seed_scale"],
            diffusion_time=self.params["gaussian_seed_diffusion_time"],
            mass=self.params["gaussian_seed_mass"],
            floor=self.params["gaussian_seed_floor"],
        )
        if self.params["verbose"]:
            logger.debug(
                "init: %s, volume of initial tumor: %s",
                cell_density.shape,
                np.sum(cell_density),
            )
        return {"cell_density": cell_density}

    def _build_diffusivity(self, state: dict[str, NDArray]) -> dict[str, FaceFields]:
        if self._axial_cropped is None:
            assert self._crop_box is not None
            self._axial_cropped = crop(self._axial_low, self._crop_box)
        axial = self._axial_cropped
        diffusivity = self.params["diffusivity"]
        faces: FaceFields = {}
        for axis, name in enumerate(_AXES):
            face = face_average(axial[:, :, :, axis], axis)
            faces[f"minus_{name}"] = face * diffusivity
            faces[f"plus_{name}"] = diffusivity * edge_roll(face, 1, axis=axis)
        return {"cell_density": faces}

    def _step(
        self,
        state: dict[str, NDArray],
        diffusivity: dict[str, FaceFields],
        dt: float,
    ) -> dict[str, NDArray]:
        u = state["cell_density"]
        sp = diffusion_term(u, diffusivity["cell_density"], self.grid_spacing)
        diff_u = (sp + logistic_growth(u, self.params["rho"])) * dt
        u += diff_u
        return state

    def _post_step_checks(
        self, state: dict[str, NDArray], previous_state: dict[str, NDArray]
    ) -> str | None:
        u = state["cell_density"]
        total_change = np.sum(u) - np.sum(previous_state["cell_density"])
        # Guard thresholds and check order are unchanged from the original.
        # Unlike the original (which reported a guard exit as a successful
        # "time" stop with final_time == stopping_time), a firing guard is
        # reported as Result(success=False, stopping_criterion="error") with
        # final_time at the actual exit step; see BaseFKPPSolver.
        if total_change < -10:
            return (
                "shrinkage guard fired: step-to-step cell-density sum "
                f"decreased by {-total_change} (> 10)"
            )
        integrated_density = self.voxel_volume * np.sum(u)
        if integrated_density < 0.000001:
            return (
                "vanishing-volume guard fired: integrated cell density "
                f"{integrated_density} < 1e-6"
            )
        return None

    def _cell_mass(self, state: dict[str, NDArray]) -> float:
        # "mass" stopping mode: integrated cell density, reproducing the
        # original DTI stopping quantity exactly. Not a physical volume.
        return self.voxel_volume * np.sum(state["cell_density"])

    def _cell_density_sum(self, state: dict[str, NDArray]) -> NDArray:
        return state["cell_density"]

    def _time_step_count(self) -> tuple[int, float]:
        stopping_time = self.params["stopping_time"]
        dw = self.params["diffusivity"]
        rho = self.params["rho"]
        dx, dy, dz = self.grid_spacing
        # NOTE: ad-hoc stability formula; the three solvers' formulas remain
        # mutually inconsistent by heritage and may be unified later. It
        # scales with the max of the FULL-resolution axial diffusivity field,
        # which is over-conservative (the simulated downsampled field's max
        # is <= the full-res max); deliberately retained from the original.
        nt = np.max(
            [
                stopping_time
                * dw
                * self._axial_full_max
                / np.power(np.min([dx, dy, dz]), 2)
                * 8
                + 100,
                stopping_time * rho * 1.1,
            ]
        )
        dt = stopping_time / nt
        return int(np.ceil(nt)), dt
