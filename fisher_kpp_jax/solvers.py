"""The JAX Fisher-KPP forward solvers: ``FKPPSolver``,
``TwoCompartmentWithNutrientFKPPSolver`` and ``AnisotropicFKPPSolver``,
each with the treatment effects of a Stupp protocol (resection,
chemotherapy, radiotherapy) built in.

A model contributes its update (one explicit Euler step); one wrapper,
``_treated_step``, applies the treatments to either update. The updates and
the wrapper's two bindings are module-level objects with a stable identity,
so the jitted time scan's cache persists across solves (see
``operators._run_time_loop``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, ClassVar, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from .base import (
    TREATMENT_DEFAULTS,
    TREATMENT_VOLUME_KEYS,
    BaseFKPPSolver,
    _SharedConstants,
    _validate_tissue_arrays,
    n_steps_from_dt,  # noqa: F401 - re-exported
)
from .operators import (
    GAUSSIAN_SEED_DIFFUSION_TIME,
    GAUSSIAN_SEED_FLOOR,
    GAUSSIAN_SEED_MASS,
    chemo_exposure,
    diffusion_term,
    elongate_tensor_along_principal_axis,
    face_diffusivities,
    logistic_growth,
    logistic_sigmoid,
)

# Tissue-occupancy threshold defining the crop mask.
CROP_TISSUE_THRESHOLD: float = 0.5

# Steepness of the smooth descending switch on the nutrient level.
NECROSIS_SWITCH_STEEPNESS: float = 50.0

_COMMON_DEFAULTS: dict[str, Any] = {
    "diffusivity_ratio": 10.0,
    # None: from the NIfTI header of the reference volume when it is given
    # as a path, else 1 mm isotropic.
    "voxel_size_mm": None,
    "gaussian_seed_scale": 1.0,
    "gaussian_seed_diffusion_time": GAUSSIAN_SEED_DIFFUSION_TIME,
    "gaussian_seed_mass": GAUSSIAN_SEED_MASS,
    "gaussian_seed_floor": GAUSSIAN_SEED_FLOOR,
    # The horizon in days, at most one of the two (see
    # base._resolve_horizon); neither: base.DEFAULT_STOPPING_TIME.
    "stopping_time": None,
    "time_after_resection": None,  # the horizon is resection_time + it
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
    # The treatment parameters at their neutral values.
    **TREATMENT_DEFAULTS,
}

# The tissue probability maps of the WM/GM mixture solvers; the white
# matter map's NIfTI header provides the voxel size and affine.
_TISSUE_VOLUME_KEYS: frozenset[str] = frozenset({"gray_matter_pbmap", "white_matter_pbmap"})


class _TreatmentConstants(TypedDict, total=False):
    """
    Device constants of a treated run, built by the base solver
    (``BaseFKPPSolver._build_treatment_constants`` and the constants
    assembly in ``_run_device_loop``) and consumed by ``_treated_step``.
    All of them are present in a treated run and none in an untreated one.

    Attributes:
        resection_time: Resection time in days, 0-d scalar at the state
            dtype.
        cavity: Boolean resection-cavity mask on the cropped grid.
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
        post_resection: The model's structural constants
            (``_structural_constants``) built with the cavity removed from
            the valid mask, under the same keys as their pre-resection
            counterparts in the flat dict; every face touching a cavity
            voxel is zero in them.
    """

    resection_time: jax.Array
    cavity: jax.Array
    chemo_times: jax.Array
    chemo_doses: jax.Array
    chemo_kill_rate: jax.Array
    chemo_decay_rate: jax.Array
    rt_times: jax.Array
    rt_log_kill: jax.Array
    post_resection: dict[str, Any]


class _SingleFieldSpecificConstants(TypedDict):
    """
    Solver-specific device constants of the single-field solvers
    (FKPPSolver and AnisotropicFKPPSolver): face_diffusivities is returned
    by their ``_structural_constants``, rho by their
    ``_build_device_constants``.

    Attributes:
        face_diffusivities: Face diffusivity arrays, keys 'fwd_x/y/z' and
            'bwd_x/y/z' (see ``diffusion_term``); constant in time up to
            the switch to the post-resection set of a treated run.
        rho: Proliferation rate, 0-d scalar at the state dtype.
    """

    face_diffusivities: dict[str, jax.Array]
    rho: jax.Array


class _SingleFieldConstants(
    _SharedConstants, _SingleFieldSpecificConstants, _TreatmentConstants
):
    """
    Flat device constants of the single-field solvers, merged by the base
    solver: the ``_SharedConstants`` keys, the
    ``_SingleFieldSpecificConstants`` keys and, in a treated run, the
    ``_TreatmentConstants`` keys in one dict.
    """


class _TwoCompartmentSpecificConstants(TypedDict):
    """
    Solver-specific device constants of
    TwoCompartmentWithNutrientFKPPSolver: tissue_mask and nutrient_faces
    are returned by its ``_structural_constants``, the rest by its
    ``_build_device_constants``.

    Attributes:
        wm: White matter fraction field on the cropped grid.
        gm: Gray matter fraction field on the cropped grid.
        tissue_mask: Boolean mask of cells with enough tissue to carry
            flux; the tumor faces are rebuilt from it every step.
        nutrient_faces: Nutrient face diffusivities, keys 'fwd_x/y/z' and
            'bwd_x/y/z'; constant in time up to the switch to the
            post-resection set of a treated run, like tissue_mask.
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


class _TwoCompartmentConstants(
    _SharedConstants, _TwoCompartmentSpecificConstants, _TreatmentConstants
):
    """
    Flat device constants of TwoCompartmentWithNutrientFKPPSolver, merged
    by the base solver: the ``_SharedConstants`` keys, the
    ``_TwoCompartmentSpecificConstants`` keys and, in a treated run, the
    ``_TreatmentConstants`` keys in one dict.
    """


# --- module-level device functions (stable identity so the jitted time
# --- loop's cache persists across solves; otherwise new parameters would
# --- trigger recompilation. see operators._run_time_scan) ---


def _single_field_update(
    state: dict[str, jax.Array],
    constants: _SingleFieldConstants,
    step_index: jax.Array,
) -> dict[str, jax.Array]:
    """
    Perform one explicit Euler step of the single-field models
    (FKPPSolver and AnisotropicFKPPSolver),
    du/dt = div(D grad u) + rho u (1 - u).

    The face diffusivities come from constants. This is the whole step of
    an untreated run.

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


def _two_compartment_update(
    state: dict[str, jax.Array],
    constants: _TwoCompartmentConstants,
    step_index: jax.Array,
) -> dict[str, jax.Array]:
    """
    Perform one explicit Euler step of the proliferative/necrotic/nutrient
    system. This is the whole step of an untreated run.

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
    tumor_faces = face_diffusivities(
        constants["white_matter_diffusivity"],
        [(constants["wm"], 1), (constants["gm"], constants["diffusivity_ratio"])],
        jnp.logical_and(constants["tissue_mask"], occupancy_valid),
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


def _treated_step(
    state: dict[str, jax.Array],
    constants: Mapping[str, Any],
    step_index: jax.Array,
    *,
    update: Callable[..., dict[str, jax.Array]],
    killed: tuple[str, ...],
    cleared: tuple[str, ...],
    sink: str | None = None,
) -> dict[str, jax.Array]:
    """
    Perform one step of a treated run: the model's update followed by the
    discrete treatment events of the step. The keyword arguments are bound
    per model (``_treated_single_field_step``,
    ``_treated_two_compartment_step``).

    The step interval is (t0, t1] with t0 = step_index dt and
    t1 = (step_index + 1) dt, both computed as products (never
    accumulated), so the intervals partition the horizon exactly at the
    state dtype. In-step operation order:

      1. post = t1 >= resection_time; from that step on the model's
         structural constants are the ``post_resection`` set, in which the
         cavity is removed from the valid mask (no flux across a face
         touching a cavity voxel);
      2. the model's update, the explicit Euler step at the pre-step
         state, on these constants;
      3. for every field in killed, the chemotherapy impulse
         field <- field exp(-chemo_kill_rate E_ct), with
         E_ct = int_{t0}^{t1} C dt the exact drug exposure of the step
         (``chemo_exposure``), so the chemotherapy kill is independent of
         the step size; then the radiotherapy impulse
         field <- field exp(-E(x) n_hits), n_hits = number of rt_times in
         (t0, t1] (exact impulse map, not part of the Euler right-hand
         side). The two factors are deliberately applied one after the
         other in this order, never combined into one. With a sink, the
         killed cells of every killed field, the field before the two
         factors minus the field after them, are added to the sink field
         (they become necrotic in the two-compartment model); without
         one they leave the system;
      4. for every field in cleared, the resection projection
         field <- 0 inside the cavity, for every step with post
         (idempotent), so that the cavity is empty at the end of every
         post-resection step.

    Every treatment term is evaluated in every step; the function has no
    branch. A run whose treatment values are all neutral never gets here:
    the base solver compiles the model's update alone for it.

    Args:
        state: The model's state dict.
        constants: Device inputs: the model's flat constants with the
            ``_TreatmentConstants`` keys.
        step_index: Scan step index, 0-d int32.
        update: The model's update.
        killed: State keys the chemotherapy and radiotherapy impulses act
            on.
        cleared: State keys the resection projection empties inside the
            cavity.
        sink: State key that receives the killed cells of every killed
            field, or None when they leave the system. The sink must not
            be a killed field.

    Returns:
        The stepped state.
    """
    dt = constants["dt"]
    t0 = step_index.astype(dt.dtype) * dt
    t1 = (step_index + 1).astype(dt.dtype) * dt

    post = t1 >= constants["resection_time"]
    post_resection = constants["post_resection"]
    pre_resection = {key: constants[key] for key in post_resection}
    structural = jax.tree_util.tree_map(
        lambda pre_value, post_value: jnp.where(post, post_value, pre_value),
        pre_resection,
        post_resection,
    )
    new_state = dict(update(state, {**constants, **structural}, step_index))

    exposure = chemo_exposure(
        t0,
        t1,
        constants["chemo_times"],
        constants["chemo_doses"],
        constants["chemo_decay_rate"],
    )
    chemo_survival = jnp.exp(-constants["chemo_kill_rate"] * exposure)
    rt_times = constants["rt_times"]
    n_hits = jnp.sum(jnp.logical_and(rt_times > t0, rt_times <= t1)).astype(dt.dtype)
    rt_survival = jnp.exp(-constants["rt_log_kill"] * n_hits)
    for key in killed:
        before = new_state[key]
        after = before * chemo_survival
        after = after * rt_survival
        new_state[key] = after
        if sink is not None:
            new_state[sink] = new_state[sink] + (before - after)

    resected = jnp.logical_and(post, constants["cavity"])
    for key in cleared:
        new_state[key] = jnp.where(resected, 0, new_state[key])
    return new_state


# The treated step of the single-field models (FKPPSolver and
# AnisotropicFKPPSolver): the impulses and the projection act on the cell
# density.
_treated_single_field_step = partial(
    _treated_step,
    update=_single_field_update,
    killed=("cell_density",),
    cleared=("cell_density",),
)

# The treated step of the two-compartment model. Killed cells become
# necrotic (decided 2026-10-01, reversing the earlier choice that they
# vanish): the chemotherapy and radiotherapy impulses act on the
# proliferative field and the killed cells, the proliferative field before
# the two factors minus the field after them, are added to the necrotic
# field. Consequences: the model has no necrotic clearance, so P + N is
# conserved under a kill and decreases only through the resection; the
# mass and volume stopping quantities and the occupancy mask (both on
# P + N) see the resection only, and the killed tissue keeps suppressing
# regrowth through the logistic factor (1 - P - N). Only the proliferative
# field responds to chemotherapy and radiotherapy, which is why the
# patient scripts score the post-op sessions against the enhancing
# tumour alone. The nutrient is not touched by a kill, and the
# consumption of the same step used the pre-kill proliferative field. The
# resection projection empties all three fields inside the cavity, and the
# post-resection tissue mask and nutrient faces block the tumor and the
# nutrient flux across the cavity boundary, so the cavity behaves like
# CSF.
_treated_two_compartment_step = partial(
    _treated_step,
    update=_two_compartment_update,
    killed=("proliferative",),
    cleared=("proliferative", "necrotic", "nutrient"),
    sink="necrotic",
)


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


class FKPPSolver(BaseFKPPSolver):
    """
    Isotropic Fisher-KPP solver on WM/GM tissue maps, with the treatment
    effects of a Stupp protocol: surgical resection, chemotherapy (CT) and
    radiotherapy (RT).

    State key: 'cell_density' (u in [0, 1]); grid in mm, time in days
    with the seed at t = 0. Diffusivity is a WM/GM mixture,
    D = white_matter_diffusivity * (wm_face + gm_face / diffusivity_ratio),
    with faces masked by min_tissue_fraction, built once on the device.

    Continuous model (explicit Euler at the pre-step state; growth and
    diffusion only)::

        du/dt = div(D grad u) + rho u (1 - u)

    Treatment parameters (``TREATMENT_KEYS``, the same for every solver).
    None of them is required: each defaults to a neutral value
    (resection_time inf, no resection_cavity, no chemo_times and
    chemo_doses, chemo_kill_rate 0, no rt_times, no rt_dose, rt_alpha 0),
    and a parameter given as None takes its default. A treatment is
    switched off by its values, not by omitting a key: an all-False (or
    no) resection_cavity leaves the dynamics untouched, an empty
    chemo_times, chemo_kill_rate = 0 or all-zero chemo_doses removes the
    chemotherapy kill, and a zero (or no) rt_dose, or rt_alpha = 0 (which
    zeroes the derived rt_beta with it), makes the radiotherapy impulse
    the identity. chemo_decay_rate (default 9.24 per day) must stay > 0
    and rt_alpha_beta_ratio (default 10 Gy) positive; both are inert
    without sessions or dose. With all three treatments neutral the run
    is the untreated run exactly: the solver compiles the update of the
    continuous model alone and builds no treatment volume (see
    ``BaseFKPPSolver._is_treated``). A non-empty resection_cavity needs a
    finite resection_time.

    The horizon is given as ``stopping_time`` or, relative to the
    surgery, as ``time_after_resection`` (the run then ends at
    resection_time + time_after_resection), at most one of the two;
    ``params['stopping_time']`` holds the horizon after construction.

    In a config the treatment volumes are NIfTI paths like the tissue
    maps: ``rt_dose`` (Gy, TOTAL over all fractions) directly, and
    ``resection_cavity`` as ``{"segmentation": <NIfTI path>, "label":
    <int>}``, the cavity being the voxels carrying that label (values
    rounded to the nearest integer first). Both may also be given as
    arrays, on the grid of the tissue maps.

    Chemotherapy acts through the drug concentration C(t), in which each
    session j at chemo_times[j] deposits its dose chemo_doses[j] (mg/m^2)
    that decays exponentially::

        C(t) = sum_j chemo_doses[j] [t >= chemo_times[j]]
                     exp(-chemo_decay_rate (t - chemo_times[j]))

    Discrete events, applied after the Euler update of the step whose
    interval (t0, t1] contains them, in this order (see
    ``_treated_step``):

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
         (computed once on the host; zero without fractions).
      3. Resection: u <- 0 inside resection_cavity for every step with
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
    )
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        **_COMMON_DEFAULTS,
        # Cells with wm + gm below this carry no flux (CSF/background).
        "min_tissue_fraction": 0.1,
    }
    _VOLUME_KEYS: ClassVar[frozenset[str]] = _TISSUE_VOLUME_KEYS | TREATMENT_VOLUME_KEYS
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "white_matter_pbmap"

    # static methods allows passing of stable module level functions as attributes
    _step_func = staticmethod(_single_field_update)
    _treated_step_func = staticmethod(_treated_single_field_step)
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

    def _valid_mask_host(self, box: tuple[slice, slice, slice]) -> NDArray:
        # Host-side float64 tissue mask, identical for both precisions.
        return (self._wm_lowres[box] + self._gm_lowres[box]) >= float(
            self.params["min_tissue_fraction"]
        )

    def _structural_constants(
        self, box: tuple[slice, slice, slice], valid_mask_host: NDArray
    ) -> dict[str, Any]:
        gm = jnp.asarray(self._gm_lowres[box], dtype=self._dtype)
        wm = jnp.asarray(self._wm_lowres[box], dtype=self._dtype)
        faces = face_diffusivities(
            float(self.params["white_matter_diffusivity"]),
            [(wm, 1), (gm, float(self.params["diffusivity_ratio"]))],
            jnp.asarray(valid_mask_host),
        )
        return {"face_diffusivities": faces}

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> dict[str, Any]:
        return {"rho": self._dynamic_scalar(self.params["rho"])}

    def _time_step_count(self) -> tuple[int, float]:
        # The treatment events are impulse maps and do not constrain dt.
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
    (see ``_two_compartment_update`` for the update-order semantics). The
    nutrient diffuses with nutrient_diffusivity, masked by tissue only,
    built once.

    The "mass" stopping quantity deliberately applies the voxel-volume
    factor to the necrotic term as well -- see ``_mass_two_compartment``.

    Treatment: the parameters, their neutral defaults, the horizon and
    the event order are those of ``FKPPSolver`` (see its docstring), the
    treatment volumes on the grid of the tissue maps. The chemotherapy
    and radiotherapy impulses act on the proliferative field and the
    killed cells become necrotic: they are moved into the necrotic field,
    so P + N is conserved under a kill and the mass and volume stopping
    quantities see the resection only (see
    ``_treated_two_compartment_step``). From the resection on, the
    projection empties the proliferative, the necrotic and the nutrient
    field inside the cavity in every step, and the cavity is removed from
    the tissue mask of the tumor faces and of the nutrient faces, so no
    tumor or nutrient flux crosses the cavity boundary: the cavity behaves
    like CSF.
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
    _VOLUME_KEYS: ClassVar[frozenset[str]] = _TISSUE_VOLUME_KEYS | TREATMENT_VOLUME_KEYS
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "white_matter_pbmap"

    _step_func = staticmethod(_two_compartment_update)
    _treated_step_func = staticmethod(_treated_two_compartment_step)
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

    def _valid_mask_host(self, box: tuple[slice, slice, slice]) -> NDArray:
        # Validity mask, computed host-side in float64 so it is identical
        # for both precisions.
        return (self._wm_lowres[box] + self._gm_lowres[box]) >= float(
            self.params["min_tissue_fraction"]
        )

    def _structural_constants(
        self, box: tuple[slice, slice, slice], valid_mask_host: NDArray
    ) -> dict[str, Any]:
        gm = jnp.asarray(self._gm_lowres[box], dtype=self._dtype)
        wm = jnp.asarray(self._wm_lowres[box], dtype=self._dtype)
        tissue_mask = jnp.asarray(valid_mask_host)
        # Nutrient faces are built once per mask (divisor 1 means gray
        # matter conducts nutrient like white matter). The tumor faces are
        # rebuilt every step inside the update, from tissue_mask.
        nutrient_faces = face_diffusivities(
            float(self.params["nutrient_diffusivity"]), [(wm, 1), (gm, 1)], tissue_mask
        )
        return {"tissue_mask": tissue_mask, "nutrient_faces": nutrient_faces}

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> dict[str, Any]:
        scalar = self._dynamic_scalar
        params = self.params
        return {
            # The fields of the per-step tumor-face rebuild.
            "wm": jnp.asarray(self._wm_lowres[box], dtype=self._dtype),
            "gm": jnp.asarray(self._gm_lowres[box], dtype=self._dtype),
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
    crop mask and the seed check come from a brain mask thresholded on that
    field. Every cell may carry flux (the field is zero outside the brain),
    so the faces of an untreated run are plain face averages.

    Treatment: the parameters, their neutral defaults, the horizon and
    the event order are those of ``FKPPSolver`` (see its docstring), the
    treatment volumes on the grid of the first three dimensions of the
    tensor field. The post-resection faces are zero wherever they touch a
    cavity voxel. A run whose whole tumor lies inside the cavity is left
    with a zero field after the resection and completes normally with the
    stopping criterion "time".
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
    _VOLUME_KEYS: ClassVar[frozenset[str]] = (
        _TISSUE_VOLUME_KEYS | {"diffusion_tensors"} | TREATMENT_VOLUME_KEYS
    )
    _REFERENCE_VOLUME_KEY: ClassVar[str] = "diffusion_tensors"
    _step_func = staticmethod(_single_field_update)
    _treated_step_func = staticmethod(_treated_single_field_step)
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

    def _valid_mask_host(self, box: tuple[slice, slice, slice]) -> NDArray:
        # Every cell may carry flux: the per-axis field is zero outside
        # the brain, so the faces need no mask of their own.
        return np.ones(self._axial_lowres[box].shape[:3], dtype=bool)

    def _structural_constants(
        self, box: tuple[slice, slice, slice], valid_mask_host: NDArray
    ) -> dict[str, Any]:
        axial = jnp.asarray(self._axial_lowres[box], dtype=self._dtype)
        faces = face_diffusivities(
            float(self.params["diffusivity"]), [(axial, 1)], jnp.asarray(valid_mask_host)
        )
        return {"face_diffusivities": faces}

    def _build_device_constants(
        self, box: tuple[slice, slice, slice]
    ) -> dict[str, Any]:
        return {"rho": self._dynamic_scalar(self.params["rho"])}

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
