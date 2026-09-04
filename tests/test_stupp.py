"""Behavior tests of the treatment-extended ``StuppFKPPSolver`` and its
config entries, self-contained.

Short f64 solves on the 24^3 tissue phantom checking (a) that the solver
reduces to ``FKPPSolver`` with neutral treatment values (every treatment
parameter is required; ``neutral_treatment_params`` switches the three
treatments off through their values), (b) closed forms of the three
treatment effects in limits where they decouple from growth/diffusion (RT
impulse under mass-conserving diffusion, the exact chemotherapy exposure
without growth and diffusion, resection projection and cavity isolation),
(c) the jit-cache and precision
behavior, (d) parameter validation and (e) the JSON config with the
treatment volumes as NIfTI paths (the cavity as a labelled segmentation).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import nibabel as nib
import numpy as np
import pytest
from loguru import logger

from fisher_kpp_jax import (
    SOLVER_KEY,
    FKPPSolver,
    StuppFKPPSolver,
    operators,
    read_config,
    solver_from_config,
    write_config,
)

_COMMON = dict(
    gaussian_seed_x_fraction=0.5,
    gaussian_seed_y_fraction=0.5,
    gaussian_seed_z_fraction=0.5,
    resolution_factor=0.6,
    precision="f64",
)
# Horizon of the Stupp runs: resection at day 2 + 8 days = 10 days.
HORIZON = 10.0

# f32 vs f64 tolerance on the final field of a short treated solve, matching
# the untreated test_solvers.py budget (observed max-abs difference ~1e-4).
F32_ATOL = 2e-3


def neutral_treatment_params(shape: tuple[int, int, int], **overrides) -> dict:
    """All three treatments present but switched off by their values: an
    empty cavity, a zero chemotherapy kill rate, a zero dose. The event
    times lie inside the 10-day horizon so that nothing warns."""
    params = dict(
        resection_time=2.0,
        time_after_resection=HORIZON - 2.0,
        resection_cavity=np.zeros(shape, dtype=bool),
        chemo_times=np.array([1.0, 3.0, 5.0]),
        chemo_doses=np.array([75.0, 75.0, 75.0]),
        chemo_kill_rate=0.0,
        chemo_decay_rate=0.5,
        rt_times=np.array([3.3, 5.3, 7.3]),
        rt_dose=np.zeros(shape),
        rt_alpha=0.1,
        rt_beta=0.01,
    )
    params.update(overrides)
    return params


def base_params(gm: np.ndarray, wm: np.ndarray, **overrides) -> dict:
    """Untreated run: the shared solver parameters plus neutral treatments."""
    params = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        gray_matter_pbmap=gm,
        white_matter_pbmap=wm,
        **_COMMON,
        **neutral_treatment_params(gm.shape),
    )
    params.update(overrides)
    return params


def off_center_cavity(shape: tuple[int, int, int]) -> np.ndarray:
    """Cube cavity overlapping one side of the centered seed."""
    cavity = np.zeros(shape, dtype=bool)
    n = shape[0]
    c = n // 2
    cavity[c - 1 : c + 3, c - 2 : c + 2, c - 2 : c + 2] = True
    return cavity


def treatment_params(shape: tuple[int, int, int], **overrides) -> dict:
    """All three treatments enabled with events inside a 10-day horizon."""
    params = dict(
        resection_time=2.0,
        time_after_resection=HORIZON - 2.0,
        resection_cavity=off_center_cavity(shape),
        chemo_times=np.array([1.0, 3.0, 5.0]),
        chemo_doses=np.array([75.0, 150.0, 200.0]),
        chemo_kill_rate=0.1 / 75,  # per mg/m^2: a 75 mg/m^2 session kills as 0.1 per unit
        chemo_decay_rate=0.5,
        rt_times=np.array([3.3, 5.3, 7.3]),
        rt_dose=np.full(shape, 6.0),
        rt_alpha=0.1,
        rt_beta=0.01,
    )
    params.update(overrides)
    return params


def lowres_seed_mass(solver: StuppFKPPSolver) -> float:
    """voxel_volume * sum of the seed on the low-resolution grid (the mass
    the stopping quantity starts from), available after a solve."""
    with jax.enable_x64():  # the seed is evaluated at the state dtype
        seed = np.asarray(solver._gaussian_seed(), dtype=np.float64)
    return solver.voxel_volume * float(seed.sum())


def test_neutral_treatment_equals_fkpp(tissue_phantom):
    """With every treatment switched off by its values the solver
    reproduces FKPPSolver: the same step count and final time, and the same
    fields up to rounding (the zero-valued treatment terms are still
    evaluated, so the compiled arithmetic is not identical; the observed
    difference is ~1e-16 relative)."""
    gm, wm = tissue_phantom
    params = base_params(gm, wm, snapshot_times=[3.0, 6.0, 10.0])
    untreated = {k: v for k, v in params.items() if k not in neutral_treatment_params(gm.shape)}
    reference = FKPPSolver({**untreated, "stopping_time": HORIZON}).solve()
    result = StuppFKPPSolver(params).solve()
    assert reference.success and result.success, (reference.error, result.error)
    assert result.final_time == reference.final_time
    np.testing.assert_allclose(
        result.final_stopping_quantity, reference.final_stopping_quantity, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.final_state["cell_density"],
        reference.final_state["cell_density"],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.time_series["cell_density"],
        reference.time_series["cell_density"],
        rtol=0,
        atol=1e-14,
    )


def test_rt_impulse_is_exact(tissue_phantom):
    """Uniform total dose, rho=0, D>0: zero-flux diffusion conserves mass,
    so the final mass is exactly the initial mass times exp(-K E) with K the
    number of fractions inside the horizon and E the per-fraction
    linear-quadratic log kill (d = total dose / number of fractions)."""
    gm, wm = tissue_phantom
    total_dose, alpha, beta = 6.0, 0.1, 0.01
    rt_times = np.array([2.5, 4.5, 6.5, 8.5, 12.0])  # 4 inside the horizon
    params = base_params(
        gm,
        wm,
        rho=0.0,
        rt_times=rt_times,
        rt_dose=np.full(gm.shape, total_dose),
        rt_alpha=alpha,
        rt_beta=beta,
    )
    solver = StuppFKPPSolver(params)
    result = solver.solve()
    assert result.success, result.error
    initial_mass = lowres_seed_mass(solver)
    dose_per_fraction = total_dose / rt_times.size
    log_kill = alpha * dose_per_fraction + beta * dose_per_fraction**2
    n_hits = int((rt_times <= HORIZON).sum())
    assert n_hits == 4
    np.testing.assert_allclose(
        result.final_stopping_quantity,
        initial_mass * np.exp(-n_hits * log_kill),
        rtol=1e-12,
    )
    # The same run without RT conserves the mass (the diffusion is the only
    # other term), which is what makes the identity above exact.
    untreated = StuppFKPPSolver(base_params(gm, wm, rho=0.0)).solve()
    np.testing.assert_allclose(
        untreated.final_stopping_quantity, initial_mass, rtol=1e-12
    )


def _ct_only_params(gm, wm, n_steps: int, doses: np.ndarray, **overrides) -> dict:
    """Chemotherapy alone: no growth, no diffusion, off-grid session times."""
    return base_params(
        gm,
        wm,
        rho=0.0,
        white_matter_diffusivity=0.0,
        n_steps=n_steps,
        chemo_times=np.array([1.37, 3.0, 5.91]),
        chemo_doses=doses,
        chemo_kill_rate=0.3 / 75,
        chemo_decay_rate=0.5,
        **overrides,
    )


def _ct_closed_form_survival(params: dict) -> float:
    """exp(-k sum_j d_j (1 - exp(-lambda (T - t_j)^+)) / lambda) over the horizon."""
    times = np.asarray(params["chemo_times"], dtype=np.float64)
    doses = np.asarray(params["chemo_doses"], dtype=np.float64)
    decay = float(params["chemo_decay_rate"])
    elapsed = np.maximum(HORIZON - times, 0.0)
    return float(np.exp(-params["chemo_kill_rate"] * np.sum(doses * (1 - np.exp(-decay * elapsed))) / decay))


@pytest.mark.parametrize("n_steps", [100, 200, 400])
def test_ct_is_exact_and_step_independent(tissue_phantom, n_steps):
    """rho=0, D=0: the per-step exposure is integrated in closed form, so
    the final mass equals m0 exp(-k sum_j d_j (1 - exp(-lambda (T - t_j)))
    / lambda) for every step count (session times off the step grid), and
    the step counts agree with each other."""
    gm, wm = tissue_phantom
    doses = np.array([75.0, 150.0, 200.0])
    results = {}
    for steps in (100, 200, 400):
        solver = StuppFKPPSolver(_ct_only_params(gm, wm, steps, doses))
        result = solver.solve()
        assert result.success, result.error
        results[steps] = (result.final_stopping_quantity, lowres_seed_mass(solver))
    final, m0 = results[n_steps]
    np.testing.assert_allclose(
        final, m0 * _ct_closed_form_survival(_ct_only_params(gm, wm, n_steps, doses)), rtol=1e-10
    )
    for other, (other_final, _) in results.items():
        np.testing.assert_allclose(final, other_final, rtol=1e-12)


def test_ct_dose_scales_log_kill(tissue_phantom):
    """Doubling every dose squares the survival factor final / m0."""
    gm, wm = tissue_phantom
    doses = np.array([75.0, 150.0, 200.0])
    survival = []
    for factor in (1.0, 2.0):
        solver = StuppFKPPSolver(_ct_only_params(gm, wm, 200, factor * doses))
        result = solver.solve()
        assert result.success, result.error
        survival.append(result.final_stopping_quantity / lowres_seed_mass(solver))
    assert survival[0] < 1.0
    np.testing.assert_allclose(survival[1], survival[0] ** 2, rtol=1e-10)


def test_resection_projection_and_isolation(tissue_phantom):
    """At resolution_factor=1 the recorded frames sit on the input grid, so
    the cavity mask applies to them directly: the density is identically
    zero inside the cavity at every snapshot at or after resection_time and
    in the final state. With rho=0 and resection at t=0 the cavity faces
    carry no flux and the total mass is conserved from the first step on
    (but is below the seed mass, which the projection reduced)."""
    gm, wm = tissue_phantom
    cavity = off_center_cavity(gm.shape)
    params = base_params(
        gm,
        wm,
        rho=0.0,
        resolution_factor=1.0,
        resection_time=0.0,
        time_after_resection=HORIZON,
        resection_cavity=cavity,
        snapshot_times=np.linspace(0, HORIZON, 6),
    )
    solver = StuppFKPPSolver(params)
    result = solver.solve()
    assert result.success, result.error
    frames = result.time_series["cell_density"]
    times = result.snapshot_times
    assert frames.shape[0] == times.size
    for frame, t in zip(frames, times):
        assert t >= params["resection_time"]
        assert np.all(frame[cavity] == 0.0)
    assert np.all(result.final_state["cell_density"][cavity] == 0.0)
    # The seed overlaps the cavity, so the projection removed some mass...
    seed_mass = lowres_seed_mass(solver)
    masses = frames.sum(axis=(1, 2, 3)) * solver.voxel_volume
    assert masses[0] < seed_mass
    # ...and nothing is lost afterwards.
    np.testing.assert_allclose(masses, masses[0], rtol=1e-12)
    np.testing.assert_allclose(result.final_stopping_quantity, masses[0], rtol=1e-12)


def test_resection_mid_run(tissue_phantom):
    """Snapshots before resection_time may be nonzero in the cavity; those
    at or after it are zero there."""
    gm, wm = tissue_phantom
    cavity = off_center_cavity(gm.shape)
    params = base_params(
        gm,
        wm,
        resolution_factor=1.0,
        resection_time=5.0,
        time_after_resection=HORIZON - 5.0,
        resection_cavity=cavity,
        snapshot_times=np.linspace(0, HORIZON, 8),
    )
    solver = StuppFKPPSolver(params)
    result = solver.solve()
    assert result.success, result.error
    frames = result.time_series["cell_density"]
    times = result.snapshot_times
    before = times < params["resection_time"]
    assert before.any() and (~before).any()
    assert np.any(frames[before][:, cavity] > 0)
    assert np.all(frames[~before][:, cavity] == 0.0)
    assert np.all(result.final_state["cell_density"][cavity] == 0.0)


def test_events_beyond_horizon_are_inert(tissue_phantom):
    """Chemotherapy and radiotherapy whose event times all lie beyond the
    horizon leave the result identical to the untreated run (the
    concentration is exactly zero, no fraction is hit). The resection can
    never lie beyond the horizon, which is measured from it."""
    gm, wm = tissue_phantom
    full = treatment_params(gm.shape)
    late = {
        "chemo_times": np.array([20.0, 30.0]),
        "chemo_doses": np.array([75.0, 150.0]),
        "chemo_kill_rate": full["chemo_kill_rate"],
        "rt_times": np.array([25.0, 26.0]),
        "rt_dose": full["rt_dose"],
    }
    params = base_params(gm, wm, snapshot_times=[3.0, 6.0, 10.0])
    reference = StuppFKPPSolver(params).solve()
    result = StuppFKPPSolver({**params, **late}).solve()
    assert reference.success and result.success
    assert result.final_time == reference.final_time
    np.testing.assert_allclose(
        result.final_state["cell_density"],
        reference.final_state["cell_density"],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.final_stopping_quantity, reference.final_stopping_quantity, rtol=1e-14
    )


def test_late_event_times_warn(tissue_phantom):
    """A warning (independent of verbose) names event times that lie beyond
    stopping_time; captured with a temporary loguru sink."""
    gm, wm = tissue_phantom
    params = base_params(gm, wm, **treatment_params(gm.shape, rt_times=np.array([3.0, 40.0])))
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        StuppFKPPSolver(params)
    finally:
        logger.remove(handler_id)
    assert any("rt_times" in m and "never fire" in m and "40.0" in m for m in messages)


def test_no_retrace_on_treatment_value_change(tissue_phantom):
    """Treatment VALUES (times, chemo doses, rates, RT dose, alpha/beta,
    cavity mask) are traced device arguments: re-solves that change only
    them, with the
    same treatment configuration and the same number of session times,
    reuse the compiled scan (operators.SCAN_TRACE_COUNT is a trace-time
    counter)."""
    gm, wm = tissue_phantom
    treated = treatment_params(gm.shape)
    params = base_params(gm, wm, **treated)
    StuppFKPPSolver(params).solve()  # warm-up: may trace (once) if not cached
    before = operators.SCAN_TRACE_COUNT
    StuppFKPPSolver(params).solve()
    other_cavity = np.roll(treated["resection_cavity"], 2, axis=0)
    StuppFKPPSolver(
        {
            **params,
            "rho": 0.16,
            "resection_time": 4.0,
            "time_after_resection": HORIZON - 4.0,  # keep the horizon (step count)
            "resection_cavity": other_cavity,
            "chemo_times": np.array([0.5, 2.5, 6.5]),
            "chemo_doses": np.array([80.0, 160.0, 210.0]),
            "chemo_kill_rate": 0.2 / 75,
            "chemo_decay_rate": 0.7,
            "rt_times": np.array([2.2, 4.2, 6.2]),
            "rt_dose": np.full(gm.shape, 9.0),
            "rt_alpha": 0.2,
            "rt_beta": 0.02,
        }
    ).solve()
    assert operators.SCAN_TRACE_COUNT == before


def test_f32_vs_f64_agreement(tissue_phantom):
    gm, wm = tissue_phantom
    params = base_params(gm, wm, **treatment_params(gm.shape))
    res32 = StuppFKPPSolver({**params, "precision": "f32"}).solve()
    res64 = StuppFKPPSolver({**params, "precision": "f64"}).solve()
    assert res32.success and res64.success, (res32.error, res64.error)
    assert res32.stopping_criterion == res64.stopping_criterion
    assert res32.final_time == res64.final_time
    np.testing.assert_allclose(
        res32.final_stopping_quantity, res64.final_stopping_quantity, rtol=1e-3
    )
    np.testing.assert_allclose(
        res32.final_state["cell_density"],
        res64.final_state["cell_density"],
        atol=F32_ATOL,
        rtol=0,
    )


def test_validation_errors(tissue_phantom):
    gm, wm = tissue_phantom
    full = treatment_params(gm.shape)

    def build(**treatment):
        return StuppFKPPSolver(base_params(gm, wm, **treatment))

    build(**full)  # the complete set is accepted
    # Every treatment parameter is required; None is not a way to disable one.
    for key in ("resection_cavity", "chemo_decay_rate", "rt_beta"):
        with pytest.raises(KeyError, match=key):
            StuppFKPPSolver({k: v for k, v in base_params(gm, wm).items() if k != key})
        with pytest.raises(ValueError, match=key):
            build(**{**full, key: None})
    # Wrong shapes / kinds.
    with pytest.raises(ValueError, match="resection_cavity"):
        build(**{**full, "resection_cavity": np.zeros((4, 4, 4), dtype=bool)})
    with pytest.raises(ValueError, match="binary"):
        build(**{**full, "resection_cavity": np.full(gm.shape, 0.5)})
    with pytest.raises(ValueError, match="rt_dose"):
        build(**{**full, "rt_dose": np.zeros((4, 4))})
    with pytest.raises(ValueError, match="rt_dose"):
        build(**{**full, "rt_dose": np.full(gm.shape, -1.0)})
    with pytest.raises(ValueError, match="rt_dose"):
        build(**{**full, "rt_dose": [[1.0]]})
    with pytest.raises(ValueError, match="1-D"):
        build(**{**full, "chemo_times": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="nonnegative"):
        build(**{**full, "chemo_times": np.array([-1.0, 2.0])})
    with pytest.raises(ValueError, match="rt_times"):
        build(**{**full, "rt_times": np.array([np.inf])})
    with pytest.raises(ValueError, match="at least one"):
        build(**{**full, "rt_times": np.array([])})
    with pytest.raises(ValueError, match="chemo_kill_rate"):
        build(**{**full, "chemo_kill_rate": -0.1})
    # Chemo doses: one finite nonnegative dose per session.
    with pytest.raises(ValueError, match="chemo_doses"):
        build(**{**full, "chemo_doses": np.array([75.0, 150.0])})
    with pytest.raises(ValueError, match="chemo_doses"):
        build(**{**full, "chemo_doses": np.array([75.0, -1.0, 200.0])})
    with pytest.raises(ValueError, match="chemo_doses"):
        build(**{**full, "chemo_doses": np.zeros((3, 1))})
    with pytest.raises(ValueError, match="chemo_doses"):
        build(**{**full, "chemo_doses": np.array([75.0, np.inf, 200.0])})
    # The decay rate divides the exposure: zero is no longer an "off" value.
    with pytest.raises(ValueError, match="chemo_decay_rate"):
        build(**{**full, "chemo_decay_rate": 0.0})
    # Chemotherapy is switched off by the kill rate or by the doses.
    build(**{**full, "chemo_kill_rate": 0.0})
    build(**{**full, "chemo_doses": np.zeros(3)})
    with pytest.raises(ValueError, match="rt_alpha"):
        build(**{**full, "rt_alpha": np.nan})
    with pytest.raises(ValueError, match="resection_time"):
        build(**{**full, "resection_time": -1.0})
    with pytest.raises(ValueError, match="time_after_resection"):
        build(**{**full, "time_after_resection": -1.0})
    with pytest.raises(ValueError, match="unknown"):
        build(**{**full, "rt_gamma": 1.0})
    # The horizon is time_after_resection only; stopping_time is not accepted.
    with pytest.raises(ValueError, match="stopping_time"):
        build(**{**full, "stopping_time": 10.0})
    solver = build(**full)
    assert solver.params["stopping_time"] == full["resection_time"] + full["time_after_resection"]


# --- configs ---


def _write_nifti(path: Path, data: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return str(path)


def _write_config(tmp_path: Path, config: dict, name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def config_dir(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """tmp dir with a label map (labels 0..4) and a dose volume."""
    rng = np.random.default_rng(3)
    segmentation = rng.integers(0, 5, size=(6, 7, 8)).astype(np.int16)
    dose = rng.random((6, 7, 8)) * 60.0
    _write_nifti(tmp_path / "seg.nii.gz", segmentation)
    _write_nifti(tmp_path / "dose.nii.gz", dose.astype(np.float32))
    return tmp_path, segmentation, dose.astype(np.float32).astype(np.float64)


def test_treatment_keys():
    """TREATMENT_KEYS are exactly the parameters FKPPSolver does not have
    (besides time_after_resection), so dropping them gives an FKPPSolver run."""
    stupp = StuppFKPPSolver._REQUIRED | set(StuppFKPPSolver._DEFAULTS)
    fkpp = FKPPSolver._REQUIRED | set(FKPPSolver._DEFAULTS)
    assert StuppFKPPSolver.TREATMENT_KEYS == stupp - fkpp - {"time_after_resection"}
    assert set(neutral_treatment_params((2, 2, 2))) == StuppFKPPSolver.TREATMENT_KEYS | {
        "time_after_resection"
    }


def test_read_config_cavity_entry(config_dir):
    """read_config keeps the treatment volumes as paths: the cavity entry
    is checked to be {"segmentation": path, "label": int} and its path
    made absolute (relative to the config's directory) but not loaded."""
    tmp_path, _, _ = config_dir
    config = {
        SOLVER_KEY: "StuppFKPPSolver",
        "_note": "ignored",
        "white_matter_pbmap": "seg.nii.gz",
        "rho": 0.12,
        "chemo_times": [24, 25, 26],
        "resection_cavity": {"segmentation": str(tmp_path / "seg.nii.gz"), "label": 4.0},
        "rt_dose": "dose.nii.gz",
        "dt": 0.1,
        "stopping_threshold": "inf",
    }
    entries = read_config(_write_config(tmp_path, config))
    assert entries == {
        SOLVER_KEY: "StuppFKPPSolver",
        "white_matter_pbmap": str(tmp_path / "seg.nii.gz"),
        "rho": 0.12,
        "chemo_times": [24, 25, 26],
        "resection_cavity": {"segmentation": str(tmp_path / "seg.nii.gz"), "label": 4},
        "rt_dose": str(tmp_path / "dose.nii.gz"),
        "dt": 0.1,
        "stopping_threshold": np.inf,
    }
    # A missing volume file is only reported on loading; a null volume stays.
    stupp = {SOLVER_KEY: "StuppFKPPSolver"}
    assert read_config(_write_config(tmp_path, {**stupp, "rt_dose": "nope.nii.gz"}))["rt_dose"].endswith(
        "nope.nii.gz"
    )
    assert read_config(_write_config(tmp_path, {**stupp, "resection_cavity": None}))["resection_cavity"] is None
    with pytest.raises(ValueError, match="unknown key.*stopping_time"):
        read_config(_write_config(tmp_path, {**stupp, "stopping_time": 5}))
    with pytest.raises(ValueError, match="rt_dose must be a NIfTI path"):
        read_config(_write_config(tmp_path, {**stupp, "rt_dose": 60.0}))
    for cavity in ("seg.nii.gz", {"segmentation": "seg.nii.gz"}, {"path": "x", "label": 4}):
        with pytest.raises(ValueError, match="resection_cavity must be an object"):
            read_config(_write_config(tmp_path, {**stupp, "resection_cavity": cavity}))


def test_cavity_from_segmentation(config_dir, tissue_phantom):
    """The solver loads a cavity entry as segmentation == label (a bool
    array), whether it comes from a config file or is passed directly;
    the dose map is loaded like the tissue maps."""
    tmp_path, _, _ = config_dir
    gm, wm = tissue_phantom
    segmentation = np.zeros(gm.shape, dtype=np.int16)
    segmentation[off_center_cavity(gm.shape)] = 4
    segmentation[0, 0, 0] = 3
    seg_path = _write_nifti(tmp_path / "seg24.nii.gz", segmentation)
    dose_path = _write_nifti(tmp_path / "dose24.nii.gz", np.full(gm.shape, 6.0, dtype=np.float32))
    params = base_params(
        gm,
        wm,
        **treatment_params(
            gm.shape,
            resection_cavity={"segmentation": seg_path, "label": 4},
            rt_dose=dose_path,
        ),
    )
    solver = StuppFKPPSolver(params)
    cavity = solver.params["resection_cavity"]
    assert cavity.dtype == bool
    np.testing.assert_array_equal(cavity, off_center_cavity(gm.shape))
    np.testing.assert_array_equal(solver.params["rt_dose"], 6.0)
    assert solver.config["resection_cavity"] == {"segmentation": seg_path, "label": 4}
    assert solver.config["rt_dose"] == dose_path
    assert solver.config["white_matter_pbmap"] == "<in-memory>"
    with pytest.raises(FileNotFoundError, match="nope.nii.gz"):
        StuppFKPPSolver({**params, "resection_cavity": {"segmentation": "nope.nii.gz", "label": 1}})
    with pytest.raises(ValueError, match="resection_cavity must be an array or"):
        StuppFKPPSolver({**params, "resection_cavity": {"segmentation": seg_path}})


def test_config_drives_solver(config_dir, tissue_phantom):
    """A complete config file is a complete StuppFKPPSolver run: dt becomes
    n_steps with the horizon resection_time + time_after_resection, the
    derived stopping_time stays out of the config, and the written config
    reads back equal. The solver's own validation reports config
    mistakes."""
    tmp_path, _, _ = config_dir
    gm, wm = tissue_phantom
    segmentation = np.zeros(gm.shape, dtype=np.int16)
    segmentation[off_center_cavity(gm.shape)] = 4
    _write_nifti(tmp_path / "wm24.nii.gz", wm)
    _write_nifti(tmp_path / "gm24.nii.gz", gm)
    _write_nifti(tmp_path / "seg24.nii.gz", segmentation)
    _write_nifti(tmp_path / "dose24.nii.gz", np.full(gm.shape, 6.0, dtype=np.float32))
    config = {
        SOLVER_KEY: "StuppFKPPSolver",
        "_note": "the phantom run of base_params / treatment_params",
        "white_matter_pbmap": "wm24.nii.gz",
        "gray_matter_pbmap": "gm24.nii.gz",
        "white_matter_diffusivity": 0.3,
        "rho": 0.15,
        **_COMMON,
        "resection_time": 2.0,
        "time_after_resection": HORIZON - 2.0,
        "resection_cavity": {"segmentation": "seg24.nii.gz", "label": 4},
        "chemo_times": [3.3, 5.3],
        "chemo_doses": [75.0, 150.0],
        "chemo_kill_rate": 0.1 / 75,
        "chemo_decay_rate": 0.5,
        "rt_times": [3.3, 5.3],
        "rt_dose": "dose24.nii.gz",
        "rt_alpha": 0.1,
        "rt_beta": 0.01,
        "dt": 0.05,
    }
    entries = read_config(_write_config(tmp_path, config))
    solver = solver_from_config(entries)
    assert isinstance(solver, StuppFKPPSolver)
    assert solver.resolve_time_stepping() == (200, 0.05)
    assert solver.params["stopping_time"] == HORIZON and "stopping_time" not in solver.config
    assert solver.config["dt"] == 0.05 and solver.config["n_steps"] is None
    result = solver.solve()
    assert result.success, result.error
    assert result.final_time == HORIZON and result.n_steps == 200
    assert result.derived == {"stopping_time": HORIZON, "voxel_size_mm": (1.0, 1.0, 1.0)}
    untreated = StuppFKPPSolver(base_params(gm, wm)).solve()
    assert result.final_stopping_quantity < untreated.final_stopping_quantity
    written = write_config(solver.config, tmp_path / "written.json")
    assert read_config(written) == solver.config
    # The solver's own validation reports config mistakes: a dose list of
    # the wrong length, a missing required parameter.
    mismatched = read_config(_write_config(tmp_path, {**config, "chemo_doses": [75.0]}))
    with pytest.raises(ValueError, match="chemo_doses"):
        solver_from_config(mismatched)
    incomplete = read_config(_write_config(tmp_path, {**config, "rt_alpha": None}))
    with pytest.raises(ValueError, match="rt_alpha"):
        solver_from_config(incomplete)
