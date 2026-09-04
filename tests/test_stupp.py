"""Behavior tests of the treatment-extended ``StuppFKPPSolver`` and its
manifest loader, self-contained.

Short f64 solves on the 24^3 tissue phantom checking (a) that the solver
reduces to ``FKPPSolver`` with neutral treatment values (every treatment
parameter is required; ``neutral_treatment_params`` switches the three
treatments off through their values), (b) closed forms of the three
treatment effects in limits where they decouple from growth/diffusion (RT
impulse under mass-conserving diffusion, the exact chemotherapy exposure
without growth and diffusion, resection projection and cavity isolation),
(c) the jit-cache and precision
behavior, (d) parameter validation and (e) the JSON manifest loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import nibabel as nib
import numpy as np
import pytest
from loguru import logger

from fisher_kpp_jax import FKPPSolver, StuppFKPPSolver, operators
from fisher_kpp_jax.solvers import (
    n_steps_from_dt,
    resolve_time_step,
    solver_params_from_manifest,
    tissue_paths_from_manifest,
    treatment_params_from_manifest,
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


def snapshot_times(solver: StuppFKPPSolver, n_snapshots: int) -> np.ndarray:
    """Simulation times of the recorded frames: a frame scheduled at step s
    holds the state after that step, i.e. at (s + 1) * dt."""
    n_steps, dt = solver._resolve_time_stepping()
    steps = np.unique(np.linspace(0, n_steps - 1, n_snapshots, dtype=np.int64))
    return (steps + 1) * dt


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
    params = base_params(gm, wm, n_time_series_snapshots=3)
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
    n_snapshots = 6
    params = base_params(
        gm,
        wm,
        rho=0.0,
        resolution_factor=1.0,
        resection_time=0.0,
        time_after_resection=HORIZON,
        resection_cavity=cavity,
        n_time_series_snapshots=n_snapshots,
    )
    solver = StuppFKPPSolver(params)
    result = solver.solve()
    assert result.success, result.error
    frames = result.time_series["cell_density"]
    times = snapshot_times(solver, n_snapshots)
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
    n_snapshots = 8
    params = base_params(
        gm,
        wm,
        resolution_factor=1.0,
        resection_time=5.0,
        time_after_resection=HORIZON - 5.0,
        resection_cavity=cavity,
        n_time_series_snapshots=n_snapshots,
    )
    solver = StuppFKPPSolver(params)
    result = solver.solve()
    assert result.success, result.error
    frames = result.time_series["cell_density"]
    times = snapshot_times(solver, n_snapshots)
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
    params = base_params(gm, wm, n_time_series_snapshots=3)
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


# --- manifest loader ---


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def manifest_dir(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """tmp dir with a label map (labels 0..4) and a dose volume."""
    rng = np.random.default_rng(3)
    segmentation = rng.integers(0, 5, size=(6, 7, 8)).astype(np.int16)
    dose = rng.random((6, 7, 8)) * 60.0
    _write_nifti(tmp_path / "seg.nii.gz", segmentation)
    _write_nifti(tmp_path / "dose.nii.gz", dose.astype(np.float32))
    return tmp_path, segmentation, dose.astype(np.float32).astype(np.float64)


def test_manifest_loader_full(manifest_dir):
    tmp_path, segmentation, dose = manifest_dir
    manifest = {
        "_note": "ignored",
        "resection": {
            "_comment": "ignored",
            "time": 10,
            "tumor_segmentation": str(tmp_path / "seg.nii.gz"),
            "cavity_label": 4,
        },
        "chemotherapy": {
            "times": [24, 25, 26],
            "doses": [75, 150, 200],
            "kill_rate": 0.05,
            "decay_rate": 1.0,
        },
        "radiotherapy": {
            "times": [24.0, 25.0],
            "dose": "dose.nii.gz",  # relative to the manifest directory
            "alpha": 0.1,
            "beta": 0.01,
            "_units": "ignored",
        },
    }
    params = treatment_params_from_manifest(_write_manifest(tmp_path, manifest))
    assert set(params) == {
        "resection_time",
        "resection_cavity",
        "chemo_times",
        "chemo_doses",
        "chemo_kill_rate",
        "chemo_decay_rate",
        "rt_times",
        "rt_dose",
        "rt_alpha",
        "rt_beta",
    }
    assert params["resection_time"] == 10.0
    assert params["resection_cavity"].dtype == bool
    np.testing.assert_array_equal(params["resection_cavity"], segmentation == 4)
    assert params["resection_cavity"].any() and not params["resection_cavity"].all()
    np.testing.assert_array_equal(params["chemo_times"], [24.0, 25.0, 26.0])
    assert params["chemo_times"].dtype == np.float64
    np.testing.assert_array_equal(params["chemo_doses"], [75.0, 150.0, 200.0])
    assert params["chemo_doses"].dtype == np.float64
    assert params["chemo_kill_rate"] == 0.05 and params["chemo_decay_rate"] == 1.0
    np.testing.assert_array_equal(params["rt_times"], [24.0, 25.0])
    np.testing.assert_array_equal(params["rt_dose"], dose)
    assert params["rt_alpha"] == 0.1 and params["rt_beta"] == 0.01


def test_manifest_loader_requires_all_sections(manifest_dir):
    tmp_path, _, _ = manifest_dir
    manifest = {
        "chemotherapy": {"times": [1.0], "doses": [75.0], "kill_rate": 0.1, "decay_rate": 0.5}
    }
    with pytest.raises(ValueError, match=r"missing treatment section.*resection.*radiotherapy"):
        treatment_params_from_manifest(_write_manifest(tmp_path, manifest))
    with pytest.raises(ValueError, match="missing treatment section"):
        treatment_params_from_manifest(_write_manifest(tmp_path, {"_only": 1}))


def test_manifest_loader_errors(manifest_dir):
    tmp_path, _, _ = manifest_dir
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        treatment_params_from_manifest(tmp_path / "missing.json")
    full = {
        "resection": {"time": 1, "tumor_segmentation": "seg.nii.gz", "cavity_label": 4},
        "chemotherapy": {"times": [1], "doses": [75], "kill_rate": 1, "decay_rate": 1},
        "radiotherapy": {"times": [1], "dose": "dose.nii.gz", "alpha": 0.1, "beta": 0.01},
    }
    treatment_params_from_manifest(_write_manifest(tmp_path, full))  # accepted
    with pytest.raises(ValueError, match=r"missing key\(s\).*doses"):
        treatment_params_from_manifest(
            _write_manifest(
                tmp_path,
                {**full, "chemotherapy": {"times": [1], "kill_rate": 1, "decay_rate": 1}},
            )
        )
    missing_volume = {
        **full,
        "resection": {**full["resection"], "tumor_segmentation": str(tmp_path / "nope.nii.gz")},
    }
    with pytest.raises(FileNotFoundError, match="nope.nii.gz"):
        treatment_params_from_manifest(_write_manifest(tmp_path, missing_volume))
    with pytest.raises(ValueError, match="unknown top-level"):
        treatment_params_from_manifest(_write_manifest(tmp_path, {**full, "surgery": {}}))
    with pytest.raises(ValueError, match="unknown key"):
        treatment_params_from_manifest(
            _write_manifest(
                tmp_path, {**full, "chemotherapy": {**full["chemotherapy"], "dose": 2}}
            )
        )
    with pytest.raises(ValueError, match="missing key"):
        treatment_params_from_manifest(
            _write_manifest(tmp_path, {**full, "radiotherapy": {"times": [1], "alpha": 0.1}})
        )
    with pytest.raises(ValueError, match="JSON object"):
        treatment_params_from_manifest(_write_manifest(tmp_path, {**full, "resection": [1, 2]}))


def test_manifest_params_drive_solver(manifest_dir, tissue_phantom):
    """The loader output merges straight into the solver params."""
    tmp_path, _, _ = manifest_dir
    gm, wm = tissue_phantom
    segmentation = np.zeros(gm.shape, dtype=np.int16)
    segmentation[off_center_cavity(gm.shape)] = 4
    _write_nifti(tmp_path / "seg24.nii.gz", segmentation)
    _write_nifti(tmp_path / "dose24.nii.gz", np.full(gm.shape, 6.0, dtype=np.float32))
    manifest = {
        "resection": {"time": 2.0, "tumor_segmentation": "seg24.nii.gz", "cavity_label": 4},
        "chemotherapy": {
            "times": [3.3, 5.3],
            "doses": [75.0, 150.0],
            "kill_rate": 0.1 / 75,
            "decay_rate": 0.5,
        },
        "radiotherapy": {"times": [3.3, 5.3], "dose": "dose24.nii.gz", "alpha": 0.1, "beta": 0.01},
    }
    treatment = treatment_params_from_manifest(_write_manifest(tmp_path, manifest))
    result = StuppFKPPSolver(base_params(gm, wm, **treatment)).solve()
    assert result.success, result.error
    untreated = StuppFKPPSolver(base_params(gm, wm)).solve()
    assert result.final_stopping_quantity < untreated.final_stopping_quantity
    # A dose list of the wrong length passes the loader and fails in the solver.
    mismatched = {**manifest, "chemotherapy": {**manifest["chemotherapy"], "doses": [75.0]}}
    treatment = treatment_params_from_manifest(_write_manifest(tmp_path, mismatched))
    with pytest.raises(ValueError, match="chemo_doses"):
        StuppFKPPSolver(base_params(gm, wm, **treatment))


def test_manifest_solver_and_tissue_sections(manifest_dir):
    """The optional 'solver' section maps 1:1 to scalar solver params (dt
    becomes n_steps with the horizon resection time + time_after_resection,
    voxel_size_mm a tuple, "inf" a float); 'tissue' gives
    the resolved pbmap paths; both are ignored by the treatment loader."""
    tmp_path, _, _ = manifest_dir
    manifest = {
        "tissue": {"wm": "seg.nii.gz", "gm": str(tmp_path / "dose.nii.gz")},
        "solver": {
            "_units": "ignored",
            "rho": 0.12,
            "white_matter_diffusivity": 0.4,
            "gaussian_seed_x_fraction": 0.4,
            "time_after_resection": 10.0,
            "dt": 0.1,
            "voxel_size_mm": [1.0, 1.5, 2.0],
            "stopping_threshold": "inf",
            "precision": "f64",
        },
        "resection": {"time": 10.0, "tumor_segmentation": "seg.nii.gz", "cavity_label": 4},
        "chemotherapy": {"times": [1.0], "doses": [75.0], "kill_rate": 0.1, "decay_rate": 0.5},
        "radiotherapy": {"times": [12.0], "dose": "dose.nii.gz", "alpha": 0.1, "beta": 0.01},
    }
    path = _write_manifest(tmp_path, manifest)
    solver = solver_params_from_manifest(path)
    assert solver == {
        "rho": 0.12,
        "white_matter_diffusivity": 0.4,
        "gaussian_seed_x_fraction": 0.4,
        "time_after_resection": 10.0,
        "n_steps": 200,
        "voxel_size_mm": (1.0, 1.5, 2.0),
        "stopping_threshold": np.inf,
        "precision": "f64",
    }
    assert tissue_paths_from_manifest(path) == {
        "wm": tmp_path / "seg.nii.gz",
        "gm": tmp_path / "dose.nii.gz",
    }
    assert set(treatment_params_from_manifest(path)) == {
        "resection_time",
        "resection_cavity",
        "chemo_times",
        "chemo_doses",
        "chemo_kill_rate",
        "chemo_decay_rate",
        "rt_times",
        "rt_dose",
        "rt_alpha",
        "rt_beta",
    }
    # Absent 'solver' / 'tissue' sections give empty results.
    minimal = _write_manifest(tmp_path, {"_only": 1})
    assert solver_params_from_manifest(minimal) == {}
    assert tissue_paths_from_manifest(minimal) == {}


def test_manifest_solver_section_drives_solver(manifest_dir, tissue_phantom):
    gm, wm = tissue_phantom
    tmp_path, _, _ = manifest_dir
    path = _write_manifest(
        tmp_path,
        {
            "resection": {"time": 2.0, "tumor_segmentation": "seg.nii.gz", "cavity_label": 4},
            "solver": {
                "rho": 0.0,
                "white_matter_diffusivity": 0.3,
                "time_after_resection": 3.0,
                "dt": 0.25,
            },
        },
    )
    params = {**base_params(gm, wm), **solver_params_from_manifest(path)}
    solver = StuppFKPPSolver(params)
    assert solver.params["n_steps"] == 20 and solver.params["rho"] == 0.0
    assert solver.params["stopping_time"] == 5.0
    result = solver.solve()
    assert result.success and result.final_time == 5


def test_manifest_solver_section_errors(manifest_dir):
    tmp_path, _, _ = manifest_dir
    with pytest.raises(ValueError, match="unknown key"):
        solver_params_from_manifest(_write_manifest(tmp_path, {"solver": {"rho": 1, "D": 2}}))
    with pytest.raises(ValueError, match="rt_alpha"):  # treatment keys have their own sections
        solver_params_from_manifest(_write_manifest(tmp_path, {"solver": {"rt_alpha": 0.1}}))
    with pytest.raises(ValueError, match="at most one"):
        solver_params_from_manifest(
            _write_manifest(
                tmp_path, {"solver": {"time_after_resection": 1, "dt": 0.1, "n_steps": 5}}
            )
        )
    with pytest.raises(ValueError, match="at most one"):
        solver_params_from_manifest(
            _write_manifest(
                tmp_path, {"solver": {"time_after_resection": 1, "dt": 0.1, "steps_per_day": 3}}
            )
        )
    with pytest.raises(ValueError, match="requires 'time_after_resection'"):
        solver_params_from_manifest(_write_manifest(tmp_path, {"solver": {"dt": 0.1}}))
    with pytest.raises(ValueError, match="requires a resection time"):
        solver_params_from_manifest(
            _write_manifest(tmp_path, {"solver": {"time_after_resection": 1, "dt": 0.1}})
        )
    with pytest.raises(ValueError, match="steps_per_day"):
        solver_params_from_manifest(
            _write_manifest(
                tmp_path, {"solver": {"time_after_resection": 1, "steps_per_day": 0}}
            )
        )
    with pytest.raises(ValueError, match="unknown key.*stopping_time"):
        solver_params_from_manifest(_write_manifest(tmp_path, {"solver": {"stopping_time": 5}}))
    with pytest.raises(FileNotFoundError, match="wm pbmap"):
        tissue_paths_from_manifest(
            _write_manifest(tmp_path, {"tissue": {"wm": "nope.nii.gz", "gm": "seg.nii.gz"}})
        )
    with pytest.raises(ValueError, match="missing"):
        tissue_paths_from_manifest(_write_manifest(tmp_path, {"tissue": {"wm": "seg.nii.gz"}}))


def test_n_steps_from_dt():
    assert n_steps_from_dt(100.0, 0.05) == 2000
    assert n_steps_from_dt(10, 0.1) == 100  # 10/0.1 is not exactly 100 in floating point
    assert n_steps_from_dt(100.0, 0.3) == 334  # rounded up, effective dt < 0.3
    with pytest.raises(ValueError):
        n_steps_from_dt(100.0, 0.0)
    with pytest.raises(ValueError):
        n_steps_from_dt(0.0, 0.1)


def test_steps_per_day_and_unresolved_manifest(manifest_dir):
    """steps_per_day is dt = 1 / steps_per_day over the horizon resection
    time + time_after_resection; with resolve=False the loader hands the
    time-step keys through for a later ``resolve_time_step`` on the merged
    params."""
    tmp_path, _, _ = manifest_dir
    resection = {"time": 10.0, "tumor_segmentation": "seg.nii.gz", "cavity_label": 4}
    path = _write_manifest(
        tmp_path,
        {"resection": resection, "solver": {"time_after_resection": 190.0, "steps_per_day": 3}},
    )
    assert solver_params_from_manifest(path) == {"time_after_resection": 190.0, "n_steps": 600}
    raw = solver_params_from_manifest(path, resolve=False)
    assert raw == {"time_after_resection": 190.0, "steps_per_day": 3}
    assert resolve_time_step({**raw, "time_after_resection": 40.0}, 10.0) == {
        "time_after_resection": 40.0,
        "n_steps": 150,
    }
    assert resolve_time_step({"rho": 1.0}, None) == {"rho": 1.0}
    assert resolve_time_step({"n_steps": 7}, None) == {"n_steps": 7}
    # Without a time step to translate no resection time is needed.
    assert solver_params_from_manifest(
        _write_manifest(tmp_path, {"solver": {"time_after_resection": 1}})
    ) == {"time_after_resection": 1}
    with pytest.raises(ValueError, match="at most one"):
        resolve_time_step({"time_after_resection": 1.0, "n_steps": 7, "dt": 0.1}, 2.0)
    with pytest.raises(ValueError, match="requires 'time_after_resection'"):
        resolve_time_step({"steps_per_day": 3}, 2.0)
    with pytest.raises(ValueError, match="requires a resection time"):
        resolve_time_step({"time_after_resection": 1.0, "steps_per_day": 3}, None)
