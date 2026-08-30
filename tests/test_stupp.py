"""Behavior tests of the treatment-extended ``StuppFKPPSolver`` and its
manifest loader, self-contained.

Short f64 solves on the 24^3 tissue phantom checking (a) that the solver
reduces to ``FKPPSolver`` without treatments, (b) closed forms of the three
treatment effects in limits where they decouple from growth/diffusion (RT
impulse under mass-conserving diffusion, chemotherapy under pure reaction,
resection projection and cavity isolation), (c) the jit-cache and precision
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
from fisher_kpp_jax.solvers import treatment_params_from_manifest

_COMMON = dict(
    gaussian_seed_x_fraction=0.5,
    gaussian_seed_y_fraction=0.5,
    gaussian_seed_z_fraction=0.5,
    resolution_factor=0.6,
    stopping_time=10,
    precision="f64",
)

# f32 vs f64 tolerance on the final field of a short treated solve, matching
# the untreated test_solvers.py budget (observed max-abs difference ~1e-4).
F32_ATOL = 2e-3


def base_params(gm: np.ndarray, wm: np.ndarray, **overrides) -> dict:
    params = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        gray_matter_pbmap=gm,
        white_matter_pbmap=wm,
        **_COMMON,
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
        resection_cavity=off_center_cavity(shape),
        chemo_times=np.array([1.0, 3.0, 5.0]),
        chemo_kill_rate=0.1,
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


def test_no_treatment_equals_fkpp(tissue_phantom):
    """With every treatment disabled the treatment branches are absent from
    the trace and the solver reproduces FKPPSolver bit-exactly."""
    gm, wm = tissue_phantom
    params = base_params(gm, wm, n_time_series_snapshots=3)
    reference = FKPPSolver(params).solve()
    result = StuppFKPPSolver(params).solve()
    assert reference.success and result.success, (reference.error, result.error)
    assert result.final_time == reference.final_time
    assert result.final_stopping_quantity == reference.final_stopping_quantity
    np.testing.assert_array_equal(
        result.final_state["cell_density"], reference.final_state["cell_density"]
    )
    np.testing.assert_array_equal(
        result.time_series["cell_density"], reference.time_series["cell_density"]
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
    n_hits = int((rt_times <= params["stopping_time"]).sum())
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


def _chemo_concentration_numpy(t, times, decay):
    elapsed = t - times
    return np.where(elapsed >= 0, np.exp(-decay * elapsed), 0.0).sum()


@pytest.mark.parametrize("n_steps", [100, 200, 400])
def test_ct_matches_euler_product_and_closed_form(tissue_phantom, n_steps):
    """rho=0, D=0: each step scales every voxel by 1 - dt k C(t0), so the
    final mass is the initial mass times the product of these factors
    exactly (f64), and approximates the closed form
    m0 exp(-k sum_j (1 - exp(-decay (T - t_j)_+)) / decay) with an O(dt)
    relative error; the bound below is verified empirically
    (test_ct_error_is_first_order)."""
    gm, wm = tissue_phantom
    times = np.array([1.0, 3.0, 5.0])
    kill_rate, decay_rate, horizon = 0.3, 0.5, 10.0
    params = base_params(
        gm,
        wm,
        rho=0.0,
        white_matter_diffusivity=0.0,
        stopping_time=horizon,
        n_steps=n_steps,
        chemo_times=times,
        chemo_kill_rate=kill_rate,
        chemo_decay_rate=decay_rate,
    )
    solver = StuppFKPPSolver(params)
    result = solver.solve()
    assert result.success, result.error
    m0 = lowres_seed_mass(solver)
    dt = horizon / n_steps
    factors = [
        1 - dt * kill_rate * _chemo_concentration_numpy(s * dt, times, decay_rate)
        for s in range(n_steps)
    ]
    np.testing.assert_allclose(
        result.final_stopping_quantity, m0 * np.prod(factors), rtol=1e-12
    )
    exposure = ((1 - np.exp(-decay_rate * np.maximum(horizon - times, 0))) / decay_rate).sum()
    closed_form = m0 * np.exp(-kill_rate * exposure)
    rel_error = abs(result.final_stopping_quantity - closed_form) / closed_form
    assert rel_error < 0.8 * dt  # empirically ~0.64 * dt for this schedule


def test_ct_error_is_first_order(tissue_phantom):
    """Doubling the step count roughly halves the closed-form error."""
    gm, wm = tissue_phantom
    times = np.array([1.0, 3.0, 5.0])
    kill_rate, decay_rate, horizon = 0.3, 0.5, 10.0
    exposure = ((1 - np.exp(-decay_rate * (horizon - times))) / decay_rate).sum()
    errors = []
    for n_steps in (100, 200, 400):
        solver = StuppFKPPSolver(
            base_params(
                gm,
                wm,
                rho=0.0,
                white_matter_diffusivity=0.0,
                stopping_time=horizon,
                n_steps=n_steps,
                chemo_times=times,
                chemo_kill_rate=kill_rate,
                chemo_decay_rate=decay_rate,
            )
        )
        result = solver.solve()
        closed_form = lowres_seed_mass(solver) * np.exp(-kill_rate * exposure)
        errors.append(abs(result.final_stopping_quantity - closed_form) / closed_form)
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 1.7) and np.all(ratios < 2.3), errors


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
    """Treatments whose event times all lie beyond stopping_time leave the
    result identical to the untreated run (the concentration is exactly
    zero, no fraction is hit, the resection never activates)."""
    gm, wm = tissue_phantom
    late = treatment_params(
        gm.shape,
        resection_time=50.0,
        chemo_times=np.array([20.0, 30.0]),
        rt_times=np.array([25.0, 26.0]),
    )
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
    """Treatment VALUES (times, rates, dose, alpha/beta, cavity mask) are
    traced device arguments: re-solves that change only them, with the
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
            "resection_cavity": other_cavity,
            "chemo_times": np.array([0.5, 2.5, 6.5]),
            "chemo_kill_rate": 0.2,
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
    # Partial treatment groups name the missing keys.
    with pytest.raises(ValueError, match="resection_cavity"):
        build(resection_time=2.0)
    with pytest.raises(ValueError, match="chemo_decay_rate"):
        build(chemo_times=full["chemo_times"], chemo_kill_rate=0.1)
    with pytest.raises(ValueError, match="rt_beta"):
        build(**{k: full[k] for k in ("rt_times", "rt_dose", "rt_alpha")})
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
    with pytest.raises(ValueError, match="rt_alpha"):
        build(**{**full, "rt_alpha": np.nan})
    with pytest.raises(ValueError, match="resection_time"):
        build(**{**full, "resection_time": -1.0})
    with pytest.raises(ValueError, match="unknown"):
        build(**{**full, "rt_gamma": 1.0})


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
        "chemotherapy": {"times": [24, 25, 26], "kill_rate": 0.05, "decay_rate": 1.0},
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
    assert params["chemo_kill_rate"] == 0.05 and params["chemo_decay_rate"] == 1.0
    np.testing.assert_array_equal(params["rt_times"], [24.0, 25.0])
    np.testing.assert_array_equal(params["rt_dose"], dose)
    assert params["rt_alpha"] == 0.1 and params["rt_beta"] == 0.01


def test_manifest_loader_partial_sections(manifest_dir):
    tmp_path, _, _ = manifest_dir
    manifest = {"chemotherapy": {"times": [1.0], "kill_rate": 0.1, "decay_rate": 0.5}}
    params = treatment_params_from_manifest(_write_manifest(tmp_path, manifest))
    assert set(params) == {"chemo_times", "chemo_kill_rate", "chemo_decay_rate"}
    assert treatment_params_from_manifest(_write_manifest(tmp_path, {"_only": 1})) == {}


def test_manifest_loader_errors(manifest_dir):
    tmp_path, _, _ = manifest_dir
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        treatment_params_from_manifest(tmp_path / "missing.json")
    missing_volume = {
        "resection": {
            "time": 1,
            "tumor_segmentation": str(tmp_path / "nope.nii.gz"),
            "cavity_label": 4,
        }
    }
    with pytest.raises(FileNotFoundError, match="nope.nii.gz"):
        treatment_params_from_manifest(_write_manifest(tmp_path, missing_volume))
    with pytest.raises(ValueError, match="unknown top-level"):
        treatment_params_from_manifest(_write_manifest(tmp_path, {"surgery": {}}))
    with pytest.raises(ValueError, match="unknown key"):
        treatment_params_from_manifest(
            _write_manifest(
                tmp_path,
                {"chemotherapy": {"times": [1], "kill_rate": 1, "decay_rate": 1, "dose": 2}},
            )
        )
    with pytest.raises(ValueError, match="missing"):
        treatment_params_from_manifest(
            _write_manifest(tmp_path, {"radiotherapy": {"times": [1], "alpha": 0.1}})
        )
    with pytest.raises(ValueError, match="JSON object"):
        treatment_params_from_manifest(_write_manifest(tmp_path, {"resection": [1, 2]}))


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
        "radiotherapy": {"times": [3.3, 5.3], "dose": "dose24.nii.gz", "alpha": 0.1, "beta": 0.01},
    }
    treatment = treatment_params_from_manifest(_write_manifest(tmp_path, manifest))
    result = StuppFKPPSolver(base_params(gm, wm, **treatment)).solve()
    assert result.success, result.error
    untreated = StuppFKPPSolver(base_params(gm, wm)).solve()
    assert result.final_stopping_quantity < untreated.final_stopping_quantity
