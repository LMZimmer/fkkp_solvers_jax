"""End-to-end solver behavior tests for fisher_kpp_jax, self-contained.

Short solves on 24^3 phantoms, checking the documented Result semantics
(stopping criteria, final_time bookkeeping, time-series recording, error
paths) and physical invariants of the models (growth under positive rho,
mass conservation of pure diffusion via the monotonicity checks, necrotic
accumulation, nutrient consumption). ``scripts/run_reference_solves.py``
checks that the reference results are matched; that is not covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fisher_kpp_jax import (
    AnisotropicFKPPSolver,
    FKPPSolver,
    TwoCompartmentWithNutrientFKPPSolver,
)
from fisher_kpp_jax import operators

# f64 keeps the invariant checks tight; the f32 path is covered by
# test_f32_vs_f64_agreement and the (default-precision) retrace test.
_COMMON = dict(
    gaussian_seed_x_fraction=0.5,
    gaussian_seed_y_fraction=0.5,
    gaussian_seed_z_fraction=0.5,
    resolution_factor=0.6,
    stopping_time=10,
    precision="f64",
)

# Loose tolerance for f32 vs f64 on a short solve (a few hundred explicit
# Euler steps of O(1) fields): observed max-abs difference is ~1e-4; assert
# 2e-3 max-abs on the final field and 1e-3 relative on the stopping quantity.
F32_ATOL = 2e-3


def fk_params(gm: np.ndarray, wm: np.ndarray, **overrides) -> dict:
    params = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        gray_matter_pbmap=gm,
        white_matter_pbmap=wm,
        **_COMMON,
    )
    params.update(overrides)
    return params


def two_compartment_params(gm: np.ndarray, wm: np.ndarray, **overrides) -> dict:
    params = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        necrosis_rate=0.4,
        nutrient_threshold=0.4,
        nutrient_diffusivity=0.5,
        nutrient_consumption_rate=0.1,
        gray_matter_pbmap=gm,
        white_matter_pbmap=wm,
        **_COMMON,
    )
    params.update(overrides)
    return params


def dti_params(tensors: np.ndarray, **overrides) -> dict:
    params = dict(
        diffusivity=0.3,
        rho=0.15,
        diffusion_tensors=tensors,
        **_COMMON,
    )
    params.update(overrides)
    return params


# Solver class + params builder over the (tissue_phantom, tensor_phantom)
# fixtures, keyed by the parametrize id.
SOLVER_CASES = {
    "fk": (FKPPSolver, lambda tissue, tensors: fk_params(*tissue)),
    "two_compartment": (
        TwoCompartmentWithNutrientFKPPSolver,
        lambda tissue, tensors: two_compartment_params(*tissue),
    ),
    "dti": (AnisotropicFKPPSolver, lambda tissue, tensors: dti_params(tensors)),
}


def assert_successful_time_solve(result, state_keys: set[str], full_shape: tuple):
    """Common checks for a solve that runs to stopping_time."""
    assert result.success, result.error
    assert result.stopping_criterion == "time"
    assert result.error is None
    assert set(result.final_state) == state_keys
    assert set(result.initial_state) == state_keys
    for key in state_keys:
        assert result.initial_state[key].shape == full_shape
        assert result.final_state[key].shape == full_shape
    assert result.final_stopping_quantity > 0


def test_fkpp_short_solve(tissue_phantom):
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, n_time_series_snapshots=4)
    result = FKPPSolver(params).solve()
    assert_successful_time_solve(result, {"cell_density"}, gm.shape)
    assert result.final_time == params["stopping_time"]
    # Positive rho: the tumor grows.
    assert (
        result.final_state["cell_density"].sum()
        > result.initial_state["cell_density"].sum()
    )
    frames = result.time_series["cell_density"]
    assert frames.shape == (4, *gm.shape)
    masses = frames.sum(axis=(1, 2, 3))
    assert np.all(np.diff(masses) > 0)  # strictly growing between snapshots


def test_two_compartment_short_solve(tissue_phantom):
    gm, wm = tissue_phantom
    params = two_compartment_params(gm, wm, n_time_series_snapshots=3)
    result = TwoCompartmentWithNutrientFKPPSolver(params).solve()
    keys = {"proliferative", "necrotic", "nutrient"}
    assert_successful_time_solve(result, keys, gm.shape)
    # Total tumor burden (P + N) grows under positive rho.
    def tumor_burden(state) -> float:
        return state["proliferative"].sum() + state["necrotic"].sum()

    assert tumor_burden(result.final_state) > tumor_burden(result.initial_state)
    # Necrosis only accumulates: the necrotic field is pointwise nondecreasing.
    necrotic = result.time_series["necrotic"]
    assert necrotic.shape == (3, *gm.shape)
    assert np.all(np.diff(necrotic, axis=0) >= -1e-12)
    # The nutrient is only consumed (its diffusion conserves total mass).
    nutrient_masses = result.time_series["nutrient"].sum(axis=(1, 2, 3))
    assert nutrient_masses[-1] < nutrient_masses[0]


def test_dti_short_solve(tensor_phantom):
    params = dti_params(tensor_phantom, n_time_series_snapshots=3)
    result = AnisotropicFKPPSolver(params).solve()
    full_shape = tensor_phantom.shape[:3]
    assert_successful_time_solve(result, {"cell_density"}, full_shape)
    assert (
        result.final_state["cell_density"].sum()
        > result.initial_state["cell_density"].sum()
    )
    assert result.time_series["cell_density"].shape == (3, *full_shape)


def test_dti_uniform_gray_matter_solve(tensor_phantom, tissue_phantom):
    gm, wm = tissue_phantom
    params = dti_params(
        tensor_phantom,
        uniform_gray_matter=True,
        gray_matter_pbmap=gm,
        white_matter_pbmap=wm,
        diffusivity_ratio=10.0,
        tensor_exponent=2,
        tensor_linear_term=0.1,
    )
    result = AnisotropicFKPPSolver(params).solve()
    assert_successful_time_solve(result, {"cell_density"}, tensor_phantom.shape[:3])


def test_stopping_threshold_early_exit(tissue_phantom):
    """Crossing the stopping threshold stops the loop at that step:
    stopping_criterion='threshold', final_time = crossing step * dt, and
    snapshots scheduled after the crossing are dropped."""
    gm, wm = tissue_phantom
    params = fk_params(
        gm, wm, stopping_time=40, n_time_series_snapshots=6, stopping_threshold=300.0
    )
    solver = FKPPSolver(params)
    result = solver.solve()
    assert result.success
    assert result.stopping_criterion == "threshold"
    assert 0.0 < result.final_time < 40.0
    assert result.final_stopping_quantity >= 300.0
    # final_time is a whole number of steps.
    _, dt = solver._time_step_count()
    n_taken = result.final_time / dt
    assert abs(n_taken - round(n_taken)) < 1e-6
    for frames in result.time_series.values():
        assert 0 < frames.shape[0] < 6  # truncated by the early exit


def test_n_steps_override(tissue_phantom):
    """An explicit n_steps pins the step count (dt = stopping_time / n_steps),
    bypassing the stability formula."""
    gm, wm = tissue_phantom
    params = fk_params(
        gm, wm, stopping_time=40, n_steps=2000, stopping_threshold=300.0
    )
    result = FKPPSolver(params).solve()
    assert result.success
    assert result.stopping_criterion == "threshold"
    # final_time is a whole number of override steps; the stability formula's
    # own dt (~0.297 here, vs the override's 0.02) would not divide it.
    n_taken = result.final_time / (40 / 2000)
    assert 0 < n_taken < 2000
    assert abs(n_taken - round(n_taken)) < 1e-9


def test_dti_guard_exit(tensor_phantom):
    """A shrinking tumor (negative rho) fires a DTI guard: success=False,
    stopping_criterion='error', final_time at the actual exit step."""
    params = dti_params(tensor_phantom, rho=-1.0, stopping_time=40)
    result = AnisotropicFKPPSolver(params).solve()
    assert result.success is False
    assert result.stopping_criterion == "error"
    assert result.error is not None and "guard fired" in result.error
    assert 0.0 < result.final_time < 40.0


def test_time_series_recording(tissue_phantom):
    """All requested snapshots are recorded on a full run, and the last one
    (scheduled at the final step) equals the final state exactly."""
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, n_time_series_snapshots=5)
    result = FKPPSolver(params).solve()
    frames = result.time_series["cell_density"]
    assert frames.shape[0] == 5
    np.testing.assert_array_equal(frames[-1], result.final_state["cell_density"])


def test_no_time_series_by_default(tissue_phantom):
    gm, wm = tissue_phantom
    result = FKPPSolver(fk_params(gm, wm)).solve()
    assert result.time_series is None


def test_seed_outside_tissue_errors(tissue_phantom):
    gm, wm = tissue_phantom
    params = fk_params(
        gm,
        wm,
        gaussian_seed_x_fraction=0.02,
        gaussian_seed_y_fraction=0.02,
        gaussian_seed_z_fraction=0.02,
    )
    result = FKPPSolver(params).solve()
    assert result.success is False
    assert result.stopping_criterion == "error"
    assert result.error == "Initial tumor position is outside the brain matter."


def test_param_validation_errors(tissue_phantom):
    gm, wm = tissue_phantom
    with pytest.raises(ValueError):
        FKPPSolver(fk_params(gm, wm, bogus_param=1))
    with pytest.raises(KeyError):
        FKPPSolver({"rho": 0.1})
    with pytest.raises(ValueError):
        FKPPSolver(fk_params(gm, wm, volume_threshold=0.5))  # mass mode
    with pytest.raises(ValueError):
        FKPPSolver(fk_params(gm, wm, stopping_mode="occupancy"))
    with pytest.raises(ValueError):
        FKPPSolver(fk_params(gm, wm, precision="f16"))
    with pytest.raises(ValueError):
        FKPPSolver(fk_params(gm, wm, n_steps=0))
    with pytest.raises(ValueError):
        FKPPSolver(fk_params(gm, wm, n_steps=12.5))
    with pytest.raises(ValueError, match="3D"):
        FKPPSolver(fk_params(gm, np.zeros((4, 4))))
    with pytest.raises(ValueError, match="diffusion_tensors"):
        AnisotropicFKPPSolver(dti_params(np.zeros((4, 4, 4, 3))))


def test_volume_stopping_mode(tissue_phantom):
    """stopping_mode='volume' thresholds a physical volume:
    voxel_volume * count(cell density > volume_threshold)."""
    gm, wm = tissue_phantom
    params = fk_params(
        gm, wm, stopping_time=40, stopping_mode="volume", stopping_threshold=50.0
    )
    solver = FKPPSolver(params)
    result = solver.solve()
    assert result.success
    assert result.stopping_criterion == "threshold"
    assert result.final_stopping_quantity >= 50.0
    # The quantity is a whole number of voxels times the voxel volume.
    count = result.final_stopping_quantity / solver.voxel_volume
    assert abs(count - round(count)) < 1e-9


def test_no_retrace_on_repeat_solve(tissue_phantom):
    """Solves must reuse the compiled time scan.

    After a warm-up solve, zero new traces (operators.SCAN_TRACE_COUNT is a
    trace-time counter) are allowed for (a) identical re-solves and (b)
    re-solves that change only PHYSICAL parameters — the sweep/inverse-loop
    case: physical scalars are dynamic arguments, never jit-static. The rho
    values sit on the diffusion-bound branch of the nt formula, so n_steps
    (a legitimate static) is unchanged."""
    gm, wm = tissue_phantom
    params = fk_params(gm, wm)
    FKPPSolver(params).solve()  # warm-up: may trace (once) if not already cached
    before = operators.SCAN_TRACE_COUNT
    FKPPSolver(params).solve()
    FKPPSolver(params).solve()
    assert operators.SCAN_TRACE_COUNT == before
    # physical-parameter-only changes (stopping threshold, rho):
    FKPPSolver({**params, "stopping_threshold": 500.0}).solve()
    FKPPSolver({**params, "rho": 0.16}).solve()
    assert operators.SCAN_TRACE_COUNT == before

    params_two_compartment = two_compartment_params(gm, wm)
    TwoCompartmentWithNutrientFKPPSolver(params_two_compartment).solve()  # warm-up
    before = operators.SCAN_TRACE_COUNT
    TwoCompartmentWithNutrientFKPPSolver(
        {**params_two_compartment, "rho": 0.16, "necrosis_rate": 0.5}
    ).solve()
    assert operators.SCAN_TRACE_COUNT == before


@pytest.mark.parametrize("solver_kind", list(SOLVER_CASES))
def test_f32_vs_f64_agreement(solver_kind, tissue_phantom, tensor_phantom):
    cls, build_params = SOLVER_CASES[solver_kind]
    params = build_params(tissue_phantom, tensor_phantom)
    res32 = cls({**params, "precision": "f32"}).solve()
    res64 = cls({**params, "precision": "f64"}).solve()
    assert res32.success and res64.success, (res32.error, res64.error)
    assert res32.stopping_criterion == res64.stopping_criterion
    assert res32.final_time == res64.final_time
    np.testing.assert_allclose(
        res32.final_stopping_quantity, res64.final_stopping_quantity, rtol=1e-3
    )
    for key in res64.final_state:
        np.testing.assert_allclose(
            res32.final_state[key], res64.final_state[key], atol=F32_ATOL, rtol=0
        )
