"""End-to-end solver comparisons: fisher_kpp_jax (f64) vs. the NumPy reference.

Short solves on 24^3 phantoms. In f64 the two implementations differ only by
XLA-vs-libm transcendental rounding and reduction order, so tolerances are
near machine precision. The f32-vs-f64 agreement test uses a loose tolerance
documented at the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from fisher_kpp import (
    AnisotropicFKPPSolver as RefDTI,
    FKPPSolver as RefFK,
    TwoCompartmentWithNutrientFKPPSolver as Ref2c,
)
from fisher_kpp_jax import (
    AnisotropicFKPPSolver as JaxDTI,
    FKPPSolver as JaxFK,
    TwoCompartmentWithNutrientFKPPSolver as Jax2c,
)

F64_ATOL = 1e-12
F64_RTOL = 1e-10


def fk_params(gm: np.ndarray, wm: np.ndarray, **overrides) -> dict:
    params = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        gray_matter=gm,
        white_matter=wm,
        gaussian_seed_x_fraction=0.5,
        gaussian_seed_y_fraction=0.5,
        gaussian_seed_z_fraction=0.5,
        resolution_factor=0.6,
        stopping_time=10,
    )
    params.update(overrides)
    return params


def fk2c_params(gm: np.ndarray, wm: np.ndarray, **overrides) -> dict:
    params = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        necrosis_rate=0.4,
        nutrient_threshold=0.4,
        nutrient_diffusivity=0.5,
        nutrient_consumption_rate=0.1,
        gray_matter=gm,
        white_matter=wm,
        gaussian_seed_x_fraction=0.5,
        gaussian_seed_y_fraction=0.5,
        gaussian_seed_z_fraction=0.5,
        resolution_factor=0.6,
        stopping_time=10,
    )
    params.update(overrides)
    return params


def dti_params(tensors: np.ndarray, **overrides) -> dict:
    params = dict(
        diffusivity=0.3,
        rho=0.15,
        diffusion_tensors=tensors,
        gaussian_seed_x_fraction=0.5,
        gaussian_seed_y_fraction=0.5,
        gaussian_seed_z_fraction=0.5,
        resolution_factor=0.6,
        stopping_time=10,
    )
    params.update(overrides)
    return params


def assert_results_close(ref, ours, atol: float = F64_ATOL, rtol: float = F64_RTOL):
    assert ours.success == ref.success, (ours.error, ref.error)
    assert ours.stopping_criterion == ref.stopping_criterion
    assert ours.final_time == ref.final_time
    np.testing.assert_allclose(
        ours.final_stopping_quantity, ref.final_stopping_quantity, rtol=rtol
    )
    assert set(ours.final_state) == set(ref.final_state)
    for key in ref.final_state:
        np.testing.assert_allclose(
            ours.initial_state[key], ref.initial_state[key], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            ours.final_state[key], ref.final_state[key], atol=atol, rtol=rtol
        )


def test_fkpp_short_solve(tissue_phantom):
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, n_time_series_snapshots=4)
    ref = RefFK(params).solve()
    ours = JaxFK({**params, "precision": "f64"}).solve()
    assert ref.success
    assert_results_close(ref, ours)
    assert set(ours.time_series) == set(ref.time_series)
    for key in ref.time_series:
        assert ours.time_series[key].shape == ref.time_series[key].shape
        np.testing.assert_allclose(
            ours.time_series[key], ref.time_series[key], atol=F64_ATOL, rtol=F64_RTOL
        )


def test_two_compartment_short_solve(tissue_phantom):
    gm, wm = tissue_phantom
    params = fk2c_params(gm, wm, n_time_series_snapshots=3)
    ref = Ref2c(params).solve()
    ours = Jax2c({**params, "precision": "f64"}).solve()
    assert ref.success
    assert_results_close(ref, ours)
    for key in ("proliferative", "necrotic", "nutrient"):
        np.testing.assert_allclose(
            ours.time_series[key], ref.time_series[key], atol=F64_ATOL, rtol=F64_RTOL
        )


def test_two_compartment_occupancy_mask_active(tissue_phantom):
    """FK_2c with the occupancy mask demonstrably ACTIVE.

    The default test parameters never reach max_tumor_occupancy=0.9
    (max(P+N) ~ 0.55 over the solve), so they cannot detect a wrong per-step
    diffusivity-rebuild variant. With max_tumor_occupancy=0.4 the mask is
    active on every one of the 315 steps (up to 53 voxels blocked; measured
    on the reference). A stale-mask variant (rebuild from PRE-step P/N)
    differs from the reference by ~5e-4 max-abs here — eight orders above
    the tolerance — so this comparison genuinely pins the post-step
    aliasing semantics of the rebuild.
    """
    gm, wm = tissue_phantom
    params = fk2c_params(gm, wm, max_tumor_occupancy=0.4)
    ref = Ref2c(params).solve()
    ours = Jax2c({**params, "precision": "f64"}).solve()
    assert ref.success
    assert_results_close(ref, ours)


def test_dti_short_solve(tensor_phantom):
    params = dti_params(tensor_phantom, n_time_series_snapshots=3)
    ref = RefDTI(params).solve()
    ours = JaxDTI({**params, "precision": "f64"}).solve()
    assert ref.success
    assert_results_close(ref, ours)
    np.testing.assert_allclose(
        ours.time_series["cell_density"],
        ref.time_series["cell_density"],
        atol=F64_ATOL,
        rtol=F64_RTOL,
    )


def test_dti_uniform_gray_matter_solve(tensor_phantom, tissue_phantom):
    gm, wm = tissue_phantom
    params = dti_params(
        tensor_phantom,
        uniform_gray_matter=True,
        gray_matter=gm,
        white_matter=wm,
        diffusivity_ratio=10.0,
        tensor_exponent=2,
        tensor_linear_term=0.1,
    )
    ref = RefDTI(params).solve()
    ours = JaxDTI({**params, "precision": "f64"}).solve()
    assert ref.success
    assert_results_close(ref, ours)


def test_stopping_threshold_early_exit(tissue_phantom):
    """The threshold crossing must happen at the same step: final_time =
    t * dt must match exactly, and snapshots recorded after the crossing
    must be dropped in both implementations. (The reference package only
    knows the old ``stopping_volume`` name.)"""
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, stopping_time=40, n_time_series_snapshots=6)
    ref = RefFK({**params, "stopping_volume": 300.0}).solve()
    ours = JaxFK(
        {**params, "stopping_threshold": 300.0, "precision": "f64"}
    ).solve()
    assert ref.stopping_criterion == "volume"
    assert 0.0 < ref.final_time < 40.0
    assert_results_close(ref, ours)
    for key in ref.time_series:
        assert ours.time_series[key].shape == ref.time_series[key].shape
        assert ours.time_series[key].shape[0] < 6  # truncated by the early exit
        np.testing.assert_allclose(
            ours.time_series[key], ref.time_series[key], atol=F64_ATOL, rtol=F64_RTOL
        )


def test_dti_guard_exit(tensor_phantom):
    """A shrinking tumor (negative rho) fires a DTI guard: success=False,
    stopping_criterion='error', final_time at the actual exit step."""
    params = dti_params(tensor_phantom, rho=-1.0, stopping_time=40)
    ref = RefDTI(params).solve()
    ours = JaxDTI({**params, "precision": "f64"}).solve()
    assert ref.success is False
    assert ref.stopping_criterion == "error"
    assert ours.success is False
    assert ours.stopping_criterion == "error"
    assert ours.final_time == ref.final_time
    assert ours.error is not None and "guard" in ours.error
    # Same guard fired (message prefix up to the formatted value).
    assert ours.error.split(":")[0] == ref.error.split(":")[0]
    np.testing.assert_allclose(
        ours.final_state["cell_density"],
        ref.final_state["cell_density"],
        atol=F64_ATOL,
        rtol=F64_RTOL,
    )


def test_snapshot_step_indices(tissue_phantom):
    """The recorded snapshot step indices are the same _record_steps
    (np.linspace over n_steps) as the reference computes."""
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, n_time_series_snapshots=5)
    ref_solver, jax_solver = RefFK(params), JaxFK({**params, "precision": "f64"})
    ref_res = ref_solver.solve()
    jax_res = jax_solver.solve()
    n_ref, _ = ref_solver._time_step_count()
    n_jax, _ = jax_solver._time_step_count()
    assert n_ref == n_jax
    np.testing.assert_array_equal(
        jax_solver._record_steps(n_jax, 5), ref_solver._record_steps(n_ref, 5)
    )
    assert jax_res.time_series["cell_density"].shape[0] == 5
    np.testing.assert_allclose(
        jax_res.time_series["cell_density"],
        ref_res.time_series["cell_density"],
        atol=F64_ATOL,
        rtol=F64_RTOL,
    )


def test_seed_outside_tissue_errors(tissue_phantom):
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, gaussian_seed_x_fraction=0.02, gaussian_seed_y_fraction=0.02,
                       gaussian_seed_z_fraction=0.02)
    ref = RefFK(params).solve()
    ours = JaxFK(params).solve()
    assert ref.success is False and ours.success is False
    assert ours.stopping_criterion == "error"
    assert ours.error == ref.error


def test_stopping_volume_deprecated_alias(tissue_phantom):
    """'stopping_volume' still works as a deprecated alias: it warns and
    produces a Result identical to 'stopping_threshold'."""
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, stopping_time=40, precision="f64")
    canonical = JaxFK({**params, "stopping_threshold": 300.0}).solve()
    with pytest.warns(DeprecationWarning, match="stopping_volume"):
        aliased_solver = JaxFK({**params, "stopping_volume": 300.0})
    aliased = aliased_solver.solve()
    assert aliased.success == canonical.success
    assert aliased.stopping_criterion == canonical.stopping_criterion
    assert aliased.final_time == canonical.final_time
    assert aliased.final_stopping_quantity == canonical.final_stopping_quantity
    for key in canonical.final_state:
        np.testing.assert_array_equal(
            aliased.final_state[key], canonical.final_state[key]
        )
        np.testing.assert_array_equal(
            aliased.initial_state[key], canonical.initial_state[key]
        )


def test_stopping_threshold_and_alias_both_rejected(tissue_phantom):
    gm, wm = tissue_phantom
    with pytest.raises(ValueError, match="only one of"):
        JaxFK(
            fk_params(gm, wm, stopping_threshold=300.0, stopping_volume=300.0)
        )


def test_param_validation_parity(tissue_phantom):
    gm, wm = tissue_phantom
    with pytest.raises(ValueError):
        JaxFK(fk_params(gm, wm, bogus_param=1))
    with pytest.raises(KeyError):
        JaxFK({"rho": 0.1})
    with pytest.raises(ValueError):
        JaxFK(fk_params(gm, wm, density_threshold=0.5))  # mass mode
    with pytest.raises(ValueError):
        JaxFK(fk_params(gm, wm, stopping_mode="occupancy"))
    with pytest.raises(ValueError):
        JaxFK(fk_params(gm, wm, precision="f16"))


def test_volume_stopping_mode(tissue_phantom):
    gm, wm = tissue_phantom
    params = fk_params(gm, wm, stopping_time=40, stopping_mode="volume")
    ref = RefFK({**params, "stopping_volume": 50.0}).solve()
    ours = JaxFK(
        {**params, "stopping_threshold": 50.0, "precision": "f64"}
    ).solve()
    assert ref.stopping_criterion == "volume"
    assert_results_close(ref, ours)


def test_no_retrace_on_repeat_solve(tissue_phantom):
    """Solves must reuse the compiled scan driver.

    After a warm-up solve, zero new traces (base.SCAN_TRACE_COUNT is a
    trace-time counter) are allowed for (a) identical re-solves and (b)
    re-solves that change only PHYSICAL parameters — the sweep/inverse-loop
    case: physical scalars are dynamic arguments, never jit-static. The rho
    values sit on the diffusion-bound branch of the nt formula, so n_steps
    (a legitimate static) is unchanged."""
    from fisher_kpp_jax import base

    gm, wm = tissue_phantom
    params = fk_params(gm, wm)
    JaxFK(params).solve()  # warm-up: may trace (once) if not already cached
    before = base.SCAN_TRACE_COUNT
    JaxFK(params).solve()
    JaxFK(params).solve()
    assert base.SCAN_TRACE_COUNT == before
    # physical-parameter-only changes (stopping threshold, rho):
    JaxFK({**params, "stopping_threshold": 500.0}).solve()
    JaxFK({**params, "rho": 0.16}).solve()
    assert base.SCAN_TRACE_COUNT == before

    params_2c = fk2c_params(gm, wm)
    Jax2c(params_2c).solve()  # warm-up
    before = base.SCAN_TRACE_COUNT
    Jax2c({**params_2c, "rho": 0.16, "necrosis_rate": 0.5}).solve()
    assert base.SCAN_TRACE_COUNT == before


# Loose tolerance for f32 vs f64 on a short solve (a few hundred explicit
# Euler steps of O(1) fields): observed max-abs difference is ~1e-4; assert
# 2e-3 max-abs on the final field and 1e-3 relative on the stopping quantity.
F32_ATOL = 2e-3


@pytest.mark.parametrize("solver_kind", ["fk", "fk2c", "dti"])
def test_f32_vs_f64_agreement(solver_kind, tissue_phantom, tensor_phantom):
    gm, wm = tissue_phantom
    if solver_kind == "fk":
        cls, params = JaxFK, fk_params(gm, wm)
    elif solver_kind == "fk2c":
        cls, params = Jax2c, fk2c_params(gm, wm)
    else:
        cls, params = JaxDTI, dti_params(tensor_phantom)
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
