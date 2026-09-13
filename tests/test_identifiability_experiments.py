"""Tests of scripts/identifiability_experiments.py (loaded with importlib
from scripts/), on the 24^3 phantom and on uniform tissue cubes, on the
CPU (every test runs under ``jax.default_device`` of the CPU device).

(1) The growth-only invariance (lambda D, lambda rho, T_r / lambda) is
exact at f64 with a fixed step count and not with a fixed step, (2) the
pre-resection frame of a treated run precedes the cavity zeroing, (3) on
the pre-resection observations alone the invariance direction is a null
direction of the weighted Jacobian (to rounding) and the Cramer-Rao error
of log T_r is infinite, while the fixed-dt variant of the T_r column is
contaminated at the discretization level, (4) the linear composition rule
reproduces a growth-only run of a light seed at a small rho to 1 % in
relative L2, (5) the metrics on synthetic spheres and half-spaces give the
set-formula Dice and the analytic surface distance, (6) the design fills
the four cells with 8 patients each at a zero seed floor, and (7) the
smaller pieces: the frame days, the total log kill, the patient
selection, the seed reparametrisations and the Cramer-Rao errors of a
singular Fisher matrix. The whole smoke pipeline on the phantom is
``slow``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import jax
import nibabel as nib
import numpy as np
import pytest

from fisher_kpp_jax import SOLVER_KEY, StuppFKPPSolver, read_config, write_config

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "identifiability_experiments.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("identifiability_experiments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ide = _load_script()
sa = ide.sa

PHANTOM_N = 24


@pytest.fixture(autouse=True)
def on_cpu():
    """Every solve of these tests runs on the CPU device."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _phantom_maps(n: int = PHANTOM_N) -> tuple[np.ndarray, np.ndarray]:
    """(wm, gm): a spherical WM core of radius 6 with a GM shell to 9."""
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    return (r < 6).astype(np.float64), ((r >= 6) & (r < 9)).astype(np.float64)


@pytest.fixture(scope="session")
def phantom_root(tmp_path_factory) -> Path:
    """A design directory on the phantom: the shipped base config (its
    Stupp schedule, 12 steps/day) with the phantom's tissue maps, the
    shipped search space, 256 Sobol' candidates."""
    tmp_path = tmp_path_factory.mktemp("ide")
    wm, gm = _phantom_maps()
    volumes = tmp_path / "volumes"
    volumes.mkdir()
    nib.save(nib.Nifti1Image(wm, np.eye(4)), str(volumes / "wm.nii.gz"))
    nib.save(nib.Nifti1Image(gm, np.eye(4)), str(volumes / "gm.nii.gz"))
    base = read_config(sa.DEFAULT_CONFIG, solver=StuppFKPPSolver)
    base["white_matter_pbmap"] = str(volumes / "wm.nii.gz")
    base["gray_matter_pbmap"] = str(volumes / "gm.nii.gz")
    base["steps_per_day"] = 12
    write_config(base, tmp_path / "base.json")
    return ide.make_design(tmp_path / "base.json", sa.DEFAULT_SEARCH_SPACE, tmp_path, "design", tissue_maps=None, log2_candidates=8)


@pytest.fixture(scope="session")
def cohort(phantom_root: Path):
    return ide.load_cohort(phantom_root)


def _fast_patient(**overrides) -> "ide.Patient":
    """A phantom tumor that reaches the cavity threshold by day 12:
    D = 0.2, rho = 0.15 (v = 0.346, lambda = 1.155), a seed of peak 0.5
    and sigma sqrt(10) mm."""
    values = dict(
        id="px",
        cell="test",
        front_speed=2.0 * np.sqrt(0.2 * 0.15),
        front_width=np.sqrt(0.2 / 0.15),
        resection_time=12.0,
        rt_alpha=0.1,
        chemo_kill_rate=0.01 / 75,
        seed_peak=0.5,
        seed_sigma=np.sqrt(10.0),
    )
    values.update(overrides)
    return ide.Patient(**values)


# --- (1) growth invariance ---


def test_growth_invariance_exact_with_fixed_n_not_with_fixed_dt(tmp_path):
    """(2 D, 2 rho, T_r / 2) against (D, rho, T_r) at f64 on a uniform
    cube (D = 0.1, rho = 0.05, T_r = 40, where 12 steps/day is stricter
    than the solver's stability estimate for both lambdas): the
    growth-only run with the reference's step count reproduces the field
    to rounding (the step products are the same numbers), the run at the
    base config's steps per day does not (a discretization-level
    difference); the treated run records every frame; the rows carry the
    metrics and invariance.csv is written."""
    cohort = _cube_cohort(tmp_path)
    rows = ide.run_invariance(cohort, tmp_path / "invariance", (1.0, 2.0), diffusivity=0.1, rho=0.05, resection_time=40.0)
    by_key = {(r["lambda"], r["run_type"], r["snapshot"]): r for r in rows}
    reference = by_key[(1.0, "growth_fixed_n", "pre")]
    assert reference["max_abs_diff"] == 0.0 and reference["dice_edema"] == 1.0 and reference["mass"] > 0
    fixed_n = by_key[(2.0, "growth_fixed_n", "pre")]
    fixed_dt = by_key[(2.0, "growth_fixed_dt", "pre")]
    assert fixed_n["n_steps"] == reference["n_steps"] == 480 and fixed_n["dt"] == pytest.approx(reference["dt"] / 2)
    assert fixed_n["max_abs_diff"] < 1e-10
    assert fixed_dt["n_steps"] == 240 and fixed_dt["max_abs_diff"] > 1e-6
    assert fixed_dt["resection_time"] == 20.0 and fixed_n["resection_time"] == 20.0
    treated = {snapshot: by_key[(2.0, "treated", snapshot)] for snapshot in ide.FISHER_FRAMES}
    assert treated["pre"]["max_abs_diff"] > 1e-6 and all(np.isfinite(r["mass"]) for r in treated.values())
    assert set(rows[0]) >= set(ide.METRIC_NAMES) | {"lambda", "run_type", "snapshot", "n_steps", "dt"}
    for run_type in ide.INVARIANCE_RUN_TYPES:
        assert (tmp_path / "invariance" / "lambda_2" / run_type / "result.json").is_file()
    assert (tmp_path / "invariance" / "lambda_2" / "treated" / "d120_cell_density.nii.gz").is_file()
    assert (tmp_path / "invariance" / "maps" / sa.CAVITY_FILE).is_file()
    saved = read_config(tmp_path / "invariance" / "lambda_2" / "treated" / "config.json")
    assert saved["precision"] == "f64" and saved["gaussian_seed_floor"] == 0.0 and saved["white_matter_diffusivity"] == pytest.approx(0.2)
    assert saved["rho"] == pytest.approx(0.1) and saved["n_steps"] == treated["pre"]["n_steps"] == 240 + 1440
    # The assembled CSV of a design directory.
    (cohort.root / "runs" / "invariance").mkdir(parents=True, exist_ok=True)
    ide.write_record(cohort.root / "runs" / "invariance" / "invariance.json", {"rows": rows})
    assert len(ide.assemble_invariance(cohort.root)) == len(rows)
    assert len(_read_csv(cohort.root / "invariance.csv")) == len(rows)
    assert (cohort.root / "figures" / "invariance_metrics.png").is_file()


# --- (2) the pre-resection frame ---


def test_pre_resection_frame_precedes_cavity_zeroing(cohort, tmp_path):
    """The pre frame is the state after step n_growth - 1 (day
    (n_growth - 1) dt): inside the cavity it holds the density the cavity
    was thresholded from (nonzero, near the threshold), the state one
    step later (the resection step) is zero there; the saved config
    reproduces the run and its frames.json records the days."""
    patient = _fast_patient()
    config = ide.patient_config(cohort, patient, ide.FISHER_HORIZON)
    growth = ide.solve_growth(cohort, ide.growth_config(config))
    density = np.asarray(growth.final_state["cell_density"])
    maps = ide.derive_maps(cohort, density, len(config["rt_times"]), tmp_path / "run")
    assert maps.cavity.sum() > 0 and np.array_equal(maps.cavity, density >= 0.6)
    run = ide.treated_run(cohort, config, maps, growth.n_steps, growth.dt, ("pre",), tmp_path / "run", growth, extra_moments={"post": growth.dt})
    n_growth, dt = growth.n_steps, growth.dt
    assert run.days["pre"] == pytest.approx((n_growth - 1) * dt) and run.days["post"] == pytest.approx(n_growth * dt)
    assert run.days["pre"] < config["resection_time"] <= run.days["post"]
    pre, post = run.frames["pre"], run.frames["post"]
    assert np.all(pre[maps.cavity] > 0) and pre[maps.cavity].min() > 0.5
    assert np.all(post[maps.cavity] == 0) and post[~maps.cavity].max() > 0
    # One step of growth separates the pre frame from the cavity's field.
    np.testing.assert_allclose(pre, density, rtol=0, atol=0.02)
    assert (run.run_dir / "pre_cell_density.nii.gz").is_file() and (run.run_dir / sa.GROWTH_DIR / "config.json").is_file()
    frames = json.loads((run.run_dir / ide.FRAMES_FILE).read_text())
    assert frames["frames"]["pre"]["recorded_day"] == pytest.approx(run.days["pre"]) and frames["n_growth"] == n_growth
    saved = read_config(run.run_dir / "config.json")
    assert saved["resection_cavity"] == {"segmentation": str(maps.cavity_path), "label": sa.CAVITY_LABEL} and saved["rt_dose"] == str(maps.dose_path)
    assert saved["snapshot_times"] == sorted(run.days.values()) and saved["resection_time"] == pytest.approx((n_growth - 0.5) * dt)
    reproduced = StuppFKPPSolver(saved).solve()
    assert reproduced.success and reproduced.snapshot_times.tolist() == pytest.approx(saved["snapshot_times"])
    stored = np.asarray(nib.load(str(run.run_dir / "pre_cell_density.nii.gz")).get_fdata())
    np.testing.assert_array_equal(stored, sa.round_field(reproduced.time_series["cell_density"][0]))


# --- (3) the invariance direction on the pre-resection observations ---


def test_invariance_direction_is_null_on_pre_resection_observations(cohort, tmp_path):
    """With the T_r column stepped at a fixed step count (scaled_dt), the
    pre-resection observations cannot separate log T_r from log v: |W e|
    is at rounding level of |W e_1| for set (a), F is singular there
    (an infinite Cramer-Rao error of log T_r, a zero eigenvalue whose
    eigenvector is e) and the T_r column has no residual; the fixed-dt
    variant is contaminated at the discretization level; the sets with
    treated frames are not singular; the outputs are written."""
    np.testing.assert_allclose(ide.INVARIANCE_DIRECTION, np.array([1, 0, -1, 0, 0]) / np.sqrt(2))
    record = ide.fisher_patient(cohort, _fast_patient(), tmp_path / "fisher", draws=2, fd_check=False, seed=3)
    scaled = record["analysis"]["scaled_dt"]
    assert scaled["a"]["We_ratio"] < 1e-6 and scaled["a"]["We1_norm"] > 0
    assert np.isinf(scaled["a"]["cr_log_T_r"]) and np.isinf(scaled["a"]["cr_log_v"]) and np.isfinite(scaled["a"]["cr_log_lambda"])
    assert scaled["a"]["resid_frac_T_r"] < 1e-6 and scaled["a"]["resid7_frac_T_r"] < 1e-6
    # Before the resection nothing depends on alpha and k_ct either: F has
    # a three-dimensional null space (e, e_alpha, e_k_ct) on set (a).
    assert np.isinf(scaled["a"]["cr_log_alpha"]) and np.isinf(scaled["a"]["cr_log_k_ct"]) and np.isinf(scaled["a"]["cond_F"])
    eigenvalues = np.array([scaled["a"][f"eig_{i}"] for i in range(1, 6)])
    assert np.all(np.abs(eigenvalues[:3]) < 1e-12 * eigenvalues[4]) and eigenvalues[3] > 1e-6 * eigenvalues[4]
    assert all(np.isfinite(scaled["f"][f"cr_{name}"]) for name in ide.THETA_NAMES)
    fixed = record["analysis"]["fixed_dt"]
    assert fixed["a"]["We_ratio"] > 1e-6 and np.isfinite(fixed["a"]["cr_log_T_r"])
    assert scaled["f"]["We_ratio"] > scaled["a"]["We_ratio"] and np.isfinite(scaled["f"]["cr_log_T_r"])
    assert set(scaled) == set(ide.OBSERVATION_SETS) and set(scaled["f"]) >= set(ide.FISHER_SET_KEYS) | {"splits"}
    splits = scaled["f"]["splits"]
    assert {s["name"] for s in splits if s["split"] == "day"} == set(ide.FISHER_FRAMES)
    assert {s["name"] for s in splits if s["split"] == "region"} == set(ide.REGION_NAMES)
    assert sum(s["n_rows"] for s in splits if s["split"] == "day") == scaled["f"]["n_rows"] == record["observation"]["n_observations"]
    patient_dir = tmp_path / "fisher" / "px"
    assert (patient_dir / "fisher.json").is_file() and (patient_dir / "sigma.npz").is_file() and (patient_dir / "W.npz").is_file()
    assert (patient_dir / "truth" / "d120_cell_density.nii.gz").is_file() and (patient_dir / "log_T_r_scaled_dt_plus" / "config.json").is_file()
    stored = np.load(patient_dir / "W.npz")
    assert stored["W"].shape == (record["observation"]["n_observations"], len(ide.COLUMN_NAMES)) and stored["W"].dtype == np.float32
    assert list(stored["columns"]) == list(ide.COLUMN_NAMES)
    variance = np.load(patient_dir / "sigma.npz")["variance"]
    assert variance.shape == (len(ide.FISHER_FRAMES), len(ide.INDICATOR_LEVELS), record["observation"]["n_voxels"]) and variance.min() >= ide.VARIANCE_FLOOR
    assert len(record["run_metrics"]) == len(ide.perturbations(_fast_patient(), 10, 0.1)) * len(ide.FISHER_FRAMES)
    assert record["fd_check"] == []
    # The scaled-dt T_r runs keep the step count, the fixed-dt runs the step.
    steps = json.loads((patient_dir / "log_T_r_scaled_dt_plus" / ide.FRAMES_FILE).read_text())
    assert steps["n_growth"] == record["n_growth"] and steps["dt"] == pytest.approx(record["dt"] * np.exp(ide.FD_STEP))
    steps = json.loads((patient_dir / "log_T_r_fixed_dt_minus" / ide.FRAMES_FILE).read_text())
    assert steps["dt"] == pytest.approx(record["dt"]) and steps["n_growth"] == int(np.rint(12.0 * np.exp(-ide.FD_STEP) / record["dt"]))


# --- (4) the linear composition rule ---


def _cube_cohort(tmp_path: Path, n: int = 40) -> "ide.Cohort":
    """A cohort on a uniform white-matter cube (in-memory maps, the
    shipped base config's schedule, floor 0, f64, 12 steps/day)."""
    base = read_config(sa.DEFAULT_CONFIG, solver=StuppFKPPSolver)
    base.update(precision="f64", gaussian_seed_floor=0.0, steps_per_day=12, snapshot_times=None)
    wm = np.ones((n, n, n))
    return ide.Cohort(
        root=tmp_path,
        spec={"treatment": sa.treatment_settings()},
        base=base,
        wm=wm,
        gm=np.zeros_like(wm),
        zooms=(1.0, 1.0, 1.0),
        affine=np.eye(4),
        tissue=np.ones_like(wm, dtype=bool),
        seed_voxel=(n // 2, n // 2, n // 2),
        seed_fractions=((n // 2 + 0.5) / n,) * 3,
        patients=[],
    )


def test_linear_composition_rule_reproduces_growth(tmp_path):
    """A seed of mass 1e-3 and diffusion time 4 mm^2 at rho = 0.001 and
    D = 0.2 grown for 10 days, against the composition rule's seed
    (m e^{rho 6}, w + 6 D) grown for 4 days: relative L2 below 1 %;
    the rule falls back to the truth's seed when the width would not be
    positive."""
    cohort = _cube_cohort(tmp_path)
    mass, tau = 1e-3, 4.0
    peak = float(sa.seed_peak_density(mass, tau))
    truth = _fast_patient(front_speed=2 * np.sqrt(0.2 * 0.001), front_width=np.sqrt(0.2 / 0.001), resection_time=10.0, seed_peak=peak, seed_sigma=np.sqrt(2 * tau))
    assert truth.diffusivity == pytest.approx(0.2) and truth.rho == pytest.approx(0.001)
    assert truth.seed["gaussian_seed_mass"] == pytest.approx(mass) and truth.seed["gaussian_seed_diffusion_time"] == pytest.approx(tau)
    peak0, sigma0, from_rule = ide.composition_seed(truth, 4.0)
    assert from_rule and sigma0 == pytest.approx(np.sqrt(2 * (tau + 0.2 * 6.0)))
    assert peak0 == pytest.approx(float(sa.seed_peak_density(mass * np.exp(0.001 * 6.0), tau + 1.2)))
    reference = ide.solve_growth(cohort, ide.growth_config(ide.patient_config(cohort, truth, 120.0))).final_state["cell_density"]
    substitute = ide.solve_growth(
        cohort, ide.growth_config(ide.patient_config(cohort, ide.replace(truth, seed_peak=peak0, seed_sigma=sigma0, resection_time=4.0), 120.0))
    ).final_state["cell_density"]
    assert reference.max() < 1e-3 and substitute.max() < 1e-3
    assert np.linalg.norm(substitute - reference) / np.linalg.norm(reference) < 0.01
    # T_0 beyond T_r + w / D: the width would be negative, the truth's seed is kept.
    fallback = ide.composition_seed(truth, 10.0 + tau / 0.2 + 1.0)
    assert fallback == (truth.seed_peak, truth.seed_sigma, False)
    # The peak reparametrisation of the fit round-trips and stays in (PEAK_MIN, PEAK_MAX].
    for value in (0.06, 0.5, 0.99):
        assert ide.peak_from_logit(ide.logit_from_peak(value)) == pytest.approx(value, abs=1e-6)
    assert ide.PEAK_MIN < ide.peak_from_logit(ide.logit_from_peak(2.0)) <= ide.PEAK_MAX
    assert ide.fit_objective("A", reference, reference) == 0.0 and ide.fit_objective("B", reference, reference) == pytest.approx(0.0, abs=1e-12)


# --- (5) the metrics ---


def test_metrics_on_spheres_and_half_spaces():
    """Concentric spheres of radii 6 and 10 (u = 1 inside): the Dice of
    both iso-surfaces is the set formula 2 n6 / (n6 + n10), the surface
    distance is the radius difference within the voxelisation, the masses
    are the voxel counts and the out-of-field mass counts the voxels
    outside a dose support of radius 8; half-spaces z >= 10 and z >= 14
    give the surface distance 4 exactly; empty sets give NaN."""
    n = 32
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    small, large = (r < 6).astype(np.float64), (r < 10).astype(np.float64)
    tissue = np.ones((n, n, n), dtype=bool)
    dose = np.where(r < 8, 60.0, 0.0)
    wm = np.ones((n, n, n))
    centre = (n // 2, n // 2, n // 2)
    metrics = ide.compare_fields(small, large, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    n6, n10, n8 = int(small.sum()), int(large.sum()), int((r < 8).sum())
    assert set(metrics) == set(ide.METRIC_NAMES)
    assert metrics["dice_core"] == metrics["dice_edema"] == pytest.approx(2 * n6 / (n6 + n10))
    assert metrics["assd_core_mm"] == metrics["assd_edema_mm"] == pytest.approx(4.0, abs=0.4)
    assert metrics["mass"] == n6 and metrics["ref_mass"] == n10
    assert metrics["mass_out_of_field"] == 0 and metrics["ref_mass_out_of_field"] == n10 - n8
    assert metrics["rel_l2"] == pytest.approx(np.sqrt((n10 - n6) / n10)) and metrics["max_abs_diff"] == 1.0
    assert metrics["rel_l2_log"] == pytest.approx(np.sqrt(n10 - n6) * abs(np.log(1e-6) - np.log(1 + 1e-6)) / np.linalg.norm(np.log(large + 1e-6)))
    # The spheres are centred at (n - 1) / 2, half a voxel from the seed voxel along each axis.
    assert metrics["qoi_n_core"] == n6 and metrics["qoi_V_core"] == n6 and metrics["qoi_centroid_drift"] == pytest.approx(np.sqrt(3) / 2)
    same = ide.compare_fields(large, large, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert same["dice_core"] == 1.0 and same["assd_edema_mm"] == 0.0 and same["rel_l2"] == 0.0 and same["rel_l2_log"] == 0.0
    # Anisotropic voxels scale the distance and the volumes.
    scaled = ide.compare_fields(small, large, tissue, (2.0, 1.0, 1.0), dose, centre, wm)
    assert 4.0 < scaled["assd_core_mm"] < 8.0 and scaled["mass"] == 2 * n6
    z = idx[2]
    a, b = (z >= 10).astype(np.float64), (z >= 14).astype(np.float64)
    planes = ide.compare_fields(a, b, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert planes["assd_core_mm"] == pytest.approx(4.0) and planes["dice_core"] == pytest.approx(2 * b.sum() / (a.sum() + b.sum()))
    empty = np.zeros((n, n, n))
    nothing = ide.compare_fields(empty, large, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert np.isnan(nothing["assd_core_mm"]) and nothing["dice_core"] == 0.0 and nothing["mass"] == 0.0
    both = ide.compare_fields(empty, empty, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert np.isnan(both["dice_core"]) and np.isnan(both["assd_edema_mm"]) and np.isnan(both["rel_l2"])
    assert ide.dice(large > 0, small > 0) == metrics["dice_core"] and np.isnan(ide.surface_distance(empty > 0, large > 0, (1.0, 1.0, 1.0)))


# --- (6) the design ---


def test_design_fills_the_cells(cohort, phantom_root):
    """32 patients, 8 per cell, ids p00-p31 in cell order, the first of
    each cell flagged for the finite-difference check; the cells follow
    rho T_r and Lambda = 60 alpha (1 + 2 / (alpha/beta)) + k_ct D_tot /
    gamma with D_tot the chemo dose within the 120-day horizon (4 900);
    the derived columns follow the script's derivations; the floor is 0,
    the precision f64; every config loads and constructs a solver once
    maps are given; the design is not overwritten."""
    records = _read_csv(phantom_root / "design.csv")
    spec = json.loads((phantom_root / "spec.json").read_text())
    base = json.loads((phantom_root / "base_config.json").read_text())
    assert len(records) == 32 and [r["patient"] for r in records] == [f"p{i:02d}" for i in range(32)]
    assert list(records[0]) == ide.DESIGN_COLUMNS
    assert base["gaussian_seed_floor"] == 0.0 and base["precision"] == "f64" and base["resolution_factor"] == 1.0
    assert spec["gaussian_seed_floor"] == 0.0 and spec["smoke"] is False and spec["n_candidates"] == 256
    # The phantom is too small for the default voxel: the base config's fractions, the grid centre.
    assert spec["seed_voxel"] == [12, 12, 12] and spec["seed_snap_distance_voxels"] == 0.0
    assert spec["seed_target_source"] == "base_config_fractions" and spec["default_seed_voxel"] == [132, 103, 90]
    assert spec["seed_target_voxel"] == [12, 12, 12]
    assert spec["log_kill"]["chemo_total_dose"] == 4900.0 and spec["schedules"]["substitute"]["chemo_total_dose"] == 6900.0
    assert spec["crt_snapshots"] == {"mid_crt": 34.0, "end_crt": 55.0}
    for cell, (old, strong) in ide.CELLS.items():
        members = [r for r in records if r["cell"] == cell]
        assert len(members) == 8 and [r["fd_check"] for r in members] == ["True"] + ["False"] * 7
        for r in members:
            assert (float(r["rho_T_r"]) >= 2.0) == old and (float(r["log_kill_total"]) >= 4.0) == strong
    for r in records:
        growth = sa.growth_parameters(float(r["front_speed_mm_per_day"]), float(r["front_width_mm"]))
        assert float(r["white_matter_diffusivity"]) == pytest.approx(growth["white_matter_diffusivity"])
        assert float(r["rho"]) == pytest.approx(growth["rho"]) and float(r["rho_T_r"]) == pytest.approx(float(r["rho"]) * float(r["resection_time"]))
        sigma = float(r["seed_relative_width"]) * float(r["front_width_mm"])
        seed = sa.seed_parameters(float(r["seed_peak_density"]), sigma)
        assert float(r["seed_sigma_mm"]) == pytest.approx(sigma) and float(r["gaussian_seed_mass"]) == pytest.approx(seed["gaussian_seed_mass"])
        alpha, kill = float(r["rt_alpha"]), float(r["chemo_kill_rate"])
        assert float(r["log_kill_rt"]) == pytest.approx(60 * alpha * (1 + 2 / 8.0))
        assert float(r["log_kill_ct"]) == pytest.approx(kill * 4900.0 / 9.24)
        assert float(r["log_kill_total"]) == pytest.approx(float(r["log_kill_rt"]) + float(r["log_kill_ct"]))
        assert 0.6 <= float(r["seed_peak_density"]) <= 1.0 and 30.0 <= float(r["resection_time"]) <= 200.0
    patient = cohort.patient("p05")
    assert patient.cell == "young_weak" and patient.solver_values()["rho"] == pytest.approx(float(records[5]["rho"]))
    config = read_config(phantom_root / "configs" / "p05.json", solver=StuppFKPPSolver)
    assert config["resection_cavity"] is None and config["rt_dose"] is None and config["time_after_resection"] == 120.0
    assert config["resection_time"] == patient.resection_time and config["chemo_times"][0] == pytest.approx(patient.resection_time + 14.0)
    assert len(config["chemo_times"]) == 52 and len(config["rt_times"]) == 30 and config["gaussian_seed_x_fraction"] == pytest.approx(12.5 / 24)
    shape = cohort.wm.shape
    StuppFKPPSolver({**config, "resection_cavity": np.zeros(shape, dtype=bool), "rt_dose": np.zeros(shape)})
    assert (phantom_root / "search_space.json").is_file() and (phantom_root / "figures").is_dir()
    with pytest.raises(FileExistsError):
        ide.make_design(spec["base_config"], sa.DEFAULT_SEARCH_SPACE, phantom_root.parent, phantom_root.name)


ATLAS_PRESENT = all(Path(path).is_file() for path in sa.DEFAULT_TISSUE_MAPS.values())


@pytest.mark.skipif(not ATLAS_PRESENT, reason="the atlas tissue maps are not present")
def test_default_seed_voxel_on_atlas(tmp_path):
    """On the atlas geometry the default seed voxel (132, 103, 90) is
    seedable itself, so the design resolves to it without snapping and
    records its source; an explicit --seed-voxel is snapped and recorded
    as an argument."""
    assert ide.DEFAULT_SEED_VOXEL == (132, 103, 90)
    wm, gm, zooms, _ = ide.load_tissue({key: str(path) for key, path in sa.DEFAULT_TISSUE_MAPS.items()})
    geometry = sa.seed_geometry(wm, gm, 0.1)
    assert ide.nearest_seedable_voxel(geometry, ide.DEFAULT_SEED_VOXEL) == ((132, 103, 90), 0.0)
    root = ide.make_design(sa.DEFAULT_CONFIG, sa.DEFAULT_SEARCH_SPACE, tmp_path, "atlas", tissue_maps=sa.DEFAULT_TISSUE_MAPS, log2_candidates=8)
    spec = json.loads((root / "spec.json").read_text())
    assert spec["seed_voxel"] == [132, 103, 90] and spec["seed_target_voxel"] == [132, 103, 90]
    assert spec["seed_target_source"] == "default" and spec["seed_snap_distance_voxels"] == 0.0
    n = spec["grid_shape"]
    assert spec["seed_fractions"] == pytest.approx([(v + 0.5) / n_i for v, n_i in zip((132, 103, 90), n)])
    cohort = ide.load_cohort(root)
    assert cohort.seed_voxel == (132, 103, 90) and cohort.tissue[132, 103, 90]
    config = read_config(root / "configs" / "p00.json", solver=StuppFKPPSolver)
    assert config["gaussian_seed_x_fraction"] == pytest.approx(132.5 / n[0])
    # An explicit voxel in the CSF is snapped to tissue and recorded as an argument.
    root = ide.make_design(sa.DEFAULT_CONFIG, sa.DEFAULT_SEARCH_SPACE, tmp_path, "explicit", tissue_maps=sa.DEFAULT_TISSUE_MAPS, seed_voxel=(91, 109, 91), log2_candidates=8)
    spec = json.loads((root / "spec.json").read_text())
    assert spec["seed_target_source"] == "argument" and spec["seed_target_voxel"] == [91, 109, 91]


# --- (7) the smaller pieces ---


def test_frame_days_and_log_kill_and_selection(cohort):
    """The frame days follow the script's rounding (pre = (n_growth - 1)
    dt, the Sundays the script's own days, the horizon frame before the
    horizon); the total log kill formula; the patient selection syntax;
    the Cramer-Rao errors of a singular matrix."""
    dt = 1.0 / 12.0
    days = ide.frame_days(100.0, dt, {name: ide.FRAME_MOMENTS[name] for name in ide.SUBSTITUTE_FRAMES})
    assert days["pre"] == pytest.approx(1199 * dt) and days["pre"] < 100.0
    script_days = sa.snapshot_days(100.0, {"mid_crt": 34.0, "end_crt": 55.0}, dt)
    assert days["d34"] == script_days["mid_crt"] and days["d55"] == script_days["end_crt"]
    assert 100.0 + 81.0 - 1.5 * dt < days["d80"] <= 100.0 + 81.0 - 0.5 * dt
    assert 100.0 + 120.0 - 1.5 * dt < days["d120"] <= 100.0 + 120.0 - 0.5 * dt and days["d180"] < 280.0
    kill = ide.total_log_kill(0.1, 8.0, 30, 2.0, 2e-3, 4900.0, 9.24)
    assert float(kill["log_kill_rt"]) == pytest.approx(60 * 0.1 * 1.25) and float(kill["log_kill_total"]) == pytest.approx(7.5 + 2e-3 * 4900 / 9.24)
    assert ide.cell_of(1.0, 3.0) == "young_weak" and ide.cell_of(2.0, 4.0) == "old_strong" and ide.cell_of(5.0, 1.0) == "old_weak"
    assert [p.id for p in ide.select_patients(cohort, "p03,p00-p02,p03", False)] == ["p03", "p00", "p01", "p02"]
    assert [p.id for p in ide.select_patients(cohort, None, True)] == ["p00", "p08", "p16", "p24"]
    assert len(ide.select_patients(cohort, "all", True)) == 32 and len(ide.select_patients(cohort, None, False)) == 32
    with pytest.raises(ValueError, match="unknown id"):
        ide.select_patients(cohort, "p99", False)
    e = ide.INVARIANCE_DIRECTION
    columns = np.random.default_rng(0).normal(size=(50, 5))
    columns[:, 2] = columns[:, 0]  # the T_r column equals the v column: e is a null direction
    errors, values, weakest, condition = ide.cramer_rao(columns.T @ columns)
    assert np.isinf(errors[0]) and np.isinf(errors[2]) and np.all(np.isfinite(errors[[1, 3, 4]])) and np.isinf(condition)
    assert values[0] == pytest.approx(0.0, abs=1e-9) and np.allclose(np.abs(weakest), np.abs(e), atol=1e-9)
    regular = np.random.default_rng(1).normal(size=(50, 5))
    errors, values, _, condition = ide.cramer_rao(regular.T @ regular)
    np.testing.assert_allclose(errors, np.sqrt(np.diag(np.linalg.inv(regular.T @ regular))))
    assert np.isfinite(condition) and values[0] > 0
    residual = ide.projection_residual(columns, 2)
    assert np.linalg.norm(residual) < 1e-9 * np.linalg.norm(columns[:, 2])


def test_patient_blocks_and_devices(cohort):
    """The device list: '' one CPU, ',' two CPU workers, ids stripped;
    the patient blocks: contiguous in design order, sizes differing by at
    most one with the first blocks longer, surplus devices idle, a single
    device one block."""
    assert ide.parse_devices("") == [""] and ide.parse_devices(",") == ["", ""] and ide.parse_devices("1, 2,6") == ["1", "2", "6"]
    patients = cohort.patients
    blocks = ide.patient_blocks(patients, 4)
    assert [len(b) for b in blocks] == [8, 8, 8, 8] and [p.id for b in blocks for p in b] == [p.id for p in patients]
    assert [p.id for p in blocks[1]] == [f"p{i:02d}" for i in range(8, 16)]
    assert [len(b) for b in ide.patient_blocks(patients[:5], 4)] == [2, 1, 1, 1]
    assert [len(b) for b in ide.patient_blocks(patients[:2], 4)] == [1, 1]
    assert [len(b) for b in ide.patient_blocks(patients, 1)] == [32] and ide.patient_blocks([], 3) == []
    assert [len(b) for b in ide.patient_blocks(patients[:7], 3)] == [3, 2, 2]
    command = ide.worker_command(cohort.root, "fisher", "3", patients[:2], ide.build_parser().parse_args(["fisher", "--name", "x", "--smoke", "--draws", "5"]))
    assert command[2:] == ["fisher", "--output-dir", str(cohort.root.parent), "--name", cohort.root.name, "--gpus", "3", "--patients", "p00,p01", "--no-assemble", "--smoke", "--draws", "5"]
    command = ide.worker_command(cohort.root, "substitute", "", patients[:1], ide.build_parser().parse_args(["substitute", "--name", "x", "--maxfev", "3", "--t0", "30,60"]))
    assert command[-7:] == ["--patients", "p00", "--no-assemble", "--maxfev", "3", "--t0", "30,60"] and command[7:9] == ["--gpus", ""]


def test_dispatch_two_cpu_workers_on_phantom(phantom_root):
    """fisher --smoke --gpus ',' over two patients: two CPU worker
    processes with their logs under logs/, the CSVs and figures
    assembled from both records by the dispatcher, the device split in
    fisher_summary.json; a plain in-process pass afterwards skips both
    patients and keeps the split; a dispatch whose workers fail
    (substitute at a negative T_0) raises after assembling and records
    the return codes."""
    base = ["--smoke", "--output-dir", str(phantom_root.parent), "--name", phantom_root.name]
    assert ide.main(["fisher", *base, "--gpus", ",", "--patients", "p01,p09", "--draws", "2"]) == 0
    logs = phantom_root / "logs"
    assert (logs / "fisher_cpu.log").is_file() and (logs / "fisher_cpu_1.log").is_file()
    assert "fisher p01" in (logs / "fisher_cpu.log").read_text() and "fisher p09" in (logs / "fisher_cpu_1.log").read_text()
    for patient in ("p01", "p09"):
        assert (phantom_root / "runs" / "fisher" / patient / "fisher.json").is_file()
    rows = _read_csv(phantom_root / "fisher.csv")
    assert {r["patient"] for r in rows} >= {"p01", "p09"}
    summary = ide.read_record(phantom_root / "fisher_summary.json")
    assert {"p01", "p09"} <= set(summary["patients"]) and summary["n_rows"] == len(rows)
    split = summary["dispatch"]
    assert split["devices"] == ["", ""] and split["n_failed"] == 0 and [b["patients"] for b in split["blocks"]] == [["p01"], ["p09"]]
    assert all(b["returncode"] == 0 and Path(b["log"]).is_file() for b in split["blocks"])
    assert (phantom_root / "figures" / "fisher_heatmap_cr_log_T_r.png").is_file()
    # A plain pass over the same patients skips them and keeps the split.
    assert ide.main(["fisher", *base, "--gpus", "", "--patients", "p01,p09"]) == 0
    assert ide.read_record(phantom_root / "fisher_summary.json")["dispatch"]["blocks"] == split["blocks"]
    # Failing workers: the dispatcher assembles, then raises naming the logs.
    with pytest.raises(RuntimeError, match="2 of 2 workers failed"):
        ide.main(["substitute", *base, "--gpus", ",", "--patients", "p01,p09", "--maxfev", "2", "--t0", "-5"])
    summary = ide.read_record(phantom_root / "substitute_summary.json")
    assert summary["dispatch"]["n_failed"] == 2 and all(b["returncode"] != 0 for b in summary["dispatch"]["blocks"])
    assert (logs / "substitute_cpu.log").is_file() and "Traceback" in (logs / "substitute_cpu_1.log").read_text()


def test_figures_tolerate_empty_iso_surfaces(tmp_path):
    """A metric that is NaN for every row (an empty iso-surface at every
    lambda, at every substitute) still gives a figure: the panels' log
    axes are scaled by hand."""
    nan = float("nan")
    invariance = [
        {"lambda": value, "run_type": run_type, "snapshot": "pre", **{name: nan for name in ide.METRIC_NAMES}, "mass": 1.0}
        for value in (0.5, 1.0, 2.0)
        for run_type in ("growth_fixed_n", "treated")
    ]
    ide.invariance_figures(invariance, tmp_path / "figures")
    assert (tmp_path / "figures" / "invariance_metrics.png").is_file() and (tmp_path / "figures" / "invariance_qois.pdf").is_file()
    substitute = [
        {"patient": f"p{i:02d}", "cell": cell, "T_0": t0, "rho_T_0": 0.01 * t0, "objective": objective, **{f"{frame}_{name}": nan for frame in ide.SUBSTITUTE_FRAMES for name in ide.METRIC_NAMES}}
        for i, cell in enumerate(ide.CELLS)
        for t0 in (30.0, 60.0)
        for objective in ide.OBJECTIVES
    ]
    ide.substitute_figures(substitute, tmp_path / "figures")
    for frame in ("d120", "d180"):
        for objective in ide.OBJECTIVES:
            assert (tmp_path / "figures" / f"substitute_{frame}_objective_{objective}.png").is_file()


@pytest.mark.slow
def test_smoke_pipeline_on_phantom(phantom_root):
    """The command line's ``all --smoke`` on the phantom design (the
    existing design is kept, one patient, three lambdas, two T_0): every
    CSV and figure is written and a second pass skips the finished
    patient. Tens of seconds (``slow``)."""
    spec = json.loads((phantom_root / "spec.json").read_text())
    args = ["all", "--smoke", "--output-dir", str(phantom_root.parent), "--name", phantom_root.name, "--config", spec["base_config"]]
    args += ["--white-matter-pbmap", "", "--gray-matter-pbmap", "", "--patients", "p00", "--lambdas", "0.5,1,2", "--t0", "30,60"]
    assert ide.main(args) == 0
    for name in ("invariance.csv", "fisher.csv", "fisher_regions.csv", "fisher_fd_check.csv", "fisher_runs.csv", "substitute.csv"):
        assert (phantom_root / name).is_file(), name
    # The CSVs hold every record of the shared design directory (other
    # tests add patients), so the checks look at p00's rows.
    fisher = [r for r in _read_csv(phantom_root / "fisher.csv") if r["patient"] == "p00"]
    assert len(fisher) == len(ide.OBSERVATION_SETS) * len(ide.T_R_VARIANTS)
    assert all(r["frames"] for r in fisher)
    checks = [r for r in _read_csv(phantom_root / "fisher_fd_check.csv") if r["patient"] == "p00"]
    assert [r["column"] for r in checks] == list(ide.THETA_NAMES)
    substitute = [r for r in _read_csv(phantom_root / "substitute.csv") if r["patient"] == "p00"]
    assert len(substitute) == 1 + 2 * len(ide.OBJECTIVES) and list(substitute[0]) == ide.SUBSTITUTE_COLUMNS
    assert all(int(r["n_evaluations"]) <= ide.SMOKE.maxfev for r in substitute if r["objective"] != "truth")
    for stem in ("fisher_heatmap_cr_log_T_r", "fisher_heatmap_resid7_frac_T_r", "substitute_d120_objective_A", "substitute_d180_objective_B", "invariance_qois"):
        assert (phantom_root / "figures" / f"{stem}.png").is_file() and (phantom_root / "figures" / f"{stem}.pdf").is_file()
    assert (phantom_root / "runs" / "fisher" / "p00" / "fisher.json").is_file()
    assert ide.main(args) == 0  # everything is skipped
