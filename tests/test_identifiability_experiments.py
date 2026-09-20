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
set-formula Dice, the analytic surface distance, the masses beyond the
reference's edema and the log volume ratios, (6) the design strips the
script factor, assigns the cells by the front width and the maturity,
rejects seeds wider than 1.5 front widths and counts them, records the
maturity column by its formula and the patients' time step, screens the
candidates by size (accepting on a synthetic field exactly within the
bands) and fills the cells, resuming from its screen records, (7) the
substitute arm's deficits (default -1, -0.5, 0.5, 1) give
T_0 = T_r - delta_a / rho and skip a T_0 outside its range with a NaN
row, (8) the free-lambda transform round-trips and the free fit recovers
the front width of a phantom truth within 10 %, (9) the smaller pieces:
the frame days, the total log kill, the patient selection and the
profile's default patients, the stability time step (an integer number
of steps per day between 2 and 12, "fixed" giving 12), the seed
reparametrisations, the Cramer-Rao errors of a singular Fisher matrix,
the device split and the worker command lines, and (10) a reused seedfix
or profile record of another fixed seed raises. The whole smoke pipeline
on the phantom is ``slow``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import jax
import nibabel as nib
import numpy as np
import pytest

from fisher_kpp_jax import StuppFKPPSolver, read_config, write_config

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
# The phantom's tissue is a ball of radius 9 voxels, so its patients are
# small: bands within the phantom in place of the provisional defaults. A
# mature compact tumour (a front travel of at least 7.5 mm) fills the
# ball (r_core = r_whole = 9.1 mm, ratio 1), which the bands accept.
PHANTOM_BANDS = ide.SizeBands(r_core_mm=(0.5, 9.5), r_whole_mm=(1.0, 9.5), whole_core_ratio=(1.0, 30.0))
PATIENTS_PER_CELL = 2  # the phantom design's, 8 patients p00-p07


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


def _phantom_base(tmp_path: Path) -> Path:
    """The shipped base config (its Stupp schedule, 12 steps/day) with the
    phantom's tissue maps, written under tmp_path."""
    wm, gm = _phantom_maps()
    volumes = tmp_path / "volumes"
    volumes.mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(wm, np.eye(4)), str(volumes / "wm.nii.gz"))
    nib.save(nib.Nifti1Image(gm, np.eye(4)), str(volumes / "gm.nii.gz"))
    base = read_config(sa.DEFAULT_CONFIG, solver=StuppFKPPSolver)
    base["white_matter_pbmap"] = str(volumes / "wm.nii.gz")
    base["gray_matter_pbmap"] = str(volumes / "gm.nii.gz")
    base["steps_per_day"] = 12
    write_config(base, tmp_path / "base.json")
    return tmp_path / "base.json"


@pytest.fixture(scope="session")
def phantom_root(tmp_path_factory) -> Path:
    """A design directory on the phantom: the shipped base config with the
    phantom's tissue maps, the script's search space, 256 Sobol'
    candidates screened with the phantom's bands, 2 patients per cell."""
    tmp_path = tmp_path_factory.mktemp("ide")
    base = _phantom_base(tmp_path)
    return ide.make_design(base, ide.DEFAULT_SEARCH_SPACE, tmp_path, "design", tissue_maps=None, log2_candidates=8, bands=PHANTOM_BANDS, patients_per_cell=PATIENTS_PER_CELL)


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


def test_linear_composition_rule_reproduces_growth(tmp_path):
    """A seed of mass 1e-3 and diffusion time 4 mm^2 at rho = 0.001 and
    D = 0.2 grown for 10 days, against the composition rule's seed
    (m e^{rho 6}, w + 6 D) grown for 4 days: relative L2 below 1 %;
    the rule falls back to the truth's seed when the width would not be
    positive; the peak reparametrisation round-trips."""
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
    are the voxel counts, the out-of-field mass counts the voxels
    outside a dose support of radius 8, the relative masses follow and the
    log volume ratios are log10((n6 + 1) / (n10 + 1)); half-spaces
    z >= 10 and z >= 14 give the surface distance 4 exactly; empty sets
    give NaN."""
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
    assert metrics["mass"] == n6 and metrics["ref_mass"] == n10 and metrics["mass_rel"] == pytest.approx(n6 / n10 - 1)
    assert metrics["mass_out_of_field"] == 0 and metrics["ref_mass_out_of_field"] == n10 - n8 and metrics["mass_out_of_field_rel"] == -1.0
    # Beyond the reference's edema (r >= 10) neither field has mass: NaN relative mass.
    assert metrics["mass_beyond_edema"] == 0 and metrics["ref_mass_beyond_edema"] == 0 and np.isnan(metrics["mass_beyond_edema_rel"])
    assert metrics["log_vol_ratio_core"] == metrics["log_vol_ratio_edema"] == pytest.approx(np.log10((n6 + 1) / (n10 + 1)))
    assert metrics["rel_l2"] == pytest.approx(np.sqrt((n10 - n6) / n10)) and metrics["max_abs_diff"] == 1.0
    assert metrics["rel_l2_log"] == pytest.approx(np.sqrt(n10 - n6) * abs(np.log(1e-6) - np.log(1 + 1e-6)) / np.linalg.norm(np.log(large + 1e-6)))
    # The spheres are centred at (n - 1) / 2, half a voxel from the seed voxel along each axis.
    assert metrics["qoi_n_core"] == n6 and metrics["qoi_V_core"] == n6 and metrics["qoi_centroid_drift"] == pytest.approx(np.sqrt(3) / 2)
    same = ide.compare_fields(large, large, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert same["dice_core"] == 1.0 and same["assd_edema_mm"] == 0.0 and same["rel_l2"] == 0.0 and same["rel_l2_log"] == 0.0
    assert same["mass_rel"] == 0.0 and same["log_vol_ratio_core"] == 0.0 and same["log_vol_ratio_edema"] == 0.0
    # Anisotropic voxels scale the distance and the volumes.
    scaled = ide.compare_fields(small, large, tissue, (2.0, 1.0, 1.0), dose, centre, wm)
    assert 4.0 < scaled["assd_core_mm"] < 8.0 and scaled["mass"] == 2 * n6
    z = idx[2]
    a, b = (z >= 10).astype(np.float64), (z >= 14).astype(np.float64)
    planes = ide.compare_fields(a, b, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert planes["assd_core_mm"] == pytest.approx(4.0) and planes["dice_core"] == pytest.approx(2 * b.sum() / (a.sum() + b.sum()))
    empty = np.zeros((n, n, n))
    nothing = ide.compare_fields(empty, large, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert np.isnan(nothing["assd_core_mm"]) and nothing["dice_core"] == 0.0 and nothing["mass"] == 0.0 and nothing["mass_rel"] == -1.0
    both = ide.compare_fields(empty, empty, tissue, (1.0, 1.0, 1.0), dose, centre, wm)
    assert np.isnan(both["dice_core"]) and np.isnan(both["assd_edema_mm"]) and np.isnan(both["rel_l2"]) and np.isnan(both["mass_rel"])
    assert ide.dice(large > 0, small > 0) == metrics["dice_core"] and np.isnan(ide.surface_distance(empty > 0, large > 0, (1.0, 1.0, 1.0)))


def test_mass_beyond_edema_and_log_vol_ratio_on_phantom_pair():
    """A reference of u* = 1 inside r < 10 with a tail of 0.1 on
    10 <= r < 14 against a field of u = 1 inside r < 6 with a tail of 0.2
    on the same shell: the region beyond the reference's edema is r >= 10
    (the tails are below 0.3), so mass_beyond_edema is 0.2 times the
    shell's voxels, the reference's 0.1 times them and the relative mass
    exactly 1; the log volume ratios are log10((n6 + 1) / (n10 + 1)) for
    the core and the edema alike; the same field against itself gives 0."""
    n = 36
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    shell = (r >= 10) & (r < 14)
    reference = (r < 10).astype(np.float64) + 0.1 * shell
    field = (r < 6).astype(np.float64) + 0.2 * shell
    tissue = np.ones((n, n, n), dtype=bool)
    dose = np.where(r < 12, 60.0, 0.0)
    metrics = ide.compare_fields(field, reference, tissue, (1.0, 1.0, 1.0), dose, (n // 2,) * 3, np.ones((n, n, n)))
    n6, n10, n_shell = int((r < 6).sum()), int((r < 10).sum()), int(shell.sum())
    assert metrics["mass_beyond_edema"] == pytest.approx(0.2 * n_shell) and metrics["ref_mass_beyond_edema"] == pytest.approx(0.1 * n_shell)
    assert metrics["mass_beyond_edema_rel"] == pytest.approx(1.0)
    assert metrics["mass_rel"] == pytest.approx((n6 + 0.2 * n_shell) / (n10 + 0.1 * n_shell) - 1)
    assert metrics["log_vol_ratio_core"] == pytest.approx(np.log10((n6 + 1) / (n10 + 1)))
    assert metrics["log_vol_ratio_edema"] == pytest.approx(np.log10((n6 + 1) / (n10 + 1)))
    # Anisotropic voxels: the volumes scale with dV, the log ratios do not.
    scaled = ide.compare_fields(field, reference, tissue, (2.0, 1.0, 1.0), dose, (n // 2,) * 3, np.ones((n, n, n)))
    assert scaled["mass_beyond_edema"] == pytest.approx(0.4 * n_shell) and scaled["log_vol_ratio_core"] == pytest.approx(metrics["log_vol_ratio_core"])
    same = ide.compare_fields(reference, reference, tissue, (1.0, 1.0, 1.0), dose, (n // 2,) * 3, np.ones((n, n, n)))
    assert same["mass_beyond_edema_rel"] == 0.0 and same["log_vol_ratio_core"] == 0.0 and same["log_vol_ratio_edema"] == 0.0


# --- (6) the design ---


def test_script_factor_stripping(tmp_path):
    """The script's search space carries growth_efolds with "script": true:
    ``load_script_search_space`` returns it as a script factor (1-12, log)
    and a space without it, whose factors are the six solver factors and
    the three seed fractions; the shared loader refuses the unstripped
    file; a script factor with a malformed range or one that is also a
    solver factor is refused."""
    space, script = ide.load_script_search_space(ide.DEFAULT_SEARCH_SPACE, StuppFKPPSolver.config_keys())
    assert list(script) == [ide.GROWTH_EFOLDS_FACTOR] and (script["growth_efolds"].low, script["growth_efolds"].high, script["growth_efolds"].scale) == (1.0, 12.0, "log")
    assert ide.GROWTH_EFOLDS_FACTOR not in space.factors and "resection_time" not in space.factors
    assert [name for name in ide.SAMPLED_FACTORS if name != ide.GROWTH_EFOLDS_FACTOR] == [name for name in space.factors if not name.startswith("gaussian_seed_")]
    assert space.overrides == {"diffusivity_ratio": 10.0, "chemo_decay_rate": 9.24, "rt_alpha_beta_ratio": 8.0, "gaussian_seed_scale": 1.0}
    assert space.factors["front_width_mm"].high == 8.0 and space.factors["seed_sigma_mm"].low == 1.0 and space.factors["chemo_kill_rate"].low == 5e-5
    with pytest.raises(ValueError, match="unknown key"):
        sa.load_search_space(ide.DEFAULT_SEARCH_SPACE, StuppFKPPSolver.config_keys())
    entries = json.loads(Path(ide.DEFAULT_SEARCH_SPACE).read_text())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**entries, "growth_efolds": {"min": 12.0, "max": 1.0, "scale": "log", "script": True}}))
    with pytest.raises(ValueError, match="min < max"):
        ide.load_script_search_space(bad, StuppFKPPSolver.config_keys())
    bad.write_text(json.dumps({**entries, "growth_efolds": {"min": 1.0, "max": 12.0, "script": True}}))
    with pytest.raises(ValueError, match="script factor"):
        ide.load_script_search_space(bad, StuppFKPPSolver.config_keys())
    # Without the script flag growth_efolds is an unknown key to the shared loader.
    bad.write_text(json.dumps({**entries, "growth_efolds": {"min": 1.0, "max": 12.0, "scale": "log"}}))
    with pytest.raises(ValueError, match="unknown key"):
        ide.load_script_search_space(bad, StuppFKPPSolver.config_keys())
    # A second script factor is stripped and recorded too.
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({**entries, "other_thing": {"min": 0.0, "max": 1.0, "scale": "linear", "script": True}}))
    space, script = ide.load_script_search_space(extra, StuppFKPPSolver.config_keys())
    assert set(script) == {"growth_efolds", "other_thing"} and "other_thing" not in space.factors


def test_cell_assignment_on_maturity_and_seed_cap(phantom_root):
    """The cells split at front_width_mm = 2 (at or below: compact) and at
    maturity = 2 growth_efolds front_width_mm^2 / seed_sigma_mm^2 = 15 (at
    or above: mature); the Sobol' candidates of the script's search space
    reject a seed wider than 1.5 front widths as seed_wide before the T_r
    rejection, and the count matches; visibility_margin stays a column
    whose sign says whether the untreated regrowth over 120 days exceeds
    the log kill."""
    assert ide.MATURITY_SPLIT == 15.0 and ide.LAMBDA_SPLIT == 2.0 and ide.SEED_WIDTH_CAP == 1.5 and ide.PATIENTS_PER_CELL == 4
    assert ide.cell_of(2.0, 15.0) == "compact_mature" and ide.cell_of(2.0, 15.0 - 1e-9) == "compact_immature"
    assert ide.cell_of(2.0 + 1e-9, 15.0) == "broad_mature" and ide.cell_of(8.0, 3.0) == "broad_immature"
    assert ide.cell_of(1.0, 50.0) == "compact_mature" and list(ide.CELLS) == ["compact_immature", "compact_mature", "broad_immature", "broad_mature"]
    assert ide.CELLS["broad_mature"] == (True, True) and ide.CELLS["compact_immature"] == (False, False)
    assert float(ide.maturity(2.0, 3.0, 1.5)) == pytest.approx(2 * 2 * 9 / 2.25) == pytest.approx(16.0)
    np.testing.assert_allclose(ide.maturity(np.array([1.0, 12.0]), np.array([1.0, 8.0]), np.array([1.0, 4.0])), [2.0, 2 * 12 * 64 / 16])
    # ell lambda / sigma^2 with ell = v T_r = 2 a lambda.
    a, width, sigma = 3.0, 2.5, 2.0
    v = 0.1
    t_r = a / sa.growth_parameters(v, width)["rho"]
    assert float(ide.maturity(a, width, sigma)) == pytest.approx(v * t_r * width / sigma**2)
    base = read_config(phantom_root / "base_config.json", solver=StuppFKPPSolver)
    space, script = ide.load_script_search_space(ide.DEFAULT_SEARCH_SPACE, StuppFKPPSolver.config_keys())
    candidates = ide.sample_candidates(space, script, base, sa.treatment_settings(), 4900.0, 30, 8, 1, tr_max=1000.0)
    wide = [c for c in candidates if c["seed_sigma_mm"] > 1.5 * c["front_width_mm"]]
    assert wide and all(c["rejected"] == "seed_wide" for c in wide)
    assert sum(1 for c in candidates if c["rejected"] == "seed_wide") == len(wide)
    assert all(c["rejected"] != "seed_wide" for c in candidates if c["seed_sigma_mm"] <= 1.5 * c["front_width_mm"])
    # The seed cap precedes the T_r rejection: a wide seed is never counted under T_r.
    assert all(c["rejected"] == "T_r_above_max" for c in candidates if c not in wide and c["resection_time"] > 1000.0)
    assert any(c["resection_time"] > 1000.0 for c in wide)
    for c in candidates:
        assert c["maturity"] == pytest.approx(2 * c["growth_efolds"] * c["front_width_mm"] ** 2 / c["seed_sigma_mm"] ** 2)
        assert c["cell"] == ide.cell_of(c["front_width_mm"], c["maturity"])
    assert float(ide.visibility_margin(0.05, 4.0)) == pytest.approx(120 * 0.05 - 4.0) and float(ide.visibility_margin(0.05, 4.0)) > 0
    assert float(ide.visibility_margin(0.01, 4.0)) == pytest.approx(1.2 - 4.0) and float(ide.visibility_margin(0.01, 4.0)) < 0
    np.testing.assert_allclose(ide.visibility_margin(np.array([0.01, 0.05]), np.array([1.2, 6.0])), [0.0, 0.0])
    kill = ide.total_log_kill(0.1, 8.0, 30, 2.0, 2e-3, 4900.0, 9.24)
    assert float(ide.visibility_margin(0.1, kill["log_kill_total"])) == pytest.approx(12.0 - 7.5 - 2e-3 * 4900 / 9.24)


def test_size_screening_acceptance_on_synthetic_field():
    """A field of 1 inside r < 4 and 0.45 on 4 <= r < 8 on an all-tissue
    grid: the core is the ball of radius 4, the whole set the ball of
    radius 8, the equivalent-sphere radii follow from the voxel counts
    (within the voxelisation, 4 and 8) and the ratio is their volume
    ratio (about 8); bands around these accept, a core band that
    excludes 4 rejects with "r_core_mm", an empty core rejects with
    "empty_core", NaN never passes; anisotropic voxels scale the radii."""
    n = 24
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    density = np.where(r < 4, 1.0, np.where(r < 8, 0.45, 0.0))
    tissue = np.ones((n, n, n), dtype=bool)
    sizes = ide.size_screen(density, tissue, (1.0, 1.0, 1.0))
    n4, n8 = int((r < 4).sum()), int((r < 8).sum())
    assert sizes["n_core"] == n4 and sizes["n_whole"] == n8 and sizes["max_density"] == 1.0
    assert sizes["r_core_mm"] == pytest.approx(ide.equivalent_radius(n4)) == pytest.approx(4.0, abs=0.3)
    assert sizes["r_whole_mm"] == pytest.approx(ide.equivalent_radius(n8)) == pytest.approx(8.0, abs=0.3)
    assert sizes["whole_core_ratio"] == pytest.approx(n8 / n4) == pytest.approx(8.0, rel=0.25)
    assert ide.equivalent_radius(4 / 3 * np.pi * 5.0**3) == pytest.approx(5.0)
    assert ide.accept_sizes(sizes, ide.SizeBands((3.0, 5.0), (7.0, 9.0), (1.2, 10.0))) is None
    assert ide.accept_sizes(sizes, ide.SizeBands((5.0, 25.0), (7.0, 9.0), (1.2, 10.0))) == "r_core_mm"
    assert ide.accept_sizes(sizes, ide.SizeBands((3.0, 5.0), (10.0, 35.0), (1.2, 10.0))) == "r_whole_mm"
    assert ide.accept_sizes(sizes, ide.SizeBands((3.0, 5.0), (7.0, 9.0), (1.2, 5.0))) == "whole_core_ratio"
    assert ide.accept_sizes(sizes, ide.DEFAULT_BANDS) == "r_core_mm"
    # The tissue mask restricts the sets; an empty core is rejected before the bands.
    half = idx[0] >= n // 2
    halved = ide.size_screen(density, half, (1.0, 1.0, 1.0))
    assert halved["n_core"] == int(((r < 4) & half).sum()) and halved["r_core_mm"] < sizes["r_core_mm"]
    empty = ide.size_screen(np.where(r < 8, 0.45, 0.0), tissue, (1.0, 1.0, 1.0))
    assert empty["n_core"] == 0 and empty["r_core_mm"] == 0.0 and np.isnan(empty["whole_core_ratio"])
    assert ide.accept_sizes(empty, ide.SizeBands((0.0, 25.0), (7.0, 9.0), (0.0, 10.0))) == "empty_core"
    assert ide.accept_sizes({**sizes, "r_whole_mm": float("nan")}, ide.SizeBands((3.0, 5.0), (7.0, 9.0), (1.2, 10.0))) == "r_whole_mm"
    scaled = ide.size_screen(density, tissue, (2.0, 2.0, 2.0))
    assert scaled["r_core_mm"] == pytest.approx(2 * sizes["r_core_mm"]) and scaled["whole_core_ratio"] == pytest.approx(sizes["whole_core_ratio"])


def test_design_fills_the_cells_by_size_screening(cohort, phantom_root):
    """8 patients (2 per cell), ids p00-p07 in cell order, the first of
    each cell flagged for the finite-difference check; the cells follow
    the front width and the maturity; no accepted seed is wider than 1.5
    front widths and the seed-wide rejections are counted;
    resection_time = growth_efolds / rho within [5, 1000] days; the
    derived columns follow the script's derivations (the seed from
    seed_sigma_mm in mm, the maturity from its formula); the time step is
    the stability mode's per patient and is in every config; the sizes
    lie within the bands and match the screen records; the counts add
    up; the floor is 0, the precision f64, the search space's overrides
    are in the base config; every config loads and constructs a solver
    once maps are given; the design is not overwritten, and a copy
    without spec.json is resumed from its screen records without a new
    solve to the same design.csv."""
    records = _read_csv(phantom_root / "design.csv")
    spec = json.loads((phantom_root / "spec.json").read_text())
    base = json.loads((phantom_root / "base_config.json").read_text())
    n_patients = PATIENTS_PER_CELL * len(ide.CELLS)
    assert len(records) == n_patients and [r["patient"] for r in records] == [f"p{i:02d}" for i in range(n_patients)]
    assert list(records[0]) == ide.DESIGN_COLUMNS
    assert base["gaussian_seed_floor"] == 0.0 and base["precision"] == "f64" and base["resolution_factor"] == 1.0
    assert base["diffusivity_ratio"] == 10.0 and base["rt_alpha_beta_ratio"] == 8.0 and base["chemo_decay_rate"] == 9.24
    assert spec["gaussian_seed_floor"] == 0.0 and spec["smoke"] is False and spec["n_candidates"] == 256 and spec["patients_per_cell"] == PATIENTS_PER_CELL
    assert spec["script_factors"] == ["growth_efolds"] and spec["sampled_factors"] == list(ide.SAMPLED_FACTORS)
    assert spec["factors"]["growth_efolds"] == {"min": 1.0, "max": 12.0, "scale": "log"} and spec["fixed_parameters"] == {"diffusivity_ratio": 10.0, "rt_alpha_beta_ratio": 8.0, "chemo_decay_rate": 9.24}
    # The phantom is too small for the default voxel: the base config's fractions, the grid centre.
    assert spec["seed_voxel"] == [12, 12, 12] and spec["seed_snap_distance_voxels"] == 0.0
    assert spec["seed_target_source"] == "base_config_fractions" and spec["default_seed_voxel"] == [132, 103, 90]
    assert spec["log_kill"]["chemo_total_dose"] == 4900.0 and spec["schedules"]["substitute"]["chemo_total_dose"] == 6900.0
    assert spec["crt_snapshots"] == {"mid_crt": 34.0, "end_crt": 55.0}
    assert spec["maturity_split"] == 15.0 and spec["lambda_split_mm"] == 2.0 and spec["seed_width_cap"]["cap"] == 1.5
    assert spec["dt_mode"] == "stability" and spec["dt_modes"]["grid_spacing_mm"] == 1.0 and spec["dt_modes"]["dt_max_days"] == 0.5
    assert spec["dt_modes"]["rho_dt_max"] == 0.01 and spec["substitute"]["T_0_min"] == 5.0 and spec["substitute"]["T_0_max"] == 3000.0
    assert spec["profile"]["sigmas"] == [1.0, 1.5, 2.0, 3.0, 4.0, 6.0] and spec["substitute"]["delta_a"] == [-1.0, -0.5, 0.5, 1.0] and spec["substitute"]["maxfev"] == 150
    assert spec["seedfix"]["lambda_bounds"] == [0.5, 8.0] and spec["seedfix"]["default_lambda_mode"] == "fixed" and spec["seedfix"]["maxfev"] == 150
    screening = spec["screening"]
    assert screening["tr_min"] == 5.0 and screening["tr_max"] == 1000.0 and screening["bands"] == PHANTOM_BANDS.record() and screening["resumed"] is False
    counts = screening["counts"]
    assert counts["n_candidates"] == 256 and counts["n_accepted"] == n_patients and counts["n_solve_failed"] == 0
    assert counts["n_screened"] == sum(c["n_screened"] for c in counts["cells"].values()) >= n_patients
    assert counts["n_rejected_seed_wide"] > 0
    assert counts["n_rejected_seed_wide"] + counts["n_rejected_T_r_below_min"] + counts["n_rejected_T_r_above_max"] + sum(c["n_admissible"] for c in counts["cells"].values()) == 256
    for cell, (broad, mature) in ide.CELLS.items():
        members = [r for r in records if r["cell"] == cell]
        assert len(members) == PATIENTS_PER_CELL and [r["fd_check"] for r in members] == ["True"] + ["False"] * (PATIENTS_PER_CELL - 1)
        cell_counts = counts["cells"][cell]
        assert cell_counts["n_accepted"] == PATIENTS_PER_CELL and spec["cells"][cell] == {"broad": broad, "mature": mature}
        assert cell_counts["n_screened"] == cell_counts["n_accepted"] + cell_counts["n_solve_failed"] + sum(cell_counts["n_rejected"].values())
        for r in members:
            assert (float(r["front_width_mm"]) > 2.0) == broad and (float(r["maturity"]) >= 15.0) == mature
    for r in records:
        assert float(r["seed_sigma_mm"]) <= 1.5 * float(r["front_width_mm"])
        assert float(r["maturity"]) == pytest.approx(2 * float(r["growth_efolds"]) * float(r["front_width_mm"]) ** 2 / float(r["seed_sigma_mm"]) ** 2)
        steps = ide.steps_per_day_for("stability", float(r["white_matter_diffusivity"]), float(r["rho"]), float(r["resection_time"]), 1.0)
        assert int(r["steps_per_day"]) == steps and float(r["dt"]) == pytest.approx(1.0 / steps) and 2 <= steps <= 12
        assert float(r["rho_dt"]) == pytest.approx(float(r["rho"]) / steps) and float(r["rho_dt"]) <= 0.01 * (1 + 1e-9)
        growth = sa.growth_parameters(float(r["front_speed_mm_per_day"]), float(r["front_width_mm"]))
        assert float(r["white_matter_diffusivity"]) == pytest.approx(growth["white_matter_diffusivity"])
        assert float(r["rho"]) == pytest.approx(growth["rho"]) and float(r["rho_T_r"]) == pytest.approx(float(r["growth_efolds"]))
        assert float(r["resection_time"]) == pytest.approx(float(r["growth_efolds"]) / float(r["rho"])) and 5.0 <= float(r["resection_time"]) <= 1000.0
        assert float(r["ell_mm"]) == pytest.approx(float(r["front_speed_mm_per_day"]) * float(r["resection_time"]))
        seed = sa.seed_parameters(float(r["seed_peak_density"]), float(r["seed_sigma_mm"]))
        assert float(r["gaussian_seed_mass"]) == pytest.approx(seed["gaussian_seed_mass"]) and float(r["gaussian_seed_diffusion_time"]) == pytest.approx(seed["gaussian_seed_diffusion_time"])
        assert float(r["s"]) == pytest.approx(float(r["seed_sigma_mm"]) / float(r["front_width_mm"])) and 1.0 <= float(r["seed_sigma_mm"]) <= 5.0
        alpha, kill = float(r["rt_alpha"]), float(r["chemo_kill_rate"])
        assert float(r["log_kill_rt"]) == pytest.approx(60 * alpha * (1 + 2 / 8.0))
        assert float(r["log_kill_ct"]) == pytest.approx(kill * 4900.0 / 9.24)
        assert float(r["log_kill_total"]) == pytest.approx(float(r["log_kill_rt"]) + float(r["log_kill_ct"]))
        assert float(r["visibility_margin"]) == pytest.approx(120.0 * float(r["rho"]) - float(r["log_kill_total"]))
        for name in ide.SIZE_COLUMNS:
            lo, hi = getattr(PHANTOM_BANDS, name)
            assert lo <= float(r[name]) <= hi
        assert float(r["R_over_lambda"]) == pytest.approx(float(r["r_whole_mm"]) / float(r["front_width_mm"]))
        screen = ide.read_record(phantom_root / "screen" / f"c{int(r['candidate']):04d}" / "screen.json")
        assert screen["failed"] is False and screen["r_core_mm"] == pytest.approx(float(r["r_core_mm"])) and screen["n_core"] > 0
        assert screen["resection_time"] == pytest.approx(float(r["resection_time"])) and screen["wall_time_s"] > 0
        assert screen["steps_per_day"] == int(r["steps_per_day"]) and screen["dt_refined"] is False
        assert screen["n_steps"] == ide.requested_steps(float(r["resection_time"]), int(r["steps_per_day"])) == int(np.ceil(float(r["resection_time"]) * int(r["steps_per_day"]) - 1e-9))
        assert screen["dt"] == pytest.approx(float(r["resection_time"]) / screen["n_steps"]) and screen["dt"] <= float(r["dt"]) * (1 + 1e-9)
        assert 0.6 <= float(r["seed_peak_density"]) <= 1.0 and 1.0 <= float(r["front_width_mm"]) <= 8.0
    assert len(list((phantom_root / "screen").glob("c*/screen.json"))) == counts["n_screened"]
    patient = cohort.patient("p05")
    assert patient.cell == "broad_immature" and patient.solver_values()["rho"] == pytest.approx(float(records[5]["rho"]))
    assert patient.r_over_lambda == pytest.approx(float(records[5]["R_over_lambda"])) and patient.fd_check is False and cohort.patient("p04").fd_check is True
    assert patient.maturity == pytest.approx(float(records[5]["maturity"])) and patient.steps_per_day == int(records[5]["steps_per_day"])
    assert patient.dt == pytest.approx(1.0 / patient.steps_per_day) and patient.solver_values()["steps_per_day"] == patient.steps_per_day
    assert patient.solver_values()["dt"] is None and patient.solver_values()["n_steps"] is None
    config = read_config(phantom_root / "configs" / "p05.json", solver=StuppFKPPSolver)
    assert config["steps_per_day"] == patient.steps_per_day and config["dt"] is None and config["n_steps"] is None
    assert config["resection_cavity"] is None and config["rt_dose"] is None and config["time_after_resection"] == 120.0
    assert config["resection_time"] == patient.resection_time and config["chemo_times"][0] == pytest.approx(patient.resection_time + 14.0)
    assert len(config["chemo_times"]) == 52 and len(config["rt_times"]) == 30 and config["gaussian_seed_x_fraction"] == pytest.approx(12.5 / 24)
    shape = cohort.wm.shape
    StuppFKPPSolver({**config, "resection_cavity": np.zeros(shape, dtype=bool), "rt_dose": np.zeros(shape)})
    assert (phantom_root / "search_space.json").is_file() and (phantom_root / "figures").is_dir()
    with pytest.raises(FileExistsError):
        ide.make_design(spec["base_config"], ide.DEFAULT_SEARCH_SPACE, phantom_root.parent, phantom_root.name, bands=PHANTOM_BANDS, patients_per_cell=PATIENTS_PER_CELL)
    # Resume: a copy without spec.json and design.csv is completed from its screen records.
    copy = phantom_root.parent / "resumed"
    shutil.copytree(phantom_root, copy, ignore=shutil.ignore_patterns("spec.json", "design.csv", "runs", "figures", "configs"))
    stamps = {path: path.stat().st_mtime_ns for path in copy.glob("screen/c*/screen.json")}
    resumed = ide.make_design(spec["base_config"], ide.DEFAULT_SEARCH_SPACE, copy.parent, copy.name, tissue_maps=None, log2_candidates=8, bands=PHANTOM_BANDS, patients_per_cell=PATIENTS_PER_CELL)
    assert (resumed / "design.csv").read_text() == (phantom_root / "design.csv").read_text()
    assert json.loads((resumed / "spec.json").read_text())["screening"]["resumed"] is True
    assert {path: path.stat().st_mtime_ns for path in copy.glob("screen/c*/screen.json")} == stamps
    # Another seed under the same screen records is refused.
    shutil.rmtree(copy / "configs")
    (copy / "spec.json").unlink()
    with pytest.raises(ValueError, match="another candidate"):
        ide.make_design(spec["base_config"], ide.DEFAULT_SEARCH_SPACE, copy.parent, copy.name, tissue_maps=None, log2_candidates=8, seed=2, bands=PHANTOM_BANDS, patients_per_cell=PATIENTS_PER_CELL)
    # So is the other time-step mode (the screen records carry the step).
    with pytest.raises(ValueError, match="another candidate"):
        ide.make_design(spec["base_config"], ide.DEFAULT_SEARCH_SPACE, copy.parent, copy.name, tissue_maps=None, log2_candidates=8, bands=PHANTOM_BANDS, patients_per_cell=PATIENTS_PER_CELL, dt_mode="fixed")


def test_design_fails_with_counts_when_a_cell_cannot_be_filled(tmp_path):
    """Bands no phantom tumour meets (a core radius of 20-30 mm on a
    24^3 grid) leave every cell short: the design raises after screening,
    naming the cells and the counts, and leaves its screen records."""
    base = _phantom_base(tmp_path)
    bands = ide.SizeBands(r_core_mm=(20.0, 30.0), r_whole_mm=(1.0, 40.0), whole_core_ratio=(1.0, 30.0))
    with pytest.raises(ValueError, match="could not be filled") as error:
        ide.make_design(base, ide.DEFAULT_SEARCH_SPACE, tmp_path, "short", tissue_maps=None, log2_candidates=3, bands=bands, patients_per_cell=1)
    assert "compact_immature" in str(error.value) and "n_screened" in str(error.value) and "maturity" in str(error.value)
    assert not (tmp_path / "short" / "spec.json").is_file() and list((tmp_path / "short" / "screen").glob("c*/screen.json"))


ATLAS_PRESENT = all(Path(path).is_file() for path in sa.DEFAULT_TISSUE_MAPS.values())


@pytest.mark.slow
@pytest.mark.skipif(not ATLAS_PRESENT, reason="the atlas tissue maps are not present")
def test_default_seed_voxel_on_atlas(tmp_path):
    """On the atlas geometry the default seed voxel (132, 103, 90) is
    seedable itself, so the design (smoke resolution, one patient per
    cell) resolves to it without snapping and records its source; an
    explicit --seed-voxel is snapped and recorded as an argument."""
    assert ide.DEFAULT_SEED_VOXEL == (132, 103, 90)
    wm, gm, zooms, _ = ide.load_tissue({key: str(path) for key, path in sa.DEFAULT_TISSUE_MAPS.items()})
    geometry = sa.seed_geometry(wm, gm, 0.1)
    assert ide.nearest_seedable_voxel(geometry, ide.DEFAULT_SEED_VOXEL) == ((132, 103, 90), 0.0)
    root = ide.make_design(sa.DEFAULT_CONFIG, ide.DEFAULT_SEARCH_SPACE, tmp_path, "atlas", tissue_maps=sa.DEFAULT_TISSUE_MAPS, log2_candidates=8, smoke=True, patients_per_cell=1)
    spec = json.loads((root / "spec.json").read_text())
    assert spec["seed_voxel"] == [132, 103, 90] and spec["seed_target_voxel"] == [132, 103, 90]
    assert spec["seed_target_source"] == "default" and spec["seed_snap_distance_voxels"] == 0.0 and spec["smoke"] is True
    n = spec["grid_shape"]
    assert spec["seed_fractions"] == pytest.approx([(v + 0.5) / n_i for v, n_i in zip((132, 103, 90), n)])
    cohort = ide.load_cohort(root)
    assert cohort.seed_voxel == (132, 103, 90) and cohort.tissue[132, 103, 90] and len(cohort.patients) == len(ide.CELLS)
    config = read_config(root / "configs" / "p00.json", solver=StuppFKPPSolver)
    assert config["gaussian_seed_x_fraction"] == pytest.approx(132.5 / n[0])
    # An explicit voxel in the CSF is snapped to tissue and recorded as an argument.
    root = ide.make_design(sa.DEFAULT_CONFIG, ide.DEFAULT_SEARCH_SPACE, tmp_path, "explicit", tissue_maps=sa.DEFAULT_TISSUE_MAPS, seed_voxel=(91, 109, 91), log2_candidates=8, smoke=True, patients_per_cell=1)
    spec = json.loads((root / "spec.json").read_text())
    assert spec["seed_target_source"] == "argument" and spec["seed_target_voxel"] == [91, 109, 91]


# --- (7) the substitute arm's deficits ---


def test_deficit_t0_and_sub_5_day_skip(tmp_path):
    """T_0 = T_r - delta_a / rho: for the fast patient (rho = 0.15,
    T_r = 12) delta_a = 0.5 gives T_0 = 8.67 and delta_a = 100 a T_0 far
    below 5 days, skipped; a negative deficit is never skipped, however
    late its T_0. ``substitute_patient`` on the cube cohort writes the truth row,
    one fitted row (T0_8.7/B/) and one skipped row with NaN metrics, and
    records the skip."""
    patient = _fast_patient()
    assert ide.SUBSTITUTE_DELTA_A == (-1.0, -0.5, 0.5, 1.0) and ide.MAXFEV == 150 and not hasattr(ide, "SUBSTITUTE_MAXFEV")
    assert ide.parse_floats(None, ide.SUBSTITUTE_DELTA_A, "--delta-a") == [-1.0, -0.5, 0.5, 1.0]
    assert [entry["delta_a"] for entry in ide.deficit_schedule(patient)] == [-1.0, -0.5, 0.5, 1.0]
    assert ide.build_parser().parse_args(["substitute", "--name", "x"]).maxfev is None  # resolved to MAXFEV by the command
    assert ide.substitute_t0(patient, 0.5) == pytest.approx(12.0 - 0.5 / 0.15) and ide.substitute_t0(patient, -0.5) == pytest.approx(12.0 + 0.5 / 0.15)
    assert ide.substitute_t0(patient, 0.0) == 12.0
    schedule = ide.deficit_schedule(patient, (0.5, 100.0, -200.0, -2.0))
    assert [entry["skipped"] for entry in schedule] == [None, "T_0_below_min", None, None]
    assert schedule[0]["T_0"] == pytest.approx(8.6667, abs=1e-3) and schedule[1]["T_0"] < 5.0 and schedule[2]["T_0"] > 1000.0
    assert [entry["delta_a"] for entry in schedule] == [0.5, 100.0, -200.0, -2.0]
    assert ide.deficit_schedule(patient, (-2.0,))[0]["skipped"] is None  # no cap above: T_0 = 25.3 > T_r
    cohort = _cube_cohort(tmp_path)
    record = ide.substitute_patient(cohort, patient, tmp_path / "substitute", maxfev=2, deltas=(0.5, 100.0))
    rows = record["rows"]
    assert [row["objective"] for row in rows] == ["truth", "B", "B"] and [row["delta_a"] for row in rows] == [0.0, 0.5, 100.0]
    assert rows[0]["T_0"] == 12.0 and rows[0]["skipped"] == "" and rows[0]["d180_mass_rel"] == 0.0 and np.isnan(rows[0]["maturity"])
    assert record["steps_per_day"] is None and record["dt"] == pytest.approx(1.0 / 12.0) and record["dt_refined"] is False
    fitted, skipped = rows[1], rows[2]
    assert fitted["skipped"] == "" and fitted["T_0"] == pytest.approx(8.6667, abs=1e-3) and fitted["rho_T_0"] == pytest.approx(0.15 * fitted["T_0"])
    assert fitted["n_evaluations"] <= 3 and np.isfinite(fitted["d180_dice_edema"]) and np.isfinite(fitted["d180_mass_rel"])
    # Before the resection the truth's edema leaves room beyond it; at d180 the fast tumour fills the cube above 0.3 (a zero reference: NaN).
    assert np.isfinite(fitted["pre_mass_beyond_edema_rel"]) and fitted["pre_ref_mass_beyond_edema"] > 0 and np.isnan(fitted["d180_mass_beyond_edema_rel"])
    assert skipped["skipped"] == "T_0_below_min" and skipped["T_0"] < 5.0 and "fitted_peak" not in skipped
    assert all(np.isnan(skipped[f"{frame}_{name}"]) for frame in ide.SUBSTITUTE_FRAMES for name in ide.METRIC_NAMES)
    assert record["skipped"] == [{"delta_a": 100.0, "T_0": skipped["T_0"], "skipped": "T_0_below_min", "objective": "B"}]
    assert record["T_0_range"] == [5.0, 3000.0]
    patient_dir = tmp_path / "substitute" / "px"
    assert (patient_dir / "T0_8.7" / "B" / "row.json").is_file() and (patient_dir / "T0_8.7" / "B" / "run" / "d180_cell_density.nii.gz").is_file()
    assert ide.read_record(patient_dir / "T0_8.7" / "B" / "fit.json")["delta_a"] == 0.5
    assert not list(patient_dir.glob("T0_-*")) and (patient_dir / "substitute.json").is_file()
    # The CSV of the rows: the skipped row's metrics are empty fields.
    sa.write_csv(tmp_path / "substitute.csv", rows, ide.SUBSTITUTE_COLUMNS)
    table = _read_csv(tmp_path / "substitute.csv")
    assert list(table[0]) == ide.SUBSTITUTE_COLUMNS and table[2]["skipped"] == "T_0_below_min" and table[2]["d180_mass"] == ""
    # The summary's medians exclude the skipped row.
    medians = ide.substitute_medians(rows)["B"]
    assert medians["0.5"]["all"]["n_skipped"] == 0 and medians["100"]["all"] == {"n_rows": 1, "n_skipped": 1, **{k: None for k in ide.frame_metric_keys()}}


# --- (8) the free-lambda fit ---


def test_free_lambda_transform_and_fit_recovers_lambda(tmp_path):
    """The bounded transform round-trips on its open range and maps a
    value at or beyond a bound to a finite coordinate; the free-lambda
    fit of the fast patient's own seed on the cube (v at the truth,
    started from the brackets' midpoints, bounds T_r 5-40 days and lambda
    0.5-4 mm) recovers the truth's front width 1.155 mm within 10 % and
    its growth time within 20 %, with the objective below the start's and
    no bound hit; the fixed-lambda fit on the same truth recovers T_r."""
    assert ide.LAMBDA_BOUNDS == (0.5, 8.0) and ide.TR_BOUNDS == (5.0, 3000.0) and ide.DEFAULT_LAMBDA_MODE == "fixed"
    for lo, hi in ((np.log(5.0), np.log(3000.0)), (np.log(0.5), np.log(8.0)), (0.05, 1.0)):
        for value in np.linspace(lo, hi, 7)[1:-1]:
            assert ide.bounded_from_unbounded(ide.unbounded_from_bounded(value, lo, hi), lo, hi) == pytest.approx(value, abs=1e-9)
        for outside in (lo - 1.0, lo, hi, hi + 1.0):
            z = ide.unbounded_from_bounded(outside, lo, hi)
            assert np.isfinite(z) and lo < ide.bounded_from_unbounded(z, lo, hi) < hi
        assert ide.bounded_from_unbounded(0.0, lo, hi) == pytest.approx((lo + hi) / 2)
    assert ide._near_bound(5.0 * np.exp(0.005), (5.0, 3000.0)) and not ide._near_bound(5.0 * np.exp(0.02), (5.0, 3000.0)) and ide._near_bound(7.95, (1.0, 8.0))
    cohort = _cube_cohort(tmp_path)
    truth = _fast_patient()
    density = np.asarray(ide.solve_growth(cohort, ide.growth_config(ide.patient_config(cohort, truth, 180.0))).final_state["cell_density"])
    region = ide.observation_region({"pre": density}, cohort.tissue, cohort.zooms)
    fit = ide.fit_growth_time_free(cohort, truth, truth.seed_peak, truth.seed_sigma, density, region, "B", maxfev=80, bounds=(5.0, 40.0), lambda_bounds=(0.5, 4.0))
    assert fit.lambda_mode == "free" and fit.start == {"T_r": pytest.approx(np.sqrt(200.0)), "lambda_mm": pytest.approx(np.sqrt(2.0))}
    assert fit.front_width == pytest.approx(truth.front_width, rel=0.10) and fit.t_r == pytest.approx(12.0, rel=0.20)
    assert fit.value < fit.value_initial and fit.value < 0.02 and not fit.bound_hit and fit.n_evaluations <= 80
    assert fit.rho == pytest.approx(truth.front_speed / (2 * fit.front_width)) and fit.rho_truth == pytest.approx(0.15)
    assert fit.history[0]["lambda_mm"] == pytest.approx(np.sqrt(2.0)) and len(fit.history) == fit.n_evaluations
    record = fit.record()
    assert record["fitted_lambda_mm"] == fit.front_width and record["lambda_bounds"] == [0.5, 4.0] and record["rho_T_r_fitted"] == pytest.approx(fit.rho * fit.t_r)
    fixed = ide.fit_growth_time(cohort, truth, truth.seed_peak, truth.seed_sigma, density, region, "B", maxfev=30, bounds=(5.0, 40.0))
    assert fixed.lambda_mode == "fixed" and fixed.front_width == truth.front_width and fixed.t_r == pytest.approx(12.0, rel=0.05) and fixed.lambda_bounds is None
    # A start at the fixed optimum is taken as given.
    started = ide.fit_growth_time_free(cohort, truth, truth.seed_peak, truth.seed_sigma, density, region, "B", maxfev=3, bounds=(5.0, 40.0), lambda_bounds=(0.5, 4.0), start=(fixed.t_r, truth.front_width))
    assert started.history[0]["T_r"] == pytest.approx(fixed.t_r) and started.history[0]["lambda_mm"] == pytest.approx(truth.front_width)
    with pytest.raises(ValueError, match="lambda bounds"):
        ide.fit_growth_time_free(cohort, truth, 0.6, 2.0, density, region, "B", maxfev=3, lambda_bounds=(8.0, 1.0))


# --- (9) the smaller pieces ---


def test_frame_days_and_log_kill_and_selection(cohort):
    """The frame days follow the script's rounding (pre = (n_growth - 1)
    dt, the Sundays the script's own days, the horizon frame before the
    horizon); the total log kill formula; the patient selection syntax
    (the smoke selection takes the first patient of each cell; the
    profile's default is the fd_check patient and the next of each cell,
    the fd_check patients alone with smoke); the lambda modes (fixed by
    default); the stability time step (an integer number of steps per
    day between 2 and 12, "fixed" giving 12, the solver's estimate
    honoured at T_r); the Cramer-Rao errors of a singular matrix."""
    dt = 1.0 / 12.0
    days = ide.frame_days(100.0, dt, {name: ide.FRAME_MOMENTS[name] for name in ide.SUBSTITUTE_FRAMES})
    assert days["pre"] == pytest.approx(1199 * dt) and days["pre"] < 100.0
    script_days = sa.snapshot_days(100.0, {"mid_crt": 34.0, "end_crt": 55.0}, dt)
    assert days["d34"] == script_days["mid_crt"] and days["d55"] == script_days["end_crt"]
    assert 100.0 + 81.0 - 1.5 * dt < days["d80"] <= 100.0 + 81.0 - 0.5 * dt
    assert 100.0 + 120.0 - 1.5 * dt < days["d120"] <= 100.0 + 120.0 - 0.5 * dt and days["d180"] < 280.0
    kill = ide.total_log_kill(0.1, 8.0, 30, 2.0, 2e-3, 4900.0, 9.24)
    assert float(kill["log_kill_rt"]) == pytest.approx(60 * 0.1 * 1.25) and float(kill["log_kill_total"]) == pytest.approx(7.5 + 2e-3 * 4900 / 9.24)
    assert [p.id for p in ide.select_patients(cohort, "p03,p00-p02,p03", False)] == ["p03", "p00", "p01", "p02"]
    assert [p.id for p in ide.select_patients(cohort, None, True)] == ["p00", "p02", "p04", "p06"]
    assert [p.id for p in ide.select_patients(cohort, None, True)] == [p.id for p in cohort.patients if p.fd_check]
    assert len(ide.select_patients(cohort, "all", True)) == 8 and len(ide.select_patients(cohort, None, False)) == 8
    with pytest.raises(ValueError, match="unknown id"):
        ide.select_patients(cohort, "p99", False)
    assert [p.id for p in ide.profile_default_patients(cohort, False)] == [f"p{i:02d}" for i in range(8)]
    assert [p.id for p in ide.profile_default_patients(cohort, True)] == ["p00", "p02", "p04", "p06"]
    assert [p.fd_check for p in ide.profile_default_patients(cohort, False)] == [True, False] * 4
    assert ide.PROFILE_SIGMAS == (1.0, 1.5, 2.0, 3.0, 4.0, 6.0) and ide.SMOKE.sigmas == (2.0, 4.0) and ide.SMOKE.patients_per_cell == 1
    assert ide.parse_lambda_modes(None) == ("fixed",) and ide.parse_lambda_modes("both") == ("fixed", "free")
    assert ide.parse_lambda_modes("free") == ("free",) and ide.parse_lambda_modes("fixed") == ("fixed",)
    with pytest.raises(ValueError, match="lambda-mode"):
        ide.parse_lambda_modes("neither")
    assert ide.parse_tr_bounds(None) == (5.0, 3000.0) and ide.parse_tr_bounds("10, 20") == (10.0, 20.0)
    # The time step: fixed is 12 steps/day; stability is dt = min(0.5, 0.25 dx^2 / (6 D), 0.01 / rho)
    # rounded to whole steps per day, raised to the solver's estimate at T_r, capped at 12.
    assert ide.DT_MODES == ("fixed", "stability") and ide.DEFAULT_DT_MODE == "stability" and ide.BASE_STEPS_PER_DAY == 12
    assert ide.DT_MAX == 0.5 and ide.DT_SAFETY == 0.25 and ide.RHO_DT_MAX == 0.01
    # The reaction bound: rho = 0.08 gives dt = 0.01 / 0.08 = 0.125 d, 8 steps/day, where the diffusion bound
    # (D = 0.05: 0.83 d) and DT_MAX would give 2; rho = 0.02 gives 0.5 d, the DT_MAX cap, and does not bind.
    assert ide.steps_per_day_for("stability", 0.05, 0.08, 1000.0, 1.0) == 8
    assert ide.steps_per_day_for("stability", 0.05, 0.02, 1000.0, 1.0) == 2
    assert ide.steps_per_day_for("stability", 0.05, 0.05, 1000.0, 1.0) == 5  # 0.2 d
    with pytest.raises(ValueError, match="rho > 0"):
        ide.steps_per_day_for("stability", 0.05, 0.0, 1000.0, 1.0)
    assert ide.steps_per_day_for("fixed", 0.05, 0.01, 300.0, 1.0) == 12 and ide.steps_per_day_for("fixed", 1.0, 0.1, 5.0, 4.0) == 12
    assert ide.steps_per_day_for("stability", 0.5, 0.01, 1000.0, 1.0) == 12  # dt = 1 / (24 D) = 1/12 d
    assert ide.steps_per_day_for("stability", 0.25, 0.01, 1000.0, 1.0) == 6  # 1 / (24 D) = 1/6 d
    assert ide.steps_per_day_for("stability", 0.2, 0.01, 1000.0, 1.0) == 5  # 1 / (24 D) = 0.208 d -> 5 steps/day
    assert ide.steps_per_day_for("stability", 0.05, 0.01, 1000.0, 1.0) == 2  # 0.83 d capped at DT_MAX 0.5 d
    assert ide.steps_per_day_for("stability", 1.0, 0.1, 1000.0, 1.0) == 12  # the floor: 1/24 d would be 24/day, never finer than 12
    assert ide.steps_per_day_for("stability", 0.2, 0.01, 1000.0, 4.0) == 2  # the smoke's 4 mm voxels: DT_MAX everywhere
    assert ide.solver_step_estimate(0.1, 0.01, 40.0, 1.0) == int(np.ceil(8 * 0.1 * 40 + 100))
    assert ide.steps_per_day_for("stability", 0.1, 0.01, 40.0, 1.0) == int(np.ceil(132 / 40)) == 4  # the solver's estimate at T_r binds (the formula gives 3)
    assert ide.steps_per_day_for("stability", 1.0, 0.1, 5.0, 1.0) == 12  # the estimate would need 28/day: capped
    for diffusivity in (0.015, 0.05, 0.2, 0.5, 1.0):
        for t_r in (5.0, 40.0, 300.0, 1000.0):
            for dx in (1.0, 4.0):
                steps = ide.steps_per_day_for("stability", diffusivity, 0.02, t_r, dx)
                assert isinstance(steps, int) and 2 <= steps <= 12 and steps >= 1
                # The formula's step, unless the floor of 12 steps/day is finer than it.
                assert 1.0 / steps <= max(min(ide.DT_MAX, ide.DT_SAFETY * dx**2 / (6 * diffusivity)), 1.0 / ide.BASE_STEPS_PER_DAY) + 1e-12
                assert steps * t_r >= min(12 * t_r, ide.solver_step_estimate(diffusivity, 0.02, t_r, dx)) - 1e-9
    with pytest.raises(ValueError, match="dt-mode"):
        ide.steps_per_day_for("adaptive", 0.1, 0.01, 100.0, 1.0)
    assert ide.grid_spacing_mm({"resolution_factor": 0.25}, (1.0, 1.0, 1.0)) == 4.0 and ide.grid_spacing_mm({"resolution_factor": 1.0}, (1.0, 2.0, 1.5)) == 1.0
    stepped = _fast_patient(steps_per_day=3)
    assert ide.requested_steps(12.0, 3) == 36 and ide.requested_steps(12.1, 3) == 37 and stepped.dt == pytest.approx(1 / 3)
    assert not ide.dt_refined(36, 12.0, stepped) and ide.dt_refined(37, 12.0, stepped) and not ide.dt_refined(1000, 12.0, _fast_patient())
    # Every patient of the design carries its step; the arms refuse another mode.
    assert all(p.steps_per_day == ide.steps_per_day_for("stability", p.diffusivity, p.rho, p.resection_time, 1.0) for p in cohort.patients)
    parse = ide.build_parser().parse_args
    assert ide.check_dt_mode(cohort, parse(["seedfix", "--name", "x"])) == "stability"
    with pytest.raises(ValueError, match="dt_mode stability"):
        ide.check_dt_mode(cohort, parse(["seedfix", "--name", "x", "--dt-mode", "fixed"]))
    assert ide.seed_record(0.6, 2.0) == {"source": "argument", "seed_peak": 0.6, "seed_sigma_mm": 2.0, "gaussian_seed_mass": pytest.approx(0.6 * (4 * np.pi * 2.0) ** 1.5), "gaussian_seed_diffusion_time": 2.0}
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
    device one block; the worker command lines forward the options
    given, per experiment."""
    assert ide.parse_devices("") == [""] and ide.parse_devices(",") == ["", ""] and ide.parse_devices("1, 2,6") == ["1", "2", "6"]
    patients = cohort.patients
    blocks = ide.patient_blocks(patients, 4)
    assert [len(b) for b in blocks] == [2, 2, 2, 2] and [p.id for b in blocks for p in b] == [p.id for p in patients]
    assert [p.id for p in blocks[1]] == ["p02", "p03"]
    assert [len(b) for b in ide.patient_blocks(patients[:5], 4)] == [2, 1, 1, 1]
    assert [len(b) for b in ide.patient_blocks(patients[:2], 4)] == [1, 1]
    assert [len(b) for b in ide.patient_blocks(patients, 1)] == [8] and ide.patient_blocks([], 3) == []
    assert [len(b) for b in ide.patient_blocks(patients[:7], 3)] == [3, 2, 2]
    parse = ide.build_parser().parse_args
    command = ide.worker_command(cohort.root, "fisher", "3", patients[:2], parse(["fisher", "--name", "x", "--smoke", "--draws", "5"]))
    assert command[2:] == ["fisher", "--output-dir", str(cohort.root.parent), "--name", cohort.root.name, "--gpus", "3", "--patients", "p00,p01", "--no-assemble", "--smoke", "--draws", "5"]
    command = ide.worker_command(cohort.root, "substitute", "", patients[:1], parse(["substitute", "--name", "x", "--maxfev", "3", "--delta-a", "0.5,2"]))
    assert command[-9:] == ["--patients", "p00", "--no-assemble", "--maxfev", "3", "--dt-mode", "stability", "--delta-a", "0.5,2"] and command[7:9] == ["--gpus", ""]
    args = parse(["seedfix", "--name", "x", "--maxfev", "4", "--dt-mode", "fixed", "--tr-bounds", "5,100", "--seed-peak", "0.7", "--seed-sigma-mm", "3", "--lambda-mode", "free"])
    command = ide.worker_command(cohort.root, "seedfix", "2", patients[:1], args)
    assert command[-12:] == ["--maxfev", "4", "--dt-mode", "fixed", "--tr-bounds", "5,100", "--seed-peak", "0.7", "--seed-sigma-mm", "3.0", "--lambda-mode", "free"]
    args = parse(["all", "--name", "x", "--maxfev", "4", "--seed-peak", "0.7", "--seed-sigma-mm", "3", "--sigmas", "2,5", "--delta-a", "0.5", "--lambda-mode", "fixed"])
    assert ide.worker_command(cohort.root, "profile", "2", patients[:1], args)[-8:] == ["--maxfev", "4", "--dt-mode", "stability", "--seed-peak", "0.7", "--sigmas", "2,5"]
    assert ide.worker_command(cohort.root, "substitute", "2", patients[:1], args)[-6:] == ["--maxfev", "4", "--dt-mode", "stability", "--delta-a", "0.5"]
    assert ide.worker_command(cohort.root, "seedfix", "2", patients[:1], args)[-10:] == ["--maxfev", "4", "--dt-mode", "stability", "--seed-peak", "0.7", "--seed-sigma-mm", "3.0", "--lambda-mode", "fixed"]
    # Unset options are not forwarded (the dt mode always is, it has a default); the profile has no seed width.
    command = ide.worker_command(cohort.root, "profile", "2", patients[:1], parse(["profile", "--name", "x"]))
    assert command[-3:] == ["--no-assemble", "--dt-mode", "stability"]
    assert ide.parse_bands(parse(["design", "--name", "x", "--r-core-band", "3,7"])) == ide.SizeBands((3.0, 7.0), (10.0, 35.0), (1.2, 8.0))
    with pytest.raises(ValueError, match="lo < hi"):
        ide.parse_bands(parse(["design", "--name", "x", "--ratio-band", "8,1"]))


def test_dispatch_two_cpu_workers_on_phantom(phantom_root):
    """fisher --smoke --gpus ',' over two patients: two CPU worker
    processes with their logs under logs/, the CSVs and figures
    assembled from both records by the dispatcher, the device split in
    fisher_summary.json; a plain in-process pass afterwards skips both
    patients and keeps the split; a dispatch whose workers fail (seedfix
    with inverted T_r bounds) raises after assembling and records the
    return codes."""
    base = ["--smoke", "--output-dir", str(phantom_root.parent), "--name", phantom_root.name]
    assert ide.main(["fisher", *base, "--gpus", ",", "--patients", "p01,p03", "--draws", "2"]) == 0
    logs = phantom_root / "logs"
    assert (logs / "fisher_cpu.log").is_file() and (logs / "fisher_cpu_1.log").is_file()
    assert "fisher p01" in (logs / "fisher_cpu.log").read_text() and "fisher p03" in (logs / "fisher_cpu_1.log").read_text()
    for patient in ("p01", "p03"):
        assert (phantom_root / "runs" / "fisher" / patient / "fisher.json").is_file()
    rows = _read_csv(phantom_root / "fisher.csv")
    assert {r["patient"] for r in rows} >= {"p01", "p03"}
    summary = ide.read_record(phantom_root / "fisher_summary.json")
    assert {"p01", "p03"} <= set(summary["patients"]) and summary["n_rows"] == len(rows)
    split = summary["dispatch"]
    assert split["devices"] == ["", ""] and split["n_failed"] == 0 and [b["patients"] for b in split["blocks"]] == [["p01"], ["p03"]]
    assert all(b["returncode"] == 0 and Path(b["log"]).is_file() for b in split["blocks"])
    assert (phantom_root / "figures" / "fisher_heatmap_cr_log_T_r.png").is_file()
    # A plain pass over the same patients skips them and keeps the split.
    assert ide.main(["fisher", *base, "--gpus", "", "--patients", "p01,p03"]) == 0
    assert ide.read_record(phantom_root / "fisher_summary.json")["dispatch"]["blocks"] == split["blocks"]
    # Failing workers: the dispatcher assembles, then raises naming the logs.
    with pytest.raises(RuntimeError, match="2 of 2 workers failed"):
        ide.main(["seedfix", *base, "--gpus", ",", "--patients", "p01,p03", "--maxfev", "2", "--tr-bounds", "10,5"])
    summary = ide.read_record(phantom_root / "seedfix_summary.json")
    assert summary["dispatch"]["n_failed"] == 2 and all(b["returncode"] != 0 for b in summary["dispatch"]["blocks"])
    assert (logs / "seedfix_cpu.log").is_file() and "Traceback" in (logs / "seedfix_cpu_1.log").read_text()


def test_figures_tolerate_empty_iso_surfaces(tmp_path):
    """A metric that is NaN for every row (an empty iso-surface at every
    lambda, at every substitute, at every patient) still gives a figure:
    the panels' axes are scaled by hand; the summaries' groups follow the
    maturity strata (at or above / below 15)."""
    nan = float("nan")
    invariance = [
        {"lambda": value, "run_type": run_type, "snapshot": "pre", **{name: nan for name in ide.METRIC_NAMES}, "mass": 1.0}
        for value in (0.5, 1.0, 2.0)
        for run_type in ("growth_fixed_n", "treated")
    ]
    ide.invariance_figures(invariance, tmp_path / "figures")
    assert (tmp_path / "figures" / "invariance_metrics.png").is_file() and (tmp_path / "figures" / "invariance_qois.pdf").is_file()
    metrics = {f"{frame}_{name}": nan for frame in ide.SUBSTITUTE_FRAMES for name in ide.METRIC_NAMES}
    substitute = [
        {"patient": f"p{i:02d}", "cell": cell, "delta_a": delta_a, "T_0": 50.0, "rho_T_0": 1.0, "objective": "B", "skipped": "", "maturity": 5.0 * (i + 1), **metrics}
        for i, cell in enumerate(ide.CELLS)
        for delta_a in (-0.5, 0.5)
    ]
    ide.substitute_figures(substitute, tmp_path / "figures")
    for frame in ("d120", "d180"):
        assert (tmp_path / "figures" / f"substitute_{frame}_objective_B.png").is_file()
    medians = ide.substitute_medians(substitute)["B"]
    assert set(medians) == {"-0.5", "0.5"} and set(medians["0.5"]) == {*ide.CELLS, "all", "maturity_ge_15", "maturity_lt_15"}
    assert medians["0.5"]["maturity_lt_15"]["n_rows"] == 2 and medians["0.5"]["maturity_ge_15"]["n_rows"] == 2
    seedfix = [
        {"patient": f"p{i:02d}", "cell": cell, "lambda_mode": mode, "objective": "B", "fitted_T_r": 30.0, "fitted_lambda_mm": 2.0, "lambda_truth_mm": 2.5, "rho_T_r_fitted": 1.0, "rho_T_r": 2.0, "R_over_lambda": 4.0 * i, "maturity": nan if i == 0 else 8.0 * i, "bound_hit": i == 1, **metrics}
        for i, cell in enumerate(ide.CELLS)
        for mode in ide.LAMBDA_MODES
    ]
    ide.seedfix_figures(seedfix, tmp_path / "figures")
    for frame in ("d120", "d180"):
        for mode in ide.LAMBDA_MODES:
            assert (tmp_path / "figures" / f"seedfix_{frame}_{mode}.png").is_file()
    medians = ide.seedfix_medians(seedfix)
    assert set(medians) == set(ide.LAMBDA_MODES) and medians["free"]["all"] == {**medians["free"]["all"], "n_patients": 4, "n_bound_hit": 1}
    assert medians["free"]["all"]["abs_rho_T_r_error"] == 1.0 and medians["free"]["all"]["abs_log_lambda_error"] == pytest.approx(abs(np.log(2.0 / 2.5)))
    assert medians["fixed"]["maturity_ge_15"]["n_patients"] == 2 and medians["fixed"]["maturity_lt_15"]["n_patients"] == 1
    profile = [
        {"patient": f"p{i:02d}", "cell": cell, "sigma_mm": sigma, "seed_peak": 0.6, "fitted_T_r": 30.0, "rho_T_r_fitted": 1.0, "rho_T_r": 2.0, "R_over_lambda": 12.0, "maturity": 20.0, "bound_hit": False, "objective_achieved": nan, **metrics}
        for i, cell in enumerate(ide.CELLS)
        for sigma in (2.0, 5.0)
    ]
    ide.profile_figures(profile, tmp_path / "figures")
    assert (tmp_path / "figures" / "profile_objective.png").is_file() and (tmp_path / "figures" / "profile_d180.pdf").is_file()
    assert set(ide.profile_medians(profile)) == {"2", "5"} and ide.profile_medians(profile)["2"]["all"]["fitted_T_r"] == 30.0
    assert ide.profile_medians(profile)["2"]["maturity_ge_15"]["n_patients"] == 4 and "maturity_lt_15" not in ide.profile_medians(profile)["2"]


# --- (10) reused records of another seed ---


def test_reused_seed_mismatch_raises(tmp_path):
    """``check_reused_seed`` raises, naming both seeds, when a reused
    record's seed_peak or seed_sigma_mm differs from the arguments;
    ``seedfix_patient`` on the cube refuses a fixed-mode row.json of
    another seed and ``profile_patient`` a sigma directory of another
    peak, both after the truth."""
    record = {"seed_peak": 0.6, "seed_sigma_mm": 2.0}
    ide.check_reused_seed(tmp_path / "row.json", record, 0.6, 2.0)
    with pytest.raises(ValueError, match=r"peak 0\.6, sigma 2\.0 mm.*peak 0\.7, sigma 2\.0 mm"):
        ide.check_reused_seed(tmp_path / "row.json", record, 0.7, 2.0)
    with pytest.raises(ValueError, match=r"sigma 2\.0 mm.*sigma 3\.0 mm"):
        ide.check_reused_seed(tmp_path / "row.json", record, 0.6, 3.0)
    cohort = _cube_cohort(tmp_path)
    patient = _fast_patient()
    seedfix_dir = tmp_path / "seedfix"
    stale = seedfix_dir / "px" / "B" / "row.json"
    stale.parent.mkdir(parents=True)
    ide.write_record(stale, {**record, "seed_peak": 0.8, "lambda_mode": "fixed"})
    with pytest.raises(ValueError, match=r"peak 0\.8.*peak 0\.6"):
        ide.seedfix_patient(cohort, patient, seedfix_dir, maxfev=2, tr_bounds=(5.0, 40.0), seed_peak=0.6, seed_sigma=2.0, lambda_modes=("fixed",))
    assert (seedfix_dir / "px" / "truth" / "frames.json").is_file() and not (seedfix_dir / "px" / "seedfix.json").is_file()
    profile_dir = tmp_path / "profile"
    stale = profile_dir / "px" / "sigma_2" / "row.json"
    stale.parent.mkdir(parents=True)
    ide.write_record(stale, {"seed_peak": 0.7, "seed_sigma_mm": 2.0})
    with pytest.raises(ValueError, match=r"peak 0\.7, sigma 2\.0 mm.*peak 0\.6, sigma 2\.0 mm"):
        ide.profile_patient(cohort, patient, profile_dir, maxfev=2, tr_bounds=(5.0, 40.0), seed_peak=0.6, sigmas=(2.0,))


@pytest.mark.slow
def test_smoke_pipeline_on_phantom(phantom_root):
    """The command line's ``all --smoke`` on the phantom design (the
    existing design is kept; one patient, one deficit, the fixed lambda
    mode, one seed width): substitute, seedfix and profile write their
    CSVs, summaries and figures, and a second pass skips the finished
    patient. Tens of seconds (``slow``)."""
    spec = json.loads((phantom_root / "spec.json").read_text())
    args = ["all", "--smoke", "--output-dir", str(phantom_root.parent), "--name", phantom_root.name, "--config", spec["base_config"]]
    args += ["--white-matter-pbmap", "", "--gray-matter-pbmap", "", "--patients", "p00", "--delta-a", "0.5", "--lambda-mode", "fixed", "--sigmas", "2"]
    assert ide.main(args) == 0
    for name in ("substitute.csv", "seedfix.csv", "profile.csv", "substitute_summary.json", "seedfix_summary.json", "profile_summary.json"):
        assert (phantom_root / name).is_file(), name
    # The CSVs hold every record of the shared design directory (other
    # tests add patients), so the checks look at p00's rows.
    substitute = [r for r in _read_csv(phantom_root / "substitute.csv") if r["patient"] == "p00"]
    assert len(substitute) == 1 + len(ide.OBJECTIVES) and list(substitute[0]) == ide.SUBSTITUTE_COLUMNS
    assert all(int(r["n_evaluations"]) <= ide.SMOKE.maxfev for r in substitute if r["objective"] != "truth")
    seedfix = [r for r in _read_csv(phantom_root / "seedfix.csv") if r["patient"] == "p00"]
    assert [r["lambda_mode"] for r in seedfix] == ["truth", "fixed"] and list(seedfix[0]) == ide.SEEDFIX_COLUMNS
    assert seedfix[1]["seed_peak"] == "0.6" and seedfix[1]["seed_sigma_mm"] == "2.0" and seedfix[1]["bound_hit"] in ("True", "False")
    p00 = ide.load_cohort(phantom_root).patient("p00")
    assert all(int(r["steps_per_day"]) == p00.steps_per_day and float(r["maturity"]) == pytest.approx(p00.maturity) for r in seedfix + substitute)
    profile = [r for r in _read_csv(phantom_root / "profile.csv") if r["patient"] == "p00"]
    assert len(profile) == 1 and profile[0]["sigma_mm"] == "2.0" and list(profile[0]) == ide.PROFILE_COLUMNS
    summary = ide.read_record(phantom_root / "seedfix_summary.json")
    assert summary["seed"]["source"] == "argument" and summary["seed"]["seed_sigma_mm"] == 2.0 and "fixed" in summary["medians"]
    for stem in ("substitute_d120_objective_B", "substitute_d180_objective_B", "seedfix_d120_fixed", "seedfix_d180_fixed", "profile_objective", "profile_d180"):
        assert (phantom_root / "figures" / f"{stem}.png").is_file() and (phantom_root / "figures" / f"{stem}.pdf").is_file()
    assert (phantom_root / "runs" / "substitute" / "p00" / "substitute.json").is_file()
    assert ide.read_record(phantom_root / "runs" / "seedfix" / "p00" / "seedfix.json")["truth_reused"] is True
    assert ide.read_record(phantom_root / "runs" / "profile" / "p00" / "profile.json")["truth_reused"] is True
    assert ide.main(args) == 0  # everything is skipped
