"""Tests of scripts/sensitivity_analysis.py (loaded with importlib from
scripts/), self-contained.

(1) The Ishigami function through the script's own design and analysis
functions against its analytic Sobol' indices (the block/row bookkeeping
end to end), (2) SALib's Saltelli row order as the script assumes it,
verified against the installed version, (3) the factor transforms, the
search-space checks, the derived groups (their parsing and evaluation
order, the growth derivation and its implied-range validation, the seed
derivation against the solver's own seed and its design-time
validation), (4) the design bookkeeping on the 24^3 phantom with the
shipped search space (every AB run differs from its block's A run only
through its factor, or through the derived parameters of its group,
seeds lie on seedable tissue voxels, every config constructs a
StuppFKPPSolver once the derived maps are given, design.csv carries the
derived and the extra columns), the nested-range seed mapping, the
time-step check, the schedule truncation and the snapshot definition,
(5) the growth stage config, the treatment maps, the snapshot days, one
two-stage run on the phantom with its pre-resection field and its two
treated-stage snapshots and one growth-only run, (6) the QoIs on
synthetic Gaussian fields against closed forms (the centroid included),
the snapshot QoIs of a synthetic run directory, the accounting, the
half-sample summary fields and the analysis of a sweep of the previous
schema, and (7) the whole pipeline on the phantom on the CPU (design,
run, qoi, analyze) with a small search space.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fisher_kpp_jax import SOLVER_KEY, FKPPSolver, StuppFKPPSolver, read_config

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sensitivity_analysis.py"
SHIPPED_SEARCH_SPACE = (
    Path(__file__).resolve().parent.parent
    / "fisher_kpp_jax"
    / "search_spaces"
    / "stupp_fkpp_search_space.json"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("sensitivity_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sa = _load_script()

CONFIG_KEYS = StuppFKPPSolver.config_keys()
SEED_KEYS = tuple(f"gaussian_seed_{axis}_fraction" for axis in "xyz")
# The phantom's flux threshold: seedable = wm + gm >= it.
MIN_TISSUE_FRACTION = 0.1


def _seed_entries(**overrides) -> dict:
    """Search-space entries: the overrides (in order), then the three seed
    factors on [0, 1] unless overridden."""
    entries = dict(overrides)
    for key in SEED_KEYS:
        entries.setdefault(key, {"min": 0.0, "max": 1.0, "scale": "linear"})
    return entries


GROWTH_GROUP = {
    "derives": ["white_matter_diffusivity", "rho"],
    "front_speed_mm_per_day": {"min": 0.03, "max": 0.25, "scale": "log"},
    "front_width_mm": {"min": 1.0, "max": 5.0, "scale": "log"},
}
SEED_GROUP = {
    "derives": ["gaussian_seed_mass", "gaussian_seed_diffusion_time"],
    "seed_peak_density": {"min": 0.6, "max": 1.0, "scale": "linear"},
    "seed_relative_width": {"min": 2.0, "max": 4.0, "scale": "log"},
}


def _group_entries(**overrides) -> dict:
    """Search-space entries with both derived groups (growth, then seed)
    and the seed factors, the overrides applied on top (in order)."""
    return _seed_entries(**{"growth": GROWTH_GROUP, "seed": SEED_GROUP, **overrides})


# The phantom's schedule: resection at day 12, the first fraction two
# days later on a "Monday" (day 14), 30 fractions on weekdays over six
# weeks (days 14-53), TMZ daily over the same weeks (days 14-55, the
# Sundays 34 and 55 included) and one adjuvant cycle (days 80-84) beyond
# the 60-day post-resection horizon (day 72), which the script drops.
PHANTOM_RESECTION = 12.0
PHANTOM_RT_START = 14.0
PHANTOM_RT_TIMES = [PHANTOM_RT_START + 7 * (i // 5) + i % 5 for i in range(30)]
PHANTOM_CHEMO_TIMES = [PHANTOM_RT_START + i for i in range(42)] + [80.0 + i for i in range(5)]
PHANTOM_CHEMO_DOSES = [75.0] * 42 + [150.0] * 5
PHANTOM_TIME_AFTER = 60.0
PHANTOM_STEPS_PER_DAY = 40


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def phantom_base(tmp_path: Path, tissue_phantom) -> dict:
    """A phantom base config on disk: WM/GM maps plus the config JSON (a
    treated run over the six-week phantom schedule with resection_cavity
    and rt_dose null, as the script derives them; 12 days of growth take
    the seed's density past the 0.6 cavity threshold; the chemotherapy
    kill rate keeps the total log kill of the 42 concomitant sessions
    below 1 so that treated tumors survive)."""
    gm, wm = tissue_phantom
    volumes = tmp_path / "volumes"
    volumes.mkdir()
    for name, data in (("wm", wm), ("gm", gm)):
        nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), str(volumes / f"{name}.nii.gz"))
    config = {
        SOLVER_KEY: "StuppFKPPSolver",
        "_note": "phantom base config of tests/test_sensitivity_analysis.py",
        "white_matter_pbmap": "volumes/wm.nii.gz",
        "gray_matter_pbmap": "volumes/gm.nii.gz",
        "white_matter_diffusivity": 0.2,
        "rho": 0.15,
        "gaussian_seed_x_fraction": 0.5,
        "gaussian_seed_y_fraction": 0.5,
        "gaussian_seed_z_fraction": 0.5,
        # Pinned so the phantom's seed (mass 250 in the search space: a peak
        # of 0.50, sigma 3.2 mm on the 24-voxel grid) does not follow the
        # solver defaults (tau 15, at which mass 250 is below the floor).
        "gaussian_seed_diffusion_time": 5.0,
        "resolution_factor": 1.0,
        "precision": "f32",
        "steps_per_day": PHANTOM_STEPS_PER_DAY,
        "min_tissue_fraction": MIN_TISSUE_FRACTION,
        "resection_time": PHANTOM_RESECTION,
        "time_after_resection": PHANTOM_TIME_AFTER,
        "resection_cavity": None,
        "chemo_times": PHANTOM_CHEMO_TIMES,
        "chemo_doses": PHANTOM_CHEMO_DOSES,
        "chemo_kill_rate": 0.01 / 75,
        "chemo_decay_rate": 0.5,
        "rt_times": PHANTOM_RT_TIMES,
        "rt_dose": None,
        "rt_alpha": 0.1,
        "rt_alpha_beta_ratio": 10.0,
    }
    path = tmp_path / "base.json"
    path.write_text(json.dumps(config, indent=1))
    seedable = (gm + wm) >= MIN_TISSUE_FRACTION
    return {"path": path, "gm": gm, "wm": wm, "seedable": seedable, "tmp_path": tmp_path}


# --- (1) Ishigami ---


def test_ishigami_round_trip():
    """Three linear factors on [-pi, pi], a = 7, b = 0.1, N = 2^12: the
    analytic first- and total-order indices are recovered within 0.03
    through the design, the response assembly (a NaN drops its whole
    block) and the SALib wrapper."""
    from SALib.test_functions import Ishigami

    names = ["x1", "x2", "x3"]
    samples = sa.saltelli_design(names, log2_n=12, seed=1)
    assert samples.shape == (4096 * 5, 3) and samples.min() >= 0 and samples.max() < 1
    x = sa.transform_factor(samples, -np.pi, np.pi, "linear")
    y = Ishigami.evaluate(x)
    values = {index: float(value) for index, value in enumerate(y)}
    result = sa.analyze_response(values, names, n_blocks=4096, n_bootstrap=50, seed=0)
    assert result["n_blocks_used"] == 4096 and result["dropped_blocks"] == []
    np.testing.assert_allclose(result["S1"], [0.3139, 0.4424, 0.0], atol=0.03)
    np.testing.assert_allclose(result["ST"], [0.5576, 0.4424, 0.2437], atol=0.03)
    np.testing.assert_allclose(result["S1_half"], [0.3139, 0.4424, 0.0], atol=0.05)
    assert np.all(result["S1_conf"] > 0) and np.all(result["ST_conf"] > 0)
    # A missing or NaN value drops its block, nothing else.
    values[5 * 7 + 2] = float("nan")
    del values[5 * 100]
    response, kept, dropped = sa.assemble_response(values, 4096, 5)
    assert dropped == [7, 100] and len(kept) == 4094 and response.shape == (4094 * 5,)
    np.testing.assert_array_equal(response[:35], y[:35])
    np.testing.assert_array_equal(response[35:40], y[40:45])
    again = sa.analyze_response(values, names, n_blocks=4096, n_bootstrap=20, seed=0)
    assert again["n_blocks_used"] == 4094 and again["dropped_blocks"] == [7, 100]
    np.testing.assert_allclose(again["ST"], [0.5576, 0.4424, 0.2437], atol=0.03)


# --- (2) SALib row order ---


@pytest.mark.parametrize("second_order", [False, True])
def test_salib_row_order(second_order):
    """In every block of the installed SALib's sample, position 0 is A_j,
    the last position is B_j and position i (1..k) differs from A_j in
    column i - 1 only, where it equals B_j (positions k + 1..2k with the
    second-order rows: B_j with column i - k - 1 from A_j), as the
    script's matrix labels and run names assume."""
    names = ["a", "b", "c", "d"]
    k, n_blocks = len(names), 8
    size = sa.block_size(k, second_order)
    assert size == (2 * k + 2 if second_order else k + 2)
    samples = sa.saltelli_design(names, log2_n=3, seed=5, second_order=second_order)
    assert samples.shape == (n_blocks * size, k)
    labels = sa.matrix_labels(names, second_order)
    assert labels[0] == "A" and labels[-1] == "B" and len(labels) == size
    assert labels[1 : k + 1] == [f"AB:{name}" for name in names]
    if second_order:
        assert labels[k + 1 : 2 * k + 1] == [f"BA:{name}" for name in names]
    for j in range(n_blocks):
        block = samples[j * size : (j + 1) * size]
        a, b = block[0], block[-1]
        assert not np.array_equal(a, b)
        for i in range(1, k + 1):
            differs = np.flatnonzero(block[i] != a)
            assert differs.tolist() == [i - 1] and block[i][i - 1] == b[i - 1]
        if second_order:
            for i in range(k + 1, 2 * k + 1):
                differs = np.flatnonzero(block[i] != b)
                assert differs.tolist() == [i - k - 1] and block[i][i - k - 1] == a[i - k - 1]
    assert sa.run_name(3, "AB:rho") == "r0003_AB-rho" and sa.run_name(12, "B") == "r0012_B"


# --- (3) transforms and search-space checks ---


def test_transforms():
    np.testing.assert_allclose(sa.transform_factor([0.0, 0.5, 1.0], 0.01, 100.0, "log"), [0.01, 1.0, 100.0])
    np.testing.assert_allclose(sa.transform_factor([0.0, 0.5, 1.0], 30.0, 200.0, "linear"), [30.0, 115.0, 200.0])
    parameter = sa.SearchSpaceParameter("rho", 0.0089228, 0.3449, "log")
    np.testing.assert_allclose(parameter.transform(0.5), np.sqrt(0.0089228 * 0.3449))
    assert sa.transform_factor(np.zeros((2, 3)), 1.0, 2.0, "linear").shape == (2, 3)
    with pytest.raises(ValueError, match="scale"):
        sa.transform_factor(0.5, 1.0, 2.0, "sqrt")


def test_load_search_space():
    """The shipped file loads with 12 factors in file order, the growth
    group's and the seed group's two sampled factors each at the group's
    position, the seed scale, the horizon and the chemotherapy decay rate
    fixed; a mapping loads too; unknown keys, a derived volume, a bad
    scale, a mismatching solver, seed entries off [0, 1] or missing each
    raise."""
    space = sa.load_search_space(SHIPPED_SEARCH_SPACE, CONFIG_KEYS)
    assert len(space.names) == 12
    assert space.overrides == {"gaussian_seed_scale": 1.0, "time_after_resection": 120.0, "chemo_decay_rate": 9.24}
    assert space.names[:3] == ["front_speed_mm_per_day", "front_width_mm", "diffusivity_ratio"]
    assert space.names[-5:] == ["seed_peak_density", "seed_relative_width", *SEED_KEYS]
    assert list(space.groups) == ["growth", "seed"]
    assert space.derived_keys == ["white_matter_diffusivity", "rho", "gaussian_seed_mass", "gaussian_seed_diffusion_time"]
    assert space.extra_keys == ["seed_sigma_mm", "seed_enhancing_radius_mm"]
    assert space.plain_names == ["diffusivity_ratio", "resection_time", "chemo_kill_rate", "rt_alpha", "rt_alpha_beta_ratio", *SEED_KEYS]
    assert space.group_factor_names == {"front_speed_mm_per_day", "front_width_mm", "seed_peak_density", "seed_relative_width"}
    shipped = json.loads(SHIPPED_SEARCH_SPACE.read_text())
    assert shipped["growth"] == GROWTH_GROUP and shipped["seed"] == SEED_GROUP and shipped["gaussian_seed_scale"] == 1.0
    assert "rho" not in shipped and "white_matter_diffusivity" not in shipped and "seed_sigma_mm" not in shipped
    assert space.factors["front_speed_mm_per_day"] == sa.SearchSpaceParameter("front_speed_mm_per_day", 0.03, 0.25, "log")
    assert space.factors["front_width_mm"] == sa.SearchSpaceParameter("front_width_mm", 1.0, 5.0, "log")
    assert space.factors["rt_alpha_beta_ratio"].scale == "linear"
    assert space.factors["seed_relative_width"] == sa.SearchSpaceParameter("seed_relative_width", 2.0, 4.0, "log")
    assert space.factors["seed_peak_density"] == sa.SearchSpaceParameter("seed_peak_density", 0.6, 1.0, "linear")
    assert "_note" in space.source and "_note" not in space.factors
    for key in ("_growth_band", "_seed_band", "_chemo_decay_rate", "_horizon", "_sources", "_scales"):
        assert key in shipped
    entries = _seed_entries(solver="StuppFKPPSolver", rho={"min": 0.1, "max": 0.2, "scale": "log"}, verbose=True)
    space = sa.load_search_space(entries, CONFIG_KEYS)
    assert space.names == ["rho", *SEED_KEYS] and space.overrides == {"verbose": True}
    with pytest.raises(ValueError, match="unknown key 'D'"):
        sa.load_search_space(_seed_entries(D={"min": 0.1, "max": 0.2, "scale": "log"}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="unknown key 'rt_beta'"):
        sa.load_search_space(_seed_entries(rt_beta=0.006), CONFIG_KEYS)
    with pytest.raises(ValueError, match="rt_dose is derived per run"):
        sa.load_search_space(_seed_entries(rt_dose="dose.nii.gz"), CONFIG_KEYS)
    with pytest.raises(ValueError, match="resection_cavity is derived per run"):
        sa.load_search_space(_seed_entries(resection_cavity={"segmentation": "s.nii.gz", "label": 1}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="scale"):
        sa.load_search_space(_seed_entries(rho={"min": 0.1, "max": 0.2, "scale": "sqrt"}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="min < max"):
        sa.load_search_space(_seed_entries(rho={"min": 0.2, "max": 0.1, "scale": "log"}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="min > 0"):
        sa.load_search_space(_seed_entries(rho={"min": 0.0, "max": 0.1, "scale": "log"}), CONFIG_KEYS)
    with pytest.raises(ValueError, match='"min"'):
        sa.load_search_space(_seed_entries(rho={"min": 0.1, "scale": "log"}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="names solver 'FKPPSolver'"):
        sa.load_search_space(_seed_entries(solver="FKPPSolver"), CONFIG_KEYS)
    bad_seed = {"min": 0.0, "max": 1.5, "scale": "linear"}
    with pytest.raises(ValueError, match="gaussian_seed_y_fraction must be a linear factor within"):
        sa.load_search_space(_seed_entries(gaussian_seed_y_fraction=bad_seed), CONFIG_KEYS)
    with pytest.raises(ValueError, match="gaussian_seed_z_fraction must be a linear factor within"):
        sa.load_search_space(_seed_entries(gaussian_seed_z_fraction={"min": 0.1, "max": 1.0, "scale": "log"}), CONFIG_KEYS)
    missing = _seed_entries()
    del missing["gaussian_seed_x_fraction"]
    with pytest.raises(ValueError, match="gaussian_seed_x_fraction must be a linear factor"):
        sa.load_search_space(missing, CONFIG_KEYS)
    with pytest.raises(ValueError, match="gaussian_seed_x_fraction must be a linear factor"):
        sa.load_search_space(_seed_entries(gaussian_seed_x_fraction=0.5), CONFIG_KEYS)


def test_derived_group_parsing():
    """A derived group: its sampled factors take the group's position in
    the factor order, its derived parameters are neither factors nor
    overrides, ``derive`` / ``solver_values`` map the records, the seed
    group takes the growth group's front width; the groups are evaluated
    growth before seed whatever the file order, and a seed group without
    a growth group raises; a group whose derived parameters are unknown,
    unregistered, derived volumes or given elsewhere, whose key or factor
    names are parameters, or whose sub-entries are not the registered
    factors, raises."""
    entries = _group_entries(diffusivity_ratio={"min": 1.0, "max": 10.0, "scale": "log"}, verbose=True)
    space = sa.load_search_space(entries, CONFIG_KEYS)
    assert space.names == [
        "front_speed_mm_per_day", "front_width_mm", "seed_peak_density", "seed_relative_width", "diffusivity_ratio", *SEED_KEYS,
    ]
    assert space.plain_names == ["diffusivity_ratio", *SEED_KEYS] and space.overrides == {"verbose": True}
    growth, seed = space.groups["growth"], space.groups["seed"]
    assert growth.derivation is sa.DERIVATIONS[tuple(GROWTH_GROUP["derives"])]
    assert seed.derivation is sa.DERIVATIONS[tuple(SEED_GROUP["derives"])]
    assert list(growth.factors) == ["front_speed_mm_per_day", "front_width_mm"] and growth.derivation.requires == ()
    assert list(seed.factors) == ["seed_peak_density", "seed_relative_width"]
    assert seed.derivation.requires == ("front_width_mm",) and seed.derivation.extras == ("seed_sigma_mm", "seed_enhancing_radius_mm")
    derived = space.derive({
        "front_speed_mm_per_day": np.array([0.1, 0.2]), "front_width_mm": np.array([2.0, 4.0]),
        "seed_peak_density": np.array([0.6, 1.0]), "seed_relative_width": np.array([2.0, 3.0]),
    })
    assert set(derived) == set(space.derived_keys) | set(space.extra_keys)
    np.testing.assert_allclose(derived["white_matter_diffusivity"], [0.1, 0.4])
    np.testing.assert_allclose(derived["rho"], [0.025, 0.025])
    np.testing.assert_allclose(derived["seed_sigma_mm"], [4.0, 12.0])  # s * lambda
    np.testing.assert_allclose(derived["gaussian_seed_diffusion_time"], [8.0, 72.0])
    np.testing.assert_allclose(derived["gaussian_seed_mass"], [0.6 * (32 * np.pi) ** 1.5, (288 * np.pi) ** 1.5])
    np.testing.assert_allclose(derived["seed_enhancing_radius_mm"], [0.0, 12.0 * np.sqrt(2 * np.log(1 / 0.6))])
    record = {"diffusivity_ratio": "3", "front_speed_mm_per_day": "0.1", "front_width_mm": "2.0", "seed_peak_density": "0.6",
              "seed_relative_width": "2.0", "white_matter_diffusivity": "0.1", "rho": "0.025", "gaussian_seed_mass": "1.5",
              "gaussian_seed_diffusion_time": "8.0", "seed_sigma_mm": "4.0", "seed_enhancing_radius_mm": "0.0",
              "gaussian_seed_x_fraction": 0.5, "gaussian_seed_y_fraction": 0.5, "gaussian_seed_z_fraction": 0.5}
    assert space.solver_values(record) == {
        "diffusivity_ratio": 3.0, "gaussian_seed_x_fraction": 0.5, "gaussian_seed_y_fraction": 0.5,
        "gaussian_seed_z_fraction": 0.5, "white_matter_diffusivity": 0.1, "rho": 0.025,
        "gaussian_seed_mass": 1.5, "gaussian_seed_diffusion_time": 8.0,
    }
    # Evaluation order: the seed group listed first is still evaluated
    # after the growth group it requires; the factor order is the file's.
    reversed_entries = _seed_entries(seed=SEED_GROUP, growth=GROWTH_GROUP)
    reordered = sa.load_search_space(reversed_entries, CONFIG_KEYS)
    assert list(reordered.groups) == ["growth", "seed"]
    assert reordered.names == ["seed_peak_density", "seed_relative_width", "front_speed_mm_per_day", "front_width_mm", *SEED_KEYS]
    assert reordered.derived_keys == space.derived_keys
    assert sa.group_evaluation_order({"seed": reordered.groups["seed"], "growth": reordered.groups["growth"]}) == ["growth", "seed"]
    with pytest.raises(ValueError, match="seed: the derivation of .* takes 'front_width_mm', which no group"):
        sa.load_search_space(_seed_entries(seed=SEED_GROUP), CONFIG_KEYS)
    with pytest.raises(ValueError, match="takes 'front_width_mm', which no group"):
        sa.load_search_space(_seed_entries(seed=SEED_GROUP, rho=0.1, white_matter_diffusivity=0.5), CONFIG_KEYS)
    # A growth group alone is fine (a growth-only sweep with a fixed seed).
    growth_only = sa.load_search_space(_seed_entries(growth=GROWTH_GROUP, gaussian_seed_mass=250.0), CONFIG_KEYS)
    assert list(growth_only.groups) == ["growth"] and growth_only.derived_keys == ["white_matter_diffusivity", "rho"]
    assert growth_only.extra_keys == []
    # A plain search space has no group.
    plain = sa.load_search_space(_seed_entries(), CONFIG_KEYS)
    assert plain.groups == {} and plain.derived_keys == [] and plain.extra_keys == [] and plain.derive({}) == {}
    with pytest.raises(ValueError, match="derives unknown parameter"):
        sa.load_search_space(_group_entries(seed={**SEED_GROUP, "derives": ["gaussian_seed_mass", "tau"]}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="no derivation registered"):
        sa.load_search_space(_group_entries(seed={**SEED_GROUP, "derives": ["gaussian_seed_mass"]}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="cannot be in a group"):
        sa.load_search_space(_group_entries(seed={**SEED_GROUP, "derives": ["rt_dose"]}), CONFIG_KEYS)
    with pytest.raises(ValueError, match='"derives" must be a nonempty list'):
        sa.load_search_space(_group_entries(seed={**SEED_GROUP, "derives": "gaussian_seed_mass"}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="must be exactly"):
        sa.load_search_space(_group_entries(seed={"derives": SEED_GROUP["derives"], "seed_peak_density": SEED_GROUP["seed_peak_density"]}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="must be exactly"):
        swapped = {"derives": SEED_GROUP["derives"], "seed_relative_width": SEED_GROUP["seed_relative_width"], "seed_peak_density": SEED_GROUP["seed_peak_density"]}
        sa.load_search_space(_group_entries(seed=swapped), CONFIG_KEYS)
    with pytest.raises(ValueError, match="must be exactly"):
        old = {"derives": SEED_GROUP["derives"], "seed_peak_density": SEED_GROUP["seed_peak_density"], "seed_sigma_mm": {"min": 1.0, "max": 5.0, "scale": "log"}}
        sa.load_search_space(_group_entries(seed=old), CONFIG_KEYS)
    with pytest.raises(ValueError, match="seed: seed_relative_width: min < max"):
        sa.load_search_space(_group_entries(seed={**SEED_GROUP, "seed_relative_width": {"min": 4.0, "max": 2.0, "scale": "log"}}), CONFIG_KEYS)
    with pytest.raises(ValueError, match="must not be a parameter name"):
        sa.load_search_space(_group_entries(rho=SEED_GROUP), CONFIG_KEYS)
    with pytest.raises(ValueError, match="given twice"):
        sa.load_search_space(_group_entries(gaussian_seed_mass=250.0), CONFIG_KEYS)
    with pytest.raises(ValueError, match="given twice"):
        sa.load_search_space(_seed_entries(rho={"min": 0.01, "max": 0.1, "scale": "log"}, growth=GROWTH_GROUP, seed=SEED_GROUP), CONFIG_KEYS)
    with pytest.raises(ValueError, match="given twice"):
        sa.load_search_space(_group_entries(white_matter_diffusivity=0.5), CONFIG_KEYS)


def test_growth_derivation():
    """D = v lambda / 2 and rho = v / (2 lambda) invert v = 2 sqrt(D rho),
    lambda = sqrt(D / rho); the design-time checks report the implied
    ranges at the rectangle's corners and refuse ranges leaving the
    admissible rho and D bounds or nonpositive ranges."""
    derived = sa.growth_parameters(0.1, 3.0)
    np.testing.assert_allclose(derived["white_matter_diffusivity"], 0.15)
    np.testing.assert_allclose(derived["rho"], 0.1 / 6)
    front = sa.front_parameters(derived["white_matter_diffusivity"], derived["rho"])
    np.testing.assert_allclose([front["front_speed_mm_per_day"], front["front_width_mm"]], [0.1, 3.0])
    front = sa.front_parameters(0.80021, 0.075777)  # the previous base config: v above the band
    assert front["front_speed_mm_per_day"] > 0.25 and 1.0 < front["front_width_mm"] < 5.0
    grid = sa.growth_parameters(np.array([[0.03], [0.25]]), np.array([[1.0, 5.0]]))
    assert grid["rho"].shape == (2, 2)
    validate = sa.DERIVATIONS[sa.GROWTH_DERIVED_KEYS].validate
    factors = {"front_speed_mm_per_day": sa.SearchSpaceParameter("front_speed_mm_per_day", 0.03, 0.25, "log"),
               "front_width_mm": sa.SearchSpaceParameter("front_width_mm", 1.0, 5.0, "log")}
    record = validate(factors, {}, {})
    np.testing.assert_allclose(record["white_matter_diffusivity_range"], [0.015, 0.625])
    np.testing.assert_allclose(record["rho_range"], [0.003, 0.125])
    assert record["rho_bounds"] == [0.001, 0.2] and record["white_matter_diffusivity_bounds"] == [0.01, 2.0]
    # The implied ranges are the extremes over the rectangle.
    np.testing.assert_allclose(record["rho_range"], [grid["rho"].min(), grid["rho"].max()])
    np.testing.assert_allclose(
        record["white_matter_diffusivity_range"], [grid["white_matter_diffusivity"].min(), grid["white_matter_diffusivity"].max()]
    )
    with pytest.raises(ValueError, match=r"implies rho in \[0.0006, 0.075\], outside"):
        validate({**factors, "front_width_mm": sa.SearchSpaceParameter("front_width_mm", 1.0, 25.0, "log"),
                  "front_speed_mm_per_day": sa.SearchSpaceParameter("front_speed_mm_per_day", 0.03, 0.15, "log")}, {}, {})
    with pytest.raises(ValueError, match="implies white_matter_diffusivity in"):
        validate({**factors, "front_speed_mm_per_day": sa.SearchSpaceParameter("front_speed_mm_per_day", 0.03, 1.0, "log")}, {}, {})
    with pytest.raises(ValueError, match="front_speed_mm_per_day: min must be positive"):
        validate({**factors, "front_speed_mm_per_day": sa.SearchSpaceParameter("front_speed_mm_per_day", -0.1, 0.2, "linear")}, {}, {})


def test_seed_derivation(tissue_phantom):
    """tau = sigma^2 / 2 and m = c_peak (4 pi tau)^(3/2) invert
    ``seed_peak_density``, and the solver's own seed built from them has
    the peak c_peak at the seed voxel and c_peak e^(-1/2) one sigma away
    in mm, on isotropic and anisotropic voxels and with downsampling (the
    solver converts voxel offsets to mm itself). The design-time checks:
    a peak range at or below the floor, above 1, a nonpositive width or a
    seed scale other than 1 raise; the group's derivation takes the
    width as seed_relative_width times the front width and reports
    sigma and the enhancing radius."""
    derived = sa.seed_parameters(0.5, np.sqrt(10.0))
    np.testing.assert_allclose(derived["gaussian_seed_diffusion_time"], 5.0)
    np.testing.assert_allclose(derived["gaussian_seed_mass"], 0.5 * (20 * np.pi) ** 1.5)
    np.testing.assert_allclose(sa.seed_peak_density(derived["gaussian_seed_mass"], 5.0), 0.5)
    np.testing.assert_allclose(sa.seed_peak_density(250.0, 5.0), 250 / (20 * np.pi) ** 1.5)
    group = sa.seed_group_parameters({"seed_peak_density": 0.8, "seed_relative_width": 2.5, "front_width_mm": 2.0})
    assert set(group) == {"gaussian_seed_mass", "gaussian_seed_diffusion_time", "seed_sigma_mm", "seed_enhancing_radius_mm"}
    np.testing.assert_allclose(group["seed_sigma_mm"], 5.0)
    np.testing.assert_allclose(group["gaussian_seed_diffusion_time"], 12.5)
    np.testing.assert_allclose(group["gaussian_seed_mass"], sa.seed_parameters(0.8, 5.0)["gaussian_seed_mass"])
    np.testing.assert_allclose(group["seed_enhancing_radius_mm"], 5.0 * np.sqrt(2 * np.log(0.8 / 0.6)))
    # The enhancing radius: the seed's density at that distance is the
    # threshold; 0 at or below the threshold peak, sigma sqrt(2 ln(1/0.6))
    # at peak 1.
    radius = float(sa.seed_enhancing_radius(0.8, 5.0))
    assert 0.8 * np.exp(-(radius**2) / (2 * 5.0**2)) == pytest.approx(0.6)
    np.testing.assert_allclose(sa.seed_enhancing_radius([0.6, 0.3, 1.0], 4.0), [0.0, 0.0, 4.0 * np.sqrt(2 * np.log(1 / 0.6))])
    n = 33
    wm, gm = np.ones((n, n, n)), np.zeros((n, n, n))
    peak, sigma = 0.6, 3.0
    d = sa.seed_parameters(peak, sigma)
    for zooms, factor in (((1.0, 1.0, 1.0), 1.0), ((1.0, 1.5, 2.0), 1.0), ((1.0, 1.0, 1.0), 0.5)):
        solver = FKPPSolver({
            "white_matter_pbmap": wm, "gray_matter_pbmap": gm, "white_matter_diffusivity": 0.1, "rho": 0.1,
            "voxel_size_mm": list(zooms), "resolution_factor": factor, "precision": "f64", "stopping_time": 1.0,
            "steps_per_day": 10, "gaussian_seed_x_fraction": 0.5, "gaussian_seed_y_fraction": 0.5,
            "gaussian_seed_z_fraction": 0.5, "gaussian_seed_scale": 1.0, "gaussian_seed_floor": 0.1,
            "gaussian_seed_mass": float(d["gaussian_seed_mass"]),
            "gaussian_seed_diffusion_time": float(d["gaussian_seed_diffusion_time"]),
        })
        solver.resolve_time_stepping()
        seed = np.asarray(solver._gaussian_seed(), dtype=np.float64)
        i, j, k = solver.seed_voxel
        dx = solver.grid_spacing[0]
        steps = int(round(sigma / dx))
        np.testing.assert_allclose(seed[i, j, k], peak, rtol=1e-6)
        np.testing.assert_allclose(seed[i + steps, j, k], peak * np.exp(-((steps * dx) ** 2) / (2 * sigma**2)), rtol=1e-6)
        np.testing.assert_allclose(seed[i, j + 1, k], peak * np.exp(-(solver.grid_spacing[1] ** 2) / (2 * sigma**2)), rtol=1e-6)
    params = {"gaussian_seed_floor": 0.1, "gaussian_seed_scale": 1.0}
    factors = {"seed_peak_density": sa.SearchSpaceParameter("seed_peak_density", 0.2, 1.0, "linear"),
               "seed_relative_width": sa.SearchSpaceParameter("seed_relative_width", 2.0, 4.0, "log")}
    required = {"front_width_mm": sa.SearchSpaceParameter("front_width_mm", 0.5, 2.5, "log")}
    validate = sa.DERIVATIONS[sa.SEED_DERIVED_KEYS].validate
    record = validate(factors, params, required)
    assert record["gaussian_seed_floor"] == 0.1 and record["gaussian_seed_scale"] == 1.0
    np.testing.assert_allclose(record["seed_sigma_mm_range"], [1.0, 10.0])
    np.testing.assert_allclose(record["gaussian_seed_diffusion_time_range"], [0.5, 50.0])
    np.testing.assert_allclose(record["gaussian_seed_mass_range"], [0.2 * (2 * np.pi) ** 1.5, (200 * np.pi) ** 1.5])
    np.testing.assert_allclose(record["seed_enhancing_radius_mm_range"], [0.0, 10.0 * np.sqrt(2 * np.log(1 / 0.6))])
    assert record["seed_enhancing_threshold"] == 0.6
    with pytest.raises(ValueError, match="min must exceed the config's gaussian_seed_floor 0.1"):
        validate({**factors, "seed_peak_density": sa.SearchSpaceParameter("seed_peak_density", 0.1, 1.0, "linear")}, params, required)
    with pytest.raises(ValueError, match="max must be at most 1"):
        validate({**factors, "seed_peak_density": sa.SearchSpaceParameter("seed_peak_density", 0.2, 1.5, "linear")}, params, required)
    with pytest.raises(ValueError, match="seed_relative_width: min must be positive"):
        validate({**factors, "seed_relative_width": sa.SearchSpaceParameter("seed_relative_width", -1.0, 5.0, "linear")}, params, required)
    with pytest.raises(ValueError, match="needs gaussian_seed_scale = 1"):
        validate(factors, {**params, "gaussian_seed_scale": 2.0}, required)


# --- (4) design bookkeeping ---


def _config_entries(path: Path) -> dict:
    return {key: value for key, value in json.loads(path.read_text()).items() if not key.startswith("_")}


def test_design_bookkeeping(phantom_base):
    """The shipped search space on the phantom base config: N (k + 2) runs
    with one A, one B and one AB per factor in every block; every AB
    config differs from its block's A config only through its factor (the
    seed fractions for a seed factor, the shifted event times for
    resection_time, the derived dynamics for a growth-group factor, the
    derived seed parameters for a seed-group factor, both for the front
    width the seed's width follows); design.csv carries the unit-cube
    coordinates, the sampled factors, the derived parameters, which the
    configs hold and which invert to the sampled peak and front, and the
    extra columns (sigma, the enhancing radius); every seed is a seedable
    tissue voxel whose fractions the config holds; the derived volumes
    stay null in the run configs and every config constructs a solver
    once they are given; spec.json records the schedule within the
    horizon and the snapshot offsets; a base config that sets the
    derived volumes, lacks the tissue maps, sets a seed scale other than
    1, has a seed floor at the peak range, or whose schedule does not
    define the snapshots is refused; the design is never overwritten."""
    tmp_path = phantom_base["tmp_path"]
    sweep_dir = sa.make_design(SHIPPED_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "design", log2_n=2, seed=3)
    assert sweep_dir == tmp_path / "sa" / "design"
    with pytest.raises(FileExistsError, match="never overwritten"):
        sa.make_design(SHIPPED_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "design", log2_n=2, seed=3)
    spec = json.loads((sweep_dir / "spec.json").read_text())
    names = spec["factor_names"]
    k, n_blocks = spec["k"], spec["N"]
    assert (k, n_blocks, spec["n_runs"], spec["block_size"]) == (12, 4, 56, 14)
    assert spec["salib_version"] and spec["seed"] == 3 and not spec["second_order"]
    assert spec["growth_solver"] == "FKPPSolver" and spec["growth_only"] is False
    assert spec["overrides"] == {"gaussian_seed_scale": 1.0, "time_after_resection": 120.0, "chemo_decay_rate": 9.24}
    assert spec["time_step"] == {"n_steps": None, "dt": None, "steps_per_day": PHANTOM_STEPS_PER_DAY}
    assert spec["derived_keys"] == ["white_matter_diffusivity", "rho", "gaussian_seed_mass", "gaussian_seed_diffusion_time"]
    assert spec["extra_keys"] == ["seed_sigma_mm", "seed_enhancing_radius_mm"]
    assert list(spec["derived_groups"]) == ["growth", "seed"]
    growth_group, seed_group = spec["derived_groups"]["growth"], spec["derived_groups"]["seed"]
    assert growth_group["derives"] == spec["derived_keys"][:2] and growth_group["factors"] == ["front_speed_mm_per_day", "front_width_mm"]
    assert growth_group["requires"] == [] and growth_group["extras"] == [] and growth_group["formula"] == sa.GROWTH_FORMULA
    np.testing.assert_allclose(growth_group["white_matter_diffusivity_range"], [0.015, 0.625])
    np.testing.assert_allclose(growth_group["rho_range"], [0.003, 0.125])
    assert seed_group["derives"] == spec["derived_keys"][2:] and seed_group["factors"] == ["seed_peak_density", "seed_relative_width"]
    assert seed_group["requires"] == ["front_width_mm"] and seed_group["extras"] == spec["extra_keys"]
    assert seed_group["gaussian_seed_floor"] == 0.1 and seed_group["gaussian_seed_scale"] == 1.0
    assert seed_group["formula"] == sa.SEED_FORMULA
    np.testing.assert_allclose(seed_group["seed_sigma_mm_range"], [2.0, 20.0])
    group_factors = set(growth_group["factors"]) | set(seed_group["factors"])
    assert spec["treatment"] == {
        "cavity_threshold": 0.6, "rt_margin_mm": 15.0, "rt_dose_per_fraction_gy": 2.0,
        "n_fractions": 30, "rt_total_dose_gy": 60.0,
    }
    # The schedule within the overridden horizon (120 days after the
    # phantom's resection at day 12 keeps every session) and the snapshot
    # Sundays 20 and 41 days after the first fraction (day 14), as
    # offsets after resection.
    assert spec["schedule"] == {
        "resection_time": PHANTOM_RESECTION, "time_after_resection": 120.0, "horizon": 132.0,
        "n_chemo_sessions": 47, "n_chemo_sessions_dropped": 0, "chemo_total_dose": 3900.0,
        "chemo_total_dose_dropped": 0.0, "n_fractions": 30, "n_fractions_dropped": 0,
    }
    assert spec["snapshots"] == {"mid_crt": 22.0, "end_crt": 43.0}
    assert (sweep_dir / "search_space.json").read_text() == SHIPPED_SEARCH_SPACE.read_text()
    base = read_config(sweep_dir / "base_config.json")
    assert Path(base["white_matter_pbmap"]).is_absolute()
    assert base["resection_cavity"] is None and base["rt_dose"] is None
    design = _read_csv(sweep_dir / "design.csv")
    assert len(design) == n_blocks * (k + 2)
    assert list(design[0]) == [
        "run_name", "index", "row", "matrix", *(f"u_{name}" for name in names), *names,
        *spec["derived_keys"], *spec["extra_keys"], "seed_voxel_i", "seed_voxel_j", "seed_voxel_k",
    ]
    n_seed_changes = 0
    assert [int(r["index"]) for r in design] == list(range(len(design)))
    assert sorted({r["run_name"] for r in design}) == sorted(p.stem for p in (sweep_dir / "configs").iterdir())
    for j in range(n_blocks):
        block = [r for r in design if int(r["row"]) == j]
        assert [r["matrix"] for r in block] == ["A", *(f"AB:{name}" for name in names), "B"]
        a = _config_entries(sweep_dir / "configs" / f"{block[0]['run_name']}.json")
        assert a["resection_cavity"] is None and a["rt_dose"] is None and a.get("snapshot_times") is None
        assert a["time_after_resection"] == 120.0 and a["chemo_decay_rate"] == 9.24
        assert block[0]["run_name"] == f"r{j:04d}_A" and block[-1]["run_name"] == f"r{j:04d}_B"
        for record in block[1:-1]:
            factor = record["matrix"].split(":")[1]
            ab = _config_entries(sweep_dir / "configs" / f"{record['run_name']}.json")
            differing = {key for key in a if a[key] != ab[key]}
            assert set(ab) == set(a)
            u_differing = {name for name in names if record[f"u_{name}"] != block[0][f"u_{name}"]}
            assert u_differing == {factor}
            if factor in SEED_KEYS:
                # The mapping may land on the A run's voxel, so only the
                # seed fractions may differ.
                assert differing <= set(SEED_KEYS)
                n_seed_changes += bool(differing)
            elif factor == "resection_time":
                assert differing == {"resection_time", "chemo_times", "rt_times"}
                shift = ab["resection_time"] - base["resection_time"]
                np.testing.assert_allclose(ab["chemo_times"], np.asarray(base["chemo_times"]) + shift)
                np.testing.assert_allclose(ab["rt_times"], np.asarray(base["rt_times"]) + shift)
            elif factor == "front_speed_mm_per_day":
                assert differing == {"white_matter_diffusivity", "rho"}
            elif factor == "front_width_mm":  # the seed's width follows the front width
                assert differing == {"white_matter_diffusivity", "rho", "gaussian_seed_mass", "gaussian_seed_diffusion_time"}
            elif factor == "seed_peak_density":
                assert differing == {"gaussian_seed_mass"}
            elif factor == "seed_relative_width":
                assert differing == {"gaussian_seed_mass", "gaussian_seed_diffusion_time"}
            else:
                assert differing == {factor}
            if factor not in group_factors:
                assert ab[factor] == float(record[factor])
    assert n_seed_changes > 0
    # design.csv: u, the sampled value, the derived parameters and the
    # extras of every row; the configs hold the derived values, which
    # invert to the sampled peak, width and front.
    for record in design:
        config = _config_entries(sweep_dir / "configs" / f"{record['run_name']}.json")
        for name in names:
            assert 0 <= float(record[f"u_{name}"]) < 1
        for key in spec["derived_keys"]:
            assert config[key] == float(record[key])
        assert config["gaussian_seed_scale"] == 1.0 and "seed_peak_density" not in config and "front_width_mm" not in config
        assert "seed_sigma_mm" not in config and "seed_enhancing_radius_mm" not in config
        speed, width = float(record["front_speed_mm_per_day"]), float(record["front_width_mm"])
        assert 0.03 <= speed <= 0.25 and 1.0 <= width <= 5.0
        np.testing.assert_allclose(config["white_matter_diffusivity"], speed * width / 2)
        np.testing.assert_allclose(config["rho"], speed / (2 * width))
        front = sa.front_parameters(config["white_matter_diffusivity"], config["rho"])
        np.testing.assert_allclose([front["front_speed_mm_per_day"], front["front_width_mm"]], [speed, width])
        peak, relative = float(record["seed_peak_density"]), float(record["seed_relative_width"])
        assert 0.6 <= peak <= 1.0 and 2.0 <= relative <= 4.0
        sigma = float(record["seed_sigma_mm"])
        np.testing.assert_allclose(sigma, relative * width)
        np.testing.assert_allclose(config["gaussian_seed_diffusion_time"], sigma**2 / 2)
        np.testing.assert_allclose(sa.seed_peak_density(config["gaussian_seed_mass"], config["gaussian_seed_diffusion_time"]), peak)
        np.testing.assert_allclose(float(record["seed_enhancing_radius_mm"]), sigma * np.sqrt(2 * np.log(peak / 0.6)))
    assert sa.seed_parameters(0.5, 2.0) == sa.seed_parameters(0.5, 2.0)  # deterministic
    # Seeds: seedable tissue voxels (wm + gm >= the solver's
    # min_tissue_fraction), the config holds their fractions.
    threshold = spec["seed_min_tissue_fraction"]
    assert threshold == FKPPSolver(sa.growth_config(base)).params["min_tissue_fraction"] == MIN_TISSUE_FRACTION
    seedable = phantom_base["seedable"]
    n = np.asarray(seedable.shape, dtype=np.float64)
    assert spec["n_seedable_voxels"] == int(seedable.sum()) == 3112
    voxels = np.argwhere(seedable)
    np.testing.assert_allclose(spec["seed_bbox_lo"], (voxels.min(axis=0) + 0.5) / n)
    np.testing.assert_allclose(spec["seed_bbox_hi"], (voxels.max(axis=0) + 0.5) / n)
    for record in design:
        voxel = np.array([int(record[f"seed_voxel_{ijk}"]) for ijk in "ijk"])
        assert seedable[tuple(voxel)]
        config = _config_entries(sweep_dir / "configs" / f"{record['run_name']}.json")
        np.testing.assert_allclose([config[key] for key in SEED_KEYS], (voxel + 0.5) / n)
        for name in names:
            if name not in SEED_KEYS:
                assert spec["factors"][name]["min"] <= float(record[name]) <= spec["factors"][name]["max"]
    geometry = sa.seed_geometry(phantom_base["wm"], phantom_base["gm"], threshold)
    assert geometry.n_voxels == 3112 and geometry.mask.sum() == 3112
    np.testing.assert_allclose(geometry.bbox_lo, spec["seed_bbox_lo"])
    np.testing.assert_allclose(geometry.bbox_hi, spec["seed_bbox_hi"])
    # The threshold decides what is seedable: a faint-tissue voxel is out
    # at 0.1 and in at 0; nothing seedable raises.
    faint_wm, faint_gm = phantom_base["wm"].copy(), phantom_base["gm"].copy()
    faint_wm[tuple(geometry.voxels[0])] = 0.05
    faint_gm[tuple(geometry.voxels[0])] = 0.0
    assert sa.seed_geometry(faint_wm, faint_gm, 0.1).n_voxels == 3111
    assert sa.seed_geometry(faint_wm, faint_gm, 0.0).n_voxels == faint_wm.size
    with pytest.raises(ValueError, match="no seedable voxel"):
        sa.seed_geometry(faint_wm, faint_gm, 2.0)
    with pytest.raises(ValueError, match="3D and alike"):
        sa.seed_geometry(faint_wm[0], faint_gm[0], 0.1)
    # The chemotherapy budget of the shipped ranges on the phantom
    # schedule (the decay rate is fixed).
    space = sa.load_search_space(SHIPPED_SEARCH_SPACE, CONFIG_KEYS)
    kill, decay = space.factors["chemo_kill_rate"], space.overrides["chemo_decay_rate"]
    assert spec["chemo_total_dose"] == 3900.0
    np.testing.assert_allclose(spec["chemo_log_kill_range"], [kill.low * 3900 / decay, kill.high * 3900 / decay])
    # Every config is a complete StuppFKPPSolver run once the derived maps
    # are given (here: empty).
    shape = seedable.shape
    for path in sorted((sweep_dir / "configs").iterdir())[:5]:
        config = read_config(path)
        assert config["resection_cavity"] is None and config["rt_dose"] is None
        solver = StuppFKPPSolver({**config, "resection_cavity": np.zeros(shape, bool), "rt_dose": np.zeros(shape)})
        assert solver.config[SOLVER_KEY] == "StuppFKPPSolver"
        FKPPSolver(sa.growth_config(config))
    # A base config with a cavity or a dose map, or without tissue maps, is refused.
    entries = json.loads(phantom_base["path"].read_text())
    with_dose = tmp_path / "with_dose.json"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), str(tmp_path / "dose.nii.gz"))
    with_dose.write_text(json.dumps({**entries, "rt_dose": "dose.nii.gz"}))
    with pytest.raises(ValueError, match=r"sets \['rt_dose'\], which every run derives"):
        sa.make_design(SHIPPED_SEARCH_SPACE, with_dose, tmp_path / "sa", "refused", log2_n=1)
    no_maps = tmp_path / "no_maps.json"
    no_maps.write_text(json.dumps({key: value for key, value in entries.items() if key != "gray_matter_pbmap"}))
    with pytest.raises(ValueError, match=r"lacks \['gray_matter_pbmap'\]"):
        sa.make_design(SHIPPED_SEARCH_SPACE, no_maps, tmp_path / "sa", "refused", log2_n=1)
    with pytest.raises(ValueError, match="cavity_threshold"):
        sa.make_design(SHIPPED_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "refused", cavity_threshold=0.0)
    # The seed derivation needs gaussian_seed_scale = 1: a base config
    # with another scale is refused (the shipped space overrides it, so
    # a space without the override is used).
    scaled = tmp_path / "scaled.json"
    scaled.write_text(json.dumps({**entries, "gaussian_seed_scale": 2.0}))
    unscaled_space = tmp_path / "unscaled_space.json"
    unscaled_space.write_text(json.dumps(_group_entries(solver="StuppFKPPSolver")))
    with pytest.raises(ValueError, match="needs gaussian_seed_scale = 1"):
        sa.make_design(unscaled_space, scaled, tmp_path / "sa", "refused", log2_n=1)
    # A peak range reaching the base config's floor is refused at design time.
    floored = tmp_path / "floored_space.json"
    floored.write_text(json.dumps(_group_entries(solver="StuppFKPPSolver", gaussian_seed_floor=0.6)))
    with pytest.raises(ValueError, match="exceed the config's gaussian_seed_floor 0.6"):
        sa.make_design(floored, phantom_base["path"], tmp_path / "sa", "refused", log2_n=1)
    # A seed group without the growth group, and a growth group whose
    # implied rho leaves the admissible range, are refused.
    no_growth = tmp_path / "no_growth_space.json"
    no_growth.write_text(json.dumps(_seed_entries(solver="StuppFKPPSolver", seed=SEED_GROUP)))
    with pytest.raises(ValueError, match="takes 'front_width_mm', which no group"):
        sa.make_design(no_growth, phantom_base["path"], tmp_path / "sa", "refused", log2_n=1)
    wide = tmp_path / "wide_space.json"
    wide.write_text(json.dumps(_group_entries(
        solver="StuppFKPPSolver", growth={**GROWTH_GROUP, "front_width_mm": {"min": 1.0, "max": 20.0, "scale": "log"}})))
    with pytest.raises(ValueError, match="implies white_matter_diffusivity in .* outside the admissible"):
        sa.make_design(wide, phantom_base["path"], tmp_path / "sa", "refused", log2_n=1)
    # The snapshots need the base schedule's six-week pattern, a
    # chemotherapy session on each snapshot Sunday and a horizon past
    # the last snapshot day; growth-only designs skip them.
    broken_rt = tmp_path / "broken_rt.json"
    broken_rt.write_text(json.dumps({**entries, "rt_times": [14.0, 16.0]}))
    with pytest.raises(ValueError, match="rt_times must be 30 fractions on 5 consecutive days a week for 6 weeks"):
        sa.make_design(SHIPPED_SEARCH_SPACE, broken_rt, tmp_path / "sa", "refused", log2_n=1)
    no_sunday = tmp_path / "no_sunday.json"
    weekday_chemo = [t for t in PHANTOM_CHEMO_TIMES if t != 34.0]
    no_sunday.write_text(json.dumps({**entries, "chemo_times": weekday_chemo, "chemo_doses": [75.0] * len(weekday_chemo)}))
    with pytest.raises(ValueError, match="snapshot mid_crt: day 34 .* has no chemotherapy session"):
        sa.make_design(SHIPPED_SEARCH_SPACE, no_sunday, tmp_path / "sa", "refused", log2_n=1)
    short_space = tmp_path / "short_space.json"
    short_space.write_text(json.dumps(_group_entries(solver="StuppFKPPSolver", time_after_resection=40.0)))
    with pytest.raises(ValueError, match=r"snapshots \{'end_crt': 43.0\} .* lie beyond time_after_resection = 40"):
        sa.make_design(short_space, phantom_base["path"], tmp_path / "sa", "refused", log2_n=1)
    growth_only_dir = sa.make_design(short_space, phantom_base["path"], tmp_path / "sa", "growth_only", log2_n=1, growth_only=True)
    assert json.loads((growth_only_dir / "spec.json").read_text())["snapshots"] is None
    assert not (tmp_path / "sa" / "refused").exists()


def test_truncate_schedule_and_snapshot_offsets():
    """The schedule within the horizon: sessions and fractions later than
    resection_time + time_after_resection are dropped with their doses
    (an override of time_after_resection, of the times or of the doses
    counts), the record holds the counts and the kept total dose; the
    shipped base config keeps 52 of 72 sessions (4 900 of 8 900 mg/m^2)
    at the shipped override. The snapshot offsets of the shipped base
    schedule are 34 and 55 days after resection (days 149 and 170); the
    checks refuse a schedule off the weekly pattern, a Sunday without a
    session or with a fraction, and no fractions at all."""
    base = read_config(sa.DEFAULT_CONFIG, solver=StuppFKPPSolver)
    full = sa.truncate_schedule(base, {})
    assert full["schedule"]["n_chemo_sessions"] == 72 and full["schedule"]["n_chemo_sessions_dropped"] == 0
    assert full["schedule"]["chemo_total_dose"] == 8900.0 and full["schedule"]["horizon"] == 375.0
    assert full["chemo_times"] == base["chemo_times"] and full["rt_times"] == base["rt_times"]
    shipped = sa.truncate_schedule(base, {"time_after_resection": 120.0})
    assert shipped["schedule"] == {
        "resection_time": 115.0, "time_after_resection": 120.0, "horizon": 235.0,
        "n_chemo_sessions": 52, "n_chemo_sessions_dropped": 20, "chemo_total_dose": 4900.0,
        "chemo_total_dose_dropped": 4000.0, "n_fractions": 30, "n_fractions_dropped": 0,
    }
    assert shipped["chemo_times"][-1] == 228.0 and shipped["chemo_doses"][-1] == 200.0 and len(shipped["chemo_doses"]) == 52
    assert shipped["chemo_times"] == base["chemo_times"][:52] and shipped["rt_times"] == base["rt_times"]
    assert sa.truncate_schedule(base, {"time_after_resection": 120.0, "chemo_doses": [1.0] * 72})["schedule"]["chemo_total_dose"] == 52.0
    tight = sa.truncate_schedule(base, {"time_after_resection": 40.0})
    assert tight["schedule"]["n_fractions"] == 20 and tight["schedule"]["n_fractions_dropped"] == 10
    assert tight["schedule"]["n_chemo_sessions"] == 27 and tight["rt_times"][-1] == 154.0 and tight["chemo_times"][-1] == 155.0
    with pytest.raises(ValueError, match="chemo_times has 72 entries, chemo_doses 2"):
        sa.truncate_schedule(base, {"chemo_doses": [1.0, 2.0]})
    # The search space's budget follows the truncated schedule.
    space = sa.load_search_space(SHIPPED_SEARCH_SPACE, CONFIG_KEYS)
    total, (low, high) = sa.chemo_log_kill_range(base, space)
    assert total == 4900.0
    np.testing.assert_allclose([low, high], [1e-4 * 4900 / 9.24, 3e-3 * 4900 / 9.24])
    # Snapshot offsets.
    assert sa.crt_snapshot_offsets(base["rt_times"], base["chemo_times"], base["resection_time"]) == {"mid_crt": 34.0, "end_crt": 55.0}
    assert sa.crt_snapshot_offsets(shipped["rt_times"], shipped["chemo_times"], 115.0) == {"mid_crt": 34.0, "end_crt": 55.0}
    assert base["rt_times"][0] + 20 == 149.0 and base["rt_times"][0] + 41 == 170.0 == base["chemo_times"][41]
    assert sa.crt_snapshot_offsets(PHANTOM_RT_TIMES, PHANTOM_CHEMO_TIMES, PHANTOM_RESECTION) == {"mid_crt": 22.0, "end_crt": 43.0}
    shuffled = list(reversed(base["rt_times"]))
    assert sa.crt_snapshot_offsets(shuffled, base["chemo_times"], 115.0) == {"mid_crt": 34.0, "end_crt": 55.0}
    with pytest.raises(ValueError, match="must be 30 fractions"):
        sa.crt_snapshot_offsets(base["rt_times"][:-1], base["chemo_times"], 115.0)
    with pytest.raises(ValueError, match="must be 30 fractions"):
        sa.crt_snapshot_offsets([t + (1 if t == 168.0 else 0) for t in base["rt_times"]], base["chemo_times"], 115.0)
    with pytest.raises(ValueError, match="snapshot end_crt: day 170 .* has no chemotherapy session"):
        sa.crt_snapshot_offsets(base["rt_times"], [t for t in base["chemo_times"] if t != 170.0], 115.0)
    with pytest.raises(ValueError, match="no rt_times"):
        sa.crt_snapshot_offsets([], base["chemo_times"], 115.0)
    # A fraction on a snapshot Sunday cannot happen on the weekly pattern,
    # so the check is exercised on the pattern check's own terms.
    assert not any(t in (149.0, 170.0) for t in base["rt_times"])


def test_snapshot_days():
    """The requested day of a snapshot is a step end m dt with
    t_end - 3/2 dt < m dt <= t_end - 1/2 dt, t_end = resection_time +
    offset + 1 the end of the snapshot day: after that day's session (a
    day earlier) and before the next day's fraction, whatever the
    alignment of the day to the steps; a step above half a day is
    refused."""
    for dt in (1 / 12, 0.083306, 0.05, 1 / 3, 0.5):
        for resection in (100.0, 71.208, 115.0):
            days = sa.snapshot_days(resection, {"mid_crt": 34.0, "end_crt": 55.0}, dt)
            assert list(days) == ["mid_crt", "end_crt"]
            for name, offset in (("mid_crt", 34.0), ("end_crt", 55.0)):
                t_end = resection + offset + 1
                m = days[name] / dt
                assert m == pytest.approx(round(m)) and t_end - 1.5 * dt < days[name] <= t_end - 0.5 * dt + 1e-12
                assert days[name] >= resection + offset  # after the day's session at resection + offset
    assert sa.snapshot_days(100.0, {"mid_crt": 34.0}, 1 / 12) == {"mid_crt": pytest.approx(1619 / 12)}
    assert sa.snapshot_days(100.0, {}, 0.1) == {}
    with pytest.raises(ValueError, match="at most half a day"):
        sa.snapshot_days(100.0, {"mid_crt": 34.0}, 1.0)
    # The run-one argument round trip.
    assert sa.format_snapshot_offsets({"mid_crt": 34.0, "end_crt": 55.0}) == "mid_crt=34.0,end_crt=55.0"
    assert sa.parse_snapshot_offsets("mid_crt=34.0,end_crt=55.0") == {"mid_crt": 34.0, "end_crt": 55.0}
    assert sa.parse_snapshot_offsets("") == {} and sa.format_snapshot_offsets({}) == ""
    assert sa.spec_snapshot_offsets({"snapshots": {"mid_crt": 22}}) == {"mid_crt": 22.0}
    assert sa.spec_snapshot_offsets({"snapshots": None}) == {} and sa.spec_snapshot_offsets({}) == {}
    with pytest.raises(ValueError, match="not one of"):
        sa.parse_snapshot_offsets("late_crt=3")
    with pytest.raises(ValueError, match="repeats"):
        sa.parse_snapshot_offsets("mid_crt=3,mid_crt=4")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        sa.parse_snapshot_offsets("mid_crt=-3")
    args = sa.build_parser().parse_args(["run-one", "--config", "c", "--run-dir", "d", "--snapshots", "mid_crt=22,end_crt=43"])
    assert sa.parse_snapshot_offsets(args.snapshots) == {"mid_crt": 22.0, "end_crt": 43.0}


def test_time_step_check(phantom_base):
    """A base config with n_steps, dt and steps_per_day all null (every
    run would fall back to the solver's horizon-dependent stability
    estimate) is refused at design time; an override of the search space
    counts; spec.json records the setting."""
    tmp_path = phantom_base["tmp_path"]
    entries = json.loads(phantom_base["path"].read_text())
    no_step = tmp_path / "no_step.json"
    no_step.write_text(json.dumps({**entries, "steps_per_day": None}))
    with pytest.raises(ValueError, match=r"sets none of \['n_steps', 'dt', 'steps_per_day'\]"):
        sa.make_design(SHIPPED_SEARCH_SPACE, no_step, tmp_path / "sa", "refused", log2_n=1)
    assert not (tmp_path / "sa" / "refused").exists()
    unlisted = tmp_path / "unlisted.json"
    unlisted.write_text(json.dumps({key: value for key, value in entries.items() if key != "steps_per_day"}))
    with pytest.raises(ValueError, match="sets none of"):
        sa.make_design(SHIPPED_SEARCH_SPACE, unlisted, tmp_path / "sa", "refused", log2_n=1)
    with_dt = tmp_path / "with_dt_space.json"
    with_dt.write_text(json.dumps(_group_entries(solver="StuppFKPPSolver", dt=0.02)))
    sweep_dir = sa.make_design(with_dt, no_step, tmp_path / "sa", "dt", log2_n=1)
    spec = json.loads((sweep_dir / "spec.json").read_text())
    assert spec["time_step"] == {"n_steps": None, "dt": 0.02, "steps_per_day": None}
    assert read_config(sweep_dir / "configs" / "r0000_A.json")["dt"] == 0.02


def test_tissue_map_override_and_cli_defaults(phantom_base):
    """The tissue maps given to make_design (the command line's default:
    the BraTS MNI152 atlas of PredictGBM) replace the base config's in
    base_config.json and the run configs; an empty entry keeps the base
    config's; an unknown key raises. The parser defaults --output-dir to
    DEFAULT_OUTPUT_DIR and the map flags to DEFAULT_TISSUE_MAPS."""
    tmp_path = phantom_base["tmp_path"]
    other = tmp_path / "other_wm.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(phantom_base["wm"]), np.eye(4)), str(other))
    sweep_dir = sa.make_design(
        sa.DEFAULT_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "maps", log2_n=1,
        tissue_maps={"white_matter_pbmap": other, "gray_matter_pbmap": ""},
    )
    base = read_config(sweep_dir / "base_config.json")
    assert base["white_matter_pbmap"] == str(other.resolve())
    assert base["gray_matter_pbmap"] == str((tmp_path / "volumes" / "gm.nii.gz").resolve())
    config = read_config(sweep_dir / "configs" / "r0000_A.json")
    assert config["white_matter_pbmap"] == str(other.resolve())
    with pytest.raises(ValueError, match="not one of"):
        sa.make_design(sa.DEFAULT_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "bad", log2_n=1,
                       tissue_maps={"csf_pbmap": other})
    args = sa.build_parser().parse_args(["design", "--name", "x"])
    assert args.output_dir == str(sa.DEFAULT_OUTPUT_DIR) == "/mnt/Drive4/lucas/stupp_sensitivity_analysis_atlas"
    assert args.white_matter_pbmap == str(sa.DEFAULT_TISSUE_MAPS["white_matter_pbmap"])
    assert args.gray_matter_pbmap.endswith("brats_mni152/brats_mni152_gm_pbmap.nii.gz")
    everything = sa.build_parser().parse_args(["all", "--name", "x", "--gray-matter-pbmap", ""])
    assert everything.gray_matter_pbmap == ""
    assert (everything.log2_n, everything.jobs_per_gpu, everything.gpus) == (11, 3, "1,2,3,6")


def test_project_seeds_nested_ranges():
    """The nested-range mapping: on a full box the three coordinates fall
    into equal integer bins per axis; a picked index that is not seedable
    snaps to the nearest seedable one of its slab / row / column; every
    result is seedable; coordinates off [0, 1] raise."""
    full = sa.SeedGeometry((4, 5, 6), np.ones((4, 5, 6), bool), np.argwhere(np.ones((4, 5, 6), bool)),
                           np.full(3, 0.5), np.full(3, 0.5))
    u = [[0.0, 0.0, 0.0], [0.24, 0.19, 0.16], [0.25, 0.2, 0.17], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
    fractions, voxels = sa.project_seeds(u, full)
    np.testing.assert_array_equal(voxels, [[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 3], [3, 4, 5]])
    np.testing.assert_allclose(fractions, (voxels + 0.5) / np.array([4.0, 5.0, 6.0]))
    assert sa.project_seeds([0.5, 0.5, 0.5], full)[1].shape == (1, 3)
    # Holes: slab 1 has seedable rows 0 and 3 only, and row (1, 3) the
    # voxels 0, 1 and 4 only.
    mask = np.zeros((3, 4, 5), bool)
    mask[0, 1, 2] = True
    mask[1, 0, :] = True
    mask[1, 3, [0, 1, 4]] = True
    mask[2, 2, 2] = True
    holes = sa.SeedGeometry(mask.shape, mask, np.argwhere(mask), np.zeros(3), np.ones(3))
    _, voxels = sa.project_seeds([[0.5, 0.5, 0.5], [0.5, 0.9, 0.7], [0.5, 0.9, 0.3], [0.0, 0.5, 0.5], [0.99, 0.0, 0.0]], holes)
    # u_y = 0.5 picks row 2 of slab 1, snapped to row 3 (nearer than 0);
    # u_z = 0.5 picks voxel 2 of that row, snapped to 1 (nearer than 4).
    np.testing.assert_array_equal(voxels, [[1, 3, 1], [1, 3, 4], [1, 3, 1], [0, 1, 2], [2, 2, 2]])
    assert all(mask[tuple(v)] for v in voxels)
    rng = np.random.default_rng(0)
    _, voxels = sa.project_seeds(rng.random((500, 3)), holes)
    assert all(mask[tuple(v)] for v in voxels)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sa.project_seeds([[1.2, 0.5, 0.5]], full)
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        sa.project_seeds([[0.5, 0.5]], full)


# --- (5) growth stage, treatment maps and one two-stage run ---


def test_growth_config(phantom_base):
    """The growth stage keeps the FKPPSolver entries of the run config,
    drops the treatment entries and the snapshots and stops at
    resection_time."""
    base = read_config(phantom_base["path"], solver=StuppFKPPSolver)
    growth = sa.growth_config({**base, "_design": "comment", "resection_time": 7.5, "snapshot_times": [20.0, 30.0]})
    assert growth[SOLVER_KEY] == "FKPPSolver" and growth["stopping_time"] == 7.5
    assert growth["snapshot_times"] is None  # the treated stage's snapshots are after resection
    assert set(growth) - {SOLVER_KEY} <= FKPPSolver.config_keys()
    assert not set(growth) & StuppFKPPSolver.TREATMENT_KEYS and "time_after_resection" not in growth
    for key in ("white_matter_pbmap", "rho", "white_matter_diffusivity", "steps_per_day", "min_tissue_fraction", *SEED_KEYS):
        assert growth[key] == base[key]
    solver = FKPPSolver(growth)
    assert solver.params["stopping_time"] == 7.5


def test_round_field():
    """Storage rounding of a density field: float32 of the input's shape,
    values below the floor (negative ones included) zero, the kept
    mantissa bits the only nonzero ones, exact for values representable
    in them, relative error within half a unit of the last kept bit,
    idempotent, and the default rounding of a random field compresses
    far better than the field itself."""
    rng = np.random.default_rng(0)
    volume = rng.random((6, 7, 8)) * np.logspace(-14, 0, 336).reshape(6, 7, 8)
    volume[0, 0, :3] = [-1e-3, 0.0, 5e-11]
    rounded = sa.round_field(volume)
    assert rounded.dtype == np.float32 and rounded.shape == volume.shape
    assert np.all(rounded[volume < 1e-10] == 0) and np.all(rounded[volume >= 1e-10] > 0)
    bits = rounded.view(np.uint32)
    assert not np.any(bits & np.uint32((1 << 16) - 1))  # 23 - 7 mantissa bits dropped
    kept = volume >= 1e-10
    np.testing.assert_allclose(rounded[kept], volume[kept], rtol=2.0**-8, atol=0)
    assert np.abs(rounded[kept] / volume[kept] - 1).max() > 2.0**-10  # it does round
    np.testing.assert_array_equal(sa.round_field(rounded), rounded)
    exact = np.array([0.5, 0.75, 1.0, 1.0 + 2.0**-7, 2.0**-30], dtype=np.float32)
    np.testing.assert_array_equal(sa.round_field(exact), exact)
    # One kept bit represents 1, 1.5, 2, 3: ties go to the even mantissa.
    np.testing.assert_array_equal(
        sa.round_field(np.array([1.5, 2.5, 1.25, 1.75, 2.75]), mantissa_bits=1), [1.5, 2.0, 1.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(sa.round_field(np.array([0.3, 0.6]), mantissa_bits=23), np.float32([0.3, 0.6]))
    assert sa.round_field(np.array([[1e-3]]), floor=1e-2)[0, 0] == 0
    with pytest.raises(ValueError, match="mantissa_bits must be 1 to 23"):
        sa.round_field(volume, mantissa_bits=0)
    field = np.exp(-rng.random((40, 40, 40)) * 60)  # a plume down to e^-60
    assert len(gzip.compress(sa.round_field(field).tobytes())) < len(gzip.compress(field.astype(np.float32).tobytes())) / 3


def test_treatment_maps():
    """A unit ball of radius 4 at the threshold gives the ball as cavity
    and, with a 6 mm margin, a dose region that holds every voxel within
    9 mm of the centre and none beyond 11 mm, at the total dose; the
    margin counts millimetres on anisotropic voxels; a density below the
    threshold everywhere gives empty maps."""
    shape = (41, 41, 41)
    centre = np.array([20.0, 20.0, 20.0])
    radius = np.linalg.norm(np.indices(shape).reshape(3, -1).T - centre, axis=1).reshape(shape)
    density = np.where(radius <= 4, 0.8, 0.59)
    cavity, dose = sa.treatment_maps(density, (1.0, 1.0, 1.0), 0.6, 6.0, 60.0)
    np.testing.assert_array_equal(cavity, radius <= 4)
    assert dose.dtype == np.float32 and set(np.unique(dose)) == {0.0, 60.0}
    assert np.all(dose[radius <= 9] == 60.0) and np.all(dose[radius > 11] == 0.0)
    assert np.all(dose[cavity] == 60.0)
    # Anisotropic voxels: 2 mm along z, so the margin reaches 3 voxels
    # from the cavity along z and 6 across.
    cavity_z, dose_z = sa.treatment_maps(density, (1.0, 1.0, 2.0), 0.6, 6.0, 4.0)
    np.testing.assert_array_equal(cavity_z, cavity)
    assert dose_z[20, 20, 24 + 3] == 4.0 and dose_z[20, 20, 24 + 4] == 0.0
    assert dose_z[20, 20 + 4 + 6, 20] == 4.0 and dose_z[20, 20 + 4 + 7, 20] == 0.0
    cavity_empty, dose_empty = sa.treatment_maps(np.full(shape, 0.59), (1.0, 1.0, 1.0), 0.6, 6.0, 60.0)
    assert not cavity_empty.any() and not dose_empty.any()
    assert sa.treatment_maps(density, (1.0, 1.0, 1.0), 0.6, 0.0, 60.0)[1].astype(bool).sum() == cavity.sum()
    with pytest.raises(ValueError, match="3D"):
        sa.treatment_maps(density[0], (1.0, 1.0), 0.6, 6.0, 60.0)


def test_align_treated_config():
    """n_steps = n_growth + ceil(time_after / dt), the horizon a multiple
    of dt, resection_time the midpoint of step n_growth; a stage that did
    not run is refused."""
    config = {"resection_time": 71.2, "time_after_resection": 260.0, "steps_per_day": 12, "dt": None, "rho": 0.1}
    aligned = sa.align_treated_config(config, 217, 71.2 / 217)
    dt = 71.2 / 217
    n_after = int(np.ceil(260.0 / dt))
    assert aligned["n_steps"] == 217 + n_after and aligned["dt"] is None and aligned["steps_per_day"] is None
    assert aligned["resection_time"] == pytest.approx(216.5 * dt)
    assert aligned["time_after_resection"] == pytest.approx((n_after + 0.5) * dt)
    horizon = aligned["resection_time"] + aligned["time_after_resection"]
    assert horizon == pytest.approx(aligned["n_steps"] * dt) and horizon / aligned["n_steps"] == pytest.approx(dt, rel=1e-15)
    assert aligned["rho"] == 0.1 and "n_steps" not in config  # the input is left alone
    # time_after already a multiple of dt: no extra step.
    exact = sa.align_treated_config({"resection_time": 10.0, "time_after_resection": 5.0}, 100, 0.1)
    assert exact["n_steps"] == 150 and exact["time_after_resection"] == pytest.approx(50.5 * 0.1)
    with pytest.raises(ValueError, match="growth stage must have run"):
        sa.align_treated_config(config, 0, 0.1)


def test_treatment_settings():
    assert sa.treatment_settings() == {"cavity_threshold": 0.6, "rt_margin_mm": 15.0, "rt_dose_per_fraction_gy": 2.0}
    assert sa.spec_treatment_settings({}) == sa.treatment_settings()
    assert sa.spec_treatment_settings({"treatment": {"cavity_threshold": 0.5, "rt_margin_mm": 3, "rt_dose_per_fraction_gy": 1}}) == {
        "cavity_threshold": 0.5, "rt_margin_mm": 3.0, "rt_dose_per_fraction_gy": 1.0
    }
    for bad in ({"cavity_threshold": 0.0}, {"cavity_threshold": 1.5}, {"rt_margin_mm": -1.0}, {"rt_dose_per_fraction": np.inf}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            sa.treatment_settings(**bad)


def _phantom_search_space(tmp_path: Path) -> Path:
    """A small search space whose ranges keep the phantom tumor growing
    past the cavity threshold by surgery: rho, D and resection_time
    factors, the seed mass fixed, the three seed factors."""
    path = tmp_path / "small_space.json"
    path.write_text(
        json.dumps(
            {
                "solver": "StuppFKPPSolver",
                "_note": "small phantom search space",
                "rho": {"min": 0.1, "max": 0.3, "scale": "log"},
                "white_matter_diffusivity": {"min": 0.05, "max": 0.3, "scale": "log"},
                "resection_time": {"min": 10.0, "max": 14.0, "scale": "linear"},
                "gaussian_seed_mass": 250.0,
                **_seed_entries(),
            }
        )
    )
    return path


def test_run_one_two_stages(phantom_base):
    """One design point on the phantom (CPU): the growth stage's record
    and config are saved into growth/ without volumes, its final density
    as the pre-resection field (float32, the final field's affine; the
    re-solved saved growth config rounded for storage, as is the final
    field), the cavity is the unrounded density at
    or above the threshold, the dose map holds the total dose within the
    margin, the treated stage's config.json references both maps and
    reproduces the run through read_config, the seed is not saved, the
    treated stage's final density vanishes inside the cavity; the two
    treated-stage snapshots are saved at the end of their Sundays (the
    re-solved saved config reproduces them), recorded in treatment.json;
    the qoi record holds the pre_, mid_crt_ and end_crt_ QoIs of the
    snapshot fields and the time stepping of both stages; a run without
    the pre-resection field (--no-keep-pre-resection-field) has no pre_
    QoIs and is counted, and a run without snapshots has no mid_crt_ /
    end_crt_ QoIs and is counted."""
    tmp_path = phantom_base["tmp_path"]
    sweep_dir = sa.make_design(
        _phantom_search_space(tmp_path), phantom_base["path"], tmp_path / "sa", "one", log2_n=1, seed=2, rt_margin_mm=3.0
    )
    config_path = sweep_dir / "configs" / "r0000_A.json"
    run_dir = sweep_dir / "runs" / "r0000_A"
    spec = json.loads((sweep_dir / "spec.json").read_text())
    treatment = sa.spec_treatment_settings(spec)
    snapshots = sa.spec_snapshot_offsets(spec)
    assert treatment["rt_margin_mm"] == 3.0 and snapshots == {"mid_crt": 22.0, "end_crt": 43.0}
    # The phantom's adjuvant sessions (days 80-84) lie beyond its horizon
    # (day 72) and are dropped from every run config.
    assert spec["schedule"]["n_chemo_sessions_dropped"] == 5 and spec["schedule"]["chemo_total_dose"] == 42 * 75.0
    assert sa.run_one(
        config_path, run_dir, treatment["cavity_threshold"], 3.0, treatment["rt_dose_per_fraction_gy"], snapshots=snapshots
    ) == 0
    growth = json.loads((run_dir / "growth" / "result.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    config = _config_entries(config_path)
    assert growth["success"] and result["success"]
    assert len(config["chemo_times"]) == 42 and config["chemo_times"][-1] <= config["resection_time"] + config["time_after_resection"]
    assert read_config(run_dir / "growth" / "config.json")[SOLVER_KEY] == "FKPPSolver"
    assert growth["final_time"] == config["resection_time"]
    # The treated stage is stepped on the growth stage's grid of times.
    n_after = int(np.ceil(config["time_after_resection"] / growth["dt"] - 1e-9))
    assert result["n_steps"] == growth["n_steps"] + n_after
    assert result["dt"] == pytest.approx(growth["dt"], rel=1e-12)
    horizon = config["resection_time"] + config["time_after_resection"]
    assert horizon - 1e-9 <= result["final_time"] <= horizon + growth["dt"]
    assert result["final_time"] == pytest.approx(result["n_steps"] * growth["dt"], rel=1e-12)
    assert sorted(p.name for p in (run_dir / "growth").iterdir()) == ["config.json", "result.json"]
    assert growth["files"] == ["config.json", "result.json"] and growth["grid_shape"] is None
    growth_density = FKPPSolver(read_config(run_dir / "growth" / "config.json")).solve().final_state["cell_density"]
    growth_density = np.asarray(growth_density, dtype=np.float64)
    pre_image = nib.load(str(run_dir / "pre_resection_cell_density.nii.gz"))
    final_image = nib.load(str(run_dir / "final_cell_density.nii.gz"))
    assert pre_image.get_data_dtype() == final_image.get_data_dtype() == np.float32
    np.testing.assert_array_equal(pre_image.affine, final_image.affine)
    assert pre_image.header.get_zooms() == final_image.header.get_zooms()
    pre_density = np.asarray(pre_image.get_fdata(), dtype=np.float64)
    np.testing.assert_array_equal(pre_density, sa.round_field(growth_density))
    np.testing.assert_allclose(pre_density, growth_density, rtol=2.0**-8, atol=1e-10)
    cavity = np.asarray(nib.load(str(run_dir / "resection_cavity.nii.gz")).get_fdata())
    dose = np.asarray(nib.load(str(run_dir / "rt_dose.nii.gz")).get_fdata())
    record = json.loads((run_dir / "treatment.json").read_text())
    assert set(np.unique(cavity)) <= {0.0, 1.0} and cavity.sum() == record["n_cavity_voxels"] > 0
    np.testing.assert_array_equal(cavity.astype(bool), growth_density >= 0.6)
    expected_cavity, expected_dose = sa.treatment_maps(growth_density, (1.0, 1.0, 1.0), 0.6, 3.0, 60.0)
    np.testing.assert_array_equal(cavity.astype(bool), expected_cavity)
    np.testing.assert_array_equal(dose, expected_dose)
    assert record["rt_total_dose_gy"] == 60.0 and record["n_fractions"] == 30
    assert record["dose_volume_mm3"] == np.count_nonzero(dose) > record["cavity_volume_mm3"] == cavity.sum()
    assert record["max_density_at_resection"] == pytest.approx(growth_density.max(), rel=1e-6)
    saved = read_config(run_dir / "config.json")
    assert saved["resection_cavity"] == {"segmentation": str(run_dir / "resection_cavity.nii.gz"), "label": 1}
    assert saved["rt_dose"] == str(run_dir / "rt_dose.nii.gz")
    assert saved["n_steps"] == result["n_steps"] and saved["steps_per_day"] is None and saved["dt"] is None
    assert saved["resection_time"] == pytest.approx((growth["n_steps"] - 0.5) * growth["dt"])
    assert record["resection_step"] == growth["n_steps"] and record["resection_time"] == config["resection_time"]
    assert record["treated_dt"] == result["dt"] and record["growth_dt"] == growth["dt"]
    assert record["resection_time_config"] == saved["resection_time"]
    reproduced = StuppFKPPSolver(saved)
    np.testing.assert_array_equal(reproduced.params["resection_cavity"], expected_cavity)
    np.testing.assert_allclose(reproduced.params["rt_dose"], expected_dose)
    assert not (run_dir / "initial_cell_density.nii.gz").exists()
    assert result["files"] == ["config.json", "final_cell_density.nii.gz", "result.json"]
    final = np.asarray(nib.load(str(run_dir / "final_cell_density.nii.gz")).get_fdata())
    assert final.max() > 0 and np.all(final[expected_cavity] == 0)
    np.testing.assert_array_equal(final, sa.round_field(final))  # stored rounded
    assert record["pre_resection_file"] == "pre_resection_cell_density.nii.gz"
    # The snapshots: the state at the end of the Sundays 22 and 43 days
    # after the sampled resection_time, requested as step ends
    # (snapshot_days), recorded by the solver at exactly those days
    # (result.json), saved rounded with the final field's affine, both
    # after the resection and inside the cavity zero; the saved config
    # holds the requested days and its re-solve reproduces the frames.
    days = sa.snapshot_days(config["resection_time"], snapshots, growth["dt"])
    assert saved["snapshot_times"] == sorted(days.values()) and result["snapshot_times"] == pytest.approx(saved["snapshot_times"])
    assert record["snapshots"].keys() == {"mid_crt", "end_crt"}
    for name, offset in snapshots.items():
        entry = record["snapshots"][name]
        assert entry["offset_days"] == offset and entry["day"] == config["resection_time"] + offset
        assert entry["requested_day"] == days[name] and entry["file"] == f"{name}_cell_density.nii.gz"
        assert entry["recorded_day"] == pytest.approx(days[name], abs=1e-9)
        t_end = entry["day"] + 1
        assert t_end - 1.5 * growth["dt"] < entry["recorded_day"] <= t_end - 0.5 * growth["dt"] + 1e-12
        assert entry["recorded_day"] > saved["resection_time"]
    frames = {name: nib.load(str(run_dir / f"{name}_cell_density.nii.gz")) for name in snapshots}
    for name, image in frames.items():
        assert image.get_data_dtype() == np.float32
        np.testing.assert_array_equal(image.affine, final_image.affine)
        frame = np.asarray(image.get_fdata(), dtype=np.float64)
        assert frame.shape == final.shape and frame.max() > 0 and np.all(frame[expected_cavity] == 0)
        np.testing.assert_array_equal(frame, sa.round_field(frame))
    assert not np.array_equal(frames["mid_crt"].get_fdata(), frames["end_crt"].get_fdata())
    assert not np.array_equal(frames["end_crt"].get_fdata(), final)
    reproduced_result = reproduced.solve()
    assert reproduced_result.success and reproduced_result.snapshot_times.tolist() == pytest.approx(saved["snapshot_times"])
    for index, day in enumerate(saved["snapshot_times"]):
        name = next(key for key, value in days.items() if value == day)
        np.testing.assert_array_equal(
            np.asarray(frames[name].get_fdata(), dtype=np.float32),
            sa.round_field(reproduced_result.time_series["cell_density"][index]),
        )
    assert not (run_dir / "time_series_cell_density.nii.gz").exists()
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.json", "end_crt_cell_density.nii.gz", "final_cell_density.nii.gz", "growth", "mid_crt_cell_density.nii.gz",
        "pre_resection_cell_density.nii.gz", "resection_cavity.nii.gz", "result.json", "rt_dose.nii.gz", "treatment.json",
    ]
    # The saved records feed run_status.csv and qoi.csv: the time
    # stepping of both stages, the volumes; the qoi record holds the
    # pre_ QoIs of the pre-resection field next to those of the final one.
    saved_records = sa.run_records(run_dir)
    assert saved_records["success"] and saved_records["solver"] == "StuppFKPPSolver"
    assert saved_records["growth_dt"] == growth["dt"] and saved_records["growth_n_steps"] == growth["n_steps"]
    assert saved_records["treated_dt"] == result["dt"] and saved_records["n_steps"] == result["n_steps"]
    assert saved_records["cavity_volume_mm3"] == record["cavity_volume_mm3"]
    design = {r["run_name"]: r for r in _read_csv(sweep_dir / "design.csv")}
    wm = phantom_base["wm"]
    qoi = sa.qoi_record(sweep_dir, design["r0000_A"], wm, (1.0, 1.0, 1.0), 0.6, 0.3)
    assert qoi["success"] and qoi["growth_dt"] == growth["dt"] and qoi["treated_dt"] == result["dt"]
    seed_voxel = tuple(int(design["r0000_A"][f"seed_voxel_{ijk}"]) for ijk in "ijk")
    expected_pre = sa.compute_qois(pre_density, (1.0, 1.0, 1.0), seed_voxel, wm, 0.6, 0.3)
    expected_final = sa.compute_qois(final, (1.0, 1.0, 1.0), seed_voxel, wm, 0.6, 0.3)
    for name in sa.QOI_NAMES:
        assert qoi[f"pre_{name}"] == expected_pre[name] and qoi[name] == expected_final[name], name
    for snapshot in snapshots:
        frame = np.asarray(frames[snapshot].get_fdata(), dtype=np.float64)
        expected = sa.compute_qois(frame, (1.0, 1.0, 1.0), seed_voxel, wm, 0.6, 0.3)
        for name in sa.QOI_NAMES:
            assert qoi[f"{snapshot}_{name}"] == expected[name], (snapshot, name)
        assert qoi[f"{snapshot}_mass"] > 0 and qoi[f"{snapshot}_mass"] != qoi["mass"]
    assert qoi["pre_V_core"] > 0 and qoi["pre_mass"] > 0 and qoi["pre_mass"] != qoi["mass"]
    assert set(qoi) == set(sa.QOI_COLUMNS)
    # Without the pre-resection field and without snapshots: no files, no
    # pre_ / mid_crt_ / end_crt_ QoIs, counted.
    run_dir_b = sweep_dir / "runs" / "r0000_B"
    assert sa.run_one(sweep_dir / "configs" / "r0000_B.json", run_dir_b, 0.6, 3.0, 2.0, keep_pre_resection_field=False) == 0
    assert not (run_dir_b / "pre_resection_cell_density.nii.gz").exists()
    assert not (run_dir_b / "mid_crt_cell_density.nii.gz").exists() and not (run_dir_b / "end_crt_cell_density.nii.gz").exists()
    record_b = json.loads((run_dir_b / "treatment.json").read_text())
    assert record_b["pre_resection_file"] is None and record_b["snapshots"] == {}
    assert read_config(run_dir_b / "config.json")["snapshot_times"] is None
    qoi_b = sa.qoi_record(sweep_dir, design["r0000_B"], wm, (1.0, 1.0, 1.0), 0.6, 0.3)
    assert qoi_b["success"] and not any(key.startswith(("pre_", "mid_crt_", "end_crt_")) for key in qoi_b)
    summary = sa.qoi_summary([qoi, qoi_b], 0.6, 0.3, n_blocks=1, size=8)  # r0000_A is index 0, r0000_B index 7
    assert summary["n_runs_without_pre_resection_field"] == 1 and summary["n_success"] == 2
    assert summary["n_runs_without_mid_crt_field"] == 1 and summary["n_runs_without_end_crt_field"] == 1
    assert summary["per_qoi"]["pre_log10_mass"]["n_runs_nan"] == 1 and summary["per_qoi"]["log10_mass"]["n_runs_nan"] == 0
    assert summary["per_qoi"]["mid_crt_log10_mass"]["n_runs_nan"] == 1 and summary["per_qoi"]["end_crt_centroid_x_mm"]["n_runs_nan"] == 1
    assert summary["per_qoi"]["pre_log10_mass"]["n_runs_failed"] == 6  # the six runs not run
    with pytest.raises(ValueError, match="unknown name"):
        sa.run_one(sweep_dir / "configs" / "r0000_B.json", sweep_dir / "runs" / "bad", snapshots={"late_crt": 3.0})
    # The run-status record of a run that failed in its growth stage.
    failed_dir = sweep_dir / "runs" / "failed"
    (failed_dir / "growth").mkdir(parents=True)
    (failed_dir / "growth" / "result.json").write_text(json.dumps({"success": False, "error": "boom"}))
    (sweep_dir / "configs" / "failed.json").write_text("not json")
    failed = sa.run_subprocess(sweep_dir, "failed", None, treatment)
    assert not failed["success"] and failed["error"] == "growth stage: boom" and failed["exit_code"] != 0
    assert list(failed) == sa.STATUS_COLUMNS and failed["growth_dt"] is None


def test_growth_only(phantom_base):
    """Growth-only mode: the design records it and refuses treatment
    factors other than resection_time; a run is the growth stage saved
    into the run directory itself (an FKPPSolver config.json, the final
    field, a pre-resection symlink to it, no maps, no growth/, no
    treatment.json), its records report the one stage's time step, and
    its qoi record has equal final and pre_ QoIs; the run pass follows
    spec.json's mode and refuses --growth-only on a two-stage design."""
    tmp_path = phantom_base["tmp_path"]
    space_path = _phantom_search_space(tmp_path)
    sweep_dir = sa.make_design(space_path, phantom_base["path"], tmp_path / "sa", "growth", log2_n=1, seed=2, growth_only=True)
    spec = json.loads((sweep_dir / "spec.json").read_text())
    assert spec["growth_only"] is True and "resection_time" in spec["factor_names"]
    treated_space = tmp_path / "treated_space.json"
    entries = json.loads(space_path.read_text())
    treated_space.write_text(json.dumps({**entries, "chemo_kill_rate": {"min": 1e-4, "max": 1e-3, "scale": "log"}}))
    with pytest.raises(ValueError, match=r"growth-only design: \['chemo_kill_rate'\] are treatment parameters"):
        sa.make_design(treated_space, phantom_base["path"], tmp_path / "sa", "refused", log2_n=1, growth_only=True)
    assert not (tmp_path / "sa" / "refused").exists()
    config_path = sweep_dir / "configs" / "r0000_A.json"
    run_dir = sweep_dir / "runs" / "r0000_A"
    assert sa.run_one(config_path, run_dir, growth_only=True) == 0
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.json", "final_cell_density.nii.gz", "pre_resection_cell_density.nii.gz", "result.json",
    ]
    pre_path = run_dir / "pre_resection_cell_density.nii.gz"
    assert pre_path.is_symlink() and pre_path.resolve() == (run_dir / "final_cell_density.nii.gz").resolve()
    result = json.loads((run_dir / "result.json").read_text())
    config = _config_entries(config_path)
    assert result["success"] and result[SOLVER_KEY] == "FKPPSolver" and result["final_time"] == config["resection_time"]
    assert result["files"] == ["config.json", "final_cell_density.nii.gz", "result.json"]
    saved = read_config(run_dir / "config.json")
    assert saved[SOLVER_KEY] == "FKPPSolver" and saved["stopping_time"] == config["resection_time"]
    records = sa.run_records(run_dir)
    assert records["success"] and records["solver"] == "FKPPSolver"
    assert records["growth_dt"] == result["dt"] and records["growth_n_steps"] == result["n_steps"]
    assert records["treated_dt"] is None and records["cavity_volume_mm3"] is None
    design = {r["run_name"]: r for r in _read_csv(sweep_dir / "design.csv")}
    qoi = sa.qoi_record(sweep_dir, design["r0000_A"], phantom_base["wm"], (1.0, 1.0, 1.0), 0.6, 0.3)
    assert qoi["success"] and qoi["mass"] > 0 and qoi["V_core"] > 0
    for name in sa.QOI_NAMES:
        assert qoi[f"pre_{name}"] == qoi[name], name
    assert not any(key.startswith(("mid_crt_", "end_crt_")) for key in qoi)  # no treated stage, no snapshots
    assert qoi["growth_dt"] == result["dt"] and qoi["treated_dt"] is None and qoi["cavity_volume_mm3"] is None
    summary = sa.qoi_summary([qoi], 0.6, 0.3, n_blocks=1, size=1)
    assert summary["n_runs_empty_cavity"] == 0 and summary["n_runs_without_pre_resection_field"] == 0
    assert summary["n_runs_without_mid_crt_field"] == 1 and summary["n_runs_without_end_crt_field"] == 1
    assert spec["snapshots"] is None and sa.spec_snapshot_offsets(spec) == {}
    # Without the pre-resection field: no symlink either.
    run_dir_b = sweep_dir / "runs" / "r0000_B"
    assert sa.run_one(sweep_dir / "configs" / "r0000_B.json", run_dir_b, growth_only=True, keep_pre_resection_field=False) == 0
    assert sorted(p.name for p in run_dir_b.iterdir()) == ["config.json", "final_cell_density.nii.gz", "result.json"]
    # The mode of a run pass.
    assert sa.resolve_growth_only(sweep_dir, dict(spec), False) is True  # the design decides
    assert sa.resolve_growth_only(sweep_dir, dict(spec), True) is True
    two_stage = {**spec, "growth_only": False}
    assert sa.resolve_growth_only(sweep_dir, two_stage, False) is False
    with pytest.raises(ValueError, match="designed as a two-stage sweep"):
        sa.resolve_growth_only(sweep_dir, two_stage, True)
    older = tmp_path / "older"
    older.mkdir()
    old_spec = {key: value for key, value in spec.items() if key != "growth_only"}
    sa.write_json(older / "spec.json", old_spec)
    assert sa.resolve_growth_only(older, dict(old_spec), True) is True
    assert json.loads((older / "spec.json").read_text())["growth_only"] is True
    sa.write_json(older / "spec.json", old_spec)
    assert sa.resolve_growth_only(older, dict(old_spec), False) is False
    assert json.loads((older / "spec.json").read_text())["growth_only"] is False
    # The command line.
    args = sa.build_parser().parse_args(["run", "--sweep-dir", "x", "--growth-only", "--no-keep-pre-resection-field"])
    assert args.growth_only and not args.keep_pre_resection_field
    args = sa.build_parser().parse_args(["all", "--name", "x"])
    assert not args.growth_only and args.keep_pre_resection_field
    args = sa.build_parser().parse_args(["run-one", "--config", "c", "--run-dir", "d", "--growth-only"])
    assert args.growth_only and args.keep_pre_resection_field


# --- (6) QoIs on synthetic fields ---
def test_chemo_log_kill_range():
    """The shipped ranges with the example config's schedule (8 900 mg/m^2
    over six cycles, 4 900 within the shipped 120-day horizon) give the
    total log kill kill * D_tot / decay range of the factor bounds at the
    fixed decay rate; an override or a fixed value stands in for a factor
    range, and the sum runs over the sessions within the horizon."""
    example = read_config(SCRIPT.parent / "stupp_config_example.json", solver=StuppFKPPSolver)
    space = sa.load_search_space(SHIPPED_SEARCH_SPACE, CONFIG_KEYS)
    total_dose, (low, high) = sa.chemo_log_kill_range(example, space)
    assert total_dose == 4900.0 and 0 < low < high
    kill, decay = space.factors["chemo_kill_rate"], space.overrides["chemo_decay_rate"]
    assert (low, high) == (kill.low * 4900 / decay, kill.high * 4900 / decay)
    fixed = sa.load_search_space(
        _seed_entries(
            chemo_kill_rate=2e-3,
            chemo_times=[110.0, 120.0, 400.0],
            chemo_doses=[100.0, 100.0, 100.0],
            chemo_decay_rate={"min": 5.0, "max": 20.0, "scale": "log"},
        ),
        CONFIG_KEYS,
    )
    assert sa.chemo_log_kill_range(example, fixed) == (200.0, (2e-3 * 200 / 20, 2e-3 * 200 / 5))
    base_only = sa.load_search_space(_seed_entries(), CONFIG_KEYS)
    expected = example["chemo_kill_rate"] * 8900 / example["chemo_decay_rate"]
    assert sa.chemo_log_kill_range(example, base_only) == (8900.0, (expected, expected))


# --- (6) QoIs on synthetic fields ---


def _gaussian(shape, zooms, centre_mm, sigmas_mm, peak=1.0) -> np.ndarray:
    grid = np.indices(shape).astype(np.float64)
    exponent = sum(
        ((grid[axis] * zooms[axis] - centre_mm[axis]) / sigmas_mm[axis]) ** 2 for axis in range(3)
    )
    return peak * np.exp(-0.5 * exponent)


def test_qois_on_gaussians():
    """Isotropic Gaussian of std sigma: R_g = sqrt(3) sigma, anisotropy 1,
    centroid_drift = the seed offset, the centroid components the
    Gaussian's centre in mm (absolute, whatever the seed), wm_fraction 1
    on an all-WM map, V_tau and r95 from the analytic ball radius
    sigma sqrt(2 ln(1 / tau)); stds (2 sigma, sigma, sigma): anisotropy 4;
    on anisotropic voxels the mm geometry still holds; an empty
    compartment gives r95 = 0; a zero field gives NaN mass-weighted
    QoIs."""
    shape, sigma = (64, 64, 64), 6.0  # 6 mm: the tau balls hold ~1000 voxels
    centre = np.array([32.0, 32.0, 32.0])
    wm = np.ones(shape)
    field = _gaussian(shape, (1.0, 1.0, 1.0), centre, (sigma,) * 3)
    qoi = sa.compute_qois(field, (1.0, 1.0, 1.0), (32, 32, 32), wm)
    assert set(qoi) == set(sa.QOI_NAMES)
    np.testing.assert_allclose(qoi["mass"], (2 * np.pi) ** 1.5 * sigma**3, rtol=1e-4)
    np.testing.assert_allclose(qoi["log10_mass"], np.log10(qoi["mass"]))
    np.testing.assert_allclose(qoi["R_g"], np.sqrt(3) * sigma, rtol=1e-3)
    np.testing.assert_allclose(qoi["anisotropy"], 1.0, atol=1e-6)
    np.testing.assert_allclose(qoi["log10_anisotropy"], 0.0, atol=1e-6)
    assert qoi["centroid_drift"] < 1e-4 and qoi["wm_fraction"] == pytest.approx(1.0)  # 0..63 grid: asymmetric tails
    np.testing.assert_allclose([qoi[key] for key in sa.CENTROID_QOIS], centre, atol=1e-4)
    shifted = _gaussian(shape, (1.0, 1.0, 1.0), np.array([30.0, 33.0, 35.0]), (sigma,) * 3)
    shifted_qoi = sa.compute_qois(shifted, (1.0, 1.0, 1.0), (32, 32, 32), wm)
    np.testing.assert_allclose([shifted_qoi[key] for key in sa.CENTROID_QOIS], [30.0, 33.0, 35.0], atol=1e-3)
    np.testing.assert_allclose(shifted_qoi["centroid_drift"], np.sqrt(4 + 1 + 9), atol=1e-3)
    # The components are absolute: the seed voxel does not enter them.
    other_seed = sa.compute_qois(shifted, (1.0, 1.0, 1.0), (10, 10, 10), wm)
    assert [other_seed[key] for key in sa.CENTROID_QOIS] == [shifted_qoi[key] for key in sa.CENTROID_QOIS]
    assert other_seed["centroid_drift"] != shifted_qoi["centroid_drift"]
    for label, tau in (("core", 0.6), ("edema", 0.3)):
        radius = sigma * np.sqrt(2 * np.log(1 / tau))
        np.testing.assert_allclose(qoi[f"V_{label}"], 4 / 3 * np.pi * radius**3, rtol=0.03)
        np.testing.assert_allclose(qoi[f"r95_{label}"], radius * 0.95 ** (1 / 3), rtol=0.03)
        assert qoi[f"n_{label}"] == qoi[f"V_{label}"]
        np.testing.assert_allclose(qoi[f"log10_V_{label}"], np.log10(qoi[f"V_{label}"] + 1.0))
    off = sa.compute_qois(field, (1.0, 1.0, 1.0), (35, 32, 28), wm)
    np.testing.assert_allclose(off["centroid_drift"], 5.0, atol=1e-6)
    assert off["r95_edema"] > qoi["r95_edema"]
    half_wm = np.zeros(shape)
    half_wm[:32] = 1.0  # the peak plane x = 32 lies in the non-WM half
    np.testing.assert_allclose(
        sa.compute_qois(field, (1.0, 1.0, 1.0), (32, 32, 32), half_wm)["wm_fraction"],
        field[:32].sum() / field.sum(),
        rtol=1e-9,
    )
    # Anisotropic Gaussian (a longer grid along x keeps the tails inside).
    stretched = _gaussian((128, 64, 64), (1.0, 1.0, 1.0), (64.0, 32.0, 32.0), (2 * sigma, sigma, sigma))
    qoi = sa.compute_qois(stretched, (1.0, 1.0, 1.0), (64, 32, 32), np.ones((128, 64, 64)))
    np.testing.assert_allclose(qoi["anisotropy"], 4.0, rtol=0.02)
    np.testing.assert_allclose(qoi["R_g"], np.sqrt(4 + 1 + 1) * sigma, rtol=1e-3)
    # Anisotropic voxels: the geometry is in mm.
    zooms = (1.0, 1.5, 2.0)
    field_mm = _gaussian((64, 48, 32), zooms, centre, (sigma,) * 3)
    qoi = sa.compute_qois(field_mm, zooms, (32, 21, 16), wm[:64, :48, :32])
    np.testing.assert_allclose(qoi["mass"], (2 * np.pi) ** 1.5 * sigma**3, rtol=1e-2)
    np.testing.assert_allclose(qoi["R_g"], np.sqrt(3) * sigma, rtol=1e-2)
    np.testing.assert_allclose(qoi["anisotropy"], 1.0, atol=0.02)
    radius = sigma * np.sqrt(2 * np.log(1 / 0.3))
    np.testing.assert_allclose(qoi["V_edema"], 4 / 3 * np.pi * radius**3, rtol=0.08)
    np.testing.assert_allclose(qoi["centroid_drift"], np.linalg.norm(centre - np.array([32, 21, 16]) * zooms), atol=1e-6)
    np.testing.assert_allclose([qoi[key] for key in sa.CENTROID_QOIS], centre, atol=1e-3)  # mm, not voxels
    # Empty core compartment: a valid outcome with r95 = 0.
    low = _gaussian(shape, (1.0, 1.0, 1.0), centre, (sigma,) * 3, peak=0.5)
    qoi = sa.compute_qois(low, (1.0, 1.0, 1.0), (32, 32, 32), wm)
    assert qoi["V_core"] == 0 and qoi["r95_core"] == 0 and qoi["n_core"] == 0
    assert qoi["V_edema"] > 0 and qoi["r95_edema"] > 0 and qoi["log10_V_core"] == 0.0
    # Zero field: nothing to locate.
    qoi = sa.compute_qois(np.zeros(shape), (1.0, 1.0, 1.0), (32, 32, 32), wm)
    assert qoi["mass"] == 0 and qoi["log10_mass"] == np.log10(sa.MASS_FLOOR_VOXELS)
    assert qoi["voxel_volume"] == 1.0
    assert all(np.isnan(qoi[key]) for key in (*sa.MASS_WEIGHTED_QOIS, "log10_anisotropy"))
    assert qoi["V_core"] == qoi["V_edema"] == qoi["r95_core"] == qoi["r95_edema"] == 0
    with pytest.raises(ValueError, match="shape"):
        sa.compute_qois(field, (1.0, 1.0, 1.0), (32, 32, 32), wm[:32])


def test_log10_mass_floor():
    """log10_mass = log10 max(M, 1e-3 dV): a zero field and an extinct
    field (M = 1e-30) sit at the floor, a field above it is unchanged; the
    floor scales with the voxel volume."""
    shape, zooms = (16, 16, 16), (1.0, 1.5, 2.0)
    voxel_volume = 3.0
    wm = np.ones(shape)
    floor = np.log10(sa.MASS_FLOOR_VOXELS * voxel_volume)
    zero = sa.compute_qois(np.zeros(shape), zooms, (8, 8, 8), wm)
    assert zero["log10_mass"] == floor and zero["mass"] == 0
    tiny = np.zeros(shape)
    tiny[8, 8, 8] = 1e-30 / voxel_volume
    extinct = sa.compute_qois(tiny, zooms, (8, 8, 8), wm)
    np.testing.assert_allclose(extinct["mass"], 1e-30)
    assert extinct["log10_mass"] == floor and np.isnan(extinct["R_g"])
    above = np.zeros(shape)
    above[8, 8, 8] = 0.5
    qoi = sa.compute_qois(above, zooms, (8, 8, 8), wm)
    np.testing.assert_allclose(qoi["log10_mass"], np.log10(0.5 * voxel_volume))
    assert np.isfinite(qoi["R_g"])
    records = [
        {"success": True, "index": i, **values} for i, values in enumerate((zero, extinct, qoi))
    ]
    status = [  # r1 failed first and was redone: only its last record counts
        {"run_name": "r0", "success": "True", "wall_time_s": "2.0"},
        {"run_name": "r1", "success": "False", "wall_time_s": "9.0"},
        {"run_name": "r1", "success": "True", "wall_time_s": "4.0"},
    ]
    summary = sa.qoi_summary(records, 0.6, 0.3, n_blocks=1, size=3, status_records=status)
    assert summary["n_runs_extinct"] == 2 and summary["n_nan_mass_weighted"] == 2
    assert summary["mean_run_wall_time_s"] == 3.0
    assert sa.qoi_summary(records, 0.6, 0.3, 1, 3)["mean_run_wall_time_s"] is None
    assert summary["per_qoi"]["log10_mass"]["n_blocks_dropped"] == 0
    assert summary["per_qoi"]["R_g"] == {
        "n_runs_total": 3, "n_runs_failed": 0, "n_runs_nan": 2, "n_blocks_total": 1,
        "n_blocks_dropped": 1, "n_blocks_used": 0, "dropped_blocks": [0],
    }


def test_snapshot_qois_on_synthetic_run(tmp_path):
    """A synthetic run directory (a result.json reporting success, a
    final field and Gaussian snapshot fields of different widths and
    centres): the qoi record holds the QoIs of every snapshot under its
    prefix (pre_, mid_crt_, end_crt_), each equal to compute_qois on that
    field, and a missing snapshot leaves its columns absent."""
    shape, zooms = (32, 32, 32), (1.0, 1.0, 1.0)
    wm = np.ones(shape)
    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "runs" / "r0000_A"
    run_dir.mkdir(parents=True)
    fields = {
        "final_cell_density.nii.gz": _gaussian(shape, zooms, (16.0, 16.0, 16.0), (3.0,) * 3, peak=0.9),
        "pre_resection_cell_density.nii.gz": _gaussian(shape, zooms, (16.0, 16.0, 16.0), (2.0,) * 3, peak=0.95),
        "mid_crt_cell_density.nii.gz": _gaussian(shape, zooms, (14.0, 16.0, 18.0), (4.0,) * 3, peak=0.7),
        "end_crt_cell_density.nii.gz": _gaussian(shape, zooms, (13.0, 17.0, 19.0), (5.0,) * 3, peak=0.5),
    }
    for name, field in fields.items():
        nib.save(nib.Nifti1Image(field.astype(np.float32), np.eye(4)), str(run_dir / name))
    sa.write_json(run_dir / "result.json", {"solver": "StuppFKPPSolver", "success": True, "final_time": 72.0, "n_steps": 10, "dt": 0.1, "wall_time_s": 1.0})
    design = {"run_name": "r0000_A", "index": 0, "row": 0, "matrix": "A", "seed_voxel_i": 16, "seed_voxel_j": 16, "seed_voxel_k": 16}
    record = sa.qoi_record(sweep_dir, design, wm, zooms, 0.6, 0.3)
    assert record["success"] and set(record) == set(sa.QOI_COLUMNS)
    seed = (16, 16, 16)
    for prefix, file in {"": "final_cell_density.nii.gz", **sa.SNAPSHOT_PREFIXES}.items():
        expected = sa.compute_qois(np.asarray(nib.load(str(run_dir / file)).get_fdata(), dtype=np.float64), zooms, seed, wm, 0.6, 0.3)
        for name in sa.QOI_NAMES:
            assert record[f"{prefix}{name}"] == expected[name], (prefix, name)
    # The grid truncates the wider Gaussians' tails, so the centroids
    # are within a few hundredths of a mm of the centres.
    np.testing.assert_allclose([record[f"mid_crt_{key}"] for key in sa.CENTROID_QOIS], [14.0, 16.0, 18.0], atol=0.05)
    np.testing.assert_allclose([record[f"end_crt_{key}"] for key in sa.CENTROID_QOIS], [13.0, 17.0, 19.0], atol=0.1)
    np.testing.assert_allclose(record["end_crt_centroid_drift"], np.sqrt(9 + 1 + 9), atol=0.1)
    assert record["pre_R_g"] < record["R_g"] < record["mid_crt_R_g"] < record["end_crt_R_g"]
    assert record["end_crt_V_core"] == 0 and record["mid_crt_V_core"] > 0  # the 0.5 peak has no core
    (run_dir / "end_crt_cell_density.nii.gz").unlink()
    partial = sa.qoi_record(sweep_dir, design, wm, zooms, 0.6, 0.3)
    assert partial["success"] and not any(key.startswith("end_crt_") for key in partial)
    assert partial["mid_crt_mass"] == record["mid_crt_mass"]
    summary = sa.qoi_summary([partial], 0.6, 0.3, n_blocks=1, size=1)
    assert summary["n_runs_without_end_crt_field"] == 1 and summary["n_runs_without_mid_crt_field"] == 0
    assert summary["n_runs_without_pre_resection_field"] == 0


# The qoi.csv and design.csv columns of the previous schema (the
# 2026-09-08 atlas sweep: rho and the diffusivity as factors, the seed
# width in mm, no treated-stage snapshots, no centroid components).
OLD_QOI_COLUMNS = (
    "run_name,index,row,matrix,success,mass,log10_mass,V_core,V_edema,log10_V_core,log10_V_edema,r95_core,r95_edema,"
    "centroid_drift,R_g,anisotropy,log10_anisotropy,wm_fraction,n_core,n_edema,voxel_volume,pre_mass,pre_log10_mass,"
    "pre_V_core,pre_V_edema,pre_log10_V_core,pre_log10_V_edema,pre_r95_core,pre_r95_edema,pre_centroid_drift,pre_R_g,"
    "pre_anisotropy,pre_log10_anisotropy,pre_wm_fraction,pre_n_core,pre_n_edema,pre_voxel_volume,final_time,n_steps,"
    "wall_time_s,growth_dt,growth_n_steps,treated_dt,cavity_volume_mm3,dose_volume_mm3"
).split(",")
OLD_FACTOR_NAMES = [
    "rho", "white_matter_diffusivity", "diffusivity_ratio", "resection_time", "chemo_kill_rate", "chemo_decay_rate",
    "rt_alpha", "rt_alpha_beta_ratio", "seed_peak_density", "seed_sigma_mm", *SEED_KEYS,
]


def test_analysis_of_previous_schema_sweep(tmp_path):
    """A sweep directory of the previous schema (13 factors, qoi.csv with
    the final and pre_ QoIs only, spec.json without schedule, snapshots
    or extra_keys) still goes through qoi_summary and analyze_sweep: the
    QoIs it holds are analysed, the new ones (centroid components, the
    treated-stage snapshots) are counted absent and skipped, and the
    figures are written."""
    names, n_blocks = OLD_FACTOR_NAMES, 4
    size = len(names) + 2
    sweep_dir = tmp_path / "old"
    sweep_dir.mkdir()
    rng = np.random.default_rng(1)
    labels = sa.matrix_labels(names, False)
    design, records = [], []
    for index in range(n_blocks * size):
        block, position = divmod(index, size)
        u = rng.random(len(names))
        design.append({"run_name": sa.run_name(block, labels[position]), "index": index, "row": block, "matrix": labels[position],
                       **{f"u_{name}": value for name, value in zip(names, u)}, **{name: value for name, value in zip(names, u)}})
        mass = 400.0 * (1 + u[0]) * (1 + 0.2 * u[1])
        record = {"run_name": design[-1]["run_name"], "index": index, "row": block, "matrix": labels[position], "success": True}
        for prefix in ("", "pre_"):
            record.update({
                f"{prefix}mass": mass, f"{prefix}log10_mass": np.log10(mass), f"{prefix}V_core": 20 * u[0], f"{prefix}V_edema": 40 * u[0],
                f"{prefix}log10_V_core": np.log10(20 * u[0] + 1), f"{prefix}log10_V_edema": np.log10(40 * u[0] + 1),
                f"{prefix}r95_core": 5 * u[3], f"{prefix}r95_edema": 8 * u[3], f"{prefix}centroid_drift": u[0] + u[3],
                f"{prefix}R_g": 3 + u[1], f"{prefix}anisotropy": 1 + u[2], f"{prefix}log10_anisotropy": np.log10(1 + u[2]),
                f"{prefix}wm_fraction": 0.5 * u[0], f"{prefix}n_core": 20 * u[0], f"{prefix}n_edema": 40 * u[0], f"{prefix}voxel_volume": 1.0,
            })
        record.update(final_time=360.0, n_steps=4320, wall_time_s=9.0, growth_dt=1 / 12, growth_n_steps=1200, treated_dt=1 / 12,
                      cavity_volume_mm3=100.0, dose_volume_mm3=900.0)
        records.append(record)
    sa.write_csv(sweep_dir / "qoi.csv", records, OLD_QOI_COLUMNS)
    assert list(_read_csv(sweep_dir / "qoi.csv")[0]) == OLD_QOI_COLUMNS
    sa.write_csv(sweep_dir / "design.csv", design)
    sa.write_json(sweep_dir / "spec.json", {
        "N": n_blocks, "k": len(names), "factor_names": names, "second_order": False, "block_size": size,
        "salib_version": "test", "factors": {n: {"min": 0.0, "max": 1.0, "scale": "linear"} for n in names},
        "derived_keys": ["gaussian_seed_mass", "gaussian_seed_diffusion_time"], "growth_only": False,
    })
    qoi_records = sa.read_csv(sweep_dir / "qoi.csv")
    summary = sa.qoi_summary(qoi_records, 0.6, 0.3, n_blocks, size)
    assert summary["n_success"] == len(records) and summary["n_runs_without_pre_resection_field"] == 0
    assert summary["n_runs_without_mid_crt_field"] == summary["n_runs_without_end_crt_field"] == len(records)
    results = sa.analyze_sweep(sweep_dir, n_bootstrap=10, seed=0)
    old_qois = [q for q in sa.ANALYSED_QOIS if q in OLD_QOI_COLUMNS]
    assert set(results) == set(old_qois) and "pre_R_g" in results and "log10_mass" in results
    sobol_summary = json.loads((sweep_dir / "sobol_summary.json").read_text())
    for qoi in sa.ANALYSED_QOIS:
        if qoi not in OLD_QOI_COLUMNS:
            assert sobol_summary["qois"][qoi]["skipped"] is True and summary["per_qoi"][qoi]["n_runs_nan"] == len(records)
    sobol = _read_csv(sweep_dir / "sobol.csv")
    assert {r["qoi"] for r in sobol} == set(old_qois) and len(sobol) == len(names) * len(old_qois)
    assert (sweep_dir / "figures" / "heatmap_ST.png").is_file() and (sweep_dir / "figures" / "scatter_pre_R_g.png").is_file()
    assert not (sweep_dir / "figures" / "scatter_centroid_x_mm.png").exists()
    _check_accounting_agreement(summary, sobol_summary, qoi_records)


# --- accounting ---

ACCOUNTING_FIELDS = (
    "n_runs_total", "n_runs_failed", "n_runs_nan", "n_blocks_total", "n_blocks_dropped",
    "n_blocks_used", "dropped_blocks",
)


def _check_accounting_agreement(qoi_summary: dict, sobol_summary: dict, qoi_records: list[dict]) -> None:
    """qoi_summary.json's per_qoi accounting equals sobol_summary.json's
    field by field, the _note's inequalities hold, and n_runs_extinct is
    the count of runs at or below the mass floor."""
    assert qoi_summary["_note"] == sobol_summary["_note"] == sa.ACCOUNTING_NOTE
    size = qoi_summary["block_size"]
    assert size == sobol_summary["block_size"]
    assert qoi_summary["n_blocks_total"] == sobol_summary["n_blocks_total"]
    for qoi in sa.ANALYSED_QOIS:
        a, b = qoi_summary["per_qoi"][qoi], sobol_summary["qois"][qoi]
        for field in ACCOUNTING_FIELDS:
            assert a[field] == b[field], (qoi, field)
        assert a["n_blocks_total"] * size == a["n_runs_total"]
        assert a["n_blocks_used"] + a["n_blocks_dropped"] == a["n_blocks_total"]
        assert len(a["dropped_blocks"]) == a["n_blocks_dropped"]
        assert -(-a["n_runs_nan"] // size) <= a["n_blocks_dropped"] <= a["n_runs_nan"] + a["n_runs_failed"]
    floor = qoi_summary["mass_floor_voxels"]
    extinct = sum(
        1 for r in qoi_records
        if r["success"] == "True" and float(r["mass"]) <= floor * float(r["voxel_volume"])
    )
    assert qoi_summary["n_runs_extinct"] == extinct


def test_accounting_on_poisoned_sweep(tmp_path):
    """A synthetic qoi.csv (N = 8 blocks of 4 runs, two factors) with one
    NaN run in block 0, two in block 1, a failed run in block 2 and an
    extinct (floored) run in block 3: qoi and analyze agree field by
    field, the four mass-weighted QoIs drop exactly blocks 0-3, log10_mass
    drops only the failed run's block, and the console lines agree."""
    names, n_blocks, size = ["f1", "f2"], 8, 4
    sweep_dir = tmp_path / "poisoned"
    sweep_dir.mkdir()
    rng = np.random.default_rng(0)
    records = []
    design = []
    labels = sa.matrix_labels(names, False)
    for index in range(n_blocks * size):
        block, position = divmod(index, size)
        f1, f2 = rng.random(2)
        design.append({"run_name": sa.run_name(block, labels[position]), "index": index, "row": block,
                       "matrix": labels[position], "f1": f1, "f2": f2})
        mass = 500.0 * (1 + f1) * (1 + 0.1 * f2)
        record = {"run_name": design[-1]["run_name"], "index": index, "row": block, "matrix": labels[position],
                  "success": True, "mass": mass, "log10_mass": np.log10(mass), "V_core": 30 * f1, "V_edema": 60 * f1,
                  "log10_V_core": np.log10(30 * f1 + 1), "log10_V_edema": np.log10(60 * f1 + 1),
                  "r95_core": 5 * f2, "r95_edema": 8 * f2, "centroid_drift": f1 + f2, "R_g": 3 + f1,
                  "centroid_x_mm": 50 + f1, "centroid_y_mm": 60 + f2, "centroid_z_mm": 70 + f1 * f2,
                  "anisotropy": 1 + f2, "log10_anisotropy": np.log10(1 + f2), "wm_fraction": 0.5 * f1,
                  "n_core": 30 * f1, "n_edema": 60 * f1, "voxel_volume": 1.0,
                  "final_time": 10.0, "n_steps": 100, "wall_time_s": 1.0}
        records.append(record)
    for index in (1, 4, 6):  # NaN mass-weighted QoIs: one in block 0, two in block 1
        for key in (*sa.MASS_WEIGHTED_QOIS, "log10_anisotropy"):
            records[index][key] = np.nan
    records[9] = {**records[9], "success": False}  # block 2: a failed run
    for key in sa.QOI_NAMES + ["final_time", "n_steps", "wall_time_s"]:
        records[9][key] = np.nan
    records[14]["mass"] = 1e-30  # block 3: extinct, floored log10_mass, NaN mass-weighted
    records[14]["log10_mass"] = np.log10(sa.MASS_FLOOR_VOXELS)
    for key in (*sa.MASS_WEIGHTED_QOIS, "log10_anisotropy"):
        records[14][key] = np.nan
    sa.write_csv(sweep_dir / "qoi.csv", records, sa.QOI_COLUMNS)
    sa.write_csv(sweep_dir / "design.csv", design)
    sa.write_json(sweep_dir / "spec.json", {
        "N": n_blocks, "k": 2, "factor_names": names, "second_order": False, "block_size": size,
        "salib_version": "test", "factors": {n: {"min": 0.0, "max": 1.0, "scale": "linear"} for n in names},
    })
    qoi_records = sa.read_csv(sweep_dir / "qoi.csv")
    summary = sa.qoi_summary(qoi_records, 0.6, 0.3, n_blocks, size)
    sa.write_json(sweep_dir / "qoi_summary.json", summary)
    sa.analyze_sweep(sweep_dir, n_bootstrap=10, seed=0)
    sobol_summary = json.loads((sweep_dir / "sobol_summary.json").read_text())
    _check_accounting_agreement(summary, sobol_summary, qoi_records)
    assert summary["n_runs_extinct"] == 1 and summary["n_failed"] == 1 and summary["n_nan_mass_weighted"] == 4
    for qoi in ("centroid_drift", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm", "R_g", "log10_anisotropy", "wm_fraction"):
        assert summary["per_qoi"][qoi] == {
            "n_runs_total": 32, "n_runs_failed": 1, "n_runs_nan": 4, "n_blocks_total": 8,
            "n_blocks_dropped": 4, "n_blocks_used": 4, "dropped_blocks": [0, 1, 2, 3],
        }, qoi
    for qoi in ("log10_mass", "V_core", "r95_edema"):
        assert summary["per_qoi"][qoi]["dropped_blocks"] == [2] and summary["per_qoi"][qoi]["n_runs_nan"] == 0
    assert sa.accounting_line("R_g", summary["per_qoi"]["R_g"]) == "  R_g: 4 NaN runs, 1 failed runs -> 4/8 blocks dropped"
    sobol = _read_csv(sweep_dir / "sobol.csv")
    assert {r["n_blocks_used"] for r in sobol if r["qoi"] == "R_g"} == {"4"}
    assert {r["n_blocks_used"] for r in sobol if r["qoi"] == "log10_mass"} == {"7"}
    # No snapshot field anywhere: the pre_, mid_crt_ and end_crt_ QoIs
    # are NaN for every successful run, counted, and skipped by the
    # analysis.
    assert summary["n_runs_without_pre_resection_field"] == 31
    assert summary["n_runs_without_mid_crt_field"] == summary["n_runs_without_end_crt_field"] == 31
    for qoi in sa.ANALYSED_QOIS:
        if qoi.startswith(tuple(sa.SNAPSHOT_PREFIXES)):
            assert summary["per_qoi"][qoi]["n_runs_nan"] == 31 and summary["per_qoi"][qoi]["n_blocks_used"] == 0
            assert sobol_summary["qois"][qoi]["skipped"] is True
    assert {r["qoi"] for r in sobol} == set(sa.FINAL_ANALYSED_QOIS)
    # The half-sample convergence fields are finite floats: the maxima of
    # |S1 - S1_half| and |ST - ST_half| over the factors.
    for qoi in sa.FINAL_ANALYSED_QOIS:
        entry = sobol_summary["qois"][qoi]
        rows = [r for r in sobol if r["qoi"] == qoi]
        for key, half in (("S1", "S1_half"), ("ST", "ST_half")):
            value = entry[f"max_abs_change_{key}_half"]
            assert isinstance(value, float) and np.isfinite(value)
            assert value == pytest.approx(max(abs(float(r[key]) - float(r[half])) for r in rows))


def test_half_sample_summary_fields():
    """qoi_summary_entry's max_abs_change_S1_half / _ST_half are the maxima
    of |S1 - S1_half| / |ST - ST_half| (finite floats, not null) whenever
    the half design was analysed, and None when it was too small."""
    from SALib.test_functions import Ishigami

    names = ["x1", "x2", "x3"]
    samples = sa.saltelli_design(names, log2_n=6, seed=1)
    y = Ishigami.evaluate(sa.transform_factor(samples, -np.pi, np.pi, "linear"))
    values = {index: float(value) for index, value in enumerate(y)}
    result = sa.analyze_response(values, names, n_blocks=64, n_bootstrap=10, seed=0)
    accounting = sa.response_accounting([{"index": i, "success": True, "y": v} for i, v in values.items()], "y", 64, 5)
    entry = sa.qoi_summary_entry(names, result, accounting)
    assert result["n_blocks_half"] == 32 and np.all(np.isfinite(result["S1_half"]))
    for key, half in (("S1", "S1_half"), ("ST", "ST_half")):
        value = entry[f"max_abs_change_{key}_half"]
        assert isinstance(value, float) and np.isfinite(value) and value > 0
        assert value == pytest.approx(float(np.max(np.abs(result[key] - result[half]))))
    assert sa._jsonable(entry)[f"max_abs_change_S1_half"] == entry["max_abs_change_S1_half"]
    # Three blocks: no half design (n_half = 1 < 2), the fields are None.
    small = {index: float(value) for index, value in enumerate(y[: 3 * 5])}
    result = sa.analyze_response(small, names, n_blocks=3, n_bootstrap=10, seed=0)
    assert result["n_blocks_half"] == 1 and np.all(np.isnan(result["S1_half"]))
    entry = sa.qoi_summary_entry(names, result, sa.response_accounting([], "y", 3, 5))
    assert entry["max_abs_change_S1_half"] is None and entry["max_abs_change_ST_half"] is None


# --- (7) end to end on the phantom ---


@pytest.mark.slow
def test_end_to_end_on_phantom(phantom_base):
    """design (N = 4, 6 factors: 32 runs, 3 mm margin), run on the CPU,
    qoi and analyze with 20 bootstrap resamples: the files of both stages
    exist, run_status.csv and qoi.csv have one line per run, the derived
    volumes are carried into qoi.csv, sobol.csv has k x #QoIs lines with
    no NaN in ST, the figures are written, and a second run pass skips
    every finished run. About 60-90 s: 32 processes, each compiling two
    scans (``slow``; deselect with -m "not slow")."""
    tmp_path = phantom_base["tmp_path"]
    sweep_dir = sa.make_design(
        _phantom_search_space(tmp_path), phantom_base["path"], tmp_path / "sa", "e2e", log2_n=2, seed=1, rt_margin_mm=3.0
    )
    spec = json.loads((sweep_dir / "spec.json").read_text())
    assert spec["k"] == 6 and spec["n_runs"] == 32 and spec["overrides"] == {"gaussian_seed_mass": 250.0}
    assert spec["treatment"]["rt_margin_mm"] == 3.0 and spec["treatment"]["rt_total_dose_gy"] == 60.0
    assert spec["derived_groups"] == {} and spec["growth_only"] is False and spec["time_step"]["steps_per_day"] == PHANTOM_STEPS_PER_DAY
    counts = sa.run_sweep(sweep_dir, gpus=[], jobs_per_gpu=8)  # 8 CPU workers
    assert counts == {"skipped": 0, "ok": 32, "failed": 0}
    status = _read_csv(sweep_dir / "run_status.csv")
    assert len(status) == 32 and all(r["success"] == "True" and r["exit_code"] == "0" for r in status)
    assert list(status[0]) == sa.STATUS_COLUMNS
    for record in status:
        run_dir = sweep_dir / "runs" / record["run_name"]
        assert (run_dir / "final_cell_density.nii.gz").is_file() and (run_dir / "config.json").is_file()
        assert (run_dir / "pre_resection_cell_density.nii.gz").is_file()
        growth = json.loads((run_dir / "growth" / "result.json").read_text())
        assert float(record["growth_dt"]) == growth["dt"] and int(record["growth_n_steps"]) == growth["n_steps"]
        assert float(record["treated_dt"]) == pytest.approx(growth["dt"], rel=1e-12)
        assert float(record["growth_dt"]) <= 1 / PHANTOM_STEPS_PER_DAY + 1e-12
        assert (run_dir / "growth" / "result.json").is_file() and (run_dir / "treatment.json").is_file()
        assert (run_dir / "resection_cavity.nii.gz").is_file() and (run_dir / "rt_dose.nii.gz").is_file()
        assert (run_dir / "mid_crt_cell_density.nii.gz").is_file() and (run_dir / "end_crt_cell_density.nii.gz").is_file()
        treatment = json.loads((run_dir / "treatment.json").read_text())
        assert {name: entry["offset_days"] for name, entry in treatment["snapshots"].items()} == spec["snapshots"] == {"mid_crt": 22.0, "end_crt": 43.0}
        assert (sweep_dir / "logs" / f"{record['run_name']}.log").is_file()
        assert json.loads((run_dir / "result.json").read_text())["success"] is True
        saved = read_config(run_dir / "config.json")
        assert saved["resection_cavity"]["segmentation"] == str(run_dir / "resection_cavity.nii.gz")
    assert sa.run_sweep(sweep_dir, gpus=[], jobs_per_gpu=1) == {"skipped": 32, "ok": 0, "failed": 0}
    sa.qoi_command(sweep_dir, 0.6, 0.3, workers=1)
    qoi = _read_csv(sweep_dir / "qoi.csv")
    assert len(qoi) == 32 and all(r["success"] == "True" for r in qoi)
    assert [r["index"] for r in qoi] == [str(i) for i in range(32)]
    assert all(float(r["mass"]) > 0 for r in qoi)
    assert list(qoi[0]) == sa.QOI_COLUMNS
    assert all(float(r["pre_mass"]) > 0 and float(r["pre_mass"]) != float(r["mass"]) for r in qoi)
    assert sum(float(r["pre_V_core"]) > 0 for r in qoi) > 16
    for prefix in ("mid_crt_", "end_crt_"):
        assert all(float(r[f"{prefix}mass"]) > 0 and float(r[f"{prefix}mass"]) != float(r["mass"]) for r in qoi)
        assert all(np.isfinite(float(r[f"{prefix}centroid_x_mm"])) for r in qoi)
    assert all(np.isfinite(float(r["centroid_z_mm"])) for r in qoi)
    status_by_run = {s["run_name"]: s for s in status}  # run_status.csv is in completion order
    assert all(
        r["growth_dt"] == status_by_run[r["run_name"]]["growth_dt"] and r["treated_dt"] == status_by_run[r["run_name"]]["treated_dt"]
        for r in qoi
    )
    # The derived volumes are carried over; most runs grow past the
    # threshold by surgery, and a dosed region always contains its cavity.
    assert all(r["cavity_volume_mm3"] != "" and r["dose_volume_mm3"] != "" for r in qoi)
    assert sum(float(r["cavity_volume_mm3"]) > 0 for r in qoi) > 16
    assert all(float(r["dose_volume_mm3"]) >= float(r["cavity_volume_mm3"]) for r in qoi)
    # The tumor core is resected, so runs may stay below the thresholds
    # afterwards (empty compartments are valid outcomes, r95 = 0).
    assert sum(float(r["V_edema"]) > 0 for r in qoi) > 16
    assert all(float(r["r95_core"]) == 0 for r in qoi if r["n_core"] == "0")
    summary = json.loads((sweep_dir / "qoi_summary.json").read_text())
    assert summary["n_success"] == 32 and summary["n_nan_mass_weighted"] == 0
    assert summary["n_runs_without_pre_resection_field"] == 0 and summary["mean_run_wall_time_s"] > 0
    assert summary["n_runs_without_mid_crt_field"] == 0 and summary["n_runs_without_end_crt_field"] == 0
    assert summary["n_runs_extinct"] == 0 and spec["chemo_total_dose"] == 42 * 75.0
    assert spec["schedule"]["n_chemo_sessions_dropped"] == 5 and spec["schedule"]["horizon"] == 72.0
    assert summary["n_runs_empty_cavity"] == sum(float(r["cavity_volume_mm3"]) == 0 for r in qoi) < 16
    assert set(summary["per_qoi"]) == set(sa.ANALYSED_QOIS) and summary["block_size"] == 8
    results = sa.analyze_sweep(sweep_dir, n_bootstrap=20, seed=0)
    assert set(results) == set(sa.ANALYSED_QOIS)
    sobol = _read_csv(sweep_dir / "sobol.csv")
    assert len(sobol) == 6 * len(sa.ANALYSED_QOIS)
    assert all(np.isfinite(float(r["ST"])) and r["n_blocks_used"] == "4" for r in sobol)
    assert {r["factor"] for r in sobol} == set(spec["factor_names"])
    sobol_summary = json.loads((sweep_dir / "sobol_summary.json").read_text())
    assert sobol_summary["N"] == 4 and set(sobol_summary["qois"]) == set(sa.ANALYSED_QOIS)
    assert sobol_summary["qois"]["log10_mass"]["dropped_blocks"] == []
    _check_accounting_agreement(summary, sobol_summary, qoi)
    figures = sweep_dir / "figures"
    for stem in ("heatmap_ST", "heatmap_S1", "bars_log10_mass", "scatter_R_g", "bars_mid_crt_centroid_x_mm", "scatter_end_crt_log10_mass"):
        assert (figures / f"{stem}.png").is_file() and (figures / f"{stem}.pdf").is_file()
