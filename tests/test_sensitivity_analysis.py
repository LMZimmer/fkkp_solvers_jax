"""Tests of scripts/sensitivity_analysis.py (loaded with importlib from
scripts/), self-contained.

(1) The Ishigami function through the script's own design and analysis
functions against its analytic Sobol' indices (the block/row bookkeeping
end to end), (2) SALib's Saltelli row order as the script assumes it,
verified against the installed version, (3) the factor transforms and the
search-space checks, (4) the design bookkeeping on the 24^3 phantom with
the shipped search space (every AB run differs from its block's A run only
through its factor, seeds lie on seedable tissue voxels, every config
constructs a StuppFKPPSolver once the derived maps are given) and the
nested-range seed mapping, (5) the growth stage config, the treatment
maps and one two-stage run on the phantom, (6) the QoIs on synthetic
Gaussian fields against closed forms and (7) the whole pipeline on the
phantom on the CPU (design, run, qoi, analyze) with a small search space.
"""

from __future__ import annotations

import csv
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def phantom_base(tmp_path: Path, tissue_phantom) -> dict:
    """A phantom base config on disk: WM/GM maps plus the config JSON (a
    short treated run with resection_cavity and rt_dose null, as the
    script derives them; 12 days of growth take the seed's density past
    the 0.6 cavity threshold)."""
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
        "resolution_factor": 1.0,
        "precision": "f32",
        "steps_per_day": 110,
        "min_tissue_fraction": MIN_TISSUE_FRACTION,
        "resection_time": 12.0,
        "time_after_resection": 30.0,
        "resection_cavity": None,
        "chemo_times": [14.0, 16.0],
        "chemo_doses": [75.0, 150.0],
        "chemo_kill_rate": 0.1 / 75,
        "chemo_decay_rate": 0.5,
        "rt_times": [14.0, 16.0],
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
    """The shipped file loads with 13 factors in file order and no
    override; a mapping loads too; unknown keys, a derived volume, a bad
    scale, a mismatching solver, seed entries off [0, 1] or missing each
    raise."""
    space = sa.load_search_space(SHIPPED_SEARCH_SPACE, CONFIG_KEYS)
    assert len(space.names) == 13 and space.overrides == {}
    assert space.names[:3] == ["rho", "white_matter_diffusivity", "diffusivity_ratio"]
    assert space.names[-3:] == list(SEED_KEYS)
    assert space.factors["rho"] == sa.SearchSpaceParameter("rho", 0.0089228, 0.3449, "log")
    assert space.factors["rt_alpha_beta_ratio"].scale == "linear"
    assert "_note" in space.source and "_note" not in space.factors
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


# --- (4) design bookkeeping ---


def _config_entries(path: Path) -> dict:
    return {key: value for key, value in json.loads(path.read_text()).items() if not key.startswith("_")}


def test_design_bookkeeping(phantom_base):
    """The shipped search space on the phantom base config: N (k + 2) runs
    with one A, one B and one AB per factor in every block; every AB
    config differs from its block's A config only through its factor (the
    seed fractions for a seed factor, the shifted event times for
    resection_time); every seed is a seedable tissue voxel whose fractions
    the config holds; the derived volumes stay null in the run configs
    and every config constructs a solver once they are given; a base
    config that sets them, or lacks the tissue maps, is refused; the
    design is never overwritten."""
    tmp_path = phantom_base["tmp_path"]
    sweep_dir = sa.make_design(SHIPPED_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "design", log2_n=2, seed=3)
    assert sweep_dir == tmp_path / "sa" / "design"
    with pytest.raises(FileExistsError, match="never overwritten"):
        sa.make_design(SHIPPED_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "design", log2_n=2, seed=3)
    spec = json.loads((sweep_dir / "spec.json").read_text())
    names = spec["factor_names"]
    k, n_blocks = spec["k"], spec["N"]
    assert (k, n_blocks, spec["n_runs"], spec["block_size"]) == (13, 4, 60, 15)
    assert spec["salib_version"] and spec["seed"] == 3 and not spec["second_order"]
    assert spec["growth_solver"] == "FKPPSolver"
    assert spec["treatment"] == {
        "cavity_threshold": 0.6, "rt_margin_mm": 15.0, "rt_dose_per_fraction_gy": 2.0,
        "n_fractions": 2, "rt_total_dose_gy": 4.0,
    }
    assert (sweep_dir / "search_space.json").read_text() == SHIPPED_SEARCH_SPACE.read_text()
    base = read_config(sweep_dir / "base_config.json")
    assert Path(base["white_matter_pbmap"]).is_absolute()
    assert base["resection_cavity"] is None and base["rt_dose"] is None
    design = _read_csv(sweep_dir / "design.csv")
    assert len(design) == n_blocks * (k + 2)
    n_seed_changes = 0
    assert [int(r["index"]) for r in design] == list(range(len(design)))
    assert sorted({r["run_name"] for r in design}) == sorted(p.stem for p in (sweep_dir / "configs").iterdir())
    for j in range(n_blocks):
        block = [r for r in design if int(r["row"]) == j]
        assert [r["matrix"] for r in block] == ["A", *(f"AB:{name}" for name in names), "B"]
        a = _config_entries(sweep_dir / "configs" / f"{block[0]['run_name']}.json")
        assert a["resection_cavity"] is None and a["rt_dose"] is None
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
            else:
                assert differing == {factor}
            assert ab[factor] == float(record[factor])
    assert n_seed_changes > 0
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
    # The chemotherapy budget of the shipped ranges on the phantom schedule.
    assert spec["chemo_total_dose"] == 225.0
    np.testing.assert_allclose(spec["chemo_log_kill_range"], [1.1e-3 * 225 / 20, 5.6e-3 * 225 / 5])
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
    assert not (tmp_path / "sa" / "refused").exists()


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
    assert sa.build_parser().parse_args(["all", "--name", "x", "--gray-matter-pbmap", ""]).gray_matter_pbmap == ""


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
    drops the treatment entries and stops at resection_time."""
    base = read_config(phantom_base["path"], solver=StuppFKPPSolver)
    growth = sa.growth_config({**base, "_design": "comment", "resection_time": 7.5})
    assert growth[SOLVER_KEY] == "FKPPSolver" and growth["stopping_time"] == 7.5
    assert set(growth) - {SOLVER_KEY} <= FKPPSolver.config_keys()
    assert not set(growth) & StuppFKPPSolver.TREATMENT_KEYS and "time_after_resection" not in growth
    for key in ("white_matter_pbmap", "rho", "white_matter_diffusivity", "steps_per_day", "min_tissue_fraction", *SEED_KEYS):
        assert growth[key] == base[key]
    solver = FKPPSolver(growth)
    assert solver.params["stopping_time"] == 7.5


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


def test_run_one_two_stages(phantom_base):
    """One design point on the phantom (CPU): the growth stage is saved
    into growth/, the cavity is the growth density at or above the
    threshold, the dose map holds the total dose within the margin, the
    treated stage's config.json references both maps and reproduces the
    run through read_config, both stages start from the same seed, and
    the treated stage's final density vanishes inside the cavity."""
    tmp_path = phantom_base["tmp_path"]
    sweep_dir = sa.make_design(
        sa.DEFAULT_SEARCH_SPACE, phantom_base["path"], tmp_path / "sa", "one", log2_n=1, seed=2, rt_margin_mm=3.0
    )
    config_path = sweep_dir / "configs" / "r0000_A.json"
    run_dir = sweep_dir / "runs" / "r0000_A"
    treatment = sa.spec_treatment_settings(json.loads((sweep_dir / "spec.json").read_text()))
    assert treatment["rt_margin_mm"] == 3.0
    assert sa.run_one(config_path, run_dir, treatment["cavity_threshold"], 3.0, treatment["rt_dose_per_fraction_gy"]) == 0
    growth = json.loads((run_dir / "growth" / "result.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    config = _config_entries(config_path)
    assert growth["success"] and result["success"]
    assert read_config(run_dir / "growth" / "config.json")[SOLVER_KEY] == "FKPPSolver"
    assert growth["final_time"] == config["resection_time"]
    # The treated stage is stepped on the growth stage's grid of times.
    n_after = int(np.ceil(config["time_after_resection"] / growth["dt"] - 1e-9))
    assert result["n_steps"] == growth["n_steps"] + n_after
    assert result["dt"] == pytest.approx(growth["dt"], rel=1e-12)
    horizon = config["resection_time"] + config["time_after_resection"]
    assert horizon - 1e-9 <= result["final_time"] <= horizon + growth["dt"]
    assert result["final_time"] == pytest.approx(result["n_steps"] * growth["dt"], rel=1e-12)
    growth_density = np.asarray(nib.load(str(run_dir / "growth" / "final_cell_density.nii.gz")).get_fdata())
    cavity = np.asarray(nib.load(str(run_dir / "resection_cavity.nii.gz")).get_fdata())
    dose = np.asarray(nib.load(str(run_dir / "rt_dose.nii.gz")).get_fdata())
    record = json.loads((run_dir / "treatment.json").read_text())
    assert set(np.unique(cavity)) <= {0.0, 1.0} and cavity.sum() == record["n_cavity_voxels"] > 0
    np.testing.assert_array_equal(cavity.astype(bool), growth_density >= 0.6)
    expected_cavity, expected_dose = sa.treatment_maps(growth_density, (1.0, 1.0, 1.0), 0.6, 3.0, 4.0)
    np.testing.assert_array_equal(cavity.astype(bool), expected_cavity)
    np.testing.assert_array_equal(dose, expected_dose)
    assert record["rt_total_dose_gy"] == 4.0 and record["n_fractions"] == 2
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
    initial = np.asarray(nib.load(str(run_dir / "initial_cell_density.nii.gz")).get_fdata())
    growth_initial = np.asarray(nib.load(str(run_dir / "growth" / "initial_cell_density.nii.gz")).get_fdata())
    np.testing.assert_array_equal(initial, growth_initial)
    final = np.asarray(nib.load(str(run_dir / "final_cell_density.nii.gz")).get_fdata())
    assert final.max() > 0 and np.all(final[expected_cavity] == 0)
    # The run-status record of a run that failed in its growth stage.
    failed_dir = sweep_dir / "runs" / "failed"
    (failed_dir / "growth").mkdir(parents=True)
    (failed_dir / "growth" / "result.json").write_text(json.dumps({"success": False, "error": "boom"}))
    (sweep_dir / "configs" / "failed.json").write_text("not json")
    failed = sa.run_subprocess(sweep_dir, "failed", None, treatment)
    assert not failed["success"] and failed["error"] == "growth stage: boom" and failed["exit_code"] != 0


# --- (6) QoIs on synthetic fields ---
def test_chemo_log_kill_range():
    """The shipped ranges with the example config's 8 900 mg/m^2 schedule
    bound the total log kill kill * D_tot / decay to [0.5, 10] within 5 %;
    an override or a fixed value stands in for a factor range."""
    example = read_config(SCRIPT.parent / "stupp_config_example.json", solver=StuppFKPPSolver)
    space = sa.load_search_space(SHIPPED_SEARCH_SPACE, CONFIG_KEYS)
    total_dose, (low, high) = sa.chemo_log_kill_range(example, space)
    assert total_dose == 8900.0
    np.testing.assert_allclose([low, high], [0.5, 10.0], rtol=0.05)
    kill, decay = space.factors["chemo_kill_rate"], space.factors["chemo_decay_rate"]
    assert (low, high) == (kill.low * 8900 / decay.high, kill.high * 8900 / decay.low)
    fixed = sa.load_search_space(
        _seed_entries(
            chemo_kill_rate=2e-3,
            chemo_doses=[100.0, 100.0],
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
    centroid_drift = the seed offset, wm_fraction 1 on an all-WM map,
    V_tau and r95 from the analytic ball radius sigma sqrt(2 ln(1 / tau));
    stds (2 sigma, sigma, sigma): anisotropy 4; on anisotropic voxels the
    mm geometry still holds; an empty compartment gives r95 = 0; a zero
    field gives NaN mass-weighted QoIs."""
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
    summary = sa.qoi_summary(records, 0.6, 0.3, n_blocks=1, size=3)
    assert summary["n_runs_extinct"] == 2 and summary["n_nan_mass_weighted"] == 2
    assert summary["per_qoi"]["log10_mass"]["n_blocks_dropped"] == 0
    assert summary["per_qoi"]["R_g"] == {
        "n_runs_total": 3, "n_runs_failed": 0, "n_runs_nan": 2, "n_blocks_total": 1,
        "n_blocks_dropped": 1, "n_blocks_used": 0, "dropped_blocks": [0],
    }


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
    for qoi in ("centroid_drift", "R_g", "log10_anisotropy", "wm_fraction"):
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
    search_space = tmp_path / "small_space.json"
    search_space.write_text(
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
    sweep_dir = sa.make_design(
        search_space, phantom_base["path"], tmp_path / "sa", "e2e", log2_n=2, seed=1, rt_margin_mm=3.0
    )
    spec = json.loads((sweep_dir / "spec.json").read_text())
    assert spec["k"] == 6 and spec["n_runs"] == 32 and spec["overrides"] == {"gaussian_seed_mass": 250.0}
    assert spec["treatment"]["rt_margin_mm"] == 3.0 and spec["treatment"]["rt_total_dose_gy"] == 4.0
    counts = sa.run_sweep(sweep_dir, gpus=[], jobs_per_gpu=8)  # 8 CPU workers
    assert counts == {"skipped": 0, "ok": 32, "failed": 0}
    status = _read_csv(sweep_dir / "run_status.csv")
    assert len(status) == 32 and all(r["success"] == "True" and r["exit_code"] == "0" for r in status)
    for record in status:
        run_dir = sweep_dir / "runs" / record["run_name"]
        assert (run_dir / "final_cell_density.nii.gz").is_file() and (run_dir / "config.json").is_file()
        assert (run_dir / "growth" / "result.json").is_file() and (run_dir / "treatment.json").is_file()
        assert (run_dir / "resection_cavity.nii.gz").is_file() and (run_dir / "rt_dose.nii.gz").is_file()
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
    assert summary["n_runs_extinct"] == 0 and spec["chemo_total_dose"] == 225.0
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
    for stem in ("heatmap_ST", "heatmap_S1", "bars_log10_mass", "scatter_R_g"):
        assert (figures / f"{stem}.png").is_file() and (figures / f"{stem}.pdf").is_file()
