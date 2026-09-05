"""Config, time-step and result-storage tests of fisher_kpp_jax,
self-contained.

Short f64 solves on the 24^3 tissue phantom checking (a) the default
configs shipped in fisher_kpp_jax/configs/, (b) the ways to construct a
solver (one mapping, keyword arguments, a config naming the solver), (c)
the config a solver records and the JSON round trip (read_config /
write_config / SolverClass(config)), (d) the three time-step forms and the
raise to the stability estimate, (e) volumes given as NIfTI paths (voxel
size and affine from the header, the in-memory marker) and (f)
Result.save / solve(store_result=True).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from loguru import logger

from fisher_kpp_jax import (
    SOLVER_KEY,
    VOLUME_IN_MEMORY,
    AnisotropicFKPPSolver,
    FKPPSolver,
    StuppFKPPSolver,
    TwoCompartmentWithNutrientFKPPSolver,
    read_config,
    solver_class,
    write_config,
)
from fisher_kpp_jax.base import DEFAULT_CONFIG_DIR, n_steps_from_dt
from fisher_kpp_jax.config import jsonable

SOLVERS = [
    FKPPSolver,
    TwoCompartmentWithNutrientFKPPSolver,
    AnisotropicFKPPSolver,
    StuppFKPPSolver,
]

_COMMON = dict(
    gaussian_seed_x_fraction=0.5,
    gaussian_seed_y_fraction=0.5,
    gaussian_seed_z_fraction=0.5,
    resolution_factor=0.6,
    stopping_time=10,
    precision="f64",
)

# A non-trivial voxel-to-world affine with 1 mm voxels.
AFFINE = np.array(
    [[-1.0, 0.0, 0.0, 90.0], [0.0, 1.0, 0.0, -126.0], [0.0, 0.0, 1.0, -72.0], [0.0, 0.0, 0.0, 1.0]]
)


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


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray = AFFINE) -> str:
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))
    return str(path)


@contextmanager
def captured_warnings():
    """Collect loguru WARNING messages emitted inside the block."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


@pytest.fixture
def phantom_paths(tmp_path: Path, tissue_phantom) -> dict[str, str]:
    """The tissue phantom written as NIfTIs (float32, AFFINE)."""
    gm, wm = tissue_phantom
    return {
        "gray_matter_pbmap": write_nifti(tmp_path / "gm.nii.gz", gm),
        "white_matter_pbmap": write_nifti(tmp_path / "wm.nii.gz", wm),
    }


# --- default configs ---


@pytest.mark.parametrize("cls", SOLVERS, ids=[cls.__name__ for cls in SOLVERS])
def test_default_config_file(cls):
    """Every parameter is defined, the optional entries equal the class
    defaults, the volume paths are absolute and point to existing files
    (or are null), and the class is registered under its name."""
    config = cls.get_default_config()
    assert config[SOLVER_KEY] == cls.__name__
    assert set(config) - {SOLVER_KEY} == cls.config_keys()
    for key, default in cls._DEFAULTS.items():
        assert jsonable(config[key]) == jsonable(default), key
    for key in cls._VOLUME_KEYS:
        value = config[key]
        if value is not None:
            assert Path(value).is_absolute() and Path(value).is_file(), key
    raw = json.loads((DEFAULT_CONFIG_DIR / f"{cls.__name__}.json").read_text())
    assert raw[SOLVER_KEY] == cls.__name__
    assert solver_class(cls.__name__) is cls


@pytest.mark.parametrize(
    "cls",
    [FKPPSolver, TwoCompartmentWithNutrientFKPPSolver],
    ids=["FKPPSolver", "TwoCompartmentWithNutrientFKPPSolver"],
)
def test_default_config_runs(cls):
    """The default configs whose volumes ship with the repository (the
    reference_solves tissue maps) construct and solve; the grid and
    horizon are reduced here to keep the test short. The voxel size comes
    from the NIfTI header and is reported as derived."""
    config = cls.get_default_config()
    solver = cls({**config, "resolution_factor": 0.2, "stopping_time": 1.0})
    assert solver.params["voxel_size_mm"] == (1.0, 1.0, 1.0)
    assert solver.config["voxel_size_mm"] is None
    assert not np.allclose(solver.affine, np.eye(4))  # the maps' header affine
    result = solver.solve()
    assert result.success, result.error
    assert result.derived == {"voxel_size_mm": (1.0, 1.0, 1.0)}


@pytest.mark.parametrize(
    "cls,key",
    [(AnisotropicFKPPSolver, "diffusion_tensors"), (StuppFKPPSolver, "resection_cavity")],
    ids=["AnisotropicFKPPSolver", "StuppFKPPSolver"],
)
def test_blank_default_config_fails_at_construction(cls, key):
    """A default config whose example volume is not available yet holds
    null there; constructing from it fails naming the parameter."""
    config = cls.get_default_config()
    assert config[key] is None
    with pytest.raises(ValueError, match=key):
        cls(config)


# --- construction and the recorded config ---


def test_solver_construction_forms(tissue_phantom):
    """One mapping, keyword arguments and a config with the "solver" entry
    build the same solver. The caller chooses the class: a config without
    the entry constructs fine, and the entry, if present, is only checked
    against the class (any mismatch raises, an unregistered name
    included)."""
    gm, wm = tissue_phantom
    params = fk_params(gm, wm)
    assert SOLVER_KEY not in params
    reference = FKPPSolver(params)
    assert reference.config[SOLVER_KEY] == "FKPPSolver"
    named = {**params, SOLVER_KEY: "FKPPSolver"}
    for solver in (FKPPSolver(**params), FKPPSolver(named), FKPPSolver(**named)):
        assert solver.config == reference.config
        assert solver.params.keys() == reference.params.keys()
    with pytest.raises(TypeError, match="not both"):
        FKPPSolver(params, rho=0.1)
    with pytest.raises(ValueError, match="names solver 'FKPPSolver'"):
        TwoCompartmentWithNutrientFKPPSolver(named)
    with pytest.raises(ValueError, match="names solver 'FKPPSolver'"):
        TwoCompartmentWithNutrientFKPPSolver(**named)
    with pytest.raises(ValueError, match="names solver 'Nope'"):
        FKPPSolver({**params, SOLVER_KEY: "Nope"})


def test_config_records_given_values(tissue_phantom):
    """The config holds the "solver" entry, the given values as given, the
    defaults for the rest and the in-memory marker for arrays; derived
    values stay out of it and are reported on the Result."""
    gm, wm = tissue_phantom
    solver = FKPPSolver(fk_params(gm, wm, stopping_mode="volume", snapshot_times=[5.0, 2.0]))
    config = solver.config
    assert next(iter(config)) == SOLVER_KEY
    assert set(config) == FKPPSolver.config_keys() | {SOLVER_KEY}
    assert config["white_matter_pbmap"] == VOLUME_IN_MEMORY
    assert config["snapshot_times"] == [5.0, 2.0]  # as given, not the sorted array
    assert config["volume_threshold"] is None and config["voxel_size_mm"] is None
    assert config["diffusivity_ratio"] == 10.0 and config["n_steps"] is None
    assert solver.params["volume_threshold"] == 0.5
    assert solver.result is None and solver.n_steps is None
    result = solver.solve()
    assert result.success, result.error
    assert result.config == config and result.config is not config
    assert result.derived == {"volume_threshold": 0.5, "voxel_size_mm": (1.0, 1.0, 1.0)}
    assert result.n_steps == solver.n_steps and result.dt == solver.dt
    assert result.wall_time_s > 0 and solver.result is result
    np.testing.assert_array_equal(result.affine, np.eye(4))
    # An in-memory volume cannot be reloaded from the config.
    with pytest.raises(ValueError, match="in-memory"):
        FKPPSolver(**config)


# --- time step ---


def test_time_step_forms(tissue_phantom):
    """n_steps, dt and steps_per_day are three spellings of one step count
    and give identical solves; at most one may be set."""
    gm, wm = tissue_phantom
    solvers = [
        FKPPSolver(fk_params(gm, wm, n_steps=200)),
        FKPPSolver(fk_params(gm, wm, dt=0.05)),
        FKPPSolver(fk_params(gm, wm, steps_per_day=20)),
    ]
    for solver in solvers:
        assert solver.resolve_time_stepping() == (200, 0.05)
        assert (solver.n_steps, solver.dt) == (200, 0.05)
    results = [solver.solve() for solver in solvers]
    for result in results[1:]:
        assert result.n_steps == 200 and result.dt == 0.05
        np.testing.assert_array_equal(
            result.final_state["cell_density"], results[0].final_state["cell_density"]
        )
    assert solvers[2].config["steps_per_day"] == 20 and solvers[2].config["n_steps"] is None
    with pytest.raises(ValueError, match="at most one"):
        FKPPSolver(fk_params(gm, wm, dt=0.1, n_steps=5))
    with pytest.raises(ValueError, match="at most one"):
        FKPPSolver(fk_params(gm, wm, dt=0.1, steps_per_day=3))
    with pytest.raises(ValueError, match="steps_per_day"):
        FKPPSolver(fk_params(gm, wm, steps_per_day=0))
    with pytest.raises(ValueError, match="dt"):
        FKPPSolver(fk_params(gm, wm, dt=-1.0))
    with pytest.raises(ValueError, match="dt"):
        FKPPSolver(fk_params(gm, wm, dt=np.inf))


def test_coarse_time_step_is_raised(tissue_phantom):
    """A requested step coarser than the stability estimate is replaced by
    the estimate's (n_steps, dt) with a warning, whichever form gave it; a
    finer request stands."""
    gm, wm = tissue_phantom
    n_formula, dt_formula = FKPPSolver(fk_params(gm, wm)).resolve_time_stepping()
    for form in ({"n_steps": 5}, {"dt": 2.0}, {"steps_per_day": 0.5}):
        with captured_warnings() as messages:
            resolved = FKPPSolver(fk_params(gm, wm, **form)).resolve_time_stepping()
        assert resolved == (n_formula, dt_formula)
        assert any("5 steps" in m and f"estimate of {n_formula}" in m for m in messages)
    with captured_warnings() as messages:
        finer = FKPPSolver(fk_params(gm, wm, n_steps=n_formula + 1)).resolve_time_stepping()
    assert finer == (n_formula + 1, 10 / (n_formula + 1)) and not messages
    result = FKPPSolver(fk_params(gm, wm, n_steps=5)).solve()
    assert result.success and result.n_steps == n_formula and result.dt == dt_formula


def test_n_steps_from_dt():
    assert n_steps_from_dt(100.0, 0.05) == 2000
    assert n_steps_from_dt(10, 0.1) == 100  # 10/0.1 is not exactly 100 in floating point
    assert n_steps_from_dt(100.0, 0.3) == 334  # rounded up, effective dt < 0.3
    with pytest.raises(ValueError):
        n_steps_from_dt(100.0, 0.0)
    with pytest.raises(ValueError):
        n_steps_from_dt(0.0, 0.1)


# --- volumes as NIfTI paths ---


def test_volumes_from_nifti_paths(tmp_path, tissue_phantom, phantom_paths):
    """Paths (str or Path) are loaded as float64 arrays, the reference
    volume's header gives the affine and voxel size, the config keeps the
    paths, and the solve equals the one with arrays."""
    gm, wm = tissue_phantom
    by_path = FKPPSolver(
        fk_params(
            gm,
            wm,
            white_matter_pbmap=phantom_paths["white_matter_pbmap"],
            gray_matter_pbmap=Path(phantom_paths["gray_matter_pbmap"]),
        )
    )
    np.testing.assert_array_equal(by_path.params["white_matter_pbmap"], wm)
    assert by_path.params["white_matter_pbmap"].dtype == np.float64
    np.testing.assert_array_equal(by_path.affine, AFFINE)
    assert by_path.config["white_matter_pbmap"] == phantom_paths["white_matter_pbmap"]
    assert by_path.config["gray_matter_pbmap"] == phantom_paths["gray_matter_pbmap"]
    by_array = FKPPSolver(fk_params(gm, wm))
    np.testing.assert_array_equal(
        by_path.solve().final_state["cell_density"], by_array.solve().final_state["cell_density"]
    )
    # Anisotropic voxels from the header; an explicit voxel_size_mm wins
    # with a warning.
    aniso = write_nifti(tmp_path / "wm_aniso.nii.gz", wm, np.diag((1.0, 1.5, 2.0, 1.0)))
    solver = FKPPSolver(fk_params(gm, wm, white_matter_pbmap=aniso))
    assert solver.params["voxel_size_mm"] == (1.0, 1.5, 2.0)
    assert solver.config["voxel_size_mm"] is None
    with captured_warnings() as messages:
        solver = FKPPSolver(fk_params(gm, wm, white_matter_pbmap=aniso, voxel_size_mm=(1, 1, 1)))
    assert solver.params["voxel_size_mm"] == (1.0, 1.0, 1.0)
    assert solver.config["voxel_size_mm"] == (1, 1, 1)
    assert any("differs from the NIfTI header" in m for m in messages)
    with pytest.raises(ValueError, match="voxel_size_mm"):
        FKPPSolver(fk_params(gm, wm, voxel_size_mm=(1.0, 0.0, 1.0)))
    with pytest.raises(FileNotFoundError):
        FKPPSolver(fk_params(gm, wm, white_matter_pbmap=str(tmp_path / "nope.nii.gz")))


# --- JSON configs ---


def test_read_write_config(tmp_path, tissue_phantom, phantom_paths):
    """Comments are dropped, keys checked, relative paths resolved against
    the config's directory, "inf" turned into a float; a written config
    reads back equal; the in-memory marker is refused on read."""
    file = {
        SOLVER_KEY: "FKPPSolver",
        "_note": "ignored",
        "white_matter_pbmap": "wm.nii.gz",
        "gray_matter_pbmap": phantom_paths["gray_matter_pbmap"],
        "white_matter_diffusivity": 0.3,
        "rho": 0.15,
        **{key: value for key, value in _COMMON.items()},
        "stopping_threshold": "inf",
        "voxel_size_mm": [1, 1, 1],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(file))
    config = read_config(path)
    expected = {key: value for key, value in file.items() if key != "_note"}
    expected["white_matter_pbmap"] = phantom_paths["white_matter_pbmap"]
    expected["stopping_threshold"] = np.inf
    assert config == expected
    assert read_config(path, solver=FKPPSolver) == config
    assert read_config(path, solver="FKPPSolver") == config
    # The solver argument stands in for a missing "solver" entry.
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({key: value for key, value in file.items() if key != SOLVER_KEY}))
    assert read_config(bare, solver=FKPPSolver) == config
    with pytest.raises(ValueError, match="no 'solver' entry"):
        read_config(bare)
    with pytest.raises(ValueError, match="names solver 'FKPPSolver', not"):
        read_config(path, solver=StuppFKPPSolver)
    with pytest.raises(FileNotFoundError, match="config not found"):
        read_config(tmp_path / "missing.json")
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        read_config(tmp_path / "list.json")
    (tmp_path / "unknown.json").write_text(json.dumps({**file, "D": 2, "chemotherapy": {}}))
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['D', 'chemotherapy'\]"):
        read_config(tmp_path / "unknown.json")
    (tmp_path / "bad_volume.json").write_text(json.dumps({**file, "white_matter_pbmap": 60.0}))
    with pytest.raises(ValueError, match="white_matter_pbmap must be a NIfTI path"):
        read_config(tmp_path / "bad_volume.json")
    # Round trip through a solver: every parameter, defaults filled in.
    solver = FKPPSolver(config)
    written = write_config(solver.config, tmp_path / "written.json")
    assert read_config(written) == solver.config
    assert set(solver.config) == FKPPSolver.config_keys() | {SOLVER_KEY}
    assert json.loads(written.read_text())["stopping_threshold"] == "inf"
    write_config({**config, "white_matter_pbmap": VOLUME_IN_MEMORY}, tmp_path / "memory.json")
    with pytest.raises(ValueError, match="in-memory"):
        read_config(tmp_path / "memory.json")


@pytest.mark.parametrize(
    "cls",
    [FKPPSolver, TwoCompartmentWithNutrientFKPPSolver],
    ids=["FKPPSolver", "TwoCompartmentWithNutrientFKPPSolver"],
)
def test_config_round_trip_reproduces_run(tmp_path, tissue_phantom, phantom_paths, cls):
    """write_config(SolverClass(cfg).config) -> read_config -> SolverClass(...)
    reproduces a phantom run: the reloaded config equals the recorded one
    (the class defaults filled in, the volume paths absolute) and the
    solve gives the identical final field. The written file names the
    solver, so read_config needs no solver argument."""
    gm, wm = tissue_phantom
    config = fk_params(gm, wm, **phantom_paths, n_steps=200)
    if cls is TwoCompartmentWithNutrientFKPPSolver:
        config.update(
            necrosis_rate=0.05,
            nutrient_threshold=0.3,
            nutrient_diffusivity=0.5,
            nutrient_consumption_rate=0.02,
        )
    reference = cls(config)
    written = write_config(reference.config, tmp_path / "round_trip.json")
    reloaded = read_config(written)
    assert reloaded[SOLVER_KEY] == cls.__name__ and reloaded == reference.config
    solver = cls(reloaded)
    assert solver.config == reference.config
    expected = reference.solve()
    result = solver.solve()
    assert expected.success and result.success, (expected.error, result.error)
    assert result.n_steps == expected.n_steps and result.final_time == expected.final_time
    for key, volume in expected.final_state.items():
        np.testing.assert_array_equal(result.final_state[key], volume)


# --- saving results ---


def test_save_and_reload(tmp_path, tissue_phantom, phantom_paths):
    """solve(store_result=True) writes config.json, result.json and the
    state volumes; the config reloads into a solver that reproduces the
    run; the time series is written on request only; an existing result
    is not overwritten unless asked; a failed run writes the JSON files."""
    gm, wm = tissue_phantom
    config = {
        SOLVER_KEY: "FKPPSolver",
        **fk_params(gm, wm, **phantom_paths, snapshot_times=[2.5, 5.0, 10.0]),
    }
    solver = FKPPSolver(**config)
    out = tmp_path / "run"
    result = solver.solve(store_result=True, outdir=out)
    assert result.success, result.error
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json",
        "final_cell_density.nii.gz",
        "initial_cell_density.nii.gz",
        "result.json",
    ]
    record = json.loads((out / "result.json").read_text())
    assert record[SOLVER_KEY] == "FKPPSolver" and record["success"] is True
    assert record["error"] is None and record["stopping_criterion"] == "time"
    assert record["n_steps"] == result.n_steps and record["dt"] == result.dt
    assert record["final_time"] == 10.0 and record["wall_time_s"] == result.wall_time_s
    assert record["grid_shape"] == list(gm.shape)
    assert record["snapshot_times"] == list(result.snapshot_times)
    assert record["derived"] == {"voxel_size_mm": [1.0, 1.0, 1.0]}
    assert set(record["files"]) == {p.name for p in out.iterdir()}
    image = nib.load(str(out / "final_cell_density.nii.gz"))
    np.testing.assert_array_equal(image.affine, AFFINE)
    np.testing.assert_allclose(
        image.get_fdata(), result.final_state["cell_density"], rtol=0, atol=1e-6
    )
    reloaded = FKPPSolver(read_config(out / "config.json"))
    assert reloaded.config == solver.config
    np.testing.assert_array_equal(
        reloaded.solve().final_state["cell_density"], result.final_state["cell_density"]
    )
    # The time series on request; overwriting on request.
    with pytest.raises(FileExistsError):
        solver.save(out)
    solver.save(out, time_series=True, overwrite=True)
    frames = nib.load(str(out / "time_series_cell_density.nii.gz")).get_fdata()
    assert frames.shape == (*gm.shape, 3)
    np.testing.assert_allclose(
        np.moveaxis(frames, -1, 0), result.time_series["cell_density"], rtol=0, atol=1e-6
    )
    assert "time_series_cell_density.nii.gz" in json.loads((out / "result.json").read_text())["files"]
    with pytest.raises(RuntimeError, match="call solve"):
        FKPPSolver(**config).save(tmp_path / "unsolved")
    with pytest.raises(ValueError, match="outdir"):
        solver.solve(store_result=True)
    # A failed run (seed outside the tissue) writes the JSON files only.
    failing = FKPPSolver(
        **{**config, **{f"gaussian_seed_{axis}_fraction": 0.02 for axis in "xyz"}}
    )
    failed = failing.solve(store_result=True, outdir=tmp_path / "failed")
    assert not failed.success and failed.n_steps is None
    assert sorted(p.name for p in (tmp_path / "failed").iterdir()) == ["config.json", "result.json"]
    record = json.loads((tmp_path / "failed" / "result.json").read_text())
    assert record["success"] is False and "outside the brain matter" in record["error"]
    assert record["n_steps"] is None and record["grid_shape"] is None
