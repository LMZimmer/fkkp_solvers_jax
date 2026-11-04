# Multi-seed extension of the solvers (2026-11-04)

Handoff record of the first step of the multi-seed work: the solver
package now seeds from a list of seed centres. Nothing is committed; the
working tree holds the changes (`git status`). Nothing has been run: the
machine the changes were written on has no JAX, so every file was only
byte-compiled (`python -m py_compile`). The test and check commands at
the end are what the remote should run first.

## Design decisions (agreed with the author, 2026-11-04)

1. **No new solver class.** Seeding is model-independent and lives in
   `BaseFKPPSolver`; all three solvers (`FKPPSolver`,
   `TwoCompartmentWithNutrientFKPPSolver`, `AnisotropicFKPPSolver`) get
   the multi-seed initial state through the shared hook, as the
   treatments were added before.
2. **Config form, no backward compatibility.** The three required
   parameters `gaussian_seed_x_fraction`, `gaussian_seed_y_fraction`,
   `gaussian_seed_z_fraction` are gone. The one required seed parameter
   is `gaussian_seed_positions`: a list of `[x, y, z]` grid fractions in
   `[0, 1]`, shape `(n, 3)`, `n >= 1`. A single-seed run is one row,
   e.g. `[[0.5, 0.5, 0.5]]`. The seed voxel of a row is
   `int(fraction * n_axis)` on the low-resolution grid, the old rule.
3. **Combination: sum, then floor, then clip.** The initial field is the
   sum of one Gaussian kernel per row, then values at or below
   `gaussian_seed_floor` are zeroed and values above 1 clipped, then the
   existing seed cut (`_seed_mask_host`, zero outside the tissue) is
   applied. With one row this is the previous single clipped Gaussian.
4. **Per-seed width and mass, as lists.** `gaussian_seed_diffusion_time`
   and `gaussian_seed_mass` are per seed: a list with one value per row
   of `gaussian_seed_positions`, or a scalar / one-entry list that
   applies to every seed. The class defaults are one-entry lists
   (`[15.0]`, `[1500.0]`), so the default config files hold lists. The
   total initial mass grows with the number of seeds; overlapping seeds
   add up and saturate at 1.
5. **Two-compartment solver: proliferative only.** Every seed goes into
   the proliferative field; necrotic starts at zero as before. Seeding
   the necrotic field from label-1 centres is a later, separate decision.
6. **The solver takes positions, not a segmentation.** Which voxels are
   seedable (labels 1 and 3 of the pre-op segmentation, with tissue) stays
   a script decision; the scripts write the positions into the config,
   so a saved config reproduces its run on its own.

## Package changes (`fisher_kpp_jax/`)

- `operators.py`: `clipped_gaussian(grid_shape, centers, grid_spacing,
  scale, dtype, diffusion_times, masses, floor)` takes `(n, 3)` centres
  and per-centre widths and masses (scalars broadcast). The kernels are
  summed by a jitted `lax.scan` (`_gaussian_sum`, memory of one grid
  regardless of `n`), then floored and clipped once. The accumulator
  starts at zero, so one centre gives its kernel exactly; the amplitude
  is still computed on the host in float64 and cast to the state dtype.
- `base.py`: `SEED_POSITIONS_KEY = "gaussian_seed_positions"` and
  `SEED_PER_SEED_KEYS` replace `GAUSSIAN_SEED_POSITION_FRACTION`.
  `_validate_seed` (replaces `_validate_unit_interval`) checks the
  positions and broadcasts the per-seed entries; after construction
  `params["gaussian_seed_positions"]` is a float64 `(n, 3)` array and
  `params["gaussian_seed_diffusion_time"]` / `params["gaussian_seed_mass"]`
  are float64 `(n,)` arrays (the config keeps the entries as given).
  `_setup_grid` sets `seed_voxels` (int64 `(n, 3)`) and warns when two
  rows land on one low-resolution voxel (their kernels then add up; no
  deduplication). `seed_voxel` is now a property returning the first
  row, kept for `util.render`, the Stupp scripts and two tests.
  `_gaussian_seed` passes the arrays through; the seed cut is unchanged.
  Class docstring documents the seed.
- `solvers.py`: `_REQUIRED` of the three classes names
  `SEED_POSITIONS_KEY`; `_COMMON_DEFAULTS` holds the per-seed defaults
  as one-entry tuples (written as lists by `_build_config`); the seed
  checks of `FKPPSolver` and `AnisotropicFKPPSolver` test every row and
  name the offending voxels (message still starts with "Initial tumor
  position is outside the brain matter" / "... brain mask").
- `configs/*.json` (all five) and `search_spaces/stupp_config_example.json`:
  `gaussian_seed_positions: [[0.5, 0.5, 0.5]]` in place of the three
  fraction entries; the five class configs hold the width and mass as
  one-entry lists.
- `__init__.py`: module docstring names the seed form.

## Script changes (boundary conversions only)

Every place that wrote the three fraction keys into a run config now
writes `gaussian_seed_positions = [[x, y, z]]`; the scripts' internal
seed handling (one voxel, `seed_voxel_i/j/k` columns, the `SeedMap`) is
unchanged.

- `scripts/sensitivity_analysis.py`: the three seed factors `SEED_KEYS`
  stay search-space factors (linear on [0, 1], projected onto the
  seedable voxels) but are no longer solver parameters, so
  `load_search_space` adds them to the known keys itself. New helpers
  `SEED_POSITIONS_KEY`, `seed_positions_entry(values)` (pops the three
  fractions of a design record, returns the one-row entry) and
  `single_seed_value(value, key)` (the scalar of a per-seed entry of a
  single-seed config). `run_config` writes the positions entry.
- `scripts/patient_sensitivity_analysis.py`: `run_config` does the same
  (imports `seed_positions_entry`).
- `scripts/identifiability_experiments.py`: the cohort config writes the
  positions entry; `Patient.from_config` reads the base config's seed
  width and mass through `single_seed_value` (the base config now holds
  one-entry lists).
- `scripts/patient_cmaes_fit.py`: `SEED_POSITIONS_KEY` replaces
  `SEED_FRACTION_KEYS` (also in `FORBIDDEN_KEYS`); `config_of` writes
  `[list(point.seed_fractions)]`. The existing single-seed fit is
  otherwise untouched; the multi-seed fit is a NEW script to be written
  (see Open items).
- `scripts/run_stupp_sweep.py`, `run_stupp_example.py`,
  `run_stupp_synthetic.py`, `run_reference_solves.py`: write the
  positions entry; `run_stupp_forward.py` reads the first row for the
  montage marker.
- Not touched: `visualize_*.py` (they read no seed entry),
  `analyze_sweep_sensitivity.py`, the `fisher_kpp/` legacy package, the
  search-space JSON files (their `gaussian_seed_{x,y,z}_fraction` factor
  entries are script factors and stay).

## Tests

Adapted fixtures only, plus two minimal checks:

- `tests/test_operators.py::test_clipped_gaussian`: new signature; a
  second centre with its own width and mass is checked against the
  NumPy sum-then-floor-then-clip.
- `tests/test_solvers.py::test_multi_seed_run`: two seeds with per-seed
  masses solve, hold more initial mass than one seed, and a per-seed
  list of the wrong length is refused. `test_seed_outside_tissue_errors`
  now matches the message prefix.
- `tests/test_config.py`, `test_stupp.py`, `test_sensitivity_analysis.py`
  (phantom base config and one direct solver construction),
  `test_identifiability_experiments.py` (two assertions read
  `config["gaussian_seed_positions"][0][0]`): fixtures converted.

## Known consequences

- **Old saved configs do not load.** Every `config.json` written before
  this change (fit directories, `results/*/base_config.json`, sweep
  configs) carries the three fraction keys, which `read_config` now
  rejects as unknown. To load one, replace the three entries by
  `"gaussian_seed_positions": [[x, y, z]]` (same numbers, same order).
  No migration helper was added on purpose; add one to `config.py` if
  the `resolve` / visualize workflows need old fit directories.
- Scripts that print or parse `gaussian_seed_mass` /
  `gaussian_seed_diffusion_time` from a config must expect a list; the
  scripts' own derivations still write scalars, which the solver
  broadcasts.
- `Result.initial_state` of a multi-seed run is the summed field; the
  mass stopping quantity starts from the sum.

## What to run on the remote (none of it has been run)

```
JAX_PLATFORMS=cpu pytest tests -x -q            # all suites; start here
JAX_PLATFORMS=cpu pytest tests/test_operators.py tests/test_solvers.py tests/test_config.py -q
python scripts/run_reference_solves.py          # reference results must still match
python scripts/patient_cmaes_fit_tum149.py fit --name smoke_multiseed_contract --gpus 0 --dry-run
```

Things to look at if something fails:

- `test_default_config_file`: every defaulted key must be classified; the
  per-seed keys are already in `PHYSICAL_DEFAULT_KEYS`, nothing new was
  added to `_DEFAULTS`.
- `run_reference_solves.py`: the single-seed path is meant to be
  bit-identical (zero accumulator, same operation order, same host
  amplitude), but the kernel now runs inside one jitted scan instead of
  eager ops; a last-bit difference in f32 would show up here first.
- The SA test suite exercises `load_search_space` with the seed factors;
  if a test asserts on the exact `known` key set, it will need the
  `SEED_KEYS` addition.

## Open items (next steps, not started)

1. The multi-seed CMA-ES script: a new script beside
   `patient_cmaes_fit.py` that fixes the seed positions once from the
   pre-op core (labels 1 and 3 with tissue, `patient_seed_geometry`),
   writes them into every run config and drops `seed_bbox_x/y/z` from
   the search space (seven factors instead of ten); shared or fixed
   `seed_sigma_mm` / `seed_peak_density` map to one-entry lists.
2. The placement rule for the seeds (lattice at about sigma,
   farthest-point sampling, component centroids, every voxel). Seeding
   every voxel should not use thousands of kernels; it is a Gaussian
   blur of the indicator and belongs in a separate host-side mode.
3. Per-field seeding of the two-compartment model (label 1 into the
   necrotic field).
4. A config migration helper for old fit directories, if needed.
