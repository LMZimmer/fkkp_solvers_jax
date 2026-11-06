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

## Open items (as of the first write-up; the status of each is below)

1. DONE 2026-11-04 (the last section of this file). The multi-seed
   CMA-ES fit (CORRECTED 2026-11-04; the earlier text here, fixed
   positions from the pre-op core and seven factors, was wrong). The
   CMA-ES scripts (`patient_cmaes_fit.py` and the tum149 wrapper) take a
   number of seeds n (an option, any n >= 1) and fit every seed with
   CMA-ES: per seed, the position through the existing seed map (three
   box factors per seed mapped onto the seedable voxels of the pre-op
   core as now). The run config receives `gaussian_seed_positions` with
   one row per seed. n = 1 is the existing single-seed fit. The four
   questions this item left open were settled as follows: the factor
   layout is a position per seed with ONE shared shape (`seed_sigma_mm`,
   `seed_peak_density` for every seed, so the width and mass go into the
   config as scalars, not per-seed lists); n is `--n-seeds`; the
   per-seed columns are `seed_voxel_i/j/k_<k>` and `seed_snapped_<k>`
   and the spec records `n_seeds`; the swap symmetry is removed by a
   CMA-ES repair that orders the seed blocks along x (not by the
   mapping alone, which cannot remove it: the summed kernels make the
   loss permutation-invariant whatever order the mapping uses).
2. Superseded by the correction of item 1: the seed positions are
   fitted, there is no placement rule for fixed seeds.
3. OPEN. Per-field seeding of the two-compartment model (label 1 into
   the necrotic field).
4. OPEN. A config migration helper for old fit directories, if needed.

## 2026-11-04, on the JAX machine: review, tests, the multi-seed fit

Everything above was committed as 2355504 before this step (the "nothing
is committed" line above is stale). Review and test results:

- Package code: correct, unchanged. The one-row `clipped_gaussian` is
  bit-identical to the pre-change single-seed kernel on GPU and in eager
  CPU mode (f32 and f64, three widths and masses); under jit on CPU only,
  some voxels differ by 1 ULP, from XLA's fused-loop code generation.
- One script bug: `identifiability_experiments.make_design` still read
  and wrote the three fraction keys (a KeyError, and unknown keys in every
  design's base config). Both lines now use the positions entry.
- Three stale assertions in `tests/test_sensitivity_analysis.py`
  (`test_design_bookkeeping`, `test_growth_config`) compared configs
  through the old keys; they read the positions row now.
- `JAX_PLATFORMS=cpu pytest tests -q`: 154 passed. The reference solves
  were not rerun. Both fixes are uncommitted.

Open item 1 was NOT done at this point (it was, later the same day:
the next section). A "lattice" seed mode (positions fixed on a
2 mm lattice over the pre-op core, box factors dropped, one shared width
and mass) was built into `scripts/patient_cmaes_fit.py` with a tum149
wrapper on 2026-11-04 and REVERTED the same day: it was a misreading of
the old item 1, which is corrected above (n fitted seeds).
`scripts/patient_cmaes_fit.py` was unchanged from 2355504 at this point.

Open items 3 (necrotic-field seeding) and 4 (the config migration
helper) stand.

## 2026-11-04, later: the multi-seed CMA-ES fit (open item 1 DONE, uncommitted)

Decisions taken with the author on 2026-11-04, after the first version
of this section's plan was read back: any number of seeds (no cap; "2-3"
was the expected use, not a limit), ONE shared seed shape for every
seed, and n = 1 must reproduce the single-seed fit. Implemented in
`scripts/patient_cmaes_fit.py`; `scripts/patient_cmaes_fit_tum149.py`
passes every option through unchanged and only its docstring was
updated; the `_seed_position` note of
`fisher_kpp_jax/search_spaces/patient_fit_search_space.json` was updated
(it still named the three fraction keys; the two-species file's notes
refer to it). Nothing is committed; the working tree also still holds
the identifiability fix and the three test assertions of the previous
section.

### The parametrisation

- **`--n-seeds n`** (`DEFAULT_N_SEEDS = 1`, any integer n >= 1) on
  `starts` and `fit`, in `_add_problem_args`; `build_problem(...,
  n_seeds)` passes it to `read_fit_search_space(path, solver_mode,
  objective_scope, n_seeds)`. On `--resume` and `resolve` the spec's
  value wins, as for `--solver` and `--objective` (the command line's
  `--n-seeds` is ignored there; the "resuming" line prints the count).
- **Factor expansion.** The search-space file keeps its three entries
  `seed_bbox_x/y/z` (linear on [0, 1], checked as before); they stand
  for ONE seed. With n > 1 `read_fit_search_space` replaces them, at the
  place of `seed_bbox_x` in the file's order (the file's `seed_bbox_y/z`
  are dropped from wherever they stand), by one block per seed:
  `seed_bbox_x_1, seed_bbox_y_1, seed_bbox_z_1, seed_bbox_x_2, ...`
  (`seed_suffix(k, n)` = '' for n = 1, else `_<k>`; `seed_bbox_names(k,
  n)`; `is_seed_bbox_factor(name)` recognises both forms), each a copy
  of the file's range with its own name. With n = 1 the names stay
  bare, so the one-seed factor order, columns, configs, starts.json and
  spec are those of before. The isotropic file gives 10 + 3 (n - 1)
  factors (16 with n = 3), the two-species file 14 + 3 (n - 1).
  `FitSpace` gained `n_seeds`, `seed_blocks` (the n name triples),
  `seed_bbox_names` (flat), and `solver_factor_names` now excludes the
  per-seed names. A file cannot clash with the expanded names: it is
  read before the expansion and rejects `seed_bbox_x_1` as unknown.
- **Shared shape.** `seed_sigma_mm` and `seed_peak_density` stay single
  factors (fitted or, the peak, fixed, as before). `config_of` writes
  the scalars `gaussian_seed_diffusion_time` and `gaussian_seed_mass`,
  which `_validate_seed` broadcasts to every row of
  `gaussian_seed_positions`; the config holds one row per seed, `[(v +
  0.5) / n]` of each seed voxel. The total initial mass is n times the
  one-seed mass where the seeds do not overlap; overlapping kernels add
  and clip at 1 (the package's rule). The evaluations.csv columns
  `gaussian_seed_mass` and `gaussian_seed_diffusion_time` stay scalars.
- **Mapping per seed.** `FitProblem.derive` maps every block through
  the one `SeedMap.voxel` (rounding, snapping) as before; `Point` holds
  `seed_voxels`, `seed_snapped` and `seed_fractions` as tuples of n
  (`Point.n_seeds`); the old scalar fields are gone. `design_checks`
  sets every seed's box factors to 0.5 at the corner.

### The seed order (the swap symmetry)

The loss is invariant under any permutation of the n seed blocks of the
unit vector: the solver sums the kernels, so the initial field is the
same. The item's note that "ordering the seeds along one axis in the
mapping removes it" was wrong as stated: a sort inside `derive` leaves
the loss landscape on the unit cube exactly as symmetric as before (it
only canonicalises the record). What removes the symmetry for the
optimizer is confining its search to one fundamental domain, done here
as a REPAIR, the same mechanism the fit already uses for the bounds:

- `CMAES(x0, sigma0, popsize, rng, repair=None)`: `ask` clips every
  member into [0, 1] and then applies `repair` to it; the repaired
  member is evaluated and is the point every update sees (y = (x - m) /
  sigma, z = invsqrtC y), exactly as the clipped member already was.
  `from_state(state, popsize, repair)` takes it too; the checkpoint
  format is unchanged.
- The repair is `FitSpace.order_seeds(u)`: a stable sort of the seed
  blocks by their x box coordinate `seed_bbox_x_<k>` (the box factors
  are linear on [0, 1], so the unit coordinate is the box coordinate),
  whole blocks permuted, every other coordinate untouched
  (`seed_order`, `order_seed_values` are the same on physical values).
  The ordered region {x_1 <= x_2 <= ... <= x_n} is convex, so the
  weighted mean of ordered members is ordered and the mean never falls
  between two mirror images of one optimum. `run_restart` passes
  `problem.space.order_seeds` with n > 1 and `None` with n = 1, so the
  one-seed path is untouched.
- Consequence for the records: seed k of every evaluated point is the
  k-th seed along the box's x axis (the voxel i index after snapping may
  break ties by a voxel). Near x_1 = x_2 the labelling flips between
  neighbouring members, which is the usual property of a fundamental
  domain and harmless (the losses are the same).

### Records, starts, output

- evaluations.csv: the per-seed columns `seed_voxel_i`, `seed_voxel_j`,
  `seed_voxel_k`, `seed_snapped` are written once per seed with the
  suffix of the factors (`seed_columns(n)`, `derived_columns(n)`
  replaces the constant `DERIVED_COLUMNS`; `evaluation_columns(names,
  prefixes, n_seeds)`); `typed_row` types them by prefix
  (`is_int_column`, `is_bool_column`; the seed names left `INT_COLUMNS`
  and `BOOL_COLUMNS`). `best_summary`, `run_restart` and `dry_run` use
  the n-dependent columns.
- The config's `_fit` record holds `n_seeds`, `seed_voxels` (a list of
  n triples) and `seed_snapped` (a list of n flags) for EVERY n; with
  n = 1 this replaces the old `seed_voxel` triple and the single flag
  (nothing reads them: the visualizers take the moments and snapshot
  days from `_fit`, not the seed). The solver-facing part of a one-seed
  config is byte-identical to before.
- spec.json: `search_space.n_seeds` and `search_space.seed_bbox_factors`
  (`FitSpace.record`); `parametrisation.n_seeds`,
  `parametrisation.seed_bbox_factors`, `parametrisation.seed_ordering`
  (the rule above) and the shared-shape wording in
  `parametrisation.seed_shape`. `problem_from_fit_dir` reads
  `parametrisation.n_seeds` (1 when absent, every spec before
  2026-11-04), and the existing factor-order check then verifies the
  expansion against the spec's `factor_names`, so resume and resolve
  work for fits of either kind.
- Starts (`convert_sweep_row`, `centre_start`): a sweep start puts
  EVERY seed at the row's one seed voxel (the box coordinates of that
  voxel in every block; `fit_seed_voxel` of the conversion record is
  that voxel) and the centre start every seed at the box centre: the
  seeds start together and the optimizer separates them. `--init-values`
  must name every per-seed factor (as it must name every factor);
  `manual_start` then orders its blocks along x and records
  `seeds_reordered` (only written with n > 1, so a one-seed starts.json
  is unchanged). `resolve_starts` records `n_seeds` and the rule in
  starts.json (n > 1 only). `format_starts` prints `seed voxel [i, j,
  k]` as before with one seed and `seed voxels [[...], ...] (snapped:
  [k, ...])` with several (`format_seed_voxels`); the `fit` start-up
  line names the seed count and the box factors.
- Module docstring: the "seed position" paragraph of the
  parametrisation, the Starts paragraph, the evaluations.csv column
  list, the CMA-ES bound-handling sentence and the examples
  (`fit --n-seeds 3 --name fit_sub01_3seeds ...`).

### Checked on 2026-11-04 on the JAX machine (CPU, no solve)

- n = 1 against the committed script (`git show HEAD:scripts/
  patient_cmaes_fit.py` imported next to the working copy) on sub-01
  with the default sweep: 25 random unit vectors give identical run
  configs (everything outside `_fit`, and `_fit` up to the three keys
  above), identical point records and identical csv columns; the 5
  sweep starts and their printed table are identical.
- n = 3 on sub-01: 16 factors in the expected order (the blocks between
  `growth_length_mm` and `rt_alpha`); `order_seeds` ascends in x and
  permutes whole blocks (the config's position rows are the same
  multiset before and after); the solver constructs from the config,
  its low-resolution seed voxels equal the script's record and the
  mass is broadcast to three equal entries; `CMAES.ask` members and the
  mean after `tell` are ordered, `from_state` keeps the repair; a
  manual start given with unordered seeds is reordered and records it;
  two sweep starts put all three seeds on the row's voxel;
  `run_design_checks` passes with n_steps 4176 confirmed at the corner.
- Through the command line: `starts --n-seeds 2 --top 2` (both seeds of
  each start at the row's voxel) and a one-seed `starts --init-values
  ... --no-sweep-init`.
- Expected noise: the solver logs "several seeds fall on one voxel of
  the low-resolution grid; their kernels add up there" whenever two
  seeds coincide, so every start evaluation (and the design check's
  centre config) logs it once.

NOT run: no GPU dry run, no fit, and the test suite was not rerun (no
test imports the fit script; the previous section's 154 passed stand).

### What to run next (nothing of it has been run)

```
python scripts/patient_cmaes_fit.py starts --n-seeds 3 --top 5
python scripts/patient_cmaes_fit.py fit --n-seeds 3 --name smoke_3seeds --gpus 0 --top 2 --dry-run
python scripts/patient_cmaes_fit_tum149.py fit --n-seeds 3 --name smoke_tum149_3seeds --gpus 0 --dry-run
```

Things to look at if something fails: the per-seed columns must appear
in `dry_run/evaluations.csv` with the suffixes; the three seeds of a
start coincide, so its initial field is three times one kernel, floored
and clipped once, i.e. the dry-run loss equals that of a one-seed run of
the same config with `gaussian_seed_mass` tripled (a hand check, not a
command of the script); a resumed or resolved OLD fit directory must
still read as one seed.
