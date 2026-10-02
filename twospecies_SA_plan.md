# Two-species sensitivity analysis: remaining verification

`scripts/patient_sensitivity_analysis.py` runs the two-species solver
(`--solver twospecies`, `TwoCompartmentWithNutrientFKPPSolver`) since
2026-10-02, with the search space
`fisher_kpp_jax/search_spaces/patient_SA_v2_search_space_twospecies.json`
and the base config
`fisher_kpp_jax/configs/TwoCompartmentWithNutrientFKPPSolver_stupp.json`.
Both files and the test update in `tests/test_patient_sensitivity_analysis.py`
are uncommitted on `stupp_all`. The scoring rules (core = thresholded P
against label 3 in every session, necrotic = thresholded N against label 1,
whole = thresholded P + N against labels 1, 2, 3; the moment QoIs on P; a
third cheap factor `necrotic_threshold` in sampled mode and the grid 0.10 to
0.85 in profiled mode) are in the script's module docstring.

## Working rules

- Branch `stupp_all` only; no commits; no ruff.
- Any file other than the script needs Lucas's permission before it is touched.
- No sweep, solve or QoI stage on real data without an explicit go from Lucas.
- Launch GPU work in tmux with `env -u LD_LIBRARY_PATH` (the shell's
  LD_LIBRARY_PATH makes JAX fall back to the CPU silently).

## Done (2026-10-02, no solve)

- Isotropic outputs unchanged: column lists, label conventions and QoI
  values on synthetic fields are identical to the previous version.
- Design step for both solvers at N = 2 into the scratchpad: standard
  k = 14 / 28 solves, twospecies k = 19 / 36 solves; the spec records the
  solver, the regions, the necrotic grid and the three cheap factors.
- A two-species run config loads through its class and resolves 12 steps
  per day on the CPU.
- One faked finished run per design (synthetic fields on the patient grid)
  through `qoi` and `analyze`: exit 0, the necrotic columns filled, the
  necrotic Dice NaN in a session without a necrotic label.
- `tests/test_patient_sensitivity_analysis.py` 20/20 and
  `tests/test_sensitivity_analysis.py` 33/33 pass; the fit script and the
  three visualizers import unchanged.

## Left to do

1. **GPU smoke of a two-species design** (needs a go). Suggested:
   ```
   env -u LD_LIBRARY_PATH python scripts/patient_sensitivity_analysis.py all \
       --solver twospecies --name smoke_twospecies_<date> --log2-n 1 \
       --output-dir <scratch or /mnt/Drive4/lucas/stupp_sensitivity_analysis_patient> --gpus <one id>
   ```
   36 solves at N = 2. Check: every run directory holds
   `<ses>_proliferative`, `<ses>_necrotic` and `<ses>_nutrient.nii.gz` per
   session plus `final_<field>.nii.gz`; `timeline.json` lists the solver,
   the fields and the files per session; `run_status.csv` has 0 failed;
   the `dt` column shows whether the two-compartment stability rule raised
   any step count; `qoi.csv` has every necrotic column filled for the
   successful rows and `qoi_summary.json` reports `n_necrotic_reference`
   per session; `analyze` completes (at N = 2 most QoIs are skipped for
   lack of complete blocks, which is expected). Record the mean run wall
   time from `qoi_summary.json` to size the full sweep: 18 432 solves at
   N = 1024.
2. **Isotropic re-run check on a real sweep** (needs a go): `qoi` and
   `analyze` on a copy of an existing isotropic sweep, or on a handful of
   its runs, to confirm `qoi.csv` is byte-identical in its columns and
   values to the version written before 2026-10-02 (the synthetic check
   above covers the functions, not the file path through `run_records` and
   `_load_field` on real NIfTI files).
3. **Fit-side check** (no solve): `python scripts/patient_cmaes_fit.py
   starts --solver twospecies --init-from-sweep <the smoke's sweep dir>`
   confirms the fit reads a two-species sweep and carries the four
   two-compartment columns instead of its fill values.

## Out of scope, noted

- `scripts/visualize_patient_best_run.py` reads one field from a sweep and
  `scripts/visualize_patient_best_run_twospecies.py` accepts fit
  directories only; rendering a two-species sweep's best run needs the
  latter to accept a sweep directory.
- The fit script keeps its own `THRESHOLD_GRID_NECROTIC` and
  `FIXED_NECROTIC_THRESHOLD`; it could import the sensitivity script's
  identical constants instead.
