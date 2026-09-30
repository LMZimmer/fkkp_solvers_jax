# Stupp treatments in all three solvers: implementation plan

Written 2026-09-17, revised 2026-09-29 after review, implemented from
2026-09-30 on in the order of section 7, which marks each step when it is
committed: steps 0 to 4 are done. Section 10 records what the
implementation added to the plan or does differently. The plan records the
design agreed for `fisher_kpp_jax` and the order in which to build it. Work
happens on branch `stupp_all`.

## 1. Goal and decisions

The treatment effects of a Stupp protocol (surgical resection, chemotherapy,
radiotherapy), today available only in `StuppFKPPSolver` on top of the
isotropic model, become part of every model: isotropic, two-compartment
with nutrient, and anisotropic. The repository keeps **three solver
classes with their current names**; `StuppFKPPSolver` disappears as a class
and stays only as a name so that saved configs still load.

Decisions taken in the design discussion (1 to 8 on 2026-09-17, revised
where marked, 9 and 10 added on 2026-09-29):

1. **Three solvers, treatment built in.** The treatment plumbing is written
   once in the base solver. A model class contributes only what is
   model-specific. No mixins, no two-base-class solvers.
2. **Disabled reproduces the untreated solver bit-exactly, and costs
   nothing.** The treatment parameters have neutral defaults. When every
   treatment value is neutral, the solver compiles the model update alone,
   which is exactly today's untreated step, so untreated runs pay nothing
   on the device and their numbers do not change. They pay nothing on the
   host either (revised 2026-09-29): a treatment volume that is not given
   stays `None` through validation and downsampling, and the arrays a
   treated program needs are built only once the run is known to be
   treated. This reverses the 2026-09-02 decision that every treatment
   parameter is required; disabling still happens through values, never
   through key presence, and there is no per-treatment branching.
3. **One face builder.** All face diffusivities come from one function that
   averages cell fields onto faces and mixes them, written so that the
   isotropic path evaluates exactly today's expression. The reference solve
   in `reference_solves/` must keep its present agreement
   (f64 max abs difference 2.776e-16).
4. **Two model updates, one treatment wrapper.** The repository holds one
   update for the single-field models (isotropic and anisotropic) and one for
   the two-compartment model, plus one wrapper that applies the treatments to
   either. No further step functions.
5. **Two-compartment model: killed cells vanish** (revised 2026-09-29).
   The kill impulses act on the proliferative field and the killed cells
   leave the system, as in the single-field model. The alternative, killed
   cells becoming necrotic, was rejected: the model has no necrotic
   clearance, so with that choice P + N would never decrease except
   through resection, and every quantity built from P + N (the mass and
   volume stopping quantities, the segmentation comparisons of the patient
   scripts) would be blind to chemotherapy and radiotherapy. The code
   carries a note recording the rejected alternative and this reason.
6. **Two-compartment cavity:** the projection zeroes proliferative, necrotic
   and nutrient inside the cavity, and the post-resection nutrient faces
   block flux across the cavity boundary, so the cavity behaves like CSF.
7. **No guards** (revised 2026-09-29). The anisotropic solver's shrinkage
   and vanishing-volume guards are deleted, and with them the guard
   mechanism of the time loop, since no solver uses one afterwards. The
   base pipeline's non-finite check is the only blow-up detector for all
   three solvers; it already was for the two isotropic ones. The guards
   never saved compute (the scan always runs its fixed length), their only
   value was the exit step in the error message.
8. **Names:** `FKPPSolver`, `TwoCompartmentWithNutrientFKPPSolver`,
   `AnisotropicFKPPSolver`.
9. **Treated runs keep today's arithmetic.** The wrapper applies the
   chemotherapy factor and then the radiotherapy factor, one after the
   other in today's order, so treated isotropic runs reproduce today's
   `StuppFKPPSolver` results bit for bit.
10. **`chemo_decay_rate` defaults to 9.24 per day**, temozolomide's plasma
    decay (a half-life of about 1.8 h; Baker et al. 1999, Clin Cancer Res
    5:309), the value every config and search space of the repository
    uses. A neutral default is only inert while there is no session; as
    soon as sessions are given, the default is a physical choice and must
    be the project's value, not a placeholder.
11. **Two isotropic configs, values unchanged, and no dead name in shipped
    files** (2026-09-29). `configs/FKPPSolver.json` is the untreated
    config and `configs/FKPPSolver_stupp.json` (today
    `StuppFKPPSolver.json`) the treated one; neither changes a parameter
    value. The shipped search spaces name `FKPPSolver`; the three
    first-version search spaces no script uses are deleted. The name
    `StuppFKPPSolver` survives only as the alias, for saved configs and
    for the copies of search spaces inside completed sweep directories.
    The demo config and the four `run_stupp_*` scripts are left for a
    later cleanup outside this plan.

## 2. Where things live

| File | Holds after the change |
|---|---|
| `fisher_kpp_jax/base.py` | the treatment parameters and their defaults, the horizon rule, the cavity config entry, treatment validation, the treated predicate, the treatment volumes on the low-resolution grid, the treatment device constants, the `post_resection` constants, the treated/untreated program selection; no guard hook |
| `fisher_kpp_jax/solvers.py` | the three model classes, the two model updates, the treatment wrapper and its two bindings, the constants TypedDicts; `_dti_guard` is deleted |
| `fisher_kpp_jax/operators.py` | the face builder, the chemotherapy and radiotherapy operators (unchanged), the time loop with the threshold stop only (the guard mechanism is deleted) |
| `fisher_kpp_jax/config.py` | the solver registry with an alias table |
| `fisher_kpp_jax/configs/*.json` | the three untreated class configs, values unchanged, each listing the treatment keys at their neutral values; `FKPPSolver_stupp.json` (today `StuppFKPPSolver.json`, values unchanged) as the treated config the scripts use |
| `fisher_kpp_jax/search_spaces/*.json` | the four live search spaces with `"solver": "FKPPSolver"`; the three first versions are deleted |
| `scripts/run_reference_solves.py` | done 2026-09-29: the isotropic section as before plus the two-compartment section (section 9) |
| `scripts/sensitivity_analysis.py`, `scripts/patient_cmaes_fit.py` | the fixes of section 5 |
| `reference_solves/` | done 2026-09-29: the three two-compartment reference fields (section 9) |
| `tests/` | the adaptations of section 5 |

## 3. Parameters

### 3.1 Treatment keys and neutral defaults

The ten treatment keys are the ones `StuppFKPPSolver` has today. They move
into the shared defaults of every solver, so `TREATMENT_KEYS` becomes an
attribute of the base class:

```
resection_time        inf      never reached; JSON spells it "inf"
time_after_resection  null     see the horizon rule
resection_cavity      null     no cavity
rt_dose               null     no dose
chemo_times           ()       no session
chemo_doses           ()
chemo_kill_rate       0.0
chemo_decay_rate      9.24     1/day, decision 10; must stay > 0
rt_times              ()       no fraction
rt_alpha              0.0
rt_alpha_beta_ratio   10.0     as today
```

The two volumes are the only entries where `null` stands for a neutral
value, because an array cannot be a default; `null` stays `null` inside
the solver (section 3.5). Everything else is neutral by its value. A
treatment parameter that is given as `None` takes its default.

The three empty sequences default to `()`, not `[]`, so that the class
defaults are immutable. The config a solver records holds a tuple default
as a new list, the form it has in a JSON file, so that a written config
still reads back equal; a config file spells them `[]`.

### 3.2 Horizon rule

`stopping_time` and `time_after_resection` both default to `null`, and at
most one may be given, the same rule as for the three time-step keys:

- `stopping_time` given: the horizon is that value.
- `time_after_resection` given: the horizon is
  `resection_time + time_after_resection`; `resection_time` must then be
  finite.
- neither given: 100 days, as today's default.

The three class configs keep `stopping_time: 100` as today and gain
`time_after_resection: null`; the default-config test treats the horizon
pair like the time-step keys, exempt from pinning and at most one
non-null. The horizon a run used is reported in
`Result.derived["stopping_time"]` whenever the config did not give it, as
it is today for treated runs.

### 3.3 Validation

The treatment validation of today's `StuppFKPPSolver._validate_extra` moves
into the base solver and runs after the model's own validation and after
the horizon rule of section 3.2 (`_validate_event_times` reads
`parameters["stopping_time"]`), with these changes:

- `resection_time` is a nonnegative scalar and may be infinite
  (`_validate_nonnegative_scalar` requires a finite value today, so the
  check is written out).
- `rt_times` may be empty. The per-fraction dose is `rt_dose / n` for n
  fractions and zero for none.
- A volume given as an array or path is checked against the shape of the
  class's reference volume, `params[_REFERENCE_VOLUME_KEY].shape[:3]`,
  which is the tissue map for the two isotropic-mixture models and the
  tensor field for the anisotropic one. A `null` volume is not checked and
  is not replaced by an array here.
- A non-empty `resection_cavity` requires a finite `resection_time`; with
  the default `inf` it is a validation error.

The cavity config entry `{"segmentation": <NIfTI path>, "label": <int>}`
and its loading move unchanged from `StuppFKPPSolver` into the base
solver's `_resolve_config_volume` and `_load_volume_entry`. Every solver's
`_VOLUME_KEYS` gains `rt_dose` and `resection_cavity`.

### 3.4 The treated predicate

After validation the base solver decides, on the host and from values only,
whether the run has any treatment:

```
treated = (cavity is not None and cavity.any())
       or (chemo_kill_rate > 0 and chemo_doses.any())
       or (rt_alpha > 0 and rt_dose is not None and rt_dose.any())
```

The predicate is deliberately conservative: it ignores event times, so a
treatment scheduled beyond the horizon still compiles the treated program,
which then does nothing. It never reports untreated while an effect exists.
It decides which program is compiled and whether the treatment volumes are
built at all.

### 3.5 Treatment volumes on the low-resolution grid

Only when the predicate says treated, and after the model's field
preparation (where the low-resolution shape is known):

- a given cavity is downsampled linearly with the tissue maps' factor and
  thresholded at 0.5, a given dose downsampled linearly and clipped at 0,
  as today;
- a `null` cavity becomes an all-false array of the low-resolution shape,
  a `null` dose an all-zero one, built directly at that shape, never at
  full resolution and never interpolated.

An untreated run builds nothing. This is what makes decision 2 hold on the
host: the growth stages of every sweep and every fit evaluation do no
treatment work.

## 4. Device side

### 4.1 One face builder

`operators.face_diffusivities(diffusivity, terms, valid_mask)` returns the
six face arrays (`fwd_x/y/z`, `bwd_x/y/z`) consumed by `diffusion_term`.
`terms` is a sequence of `(field, divisor)` pairs; a field is 3D, or 4D
with a trailing axis of three for per-axis values. For every axis:

```
fwd = diffusivity * ( avg(field_1) / divisor_1 + avg(field_2) / divisor_2 + ... )
bwd = the edge-replicated shift of fwd
```

with `avg` the masked face average and the sum accumulated left to right.
The division is always performed, a divisor of 1 included: dividing by 1
is exact in floating point, so the white-matter term and the nutrient
faces come out bit-identical to today's expressions without a special
case, and the two-compartment step can pass a traced device scalar as the
ratio in its per-step rebuild inside the jitted program. The three uses:

```
isotropic tumour   D    [(wm, 1), (gm, diffusivity_ratio)]   mask: tissue (per step with occupancy in the two-compartment model)
nutrient           D_s  [(wm, 1), (gm, 1)]                   mask: tissue
anisotropic        D    [(axial, 1)]                          mask: all true
```

The isotropic case is exactly today's `diffusivity * (wm_face + gm_face /
ratio)`; the anisotropic case reduces to `diffusivity * face_average(...)`
because a masked average under an all-true mask is the plain average and
shifting the scaled face equals scaling the shifted face. Both are
bit-identical to today. `_mixture_face_fields` and the anisotropic face
loop are deleted.

### 4.2 Model hooks

Each model class implements, besides the hooks it has today:

- `_valid_mask_host(box) -> NDArray`: the cells that may carry flux.
  Tissue above `min_tissue_fraction` for the two mixture models, all true
  for the anisotropic model.
- `_structural_constants(box, valid_mask_host) -> dict`: the constants
  that depend on that mask. Single-field models: `face_diffusivities`.
  Two-compartment model: `tissue_mask` and `nutrient_faces` (the tumour
  faces are rebuilt every step from `tissue_mask`, as today).
- `_build_device_constants(box) -> dict`: everything else (scalars, and for
  the two-compartment model the `wm` and `gm` fields the per-step rebuild
  needs).

The base solver assembles the constants:

```
untreated:  {**model, **structural(valid), **shared}
treated:    {**model, **structural(valid), **treatment,
             "post_resection": structural(valid & ~cavity), **shared}
```

`treatment` holds `resection_time`, `cavity`, `chemo_times`, `chemo_doses`,
`chemo_kill_rate`, `chemo_decay_rate`, `rt_times` and `rt_log_kill`,
exactly today's Stupp constants minus the post faces, which now live under
`post_resection` with the same keys as their pre-resection counterparts.
A `_TreatmentConstants` TypedDict documents them; each model's flat
constants type inherits from its own specific type, the shared type and
the treatment type.

### 4.3 Step functions

Today's `_single_field_step` and `_two_compartment_step` become
`_single_field_update` and `_two_compartment_update`, unchanged in body.
They are the `_step_func` of the untreated program.

One wrapper, `_treated_step(state, constants, step_index, *, update,
killed, cleared)`, does in this order:

1. `post = t1 >= resection_time` with `t1 = (step_index + 1) * dt`;
   patch the constants with `post_resection` selected by `post`
   (`jax.tree_util.tree_map` with `jnp.where` over the pre and post
   structures);
2. run `update` on the patched constants (the explicit Euler step at the
   pre-step state);
3. for every field in `killed`, first
   `field <- field * exp(-chemo_kill_rate * E_ct)` with `E_ct` the exact
   exposure of the step (`chemo_exposure`), then
   `field <- field * exp(-E_rt * n_hits)` with `n_hits` the number of
   `rt_times` in `(t0, t1]`; the two factors are applied one after the
   other in this order, as today (decision 9), never combined into one;
4. for every field in `cleared`: zero it where `post & cavity`.

The two bindings are module-level objects, so the jit cache keeps working:

```python
_treated_single_field_step = partial(
    _treated_step, update=_single_field_update,
    killed=("cell_density",), cleared=("cell_density",))
_treated_two_compartment_step = partial(
    _treated_step, update=_two_compartment_update,
    killed=("proliferative",), cleared=("proliferative", "necrotic", "nutrient"))
```

The single-field binding is today's `_stupp_step` operation for
operation. Each class declares `_step_func` (its update) and
`_treated_step_func` (its binding); `_run_device_loop` passes one or the
other to the time loop according to the treated predicate. The wrapper
itself has no branch.

The two-compartment binding carries the note required by decision 5:
killed cells vanish; the rejected alternative (moving them into the
necrotic field) and its reason are stated. Consequences worth stating in
that note: P + N drops under a kill, so the mass and volume stopping
quantities see it and the occupancy mask frees up; the nutrient is not
touched by the kill, and the consumption of the same step used the
pre-kill proliferative field.

### 4.4 Model specifics

**Isotropic.** Identical to today's `StuppFKPPSolver` behaviour once the
treatment values are non-neutral, and identical to today's `FKPPSolver`
otherwise.

**Two-compartment.** `post_resection` holds the tissue mask with the cavity
removed and the nutrient faces built with that mask, so after the
resection no tumour or nutrient flux crosses the cavity boundary. The
projection empties all three fields inside the cavity every post-resection
step. The initial nutrient is unchanged (one inside tissue, zero outside).

**Anisotropic.** The untreated faces use the all-true mask, so nothing
changes for untreated runs. The post-resection faces use `~cavity`, which
zeroes every face touching a cavity voxel, the same rule as the isotropic
model. There is no guard: a treated run whose entire seed lies inside the
cavity leaves a zero field after the resection and completes normally
with the stopping criterion "time".

### 4.5 Guard removal

Deleted in `operators.py`: `SHRINKAGE_LIMIT`, `VANISHING_DENSITY_LIMIT`,
`_STOP_SHRINKAGE`, `_STOP_VANISHING`, `_no_guard`, the `guard_func`
argument and its signature description in `_time_step`, `_run_time_scan`
and `_run_time_loop`, the guard branches of `_time_step`'s stop-kind
select, and the `guard_mass_change` and `guard_density` entries of the
scan carry. The stop kinds reduce to `_RUNNING` and `_STOP_THRESHOLD`.
Deleted in `base.py`: the `_guard_func` class attribute, the two guard
fields of `_TimeLoopOutputs`, `_guard_error_message` and its call; the
imports go with them. Deleted in `solvers.py`: `_dti_guard` and the
anisotropic class's guard binding. The stopping criterion "error" then
arises only from an exception or from the non-finite check.

## 5. Backward compatibility

- **Saved configs.** Every `config.json` in `results/` and `runs/` names
  `StuppFKPPSolver` and gives `time_after_resection` without
  `stopping_time`. Both stay valid: the registry gets an alias table
  (`"StuppFKPPSolver"` resolves to `FKPPSolver`), `solver_class`,
  `_resolve_solver` and the constructor's name check compare classes, not
  names, and the horizon rule accepts the old key. `fisher_kpp_jax`
  exports `StuppFKPPSolver = FKPPSolver`. New configs are written with the
  class's own name.
- **Configs: one untreated, one treated, values unchanged.**
  `configs/FKPPSolver.json` stays the untreated config; like the other two
  class configs it only gains the ten treatment keys at their neutral
  values and `time_after_resection: null`. `configs/StuppFKPPSolver.json`
  is renamed to `configs/FKPPSolver_stupp.json` with
  `"solver": "FKPPSolver"`; every parameter value stays. It is the treated
  config of the isotropic model and the base config of the sweep,
  identifiability and patient scripts, which reach it through one
  constant, `sensitivity_analysis.DEFAULT_CONFIG` (the fit and
  identifiability scripts import it); the constant, the docstrings naming
  the file in the four scripts and the tests' `BASE_CONFIG` follow. Its
  `_note` entries say it is the treated config, not a class default, and
  that with its `null` cavity and dose it constructs and runs as a
  chemotherapy-only run (today it fails). A treated two-compartment or
  anisotropic config, when a script needs one, is that model's class
  config plus the schedule block of this file.
- **Search spaces.** The four live files (`stupp_fkpp_sigma_v2`,
  `stupp_identifiability`, `sailor_patient_v2`, `sailor_patient_fit`) get
  `"solver": "FKPPSolver"`; nothing else in them changes. The three first
  versions (`stupp_fkpp`, `stupp_fkpp_sigma`, `sailor_patient`) are
  deleted: no script uses them, and the sweeps that did keep their own
  copies. Where a `_note`, `_revision` or docstring names a deleted search
  space, "(deleted 2026-09-29)" is appended instead of rewriting it (a
  deviation from "nothing else in them changes" limited to those comment
  entries). All loaders funnel into `sensitivity_analysis.load_search_space`
  (the identifiability and patient loaders call it), whose `solver_name`
  check becomes a class check: the entry is resolved through the registry
  and must be `FKPPSolver`, so the copies in old sweep and result
  directories, which say `StuppFKPPSolver`, still load through the alias.
  The fit script's own loader check does the same.
- **Scripts.** Strict backward compatibility is not required for
  `sensitivity_analysis.py` and `patient_cmaes_fit.py`.
  - `StuppFKPPSolver.TREATMENT_KEYS` keeps working through the alias.
  - `sensitivity_analysis.growth_config` must drop the treatment keys and
    `time_after_resection` explicitly, since `FKPPSolver.config_keys()`
    now contains them; today it relies on their absence.
  - `sensitivity_analysis.run_records` classifies a run's `result.json`
    by the solver name today (`FKPPSolver` means growth-only). Treated
    runs will carry that name too, so it takes the mode instead:
    `run_records(run_dir, growth_only)`; `result.json` is the growth stage
    in growth-only mode and the treated stage otherwise, and the solver
    name is recorded but no longer interpreted. `run_subprocess` passes
    its own `growth_only`, the collect stage passes the mode it resolves
    from `spec.json` (`resolve_growth_only`), and the two callers in
    `patient_sensitivity_analysis.py` pass `False` (patient runs are
    always treated).
  - `patient_cmaes_fit.resolved_constants` calls
    `StuppFKPPSolver.get_default_config()`, which under the alias returns
    the untreated `FKPPSolver.json`; it reads the treated config by path
    instead, `read_config(DEFAULT_CONFIG)`.
  - `SOLVER_NAME` in `sensitivity_analysis.py` and
    `patient_sensitivity_analysis.py` becomes `FKPPSolver.__name__`; the
    search-space loaders are covered under "Search spaces" above.
  - The demo config `scripts/stupp_config_example.json` and the four
    `run_stupp_*` scripts are left as they are (decision 11); they load
    through the alias, and `run_stupp_synthetic.py`'s name check compares
    against `StuppFKPPSolver.__name__`, which is `FKPPSolver`, so it
    passes.
  - Docstrings that state that the Stupp solver rejects `stopping_time`
    or that its treatment parameters are required are corrected: the
    `StuppFKPPSolver` class docstring (which becomes the treatment part of
    the `FKPPSolver` docstring), the `config.py` module docstring, the
    "Four solvers" of `__init__.py`, and the scripts' mentions.
- **Tests to adapt** (no new test files; the existing cases are adapted):
  - `tests/test_stupp.py`: the module docstring; `test_neutral_treatment_equals_fkpp`
    asserts exact equality of every field and snapshot
    (`assert_array_equal`) and that the neutral solve compiled no new
    program (`operators.SCAN_TRACE_COUNT` unchanged by it once the
    untreated program is cached); `test_validation_errors` replaces the
    "every treatment parameter is required" block by "omitting a
    treatment parameter, or giving `None`, yields the neutral value",
    loses the "at least one fraction" case, accepts an empty `rt_times`
    (zero log kill), an infinite `resection_time`, and checks the horizon
    rule (both keys given raises, neither gives 100 days);
    `test_treatment_keys` asserts one `TREATMENT_KEYS` set on all three
    classes, contained in every class's `config_keys()`, with
    `time_after_resection` a base key; the assertion that a constructed
    config records `"StuppFKPPSolver"` becomes `"FKPPSolver"`, while the
    test configs written with the old name stay so that the alias is
    covered; `test_read_config_rejects_rt_beta` stays.
  - `tests/test_config.py`: `SOLVERS` holds the three classes;
    `PINNED_DEFAULT_KEYS` gains `resection_time`, `resection_cavity`,
    `rt_dose`, `chemo_times`, `chemo_doses`, `chemo_kill_rate`, `rt_times`
    and `rt_alpha` and loses `stopping_time`, `PHYSICAL_DEFAULT_KEYS`
    gains `chemo_decay_rate`, and the horizon pair `stopping_time` /
    `time_after_resection` joins the time-step keys in the exemption (at
    most one non-null); `test_blank_default_config_fails_at_construction`
    keeps only the anisotropic case, and its former Stupp case becomes the
    check that `configs/FKPPSolver_stupp.json` loads by path, names
    `FKPPSolver` and constructs, so the file stays covered; the comments
    naming the old file follow the rename.
  - `tests/test_solvers.py`: `test_dti_guard_exit` is deleted; no guard
    exists, and a decaying anisotropic tumour runs to the horizon with
    success.
  - `tests/test_sensitivity_analysis.py`: the two assertions that a
    saved record names `"StuppFKPPSolver"` become `"FKPPSolver"`; the
    `run_records` calls pass the mode; `SHIPPED_SEARCH_SPACE`, the fixture
    of the design and loader tests, becomes the shipped default
    `stupp_fkpp_sigma_v2_search_space.json`; the tests that compare the
    first version with the sigma version and the sigma version with v2
    (the readers of `SIGMA_SEARCH_SPACE`) are deleted with their files,
    keeping what checks v2 on its own.
  - `tests/test_patient_sensitivity_analysis.py`: `BASE_CONFIG` follows
    the rename; `SEARCH_SPACE_V1` and the v1-versus-v2 comparison tests
    are deleted with their file.
  - The other test modules use `StuppFKPPSolver` through the alias and
    read the treated config by path through `sa.DEFAULT_CONFIG`; they are
    unchanged.

## 6. Numerical guarantees

- Untreated runs of all three models compile the same program as today
  and are bit-identical.
- A run with neutral treatment values is exactly the untreated run, because
  it compiles the untreated program and builds no treatment volume.
- The isotropic reference solve keeps its current agreement.
- Treated isotropic runs reproduce today's `StuppFKPPSolver` results bit
  for bit: the wrapper binding is the old step operation for operation
  (decision 9), and the face construction is the old arithmetic.
- Untreated two-compartment runs reproduce the stored reference fields of
  section 9; `scripts/run_reference_solves.py --solvers two_compartment`
  reports max|d| and relL2 per field, and max|d| == 0 per field on the GPU
  at f64 is the acceptance (step 9 of section 7).
- The anisotropic model has no reference solve (section 8); the phantom
  tests are its only check.

## 7. Implementation order

0. Done 2026-09-29: the two-compartment reference solve stored
   (section 9), the script extended.
1. Done 2026-09-30: `operators.py`: add `face_diffusivities`; delete the guard mechanism
   (section 4.5) and shrink the scan carry accordingly.
2. Done 2026-09-30: `solvers.py`: switch the three models to the builder; add
   `_valid_mask_host` and `_structural_constants`; rename the two steps to
   updates; add `_treated_step` and the two bindings; delete
   `_stupp_step`, `_mixture_face_fields`, `_dti_guard`, the
   `StuppFKPPSolver` class and its TypedDicts; add `_TreatmentConstants`;
   set `_step_func` and `_treated_step_func` on each class.
3. Done 2026-09-30: `base.py`: treatment keys and defaults in the shared defaults, the
   horizon rule, the cavity config entry methods, the treatment validation
   with `null` volumes left alone, the treated predicate, the treatment
   volumes on the low-resolution grid (section 3.5), the constants
   assembly and the step selection, `TREATMENT_KEYS` on the class; remove
   the guard hook and diagnostics.
4. Done 2026-09-30: `config.py`: the alias table and class-based name checks.
5. `__init__.py`: export the alias; update the module docstring.
6. `configs/` and `search_spaces/`: the three class configs; the rename
   to `FKPPSolver_stupp.json` and its notes; the solver entry of the four
   live search spaces; delete the three first versions.
7. `scripts/`: `DEFAULT_CONFIG` and the docstrings naming the file;
   `growth_config`; `run_records` and its callers; `resolved_constants`;
   the class check of `load_search_space` and of the fit loader;
   `SOLVER_NAME`; docstring corrections.
8. `tests/`: the adaptations listed in section 5.
9. Open. Run `scripts/run_reference_solves.py --solvers all` and the test
   suite; confirm the isotropic agreement is unchanged, the two-compartment
   fields match the stored reference, and a config from a completed sweep
   still loads and re-solves. Acceptance for the two-compartment
   regression: max|d| == 0 per field on the GPU at f64; the numbers are
   reported either way.

## 8. Inputs of the two newly treated models

Both take the treatment inputs the isotropic solver takes today, on the grid
of the model's reference volume:

- `resection_cavity`: in a config `{"segmentation": <NIfTI path>, "label":
  <int>}`, a 3D integer-labelled segmentation (values rounded to integers,
  the cavity being the voxels carrying the label); in memory a 3D bool or
  0/1 array. SAILOR: `longitudinal/recurrence_preop.nii.gz`, label 4.
- `rt_dose`: a 3D float NIfTI in Gy holding the TOTAL dose over all
  fractions, nonnegative and finite. SAILOR:
  `longitudinal/dose_warped_longitudinal.nii.gz`.
- the schedule and response scalars: `resection_time`,
  `time_after_resection` or `stopping_time`, `chemo_times` with one
  `chemo_doses` entry per session, `chemo_kill_rate`, `chemo_decay_rate`,
  `rt_times`, `rt_alpha`, `rt_alpha_beta_ratio`.

Model-specific:

- **Two-compartment:** nothing beyond the tissue maps and the four nutrient
  and necrosis scalars it already needs; the treatment volumes must match
  the tissue maps' shape.
- **Anisotropic:** the tensor field it already needs, a 5D NIfTI of shape
  (Nx, Ny, Nz, 3, 3) whose header gives the voxel size and affine; the
  treatment volumes must match its first three dimensions. No tensor field
  exists in the project or in the processed SAILOR tree; the raw SAILOR
  data holds diffusion-weighted images with bval and bvec files for some
  subjects, so a tensor fit and a registration to the tissue-map grid are
  needed outside the solver first. Until then the treated anisotropic path
  is exercised by the phantom tests only.

## 9. Reference solves

- **Isotropic** (unchanged): the GliODIL configuration on the SAILOR sub-01
  session-1 tissue maps, checked by `scripts/run_reference_solves.py
  --solvers isotropic` against `reference_solves/result_reference.nii.gz`.
- **Two-compartment** (done 2026-09-29): a regression baseline of
  `fisher_kpp_jax` itself, taken before the change. The run is the class
  default config `configs/TwoCompartmentWithNutrientFKPPSolver.json`
  unchanged (it already matches `FKPPSolver.json` on every shared key:
  the same maps, D 0.80021, rho 0.075777, seed, grid and stopping
  settings, plus the placeholder nutrient and necrosis values) at
  precision f64 on a GPU, at the solver's own stability step: 941 steps,
  dt 0.106364 d, final time 100, final mass P + N 42880.563 (sum P
  31666.031, sum N 11214.532, sum nutrient 1562492; max P 0.349407, max N
  0.875379). Stored: `reference_solves/reference_two_compartment_{proliferative,necrotic,nutrient}.nii.gz`,
  float64 with the maps' affine; deliberately no config and no log.
  `scripts/run_reference_solves.py --solvers two_compartment` re-solves
  the same config and prints max|d| and relL2 per field against the
  stored ones; `--write-two-compartment-reference` rewrites them.
- **Anisotropic:** none; a tensor field has to be processed first
  (section 8).

## 10. Implementation notes (2026-09-30)

- **Steps 1 to 3.** Old against new on the 24^3 phantoms, 90 arrays
  (final fields, snapshots, step counts and stopping quantities of the
  untreated isotropic, two-compartment and anisotropic runs with the time,
  threshold and volume stops, and of treated isotropic runs, the former
  `StuppFKPPSolver` against `FKPPSolver`; f32 and f64): max|d| = 0 for
  every array on the GPU and on the CPU.
- `base.py` holds `TREATMENT_DEFAULTS`, `TREATMENT_VOLUME_KEYS`,
  `HORIZON_KEYS` and `DEFAULT_STOPPING_TIME`; the shared defaults
  (`_COMMON_DEFAULTS` in `solvers.py`) take the treatment defaults from
  there. The horizon rule is `_resolve_horizon`, the treatment validation
  `_validate_treatment`, the treated predicate `BaseFKPPSolver._is_treated`
  (evaluated at solve time from `params`), the volumes
  `_treatment_fields`, the device constants `_build_treatment_constants`.
- The treatment volumes are built in `_run_device_loop`, right before the
  constants, not during the grid setup: `resolve_time_stepping()` builds
  none, a treated solve builds them once.
- `_TreatmentConstants` lives in `solvers.py` as planned, while its values
  are built in `base.py`, whose builders therefore return plain dicts.
- A default that is a tuple is recorded in `solver.config` as a list
  (section 3.1).
- `stopping_time` itself is still not validated (as before); the horizon
  rule only decides which of the two keys gives it.
- **Step 4.** `config.names_solver(named, cls)` is the class-based check
  the constructor and `_resolve_solver` use; an unregistered name names no
  class, so the constructor's message for it is unchanged.
  `register_solver` refuses a class named like an alias. `read_config`
  returns the class's own name in the "solver" entry, also for a file that
  names the class by its former name.
