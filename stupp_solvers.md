# Stupp treatments in all three solvers: implementation plan

Written 2026-09-17. Nothing in this plan is implemented yet. It records the
design agreed for `fisher_kpp_jax` and the order in which to build it.

## 1. Goal and decisions

The treatment effects of a Stupp protocol (surgical resection, chemotherapy,
radiotherapy), today available only in `StuppFKPPSolver` on top of the
isotropic model, become part of every model: isotropic, two-compartment
with nutrient, and anisotropic. The repository keeps **three solver
classes with their current names**; `StuppFKPPSolver` disappears as a class
and stays only as a name so that saved configs still load.

Decisions taken in the design discussion:

1. **Three solvers, treatment built in.** The treatment plumbing is written
   once in the base solver. A model class contributes only what is
   model-specific. No mixins, no two-base-class solvers.
2. **Disabled reproduces the untreated solver bit-exactly.** The treatment
   parameters have neutral defaults. When every treatment value is neutral,
   the solver compiles the model update alone, which is exactly today's
   untreated step, so untreated runs pay nothing and their numbers do not
   change. This reverses the 2026-09-02 decision that every treatment
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
5. **Two-compartment model:** the kill impulses act on the proliferative
   field and the killed cells **become necrotic**. The code carries a note
   that this is an open modelling question (the alternative is that they
   vanish, as in the single-field model).
6. **Two-compartment cavity:** the projection zeroes proliferative, necrotic
   and nutrient inside the cavity, and the post-resection nutrient faces
   block flux across the cavity boundary, so the cavity behaves like CSF.
7. **Anisotropic guard:** only the vanishing-volume guard is kept. The
   shrinkage guard is deleted (a resection or a fraction would fire it). The
   base pipeline's non-finite check remains the blow-up detector.
8. **Names:** `FKPPSolver`, `TwoCompartmentWithNutrientFKPPSolver`,
   `AnisotropicFKPPSolver`.

## 2. Where things live

| File | Holds after the change |
|---|---|
| `fisher_kpp_jax/base.py` | the treatment parameters and their defaults, the horizon rule, the cavity config entry, treatment validation, cavity and dose downsampling, the treatment device constants, the `post_resection` constants, the treated/untreated program selection |
| `fisher_kpp_jax/solvers.py` | the three model classes, the two model updates, the treatment wrapper and its two bindings, the constants TypedDicts |
| `fisher_kpp_jax/operators.py` | the face builder, the chemotherapy and radiotherapy operators (unchanged), the time loop with the vanishing guard only |
| `fisher_kpp_jax/config.py` | the solver registry with an alias table |
| `fisher_kpp_jax/configs/*.json` | the three default configs, each listing the treatment keys at their neutral values; `StuppFKPPSolver.json` stays as the treated example config the scripts use |

## 3. Parameters

### 3.1 Treatment keys and neutral defaults

The ten treatment keys are the ones `StuppFKPPSolver` has today. They move
into the shared defaults of every solver, so `TREATMENT_KEYS` becomes an
attribute of the base class:

```
resection_time        inf      never reached; JSON spells it "inf"
time_after_resection  null     see the horizon rule
resection_cavity      null     no cavity (an all-false mask)
rt_dose               null     zero dose everywhere
chemo_times           []       no session
chemo_doses           []
chemo_kill_rate       0.0
chemo_decay_rate      1.0      must stay > 0; inert without sessions
rt_times              []       no fraction
rt_alpha              0.0
rt_alpha_beta_ratio   10.0     as today
```

The two volumes are the only entries where `null` stands for a neutral
value, because an array cannot be a default. Everything else is neutral by
its value. The `chemo_decay_rate` default of 1.0 is a placeholder choice;
any positive number is inert while there is no session.

### 3.2 Horizon rule

`stopping_time` and `time_after_resection` both default to `null`, and at
most one may be given, the same rule as for the three time-step keys:

- `stopping_time` given: the horizon is that value.
- `time_after_resection` given: the horizon is
  `resection_time + time_after_resection`; `resection_time` must then be
  finite.
- neither given: 100 days, as today's default.

The horizon a run used is reported in `Result.derived["stopping_time"]`
whenever the config did not give it, as it is today for treated runs.

### 3.3 Validation

The treatment validation of today's `StuppFKPPSolver._validate_extra` moves
into the base solver and runs after the model's own validation, with these
changes:

- `resection_time` is a nonnegative scalar and may be infinite.
- `rt_times` may be empty. The per-fraction dose is `rt_dose / n` for n
  fractions and zero for none.
- The volume shape every treatment volume is checked against is the shape
  of the class's reference volume, `params[_REFERENCE_VOLUME_KEY].shape[:3]`,
  which is the tissue map for the two isotropic-mixture models and the
  tensor field for the anisotropic one.
- A `null` cavity becomes an all-false array and a `null` dose an all-zero
  array of that shape, before validation.

The cavity config entry `{"segmentation": <NIfTI path>, "label": <int>}`
and its loading move unchanged from `StuppFKPPSolver` into the base
solver's `_resolve_config_volume` and `_load_volume_entry`. Every solver's
`_VOLUME_KEYS` gains `rt_dose` and `resection_cavity`.

### 3.4 The treated predicate

After validation the base solver decides, on the host and from values only,
whether the run has any treatment:

```
treated = cavity.any()
       or (chemo_kill_rate > 0 and chemo_doses.any())
       or (rt_alpha > 0 and rt_dose.any())
```

The predicate is deliberately conservative: it ignores event times, so a
treatment scheduled beyond the horizon still compiles the treated program,
which then does nothing. It never reports untreated while an effect exists.
It decides only which program is compiled; results of a neutral run are the
same either way, and exactly the same as the untreated model.

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
The division is skipped when the divisor is exactly 1. The three uses:

```
isotropic tumour   D    [(wm, 1), (gm, diffusivity_ratio)]   mask: tissue (per step with occupancy in the two-compartment model)
nutrient           D_s  [(wm, 1), (gm, 1)]                   mask: tissue
anisotropic        D    [(axial, 1)]                          mask: all true
```

The isotropic case is exactly today's `diffusivity * (wm_face + gm_face /
ratio)`; the anisotropic case reduces to `diffusivity * face_average(...)`
because a masked average under an all-true mask is the plain average. Both
are bit-identical to today. `_mixture_face_fields` and the anisotropic face
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
killed, absorbed_into, cleared)`, does in this order:

1. `post = t1 >= resection_time` with `t1 = (step_index + 1) * dt`;
   patch the constants with `post_resection` selected by `post`
   (`jax.tree_util.tree_map` with `jnp.where` over the pre and post
   structures);
2. run `update` on the patched constants (the explicit Euler step at the
   pre-step state);
3. survival factor `s = exp(-chemo_kill_rate * E_ct) * exp(-E_rt * n_hits)`
   with `E_ct` the exact exposure of the step (`chemo_exposure`) and
   `n_hits` the number of `rt_times` in `(t0, t1]`;
4. for every field in `killed`: `field <- field * s`; if `absorbed_into`
   is set: `absorbed <- absorbed + field_before * (1 - s)`;
5. for every field in `cleared`: zero it where `post & cavity`.

The two bindings are module-level objects, so the jit cache keeps working:

```python
_treated_single_field_step = partial(
    _treated_step, update=_single_field_update,
    killed=("cell_density",), absorbed_into=None, cleared=("cell_density",))
_treated_two_compartment_step = partial(
    _treated_step, update=_two_compartment_update,
    killed=("proliferative",), absorbed_into="necrotic",
    cleared=("proliferative", "necrotic", "nutrient"))
```

The single-field binding is today's `_stupp_step` verbatim. Each class
declares `_step_func` (its update) and `_treated_step_func` (its binding);
`_run_device_loop` passes one or the other to the time loop according to
the treated predicate. The wrapper itself has no branch.

The two-compartment binding carries the code note required by decision 5:
killed cells are moved into the necrotic field; whether they should vanish
instead is an open question. Consequences worth stating in that note: the
"mass" stopping quantity (P + N) does not see the kill, and the occupancy
mask does not free up.

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
model. The guard is the vanishing-volume check only: `SHRINKAGE_LIMIT`,
`_STOP_SHRINKAGE`, the shrinkage branch of `_time_step`'s stop-kind
select, the `guard_mass_change` diagnostic and the shrinkage error message
are deleted. The guard tuple shrinks to `(code, integrated_density)`. Note
for review: a treated run whose entire seed lies inside the cavity and
never diffuses out is stopped by the vanishing guard after the resection
and reported as an error.

## 5. Backward compatibility

- **Saved configs.** Every `config.json` in `results/` and `runs/` names
  `StuppFKPPSolver` and gives `time_after_resection` without
  `stopping_time`. Both stay valid: the registry gets an alias table
  (`"StuppFKPPSolver"` resolves to `FKPPSolver`), `solver_class`,
  `_resolve_solver` and the constructor's name check compare classes, not
  names, and the horizon rule accepts the old key. `fisher_kpp_jax`
  exports `StuppFKPPSolver = FKPPSolver`. New configs are written with the
  class's own name.
- **`configs/StuppFKPPSolver.json`** stays where it is, because
  `scripts/sensitivity_analysis.py` and
  `scripts/identifiability_experiments.py` use it by path as their base
  config. It is no longer a class's default config; its `_note` entries are
  updated to say so. The three class configs gain the treatment keys at
  the neutral values and `time_after_resection: null`.
- **Scripts.** `StuppFKPPSolver.TREATMENT_KEYS` keeps working through the
  alias. `sensitivity_analysis.growth_config` must drop the treatment keys
  and `time_after_resection` explicitly, since `FKPPSolver.config_keys()`
  now contains them; today it relies on their absence. Docstrings that
  state that the Stupp solver rejects `stopping_time` or that its treatment
  parameters are required are corrected.
- **Existing tests to adapt** (no new tests unless asked):
  `tests/test_stupp.py` constructs `StuppFKPPSolver`, which the alias keeps
  working; `test_treatment_keys` and `test_neutral_treatment_equals_fkpp`
  need their expectations updated (the neutral run is now exactly equal,
  and the treatment keys are shared); `test_validation_errors` loses the
  "at least one fraction" case; `test_read_config_rejects_rt_beta` stays.
  `tests/test_config.py`: `test_default_config_file` classifies the new
  defaulted keys (volumes pinned at null, the rest physical), and
  `test_blank_default_config_fails_at_construction` keeps only the
  anisotropic case. `tests/test_solvers.py::test_dti_guard_exit` asserts
  only that a guard fired, which the vanishing guard still does for a
  decaying tumour, so it should pass unchanged.

## 6. Numerical guarantees

- Untreated runs of all three models compile the same program as today
  and are bit-identical.
- A run with neutral treatment values is exactly the untreated run, because
  it compiles the untreated program.
- The isotropic reference solve keeps its current agreement.
- Treated isotropic runs reproduce today's `StuppFKPPSolver` results: the
  wrapper binding is the old step, and the face construction is the old
  arithmetic.

## 7. Implementation order

1. `operators.py`: add `face_diffusivities`; remove the shrinkage guard and
   its constants; shrink the guard tuple and the scan carry accordingly.
2. `solvers.py`: switch the three models to the builder; add
   `_valid_mask_host` and `_structural_constants`; rename the two steps to
   updates; add `_treated_step` and the two bindings; delete
   `_stupp_step`, `_mixture_face_fields`, the `StuppFKPPSolver` class and
   its TypedDicts; add `_TreatmentConstants`; set `_step_func` and
   `_treated_step_func` on each class.
3. `base.py`: treatment keys and defaults in the shared defaults, the
   horizon rule, the cavity config entry methods, the treatment validation
   including the `null` volume handling, cavity and dose downsampling after
   the model's field preparation, the constants assembly, the treated
   predicate and the step selection, `TREATMENT_KEYS` on the class.
4. `config.py`: the alias table and class-based name checks.
5. `__init__.py`: export the alias; update the module docstring.
6. `configs/`: the three class configs; the notes of
   `StuppFKPPSolver.json`.
7. `scripts/`: `growth_config`; docstring corrections.
8. `tests/`: the adaptations listed in section 5.
9. Run `scripts/run_reference_solves.py` and the test suite; confirm the
   reference agreement is unchanged and that a config from a completed
   sweep still loads and re-solves.

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
  needed outside the solver first.
