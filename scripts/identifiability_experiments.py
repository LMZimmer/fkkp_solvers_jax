#!/usr/bin/env python
"""Five identifiability experiments on the pre-resection growth time of
the Stupp-protocol forward model, fisher_kpp_jax.StuppFKPPSolver, on
atlas tissue maps, over a cohort of formed-front, patient-sized tumours
grown from small seeds, stratified by front width and by how far the
front has travelled beyond the seed (maturity).

Background. The plain Fisher-KPP model du/dt = D lap u + rho u (1 - u)
maps the parameters (D, rho, T) to the same field as (lambda D,
lambda rho, T / lambda) for any lambda > 0: rescaling time by lambda
multiplies both rates by lambda and divides the growth time by it, so a
density observed once, at an unknown age, fixes only the products D T
and rho T. In the growth band of the sensitivity script (v = 2 sqrt(D rho)
the front speed, lambda_f = sqrt(D / rho) the front width) the rescaling
is (lambda v, lambda_f, T / lambda): the speed and the age trade off, the
width is untouched. The treated model does not have this invariance. Its
treatment happens on calendar days counted from the surgery (the base
config's rt_times and chemo_times are shifted with resection_time, the
concomitant block starts 14 days after it, the adjuvant cycles 81 days
after it), so a tumor twice as fast that is resected at half its age is
irradiated and dosed for the same number of days while it regrows twice
as fast. Experiment 0 measures how far the treated frames move under the
rescaling that leaves the pre-resection field unchanged. Experiment 1
asks, locally at each truth, whether the growth time T_r has an
effect on the observations (smoothed threshold maps of the density at
five days) that the other parameters (v, lambda_f, alpha, k_ct, and the
seed's peak and width) cannot reproduce: the Fisher information of the
observations with a noise model of threshold and registration
uncertainty, its weakest direction and the Cramer-Rao error of log T_r
per set of observed days. Experiment 2 asks, globally, whether a seed of
a different mass and width grown for a wrong time T_0 = T_r - delta_a /
rho (a deficit of delta_a e-folds against the truth's age a = rho T_r)
can reproduce the state the truth reaches at T_r, and how the treated
frames of that substitute then differ from the truth's: the linear
composition rule says it can when the growth is still linear (the seed
grown for T_r - T_0 days is again a Gaussian), the logistic saturation
and the tissue boundaries say it cannot for old tumors. Experiment 3 is
its mirror: the seed is held at a small standard seed and the growth
time is fitted, with the front width at the truth's or fitted along, so
that the fitted T_r says what age a standard seed needs to reach the
truth's pre-operative state, and the treated frames say what that wrong
age costs. Experiment 4 profiles that fit over the standard seed's
width.

Conventions. A subcommand solves in one Python process (the jitted
time scan is cached across solves of the same step count) at precision
f64; ``--gpus`` names the CUDA devices like the sensitivity script's
slots ('' the CPU): with one device the subcommand solves in this
process, with several the fisher, substitute, seedfix and profile
subcommands dispatch one worker process per device (see Devices below);
the design and the invariance take one device. The parameters of the
growth band are the front speed
v = front_speed_mm_per_day and the front width lambda = front_width_mm;
D = v lambda / 2 and rho = v / (2 lambda) follow from the script's growth
derivation (``growth_parameters``). The fit and the derivatives are in
theta = (log v, log lambda, log T_r, log alpha, log k_ct) with
alpha = rt_alpha and k_ct = chemo_kill_rate. The cavity and the dose map
of a truth come from its growth stage exactly as the sensitivity script
derives them (``treatment_maps``: the cavity is the density at
resection_time at or above 0.6, the dose map the cavity grown by 15 mm
carrying 2 Gy times the 30 fractions, 60 Gy in total; --cavity-threshold,
--rt-margin-mm, --rt-dose-per-fraction of the design) and are then held
fixed as data of every run compared with that truth. The tissue maps are
the script's default atlas (--white-matter-pbmap, --gray-matter-pbmap:
the BraTS MNI152 maps of PredictGBM, 182 x 218 x 182 at 1 mm). The base
config is the script's default, fisher_kpp_jax/configs/StuppFKPPSolver.json
(--config), whose resection_cavity and rt_dose must be null and which
must set a time step (the fixed mode's, see Time step below); the
design sets its precision to f64, its
gaussian_seed_floor to 0 (recorded in spec.json and base_config.json;
the base config's 0.1 would erase the light seeds of experiment 2 and
the seed derivation requires the peak above the floor) and, with
--smoke, its resolution_factor to 0.25 (the solver's zoom factor: 4 mm
voxels; the results are upsampled to the 1 mm grid). The seed voxel is
one fixed seedable voxel for every patient (--seed-voxel i,j,k; default
DEFAULT_SEED_VOXEL = (132, 103, 90), right-hemisphere deep white matter
on the atlas, the voxel scripts/run_stupp_synthetic.py uses; snapped to
the nearest voxel with wm + gm >= min_tissue_fraction,
``seed_geometry``, which leaves it where it is on the atlas; a grid too
small to hold it, such as the tests' 24^3 phantom, falls back to the
base config's fractions, the grid centre); its fractions (v + 0.5) / n
go into every config. The time step is the patient's (Time step below;
steps_per_day in every config). The schedule of every run is the base
config's truncated to the run's horizon (``truncate_schedule``) and
shifted by resection_time - 100 (``SHIFTED_TIME_KEYS``), so the
treatment keeps its offsets after surgery; the treated stage is stepped
on the growth stage's grid of times (``align_treated_config``: n_steps =
n_growth + ceil(time_after_resection / dt), resection_time stated as
(n_growth - 1/2) dt so that step n_growth resects; the treated stage
checks that the solver kept that dt and refuses to continue otherwise).
Every run's config
is saved (config.json in its directory: the aligned config with the
maps' paths and the frames' days as snapshot_times) so that
StuppFKPPSolver(read_config(path)) reproduces it; the solves themselves
take the loaded tissue arrays in place of the paths, which gives the
same fields and skips reading the NIfTIs. Fields are saved as float32
NIfTI rounded for storage (``round_field``: 7 mantissa bits, values
below 1e-10 zeroed) with the atlas affine; every Jacobian, objective and
metric is computed from the unrounded fields in memory. Runs are
resumable: a patient whose result JSON exists (runs/fisher/<patient>/
fisher.json, runs/substitute/<patient>/substitute.json,
runs/seedfix/<patient>/seedfix.json, runs/profile/<patient>/
profile.json; within experiments 2-4 also a T_0, lambda mode or sigma_0
whose row.json exists), an invariance experiment whose invariance.json
exists, are skipped, and the CSVs and figures are assembled from every
record present; the design's size screening resumes from its screen/
records (a design directory holding spec.json is never overwritten).
Nothing is written outside <output-dir>/<name>/.

Frames. A treated run records the state at named moments after the
resection (FRAME_MOMENTS; ``frame_days``): each frame is the state after
the last step whose end lies at least half a step before the moment,
the sensitivity script's rounding (``snapshot_days`` takes the moment as
resection_time + offset + 1, so it receives moment - 1).
  pre    the last step end before resection_time, i.e. after step
         n_growth - 1 at (n_growth - 1) dt, before the resection step
         zeroes the cavity (the pre-resection state; it equals the
         growth stage's state one step before the density the cavity is
         thresholded from: 2 h earlier at 12 steps/day, 12 h at 2);
         tests/test_identifiability_experiments.py checks that it is
         nonzero inside the cavity and that the state one step later is
         zero there
  d34    the end of the Sunday closing the third CRT week (offset 34,
         moment 35: before Monday's fraction), as the script's mid_crt
  d55    the end of the Sunday closing the sixth CRT week (offset 55,
         moment 56), the script's end_crt; the design checks both
         against ``crt_snapshot_offsets`` of the base schedule
  d80    the end of day 80, the day before the first adjuvant dose at
         offset 81 (the design checks the base schedule), moment 81
  d120   the moment resection_time + 120, the horizon of experiment 1
         (between half a step and one and a half steps before it; no
         dose is scheduled within a day of it)
  d180   the moment resection_time + 180, the horizon of experiment 2
Experiment 1 records pre, d34, d55, d80, d120; experiment 2 also d180.
Experiments 3 and 4 record the six frames of experiment 2.
Each frame is saved as <name>_cell_density.nii.gz and frames.json holds
the requested and the recorded days.

Metrics (``compare_fields``, one function for every experiment),
between a field u and a reference u* on the tissue mask (wm + gm >=
min_tissue_fraction), with dV the voxel volume:
  dice_core, dice_edema        Dice of {u >= 0.6} and of {u >= 0.3}
                               against the same sets of u*; NaN when
                               both sets are empty
  assd_core_mm, assd_edema_mm  the average symmetric surface distance in
                               mm of each iso-surface: the mean, over
                               the surface voxels of both sets (a voxel
                               with a face neighbour outside the set), of
                               the Euclidean distance
                               (distance_transform_edt with the zooms) to
                               the other set's surface; NaN when either
                               set is empty
  rel_l2_log                   ||log(u + 1e-6) - log(u* + 1e-6)||_2 /
                               ||log(u* + 1e-6)||_2
  rel_l2                       ||u - u*||_2 / ||u*||_2
  max_abs_diff                 max |u - u*|
  mass                         dV sum u (the total mass of u)
  mass_out_of_field            dV sum u over the voxels where the dose
                               map is zero (the out-of-field tail)
  mass_beyond_edema            dV sum u over the tissue voxels where
                               u* < 0.3 (the mass outside the
                               reference's edema)
  ref_mass, ref_mass_out_of_field, ref_mass_beyond_edema
                               the same three of u*
  mass_rel, mass_out_of_field_rel, mass_beyond_edema_rel
                               <name> / ref_<name> - 1 (NaN for a zero
                               reference)
  log_vol_ratio_core, log_vol_ratio_edema
                               log10(V + dV) - log10(V* + dV) of the
                               volumes of {u >= 0.6} / {u >= 0.3} and of
                               the same sets of u*
  qoi_<name>                   the script's QoIs of u (``compute_qois``:
                               mass, log10_mass, V_core, V_edema,
                               log10_V_core, log10_V_edema, r95_core,
                               r95_edema, centroid_drift, centroid_x/y/z_mm,
                               R_g, anisotropy, log10_anisotropy,
                               wm_fraction, n_core, n_edema, voxel_volume)
                               with the cohort's seed voxel and the atlas
                               white-matter map

Subcommand design (the cohort). The search space is the script's
(--search-space, fisher_kpp_jax/search_spaces/
stupp_identifiability_search_space.json): the growth group
(front_speed_mm_per_day 0.03-0.25 mm/day and front_width_mm 1-8 mm,
both log-uniform), the seed group (seed_peak_density 0.6-1 uniform,
seed_sigma_mm 1-5 mm log-uniform: the width in mm, independent of the
front width), chemo_kill_rate (5e-5-1e-2 /day per mg/m^2, log), rt_alpha
(5e-4-0.1 /Gy, log), the fixed overrides chemo_decay_rate 9.24 /day,
rt_alpha_beta_ratio 8 Gy and diffusivity_ratio 10 (written into the base
config) and the script factor growth_efolds (1-12, log-uniform), the age
a = rho T_r: an entry carrying "script": true is a factor of this script
and not a solver parameter, so ``load_script_search_space`` strips it
before the sensitivity script's loader sees the file and spec.json
records it (script_factors). The seed position is the fixed voxel. A
scrambled Sobol' sequence (scipy.stats.qmc.Sobol, seed --design-seed,
default 1) gives 2 ** --log2-candidates points (default 1024) on the
unit cube of the seven factors (SAMPLED_FACTORS' order: v, lambda,
growth_efolds, chemo_kill_rate, rt_alpha, seed_peak_density,
seed_sigma_mm), transformed with the ranges (``transform_factor``) and
derived with the groups (``SearchSpace.derive``: white_matter_diffusivity,
rho, gaussian_seed_mass, gaussian_seed_diffusion_time,
seed_enhancing_radius_mm, s = sigma / lambda); resection_time =
growth_efolds / rho. A candidate whose seed is wider than
SEED_WIDTH_CAP = 1.5 front widths (seed_sigma_mm > 1.5 front_width_mm)
is rejected first (seed_wide: the pre-operative shape would be the
seed's, not the growth's), then one with resection_time below 5 days or
above --tr-max (default 1000); both are counted. Per candidate the
maturity
  maturity = 2 growth_efolds front_width_mm^2 / seed_sigma_mm^2
           = ell lambda / sigma^2
with ell = v T_r = 2 a lambda the front travel: the travel over the
seed's forgetting distance sigma^2 / lambda (``maturity``; a mature
truth has forgotten its seed, an immature one still carries it), and
the total log kill of the 120-day schedule
  Lambda = n d alpha (1 + d / (alpha/beta)) + k_ct D_tot / gamma
         = 60 alpha (1 + 2 / 8) + k_ct 4900 / 9.24
with n = 30 fractions of d = 2 Gy, D_tot the chemotherapy dose within the
120-day horizon of experiment 1 (4 900 mg/m^2: 42 x 75 + 5 x 150 +
5 x 200) and gamma = chemo_decay_rate, and the visibility margin
120 rho - Lambda (the e-folds the untreated regrowth gains on the log
kill by day 120; its sign says whether the truth regrows past the
treatment by then; a design column, not a cell split). The cells
(``cell_of``; LAMBDA_SPLIT = 2 mm, MATURITY_SPLIT = 15):
  compact_immature    front_width_mm <= 2 mm, maturity < 15
  compact_mature      front_width_mm <= 2 mm, maturity >= 15
  broad_immature      front_width_mm > 2 mm,  maturity < 15
  broad_mature        front_width_mm > 2 mm,  maturity >= 15
Size screening (``screen_cohort``): per cell the admissible candidates
are taken in sequence order and each one gets a growth-only solve (the
truth's growth path, ``solve_growth`` at the candidate's time step) to
its resection_time; from its density on the 1 mm grid (the solver upsamples
a coarser grid) the equivalent-sphere radii r_core_mm of {u >= 0.6} and
r_whole_mm of {u >= 0.3} on the tissue mask ((3 V / 4 pi)^(1/3)) and the
volume ratio whole / core (``size_screen``); a candidate is accepted
when r_core_mm lies in --r-core-band, r_whole_mm in --r-whole-band and
the ratio in --ratio-band ("lo,hi"; the defaults 5,25 / 10,35 / 1.2,8
are provisional), one with an empty core is rejected, and a solve that
fails rejects the candidate (counted; three failures in a row stop the
design). The screening runs on one device (--gpus, '' the CPU), is
resumable (screen/c<candidate>/screen.json holds the candidate's
factors, step, sizes and wall time; a record of another candidate under
the same index, i.e. another seed, search space or time step, is
refused) and stops when every cell holds --patients-per-cell patients
(PATIENTS_PER_CELL = 4, 16 patients; 1 with --smoke); a cell that cannot
be filled from the points fails the design with the counts. The
patients are p00-p15 in cell order, the first of each cell flagged for
the finite-difference check of experiment 1 (fd_check).
  design.csv    patient, cell, candidate (the Sobol' index), u_<factor>
                for the seven factors, the seven factors
                (front_speed_mm_per_day, front_width_mm, growth_efolds,
                chemo_kill_rate, rt_alpha, seed_peak_density,
                seed_sigma_mm), resection_time, white_matter_diffusivity,
                rho, gaussian_seed_mass, gaussian_seed_diffusion_time,
                seed_enhancing_radius_mm, s (= sigma / lambda), rho_T_r
                (= growth_efolds), ell_mm (= v resection_time),
                log_kill_rt, log_kill_ct, log_kill_total,
                visibility_margin, maturity, r_core_mm, r_whole_mm,
                whole_core_ratio, R_over_lambda (= r_whole_mm / lambda),
                steps_per_day, dt (the patient's time step), fd_check
  spec.json     the settings: base config and search space paths, the
                tissue maps, grid shape, precision, gaussian_seed_floor,
                resolution_factor, smoke and the smoke settings, the
                design seed and candidate count, patients_per_cell, the
                cells and their splits (lambda_split_mm,
                maturity_split, the maturity formula, the seed width cap;
                visibility_horizon_days), the sampled and the script
                factors with their ranges, the fixed overrides and
                parameters, the seed voxel (seed_target_voxel and
                seed_target_source: "default" for DEFAULT_SEED_VOXEL,
                "base_config_fractions" for the fallback, "argument" for
                --seed-voxel; the snapped seed_voxel,
                seed_snap_distance_voxels, seed_fractions,
                default_seed_voxel), the tissue threshold and seedable
                count, the time step (the base config's setting, dt_mode
                and the modes' formulas), the treatment derivation settings,
                the schedules within both horizons (``truncate_schedule``
                records), the log-kill formula and its constants, the
                visibility formula, the screening (tr_min, tr_max, the
                bands, the levels, the device, the counts: candidates,
                rejections by seed width and by resection_time, solves,
                acceptances,
                failures, and per cell the candidates, the admissible
                ones, the screened, accepted and failed ones and the
                rejections by reason; the wall time), the frame moments,
                the CRT snapshot offsets, the horizons and frames per
                experiment, the lambda set, the experiment 1-4 settings
                and the summaries' maturity strata
  screen/c<candidate>/screen.json   the screening record of every solved
                candidate (accepted or not)
  base_config.json, search_space.json
  configs/<patient>.json   the truth config of experiment 1 (120-day
                horizon, maps null, schedule truncated and shifted)

Time step (--dt-mode fixed | stability, default stability; the design
records it as spec.json's dt_mode and the patients' steps in
design.csv, and substitute, seedfix, profile and all must name the same
mode, ``check_dt_mode``, so that every solve of a patient, the truth,
the screening, every fit evaluation and every treated run, uses that
patient's step; the fisher and invariance subcommands step at the fixed
12 steps/day whatever the mode).
  fixed      BASE_STEPS_PER_DAY = 12 steps/day for every patient (the
             base config's dt = 1/12 day).
  stability  per patient dt = min(DT_MAX, DT_SAFETY dx^2 / (6 D_wm)),
             DT_MAX = 0.5 d, DT_SAFETY = 0.5, dx the grid spacing (1 mm
             on the atlas, 4 mm with --smoke) and D_wm the patient's
             white_matter_diffusivity, rounded down to an integer number
             of steps per day (steps_per_day = ceil(1 / dt)), raised to
             the solver's own estimate at the truth's resection_time
             (ceil(max(8 D T_r / dx^2 + 100, 1.1 rho T_r) / T_r),
             ``solver_step_estimate``) so that the truth is stepped as
             requested, and capped at 12 steps/day, the floor of the
             step (``steps_per_day_for``). On the atlas that is
             ceil(12 D) steps/day between 2 and 12: 2 for D below
             1/6 mm^2/day (about half of the admissible candidates), 12
             at the corner D = 1. DT_MAX is half a day because the frame
             rounding needs a step of at most half a day.
The solver takes a run's step as ceil(horizon steps_per_day) steps, so
the effective step divides the horizon exactly and is at most
1 / steps_per_day (``requested_steps``; a warning names the rounding).
The schedule's sessions and fractions lie on whole days after the
resection, so with a whole number of steps per day every impulse falls
at a step boundary to that rounding, as at 12 steps/day; the cavity and
dose maps are derived from the growth stage's final state as before. A
solve whose horizon needs more than the patient's steps (a fit
evaluation at a short T_0 or T_r, where the solver's estimate of
8 D T / dx^2 + 100 steps exceeds T steps_per_day) is refined by the
solver to its estimate, as it is in the fixed mode; the record then
carries the solver's dt beside the patient's steps_per_day, and the
truth's record flags it (dt_refined, ``dt_refined``: more steps than
``requested_steps``; the screening's screen.json too, and the design
reports any accepted patient it happened to, which needs a T_r below 25
days). Every fit record and every row of experiments 2-4 carries
growth_n_steps, steps_per_day (the patient's) and dt (the solver's).

Subcommand dt-check (--patients ids or ranges, default the fd_check
patients; one device). Per patient (``dt_check_patient``) the truth of
experiments 2-4 (the growth stage, its own cavity and dose map, the
treated stage to d180 with the six frames) once at the fixed step and
once at the stability step (``steps_per_day_for`` per mode, whatever
the design's mode; runs/dt_check/<patient>/fixed/ and stability/), and
the stability run's pre and d180 frames compared with the fixed run's
(``compare_to``, the fixed run's dose map).
  dt_check.csv     patient, cell, maturity, resection_time,
                   white_matter_diffusivity, grid_spacing_mm, per mode
                   steps_per_day, dt, dt_refined, n_growth, n_steps and
                   wall_time_s (<key>_fixed, <key>_stability), and
                   pre_ / d180_ dice_core, dice_edema, mass_rel,
                   mass_beyond_edema_rel of the stability run against
                   the fixed run
  runs/dt_check/<patient>/   fixed/, stability/ (each as a truth
                   directory) and dt_check.json (the record with the
                   row, the resume marker)

Subcommand invariance (experiment 0). One patient: the base config's D,
rho and resection_time (--white-matter-diffusivity, --rho,
--resection-time override them; the base values are D = 0.0968 mm^2/day,
rho = 0.0194 /day, T_r = 100 days), its seed (peak 0.8, sigma 6.3 mm),
alpha and k_ct, at the cohort's seed voxel. For every lambda in
--lambdas (default 0.25, 0.5, 1, 2, 4) the run (lambda D, lambda rho,
T_r / lambda), i.e. (lambda v, lambda_f, T_r / lambda), is solved three
ways (``run_invariance``):
  growth_fixed_n   (A) growth-only FKPPSolver with n_steps fixed to the
                   lambda = 1 run's count (dt = T_r / (lambda n)): the
                   step products dt D, dt rho are the same numbers, so
                   the fields agree to floating-point rounding (exactly,
                   for a power-of-two lambda)
  growth_fixed_dt  (B) growth-only FKPPSolver at the base config's 12
                   steps/day (dt fixed, n = 12 T_r / lambda): the
                   discretization error changes with lambda
  treated          (C) StuppFKPPSolver with the cavity and dose map
                   derived once from the lambda = 1 growth stage
                   (runs/invariance/maps/) and reused for every lambda,
                   resection_time T_r / lambda, the schedule truncated
                   to the 120-day horizon and shifted with it, stepped on
                   (B)'s grid, recording the five frames of experiment 1
Each lambda is compared with lambda = 1 by ``compare_fields``: (A) and
(B) on their final field (the pre-resection field, snapshot "pre"), (C)
on every frame. (A) is expected at rounding level; the max abs
difference is reported.
  invariance.csv   lambda, run_type, snapshot, n_steps, dt, resection_time
                   (T_r / lambda), wall_time_s and the metrics
  figures/invariance_metrics.{png,pdf}, invariance_qois.{png,pdf}
                   every metric against lambda (log axis), one line per
                   (run type, snapshot)
  runs/invariance/lambda_<lambda>/<run_type>/   config.json, result.json,
                   the field(s); maps/ the shared cavity and dose

Subcommand fisher (experiment 1). Per patient (``fisher_patient``):
  1. The truth: the growth stage at the fixed step of 12 steps/day,
     whatever the design's dt mode (the scaled-dt T_r column below steps
     at dt e^{+-h}, which a half-day step would push past the frame
     rounding's limit; runs/fisher/<patient>/truth/growth/), its density
     at resection_time
     (pre_resection_cell_density.nii.gz), the maps derived from it
     (resection_cavity.nii.gz, rt_dose.nii.gz, treatment.json) and the
     treated stage over the 120-day horizon with the frames pre, d34,
     d55, d80, d120.
  2. The observation region: the tissue voxels within 30 mm
     (--observation margin OBSERVATION_MARGIN_MM) of the union over the
     five frames of the truth's 0.3 iso-surface (the Euclidean distance
     to the union for a voxel outside it, to its complement for a voxel
     inside it; ``observation_region``); observation.json records its
     voxel count and bounding box. The observation vector of a run is
     the smoothed indicator s(u) = 1 / (1 + exp(-(u - c) / 0.02)) at
     c = 0.6 and c = 0.3 of each frame on the region's voxels, in the
     row order (frame, level, voxel): 10 rows per voxel.
  3. The noise: K = --draws draws (default 64, 8 with --smoke), each
     with the thresholds c drawn uniformly from [0.50, 0.85] and
     [0.19, 0.50] (one pair per draw, shared by its frames) and, per
     frame, a random displacement field (three independent Gaussian
     random fields: white noise smoothed with a Gaussian kernel of
     standard deviation 5 mm, the correlation length, scaled to a
     standard deviation of 1 mm per component); the truth's frame is
     sampled at x + d(x) by scipy.ndimage.map_coordinates (linear) on
     the region's voxels and the indicators are taken at the drawn
     thresholds. Sigma = diag(the per-row variance over the draws,
     floored at 1e-3) (``noise_variance``; sigma.npz; the draws are
     seeded with 1000 design_seed + the patient's index). Far from any
     iso-surface the indicators do not move and the floor applies (about
     99 % of the rows on the atlas); near a surface the variance is that
     of a 1 mm registration error and a threshold uncertainty.
  4. The Jacobian J by central differences in theta with the step
     h = 0.05 (``perturbations``): column i is (y(theta + h e_i) -
     y(theta - h e_i)) / (2h) with y the observation vector of a treated
     run of the perturbed patient (each run a separate directory
     <column>_plus / <column>_minus with its config, frames and result)
     with the truth's cavity and dose, the schedule shifted with its
     resection_time, and the truth's seed (the log lambda column changes
     D and rho at a fixed absolute seed width). The T_r column is stepped
     two ways, both computed and analysed (fisher.csv's t_r_column):
       scaled_dt  T_r e^{+-h} with the truth's step count n_growth kept,
                  so dt' = T_r e^{+-h} / n_growth and the treated stage
                  is aligned to dt' (n_steps = n_growth + ceil(120 / dt'));
                  the growth invariance then holds exactly (the pre
                  frame of the T_r run equals that of the v run to
                  rounding, as (A) of experiment 0), so on the
                  pre-resection observations the invariance direction is
                  a null direction of J to rounding and any T_r
                  information in the treated frames is the calendar
                  effect plus the change of the treated stage's
                  discretization error with dt' (5 % of it). This is the
                  primary column (the heatmaps).
       fixed_dt   T_r e^{+-h} rounded to a whole number of the truth's
                  steps (T_r' = round(T_r e^{+-h} / dt) dt, n_growth' =
                  T_r' / dt; the effective step is log(T_r' / T_r), and
                  the divided difference uses h_+ + h_-); the whole run
                  stays on the truth's time grid, but the pre-resection
                  field then differs from the v run's by the growth
                  stage's discretization error (the (B) difference of
                  experiment 0), which shows up as a spurious T_r
                  information of order 1e-3 in |W e| / |W e_1| on the
                  pre-resection observations alone (a Cramer-Rao error of
                  a few units instead of infinity).
     A second Jacobian adds two columns of the seed: the peak density
     (additive step 0.05; the plus step is shortened so that the peak
     stays at most 1 and dropped, leaving a one-sided difference against
     the truth, when the peak is 1) and log sigma (step 0.05), converting
     (peak, sigma) to (m, w) with the script's seed derivation
     (``seed_parameters``: w = sigma^2 / 2, m = peak (4 pi w)^(3/2)).
     For the fd_check patients (the first of each cell) the five theta
     columns are repeated at the step 0.025 (scaled_dt T_r column) as a
     finite-difference check: fisher_fd_check.csv reports per column
     rel_diff = ||W_h - W_{h/2}|| / ||W_{h/2}|| of the weighted columns
     and |W e| / |W e_1| and the Cramer-Rao error of log T_r at both
     steps for the sets (a) and (f). On the atlas the columns agree to
     0.4-1.5 % (p00); on a small phantom that the tumor saturates the
     indicators are far from linear in a 5 % step and the check reports
     tens of per cent.
  5. The analysis (``fisher_analysis``) of the weighted Jacobian
     W = Sigma^{-1/2} J restricted to each observation set of frames
       (a) pre; (b) pre + d120; (c) pre + d55 + d120;
       (d) pre + d34 + d55 + d120; (e) pre + d80 + d120; (f) all five.
     With the five theta columns: |W e| for the growth invariance
     direction e = (1, 0, -1, 0, 0) / sqrt 2 (over log v, log lambda,
     log T_r: the rescaling multiplies v by lambda, leaves the front
     width and divides T_r; the request's (1, 1, -1) / sqrt 3 is the
     same direction over (log D, log rho, log T_r)) and |W e_1| for
     scale; the eigenvalues of F = W^T W (eig_1 <= ... <= eig_5), the
     loadings of the weakest eigenvector (weak_<theta>; on set (a) the
     null space is three-dimensional, e, e_alpha and e_k_ct, since
     nothing before the resection depends on the treatment, and the
     reported vector is one of it) and the condition number; the
     Cramer-Rao standard errors sqrt((F^-1)_ii) of every theta parameter
     (cr_<theta>; cr_log_T_r = sqrt((F^-1)_33); an eigenvalue below
     1e-14 of the largest counts as zero and a parameter with a loading
     on its eigenvector gets an infinite error, ``cramer_rao``); the
     residual fraction of the T_r column after its least-squares
     projection onto the other four columns (resid_frac_T_r =
     ||W_T - P W_T|| / ||W_T||). With all seven columns: the Cramer-Rao
     errors of log T_r and of the two seed parameters (cr7_log_T_r,
     cr7_peak, cr7_log_sigma), the residual fraction of the T_r column
     after projection onto the other six (resid7_frac_T_r, with the
     norm norm7_T_r), and that fraction split by region and by day
     (fisher_regions.csv: the residual and the column restricted to the
     rows of the split, the projection being the global one; a region
     is core (the truth's density at that row's frame and voxel >= 0.6),
     rim (0.3 <= u* < 0.6) or out_of_field (the voxel outside the dose
     support); a day is a frame; a subset's fraction may exceed 1).
  Outputs:
  fisher.csv           one row per patient x t_r_column x observation
                       set: patient, cell, t_r_column, observation_set,
                       frames (joined with +), resection_time, n_growth,
                       dt, n_region_voxels, n_observations, n_rows,
                       We_norm, We1_norm, We_ratio, eig_1..eig_5,
                       weak_log_v .. weak_log_k_ct, cond_F, cr_log_v ..
                       cr_log_k_ct, resid_frac_T_r, resid7_frac_T_r,
                       norm7_T_r, cr7_log_T_r, cr7_peak, cr7_log_sigma
  fisher_regions.csv   patient, cell, t_r_column, observation_set, split
                       (region | day), name, resid7_frac, norm_T_r, n_rows
  fisher_fd_check.csv  patient, cell, column, step, step_half, rel_diff,
                       We_ratio_a, We_ratio_a_half, We_ratio_f,
                       We_ratio_f_half, cr_log_T_r_f, cr_log_T_r_f_half
  fisher_runs.csv      the metrics of every perturbed run against the
                       truth per frame: patient, cell, run, column, sign,
                       variant, step, frame, n_steps, dt, the metrics
  runs/fisher/<patient>/   truth/ (growth/, the maps, the pre-resection
                       density, config.json, result.json,
                       final_cell_density.nii.gz, the frames,
                       frames.json), <column>[_<variant>]_{plus,minus}/
                       and the fd_check runs (config.json, result.json,
                       final field, frames, frames.json),
                       observation.json (region, noise settings, the
                       rows at the floor, the size of W), sigma.npz (the
                       variance, float32, (frames, levels, voxels)),
                       W.npz (the seven-column W in float32 with the
                       column names, frame names, levels, the region's
                       voxel indices and each row's frame; written only
                       below 500 MB, about 116 MB for a 414 000-voxel
                       region), fisher.json (everything above, the
                       resume marker)
  figures/fisher_heatmap_<key>.{png,pdf}   patients x observation sets
                       for the scaled_dt column: cr_log_T_r, cr7_log_T_r
                       (log10 colour), resid_frac_T_r, resid7_frac_T_r,
                       We_ratio (log10 colour)
  fisher_summary.json  the assembly record: the patients and cells with
                       a record, the row counts and the device split of
                       the last dispatch (Devices below)

Subcommand substitute (experiment 2). The horizon is 180 days after
surgery for the truth and the substitutes (the base config's six-cycle
schedule truncated to it: 62 sessions, 6 900 mg/m^2, all 30 fractions;
cycle 4 ends at offset 169, cycle 5 is dropped). Per patient
(``substitute_patient``): the truth as in experiment 1 with the frames
pre, d34, d55, d80, d120, d180 and the observation region of its six
frames; then for each deficit delta_a in --delta-a (default -1, -0.5,
0.5, 1 e-folds) the growth time T_0 = T_r - delta_a / rho (delta_a > 0:
T_0 earlier than the truth; ``deficit_schedule``), skipped when T_0 lies
below 5 days or above the design's --tr-max (recorded in substitute.json
and as a row with NaN metrics), and per objective, with v and lambda at
the truth, the seed position fixed and the floor 0, the seed (logit of
the peak on (0.05, 1], log sigma; peak = 0.05 + 0.95 sigmoid(z),
``bounded_from_unbounded``) is fitted by scipy's Nelder-Mead (--maxfev
evaluations at most, default 60 (SUBSTITUTE_MAXFEV), 20 with --smoke;
the initial simplex x0, x0 + (0.5, 0), x0 + (0, 0.2); xatol 1e-3,
fatol 1e-6) so that the growth-only field at T_0 (FKPPSolver at the
patient's step) matches the truth's
density at resection_time on the region's voxels (``fit_seed``):
  B   1 - the mean over c in {0.6, 0.3} of the soft Dice
      2 sum s s* / (sum s^2 + sum s*^2) of the smoothed indicators
      (1 for identical fields, the Dice for binary ones)
(``fit_objective`` also knows A, the relative L2 of log(u + 1e-6) -
log(u* + 1e-6) on the region, which OBJECTIVES does not run). The
initial guess is the linear composition rule (``composition_seed``):
m' = m e^{rho (T_r - T_0)}, w' = w + D (T_r - T_0) (a Gaussian seed of
diffusion time w grown for T_r - T_0 days by the linearised equation is a
Gaussian of diffusion time w' with the mass m'), as (peak, sigma) =
(m' / (4 pi w')^(3/2), sqrt(2 w')), falling back to the truth's seed when
w' is not positive (T_0 > T_r + w / D); a peak above 1 (the truth's
saturated core cannot be a Gaussian of peak at most 1) enters the fit at
the top of the range (rule_peak keeps the rule's value, initial_peak the
one used). The best evaluation is the fitted seed (fit.json holds the
history). The treated stage is then run from it with resection_time T_0,
the truth's cavity, dose, alpha and k_ct, the schedule truncated to 180
days and shifted for T_0, on its growth stage's grid, recording the six
frames, and compared with the truth's frames by ``compare_fields``.
  substitute.csv   one row per patient x delta_a x objective (a skipped
                   pair with its reason in skipped and NaN metrics), plus
                   one row per patient with objective "truth" (delta_a 0,
                   T_0 = T_r, the truth against itself, for the reference
                   masses): patient, cell, delta_a, T_0, rho_T_0, rho_T_r,
                   maturity, objective, skipped, rule_peak, rule_sigma_mm,
                   initial_peak, initial_sigma_mm, initial_from_rule,
                   fitted_peak, fitted_sigma_mm, n_evaluations,
                   objective_initial, objective_achieved, growth_n_steps,
                   steps_per_day, dt, wall_time_s (the fit), and
                   <frame>_<metric> for the six frames
  figures/substitute_<frame>_objective_B.{png,pdf}   for d120 and d180:
                   every comparison metric against delta_a (linear
                   axis), the (patient, delta_a) points coloured by cell
                   and one line per cell through the cell's median at
                   each delta_a
  runs/substitute/<patient>/   truth/ (as in experiment 1, six frames),
                   observation.json, T0_<T_0>/<objective>/ (fit.json,
                   row.json, run/ with the substitute's treated run;
                   T_0 with one decimal), substitute.json
  substitute_summary.json   the assembly record (as fisher_summary.json)
                   with the delta_a values, the skipped pairs and the
                   medians (``substitute_medians``): per objective and
                   delta_a, per cell, over all patients and per stratum
                   maturity >= 15 / < 15 (maturity_ge_15, maturity_lt_15;
                   ``row_groups``), the row count, the skipped rows
                   (excluded) and the median of every d120_ and d180_
                   metric

Subcommand seedfix (experiment 3). The mirror of experiment 2: the seed
is held at a small standard seed, --seed-peak (default 0.6) and
--seed-sigma-mm (default 2 mm; SEED_SOURCE "argument", recorded in
spec.json's seedfix entry with the defaults, in seedfix_summary.json and
per row as seed_peak and seed_sigma_mm) and the growth time is fitted
instead. Per patient (``seedfix_patient``): the truth as in experiment 2
(runs/substitute/<patient>/truth is read back when it holds the six
frames, ``load_truth_run``, its stored fields being the float32 ones
rounded for storage; otherwise the truth is solved into
runs/seedfix/<patient>/truth) and the observation region of its six
frames; then per lambda mode of --lambda-mode (fixed | free | both, the
default fixed; the fixed mode runs first), with objective B, the seed
voxel at the truth and the growth-only solve path of ``fit_seed``
(FKPPSolver at the patient's step, n_steps = ceil(T_r steps_per_day), so
every evaluation has its own step count and compiles its own scan):
  fixed   with v and lambda at the truth, log T_r is fitted on
          [log 5, log 3000] (--tr-bounds, days) by scipy's bounded Brent
          search (minimize_scalar; at most --maxfev evaluations, xatol
          1e-3 in log T_r; the search starts from the bracket's
          golden-section point, never from the truth's T_r)
          (``fit_growth_time``); directory B/
  free    with v at the truth, (log T_r, log lambda) is fitted by
          Nelder-Mead (the options of ``fit_seed``: --maxfev evaluations
          at most, the initial simplex x0, x0 + (0.5, 0), x0 + (0, 0.2),
          xatol 1e-3, fatol 1e-6) in the unbounded coordinates of
          ``bounded_from_unbounded``, log T_r on [log 5, log 3000] and
          log lambda on [log 0.5, log 8] mm (LAMBDA_BOUNDS),
          D = v lambda / 2 and rho = v / (2 lambda) re-derived from the
          current lambda at every evaluation; the start is the fixed
          mode's optimum (T_r*, the truth's lambda) when its fit.json
          exists in the patient's directory and did not hit a bound,
          else the geometric midpoints of both brackets
          (``fit_growth_time_free``); directory B_free_lambda/
so that the growth-only field at T_r from the fixed seed matches the
truth's density at resection on the region's voxels; the best evaluation
is the fit, and bound_hit flags a fit whose log T_r (or, in the free
mode, log lambda) lies within 1e-2 of a bound. The treated stage is then
run from the fixed seed with the fitted lambda's D and rho, resection_time
the fitted T_r, the truth's cavity, dose, alpha and k_ct, the schedule
truncated to 180 days and shifted for the fitted T_r (``patient_config``,
as experiment 2 shifts it for T_0), on its growth stage's grid,
recording the six frames, and compared with the truth's frames by
``compare_fields``.
  seedfix.csv      one row per patient x lambda mode, plus one row per
                   patient with lambda_mode and objective "truth" (the
                   truth against itself, fitted_T_r = T_r, the truth's
                   own seed and lambda): patient, cell, lambda_mode,
                   objective, fitted_T_r, fitted_lambda_mm,
                   lambda_truth_mm, rho_T_r_fitted (with the fitted rho),
                   rho_T_r, R_over_lambda and maturity (the design's),
                   seed_peak, seed_sigma_mm, n_evaluations,
                   objective_initial, objective_achieved, bound_hit,
                   growth_n_steps, steps_per_day, dt, wall_time_s (the
                   fit), and <frame>_<metric> for the six frames
  seedfix_summary.json   the assembly record (as substitute_summary.json)
                   with the seed's record, the lambda modes and the
                   medians (``seedfix_medians``): per lambda mode, per
                   cell, over all patients and per stratum
                   maturity >= 15 / < 15, the patient count,
                   n_bound_hit and, over the patients whose fit did not
                   hit a bound, the median of |rho_fitted T_r_fitted -
                   rho T_r| (abs_rho_T_r_error), of |log(lambda_fitted /
                   lambda)| (abs_log_lambda_error) and of every d120_ and
                   d180_ metric
  figures/seedfix_<frame>_<lambda_mode>.{png,pdf}   for d120 and d180:
                   every comparison metric against the truth's maturity
                   (log axis), one marker per patient coloured by cell
                   (hollow for a bound hit)
  runs/seedfix/<patient>/   truth/ (unless experiment 2's is reused),
                   observation.json, B/ and B_free_lambda/ (fit.json:
                   the fit's record with the bounds, the start and the
                   history; row.json; run/ with the treated run),
                   seedfix.json (the record with wall_time_s, the resume
                   marker; a mode whose row.json exists is reused, and a
                   reused row.json or fit.json must carry the current
                   seed_peak and seed_sigma_mm, else the pass raises
                   naming both seeds, ``check_reused_seed``)

Subcommand profile (experiment 4). Experiment 3's fixed-lambda fit
profiled over the width of the standard seed: per patient of --patients
(default per cell the fd_check patient and the next accepted patient in
design order, 8 patients; with --smoke the fd_check patients alone, 4;
``profile_default_patients``) and per sigma_0 in --sigmas (default 1,
1.5, 2, 3, 4, 6 mm; 2, 4 with --smoke), with the peak --seed-peak
(default 0.6), the fixed-lambda fit of the growth time (``fit_growth_time``,
objective B, --tr-bounds, --maxfev), the treated run at the fitted T_r
with the truth's maps, alpha and k_ct and the six-frame comparison,
exactly the path of ``seedfix_patient``'s fixed mode
(``profile_patient``); the truth is read back from runs/substitute/
<patient>/truth, else runs/seedfix/<patient>/truth, else solved into
runs/profile/<patient>/truth.
  profile.csv      one row per patient x sigma_0: patient, cell,
                   sigma_mm, seed_peak, fitted_T_r, rho_T_r_fitted,
                   rho_T_r, R_over_lambda, maturity, bound_hit,
                   objective_initial, objective_achieved, n_evaluations,
                   growth_n_steps, steps_per_day, dt, wall_time_s (the
                   fit), and <frame>_<metric> for the six frames
  figures/profile_objective.{png,pdf}   objective_achieved against
                   sigma_0, one line per patient coloured by cell (a
                   bound hit hollow)
  figures/profile_d180.{png,pdf}   d180 mass_beyond_edema_rel and
                   mass_rel against sigma_0, the same way
  profile_summary.json   the assembly record with the seed peak, the
                   sigmas and the medians (``profile_medians``): per
                   sigma_0 and group (cells, all, the maturity strata)
                   the patient count, n_bound_hit and, excluding
                   the bound hits, the medians of fitted_T_r,
                   rho_T_r_fitted, objective_achieved and the d120_ and
                   d180_ metrics
  runs/profile/<patient>/   truth/ (unless reused), observation.json,
                   sigma_<sigma_0>/ (fit.json, row.json, run/),
                   profile.json (the resume marker; a sigma_0 whose
                   row.json exists is reused when it carries the current
                   peak and that sigma_0, ``check_reused_seed``)

Subcommand all runs design (skipped when spec.json exists; on the first
device of --gpus), then substitute, seedfix (the fixed lambda mode by
default) and profile in order over the devices, each with its own
defaults (--maxfev bounds every fit; --dt-mode, --delta-a, --seed-peak,
--seed-sigma-mm, --lambda-mode, --sigmas, --tr-bounds, --tr-max,
--patients-per-cell and the band arguments are taken as by the
subcommands). --patients restricts substitute, seedfix and profile to
ids or ranges (p03, p00-p07, all); without it the profile takes its
default patients. The invariance, fisher and dt-check subcommands run
on their own.

Devices. --gpus is a comma-separated list of CUDA device ids, '' the
CPU (the default is the sensitivity script's slots, 1,2,3,6; ','
names two CPU workers). The design's screening, the invariance
experiment and dt-check run on one device (the first of the list under
all; their subcommands refuse more). With one device, or one selected patient,
fisher, substitute, seedfix and profile solve in this process. With
several devices they dispatch
(``dispatch``): the selected patients, in design order, are cut into
contiguous blocks of sizes differing by at most one, one block per
device (``patient_blocks``; more devices than patients leave the
surplus idle), and one worker process per device runs the same
subcommand with --gpus <device> --patients <block> and the other
options, with the sensitivity script's environment handling
(XLA_PYTHON_CLIENT_PREALLOCATE=false and CUDA_VISIBLE_DEVICES=<device>,
or JAX_PLATFORMS=cpu for ''), its output in
logs/<experiment>_<device>.log (cpu for ''; a repeated device gets a
_<k> suffix). The dispatcher waits for every worker, assembles the CSVs,
figures and summary JSON from every record present (as a plain pass
does; the workers skip that step) and then fails, naming the logs, when
any worker returned nonzero. The device split (devices, blocks, return
codes, logs, wall times) goes into <experiment>_summary.json; a later
single-process pass keeps the last split. Workers are ordinary passes,
so a dispatch is resumable like any other. --smoke uses
resolution_factor 0.25, one patient per cell (the design accepts one,
4 patients, and the arms take the first of each cell), K = 8, maxfev 20,
the profile widths 2 and 4 mm and the CPU, so that the whole pipeline
finishes in minutes; the design records it (spec.json: smoke) and the
later subcommands follow the record. The tests use the same settings on
a phantom.

Output layout (--output-dir, default DEFAULT_OUTPUT_DIR
/mnt/Drive4/lucas/stupp_identifiability; --name required):
  <output-dir>/<name>/
    spec.json, base_config.json, search_space.json, design.csv
    screen/c<candidate>/screen.json
    configs/<patient>.json
    runs/invariance/, runs/fisher/<patient>/, runs/substitute/<patient>/,
    runs/seedfix/<patient>/, runs/profile/<patient>/,
    runs/dt_check/<patient>/
    invariance.csv, fisher.csv, fisher_regions.csv, fisher_fd_check.csv,
    fisher_runs.csv, substitute.csv, seedfix.csv, profile.csv,
    dt_check.csv
    <experiment>_summary.json
    figures/

Cost (an estimate). The sensitivity script's fit for a two-stage run in
its own process is 6.6 s + 1.4 ms per time step on one Quadro RTX 8000.
In one process on the same class of card a growth-only atlas solve takes
0.5 s + 1.4 ms per step (1 200 steps: 2.2 s) and a treated run recording
frames 3.5 s + 2.2 ms per step (1 908 steps: 7.7 s; the treatment terms,
the frame upsamplings and the maps' downsampling are the extra). A new
step count compiles the scan again (1-3 s). With T_r the resection time
in days and k the patient's steps per day (12 in the fixed mode, 2-12
in the stability mode, 2 for about half of the cohort), a growth stage
is k T_r steps and a treated run k (T_r + 120) steps (experiment 1) or
k (T_r + 180) (experiments 2-4); the figures below are for 12 steps/day
and shrink with k.
  design       one growth-only solve per screened candidate (k T_r
               steps, T_r up to --tr-max: 2-20 s each with the
               recompilation); the number screened depends on the
               acceptance rate of the bands (a few solves per accepted
               patient in a cell that fits the bands, tens in one that
               does not), so tens of minutes to a few hours on one GPU.
  invariance   5 lambdas x (2 growth + 1 treated) solves: 1-2 min.
  fisher       per patient the truth (2 solves), 16 perturbed treated
               runs (10 more for the four fd_check patients), the noise
               (K = 64: about 2 min on the CPU side), the metrics of 80
               (130) frames and the analysis (least squares on 4 million
               rows): 5-12 min per patient, about 2-3 h for the 16
               patients on one GPU; about 500 MB per fd_check patient
               (W.npz 116 MB, 29 run directories of about 12 MB), about
               8 GB in total.
  substitute   per patient the truth and 4 fits of up to 60 growth-only
               solves of k T_0 steps (0.85 s per evaluation at T_0 = 30
               days, 3.1 s at 150, about 15 s at 1 000 at 12 steps/day)
               plus 4 treated runs: 10-30 min per patient depending on
               T_0, so 3-8 h for the cohort on one GPU; --gpus 1,2,3,6
               dispatches 4 patients to each of four devices. About
               33 MB per patient.
  seedfix      per patient 1 fixed-mode fit (10-20 Brent evaluations,
               each a growth-only solve with its own step count and
               compilation, 5-40 s; the free mode, when asked for, up to
               150 Nelder-Mead evaluations) plus 1 treated run: 5-30 min
               per patient.
  profile      per patient 6 fixed-mode fits and 6 treated runs, 8
               patients: 1-3 h in total over the devices.
  dt-check     per patient two truths (2 growth and 2 treated solves):
               1-2 min per patient, 4 patients by default.
  --smoke      the whole pipeline on the CPU (maxfev 20, 4 mm voxels):
               tens of minutes; the design's screening at 4 mm takes
               2-5 s per candidate.

Run from the project root, e.g. (ID = /mnt/Drive4/lucas/stupp_identifiability):
  python scripts/identifiability_experiments.py design --name id_2026-09-19 --gpus 1
  python scripts/identifiability_experiments.py dt-check --name id_2026-09-19 --gpus 1
  python scripts/identifiability_experiments.py substitute --name id_2026-09-19 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py seedfix --name id_2026-09-19 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py profile --name id_2026-09-19 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py invariance --name id_2026-09-19 --gpus 1
  python scripts/identifiability_experiments.py fisher --name id_2026-09-19 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py all --name id_2026-09-19 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py all --smoke --output-dir runs/ --name smoke
Every subcommand but design needs the design directory; the CSVs and
figures are reassembled from the records present at the end of each
pass, so a pass over a subset of patients updates them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, cast

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes its backend, which importing fisher_kpp_jax does not do
# (the backend starts with the first array operation, i.e. the first
# solve; ``configure_device`` sets the device before that).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, map_coordinates  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from fisher_kpp_jax import SOLVER_KEY, FKPPSolver, Result, StuppFKPPSolver, read_config, write_config  # noqa: E402
from fisher_kpp_jax.config import jsonable  # noqa: E402

SENSITIVITY_SCRIPT = _ROOT / "scripts" / "sensitivity_analysis.py"


def load_sensitivity_analysis() -> ModuleType:
    """
    scripts/sensitivity_analysis.py as a module, loaded from its file the
    way tests/test_sensitivity_analysis.py loads it (scripts/ is not a
    package); a copy already registered in sys.modules is reused.
    """
    name = "sensitivity_analysis"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, SENSITIVITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SENSITIVITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sa = load_sensitivity_analysis()

# --- settings ---

EXPERIMENTS: tuple[str, ...] = ("design", "invariance", "fisher", "substitute", "seedfix", "profile", "dt-check")
DEFAULT_OUTPUT_DIR = Path("/mnt/Drive4/lucas/stupp_identifiability")
DEFAULT_SEARCH_SPACE = _ROOT / "fisher_kpp_jax" / "search_spaces" / "stupp_identifiability_search_space.json"
DEFAULT_GPUS: str = str(sa.DEFAULT_GPUS)  # the sensitivity script's slots, "1,2,3,6"
# The seed voxel of every patient: right-hemisphere deep white matter on
# the atlas, the voxel scripts/run_stupp_synthetic.py uses (snapped to the
# nearest seedable voxel; a grid too small to hold it falls back to the
# base config's fractions, see ``make_design``).
DEFAULT_SEED_VOXEL: tuple[int, int, int] = (132, 103, 90)
LOG_DIR = "logs"  # the dispatcher's worker logs, <experiment>_<device>.log
NO_ASSEMBLE_FLAG = "--no-assemble"  # the workers skip the assembly
PRECISION = "f64"
SEED_FLOOR = 0.0  # gaussian_seed_floor of the whole cohort (the base config's 0.1 is replaced)
DESIGN_SEED = 1
DEFAULT_LOG2_CANDIDATES = 10  # 1024 Sobol' points to fill the four cells from
PATIENTS_PER_CELL = 4  # 16 patients
# A search-space entry carrying this key with a true value is a factor of
# this script, not a solver parameter; ``load_script_search_space`` strips
# it before the shared loader sees the file.
SCRIPT_FACTOR_KEY = "script"
GROWTH_EFOLDS_FACTOR = "growth_efolds"  # a = rho T_r; resection_time = growth_efolds / rho
LAMBDA_SPLIT = 2.0  # front_width_mm at or below / above: compact / broad
# maturity = 2 growth_efolds front_width_mm^2 / seed_sigma_mm^2 = ell lambda / sigma^2
# (ell = v T_r = 2 a lambda the front travel): below / at or above: immature / mature.
MATURITY_SPLIT = 15.0
SEED_WIDTH_CAP = 1.5  # a candidate with seed_sigma_mm > SEED_WIDTH_CAP front_width_mm is rejected (seed_wide)
# The time step (--dt-mode, ``steps_per_day_for``). fixed: BASE_STEPS_PER_DAY
# steps per day for every patient (the base config's 12). stability: per
# patient dt = min(DT_MAX, DT_SAFETY dx^2 / (6 D_wm)) rounded down to an
# integer number of steps per day, never finer than BASE_STEPS_PER_DAY
# (the floor of the step) and never coarser than the solver's own
# estimate at the truth's resection_time (``solver_step_estimate``), so
# that the truth is stepped as requested; DT_MAX is half a day because
# the frame rounding (``frame_days``) needs a step of at most half a day.
DT_MODES: tuple[str, ...] = ("fixed", "stability")
DEFAULT_DT_MODE = "stability"
BASE_STEPS_PER_DAY = 12
DT_MAX = 0.5  # days
DT_SAFETY = 0.5  # of the explicit diffusion limit dx^2 / (6 D)
SOLVER_STEP_FLOOR = 100  # the solver's estimate over a horizon T: max(8 D T / dx^2 + 100, 1.1 rho T) steps
VISIBILITY_HORIZON = 120.0  # days; visibility_margin = VISIBILITY_HORIZON rho - log_kill_total (a design column, not a cell split)
TR_MIN = 5.0  # days; a candidate below it is rejected, a substitute T_0 below it skipped
DEFAULT_TR_MAX = 1000.0  # days; --tr-max, the candidates above it are rejected
MAX_CONSECUTIVE_SCREEN_FAILURES = 3  # the screening aborts after this many failed solves in a row
# The sampled factors (the search space's order, growth_efolds after the
# growth group); everything else is fixed: diffusivity_ratio,
# rt_alpha_beta_ratio and chemo_decay_rate at the search space's
# overrides, the seed position at one voxel.
SAMPLED_FACTORS: tuple[str, ...] = (
    sa.GROWTH_SPEED_FACTOR,
    sa.GROWTH_WIDTH_FACTOR,
    GROWTH_EFOLDS_FACTOR,
    "chemo_kill_rate",
    "rt_alpha",
    sa.SEED_PEAK_FACTOR,
    sa.SEED_SIGMA_FACTOR,
)
FIXED_PARAMETERS: tuple[str, ...] = ("diffusivity_ratio", "rt_alpha_beta_ratio", "chemo_decay_rate")
CELLS: dict[str, tuple[bool, bool]] = {  # cell name -> (broad: front_width_mm > LAMBDA_SPLIT, mature: maturity >= MATURITY_SPLIT)
    "compact_immature": (False, False),
    "compact_mature": (False, True),
    "broad_immature": (True, False),
    "broad_mature": (True, True),
}
DESIGN_DERIVED: tuple[str, ...] = (
    "white_matter_diffusivity",
    "rho",
    "gaussian_seed_mass",
    "gaussian_seed_diffusion_time",
    sa.SEED_RADIUS_COLUMN,
    sa.SEED_RATIO_COLUMN,
)
SIZE_COLUMNS: tuple[str, ...] = ("r_core_mm", "r_whole_mm", "whole_core_ratio")
DESIGN_COLUMNS: list[str] = [
    "patient",
    "cell",
    "candidate",
    *(f"u_{name}" for name in SAMPLED_FACTORS),
    *SAMPLED_FACTORS,
    "resection_time",
    *DESIGN_DERIVED,
    "rho_T_r",
    "ell_mm",
    "log_kill_rt",
    "log_kill_ct",
    "log_kill_total",
    "visibility_margin",
    "maturity",
    *SIZE_COLUMNS,
    "R_over_lambda",
    "steps_per_day",
    "dt",
    "fd_check",
]
SCREEN_DIR = "screen"  # <design>/screen/c<candidate>/screen.json, the screening's resume records
SCREEN_FILE = "screen.json"


@dataclass(frozen=True)
class SizeBands:
    """The size screening's acceptance bands (lo, hi), applied to the
    growth stage's density at resection_time on the 1 mm grid: the
    equivalent-sphere radius of {u >= 0.6} and of {u >= 0.3} on the tissue
    mask in mm and the volume ratio whole / core. The defaults are
    provisional (to be replaced from BraTS)."""

    r_core_mm: tuple[float, float] = (5.0, 25.0)
    r_whole_mm: tuple[float, float] = (10.0, 35.0)
    whole_core_ratio: tuple[float, float] = (1.2, 8.0)

    def record(self) -> dict[str, list[float]]:
        return {name: [float(lo), float(hi)] for name, (lo, hi) in ((n, getattr(self, n)) for n in SIZE_COLUMNS)}


DEFAULT_BANDS = SizeBands()
BAND_ARGS: dict[str, str] = {"r_core_mm": "r_core_band", "r_whole_mm": "r_whole_band", "whole_core_ratio": "ratio_band"}

# Horizons after resection (days) and the schedule truncation.
FISHER_HORIZON = 120.0  # the shipped search space's time_after_resection
SUBSTITUTE_HORIZON = 180.0
# The frames a treated run records, by name: the moment (days after
# resection_time) BEFORE which the state is taken, as the last step end at
# least half a step earlier (``frame_days``): pre = the last step end before
# the resection; d34 / d55 = the end of the CRT Sundays 34 and 55 (moment
# 35 / 56, before Monday's fraction; the script's snapshot definition);
# d80 = the end of day 80, before the first adjuvant dose at offset 81;
# d120 / d180 = the moment T_r + 120 / + 180 (the horizons).
FRAME_MOMENTS: dict[str, float] = {"pre": 0.0, "d34": 35.0, "d55": 56.0, "d80": 81.0, "d120": 120.0, "d180": 180.0}
FISHER_FRAMES: tuple[str, ...] = ("pre", "d34", "d55", "d80", "d120")
SUBSTITUTE_FRAMES: tuple[str, ...] = ("pre", "d34", "d55", "d80", "d120", "d180")
FRAME_FILE = "{name}_cell_density.nii.gz"
FRAMES_FILE = "frames.json"
PATIENT_FILE = "{experiment}.json"  # the per-patient result record, the resume marker

# Experiment 0.
INVARIANCE_LAMBDAS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
INVARIANCE_RUN_TYPES: dict[str, str] = {
    "growth_fixed_n": "A: growth-only FKPPSolver, n_steps fixed to the lambda = 1 count",
    "growth_fixed_dt": "B: growth-only FKPPSolver at the base config's steps per day",
    "treated": "C: StuppFKPPSolver with the lambda = 1 cavity and dose, schedule shifted with T_r / lambda",
}

# Experiment 1.
THETA_NAMES: tuple[str, ...] = ("log_v", "log_lambda", "log_T_r", "log_alpha", "log_k_ct")
SEED_THETA_NAMES: tuple[str, ...] = ("peak", "log_sigma")
COLUMN_NAMES: tuple[str, ...] = (*THETA_NAMES, *SEED_THETA_NAMES)
T_R_INDEX = THETA_NAMES.index("log_T_r")
T_R_VARIANTS: tuple[str, ...] = ("scaled_dt", "fixed_dt")  # how the T_r column is stepped, see the docstring
FD_STEP = 0.05  # the finite-difference step in theta (and in the peak density, additive)
FD_CHECK_STEP = 0.025
# The growth invariance (lambda D, lambda rho, T_r / lambda) in theta:
# v = 2 sqrt(D rho) scales with lambda, the front width sqrt(D / rho)
# does not, so the direction is (1, 0, -1) over (log v, log lambda,
# log T_r); the request's (1, 1, -1) / sqrt 3 is the same direction over
# (log D, log rho, log T_r).
INVARIANCE_DIRECTION: NDArray = np.array([1.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
OBSERVATION_SETS: dict[str, tuple[str, ...]] = {
    "a": ("pre",),
    "b": ("pre", "d120"),
    "c": ("pre", "d55", "d120"),
    "d": ("pre", "d34", "d55", "d120"),
    "e": ("pre", "d80", "d120"),
    "f": ("pre", "d34", "d55", "d80", "d120"),
}
INDICATOR_LEVELS: tuple[float, ...] = (sa.TAU_CORE, sa.TAU_EDEMA)  # 0.6, 0.3
INDICATOR_WIDTH = 0.02  # s(u) = 1 / (1 + exp(-(u - c) / width))
OBSERVATION_MARGIN_MM = 30.0  # tissue voxels within it of the union of the truth's 0.3 iso-surfaces
NOISE_DRAWS = 64
THRESHOLD_RANGES: dict[float, tuple[float, float]] = {sa.TAU_CORE: (0.50, 0.85), sa.TAU_EDEMA: (0.19, 0.50)}
DISPLACEMENT_STD_MM = 1.0
DISPLACEMENT_LENGTH_MM = 5.0  # the smoothing kernel's standard deviation
VARIANCE_FLOOR = 1e-3
W_MAX_BYTES = 500e6  # W.npz is written only below this size
REGION_NAMES: tuple[str, ...] = ("core", "rim", "out_of_field")
SINGULAR_TOLERANCE = 1e-14  # an eigenvalue of F below it times the largest counts as zero

# Experiment 2: the substitute's growth time T_0 = T_r - delta_a / rho per
# deficit delta_a in e-folds (delta_a > 0: T_0 earlier than the truth).
SUBSTITUTE_DELTA_A: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)
MAXFEV = 150  # --maxfev of the seedfix and profile fits
SUBSTITUTE_MAXFEV = 60  # --maxfev of the substitute fits
PEAK_MIN = 0.05  # the fitted peak lies in (PEAK_MIN, PEAK_MAX]
PEAK_MAX = 1.0
LOGIT_CLIP = 1e-4  # the inverse map keeps the unit-interval argument in [LOGIT_CLIP, 1 - LOGIT_CLIP]
SIMPLEX_STEPS: tuple[float, float] = (0.5, 0.2)  # the initial simplex: x0, x0 + (0.5, 0), x0 + (0, 0.2) in the fit's two coordinates
NELDER_MEAD_XATOL = 1e-3
NELDER_MEAD_FATOL = 1e-6
OBJECTIVES: tuple[str, ...] = ("B",)  # the fit objective(s) run; ``fit_objective`` also knows "A"
LOG_EPS = 1e-6  # log(u + LOG_EPS) in the log metrics and objective A

# Experiment 3.
TR_BOUNDS: tuple[float, float] = (5.0, 3000.0)  # the fitted growth time's range in days (--tr-bounds)
LAMBDA_BOUNDS: tuple[float, float] = (0.5, 8.0)  # the fitted front width's range in mm (the free-lambda mode)
LAMBDA_MODES: tuple[str, ...] = ("fixed", "free")  # --lambda-mode fixed | free | both
DEFAULT_LAMBDA_MODE = "fixed"
LAMBDA_MODE_DIRS: dict[str, str] = {"fixed": "B", "free": "B_free_lambda"}  # runs/seedfix/<patient>/<dir>/
SCALAR_XATOL = 1e-3  # the bounded search's tolerance in log T_r
SEED_SOURCE = "argument"  # the fixed seed comes from --seed-peak and --seed-sigma-mm
DEFAULT_SEED_PEAK = 0.6
DEFAULT_SEED_SIGMA_MM = 2.0
BOUND_TOLERANCE = 1e-2  # a fitted log parameter within it of a bound counts as a bound hit

# Experiment 4 (profile).
PROFILE_SIGMAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # the fixed seed widths sigma_0 in mm (--sigmas)

# Metrics (``compare_fields``).
MASS_NAMES: tuple[str, ...] = ("mass", "mass_out_of_field", "mass_beyond_edema")  # each with ref_<name> and <name>_rel
METRIC_NAMES: list[str] = [
    "dice_core",
    "dice_edema",
    "assd_core_mm",
    "assd_edema_mm",
    "rel_l2_log",
    "rel_l2",
    "max_abs_diff",
    *MASS_NAMES,
    *(f"ref_{name}" for name in MASS_NAMES),
    *(f"{name}_rel" for name in MASS_NAMES),
    "log_vol_ratio_core",
    "log_vol_ratio_edema",
    *(f"qoi_{name}" for name in sa.QOI_NAMES),
]
# The metrics plotted against lambda / delta_a / maturity (the QoIs are
# plotted in a second figure).
PLOTTED_METRICS: list[str] = [name for name in METRIC_NAMES if not name.startswith("qoi_") and not name.startswith("ref_")]
PLOTTED_QOIS: list[str] = [f"qoi_{name}" for name in sa.FINAL_ANALYSED_QOIS]

COLOR_CELLS: dict[str, str] = {"compact_immature": "#2a78d6", "compact_mature": "#eb6834", "broad_immature": "#2a9d8f", "broad_mature": "#8a4fbf"}
COLOR_TEXT = "#0b0b0b"
COLOR_LINES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#2a9d8f", "#8a4fbf", "#c9a227", "#52514e", "#d64a8a", "#7a5c2e")


@dataclass(frozen=True)
class Smoke:
    """The --smoke settings: a coarse grid, few patients, few draws, few
    evaluations, the CPU."""

    resolution_factor: float = 0.25  # the solver's zoom factor: 4 mm voxels on the 1 mm atlas
    patients_per_cell: int = 1  # the smoke design's cohort: one patient per cell
    n_patients: int = 4  # the first patient of each cell
    draws: int = 8
    maxfev: int = 20
    sigmas: tuple[float, ...] = (2.0, 4.0)  # the profile's seed widths


SMOKE = Smoke()


# --- patients and cohort ---


@dataclass(frozen=True)
class Patient:
    """
    One truth: the growth band (front speed v, front width lambda), the
    resection time, the two treatment parameters and the seed (its peak
    density and absolute width sigma in mm); the seed position is the
    cohort's.

    The solver parameters follow from the script's derivations:
    D = v lambda / 2, rho = v / (2 lambda) (``growth_parameters``),
    tau = sigma^2 / 2, m = peak (4 pi tau)^(3/2) (``seed_parameters``).
    """

    id: str
    cell: str
    front_speed: float
    front_width: float
    resection_time: float
    rt_alpha: float
    chemo_kill_rate: float
    seed_peak: float
    seed_sigma: float
    r_over_lambda: float = float("nan")  # the design's R_over_lambda (r_whole_mm / front_width_mm); NaN off the design
    maturity: float = float("nan")  # the design's maturity (``maturity``); NaN off the design
    fd_check: bool = False  # the design's fd_check flag (the first patient of each cell)
    steps_per_day: int | None = None  # the patient's time step (``steps_per_day_for``); None: the base config's setting

    @property
    def dt(self) -> float | None:
        """1 / steps_per_day in days; None without a step of its own."""
        return None if self.steps_per_day is None else 1.0 / float(self.steps_per_day)

    @property
    def growth(self) -> dict[str, float]:
        """white_matter_diffusivity and rho."""
        derived = sa.growth_parameters(self.front_speed, self.front_width)
        return {key: float(value) for key, value in derived.items()}

    @property
    def diffusivity(self) -> float:
        return self.growth["white_matter_diffusivity"]

    @property
    def rho(self) -> float:
        return self.growth["rho"]

    @property
    def seed(self) -> dict[str, float]:
        """gaussian_seed_mass and gaussian_seed_diffusion_time."""
        derived = sa.seed_parameters(self.seed_peak, self.seed_sigma)
        return {key: float(value) for key, value in derived.items()}

    @property
    def theta(self) -> NDArray:
        """(log v, log lambda, log T_r, log alpha, log k_ct)."""
        return np.log([self.front_speed, self.front_width, self.resection_time, self.rt_alpha, self.chemo_kill_rate])

    def solver_values(self) -> dict[str, Any]:
        """The StuppFKPPSolver parameters the patient sets (its time step
        as steps_per_day, with n_steps and dt unset, when it has one)."""
        values: dict[str, Any] = {
            **self.growth,
            **self.seed,
            "resection_time": float(self.resection_time),
            "rt_alpha": float(self.rt_alpha),
            "chemo_kill_rate": float(self.chemo_kill_rate),
        }
        if self.steps_per_day is not None:
            values.update({"n_steps": None, "dt": None, "steps_per_day": int(self.steps_per_day)})
        return values

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Patient:
        """A patient from a design.csv record (strings or numbers)."""
        return cls(
            id=str(record["patient"]),
            cell=str(record["cell"]),
            front_speed=float(record[sa.GROWTH_SPEED_FACTOR]),
            front_width=float(record[sa.GROWTH_WIDTH_FACTOR]),
            resection_time=float(record["resection_time"]),
            rt_alpha=float(record["rt_alpha"]),
            chemo_kill_rate=float(record["chemo_kill_rate"]),
            seed_peak=float(record[sa.SEED_PEAK_FACTOR]),
            seed_sigma=float(record[sa.SEED_SIGMA_FACTOR]),
            r_over_lambda=sa.as_float(record.get("R_over_lambda")),
            maturity=sa.as_float(record.get("maturity")),
            fd_check=str(record.get("fd_check")) == "True",
            steps_per_day=None if str(record.get("steps_per_day", "")).strip() == "" else int(float(record["steps_per_day"])),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], id: str = "base", cell: str = "base") -> Patient:
        """The patient a StuppFKPPSolver config describes (its D, rho,
        seed, resection_time, rt_alpha, chemo_kill_rate)."""
        front = sa.front_parameters(config["white_matter_diffusivity"], config["rho"])
        tau = float(config["gaussian_seed_diffusion_time"])
        return cls(
            id=id,
            cell=cell,
            front_speed=float(front[sa.GROWTH_SPEED_FACTOR]),
            front_width=float(front[sa.GROWTH_WIDTH_FACTOR]),
            resection_time=float(config["resection_time"]),
            rt_alpha=float(config["rt_alpha"]),
            chemo_kill_rate=float(config["chemo_kill_rate"]),
            seed_peak=float(sa.seed_peak_density(config["gaussian_seed_mass"], tau)),
            seed_sigma=float(np.sqrt(2.0 * tau)),
            steps_per_day=None if config.get("steps_per_day") is None else int(config["steps_per_day"]),
        )


@dataclass(frozen=True)
class Cohort:
    """
    A design directory as loaded (``load_cohort``): its spec, the base
    config with absolute paths, the tissue maps and grid, the fixed seed
    voxel and the patients.
    """

    root: Path
    spec: dict[str, Any]
    base: dict[str, Any]
    wm: NDArray
    gm: NDArray
    zooms: tuple[float, float, float]
    affine: NDArray
    tissue: NDArray
    seed_voxel: tuple[int, int, int]
    seed_fractions: tuple[float, float, float]
    patients: list[Patient]

    def patient(self, id: str) -> Patient:
        for patient in self.patients:
            if patient.id == id:
                return patient
        raise KeyError(f"no patient {id!r} in {self.root / 'design.csv'}; ids: {[p.id for p in self.patients]}")

    @property
    def treatment(self) -> dict[str, float]:
        """The treatment derivation settings (``treatment_settings``)."""
        return dict(sa.spec_treatment_settings(self.spec))

    @property
    def smoke(self) -> bool:
        return bool(self.spec.get("smoke", False))


def load_image(path: str | Path) -> nib.Nifti1Image:
    """A NIfTI image (nibabel's loader, typed as the NIfTI-1 class)."""
    return cast(nib.Nifti1Image, nib.load(str(path)))


def load_tissue(base: Mapping[str, Any]) -> tuple[NDArray, NDArray, tuple[float, float, float], NDArray]:
    """The base config's white- and gray-matter maps (float64), the zooms
    and the affine of the white-matter map's header."""
    wm_image = load_image(base["white_matter_pbmap"])
    gm_image = load_image(base["gray_matter_pbmap"])
    wm = np.asarray(wm_image.get_fdata(), dtype=np.float64)
    gm = np.asarray(gm_image.get_fdata(), dtype=np.float64)
    if wm.shape != gm.shape or wm.ndim != 3:
        raise ValueError(f"the tissue maps must be 3D and alike, got {wm.shape} and {gm.shape}.")
    zooms = tuple(float(z) for z in wm_image.header.get_zooms()[:3])
    return wm, gm, (zooms[0], zooms[1], zooms[2]), np.asarray(wm_image.affine, dtype=np.float64)


def load_cohort(root: str | Path) -> Cohort:
    """Load a design directory written by ``make_design``."""
    root = Path(root)
    spec = sa.read_json(root / "spec.json")
    base = read_config(root / "base_config.json", solver=StuppFKPPSolver)
    wm, gm, zooms, affine = load_tissue(base)
    tissue = (wm + gm) >= float(spec["min_tissue_fraction"])
    voxel = tuple(int(v) for v in spec["seed_voxel"])
    fractions = tuple(float(v) for v in spec["seed_fractions"])
    patients = [Patient.from_record(record) for record in sa.read_csv(root / "design.csv")]
    if base.get("steps_per_day") is not None:  # a design without the column (before --dt-mode) steps at the base config's rate
        patients = [replace(p, steps_per_day=int(base["steps_per_day"])) if p.steps_per_day is None else p for p in patients]
    return Cohort(
        root=root,
        spec=dict(spec),
        base=base,
        wm=wm,
        gm=gm,
        zooms=zooms,
        affine=affine,
        tissue=tissue,
        seed_voxel=(voxel[0], voxel[1], voxel[2]),
        seed_fractions=(fractions[0], fractions[1], fractions[2]),
        patients=patients,
    )


def select_patients(cohort: Cohort, selection: str | None, smoke: bool) -> list[Patient]:
    """
    The patients of a run pass: --patients as comma-separated ids or
    ranges (p03, p00-p07; 'all'), else the first patient of each cell
    with --smoke, else every patient.
    """
    if selection is None or selection.strip() == "":
        if smoke:
            chosen: list[Patient] = []
            for cell in CELLS:
                chosen.extend([p for p in cohort.patients if p.cell == cell][:1])
            return chosen[: SMOKE.n_patients]
        return list(cohort.patients)
    if selection.strip() == "all":
        return list(cohort.patients)
    ids: list[str] = []
    order = [p.id for p in cohort.patients]
    for item in (part.strip() for part in selection.split(",") if part.strip()):
        first, _, last = item.partition("-")
        if last:
            if first not in order or last not in order:
                raise ValueError(f"--patients: unknown id in range {item!r}; ids: {order}")
            lo, hi = order.index(first), order.index(last)
            ids.extend(order[lo : hi + 1])
        else:
            if first not in order:
                raise ValueError(f"--patients: unknown id {first!r}; ids: {order}")
            ids.append(first)
    seen: dict[str, None] = {}
    for id in ids:
        seen.setdefault(id, None)
    return [cohort.patient(id) for id in seen]


# --- JSON / files ---


def write_record(path: Path, record: Mapping[str, Any]) -> None:
    """A JSON file of a record (numpy values converted, inf and nan kept as
    the strings "inf" / "nan", which ``read_record`` turns back)."""
    path.write_text(json.dumps(jsonable(record), indent=2) + "\n", encoding="utf-8")


def _from_json(value: Any) -> Any:
    if isinstance(value, str) and value in ("inf", "-inf", "nan"):
        return float(value)
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_json(v) for k, v in value.items()}
    return value


def read_record(path: Path) -> dict[str, Any]:
    """The inverse of ``write_record``."""
    record = _from_json(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(record, dict):
        raise ValueError(f"{path}: must hold a JSON object.")
    return record


def save_field(path: Path, field: NDArray, affine: NDArray) -> None:
    """A density field as float32 NIfTI, rounded for storage
    (``round_field``: 7 mantissa bits, values below 1e-10 zeroed)."""
    image = nib.Nifti1Image(sa.round_field(field), affine)
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))


def configure_device(gpu: str | None) -> None:
    """
    Choose the device of this process before JAX starts its backend, like
    the sensitivity script's slots: a CUDA device id (CUDA_VISIBLE_DEVICES)
    or '' / None for the CPU (JAX_PLATFORMS=cpu).
    """
    if gpu is None or gpu.strip() == "":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu.strip()


# --- design ---


def nearest_seedable_voxel(geometry: Any, target: Sequence[int]) -> tuple[tuple[int, int, int], float]:
    """The seedable voxel (``seed_geometry``) nearest to a target voxel
    (Euclidean distance in voxels; the lowest index on a tie) and that
    distance."""
    voxels = np.asarray(geometry.voxels, dtype=np.int64)
    distances = np.linalg.norm(voxels - np.asarray(target, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    chosen = voxels[index]
    return (int(chosen[0]), int(chosen[1]), int(chosen[2])), float(distances[index])


def total_log_kill(
    rt_alpha: NDArray | float,
    alpha_beta_ratio: float,
    n_fractions: int,
    dose_per_fraction: float,
    chemo_kill_rate: NDArray | float,
    chemo_total_dose: float,
    chemo_decay_rate: float,
) -> dict[str, NDArray]:
    """
    The total log kill of the schedule: the radiotherapy part
    n d alpha (1 + d / (alpha / beta)) (60 alpha (1 + 2 / (alpha/beta)) for
    30 fractions of 2 Gy), the chemotherapy part k_ct D_tot / gamma (the
    script's chemotherapy budget) and their sum Lambda.
    """
    alpha = np.asarray(rt_alpha, dtype=np.float64)
    kill = np.asarray(chemo_kill_rate, dtype=np.float64)
    rt = n_fractions * dose_per_fraction * alpha * (1.0 + dose_per_fraction / alpha_beta_ratio)
    ct = kill * chemo_total_dose / chemo_decay_rate
    return {"log_kill_rt": rt, "log_kill_ct": ct, "log_kill_total": rt + ct}


def first_adjuvant_offset(chemo_times: Sequence[float], rt_times: Sequence[float], resection_time: float) -> float | None:
    """The offset after resection_time of the first adjuvant chemotherapy
    session: the first session after the last fraction that does not
    follow a session on the previous day (the concomitant block is daily
    and ends two days after the last fraction). None without one."""
    sessions = np.sort(np.asarray(chemo_times, dtype=np.float64))
    last_fraction = float(np.max(rt_times))
    for i in range(1, sessions.size):
        if sessions[i] > last_fraction and sessions[i] - sessions[i - 1] > 1.0 + 1e-9:
            return float(sessions[i]) - float(resection_time)
    return None


def load_script_search_space(path: str | Path, config_keys: Iterable[str]) -> tuple[Any, dict[str, Any]]:
    """
    The search space of a design: the file's entries carrying
    SCRIPT_FACTOR_KEY ("script": true), which are factors of this script
    and not solver parameters, are taken out and parsed here as
    {"min", "max", "scale"} ranges; the rest is handed to the sensitivity
    script's ``load_search_space`` (which would refuse the unknown keys).

    Returns:
        (space, script_factors): the SearchSpace and the script factors by
        name (``SearchSpaceParameter``: name, low, high, scale).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"search space not found: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, Mapping):
        raise ValueError(f"search space {path}: must be a JSON object.")
    stripped: dict[str, Any] = {}
    script: dict[str, Any] = {}
    for key, value in entries.items():
        if not (isinstance(value, Mapping) and value.get(SCRIPT_FACTOR_KEY)):
            stripped[key] = value
            continue
        entry = {k: v for k, v in value.items() if k != SCRIPT_FACTOR_KEY}
        if set(entry) != {"min", "max", "scale"} or entry["scale"] not in ("linear", "log"):
            raise ValueError(
                f'search space {path}: the script factor {key} must be {{"min", "max", "scale": "linear" | "log", '
                f'"{SCRIPT_FACTOR_KEY}": true}}, got {dict(value)!r}.'
            )
        low, high = float(entry["min"]), float(entry["max"])
        if not (np.isfinite(low) and np.isfinite(high) and low < high) or (entry["scale"] == "log" and low <= 0):
            raise ValueError(f"search space {path}: {key}: min < max must be finite (and min > 0 for log), got {low!r}, {high!r}.")
        script[key] = sa.SearchSpaceParameter(key, low, high, entry["scale"])
    space = sa.load_search_space(stripped, config_keys)
    taken = sorted(set(script) & set(space.factors))
    if taken:
        raise ValueError(f"search space {path}: {taken} are script factors and factors of the space at once.")
    return space, script


def visibility_margin(rho: NDArray | float, log_kill_total: NDArray | float) -> NDArray:
    """VISIBILITY_HORIZON rho - Lambda: the e-folds the untreated growth
    gains over the 120-day log kill; at or above 0 the truth regrows past
    the treatment by day 120 (visible), below it does not (invisible)."""
    return VISIBILITY_HORIZON * np.asarray(rho, dtype=np.float64) - np.asarray(log_kill_total, dtype=np.float64)


def solver_step_estimate(diffusivity: float, rho: float, horizon: float, dx: float) -> int:
    """The solver's own stability estimate of the step count over a
    horizon in days: ceil(max(8 D T / dx^2 + SOLVER_STEP_FLOOR, 1.1 rho T))
    (``_time_step_count`` of FKPPSolver and StuppFKPPSolver, dx the
    smallest grid spacing in mm); the solver raises a coarser request to
    it."""
    t = float(horizon)
    n = max(t * float(diffusivity) / float(dx) ** 2 * 8.0 + SOLVER_STEP_FLOOR, t * float(rho) * 1.1)
    return int(np.ceil(n))


def steps_per_day_for(mode: str, diffusivity: float, rho: float, resection_time: float, dx: float) -> int:
    """
    The steps per day of a patient's solves. "fixed": BASE_STEPS_PER_DAY.
    "stability": dt = min(DT_MAX, DT_SAFETY dx^2 / (6 D)) with D the
    patient's white-matter diffusivity (mm^2/day) and dx the grid spacing
    (mm), rounded down to an integer number of steps per day
    (ceil(1 / dt)), then raised to the solver's estimate at the truth's
    resection_time (ceil(``solver_step_estimate`` / resection_time)) so
    that the truth's growth stage is stepped as requested, and capped at
    BASE_STEPS_PER_DAY, the finest step (the solver may still refine a
    solve that needs more than that many steps per day, as it does in
    the fixed mode).
    """
    if mode not in DT_MODES:
        raise ValueError(f"--dt-mode must be one of {DT_MODES}, got {mode!r}.")
    if mode == "fixed":
        return BASE_STEPS_PER_DAY
    diffusivity, dx, t_r = float(diffusivity), float(dx), float(resection_time)
    if not (diffusivity > 0 and dx > 0 and t_r > 0):
        raise ValueError(f"the stability step needs D > 0, dx > 0 and resection_time > 0, got {diffusivity!r}, {dx!r}, {t_r!r}.")
    dt = min(DT_MAX, DT_SAFETY * dx**2 / (6.0 * diffusivity))
    steps = int(np.ceil(1.0 / dt - 1e-9))
    steps = max(steps, int(np.ceil(solver_step_estimate(diffusivity, rho, t_r, dx) / t_r - 1e-9)))
    return int(min(max(steps, 1), BASE_STEPS_PER_DAY))


def grid_spacing_mm(base: Mapping[str, Any], zooms: Sequence[float]) -> float:
    """The smallest spacing of the solver's grid in mm: the smallest voxel
    size over the base config's resolution_factor (4 mm in the smoke)."""
    return float(min(float(z) for z in zooms)) / float(base["resolution_factor"])


def maturity(growth_efolds: NDArray | float, front_width: NDArray | float, seed_sigma: NDArray | float) -> NDArray:
    """2 growth_efolds front_width_mm^2 / seed_sigma_mm^2 = ell lambda /
    sigma^2 (ell = v T_r = 2 a lambda the front travel): the front travel
    over the seed's forgetting distance sigma^2 / lambda; at or above
    MATURITY_SPLIT the truth is mature, below it immature."""
    a = np.asarray(growth_efolds, dtype=np.float64)
    width = np.asarray(front_width, dtype=np.float64)
    sigma = np.asarray(seed_sigma, dtype=np.float64)
    return 2.0 * a * width**2 / sigma**2


def cell_of(front_width: float, maturity: float) -> str:
    """The cell of a candidate: (front_width_mm at or below / above
    LAMBDA_SPLIT: compact / broad) x (maturity below / at or above
    MATURITY_SPLIT: immature / mature)."""
    broad, mature = front_width > LAMBDA_SPLIT, maturity >= MATURITY_SPLIT
    for name, (is_broad, is_mature) in CELLS.items():
        if is_broad == broad and is_mature == mature:
            return name
    raise AssertionError("unreachable")


def sample_candidates(
    space: Any,
    script_factors: Mapping[str, Any],
    base: Mapping[str, Any],
    treatment: Mapping[str, float],
    chemo_total_dose: float,
    n_fractions: int,
    log2_candidates: int,
    seed: int,
    tr_max: float = DEFAULT_TR_MAX,
) -> list[dict[str, Any]]:
    """
    The candidates of a design: a scrambled Sobol' sequence over the
    sampled factors (unit cube, 2 ** log2_candidates points) transformed
    with the factors' ranges (the search space's and the script's) and
    derived with the space's groups; resection_time = growth_efolds / rho,
    ell_mm = v resection_time, the log kill (``total_log_kill``), the
    visibility margin and the maturity (``maturity``); the cell
    (``cell_of``) and "rejected": None, or "seed_wide" for a seed_sigma_mm
    above SEED_WIDTH_CAP front_width_mm (checked first), else
    "T_r_below_min" / "T_r_above_max" for a resection_time outside
    [TR_MIN, tr_max].

    Returns:
        One record per Sobol' point in sequence order (the DESIGN_COLUMNS
        without patient, fd_check, the sizes and R_over_lambda, plus
        "rejected").
    """
    factors = {**space.factors, **script_factors}
    missing = [name for name in SAMPLED_FACTORS if name not in factors]
    if missing:
        raise ValueError(f"the search space lacks the factors {missing}.")
    if GROWTH_EFOLDS_FACTOR not in script_factors:
        raise ValueError(f"{GROWTH_EFOLDS_FACTOR} must be a factor of the script ({SCRIPT_FACTOR_KEY!r}: true), not a solver parameter.")
    sampler = qmc.Sobol(d=len(SAMPLED_FACTORS), scramble=True, seed=int(seed))
    u = np.asarray(sampler.random_base2(int(log2_candidates)), dtype=np.float64)
    values = {name: factors[name].transform(u[:, column]) for column, name in enumerate(SAMPLED_FACTORS)}
    derived = space.derive(values)
    rho = np.asarray(derived["rho"], dtype=np.float64)
    resection_time = values[GROWTH_EFOLDS_FACTOR] / rho
    kill = total_log_kill(
        values["rt_alpha"],
        float(base["rt_alpha_beta_ratio"]),
        n_fractions,
        float(treatment["rt_dose_per_fraction_gy"]),
        values["chemo_kill_rate"],
        chemo_total_dose,
        float(base["chemo_decay_rate"]),
    )
    margin = visibility_margin(rho, kill["log_kill_total"])
    widths = values[sa.GROWTH_WIDTH_FACTOR]
    sigmas = values[sa.SEED_SIGMA_FACTOR]
    mature = maturity(values[GROWTH_EFOLDS_FACTOR], widths, sigmas)
    records: list[dict[str, Any]] = []
    for i in range(len(u)):
        t_r = float(resection_time[i])
        if float(sigmas[i]) > SEED_WIDTH_CAP * float(widths[i]):
            rejected: str | None = "seed_wide"
        else:
            rejected = "T_r_below_min" if t_r < TR_MIN else ("T_r_above_max" if t_r > float(tr_max) else None)
        record: dict[str, Any] = {
            "candidate": int(i),
            "cell": cell_of(float(widths[i]), float(mature[i])),
            "rejected": rejected,
        }
        record.update({f"u_{name}": float(u[i, column]) for column, name in enumerate(SAMPLED_FACTORS)})
        record.update({name: float(values[name][i]) for name in SAMPLED_FACTORS})
        record["resection_time"] = t_r
        record.update({key: float(derived[key][i]) for key in DESIGN_DERIVED})
        record["rho_T_r"] = float(values[GROWTH_EFOLDS_FACTOR][i])
        record["ell_mm"] = float(values[sa.GROWTH_SPEED_FACTOR][i]) * t_r
        record.update({key: float(kill[key][i]) for key in ("log_kill_rt", "log_kill_ct", "log_kill_total")})
        record["visibility_margin"] = float(margin[i])
        record["maturity"] = float(mature[i])
        records.append(record)
    return records


def candidate_patient(record: Mapping[str, Any]) -> Patient:
    """The Patient of a candidate record (id c<candidate>)."""
    return Patient(
        id=f"c{int(record['candidate']):04d}",
        cell=str(record["cell"]),
        front_speed=float(record[sa.GROWTH_SPEED_FACTOR]),
        front_width=float(record[sa.GROWTH_WIDTH_FACTOR]),
        resection_time=float(record["resection_time"]),
        rt_alpha=float(record["rt_alpha"]),
        chemo_kill_rate=float(record["chemo_kill_rate"]),
        seed_peak=float(record[sa.SEED_PEAK_FACTOR]),
        seed_sigma=float(record[sa.SEED_SIGMA_FACTOR]),
        maturity=float(record["maturity"]),
        steps_per_day=None if record.get("steps_per_day") is None else int(record["steps_per_day"]),
    )


def equivalent_radius(volume: float) -> float:
    """The radius in mm of the sphere of a volume in mm^3: (3 V / 4 pi)^(1/3)."""
    return float((3.0 * float(volume) / (4.0 * np.pi)) ** (1.0 / 3.0))


def size_screen(
    density: NDArray, tissue: NDArray, zooms: Sequence[float], core_level: float = sa.TAU_CORE, whole_level: float = sa.TAU_EDEMA
) -> dict[str, float]:
    """
    The size measures of a density on the tissue mask: the voxel counts
    n_core and n_whole of {u >= core_level} and {u >= whole_level}, the
    equivalent-sphere radii r_core_mm and r_whole_mm of their volumes
    (``equivalent_radius``; 0 for an empty set), the volume ratio
    whole_core_ratio (NaN for an empty core) and max_density.
    """
    density = np.asarray(density, dtype=np.float64)
    voxel_volume = float(np.prod(np.asarray(zooms, dtype=np.float64)))
    n_core = int(np.count_nonzero(np.logical_and(density >= core_level, tissue)))
    n_whole = int(np.count_nonzero(np.logical_and(density >= whole_level, tissue)))
    return {
        "n_core": n_core,
        "n_whole": n_whole,
        "r_core_mm": equivalent_radius(voxel_volume * n_core),
        "r_whole_mm": equivalent_radius(voxel_volume * n_whole),
        "whole_core_ratio": n_whole / n_core if n_core > 0 else float("nan"),
        "max_density": float(density.max()) if density.size else 0.0,
    }


def accept_sizes(sizes: Mapping[str, Any], bands: SizeBands = DEFAULT_BANDS) -> str | None:
    """None when the sizes lie within every band, else the reason of the
    rejection: "empty_core", or the first of r_core_mm, r_whole_mm,
    whole_core_ratio (SIZE_COLUMNS' order) outside its band."""
    if int(sizes["n_core"]) == 0:
        return "empty_core"
    for name in SIZE_COLUMNS:
        lo, hi = getattr(bands, name)
        value = float(sizes[name])
        if not (np.isfinite(value) and lo <= value <= hi):
            return name
    return None


def requested_steps(horizon: float, steps_per_day: int) -> int:
    """The step count of a horizon in days at a whole number of steps per
    day, as the solver translates it: ceil(horizon steps_per_day) (the
    effective step horizon / n is then at most 1 / steps_per_day)."""
    return int(np.ceil(float(horizon) * int(steps_per_day) - 1e-9))


def dt_refined(n_steps: int, horizon: float, patient: Patient) -> bool:
    """Whether a growth solve over the horizon took more steps than the
    patient's own step asks for (the solver's estimate was stricter);
    False for a patient without a step of its own."""
    return patient.steps_per_day is not None and int(n_steps) > requested_steps(horizon, patient.steps_per_day)


def screen_candidate(cohort: Cohort, record: Mapping[str, Any], screen_dir: Path) -> dict[str, Any]:
    """
    The screening record of a candidate: read back from
    screen_dir/c<candidate>/screen.json when it exists (its factor values
    must match the candidate's, else the design was resumed with another
    seed or search space), otherwise the growth-only solve of the
    candidate to its resection_time (the truth's growth path: the
    candidate's time step, the cohort's seed voxel; ``solve_growth``) measured by
    ``size_screen`` on the 1 mm grid and written there.

    Returns:
        candidate, the sampled factors, resection_time, rho,
        steps_per_day (the candidate's step), n_core, n_whole, r_core_mm,
        r_whole_mm, whole_core_ratio, max_density, n_steps, dt (the
        solver's), dt_refined (the solver stepped finer than requested),
        wall_time_s, failed and error (the solver's message when the solve
        raised; the sizes are then NaN).
    """
    keys = (*SAMPLED_FACTORS, "resection_time", "rho", "steps_per_day")
    path = screen_dir / f"c{int(record['candidate']):04d}" / SCREEN_FILE
    if path.is_file():
        stored = read_record(path)
        for name in keys:
            if not np.isclose(float(stored[name]), float(record[name]), rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"{path} holds another candidate ({name} {stored[name]!r}, the design's {record[name]!r}): "
                    "the design was resumed with a different seed, search space or time step; remove the directory."
                )
        return stored
    patient = candidate_patient(record)
    out: dict[str, Any] = {"candidate": int(record["candidate"]), "cell": str(record["cell"])}
    out.update({name: float(record[name]) for name in keys if name != "steps_per_day"})
    out["steps_per_day"] = int(record["steps_per_day"])
    start = time.perf_counter()
    try:
        result = solve_growth(cohort, growth_config(patient_config(cohort, patient, FISHER_HORIZON)))
    except RuntimeError as error:
        out.update({name: float("nan") for name in ("n_core", "n_whole", *SIZE_COLUMNS, "max_density", "n_steps", "dt")})
        out.update({"dt_refined": None, "failed": True, "error": str(error)})
    else:
        out.update(size_screen(np.asarray(result.final_state["cell_density"], dtype=np.float64), cohort.tissue, cohort.zooms))
        assert result.dt is not None
        assert result.n_steps is not None
        out.update({"n_steps": result.n_steps, "dt": result.dt, "dt_refined": dt_refined(result.n_steps, patient.resection_time, patient), "failed": False, "error": None})
    out["wall_time_s"] = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    write_record(path, out)
    return out


def screen_cohort(
    cohort: Cohort,
    candidates: Sequence[Mapping[str, Any]],
    bands: SizeBands,
    patients_per_cell: int,
    screen_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    The size screening: per cell (CELLS' order) the admissible candidates
    (not rejected by their seed width or resection_time) are screened in
    sequence order
    (``screen_candidate``, ``accept_sizes``) until patients_per_cell are
    accepted. A failed solve rejects the candidate and is counted;
    MAX_CONSECUTIVE_SCREEN_FAILURES failures in a row raise. When a cell
    cannot be filled the screening of the other cells still runs and the
    error then names every short cell with the counts.

    Returns:
        (records, counts): the design records (``DESIGN_COLUMNS``; ids
        p00, p01, ... in cell order, the first of each cell flagged
        fd_check, R_over_lambda = r_whole_mm / front_width_mm) and the
        counts: n_candidates, n_rejected_seed_wide,
        n_rejected_T_r_below_min, n_rejected_T_r_above_max, n_screened,
        n_accepted, n_solve_failed and per cell n_candidates,
        n_admissible, n_screened, n_accepted, n_solve_failed and
        n_rejected by reason.
    """
    counts: dict[str, Any] = {
        "n_candidates": len(candidates),
        "n_rejected_seed_wide": sum(1 for c in candidates if c["rejected"] == "seed_wide"),
        "n_rejected_T_r_below_min": sum(1 for c in candidates if c["rejected"] == "T_r_below_min"),
        "n_rejected_T_r_above_max": sum(1 for c in candidates if c["rejected"] == "T_r_above_max"),
        "n_screened": 0,
        "n_accepted": 0,
        "n_solve_failed": 0,
        "cells": {},
    }
    accepted_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell in CELLS:
        queue = [c for c in candidates if c["cell"] == cell and c["rejected"] is None]
        cell_counts: dict[str, Any] = {
            "n_candidates": sum(1 for c in candidates if c["cell"] == cell),
            "n_admissible": len(queue),
            "n_screened": 0,
            "n_accepted": 0,
            "n_solve_failed": 0,
            "n_rejected": {reason: 0 for reason in ("empty_core", *SIZE_COLUMNS)},
        }
        accepted: list[dict[str, Any]] = []
        consecutive = 0
        for candidate in queue:
            if len(accepted) >= patients_per_cell:
                break
            screened = screen_candidate(cohort, candidate, screen_dir)
            cell_counts["n_screened"] += 1
            label = f"c{int(candidate['candidate']):04d} ({cell}, T_r {candidate['resection_time']:.0f} d, maturity {candidate['maturity']:.1f})"
            if screened["failed"]:
                cell_counts["n_solve_failed"] += 1
                consecutive += 1
                print(f"  screen {label}: solve failed: {screened['error']}", flush=True)
                if consecutive >= MAX_CONSECUTIVE_SCREEN_FAILURES:
                    raise RuntimeError(f"the screening failed {consecutive} solves in a row (last: {screened['error']}); the design stops.")
                continue
            consecutive = 0
            reason = accept_sizes(screened, bands)
            verdict = "accepted" if reason is None else f"rejected ({reason})"
            print(
                f"  screen {label}: r_core {screened['r_core_mm']:.1f} mm, r_whole {screened['r_whole_mm']:.1f} mm, "
                f"ratio {screened['whole_core_ratio']:.2f}: {verdict} ({screened['wall_time_s']:.1f} s)",
                flush=True,
            )
            if reason is None:
                accepted.append({**candidate, **{key: screened[key] for key in SIZE_COLUMNS}})
            else:
                cell_counts["n_rejected"][reason] += 1
        cell_counts["n_accepted"] = len(accepted)
        counts["cells"][cell] = cell_counts
        accepted_by_cell[cell] = accepted
    for key in ("n_screened", "n_accepted", "n_solve_failed"):
        counts[key] = sum(int(c[key]) for c in counts["cells"].values())
    short = [cell for cell, accepted in accepted_by_cell.items() if len(accepted) < patients_per_cell]
    if short:
        raise ValueError(
            f"the cells {short} could not be filled with {patients_per_cell} patients each from {len(candidates)} Sobol' points "
            f"(compact / broad at front_width_mm {LAMBDA_SPLIT:g}, immature / mature at maturity {MATURITY_SPLIT:g}, seeds wider than "
            f"{SEED_WIDTH_CAP:g} front widths rejected; counts {json.dumps(jsonable(counts))}); raise --log2-candidates or widen the bands."
        )
    records: list[dict[str, Any]] = []
    for cell in CELLS:
        for position, candidate in enumerate(accepted_by_cell[cell]):
            record = {key: value for key, value in candidate.items() if key != "rejected"}
            record["patient"] = f"p{len(records):02d}"
            record["fd_check"] = position == 0
            record["R_over_lambda"] = float(record["r_whole_mm"]) / float(record[sa.GROWTH_WIDTH_FACTOR])
            records.append(record)
    return records, counts


def make_design(
    config_path: str | Path,
    search_space_path: str | Path,
    output_dir: str | Path,
    name: str,
    tissue_maps: Mapping[str, str | Path | None] | None = None,
    seed_voxel: Sequence[int] | None = None,
    log2_candidates: int = DEFAULT_LOG2_CANDIDATES,
    seed: int = DESIGN_SEED,
    smoke: bool = False,
    cavity_threshold: float = sa.CAVITY_THRESHOLD,
    rt_margin_mm: float = sa.RT_MARGIN_MM,
    rt_dose_per_fraction: float = sa.RT_DOSE_PER_FRACTION_GY,
    tr_max: float = DEFAULT_TR_MAX,
    bands: SizeBands = DEFAULT_BANDS,
    patients_per_cell: int = PATIENTS_PER_CELL,
    dt_mode: str = DEFAULT_DT_MODE,
    device: str = "",
) -> Path:
    """
    Draw and screen the cohort and write the design directory
    <output_dir>/<name>: base_config.json, search_space.json, screen/
    (the screening records), spec.json, design.csv and
    configs/<patient>.json. A directory holding spec.json is never
    overwritten; one without it (an interrupted screening) is resumed
    from its screen/ records.

    Args:
        config_path: The base config (resection_cavity and rt_dose null).
        search_space_path: The search space (the script's default,
            ``load_script_search_space``).
        output_dir: Parent of the design directory.
        name: The design directory name.
        tissue_maps: NIfTI paths replacing the base config's
            white_matter_pbmap / gray_matter_pbmap (None or '' keeps the
            base config's).
        seed_voxel: The seed voxel (i, j, k) of every patient, snapped to
            the nearest seedable voxel; default DEFAULT_SEED_VOXEL where
            the grid holds it, else the base config's fractions.
        log2_candidates: 2 ** log2_candidates Sobol' points to fill the
            cells from.
        seed: Seed of the scrambled Sobol' sequence.
        smoke: Record the smoke mode and set the base config's
            resolution_factor to the smoke setting.
        cavity_threshold: See ``treatment_settings``.
        rt_margin_mm: See ``treatment_settings``.
        rt_dose_per_fraction: See ``treatment_settings``.
        tr_max: Candidates with resection_time above it (days) are
            rejected; also the cap of the substitute arm's T_0.
        bands: The size screening's acceptance bands (``SizeBands``).
        patients_per_cell: Patients accepted per cell.
        dt_mode: The time step of every patient's solves ("fixed" |
            "stability", ``steps_per_day_for``; recorded in spec.json and
            per patient in design.csv).
        device: The screening device, for the record ('' the CPU).

    Returns:
        The design directory.
    """
    root = Path(output_dir) / name
    if (root / "spec.json").is_file():
        raise FileExistsError(f"{root} holds a design (spec.json); a design is never overwritten.")
    resumed = (root / SCREEN_DIR).is_dir()
    config_path = Path(config_path).resolve()
    search_space_path = Path(search_space_path).resolve()
    base = read_config(config_path, solver=StuppFKPPSolver)
    for key, path in (tissue_maps or {}).items():
        if key not in sa.DEFAULT_TISSUE_MAPS:
            raise ValueError(f"tissue_maps: {key!r} is not one of {sorted(sa.DEFAULT_TISSUE_MAPS)}.")
        if path is not None and str(path):
            base[key] = str(Path(path).resolve())
    missing = [key for key in sa.BASE_KEYS_NEEDED if base.get(key) is None]
    if missing:
        raise ValueError(f"the base config {config_path} lacks {missing}, which the design needs.")
    given = [key for key in sa.DERIVED_VOLUME_KEYS if base.get(key) is not None]
    if given:
        raise ValueError(f"the base config {config_path} sets {given}, which every run derives; set them to null.")
    if all(base.get(key) is None for key in sa.TIME_STEP_KEYS):
        raise ValueError(f"the base config {config_path} sets none of {list(sa.TIME_STEP_KEYS)}; set steps_per_day.")
    if int(patients_per_cell) < 1:
        raise ValueError(f"patients_per_cell must be at least 1, got {patients_per_cell!r}.")
    if dt_mode not in DT_MODES:
        raise ValueError(f"dt_mode must be one of {DT_MODES}, got {dt_mode!r}.")
    base["precision"] = PRECISION
    base["gaussian_seed_floor"] = SEED_FLOOR
    base["snapshot_times"] = None
    if smoke:
        base["resolution_factor"] = SMOKE.resolution_factor
    space, script_factors = load_script_search_space(search_space_path, StuppFKPPSolver.config_keys())
    scale = space.overrides.get("gaussian_seed_scale", base["gaussian_seed_scale"])
    if float(scale) != 1.0:
        raise ValueError(f"the seed derivation needs gaussian_seed_scale = 1, got {scale!r}.")
    horizon_override = space.overrides.get("time_after_resection")
    if horizon_override is not None and float(horizon_override) != FISHER_HORIZON:
        raise ValueError(
            f"the search space fixes time_after_resection at {horizon_override!r}, the fisher horizon is {FISHER_HORIZON:g}."
        )
    # The search space's fixed overrides go into the base config (the
    # horizon is set per run; the seed scale is 1 either way).
    for key, value in space.overrides.items():
        if key not in ("time_after_resection", "gaussian_seed_scale"):
            base[key] = value
    base["gaussian_seed_scale"] = 1.0
    treatment = sa.treatment_settings(cavity_threshold, rt_margin_mm, rt_dose_per_fraction)
    # The growth stage's instance of the base config validates the entries
    # and holds the loaded maps and the resolved flux threshold.
    growth = FKPPSolver(sa.growth_config(base))
    wm, gm = growth.params["white_matter_pbmap"], growth.params["gray_matter_pbmap"]
    min_tissue_fraction = float(growth.params["min_tissue_fraction"])
    geometry = sa.seed_geometry(wm, gm, min_tissue_fraction)
    shape = np.asarray(geometry.shape, dtype=np.float64)
    if seed_voxel is None:
        if all(0 <= v < n for v, n in zip(DEFAULT_SEED_VOXEL, geometry.shape)):
            target, target_source = list(DEFAULT_SEED_VOXEL), "default"
        else:
            target = [int(float(base[key]) * n) for key, n in zip(sa.SEED_KEYS, geometry.shape)]
            target_source = "base_config_fractions"
    else:
        target = [int(v) for v in seed_voxel]
        if len(target) != 3 or any(v < 0 or v >= n for v, n in zip(target, geometry.shape)):
            raise ValueError(f"--seed-voxel must be three indices within the grid {geometry.shape}, got {seed_voxel!r}.")
        target_source = "argument"
    voxel, snap_distance = nearest_seedable_voxel(geometry, target)
    fractions = tuple(float(v) for v in (np.asarray(voxel, dtype=np.float64) + 0.5) / shape)
    for key, fraction in zip(sa.SEED_KEYS, fractions):
        base[key] = fraction
    schedules = {
        "fisher": sa.truncate_schedule(base, {"time_after_resection": FISHER_HORIZON})["schedule"],
        "substitute": sa.truncate_schedule(base, {"time_after_resection": SUBSTITUTE_HORIZON})["schedule"],
    }
    snapshots = sa.crt_snapshot_offsets(base["rt_times"], base["chemo_times"], base["resection_time"])
    expected = {"mid_crt": FRAME_MOMENTS["d34"] - 1.0, "end_crt": FRAME_MOMENTS["d55"] - 1.0}
    if {k: float(v) for k, v in snapshots.items()} != expected:
        raise ValueError(f"the base schedule's CRT Sundays are {snapshots}, the frames assume {expected}.")
    first_adjuvant = first_adjuvant_offset(base["chemo_times"], base["rt_times"], base["resection_time"])
    if first_adjuvant != FRAME_MOMENTS["d80"]:
        raise ValueError(f"the first adjuvant dose is at offset {first_adjuvant}, the d80 frame assumes {FRAME_MOMENTS['d80']:g}.")
    n_fractions = int(schedules["fisher"]["n_fractions"])
    chemo_total_dose = float(schedules["fisher"]["chemo_total_dose"])
    candidates = sample_candidates(space, script_factors, base, treatment, chemo_total_dose, n_fractions, log2_candidates, seed, tr_max)
    wm_array, gm_array, zooms, affine = load_tissue(base)
    dx = grid_spacing_mm(base, zooms)
    for candidate in candidates:
        steps = steps_per_day_for(dt_mode, candidate["white_matter_diffusivity"], candidate["rho"], candidate["resection_time"], dx)
        candidate["steps_per_day"], candidate["dt"] = steps, 1.0 / steps
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("configs", "runs", "figures", SCREEN_DIR):
        (root / sub).mkdir(exist_ok=True)
    shutil.copyfile(search_space_path, root / "search_space.json")
    write_config(base, root / "base_config.json")
    screening_cohort = Cohort(
        root=root,
        spec={"treatment": treatment, "min_tissue_fraction": min_tissue_fraction},
        base=base,
        wm=wm_array,
        gm=gm_array,
        zooms=zooms,
        affine=affine,
        tissue=(wm_array + gm_array) >= min_tissue_fraction,
        seed_voxel=(voxel[0], voxel[1], voxel[2]),
        seed_fractions=(fractions[0], fractions[1], fractions[2]),
        patients=[],
    )
    print(
        f"screening {len(candidates)} candidates ({sum(1 for c in candidates if c['rejected'] is None)} with T_r in "
        f"[{TR_MIN:g}, {tr_max:g}] days and a seed at most {SEED_WIDTH_CAP:g} front widths wide) for {patients_per_cell} patients per cell on "
        f"{device or 'cpu'}, dt mode {dt_mode} (dx {dx:g} mm){' (resumed)' if resumed else ''}",
        flush=True,
    )
    start = time.perf_counter()
    records, counts = screen_cohort(screening_cohort, candidates, bands, int(patients_per_cell), root / SCREEN_DIR)
    screening_time = time.perf_counter() - start
    factors = {**space.factors, **script_factors}
    spec: dict[str, Any] = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "base_config": str(config_path),
        "search_space_path": str(search_space_path),
        "tissue_maps": {key: base[key] for key in sa.DEFAULT_TISSUE_MAPS},
        "grid_shape": list(geometry.shape),
        "precision": PRECISION,
        "gaussian_seed_floor": SEED_FLOOR,
        "resolution_factor": float(base["resolution_factor"]),
        "smoke": bool(smoke),
        "smoke_settings": {
            "resolution_factor": SMOKE.resolution_factor,
            "patients_per_cell": SMOKE.patients_per_cell,
            "n_patients": SMOKE.n_patients,
            "draws": SMOKE.draws,
            "maxfev": SMOKE.maxfev,
            "sigmas": list(SMOKE.sigmas),
        },
        "design_seed": int(seed),
        "log2_candidates": int(log2_candidates),
        "n_candidates": 2 ** int(log2_candidates),
        "patients_per_cell": int(patients_per_cell),
        "cells": {name: {"broad": broad, "mature": mature} for name, (broad, mature) in CELLS.items()},
        "lambda_split_mm": LAMBDA_SPLIT,
        "maturity_split": MATURITY_SPLIT,
        "maturity": {"formula": "maturity = 2 growth_efolds front_width_mm^2 / seed_sigma_mm^2 (= ell lambda / sigma^2); mature at or above maturity_split"},
        "seed_width_cap": {"formula": f"a candidate with seed_sigma_mm > {SEED_WIDTH_CAP:g} front_width_mm is rejected (seed_wide)", "cap": SEED_WIDTH_CAP},
        "visibility_horizon_days": VISIBILITY_HORIZON,
        "sampled_factors": list(SAMPLED_FACTORS),
        "script_factors": list(script_factors),
        "factors": {name: {"min": factors[name].low, "max": factors[name].high, "scale": factors[name].scale} for name in SAMPLED_FACTORS},
        "fixed_overrides": dict(space.overrides),
        "fixed_parameters": {key: base[key] for key in FIXED_PARAMETERS},
        "seed_voxel": list(voxel),
        "seed_fractions": list(fractions),
        "seed_target_voxel": target,
        "seed_target_source": target_source,
        "default_seed_voxel": list(DEFAULT_SEED_VOXEL),
        "seed_snap_distance_voxels": snap_distance,
        "min_tissue_fraction": min_tissue_fraction,
        "n_seedable_voxels": geometry.n_voxels,
        "time_step": {key: base[key] for key in sa.TIME_STEP_KEYS},
        "dt_mode": dt_mode,
        "dt_modes": {
            "fixed": f"{BASE_STEPS_PER_DAY} steps per day for every patient (the base config's)",
            "stability": (
                f"per patient dt = min({DT_MAX:g}, {DT_SAFETY:g} dx^2 / (6 D_wm)) rounded down to an integer number of steps per day, raised to the "
                f"solver's estimate ceil(max(8 D T_r / dx^2 + {SOLVER_STEP_FLOOR}, 1.1 rho T_r) / T_r) and capped at {BASE_STEPS_PER_DAY} steps per day"
            ),
            "dt_max_days": DT_MAX,
            "dt_safety": DT_SAFETY,
            "base_steps_per_day": BASE_STEPS_PER_DAY,
            "grid_spacing_mm": dx,
        },
        "treatment": {**treatment, "n_fractions": n_fractions, "rt_total_dose_gy": treatment["rt_dose_per_fraction_gy"] * n_fractions},
        "schedules": schedules,
        "log_kill": {
            "formula": "Lambda = n_fractions d alpha (1 + d / rt_alpha_beta_ratio) + chemo_kill_rate D_tot / chemo_decay_rate, D_tot the chemo dose sum within the fisher horizon",
            "chemo_total_dose": chemo_total_dose,
            "chemo_decay_rate": float(base["chemo_decay_rate"]),
            "rt_alpha_beta_ratio": float(base["rt_alpha_beta_ratio"]),
        },
        "visibility": {"formula": f"visibility_margin = {VISIBILITY_HORIZON:g} rho - Lambda; at or above 0 the untreated regrowth outruns the log kill by day {VISIBILITY_HORIZON:g} (a design column, not a cell split)"},
        "screening": {
            "tr_min": TR_MIN,
            "tr_max": float(tr_max),
            "bands": bands.record(),
            "levels": {"core": sa.TAU_CORE, "whole": sa.TAU_EDEMA},
            "radius": "equivalent-sphere radius (3 V / 4 pi)^(1/3) of the set on the tissue mask, mm, on the 1 mm grid",
            "device": device or "cpu",
            "max_consecutive_failures": MAX_CONSECUTIVE_SCREEN_FAILURES,
            "resumed": resumed,
            "wall_time_s": screening_time,
            "counts": counts,
        },
        "frame_moments": FRAME_MOMENTS,
        "crt_snapshots": snapshots,
        "horizons": {"fisher": FISHER_HORIZON, "substitute": SUBSTITUTE_HORIZON},
        "frames": {"fisher": list(FISHER_FRAMES), "substitute": list(SUBSTITUTE_FRAMES)},
        "invariance_lambdas": list(INVARIANCE_LAMBDAS),
        "fisher": {
            "theta": list(THETA_NAMES),
            "seed_theta": list(SEED_THETA_NAMES),
            "fd_step": FD_STEP,
            "fd_check_step": FD_CHECK_STEP,
            "t_r_variants": list(T_R_VARIANTS),
            "invariance_direction": INVARIANCE_DIRECTION.tolist(),
            "observation_sets": {key: list(value) for key, value in OBSERVATION_SETS.items()},
            "indicator_levels": list(INDICATOR_LEVELS),
            "indicator_width": INDICATOR_WIDTH,
            "observation_margin_mm": OBSERVATION_MARGIN_MM,
            "noise_draws": SMOKE.draws if smoke else NOISE_DRAWS,
            "threshold_ranges": {str(level): list(bounds) for level, bounds in THRESHOLD_RANGES.items()},
            "displacement_std_mm": DISPLACEMENT_STD_MM,
            "displacement_length_mm": DISPLACEMENT_LENGTH_MM,
            "variance_floor": VARIANCE_FLOOR,
        },
        "substitute": {
            "delta_a": list(SUBSTITUTE_DELTA_A),
            "T_0": "T_r - delta_a / rho; skipped below T_0_min or above T_0_max",
            "T_0_min": TR_MIN,
            "T_0_max": float(tr_max),
            "maxfev": SMOKE.maxfev if smoke else SUBSTITUTE_MAXFEV,
            "peak_range": [PEAK_MIN, PEAK_MAX],
            "objectives": list(OBJECTIVES),
            "simplex_steps": list(SIMPLEX_STEPS),
        },
        "seedfix": {
            "seed": seed_record(DEFAULT_SEED_PEAK, DEFAULT_SEED_SIGMA_MM),
            "T_r_bounds": list(TR_BOUNDS),
            "lambda_bounds": list(LAMBDA_BOUNDS),
            "lambda_modes": list(LAMBDA_MODES),
            "maxfev": SMOKE.maxfev if smoke else MAXFEV,
            "objectives": list(OBJECTIVES),
            "bound_tolerance": BOUND_TOLERANCE,
            "simplex_steps": list(SIMPLEX_STEPS),
            "default_lambda_mode": DEFAULT_LAMBDA_MODE,
        },
        "profile": {
            "sigmas": list(SMOKE.sigmas if smoke else PROFILE_SIGMAS),
            "seed_peak": DEFAULT_SEED_PEAK,
            "objective": OBJECTIVES[0],
            "maxfev": SMOKE.maxfev if smoke else MAXFEV,
            "patients": "per cell the fd_check patient and the next accepted one (the fd_check patients alone with smoke)",
        },
        "summary_strata": {"maturity_split": MATURITY_SPLIT, "groups": [f"maturity_ge_{MATURITY_SPLIT:g}", f"maturity_lt_{MATURITY_SPLIT:g}"]},
        "n_patients": len(records),
    }
    sa.write_json(root / "spec.json", spec)
    sa.write_csv(root / "design.csv", records, DESIGN_COLUMNS)
    cohort = load_cohort(root)
    for patient in cohort.patients:
        write_config(patient_config(cohort, patient, FISHER_HORIZON), root / "configs" / f"{patient.id}.json")
    return root


# --- run configs and solves ---


def patient_config(cohort: Cohort, patient: Patient, horizon: float) -> dict[str, Any]:
    """
    The treated config of a patient: the base config with the patient's
    parameters and the cohort's seed voxel, time_after_resection = horizon,
    the schedule truncated to the horizon (``truncate_schedule``) and
    shifted by resection_time - base resection_time; resection_cavity and
    rt_dose null, snapshot_times null (the run sets them).
    """
    base = cohort.base
    config: dict[str, Any] = {SOLVER_KEY: sa.SOLVER_NAME}
    config.update({key: value for key, value in base.items() if key != SOLVER_KEY})
    config.update(patient.solver_values())
    for key, fraction in zip(sa.SEED_KEYS, cohort.seed_fractions):
        config[key] = fraction
    config["time_after_resection"] = float(horizon)
    schedule = sa.truncate_schedule(base, {"time_after_resection": float(horizon)})
    shift = patient.resection_time - float(base["resection_time"])
    config["chemo_doses"] = schedule["chemo_doses"]
    for key in sa.SHIFTED_TIME_KEYS:
        config[key] = [float(t) + shift for t in schedule[key]]
    for key in sa.DERIVED_VOLUME_KEYS:
        config[key] = None
    config["snapshot_times"] = None
    return config


def growth_config(config: Mapping[str, Any], n_steps: int | None = None) -> dict[str, Any]:
    """The growth stage's FKPPSolver config of a treated config
    (``growth_config`` of the script: stopping_time = resection_time), with
    n_steps in place of the base time step when given."""
    growth = dict(sa.growth_config(config))
    if n_steps is not None:
        for key in sa.TIME_STEP_KEYS:
            growth[key] = None
        growth["n_steps"] = int(n_steps)
    return growth


@dataclass(frozen=True)
class TreatmentMaps:
    """The derived cavity and dose map of a truth, in memory and on disk."""

    cavity: NDArray
    dose: NDArray
    cavity_path: Path
    dose_path: Path
    record: dict[str, Any]

    def config_entries(self) -> dict[str, Any]:
        return {
            "resection_cavity": {"segmentation": str(self.cavity_path), "label": sa.CAVITY_LABEL},
            "rt_dose": str(self.dose_path),
        }


def _in_memory(cohort: Cohort, config: Mapping[str, Any], maps: TreatmentMaps | None) -> dict[str, Any]:
    """The config with the cohort's tissue arrays (and the maps' arrays) in
    place of the paths, so a solve reads no NIfTI; the fields are the
    same. voxel_size_mm is set to the header's zooms, which the path form
    resolves itself."""
    params = dict(config)
    params["white_matter_pbmap"] = cohort.wm
    params["gray_matter_pbmap"] = cohort.gm
    params["voxel_size_mm"] = list(cohort.zooms)
    if maps is not None and "resection_cavity" in params:
        params["resection_cavity"] = maps.cavity
        params["rt_dose"] = maps.dose
    return params


def _check_result(result: Result, what: str, dt: float | None = None) -> None:
    if not result.success:
        raise RuntimeError(f"{what}: the solver failed: {result.error}")
    if dt is not None and result.dt is not None and abs(result.dt - dt) > 1e-9 * dt:
        raise RuntimeError(
            f"{what}: the solver stepped at dt={result.dt:g} instead of the requested {dt:g} "
            "(its stability estimate was stricter); the run is not on the intended time grid."
        )


def solve_growth(cohort: Cohort, config: Mapping[str, Any], dt: float | None = None) -> Result:
    """Solve a growth-only FKPPSolver config (tissue maps as paths) with
    the cohort's arrays; the Result carries the path config and the
    cohort's affine. Raises on failure or when the solver changed dt."""
    result = FKPPSolver(_in_memory(cohort, config, None)).solve()
    _check_result(result, "growth stage", dt)
    result.config = dict(config)
    result.affine = cohort.affine
    return result


def save_growth(result: Result, run_dir: Path, keep_final: bool = False) -> None:
    """The growth stage's record and config into run_dir/growth (no
    volumes, like the script; the final field rounded when kept)."""
    result.initial_state = {}
    result.final_state = {key: sa.round_field(value) for key, value in result.final_state.items()} if keep_final else {}
    result.time_series = None
    result.save(run_dir / sa.GROWTH_DIR, overwrite=True)


def derive_maps(cohort: Cohort, density: NDArray, n_fractions: int, run_dir: Path) -> TreatmentMaps:
    """
    The cavity and dose map of a density at resection_time
    (``treatment_maps`` with the cohort's treatment settings; the dose is
    rt_dose_per_fraction_gy times n_fractions), saved into run_dir as
    the script saves them (resection_cavity.nii.gz, rt_dose.nii.gz,
    treatment.json).
    """
    treatment = cohort.treatment
    total_dose = float(treatment["rt_dose_per_fraction_gy"]) * int(n_fractions)
    cavity, dose = sa.treatment_maps(density, cohort.zooms, treatment["cavity_threshold"], treatment["rt_margin_mm"], total_dose)
    run_dir.mkdir(parents=True, exist_ok=True)
    cavity_path, dose_path = run_dir / sa.CAVITY_FILE, run_dir / sa.DOSE_FILE
    nib.save(nib.Nifti1Image(cavity.astype(np.uint8) * sa.CAVITY_LABEL, cohort.affine), str(cavity_path))
    nib.save(nib.Nifti1Image(dose, cohort.affine), str(dose_path))
    voxel_volume = float(np.prod(cohort.zooms))
    n_cavity, n_dose = int(cavity.sum()), int(np.count_nonzero(dose))
    record = {
        **treatment,
        "n_fractions": int(n_fractions),
        "rt_total_dose_gy": total_dose,
        "max_density_at_resection": float(density.max()),
        "voxel_volume_mm3": voxel_volume,
        "n_cavity_voxels": n_cavity,
        "n_dose_voxels": n_dose,
        "cavity_volume_mm3": voxel_volume * n_cavity,
        "dose_volume_mm3": voxel_volume * n_dose,
        "cavity_file": sa.CAVITY_FILE,
        "dose_file": sa.DOSE_FILE,
    }
    sa.write_json(run_dir / sa.TREATMENT_FILE, record)
    return TreatmentMaps(cavity=cavity, dose=dose, cavity_path=cavity_path, dose_path=dose_path, record=record)


def frame_days(resection_time: float, dt: float, moments: Mapping[str, float]) -> dict[str, float]:
    """
    The days the treated stage records its frames on, for snapshot_times:
    for each frame the state after the last step whose end lies at least
    half a step before the moment resection_time + moment (the script's
    snapshot rounding, ``snapshot_days``, which takes the moment as
    resection_time + offset + 1: the offset handed to it is moment - 1).
    """
    return {
        str(name): float(day)
        for name, day in sa.snapshot_days(float(resection_time), {name: float(moment) - 1.0 for name, moment in moments.items()}, dt).items()
    }


@dataclass
class RunOutput:
    """A treated run: its saved config, the recorded frames (unrounded,
    full-resolution float64) and the time stepping."""

    run_dir: Path
    config: dict[str, Any]
    frames: dict[str, NDArray]
    days: dict[str, float]
    n_growth: int
    dt: float
    n_steps: int
    wall_time_s: float


def treated_run(
    cohort: Cohort,
    config: Mapping[str, Any],
    maps: TreatmentMaps,
    n_growth: int,
    dt: float,
    frames: Sequence[str],
    run_dir: Path,
    growth: Result | None = None,
    extra_moments: Mapping[str, float] | None = None,
) -> RunOutput:
    """
    The treated stage of a config on the growth stage's time grid
    (``align_treated_config``: n_steps = n_growth + ceil(time_after / dt),
    resection_time stated as the midpoint of step n_growth), recording the
    frames (``frame_days``; FRAME_MOMENTS plus extra moments by name) and
    saving into run_dir: config.json (the aligned config with the maps'
    paths, which StuppFKPPSolver(read_config(...)) reproduces),
    result.json, final_cell_density.nii.gz and <frame>_cell_density.nii.gz
    (float32, rounded for storage), frames.json (the requested and the
    recorded days), and the growth stage's record into growth/ if given.

    Raises when the solver fails, changes dt or drops a frame.
    """
    moments = {name: FRAME_MOMENTS[name] for name in frames}
    moments.update({str(k): float(v) for k, v in (extra_moments or {}).items()})
    treated = dict(config)
    treated.update(maps.config_entries())
    aligned = dict(sa.align_treated_config(treated, n_growth, dt))
    days = frame_days(float(config["resection_time"]), dt, moments)
    horizon = float(aligned["resection_time"]) + float(aligned["time_after_resection"])
    late = {name: day for name, day in days.items() if day > horizon}
    if late:
        raise ValueError(f"the frames {late} lie beyond the treated stage's horizon {horizon:g}.")
    aligned["snapshot_times"] = sorted(set(days.values()))
    result = StuppFKPPSolver(_in_memory(cohort, aligned, maps)).solve()
    _check_result(result, f"treated stage {run_dir}", dt)
    if result.time_series is None or result.snapshot_times is None:
        raise RuntimeError(f"treated stage {run_dir}: no frames were recorded.")
    recorded = np.asarray(result.snapshot_times, dtype=np.float64)
    series = result.time_series["cell_density"]
    fields: dict[str, NDArray] = {}
    record: dict[str, Any] = {}
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, day in days.items():
        index = int(np.argmin(np.abs(recorded - day)))
        if not np.isclose(recorded[index], day, rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"treated stage {run_dir}: frame {name} at day {day:g} was not recorded (recorded {recorded.tolist()}).")
        fields[name] = np.asarray(series[index], dtype=np.float64)
        file = FRAME_FILE.format(name=name)
        save_field(run_dir / file, fields[name], cohort.affine)
        record[name] = {"moment_offset": moments[name], "requested_day": day, "recorded_day": float(recorded[index]), "file": file}
    result.initial_state = {}
    result.final_state = {key: sa.round_field(value) for key, value in result.final_state.items()}
    result.time_series = None
    result.config = aligned
    result.affine = cohort.affine
    result.save(run_dir, overwrite=True)
    sa.write_json(
        run_dir / FRAMES_FILE,
        {"resection_time": float(config["resection_time"]), "dt": dt, "steps_per_day": config.get("steps_per_day"), "n_growth": n_growth, "frames": record},
    )
    if growth is not None:
        save_growth(growth, run_dir)
    assert result.n_steps is not None and result.wall_time_s is not None
    return RunOutput(
        run_dir=run_dir,
        config=aligned,
        frames=fields,
        days=days,
        n_growth=int(n_growth),
        dt=float(dt),
        n_steps=int(result.n_steps),
        wall_time_s=float(result.wall_time_s),
    )


@dataclass
class TruthRun:
    """A truth: its treated run, the derived maps and the growth stage's
    density at resection_time (the field the cavity is thresholded from)."""

    patient: Patient
    config: dict[str, Any]
    run: RunOutput
    maps: TreatmentMaps
    density: NDArray

    @property
    def n_growth(self) -> int:
        return self.run.n_growth

    @property
    def dt(self) -> float:
        return self.run.dt

    def stepping(self) -> dict[str, Any]:
        """The record entries of the truth's time step: n_growth, dt (the
        solver's), steps_per_day (the patient's) and dt_refined."""
        return {
            "n_growth": self.n_growth,
            "dt": self.dt,
            "steps_per_day": self.patient.steps_per_day,
            "dt_refined": dt_refined(self.n_growth, self.patient.resection_time, self.patient),
        }


def truth_run(cohort: Cohort, patient: Patient, horizon: float, frames: Sequence[str], run_dir: Path) -> TruthRun:
    """
    A patient's truth: the growth stage at the patient's time step (the
    base config's without one; saved into run_dir/growth), its density at
    resection_time (saved as
    pre_resection_cell_density.nii.gz), the maps derived from it
    (``derive_maps``) and the treated stage with them (``treated_run``).
    """
    config = patient_config(cohort, patient, horizon)
    growth = solve_growth(cohort, growth_config(config))
    density = np.asarray(growth.final_state["cell_density"], dtype=np.float64)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_field(run_dir / sa.PRE_RESECTION_FILE, density, cohort.affine)
    maps = derive_maps(cohort, density, len(config["rt_times"]), run_dir)
    assert growth.n_steps is not None and growth.dt is not None
    if dt_refined(growth.n_steps, patient.resection_time, patient):
        print(f"  {patient.id}: the solver stepped the truth at dt={growth.dt:g} instead of {patient.dt:g} (its estimate was stricter)", flush=True)
    run = treated_run(cohort, config, maps, growth.n_steps, growth.dt, frames, run_dir, growth)
    return TruthRun(patient=patient, config=config, run=run, maps=maps, density=density)


# --- metrics ---


def _bounding_box(mask: NDArray, margin: int = 1) -> tuple[slice, slice, slice]:
    """The bounding box of a nonempty mask, grown by margin voxels within
    the array."""
    indices = np.argwhere(mask)
    lo = np.maximum(indices.min(axis=0) - margin, 0)
    hi = np.minimum(indices.max(axis=0) + margin + 1, mask.shape)
    return (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2])))


def dice(a: NDArray, b: NDArray) -> float:
    """2 |A and B| / (|A| + |B|); NaN when both sets are empty."""
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return float("nan")
    return 2.0 * int(np.logical_and(a, b).sum()) / total


def surface_distance(a: NDArray, b: NDArray, zooms: Sequence[float]) -> float:
    """
    The average symmetric surface distance in mm of two masks: the mean,
    over the surface voxels of both (a voxel of the set with a face
    neighbour outside it; the array boundary is not a surface), of the
    Euclidean distance (distance_transform_edt with the zooms) to the
    other set's surface. NaN when either set is empty.
    """
    if not a.any() or not b.any():
        return float("nan")
    box = _bounding_box(np.logical_or(a, b))
    a_box, b_box = a[box], b[box]
    surface_a = np.logical_and(a_box, ~binary_erosion(a_box, border_value=1))
    surface_b = np.logical_and(b_box, ~binary_erosion(b_box, border_value=1))
    sampling = np.asarray(zooms, dtype=np.float64)
    to_b = distance_transform_edt(~surface_b, sampling=sampling)
    to_a = distance_transform_edt(~surface_a, sampling=sampling)
    distances = np.concatenate([to_b[surface_a], to_a[surface_b]])
    return float(distances.mean())


def rel_l2_log(field: NDArray, reference: NDArray) -> float:
    """||log(u + eps) - log(u* + eps)||_2 / ||log(u* + eps)||_2 over the
    given voxels (eps = LOG_EPS)."""
    log_reference = np.log(reference + LOG_EPS)
    return float(np.linalg.norm(np.log(field + LOG_EPS) - log_reference) / np.linalg.norm(log_reference))


def compare_fields(
    field: NDArray,
    reference: NDArray,
    tissue: NDArray,
    zooms: Sequence[float],
    dose: NDArray,
    seed_voxel: Sequence[int],
    wm: NDArray,
) -> dict[str, float]:
    """
    The metrics between a field u and a reference u* on the tissue mask
    (``METRIC_NAMES``): Dice of {u >= 0.6} and of {u >= 0.3}, the average
    symmetric surface distance in mm of each iso-surface
    (``surface_distance``; NaN when either set is empty), the relative L2
    of log(u + 1e-6) - log(u* + 1e-6) (``rel_l2_log``), the relative L2
    ||u - u*|| / ||u*||, the max abs difference, the masses of u (dV sum u
    in total, outside the dose map's support (dose == 0: the out-of-field
    tail) and beyond the reference's edema (the tissue voxels with
    u* < 0.3)), the same three of u* (ref_<name>), the relative masses
    <name> / ref_<name> - 1 (NaN for a zero reference), the log volume
    ratios log10(V + dV) - log10(V* + dV) of the 0.6 and the 0.3 sets, and
    the script's QoIs of u (``compute_qois``, prefixed qoi_).
    """
    field = np.asarray(field, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    voxel_volume = float(np.prod(np.asarray(zooms, dtype=np.float64)))
    u, u_ref = field[tissue], reference[tissue]
    out_of_field = np.logical_and(tissue, np.asarray(dose) <= 0)
    beyond_edema = np.logical_and(tissue, reference < sa.TAU_EDEMA)
    out: dict[str, float] = {}
    for label, level in (("core", sa.TAU_CORE), ("edema", sa.TAU_EDEMA)):
        a = np.logical_and(field >= level, tissue)
        b = np.logical_and(reference >= level, tissue)
        out[f"dice_{label}"] = dice(a, b)
        out[f"assd_{label}_mm"] = surface_distance(a, b, zooms)
        out[f"log_vol_ratio_{label}"] = float(np.log10(voxel_volume * (int(a.sum()) + 1)) - np.log10(voxel_volume * (int(b.sum()) + 1)))
    out["rel_l2_log"] = rel_l2_log(u, u_ref)
    norm_ref = float(np.linalg.norm(u_ref))
    out["rel_l2"] = float(np.linalg.norm(u - u_ref) / norm_ref) if norm_ref > 0 else float("nan")
    out["max_abs_diff"] = float(np.max(np.abs(u - u_ref))) if u.size else 0.0
    out["mass"] = voxel_volume * float(u.sum())
    out["mass_out_of_field"] = voxel_volume * float(field[out_of_field].sum())
    out["mass_beyond_edema"] = voxel_volume * float(field[beyond_edema].sum())
    out["ref_mass"] = voxel_volume * float(u_ref.sum())
    out["ref_mass_out_of_field"] = voxel_volume * float(reference[out_of_field].sum())
    out["ref_mass_beyond_edema"] = voxel_volume * float(reference[beyond_edema].sum())
    for name in MASS_NAMES:
        ref = out[f"ref_{name}"]
        out[f"{name}_rel"] = out[name] / ref - 1.0 if ref > 0 else float("nan")
    qois = sa.compute_qois(field, zooms, seed_voxel, wm)
    out.update({f"qoi_{name}": float(value) for name, value in qois.items()})
    return out


def compare_to(cohort: Cohort, field: NDArray, reference: NDArray, dose: NDArray) -> dict[str, float]:
    """``compare_fields`` with the cohort's tissue mask, zooms, seed voxel
    and white-matter map."""
    return compare_fields(field, reference, cohort.tissue, cohort.zooms, dose, cohort.seed_voxel, cohort.wm)


# --- observations (experiment 1) ---


@dataclass(frozen=True)
class Region:
    """The observed voxels: a mask, its bounding box and the voxel
    indices (n, 3)."""

    mask: NDArray
    box: tuple[slice, slice, slice]
    voxels: NDArray

    @property
    def n(self) -> int:
        return int(len(self.voxels))

    def record(self) -> dict[str, Any]:
        return {
            "n_voxels": self.n,
            "box_lo": [int(s.start) for s in self.box],
            "box_hi": [int(s.stop) for s in self.box],
        }


def observation_region(
    frames: Mapping[str, NDArray],
    tissue: NDArray,
    zooms: Sequence[float],
    margin_mm: float = OBSERVATION_MARGIN_MM,
    level: float = sa.TAU_EDEMA,
) -> Region:
    """
    The tissue voxels within margin_mm of the union over the frames of
    the truth's iso-surface {u* >= level}: the Euclidean distance (in mm,
    distance_transform_edt with the zooms) of a voxel outside the union
    to the union, of a voxel inside to its complement.
    """
    union = np.zeros(tissue.shape, dtype=bool)
    for frame in frames.values():
        union |= np.asarray(frame) >= level
    if not union.any():
        raise ValueError(f"no voxel of the truth reaches {level:g} at any frame; the observation region is empty.")
    sampling = np.asarray(zooms, dtype=np.float64)
    distance = distance_transform_edt(~union, sampling=sampling) + distance_transform_edt(union, sampling=sampling)
    mask = np.logical_and(tissue, distance <= float(margin_mm))
    voxels = np.argwhere(mask).astype(np.int64)
    return Region(mask=mask, box=_bounding_box(mask, margin=0), voxels=voxels)


def smoothed_indicator(u: NDArray, level: float, width: float = INDICATOR_WIDTH) -> NDArray:
    """s(u) = 1 / (1 + exp(-(u - level) / width))."""
    return np.asarray(expit((np.asarray(u, dtype=np.float64) - level) / width), dtype=np.float64)


def observation_vector(
    frames: Mapping[str, NDArray],
    names: Sequence[str],
    region: Region,
    levels: Sequence[float] = INDICATOR_LEVELS,
    width: float = INDICATOR_WIDTH,
) -> NDArray:
    """The observations of a run: the smoothed indicators at each level of
    each frame on the region's voxels, (n_frames, n_levels, n_voxels)."""
    index = tuple(region.voxels.T)
    out = np.empty((len(names), len(levels), region.n), dtype=np.float64)
    for f, name in enumerate(names):
        values = np.asarray(frames[name], dtype=np.float64)[index]
        for l, level in enumerate(levels):
            out[f, l] = smoothed_indicator(values, level, width)
    return out


def noise_variance(
    frames: Mapping[str, NDArray],
    names: Sequence[str],
    region: Region,
    zooms: Sequence[float],
    draws: int,
    rng: np.random.Generator,
    levels: Sequence[float] = INDICATOR_LEVELS,
    width: float = INDICATOR_WIDTH,
    threshold_ranges: Mapping[float, tuple[float, float]] = THRESHOLD_RANGES,
    displacement_std_mm: float = DISPLACEMENT_STD_MM,
    displacement_length_mm: float = DISPLACEMENT_LENGTH_MM,
    variance_floor: float = VARIANCE_FLOOR,
) -> NDArray:
    """
    The per-observation noise variance: over `draws` draws, each with one
    threshold per level drawn uniformly from threshold_ranges (shared by
    the frames of the draw) and, per frame, a random displacement field
    (three independent Gaussian random fields: white noise smoothed with
    a Gaussian kernel of standard deviation displacement_length_mm,
    scaled to a standard deviation of displacement_std_mm per component),
    the truth's frame is sampled at x + d(x) by map_coordinates (linear)
    on the region's voxels and the indicators are taken at the drawn
    thresholds; the variance over the draws, floored at variance_floor,
    is returned as (n_frames, n_levels, n_voxels).
    """
    zooms_array = np.asarray(zooms, dtype=np.float64)
    smooth = displacement_length_mm / zooms_array  # kernel std per axis, voxels
    # Filtered white noise has the variance prod_i 1 / (2 sqrt(pi) s_i).
    scale = float(np.sqrt(np.prod(2.0 * np.sqrt(np.pi) * smooth))) * displacement_std_mm
    pad = int(np.ceil((3.0 * displacement_length_mm + 4.0 * displacement_std_mm) / zooms_array.min()))
    shape = np.asarray(region.mask.shape)
    lo = np.maximum([s.start for s in region.box], 0)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum([s.stop for s in region.box] + np.array([pad] * 3), shape)
    sub = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2])))
    sub_shape = tuple(int(h - l) for l, h in zip(lo, hi))
    local = region.voxels - lo  # the region's voxels within the sub-box
    local_index = tuple(local.T)
    coords0 = local.T.astype(np.float64)
    cropped = {name: np.asarray(frames[name], dtype=np.float64)[sub] for name in names}
    total = np.zeros((len(names), len(levels), region.n), dtype=np.float64)
    squares = np.zeros_like(total)
    for _ in range(int(draws)):
        thresholds = [float(rng.uniform(*threshold_ranges[level])) for level in levels]
        for f, name in enumerate(names):
            coords = coords0.copy()
            for axis in range(3):
                field = gaussian_filter(rng.standard_normal(sub_shape), smooth, truncate=3.0)
                coords[axis] += field[local_index] * scale / zooms_array[axis]
            warped = map_coordinates(cropped[name], coords, order=1, mode="nearest")
            for l, threshold in enumerate(thresholds):
                s = smoothed_indicator(warped, threshold, width)
                total[f, l] += s
                squares[f, l] += s * s
    mean = total / draws
    variance = np.maximum(squares / draws - mean * mean, 0.0)
    return np.maximum(variance, variance_floor)


# --- Fisher analysis (experiment 1) ---


def cramer_rao(fisher: NDArray, tolerance: float = SINGULAR_TOLERANCE) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    The Cramer-Rao standard errors sqrt((F^-1)_ii) of a Fisher matrix,
    its eigenvalues (ascending) and the loadings of its weakest
    eigenvector (the component of largest magnitude made positive) and
    its condition number. An eigenvalue at or below tolerance times the
    largest counts as zero: the inverse is taken on the other
    eigenvectors and a parameter with a loading above sqrt(tolerance) on
    a zero eigenvector gets an infinite error (the condition number is
    then inf).
    """
    values, vectors = np.linalg.eigh(np.asarray(fisher, dtype=np.float64))
    largest = float(values[-1]) if values.size else 0.0
    zero = values <= tolerance * max(largest, np.finfo(np.float64).tiny)
    inverse_values = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, values))
    covariance = (vectors * inverse_values) @ vectors.T
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if zero.any():
        unidentifiable = np.any(np.abs(vectors[:, zero]) > np.sqrt(tolerance), axis=1)
        errors = np.where(unidentifiable, np.inf, errors)
        condition = float("inf")
    else:
        condition = float(values[-1] / values[0])
    weakest = vectors[:, 0].copy()
    weakest *= np.sign(weakest[np.argmax(np.abs(weakest))]) or 1.0
    return errors, values, weakest, condition


def projection_residual(matrix: NDArray, column: int) -> NDArray:
    """The residual of one column after its least-squares projection onto
    the other columns."""
    others = np.delete(matrix, column, axis=1)
    target = matrix[:, column]
    coefficients = np.linalg.lstsq(others, target, rcond=None)[0]
    return np.asarray(target - others @ coefficients, dtype=np.float64)


def _fraction(residual: NDArray, target: NDArray, rows: NDArray | None = None) -> tuple[float, float, int]:
    """(||r|| / ||t||, ||t||, n) over the given rows (all when None); the
    fraction is NaN when ||t|| is 0."""
    r = residual if rows is None else residual[rows]
    t = target if rows is None else target[rows]
    norm = float(np.linalg.norm(t))
    return (float(np.linalg.norm(r) / norm) if norm > 0 else float("nan"), norm, int(t.size))


def fisher_analysis(
    weighted: NDArray,
    frame_of_row: NDArray,
    frame_names: Sequence[str],
    region_of_row: Mapping[str, NDArray],
    sets: Mapping[str, Sequence[str]] = OBSERVATION_SETS,
    direction: NDArray = INVARIANCE_DIRECTION,
) -> dict[str, dict[str, Any]]:
    """
    The Fisher analysis of a weighted Jacobian W = Sigma^{-1/2} J with the
    columns ``COLUMN_NAMES`` (theta, then the two seed parameters), per
    observation set (the rows of its frames):

    - |W e| for the invariance direction e (over the theta columns) and
      |W e_1| (the log v column) for scale;
    - the eigenvalues of F = W^T W over the theta columns, the loadings
      of its weakest eigenvector and its condition number;
    - the Cramer-Rao standard errors of the theta parameters
      (``cramer_rao``; that of log T_r is sqrt((F^-1)_33));
    - the residual fraction of the T_r column after its least-squares
      projection onto the other four theta columns;
    - from all seven columns: the Cramer-Rao errors of log T_r and of the
      two seed parameters, and the residual fraction of the T_r column
      after projection onto the other six, overall and split by region
      (``region_of_row``: the rows whose truth density is at or above
      0.6, between 0.3 and 0.6, or whose voxel lies outside the dose
      support) and by frame.

    Returns:
        A record per set: n_rows, We_norm, We1_norm, We_ratio, eig_<i>,
        weak_<theta>, cond_F, cr_<theta>, resid_frac_T_r,
        resid7_frac_T_r, cr7_log_T_r, cr7_peak, cr7_log_sigma, and
        "splits": [{"split": "region" | "day", "name", "resid7_frac",
        "norm_T_r", "n_rows"}, ...].
    """
    n_theta = len(THETA_NAMES)
    out: dict[str, dict[str, Any]] = {}
    for set_name, set_frames in sets.items():
        frame_indices = [frame_names.index(name) for name in set_frames if name in frame_names]
        if len(frame_indices) != len(set_frames):
            continue
        rows = np.isin(frame_of_row, frame_indices)
        w_all = weighted[rows]
        w_theta = w_all[:, :n_theta]
        record: dict[str, Any] = {"n_rows": int(rows.sum()), "frames": list(set_frames)}
        record["We_norm"] = float(np.linalg.norm(w_theta @ direction))
        record["We1_norm"] = float(np.linalg.norm(w_theta[:, 0]))
        record["We_ratio"] = record["We_norm"] / record["We1_norm"] if record["We1_norm"] > 0 else float("nan")
        errors, values, weakest, condition = cramer_rao(w_theta.T @ w_theta)
        record.update({f"eig_{i + 1}": float(values[i]) for i in range(n_theta)})
        record.update({f"weak_{name}": float(weakest[i]) for i, name in enumerate(THETA_NAMES)})
        record["cond_F"] = condition
        record.update({f"cr_{name}": float(errors[i]) for i, name in enumerate(THETA_NAMES)})
        residual = projection_residual(w_theta, T_R_INDEX)
        record["resid_frac_T_r"] = _fraction(residual, w_theta[:, T_R_INDEX])[0]
        errors7, _, _, _ = cramer_rao(w_all.T @ w_all)
        record["cr7_log_T_r"] = float(errors7[T_R_INDEX])
        for i, name in enumerate(SEED_THETA_NAMES):
            record[f"cr7_{name}"] = float(errors7[n_theta + i])
        residual7 = projection_residual(w_all, T_R_INDEX)
        target7 = w_all[:, T_R_INDEX]
        record["resid7_frac_T_r"], record["norm7_T_r"], _ = _fraction(residual7, target7)
        splits: list[dict[str, Any]] = []
        for region_name, region_rows in region_of_row.items():
            fraction, norm, n = _fraction(residual7, target7, region_rows[rows])
            splits.append({"split": "region", "name": region_name, "resid7_frac": fraction, "norm_T_r": norm, "n_rows": n})
        frames_rows = frame_of_row[rows]
        for f in frame_indices:
            fraction, norm, n = _fraction(residual7, target7, frames_rows == f)
            splits.append({"split": "day", "name": frame_names[f], "resid7_frac": fraction, "norm_T_r": norm, "n_rows": n})
        record["splits"] = splits
        out[set_name] = record
    return out


@dataclass(frozen=True)
class Perturbation:
    """One finite-difference run: the perturbed patient, the size of the
    step in its column's parameter and the growth stage's time stepping."""

    column: str
    sign: str
    patient: Patient
    step: float
    growth_n_steps: int
    dt: float
    variant: str = ""

    @property
    def name(self) -> str:
        suffix = f"_{self.variant}" if self.variant else ""
        return f"{self.column}{suffix}_{self.sign}"


def perturbations(
    patient: Patient, n_growth: int, dt: float, step: float = FD_STEP, columns: Sequence[str] = COLUMN_NAMES, t_r_variants: Sequence[str] = T_R_VARIANTS
) -> list[Perturbation]:
    """
    The finite-difference runs of a patient: for each column the plus
    and the minus perturbation of size step (in the log parameter; in the
    peak density additively, the plus step shortened so that the peak
    stays at most 1 and dropped when the peak is 1). The growth columns
    keep the truth's time grid (n_growth, dt). The T_r column comes in
    the variants T_R_VARIANTS: "scaled_dt" keeps n_growth steps, so the
    growth stage runs at dt' = T_r e^{+-h} / n_growth (the growth
    invariance holds exactly); "fixed_dt" keeps dt and rounds T_r e^{+-h}
    to a whole number of steps (the effective step is the rounded log
    ratio).
    """
    out: list[Perturbation] = []
    for column in columns:
        for sign, direction in (("plus", 1.0), ("minus", -1.0)):
            factor = float(np.exp(direction * step))
            if column == "log_v":
                out.append(Perturbation(column, sign, replace(patient, front_speed=patient.front_speed * factor), step, n_growth, dt))
            elif column == "log_lambda":
                out.append(Perturbation(column, sign, replace(patient, front_width=patient.front_width * factor), step, n_growth, dt))
            elif column == "log_alpha":
                out.append(Perturbation(column, sign, replace(patient, rt_alpha=patient.rt_alpha * factor), step, n_growth, dt))
            elif column == "log_k_ct":
                out.append(Perturbation(column, sign, replace(patient, chemo_kill_rate=patient.chemo_kill_rate * factor), step, n_growth, dt))
            elif column == "log_sigma":
                out.append(Perturbation(column, sign, replace(patient, seed_sigma=patient.seed_sigma * factor), step, n_growth, dt))
            elif column == "peak":
                h = min(step, PEAK_MAX - patient.seed_peak) if direction > 0 else step
                if h <= 0:
                    continue
                out.append(Perturbation(column, sign, replace(patient, seed_peak=patient.seed_peak + direction * h), h, n_growth, dt))
            elif column == "log_T_r":
                for variant in t_r_variants:
                    if variant == "scaled_dt":
                        t_r = patient.resection_time * factor
                        out.append(Perturbation(column, sign, replace(patient, resection_time=t_r), step, n_growth, t_r / n_growth, variant))
                    elif variant == "fixed_dt":
                        k = int(np.rint(patient.resection_time * factor / dt))
                        if k == n_growth or k < 1:
                            raise ValueError(f"the fixed-dt T_r step of {patient.id} rounds to no whole step (dt={dt:g}).")
                        t_r = k * dt
                        out.append(Perturbation(column, sign, replace(patient, resection_time=t_r), abs(float(np.log(t_r / patient.resection_time))), k, dt, variant))
                    else:
                        raise ValueError(f"unknown T_r variant {variant!r}.")
            else:
                raise ValueError(f"unknown column {column!r}.")
    return out


def _difference(plus: NDArray | None, minus: NDArray | None, truth: NDArray, h_plus: float, h_minus: float) -> NDArray:
    """The central difference (y+ - y-) / (h+ + h-), one-sided against the
    truth when one side is missing."""
    if plus is None and minus is None:
        raise ValueError("a column needs at least one perturbation.")
    y_plus = truth if plus is None else plus
    y_minus = truth if minus is None else minus
    return (y_plus - y_minus) / (h_plus + h_minus)


def fisher_patient(cohort: Cohort, patient: Patient, out_dir: Path, draws: int, fd_check: bool, seed: int) -> dict[str, Any]:
    """
    Experiment 1 for one patient: the truth (``truth_run``, horizon
    FISHER_HORIZON, frames FISHER_FRAMES), the observation region and
    vector, the noise variance (``noise_variance``, `draws` draws), the
    finite-difference runs (``perturbations``; each a treated run with the
    truth's maps), the Jacobians (the theta columns with the T_r column in
    both variants, plus the two seed columns), the metrics of every run
    against the truth per frame, the Fisher analysis per T_r variant
    (``fisher_analysis``) and, with fd_check, the theta columns again at
    half the step. Writes out_dir/<patient>/: truth/, <run>/, observation.json,
    sigma.npz, W.npz (float32, when below W_MAX_BYTES) and fisher.json
    (the record returned, the resume marker).
    """
    patient_dir = out_dir / patient.id
    start = time.perf_counter()
    truth = truth_run(cohort, patient, FISHER_HORIZON, FISHER_FRAMES, patient_dir / "truth")
    frames = list(FISHER_FRAMES)
    region = observation_region(truth.run.frames, cohort.tissue, cohort.zooms)
    y0 = observation_vector(truth.run.frames, frames, region)
    rng = np.random.default_rng(seed)
    noise_start = time.perf_counter()
    variance = noise_variance(truth.run.frames, frames, region, cohort.zooms, draws, rng)
    noise_time = time.perf_counter() - noise_start
    np.savez_compressed(patient_dir / "sigma.npz", variance=variance.astype(np.float32), frames=np.asarray(frames), levels=np.asarray(INDICATOR_LEVELS))
    dose = truth.maps.dose
    run_metrics: list[dict[str, Any]] = []
    observations: dict[str, NDArray] = {}
    steps: dict[str, float] = {}
    runs = perturbations(patient, truth.n_growth, truth.dt)
    if fd_check:
        runs += [replace(p, variant="fd_check") for p in perturbations(patient, truth.n_growth, truth.dt, FD_CHECK_STEP, THETA_NAMES, ("scaled_dt",))]
    solve_time = 0.0
    for perturbation in runs:
        config = patient_config(cohort, perturbation.patient, FISHER_HORIZON)
        run = treated_run(cohort, config, truth.maps, perturbation.growth_n_steps, perturbation.dt, frames, patient_dir / perturbation.name)
        solve_time += run.wall_time_s
        observations[perturbation.name] = observation_vector(run.frames, frames, region)
        steps[perturbation.name] = perturbation.step
        for name in frames:
            metrics = compare_to(cohort, run.frames[name], truth.run.frames[name], dose)
            run_metrics.append({"patient": patient.id, "cell": patient.cell, "run": perturbation.name, "column": perturbation.column, "sign": perturbation.sign, "variant": perturbation.variant, "step": perturbation.step, "frame": name, "n_steps": run.n_steps, "dt": run.dt, **metrics})
        print(f"  {patient.id} {perturbation.name}: {run.n_steps} steps, dt {run.dt:.5f}, {run.wall_time_s:.1f} s", flush=True)

    def column(name: str, variant: str = "") -> NDArray:
        suffix = f"_{variant}" if variant else ""
        plus = observations.get(f"{name}{suffix}_plus")
        minus = observations.get(f"{name}{suffix}_minus")
        return _difference(plus, minus, y0, steps.get(f"{name}{suffix}_plus", 0.0), steps.get(f"{name}{suffix}_minus", 0.0))

    n_rows = y0.size
    frame_of_row = np.repeat(np.arange(len(frames)), len(INDICATOR_LEVELS) * region.n)
    truth_of_row = np.concatenate([np.tile(np.asarray(truth.run.frames[name])[tuple(region.voxels.T)], len(INDICATOR_LEVELS)) for name in frames])
    out_of_field = np.tile(np.asarray(dose)[tuple(region.voxels.T)] <= 0, len(frames) * len(INDICATOR_LEVELS))
    region_of_row = {
        "core": truth_of_row >= sa.TAU_CORE,
        "rim": np.logical_and(truth_of_row >= sa.TAU_EDEMA, truth_of_row < sa.TAU_CORE),
        "out_of_field": out_of_field,
    }
    inverse_sigma = 1.0 / np.sqrt(variance.reshape(-1))
    jacobians: dict[str, NDArray] = {}
    analysis: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in T_R_VARIANTS:
        columns = [column(name, variant if name == "log_T_r" else "").reshape(-1) for name in COLUMN_NAMES]
        jacobian = np.stack(columns, axis=1)
        jacobians[variant] = jacobian
        weighted = jacobian * inverse_sigma[:, None]
        analysis[variant] = fisher_analysis(weighted, frame_of_row, frames, region_of_row)
    check: list[dict[str, Any]] = []
    if fd_check:
        reference = jacobians["scaled_dt"]
        half_columns = [column(name, "fd_check").reshape(-1) for name in THETA_NAMES]
        half = np.stack(half_columns, axis=1)
        weighted_half = np.concatenate([half, reference[:, len(THETA_NAMES):]], axis=1) * inverse_sigma[:, None]
        half_analysis = fisher_analysis(weighted_half, frame_of_row, frames, region_of_row)
        for i, name in enumerate(THETA_NAMES):
            full = reference[:, i] * inverse_sigma
            finer = half[:, i] * inverse_sigma
            norm = float(np.linalg.norm(finer))
            check.append({
                "patient": patient.id,
                "cell": patient.cell,
                "column": name,
                "step": FD_STEP,
                "step_half": FD_CHECK_STEP,
                "rel_diff": float(np.linalg.norm(full - finer) / norm) if norm > 0 else float("nan"),
                "We_ratio_a": analysis["scaled_dt"]["a"]["We_ratio"],
                "We_ratio_a_half": half_analysis["a"]["We_ratio"],
                "We_ratio_f": analysis["scaled_dt"]["f"]["We_ratio"],
                "We_ratio_f_half": half_analysis["f"]["We_ratio"],
                "cr_log_T_r_f": analysis["scaled_dt"]["f"]["cr_log_T_r"],
                "cr_log_T_r_f_half": half_analysis["f"]["cr_log_T_r"],
            })
    weighted7 = (jacobians["scaled_dt"] * inverse_sigma[:, None]).astype(np.float32)
    w_bytes = int(weighted7.nbytes)
    if w_bytes < W_MAX_BYTES:
        np.savez_compressed(
            patient_dir / "W.npz",
            W=weighted7,
            columns=np.asarray(COLUMN_NAMES),
            frames=np.asarray(frames),
            levels=np.asarray(INDICATOR_LEVELS),
            voxels=region.voxels.astype(np.int32),
            frame_of_row=frame_of_row.astype(np.int8),
        )
    observation = {
        **region.record(),
        "n_observations": int(n_rows),
        "frames": frames,
        "levels": list(INDICATOR_LEVELS),
        "indicator_width": INDICATOR_WIDTH,
        "margin_mm": OBSERVATION_MARGIN_MM,
        "noise_draws": int(draws),
        "noise_seed": int(seed),
        "variance_floor": VARIANCE_FLOOR,
        "n_rows_at_floor": int((variance <= VARIANCE_FLOOR).sum()),
        "W_bytes": w_bytes,
        "W_saved": w_bytes < W_MAX_BYTES,
    }
    sa.write_json(patient_dir / "observation.json", observation)
    record: dict[str, Any] = {
        "patient": patient.id,
        "cell": patient.cell,
        "theta": dict(zip(THETA_NAMES, patient.theta.tolist())),
        "seed_theta": {"peak": patient.seed_peak, "log_sigma": float(np.log(patient.seed_sigma))},
        "resection_time": patient.resection_time,
        "n_growth": truth.n_growth,
        "dt": truth.dt,
        "cavity_volume_mm3": truth.maps.record["cavity_volume_mm3"],
        "dose_volume_mm3": truth.maps.record["dose_volume_mm3"],
        "observation": observation,
        "steps": steps,
        "analysis": analysis,
        "fd_check": check,
        "run_metrics": run_metrics,
        "wall_time_s": time.perf_counter() - start,
        "solve_time_s": solve_time,
        "noise_time_s": noise_time,
    }
    write_record(patient_dir / PATIENT_FILE.format(experiment="fisher"), record)
    return record


# --- substitute seeds (experiment 2) ---


def bounded_from_unbounded(z: float, lo: float, hi: float) -> float:
    """The bounded coordinate of an unbounded one: lo + (hi - lo) sigmoid(z),
    the fits' transform onto an open range (the peak on (PEAK_MIN,
    PEAK_MAX], log T_r and log lambda on their log bounds)."""
    return float(lo) + (float(hi) - float(lo)) * float(expit(z))


def unbounded_from_bounded(x: float, lo: float, hi: float) -> float:
    """The inverse of ``bounded_from_unbounded``, the unit-interval argument
    clipped to [LOGIT_CLIP, 1 - LOGIT_CLIP] (a value at or beyond a bound
    maps to a finite z)."""
    p = float(np.clip((float(x) - float(lo)) / (float(hi) - float(lo)), LOGIT_CLIP, 1.0 - LOGIT_CLIP))
    return float(np.log(p / (1.0 - p)))


def peak_from_logit(z: float) -> float:
    """The peak density of its logit coordinate: PEAK_MIN + (PEAK_MAX - PEAK_MIN) sigmoid(z)."""
    return bounded_from_unbounded(z, PEAK_MIN, PEAK_MAX)


def logit_from_peak(peak: float) -> float:
    """The inverse of ``peak_from_logit`` (``unbounded_from_bounded``)."""
    return unbounded_from_bounded(peak, PEAK_MIN, PEAK_MAX)


def composition_seed(patient: Patient, t0: float) -> tuple[float, float, bool]:
    """
    The initial guess of the substitute seed at T_0 from the linear
    composition rule: m' = m e^{rho (T_r - T_0)}, w' = w + D (T_r - T_0)
    (a Gaussian seed of diffusion time w grown linearly for T_r - T_0
    days), as (peak, sigma) through the script's seed derivation; the
    truth's seed, flagged False, when w' is not positive.
    """
    seed = patient.seed
    m = seed["gaussian_seed_mass"] * float(np.exp(patient.rho * (patient.resection_time - t0)))
    w = seed["gaussian_seed_diffusion_time"] + patient.diffusivity * (patient.resection_time - t0)
    if w <= 0:
        return patient.seed_peak, patient.seed_sigma, False
    return float(sa.seed_peak_density(m, w)), float(np.sqrt(2.0 * w)), True


def soft_dice(s: NDArray, s_ref: NDArray) -> float:
    """2 sum(s s*) / (sum s^2 + sum s*^2): the Dice of binary fields, 1
    for identical fields; NaN when both fields are 0."""
    total = float((s * s).sum() + (s_ref * s_ref).sum())
    return 2.0 * float((s * s_ref).sum()) / total if total > 0 else float("nan")


def fit_objective(kind: str, field: NDArray, reference: NDArray, levels: Sequence[float] = INDICATOR_LEVELS, width: float = INDICATOR_WIDTH) -> float:
    """
    The fit objectives on the observed voxels: "A" = the relative L2 of
    log(u + 1e-6) - log(u* + 1e-6) (``rel_l2_log``), "B" = 1 - the mean
    over the levels of the soft Dice of the smoothed indicators.
    """
    if kind == "A":
        return rel_l2_log(field, reference)
    if kind == "B":
        dices = [soft_dice(smoothed_indicator(field, level, width), smoothed_indicator(reference, level, width)) for level in levels]
        return 1.0 - float(np.mean(dices))
    raise ValueError(f"unknown objective {kind!r}.")


@dataclass
class SeedFit:
    """The result of ``fit_seed``."""

    objective: str
    t0: float
    rule_peak: float
    rule_sigma: float
    initial_peak: float
    initial_sigma: float
    initial_from_rule: bool
    peak: float
    sigma: float
    n_evaluations: int
    value_initial: float
    value: float
    growth_n_steps: int
    dt: float
    steps_per_day: int | None
    history: list[dict[str, float]]
    wall_time_s: float

    def record(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "T_0": self.t0,
            "rule_peak": self.rule_peak,
            "rule_sigma_mm": self.rule_sigma,
            "initial_peak": self.initial_peak,
            "initial_sigma_mm": self.initial_sigma,
            "initial_from_rule": self.initial_from_rule,
            "fitted_peak": self.peak,
            "fitted_sigma_mm": self.sigma,
            "n_evaluations": self.n_evaluations,
            "objective_initial": self.value_initial,
            "objective_achieved": self.value,
            "growth_n_steps": self.growth_n_steps,
            "steps_per_day": self.steps_per_day,
            "dt": self.dt,
            "wall_time_s": self.wall_time_s,
            "history": self.history,
        }


def fit_seed(
    cohort: Cohort,
    patient: Patient,
    t0: float,
    reference: NDArray,
    region: Region,
    objective: str,
    maxfev: int,
    horizon: float = SUBSTITUTE_HORIZON,
) -> SeedFit:
    """
    Fit the seed (logit of the peak on (PEAK_MIN, PEAK_MAX], log sigma)
    of a growth-only run of T_0 days (the patient's v, lambda, the
    cohort's seed voxel and time step) to the truth's
    density at resection (reference) on the region's voxels, by
    Nelder-Mead (scipy.optimize.minimize; at most maxfev evaluations, the
    initial simplex x0, x0 + (SIMPLEX_STEPS[0], 0), x0 + (0,
    SIMPLEX_STEPS[1]), xatol NELDER_MEAD_XATOL, fatol NELDER_MEAD_FATOL),
    starting from ``composition_seed`` (its peak brought into the fit's
    range by ``logit_from_peak``; the rule's values are recorded too).
    """
    rule_peak, sigma0, from_rule = composition_seed(patient, t0)
    x0 = np.array([logit_from_peak(rule_peak), np.log(sigma0)])
    peak0 = peak_from_logit(float(x0[0]))
    index = tuple(region.voxels.T)
    reference_values = np.asarray(reference, dtype=np.float64)[index]
    history: list[dict[str, float]] = []
    stepping: dict[str, float] = {}

    def evaluate(x: NDArray) -> float:
        peak, sigma = peak_from_logit(float(x[0])), float(np.exp(x[1]))
        candidate = replace(patient, seed_peak=peak, seed_sigma=sigma, resection_time=float(t0))
        result = solve_growth(cohort, growth_config(patient_config(cohort, candidate, horizon)))
        assert result.n_steps is not None and result.dt is not None
        stepping["n_steps"], stepping["dt"] = float(result.n_steps), float(result.dt)
        value = fit_objective(objective, np.asarray(result.final_state["cell_density"], dtype=np.float64)[index], reference_values)
        history.append({"peak": peak, "sigma_mm": sigma, "value": value})
        return value

    start = time.perf_counter()
    simplex = np.array([x0, x0 + np.array([SIMPLEX_STEPS[0], 0.0]), x0 + np.array([0.0, SIMPLEX_STEPS[1]])])
    optimum = minimize(
        evaluate,
        x0,
        method="Nelder-Mead",
        options={"maxfev": int(maxfev), "initial_simplex": simplex, "xatol": NELDER_MEAD_XATOL, "fatol": NELDER_MEAD_FATOL},
    )
    best = min(history, key=lambda h: h["value"])
    return SeedFit(
        objective=objective,
        t0=float(t0),
        rule_peak=rule_peak,
        rule_sigma=sigma0,
        initial_peak=peak0,
        initial_sigma=sigma0,
        initial_from_rule=from_rule,
        peak=best["peak"],
        sigma=best["sigma_mm"],
        n_evaluations=int(optimum.nfev),
        value_initial=history[0]["value"],
        value=best["value"],
        growth_n_steps=int(stepping["n_steps"]),
        dt=stepping["dt"],
        steps_per_day=patient.steps_per_day,
        history=history,
        wall_time_s=time.perf_counter() - start,
    )


def substitute_t0(patient: Patient, delta_a: float) -> float:
    """The substitute's growth time of a deficit delta_a in e-folds:
    T_0 = T_r - delta_a / rho (delta_a > 0: earlier than the truth)."""
    return patient.resection_time - float(delta_a) / patient.rho


def deficit_schedule(patient: Patient, deltas: Sequence[float] = SUBSTITUTE_DELTA_A, tr_max: float = DEFAULT_TR_MAX) -> list[dict[str, Any]]:
    """The substitute arm's (delta_a, T_0) pairs of a patient
    (``substitute_t0``) with "skipped": None, or "T_0_below_min" for a
    T_0 below TR_MIN, "T_0_above_max" for one above tr_max (the design's
    --tr-max)."""
    out: list[dict[str, Any]] = []
    for delta_a in deltas:
        t0 = substitute_t0(patient, float(delta_a))
        skipped = "T_0_below_min" if t0 < TR_MIN else ("T_0_above_max" if t0 > float(tr_max) else None)
        out.append({"delta_a": float(delta_a), "T_0": t0, "skipped": skipped})
    return out


def substitute_row(
    cohort: Cohort,
    patient: Patient,
    delta_a: float,
    t0: float,
    objective: str,
    fit: SeedFit | None,
    metrics: Mapping[str, Mapping[str, float]],
    skipped: str | None = None,
) -> dict[str, Any]:
    """A substitute.csv record: the patient, delta_a, T_0, rho T_0, rho T_r,
    the design's maturity, the objective, skipped (the reason, '' for a run pair), the fit (the
    truth's seed for the "truth" row) and the metrics per frame as
    <frame>_<metric> (NaN for the frames not given, i.e. a skipped
    pair)."""
    row: dict[str, Any] = {
        "patient": patient.id,
        "cell": patient.cell,
        "delta_a": float(delta_a),
        "T_0": float(t0),
        "rho_T_0": patient.rho * float(t0),
        "rho_T_r": patient.rho * patient.resection_time,
        "maturity": patient.maturity,
        "objective": objective,
        "skipped": skipped or "",
    }
    if fit is not None:
        row.update({key: value for key, value in fit.record().items() if key not in ("history", "objective", "T_0")})
    elif objective == "truth":
        row.update({"fitted_peak": patient.seed_peak, "fitted_sigma_mm": patient.seed_sigma, "steps_per_day": patient.steps_per_day})
    for frame in SUBSTITUTE_FRAMES:
        values = metrics.get(frame)
        row.update({f"{frame}_{name}": (float("nan") if values is None else values[name]) for name in METRIC_NAMES})
    return row


SUBSTITUTE_COLUMNS: list[str] = [
    "patient",
    "cell",
    "delta_a",
    "T_0",
    "rho_T_0",
    "rho_T_r",
    "maturity",
    "objective",
    "skipped",
    "rule_peak",
    "rule_sigma_mm",
    "initial_peak",
    "initial_sigma_mm",
    "initial_from_rule",
    "fitted_peak",
    "fitted_sigma_mm",
    "n_evaluations",
    "objective_initial",
    "objective_achieved",
    "growth_n_steps",
    "steps_per_day",
    "dt",
    "wall_time_s",
    *(f"{frame}_{name}" for frame in SUBSTITUTE_FRAMES for name in METRIC_NAMES),
]


def substitute_patient(
    cohort: Cohort,
    patient: Patient,
    out_dir: Path,
    maxfev: int,
    deltas: Sequence[float] = SUBSTITUTE_DELTA_A,
    tr_max: float = DEFAULT_TR_MAX,
) -> dict[str, Any]:
    """
    Experiment 2 for one patient: the truth over SUBSTITUTE_HORIZON with
    the frames SUBSTITUTE_FRAMES, then per deficit delta_a
    (``deficit_schedule``: T_0 = T_r - delta_a / rho; a T_0 below TR_MIN
    or above tr_max is skipped, recorded and given a row of NaN metrics)
    and objective the fitted seed (``fit_seed``; T0_<T_0>/<objective>/
    fit.json), the treated run from it with resection_time T_0, the
    truth's maps, alpha and k_ct and the schedule shifted for T_0
    (T0_<T_0>/<objective>/run/) and the metrics against the truth per
    frame; a (T_0, objective) whose row.json exists is reused. Writes
    substitute.json (the record returned).
    """
    patient_dir = out_dir / patient.id
    start = time.perf_counter()
    truth = truth_run(cohort, patient, SUBSTITUTE_HORIZON, SUBSTITUTE_FRAMES, patient_dir / "truth")
    region = observation_region(truth.run.frames, cohort.tissue, cohort.zooms)
    sa.write_json(patient_dir / "observation.json", {**region.record(), "margin_mm": OBSERVATION_MARGIN_MM})
    dose = truth.maps.dose
    rows: list[dict[str, Any]] = []
    truth_metrics = {name: compare_to(cohort, truth.run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
    rows.append(substitute_row(cohort, patient, 0.0, patient.resection_time, "truth", None, truth_metrics))
    schedule = deficit_schedule(patient, deltas, tr_max)
    skipped: list[dict[str, Any]] = []
    for entry in schedule:
        delta_a, t0 = float(entry["delta_a"]), float(entry["T_0"])
        for objective in OBJECTIVES:
            if entry["skipped"]:
                rows.append(substitute_row(cohort, patient, delta_a, t0, objective, None, {}, skipped=str(entry["skipped"])))
                skipped.append({**entry, "objective": objective})
                print(f"  {patient.id} delta_a={delta_a:g} (T_0={t0:.1f}) {objective}: skipped ({entry['skipped']})", flush=True)
                continue
            fit_dir = patient_dir / f"T0_{t0:.1f}" / objective
            row_path = fit_dir / "row.json"
            if row_path.is_file():
                rows.append(read_record(row_path))
                print(f"  {patient.id} delta_a={delta_a:g} (T_0={t0:.1f}) {objective}: row exists, kept", flush=True)
                continue
            fit = fit_seed(cohort, patient, t0, truth.density, region, objective, maxfev)
            fit_dir.mkdir(parents=True, exist_ok=True)
            write_record(fit_dir / "fit.json", {**fit.record(), "delta_a": delta_a})
            substitute = replace(patient, seed_peak=fit.peak, seed_sigma=fit.sigma, resection_time=t0)
            config = patient_config(cohort, substitute, SUBSTITUTE_HORIZON)
            run = treated_run(cohort, config, truth.maps, fit.growth_n_steps, fit.dt, SUBSTITUTE_FRAMES, fit_dir / "run")
            metrics = {name: compare_to(cohort, run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
            row = substitute_row(cohort, patient, delta_a, t0, objective, fit, metrics)
            write_record(row_path, row)
            rows.append(row)
            print(
                f"  {patient.id} delta_a={delta_a:g} (T_0={t0:.1f}) {objective}: peak {fit.initial_peak:.3f} -> {fit.peak:.3f}, sigma "
                f"{fit.initial_sigma:.2f} -> {fit.sigma:.2f} mm, objective {fit.value_initial:.4f} -> {fit.value:.4f} in {fit.n_evaluations} "
                f"evaluations ({fit.wall_time_s:.0f} s); d120 Dice {metrics['d120']['dice_edema']:.3f}",
                flush=True,
            )
    record = {
        "patient": patient.id,
        "cell": patient.cell,
        "resection_time": patient.resection_time,
        "rho": patient.rho,
        "maturity": patient.maturity,
        **truth.stepping(),
        "cavity_volume_mm3": truth.maps.record["cavity_volume_mm3"],
        "dose_volume_mm3": truth.maps.record["dose_volume_mm3"],
        "observation": region.record(),
        "delta_a": [float(v) for v in deltas],
        "schedule": schedule,
        "T_0_range": [TR_MIN, float(tr_max)],
        "skipped": skipped,
        "rows": rows,
        "wall_time_s": time.perf_counter() - start,
    }
    write_record(patient_dir / PATIENT_FILE.format(experiment="substitute"), record)
    return record


# --- fixed-seed substitution (experiment 3) ---


def seed_record(peak: float, sigma: float) -> dict[str, Any]:
    """The fixed seed's record (spec.json, seedfix_summary.json): its
    source (SEED_SOURCE: --seed-peak and --seed-sigma-mm), peak density,
    sigma in mm and the solver's mass and diffusion time
    (``seed_parameters``)."""
    derived = sa.seed_parameters(float(peak), float(sigma))
    return {
        "source": SEED_SOURCE,
        "seed_peak": float(peak),
        "seed_sigma_mm": float(sigma),
        "gaussian_seed_mass": float(derived["gaussian_seed_mass"]),
        "gaussian_seed_diffusion_time": float(derived["gaussian_seed_diffusion_time"]),
    }


def load_truth_run(cohort: Cohort, patient: Patient, horizon: float, frames: Sequence[str], run_dir: Path) -> TruthRun | None:
    """
    A truth written by ``truth_run`` read back from run_dir: the maps
    (resection_cavity.nii.gz as the boolean mask, rt_dose.nii.gz,
    treatment.json), the density at resection
    (pre_resection_cell_density.nii.gz), the frames
    (<frame>_cell_density.nii.gz) and the stepping (frames.json,
    result.json). The fields are the stored ones, float32 rounded for
    storage (``round_field``), not the unrounded fields ``truth_run``
    keeps in memory. None when the directory lacks frames.json,
    result.json or one of the frames.
    """
    frames_path, result_path = run_dir / FRAMES_FILE, run_dir / "result.json"
    if not (frames_path.is_file() and result_path.is_file()):
        return None
    frames_record = sa.read_json(frames_path)
    recorded = frames_record["frames"]
    if any(name not in recorded or not (run_dir / recorded[name]["file"]).is_file() for name in frames):
        return None
    result = sa.read_json(result_path)
    cavity_path, dose_path = run_dir / sa.CAVITY_FILE, run_dir / sa.DOSE_FILE
    cavity = np.asarray(load_image(cavity_path).get_fdata()) == sa.CAVITY_LABEL
    dose = np.asarray(load_image(dose_path).get_fdata(), dtype=np.float32)
    maps = TreatmentMaps(cavity=cavity, dose=dose, cavity_path=cavity_path, dose_path=dose_path, record=dict(sa.read_json(run_dir / sa.TREATMENT_FILE)))
    density = np.asarray(load_image(run_dir / sa.PRE_RESECTION_FILE).get_fdata(), dtype=np.float64)
    fields = {name: np.asarray(load_image(run_dir / recorded[name]["file"]).get_fdata(), dtype=np.float64) for name in frames}
    run = RunOutput(
        run_dir=run_dir,
        config=dict(sa.read_json(run_dir / "config.json")),
        frames=fields,
        days={name: float(recorded[name]["recorded_day"]) for name in frames},
        n_growth=int(frames_record["n_growth"]),
        dt=float(frames_record["dt"]),
        n_steps=int(result["n_steps"]),
        wall_time_s=float(result["wall_time_s"]),
    )
    return TruthRun(patient=patient, config=patient_config(cohort, patient, horizon), run=run, maps=maps, density=density)


@dataclass
class TimeFit:
    """The result of ``fit_growth_time`` (lambda_mode "fixed") or
    ``fit_growth_time_free`` ("free"): the fitted growth time, the front
    width (the truth's in the fixed mode) and the rho they imply."""

    objective: str
    lambda_mode: str
    seed_peak: float
    seed_sigma: float
    bounds: tuple[float, float]
    lambda_bounds: tuple[float, float] | None
    t_r: float
    front_width: float
    front_width_truth: float
    bound_hit: bool
    rho: float  # rho of the fitted front width (the truth's in the fixed mode)
    rho_truth: float
    t_r_truth: float
    start: dict[str, float] | None
    n_evaluations: int
    value_initial: float
    value: float
    growth_n_steps: int
    dt: float
    steps_per_day: int | None
    history: list[dict[str, float]]
    wall_time_s: float

    def record(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "lambda_mode": self.lambda_mode,
            "fitted_T_r": self.t_r,
            "fitted_lambda_mm": self.front_width,
            "lambda_truth_mm": self.front_width_truth,
            "rho_T_r_fitted": self.rho * self.t_r,
            "rho_T_r_truth": self.rho_truth * self.t_r_truth,
            "objective_initial": self.value_initial,
            "objective_achieved": self.value,
            "n_evaluations": self.n_evaluations,
            "bound_hit": self.bound_hit,
            "T_r_bounds": [self.bounds[0], self.bounds[1]],
            "lambda_bounds": None if self.lambda_bounds is None else [self.lambda_bounds[0], self.lambda_bounds[1]],
            "start": self.start,
            "seed_peak": self.seed_peak,
            "seed_sigma_mm": self.seed_sigma,
            "growth_n_steps": self.growth_n_steps,
            "steps_per_day": self.steps_per_day,
            "dt": self.dt,
            "wall_time_s": self.wall_time_s,
            "history": self.history,
        }


def _near_bound(value: float, bounds: tuple[float, float], tolerance: float = BOUND_TOLERANCE) -> bool:
    """Whether log(value) lies within tolerance of log of either bound."""
    lo, hi = float(np.log(bounds[0])), float(np.log(bounds[1]))
    x = float(np.log(value))
    return bool(abs(x - lo) < tolerance or abs(x - hi) < tolerance)


def fit_growth_time(
    cohort: Cohort,
    patient: Patient,
    seed_peak: float,
    seed_sigma: float,
    reference: NDArray,
    region: Region,
    objective: str,
    maxfev: int,
    bounds: tuple[float, float] = TR_BOUNDS,
    horizon: float = SUBSTITUTE_HORIZON,
) -> TimeFit:
    """
    Fit log T_r of a growth-only run from the fixed seed (peak, sigma in
    mm; the patient's v, lambda, the cohort's seed voxel and the patient's
    time step: the solve path of ``fit_seed``, one step count per T_r) to the truth's density at resection (reference) on the
    region's voxels, by the bounded Brent search of
    scipy.optimize.minimize_scalar on [log lo, log hi] of the bounds in
    days (at most maxfev evaluations, xatol SCALAR_XATOL in log T_r). The
    search starts from the bracket's golden-section point, not from the
    truth's T_r. The best evaluation is the fitted T_r; its step count
    and dt are the grid of the treated stage run from it; bound_hit says
    whether its log lies within BOUND_TOLERANCE of a bound.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not 0.0 < lo < hi:
        raise ValueError(f"the T_r bounds must be 0 < lo < hi days, got {bounds}.")
    index = tuple(region.voxels.T)
    reference_values = np.asarray(reference, dtype=np.float64)[index]
    history: list[dict[str, float]] = []

    def evaluate(x: float) -> float:
        t_r = float(np.exp(x))
        candidate = replace(patient, seed_peak=seed_peak, seed_sigma=seed_sigma, resection_time=t_r)
        result = solve_growth(cohort, growth_config(patient_config(cohort, candidate, horizon)))
        assert result.n_steps is not None and result.dt is not None
        value = fit_objective(objective, np.asarray(result.final_state["cell_density"], dtype=np.float64)[index], reference_values)
        history.append({"T_r": t_r, "value": value, "n_steps": float(result.n_steps), "dt": float(result.dt)})
        return value

    start = time.perf_counter()
    optimum = minimize_scalar(evaluate, bounds=(float(np.log(lo)), float(np.log(hi))), method="bounded", options={"maxiter": int(maxfev), "xatol": SCALAR_XATOL})
    best = min(history, key=lambda h: h["value"])
    return TimeFit(
        objective=objective,
        lambda_mode="fixed",
        seed_peak=float(seed_peak),
        seed_sigma=float(seed_sigma),
        bounds=(lo, hi),
        lambda_bounds=None,
        t_r=best["T_r"],
        front_width=patient.front_width,
        front_width_truth=patient.front_width,
        bound_hit=_near_bound(best["T_r"], (lo, hi)),
        rho=patient.rho,
        rho_truth=patient.rho,
        t_r_truth=patient.resection_time,
        start=None,
        n_evaluations=int(optimum.nfev),
        value_initial=history[0]["value"],
        value=best["value"],
        growth_n_steps=int(best["n_steps"]),
        dt=best["dt"],
        steps_per_day=patient.steps_per_day,
        history=history,
        wall_time_s=time.perf_counter() - start,
    )


def fit_growth_time_free(
    cohort: Cohort,
    patient: Patient,
    seed_peak: float,
    seed_sigma: float,
    reference: NDArray,
    region: Region,
    objective: str,
    maxfev: int,
    bounds: tuple[float, float] = TR_BOUNDS,
    lambda_bounds: tuple[float, float] = LAMBDA_BOUNDS,
    start: tuple[float, float] | None = None,
    horizon: float = SUBSTITUTE_HORIZON,
) -> TimeFit:
    """
    Fit (log T_r, log lambda) of a growth-only run from the fixed seed
    with v at the truth's, D = v lambda / 2 and rho = v / (2 lambda)
    re-derived from the current lambda at every evaluation (the
    patient's growth derivation), to the truth's density at resection on
    the region's voxels, by Nelder-Mead (the solve path and the options of
    ``fit_seed``: at most maxfev evaluations, the initial simplex x0,
    x0 + (SIMPLEX_STEPS[0], 0), x0 + (0, SIMPLEX_STEPS[1]), xatol
    NELDER_MEAD_XATOL, fatol NELDER_MEAD_FATOL) in the unbounded
    coordinates of ``bounded_from_unbounded`` over [log lo, log hi] of
    the T_r bounds (days) and of the lambda bounds (mm). The start is
    (T_r, lambda) in days and mm, default the geometric midpoints of both
    brackets. The best evaluation is the fit; bound_hit says whether its
    log T_r or its log lambda lies within BOUND_TOLERANCE of a bound.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    llo, lhi = float(lambda_bounds[0]), float(lambda_bounds[1])
    if not 0.0 < lo < hi:
        raise ValueError(f"the T_r bounds must be 0 < lo < hi days, got {bounds}.")
    if not 0.0 < llo < lhi:
        raise ValueError(f"the lambda bounds must be 0 < lo < hi mm, got {lambda_bounds}.")
    log_t = (float(np.log(lo)), float(np.log(hi)))
    log_l = (float(np.log(llo)), float(np.log(lhi)))
    if start is None:
        start = (float(np.sqrt(lo * hi)), float(np.sqrt(llo * lhi)))
    x0 = np.array([unbounded_from_bounded(float(np.log(start[0])), *log_t), unbounded_from_bounded(float(np.log(start[1])), *log_l)])
    index = tuple(region.voxels.T)
    reference_values = np.asarray(reference, dtype=np.float64)[index]
    history: list[dict[str, float]] = []

    def evaluate(x: NDArray) -> float:
        t_r = float(np.exp(bounded_from_unbounded(float(x[0]), *log_t)))
        width = float(np.exp(bounded_from_unbounded(float(x[1]), *log_l)))
        candidate = replace(patient, front_width=width, seed_peak=seed_peak, seed_sigma=seed_sigma, resection_time=t_r)
        result = solve_growth(cohort, growth_config(patient_config(cohort, candidate, horizon)))
        assert result.n_steps is not None and result.dt is not None
        value = fit_objective(objective, np.asarray(result.final_state["cell_density"], dtype=np.float64)[index], reference_values)
        history.append({"T_r": t_r, "lambda_mm": width, "value": value, "n_steps": float(result.n_steps), "dt": float(result.dt)})
        return value

    started = time.perf_counter()
    simplex = np.array([x0, x0 + np.array([SIMPLEX_STEPS[0], 0.0]), x0 + np.array([0.0, SIMPLEX_STEPS[1]])])
    optimum = minimize(
        evaluate,
        x0,
        method="Nelder-Mead",
        options={"maxfev": int(maxfev), "initial_simplex": simplex, "xatol": NELDER_MEAD_XATOL, "fatol": NELDER_MEAD_FATOL},
    )
    best = min(history, key=lambda h: h["value"])
    width = float(best["lambda_mm"])
    return TimeFit(
        objective=objective,
        lambda_mode="free",
        seed_peak=float(seed_peak),
        seed_sigma=float(seed_sigma),
        bounds=(lo, hi),
        lambda_bounds=(llo, lhi),
        t_r=best["T_r"],
        front_width=width,
        front_width_truth=patient.front_width,
        bound_hit=_near_bound(best["T_r"], (lo, hi)) or _near_bound(width, (llo, lhi)),
        rho=float(sa.growth_parameters(patient.front_speed, width)["rho"]),
        rho_truth=patient.rho,
        t_r_truth=patient.resection_time,
        start={"T_r": float(start[0]), "lambda_mm": float(start[1])},
        n_evaluations=int(optimum.nfev),
        value_initial=history[0]["value"],
        value=best["value"],
        growth_n_steps=int(best["n_steps"]),
        dt=best["dt"],
        steps_per_day=patient.steps_per_day,
        history=history,
        wall_time_s=time.perf_counter() - started,
    )


def seedfix_row(cohort: Cohort, patient: Patient, lambda_mode: str, fit: TimeFit | None, metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """A seedfix.csv record: the patient, the lambda mode ("truth" for
    the truth row), the fitted T_r and lambda (the truth's for the truth
    row), rho T_r fitted (with the fitted rho) and true, the design's
    R_over_lambda and maturity, the seed, the fit (with bound_hit; False for the
    truth row) and the metrics per frame as <frame>_<metric>."""
    t_r = fit.t_r if fit is not None else patient.resection_time
    width = fit.front_width if fit is not None else patient.front_width
    rho = fit.rho if fit is not None else patient.rho
    row: dict[str, Any] = {
        "patient": patient.id,
        "cell": patient.cell,
        "lambda_mode": lambda_mode,
        "objective": fit.objective if fit is not None else "truth",
        "fitted_T_r": float(t_r),
        "fitted_lambda_mm": float(width),
        "lambda_truth_mm": patient.front_width,
        "rho_T_r_fitted": float(rho) * float(t_r),
        "rho_T_r": patient.rho * patient.resection_time,
        "R_over_lambda": patient.r_over_lambda,
        "maturity": patient.maturity,
    }
    if fit is not None:
        skipped = ("history", "objective", "lambda_mode", "fitted_T_r", "fitted_lambda_mm", "lambda_truth_mm", "rho_T_r_fitted", "rho_T_r_truth", "T_r_bounds", "lambda_bounds", "start")
        row.update({key: value for key, value in fit.record().items() if key not in skipped})
    else:
        row.update({"seed_peak": patient.seed_peak, "seed_sigma_mm": patient.seed_sigma, "bound_hit": False, "steps_per_day": patient.steps_per_day})
    for frame, values in metrics.items():
        row.update({f"{frame}_{name}": value for name, value in values.items()})
    return row


SEEDFIX_COLUMNS: list[str] = [
    "patient",
    "cell",
    "lambda_mode",
    "objective",
    "fitted_T_r",
    "fitted_lambda_mm",
    "lambda_truth_mm",
    "rho_T_r_fitted",
    "rho_T_r",
    "R_over_lambda",
    "maturity",
    "seed_peak",
    "seed_sigma_mm",
    "n_evaluations",
    "objective_initial",
    "objective_achieved",
    "bound_hit",
    "growth_n_steps",
    "steps_per_day",
    "dt",
    "wall_time_s",
    *(f"{frame}_{name}" for frame in SUBSTITUTE_FRAMES for name in METRIC_NAMES),
]


def check_reused_seed(path: Path, record: Mapping[str, Any], seed_peak: float, seed_sigma: float) -> None:
    """Raise when a reused record (a seedfix or profile row.json or
    fit.json) was written for another fixed seed than the current
    arguments (seed_peak, seed_sigma_mm), naming both."""
    stored = {"seed_peak": sa.as_float(record.get("seed_peak")), "seed_sigma_mm": sa.as_float(record.get("seed_sigma_mm"))}
    current = {"seed_peak": float(seed_peak), "seed_sigma_mm": float(seed_sigma)}
    if any(not np.isclose(stored[key], current[key], rtol=1e-9, atol=0.0) for key in current):
        raise ValueError(
            f"{path} was written for the seed peak {stored['seed_peak']!r}, sigma {stored['seed_sigma_mm']!r} mm; the current arguments are "
            f"peak {current['seed_peak']!r}, sigma {current['seed_sigma_mm']!r} mm. Run into another design, or remove the record."
        )


def load_or_solve_truth(cohort: Cohort, patient: Patient, patient_dir: Path, sources: Sequence[Path]) -> tuple[TruthRun, bool]:
    """
    The patient's truth over SUBSTITUTE_HORIZON with the frames
    SUBSTITUTE_FRAMES: read back (``load_truth_run``) from the first of
    the source directories holding the six frames (their stored fields
    are the float32 ones rounded for storage), else solved into
    patient_dir/truth. Returns (truth, reused).
    """
    for source in sources:
        truth = load_truth_run(cohort, patient, SUBSTITUTE_HORIZON, SUBSTITUTE_FRAMES, source)
        if truth is not None:
            print(f"  {patient.id}: truth read from {source}", flush=True)
            return truth, True
    return truth_run(cohort, patient, SUBSTITUTE_HORIZON, SUBSTITUTE_FRAMES, patient_dir / "truth"), False


def parse_lambda_modes(text: str | None) -> tuple[str, ...]:
    """--lambda-mode as the modes run, in LAMBDA_MODES' order: fixed (the
    default, DEFAULT_LAMBDA_MODE) | free | both."""
    mode = DEFAULT_LAMBDA_MODE if text is None or str(text).strip() == "" else str(text).strip()
    if mode == "both":
        return tuple(LAMBDA_MODES)
    if mode not in LAMBDA_MODES:
        raise ValueError(f"--lambda-mode must be one of {(*LAMBDA_MODES, 'both')}, got {text!r}.")
    return (mode,)


def seedfix_patient(
    cohort: Cohort,
    patient: Patient,
    out_dir: Path,
    maxfev: int,
    tr_bounds: tuple[float, float] = TR_BOUNDS,
    seed_peak: float = DEFAULT_SEED_PEAK,
    seed_sigma: float = DEFAULT_SEED_SIGMA_MM,
    lambda_modes: Sequence[str] = LAMBDA_MODES,
    lambda_bounds: tuple[float, float] = LAMBDA_BOUNDS,
) -> dict[str, Any]:
    """
    Experiment 3 for one patient: the truth (experiment 2's, runs/
    substitute/<patient>/truth, read back when complete; else solved into
    <patient>/truth; ``load_or_solve_truth``), then per lambda mode (fixed
    before free) the fitted growth time of the fixed seed (seed_peak,
    seed_sigma; ``fit_growth_time`` with v and lambda at the truth, or
    ``fit_growth_time_free`` with lambda fitted too, started from the
    fixed mode's optimum (log T_r*, log lambda_truth) when its fit.json
    exists in this patient's directory and did not hit a bound, else the
    brackets' midpoints), the
    treated run from that seed with the fitted lambda's D and rho,
    resection_time the fitted T_r, the truth's maps, alpha and k_ct and
    the schedule shifted for it (<LAMBDA_MODE_DIRS[mode]>/run/) and the
    metrics against the truth per frame; a mode whose row.json exists is
    reused (its seed must be the current one, ``check_reused_seed``).
    Writes seedfix.json (the record returned).
    """
    patient_dir = out_dir / patient.id
    start = time.perf_counter()
    truth, reused = load_or_solve_truth(cohort, patient, patient_dir, [out_dir.parent / "substitute" / patient.id / "truth"])
    region = observation_region(truth.run.frames, cohort.tissue, cohort.zooms)
    patient_dir.mkdir(parents=True, exist_ok=True)
    sa.write_json(patient_dir / "observation.json", {**region.record(), "margin_mm": OBSERVATION_MARGIN_MM})
    dose = truth.maps.dose
    objective = OBJECTIVES[0]
    rows: list[dict[str, Any]] = []
    truth_metrics = {name: compare_to(cohort, truth.run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
    rows.append(seedfix_row(cohort, patient, "truth", None, truth_metrics))
    fixed_fit_path = patient_dir / LAMBDA_MODE_DIRS["fixed"] / "fit.json"
    for mode in (m for m in LAMBDA_MODES if m in lambda_modes):
        fit_dir = patient_dir / LAMBDA_MODE_DIRS[mode]
        row_path = fit_dir / "row.json"
        if row_path.is_file():
            stored = read_record(row_path)
            check_reused_seed(row_path, stored, seed_peak, seed_sigma)
            rows.append(stored)
            print(f"  {patient.id} {mode}: row exists, kept", flush=True)
            continue
        if mode == "fixed":
            fit = fit_growth_time(cohort, patient, seed_peak, seed_sigma, truth.density, region, objective, maxfev, tr_bounds)
        else:
            free_start = None
            if fixed_fit_path.is_file():
                fixed_fit = read_record(fixed_fit_path)
                check_reused_seed(fixed_fit_path, fixed_fit, seed_peak, seed_sigma)
                if str(fixed_fit.get("bound_hit")) != "True":
                    free_start = (float(fixed_fit["fitted_T_r"]), patient.front_width)
            fit = fit_growth_time_free(cohort, patient, seed_peak, seed_sigma, truth.density, region, objective, maxfev, tr_bounds, lambda_bounds, free_start)
        fit_dir.mkdir(parents=True, exist_ok=True)
        write_record(fit_dir / "fit.json", fit.record())
        substitute = replace(patient, front_width=fit.front_width, seed_peak=seed_peak, seed_sigma=seed_sigma, resection_time=fit.t_r)
        config = patient_config(cohort, substitute, SUBSTITUTE_HORIZON)
        run = treated_run(cohort, config, truth.maps, fit.growth_n_steps, fit.dt, SUBSTITUTE_FRAMES, fit_dir / "run")
        metrics = {name: compare_to(cohort, run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
        row = seedfix_row(cohort, patient, mode, fit, metrics)
        write_record(row_path, row)
        rows.append(row)
        print(
            f"  {patient.id} {mode}: T_r {fit.t_r:.1f} (truth {patient.resection_time:.1f}), lambda {fit.front_width:.2f} (truth "
            f"{patient.front_width:.2f}) mm, rho T_r {fit.rho * fit.t_r:.2f} (truth {patient.rho * patient.resection_time:.2f}), "
            f"objective {fit.value_initial:.4f} -> {fit.value:.4f} in {fit.n_evaluations} evaluations ({fit.wall_time_s:.0f} s)"
            f"{', bound hit' if fit.bound_hit else ''}; d120 Dice {metrics['d120']['dice_edema']:.3f}",
            flush=True,
        )
    record = {
        "patient": patient.id,
        "cell": patient.cell,
        "resection_time": patient.resection_time,
        "rho": patient.rho,
        "front_width": patient.front_width,
        "R_over_lambda": patient.r_over_lambda,
        "maturity": patient.maturity,
        **truth.stepping(),
        "cavity_volume_mm3": truth.maps.record["cavity_volume_mm3"],
        "dose_volume_mm3": truth.maps.record["dose_volume_mm3"],
        "truth_dir": str(truth.run.run_dir),
        "truth_reused": reused,
        "seed": seed_record(seed_peak, seed_sigma),
        "lambda_modes": list(lambda_modes),
        "T_r_bounds": [float(tr_bounds[0]), float(tr_bounds[1])],
        "lambda_bounds": [float(lambda_bounds[0]), float(lambda_bounds[1])],
        "observation": region.record(),
        "rows": rows,
        "wall_time_s": time.perf_counter() - start,
    }
    write_record(patient_dir / PATIENT_FILE.format(experiment="seedfix"), record)
    return record


# --- seed-width profile (experiment 4) ---


PROFILE_COLUMNS: list[str] = [
    "patient",
    "cell",
    "sigma_mm",
    "seed_peak",
    "fitted_T_r",
    "rho_T_r_fitted",
    "rho_T_r",
    "R_over_lambda",
    "maturity",
    "bound_hit",
    "objective_initial",
    "objective_achieved",
    "n_evaluations",
    "growth_n_steps",
    "steps_per_day",
    "dt",
    "wall_time_s",
    *(f"{frame}_{name}" for frame in SUBSTITUTE_FRAMES for name in METRIC_NAMES),
]


def profile_row(patient: Patient, sigma: float, fit: TimeFit, metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """A profile.csv record: the patient, the seed width sigma_0 and peak,
    the design's R_over_lambda and maturity, the fit (fitted_T_r, rho_T_r_fitted, bound_hit, the objective values,
    the evaluations, the stepping) and the metrics per frame as
    <frame>_<metric>."""
    row: dict[str, Any] = {
        "patient": patient.id,
        "cell": patient.cell,
        "sigma_mm": float(sigma),
        "seed_peak": fit.seed_peak,
        "fitted_T_r": fit.t_r,
        "rho_T_r_fitted": fit.rho * fit.t_r,
        "rho_T_r": patient.rho * patient.resection_time,
        "R_over_lambda": patient.r_over_lambda,
        "maturity": patient.maturity,
        "bound_hit": fit.bound_hit,
        "objective_initial": fit.value_initial,
        "objective_achieved": fit.value,
        "n_evaluations": fit.n_evaluations,
        "growth_n_steps": fit.growth_n_steps,
        "steps_per_day": fit.steps_per_day,
        "dt": fit.dt,
        "wall_time_s": fit.wall_time_s,
    }
    for frame, values in metrics.items():
        row.update({f"{frame}_{name}": value for name, value in values.items()})
    return row


def profile_patient(
    cohort: Cohort,
    patient: Patient,
    out_dir: Path,
    maxfev: int,
    tr_bounds: tuple[float, float] = TR_BOUNDS,
    seed_peak: float = DEFAULT_SEED_PEAK,
    sigmas: Sequence[float] = PROFILE_SIGMAS,
) -> dict[str, Any]:
    """
    Experiment 4 for one patient: the truth (runs/substitute/<patient>/
    truth, else runs/seedfix/<patient>/truth, read back when complete;
    else solved into <patient>/truth; ``load_or_solve_truth``), then per
    seed width sigma_0 the fixed-lambda fit of the growth time of the seed
    (seed_peak, sigma_0) (``fit_growth_time``, objective OBJECTIVES[0]),
    the treated run from it at the fitted T_r with the truth's maps,
    alpha and k_ct (sigma_<sigma_0>/run/) and the metrics against the
    truth per frame, the path of ``seedfix_patient``'s fixed mode; a
    sigma_0 whose row.json exists is reused (its seed must be the current
    peak and that sigma_0, ``check_reused_seed``). Writes profile.json
    (the record returned).
    """
    patient_dir = out_dir / patient.id
    start = time.perf_counter()
    sources = [out_dir.parent / experiment / patient.id / "truth" for experiment in ("substitute", "seedfix")]
    truth, reused = load_or_solve_truth(cohort, patient, patient_dir, sources)
    region = observation_region(truth.run.frames, cohort.tissue, cohort.zooms)
    patient_dir.mkdir(parents=True, exist_ok=True)
    sa.write_json(patient_dir / "observation.json", {**region.record(), "margin_mm": OBSERVATION_MARGIN_MM})
    dose = truth.maps.dose
    objective = OBJECTIVES[0]
    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        fit_dir = patient_dir / f"sigma_{float(sigma):g}"
        row_path = fit_dir / "row.json"
        if row_path.is_file():
            stored = read_record(row_path)
            check_reused_seed(row_path, stored, seed_peak, float(sigma))
            rows.append(stored)
            print(f"  {patient.id} sigma_0={sigma:g}: row exists, kept", flush=True)
            continue
        fit = fit_growth_time(cohort, patient, seed_peak, float(sigma), truth.density, region, objective, maxfev, tr_bounds)
        fit_dir.mkdir(parents=True, exist_ok=True)
        write_record(fit_dir / "fit.json", fit.record())
        substitute = replace(patient, seed_peak=seed_peak, seed_sigma=float(sigma), resection_time=fit.t_r)
        config = patient_config(cohort, substitute, SUBSTITUTE_HORIZON)
        run = treated_run(cohort, config, truth.maps, fit.growth_n_steps, fit.dt, SUBSTITUTE_FRAMES, fit_dir / "run")
        metrics = {name: compare_to(cohort, run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
        row = profile_row(patient, float(sigma), fit, metrics)
        write_record(row_path, row)
        rows.append(row)
        print(
            f"  {patient.id} sigma_0={sigma:g} mm: T_r {fit.t_r:.1f} (truth {patient.resection_time:.1f}), rho T_r {fit.rho * fit.t_r:.2f} "
            f"(truth {patient.rho * patient.resection_time:.2f}), objective {fit.value_initial:.4f} -> {fit.value:.4f} in {fit.n_evaluations} "
            f"evaluations ({fit.wall_time_s:.0f} s){', bound hit' if fit.bound_hit else ''}; d180 mass beyond edema rel "
            f"{metrics['d180']['mass_beyond_edema_rel']:.3f}",
            flush=True,
        )
    record = {
        "patient": patient.id,
        "cell": patient.cell,
        "resection_time": patient.resection_time,
        "rho": patient.rho,
        "R_over_lambda": patient.r_over_lambda,
        "maturity": patient.maturity,
        **truth.stepping(),
        "truth_dir": str(truth.run.run_dir),
        "truth_reused": reused,
        "seed_peak": float(seed_peak),
        "sigmas": [float(s) for s in sigmas],
        "objective": objective,
        "T_r_bounds": [float(tr_bounds[0]), float(tr_bounds[1])],
        "observation": region.record(),
        "rows": rows,
        "wall_time_s": time.perf_counter() - start,
    }
    write_record(patient_dir / PATIENT_FILE.format(experiment="profile"), record)
    return record


# --- dt-check ---


DT_CHECK_METRICS: tuple[str, ...] = ("dice_core", "dice_edema", "mass_rel", "mass_beyond_edema_rel")
DT_CHECK_FRAMES: tuple[str, ...] = ("pre", "d180")
DT_CHECK_COLUMNS: list[str] = [
    "patient",
    "cell",
    "maturity",
    "resection_time",
    "white_matter_diffusivity",
    "grid_spacing_mm",
    *(f"{key}_{mode}" for mode in DT_MODES for key in ("steps_per_day", "dt", "dt_refined", "n_growth", "n_steps", "wall_time_s")),
    *(f"{frame}_{name}" for frame in DT_CHECK_FRAMES for name in DT_CHECK_METRICS),
]


def dt_check_patient(cohort: Cohort, patient: Patient, out_dir: Path) -> dict[str, Any]:
    """
    The time-step check of one patient: the truth (``truth_run``: the
    growth stage, its own cavity and dose map and the treated stage over
    SUBSTITUTE_HORIZON with the frames SUBSTITUTE_FRAMES) once at the
    fixed step and once at the stability step (``steps_per_day_for`` per
    mode, on the design's grid spacing) into out_dir/<patient>/<mode>/,
    and the metrics (``compare_to``) of the stability run's pre and d180
    frames against the fixed run's, the fixed run's dose map defining the
    field. Writes dt_check.json (the record returned, the resume marker):
    per mode steps_per_day, dt, dt_refined, n_growth, n_steps and
    wall_time_s, and the row of dt_check.csv.
    """
    patient_dir = out_dir / patient.id
    start = time.perf_counter()
    dx = grid_spacing_mm(cohort.base, cohort.zooms)
    truths: dict[str, TruthRun] = {}
    for mode in DT_MODES:
        steps = steps_per_day_for(mode, patient.diffusivity, patient.rho, patient.resection_time, dx)
        stepped = replace(patient, steps_per_day=steps)
        truths[mode] = truth_run(cohort, stepped, SUBSTITUTE_HORIZON, SUBSTITUTE_FRAMES, patient_dir / mode)
        print(f"  {patient.id} {mode}: {steps} steps/day, dt {truths[mode].dt:g}, {truths[mode].n_growth} growth steps, {truths[mode].run.n_steps} steps in total ({truths[mode].run.wall_time_s:.0f} s)", flush=True)
    fine, coarse = truths["fixed"], truths["stability"]
    metrics = {frame: compare_to(cohort, coarse.run.frames[frame], fine.run.frames[frame], fine.maps.dose) for frame in SUBSTITUTE_FRAMES}
    row: dict[str, Any] = {
        "patient": patient.id,
        "cell": patient.cell,
        "maturity": patient.maturity,
        "resection_time": patient.resection_time,
        "white_matter_diffusivity": patient.diffusivity,
        "grid_spacing_mm": dx,
    }
    for mode, truth in truths.items():
        stepping = truth.stepping()
        row.update({f"{key}_{mode}": stepping[key] for key in ("steps_per_day", "dt", "dt_refined", "n_growth")})
        row[f"n_steps_{mode}"] = truth.run.n_steps
        row[f"wall_time_s_{mode}"] = truth.run.wall_time_s
    for frame in DT_CHECK_FRAMES:
        row.update({f"{frame}_{name}": metrics[frame][name] for name in DT_CHECK_METRICS})
    record = {
        "patient": patient.id,
        "cell": patient.cell,
        "resection_time": patient.resection_time,
        "grid_spacing_mm": dx,
        "modes": {mode: {**truth.stepping(), "n_steps": truth.run.n_steps, "wall_time_s": truth.run.wall_time_s, "run_dir": str(truth.run.run_dir)} for mode, truth in truths.items()},
        "metrics": metrics,
        "row": row,
        "wall_time_s": time.perf_counter() - start,
    }
    write_record(patient_dir / PATIENT_FILE.format(experiment="dt_check"), record)
    return record


def assemble_dt_check(root: Path) -> list[dict[str, Any]]:
    """dt_check.csv from the records present (runs/dt_check/<patient>/dt_check.json)."""
    rows = [record["row"] for record in patient_records(root, "dt_check")]
    sa.write_csv(root / "dt_check.csv", rows, DT_CHECK_COLUMNS)
    return rows


# --- invariance (experiment 0) ---


INVARIANCE_COLUMNS: list[str] = ["lambda", "run_type", "snapshot", "n_steps", "dt", "resection_time", "wall_time_s", *METRIC_NAMES]


def run_invariance(
    cohort: Cohort,
    out_dir: Path,
    lambdas: Sequence[float] = INVARIANCE_LAMBDAS,
    diffusivity: float | None = None,
    rho: float | None = None,
    resection_time: float | None = None,
) -> list[dict[str, Any]]:
    """
    Experiment 0: one patient (the base config's D, rho, resection_time,
    seed, alpha and k_ct unless overridden) run for every lambda as
    (lambda D, lambda rho, T_r / lambda) three ways (``INVARIANCE_RUN_TYPES``)
    and compared to lambda = 1 with ``compare_fields``: the growth-only
    runs on their final field ("pre"), the treated run on every frame.
    Writes out_dir/lambda_<lambda>/<run_type>/ and out_dir/invariance.json
    (the rows returned; the resume marker).
    """
    base = dict(cohort.base)
    if diffusivity is not None:
        base["white_matter_diffusivity"] = float(diffusivity)
    if rho is not None:
        base["rho"] = float(rho)
    if resection_time is not None:
        base["resection_time"] = float(resection_time)
    patient = Patient.from_config(base, id="invariance", cell="base")
    ordered = [1.0, *(float(v) for v in lambdas if float(v) != 1.0)]
    if 1.0 not in [float(v) for v in lambdas]:
        raise ValueError("the lambda set must contain 1 (the reference).")
    reference: dict[tuple[str, str], NDArray] = {}
    n_reference: int | None = None
    maps: TreatmentMaps | None = None
    rows: list[dict[str, Any]] = []
    for value in ordered:
        scaled = replace(patient, front_speed=patient.front_speed * value, resection_time=patient.resection_time / value)
        config = patient_config(cohort, scaled, FISHER_HORIZON)
        lambda_dir = out_dir / f"lambda_{value:g}"
        fields: dict[tuple[str, str], NDArray] = {}
        stepping: dict[str, tuple[int, float, float]] = {}
        growth_fixed_dt = solve_growth(cohort, growth_config(config))
        assert growth_fixed_dt.n_steps is not None and growth_fixed_dt.dt is not None
        n_dt, dt_dt = growth_fixed_dt.n_steps, growth_fixed_dt.dt
        if n_reference is None:
            n_reference = n_dt
        fields[("growth_fixed_dt", "pre")] = np.asarray(growth_fixed_dt.final_state["cell_density"], dtype=np.float64)
        stepping["growth_fixed_dt"] = (n_dt, dt_dt, float(growth_fixed_dt.wall_time_s or 0.0))
        growth_fixed_n = solve_growth(cohort, growth_config(config, n_steps=n_reference), scaled.resection_time / n_reference)
        assert growth_fixed_n.n_steps is not None and growth_fixed_n.dt is not None
        fields[("growth_fixed_n", "pre")] = np.asarray(growth_fixed_n.final_state["cell_density"], dtype=np.float64)
        stepping["growth_fixed_n"] = (growth_fixed_n.n_steps, growth_fixed_n.dt, float(growth_fixed_n.wall_time_s or 0.0))
        for name, result in (("growth_fixed_dt", growth_fixed_dt), ("growth_fixed_n", growth_fixed_n)):
            run_dir = lambda_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            save_field(run_dir / FRAME_FILE.format(name="pre"), fields[(name, "pre")], cohort.affine)
            result.initial_state = {}
            result.final_state = {}
            result.time_series = None
            result.save(run_dir, overwrite=True)
        if maps is None:
            maps = derive_maps(cohort, fields[("growth_fixed_dt", "pre")], len(config["rt_times"]), out_dir / "maps")
        treated = treated_run(cohort, config, maps, n_dt, dt_dt, FISHER_FRAMES, lambda_dir / "treated")
        for name in FISHER_FRAMES:
            fields[("treated", name)] = treated.frames[name]
        stepping["treated"] = (treated.n_steps, treated.dt, treated.wall_time_s)
        if value == 1.0:
            reference = dict(fields)
        for (run_type, snapshot), field in fields.items():
            n_steps, dt, wall = stepping[run_type]
            metrics = compare_to(cohort, field, reference[(run_type, snapshot)], maps.dose)
            rows.append({"lambda": value, "run_type": run_type, "snapshot": snapshot, "n_steps": n_steps, "dt": dt, "resection_time": scaled.resection_time, "wall_time_s": wall, **metrics})
        print(
            f"  lambda {value:g}: growth {n_dt} steps at dt {dt_dt:.5f} / {n_reference} steps at dt {growth_fixed_n.dt:.5f}, treated {treated.n_steps} steps; "
            f"max abs diff A {rows[-len(fields) + 1]['max_abs_diff']:.3g}, B {rows[-len(fields)]['max_abs_diff']:.3g}, "
            f"C d120 Dice {rows[-1]['dice_edema']:.3f}",
            flush=True,
        )
    record = {"patient": patient.__dict__, "lambdas": ordered, "run_types": INVARIANCE_RUN_TYPES, "rows": rows}
    write_record(out_dir / "invariance.json", record)
    return rows


# --- assembly and figures ---


FISHER_SET_KEYS: list[str] = [
    "n_rows",
    "We_norm",
    "We1_norm",
    "We_ratio",
    *(f"eig_{i + 1}" for i in range(len(THETA_NAMES))),
    *(f"weak_{name}" for name in THETA_NAMES),
    "cond_F",
    *(f"cr_{name}" for name in THETA_NAMES),
    "resid_frac_T_r",
    "resid7_frac_T_r",
    "norm7_T_r",
    "cr7_log_T_r",
    *(f"cr7_{name}" for name in SEED_THETA_NAMES),
]
FISHER_COLUMNS: list[str] = [
    "patient",
    "cell",
    "t_r_column",
    "observation_set",
    "frames",
    "resection_time",
    "n_growth",
    "dt",
    "n_region_voxels",
    "n_observations",
    *FISHER_SET_KEYS,
]
FISHER_REGION_COLUMNS: list[str] = ["patient", "cell", "t_r_column", "observation_set", "split", "name", "resid7_frac", "norm_T_r", "n_rows"]
FD_CHECK_COLUMNS: list[str] = [
    "patient",
    "cell",
    "column",
    "step",
    "step_half",
    "rel_diff",
    "We_ratio_a",
    "We_ratio_a_half",
    "We_ratio_f",
    "We_ratio_f_half",
    "cr_log_T_r_f",
    "cr_log_T_r_f_half",
]
FISHER_RUN_COLUMNS: list[str] = ["patient", "cell", "run", "column", "sign", "variant", "step", "frame", "n_steps", "dt", *METRIC_NAMES]
HEATMAP_KEYS: dict[str, tuple[str, bool]] = {  # fisher.csv column -> (title, log10)
    "cr_log_T_r": ("Cramer-Rao standard error of log T_r (theta columns)", True),
    "cr7_log_T_r": ("Cramer-Rao standard error of log T_r (with the seed columns)", True),
    "resid_frac_T_r": ("residual fraction of the T_r column (other four theta columns)", False),
    "resid7_frac_T_r": ("residual fraction of the T_r column (other six columns)", False),
    "We_ratio": ("|W e| / |W e_1|, e the growth invariance direction", True),
}


def patient_records(root: Path, experiment: str) -> list[dict[str, Any]]:
    """The per-patient records of an experiment (runs/<experiment>/<patient>/<experiment>.json), in design order."""
    out_dir = root / "runs" / experiment
    records = []
    for patient_dir in sorted(out_dir.glob("p*")):
        path = patient_dir / PATIENT_FILE.format(experiment=experiment)
        if path.is_file():
            records.append(read_record(path))
    return records


def _save_figure(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=150)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def write_summary(root: Path, experiment: str, records: Sequence[Mapping[str, Any]], n_rows: int, dispatch: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    <root>/<experiment>_summary.json: the patients and cells with a
    record, the CSV row count, the assembly time and the device split of
    the last dispatch (the given one, else the previous summary's).
    """
    path = root / f"{experiment}_summary.json"
    previous = read_record(path) if path.is_file() else {}
    cells: dict[str, int] = {}
    for record in records:
        cells[str(record["cell"])] = cells.get(str(record["cell"]), 0) + 1
    summary = {
        "experiment": experiment,
        "assembled": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_patients": len(records),
        "patients": [str(record["patient"]) for record in records],
        "cells": cells,
        "n_rows": int(n_rows),
        "dispatch": dict(dispatch) if dispatch is not None else previous.get("dispatch"),
    }
    write_record(path, summary)
    return summary


def assemble_fisher(root: Path, dispatch: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """fisher.csv, fisher_regions.csv, fisher_fd_check.csv, fisher_runs.csv,
    the heatmaps and fisher_summary.json from the patient records
    present (``write_summary``; dispatch: the device split to record)."""
    rows: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    records = patient_records(root, "fisher")
    for record in records:
        head = {"patient": record["patient"], "cell": record["cell"]}
        for variant, sets in record["analysis"].items():
            for set_name, analysis in sets.items():
                rows.append({
                    **head,
                    "t_r_column": variant,
                    "observation_set": set_name,
                    "frames": "+".join(analysis["frames"]),
                    "resection_time": record["resection_time"],
                    "n_growth": record["n_growth"],
                    "dt": record["dt"],
                    "n_region_voxels": record["observation"]["n_voxels"],
                    "n_observations": record["observation"]["n_observations"],
                    **{key: analysis.get(key) for key in FISHER_SET_KEYS},
                })
                for split in analysis["splits"]:
                    regions.append({**head, "t_r_column": variant, "observation_set": set_name, **split})
        checks.extend(record["fd_check"])
        runs.extend(record["run_metrics"])
    sa.write_csv(root / "fisher.csv", rows, FISHER_COLUMNS)
    sa.write_csv(root / "fisher_regions.csv", regions, FISHER_REGION_COLUMNS)
    sa.write_csv(root / "fisher_fd_check.csv", checks, FD_CHECK_COLUMNS)
    sa.write_csv(root / "fisher_runs.csv", runs, FISHER_RUN_COLUMNS)
    if rows:
        fisher_heatmaps(rows, root / "figures")
    write_summary(root, "fisher", records, len(rows), dispatch)
    return rows


def fisher_heatmaps(rows: Sequence[Mapping[str, Any]], figure_dir: Path, t_r_column: str = "scaled_dt") -> None:
    """Heatmaps (patients x observation sets) of the Cramer-Rao errors,
    the residual fractions and |W e| / |W e_1| for one T_r variant."""
    figure_dir.mkdir(exist_ok=True)
    selected = [row for row in rows if row["t_r_column"] == t_r_column]
    patients = list(dict.fromkeys(str(row["patient"]) for row in selected))
    sets = list(dict.fromkeys(str(row["observation_set"]) for row in selected))
    cells = {str(row["patient"]): str(row["cell"]) for row in selected}
    for key, (title, log_scale) in HEATMAP_KEYS.items():
        matrix = np.full((len(patients), len(sets)), np.nan)
        for row in selected:
            value = row.get(key)
            matrix[patients.index(str(row["patient"])), sets.index(str(row["observation_set"]))] = float("nan") if value is None else float(value)
        shown = np.log10(np.maximum(matrix, 1e-300)) if log_scale else matrix
        finite = np.isfinite(shown)
        figure, axis = plt.subplots(figsize=(1.6 + 0.8 * len(sets), 1.2 + 0.28 * len(patients)))
        image = axis.imshow(np.where(finite, shown, np.nan), cmap="Blues", aspect="auto")
        low, high = (float(np.nanmin(shown[finite])), float(np.nanmax(shown[finite]))) if finite.any() else (0.0, 1.0)
        axis.set_xticks(range(len(sets)), [f"({s})" for s in sets], fontsize=8)
        axis.set_yticks(range(len(patients)), [f"{p} {cells[p]}" for p in patients], fontsize=7)
        for i in range(len(patients)):
            for j in range(len(sets)):
                value = matrix[i, j]
                text = "inf" if np.isinf(value) else ("" if np.isnan(value) else (f"{value:.2g}" if not log_scale else f"{value:.1e}"))
                dark = finite[i, j] and high > low and (shown[i, j] - low) / (high - low) > 0.55
                axis.text(j, i, text, ha="center", va="center", fontsize=6, color="white" if dark else COLOR_TEXT)
        axis.set_title(f"{title}{' (log10 colour)' if log_scale else ''}, T_r column {t_r_column}", fontsize=9)
        figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
        figure.tight_layout()
        _save_figure(figure, figure_dir / f"fisher_heatmap_{key}")


def assemble_invariance(root: Path) -> list[dict[str, Any]]:
    """invariance.csv and its figures from runs/invariance/invariance.json."""
    path = root / "runs" / "invariance" / "invariance.json"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = list(read_record(path)["rows"])
    sa.write_csv(root / "invariance.csv", rows, INVARIANCE_COLUMNS)
    invariance_figures(rows, root / "figures")
    return rows


def invariance_figures(rows: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    """Every metric against lambda (log axis), one line per (run type,
    snapshot): the comparison metrics in one figure, the QoIs in another."""
    figure_dir.mkdir(exist_ok=True)
    series = list(dict.fromkeys((str(row["run_type"]), str(row["snapshot"])) for row in rows))
    for stem, names in (("invariance_metrics", PLOTTED_METRICS), ("invariance_qois", PLOTTED_QOIS)):
        n_cols = 4
        n_rows = int(np.ceil(len(names) / n_cols))
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.6 * n_rows), squeeze=False)
        values = sorted({float(r["lambda"]) for r in rows})
        for axis, name in zip(axes.flat, names):
            for index, (run_type, snapshot) in enumerate(series):
                points = sorted((float(r["lambda"]), sa.as_float(r.get(name))) for r in rows if r["run_type"] == run_type and r["snapshot"] == snapshot)
                x = np.array([p[0] for p in points])
                y = np.array([p[1] for p in points])
                style = "-" if run_type == "treated" else ("--" if run_type == "growth_fixed_dt" else ":")
                axis.plot(x, y, style, marker="o", markersize=3, color=COLOR_LINES[index % len(COLOR_LINES)], label=f"{run_type} {snapshot}")
            # The limits are set by hand: a panel whose values are all NaN
            # (an empty iso-surface at every lambda) has no data to scale.
            axis.set_xlim(min(values) / 1.5, max(values) * 1.5)
            axis.set_xscale("log")
            axis.set_xticks(values, [f"{v:g}" for v in values])
            axis.set_xticks([], minor=True)
            axis.set_xlabel("lambda", fontsize=8)
            axis.set_title(name, fontsize=8)
            axis.tick_params(labelsize=7)
            axis.spines[["top", "right"]].set_visible(False)
        for axis in axes.flat[len(names):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower right", fontsize=7, frameon=False, ncol=2)
        figure.suptitle("experiment 0: (lambda D, lambda rho, T_r / lambda) against lambda = 1", fontsize=10)
        figure.tight_layout()
        _save_figure(figure, figure_dir / stem)


def row_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """The summaries' groups of rows: each cell, "all", and the strata
    maturity >= / < MATURITY_SPLIT (maturity_ge_15, maturity_lt_15; a row
    without a finite value is in neither); empty groups are left out."""
    groups: dict[str, list[Mapping[str, Any]]] = {cell: [row for row in rows if row["cell"] == cell] for cell in CELLS}
    groups["all"] = list(rows)
    values = {id(row): sa.as_float(row.get("maturity")) for row in rows}
    groups[f"maturity_ge_{MATURITY_SPLIT:g}"] = [row for row in rows if values[id(row)] >= MATURITY_SPLIT]
    groups[f"maturity_lt_{MATURITY_SPLIT:g}"] = [row for row in rows if values[id(row)] < MATURITY_SPLIT]
    return {name: group for name, group in groups.items() if group}


def median_of(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """The median of a column over the rows with a finite value; None
    without one."""
    values = np.array([sa.as_float(row.get(key)) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def frame_metric_keys(frames: Sequence[str] = ("d120", "d180")) -> list[str]:
    """<frame>_<metric> for the summarised frames and every metric."""
    return [f"{frame}_{name}" for frame in frames for name in METRIC_NAMES]


def substitute_medians(rows: Sequence[Mapping[str, Any]], frames: Sequence[str] = ("d120", "d180")) -> dict[str, Any]:
    """Per objective and delta_a, per group (``row_groups``: the cells,
    "all", the two maturity strata): the row count, the skipped rows
    (n_skipped, a T_0 outside its range; excluded) and the median of
    every <frame>_<metric> of the frames (NaN entries left out; None
    when no value is finite)."""
    keys = frame_metric_keys(frames)
    out: dict[str, Any] = {}
    for objective in OBJECTIVES:
        selected = [row for row in rows if row["objective"] == objective]
        out[objective] = {}
        for delta_a in sorted({float(row["delta_a"]) for row in selected}):
            at = [row for row in selected if float(row["delta_a"]) == delta_a]
            entries: dict[str, Any] = {}
            for group, group_rows in row_groups(at).items():
                kept = [row for row in group_rows if not str(row.get("skipped") or "")]
                entry: dict[str, Any] = {"n_rows": len(group_rows), "n_skipped": len(group_rows) - len(kept)}
                entry.update({key: median_of(kept, key) for key in keys})
                entries[group] = entry
            out[objective][f"{delta_a:g}"] = entries
    return out


def assemble_substitute(root: Path, dispatch: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """substitute.csv, its figures and substitute_summary.json
    (``write_summary`` plus the medians, ``substitute_medians``, and the
    skipped pairs) from the patient records present."""
    rows: list[dict[str, Any]] = []
    records = patient_records(root, "substitute")
    for record in records:
        rows.extend(record["rows"])
    sa.write_csv(root / "substitute.csv", rows, SUBSTITUTE_COLUMNS)
    if rows:
        substitute_figures(rows, root / "figures")
    summary = write_summary(root, "substitute", records, len(rows), dispatch)
    summary["delta_a"] = sorted({float(row["delta_a"]) for row in rows if row["objective"] != "truth"})
    summary["skipped"] = [{"patient": record["patient"], **entry} for record in records for entry in record.get("skipped", [])]
    summary["n_skipped"] = len(summary["skipped"])
    summary["medians"] = substitute_medians(rows)
    write_record(root / "substitute_summary.json", summary)
    return rows


def _metric_axes(names: Sequence[str]) -> tuple[Any, Any]:
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    return plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.6 * n_rows), squeeze=False)


def substitute_figures(rows: Sequence[Mapping[str, Any]], figure_dir: Path, frames: Sequence[str] = ("d120", "d180")) -> None:
    """Per frame and objective: every comparison metric against delta_a
    (linear axis), points per (patient, delta_a) coloured by cell, one
    line per cell through the cell's median at each delta_a; skipped
    pairs have no finite value and are not drawn."""
    figure_dir.mkdir(exist_ok=True)
    for frame in frames:
        for objective in OBJECTIVES:
            selected = [row for row in rows if row["objective"] == objective]
            if not selected:
                continue
            deltas = sorted({float(row["delta_a"]) for row in selected})
            figure, axes = _metric_axes(PLOTTED_METRICS)
            for axis, name in zip(axes.flat, PLOTTED_METRICS):
                key = f"{frame}_{name}"
                for cell, color in COLOR_CELLS.items():
                    cell_rows = [row for row in selected if row["cell"] == cell]
                    if not cell_rows:
                        continue
                    x = np.array([float(row["delta_a"]) for row in cell_rows])
                    y = np.array([sa.as_float(row.get(key)) for row in cell_rows])
                    finite = np.isfinite(y)
                    axis.scatter(x[finite], y[finite], s=10, color=color, alpha=0.5, linewidths=0, label=cell)
                    medians = [(delta_a, median_of([row for row in cell_rows if float(row["delta_a"]) == delta_a], key)) for delta_a in deltas]
                    points = [(x_, m) for x_, m in medians if m is not None]
                    if points:
                        axis.plot([p[0] for p in points], [p[1] for p in points], "-", color=color, linewidth=1.5)
                # Limits by hand: a panel whose values are all NaN (an empty
                # iso-surface for every substitute) has no data to scale.
                span = max(deltas[-1] - deltas[0], 1.0)
                axis.set_xlim(deltas[0] - 0.15 * span, deltas[-1] + 0.15 * span)
                axis.set_xticks(deltas, [f"{d:g}" for d in deltas])
                axis.axvline(0.0, color="#52514e", linewidth=0.6, linestyle=":")
                axis.set_xlabel("delta_a = rho (T_r - T_0)", fontsize=8)
                axis.set_title(f"{frame} {name}", fontsize=8)
                axis.tick_params(labelsize=7)
                axis.spines[["top", "right"]].set_visible(False)
            for axis in axes.flat[len(PLOTTED_METRICS):]:
                axis.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower right", fontsize=7, frameon=False, ncol=2)
            figure.suptitle(f"experiment 2, objective {objective}: substitute against truth at {frame}; lines: cell medians per delta_a", fontsize=10)
            figure.tight_layout()
            _save_figure(figure, figure_dir / f"substitute_{frame}_objective_{objective}")


def seedfix_medians(rows: Sequence[Mapping[str, Any]], frames: Sequence[str] = ("d120", "d180")) -> dict[str, dict[str, dict[str, Any]]]:
    """Per lambda mode and group (``row_groups``: the cells, "all", the
    two maturity strata): the patient count, the number of fits that
    hit a bound (n_bound_hit) and, over the patients whose fit did not,
    the median of |rho_fitted T_r_fitted - rho T_r| (abs_rho_T_r_error),
    of |log(lambda_fitted / lambda)| (abs_log_lambda_error) and of every
    <frame>_<metric> of the frames (NaN entries left out; None when no
    value is finite)."""
    keys = frame_metric_keys(frames)
    medians: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in LAMBDA_MODES:
        selected = [row for row in rows if row.get("lambda_mode") == mode]
        if not selected:
            continue
        medians[mode] = {}
        for group, group_rows in row_groups(selected).items():
            kept = [row for row in group_rows if str(row.get("bound_hit")) != "True"]
            entry: dict[str, Any] = {"n_patients": len(group_rows), "n_bound_hit": len(group_rows) - len(kept)}
            errors = np.array([abs(float(row["rho_T_r_fitted"]) - float(row["rho_T_r"])) for row in kept])
            entry["abs_rho_T_r_error"] = float(np.median(errors)) if errors.size else None
            widths = np.array([abs(np.log(float(row["fitted_lambda_mm"]) / float(row["lambda_truth_mm"]))) for row in kept])
            entry["abs_log_lambda_error"] = float(np.median(widths)) if widths.size else None
            entry.update({key: median_of(kept, key) for key in keys})
            medians[mode][group] = entry
    return medians


def assemble_seedfix(root: Path, dispatch: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """seedfix.csv, its figures and seedfix_summary.json (``write_summary``
    plus the seed's record, the lambda modes and the medians,
    ``seedfix_medians``) from the patient records present."""
    rows: list[dict[str, Any]] = []
    records = patient_records(root, "seedfix")
    for record in records:
        rows.extend(record["rows"])
    sa.write_csv(root / "seedfix.csv", rows, SEEDFIX_COLUMNS)
    if rows:
        seedfix_figures(rows, root / "figures")
    summary = write_summary(root, "seedfix", records, len(rows), dispatch)
    summary["seed"] = records[0]["seed"] if records else None
    summary["lambda_modes"] = sorted({str(row["lambda_mode"]) for row in rows if row["lambda_mode"] != "truth"}, key=lambda m: LAMBDA_MODES.index(m) if m in LAMBDA_MODES else 99)
    summary["medians"] = seedfix_medians(rows)
    write_record(root / "seedfix_summary.json", summary)
    return rows


def _log_axis(axis: Any, values: NDArray, label: str, fallback: tuple[float, float] = (1.0, 100.0)) -> None:
    """A log x axis scaled by hand around the finite values (the fallback
    range without any), with plain tick labels."""
    finite = values[np.isfinite(values) & (values > 0)]
    low, high = (float(finite.min()) / 1.5, float(finite.max()) * 1.5) if finite.size else fallback
    if low == high:
        low, high = low / 1.5, high * 1.5
    axis.set_xlim(low, high)
    axis.set_xscale("log")
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if low <= t <= high]
    axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
    axis.set_xticks([], minor=True)
    axis.set_xlabel(label, fontsize=8)


def seedfix_figures(rows: Sequence[Mapping[str, Any]], figure_dir: Path, frames: Sequence[str] = ("d120", "d180")) -> None:
    """Per frame and lambda mode: every comparison metric against the
    truth's maturity (log axis), one marker per patient coloured by cell
    (a bound hit hollow)."""
    figure_dir.mkdir(exist_ok=True)
    for frame in frames:
        for mode in LAMBDA_MODES:
            selected = [row for row in rows if row.get("lambda_mode") == mode]
            if not selected:
                continue
            maturities = np.array([sa.as_float(row.get("maturity")) for row in selected])
            figure, axes = _metric_axes(PLOTTED_METRICS)
            for axis, name in zip(axes.flat, PLOTTED_METRICS):
                key = f"{frame}_{name}"
                for cell, color in COLOR_CELLS.items():
                    cell_rows = [row for row in selected if row["cell"] == cell]
                    if not cell_rows:
                        continue
                    x = np.array([sa.as_float(row.get("maturity")) for row in cell_rows])
                    y = np.array([sa.as_float(row.get(key)) for row in cell_rows])
                    hit = np.array([str(row.get("bound_hit")) == "True" for row in cell_rows])
                    finite = np.isfinite(x) & np.isfinite(y)
                    axis.scatter(x[finite & ~hit], y[finite & ~hit], s=16, color=color, alpha=0.7, linewidths=0, label=cell)
                    axis.scatter(x[finite & hit], y[finite & hit], s=16, facecolors="none", edgecolors=color, linewidths=0.8)
                # Limits by hand: a panel whose values are all NaN has no data to scale.
                _log_axis(axis, maturities, "maturity (truth)")
                axis.set_title(f"{frame} {name}", fontsize=8)
                axis.tick_params(labelsize=7)
                axis.spines[["top", "right"]].set_visible(False)
            for axis in axes.flat[len(PLOTTED_METRICS):]:
                axis.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower right", fontsize=7, frameon=False, ncol=2)
            figure.suptitle(f"experiment 3, lambda {mode}: fixed seed with fitted T_r against truth at {frame}; one marker per patient (hollow: bound hit)", fontsize=10)
            figure.tight_layout()
            _save_figure(figure, figure_dir / f"seedfix_{frame}_{mode}")


def profile_medians(rows: Sequence[Mapping[str, Any]], frames: Sequence[str] = ("d120", "d180")) -> dict[str, Any]:
    """Per sigma_0 and group (``row_groups``): the patient count, the
    bound hits (excluded) and the medians of fitted_T_r, rho_T_r_fitted,
    objective_achieved and every <frame>_<metric> of the frames."""
    keys = ["fitted_T_r", "rho_T_r_fitted", "objective_achieved", *frame_metric_keys(frames)]
    out: dict[str, Any] = {}
    for sigma in sorted({float(row["sigma_mm"]) for row in rows}):
        at = [row for row in rows if float(row["sigma_mm"]) == sigma]
        entries: dict[str, Any] = {}
        for group, group_rows in row_groups(at).items():
            kept = [row for row in group_rows if str(row.get("bound_hit")) != "True"]
            entry: dict[str, Any] = {"n_patients": len(group_rows), "n_bound_hit": len(group_rows) - len(kept)}
            entry.update({key: median_of(kept, key) for key in keys})
            entries[group] = entry
        out[f"{sigma:g}"] = entries
    return out


def assemble_profile(root: Path, dispatch: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """profile.csv, its figures and profile_summary.json (``write_summary``
    plus the seed peak, the sigmas and the medians, ``profile_medians``)
    from the patient records present."""
    rows: list[dict[str, Any]] = []
    records = patient_records(root, "profile")
    for record in records:
        rows.extend(record["rows"])
    sa.write_csv(root / "profile.csv", rows, PROFILE_COLUMNS)
    if rows:
        profile_figures(rows, root / "figures")
    summary = write_summary(root, "profile", records, len(rows), dispatch)
    summary["seed_peak"] = records[0]["seed_peak"] if records else None
    summary["sigmas"] = sorted({float(row["sigma_mm"]) for row in rows})
    summary["medians"] = profile_medians(rows)
    write_record(root / "profile_summary.json", summary)
    return rows


def profile_figures(rows: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    """profile_objective: objective_achieved against sigma_0, one line per
    patient coloured by cell (a bound hit as a hollow marker);
    profile_d180: d180 mass_beyond_edema_rel and mass_rel against
    sigma_0 the same way."""
    figure_dir.mkdir(exist_ok=True)
    patients = list(dict.fromkeys(str(row["patient"]) for row in rows))
    sigmas = sorted({float(row["sigma_mm"]) for row in rows})
    panels = {"profile_objective": ["objective_achieved"], "profile_d180": ["d180_mass_beyond_edema_rel", "d180_mass_rel"]}
    for stem, keys in panels.items():
        figure, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2), squeeze=False)
        for axis, key in zip(axes.flat, keys):
            for patient in patients:
                patient_rows = sorted((row for row in rows if str(row["patient"]) == patient), key=lambda row: float(row["sigma_mm"]))
                cell = str(patient_rows[0]["cell"])
                color = COLOR_CELLS.get(cell, "#52514e")
                x = np.array([float(row["sigma_mm"]) for row in patient_rows])
                y = np.array([sa.as_float(row.get(key)) for row in patient_rows])
                hit = np.array([str(row.get("bound_hit")) == "True" for row in patient_rows])
                finite = np.isfinite(y)
                axis.plot(x[finite], y[finite], "-", color=color, linewidth=1.2, alpha=0.8, label=f"{patient} {cell}")
                axis.scatter(x[finite & ~hit], y[finite & ~hit], s=18, color=color, linewidths=0)
                axis.scatter(x[finite & hit], y[finite & hit], s=18, facecolors="none", edgecolors=color, linewidths=0.8)
            axis.set_xticks(sigmas, [f"{s:g}" for s in sigmas])
            axis.set_xlim(min(sigmas) - 0.5, max(sigmas) + 0.5)
            axis.set_xlabel("seed width sigma_0 (mm)", fontsize=8)
            axis.set_title(key, fontsize=8)
            axis.tick_params(labelsize=7)
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", fontsize=6, frameon=False, ncol=1)
        figure.suptitle("experiment 4: fixed seed of width sigma_0 with fitted T_r (hollow: bound hit)", fontsize=10)
        figure.tight_layout()
        _save_figure(figure, figure_dir / stem)


# --- commands ---


def parse_band(text: str | None, default: tuple[float, float], flag: str) -> tuple[float, float]:
    """A band argument "lo,hi" with lo < hi; the default when not given."""
    if text is None or str(text).strip() == "":
        return default
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"{flag}: expected lo,hi, got {text!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not lo < hi:
        raise ValueError(f"{flag}: need lo < hi, got {text!r}.")
    return lo, hi


def parse_bands(args: argparse.Namespace) -> SizeBands:
    """The size bands of the command line (--r-core-band, --r-whole-band,
    --ratio-band; DEFAULT_BANDS where not given)."""
    values = {
        name: parse_band(getattr(args, attribute, None), getattr(DEFAULT_BANDS, name), "--" + attribute.replace("_", "-"))
        for name, attribute in BAND_ARGS.items()
    }
    return SizeBands(**values)


def design_command(args: argparse.Namespace, device: str = "") -> Path:
    root = make_design(
        args.config,
        args.search_space,
        args.output_dir,
        args.name,
        {key: getattr(args, key) for key in sa.DEFAULT_TISSUE_MAPS},
        None if args.seed_voxel is None else [int(v) for v in args.seed_voxel.split(",")],
        args.log2_candidates,
        args.design_seed,
        args.smoke,
        args.cavity_threshold,
        args.rt_margin_mm,
        args.rt_dose_per_fraction,
        tr_max=float(args.tr_max),
        bands=parse_bands(args),
        patients_per_cell=SMOKE.patients_per_cell if args.smoke else int(args.patients_per_cell),
        dt_mode=str(args.dt_mode),
        device=device,
    )
    spec = sa.read_json(root / "spec.json")
    print(f"design directory: {root}")
    print(f"tissue maps: {spec['tissue_maps']}")
    print(
        f"seed voxel {spec['seed_voxel']} (target {spec['seed_target_voxel']}, snapped {spec['seed_snap_distance_voxels']:g} voxels); "
        f"floor {spec['gaussian_seed_floor']:g}, precision {spec['precision']}, resolution_factor {spec['resolution_factor']:g}"
        f"{' (smoke)' if spec['smoke'] else ''}; dt mode {spec['dt_mode']} (dx {spec['dt_modes']['grid_spacing_mm']:g} mm)"
    )
    screening = spec["screening"]
    counts = screening["counts"]
    print(
        f"screening: {counts['n_candidates']} candidates, {counts['n_rejected_seed_wide']} with a seed wider than {spec['seed_width_cap']['cap']:g} "
        f"front widths, {counts['n_rejected_T_r_below_min']} with T_r < {screening['tr_min']:g} d and "
        f"{counts['n_rejected_T_r_above_max']} with T_r > {screening['tr_max']:g} d rejected; {counts['n_screened']} solved, "
        f"{counts['n_accepted']} accepted, {counts['n_solve_failed']} failed; bands {screening['bands']}; {screening['wall_time_s'] / 60:.1f} min"
    )
    records = sa.read_csv(root / "design.csv")
    for cell in CELLS:
        members = [r for r in records if r["cell"] == cell]
        cell_counts = counts["cells"][cell]
        rejected = ", ".join(f"{reason} {n}" for reason, n in cell_counts["n_rejected"].items() if n)
        print(
            f"  {cell}: {cell_counts['n_candidates']} candidates ({cell_counts['n_admissible']} admissible), {cell_counts['n_screened']} screened, "
            f"{cell_counts['n_accepted']} accepted, {cell_counts['n_solve_failed']} failed; rejected: {rejected or '-'}"
        )
        if members:
            spans = {
                key: (min(float(r[key]) for r in members), max(float(r[key]) for r in members))
                for key in ("resection_time", "rho_T_r", "maturity", "r_core_mm", "r_whole_mm", "whole_core_ratio", "R_over_lambda", "visibility_margin")
            }
            print(
                f"    {len(members)} patients: T_r {spans['resection_time'][0]:.0f}-{spans['resection_time'][1]:.0f} d, rho T_r "
                f"{spans['rho_T_r'][0]:.1f}-{spans['rho_T_r'][1]:.1f}, maturity {spans['maturity'][0]:.1f}-{spans['maturity'][1]:.1f}, "
                f"r_core {spans['r_core_mm'][0]:.1f}-{spans['r_core_mm'][1]:.1f} mm, r_whole "
                f"{spans['r_whole_mm'][0]:.1f}-{spans['r_whole_mm'][1]:.1f} mm, ratio {spans['whole_core_ratio'][0]:.2f}-{spans['whole_core_ratio'][1]:.2f}, "
                f"R/lambda {spans['R_over_lambda'][0]:.1f}-{spans['R_over_lambda'][1]:.1f}, margin {spans['visibility_margin'][0]:.2f}-{spans['visibility_margin'][1]:.2f}"
            )
            for r in members:
                print(
                    f"      {r['patient']}: T_r {float(r['resection_time']):.0f} d, lambda {float(r['front_width_mm']):.2f} mm, sigma "
                    f"{float(r['seed_sigma_mm']):.2f} mm, maturity {float(r['maturity']):.1f}, {r['steps_per_day']} steps/day"
                )
    refined = [r["patient"] for r in records if str(read_record(root / SCREEN_DIR / f"c{int(r['candidate']):04d}" / SCREEN_FILE).get("dt_refined")) == "True"]
    if refined:
        print(f"the solver refined the requested step of {refined} in the screening (its estimate needs more than {BASE_STEPS_PER_DAY} steps/day)")
    print(f"{spec['n_patients']} patients from {spec['n_candidates']} Sobol' points (seed {spec['design_seed']})")
    return root


def invariance_command(root: Path, args: argparse.Namespace) -> None:
    cohort = load_cohort(root)
    out_dir = root / "runs" / "invariance"
    if (out_dir / "invariance.json").is_file():
        print(f"invariance: {out_dir / 'invariance.json'} exists, kept")
    else:
        lambdas = [float(v) for v in args.lambdas.split(",")] if args.lambdas else list(INVARIANCE_LAMBDAS)
        start = time.perf_counter()
        run_invariance(cohort, out_dir, lambdas, args.white_matter_diffusivity, args.rho, args.resection_time)
        print(f"invariance: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    assemble_invariance(root)


def parse_devices(text: str) -> list[str]:
    """--gpus as a list of devices: comma-separated CUDA ids, '' the CPU
    ('' alone is one CPU process, ',' two)."""
    return [item.strip() for item in str(text).split(",")]


def patient_blocks(patients: Sequence[Patient], n_blocks: int) -> list[list[Patient]]:
    """Contiguous blocks of the patients in their order, one per device,
    of sizes differing by at most one (the first blocks longer); the
    empty blocks of more devices than patients are dropped."""
    if not patients:
        return []
    pieces = np.array_split(np.arange(len(patients)), max(1, int(n_blocks)))
    return [[patients[int(i)] for i in piece] for piece in pieces if len(piece)]


def worker_command(root: Path, experiment: str, device: str, patients: Sequence[Patient], args: argparse.Namespace) -> list[str]:
    """The command line of one dispatched worker: this script's
    experiment subcommand on one device and one block of patients, the
    other options passed through (those given: --draws for fisher;
    --maxfev and --dt-mode for the fits; --delta-a for substitute; --tr-bounds and
    --seed-peak for seedfix and profile; --seed-sigma-mm and
    --lambda-mode for seedfix; --sigmas for profile), the assembly
    skipped."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        experiment,
        "--output-dir",
        str(root.parent),
        "--name",
        root.name,
        "--gpus",
        device,
        "--patients",
        ",".join(patient.id for patient in patients),
        NO_ASSEMBLE_FLAG,
    ]
    if getattr(args, "smoke", False):
        command.append("--smoke")
    forwarded: list[tuple[tuple[str, ...], str, str]] = [  # (experiments, attribute, flag)
        (("fisher",), "draws", "--draws"),
        (("substitute", "seedfix", "profile"), "maxfev", "--maxfev"),
        (("substitute", "seedfix", "profile"), "dt_mode", "--dt-mode"),
        (("substitute",), "delta_a", "--delta-a"),
        (("seedfix", "profile"), "tr_bounds", "--tr-bounds"),
        (("seedfix", "profile"), "seed_peak", "--seed-peak"),
        (("seedfix",), "seed_sigma_mm", "--seed-sigma-mm"),
        (("seedfix",), "lambda_mode", "--lambda-mode"),
        (("profile",), "sigmas", "--sigmas"),
    ]
    for experiments, attribute, flag in forwarded:
        value = getattr(args, attribute, None)
        if experiment in experiments and value is not None and str(value) != "":
            command += [flag, str(value)]
    return command


def dispatch(root: Path, experiment: str, devices: Sequence[str], patients: Sequence[Patient], args: argparse.Namespace) -> dict[str, Any]:
    """
    Run an experiment over several devices: the patients cut into
    contiguous blocks (``patient_blocks``), one worker process per
    device (``worker_command``) with the sensitivity script's
    environment handling (XLA_PYTHON_CLIENT_PREALLOCATE=false and
    CUDA_VISIBLE_DEVICES=<device>, or JAX_PLATFORMS=cpu for ''), its
    output in <root>/logs/<experiment>_<device>.log (cpu for ''; a
    repeated device gets a _<k> suffix); waits for every worker.

    Returns:
        The dispatch record: devices, started, finished, wall_time_s,
        n_failed and blocks (device, log, patients, returncode,
        wall_time_s per worker). The caller assembles and then raises
        on failures (``check_dispatch``).
    """
    blocks = patient_blocks(patients, len(devices))
    log_dir = root / LOG_DIR
    log_dir.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    workers: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    for device, block in zip(devices, blocks):
        label = device or "cpu"
        count = used.get(label, 0)
        used[label] = count + 1
        log_path = log_dir / f"{experiment}_{label}{'' if count == 0 else f'_{count}'}.log"
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if device:
            env["CUDA_VISIBLE_DEVICES"] = device
        else:
            env["JAX_PLATFORMS"] = "cpu"
        handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(worker_command(root, experiment, device, block, args), stdout=handle, stderr=subprocess.STDOUT, env=env)
        workers.append({"device": device, "log": str(log_path), "patients": [p.id for p in block], "process": process, "handle": handle, "start": time.perf_counter()})
        print(f"{experiment}: worker on {label} for {', '.join(p.id for p in block)} -> {log_path}", flush=True)
    records: list[dict[str, Any]] = []
    for worker in workers:
        code = int(worker["process"].wait())
        worker["handle"].close()
        wall = time.perf_counter() - worker["start"]
        records.append({"device": worker["device"], "log": worker["log"], "patients": worker["patients"], "returncode": code, "wall_time_s": wall})
        print(f"{experiment}: worker on {worker['device'] or 'cpu'} {'ok' if code == 0 else f'FAILED (exit {code})'} in {wall / 60:.1f} min", flush=True)
    return {
        "experiment": experiment,
        "devices": list(devices),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": time.perf_counter() - start,
        "n_failed": sum(1 for r in records if r["returncode"] != 0),
        "blocks": records,
    }


def check_dispatch(record: Mapping[str, Any]) -> None:
    """Raise when a worker of a dispatch failed, naming its log."""
    failed = [block for block in record["blocks"] if block["returncode"] != 0]
    if failed:
        raise RuntimeError(
            f"{record['experiment']}: {len(failed)} of {len(record['blocks'])} workers failed: "
            + "; ".join(f"{block['device'] or 'cpu'} (exit {block['returncode']}, see {block['log']})" for block in failed)
        )


def fisher_command(root: Path, args: argparse.Namespace, smoke: bool, devices: Sequence[str]) -> None:
    cohort = load_cohort(root)
    # The fixed step whatever the design's mode: the scaled-dt T_r column
    # steps at dt e^{+-h}, which a half-day step would push past the frame
    # rounding's limit of half a day.
    patients = [replace(patient, steps_per_day=BASE_STEPS_PER_DAY) for patient in select_patients(cohort, args.patients, smoke)]
    if min(len(devices), len(patients)) > 1:
        record = dispatch(root, "fisher", devices, patients, args)
        assemble_fisher(root, record)
        check_dispatch(record)
        return
    draws = int(args.draws) if args.draws is not None else (SMOKE.draws if smoke else NOISE_DRAWS)
    out_dir = root / "runs" / "fisher"
    fd_flags = {r["patient"]: r.get("fd_check") == "True" for r in sa.read_csv(root / "design.csv")}
    start = time.perf_counter()
    for index, patient in enumerate(patients):
        marker = out_dir / patient.id / PATIENT_FILE.format(experiment="fisher")
        if marker.is_file():
            print(f"fisher {patient.id}: {marker} exists, skipped", flush=True)
            continue
        print(f"fisher {patient.id} ({patient.cell}): T_r {patient.resection_time:.1f}, K = {draws}", flush=True)
        seed = int(cohort.spec["design_seed"]) * 1000 + [p.id for p in cohort.patients].index(patient.id)
        record = fisher_patient(cohort, patient, out_dir, draws, fd_flags.get(patient.id, False), seed)
        summary = record["analysis"]["scaled_dt"]
        print(
            f"fisher {patient.id}: {record['observation']['n_observations']} observations on {record['observation']['n_voxels']} voxels; "
            f"|We|/|We1| (a) {summary['a']['We_ratio']:.2e}, (f) {summary['f']['We_ratio']:.2e}; CR(log T_r) (b) {summary['b']['cr_log_T_r']:.3g}, "
            f"(f) {summary['f']['cr_log_T_r']:.3g}; {record['wall_time_s'] / 60:.1f} min ({index + 1} done)",
            flush=True,
        )
    print(f"fisher: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    if not getattr(args, "no_assemble", False):
        assemble_fisher(root)


def check_dt_mode(cohort: Cohort, args: argparse.Namespace) -> str:
    """The run's --dt-mode (DEFAULT_DT_MODE when not given), which must be
    the design's recorded dt_mode (spec.json; "fixed" for a design made
    before the modes existed, which stepped at the base config's rate):
    every solve of a patient uses the design's step, so another mode
    raises with both values. Returns the mode."""
    mode = str(getattr(args, "dt_mode", None) or DEFAULT_DT_MODE)
    recorded = str(cohort.spec.get("dt_mode", "fixed"))
    if mode != recorded:
        raise ValueError(
            f"--dt-mode {mode} but the design {cohort.root} was made with dt_mode {recorded}: every run of a patient uses the design's "
            f"time step (design.csv's steps_per_day); pass --dt-mode {recorded}, or make a design with --dt-mode {mode}."
        )
    return mode


def dt_check_command(root: Path, args: argparse.Namespace, smoke: bool) -> None:
    cohort = load_cohort(root)
    patients = select_patients(cohort, args.patients, smoke) if args.patients else [patient for patient in cohort.patients if patient.fd_check]
    out_dir = root / "runs" / "dt_check"
    start = time.perf_counter()
    for index, patient in enumerate(patients):
        marker = out_dir / patient.id / PATIENT_FILE.format(experiment="dt_check")
        if marker.is_file():
            print(f"dt-check {patient.id}: {marker} exists, skipped", flush=True)
            continue
        print(f"dt-check {patient.id} ({patient.cell}): T_r {patient.resection_time:.1f}, D {patient.diffusivity:.3g} mm^2/day, maturity {patient.maturity:.1f}", flush=True)
        record = dt_check_patient(cohort, patient, out_dir)
        row = record["row"]
        print(
            f"dt-check {patient.id}: dt {row['dt_fixed']:g} -> {row['dt_stability']:g}; pre Dice edema {row['pre_dice_edema']:.4f}, mass rel {row['pre_mass_rel']:.2e}; "
            f"d180 Dice edema {row['d180_dice_edema']:.4f}, mass beyond edema rel {row['d180_mass_beyond_edema_rel']:.3f}; {record['wall_time_s'] / 60:.1f} min ({index + 1} done)",
            flush=True,
        )
    print(f"dt-check: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    assemble_dt_check(root)


def spec_tr_max(cohort: Cohort) -> float:
    """The design's --tr-max (spec.json, screening.tr_max), the cap of the
    substitute arm's T_0."""
    return float(cohort.spec.get("screening", {}).get("tr_max", DEFAULT_TR_MAX))


def parse_floats(text: str | None, default: Sequence[float], flag: str) -> list[float]:
    """A comma-separated list of floats; the default when not given."""
    if text is None or str(text).strip() == "":
        return [float(v) for v in default]
    try:
        return [float(part) for part in str(text).split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"{flag}: expected comma-separated numbers, got {text!r}.") from error


def substitute_command(root: Path, args: argparse.Namespace, smoke: bool, devices: Sequence[str]) -> None:
    cohort = load_cohort(root)
    check_dt_mode(cohort, args)
    patients = select_patients(cohort, args.patients, smoke)
    if min(len(devices), len(patients)) > 1:
        record = dispatch(root, "substitute", devices, patients, args)
        assemble_substitute(root, record)
        check_dispatch(record)
        return
    maxfev = int(args.maxfev) if args.maxfev is not None else (SMOKE.maxfev if smoke else SUBSTITUTE_MAXFEV)
    deltas = parse_floats(getattr(args, "delta_a", None), SUBSTITUTE_DELTA_A, "--delta-a")
    tr_max = spec_tr_max(cohort)
    out_dir = root / "runs" / "substitute"
    start = time.perf_counter()
    for index, patient in enumerate(patients):
        marker = out_dir / patient.id / PATIENT_FILE.format(experiment="substitute")
        if marker.is_file():
            print(f"substitute {patient.id}: {marker} exists, skipped", flush=True)
            continue
        print(
            f"substitute {patient.id} ({patient.cell}): T_r {patient.resection_time:.1f}, rho T_r {patient.rho * patient.resection_time:.2f}, "
            f"delta_a {','.join(f'{d:g}' for d in deltas)}, T_0 in [{TR_MIN:g}, {tr_max:g}], maxfev {maxfev}",
            flush=True,
        )
        record = substitute_patient(cohort, patient, out_dir, maxfev, deltas, tr_max)
        print(f"substitute {patient.id}: {record['wall_time_s'] / 60:.1f} min ({index + 1} done)", flush=True)
    print(f"substitute: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    if not getattr(args, "no_assemble", False):
        assemble_substitute(root)


def parse_tr_bounds(text: str | None) -> tuple[float, float]:
    """--tr-bounds as (lo, hi) days: two comma-separated values with
    0 < lo < hi; TR_BOUNDS when not given."""
    if text is None or str(text).strip() == "":
        return TR_BOUNDS
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"--tr-bounds: expected lo,hi in days, got {text!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not 0.0 < lo < hi:
        raise ValueError(f"--tr-bounds: need 0 < lo < hi, got {text!r}.")
    return lo, hi


def fixed_seed_args(args: argparse.Namespace) -> tuple[float, float]:
    """(--seed-peak, --seed-sigma-mm) with the defaults DEFAULT_SEED_PEAK
    and DEFAULT_SEED_SIGMA_MM; the peak must lie in (0, 1], the width be
    positive."""
    peak = float(args.seed_peak) if getattr(args, "seed_peak", None) is not None else DEFAULT_SEED_PEAK
    sigma = float(args.seed_sigma_mm) if getattr(args, "seed_sigma_mm", None) is not None else DEFAULT_SEED_SIGMA_MM
    if not 0.0 < peak <= 1.0:
        raise ValueError(f"--seed-peak must lie in (0, 1], got {peak!r}.")
    if not sigma > 0.0:
        raise ValueError(f"--seed-sigma-mm must be positive, got {sigma!r}.")
    return peak, sigma


def seedfix_command(root: Path, args: argparse.Namespace, smoke: bool, devices: Sequence[str]) -> None:
    cohort = load_cohort(root)
    check_dt_mode(cohort, args)
    patients = select_patients(cohort, args.patients, smoke)
    if min(len(devices), len(patients)) > 1:
        record = dispatch(root, "seedfix", devices, patients, args)
        assemble_seedfix(root, record)
        check_dispatch(record)
        return
    maxfev = int(args.maxfev) if args.maxfev is not None else (SMOKE.maxfev if smoke else MAXFEV)
    tr_bounds = parse_tr_bounds(getattr(args, "tr_bounds", None))
    peak, sigma = fixed_seed_args(args)
    modes = parse_lambda_modes(getattr(args, "lambda_mode", None))
    out_dir = root / "runs" / "seedfix"
    start = time.perf_counter()
    for index, patient in enumerate(patients):
        marker = out_dir / patient.id / PATIENT_FILE.format(experiment="seedfix")
        if marker.is_file():
            print(f"seedfix {patient.id}: {marker} exists, skipped", flush=True)
            continue
        print(
            f"seedfix {patient.id} ({patient.cell}): T_r {patient.resection_time:.1f}, rho T_r {patient.rho * patient.resection_time:.2f}, "
            f"lambda {patient.front_width:.2f} mm, R/lambda {patient.r_over_lambda:.1f}, seed peak {peak:.3f} sigma {sigma:.2f} mm, T_r in "
            f"[{tr_bounds[0]:g}, {tr_bounds[1]:g}], lambda in [{LAMBDA_BOUNDS[0]:g}, {LAMBDA_BOUNDS[1]:g}], modes {','.join(modes)}, maxfev {maxfev}",
            flush=True,
        )
        record = seedfix_patient(cohort, patient, out_dir, maxfev, tr_bounds, peak, sigma, modes)
        print(f"seedfix {patient.id}: {record['wall_time_s'] / 60:.1f} min ({index + 1} done)", flush=True)
    print(f"seedfix: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    if not getattr(args, "no_assemble", False):
        assemble_seedfix(root)


def profile_default_patients(cohort: Cohort, smoke: bool) -> list[Patient]:
    """The profile's default patients: per cell (CELLS' order) the
    fd_check patient and the next accepted patient in design order (8 for
    a full design); with smoke the first patient of each cell alone (4)."""
    if smoke:
        return select_patients(cohort, None, True)
    chosen: list[Patient] = []
    for cell in CELLS:
        members = [p for p in cohort.patients if p.cell == cell]
        flagged = [p for p in members if p.fd_check]
        first = flagged[0] if flagged else (members[0] if members else None)
        if first is None:
            continue
        chosen.extend(members[members.index(first) : members.index(first) + 2])
    return chosen


def profile_command(root: Path, args: argparse.Namespace, smoke: bool, devices: Sequence[str]) -> None:
    cohort = load_cohort(root)
    check_dt_mode(cohort, args)
    if args.patients:
        patients = select_patients(cohort, args.patients, smoke)
    else:
        patients = profile_default_patients(cohort, smoke)
    if min(len(devices), len(patients)) > 1:
        record = dispatch(root, "profile", devices, patients, args)
        assemble_profile(root, record)
        check_dispatch(record)
        return
    maxfev = int(args.maxfev) if args.maxfev is not None else (SMOKE.maxfev if smoke else MAXFEV)
    tr_bounds = parse_tr_bounds(getattr(args, "tr_bounds", None))
    peak, _ = fixed_seed_args(args)
    sigmas = parse_floats(getattr(args, "sigmas", None), SMOKE.sigmas if smoke else PROFILE_SIGMAS, "--sigmas")
    if any(s <= 0 for s in sigmas):
        raise ValueError(f"--sigmas must be positive widths in mm, got {args.sigmas!r}.")
    out_dir = root / "runs" / "profile"
    start = time.perf_counter()
    for index, patient in enumerate(patients):
        marker = out_dir / patient.id / PATIENT_FILE.format(experiment="profile")
        if marker.is_file():
            print(f"profile {patient.id}: {marker} exists, skipped", flush=True)
            continue
        print(
            f"profile {patient.id} ({patient.cell}): T_r {patient.resection_time:.1f}, rho T_r {patient.rho * patient.resection_time:.2f}, "
            f"seed peak {peak:.3f}, sigma_0 {','.join(f'{s:g}' for s in sigmas)} mm, T_r in [{tr_bounds[0]:g}, {tr_bounds[1]:g}], maxfev {maxfev}",
            flush=True,
        )
        record = profile_patient(cohort, patient, out_dir, maxfev, tr_bounds, peak, sigmas)
        print(f"profile {patient.id}: {record['wall_time_s'] / 60:.1f} min ({index + 1} done)", flush=True)
    print(f"profile: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    if not getattr(args, "no_assemble", False):
        assemble_profile(root)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="parent of the design directory")
    parser.add_argument("--name", required=True, help="design directory name")
    parser.add_argument("--smoke", action="store_true", help=f"smoke settings: resolution_factor {SMOKE.resolution_factor:g}, {SMOKE.n_patients} patients, K = {SMOKE.draws}, maxfev {SMOKE.maxfev}, CPU")


def _add_device_arg(parser: argparse.ArgumentParser, single: bool = False) -> None:
    if single:
        parser.add_argument("--gpus", default=DEFAULT_GPUS.split(",")[0], help="the CUDA device id of this process; '' = CPU")
    else:
        parser.add_argument(
            "--gpus",
            default=DEFAULT_GPUS,
            help="comma-separated CUDA device ids ('' = CPU; ',' = two CPU workers): one device solves in this process, several dispatch one worker per device",
        )


def _add_worker_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(NO_ASSEMBLE_FLAG, dest="no_assemble", action="store_true", help="skip the CSV and figure assembly at the end (the dispatcher's workers)")


def _add_design_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(sa.DEFAULT_CONFIG), help="base config JSON (resection_cavity and rt_dose null)")
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE), help="search-space JSON (growth_efolds a script factor)")
    for key, path in sa.DEFAULT_TISSUE_MAPS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=str(path), help=f"NIfTI replacing the base config's {key} ('' keeps the base config's)")
    parser.add_argument(
        "--seed-voxel",
        default=None,
        help=f"the seed voxel i,j,k of every patient (snapped to the nearest seedable voxel); default {','.join(str(v) for v in DEFAULT_SEED_VOXEL)}, the base config's fractions on a grid too small for it",
    )
    parser.add_argument("--log2-candidates", type=int, default=DEFAULT_LOG2_CANDIDATES, help="2 ** it Sobol' points to fill the cells from")
    parser.add_argument("--design-seed", type=int, default=DESIGN_SEED, help="seed of the scrambled Sobol' sequence")
    parser.add_argument("--cavity-threshold", type=float, default=sa.CAVITY_THRESHOLD, help="cell density at or above which a voxel at resection_time is resected")
    parser.add_argument("--rt-margin-mm", type=float, default=sa.RT_MARGIN_MM, help="margin around the cavity of the dose region, mm")
    parser.add_argument("--rt-dose-per-fraction", type=float, default=sa.RT_DOSE_PER_FRACTION_GY, help="dose per fraction in Gy")
    parser.add_argument("--tr-max", type=float, default=DEFAULT_TR_MAX, help=f"candidates with resection_time above it (days) are rejected; also caps the substitute arm's T_0 (default {DEFAULT_TR_MAX:g}; below {TR_MIN:g} d is rejected too)")
    parser.add_argument("--patients-per-cell", type=int, default=PATIENTS_PER_CELL, help=f"patients accepted per cell (default {PATIENTS_PER_CELL}; {SMOKE.patients_per_cell} with --smoke)")
    for name, attribute in BAND_ARGS.items():
        lo, hi = getattr(DEFAULT_BANDS, name)
        parser.add_argument(
            f"--{attribute.replace('_', '-')}",
            default=None,
            help=f"size screening: accepted range lo,hi of {name} (PROVISIONAL default {lo:g},{hi:g}, to be replaced from BraTS)",
        )


def _add_invariance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lambdas", default=None, help=f"comma-separated lambda values (must contain 1); default {','.join(f'{v:g}' for v in INVARIANCE_LAMBDAS)}")
    parser.add_argument("--white-matter-diffusivity", type=float, default=None, help="D of the invariance patient (default the base config's)")
    parser.add_argument("--rho", type=float, default=None, help="rho of the invariance patient (default the base config's)")
    parser.add_argument("--resection-time", type=float, default=None, help="T_r of the invariance patient (default the base config's)")


def _add_patient_args(parser: argparse.ArgumentParser, profile: bool = False) -> None:
    default = "per cell the fd_check patient and the next one (the fd_check patients alone with --smoke)" if profile else "all, or the first of each cell with --smoke"
    parser.add_argument("--patients", default=None, help=f"comma-separated patient ids or ranges (p03, p00-p07); default {default}")


def _add_fisher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draws", type=int, default=None, help=f"noise draws K (default {NOISE_DRAWS}, {SMOKE.draws} with --smoke)")


def _add_dt_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dt-mode",
        default=DEFAULT_DT_MODE,
        choices=DT_MODES,
        help=f"the time step of every patient's solves: fixed ({BASE_STEPS_PER_DAY} steps/day) or stability (per patient, see the docstring); default {DEFAULT_DT_MODE}; "
        "the design records it and the other subcommands must name the same mode",
    )


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--maxfev", type=int, default=None, help=f"evaluations per fit (default {SUBSTITUTE_MAXFEV} for substitute, {MAXFEV} for seedfix and profile, {SMOKE.maxfev} with --smoke)"
    )


def _add_substitute_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--delta-a", default=None, help=f"comma-separated deficits delta_a in e-folds (T_0 = T_r - delta_a / rho); default {','.join(f'{v:g}' for v in SUBSTITUTE_DELTA_A)}")


def _add_seed_peak_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed-peak", type=float, default=None, help=f"the fixed seed's peak density (default {DEFAULT_SEED_PEAK:g})")


def _add_tr_bounds_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tr-bounds", default=None, help=f"the fitted growth time's range lo,hi in days; default {TR_BOUNDS[0]:g},{TR_BOUNDS[1]:g}")


def _add_seedfix_args(parser: argparse.ArgumentParser) -> None:
    _add_tr_bounds_arg(parser)
    _add_seed_peak_arg(parser)
    parser.add_argument("--seed-sigma-mm", type=float, default=None, help=f"the fixed seed's width sigma in mm (default {DEFAULT_SEED_SIGMA_MM:g})")
    parser.add_argument("--lambda-mode", default=None, choices=(*LAMBDA_MODES, "both"), help=f"fit T_r with lambda at the truth (fixed), with lambda fitted too (free), or both (default {DEFAULT_LAMBDA_MODE})")


def _add_profile_args(parser: argparse.ArgumentParser, shared: bool = False) -> None:
    if not shared:
        _add_tr_bounds_arg(parser)
        _add_seed_peak_arg(parser)
    parser.add_argument("--sigmas", default=None, help=f"comma-separated fixed seed widths sigma_0 in mm; default {','.join(f'{v:g}' for v in PROFILE_SIGMAS)} ({','.join(f'{v:g}' for v in SMOKE.sigmas)} with --smoke)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    design = commands.add_parser("design", help="draw and screen the cohort and write the design directory")
    _add_common_args(design)
    _add_device_arg(design, single=True)
    _add_design_args(design)
    _add_dt_mode_arg(design)
    invariance = commands.add_parser("invariance", help="experiment 0")
    _add_common_args(invariance)
    _add_device_arg(invariance, single=True)
    _add_invariance_args(invariance)
    fisher = commands.add_parser("fisher", help="experiment 1")
    _add_common_args(fisher)
    _add_device_arg(fisher)
    _add_patient_args(fisher)
    _add_fisher_args(fisher)
    _add_worker_arg(fisher)
    substitute = commands.add_parser("substitute", help="experiment 2")
    _add_common_args(substitute)
    _add_device_arg(substitute)
    _add_patient_args(substitute)
    _add_dt_mode_arg(substitute)
    _add_fit_args(substitute)
    _add_substitute_args(substitute)
    _add_worker_arg(substitute)
    seedfix = commands.add_parser("seedfix", help="experiment 3")
    _add_common_args(seedfix)
    _add_device_arg(seedfix)
    _add_patient_args(seedfix)
    _add_dt_mode_arg(seedfix)
    _add_fit_args(seedfix)
    _add_seedfix_args(seedfix)
    _add_worker_arg(seedfix)
    profile = commands.add_parser("profile", help="experiment 4")
    _add_common_args(profile)
    _add_device_arg(profile)
    _add_patient_args(profile, profile=True)
    _add_dt_mode_arg(profile)
    _add_fit_args(profile)
    _add_profile_args(profile)
    _add_worker_arg(profile)
    dt_check = commands.add_parser("dt-check", help="the truth of each patient at the fixed and at the stability step, compared")
    _add_common_args(dt_check)
    _add_device_arg(dt_check, single=True)
    dt_check.add_argument("--patients", default=None, help="comma-separated patient ids or ranges (p03, p00-p07); default the fd_check patients")
    everything = commands.add_parser("all", help="design (one device), then substitute, seedfix and profile")
    _add_common_args(everything)
    _add_device_arg(everything)
    _add_design_args(everything)
    _add_dt_mode_arg(everything)
    _add_patient_args(everything)
    _add_fit_args(everything)
    _add_substitute_args(everything)
    _add_seedfix_args(everything)
    _add_profile_args(everything, shared=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_dir) / args.name
    devices = parse_devices(args.gpus)
    if args.command == "design":
        if len(devices) > 1:
            raise ValueError(f"design screens on one device; give --gpus one entry, not {args.gpus!r}.")
        device = "" if args.smoke else devices[0]
        configure_device(device)
        design_command(args, device)
        return 0
    smoke = bool(args.smoke) or ((root / "spec.json").is_file() and bool(sa.read_json(root / "spec.json").get("smoke", False)))
    if smoke:
        devices = ["" for _ in devices]  # the CPU, as many workers as devices given
    if args.command in ("invariance", "dt-check") and len(devices) > 1:
        raise ValueError(f"{args.command} runs on one device; give --gpus one entry, not {args.gpus!r}.")
    configure_device(devices[0])
    if args.command == "all" and not (root / "spec.json").is_file():
        design_command(args, devices[0])
    elif args.command == "all":
        print(f"design directory {root} exists, kept")
    if not (root / "spec.json").is_file():
        raise FileNotFoundError(f"{root / 'spec.json'} not found; run the design first.")
    if args.command == "invariance":
        invariance_command(root, args)
    if args.command == "dt-check":
        dt_check_command(root, args, smoke)
    if args.command == "fisher":
        fisher_command(root, args, smoke, devices)
    if args.command in ("substitute", "all"):
        substitute_command(root, args, smoke, devices)
    if args.command in ("seedfix", "all"):
        seedfix_command(root, args, smoke, devices)
    if args.command in ("profile", "all"):
        profile_command(root, args, smoke, devices)
    return 0


if __name__ == "__main__":
    sys.exit(main())
