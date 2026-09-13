#!/usr/bin/env python
"""Three identifiability experiments on the pre-resection growth time of
the Stupp-protocol forward model, fisher_kpp_jax.StuppFKPPSolver, on
atlas tissue maps.

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
asks, locally at each of 32 truths, whether the growth time T_r has an
effect on the observations (smoothed threshold maps of the density at
five days) that the other parameters (v, lambda_f, alpha, k_ct, and the
seed's peak and width) cannot reproduce: the Fisher information of the
observations with a noise model of threshold and registration
uncertainty, its weakest direction and the Cramer-Rao error of log T_r
per set of observed days. Experiment 2 asks, globally, whether a seed of
a different mass and width grown for a fixed, wrong time T_0 can
reproduce the state the truth reaches at T_r, and how the treated frames
of that substitute then differ from the truth's: the linear composition
rule says it can when the growth is still linear (the seed grown for
T_r - T_0 days is again a Gaussian), the logistic saturation and the
tissue boundaries say it cannot for old tumors.

Conventions. A subcommand solves in one Python process (the jitted
time scan is cached across solves of the same step count) at precision
f64; ``--gpus`` names the CUDA devices like the sensitivity script's
slots ('' the CPU): with one device the subcommand solves in this
process, with several the fisher and substitute subcommands dispatch
one worker process per device (see Devices below). The parameters of the growth band are the front speed
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
must set a time step; the design sets its precision to f64, its
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
go into every config. The time step is the
base config's 12 steps/day (dt = 1/12 day; the solver raises a coarser
request to its stability estimate, which never binds in the growth band
on the atlas but does for short horizons on small test grids: every run
checks that the solver kept the intended dt and refuses to continue
otherwise). The schedule of every run is the base config's truncated to
the run's horizon (``truncate_schedule``) and shifted by
resection_time - 100 (``SHIFTED_TIME_KEYS``), so the treatment keeps its
offsets after surgery; the treated stage is stepped on the growth
stage's grid of times (``align_treated_config``: n_steps = n_growth +
ceil(time_after_resection / dt), resection_time stated as
(n_growth - 1/2) dt so that step n_growth resects). Every run's config
is saved (config.json in its directory: the aligned config with the
maps' paths and the frames' days as snapshot_times) so that
StuppFKPPSolver(read_config(path)) reproduces it; the solves themselves
take the loaded tissue arrays in place of the paths, which gives the
same fields and skips reading the NIfTIs. Fields are saved as float32
NIfTI rounded for storage (``round_field``: 7 mantissa bits, values
below 1e-10 zeroed) with the atlas affine; every Jacobian, objective and
metric is computed from the unrounded fields in memory. Runs are
resumable: a patient whose result JSON exists (runs/fisher/<patient>/
fisher.json, runs/substitute/<patient>/substitute.json; within
experiment 2 also a (T_0, objective) whose row.json exists), an
invariance experiment whose invariance.json exists, are skipped, and the
CSVs and figures are assembled from every record present. Nothing is
written outside <output-dir>/<name>/.

Frames. A treated run records the state at named moments after the
resection (FRAME_MOMENTS; ``frame_days``): each frame is the state after
the last step whose end lies at least half a step before the moment,
the sensitivity script's rounding (``snapshot_days`` takes the moment as
resection_time + offset + 1, so it receives moment - 1).
  pre    the last step end before resection_time, i.e. after step
         n_growth - 1 at (n_growth - 1) dt, before the resection step
         zeroes the cavity (the pre-resection state; it equals the
         growth stage's state one step before the density the cavity is
         thresholded from); tests/test_identifiability_experiments.py
         checks that it is nonzero inside the cavity and that the state
         one step later is zero there
  d34    the end of the Sunday closing the third CRT week (offset 34,
         moment 35: before Monday's fraction), as the script's mid_crt
  d55    the end of the Sunday closing the sixth CRT week (offset 55,
         moment 56), the script's end_crt; the design checks both
         against ``crt_snapshot_offsets`` of the base schedule
  d80    the end of day 80, the day before the first adjuvant dose at
         offset 81 (the design checks the base schedule), moment 81
  d120   the moment resection_time + 120, the horizon of experiment 1
         (1-3 h before it at 12 steps/day; no dose is scheduled within a
         day of it)
  d180   the moment resection_time + 180, the horizon of experiment 2
Experiment 1 records pre, d34, d55, d80, d120; experiment 2 also d180.
Each frame is saved as <name>_cell_density.nii.gz and frames.json holds
the requested and the recorded days.

Metrics (``compare_fields``, one function for the three experiments),
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
  ref_mass, ref_mass_out_of_field
                               the same two of u*
  qoi_<name>                   the script's QoIs of u (``compute_qois``:
                               mass, log10_mass, V_core, V_edema,
                               log10_V_core, log10_V_edema, r95_core,
                               r95_edema, centroid_drift, centroid_x/y/z_mm,
                               R_g, anisotropy, log10_anisotropy,
                               wm_fraction, n_core, n_edema, voxel_volume)
                               with the cohort's seed voxel and the atlas
                               white-matter map

Subcommand design (the cohort). The sampled factors are those of the
shipped search space (--search-space, fisher_kpp_jax/search_spaces/
stupp_fkpp_search_space.json) that vary here: the growth group
(front_speed_mm_per_day 0.03-0.25 mm/day, front_width_mm 1-5 mm, both
log-uniform), resection_time (30-200 days), chemo_kill_rate
(5e-4-1e-2 /day per mg/m^2, log), rt_alpha (0.01-0.2 /Gy, log) and the
seed group (seed_peak_density 0.6-1, seed_relative_width 2-4 log, the
width sigma = seed_relative_width front_width_mm in mm); diffusivity_ratio,
rt_alpha_beta_ratio and chemo_decay_rate are fixed at the base config
(10, 8 Gy, 9.24 /day) and the seed position is the fixed voxel. A
scrambled Sobol' sequence (scipy.stats.qmc.Sobol, seed --design-seed,
default 1) gives 2 ** --log2-candidates points (default 1024) on the unit
cube of the seven factors in the search space's order, transformed with
its ranges (``transform_factor``) and derived with its groups
(``SearchSpace.derive``: white_matter_diffusivity, rho, gaussian_seed_mass,
gaussian_seed_diffusion_time, seed_sigma_mm, seed_enhancing_radius_mm).
Per candidate the age rho T_r and the total log kill
  Lambda = n d alpha (1 + d / (alpha/beta)) + k_ct D_tot / gamma
         = 60 alpha (1 + 2 / 8) + k_ct 4900 / 9.24
with n = 30 fractions of d = 2 Gy, D_tot the chemotherapy dose within the
120-day horizon of experiment 1 (4 900 mg/m^2: 42 x 75 + 5 x 150 +
5 x 200) and gamma = chemo_decay_rate, and the first 8 candidates (in
sequence order) of each of the four cells
  young_weak    rho T_r < 2, Lambda < 4
  young_strong  rho T_r < 2, Lambda >= 4
  old_weak      rho T_r >= 2, Lambda < 4
  old_strong    rho T_r >= 2, Lambda >= 4
are kept: 32 patients p00-p31 in that cell order (with the default seed
each cell has 150-400 candidates among the 1024 points). The first
patient of each cell is flagged for the finite-difference check of
experiment 1 (fd_check).
  design.csv    patient, cell, candidate (the Sobol' index), u_<factor>
                for the seven factors, the seven factors
                (front_speed_mm_per_day, front_width_mm, resection_time,
                chemo_kill_rate, rt_alpha, seed_peak_density,
                seed_relative_width), white_matter_diffusivity, rho,
                gaussian_seed_mass, gaussian_seed_diffusion_time,
                seed_sigma_mm, seed_enhancing_radius_mm, rho_T_r,
                log_kill_rt, log_kill_ct, log_kill_total, fd_check
  spec.json     the settings: base config and search space paths, the
                tissue maps, grid shape, precision, gaussian_seed_floor,
                resolution_factor, smoke and the smoke settings, the
                design seed and candidate count, the cells and splits,
                the factors' ranges, the fixed parameters, the seed
                voxel (seed_target_voxel and seed_target_source: "default"
                for DEFAULT_SEED_VOXEL, "base_config_fractions" for the
                fallback, "argument" for --seed-voxel; the snapped
                seed_voxel, seed_snap_distance_voxels, seed_fractions,
                default_seed_voxel), the tissue
                threshold and seedable count, the time step, the
                treatment derivation settings, the schedules within both
                horizons (``truncate_schedule`` records), the log-kill
                formula and its constants, the frame moments, the CRT
                snapshot offsets, the horizons and frames per experiment,
                the lambda set, the experiment 1 and 2 settings
  base_config.json, search_space.json
  configs/<patient>.json   the truth config of experiment 1 (120-day
                horizon, maps null, schedule truncated and shifted)

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
  1. The truth: the growth stage at 12 steps/day (runs/fisher/<patient>/
     truth/growth/), its density at resection_time
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
frames; then for T_0 in --t0 (default 30, 60, 100, 150 days) and each
objective, with v and lambda at the truth, the seed position fixed and
the floor 0, the seed (logit of the peak on (0.05, 1], log sigma; peak =
0.05 + 0.95 sigmoid(z)) is fitted by scipy's Nelder-Mead (--maxfev
evaluations at most, default 150, 20 with --smoke; the initial simplex
x0, x0 + (0.5, 0), x0 + (0, 0.2); xatol 1e-3, fatol 1e-6) so that the
growth-only field at T_0 (FKPPSolver, 12 steps/day) matches the truth's
density at resection_time on the region's voxels (``fit_seed``):
  A   the relative L2 of log(u + 1e-6) - log(u* + 1e-6) on the region
  B   1 - the mean over c in {0.6, 0.3} of the soft Dice
      2 sum s s* / (sum s^2 + sum s*^2) of the smoothed indicators
      (1 for identical fields, the Dice for binary ones)
The initial guess is the linear composition rule (``composition_seed``):
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
  substitute.csv   one row per patient x T_0 x objective, plus one row
                   per patient with objective "truth" (T_0 = T_r, the
                   truth against itself, for the reference masses):
                   patient, cell, T_0, rho_T_0, rho_T_r, objective,
                   rule_peak, rule_sigma_mm, initial_peak,
                   initial_sigma_mm, initial_from_rule, fitted_peak,
                   fitted_sigma_mm, n_evaluations, objective_initial,
                   objective_achieved, growth_n_steps, dt, wall_time_s
                   (the fit), and <frame>_<metric> for the six frames
  figures/substitute_<frame>_objective_<A|B>.{png,pdf}   for d120 and
                   d180: every comparison metric against rho T_0 (log
                   axis), the (patient, T_0) points coloured by cell and
                   one line per cell through the cell's median at each
                   T_0 (at the cell's median rho T_0)
  runs/substitute/<patient>/   truth/ (as in experiment 1, six frames),
                   observation.json, T0_<T_0>/<objective>/ (fit.json,
                   row.json, run/ with the substitute's treated run),
                   substitute.json
  substitute_summary.json   the assembly record, as fisher_summary.json

Subcommand all runs design (skipped when spec.json exists), invariance,
fisher and substitute in order. --patients restricts fisher and
substitute to ids or ranges (p03, p00-p07, all).

Devices. --gpus is a comma-separated list of CUDA device ids, '' the
CPU (the default is the sensitivity script's slots, 1,2,3,6; ','
names two CPU workers). The invariance experiment runs on one device
(the first of the list under all; the invariance subcommand refuses
more). With one device, or one selected patient, fisher and substitute
solve in this process. With several devices they dispatch
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
resolution_factor 0.25, the first patient of each cell (4 patients),
K = 8, maxfev 20 and the CPU, so that the whole pipeline finishes in
minutes; the design records it (spec.json: smoke) and the later
subcommands follow the record. The tests use the same settings on a
phantom.

Output layout (--output-dir, default DEFAULT_OUTPUT_DIR
/mnt/Drive4/lucas/stupp_identifiability; --name required):
  <output-dir>/<name>/
    spec.json, base_config.json, search_space.json, design.csv
    configs/<patient>.json
    runs/invariance/, runs/fisher/<patient>/, runs/substitute/<patient>/
    invariance.csv, fisher.csv, fisher_regions.csv, fisher_fd_check.csv,
    fisher_runs.csv, substitute.csv
    figures/

Cost (an estimate). The sensitivity script's fit for a two-stage run in
its own process is 6.6 s + 1.4 ms per time step on one Quadro RTX 8000
(2026-09-07). In one process on the same class of card (GPU 2,
2026-09-13) a growth-only atlas solve took 0.5 s + 1.4 ms per step
(1 200 steps: 2.2 s) and a treated run recording five frames 3.5 s +
2.2 ms per step (1 908 steps: 7.7 s; the treatment terms, the six
frame upsamplings and the maps' downsampling are the extra). A new step
count compiles the scan again (1-3 s). With T_r the resection time in
days, a growth stage is 12 T_r steps and a treated run 12 (T_r + 120)
steps (experiment 1) or 12 (T_r + 180) (experiment 2).
  invariance   5 lambdas x (2 growth + 1 treated) solves: 1-2 min.
  fisher       per patient the truth (2 solves), 16 perturbed treated
               runs (10 more for the four fd_check patients), the noise
               (K = 64: about 2 min on the CPU side), the metrics of 80
               (130) frames and the analysis (least squares on 4 million
               rows): measured 7.9 min for p00 (T_r = 39 days, with the
               fd check; 204 s of solves, 115 s of noise); at T_r = 200
               a run is 3 840 steps and 12 s, so 5-12 min per patient,
               about 4-5 h for the 32 patients on one GPU; about 500 MB
               per fd_check patient (W.npz 116 MB, 29 run directories of
               about 12 MB), about 16 GB in total.
  substitute   per patient the truth and 8 fits of up to 150 growth-only
               solves of 12 T_0 steps (measured 0.85 s per evaluation at
               T_0 = 30, 3.1 s at T_0 = 150; a fit that converges earlier
               stops earlier) plus 8 treated runs: measured 34.7 min for
               p16 (T_r = 94 days), so about 35 min per patient and
               about 19 h for the cohort on one GPU; --gpus 1,2,3,6
               dispatches 8 patients to each of four devices (5 h). About
               33 MB per patient.
  --smoke      the whole pipeline on the CPU (K = 8, maxfev 20, 4 mm
               voxels): measured 30 min on 2026-09-13, the invariance in
               1 min, the four fisher patients in 15 min (3.4-4.5 min
               each; the 4 mm solves take 2.6 s, the noise and the
               metrics on the 1 mm grid the rest), the four substitute
               patients in 13 min (3.2-3.4 min each); 1.8 GB.

Run from the project root, e.g. (ID = /mnt/Drive4/lucas/stupp_identifiability):
  python scripts/identifiability_experiments.py design --name id_2026-09-13
  python scripts/identifiability_experiments.py invariance --name id_2026-09-13 --gpus 1
  python scripts/identifiability_experiments.py fisher --name id_2026-09-13 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py substitute --name id_2026-09-13 --gpus 1,2,3,6
  python scripts/identifiability_experiments.py all --name id_2026-09-13 --gpus 1,2,3,6
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
from collections.abc import Mapping, Sequence
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
from scipy.optimize import minimize  # noqa: E402
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

EXPERIMENTS: tuple[str, ...] = ("design", "invariance", "fisher", "substitute")
DEFAULT_OUTPUT_DIR = Path("/mnt/Drive4/lucas/stupp_identifiability")
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
PATIENTS_PER_CELL = 8
AGE_SPLIT = 2.0  # rho T_r below / above
KILL_SPLIT = 4.0  # Lambda below / above
# The sampled factors of the shipped search space (its order); everything
# else is fixed: diffusivity_ratio, rt_alpha_beta_ratio and
# chemo_decay_rate at the base config, the seed position at one voxel.
SAMPLED_FACTORS: tuple[str, ...] = (
    sa.GROWTH_SPEED_FACTOR,
    sa.GROWTH_WIDTH_FACTOR,
    "resection_time",
    "chemo_kill_rate",
    "rt_alpha",
    sa.SEED_PEAK_FACTOR,
    sa.SEED_RELATIVE_WIDTH_FACTOR,
)
FIXED_AT_BASE: tuple[str, ...] = ("diffusivity_ratio", "rt_alpha_beta_ratio", "chemo_decay_rate")
CELLS: dict[str, tuple[bool, bool]] = {  # cell name -> (old: rho T_r >= AGE_SPLIT, strong: Lambda >= KILL_SPLIT)
    "young_weak": (False, False),
    "young_strong": (False, True),
    "old_weak": (True, False),
    "old_strong": (True, True),
}
DESIGN_DERIVED: tuple[str, ...] = (
    "white_matter_diffusivity",
    "rho",
    "gaussian_seed_mass",
    "gaussian_seed_diffusion_time",
    sa.SEED_SIGMA_COLUMN,
    sa.SEED_RADIUS_COLUMN,
)
DESIGN_COLUMNS: list[str] = [
    "patient",
    "cell",
    "candidate",
    *(f"u_{name}" for name in SAMPLED_FACTORS),
    *SAMPLED_FACTORS,
    *DESIGN_DERIVED,
    "rho_T_r",
    "log_kill_rt",
    "log_kill_ct",
    "log_kill_total",
    "fd_check",
]

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

# Experiment 2.
SUBSTITUTE_T0: tuple[float, ...] = (30.0, 60.0, 100.0, 150.0)
MAXFEV = 150
PEAK_MIN = 0.05  # the fitted peak lies in (PEAK_MIN, PEAK_MAX]
PEAK_MAX = 1.0
LOGIT_CLIP = 1e-4  # the inverse map keeps the unit-interval argument in [LOGIT_CLIP, 1 - LOGIT_CLIP]
SIMPLEX_STEPS: tuple[float, float] = (0.5, 0.2)  # the initial simplex: x0, x0 + (0.5, 0), x0 + (0, 0.2) in (logit peak, log sigma)
NELDER_MEAD_XATOL = 1e-3
NELDER_MEAD_FATOL = 1e-6
OBJECTIVES: tuple[str, ...] = ("A", "B")
LOG_EPS = 1e-6  # log(u + LOG_EPS) in the log metrics and objective A

# Metrics (``compare_fields``).
METRIC_NAMES: list[str] = [
    "dice_core",
    "dice_edema",
    "assd_core_mm",
    "assd_edema_mm",
    "rel_l2_log",
    "rel_l2",
    "max_abs_diff",
    "mass",
    "mass_out_of_field",
    "ref_mass",
    "ref_mass_out_of_field",
    *(f"qoi_{name}" for name in sa.QOI_NAMES),
]
# The metrics plotted against lambda / rho T_0 (the QoIs are plotted in a
# second figure).
PLOTTED_METRICS: list[str] = [name for name in METRIC_NAMES if not name.startswith("qoi_") and not name.startswith("ref_")]
PLOTTED_QOIS: list[str] = [f"qoi_{name}" for name in sa.FINAL_ANALYSED_QOIS]

COLOR_CELLS: dict[str, str] = {"young_weak": "#2a78d6", "young_strong": "#eb6834", "old_weak": "#2a9d8f", "old_strong": "#8a4fbf"}
COLOR_TEXT = "#0b0b0b"
COLOR_LINES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#2a9d8f", "#8a4fbf", "#c9a227", "#52514e", "#d64a8a", "#7a5c2e")


@dataclass(frozen=True)
class Smoke:
    """The --smoke settings: a coarse grid, few patients, few draws, few
    evaluations, the CPU."""

    resolution_factor: float = 0.25  # the solver's zoom factor: 4 mm voxels on the 1 mm atlas
    n_patients: int = 4  # the first patient of each cell
    draws: int = 8
    maxfev: int = 20


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

    def solver_values(self) -> dict[str, float]:
        """The StuppFKPPSolver parameters the patient sets."""
        return {
            **self.growth,
            **self.seed,
            "resection_time": float(self.resection_time),
            "rt_alpha": float(self.rt_alpha),
            "chemo_kill_rate": float(self.chemo_kill_rate),
        }

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
            seed_sigma=float(record[sa.SEED_SIGMA_COLUMN]),
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


def cell_of(age: float, log_kill: float) -> str:
    """The cell of a candidate: (rho T_r below / above AGE_SPLIT) x
    (Lambda below / above KILL_SPLIT)."""
    old, strong = age >= AGE_SPLIT, log_kill >= KILL_SPLIT
    for name, (is_old, is_strong) in CELLS.items():
        if is_old == old and is_strong == strong:
            return name
    raise AssertionError("unreachable")


def sample_cohort(
    space: Any,
    base: Mapping[str, Any],
    treatment: Mapping[str, float],
    chemo_total_dose: float,
    n_fractions: int,
    log2_candidates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    The design records: a scrambled Sobol' sequence over the sampled
    factors (unit cube, 2 ** log2_candidates points), transformed with the
    search space's ranges and derived with its groups, then the first
    PATIENTS_PER_CELL candidates of each cell in sequence order.

    Returns:
        One record per patient (``DESIGN_COLUMNS``), cells in CELLS' order,
        ids p00, p01, ...
    """
    missing = [name for name in SAMPLED_FACTORS if name not in space.factors]
    if missing:
        raise ValueError(f"the search space lacks the factors {missing}.")
    sampler = qmc.Sobol(d=len(SAMPLED_FACTORS), scramble=True, seed=int(seed))
    u = np.asarray(sampler.random_base2(int(log2_candidates)), dtype=np.float64)
    values = {name: space.factors[name].transform(u[:, column]) for column, name in enumerate(SAMPLED_FACTORS)}
    derived = space.derive(values)
    age = derived["rho"] * values["resection_time"]
    kill = total_log_kill(
        values["rt_alpha"],
        float(base["rt_alpha_beta_ratio"]),
        n_fractions,
        float(treatment["rt_dose_per_fraction_gy"]),
        values["chemo_kill_rate"],
        chemo_total_dose,
        float(base["chemo_decay_rate"]),
    )
    cells = [cell_of(float(age[i]), float(kill["log_kill_total"][i])) for i in range(len(u))]
    records: list[dict[str, Any]] = []
    for cell in CELLS:
        candidates = [i for i, name in enumerate(cells) if name == cell][:PATIENTS_PER_CELL]
        if len(candidates) < PATIENTS_PER_CELL:
            counts = {name: cells.count(name) for name in CELLS}
            raise ValueError(
                f"cell {cell} has {len(candidates)} candidates among {len(u)} Sobol' points (counts {counts}); "
                "raise --log2-candidates."
            )
        for position, i in enumerate(candidates):
            record: dict[str, Any] = {
                "patient": f"p{len(records):02d}",
                "cell": cell,
                "candidate": int(i),
                "fd_check": position == 0,
            }
            record.update({f"u_{name}": float(u[i, column]) for column, name in enumerate(SAMPLED_FACTORS)})
            record.update({name: float(values[name][i]) for name in SAMPLED_FACTORS})
            record.update({key: float(derived[key][i]) for key in DESIGN_DERIVED})
            record["rho_T_r"] = float(age[i])
            record.update({key: float(kill[key][i]) for key in ("log_kill_rt", "log_kill_ct", "log_kill_total")})
            records.append(record)
    return records


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
) -> Path:
    """
    Draw the cohort and write the design directory <output_dir>/<name>:
    base_config.json, search_space.json, spec.json, design.csv and
    configs/<patient>.json. Refuses to overwrite an existing directory.

    Args:
        config_path: The base config (resection_cavity and rt_dose null).
        search_space_path: The search space (the shipped one).
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

    Returns:
        The design directory.
    """
    root = Path(output_dir) / name
    if root.exists():
        raise FileExistsError(f"{root} exists; a design is never overwritten.")
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
    base["precision"] = PRECISION
    base["gaussian_seed_floor"] = SEED_FLOOR
    base["snapshot_times"] = None
    if smoke:
        base["resolution_factor"] = SMOKE.resolution_factor
    space = sa.load_search_space(search_space_path, StuppFKPPSolver.config_keys())
    scale = space.overrides.get("gaussian_seed_scale", base["gaussian_seed_scale"])
    if float(scale) != 1.0:
        raise ValueError(f"the seed derivation needs gaussian_seed_scale = 1, got {scale!r}.")
    base["gaussian_seed_scale"] = 1.0
    horizon_override = space.overrides.get("time_after_resection")
    if horizon_override is not None and float(horizon_override) != FISHER_HORIZON:
        raise ValueError(
            f"the search space fixes time_after_resection at {horizon_override!r}, the fisher horizon is {FISHER_HORIZON:g}."
        )
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
    records = sample_cohort(space, base, treatment, float(schedules["fisher"]["chemo_total_dose"]), n_fractions, log2_candidates, seed)
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
        "smoke_settings": {"resolution_factor": SMOKE.resolution_factor, "n_patients": SMOKE.n_patients, "draws": SMOKE.draws, "maxfev": SMOKE.maxfev},
        "design_seed": int(seed),
        "log2_candidates": int(log2_candidates),
        "n_candidates": 2 ** int(log2_candidates),
        "patients_per_cell": PATIENTS_PER_CELL,
        "cells": {name: {"old": old, "strong": strong} for name, (old, strong) in CELLS.items()},
        "age_split_rho_T_r": AGE_SPLIT,
        "kill_split_lambda": KILL_SPLIT,
        "sampled_factors": list(SAMPLED_FACTORS),
        "factors": {name: {"min": space.factors[name].low, "max": space.factors[name].high, "scale": space.factors[name].scale} for name in SAMPLED_FACTORS},
        "fixed_at_base": {key: base[key] for key in FIXED_AT_BASE},
        "seed_voxel": list(voxel),
        "seed_fractions": list(fractions),
        "seed_target_voxel": target,
        "seed_target_source": target_source,
        "default_seed_voxel": list(DEFAULT_SEED_VOXEL),
        "seed_snap_distance_voxels": snap_distance,
        "min_tissue_fraction": min_tissue_fraction,
        "n_seedable_voxels": geometry.n_voxels,
        "time_step": {key: base[key] for key in sa.TIME_STEP_KEYS},
        "treatment": {**treatment, "n_fractions": n_fractions, "rt_total_dose_gy": treatment["rt_dose_per_fraction_gy"] * n_fractions},
        "schedules": schedules,
        "log_kill": {
            "formula": "Lambda = n_fractions d alpha (1 + d / rt_alpha_beta_ratio) + chemo_kill_rate D_tot / chemo_decay_rate, D_tot the chemo dose sum within the fisher horizon",
            "chemo_total_dose": float(schedules["fisher"]["chemo_total_dose"]),
            "chemo_decay_rate": float(base["chemo_decay_rate"]),
            "rt_alpha_beta_ratio": float(base["rt_alpha_beta_ratio"]),
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
            "T0": list(SUBSTITUTE_T0),
            "maxfev": SMOKE.maxfev if smoke else MAXFEV,
            "peak_range": [PEAK_MIN, PEAK_MAX],
            "objectives": list(OBJECTIVES),
            "simplex_steps": list(SIMPLEX_STEPS),
        },
        "n_patients": len(records),
    }
    root.mkdir(parents=True, exist_ok=False)
    for sub in ("configs", "runs", "figures"):
        (root / sub).mkdir()
    shutil.copyfile(search_space_path, root / "search_space.json")
    write_config(base, root / "base_config.json")
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
    sa.write_json(run_dir / FRAMES_FILE, {"resection_time": float(config["resection_time"]), "dt": dt, "n_growth": n_growth, "frames": record})
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


def truth_run(cohort: Cohort, patient: Patient, horizon: float, frames: Sequence[str], run_dir: Path) -> TruthRun:
    """
    A patient's truth: the growth stage at the base config's time step
    (saved into run_dir/growth), its density at resection_time (saved as
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
    ||u - u*|| / ||u*||, the max abs difference, the total mass of u
    (dV sum u), its mass outside the dose map's support (dose == 0: the
    out-of-field tail), the same two of u*, and the script's QoIs of u
    (``compute_qois``, prefixed qoi_).
    """
    field = np.asarray(field, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    voxel_volume = float(np.prod(np.asarray(zooms, dtype=np.float64)))
    u, u_ref = field[tissue], reference[tissue]
    out_of_field = np.logical_and(tissue, np.asarray(dose) <= 0)
    out: dict[str, float] = {}
    for label, level in (("core", sa.TAU_CORE), ("edema", sa.TAU_EDEMA)):
        a = np.logical_and(field >= level, tissue)
        b = np.logical_and(reference >= level, tissue)
        out[f"dice_{label}"] = dice(a, b)
        out[f"assd_{label}_mm"] = surface_distance(a, b, zooms)
    out["rel_l2_log"] = rel_l2_log(u, u_ref)
    norm_ref = float(np.linalg.norm(u_ref))
    out["rel_l2"] = float(np.linalg.norm(u - u_ref) / norm_ref) if norm_ref > 0 else float("nan")
    out["max_abs_diff"] = float(np.max(np.abs(u - u_ref))) if u.size else 0.0
    out["mass"] = voxel_volume * float(u.sum())
    out["mass_out_of_field"] = voxel_volume * float(field[out_of_field].sum())
    out["ref_mass"] = voxel_volume * float(u_ref.sum())
    out["ref_mass_out_of_field"] = voxel_volume * float(reference[out_of_field].sum())
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


def peak_from_logit(z: float) -> float:
    """The peak density of its logit coordinate: PEAK_MIN + (PEAK_MAX - PEAK_MIN) sigmoid(z)."""
    return PEAK_MIN + (PEAK_MAX - PEAK_MIN) * float(expit(z))


def logit_from_peak(peak: float) -> float:
    """The inverse of ``peak_from_logit``, the unit-interval argument
    clipped to [LOGIT_CLIP, 1 - LOGIT_CLIP]."""
    p = float(np.clip((peak - PEAK_MIN) / (PEAK_MAX - PEAK_MIN), LOGIT_CLIP, 1.0 - LOGIT_CLIP))
    return float(np.log(p / (1.0 - p)))


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
    cohort's seed voxel, the base config's time step) to the truth's
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
        history=history,
        wall_time_s=time.perf_counter() - start,
    )


def substitute_row(cohort: Cohort, patient: Patient, t0: float, objective: str, fit: SeedFit | None, metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """A substitute.csv record: the patient, T_0, rho T_0, rho T_r, the fit
    and the metrics per frame as <frame>_<metric>."""
    row: dict[str, Any] = {
        "patient": patient.id,
        "cell": patient.cell,
        "T_0": float(t0),
        "rho_T_0": patient.rho * float(t0),
        "rho_T_r": patient.rho * patient.resection_time,
        "objective": objective,
    }
    if fit is not None:
        row.update({key: value for key, value in fit.record().items() if key not in ("history", "objective", "T_0")})
    else:
        row.update({"fitted_peak": patient.seed_peak, "fitted_sigma_mm": patient.seed_sigma})
    for frame, values in metrics.items():
        row.update({f"{frame}_{name}": value for name, value in values.items()})
    return row


SUBSTITUTE_COLUMNS: list[str] = [
    "patient",
    "cell",
    "T_0",
    "rho_T_0",
    "rho_T_r",
    "objective",
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
    "dt",
    "wall_time_s",
    *(f"{frame}_{name}" for frame in SUBSTITUTE_FRAMES for name in METRIC_NAMES),
]


def substitute_patient(cohort: Cohort, patient: Patient, out_dir: Path, maxfev: int, t0_values: Sequence[float] = SUBSTITUTE_T0) -> dict[str, Any]:
    """
    Experiment 2 for one patient: the truth over SUBSTITUTE_HORIZON with
    the frames SUBSTITUTE_FRAMES, then per T_0 and objective the fitted
    seed (``fit_seed``; T0_<T_0>/<objective>/fit.json), the treated run
    from it with resection_time T_0, the truth's maps, alpha and k_ct and
    the schedule shifted for T_0 (T0_<T_0>/<objective>/run/) and the
    metrics against the truth per frame; a (T_0, objective) whose row.json
    exists is reused. Writes substitute.json (the record returned).
    """
    patient_dir = out_dir / patient.id
    start = time.perf_counter()
    truth = truth_run(cohort, patient, SUBSTITUTE_HORIZON, SUBSTITUTE_FRAMES, patient_dir / "truth")
    region = observation_region(truth.run.frames, cohort.tissue, cohort.zooms)
    sa.write_json(patient_dir / "observation.json", {**region.record(), "margin_mm": OBSERVATION_MARGIN_MM})
    dose = truth.maps.dose
    rows: list[dict[str, Any]] = []
    truth_metrics = {name: compare_to(cohort, truth.run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
    rows.append(substitute_row(cohort, patient, patient.resection_time, "truth", None, truth_metrics))
    for t0 in t0_values:
        for objective in OBJECTIVES:
            fit_dir = patient_dir / f"T0_{int(round(t0)):03d}" / objective
            row_path = fit_dir / "row.json"
            if row_path.is_file():
                rows.append(read_record(row_path))
                print(f"  {patient.id} T_0={t0:g} {objective}: row exists, kept", flush=True)
                continue
            fit = fit_seed(cohort, patient, t0, truth.density, region, objective, maxfev)
            fit_dir.mkdir(parents=True, exist_ok=True)
            write_record(fit_dir / "fit.json", fit.record())
            substitute = replace(patient, seed_peak=fit.peak, seed_sigma=fit.sigma, resection_time=float(t0))
            config = patient_config(cohort, substitute, SUBSTITUTE_HORIZON)
            run = treated_run(cohort, config, truth.maps, fit.growth_n_steps, fit.dt, SUBSTITUTE_FRAMES, fit_dir / "run")
            metrics = {name: compare_to(cohort, run.frames[name], truth.run.frames[name], dose) for name in SUBSTITUTE_FRAMES}
            row = substitute_row(cohort, patient, t0, objective, fit, metrics)
            write_record(row_path, row)
            rows.append(row)
            print(
                f"  {patient.id} T_0={t0:g} {objective}: peak {fit.initial_peak:.3f} -> {fit.peak:.3f}, sigma {fit.initial_sigma:.2f} -> "
                f"{fit.sigma:.2f} mm, objective {fit.value_initial:.4f} -> {fit.value:.4f} in {fit.n_evaluations} evaluations "
                f"({fit.wall_time_s:.0f} s); d120 Dice {metrics['d120']['dice_edema']:.3f}",
                flush=True,
            )
    record = {
        "patient": patient.id,
        "cell": patient.cell,
        "resection_time": patient.resection_time,
        "rho": patient.rho,
        "n_growth": truth.n_growth,
        "dt": truth.dt,
        "cavity_volume_mm3": truth.maps.record["cavity_volume_mm3"],
        "dose_volume_mm3": truth.maps.record["dose_volume_mm3"],
        "observation": region.record(),
        "rows": rows,
        "wall_time_s": time.perf_counter() - start,
    }
    write_record(patient_dir / PATIENT_FILE.format(experiment="substitute"), record)
    return record


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


def assemble_substitute(root: Path, dispatch: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """substitute.csv, its figures and substitute_summary.json from the
    patient records present (``write_summary``)."""
    rows: list[dict[str, Any]] = []
    records = patient_records(root, "substitute")
    for record in records:
        rows.extend(record["rows"])
    sa.write_csv(root / "substitute.csv", rows, SUBSTITUTE_COLUMNS)
    if rows:
        substitute_figures(rows, root / "figures")
    write_summary(root, "substitute", records, len(rows), dispatch)
    return rows


def substitute_figures(rows: Sequence[Mapping[str, Any]], figure_dir: Path, frames: Sequence[str] = ("d120", "d180")) -> None:
    """Per frame and objective: every comparison metric against rho T_0,
    points per (patient, T_0) coloured by cell, one line per cell through
    the cell's median at each T_0 (at the cell's median rho T_0)."""
    figure_dir.mkdir(exist_ok=True)
    for frame in frames:
        for objective in OBJECTIVES:
            selected = [row for row in rows if row["objective"] == objective]
            if not selected:
                continue
            names = [name for name in PLOTTED_METRICS if not name.startswith("ref_")]
            n_cols = 4
            n_rows = int(np.ceil(len(names) / n_cols))
            figure, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.6 * n_rows), squeeze=False)
            ages = np.array([float(row["rho_T_0"]) for row in selected])
            for axis, name in zip(axes.flat, names):
                key = f"{frame}_{name}"
                for cell, color in COLOR_CELLS.items():
                    cell_rows = [row for row in selected if row["cell"] == cell]
                    if not cell_rows:
                        continue
                    x = np.array([float(row["rho_T_0"]) for row in cell_rows])
                    y = np.array([sa.as_float(row.get(key)) for row in cell_rows])
                    finite = np.isfinite(y)
                    axis.scatter(x[finite], y[finite], s=10, color=color, alpha=0.5, linewidths=0, label=cell)
                    t0_values = sorted(set(float(row["T_0"]) for row in cell_rows))
                    medians = []
                    for t0 in t0_values:
                        at = [row for row in cell_rows if float(row["T_0"]) == t0]
                        values = np.array([sa.as_float(row.get(key)) for row in at])
                        at_ages = np.array([float(row["rho_T_0"]) for row in at])
                        if np.isfinite(values).any():
                            medians.append((float(np.median(at_ages)), float(np.nanmedian(values))))
                    if medians:
                        axis.plot([m[0] for m in medians], [m[1] for m in medians], "-", color=color, linewidth=1.5)
                # Limits by hand: a panel whose values are all NaN (an empty
                # iso-surface for every substitute) has no data to scale.
                low, high = float(ages.min()) / 1.5, float(ages.max()) * 1.5
                axis.set_xlim(low, high)
                axis.set_xscale("log")
                ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if low <= t <= high]
                axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
                axis.set_xticks([], minor=True)
                axis.set_xlabel("rho T_0", fontsize=8)
                axis.set_title(f"{frame} {name}", fontsize=8)
                axis.tick_params(labelsize=7)
                axis.spines[["top", "right"]].set_visible(False)
            for axis in axes.flat[len(names):]:
                axis.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower right", fontsize=7, frameon=False, ncol=2)
            figure.suptitle(f"experiment 2, objective {objective}: substitute against truth at {frame}; lines: cell medians per T_0", fontsize=10)
            figure.tight_layout()
            _save_figure(figure, figure_dir / f"substitute_{frame}_objective_{objective}")


# --- commands ---


def design_command(args: argparse.Namespace) -> Path:
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
    )
    spec = sa.read_json(root / "spec.json")
    print(f"design directory: {root}")
    print(f"tissue maps: {spec['tissue_maps']}")
    print(
        f"seed voxel {spec['seed_voxel']} (target {spec['seed_target_voxel']}, snapped {spec['seed_snap_distance_voxels']:g} voxels); "
        f"floor {spec['gaussian_seed_floor']:g}, precision {spec['precision']}, resolution_factor {spec['resolution_factor']:g}"
        f"{' (smoke)' if spec['smoke'] else ''}"
    )
    records = sa.read_csv(root / "design.csv")
    for cell in CELLS:
        members = [r for r in records if r["cell"] == cell]
        ages = [float(r["rho_T_r"]) for r in members]
        kills = [float(r["log_kill_total"]) for r in members]
        print(f"  {cell}: {len(members)} patients, rho T_r {min(ages):.2f}-{max(ages):.2f}, Lambda {min(kills):.2f}-{max(kills):.2f}")
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
    other options passed through, the assembly skipped."""
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
    if experiment == "fisher" and getattr(args, "draws", None) is not None:
        command += ["--draws", str(args.draws)]
    if experiment == "substitute":
        if getattr(args, "maxfev", None) is not None:
            command += ["--maxfev", str(args.maxfev)]
        if getattr(args, "t0", None):
            command += ["--t0", str(args.t0)]
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
    patients = select_patients(cohort, args.patients, smoke)
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
        record = fisher_patient(cohort, patient, out_dir, draws, fd_flags.get(patient.id, False), int(cohort.spec["design_seed"]) * 1000 + cohort.patients.index(patient))
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


def substitute_command(root: Path, args: argparse.Namespace, smoke: bool, devices: Sequence[str]) -> None:
    cohort = load_cohort(root)
    patients = select_patients(cohort, args.patients, smoke)
    if min(len(devices), len(patients)) > 1:
        record = dispatch(root, "substitute", devices, patients, args)
        assemble_substitute(root, record)
        check_dispatch(record)
        return
    maxfev = int(args.maxfev) if args.maxfev is not None else (SMOKE.maxfev if smoke else MAXFEV)
    t0_values = [float(v) for v in args.t0.split(",")] if args.t0 else list(SUBSTITUTE_T0)
    out_dir = root / "runs" / "substitute"
    start = time.perf_counter()
    for index, patient in enumerate(patients):
        marker = out_dir / patient.id / PATIENT_FILE.format(experiment="substitute")
        if marker.is_file():
            print(f"substitute {patient.id}: {marker} exists, skipped", flush=True)
            continue
        print(f"substitute {patient.id} ({patient.cell}): T_r {patient.resection_time:.1f}, rho T_r {patient.rho * patient.resection_time:.2f}, maxfev {maxfev}", flush=True)
        record = substitute_patient(cohort, patient, out_dir, maxfev, t0_values)
        print(f"substitute {patient.id}: {record['wall_time_s'] / 60:.1f} min ({index + 1} done)", flush=True)
    print(f"substitute: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    if not getattr(args, "no_assemble", False):
        assemble_substitute(root)


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
    parser.add_argument("--search-space", default=str(sa.DEFAULT_SEARCH_SPACE), help="search-space JSON")
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


def _add_invariance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lambdas", default=None, help=f"comma-separated lambda values (must contain 1); default {','.join(f'{v:g}' for v in INVARIANCE_LAMBDAS)}")
    parser.add_argument("--white-matter-diffusivity", type=float, default=None, help="D of the invariance patient (default the base config's)")
    parser.add_argument("--rho", type=float, default=None, help="rho of the invariance patient (default the base config's)")
    parser.add_argument("--resection-time", type=float, default=None, help="T_r of the invariance patient (default the base config's)")


def _add_patient_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patients", default=None, help="comma-separated patient ids or ranges (p03, p00-p07); default all, or the first of each cell with --smoke")


def _add_fisher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draws", type=int, default=None, help=f"noise draws K (default {NOISE_DRAWS}, {SMOKE.draws} with --smoke)")


def _add_substitute_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--maxfev", type=int, default=None, help=f"Nelder-Mead evaluations per fit (default {MAXFEV}, {SMOKE.maxfev} with --smoke)")
    parser.add_argument("--t0", default=None, help=f"comma-separated T_0 values in days; default {','.join(f'{v:g}' for v in SUBSTITUTE_T0)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    design = commands.add_parser("design", help="draw the cohort and write the design directory")
    _add_common_args(design)
    _add_design_args(design)
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
    _add_substitute_args(substitute)
    _add_worker_arg(substitute)
    everything = commands.add_parser("all", help="design, invariance, fisher and substitute in order")
    _add_common_args(everything)
    _add_device_arg(everything)
    _add_design_args(everything)
    _add_invariance_args(everything)
    _add_patient_args(everything)
    _add_fisher_args(everything)
    _add_substitute_args(everything)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_dir) / args.name
    if args.command == "design":
        design_command(args)
        return 0
    if args.command == "all" and not (root / "spec.json").is_file():
        design_command(args)
    elif args.command == "all":
        print(f"design directory {root} exists, kept")
    if not (root / "spec.json").is_file():
        raise FileNotFoundError(f"{root / 'spec.json'} not found; run the design first.")
    smoke = bool(args.smoke) or bool(sa.read_json(root / "spec.json").get("smoke", False))
    devices = parse_devices(args.gpus)
    if smoke:
        devices = ["" for _ in devices]  # the CPU, as many workers as devices given
    if args.command == "invariance" and len(devices) > 1:
        raise ValueError(f"invariance runs on one device; give --gpus one entry, not {args.gpus!r}.")
    configure_device(devices[0])
    if args.command in ("invariance", "all"):
        invariance_command(root, args)
    if args.command in ("fisher", "all"):
        fisher_command(root, args, smoke, devices)
    if args.command in ("substitute", "all"):
        substitute_command(root, args, smoke, devices)
    return 0


if __name__ == "__main__":
    sys.exit(main())
