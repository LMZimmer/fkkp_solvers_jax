#!/usr/bin/env python
"""Forward-solve the GliODIL single-patient config with the JAX FKPP solver.

Mirrors the GliODIL run captured in reference_solves/reference.log (produced
by /home/home/lucas/projects/dockerize/GliODIL/forward_solve_single.py): same
D, rho, ratio, time horizon, timestep count, seed position, tissue maps and
full-volume grid, but runs fisher_kpp_jax.FKPPSolver instead of GliODIL's
synthetic_generator.py:

  1. fkpp_jax_f64_cpu — f64 on CPU,
  2. fkpp_jax_f64_gpu — f64 on GPU (skipped when no GPU is visible),
  3. fkpp_jax_f32_gpu — f32 on GPU (falls back to CPU, named _cpu then).

Only the plain FKPP solver is run: the GliODIL config does not define the
nutrient/necrosis parameters of FK_2c or a DTI tensor field, so the other two
solvers have no clean parameter mapping and are skipped.

Two parameter choices pin the run to the GliODIL configuration:

  - n_steps=4608: GliODIL's Nt (dt = 100/4608 days; smaller than the
    solver's own stability choice of ~843 steps, so still stable);
  - gaussian_seed_diffusion_time=15, gaussian_seed_mass=1500: GliODIL's
    gauss_sol3d seed ("experimentally chosen" there), much wider and heavier
    than the TumorGrowthToolkit default of Dt=5, M=250 (which yields ~44%
    less final mass over this run). The solver evaluates the seed on the
    device at the state dtype; differences to GliODIL's host float64 profile
    are at the ULP level.

Inputs: reference_solves/{gm,wm}_pbmap.nii.gz (the patient's tissue
probability maps). Outputs per run, GliODIL-style with identity affine:
result_<run>.nii.gz and segmentation_<run>.nii.gz, next to the GliODIL
reference pair result_reference.nii.gz / segmentation_reference.nii.gz.

Everything printed is also written to reference_solves/reference_solves.log
(the counterpart of reference.log): the config, available CPUs, wall-clock and
CPU time per solver, and the comparison table against the GliODIL reference.

Run from the project root:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/run_reference_solves.py
"""

from __future__ import annotations

import os
import resource
import sys
import time

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import jax  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

import fisher_kpp_jax as jax_pkg  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "reference_solves")
LOG_PATH = os.path.join(OUT_DIR, "reference_solves.log")
REFERENCE_RESULT = os.path.join(OUT_DIR, "result_reference.nii.gz")

# --- GliODIL forward_solve_single.py configuration, verbatim ---
# D, rho and the two thresholds are the optimized coefficients of an example
# GliODIL run (see that script for provenance).
DW = 0.927641
RHO = 0.091541
RATIO_DW_DG = 100
TH_UP = 0.554089
TH_DOWN = 0.303587
TH_NECRO = 0.8

DAYS = 100  # --days of the GliODIL run
NT = 4608  # --Nt of the GliODIL run, i.e. dt = 100/4608 days

# GliODIL's gauss_sol3d seed profile ("experimentally chosen" there).
GLIODIL_SEED_DIFFUSION_TIME = 15.0
GLIODIL_SEED_MASS = 1500.0

# Center of mass of the enhancing core (label 3) at voxel (140, 116, 55),
# offset by half a voxel so int(pct*N) truncation lands on it (both the
# generator and our solvers compute the seed voxel as int(fraction * N)).
SEED_X_FRACTION = (140 + 0.5) / 182
SEED_Y_FRACTION = (116 + 0.5) / 218
SEED_Z_FRACTION = (55 + 0.5) / 182

AFFINE = np.eye(4)  # the generator saves everything with an identity affine

CPU_DEVICE = jax.devices("cpu")[0]
try:
    GPU_DEVICE = jax.devices("gpu")[0]
except RuntimeError:
    GPU_DEVICE = None

_log_file = None


def emit(message: str = "") -> None:
    """Print and append to the run log (the counterpart of reference.log)."""
    print(message)
    if _log_file is not None:
        _log_file.write(message + "\n")
        _log_file.flush()


def cpu_seconds() -> float:
    """User + system CPU time of this process (all threads)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def load_tissue_maps() -> tuple[np.ndarray, np.ndarray]:
    """The patient's tissue probability maps from reference_solves/."""
    arrays = {}
    for tissue in ("gm", "wm"):
        path = os.path.join(OUT_DIR, f"{tissue}_pbmap.nii.gz")
        arrays[tissue] = np.asarray(nib.load(path).get_fdata(), dtype=np.float64)
    return arrays["gm"], arrays["wm"]


def segment(cell_density: np.ndarray) -> np.ndarray:
    """GliODIL's segment_BRATS_volume_cell_distribusion, verbatim."""
    seg = np.zeros_like(cell_density)
    seg[cell_density >= TH_UP] = 1
    seg[np.logical_and(cell_density < TH_UP, cell_density >= TH_DOWN)] = 3
    seg[cell_density >= TH_NECRO] = 4
    return seg


def rel_l2(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    max_abs = float(np.max(np.abs(ours - theirs)))
    denom = float(np.linalg.norm(theirs.ravel()))
    rel = float(np.linalg.norm((ours - theirs).ravel())) / denom if denom > 0 else max_abs
    return max_abs, rel


def save_outputs(name: str, cell_density: np.ndarray) -> None:
    result_path = os.path.join(OUT_DIR, f"result_{name}.nii.gz")
    segm_path = os.path.join(OUT_DIR, f"segmentation_{name}.nii.gz")
    nib.save(nib.Nifti1Image(cell_density, AFFINE), result_path)
    nib.save(nib.Nifti1Image(segment(cell_density), AFFINE), segm_path)
    emit(f"  saved {result_path}")
    emit(f"  saved {segm_path}")


def main() -> None:
    global _log_file
    os.makedirs(OUT_DIR, exist_ok=True)
    _log_file = open(LOG_PATH, "w")

    emit(f"jax {jax.__version__}, default backend: {jax.default_backend()}")
    if GPU_DEVICE is None:
        emit("No GPU visible: the f64 GPU run is skipped, f32 executes on CPU.")
    emit(
        f"CPUs: {len(os.sched_getaffinity(0))} available of {os.cpu_count()} "
        "on this host"
    )

    gm, wm = load_tissue_maps()
    emit(
        f"config: Dw={DW} rho={RHO} RatioDw_Dg={RATIO_DW_DG} days={DAYS} Nt={NT} "
        f"(dt={DAYS / NT:.6g} d) th_necro={TH_NECRO} th_up={TH_UP} th_down={TH_DOWN}"
    )
    emit(
        f"grid: {gm.shape}, seed voxel "
        f"({int(SEED_X_FRACTION * gm.shape[0])}, {int(SEED_Y_FRACTION * gm.shape[1])}, "
        f"{int(SEED_Z_FRACTION * gm.shape[2])}), GliODIL seed "
        f"(Dt={GLIODIL_SEED_DIFFUSION_TIME:g}, M={GLIODIL_SEED_MASS:g})"
    )

    base_params = {
        "white_matter_diffusivity": DW,
        "rho": RHO,
        "diffusivity_ratio": RATIO_DW_DG,
        "gray_matter_pbmap": gm,
        "white_matter_pbmap": wm,
        "gaussian_seed_x_fraction": SEED_X_FRACTION,
        "gaussian_seed_y_fraction": SEED_Y_FRACTION,
        "gaussian_seed_z_fraction": SEED_Z_FRACTION,
        "resolution_factor": 1.0,
        "stopping_time": DAYS,
        "n_steps": NT,
        "gaussian_seed_diffusion_time": GLIODIL_SEED_DIFFUSION_TIME,
        "gaussian_seed_mass": GLIODIL_SEED_MASS,
    }

    runs = [("fkpp_jax_f64_cpu", "f64", CPU_DEVICE)]
    if GPU_DEVICE is not None:
        runs.append(("fkpp_jax_f64_gpu", "f64", GPU_DEVICE))
        runs.append(("fkpp_jax_f32_gpu", "f32", GPU_DEVICE))
    else:
        runs.append(("fkpp_jax_f32_cpu", "f32", CPU_DEVICE))

    finals: dict[str, np.ndarray] = {}
    failures: list[str] = []
    for name, precision, device in runs:
        emit(f"\n== {name} ==")
        solver = jax_pkg.FKPPSolver({**base_params, "precision": precision})
        wall_start = time.perf_counter()
        cpu_start = cpu_seconds()
        with jax.default_device(device):
            result = solver.solve()
        wall = time.perf_counter() - wall_start
        cpu = cpu_seconds() - cpu_start
        if not result.success:
            failures.append(f"{name}: {result.error}")
            emit(f"  FAILED after {wall / 60:.1f} min: {result.error}")
            continue
        cell_density = np.asarray(result.final_state["cell_density"], dtype=np.float64)
        finals[name] = cell_density
        emit(f"  wall time: {wall / 60:.1f} min")
        emit(f"  CPU time:  {cpu / 60:.1f} min -> {cpu / wall:.2f} cores busy on average")
        emit(
            f"  final_time={result.final_time:g}, "
            f"criterion={result.stopping_criterion}, "
            f"final mass={result.final_stopping_quantity:.8g}, "
            f"max density={cell_density.max():.6g}"
        )
        save_outputs(name, cell_density)

    reference = None
    reference_label = None
    if os.path.exists(REFERENCE_RESULT):
        reference = np.asarray(
            nib.load(REFERENCE_RESULT).get_fdata(), dtype=np.float64
        )
        reference_label = os.path.basename(REFERENCE_RESULT)
    elif "fkpp_jax_f64_cpu" in finals:
        reference = finals["fkpp_jax_f64_cpu"]
        reference_label = f"fkpp_jax_f64_cpu (no {os.path.basename(REFERENCE_RESULT)} found)"

    if reference is not None:
        emit(f"\n== comparison vs {reference_label} ==")
        for name, field in finals.items():
            if field is reference:
                continue
            max_abs, rel = rel_l2(field, reference)
            emit(f"  {name:20s} max|d|={max_abs:.3e}  relL2={rel:.3e}")

    emit()
    if failures:
        emit("FAILED runs:")
        for failure in failures:
            emit(f" - {failure}")
        _log_file.close()
        sys.exit(1)
    emit("done.")
    _log_file.close()


if __name__ == "__main__":
    main()
