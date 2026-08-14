"""Validate fisher_kpp_jax against the NumPy reference on realistic sizes.

Mirrors scripts/compare_to_original.py in fisher-kpp, but compares the JAX
port against the ``fisher_kpp`` reference package on ~128^3 synthetic inputs:

  1. reference NumPy vs. fisher_kpp_jax CPU f64 — final fields, final_time,
     stopping quantity; near machine precision expected, hard FAIL above
     1e-8 relative L2,
  2. fisher_kpp_jax f32 (GPU if available, else CPU) vs. CPU f64 — same
     report, with the stopping-step shift (+-k steps) reported explicitly,
  3. wall-clock timing: NumPy vs JAX-CPU vs JAX-GPU, first solve (including
     jit compile) and second solve (warm persistent cache) separately.

Run from the project root:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/validate_against_original.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "fisher-kpp"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import fisher_kpp as ref_pkg  # noqa: E402
import fisher_kpp_jax as jax_pkg  # noqa: E402

N = 128
STRICT_REL_TOL = 1e-8  # part 1: reference vs JAX CPU f64
F32_REL_TOL = 1e-2  # part 2 sanity bound: f32 vs f64
INFO_DTI_ELONGATE_TOL = 1e-3  # cross-library float32 eigh, informational

FAILURES: list[str] = []
TIMINGS: list[tuple[str, str, float, float | None]] = []  # solver, impl, cold, warm

CPU_DEVICE = jax.devices("cpu")[0]
try:
    GPU_DEVICE: Any | None = jax.devices("gpu")[0]
except RuntimeError:
    GPU_DEVICE = None


def rel_l2(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    max_abs = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0
    denom = float(np.linalg.norm(theirs.ravel()))
    rel = float(np.linalg.norm((ours - theirs).ravel())) / denom if denom > 0 else max_abs
    return max_abs, rel


def report_field(label: str, ours: np.ndarray, theirs: np.ndarray, tol: float) -> None:
    if np.shape(ours) != np.shape(theirs):
        FAILURES.append(f"{label}: shape mismatch {np.shape(ours)} vs {np.shape(theirs)}")
        print(f"  {label:48s} SHAPE MISMATCH")
        return
    max_abs, rel = rel_l2(ours, theirs)
    ok = rel < tol
    if not ok:
        FAILURES.append(f"{label}: rel L2 {rel:.3e} (tol {tol:.0e})")
    print(f"  {label:48s} {'OK ' if ok else 'FAIL'} max|d|={max_abs:.3e}  relL2={rel:.3e}")


def report_scalar(label: str, ours: Any, theirs: Any, rel_tol: float = 1e-12) -> None:
    if isinstance(ours, str) or isinstance(theirs, str):
        ok = ours == theirs
    else:
        ok = ours == theirs or abs(ours - theirs) <= rel_tol * max(1.0, abs(theirs))
    if not ok:
        FAILURES.append(f"{label}: {ours!r} vs {theirs!r}")
    print(f"  {label:48s} {'OK ' if ok else 'FAIL'} ours={ours!r} theirs={theirs!r}")


def make_tissue(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    """Spherical WM core with a GM shell."""
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    wm = (r < n * 0.25).astype(np.float64)
    gm = ((r >= n * 0.25) & (r < n * 0.375)).astype(np.float64)
    return gm, wm


def make_tensor_field(n: int = N) -> np.ndarray:
    """Symmetric positive-definite tensors inside a sphere, zero outside."""
    rng = np.random.default_rng(42)
    b = rng.normal(size=(n, n, n, 3, 3))
    tensors = b @ b.transpose(0, 1, 2, 4, 3) * 0.05 + 0.3 * np.eye(3)
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    tensors[r >= n * 0.375] = 0.0
    return tensors


def timed_solve(solver_cls: type, params: dict) -> tuple[Any, float]:
    solver = solver_cls(params)
    start = time.perf_counter()
    result = solver.solve()
    return result, time.perf_counter() - start


def solve_jax(
    solver_cls: type, params: dict, precision: str, device: Any
) -> tuple[Any, float, float]:
    """Two identical solves on the given device: (result, cold_s, warm_s)."""
    with jax.default_device(device):
        result, cold = timed_solve(solver_cls, {**params, "precision": precision})
        _, warm = timed_solve(solver_cls, {**params, "precision": precision})
    return result, cold, warm


def steps_of(result: Any, dt: float) -> int:
    return int(round(result.final_time / dt))


def validate_solver(
    name: str,
    ref_cls: type,
    jax_cls: type,
    params: dict,
    state_keys: tuple[str, ...],
) -> None:
    print(f"\n== {name} ==")
    ref_result, ref_time = timed_solve(ref_cls, params)
    if not ref_result.success and ref_result.stopping_criterion != "error":
        FAILURES.append(f"{name}: reference solve failed: {ref_result.error}")
        return
    TIMINGS.append((name, "NumPy (f64)", ref_time, None))

    # dt for step-shift reporting (host-only, cheap).
    probe = jax_cls({**params, "precision": "f64"})
    probe._prepare_fields()
    vx, vy, vz = probe.params["voxel_size_mm"]
    factor = probe.params["resolution_factor"]
    probe.grid_spacing = (vx / factor, vy / factor, vz / factor)
    _, dt = probe._time_step_count()
    dt = float(dt)

    # --- part 1: reference vs JAX CPU f64 (strict) ---
    f64_result, f64_cold, f64_warm = solve_jax(jax_cls, params, "f64", CPU_DEVICE)
    TIMINGS.append((name, "JAX-CPU (f64)", f64_cold, f64_warm))
    print(" reference NumPy vs JAX CPU f64 (strict):")
    report_scalar("success", f64_result.success, ref_result.success)
    report_scalar("stopping_criterion", f64_result.stopping_criterion,
                  ref_result.stopping_criterion)
    report_scalar("final_time", f64_result.final_time, ref_result.final_time)
    report_scalar(
        "final_stopping_quantity",
        f64_result.final_stopping_quantity,
        ref_result.final_stopping_quantity,
        rel_tol=STRICT_REL_TOL,
    )
    for key in state_keys:
        report_field(
            f"final_state[{key}]",
            f64_result.final_state[key],
            ref_result.final_state[key],
            STRICT_REL_TOL,
        )

    # --- part 2: JAX f32 (GPU if available) vs JAX CPU f64 ---
    f32_device = GPU_DEVICE if GPU_DEVICE is not None else CPU_DEVICE
    f32_label = "JAX-GPU (f32)" if GPU_DEVICE is not None else "JAX-CPU (f32)"
    f32_result, f32_cold, f32_warm = solve_jax(jax_cls, params, "f32", f32_device)
    TIMINGS.append((name, f32_label, f32_cold, f32_warm))
    print(f" {f32_label} vs JAX CPU f64:")
    report_scalar("stopping_criterion", f32_result.stopping_criterion,
                  f64_result.stopping_criterion)
    shift = steps_of(f32_result, dt) - steps_of(f64_result, dt)
    print(f"  {'stopping-step shift (f32 - f64)':48s} {shift:+d} steps "
          f"(final_time {f32_result.final_time:.6g} vs {f64_result.final_time:.6g}, "
          f"dt={dt:.6g})")
    q64 = f64_result.final_stopping_quantity
    qrel = abs(f32_result.final_stopping_quantity - q64) / max(1.0, abs(q64))
    print(f"  {'final_stopping_quantity rel diff':48s} {qrel:.3e} "
          f"({f32_result.final_stopping_quantity:.8g} vs {q64:.8g})")
    for key in state_keys:
        report_field(
            f"final_state[{key}] (f32 vs f64)",
            f32_result.final_state[key],
            f64_result.final_state[key],
            F32_REL_TOL,
        )

    # --- extra GPU f64 timing, when a GPU is present ---
    if GPU_DEVICE is not None:
        _, gcold, gwarm = solve_jax(jax_cls, params, "f64", GPU_DEVICE)
        TIMINGS.append((name, "JAX-GPU (f64)", gcold, gwarm))


def validate_fk_volume_stop() -> None:
    """FK with a finite stopping threshold: the crossing step must match the
    reference exactly in f64; the f32 shift is reported explicitly. The
    JAX port uses the canonical ``stopping_threshold`` name; the reference
    package only knows the old ``stopping_volume`` name."""
    print("\n== FKPPSolver, stopping-volume early exit ==")
    gm, wm = make_tissue()
    base = dict(
        white_matter_diffusivity=0.3,
        rho=0.15,
        gray_matter=gm,
        white_matter=wm,
        seed_x_fraction=0.5,
        seed_y_fraction=0.5,
        seed_z_fraction=0.5,
        resolution_factor=1.0,
        stopping_time=30,
    )
    # Calibrate a mid-run threshold from a probe solve (JAX CPU f64).
    probe_result, _, _ = solve_jax(jax_pkg.FKPPSolver, base, "f64", CPU_DEVICE)
    threshold = 0.75 * probe_result.final_stopping_quantity
    ref_params = {**base, "stopping_volume": threshold}
    params = {**base, "stopping_threshold": threshold}

    ref_result, _ = timed_solve(ref_pkg.FKPPSolver, ref_params)
    if ref_result.stopping_criterion != "volume":
        FAILURES.append(
            "volume-stop scenario: reference did not stop on volume "
            f"(criterion={ref_result.stopping_criterion!r})"
        )
        return
    f64_result, _, _ = solve_jax(jax_pkg.FKPPSolver, params, "f64", CPU_DEVICE)
    f32_device = GPU_DEVICE if GPU_DEVICE is not None else CPU_DEVICE
    f32_result, _, _ = solve_jax(jax_pkg.FKPPSolver, params, "f32", f32_device)

    solver = jax_pkg.FKPPSolver({**params, "precision": "f64"})
    solver._prepare_fields()
    vx, vy, vz = solver.params["voxel_size_mm"]
    solver.grid_spacing = (vx, vy, vz)
    _, dt = solver._time_step_count()
    dt = float(dt)

    print(f" threshold={threshold:.6g} (75% of unconstrained final mass)")
    report_scalar("criterion (f64 vs reference)", f64_result.stopping_criterion,
                  ref_result.stopping_criterion)
    report_scalar("crossing final_time (f64 vs reference)", f64_result.final_time,
                  ref_result.final_time)
    report_scalar(
        "crossing quantity (f64 vs reference)",
        f64_result.final_stopping_quantity,
        ref_result.final_stopping_quantity,
        rel_tol=STRICT_REL_TOL,
    )
    report_field(
        "final_state at crossing (f64 vs reference)",
        f64_result.final_state["cell_density"],
        ref_result.final_state["cell_density"],
        STRICT_REL_TOL,
    )
    shift = steps_of(f32_result, dt) - steps_of(f64_result, dt)
    print(f"  {'crossing-step shift (f32 - f64)':48s} {shift:+d} steps "
          f"(final_time {f32_result.final_time:.6g} vs {f64_result.final_time:.6g})")
    report_scalar("criterion (f32 vs f64)", f32_result.stopping_criterion,
                  f64_result.stopping_criterion)


def validate_dti_elongated_info() -> None:
    """Informational: DTI with ellipsoid_scaling=1.5. Both implementations
    elongate in float32, but through different eigh backends (torch LAPACK vs
    jnp.linalg.eigh), so the diffusivity fields — and hence the solves —
    agree only to float32 eigendecomposition accuracy, not 1e-8. Reported
    with a loose 1e-3 bound that would only trip on a genuine port bug."""
    print("\n== AnisotropicFKPPSolver, ellipsoid_scaling=1.5 (informational) ==")
    tensors = make_tensor_field(64)  # elongation path only; smaller is enough
    params = dict(
        diffusivity=0.3,
        rho=0.15,
        diffusion_tensors=tensors,
        ellipsoid_scaling=1.5,
        tensor_exponent=2,
        tensor_linear_term=0.1,
        seed_x_fraction=0.5,
        seed_y_fraction=0.5,
        seed_z_fraction=0.5,
        resolution_factor=1.0,
        stopping_time=15,
    )
    ref_result, _ = timed_solve(ref_pkg.AnisotropicFKPPSolver, params)
    f64_result, _, _ = solve_jax(
        jax_pkg.AnisotropicFKPPSolver, params, "f64", CPU_DEVICE
    )
    report_scalar("stopping_criterion", f64_result.stopping_criterion,
                  ref_result.stopping_criterion)
    report_scalar("final_time", f64_result.final_time, ref_result.final_time)
    report_field(
        "final_state (float32-eigh-limited)",
        f64_result.final_state["cell_density"],
        ref_result.final_state["cell_density"],
        INFO_DTI_ELONGATE_TOL,
    )


def print_timings() -> None:
    print("\n== wall-clock timings ==")
    print(f"  {'solver':38s} {'implementation':16s} {'1st solve (incl compile)':>26s} "
          f"{'2nd solve (warm cache)':>24s}")
    for solver_name, impl, cold, warm in TIMINGS:
        warm_str = f"{warm:20.2f} s" if warm is not None else f"{'—':>22s}"
        print(f"  {solver_name:38s} {impl:16s} {cold:22.2f} s {warm_str}")


def main() -> None:
    print(f"jax {jax.__version__}, default backend: {jax.default_backend()}")
    if GPU_DEVICE is not None:
        print(f"GPU: {GPU_DEVICE.device_kind} "
              f"(compute capability {GPU_DEVICE.compute_capability})")
    else:
        print("No GPU visible: the f32 comparison and timings run on CPU.")

    gm, wm = make_tissue()
    tensors = make_tensor_field()

    validate_solver(
        "FKPPSolver",
        ref_pkg.FKPPSolver,
        jax_pkg.FKPPSolver,
        dict(
            white_matter_diffusivity=0.3,
            rho=0.15,
            gray_matter=gm,
            white_matter=wm,
            seed_x_fraction=0.5,
            seed_y_fraction=0.5,
            seed_z_fraction=0.5,
            resolution_factor=1.0,
            stopping_time=30,
        ),
        ("cell_density",),
    )
    validate_solver(
        "TwoCompartmentWithNutrientFKPPSolver",
        ref_pkg.TwoCompartmentWithNutrientFKPPSolver,
        jax_pkg.TwoCompartmentWithNutrientFKPPSolver,
        dict(
            white_matter_diffusivity=0.3,
            rho=0.15,
            necrosis_rate=0.4,
            nutrient_threshold=0.4,
            nutrient_diffusivity=0.5,
            nutrient_consumption_rate=0.1,
            gray_matter=gm,
            white_matter=wm,
            seed_x_fraction=0.5,
            seed_y_fraction=0.5,
            seed_z_fraction=0.5,
            resolution_factor=1.0,
            stopping_time=30,
        ),
        ("proliferative", "necrotic", "nutrient"),
    )
    validate_solver(
        "AnisotropicFKPPSolver",
        ref_pkg.AnisotropicFKPPSolver,
        jax_pkg.AnisotropicFKPPSolver,
        dict(
            diffusivity=0.3,
            rho=0.15,
            diffusion_tensors=tensors,
            seed_x_fraction=0.5,
            seed_y_fraction=0.5,
            seed_z_fraction=0.5,
            resolution_factor=1.0,
            stopping_time=30,
        ),
        ("cell_density",),
    )
    validate_fk_volume_stop()
    validate_dti_elongated_info()
    print_timings()

    print()
    if FAILURES:
        print("FAIL — the following checks exceeded tolerance:")
        for failure in FAILURES:
            print(" -", failure)
        sys.exit(1)
    print("PASS — all comparisons within tolerance.")


if __name__ == "__main__":
    main()
