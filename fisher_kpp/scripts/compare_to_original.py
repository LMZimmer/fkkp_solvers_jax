"""Throwaway comparison of the fisher_kpp refactor against TumorGrowthToolkit.

Runs the original solvers and the refactored ones on synthetic phantoms with
matched parameters (stopping_mode="mass") and reports max-abs / relative-L2
differences. After the bug-fix pass, deviations from the originals are
expected ONLY from:
  (a) zero-flux boundary handling near the crop margin (fix 4),
  (b) the FK_2c voxel-volume stopping fix and reaction-rate dt guard (fixes
      2/3) — the guard is non-binding for these parameters, so only the
      reported stopping quantity deviates,
  (c) the DTI guard-exit failure semantics (fix 1).
Operator stencils are additionally checked to be bitwise identical to the
roll-based originals on the interior (excluding a 1-voxel shell). Anything
else should still be at floating-point noise level (rel L2 < 1e-8).

Run from the project root:  uv run python scripts/compare_to_original.py
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

TGT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "TumorGrowthToolkit",
)
sys.path.insert(0, TGT_ROOT)


def _stub_module(name: str, attrs: tuple[str, ...] = ()) -> types.ModuleType:
    mod = types.ModuleType(name)
    for attr in attrs:
        setattr(mod, attr, lambda *a, **k: None)
    sys.modules[name] = mod
    return mod


# The original FK_DTI module (and its package __init__ via tools.py) imports
# matplotlib, nibabel and dipy at module level; none of those code paths run
# with doPlot=False, so stub them out instead of installing them.
_mpl = _stub_module("matplotlib")
_mpl.pyplot = _stub_module("matplotlib.pyplot")
_stub_module("nibabel", ("load", "save", "Nifti1Image"))
_dipy = _stub_module("dipy")
_dipy.core = _stub_module("dipy.core")
_dipy.core.gradients = _stub_module("dipy.core.gradients", ("gradient_table",))
_dipy.data = _stub_module(
    "dipy.data", ("fetch_sherbrooke_3shell", "fetch_bundle_atlas_hcp842")
)
_dipy.io = _stub_module("dipy.io", ("read_bvals_bvecs",))
_dipy.io.image = _stub_module("dipy.io.image", ("load_nifti", "save_nifti"))
_dipy.reconst = _stub_module("dipy.reconst")
_dipy.reconst.dti = _stub_module(
    "dipy.reconst.dti", ("TensorModel", "fractional_anisotropy", "color_fa")
)
_dipy.segment = _stub_module("dipy.segment")
_dipy.segment.mask = _stub_module("dipy.segment.mask", ("median_otsu",))

from TumorGrowthToolkit.FK.FK import Solver as OriginalFK  # noqa: E402
from TumorGrowthToolkit.FK_2c.FK_2c import Solver as OriginalFK2c  # noqa: E402
from TumorGrowthToolkit.FK_DTI import tools as original_tools  # noqa: E402
from TumorGrowthToolkit.FK_DTI.FK_DTI import FK_DTI_Solver as OriginalDTI  # noqa: E402

from fisher_kpp import (  # noqa: E402
    AnisotropicFKPPSolver,
    FKPPSolver,
    TwoCompartmentWithNutrientFKPPSolver,
)
from fisher_kpp.operators import (  # noqa: E402
    diffusion_term,
    edge_roll,
    elongate_tensor_along_principal_axis,
    face_average,
    masked_face_average,
)

FAILURES: list[str] = []


def report(label: str, ours: np.ndarray, theirs: np.ndarray, tol: float = 1e-8) -> None:
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    if ours.shape != theirs.shape:
        FAILURES.append(f"{label}: shape mismatch {ours.shape} vs {theirs.shape}")
        print(f"  {label:45s} SHAPE MISMATCH {ours.shape} vs {theirs.shape}")
        return
    max_abs = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0
    denom = float(np.linalg.norm(theirs.ravel()))
    rel_l2 = float(np.linalg.norm((ours - theirs).ravel())) / denom if denom > 0 else max_abs
    status = "OK " if rel_l2 < tol else "FAIL"
    if rel_l2 >= tol:
        FAILURES.append(f"{label}: rel L2 {rel_l2:.3e}")
    print(f"  {label:45s} {status} max|d|={max_abs:.3e}  relL2={rel_l2:.3e}")


def report_exact(label: str, ours: np.ndarray, theirs: np.ndarray) -> None:
    """Bitwise equality (max|d| must be exactly 0)."""
    max_abs = float(np.max(np.abs(np.asarray(ours) - np.asarray(theirs))))
    if max_abs != 0.0:
        FAILURES.append(f"{label}: not bitwise identical, max|d|={max_abs:.3e}")
    print(f"  {label:45s} {'OK ' if max_abs == 0.0 else 'FAIL'} max|d|={max_abs:.3e}")


def report_scalar(label: str, ours: float, theirs: float) -> None:
    ok = ours == theirs or abs(ours - theirs) <= 1e-12 * max(1.0, abs(theirs))
    if not ok:
        FAILURES.append(f"{label}: {ours!r} vs {theirs!r}")
    print(f"  {label:45s} {'OK ' if ok else 'FAIL'} ours={ours!r} theirs={theirs!r}")


def make_tissue(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Spherical WM core with a GM shell."""
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    wm = (r < 10).astype(np.float64)
    gm = ((r >= 10) & (r < 15)).astype(np.float64)
    return gm, wm


def make_tensor_field(n: int = 40) -> np.ndarray:
    """Symmetric positive-definite tensors inside a sphere, zero outside."""
    rng = np.random.default_rng(42)
    b = rng.normal(size=(n, n, n, 3, 3))
    tensors = b @ b.transpose(0, 1, 2, 4, 3) * 0.05 + 0.3 * np.eye(3)
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    tensors[r >= 15] = 0.0
    return tensors


def compare_stencils() -> None:
    """Fix 4 check: zero-flux stencils vs the roll-based (periodic) originals
    must be bitwise identical on the interior (excluding a 1-voxel shell)."""
    print("\n== operator stencils vs roll-based originals (interior) ==")
    rng = np.random.default_rng(7)
    shape = (12, 13, 14)
    field = rng.random(shape)
    mask = rng.random(shape) > 0.3
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    def roll_diffusion_term(u, d, spacing):
        dx, dy, dz = spacing
        sp_x = 1 / (dx * dx) * (
            d["plus_x"] * (np.roll(u, 1, axis=0) - u)
            - d["minus_x"] * (u - np.roll(u, -1, axis=0))
        )
        sp_y = 1 / (dy * dy) * (
            d["plus_y"] * (np.roll(u, 1, axis=1) - u)
            - d["minus_y"] * (u - np.roll(u, -1, axis=1))
        )
        sp_z = 1 / (dz * dz) * (
            d["plus_z"] * (np.roll(u, 1, axis=2) - u)
            - d["minus_z"] * (u - np.roll(u, -1, axis=2))
        )
        return sp_x + sp_y + sp_z

    for axis in range(3):
        ours = face_average(field, axis)
        theirs = (np.roll(field, -1, axis=axis) + field) / 2
        report_exact(f"face_average interior (axis={axis})", ours[interior], theirs[interior])

        ours = masked_face_average(field, mask, axis)
        cond = np.logical_and(np.roll(mask, -1, axis=axis), mask)
        theirs = np.where(cond, (np.roll(field, -1, axis=axis) + field) / 2, 0)
        report_exact(
            f"masked_face_average interior (axis={axis})", ours[interior], theirs[interior]
        )

    # End-to-end diffusion term: minus faces shared, plus faces built the way
    # each implementation builds them (edge_roll vs np.roll).
    u = rng.random(shape)
    spacing = (1.3, 0.9, 1.1)
    minus = {name: rng.random(shape) for name in ("x", "y", "z")}
    ours_faces = {}
    theirs_faces = {}
    for axis, name in enumerate(("x", "y", "z")):
        ours_faces[f"minus_{name}"] = minus[name]
        theirs_faces[f"minus_{name}"] = minus[name]
        ours_faces[f"plus_{name}"] = edge_roll(minus[name], 1, axis=axis)
        theirs_faces[f"plus_{name}"] = np.roll(minus[name], 1, axis=axis)
    ours = diffusion_term(u, ours_faces, spacing)
    theirs = roll_diffusion_term(u, theirs_faces, spacing)
    report_exact("diffusion_term interior", ours[interior], theirs[interior])


def compare_fk(stopping_volume: float, tag: str) -> None:
    print(f"\n== FK vs FKPPSolver ({tag}) ==")
    gm, wm = make_tissue()
    common = dict(rho=0.15, stopping_time=40, stopping_volume=stopping_volume)
    orig = OriginalFK(
        {
            "Dw": 0.3,
            "gm": gm,
            "wm": wm,
            "NxT1_pct": 0.5,
            "NyT1_pct": 0.5,
            "NzT1_pct": 0.5,
            "resolution_factor": 0.6,
            "RatioDw_Dg": 10.0,
            "th_matter": 0.1,
            "init_scale": 1.0,
            "time_series_solution_Nt": 4,
            **common,
        }
    ).solve()
    ours = FKPPSolver(
        {
            "white_matter_diffusivity": 0.3,
            "gray_matter": gm,
            "white_matter": wm,
            "gaussian_seed_x_fraction": 0.5,
            "gaussian_seed_y_fraction": 0.5,
            "gaussian_seed_z_fraction": 0.5,
            "resolution_factor": 0.6,
            "diffusivity_ratio": 10.0,
            "min_tissue_fraction": 0.1,
            "gaussian_seed_scale": 1.0,
            "n_time_series_snapshots": 4,
            "stopping_mode": "mass",
            **common,
        }
    ).solve()
    assert orig["success"] and ours.success, (orig.get("error"), ours.error)
    report("initial_state", ours.initial_state["cell_density"], orig["initial_state"])
    report("final_state", ours.final_state["cell_density"], orig["final_state"])
    report("time_series", ours.time_series["cell_density"], orig["time_series"])
    report_scalar("final_time", ours.final_time, orig["final_time"])
    report_scalar(
        "final_stopping_quantity", ours.final_stopping_quantity, orig["final_volume"]
    )
    report_scalar(
        "stopping_criterion",
        ours.stopping_criterion,
        orig["stopping_criteria"],
    )


def compare_fk_2c() -> None:
    print("\n== FK_2c vs TwoCompartmentWithNutrientFKPPSolver ==")
    gm, wm = make_tissue()
    orig = OriginalFK2c(
        {
            "Dw": 0.3,
            "rho": 0.15,
            "lambda_np": 0.4,
            "sigma_np": 0.4,
            "D_s": 0.5,
            "lambda_s": 0.1,
            "gm": gm,
            "wm": wm,
            "NxT1_pct": 0.5,
            "NyT1_pct": 0.5,
            "NzT1_pct": 0.5,
            "resolution_factor": 0.6,
            "th_matter": 0.1,
            "th_necro": 0.9,
            "stopping_time": 40,
            "time_series_solution_Nt": 4,
            "Nt_multiplier": 8,
        }
    ).solve()
    ours = TwoCompartmentWithNutrientFKPPSolver(
        {
            "white_matter_diffusivity": 0.3,
            "rho": 0.15,
            "necrosis_rate": 0.4,
            "nutrient_threshold": 0.4,
            "nutrient_diffusivity": 0.5,
            "nutrient_consumption_rate": 0.1,
            "gray_matter": gm,
            "white_matter": wm,
            "gaussian_seed_x_fraction": 0.5,
            "gaussian_seed_y_fraction": 0.5,
            "gaussian_seed_z_fraction": 0.5,
            "resolution_factor": 0.6,
            "min_tissue_fraction": 0.1,
            "max_tumor_occupancy": 0.9,
            "stopping_time": 40,
            "n_time_series_snapshots": 4,
            "nt_multiplier": 8,
            "stopping_mode": "mass",
        }
    ).solve()
    assert orig["success"] and ours.success, (orig.get("error"), ours.error)
    for old_key, new_key in [
        ("P", "proliferative"),
        ("N", "necrotic"),
        ("S", "nutrient"),
    ]:
        report(
            f"initial_state[{new_key}]",
            ours.initial_state[new_key],
            orig["initial_state"][old_key],
        )
        report(
            f"final_state[{new_key}]",
            ours.final_state[new_key],
            orig["final_state"][old_key],
        )
        report(
            f"time_series[{new_key}]",
            ours.time_series[new_key],
            np.array(orig["time_series"][old_key]),
        )
    report_scalar("final_time", ours.final_time, orig["final_time"])
    report_scalar(
        "stopping_criterion", ours.stopping_criterion, orig["stopping_criteria"]
    )

    # Expected deviation (b): ours = vv*(sum P + sum N), theirs = vv*sum P +
    # sum N (the original FK_2c drops the voxel-volume factor on N). Check
    # the deviation carries exactly that algebraic signature: the implied
    # low-res sums of P and N must be non-negative, and the implied sum of N
    # must roughly match the one recovered from the original's final N field
    # (upsampling scales a field's sum by ~vv for 1mm voxels).
    vv = (1.0 / 0.6) ** 3
    ours_q = ours.final_stopping_quantity
    theirs_q = orig["final_volume"]
    n_sum = (ours_q - theirs_q) / (vv - 1)
    p_sum = (theirs_q - n_sum) / vv
    n_sum_est = float(np.sum(orig["final_state"]["N"])) / vv
    consistent = (
        n_sum >= 0
        and p_sum >= 0
        and (n_sum_est == 0 or abs(n_sum - n_sum_est) / n_sum_est < 0.5)
    )
    if not consistent:
        FAILURES.append(
            "FK_2c stopping quantity deviation not attributable to the "
            f"voxel-volume fix: ours={ours_q!r} theirs={theirs_q!r} "
            f"implied sumP={p_sum!r} sumN={n_sum!r} est sumN={n_sum_est!r}"
        )
    print(
        f"  {'final_stopping_quantity (fix 2b)':45s} "
        f"{'OK ' if consistent else 'FAIL'} ours={ours_q:.6g} orig={theirs_q:.6g} "
        f"deviation attributed to voxel-volume fix "
        f"(implied sumN={n_sum:.6g} vs est {n_sum_est:.6g})"
    )


def compare_dti_helpers() -> None:
    print("\n== DTI helper functions ==")
    tensors = make_tensor_field()
    gm, wm = make_tissue()

    ours_elong = elongate_tensor_along_principal_axis(tensors, 1.5)
    theirs_elong = original_tools.elongate_tensor_along_main_axis_torch(tensors, 1.5)
    report("elongate_tensor (factor 1.5)", ours_elong, theirs_elong)

    # Original makeXYZ_rgb_from_tensor is a Solver method reading limit params.
    limit_params = {"relative_upper_limit_DTI": 2, "relative_lower_limit_DTI": 0.05}
    orig_solver = OriginalDTI(dict(limit_params))
    new_solver = AnisotropicFKPPSolver(
        {
            "diffusivity": 0.3,
            "rho": 0.15,
            "diffusion_tensors": tensors,
            "gaussian_seed_x_fraction": 0.5,
            "gaussian_seed_y_fraction": 0.5,
            "gaussian_seed_z_fraction": 0.5,
            "resolution_factor": 0.6,
            "diffusivity_upper_limit": 2,
            "diffusivity_lower_limit": 0.05,
        }
    )

    theirs = orig_solver.makeXYZ_rgb_from_tensor(
        tensors, exponent=2, linear=0.1, desiredSTD=None
    )
    mine = new_solver._axial_diffusivity_from_tensor(
        tensors, exponent=2, linear_term=0.1, wm=None, gm=None,
        diffusivity_ratio=None, normalization_std=None,
    )
    report("axial diffusivity (plain, exp=2)", mine, theirs)

    theirs = orig_solver.makeXYZ_rgb_from_tensor(
        tensors, exponent=1, linear=0, desiredSTD=0.2
    )
    mine = new_solver._axial_diffusivity_from_tensor(
        tensors, exponent=1, linear_term=0, wm=None, gm=None,
        diffusivity_ratio=None, normalization_std=0.2,
    )
    report("axial diffusivity (desiredSTD=0.2)", mine, theirs)

    theirs = orig_solver.makeXYZ_rgb_from_tensor(
        tensors, exponent=1, linear=0, wm=wm, gm=gm, ratioDw_Dg=10.0, desiredSTD=None
    )
    mine = new_solver._axial_diffusivity_from_tensor(
        tensors, exponent=1, linear_term=0, wm=wm, gm=gm,
        diffusivity_ratio=10.0, normalization_std=None,
    )
    report("axial diffusivity (uniform gm branch)", mine, theirs)


def compare_dti_solve(uniform_gm: bool) -> None:
    print(f"\n== FK_DTI vs AnisotropicFKPPSolver (uniform_gray_matter={uniform_gm}) ==")
    tensors = make_tensor_field()
    gm, wm = make_tissue()
    orig_params = {
        "Dw": 0.3,
        "rho": 0.15,
        "diffusionTensors": tensors,
        "diffusionEllipsoidScaling": 1.5,
        "diffusionTensorExponent": 2,
        "diffusionTensorLinear": 0.1,
        "NxT1_pct": 0.5,
        "NyT1_pct": 0.5,
        "NzT1_pct": 0.5,
        "resolution_factor": 0.6,
        "stopping_time": 40,
        "time_series_solution_Nt": 4,
    }
    new_params = {
        "diffusivity": 0.3,
        "rho": 0.15,
        "diffusion_tensors": tensors,
        "ellipsoid_scaling": 1.5,
        "tensor_exponent": 2,
        "tensor_linear_term": 0.1,
        "gaussian_seed_x_fraction": 0.5,
        "gaussian_seed_y_fraction": 0.5,
        "gaussian_seed_z_fraction": 0.5,
        "resolution_factor": 0.6,
        "stopping_time": 40,
        "n_time_series_snapshots": 4,
        "stopping_mode": "mass",
    }
    if uniform_gm:
        orig_params.update(
            {"use_homogen_gm": True, "gm": gm, "wm": wm, "RatioDw_Dg": 10.0}
        )
        new_params.update(
            {
                "uniform_gray_matter": True,
                "gray_matter": gm,
                "white_matter": wm,
                "diffusivity_ratio": 10.0,
            }
        )
    orig = OriginalDTI(orig_params).solve()
    ours = AnisotropicFKPPSolver(new_params).solve()
    assert orig["success"] and ours.success, (orig.get("error"), ours.error)
    report("initial_state", ours.initial_state["cell_density"], orig["initial_state"])
    report("final_state", ours.final_state["cell_density"], orig["final_state"])
    report("time_series", ours.time_series["cell_density"], orig["time_series"])
    report_scalar("final_time", ours.final_time, orig["final_time"])
    report_scalar(
        "final_stopping_quantity", ours.final_stopping_quantity, orig["final_volume"]
    )
    report_scalar(
        "stopping_criterion", ours.stopping_criterion, orig["stopping_criteria"]
    )


def compare_dti_guard_exit() -> None:
    """Fix 1 check (expected deviation (c)): with a shrinking tumor (negative
    rho) a DTI guard fires. The original reports that as a successful "time"
    stop with final_time == stopping_time; ours must report a failure with
    the actual exit time. The simulated fields at the exit step must still
    match the original's."""
    print("\n== FK_DTI guard exit (fix 1) ==")
    tensors = make_tensor_field()
    orig = OriginalDTI(
        {
            "Dw": 0.3,
            "rho": -1.0,
            "diffusionTensors": tensors,
            "NxT1_pct": 0.5,
            "NyT1_pct": 0.5,
            "NzT1_pct": 0.5,
            "resolution_factor": 0.6,
            "stopping_time": 40,
        }
    ).solve()
    ours = AnisotropicFKPPSolver(
        {
            "diffusivity": 0.3,
            "rho": -1.0,
            "diffusion_tensors": tensors,
            "gaussian_seed_x_fraction": 0.5,
            "gaussian_seed_y_fraction": 0.5,
            "gaussian_seed_z_fraction": 0.5,
            "resolution_factor": 0.6,
            "stopping_time": 40,
        }
    ).solve()
    report("final_state (at guard exit)", ours.final_state["cell_density"], orig["final_state"])
    checks = [
        ("original reports success", orig["success"] is True),
        ("original final_time == stopping_time", orig["final_time"] == 40),
        ("ours reports failure", ours.success is False),
        ("ours stopping_criterion == 'error'", ours.stopping_criterion == "error"),
        ("ours error names a guard", ours.error is not None and "guard" in ours.error),
        ("ours final_time < stopping_time", ours.final_time < 40),
    ]
    for label, ok in checks:
        if not ok:
            FAILURES.append(f"DTI guard exit: {label}")
        print(f"  {label:45s} {'OK ' if ok else 'FAIL'}")
    print(f"  ours: final_time={ours.final_time!r} error={ours.error!r}")


def check_stopping_mode_validation() -> None:
    """Fix 2 checks with no original counterpart: 'volume' mode and the
    density_threshold validation."""
    print("\n== stopping_mode='volume' and validation ==")
    gm, wm = make_tissue()
    base = {
        "white_matter_diffusivity": 0.3,
        "rho": 0.15,
        "gray_matter": gm,
        "white_matter": wm,
        "gaussian_seed_x_fraction": 0.5,
        "gaussian_seed_y_fraction": 0.5,
        "gaussian_seed_z_fraction": 0.5,
        "resolution_factor": 0.6,
        "stopping_time": 40,
        "stopping_volume": 50.0,
    }
    res = FKPPSolver({**base, "stopping_mode": "volume"}).solve()
    checks = [
        ("volume mode solve succeeds", res.success),
        ("volume-mode stop reached", res.stopping_criterion == "volume"),
        (
            "quantity is a voxel-count volume",
            res.final_stopping_quantity >= 50.0
            and abs(
                res.final_stopping_quantity / ((1 / 0.6) ** 3)
                - round(res.final_stopping_quantity / ((1 / 0.6) ** 3))
            )
            < 1e-9,
        ),
    ]
    try:
        FKPPSolver({**base, "density_threshold": 0.5})
        checks.append(("density_threshold rejected in mass mode", False))
    except ValueError:
        checks.append(("density_threshold rejected in mass mode", True))
    try:
        FKPPSolver({**base, "stopping_mode": "occupancy"})
        checks.append(("bad stopping_mode rejected", False))
    except ValueError:
        checks.append(("bad stopping_mode rejected", True))
    for label, ok in checks:
        if not ok:
            FAILURES.append(f"stopping_mode: {label}")
        print(f"  {label:45s} {'OK ' if ok else 'FAIL'}")


def main() -> None:
    compare_stencils()
    compare_fk(stopping_volume=np.inf, tag="time stop")
    compare_fk(stopping_volume=300.0, tag="volume stop")
    compare_fk_2c()
    compare_dti_helpers()
    compare_dti_solve(uniform_gm=False)
    compare_dti_solve(uniform_gm=True)
    compare_dti_guard_exit()
    check_stopping_mode_validation()
    print()
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("All comparisons within tolerance.")


if __name__ == "__main__":
    main()
