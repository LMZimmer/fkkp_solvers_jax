#!/usr/bin/env python
"""Re-solve the best run of a scripts/patient_cmaes_fit.py fit of the
two-species model (--solver twospecies,
fisher_kpp_jax.TwoCompartmentWithNutrientFKPPSolver) and render its three
fields next to the patient's longitudinal tumour segmentations, one
figure per field (proliferative, necrotic, nutrient).

The counterpart of scripts/visualize_patient_best_run.py for the
two-species solver: the fit-directory path, the session panels, the
backgrounds, the palette, the masks and the volume panel are that
script's (its helpers are imported, nothing is copied); what differs is
below. The fit directory (--fit-dir, the only source: the two-species
model has been run in fits only) must have been run with --solver
twospecies and --objective full (the figures carry the treated timeline);
the isotropic solver's fits are rendered by the other script.

Run. The best run is the fit's re-solved best evaluation (best/config.json,
written by the fit's resolve). It is solved once more as the fit saved
it: its schedule, its fixed n_steps and the session moments and snapshot
days of the config's "_fit" record, over the sessions the fit simulated
(plus the curve frames with --volume-curve), recording the three state
fields at every snapshot; the patient and the sessions come from the
fit's spec.json. The criterion is the re-solve's J (best/objective.json).

Scored densities, thresholds and Dice (the fit's loss, spec.json
objective.mode). With --loss core+necrotic (the two-species default
since 2026-10-02) the core Dice is that of patient_cmaes_fit's
``model_core_mask``: the thresholded P joined with the thresholded N
against the pre-op core (labels 1 and 3), and the thresholded P alone
against the enhancing tumour (label 3) of every later session; the
necrotic Dice is the thresholded N against the necrotic label (label 1
after the cavity correction), NaN in a session without one (dropped by
the fit). With --loss core (the rule until 2026-10-02) the core Dice is
that of P + N against the pre-op core and of P against the later
enhancing tumour, and there is no necrotic Dice. All on the fit's masks
(patient_cmaes_fit.fit_session_references: the session's own cavity,
the earlier sessions' cavities and the resection cavity excluded).
--core-threshold logged (the default) takes the fit's profiled core
threshold and --necrotic-threshold logged the profiled necrotic one, so
the per-session Dice equal the fit's dice_core_star and
dice_necrotic_star (printed next to them and recorded as fit_dice_*)
when the fit was scored with the current masks (a note is printed
otherwise); a number fixes either; --core-threshold infer (core-loss
fits only) maximises the mean over the sessions of the core Dice on the
grid 0.05..0.95 step 0.01. The session frames are rounded for storage
as the fit rounds its fields before the masks are taken.

Figures, all in the layout of the other script (one panel per session on
the axial slice through the centre of mass of the pre-op reference core,
or --slice-z; the session's T1c as the background; rows of --columns;
titles "<session> <label> (day <n>)" with the clinical day relative to
the post-op scan; a volume-vs-time panel below in model days with the
treatment events marked; no figure title):
  segmentations   the other script's segmentation figure, unchanged
  proliferative   the P frame overlaid (inferno at alpha 0.75, values
                  below --display-threshold transparent) with the
                  session's segmented tumour core (labels 1 + 3) as a red
                  contour and, from the post-op session on, the resection
                  cavity (the post-op session's label 4) in green; in each
                  panel the fit's core Dice (the rule of the fit's loss,
                  named in the legend above the panels); below, the
                  volume of P >= core threshold
                  per session (cavity excluded) as hollow red circles,
                  with --volume-curve the same quantity along the run
                  (latest scan's mask) as a dashed red line, and the
                  segmented core volumes as filled black squares
  necrotic        the same with the N frame: with core+necrotic the
                  necrotic Dice in each panel ("n/a" where the session
                  has no necrotic label), the volume of N >= necrotic
                  threshold and the segmented NECROTIC volumes (label 1
                  after the correction) as the black squares; with the
                  core loss the core Dice again, N >= core threshold and
                  the segmented core volumes
  nutrient        the nutrient DEPLETION 1 - nutrient overlaid, in tissue
                  only (where the solver's initial nutrient is 1: wm + gm
                  >= min_tissue_fraction; the CSF, where the nutrient is
                  0 throughout, stays transparent), with the same colormap
                  and display threshold and the contours; no Dice; below,
                  the depleted tissue volume per session (nutrient <
                  the config's nutrient_threshold, the necrosis switch
                  level, in tissue, cavity excluded; the resection clears
                  the nutrient in the cavity, which must not count as
                  depleted) and, with --volume-curve, along the run
Written into <output-dir>/<run-name>/ (exist_ok=False, nothing outside
it): config.json, result.json, initial_<field>.nii.gz and
final_<field>.nii.gz per field (fisher_kpp_jax.Result.save), the session
frames <session>_<field>.nii.gz per field (the fit's resolve naming),
segmentations.pdf/.png, proliferative.pdf/.png, necrotic.pdf/.png,
nutrient.pdf/.png and run_summary.json (the fit's best restart and
evaluation and its J, the thresholds and how they were chosen, the
nutrient threshold, the slice, per session the model moment, post-op day,
recorded day, the fit's Dice and the re-solve's, the three model volumes
and the segmented core volume, excluded cavity and relabelled voxel
counts, the files, and with --volume-curve the curve days and volumes
per field).

Run from the project root, e.g.:
  CUDA_VISIBLE_DEVICES=<free gpu> python scripts/visualize_patient_best_run_twospecies.py \\
      --fit-dir /mnt/Drive4/lucas/stupp_patient_fit/fit_sub01_twospecies_2026-10-01 --output-dir runs/
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
# numpy's transparent-hugepage request stalls on a fragmented host (see
# visualize_patient_best_run.py). Read by numpy at import.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from fisher_kpp_jax import TwoCompartmentWithNutrientFKPPSolver, read_config  # noqa: E402
from fisher_kpp_jax.config import jsonable  # noqa: E402
from patient_cmaes_fit import (  # noqa: E402
    fit_session_references,
    model_core_mask,
    region_densities,
    same_label_conventions,
)
from patient_sensitivity_analysis import (  # noqa: E402
    LABEL_CAVITY,
    LABEL_POSTOP,
    LABEL_PREOP,
    Reference,
    check_preop_labels,
    dice,
    load_segmentation,
    snapshot_file,
)
from sensitivity_analysis import read_json, round_field, write_json  # noqa: E402
from visualize_patient_best_run import (  # noqa: E402
    CAVITY_OUTLINE_COLOR,
    CONTOUR_LEGEND,
    FIELD_ALPHA,
    FIT_CRITERION,
    MODEL_COLOR,
    OBSERVED_COLOR,
    OBSERVED_CORE_CONTOUR_COLOR,
    OBSERVED_CORE_LABELS,
    REFERENCE_CORE_LABEL,
    _figure,
    _finish_curve,
    fit_best_run,
    infer_threshold,
    load_volume,
    postop_days,
    recorded_frame,
    render_segmentations,
    session_background,
    session_title,
)

SOLVER_MODE = "twospecies"
# The state fields, in figure order, with the colorbar label of each.
FIELDS: tuple[str, ...] = ("proliferative", "necrotic", "nutrient")
COLORBAR_LABELS: dict[str, str] = {
    "proliferative": "proliferative density P",
    "necrotic": "necrotic density N",
    "nutrient": "nutrient depletion (1 - nutrient, tissue only)",
}
# The legend line naming the fit's Dice rule on the P and N figures, per loss.
NECROTIC_LOSS = "core+necrotic"
DICE_RULES: dict[str, str] = {
    "core": "Dice core: P + N vs pre-op core (labels 1 + 3), P vs post-op enhancing (label 3), at the core threshold",
    NECROTIC_LOSS: (
        "Dice core: thresholded P or thresholded N vs pre-op core (labels 1 + 3), thresholded P vs post-op enhancing (label 3); "
        "Dice necrotic: thresholded N vs label 1"
    ),
}
REFERENCE_NECROTIC_LABEL = "segmented necrotic (label 1 after the cavity correction)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fit-dir",
        required=True,
        help="scripts/patient_cmaes_fit.py --solver twospecies fit directory holding spec.json and the re-solved best/",
    )
    parser.add_argument(
        "--core-threshold",
        default="logged",
        help="'logged' (the fit's profiled core threshold; default), 'infer' (core-loss fits only: mean core Dice over "
        "the sessions maximised on a grid) or a number",
    )
    parser.add_argument(
        "--necrotic-threshold",
        default="logged",
        help="core+necrotic fits: 'logged' (the fit's profiled necrotic threshold; default) or a number",
    )
    parser.add_argument(
        "--volume-curve",
        action="store_true",
        help="draw each field's volume along the run as a line on its figure's volume panel",
    )
    parser.add_argument(
        "--snapshot-interval-days",
        type=float,
        default=2.0,
        help="spacing of the curve frames (default 2; --volume-curve only)",
    )
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name", default=None, help="run directory name (default: best_<fit>_restart<k>_eval<n>_<UTC>_<pid>)"
    )
    parser.add_argument(
        "--patient-root", default=None, help="patient data root (default: spec.json's)"
    )
    parser.add_argument(
        "--slice-z", type=int, default=None, help="axial slice (default: pre-op core centre of mass)"
    )
    parser.add_argument(
        "--display-threshold", type=float, default=0.01, help="overlay value below which the overlay is transparent"
    )
    parser.add_argument("--columns", type=int, default=4, help="panels per row (default 4)")
    return parser.parse_args(argv)


def render_field(
    outfile_stem: Path,
    sessions: list[dict[str, Any]],
    overlays: list[NDArray],
    resection_cavity: NDArray,
    labels: list[NDArray],
    backgrounds: list[NDArray],
    z: int,
    display_threshold: float,
    colorbar_label: str,
    moments: list[float],
    days: list[int],
    dices: list[float] | None,
    dice_label: str,
    dice_rule: str,
    volumes: list[float],
    volume_label: str,
    reference_volumes: list[float],
    reference_label: str,
    curve_times: NDArray | None,
    curve_volumes: NDArray | None,
    curve_label: str,
    params: dict[str, Any],
    stopping_time: float,
    n_col: int,
) -> None:
    """
    One field's figure: per session the overlay (the field, or the
    depletion for the nutrient) over the T1c with the segmented tumour
    core (red) and, in every session but the pre-op one, the resection
    cavity (green); the fit's Dice in each panel when given (P and N
    figures: "<dice_label> <value>", "n/a" for a NaN; the rule is named
    in the legend above the panels); below, the field's volume per
    session as hollow red circles, the curve along the run when given,
    and the reference volumes (reference_label) as filled black squares.
    """
    fig, axes, (bottom, colorbar_ax) = _figure(len(sessions), n_col)
    handles = [
        Line2D([], [], color=color, linewidth=width, label=label)
        for label, (color, width) in CONTOUR_LEGEND.items()
    ]
    if dices is not None:
        handles.append(Line2D([], [], color="none", label=dice_rule))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(CONTOUR_LEGEND), fontsize=10, frameon=False)  # the rule on its own row
    image = None
    cavity = np.rot90(np.asarray(resection_cavity, dtype=bool)[:, :, z])
    for k, (ax, session, overlay, label_volume, background, day) in enumerate(
        zip(axes, sessions, overlays, labels, backgrounds, days, strict=True)
    ):
        ax.imshow(np.rot90(background[:, :, z]), cmap="gray", interpolation="none")
        image = ax.imshow(
            np.ma.masked_less(np.rot90(overlay[:, :, z]), display_threshold), cmap="inferno", alpha=FIELD_ALPHA,
            vmin=0.0, vmax=1.0, interpolation="none",
        )
        observed_core = np.rot90(np.isin(label_volume, OBSERVED_CORE_LABELS)[:, :, z])
        if observed_core.any():
            ax.contour(
                observed_core.astype(float), levels=[0.5], colors=[OBSERVED_CORE_CONTOUR_COLOR], linewidths=1.2
            )
        if session["label"] != LABEL_PREOP and cavity.any():
            ax.contour(cavity.astype(float), levels=[0.5], colors=[CAVITY_OUTLINE_COLOR], linewidths=1.2)
        ax.set_title(session_title(session, day), fontsize=12, fontweight="bold", pad=8)
        if dices is not None:
            value = "n/a" if not np.isfinite(dices[k]) else f"{dices[k]:.2f}"
            ax.text(0.02, 0.02, f"{dice_label} {value}", transform=ax.transAxes, color="white", fontsize=10, va="bottom")
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, cax=colorbar_ax, label=colorbar_label)
    if curve_times is not None and curve_volumes is not None:
        bottom.plot(curve_times, curve_volumes, "--", color=MODEL_COLOR, linewidth=1.2, label=curve_label)
    bottom.plot(
        moments, volumes, "o", markerfacecolor="none", markeredgecolor=MODEL_COLOR, markersize=7,
        markeredgewidth=1.5, label=volume_label,
    )
    bottom.plot(moments, reference_volumes, "s", color=OBSERVED_COLOR, markersize=6, label=reference_label)
    _finish_curve(bottom, params, stopping_time)
    bottom.set_ylabel("volume [mL]", fontsize=12)
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fit_dir = Path(args.fit_dir)
    spec = read_json(fit_dir / "spec.json")
    solver_mode = spec.get("solver", {}).get("mode", "standard")
    if solver_mode != SOLVER_MODE:
        raise ValueError(
            f"the fit {fit_dir} was run with --solver {solver_mode}; this script renders the fields of the two-species "
            f"solver (--solver {SOLVER_MODE}) only; scripts/visualize_patient_best_run.py renders the isotropic solver's."
        )
    scope = spec.get("objective", {}).get("scope", "full")
    if scope != "full":
        raise NotImplementedError(
            f"the fit {fit_dir} was run with --objective {scope}; this script renders the treated timeline "
            "(resection, fractions, chemotherapy) of a full-series fit only."
        )
    loss = spec.get("objective", {}).get("mode", "core")
    necrotic_scored = loss == NECROTIC_LOSS
    if loss not in DICE_RULES:
        raise NotImplementedError(f"the fit {fit_dir} was run with --loss {loss}; this script knows {sorted(DICE_RULES)}.")
    if necrotic_scored and args.core_threshold == "infer":
        raise ValueError("--core-threshold infer is not available for a core+necrotic fit (two profiled thresholds); use logged or a number.")
    config_path, fit, objective = fit_best_run(fit_dir)
    resolved = objective["resolved"]
    criterion_value = float(resolved[FIT_CRITERION])

    def recorded(key: str) -> float | None:
        value = resolved.get(key)
        return None if value is None or not np.isfinite(float(value)) else float(value)

    recorded_threshold = recorded("core_threshold_star")
    recorded_necrotic = recorded("necrotic_threshold_star")
    fit_logged_dice = {sid: float(v["dice_core_star"]) for sid, v in resolved["per_session"].items()}
    fit_logged_necrotic = {
        sid: float(v["dice_necrotic_star"]) for sid, v in resolved["per_session"].items() if v.get("dice_necrotic_star") is not None
    }
    print(
        f"best of fit {spec['name']} (--loss {loss}): restart {objective['restart']}, evaluation {objective['eval_id']}: "
        f"{FIT_CRITERION} = {criterion_value:.4f}"
    )
    run_name = args.run_name or (
        f"best_{spec['name']}_restart{objective['restart']}_eval{objective['eval_id']}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # The fit's own snapshot days for the session moments, plus the curve frames.
    config = read_config(config_path, solver=TwoCompartmentWithNutrientFKPPSolver)
    n_steps, dt = int(fit["n_steps"]), float(fit["dt"])
    moments = {sid: float(moment) for sid, moment in fit["snapshot_moments"].items()}
    session_days = {sid: float(day) for sid, day in fit["snapshot_days"].items()}
    sessions = [s for s in spec["patient"]["sessions"] if s["id"] in session_days]
    if len(sessions) != len(session_days):
        raise ValueError(f"spec.json lacks sessions of the fit's snapshot days {sorted(session_days)}.")
    stopping_time = float(config["resection_time"]) + float(config["time_after_resection"])
    if args.volume_curve:
        curve_days = np.arange(0.0, stopping_time, args.snapshot_interval_days)
        config["snapshot_times"] = sorted({*session_days.values(), *curve_days.tolist(), stopping_time})
    else:
        config["snapshot_times"] = sorted(set(session_days.values()))
    solver = TwoCompartmentWithNutrientFKPPSolver(config)
    params = solver.params
    nutrient_threshold = float(params["nutrient_threshold"])
    print(f"run directory: {run_dir}")
    print(f"{n_steps} steps (dt={dt:.4g} d), {len(config['snapshot_times'])} snapshots requested")
    result = solver.solve(store_result=True, outdir=run_dir)
    if not result.success:
        raise RuntimeError(f"solve failed: {result.error}")
    if result.n_steps != n_steps:
        raise RuntimeError(f"the solver used n_steps={result.n_steps}, not the planned {n_steps}.")
    times = np.asarray(result.snapshot_times, dtype=np.float64)
    frames = {field: result.time_series[field] for field in FIELDS}
    # The solver's tissue: where the initial nutrient is 1 (the CSF holds
    # none throughout, so it must not read as depleted).
    tissue = np.asarray(result.initial_state["nutrient"], dtype=np.float64) > 0
    affine = np.eye(4) if result.affine is None else np.asarray(result.affine, dtype=np.float64)
    shape = tuple(int(n) for n in frames[FIELDS[0]].shape[1:])
    voxel_volume_ml = float(np.prod(np.asarray(params["voxel_size_mm"], dtype=np.float64))) / 1000.0
    print(f"solve ({result.n_steps} steps, {times.size} frames per field): {result.wall_time_s:.1f} s")

    # Per session: the recorded frames (saved as the fit's resolve saves
    # them), the scored density, the reference masks, the display labels
    # and the background.
    patient_root = Path(args.patient_root) if args.patient_root else Path(spec["patient"]["root"])
    patient = spec["patient"]["patient"]
    fields: dict[str, list[NDArray]] = {field: [] for field in FIELDS}
    scored: list[NDArray] = []  # the core density per session (P + N or P under the core loss, P under core+necrotic)
    recorded_days: list[float] = []
    segmentations: list[NDArray] = []
    labels: list[NDArray] = []
    backgrounds: list[NDArray] = []
    for session in sessions:
        sid = session["id"]
        preop = session["label"] == LABEL_PREOP
        day = None
        for field in FIELDS:
            day, frame = recorded_frame(times, frames[field], session_days[sid])
            # Rounded for storage as the fit rounds its fields, so the Dice
            # and the volumes below equal the fit's.
            frame = round_field(frame)
            image = nib.Nifti1Image(frame, affine)
            image.set_data_dtype(np.float32)
            nib.save(image, str(run_dir / snapshot_file(sid, field)))
            fields[field].append(np.asarray(frame, dtype=np.float64))
        recorded_days.append(float(day))
        scored.append(region_densities({field: fields[field][-1] for field in FIELDS}, preop, SOLVER_MODE, loss)["core"])
        segmentation, _ = load_segmentation(session["segmentation"])
        if segmentation.shape != shape:
            raise ValueError(f"{session['segmentation']}: shape {segmentation.shape} differs from the run's grid {shape}.")
        segmentations.append(segmentation)
        labels.append(check_preop_labels(segmentation, session["segmentation"]) if preop else segmentation)
        backgrounds.append(load_volume(session_background(patient_root, patient, session), shape))
    # The fit's reference masks (the resection cavity excluded in every
    # later session); the overlay (labels) stays the raw segmentation. The
    # resection cavity drawn: the post-op session's label-4 voxels.
    postop_index = next(i for i, s in enumerate(sessions) if s["label"] == LABEL_POSTOP)
    resection_cavity = segmentations[postop_index] == LABEL_CAVITY
    cavity_spec = config["resection_cavity"]
    fit_cavity = load_segmentation(cavity_spec["segmentation"])[0] == int(cavity_spec["label"])
    if not np.array_equal(fit_cavity, resection_cavity):
        print(
            f"note: the fit's resection cavity ({cavity_spec['segmentation']}, label {cavity_spec['label']}) differs "
            f"from the post-op session's label {LABEL_CAVITY}; the masks use the fit's, the figures outline the latter"
        )
    references: list[Reference] = fit_session_references(segmentations, sessions, fit_cavity)
    if not same_label_conventions(spec.get("label_conventions")):
        print(
            "note: the fit was scored with older reference masks (its spec.json's label_conventions differ from "
            "FIT_LABEL_CONVENTIONS), so the Dice below differ from the fit's"
        )
    for reference in references:
        if reference.n_relabelled:
            print(f"  {reference.session}: {reference.n_relabelled} necrotic voxels counted as cavity (earlier cavities)")

    if args.core_threshold == "logged":
        threshold = recorded_threshold
        threshold_source = "logged"
        if threshold is None:
            if necrotic_scored:
                raise ValueError(f"{fit_dir}: the re-solve records no core threshold; pass --core-threshold <number>.")
            print("the fit records no core threshold; inferring it.")
            threshold, threshold_source = infer_threshold(scored, references), "inferred"
    elif args.core_threshold == "infer":
        threshold, threshold_source = infer_threshold(scored, references), "inferred"
    else:
        threshold, threshold_source = float(args.core_threshold), "given"
    necrotic_threshold: float | None = None
    necrotic_source = "not scored"
    if necrotic_scored:
        if args.necrotic_threshold == "logged":
            necrotic_threshold, necrotic_source = recorded_necrotic, "logged"
            if necrotic_threshold is None:
                raise ValueError(f"{fit_dir}: the re-solve records no necrotic threshold; pass --necrotic-threshold <number>.")
        else:
            necrotic_threshold, necrotic_source = float(args.necrotic_threshold), "given"
    print(
        f"core threshold {threshold:g} ({threshold_source}); "
        + (f"necrotic threshold {necrotic_threshold:g} ({necrotic_source}); " if necrotic_scored else "")
        + f"{DICE_RULES[loss]}; nutrient threshold {nutrient_threshold:g} (the config's)"
    )

    def masked_volume(mask: NDArray, valid: NDArray) -> float:
        return float((mask & valid).sum()) * voxel_volume_ml

    def field_mask(field: str, frame: NDArray) -> NDArray:
        """The voxels a field's volume counts: P at or above the core
        threshold, N at or above the necrotic threshold (the core one
        under the core loss), for the nutrient the depleted tissue."""
        if field == "nutrient":
            return (frame < nutrient_threshold) & tissue
        if field == "necrotic" and necrotic_scored:
            return frame >= necrotic_threshold
        return frame >= threshold

    dices = [
        dice(
            model_core_mask(
                core, fields["necrotic"][k] if necrotic_scored else None, s["label"] == LABEL_PREOP, threshold,
                necrotic_threshold if necrotic_threshold is not None else float("nan"),
            ) & r.valid,
            r.core,
        )
        for k, (core, s, r) in enumerate(zip(scored, sessions, references, strict=True))
    ]
    necrotic_dices: list[float] | None = None
    reference_necrotic_volumes: list[float] | None = None
    if necrotic_scored:
        # NaN where the session has no necrotic label (dropped by the fit).
        necrotic_dices = [
            dice((n >= necrotic_threshold) & r.valid, r.necrotic) if r.necrotic.any() else float("nan")
            for n, r in zip(fields["necrotic"], references, strict=True)
        ]
        reference_necrotic_volumes = [float(r.necrotic.sum()) * voxel_volume_ml for r in references]
    volumes = {
        field: [masked_volume(field_mask(field, f), r.valid) for f, r in zip(fields[field], references, strict=True)]
        for field in FIELDS
    }
    reference_volumes = [float(r.core.sum()) * voxel_volume_ml for r in references]
    curves: dict[str, NDArray] = {}
    if args.volume_curve:
        # Each curve frame is masked as the most recent session at or before
        # it masks its own frame (nothing before the first session), so the
        # curve passes through the session markers.
        session_frame_days = np.asarray(recorded_days, dtype=np.float64)
        for field in FIELDS:
            curve = np.empty(times.size, dtype=np.float64)
            for i, (day, frame) in enumerate(zip(times, frames[field], strict=True)):
                mask = field_mask(field, np.asarray(frame, dtype=np.float64))
                earlier = np.flatnonzero(session_frame_days <= day)
                if earlier.size:
                    mask &= references[int(earlier[np.argmax(session_frame_days[earlier])])].valid
                curve[i] = float(mask.sum()) * voxel_volume_ml
            curves[field] = curve
    preop_core = references[0].core
    if not preop_core.any():
        raise ValueError("the pre-op reference core is empty; pass --slice-z.")
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(preop_core)[2]))
    session_moments = [float(moments[s["id"]]) for s in sessions]
    session_postop_days = postop_days(sessions)
    for k, session in enumerate(sessions):
        logged = fit_logged_dice.get(session["id"])
        fit_note = "" if logged is None else f" (fit {logged:.3f})"
        necrotic_note = ""
        if necrotic_dices is not None:
            logged_n = fit_logged_necrotic.get(session["id"])
            necrotic_note = f", Dice necrotic {necrotic_dices[k]:.3f}" + ("" if logged_n is None else f" (fit {logged_n:.3f})")
            necrotic_note += f", segmented necrotic {reference_necrotic_volumes[k]:.2f} mL"
        print(
            f"  {session['id']}: Dice core {dices[k]:.3f}{fit_note}{necrotic_note}, P {volumes['proliferative'][k]:.2f} mL, "
            f"N {volumes['necrotic'][k]:.2f} mL, depleted {volumes['nutrient'][k]:.2f} mL, "
            f"segmented core {reference_volumes[k]:.2f} mL"
        )

    render_segmentations(
        run_dir / "segmentations", sessions, labels, backgrounds, z, session_moments, session_postop_days,
        reference_volumes, params, stopping_time, args.columns,
    )
    n_threshold = necrotic_threshold if necrotic_scored else threshold
    volume_labels = {
        "proliferative": f"model P >= {threshold:g} at scan (cavity excluded)",
        "necrotic": f"model N >= {n_threshold:g} at scan (cavity excluded)",
        "nutrient": f"depleted tissue (nutrient < {nutrient_threshold:g}) at scan (cavity excluded)",
    }
    for field in FIELDS:
        if field == "nutrient":
            overlays = [np.where(tissue, 1.0 - frame, 0.0) for frame in fields[field]]
            panel_dices, dice_label = None, ""
        elif field == "necrotic" and necrotic_scored:
            overlays, panel_dices, dice_label = fields[field], necrotic_dices, "Dice necrotic"
        else:
            overlays, panel_dices, dice_label = fields[field], dices, "Dice core"
        if field == "necrotic" and necrotic_scored:
            references_volumes, reference_label = reference_necrotic_volumes, REFERENCE_NECROTIC_LABEL
        else:
            references_volumes, reference_label = reference_volumes, REFERENCE_CORE_LABEL
        render_field(
            run_dir / field, sessions, overlays, resection_cavity, labels, backgrounds, z, args.display_threshold,
            COLORBAR_LABELS[field], session_moments, session_postop_days, panel_dices, dice_label, DICE_RULES[loss],
            volumes[field], volume_labels[field], references_volumes, reference_label, times if field in curves else None,
            curves.get(field), volume_labels[field].replace(" at scan", " along the run (latest scan's cavity excluded)"),
            params, stopping_time, args.columns,
        )
    write_json(
        run_dir / "run_summary.json",
        {
            "run_name": run_name,
            "cli_args": jsonable(vars(args)),
            "fit_dir": str(fit_dir.resolve()),
            "best_restart": objective["restart"],
            "best_eval_id": objective["eval_id"],
            "fit_loss": float(resolved["loss"]),
            "run_config": str(config_path),
            "criterion": FIT_CRITERION,
            "criterion_value": criterion_value,
            "solver": SOLVER_MODE,
            "fields": list(FIELDS),
            "loss": loss,
            "dice_rule": DICE_RULES[loss],
            "core_threshold": threshold,
            "core_threshold_source": threshold_source,
            "necrotic_threshold": necrotic_threshold,
            "necrotic_threshold_source": necrotic_source,
            "nutrient_threshold": nutrient_threshold,
            "depleted_rule": "nutrient < nutrient_threshold in tissue (initial nutrient 1), the session's excluded cavity removed",
            "slice_z": z,
            "voxel_volume_ml": voxel_volume_ml,
            "dt": result.dt,
            "n_steps": result.n_steps,
            "stopping_time": stopping_time,
            "sessions": [
                {
                    "id": s["id"],
                    "label": s["label"],
                    "date": s["date"],
                    "moment": m,
                    "postop_day": pd,
                    "recorded_day": d,
                    "dice_core": v,
                    "fit_dice_core": fit_logged_dice.get(s["id"]),
                    "dice_necrotic": None if necrotic_dices is None else necrotic_dices[k],
                    "fit_dice_necrotic": fit_logged_necrotic.get(s["id"]),
                    "reference_necrotic_volume_ml": None if reference_necrotic_volumes is None else reference_necrotic_volumes[k],
                    "model_proliferative_volume_ml": volumes["proliferative"][k],
                    "model_necrotic_volume_ml": volumes["necrotic"][k],
                    "depleted_tissue_volume_ml": volumes["nutrient"][k],
                    "reference_core_volume_ml": rv,
                    "n_cavity_excluded": r.n_cavity,
                    "n_necrotic_relabelled_cavity": r.n_relabelled,
                    "background": str(session_background(patient_root, patient, s)),
                    "files": {field: snapshot_file(s["id"], field) for field in FIELDS},
                }
                for k, (s, m, pd, d, v, rv, r) in enumerate(
                    zip(
                        sessions, session_moments, session_postop_days, recorded_days, dices, reference_volumes,
                        references, strict=True,
                    )
                )
            ],
            **(
                {"curve_days": times.tolist(), "curve_volume_ml": {field: curve.tolist() for field, curve in curves.items()}}
                if curves
                else {}
            ),
        },
    )
    print(
        f"saved {run_dir / 'segmentations.pdf'}, " + ", ".join(str(run_dir / f"{field}.pdf") for field in FIELDS)
        + " (+ .png, run_summary.json)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
