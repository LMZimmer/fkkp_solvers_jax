#!/usr/bin/env python
"""Copy-last baseline for a patient's longitudinal tumour core: every
session is predicted by the previous session's observation, the pre-op
session by itself. Scored and drawn like scripts/visualize_patient_best_run.py
renders a scripts/patient_cmaes_fit.py fit.

Input (--fit-dir): a full-objective fit directory of the isotropic solver
(spec.json objective.scope "full", solver.mode "standard"). It supplies
the patient, the sessions (those the fit simulated, from best/config.json's
"_fit" record), the resection cavity of the masks (best/config.json's
resection_cavity) and the volume panel's time axis: the session moments
and the treatment events of the fit's best config, in model days (the seed
is day 0, so the axis carries the fit's pre-op time; the prediction itself
does not depend on the fit). Nothing is solved.

Masks: the fit's reference masks (patient_cmaes_fit.fit_session_references):
the reference core is labels 1 + 3 (necrotic + enhancing) in the pre-op
session and label 3 (enhancing) in every later one; in a later session the
session's cavity, the earlier sessions' cavities and the resection cavity
are excluded (valid). Prediction of session k > 0: the reference core of
session k - 1; of the pre-op session: its own reference core (Dice 1).
Dice of session k: dice(prediction & valid_k, core_k), NaN when both are
empty (left out of every mean), 0 when one is.

Two figures, the layout of visualize_patient_best_run.py:
  segmentations   identical to that script's (the raw labels per session
                  and the reference core volumes)
  model           the prediction mask (value 1, inferno) over the session's
                  T1c with the observed core (labels 1 + 3, red) and the
                  resection cavity (green) outlined and the per-session
                  Dice; below, the predicted core volume per session
                  (valid voxels only) as filled black squares and the
                  reference core volumes as hollow red circles
Written into <output-dir>/<run-name>/ (exist_ok=False): segmentations.pdf/.png,
model.pdf/.png and run_summary.json (per session the predecessor, Dice,
predicted and reference core volume; the mean Dice over the sessions after
the pre-op one and over all sessions).

Run from the project root, e.g.:
  python scripts/copy_last_eval.py \\
      --fit-dir /mnt/Drive4/lucas/stupp_patient_fit_respond/GR149_1seed \\
      --output-dir runs/copy_last_2026-11-09 --run-name GR149
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Nothing is solved; keep jax (imported through fisher_kpp_jax) off the GPUs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver, read_config  # noqa: E402
from fisher_kpp_jax.config import jsonable  # noqa: E402
from patient_cmaes_fit import fit_session_references  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    LABEL_CAVITY,
    LABEL_POSTOP,
    LABEL_PREOP,
    check_preop_labels,
    dice,
    load_segmentation,
)
from sensitivity_analysis import read_json, write_json  # noqa: E402
from visualize_patient_best_run import (  # noqa: E402
    fit_best_run,
    load_volume,
    postop_days_of,
    render_model,
    render_segmentations,
    session_background,
)

PREDICTION_LABEL = "copy-last prediction (previous scan's core, cavity excluded)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fit-dir", required=True,
        help="scripts/patient_cmaes_fit.py fit directory (spec.json, best/config.json): patient, sessions, time axis",
    )
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument("--run-name", default=None, help="run directory name (default: copy_last_<patient>)")
    parser.add_argument("--patient-root", default=None, help="patient data root (default: spec.json's)")
    parser.add_argument("--slice-z", type=int, default=None, help="axial slice (default: pre-op core centre of mass)")
    parser.add_argument("--columns", type=int, default=4, help="panels per row (default 4)")
    parser.add_argument(
        "--yscale", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
        help="y limits in mL of the volume panels of both figures (default: matplotlib's autoscale)",
    )
    return parser.parse_args(argv)


def nanmean_or_none(values: list[float]) -> float | None:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fit_dir = Path(args.fit_dir)
    spec = read_json(fit_dir / "spec.json")
    solver_mode = spec.get("solver", {}).get("mode", "standard")
    scope = spec.get("objective", {}).get("scope", "full")
    if solver_mode != "standard" or scope != "full":
        raise NotImplementedError(
            f"the fit {fit_dir} was run with --solver {solver_mode} --objective {scope}; this script takes the "
            "masks and sessions of a standard full-objective fit."
        )
    config_path, fit, _ = fit_best_run(fit_dir)
    config = read_config(config_path, solver=StuppFKPPSolver)
    moments = {sid: float(moment) for sid, moment in fit["snapshot_moments"].items()}
    sessions = [s for s in spec["patient"]["sessions"] if s["id"] in moments]
    if len(sessions) != len(moments):
        raise ValueError(f"spec.json lacks sessions of the fit's snapshot moments {sorted(moments)}.")
    if sessions[0]["label"] != LABEL_PREOP:
        raise ValueError(f"the first session {sessions[0]['id']} is not labelled {LABEL_PREOP}.")
    stopping_time = float(config["resection_time"]) + float(config["time_after_resection"])

    patient = spec["patient"]["patient"]
    run_name = args.run_name or f"copy_last_{patient}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    patient_root = Path(args.patient_root) if args.patient_root else Path(spec["patient"]["root"])

    segmentations: list[NDArray] = []
    labels: list[NDArray] = []
    backgrounds: list[NDArray] = []
    voxel_size = None
    for session in sessions:
        segmentation, image = load_segmentation(session["segmentation"])
        if segmentations and segmentation.shape != segmentations[0].shape:
            raise ValueError(f"{session['segmentation']}: shape {segmentation.shape} differs from the pre-op {segmentations[0].shape}.")
        if voxel_size is None:
            voxel_size = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
        segmentations.append(segmentation)
        labels.append(check_preop_labels(segmentation, session["segmentation"]) if session["label"] == LABEL_PREOP else segmentation)
        backgrounds.append(load_volume(session_background(patient_root, patient, session), segmentation.shape))
    voxel_volume_ml = float(np.prod(voxel_size)) / 1000.0

    postop_index = next(i for i, s in enumerate(sessions) if s["label"] == LABEL_POSTOP)
    resection_cavity = segmentations[postop_index] == LABEL_CAVITY
    cavity_spec = config["resection_cavity"]
    fit_cavity = load_segmentation(cavity_spec["segmentation"])[0] == int(cavity_spec["label"])
    references = fit_session_references(segmentations, sessions, fit_cavity)

    # Session k is predicted by session k - 1's reference core, the pre-op session by its own.
    predecessors = [0] + list(range(len(sessions) - 1))
    predictions = [references[j].core & references[k].valid for k, j in enumerate(predecessors)]
    dices = [dice(p, r.core) for p, r in zip(predictions, references, strict=True)]
    predicted_volumes = [float(p.sum()) * voxel_volume_ml for p in predictions]
    reference_volumes = [float(r.core.sum()) * voxel_volume_ml for r in references]

    preop_core = references[0].core
    if not preop_core.any():
        raise ValueError("the pre-op reference core is empty; pass --slice-z.")
    z = args.slice_z if args.slice_z is not None else int(round(center_of_mass(preop_core)[2]))
    session_moments = [moments[s["id"]] for s in sessions]
    session_postop_days = postop_days_of(spec["patient"]["sessions"], sessions)
    for k, (session, value) in enumerate(zip(sessions, dices, strict=True)):
        print(
            f"  {session['id']} <- {sessions[predecessors[k]]['id']}: Dice core {value:.3f}, "
            f"predicted {predicted_volumes[k]:.2f} mL, segmented {reference_volumes[k]:.2f} mL"
        )
    mean_followup = nanmean_or_none(dices[1:])
    mean_all = nanmean_or_none(dices)
    print(f"mean Dice core after the pre-op session: {mean_followup}; over all sessions: {mean_all}")

    render_segmentations(
        run_dir / "segmentations", sessions, labels, backgrounds, z, session_moments, session_postop_days,
        reference_volumes, config, stopping_time, args.columns, args.yscale,
    )
    render_model(
        run_dir / "model", sessions, [p.astype(np.float64) for p in predictions], resection_cavity, labels,
        backgrounds, z, 0.5, 0.5, session_moments, session_postop_days, dices, predicted_volumes,
        reference_volumes, None, None, config, stopping_time, args.columns, args.yscale,
        model_label=PREDICTION_LABEL, colorbar_label="copy-last prediction",
    )
    summary: dict[str, Any] = {
        "run_name": run_name,
        "cli_args": jsonable(vars(args)),
        "fit_dir": str(fit_dir.resolve()),
        "patient": patient,
        "method": "copy-last: session k predicted by session k-1's reference core, the pre-op session by itself",
        "slice_z": z,
        "voxel_volume_ml": voxel_volume_ml,
        "mean_dice_core_after_preop": mean_followup,
        "mean_dice_core_all": mean_all,
        "sessions": [
            {
                "id": s["id"],
                "label": s["label"],
                "date": s["date"],
                "moment": m,
                "postop_day": pd,
                "predicted_from": sessions[j]["id"],
                "dice_core": v,
                "predicted_core_volume_ml": pv,
                "reference_core_volume_ml": rv,
                "n_cavity_excluded": r.n_cavity,
                "n_necrotic_relabelled_cavity": r.n_relabelled,
            }
            for s, m, pd, j, v, pv, rv, r in zip(
                sessions, session_moments, session_postop_days, predecessors, dices, predicted_volumes,
                reference_volumes, references, strict=True,
            )
        ],
    }
    write_json(run_dir / "run_summary.json", summary)
    print(f"saved {run_dir / 'segmentations.pdf'} and {run_dir / 'model.pdf'} (+ .png, run_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
