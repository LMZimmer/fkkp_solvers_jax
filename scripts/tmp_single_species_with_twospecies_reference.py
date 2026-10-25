#!/usr/bin/env python
"""TEMPORARY. Redraw the model figure of a rendered single-species run
(scripts/visualize_patient_best_run.py output directory) with the segmented
core volumes (the hollow red circles of the volume panel) taken from a
rendered two-species run (scripts/visualize_patient_best_run_twospecies.py
output directory), so a fit made on the old sub-01 processing is shown
against the reprocessed segmentation's volumes. Nothing is re-solved: the
session frames, the Dice, the model volumes, the slice and the backgrounds
are the single-species run's (its run_summary.json and saved frames); only
the reference volumes of the volume panel are swapped, matched by session
id. The panels' red contours remain the single-species run's segmentations.

  python scripts/tmp_single_species_with_twospecies_reference.py \\
      --single-run runs/best_sub01_fit_2026-09-28_v4 \\
      --twospecies-run runs/best_sub01_twospecies_fit_2026-10-01_v2 \\
      --output runs/best_sub01_fit_2026-09-28_v4/model_twospecies_reference --yscale 0 33
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from patient_sensitivity_analysis import LABEL_CAVITY, LABEL_POSTOP, LABEL_PREOP, check_preop_labels, load_segmentation  # noqa: E402
from visualize_patient_best_run import load_volume, render_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single-run", required=True, help="rendered single-species run directory")
    parser.add_argument("--twospecies-run", required=True, help="rendered two-species run directory")
    parser.add_argument("--output", required=True, help="output stem (.pdf and .png are appended)")
    parser.add_argument("--yscale", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--display-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)

    single = Path(args.single_run)
    summary = json.load(open(single / "run_summary.json"))
    other = json.load(open(Path(args.twospecies_run) / "run_summary.json"))
    other_reference = {s["id"]: float(s["reference_core_volume_ml"]) for s in other["sessions"]}
    config = json.load(open(single / "config.json"))
    spec = json.load(open(Path(summary["fit_dir"]) / "spec.json"))
    spec_sessions = {s["id"]: s for s in spec["patient"]["sessions"]}

    sessions = [spec_sessions[s["id"]] for s in summary["sessions"]]
    shape = None
    fields, labels, backgrounds, segmentations = [], [], [], []
    for record, session in zip(summary["sessions"], sessions, strict=True):
        frame = np.asarray(nib.load(str(single / record["file"])).get_fdata(), dtype=np.float64)
        shape = frame.shape
        fields.append(frame)
        segmentation, _ = load_segmentation(session["segmentation"])
        segmentations.append(segmentation)
        labels.append(check_preop_labels(segmentation, session["segmentation"]) if session["label"] == LABEL_PREOP else segmentation)
        backgrounds.append(load_volume(Path(record["background"]), shape))
    postop_index = next(i for i, s in enumerate(sessions) if s["label"] == LABEL_POSTOP)
    resection_cavity = segmentations[postop_index] == LABEL_CAVITY

    reference_volumes = [other_reference[s["id"]] for s in sessions]
    for record, volume in zip(summary["sessions"], reference_volumes, strict=True):
        print(f"  {record['id']}: segmented core {record['reference_core_volume_ml']:.2f} mL -> {volume:.2f} mL (two-species run)")
    params = {"resection_time": config["resection_time"], "rt_times": config["rt_times"], "chemo_times": config["chemo_times"]}
    render_model(
        Path(args.output), sessions, fields, resection_cavity, labels, backgrounds, int(summary["slice_z"]),
        float(summary["core_threshold"]), args.display_threshold,
        [float(s["moment"]) for s in summary["sessions"]], [int(s["postop_day"]) for s in summary["sessions"]],
        [float(s["dice_core"]) for s in summary["sessions"]], [float(s["model_core_volume_ml"]) for s in summary["sessions"]],
        reference_volumes, None, None, params, float(summary["stopping_time"]), args.columns,
        None if args.yscale is None else tuple(args.yscale),
    )
    print(f"saved {args.output}.pdf and {args.output}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
