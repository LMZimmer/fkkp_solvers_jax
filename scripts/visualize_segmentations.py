#!/usr/bin/env python
"""Render one patient's tumour segmentations, session by session, into
<output-dir>/patient_summary.pdf (on the T1c) and
<output-dir>/patient_summary_flair.pdf (the same figure on the FLAIR):
the grid of scripts/visualize_patient_best_run.py's segmentations.pdf
with a row per session and the segmented core volume over time below
it, without any model run.

Sessions. The patient's rows of the session-labels tsv (default
/mnt/Drive4/lucas/SAILOR/session_labels.tsv), sorted by date; a session
without all three native images (below) is left out of both figures
and printed. Exactly one
session must be labelled postop: it is day 0 of the clinical day of every
row (the pre-op scan negative, from the tsv dates).

Images. Each session's own images, not the longitudinal (pre-op
registered) ones, below <root>/<patient>/<session>/ (default root
/mnt/Drive4/lucas/SAILOR/processed):
  skull_stripped/t1c_skullstripped.nii.gz      the T1c (patient_summary.pdf)
  skull_stripped/flair_skullstripped.nii.gz    the FLAIR (patient_summary_flair.pdf)
  tumor_segmentation/tumor_seg.nii.gz          the segmentation
checked to share their shape. Each image is drawn in gray from 0 to the
--window-percentile (default 99) of its non-zero voxels.

Grid. One row per session, titled on the left with the best-run
panels' "<session> <label> (day <n>)" broken after the session id (long
ids such as ses-20210406 would not fit one line); six columns, from left to right: the
axial image (T1c or FLAIR), the axial image with the segmentation
overlaid, then the same pair sagittal and coronal. The overlay is the raw labels in the
best-run script's palette and alpha (necrosis orange, edema blue,
enhancing violet, cavity green; alpha 0.6), named in a legend above the
grid. The slices go through a voxel per row (printed as x/y/z in the
plain image panels): --center preop (the default) uses the pre-op
session's tumour core centre of mass in every row, as segmentations.pdf
does (all sessions must then share the pre-op grid; on SAILOR's
processed data they are on one common grid, not registered to each
other beyond that); --center session uses each session's own. The core
centre is labels 1 + 3, falling back to the whole tumour (1-3), then to
any labelled voxel, then to the grid centre. The arrays are drawn as
segmentations.pdf draws them, without reorientation: axial
np.rot90(volume[:, :, z]), coronal np.rot90(volume[:, y, :]), sagittal
np.rot90(volume[x, :, :]) mirrored so the second axis runs right to left
(anterior on the left for the processed data's A second axis); each view
is centred on a square canvas of the grid's largest extent so the views
share one scale (scripts/visualize_sailor.py's oriented_slices).

Volume panel. Below the grid, as at the bottom of segmentations.pdf
(its legend above the panel rather than inside, where it would hide the
pre-op point), the
segmented core volume per session (red circles joined by a line, each
labelled with its session id) against the clinical day, with the
treatment events marked (fisher_kpp_jax.util.mark_treatment_events): the
patient sensitivity analysis's timeline (``build_timeline``: resection 3
days before the post-op scan, CRT from the first follow-up after it,
adjuvant TMZ cycles; the doses from its default config, they are not
drawn) over the pre-op session and the post-op / follow-up sessions after
it, shifted to the post-op day. A patient whose timeline cannot be built
(no pre-op session, no follow-up after the post-op one) gets the volumes
without events, and a note is printed. The core follows the segmented
core of segmentations.pdf: labels 1 + 3 in the pre-op session (and any
session not labelled postop or followup), label 3 alone in the post-op
and follow-up sessions, counted on the native segmentation (so no
cavity correction across sessions, which changes only necrotic voxels;
the volumes differ from the best-run script's by the registration
resampling of its longitudinal masks). A pre-op segmentation with label
4 (cavity) is drawn as it is and a note is printed (the patient scripts
refuse one). --yscale LOW HIGH fixes the panel's y limits in mL.

Run from the project root, e.g.:
  python scripts/visualize_segmentations.py --output-dir runs/segmentations_sub-01
  python scripts/visualize_segmentations.py --patient sub-03 --center session --output-dir runs/segmentations_sub-03
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from fisher_kpp_jax import read_config  # noqa: E402
from fisher_kpp_jax.util import mark_treatment_events  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    CORE_LABELS,
    DEFAULT_CONFIGS,
    DEFAULT_PATIENT,
    DEFAULT_PATIENT_ROOT,
    DEFAULT_SESSION_LABELS,
    DEFAULT_SOLVER_MODE,
    LABEL_CAVITY,
    LABEL_FOLLOWUP,
    LABEL_POSTOP,
    LABEL_PREOP,
    POSTOP_CORE_LABELS,
    PREOP_SEGMENTATION_FILE,
    SOLVER_MODES,
    Session,
    build_timeline,
    load_segmentation,
    protocol_from_config,
    read_session_labels,
)
from visualize_patient_best_run import (  # noqa: E402
    LABEL_COLORS,
    LABEL_NAMES,
    PREOP_T1C,
    REFERENCE_COLOR,
    REFERENCE_CORE_LABEL,
    SEGMENTATION_ALPHA,
    session_title,
)
from visualize_sailor import display_window, oriented_slices, tumor_center  # noqa: E402

SEGMENTATION_FILE = Path(PREOP_SEGMENTATION_FILE)
# The background images (name in the panel titles, file below the
# session directory) and the figure each one is drawn in.
MODALITIES: dict[str, tuple[str, Path]] = {
    "t1c": ("T1c", PREOP_T1C),
    "flair": ("FLAIR", Path("skull_stripped") / "flair_skullstripped.nii.gz"),
}
OUTPUT_FILES: dict[str, str] = {"t1c": "patient_summary.pdf", "flair": "patient_summary_flair.pdf"}
CENTERS = ("preop", "session")
DEFAULT_WINDOW_PERCENTILE = 99.0
# Column order: per view the plain T1c, then the T1c with the overlay.
VIEWS = ("axial", "sagittal", "coronal")
VIEW_INDEX = {"axial": 2, "sagittal": 0, "coronal": 1}
LATER_LABELS = (LABEL_POSTOP, LABEL_FOLLOWUP)
PANEL_INCHES = 3.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", required=True, help=f"directory of {' and '.join(OUTPUT_FILES.values())} (created)"
    )
    parser.add_argument("--patient", default=DEFAULT_PATIENT, help=f"subject id (default {DEFAULT_PATIENT})")
    parser.add_argument("--patient-root", default=str(DEFAULT_PATIENT_ROOT), help="processed data root")
    parser.add_argument("--session-labels", default=str(DEFAULT_SESSION_LABELS), help="session-labels tsv")
    parser.add_argument(
        "--center",
        choices=CENTERS,
        default="preop",
        help="'preop' (default): every row through the pre-op core centre of mass; "
        "'session': each row through its own",
    )
    parser.add_argument(
        "--window-percentile",
        type=float,
        default=DEFAULT_WINDOW_PERCENTILE,
        help=f"image display window: black at 0, white at this percentile of the non-zero voxels (default {DEFAULT_WINDOW_PERCENTILE:g})",
    )
    parser.add_argument(
        "--yscale", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
        help="y limits in mL of the volume panel (default: matplotlib's autoscale)",
    )
    return parser.parse_args(argv)


@dataclass
class Row:
    """A drawn session: its background images and their display maxima
    by modality, slice centre and segmented core volume."""

    session: Session
    day: int
    images: dict[str, NDArray]
    vmax: dict[str, float]
    labels: NDArray
    core_ml: float
    center: tuple[int, int, int] | None = None
    center_source: str | None = None


def session_files(root: Path, patient: str, session: Session) -> tuple[dict[str, Path], Path]:
    """The session's background images by modality and its segmentation."""
    base = root / patient / session.id
    return {key: base / relative for key, (_, relative) in MODALITIES.items()}, base / SEGMENTATION_FILE


def core_labels(session: Session) -> tuple[int, ...]:
    """The segmented core of segmentations.pdf: enhancing alone after the
    resection, necrotic + enhancing otherwise."""
    return POSTOP_CORE_LABELS if session.label in LATER_LABELS else CORE_LABELS


def load_row(root: Path, patient: str, session: Session, day: int, window_percentile: float) -> Row:
    image_paths, segmentation_path = session_files(root, patient, session)
    segmentation, segmentation_image = load_segmentation(segmentation_path)
    images = {}
    for key, path in image_paths.items():
        images[key] = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if images[key].shape != segmentation.shape:
            raise ValueError(f"{path}: shape {images[key].shape} differs from the segmentation's {segmentation.shape}.")
    if session.label == LABEL_PREOP and (segmentation == LABEL_CAVITY).any():
        print(
            f"note: {segmentation_path}: the pre-op segmentation has {int((segmentation == LABEL_CAVITY).sum())} "
            f"label-{LABEL_CAVITY} (cavity) voxels; drawn as they are, not counted as core"
        )
    voxel_ml = float(np.prod(segmentation_image.header.get_zooms()[:3])) / 1000.0
    core_ml = float(np.isin(segmentation, core_labels(session)).sum()) * voxel_ml
    vmax = {key: display_window(image, window_percentile) for key, image in images.items()}
    return Row(session, day, images, vmax, segmentation.astype(np.uint8), core_ml)


def assign_centers(rows: list[Row], mode: str) -> None:
    """Every row's slice centre (``tumor_center`` of the pre-op row or of
    the row itself, the grid centre without labelled voxels)."""
    preop = next((r for r in rows if r.session.label == LABEL_PREOP), None)
    if mode == "preop":
        if preop is None:
            raise ValueError("--center preop: no pre-op session is drawn; pass --center session.")
        shapes = sorted({r.labels.shape for r in rows if r.labels.shape != preop.labels.shape})
        if shapes:
            raise ValueError(
                f"--center preop: shape(s) {shapes} differ from the pre-op grid {preop.labels.shape}; pass --center session."
            )
    for row in rows:
        source_row = preop if mode == "preop" else row
        center, source = tumor_center(source_row.labels, source_row.session.label == LABEL_PREOP)
        if center is None:
            center, source = tuple(int(n // 2) for n in row.labels.shape[:3]), "grid"
        row.center, row.center_source = center, source


def treatment_days(sessions: list[Session], postop: Session) -> dict[str, Any] | None:
    """
    The treatment events of the patient's timeline (``build_timeline``
    over the pre-op session and the post-op / follow-up sessions after it)
    as days relative to the post-op scan; None, with a printed note, when
    it cannot be built.
    """
    preops = [s for s in sessions if s.label == LABEL_PREOP]
    if len(preops) != 1:
        print(f"note: {len(preops)} pre-op sessions; no treatment events drawn")
        return None
    preop = preops[0]
    later = sorted((s for s in sessions if s.label in LATER_LABELS and s.date > preop.date), key=lambda s: s.date)
    base = read_config(DEFAULT_CONFIGS[DEFAULT_SOLVER_MODE], solver=SOLVER_MODES[DEFAULT_SOLVER_MODE])
    try:
        timeline = build_timeline([preop, *later], protocol_from_config(base))
    except ValueError as exc:
        print(f"note: no treatment timeline ({exc}); no treatment events drawn")
        return None
    return timeline.model_days(-float((postop.date - preop.date).days))


def slice_text(view: str, center: tuple[int, int, int]) -> str:
    return f"{'xyz'[VIEW_INDEX[view]]}={center[VIEW_INDEX[view]]}"


def render(
    outfile: Path,
    modality: str,
    rows: list[Row],
    events: dict[str, Any] | None,
    yscale: tuple[float, float] | None = None,
) -> None:
    n_col = 2 * len(VIEWS)
    n_row = len(rows)
    fig = plt.figure(figsize=(PANEL_INCHES * n_col + 0.8, PANEL_INCHES * n_row + 4.0), constrained_layout=True)
    grid = fig.add_gridspec(n_row + 1, n_col, height_ratios=[1.0] * n_row + [4.0 / PANEL_INCHES])
    fig.legend(
        handles=[Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k]) for k in sorted(LABEL_COLORS)],
        loc="outside upper center", ncol=len(LABEL_COLORS), fontsize=10, frameon=False,
    )
    colors = [(0, 0, 0, 0)] + [LABEL_COLORS[k] for k in sorted(LABEL_COLORS)]
    cmap = ListedColormap(colors)
    name = MODALITIES[modality][0]
    for i, row in enumerate(rows):
        backgrounds = oriented_slices(row.images[modality], row.center)
        overlays = oriented_slices(row.labels, row.center)
        for j, view in enumerate(VIEWS):
            plain = fig.add_subplot(grid[i, 2 * j])
            overlaid = fig.add_subplot(grid[i, 2 * j + 1])
            for ax in (plain, overlaid):
                ax.imshow(backgrounds[view], cmap="gray", vmin=0.0, vmax=row.vmax[modality], interpolation="none")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            overlaid.imshow(
                overlays[view], cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5,
                alpha=SEGMENTATION_ALPHA, interpolation="none",
            )
            plain.text(
                0.02, 0.02, slice_text(view, row.center), transform=plain.transAxes, color="white", fontsize=9, va="bottom"
            )
            if i == 0:
                plain.set_title(f"{view} {name}", fontsize=12, fontweight="bold", pad=8)
                overlaid.set_title(f"{view} {name} + segmentation", fontsize=12, fontweight="bold", pad=8)
            if j == 0:
                title = session_title({"id": row.session.id, "label": row.session.label}, row.day)
                plain.set_ylabel(title.replace(" ", "\n", 1), fontsize=12, fontweight="bold")
    bottom = fig.add_subplot(grid[n_row, :])
    days = [row.day for row in rows]
    volumes = [row.core_ml for row in rows]
    bottom.plot(days, volumes, "o-", color=REFERENCE_COLOR, markersize=6, label=REFERENCE_CORE_LABEL)
    for row in rows:
        bottom.annotate(
            row.session.id, (row.day, row.core_ml), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=REFERENCE_COLOR,
        )
    low, high = bottom.get_ylim()
    bottom.set_ylim(low, high + 0.08 * (high - low))  # room for the label above the highest point
    if events is not None:
        mark_treatment_events(bottom, events)
    first, last = min(days), max(days)
    margin = 0.01 * max(last - first, 1)
    bottom.set_xlim(first - margin, last + margin)
    if yscale is not None:
        bottom.set_ylim(*yscale)
    bottom.set_xlabel("day relative to the post-op scan", fontsize=12)
    bottom.set_ylabel("core volume [mL]", fontsize=12)
    bottom.grid(alpha=0.3)
    bottom.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, frameon=False)
    fig.savefig(str(outfile), format="pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.patient_root).resolve()
    sessions = sorted(read_session_labels(args.session_labels, args.patient), key=lambda s: s.date)
    postops = [s for s in sessions if s.label == LABEL_POSTOP]
    if len(postops) != 1:
        raise ValueError(f"expected one session labelled {LABEL_POSTOP}, got {[s.id for s in postops]}.")
    postop = postops[0]
    rows: list[Row] = []
    for session in sessions:
        image_paths, segmentation_path = session_files(root, args.patient, session)
        missing = [str(path) for path in (*image_paths.values(), segmentation_path) if not path.is_file()]
        if missing:
            print(f"  {session.id} ({session.label}, {session.date}): left out, missing {missing}")
            continue
        rows.append(load_row(root, args.patient, session, (session.date - postop.date).days, args.window_percentile))
    if not rows:
        raise FileNotFoundError(f"no session of {args.patient} has all native images below {root / args.patient}.")
    assign_centers(rows, args.center)
    events = treatment_days(sessions, postop)
    for row in rows:
        x, y, z = row.center
        print(
            f"  {row.session.id} {row.session.label} (day {row.day}): core {row.core_ml:.2f} mL, "
            f"slices x={x} y={y} z={z} ({args.center} {row.center_source} centre)"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for modality, filename in OUTPUT_FILES.items():
        outfile = output_dir / filename
        render(outfile, modality, rows, events, args.yscale)
        print(f"saved {outfile.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
