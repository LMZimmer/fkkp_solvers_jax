#!/usr/bin/env python
"""Render every SAILOR subject's longitudinal tumour segmentations into
one PDF: a page per subject, a row per session, and in each row the
axial, coronal and sagittal T1c slice through the session's tumour core
centre of mass with the segmentation overlaid; a last row per subject
overlays the later exams' resection cavities as contours on the pre-op
T1c.

Sessions come from the session-labels tsv (default
/mnt/Drive4/lucas/SAILOR/session_labels.tsv, columns subject, session,
label, date and, when present, day_offset), in file order; --subjects
restricts the pages. A session's images are taken below
<root>/<subject>/<session>/ (default root /mnt/Drive4/lucas/SAILOR/processed):
  longitudinal  longitudinal/t1c_warped_longitudinal.nii.gz with
                longitudinal/recurrence_preop.nii.gz: the session registered
                to the subject's pre-op space, as the patient sensitivity
                analysis and scripts/visualize_patient_best_run.py use it;
                exists for the later sessions of subjects with a pre-op scan
  native        skull_stripped/t1c_skullstripped.nii.gz with
                tumor_segmentation/tumor_seg.nii.gz: the session's own
                images on the common 1 mm grid; every processed session has
                them, the pre-op sessions only these
--space longitudinal (the default) draws the longitudinal pair when both
files exist and the native pair otherwise; --space native always draws
the native pair. Each row names the pair it shows. A tsv session without
a processed directory or without both files of a pair is listed as
missing in the page header instead of getting a row ("old" exams and a
few follow-ups). All volumes must share the grid of the first one
loaded and the LAS axis orientation of the processed data (checked from
the headers).

Slices. The tumour core is labels 1 (necrotic) and 3 (enhancing), a
pre-op segmentation relabelled 4 -> 3 as the patient sensitivity
analysis does (``relabel_preop``); the cavity (label 4) of a later
session is not core. --center session (the default) takes the session's
own core centre of mass, rounded to a voxel; a session without core
voxels falls back to its whole tumour (labels 1-3), then to any labelled
voxel, then to the subject's reference centre (the first session with
a core), then to the grid centre, and its row says which. --center
subject uses the reference centre for every row, so the rows show the
same anatomy. The axial and coronal panels are in radiological view
(the subject's right on the left of the image, anterior / superior up,
np.rot90 of the array slice as in the best-run script); the sagittal
panel has anterior on the left.

Overlay. The segmentation labels in the palette of
scripts/visualize_patient_best_run.py (necrosis orange, edema blue,
enhancing violet, cavity green) at --alpha (default 0.6) over the T1c
in gray, windowed per session from 0 to the --window-percentile (default
99) of its non-zero voxels; the three views are centred on a square
canvas of the grid's largest extent so they share one scale; a legend
on each page. The row label carries the session id,
its tsv label, the date (a trailing '*' marks an approximate tsv date),
the tsv day offset when the column exists, the image pair, the slice
indices and the core and whole-tumour volumes in mL.

Cavity overview. Below a subject's last session one more row shows the
pre-op session's T1c (without a drawn pre-op session, the first drawn
exam, and the row says so) through the pre-op tumour core centre of
mass (the same fallbacks as above) with the cavity, label 4, of every
later session labelled postop or follow-up as one contour, coloured by
exam order along the viridis colormap and listed in the row's legend
(id, date, day offset); an exam without cavity voxels on the three
slices leaves no contour. The masks are taken as drawn, so under
--space longitudinal they lie in the pre-op space; a subject without a
pre-op scan has native pairs only, all on the common grid but not
registered to each other beyond that.

Pages. One page per subject; --rows-per-page splits a subject with more
rows (sessions plus the overview) over several pages (header "page k/n"). The PDF is written to
--output (its parent is created); the document metadata records the
data root and the options. A session whose files fail to load is drawn
as an empty row carrying the error text so the rest of the document
still renders; every such row is also printed.

Run from the project root, e.g.:
  python scripts/visualize_sailor.py --output /mnt/Drive4/lucas/SAILOR/sailor_sessions.pdf
  python scripts/visualize_sailor.py --output sailor_sub-01.pdf --subjects sub-01 --center subject
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from patient_sensitivity_analysis import (  # noqa: E402
    CORE_LABELS,
    DEFAULT_PATIENT_ROOT,
    DEFAULT_SESSION_LABELS,
    LABEL_CAVITY,
    LABEL_FOLLOWUP,
    LABEL_POSTOP,
    LABEL_PREOP,
    LATER_SEGMENTATION_FILE,
    PREOP_SEGMENTATION_FILE,
    SESSION_COLUMNS,
    WHOLE_LABELS,
    load_segmentation,
    normalise_label,
    relabel_preop,
)
from visualize_patient_best_run import (  # noqa: E402
    LABEL_COLORS,
    LABEL_NAMES,
    LATER_T1C,
    PREOP_T1C,
    SEGMENTATION_ALPHA,
)

DAY_OFFSET_COLUMN = "day_offset"
# The image pairs (T1c, segmentation) below <root>/<subject>/<session>/.
PAIRS: dict[str, tuple[Path, Path]] = {
    "longitudinal": (LATER_T1C, Path(LATER_SEGMENTATION_FILE)),
    "native": (PREOP_T1C, Path(PREOP_SEGMENTATION_FILE)),
}
SPACES = ("longitudinal", "native")
CENTERS = ("session", "subject")
# The processed data's axis orientation (nibabel axis codes): x runs to
# the subject's left, y anterior, z superior. The slice views below
# assume it.
EXPECTED_AXCODES = ("L", "A", "S")
VIEWS = ("axial", "coronal", "sagittal")
DEFAULT_WINDOW_PERCENTILE = 99.0
# The cavity-overview row: one contour per post-op / follow-up exam,
# coloured by exam order along this colormap.
CAVITY_LABELS = (LABEL_POSTOP, LABEL_FOLLOWUP)
CAVITY_CMAP = "viridis"
CAVITY_LINEWIDTH = 0.9
PANEL_INCHES = 3.0
LABEL_COLUMN_INCHES = 2.1
HEADER_INCHES = 1.1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, help="the PDF written (parent directories are created)")
    parser.add_argument("--patient-root", default=str(DEFAULT_PATIENT_ROOT), help="processed data root")
    parser.add_argument("--session-labels", default=str(DEFAULT_SESSION_LABELS), help="session-labels tsv")
    parser.add_argument(
        "--subjects",
        default=None,
        help="subjects to render: a comma-separated list of ids ('sub-01,sub-03') or numbers ('1,3') "
        "or a range ('01-05'); default all subjects of the tsv",
    )
    parser.add_argument(
        "--space",
        choices=SPACES,
        default="longitudinal",
        help="'longitudinal' (default): the pre-op-registered pair when it exists, else the session's own; "
        "'native': always the session's own pair",
    )
    parser.add_argument(
        "--center",
        choices=CENTERS,
        default="session",
        help="'session' (default): each row through its own core centre of mass; "
        "'subject': every row through the subject's reference centre",
    )
    parser.add_argument(
        "--rows-per-page", type=int, default=None, help="split a subject over pages of this many rows (default: one page)"
    )
    parser.add_argument("--alpha", type=float, default=SEGMENTATION_ALPHA, help=f"overlay alpha (default {SEGMENTATION_ALPHA})")
    parser.add_argument(
        "--window-percentile",
        type=float,
        default=DEFAULT_WINDOW_PERCENTILE,
        help=f"T1c display window: black at 0, white at this percentile of the non-zero voxels (default {DEFAULT_WINDOW_PERCENTILE})",
    )
    parser.add_argument("--dpi", type=int, default=100, help="raster resolution of the panels in the PDF (default 100)")
    args = parser.parse_args(argv)
    if args.rows_per_page is not None and args.rows_per_page < 1:
        parser.error("--rows-per-page must be at least 1.")
    return args


def parse_subjects(text: str) -> list[str]:
    """
    The subject ids of a --subjects argument: a range "01-05" (sub-01 to
    sub-05) or a comma-separated list of ids ("sub-01,sub-03") or
    numbers ("1,3").
    """
    text = text.strip()
    if not text:
        raise ValueError("--subjects must name at least one subject.")
    if "," not in text and "-" in text and not text.startswith("sub"):
        first, _, last = text.partition("-")
        lo, hi = int(first), int(last)
        if lo > hi:
            raise ValueError(f"--subjects: the range {text!r} is empty.")
        return [f"sub-{n:02d}" for n in range(lo, hi + 1)]
    ids = []
    for item in (part.strip() for part in text.split(",") if part.strip()):
        ids.append(item if item.startswith("sub") else f"sub-{int(item):02d}")
    return ids


@dataclass(frozen=True)
class SessionRow:
    """
    One tsv row.

    Attributes:
        subject: The subject id.
        id: The session id.
        label: The normalised label (preop, postop, followup, old, ...).
        raw_label: The tsv label as written.
        date: The tsv date without the '*' mark.
        approximate: Whether the tsv marks the date with a trailing '*'.
        day_offset: The tsv day offset, None without the column or value.
    """

    subject: str
    id: str
    label: str
    raw_label: str
    date: str
    approximate: bool
    day_offset: int | None

    @property
    def preop(self) -> bool:
        return self.label == LABEL_PREOP

    def title(self) -> str:
        approx = "*" if self.approximate else ""
        day = "" if self.day_offset is None else f" (day {self.day_offset})"
        return f"{self.id} {self.raw_label}\n{self.date}{approx}{day}"


def read_sessions(path: str | Path) -> dict[str, list[SessionRow]]:
    """
    The tsv rows grouped by subject, subjects and sessions in file order.

    Raises:
        ValueError: The tsv lacks one of SESSION_COLUMNS.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"session labels not found: {path}")
    groups: dict[str, list[SessionRow]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in SESSION_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: lacks the column(s) {missing}; found {reader.fieldnames}.")
        has_offset = DAY_OFFSET_COLUMN in (reader.fieldnames or [])
        for row in reader:
            date_text = row["date"].strip()
            offset_text = (row.get(DAY_OFFSET_COLUMN) or "").strip() if has_offset else ""
            session = SessionRow(
                subject=row["subject"].strip(),
                id=row["session"].strip(),
                label=normalise_label(row["label"]),
                raw_label=row["label"].strip(),
                date=date_text.rstrip("*"),
                approximate=date_text.endswith("*"),
                day_offset=int(float(offset_text)) if offset_text else None,
            )
            groups.setdefault(session.subject, []).append(session)
    return groups


def session_pair(root: Path, session: SessionRow, space: str) -> tuple[str, Path, Path] | None:
    """
    The image pair drawn for a session: (pair name, T1c, segmentation),
    or None when the session has no complete pair (no processed
    directory, or a file of every admissible pair missing).
    """
    base = root / session.subject / session.id
    order = ("longitudinal", "native") if space == "longitudinal" else ("native",)
    for name in order:
        t1c, segmentation = (base / relative for relative in PAIRS[name])
        if t1c.is_file() and segmentation.is_file():
            return name, t1c, segmentation
    return None


def check_orientation(image: nib.Nifti1Image, path: Path) -> None:
    codes = tuple(nib.aff2axcodes(image.affine))
    if codes != EXPECTED_AXCODES:
        raise ValueError(f"{path}: axis orientation {codes} differs from the expected {EXPECTED_AXCODES}.")


def tumor_center(segmentation: NDArray, preop: bool) -> tuple[tuple[int, int, int] | None, str | None]:
    """
    The voxel centre of mass of the session's tumour core (labels
    CORE_LABELS, a pre-op segmentation relabelled 4 -> 3), falling back
    to the whole tumour (WHOLE_LABELS) and then to any labelled voxel.

    Returns:
        (centre, source) with source in "core", "whole", "any", or
        (None, None) for an empty segmentation.
    """
    labels = relabel_preop(segmentation) if preop else np.asarray(segmentation)
    for source, mask in (
        ("core", np.isin(labels, CORE_LABELS)),
        ("whole", np.isin(labels, WHOLE_LABELS)),
        ("any", labels != 0),
    ):
        if mask.any():
            centre = tuple(int(round(c)) for c in center_of_mass(mask))
            return (centre[0], centre[1], centre[2]), source
    return None, None


def pad_to_square(plane: NDArray, size: int, fill: float = 0.0) -> NDArray:
    """The plane centred on a size x size canvas filled with ``fill``
    (voxels are isotropic, so the views share one scale)."""
    if plane.shape[0] > size or plane.shape[1] > size:
        raise ValueError(f"plane {plane.shape} exceeds the canvas {size}.")
    top = (size - plane.shape[0]) // 2
    left = (size - plane.shape[1]) // 2
    canvas = np.full((size, size), fill, dtype=plane.dtype)
    canvas[top : top + plane.shape[0], left : left + plane.shape[1]] = plane
    return canvas


def oriented_slices(volume: NDArray, centre: tuple[int, int, int]) -> dict[str, NDArray]:
    """
    The three views through a voxel of an LAS volume, each centred on a
    square canvas of the grid's largest extent (``pad_to_square``): axial
    and coronal in radiological view (the subject's right on the left,
    anterior / superior up), the sagittal with anterior on the left and
    superior up.
    """
    x, y, z = centre
    size = int(max(volume.shape[:3]))
    return {
        "axial": pad_to_square(np.rot90(volume[:, :, z]), size),
        "coronal": pad_to_square(np.rot90(volume[:, y, :]), size),
        "sagittal": pad_to_square(np.rot90(volume[x, :, :])[:, ::-1], size),
    }


def display_window(t1c: NDArray, percentile: float) -> float:
    """The upper display limit of a T1c: the percentile of its non-zero
    voxels (the brain), 1.0 for an empty volume."""
    inside = t1c[t1c > 0]
    return float(np.percentile(inside, percentile)) if inside.size else 1.0


@dataclass
class LoadedSession:
    """A session's row data, or the reason it has none."""

    session: SessionRow
    pair: str | None = None
    t1c: NDArray | None = None
    labels: NDArray | None = None
    own_center: tuple[int, int, int] | None = None
    own_source: str | None = None
    core_ml: float = float("nan")
    whole_ml: float = float("nan")
    vmax: float = 1.0
    error: str | None = None
    center: tuple[int, int, int] | None = None
    center_source: str | None = None


def load_session(
    root: Path, session: SessionRow, space: str, grid: dict[str, Any], window_percentile: float
) -> LoadedSession:
    """
    Load a session's pair and its own tumour centre. ``grid`` carries the
    shape and voxel volume of the first volume loaded (filled here on
    first use); later volumes are checked against it. Load errors are
    recorded on the row, not raised.
    """
    found = session_pair(root, session, space)
    if found is None:
        return LoadedSession(session, error="no complete image pair")
    pair, t1c_path, segmentation_path = found
    loaded = LoadedSession(session, pair=pair)
    try:
        t1c_image = nib.load(str(t1c_path))
        check_orientation(t1c_image, t1c_path)
        segmentation, segmentation_image = load_segmentation(segmentation_path)
        check_orientation(segmentation_image, segmentation_path)
        t1c = np.asarray(t1c_image.get_fdata(), dtype=np.float32)
        if t1c.shape != segmentation.shape:
            raise ValueError(f"T1c shape {t1c.shape} differs from the segmentation's {segmentation.shape}.")
        if "shape" not in grid:
            grid["shape"] = t1c.shape
            grid["voxel_ml"] = float(np.prod(t1c_image.header.get_zooms()[:3])) / 1000.0
        elif t1c.shape != grid["shape"]:
            raise ValueError(f"shape {t1c.shape} differs from the document's grid {grid['shape']}.")
    except (OSError, ValueError) as exc:
        loaded.error = f"{type(exc).__name__}: {exc}"
        return loaded
    voxel_ml = grid["voxel_ml"]
    labels = (relabel_preop(segmentation) if session.preop else segmentation).astype(np.uint8)
    loaded.t1c = t1c
    loaded.labels = labels
    loaded.vmax = display_window(t1c, window_percentile)
    loaded.own_center, loaded.own_source = tumor_center(segmentation, session.preop)
    loaded.core_ml = float(np.isin(labels, CORE_LABELS).sum()) * voxel_ml
    loaded.whole_ml = float(np.isin(labels, WHOLE_LABELS).sum()) * voxel_ml
    return loaded


def assign_centers(rows: list[LoadedSession], mode: str, shape: tuple[int, ...] | None) -> tuple[int, int, int] | None:
    """
    Set every loaded row's slice centre. The reference centre is the
    first row whose own centre came from the core (then whole, then any).
    'session' mode gives each row its own centre with the reference and
    the grid centre as fallbacks; 'subject' mode gives every row the
    reference centre (the grid centre when the subject has none).

    Returns:
        The reference centre, None when no row has labelled voxels.
    """
    reference: tuple[int, int, int] | None = None
    for source in ("core", "whole", "any"):
        match = next((r for r in rows if r.error is None and r.own_source == source), None)
        if match is not None:
            reference = match.own_center
            break
    grid_center = None if shape is None else tuple(int(n // 2) for n in shape[:3])
    for row in rows:
        if row.error is not None:
            continue
        if mode == "session" and row.own_center is not None:
            row.center, row.center_source = row.own_center, row.own_source
        elif reference is not None:
            row.center, row.center_source = reference, "reference"
        else:
            row.center, row.center_source = grid_center, "grid"
    return reference


@dataclass
class CavityOverview:
    """
    The subject's extra row: the base session's T1c through the pre-op
    core centre with the cavity (label 4) of every later post-op /
    follow-up exam as a contour.

    Attributes:
        base: The background session (the pre-op one, else the first
            drawn session, see ``note``).
        center: The slice centre and where it came from.
        exams: The contoured sessions with their cavity masks, in
            session order.
        note: Why the base is not a pre-op session, or None.
    """

    base: LoadedSession
    center: tuple[int, int, int]
    center_source: str
    exams: list[tuple[LoadedSession, NDArray]]
    note: str | None = None

    def color(self, k: int) -> tuple[float, float, float, float]:
        cmap = plt.get_cmap(CAVITY_CMAP)
        return cmap(0.0 if len(self.exams) < 2 else k / (len(self.exams) - 1))

    def label(self) -> str:
        x, y, z = self.center
        which = "pre-op" if self.base.session.preop else self.base.session.id
        centre = f"{which} core CoM" if self.center_source == "core" else f"{self.center_source} centre"
        lines = [
            "cavity contours",
            f"on {self.base.session.id} {self.base.session.raw_label} T1c",
            f"{centre}:",
            f"x={x} y={y} z={z}",
            f"{len(self.exams)} post-op / follow-up exam(s)",
        ]
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)


def cavity_overview(
    drawn: list[LoadedSession], reference: tuple[int, int, int] | None, shape: tuple[int, ...] | None
) -> CavityOverview | None:
    """
    The overview row of a subject: the pre-op session (the first drawn
    row labelled preop; without one, the first drawn row, noted) as the
    background, its own centre (falling back to the reference and the
    grid centre), and a cavity mask for every later drawn row labelled
    postop or followup. None when nothing was drawn.
    """
    rows = [r for r in drawn if r.error is None]
    if not rows:
        return None
    base = next((r for r in rows if r.session.preop), None)
    note = None
    if base is None:
        base = rows[0]
        note = "no pre-op session drawn;\nthe first exam is the background"
    if base.own_center is not None:
        center, source = base.own_center, base.own_source
    elif reference is not None:
        center, source = reference, "reference"
    else:
        center, source = tuple(int(n // 2) for n in (shape or base.t1c.shape)[:3]), "grid"  # type: ignore[union-attr]
    later = rows[rows.index(base) + 1 :]
    exams = [(r, r.labels == LABEL_CAVITY) for r in later if r.session.label in CAVITY_LABELS]  # type: ignore[operator]
    return CavityOverview(base, center, source or "grid", exams, note)  # type: ignore[arg-type]


def row_label(row: LoadedSession) -> str:
    lines = [row.session.title()]
    if row.error is not None:
        lines.append(f"pair: {row.pair or 'none'}")
        lines.append(f"not drawn: {row.error}")
        return "\n".join(lines)
    x, y, z = row.center  # type: ignore[misc]
    centre = "core CoM" if row.center_source == "core" else f"{row.center_source} centre"
    lines.append(f"{row.pair} pair")
    lines.append(f"{centre}: x={x} y={y} z={z}")
    lines.append(f"core {row.core_ml:.1f} mL, whole {row.whole_ml:.1f} mL")
    return "\n".join(lines)


def page_header(subject: str, rows: list[SessionRow], missing: list[SessionRow], page: tuple[int, int]) -> str:
    n_total = len(rows)
    n_shown = n_total - len(missing)
    text = f"{subject}: {n_total} sessions in the tsv, {n_shown} drawn"
    if page[1] > 1:
        text += f" (page {page[0]}/{page[1]})"
    if missing:
        text += "\nmissing: " + ", ".join(f"{s.id} ({s.raw_label}, {s.date}{'*' if s.approximate else ''})" for s in missing)
    return text


def render_page(
    pdf: PdfPages,
    subject: str,
    all_rows: list[SessionRow],
    drawn: list[LoadedSession | CavityOverview],
    missing: list[SessionRow],
    page: tuple[int, int],
    alpha: float,
    dpi: int,
) -> None:
    """One PDF page: a header, a legend, one row per drawn session and,
    where the chunk holds it, the subject's cavity-overview row."""
    n_rows = max(1, len(drawn))
    fig = plt.figure(
        figsize=(LABEL_COLUMN_INCHES + PANEL_INCHES * len(VIEWS), HEADER_INCHES + PANEL_INCHES * n_rows), dpi=dpi
    )
    header_fraction = HEADER_INCHES / fig.get_figheight()
    grid = fig.add_gridspec(
        n_rows, 1 + len(VIEWS), width_ratios=[LABEL_COLUMN_INCHES] + [PANEL_INCHES] * len(VIEWS),
        left=0.01, right=0.99, bottom=0.005, top=1.0 - header_fraction, wspace=0.02, hspace=0.06,
    )
    fig.text(0.01, 1.0 - 0.25 * header_fraction, page_header(subject, all_rows, missing, page), fontsize=11, fontweight="bold", va="top", ha="left")
    fig.legend(
        handles=[Patch(color=LABEL_COLORS[k], label=LABEL_NAMES[k]) for k in sorted(LABEL_COLORS)],
        loc="upper right", bbox_to_anchor=(0.99, 1.0 - 0.1 * header_fraction), fontsize=9, frameon=False, ncol=2,
    )
    colors = [(0, 0, 0, 0)] + [LABEL_COLORS[k] for k in sorted(LABEL_COLORS)]
    cmap = ListedColormap(colors)
    for k, row in enumerate(drawn):
        label_ax = fig.add_subplot(grid[k, 0])
        label_ax.axis("off")
        if isinstance(row, CavityOverview):
            render_overview_row(fig, grid, k, row, label_ax)
            continue
        label_ax.text(0.02, 0.5, row_label(row), fontsize=8.5, va="center", ha="left", wrap=True, transform=label_ax.transAxes)
        for j, view in enumerate(VIEWS):
            ax = fig.add_subplot(grid[k, 1 + j])
            ax.axis("off")
            if row.error is not None or row.center is None:
                ax.text(0.5, 0.5, "not drawn", ha="center", va="center", fontsize=9, color="gray", transform=ax.transAxes)
                continue
            background = oriented_slices(row.t1c, row.center)[view]  # type: ignore[arg-type]
            overlay = oriented_slices(row.labels, row.center)[view]  # type: ignore[arg-type]
            ax.imshow(background, cmap="gray", vmin=0.0, vmax=row.vmax, interpolation="none")
            ax.imshow(
                np.ma.masked_equal(overlay, 0), cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5, alpha=alpha, interpolation="none"
            )
            index = {"axial": f"z={row.center[2]}", "coronal": f"y={row.center[1]}", "sagittal": f"x={row.center[0]}"}[view]
            ax.set_title(f"{view} ({index})", fontsize=8.5, pad=2)
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def render_overview_row(fig: Any, grid: Any, k: int, overview: CavityOverview, label_ax: Any) -> None:
    """The cavity-overview row: the base T1c in the three views with one
    contour per exam, and the exam legend in the label column."""
    label_ax.text(0.02, 0.98, overview.label(), fontsize=8.5, va="top", ha="left", wrap=True, transform=label_ax.transAxes)
    handles = []
    for n, (exam, _) in enumerate(overview.exams):
        session = exam.session
        day = "" if session.day_offset is None else f", day {session.day_offset}"
        handles.append(Line2D([], [], color=overview.color(n), linewidth=1.5, label=f"{session.id} ({session.date}{'*' if session.approximate else ''}{day})"))
    if handles:
        label_ax.legend(
            handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.5 if len(handles) > 12 else 7.5,
            frameon=False, ncol=1, handlelength=1.5, borderaxespad=0.0, labelspacing=0.25,
        )
    base = overview.base
    backgrounds = oriented_slices(base.t1c, overview.center)  # type: ignore[arg-type]
    masks = [(oriented_slices(mask.astype(np.uint8), overview.center), n) for n, (_, mask) in enumerate(overview.exams)]
    for j, view in enumerate(VIEWS):
        ax = fig.add_subplot(grid[k, 1 + j])
        ax.axis("off")
        ax.imshow(backgrounds[view], cmap="gray", vmin=0.0, vmax=base.vmax, interpolation="none")
        for slices, n in masks:
            plane = slices[view]
            if plane.any():
                ax.contour(plane.astype(float), levels=[0.5], colors=[overview.color(n)], linewidths=CAVITY_LINEWIDTH)
        index = {"axial": f"z={overview.center[2]}", "coronal": f"y={overview.center[1]}", "sagittal": f"x={overview.center[0]}"}[view]
        ax.set_title(f"{view} ({index}), cavities", fontsize=8.5, pad=2)


def chunks(items: list[Any], size: int | None) -> list[list[Any]]:
    if size is None or size >= len(items):
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.patient_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"patient root not found: {root}")
    groups = read_sessions(args.session_labels)
    subjects = list(groups)
    if args.subjects:
        wanted = parse_subjects(args.subjects)
        unknown = [s for s in wanted if s not in groups]
        if unknown:
            raise ValueError(f"subject(s) {unknown} are not in {args.session_labels}; known: {subjects}.")
        subjects = wanted
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(subjects)} subjects, {sum(len(groups[s]) for s in subjects)} tsv sessions; data root {root}")

    grid: dict[str, Any] = {}
    n_drawn = n_missing = n_failed = 0
    metadata = {
        "Title": "SAILOR longitudinal tumour segmentations",
        "Subject": f"root {root}; space {args.space}; center {args.center}; rows per page {args.rows_per_page}",
        "Creator": "scripts/visualize_sailor.py",
        "CreationDate": datetime.now(timezone.utc),
    }
    with PdfPages(str(output), metadata=metadata) as pdf:
        for subject in subjects:
            sessions = groups[subject]
            loaded = [load_session(root, s, args.space, grid, args.window_percentile) for s in sessions]
            missing = [r.session for r in loaded if r.pair is None]
            drawn = [r for r in loaded if r.pair is not None]
            reference = assign_centers(drawn, args.center, grid.get("shape"))
            failed = [r for r in drawn if r.error is not None]
            for row in failed:
                print(f"  {subject} {row.session.id}: not drawn: {row.error}", file=sys.stderr)
            overview = cavity_overview(drawn, reference, grid.get("shape"))
            rows: list[LoadedSession | CavityOverview] = [*drawn, *([overview] if overview is not None else [])]
            pages = chunks(rows, args.rows_per_page)
            for k, page_rows in enumerate(pages, start=1):
                render_page(pdf, subject, sessions, page_rows, missing, (k, len(pages)), args.alpha, args.dpi)
            n_drawn += len(drawn) - len(failed)
            n_missing += len(missing)
            n_failed += len(failed)
            pairs = sorted({r.pair for r in drawn if r.pair})
            reference_text = "none" if reference is None else "x={} y={} z={}".format(*reference)
            print(
                f"{subject}: {len(drawn) - len(failed)}/{len(sessions)} sessions drawn on {len(pages)} page(s) "
                f"({'+'.join(pairs) or 'no pairs'}), reference centre {reference_text}, "
                f"{0 if overview is None else len(overview.exams)} cavity contour(s)"
                + (f", missing {[s.id for s in missing]}" if missing else "")
            )
    print(f"saved {output.resolve()}: {n_drawn} sessions drawn, {n_missing} without images, {n_failed} failed to load")
    return 0


if __name__ == "__main__":
    sys.exit(main())
