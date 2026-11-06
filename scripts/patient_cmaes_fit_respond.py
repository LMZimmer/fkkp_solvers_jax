"""
patient_cmaes_fit.py for the RESPOND long-follow-up patients, one fit per
patient and seed count, plus a queue that runs them on a few GPUs.

The fit itself is patient_cmaes_fit.py: this script prepares one patient's
inputs in the layout that script reads, replaces the clinical timeline rule
by the one recorded below, injects the patient defaults into the command
line and hands over to patient_cmaes_fit.main. Every option of
patient_cmaes_fit.py works unchanged and given options override the injected
defaults (argparse keeps the last occurrence).

  python scripts/patient_cmaes_fit_respond.py starts --subject GR149
  python scripts/patient_cmaes_fit_respond.py fit --subject GR149 --name GR149_1seed --gpus 0
  python scripts/patient_cmaes_fit_respond.py fit --subject GR149 --name GR149_2seed --n-seeds 2 --gpus 0
  python scripts/patient_cmaes_fit_respond.py resolve --fit-dir /mnt/Drive4/lucas/stupp_patient_fit_respond/GR149_1seed --gpu 0

  python scripts/patient_cmaes_fit_respond.py queue --gpus 4,6 --list
  python scripts/patient_cmaes_fit_respond.py queue --gpus 4,6 --subjects GR149,GR041 --n-seeds 1 --dry-run
  python scripts/patient_cmaes_fit_respond.py queue --gpus 4,6

Patient data
------------
ROOT = /mnt/Drive4/lucas/respond_longitudinal/lucas_long_followup holds
sub-<subject>/ses-<yyyymmdd> (raw), sub-<subject>/processed/ses-<yyyymmdd>
(per-session layout of patient_sensitivity_analysis.py),
sub-<subject>/processed/session_labels.tsv (subject, session, timepoint,
processed, reference, label_source) and dates.xlsx (one row per subject: ID,
Baseline, Resection, intraop, postop, RT planning, first image after RT,
recurrence; read with the standard library, column positions by cell
reference, so an empty cell stays empty). The RT dose map is
sub-<subject>/ses-*/sub-<subject>_ses-<yyyymmdd>_space-preop_rtdose.nii.gz.

``prepare_inputs`` builds INPUTS_ROOT/<subject>/<session>/ as a file-level
symbolic-link mirror of processed/<session>/ for the fitted sessions, links
the dose map into the dose session's longitudinal/ and writes
INPUTS_ROOT/session_labels_<subject>.tsv. The mirror is built once
(atomically, by renaming a finished temporary directory) and never removed:
a later call only verifies it, so the one- and two-seed fits of a patient can
run at the same time. A mirror that no longer matches the processed files (or
a tsv that no longer matches) raises, naming what differs; remove it by hand
when no fit of that patient is running. This differs from
patient_cmaes_fit_tum149.py, which rebuilds its mirror on every call.

Design choices (decided with the author on 2026-11-06)
------------------------------------------------------
1. Patients: EXCLUDED_SUBJECTS are refused, with the reason recorded there:
   GR115 and GR147 have no dose map (GR147 also has no RT planning date),
   GR138's pre-op mask holds no label 1 or 3, so it has no seedable voxel.
2. Sessions fitted: the session with reference = 1 (it must be labelled
   preop and equal the sheet's Baseline) and every later row with processed
   = 1 labelled postop or followup; there must be exactly one postop row, and
   it must equal the sheet's postop. Skipped and recorded: rows labelled
   intraop and any other preop row (GR106's ses-20191010). Follow-ups after
   the sheet's recurrence date are KEPT (GR008's ses-20200922 and GR108's
   ses-20220524), although treatment after a recurrence is not modelled.
3. CRT start: CRT_START_DAYS_AFTER_RT_DATE (6) days after the RT date. The RT
   date is the sheet's RT planning date, or the dose file's date when the
   sheet has none (GR010). For GR149 this is 2021-04-12, the day of
   patient_cmaes_fit_tum149.py's rule. A CRT that ends on or after the sheet's
   "first image after RT" is printed and recorded, not refused (GR027: 6 days
   after).
4. Resection: the sheet's Resection date for each patient, not
   RESECTION_DAYS_BEFORE_POSTOP. This moves GR149's resection 2 days later
   than in patient_cmaes_fit_tum149.py, so GR149 is refitted here.
5. Treatment otherwise as patient_cmaes_fit_tum149.py: the Stupp schedule
   constants and base config doses of patient_sensitivity_analysis.py, at
   most ADJUVANT_CYCLES adjuvant cycles, nothing after the last session.
6. Starts: --no-sweep-init is always set (no sweep exists for these
   patients), so without --init-values CMA-ES starts at the centre of the
   unit cube; with --n-seeds 2 both seeds start at the box centre and the
   sampling separates them. The two seed counts are independent fits.
7. --popsize defaults to POPSIZE here because patient_cmaes_fit.py's "auto"
   would give 4 members on one worker. POPSIZE is the value of
   fit_tum149_2026-10-29.
8. Output: OUTPUT_DIR unless --output-dir is given; the fit directory
   receives DESIGN_FILE with this record.

Queue
-----
``queue`` runs one fit per job, one worker each, on the devices of --gpus:
a device id given twice takes two jobs at a time. Jobs run one seed count
after the other (--n-seeds order), and within one the shortest horizon first.
A job's log is OUTPUT_DIR/<name>.log and ends with EXIT_CODE=. Every subject is
prepared before the first job starts, so a data problem stops the queue before
any solve. Each job's environment drops LD_LIBRARY_PATH (a JAX CPU fallback
otherwise) and sets NUMPY_MADVISE_HUGEPAGE=0.

Started again, the queue skips finished fits (summary.json and best/ exist)
and continues interrupted ones with --resume. A fit directory without
spec.json stops the queue before any job starts and is not touched. A failed
job does not stop the others, is never retried within a run, and makes the
queue exit with 1. --dry-run evaluates each start once under <name>_dryrun;
those names are not skipped or resumed, so an existing one stops the queue in
the same way: remove it before repeating the dry run.
--list prints the jobs with a time estimate and starts nothing: REFERENCE_MINUTES
of fit_tum149_2026-10-29 scaled by n_steps alone, one GPU, so a rough number.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue as queue_module
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import patient_cmaes_fit  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    ADJUVANT_CYCLE_DAYS,
    ADJUVANT_CYCLES,
    ADJUVANT_DAYS_ON,
    ADJUVANT_DELAY_DAYS,
    CONCOMITANT_DAYS,
    DOSE_FILE,
    LABEL_FOLLOWUP,
    LABEL_POSTOP,
    LABEL_PREOP,
    N_FRACTIONS,
    AdjuvantCycle,
    Protocol,
    Session,
    Timeline,
    crt_start_session,
    fraction_offsets,
)

# --- the data ---

ROOT = Path("/mnt/Drive4/lucas/respond_longitudinal/lucas_long_followup")
DATES_SHEET = ROOT / "dates.xlsx"
INPUTS_ROOT = ROOT / "fit_inputs"  # INPUTS_ROOT/<subject>/<session>/, the layout the fit script reads
OUTPUT_DIR = Path("/mnt/Drive4/lucas/stupp_patient_fit_respond")
DESIGN_FILE = "respond_design.json"

EXCLUDED_SUBJECTS: dict[str, str] = {  # design choice 1
    "GR115": "no RT dose map",
    "GR147": "no RT dose map and no RT planning date",
    "GR138": "the pre-op mask holds no label 1 or 3, so no voxel can be seeded",
}

LABEL_INTRAOP = "intraop"
FITTED_LATER_LABELS = (LABEL_POSTOP, LABEL_FOLLOWUP)

CRT_START_DAYS_AFTER_RT_DATE = 6  # design choice 3

POPSIZE = 12  # design choice 7
TSV_COLUMNS = ("subject", "session", "label", "date")

# The --list estimate: fit_tum149_2026-10-29, 100 generations of 12, one GPU.
REFERENCE_MINUTES = 168.4
REFERENCE_N_STEPS = 7500
ESTIMATE_PREOP_TIME_MAX = 200
ESTIMATE_STEPS_PER_DAY = 12

XLSX_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
EXCEL_EPOCH = date(1899, 12, 30)


def session_date(session_id: str) -> date:
    """The scan date of a session id 'ses-yyyymmdd'."""
    digits = session_id.removeprefix("ses-")
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"session id {session_id!r} is not of the form ses-yyyymmdd.")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def read_dates_sheet(path: Path = DATES_SHEET) -> dict[str, dict[str, date | None]]:
    """The first sheet of dates.xlsx as {subject: {column header: date or None}}."""
    tag = f"{{{XLSX_NAMESPACE}}}"
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(t.text or "" for t in item.iter(f"{tag}t")) for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(f"{tag}si")]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, str]] = []
    for row in sheet.iter(f"{tag}row"):
        cells: dict[str, str] = {}
        for cell in row.findall(f"{tag}c"):
            value = cell.find(f"{tag}v")
            if value is not None and value.text is not None:
                column = re.match(r"[A-Z]+", cell.get("r", "")).group()
                cells[column] = strings[int(value.text)] if cell.get("t") == "s" else value.text
        rows.append(cells)
    if not rows:
        raise ValueError(f"{path}: the first sheet is empty.")
    header = rows[0]
    id_column = next((c for c, name in header.items() if name == "ID"), None)
    if id_column is None:
        raise ValueError(f"{path}: no column named ID; found {list(header.values())}.")
    table: dict[str, dict[str, date | None]] = {}
    for cells in rows[1:]:
        if id_column not in cells:
            continue
        table[cells[id_column]] = {
            name: EXCEL_EPOCH + timedelta(days=int(float(cells[column]))) if column in cells else None
            for column, name in header.items()
            if column != id_column
        }
    return table


@dataclass(frozen=True)
class Subject:
    """
    One patient's design (design choices 2 to 4).

    Attributes:
        name: The subject id, "GR149".
        sessions: The fitted sessions, the pre-op one first, then by date.
        skipped: The tsv's other sessions as {session id: reason}.
        dose_source: The RT dose map.
        resection: The sheet's Resection date.
        rt_date: The date the CRT start counts from.
        rt_date_source: Where rt_date comes from.
        first_image_after_rt: The sheet's date.
        recurrence: The sheet's date.
    """

    name: str
    sessions: tuple[Session, ...]
    skipped: dict[str, str]
    dose_source: Path
    resection: date
    rt_date: date
    rt_date_source: str
    first_image_after_rt: date | None
    recurrence: date | None

    @property
    def processed_root(self) -> Path:
        return ROOT / f"sub-{self.name}" / "processed"

    @property
    def session_labels(self) -> Path:
        return INPUTS_ROOT / f"session_labels_{self.name}.tsv"

    @property
    def crt_start(self) -> date:
        return self.rt_date + timedelta(days=CRT_START_DAYS_AFTER_RT_DATE)

    @property
    def horizon_days(self) -> int:
        return (self.sessions[-1].date - self.sessions[0].date).days

    @property
    def crt_last_day(self) -> date:
        """The last CRT day of the schedule (fractions or concomitant TMZ, whichever ends later)."""
        offset = max(fraction_offsets(0, N_FRACTIONS)[-1], CONCOMITANT_DAYS - 1)
        return self.crt_start + timedelta(days=offset)

    def notes(self) -> list[str]:
        """What the design keeps although it looks odd (design choices 2 and 3)."""
        notes = []
        if self.first_image_after_rt is not None and self.crt_last_day >= self.first_image_after_rt:
            days = (self.crt_last_day - self.first_image_after_rt).days
            notes.append(f"the CRT ends {self.crt_last_day}, {days} day(s) after the first image after RT ({self.first_image_after_rt})")
        if self.recurrence is not None:
            after = [s.id for s in self.sessions if s.date > self.recurrence]
            if after:
                notes.append(f"session(s) {after} are after the recurrence date {self.recurrence} and are kept")
        return notes


def find_dose(name: str) -> Path:
    """The subject's one RT dose map."""
    found = sorted((ROOT / f"sub-{name}").glob(f"ses-*/sub-{name}_ses-*_space-preop_rtdose.nii.gz"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one RT dose map for {name} under {ROOT / f'sub-{name}'}, found {[str(p) for p in found]}.")
    return found[0]


def dose_file_date(dose: Path) -> date:
    """The date in a dose file's name."""
    match = re.search(r"_(ses-\d{8})_space-preop_rtdose", dose.name)
    if match is None:
        raise ValueError(f"{dose}: no session date in the file name.")
    return session_date(match.group(1))


def load_subject(name: str, sheet: dict[str, dict[str, date | None]] | None = None) -> Subject:
    """
    The subject's design from its tsv, the dates sheet and the dose map.

    Raises:
        ValueError: The subject is excluded or unknown, or the tsv and the
            sheet disagree on the pre-op or the post-op session.
    """
    if name in EXCLUDED_SUBJECTS:
        raise ValueError(f"{name} is excluded: {EXCLUDED_SUBJECTS[name]}.")
    sheet = read_dates_sheet() if sheet is None else sheet
    if name not in sheet:
        raise ValueError(f"{name} is not in {DATES_SHEET}; known: {sorted(sheet)}.")
    dates = sheet[name]
    tsv = ROOT / f"sub-{name}" / "processed" / "session_labels.tsv"
    with open(tsv, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["subject"].strip() == name]
    references = [row for row in rows if row["reference"].strip() == "1"]
    if len(references) != 1 or references[0]["timepoint"].strip() != LABEL_PREOP:
        raise ValueError(f"{tsv}: expected one reference session labelled {LABEL_PREOP}, got {references}.")
    preop_id = references[0]["session"].strip()
    if session_date(preop_id) != dates["Baseline"]:
        raise ValueError(f"{name}: the reference session {preop_id} is not the sheet's Baseline {dates['Baseline']}.")
    fitted = [Session(preop_id, LABEL_PREOP, session_date(preop_id))]
    skipped: dict[str, str] = {}
    for row in rows:
        sid, label = row["session"].strip(), row["timepoint"].strip()
        if sid == preop_id:
            continue
        if row["processed"].strip() != "1":
            skipped[sid] = "not processed"
        elif label == LABEL_PREOP:
            skipped[sid] = "another pre-op scan; the reference session is the pre-op session"
        elif label == LABEL_INTRAOP:
            skipped[sid] = "intra-operative scan"
        elif label in FITTED_LATER_LABELS:
            fitted.append(Session(sid, label, session_date(sid)))
        else:
            raise ValueError(f"{tsv}: session {sid} has the unknown timepoint {label!r}.")
    later = sorted(fitted[1:], key=lambda s: s.date)
    postops = [s for s in later if s.label == LABEL_POSTOP]
    if len(postops) != 1:
        raise ValueError(f"{tsv}: expected one {LABEL_POSTOP} session, got {[s.id for s in postops]}.")
    if postops[0].date != dates["postop"]:
        raise ValueError(f"{name}: the tsv's post-op session {postops[0].id} is not the sheet's postop {dates['postop']}.")
    if dates["Resection"] is None:
        raise ValueError(f"{name}: the sheet has no Resection date.")
    dose = find_dose(name)
    planning = dates["RT planning"]
    return Subject(
        name=name,
        sessions=(fitted[0], *later),
        skipped=skipped,
        dose_source=dose,
        resection=dates["Resection"],
        rt_date=planning if planning is not None else dose_file_date(dose),
        rt_date_source="RT planning (dates.xlsx)" if planning is not None else f"dose file date ({dose.name}); the sheet has no RT planning date",
        first_image_after_rt=dates["first image after RT"],
        recurrence=dates["recurrence"],
    )


def all_subjects() -> list[str]:
    """The sheet's subjects without the excluded ones."""
    return sorted(set(read_dates_sheet()) - set(EXCLUDED_SUBJECTS))


# --- the inputs ---


def expected_links(subject: Subject) -> dict[Path, Path]:
    """The mirror as {path relative to INPUTS_ROOT/<subject>: link target}."""
    links: dict[Path, Path] = {}
    for session in subject.sessions:
        source = subject.processed_root / session.id
        if not source.is_dir():
            raise FileNotFoundError(f"processed session not found: {source}")
        for root, _dirs, files in os.walk(source):
            relative = Path(session.id) / Path(root).relative_to(source)
            for name in files:
                links[relative / name] = Path(root) / name
    dose_target = Path(crt_start_session(subject.sessions).id) / DOSE_FILE
    if dose_target in links:
        raise RuntimeError(f"{links[dose_target]} already exists in the processed data; this script expected to supply it from {subject.dose_source}.")
    links[dose_target] = subject.dose_source
    return links


def _mirror_differences(mirror: Path, links: dict[Path, Path]) -> list[str]:
    """How the existing mirror differs from the expected links (empty = no difference)."""
    differences = []
    found: set[Path] = set()
    for root, _dirs, files in os.walk(mirror):
        for name in files:
            path = Path(root) / name
            relative = path.relative_to(mirror)
            found.add(relative)
            if relative not in links:
                differences.append(f"unexpected {relative}")
            elif not path.is_symlink() or Path(os.readlink(path)) != links[relative]:
                differences.append(f"{relative} does not link to {links[relative]}")
    differences += [f"missing {relative}" for relative in sorted(set(links) - found)]
    return differences


def _write_if_absent(path: Path, text: str) -> None:
    """Write a file atomically if absent; an existing file must hold the same text."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"{path} differs from what this script would write; remove it by hand when no fit of the patient is running.")
        return
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def prepare_inputs(subject: Subject, quiet: bool = False) -> None:
    """Build the subject's mirror and tsv if absent and verify them if not (module docstring)."""
    links = expected_links(subject)
    INPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    mirror = INPUTS_ROOT / subject.name
    built = False
    if not mirror.exists():
        temporary = INPUTS_ROOT / f".{subject.name}.tmp-{os.getpid()}"
        for relative, target in links.items():
            (temporary / relative).parent.mkdir(parents=True, exist_ok=True)
            (temporary / relative).symlink_to(target)
        try:
            os.rename(temporary, mirror)
            built = True
        except OSError:  # another process finished first: verify its mirror below
            shutil.rmtree(temporary)
    if not built:
        differences = _mirror_differences(mirror, links)
        if differences:
            raise RuntimeError(
                f"{mirror} no longer matches {subject.processed_root} ({len(differences)} difference(s), first: {differences[:5]}); "
                "remove it by hand when no fit of the patient is running."
            )
    lines = ["\t".join(TSV_COLUMNS)]
    lines += ["\t".join((subject.name, s.id, s.label, s.date.isoformat())) for s in subject.sessions]
    _write_if_absent(subject.session_labels, "\n".join(lines) + "\n")
    if not quiet:
        print(f"inputs: {mirror} ({'built' if built else 'verified'}; {len(subject.sessions)} sessions; skipped {subject.skipped})")
        print(f"        dose map {subject.dose_source}")
        print(f"        session labels {subject.session_labels}")
        print(f"        resection {subject.resection}, CRT start {subject.crt_start} ({subject.rt_date_source} + {CRT_START_DAYS_AFTER_RT_DATE} d), horizon {subject.horizon_days} d")
        for note in subject.notes():
            print(f"        note: {note}")


# --- the timeline (design choices 3 and 4) ---


def timeline_builder(subject: Subject):
    """
    patient_sensitivity_analysis.build_timeline with the resection on the
    subject's sheet date and the CRT start CRT_START_DAYS_AFTER_RT_DATE days
    after its RT date; everything else (fractions, concomitant days, adjuvant
    cycles, dropping after the horizon) identical.
    """

    def build_timeline_respond(sessions_: Sequence[Session], protocol: Protocol) -> Timeline:
        sessions_ = tuple(sessions_)
        preop = sessions_[0]
        if preop.label != LABEL_PREOP:
            raise ValueError(f"the first session must be the pre-op one, got {preop.id} ({preop.label}).")
        if not any(s.label == LABEL_POSTOP for s in sessions_):
            raise ValueError(f"no session labelled {LABEL_POSTOP}.")
        horizon = max((s.date - preop.date).days for s in sessions_)
        resection = (subject.resection - preop.date).days
        crt_start = (subject.crt_start - preop.date).days
        if resection < 0:
            raise ValueError(f"the resection ({resection} days after the pre-op scan) precedes the pre-op scan.")
        if resection >= crt_start:
            raise ValueError(f"the resection (day {resection}) is not before the CRT start (day {crt_start}).")
        rt_all = fraction_offsets(crt_start, protocol.n_fractions)
        concomitant_all = tuple(crt_start + i for i in range(protocol.concomitant_days))
        last_crt = max(rt_all[-1], concomitant_all[-1])
        adjuvant_start = last_crt + protocol.adjuvant_delay_days
        cycles: list[AdjuvantCycle] = []
        number, start = 1, adjuvant_start
        while start <= horizon and (protocol.adjuvant_cycles is None or number <= protocol.adjuvant_cycles):
            days = [start + i for i in range(protocol.adjuvant_days_on)]
            kept = tuple(d for d in days if d <= horizon)
            dose = protocol.adjuvant_first_dose if number == 1 else protocol.adjuvant_later_dose
            cycles.append(AdjuvantCycle(number, start, kept, dose, len(days) - len(kept)))
            number += 1
            start += protocol.adjuvant_cycle_days
        rt_kept = tuple(o for o in rt_all if o <= horizon)
        concomitant_kept = tuple(o for o in concomitant_all if o <= horizon)
        return Timeline(
            sessions=sessions_,
            resection_offset=resection,
            crt_start_offset=crt_start,
            crt_session_id=f"RT date {subject.rt_date} + {CRT_START_DAYS_AFTER_RT_DATE} d",
            rt_offsets=rt_kept,
            concomitant_offsets=concomitant_kept,
            adjuvant_cycles=tuple(cycles),
            horizon_offset=horizon,
            n_rt_dropped=len(rt_all) - len(rt_kept),
            n_concomitant_dropped=len(concomitant_all) - len(concomitant_kept),
            concomitant_dose=protocol.concomitant_dose,
        )

    return build_timeline_respond


# --- the record and the command line of one fit ---


def design_record(subject: Subject) -> dict:
    return {
        "script": Path(__file__).name,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subject": subject.name,
        "root": str(ROOT),
        "processed_root": str(subject.processed_root),
        "inputs_root": str(INPUTS_ROOT),
        "session_labels": str(subject.session_labels),
        "sessions": [s.record() for s in subject.sessions],
        "skipped_sessions": subject.skipped,
        "dose_source": str(subject.dose_source),
        "dose_session": crt_start_session(subject.sessions).id,
        "resection": {"rule": "the Resection date of dates.xlsx", "date": subject.resection.isoformat()},
        "crt_start": {
            "rule": f"{CRT_START_DAYS_AFTER_RT_DATE} days after the RT date",
            "rt_date": subject.rt_date.isoformat(),
            "rt_date_source": subject.rt_date_source,
            "date": subject.crt_start.isoformat(),
        },
        "first_image_after_rt": None if subject.first_image_after_rt is None else subject.first_image_after_rt.isoformat(),
        "recurrence": None if subject.recurrence is None else subject.recurrence.isoformat(),
        "notes_on_this_patient": subject.notes(),
        "protocol_constants": {
            "n_fractions": N_FRACTIONS,
            "concomitant_days": CONCOMITANT_DAYS,
            "adjuvant_delay_days": ADJUVANT_DELAY_DAYS,
            "adjuvant_days_on": ADJUVANT_DAYS_ON,
            "adjuvant_cycle_days": ADJUVANT_CYCLE_DAYS,
            "adjuvant_cycles": ADJUVANT_CYCLES,
        },
        "starts": "no sweep starts (--no-sweep-init); centre of the unit cube unless --init-values",
        "notes": "see the module docstring of the script (Design choices, 2026-11-06)",
    }


FIT_COMMANDS = ("starts", "fit", "resolve")


def pop_subject(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """--subject's value and the argv without it (patient_cmaes_fit.py does not know the option)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--subject", default=None)
    args, rest = parser.parse_known_args(list(argv))
    return args.subject, rest


def subject_of_fit_dir(argv: Sequence[str]) -> str:
    """The subject a 'resolve' command line's fit directory was fitted on."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fit-dir", required=True)
    args, _rest = parser.parse_known_args(list(argv))
    spec = json.loads((Path(args.fit_dir) / "spec.json").read_text(encoding="utf-8"))
    return spec["patient"]["patient"]


def inject_defaults(argv: Sequence[str], subject: Subject) -> list[str]:
    """The argv with the patient defaults inserted after the subcommand (given options come later and win)."""
    argv = list(argv)
    position = next(i for i, a in enumerate(argv) if a in FIT_COMMANDS)
    command = argv[position]
    defaults: list[str] = []
    if command in ("starts", "fit"):
        defaults += [
            "--patient", subject.name,
            "--patient-root", str(INPUTS_ROOT),
            "--session-labels", str(subject.session_labels),
            "--sessions", ",".join(s.id for s in subject.sessions),
            "--no-sweep-init",
        ]
    if command == "fit":
        defaults += ["--output-dir", str(OUTPUT_DIR), "--popsize", str(POPSIZE)]
    return argv[: position + 1] + defaults + argv[position + 1 :]


def fit_dir_of(argv: Sequence[str]) -> Path | None:
    """The fit directory a 'fit' command line names (--output-dir / --name), or None."""
    args, _unknown = patient_cmaes_fit.build_parser().parse_known_args(list(argv))
    if args.command != "fit":
        return None
    return Path(args.output_dir).resolve() / args.name


def run_one_fit(argv: Sequence[str]) -> int:
    """A starts, fit or resolve command of one subject."""
    name, argv = pop_subject(argv)
    command = next((a for a in argv if a in FIT_COMMANDS), None)
    if command is None:
        return patient_cmaes_fit.main(argv)  # argparse reports the missing subcommand or prints the help
    if command == "resolve":
        name = subject_of_fit_dir(argv)
    elif name is None:
        raise SystemExit(f"{command} needs --subject; known: {all_subjects()}.")
    subject = load_subject(name)
    patient_cmaes_fit.build_timeline = timeline_builder(subject)  # looked up in the fit script's namespace (build_problem)
    if command != "resolve":
        prepare_inputs(subject)
    argv = inject_defaults(argv, subject)
    code = patient_cmaes_fit.main(argv)
    fit_dir = fit_dir_of(argv)
    if fit_dir is not None and fit_dir.is_dir():
        (fit_dir / DESIGN_FILE).write_text(json.dumps(design_record(subject), indent=2) + "\n", encoding="utf-8")
        print(f"design record: {fit_dir / DESIGN_FILE}")
    return code


# --- the queue ---


@dataclass(frozen=True)
class Job:
    """One fit of the queue."""

    subject: Subject
    n_seeds: int
    output_dir: Path
    dry_run: bool

    @property
    def name(self) -> str:
        return f"{self.subject.name}_{self.n_seeds}seed" + ("_dryrun" if self.dry_run else "")

    @property
    def fit_dir(self) -> Path:
        return self.output_dir / self.name

    @property
    def log(self) -> Path:
        return self.output_dir / f"{self.name}.log"

    @property
    def estimated_hours(self) -> float:
        n_steps = math.ceil(ESTIMATE_STEPS_PER_DAY * (ESTIMATE_PREOP_TIME_MAX + self.subject.horizon_days))
        return REFERENCE_MINUTES * n_steps / REFERENCE_N_STEPS / 60.0

    def state(self) -> str:
        """new, finished, interrupted or broken (module docstring, Queue)."""
        if not self.fit_dir.exists():
            return "new"
        if self.dry_run:
            return "broken"
        if (self.fit_dir / "summary.json").is_file() and (self.fit_dir / "best").is_dir():
            return "finished"
        return "interrupted" if (self.fit_dir / "spec.json").is_file() else "broken"

    def command(self, gpu: str) -> list[str]:
        command = [
            sys.executable, str(Path(__file__).resolve()), "fit",
            "--subject", self.subject.name,
            "--name", self.name,
            "--n-seeds", str(self.n_seeds),
            "--output-dir", str(self.output_dir),
            "--gpus", gpu,
        ]
        if self.dry_run:
            command.append("--dry-run")
        elif self.state() == "interrupted":
            command.append("--resume")
        return command


def job_environment() -> dict[str, str]:
    """The environment of a job (module docstring, Queue)."""
    environment = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    environment["NUMPY_MADVISE_HUGEPAGE"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def run_job(job: Job, gpu: str) -> int:
    """Run one fit with its log; the log ends with EXIT_CODE=."""
    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = job.command(gpu)
    with open(job.log, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=job_environment(), cwd=_ROOT, check=False).returncode
        log.write(f"EXIT_CODE={code}\n")
    return code


def run_queue(jobs: Sequence[Job], gpus: Sequence[str]) -> int:
    """One thread per device slot takes jobs in order; the exit code is 1 if a job failed."""
    pending: queue_module.Queue[Job] = queue_module.Queue()
    for job in jobs:
        pending.put(job)
    results: dict[str, str] = {}
    lock = threading.Lock()

    def say(text: str) -> None:
        with lock:
            print(f"{datetime.now().isoformat(timespec='seconds')} {text}", flush=True)

    def slot(gpu: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue_module.Empty:
                return
            start = time.monotonic()
            say(f"gpu {gpu}: start {job.name} (log {job.log})")
            try:
                code = run_job(job, gpu)
            except OSError as error:
                say(f"gpu {gpu}: {job.name} could not start: {error}")
                code = 1
            hours = (time.monotonic() - start) / 3600.0
            results[job.name] = f"exit {code}, {hours:.2f} h"
            say(f"gpu {gpu}: end {job.name}: exit {code}, {hours:.2f} h")

    threads = [threading.Thread(target=slot, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("queue done:")
    for job in jobs:
        print(f"  {job.name}: {results.get(job.name, 'not run')}")
    failed = [name for name, result in results.items() if not result.startswith("exit 0,")]
    if failed:
        print(f"FAILED: {failed}")
    return 1 if failed else 0


def queue_main(argv: Sequence[str]) -> int:
    """The queue subcommand (module docstring, Queue)."""
    parser = argparse.ArgumentParser(prog="patient_cmaes_fit_respond.py queue", description="Run the fits of several subjects on a few GPUs.")
    parser.add_argument("--gpus", required=True, help="comma-separated device ids, one job at a time per entry (an id given twice takes two jobs)")
    parser.add_argument("--subjects", default="", help="comma-separated subjects (default: every subject of dates.xlsx except the excluded ones)")
    parser.add_argument("--n-seeds", default="1,2", help="comma-separated seed counts, run in this order (default 1,2)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help=f"the fit directories and logs (default {OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="evaluate each start once under <name>_dryrun")
    parser.add_argument("--list", action="store_true", help="print the jobs and an estimate; start nothing")
    args = parser.parse_args(list(argv))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        parser.error("--gpus names no device.")
    names = [s.strip() for s in args.subjects.split(",") if s.strip()] or all_subjects()
    seed_counts = [int(n) for n in args.n_seeds.split(",")]
    sheet = read_dates_sheet()
    subjects = sorted((load_subject(name, sheet) for name in names), key=lambda s: (s.horizon_days, s.name))
    output_dir = Path(args.output_dir).resolve()
    jobs = [Job(subject, n, output_dir, bool(args.dry_run)) for n in seed_counts for subject in subjects]

    states = {job.name: job.state() for job in jobs}
    print(f"{len(jobs)} job(s) on {gpus}; excluded {EXCLUDED_SUBJECTS}")
    for job in jobs:
        print(f"  {job.name:<22} horizon {job.subject.horizon_days:>5} d  sessions {len(job.subject.sessions):>3}  ~{job.estimated_hours:5.1f} h  {states[job.name]}")
    remaining = [job for job in jobs if states[job.name] != "finished"]
    hours = sum(job.estimated_hours for job in remaining)
    print(f"estimate for the {len(remaining)} unfinished job(s): {hours:.0f} GPU-hours, about {hours / len(gpus):.0f} h on {len(gpus)} slot(s) (rough, module docstring)")
    if args.list:
        return 0

    broken = [job.name for job in remaining if states[job.name] == "broken"]
    if broken:
        print(f"these fit directories exist but cannot be continued; nothing was started, and they were not touched: {broken}")
        return 1
    for subject in subjects:
        prepare_inputs(subject)
    return run_queue(remaining, gpus)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "queue":
        return queue_main(argv[1:])
    return run_one_fit(argv)


if __name__ == "__main__":
    sys.exit(main())
