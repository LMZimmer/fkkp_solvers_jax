"""
patient_cmaes_fit.py for the RESPOND patient tum_149 (subject GR149).

The fit itself is patient_cmaes_fit.py: this script only prepares the
patient's inputs in the layout that script reads, replaces the clinical
timeline rule by the one recorded below, injects the patient defaults into
the command line and then hands over to patient_cmaes_fit.main. Every
option of patient_cmaes_fit.py works unchanged and given options override
the injected defaults (argparse keeps the last occurrence).

  python scripts/patient_cmaes_fit_tum149.py starts
  python scripts/patient_cmaes_fit_tum149.py fit --name fit_tum149 --gpus 0,1,2,3 --dry-run
  python scripts/patient_cmaes_fit_tum149.py fit --name fit_tum149 --gpus 0,1,2,3
  python scripts/patient_cmaes_fit_tum149.py fit --name fit_tum149_preop --objective preop --gpus 0
  python scripts/patient_cmaes_fit_tum149.py resolve --fit-dir /mnt/Drive4/lucas/stupp_patient_fit_tum149/fit_tum149 --gpu 0

Patient data
------------
RAW_ROOT = /mnt/Drive4/lucas/respond_longitudinal/tum_149 holds the raw
sessions (ses-<yyyymmdd>) and processed/<session>/ with the per-session
layout of patient_sensitivity_analysis.py (tissue_segmentation,
tumor_segmentation, longitudinal/recurrence_preop.nii.gz), every image on
the 240 x 240 x 155 1 mm grid of the pre-op scan. The fit script expects
<root>/<patient>/<session>/ and a session-labels tsv with the columns
subject, session, label, date; processed/session_labels.tsv has subject
GR149 and the columns subject, session, timepoint, processed (no date).

``prepare_inputs`` therefore builds INPUTS_ROOT/tum_149/<session>/ as a
mirror of processed/<session>/ (directories created, every file a symbolic
link to the processed file; the mirror is rebuilt on every call and holds
nothing else), adds the dose map (below) and writes
INPUTS_ROOT/session_labels.tsv with the date taken from the session id.

Design choices (2026-10-24)
---------------------------
1. Post-op session: ses-20210313 (cavity label 4, 33 045 voxels). The
   other post-op scan ses-20210312 (one day earlier; 10 918 cavity voxels
   and residual enhancing tissue) is SKIPPED: the fit script allows one
   post-op session, and the later scan shows the completed resection.
2. Sessions fitted (the full objective): the pre-op scan ses-20210303,
   the post-op scan ses-20210313 and all six follow-ups ses-20210406,
   ses-20210510, ses-20210623, ses-20211004, ses-20211228, ses-20220502
   (horizon 425 days after the pre-op scan). ses-20210309 and
   ses-20220412 exist raw only (not processed, not in the tsv) and are
   not used.
3. Dose map: RAW_ROOT/ses-20210303/sub-GR149_ses-20210406_space-preop_rtdose.nii.gz
   (total dose, maximum 63.8 Gy), already on the pre-op grid (same affine
   as the processed images). It is linked to
   <dose session>/longitudinal/dose_warped_longitudinal.nii.gz, the file
   the fit script reads; the dose session is the first follow-up after
   the post-op scan (ses-20210406), as in patient_sensitivity_analysis.py.
4. Treatment: the Stupp protocol exactly as for SAILOR (the schedule
   constants and the base config's doses of patient_sensitivity_analysis.py:
   30 fractions in 6 weeks, 42 concomitant TMZ days, adjuvant cycles of
   5 on / 23 off from 28 days after the last CRT day), with ONE change:
   the CRT starts CRT_START_DAYS_AFTER_POSTOP = 30 days after the
   post-op scan (day 40 after the pre-op scan, 2021-04-12) instead of on
   the first follow-up scan date (SAILOR rule; here ses-20210406, day
   34). The resection stays RESECTION_DAYS_BEFORE_POSTOP days before the
   post-op scan. The actual treatment dates of the patient are unknown.
5. Adjuvant cycles are NOT capped at the six cycles of the protocol: as
   in patient_sensitivity_analysis.py every cycle that starts before the
   last session is scheduled (12 cycles over the 425-day horizon).
6. Starts: --no-sweep-init is always set (no sensitivity-analysis sweep
   exists for this patient), so without --init-values CMA-ES starts at
   the centre of the unit cube of the search space.
7. Output: the fit directories go to OUTPUT_DIR
   (/mnt/Drive4/lucas/stupp_patient_fit_tum149) unless --output-dir is
   given; the fit directory receives DESIGN_FILE with this record.

Everything else (search space, base config, loss, --solver, --objective,
preop_time range, steps per day, CMA-ES settings) is patient_cmaes_fit.py's.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import patient_cmaes_fit  # noqa: E402
from patient_sensitivity_analysis import (  # noqa: E402
    ADJUVANT_CYCLE_DAYS,
    ADJUVANT_DAYS_ON,
    ADJUVANT_DELAY_DAYS,
    CONCOMITANT_DAYS,
    DOSE_FILE,
    LABEL_FOLLOWUP,
    LABEL_POSTOP,
    LABEL_PREOP,
    N_FRACTIONS,
    RESECTION_DAYS_BEFORE_POSTOP,
    AdjuvantCycle,
    Protocol,
    Session,
    Timeline,
    crt_start_session,
    fraction_offsets,
)

# --- the patient ---

PATIENT = "tum_149"
RAW_SUBJECT = "GR149"
RAW_ROOT = Path("/mnt/Drive4/lucas/respond_longitudinal/tum_149")
PROCESSED_ROOT = RAW_ROOT / "processed"
RAW_SESSION_LABELS = PROCESSED_ROOT / "session_labels.tsv"
DOSE_SOURCE = RAW_ROOT / "ses-20210303" / "sub-GR149_ses-20210406_space-preop_rtdose.nii.gz"
INPUTS_ROOT = RAW_ROOT / "fit_inputs"  # the mirror the fit script reads: INPUTS_ROOT/<PATIENT>/<session>/
SESSION_LABELS = INPUTS_ROOT / "session_labels.tsv"
OUTPUT_DIR = Path("/mnt/Drive4/lucas/stupp_patient_fit_tum149")
DESIGN_FILE = "tum149_design.json"

PREOP_SESSION = "ses-20210303"
POSTOP_SESSION = "ses-20210313"
SKIPPED_SESSIONS: tuple[str, ...] = ("ses-20210312",)  # design choice 1
FOLLOWUP_SESSIONS: tuple[str, ...] = (
    "ses-20210406",
    "ses-20210510",
    "ses-20210623",
    "ses-20211004",
    "ses-20211228",
    "ses-20220502",
)
SESSIONS: tuple[str, ...] = (PREOP_SESSION, POSTOP_SESSION, *FOLLOWUP_SESSIONS)
LABELS: dict[str, str] = {PREOP_SESSION: LABEL_PREOP, POSTOP_SESSION: LABEL_POSTOP, **{s: LABEL_FOLLOWUP for s in FOLLOWUP_SESSIONS}}

CRT_START_DAYS_AFTER_POSTOP = 30  # design choice 4
PROTOCOL_ANCHOR = f"{POSTOP_SESSION} + {CRT_START_DAYS_AFTER_POSTOP} d"  # the Timeline's crt_session_id (printed, recorded)

TSV_COLUMNS = ("subject", "session", "label", "date")


def session_date(session_id: str) -> date:
    """The scan date of a session id 'ses-yyyymmdd'."""
    digits = session_id.removeprefix("ses-")
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"session id {session_id!r} is not of the form ses-yyyymmdd.")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def sessions() -> list[Session]:
    """The fitted sessions, the pre-op one first, then by date."""
    rows = [Session(sid, LABELS[sid], session_date(sid)) for sid in SESSIONS]
    return [rows[0], *sorted(rows[1:], key=lambda s: s.date)]


# --- the inputs ---


def check_raw_layout() -> None:
    """Every fitted session is processed and listed in the raw tsv; the dose map exists."""
    missing = [sid for sid in SESSIONS if not (PROCESSED_ROOT / sid).is_dir()]
    if missing:
        raise FileNotFoundError(f"processed session(s) not found under {PROCESSED_ROOT}: {missing}.")
    if not DOSE_SOURCE.is_file():
        raise FileNotFoundError(f"dose map not found: {DOSE_SOURCE}")
    if RAW_SESSION_LABELS.is_file():
        raw_rows: dict[str, dict[str, str]] = {}
        with open(RAW_SESSION_LABELS, encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                if row.get("subject", "").strip() == RAW_SUBJECT:
                    raw_rows[row["session"].strip()] = row
        unlisted = [sid for sid in SESSIONS if raw_rows.get(sid, {}).get("processed", "1").strip() != "1"]
        if unlisted:
            raise ValueError(f"{RAW_SESSION_LABELS}: session(s) {unlisted} are not listed as processed for {RAW_SUBJECT}.")
        wrong = {sid: raw_rows[sid].get("timepoint", "").strip() for sid in SESSIONS if raw_rows[sid].get("timepoint", "").strip() != LABELS[sid]}
        if wrong:
            raise ValueError(f"{RAW_SESSION_LABELS}: the timepoint of {wrong} differs from this script's labels.")


def _clear_mirror(patient_dir: Path) -> None:
    """Remove the mirror; refuse if it holds anything but directories and symbolic links."""
    if not patient_dir.exists():
        return
    for root, _dirs, files in os.walk(patient_dir):
        for name in files:
            path = Path(root) / name
            if not path.is_symlink():
                raise RuntimeError(f"{path} is a regular file, not a link: {patient_dir} is not a mirror this script built; not touching it.")
    shutil.rmtree(patient_dir)


def prepare_inputs(quiet: bool = False) -> None:
    """
    Build INPUTS_ROOT/<PATIENT>/<session>/ as a file-level symbolic-link
    mirror of PROCESSED_ROOT/<session>/ for the fitted sessions, link the
    dose map into the dose session's longitudinal/ and write the
    session-labels tsv (TSV_COLUMNS). Rebuilt from scratch on every call.
    """
    check_raw_layout()
    patient_dir = INPUTS_ROOT / PATIENT
    _clear_mirror(patient_dir)
    for sid in SESSIONS:
        source = PROCESSED_ROOT / sid
        for root, _dirs, files in os.walk(source):
            relative = Path(root).relative_to(source)
            target_dir = patient_dir / sid / relative
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in files:
                (target_dir / name).symlink_to(Path(root) / name)
    dose_session = crt_start_session(sessions())
    dose_target = patient_dir / dose_session.id / DOSE_FILE
    dose_target.parent.mkdir(parents=True, exist_ok=True)
    if dose_target.exists() or dose_target.is_symlink():
        raise RuntimeError(f"{dose_target} already exists in the processed data; this script expected to supply it from {DOSE_SOURCE}.")
    dose_target.symlink_to(DOSE_SOURCE)
    lines = ["\t".join(TSV_COLUMNS)]
    for s in sessions():
        lines.append("\t".join((PATIENT, s.id, s.label, s.date.isoformat())))
    SESSION_LABELS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not quiet:
        print(f"inputs: {patient_dir} ({len(SESSIONS)} sessions mirrored from {PROCESSED_ROOT}; skipped {list(SKIPPED_SESSIONS)})")
        print(f"        dose map {DOSE_SOURCE} -> {dose_target}")
        print(f"        session labels {SESSION_LABELS}")


# --- the timeline (design choice 4) ---


def build_timeline_tum149(sessions_: Sequence[Session], protocol: Protocol) -> Timeline:
    """
    patient_sensitivity_analysis.build_timeline with the CRT start moved
    from the first follow-up scan to CRT_START_DAYS_AFTER_POSTOP days after
    the post-op scan; everything else (resection, fractions, concomitant
    days, adjuvant cycles, dropping after the horizon) identical.
    """
    sessions_ = tuple(sessions_)
    preop = sessions_[0]
    if preop.label != LABEL_PREOP:
        raise ValueError(f"the first session must be the pre-op one, got {preop.id} ({preop.label}).")
    postop = next((s for s in sessions_ if s.label == LABEL_POSTOP), None)
    if postop is None:
        raise ValueError(f"no session labelled {LABEL_POSTOP}.")
    offset = {s.id: (s.date - preop.date).days for s in sessions_}
    horizon = max(offset.values())
    resection = offset[postop.id] - RESECTION_DAYS_BEFORE_POSTOP
    crt_start = offset[postop.id] + CRT_START_DAYS_AFTER_POSTOP
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
    while start <= horizon:  # design choice 5: no cap on the number of cycles
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
        crt_session_id=PROTOCOL_ANCHOR,
        rt_offsets=rt_kept,
        concomitant_offsets=concomitant_kept,
        adjuvant_cycles=tuple(cycles),
        horizon_offset=horizon,
        n_rt_dropped=len(rt_all) - len(rt_kept),
        n_concomitant_dropped=len(concomitant_all) - len(concomitant_kept),
        concomitant_dose=protocol.concomitant_dose,
    )


# The fit script looks build_timeline up in its own namespace (build_problem,
# problem_from_fit_dir), so the replacement covers fit, starts and resolve.
patient_cmaes_fit.build_timeline = build_timeline_tum149


# --- the record and the command line ---


def design_record() -> dict:
    return {
        "script": Path(__file__).name,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient": PATIENT,
        "raw_subject": RAW_SUBJECT,
        "raw_root": str(RAW_ROOT),
        "processed_root": str(PROCESSED_ROOT),
        "inputs_root": str(INPUTS_ROOT),
        "session_labels": str(SESSION_LABELS),
        "sessions": [s.record() for s in sessions()],
        "skipped_sessions": list(SKIPPED_SESSIONS),
        "dose_source": str(DOSE_SOURCE),
        "dose_session": crt_start_session(sessions()).id,
        "crt_start": {
            "rule": f"{CRT_START_DAYS_AFTER_POSTOP} days after the post-op scan (SAILOR rule: the first follow-up scan)",
            "days_after_postop": CRT_START_DAYS_AFTER_POSTOP,
            "date": (session_date(POSTOP_SESSION) + timedelta(days=CRT_START_DAYS_AFTER_POSTOP)).isoformat(),
        },
        "resection_days_before_postop": RESECTION_DAYS_BEFORE_POSTOP,
        "protocol_constants": {
            "n_fractions": N_FRACTIONS,
            "concomitant_days": CONCOMITANT_DAYS,
            "adjuvant_delay_days": ADJUVANT_DELAY_DAYS,
            "adjuvant_days_on": ADJUVANT_DAYS_ON,
            "adjuvant_cycle_days": ADJUVANT_CYCLE_DAYS,
            "adjuvant_cycle_cap": None,
        },
        "starts": "no sweep starts (--no-sweep-init); centre of the unit cube unless --init-values",
        "notes": "see the module docstring of the script (Design choices, 2026-10-24)",
    }


COMMANDS = ("starts", "fit", "resolve")


def inject_defaults(argv: Sequence[str]) -> list[str]:
    """The argv with the patient defaults inserted after the subcommand (given options come later and win)."""
    argv = list(argv)
    position = next((i for i, a in enumerate(argv) if a in COMMANDS), None)
    if position is None:
        return argv  # argparse reports the missing subcommand
    command = argv[position]
    defaults: list[str] = []
    if command in ("starts", "fit"):
        defaults += [
            "--patient", PATIENT,
            "--patient-root", str(INPUTS_ROOT),
            "--session-labels", str(SESSION_LABELS),
            "--sessions", ",".join(SESSIONS),
            "--no-sweep-init",
        ]
    if command == "fit":
        defaults += ["--output-dir", str(OUTPUT_DIR)]
    return argv[: position + 1] + defaults + argv[position + 1 :]


def fit_dir_of(argv: Sequence[str]) -> Path | None:
    """The fit directory a 'fit' command line names (--output-dir / --name), or None."""
    args, _unknown = patient_cmaes_fit.build_parser().parse_known_args(argv)
    if args.command != "fit":
        return None
    return Path(args.output_dir).resolve() / args.name


def main(argv: Sequence[str] | None = None) -> int:
    argv = inject_defaults(sys.argv[1:] if argv is None else argv)
    if any(a in ("starts", "fit") for a in argv):
        prepare_inputs()
    code = patient_cmaes_fit.main(argv)
    fit_dir = fit_dir_of(argv)
    if fit_dir is not None and fit_dir.is_dir():
        (fit_dir / DESIGN_FILE).write_text(json.dumps(design_record(), indent=2) + "\n", encoding="utf-8")
        print(f"design record: {fit_dir / DESIGN_FILE}")
    return code


if __name__ == "__main__":
    sys.exit(main())
