"""Tests of scripts/patient_sensitivity_analysis.py (loaded with importlib
from scripts/), fast only: no patient data, no solver run.

(1) The clinical timeline from synthetic session dates, anchored on the
CRT start session's day t3 whatever its weekday (resection three days
before the post-op scan, 30 fractions at t3 + 7 w + d, 42 TMZ days from
t3, the adjuvant start at t3 + 69, 28-day cycles of 5 days on with the
config's per-cycle doses, truncation at the last session, every model
day shifted by preop_time, the CRT-relative offsets and the anchor line
of the spec record), the protocol doses read from the shipped base
config, (2) the pre-op relabelling 4 -> 3 and the
cavity exclusion in the Dice, (3) Dice / msd / hd95 on toy masks with the
empty-mask conventions, (4) the threshold-pair grid, the profiled
thresholds and the row thresholds on a synthetic field, (5) the
cheap-factor dedup (N (k_dyn + 2) distinct solves, every row mapped to a
run of equal dynamics) and edema_threshold < core_threshold on every
sampled row, (6) the shipped search space in both modes (factor order,
cheap factors, the refusal of timeline keys, the v1 space's shared ranges
equal to sigma_v2's, the v2 space differing from v1 in exactly its three
widened ranges, the script's defaults) and the snapshot-day rule.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from fisher_kpp_jax import StuppFKPPSolver, read_config

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "patient_sensitivity_analysis.py"
# The script's default (v2, 2026-09-17) and its predecessor (v1, 2026-09-15).
SEARCH_SPACE = Path(__file__).resolve().parent.parent / "fisher_kpp_jax" / "search_spaces" / "sailor_patient_v2_search_space.json"
SEARCH_SPACE_V1 = SEARCH_SPACE.with_name("sailor_patient_search_space.json")
BASE_CONFIG = Path(__file__).resolve().parent.parent / "fisher_kpp_jax" / "configs" / "StuppFKPPSolver.json"
SIGMA_V2_SEARCH_SPACE = SEARCH_SPACE.with_name("stupp_fkpp_sigma_v2_search_space.json")
# The entries v2 changes against v1 (patient-fit priors widened toward non-response).
V2_CHANGES = {
    "growth.front_width_mm": {"min": 1.0, "max": 8.0, "scale": "log"},
    "chemo_kill_rate": {"min": 5.0e-5, "max": 1.0e-2, "scale": "log"},
    "rt_alpha": {"min": 5.0e-4, "max": 0.1, "scale": "log"},
}


def _load_script():
    spec = importlib.util.spec_from_file_location("patient_sensitivity_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


psa = _load_script()
sa = sys.modules["sensitivity_analysis"]

PROTOCOL = psa.Protocol(concomitant_dose=75.0, adjuvant_first_dose=150.0, adjuvant_later_dose=200.0, base_cycle_days=28.0)
FACTOR_ORDER_PROFILED = [
    "front_speed_mm_per_day",
    "front_width_mm",
    "seed_peak_density",
    "seed_sigma_mm",
    "preop_time",
    "diffusivity_ratio",
    "chemo_kill_rate",
    "rt_alpha",
    "rt_alpha_beta_ratio",
    "gaussian_seed_x_fraction",
    "gaussian_seed_y_fraction",
    "gaussian_seed_z_fraction",
]
FACTOR_ORDER_SAMPLED = [*FACTOR_ORDER_PROFILED, "core_threshold", "edema_threshold_ratio"]


def _sessions(preop: date, later_offsets: dict[str, int], labels: dict[str, str]) -> list[psa.Session]:
    sessions = [psa.Session("ses-01", "preop", preop)]
    for sid, offset in later_offsets.items():
        sessions.append(psa.Session(sid, labels.get(sid, "followup"), preop + timedelta(days=offset)))
    return sessions


# --- (1) timeline ---


def _crt_offsets(t3: int) -> tuple[list[int], list[int], list[int]]:
    """The protocol's fraction, concomitant TMZ and adjuvant start offsets
    for a CRT start at offset t3: t3 + 7 w + d, t3 .. t3 + 41, t3 + 69."""
    fractions = [t3 + 7 * w + d for w in range(6) for d in range(5)]
    concomitant = [t3 + i for i in range(42)]
    return fractions, concomitant, [t3 + 69 + 28 * c for c in range(20)]


@pytest.mark.parametrize("t3", [40, 43])  # Monday and Thursday for a pre-op on Wednesday 2020-01-01
def test_timeline_protocol_anchored_on_crt_start(t3: int):
    # Pre-op on a Wednesday; post-op 15 days later; further scans at 60 and
    # 100 (the last one). The schedule must not depend on the weekday of
    # the CRT start date.
    preop = date(2020, 1, 1)
    assert (preop + timedelta(days=t3)).weekday() in (0, 3)
    sessions = _sessions(preop, {"ses-02": 15, "ses-03": t3, "ses-04": 60, "ses-05": 100}, {"ses-02": "postop"})
    timeline = psa.build_timeline(sessions, PROTOCOL)
    fractions, concomitant, adjuvant_starts = _crt_offsets(t3)
    assert timeline.resection_offset == 15 - 3
    assert timeline.crt_start_offset == t3 and timeline.crt_session_id == "ses-03"
    # 30 fractions at t3 + 7 w + d (w = 0..5, d = 0..4), the last on t3 + 39.
    assert list(timeline.rt_offsets) == fractions
    assert list(timeline.rt_offsets_total) == fractions and fractions[-1] == t3 + 39
    assert [timeline.crt_offset(o) for o in timeline.rt_offsets] == [7 * w + d for w in range(6) for d in range(5)]
    # 42 consecutive TMZ days from t3; the last CRT day is t3 + 41.
    assert list(timeline.concomitant_offsets) == concomitant
    assert timeline.last_crt_offset == t3 + 41
    # Adjuvant start t3 + 69 (= t3 + 41 + 28): after the horizon 100 for
    # both starts, so no cycle runs.
    assert adjuvant_starts[0] == t3 + 69 > 100
    assert timeline.adjuvant_cycles == ()
    assert timeline.horizon_offset == 100
    assert timeline.n_rt_dropped == 0 and timeline.n_concomitant_dropped == 0
    chemo_offsets, chemo_doses = timeline.chemo_schedule
    assert chemo_offsets == concomitant
    assert chemo_doses == [75.0] * 42
    assert timeline.snapshot_offsets == {"ses-01": 0, "ses-02": 15, "ses-03": t3, "ses-04": 60, "ses-05": 100}
    # Every model day shifts by preop_time.
    days = timeline.model_days(137.5)
    assert days["resection_time"] == pytest.approx(137.5 + 12)
    assert days["time_after_resection"] == pytest.approx(100 - 12)
    assert days["rt_times"] == pytest.approx([137.5 + o for o in fractions])
    assert days["chemo_times"] == pytest.approx([137.5 + o for o in concomitant])
    assert days["snapshots"]["ses-03"] == pytest.approx(137.5 + t3)
    assert days["snapshots"]["ses-05"] == pytest.approx(237.5)
    assert days["stopping_time"] == pytest.approx(days["resection_time"] + days["time_after_resection"])
    # The spec record: calendar dates for reference, offsets after the
    # pre-op scan and relative to the CRT start, the anchor line.
    record = timeline.record()
    assert record["protocol_anchor"] == "ses-03 = day 0 of CRT week 1; weekdays from the calendar are not used"
    assert record["resection"]["date"] == (preop + timedelta(days=12)).isoformat()
    assert record["crt_start"] == {
        "session": "ses-03", "date": (preop + timedelta(days=t3)).isoformat(), "offset_days": t3, "crt_offset_days": 0,
    }
    assert [f["crt_offset_days"] for f in record["rt_fractions"]] == [o - t3 for o in fractions]
    assert [f["offset_days"] for f in record["rt_fractions"]] == fractions
    assert record["rt_fractions"][-1]["date"] == (preop + timedelta(days=t3 + 39)).isoformat()
    assert [d["crt_offset_days"] for d in record["concomitant_tmz"]] == list(range(42))
    assert record["last_crt_day"]["crt_offset_days"] == 41
    assert record["adjuvant_start"]["crt_offset_days"] == 69 and record["adjuvant_start"]["offset_days"] == t3 + 69
    assert record["n_rt_fractions"] == 30 and record["adjuvant_cycles"] == []
    table = psa.format_timeline(timeline)
    assert "adjuvant cycle table" in table and "CRT day" in table
    assert "fractions: 30 at CRT days 0-4, 7-11, 14-18, 21-25, 28-32, 35-39" in table
    assert "concomitant TMZ: 42 days at CRT days 0-41" in table
    assert "Mon" not in table and "Thu" not in table


@pytest.mark.parametrize("t3", [40, 43])
def test_timeline_adjuvant_cycles_and_truncation(t3: int):
    preop = date(2020, 1, 1)
    fractions, concomitant, adjuvant_starts = _crt_offsets(t3)
    # Last scan at t3 + 137: the adjuvant phase starts at t3 + 69 and the
    # cycles at t3 + 69, 97, 125 (5 days each, the third t3 + 125..129); the
    # fourth (t3 + 153) starts after the horizon.
    horizon = t3 + 137
    sessions = _sessions(preop, {"ses-02": 15, "ses-03": t3, "ses-04": horizon}, {"ses-02": "postop"})
    timeline = psa.build_timeline(sessions, PROTOCOL)
    cycles = timeline.adjuvant_cycles
    assert [c.start_offset for c in cycles] == adjuvant_starts[:3] == [t3 + 69, t3 + 97, t3 + 125]
    assert [c.dose for c in cycles] == [150.0, 200.0, 200.0]
    assert all(c.offsets == tuple(c.start_offset + i for i in range(5)) for c in cycles)
    assert [timeline.crt_offset(o) for c in cycles for o in c.offsets] == [
        *range(69, 74), *range(97, 102), *range(125, 130)
    ]
    assert all(c.n_dropped == 0 for c in cycles)
    chemo_offsets, chemo_doses = timeline.chemo_schedule
    assert len(chemo_offsets) == 42 + 15
    assert chemo_offsets == sorted(chemo_offsets) and chemo_offsets[:42] == concomitant
    assert sum(chemo_doses) == pytest.approx(42 * 75 + 5 * 150 + 10 * 200)
    record = timeline.record()
    assert [c["start_crt_offset_days"] for c in record["adjuvant_cycles"]] == [69, 97, 125]
    assert [d["crt_offset_days"] for d in record["adjuvant_cycles"][0]["days"]] == list(range(69, 74))
    assert record["chemo_offsets"] == chemo_offsets and record["chemo_total_dose_mg_m2"] == pytest.approx(sum(chemo_doses))
    # All offsets shift by preop_time; the resection offset is unchanged.
    days = timeline.model_days(60.0)
    assert days["chemo_times"] == pytest.approx([60.0 + o for o in chemo_offsets])
    assert days["rt_times"] == pytest.approx([60.0 + o for o in fractions])
    assert days["resection_time"] == pytest.approx(60.0 + 12)
    # A last scan inside a cycle truncates it: t3 + 127 keeps t3 + 125, 126, 127.
    sessions = _sessions(preop, {"ses-02": 15, "ses-03": t3, "ses-04": t3 + 127}, {"ses-02": "postop"})
    truncated = psa.build_timeline(sessions, PROTOCOL).adjuvant_cycles
    assert truncated[-1].offsets == (t3 + 125, t3 + 126, t3 + 127) and truncated[-1].n_dropped == 2
    assert len(truncated) == 3
    # A last scan before the CRT end drops fractions and TMZ days: at
    # t3 + 7 the fractions 0-4 and 7 (6) and the TMZ days 0-7 (8) remain.
    sessions = _sessions(preop, {"ses-02": 15, "ses-03": t3, "ses-04": t3 + 7}, {"ses-02": "postop"})
    short = psa.build_timeline(sessions, PROTOCOL)
    assert list(short.rt_offsets) == [t3 + d for d in (0, 1, 2, 3, 4, 7)] and short.n_rt_dropped == 24
    assert short.n_concomitant_dropped == 42 - 8
    assert short.adjuvant_cycles == ()
    # A post-op scan within three days of the pre-op one puts the resection
    # before the pre-op scan; a CRT start needs a follow-up after the post-op scan.
    with pytest.raises(ValueError, match="precedes"):
        psa.build_timeline(_sessions(preop, {"ses-02": 2, "ses-03": t3}, {"ses-02": "postop"}), PROTOCOL)
    with pytest.raises(ValueError, match="no followup"):
        psa.build_timeline(_sessions(preop, {"ses-02": 45, "ses-03": 43}, {"ses-02": "postop"}), PROTOCOL)


def test_fraction_offsets_pattern():
    assert psa.fraction_offsets(0, 30) == tuple(7 * w + d for w in range(6) for d in range(5))
    assert psa.fraction_offsets(10, 7) == (10, 11, 12, 13, 14, 17, 18)
    assert psa.fraction_offsets(3, 0) == ()
    assert psa.N_FRACTIONS == 30 and psa.CONCOMITANT_DAYS == 42 and psa.ADJUVANT_DELAY_DAYS == 28


def test_protocol_from_shipped_config():
    base = read_config(BASE_CONFIG, solver=StuppFKPPSolver)
    protocol = psa.protocol_from_config(base)
    assert protocol.concomitant_dose == 75.0
    assert protocol.adjuvant_first_dose == 150.0
    assert protocol.adjuvant_later_dose == 200.0
    assert protocol.base_cycle_days == 28.0
    assert protocol.n_fractions == 30 and protocol.concomitant_days == 42
    assert protocol.adjuvant_cycle_days == 28 and protocol.adjuvant_days_on == 5 and protocol.adjuvant_delay_days == 28
    bad = dict(base)
    bad["rt_times"] = base["rt_times"][:-1]
    with pytest.raises(ValueError, match="fractions"):
        psa.protocol_from_config(bad)


def test_session_parsing_and_selection():
    assert psa.parse_sessions("02-04") == ["ses-02", "ses-03", "ses-04"]
    assert psa.parse_sessions("ses-02,ses-05") == ["ses-02", "ses-05"]
    assert psa.parse_sessions("2, 3") == ["ses-02", "ses-03"]
    assert psa.normalise_label("follow-up") == "followup"
    assert psa.parse_tsv_date("2010-04-29*") == (date(2010, 4, 29), True)
    preop = date(2020, 1, 1)
    sessions = _sessions(preop, {"ses-02": 15, "ses-03": 43, "ses-04": 60}, {"ses-02": "postop"})
    chosen = psa.select_sessions(sessions, ["ses-04", "ses-02"])
    assert [s.id for s in chosen] == ["ses-01", "ses-02", "ses-04"]
    with pytest.raises(ValueError, match="postop"):
        psa.select_sessions(sessions, ["ses-03", "ses-04"])
    with pytest.raises(ValueError, match="not in the tsv"):
        psa.select_sessions(sessions, ["ses-09"])
    assert psa.crt_start_session(chosen).id == "ses-04"


# --- (2) relabelling and cavity exclusion ---


def test_relabel_preop_and_cavity_exclusion():
    seg = np.zeros((6, 6, 6), dtype=np.int64)
    seg[1:3, 1:3, 1:3] = 4
    seg[3:5, 1:3, 1:3] = 1
    seg[1:5, 3:5, 1:3] = 2
    relabelled = psa.relabel_preop(seg)
    assert not (relabelled == 4).any() and (relabelled == 3).sum() == 8 and (seg == 4).sum() == 8
    preop = psa.reference_masks(seg, preop=True)
    assert preop.n_cavity == 0 and preop.valid.all()
    assert preop.core.sum() == 16 and preop.whole.sum() == 32
    postop = psa.reference_masks(seg, preop=False)
    assert postop.n_cavity == 8 and (~postop.valid).sum() == 8
    assert postop.core.sum() == 8 and postop.whole.sum() == 24
    # A model mask covering the cavity and the necrotic block: the cavity
    # voxels count neither for the model nor for the reference.
    density = np.zeros(seg.shape)
    density[1:5, 1:3, 1:3] = 1.0
    qois = psa.threshold_qois(density, postop, (1.0, 1.0, 1.0), 0.6, 0.3, distances=False)
    assert qois["dice_core"] == pytest.approx(1.0)
    assert qois["log10_V_core"] == pytest.approx(np.log10(8 + 1))
    assert qois["log_vol_ratio_core"] == pytest.approx(0.0)
    qois_pre = psa.threshold_qois(density, preop, (1.0, 1.0, 1.0), 0.6, 0.3, distances=False)
    assert qois_pre["dice_core"] == pytest.approx(1.0) and qois_pre["log10_V_core"] == pytest.approx(np.log10(17))


# --- (3) Dice and surface distances ---


def test_dice_and_distances_toy_masks():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[2:6, 2:6, 2:6] = True
    assert psa.dice(a, a) == pytest.approx(1.0)
    empty = np.zeros_like(a)
    assert np.isnan(psa.dice(empty, empty))
    assert psa.dice(a, empty) == 0.0 and psa.dice(empty, a) == 0.0
    b = np.zeros_like(a)
    b[2:6, 2:6, 4:8] = True  # half overlap along z
    assert psa.dice(a, b) == pytest.approx(0.5)
    # Surfaces: a 4^3 cube has 64 - 8 = 56 surface voxels.
    assert psa.surface(a).sum() == 56
    msd, hd95 = psa.symmetric_surface_distances(a, a, (1.0, 1.0, 1.0))
    assert msd == 0.0 and hd95 == 0.0
    # A cube shifted by 2 voxels along z: the far faces are 2 mm from the
    # other's surface, the side faces at most 2 mm away.
    msd, hd95 = psa.symmetric_surface_distances(a, b, (1.0, 1.0, 1.0))
    assert 0 < msd < 2.0 and hd95 == pytest.approx(2.0)
    msd_z, _ = psa.symmetric_surface_distances(a, b, (1.0, 1.0, 2.0))
    assert msd_z > msd  # anisotropic zooms scale the distances
    assert all(np.isnan(v) for v in psa.symmetric_surface_distances(a, empty, (1.0, 1.0, 1.0)))
    assert all(np.isnan(v) for v in psa.symmetric_surface_distances(empty, empty, (1.0, 1.0, 1.0)))
    assert psa.surface_distances(a, empty, (1.0, 1.0, 1.0)).size == 0
    # A mask on the array border counts the border as outside.
    edge = np.zeros_like(a)
    edge[0:2, 0:2, 0:2] = True
    assert psa.surface(edge).sum() == 8


def test_threshold_qois_empty_conventions():
    zooms = (1.0, 1.0, 1.0)
    seg = np.zeros((8, 8, 8), dtype=np.int64)
    seg[2:5, 2:5, 2:5] = 2  # edema only: the reference core is empty
    reference = psa.reference_masks(seg, preop=False)
    density = np.zeros(seg.shape)
    qois = psa.threshold_qois(density, reference, zooms, 0.6, 0.3)
    assert np.isnan(qois["dice_core"])  # both empty
    assert qois["dice_whole"] == 0.0  # model empty, reference not
    assert np.isnan(qois["log_vol_ratio_core"]) and np.isfinite(qois["log_vol_ratio_whole"])
    assert np.isnan(qois["msd_core"]) and np.isnan(qois["hd95_whole"])
    assert qois["log10_V_core"] == pytest.approx(0.0)  # log10(0 + 1 mm^3)


# --- (4) threshold grid and profiled thresholds ---


def test_threshold_pair_grid():
    pairs = psa.threshold_pairs()
    assert all(edema < core for core, edema in pairs)
    assert len(pairs) == sum(1 for c in psa.THRESHOLD_GRID_CORE for e in psa.THRESHOLD_GRID_EDEMA if e < c)
    assert (0.30, 0.10) in pairs and (0.85, 0.60) in pairs and (0.30, 0.30) not in pairs and (0.30, 0.35) not in pairs
    assert psa.THRESHOLD_GRID_CORE == tuple(round(0.30 + 0.05 * i, 2) for i in range(12))
    assert psa.THRESHOLD_GRID_EDEMA == tuple(round(0.10 + 0.05 * i, 2) for i in range(11))
    assert psa.THRESHOLD_GRID_CORE[0] == 0.30 and psa.THRESHOLD_GRID_CORE[-1] == 0.85
    assert psa.THRESHOLD_GRID_EDEMA[0] == 0.10 and psa.THRESHOLD_GRID_EDEMA[-1] == 0.60


def test_profiled_thresholds_on_synthetic_field():
    # A radial field; the reference core is the region >= 0.55, the whole
    # tumour the region >= 0.2: the best grid pair is (0.55, 0.20) with
    # Dice 1 for both.
    n = 24
    idx = np.indices((n, n, n))
    r = np.sqrt(((idx - (n - 1) / 2) ** 2).sum(axis=0))
    density = np.clip(1.0 - r / 10.0, 0.0, 1.0)
    seg = np.zeros((n, n, n), dtype=np.int64)
    seg[density >= 0.2] = 2
    seg[density >= 0.55] = 3
    reference = psa.reference_masks(seg, preop=True)
    star = psa.profiled_qois(density, reference)
    assert star["core_threshold_star"] == 0.55 and star["edema_threshold_star"] == 0.20
    assert star["dice_star_core"] == pytest.approx(1.0) and star["dice_star_whole"] == pytest.approx(1.0)
    # The row thresholds 0.6 / 0.3 give smaller model masks and Dice below 1.
    qois = psa.threshold_qois(density, reference, (1.0, 1.0, 1.0), 0.6, 0.3)
    assert 0 < qois["dice_core"] < 1 and 0 < qois["dice_whole"] < 1
    assert qois["log_vol_ratio_core"] < 0 and qois["msd_core"] > 0 and qois["hd95_core"] >= qois["msd_core"]
    # An empty reference core against a nonempty model core is a Dice of 0
    # at every threshold: the whole tumour's best threshold is found, the
    # core's is the first admissible one in grid order, the core's Dice* 0.
    seg_edema = np.where(seg == 3, 2, seg)
    star = psa.profiled_qois(density, psa.reference_masks(seg_edema, preop=True))
    assert star["dice_star_core"] == 0.0 and star["dice_star_whole"] == pytest.approx(1.0)
    assert star["edema_threshold_star"] == 0.20 and star["core_threshold_star"] == 0.30
    # An empty field against an empty reference: both masks empty on the
    # whole grid, everything NaN; against the reference, Dice 0 at the
    # first grid pair.
    star = psa.profiled_qois(np.zeros_like(density), psa.reference_masks(np.zeros_like(seg), preop=True))
    assert all(np.isnan(v) for v in star.values())
    star = psa.profiled_qois(np.zeros_like(density), reference)
    assert star["dice_star_core"] == 0.0 and star["dice_star_whole"] == 0.0
    assert (star["core_threshold_star"], star["edema_threshold_star"]) == psa.threshold_pairs()[0]
    # The crop box holds every mask, and the QoIs on it equal the full-grid ones.
    box = psa.crop_box(density, reference, 0.1)
    cropped = psa.Reference("", reference.core[box], reference.whole[box], reference.valid[box], 0)
    assert psa.threshold_qois(density[box], cropped, (1.0, 1.0, 1.0), 0.6, 0.3) == pytest.approx(qois, nan_ok=True)
    assert psa.profiled_qois(density[box], cropped) == pytest.approx(psa.profiled_qois(density, reference), nan_ok=True)


# --- (5) dedup and the sampled thresholds ---


def _synthetic_geometry():
    n = 16
    core = np.zeros((n, n, n), dtype=bool)
    core[5:11, 5:11, 5:11] = True
    wm = np.ones((n, n, n)) * 0.6
    gm = np.ones((n, n, n)) * 0.3
    wm[7, 7, 7] = gm[7, 7, 7] = 0.0  # a hole
    return wm, gm, core


def test_patient_seed_geometry():
    wm, gm, core = _synthetic_geometry()
    geometry = psa.patient_seed_geometry(wm, gm, core, 0.1)
    assert geometry.n_voxels == 6**3 - 1
    assert geometry.mask.sum() == 6**3 - 1 and not geometry.mask[7, 7, 7]
    assert (geometry.mask <= core).all()
    with pytest.raises(ValueError, match="shape"):
        psa.patient_seed_geometry(wm, gm, core[:8], 0.1)


def test_cheap_factor_dedup():
    names = ["a", "b", "c", "cheap1", "cheap2"]
    log2_n = 2
    samples = sa.saltelli_design(names, log2_n, seed=3)
    representative = psa.dedup_runs(samples, [3, 4])
    n_blocks, k, k_dyn = 2**log2_n, len(names), 3
    assert samples.shape[0] == n_blocks * (k + 2)
    assert len(set(representative.tolist())) == n_blocks * (k_dyn + 2)
    # Every row maps to a row of equal dynamics, its own or an earlier one,
    # and the A_B rows of the cheap columns map to their block's A row.
    for index, rep in enumerate(representative):
        assert rep <= index
        assert np.array_equal(samples[index, :3], samples[rep, :3])
    size = k + 2
    for block in range(n_blocks):
        base_row = block * size
        assert representative[base_row + 1 + 3] == base_row and representative[base_row + 1 + 4] == base_row
        assert representative[base_row + 1] == base_row + 1 and representative[base_row + size - 1] == base_row + size - 1
    # Without cheap columns every row is its own run.
    assert np.array_equal(psa.dedup_runs(samples, []), np.arange(samples.shape[0]))


def test_sampled_design_table():
    space, meta = psa.read_patient_search_space(SEARCH_SPACE, "sampled")
    cheap_columns = [space.names.index(f) for f in meta["cheap_factors"]]
    wm, gm, core = _synthetic_geometry()
    geometry = psa.patient_seed_geometry(wm, gm, core, 0.1)
    log2_n = 2
    samples = sa.saltelli_design(space.names, log2_n, seed=1)
    table = psa.patient_design_table(samples, space, geometry, cheap_columns)
    n_blocks, k, k_dyn = 2**log2_n, len(space.names), len(space.names) - 2
    assert len(table) == n_blocks * (k + 2)
    runs = {record["run_name"] for record in table}
    assert len(runs) == n_blocks * (k_dyn + 2)
    by_row = {record["row_name"]: record for record in table}
    assert runs <= set(by_row)  # every run is a row's own name
    for record in table:
        assert record["row_name"] == sa.run_name(record["row"], record["matrix"])
        assert 0 < record["edema_threshold"] < record["core_threshold"]
        assert record["edema_threshold"] == pytest.approx(record["edema_threshold_ratio"] * record["core_threshold"])
        assert 0.45 <= record["core_threshold"] <= 0.75 and 0.25 <= record["edema_threshold_ratio"] <= 0.6
        run = by_row[record["run_name"]]
        for key in (*space.derived_keys, "preop_time", "diffusivity_ratio", "seed_voxel_i", "seed_voxel_j", "seed_voxel_k"):
            assert run[key] == record[key]
        assert core[record["seed_voxel_i"], record["seed_voxel_j"], record["seed_voxel_k"]]
    # The solver values of a row hold no script factor.
    values = psa.solver_values(space, table[0])
    assert not set(values) & set(psa.SCRIPT_FACTORS)
    assert {"white_matter_diffusivity", "rho", "gaussian_seed_mass", "gaussian_seed_diffusion_time", "rt_alpha"} <= set(values)
    # The distinct runs are recoverable from the CSV-like records.
    distinct = psa.distinct_runs(table)
    assert set(distinct) == runs


# --- (6) search space and snapshot days ---


def test_search_space_modes():
    profiled, meta = psa.read_patient_search_space(SEARCH_SPACE, "profiled")
    assert profiled.names == FACTOR_ORDER_PROFILED
    assert meta["cheap_factors"] == [] and meta["threshold_mode"] == "profiled"
    assert set(meta["threshold_factors"]) == {"core_threshold", "edema_threshold_ratio"}
    assert profiled.overrides == {"chemo_decay_rate": 9.24, "gaussian_seed_scale": 1.0}
    assert list(profiled.groups) == ["growth", "seed"]
    assert profiled.factors["seed_sigma_mm"].high == 16.0 and profiled.factors["preop_time"].low == 30.0
    sampled, meta = psa.read_patient_search_space(SEARCH_SPACE, "sampled")
    assert sampled.names == FACTOR_ORDER_SAMPLED
    assert meta["cheap_factors"] == ["core_threshold", "edema_threshold_ratio"]
    assert sampled.factors["core_threshold"].scale == "linear" and sampled.factors["edema_threshold_ratio"].high == 0.6
    assert set(psa.analysed_qois(["ses01_"], "profiled")) < set(psa.analysed_qois(["ses01_"], "sampled"))
    assert len(psa.analysed_qois(["ses01_", "ses02_"], "profiled")) == 2 * (10 + 4 + 5) + 1
    assert len(psa.analysed_qois(["ses01_"], "sampled")) == 10 + 4 + 10 + 5 + 1
    with pytest.raises(ValueError, match="threshold mode"):
        psa.read_patient_search_space(SEARCH_SPACE, "fixed")


def test_shared_ranges_equal_sigma_v2():
    """Every range and scale the v1 patient space shares with the sigma_v2
    atlas space (growth, seed, diffusivity_ratio, chemo_kill_rate,
    chemo_decay_rate, rt_alpha, rt_alpha_beta_ratio, the seed fractions,
    gaussian_seed_scale) equals it; preop_time stands in for
    resection_time with its range, and only the threshold factors are
    the patient space's own. This documents the sigma_v2 alignment of v1
    (2026-09-15); v2 departs from it in three ranges (below)."""
    patient = {k: v for k, v in json.loads(SEARCH_SPACE_V1.read_text()).items() if not k.startswith("_")}
    atlas = {k: v for k, v in json.loads(SIGMA_V2_SEARCH_SPACE.read_text()).items() if not k.startswith("_")}
    shared = set(patient) & set(atlas)
    assert shared == set(atlas) - {"resection_time", "time_after_resection"}
    for key in shared:
        assert patient[key] == atlas[key], key
    assert patient["preop_time"] == atlas["resection_time"]
    assert set(patient) - shared == {"preop_time", "core_threshold", "edema_threshold_ratio"}
    space, _ = psa.read_patient_search_space(SEARCH_SPACE_V1, "sampled")
    assert space.factors["front_width_mm"].high == 4.0 and space.factors["seed_sigma_mm"].low == 5.0
    assert space.factors["chemo_kill_rate"].low == 1e-3 and space.factors["chemo_kill_rate"].high == 3.5e-2
    assert space.factors["rt_alpha"].low == 1e-3 and space.factors["rt_alpha"].high == 0.2


def _flat_entries(path: Path) -> dict[str, object]:
    """The non-comment entries of a search-space file with the derived
    groups' sub-entries flattened to '<group>.<factor>' (their 'derives'
    lists kept under '<group>.derives')."""
    flat: dict[str, object] = {}
    for key, value in json.loads(path.read_text()).items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and "derives" in value:
            for sub, entry in value.items():
                flat[f"{key}.{sub}"] = entry
        else:
            flat[key] = value
    return flat


def test_v2_differs_from_v1_in_three_ranges():
    """The v2 space (the script's default) equals v1 in every non-comment
    entry except rt_alpha, chemo_kill_rate and growth.front_width_mm, whose
    ranges are V2_CHANGES; the loader sees the same factors in the same
    order, and the parser's default search space is the v2 file."""
    v1, v2 = _flat_entries(SEARCH_SPACE_V1), _flat_entries(SEARCH_SPACE)
    assert set(v1) == set(v2)
    changed = {key for key in v1 if v1[key] != v2[key]}
    assert changed == set(V2_CHANGES)
    for key, entry in V2_CHANGES.items():
        assert v2[key] == entry, key
        assert v1[key]["scale"] == entry["scale"] == "log", key
    for key in set(v1) - changed:
        assert v1[key] == v2[key], key
    for mode in ("profiled", "sampled"):
        space_v1, meta_v1 = psa.read_patient_search_space(SEARCH_SPACE_V1, mode)
        space_v2, meta_v2 = psa.read_patient_search_space(SEARCH_SPACE, mode)
        assert space_v1.names == space_v2.names and meta_v1["cheap_factors"] == meta_v2["cheap_factors"]
        assert space_v2.overrides == space_v1.overrides
    space, _ = psa.read_patient_search_space(SEARCH_SPACE, "sampled")
    assert (space.factors["rt_alpha"].low, space.factors["rt_alpha"].high) == (5e-4, 0.1)
    assert (space.factors["chemo_kill_rate"].low, space.factors["chemo_kill_rate"].high) == (5e-5, 1e-2)
    assert (space.factors["front_width_mm"].low, space.factors["front_width_mm"].high) == (1.0, 8.0)
    assert psa.DEFAULT_SEARCH_SPACE == SEARCH_SPACE
    args = psa.build_parser().parse_args(["design", "--name", "x"])
    assert Path(args.search_space) == SEARCH_SPACE


def test_defaults():
    assert psa.DEFAULT_SESSIONS == "02-08"
    assert psa.DEFAULT_LOG2_N == 10 and psa.DEFAULT_LOG2_N != sa.DEFAULT_LOG2_N
    assert psa.DEFAULT_THRESHOLD_MODE == "sampled"
    args = psa.build_parser().parse_args(["design", "--name", "x"])
    assert args.sessions == "02-08" and args.log2_n == 10 and args.threshold_mode == "sampled"
    assert args.search_space == str(psa.DEFAULT_SEARCH_SPACE) and args.patient == "sub-01"
    assert psa.DEFAULT_SEARCH_SPACE.name == "sailor_patient_v2_search_space.json"
    assert psa.parse_sessions(args.sessions) == [f"ses-{n:02d}" for n in range(2, 9)]


def test_search_space_refusals(tmp_path: Path):
    entries = json.loads(SEARCH_SPACE.read_text())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**entries, "resection_time": {"min": 30.0, "max": 200.0, "scale": "linear"}}))
    with pytest.raises(ValueError, match="resection_time"):
        psa.read_patient_search_space(bad, "profiled")
    bad.write_text(json.dumps({**entries, "rt_dose": "x.nii.gz"}))
    with pytest.raises(ValueError, match="rt_dose"):
        psa.read_patient_search_space(bad, "profiled")
    no_cheap = {k: ({kk: vv for kk, vv in v.items() if kk != "cheap"} if isinstance(v, dict) else v) for k, v in entries.items()}
    bad.write_text(json.dumps(no_cheap))
    with pytest.raises(ValueError, match="cheap"):
        psa.read_patient_search_space(bad, "sampled")
    bad_ratio = dict(entries)
    bad_ratio["edema_threshold_ratio"] = {"min": 0.25, "max": 1.0, "scale": "linear", "cheap": True}
    bad.write_text(json.dumps(bad_ratio))
    with pytest.raises(ValueError, match="below 1"):
        psa.read_patient_search_space(bad, "sampled")
    psa.read_patient_search_space(bad, "profiled")  # the ratio is not used in profiled mode
    no_preop = {k: v for k, v in entries.items() if k != "preop_time"}
    bad.write_text(json.dumps(no_preop))
    with pytest.raises(ValueError, match="preop_time"):
        psa.read_patient_search_space(bad, "profiled")


def test_session_snapshot_days_and_files():
    dt = 1.0 / 12
    moments = {"ses-01": 137.3, "ses-02": 152.3, "ses-06": 223.3}
    days = psa.session_snapshot_days(moments, dt)
    for name, t in moments.items():
        m = days[name] / dt
        assert m == pytest.approx(round(m))
        assert days[name] <= t - dt / 2 + 1e-9 and days[name] > t - 1.5 * dt
    assert psa.session_snapshot_days({"ses-01": 30.0}, dt)["ses-01"] == pytest.approx((30 * 12 - 1) * dt)
    with pytest.raises(ValueError, match="half a day"):
        psa.session_snapshot_days(moments, 1.0)
    assert psa.snapshot_file("ses-03") == "ses03_cell_density.nii.gz"
    assert psa.parse_snapshots(psa.format_snapshots(moments)) == moments
    with pytest.raises(ValueError, match="repeated"):
        psa.parse_snapshots("ses-01=1,ses-01=2")
    assert psa.Session("ses-03", "followup", date(2020, 1, 1)).prefix == "ses03_"
