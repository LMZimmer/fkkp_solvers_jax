#!/usr/bin/env python
"""Global sensitivity analysis of a scripts/run_stupp_sweep.py sweep.

Reads <sweep-dir>/sweep_results.csv (one row per configuration) and analyzes
how the quantity of interest -- the FINAL TUMOR MASS, column 'final_mass',
i.e. the run's final_stopping_quantity under stopping_mode="mass"
(voxel_volume * sum of the cell density) -- responds to the seven sampled
factors: rho, white_matter_diffusivity, diffusivity_ratio, resection_time,
chemo_kill_rate, chemo_decay_rate, rt_alpha. The quadratic radiosensitivity
is skipped (the analysed sweep sampled it as 0.1 * rt_alpha, perfectly
collinear; sweeps since the rt_alpha_beta_ratio refactor carry the
alpha/beta ratio column instead, which is not analysed either); the seed
voxel is not a scalar factor,
but the WM probability at the seed voxel (read from the sweep's WM pbmap) is
included as one scalar covariate when that map is readable. A dummy factor of
uniform noise is appended as a null baseline for every method.

Three analyses, written into <output-dir>/:

  1. Scatterplots: one panel per factor, QoI vs factor with all points
     (rasterized, low alpha) and a binned-median trend line;
     scatter_qoi.png/.pdf and the same grid with log10(QoI) on y,
     scatter_log10_qoi.png/.pdf.
  2. Monotone-regression diagnostics: per factor the Spearman rank
     correlation, the standardized rank regression coefficient (SRRC) and
     the partial rank correlation coefficient (PRCC), the latter two with
     bootstrap 95% confidence intervals. The R^2 of the full rank
     regression is reported prominently: below 0.7 the SRRC/PRCC ranking
     is unreliable and must not be interpreted (the numbers are still
     written, labelled as such).
  3. Regional SA / Monte Carlo filtering: configurations are split into
     behavioral (--behavioral-min <= QoI <= --behavioral-max) vs the rest;
     per factor the two marginal distributions are compared with a
     two-sample Kolmogorov-Smirnov test (D and p, with a Bonferroni note
     for the multiple tests) and plotted as paired empirical CDFs,
     regional_cdfs.png/.pdf. The default window is a PLACEHOLDER without
     clinical justification and requires review; a small table of the KS
     ranking under three alternative quantile-based windows
     (ks_window_sensitivity.csv) makes the threshold sensitivity visible.

Tabular output: sensitivity.csv (one row per factor: Spearman rho/p, SRRC
and PRCC with CIs, KS D and p) and summary.json (configuration count, QoI
column, rank-regression R^2, behavioral window and count, RNG seed).

Run from the project root, e.g.:
  python scripts/analyze_sweep_sensitivity.py --output-dir runs/sweep_sub15_sa
  python scripts/analyze_sweep_sensitivity.py --output-dir runs/sa \
      --behavioral-min 0 --behavioral-max 1e5 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

import numpy as np  # noqa: E402

DEFAULT_SWEEP_DIR = "/mnt/Drive4/lucas/SAILOR/sweep/sweep_sub15"
# PLACEHOLDER behavioral window (units of the QoI, i.e. integrated cell
# density in voxel volumes): "final mass at most 1e4", roughly the lower
# quartile of sweep_sub15. NOT clinically justified -- requires review.
DEFAULT_BEHAVIORAL_MIN = 0.0
DEFAULT_BEHAVIORAL_MAX = 1.0e4

QOI_COLUMN = "final_mass"
FACTORS = [
    "rho",
    "white_matter_diffusivity",
    "diffusivity_ratio",
    "resection_time",
    "chemo_kill_rate",
    "chemo_decay_rate",
    "rt_alpha",
]
SEED_WM = "seed_wm_probability"
DUMMY = "dummy_noise"
FACTOR_LABELS = {
    "rho": "rho [1/day]",
    "white_matter_diffusivity": "D_wm [mm^2/day]",
    "diffusivity_ratio": "diffusivity ratio",
    "resection_time": "resection time [day]",
    "chemo_kill_rate": "chemo kill rate [1/day per mg/m^2]",
    "chemo_decay_rate": "chemo decay rate [1/day]",
    "rt_alpha": "rt alpha [1/Gy]",
    SEED_WM: "WM prob. at seed voxel",
    DUMMY: "dummy (uniform noise)",
}
N_TREND_BINS = 30
ALTERNATIVE_WINDOW_QUANTILES = (0.25, 0.50, 0.75)  # behavioral = QoI <= q


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sweep-dir", default=DEFAULT_SWEEP_DIR, help="sweep directory with sweep_results.csv"
    )
    parser.add_argument("--output-dir", required=True, help="directory the analysis writes into")
    parser.add_argument(
        "--behavioral-min",
        type=float,
        default=DEFAULT_BEHAVIORAL_MIN,
        help="lower edge of the behavioral QoI window (PLACEHOLDER default, requires review)",
    )
    parser.add_argument(
        "--behavioral-max",
        type=float,
        default=DEFAULT_BEHAVIORAL_MAX,
        help="upper edge of the behavioral QoI window (PLACEHOLDER default, requires review)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed of the bootstrap and the dummy factor"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="bootstrap resamples for the SRRC/PRCC CIs"
    )
    return parser.parse_args(argv)


def load_sweep(sweep_dir: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    """Read sweep_results.csv and return (successful rows, factor matrix, QoI).

    Args:
        sweep_dir: Sweep directory containing sweep_results.csv.

    Returns:
        The successful CSV rows, the (n, len(FACTORS)) factor matrix in
        FACTORS order and the (n,) QoI vector.
    """
    csv_path = sweep_dir / "sweep_results.csv"
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [c for c in (*FACTORS, QOI_COLUMN, "success") if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} lacks the columns {missing}.")
    kept = [row for row in rows if row["success"] == "True"]
    if len(kept) < len(rows):
        print(f"dropping {len(rows) - len(kept)} non-successful configurations")
    if not kept:
        raise ValueError(f"no successful configurations in {csv_path}.")
    factors = np.array([[float(row[name]) for name in FACTORS] for row in kept])
    qoi = np.array([float(row[QOI_COLUMN]) for row in kept])
    return kept, factors, qoi


def seed_wm_covariate(sweep_dir: Path, rows: list[dict[str, str]]) -> np.ndarray | None:
    """WM probability at each configuration's seed voxel, or None.

    The WM pbmap path is taken from the first configuration's config.json
    (run_config.json in sweeps written before the config refactor); if the
    map (or the CSV's seed_voxel column) is not usable the covariate is
    dropped with a printed note.

    Args:
        sweep_dir: Sweep directory with the per-configuration run folders.
        rows: Successful sweep_results.csv rows.

    Returns:
        The (n,) WM probabilities, or None when the covariate is dropped.
    """
    import nibabel as nib

    if "seed_voxel" not in rows[0]:
        print("NOTE: no seed_voxel column; the seed-WM covariate is dropped.")
        return None
    try:
        run_dir = sweep_dir / rows[0]["config"]
        if (run_dir / "config.json").is_file():
            with open(run_dir / "config.json", encoding="utf-8") as handle:
                wm_path = json.load(handle)["white_matter_pbmap"]
        else:
            with open(run_dir / "run_config.json", encoding="utf-8") as handle:
                wm_path = json.load(handle)["tissue"]["wm"]
        wm = np.asarray(nib.load(wm_path).get_fdata(), dtype=np.float64)
    except Exception as error:  # noqa: BLE001 - any unreadable map drops the covariate
        print(f"NOTE: WM pbmap not readable ({error}); the seed-WM covariate is dropped.")
        return None
    voxels = np.array([[int(v) for v in row["seed_voxel"].split()] for row in rows])
    print(f"seed-WM covariate from {wm_path}")
    return wm[voxels[:, 0], voxels[:, 1], voxels[:, 2]]


def rank_correlations(ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Spearman, SRRC and PRCC of the QoI in one rank-transformed matrix.

    Args:
        ranked: (n, k+1) matrix of ranks, factors in the first k columns and
            the QoI in the last.

    Returns:
        (spearman, srrc, prcc) per factor and the rank-regression R^2, all
        derived from the correlation matrix of the ranks: SRRC are the
        standardized regression coefficients solve(R_xx, r_xy), R^2 is
        r_xy @ srrc, and PRCC comes from the inverse correlation matrix.
    """
    corr = np.corrcoef(ranked, rowvar=False)
    spearman = corr[:-1, -1]
    srrc = np.linalg.solve(corr[:-1, :-1], spearman)
    r_squared = float(spearman @ srrc)
    precision = np.linalg.inv(corr)
    prcc = -precision[:-1, -1] / np.sqrt(np.diag(precision)[:-1] * precision[-1, -1])
    return spearman, srrc, prcc, r_squared


def bootstrap_cis(
    data: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile 95% CIs of SRRC and PRCC under row resampling.

    Args:
        data: (n, k+1) raw matrix, factors then QoI; each resample is
            rank-transformed anew.
        n_bootstrap: Number of resamples.
        rng: Bootstrap random generator.

    Returns:
        (srrc_ci, prcc_ci), each (k, 2) with the 2.5% and 97.5% quantiles.
    """
    from scipy.stats import rankdata

    n = data.shape[0]
    srrc_samples = np.empty((n_bootstrap, data.shape[1] - 1))
    prcc_samples = np.empty_like(srrc_samples)
    for i in range(n_bootstrap):
        resample = rankdata(data[rng.integers(n, size=n)], axis=0)
        _, srrc_samples[i], prcc_samples[i], _ = rank_correlations(resample)
    quantiles = (0.025, 0.975)
    return (
        np.quantile(srrc_samples, quantiles, axis=0).T,
        np.quantile(prcc_samples, quantiles, axis=0).T,
    )


def binned_medians(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Median of y in n_bins equal-count bins of x.

    Args:
        x: Factor values.
        y: QoI values.
        n_bins: Number of equal-count bins.

    Returns:
        (bin centers as the median x per bin, median y per bin).
    """
    order = np.argsort(x)
    splits = np.array_split(order, n_bins)
    centers = np.array([np.median(x[s]) for s in splits])
    medians = np.array([np.median(y[s]) for s in splits])
    return centers, medians


def scatter_grid(
    path_stem: Path,
    names: list[str],
    factors: np.ndarray,
    y: np.ndarray,
    y_label: str,
    title: str,
) -> None:
    """One scatter panel per factor with a binned-median trend, PNG + PDF.

    Args:
        path_stem: Output path without extension.
        names: Factor names, one panel each.
        factors: (n, len(names)) factor matrix.
        y: Values on the y axes.
        y_label: Shared y-axis label.
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    n_cols = 3
    n_rows = -(-len(names) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), sharey=True, constrained_layout=True
    )
    for ax, name, column in zip(axes.flat, names, factors.T):
        ax.scatter(column, y, s=3, alpha=0.08, color="tab:blue", rasterized=True, linewidths=0)
        centers, medians = binned_medians(column, y, N_TREND_BINS)
        ax.plot(centers, medians, color="tab:red", linewidth=2, label="binned median")
        ax.set_xlabel(FACTOR_LABELS.get(name, name))
    for ax in axes.flat[: len(names) : n_cols]:
        ax.set_ylabel(y_label)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best", fontsize="small")
    fig.suptitle(title)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


def cdf_grid(
    path_stem: Path,
    names: list[str],
    factors: np.ndarray,
    behavioral: np.ndarray,
    ks_stats: list[tuple[float, float]],
    title: str,
) -> None:
    """Per factor the behavioral vs non-behavioral empirical CDFs, PNG + PDF.

    Args:
        path_stem: Output path without extension.
        names: Factor names, one panel each.
        factors: (n, len(names)) factor matrix.
        behavioral: Boolean behavioral mask.
        ks_stats: Per factor the (D, p) annotated into the panel.
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    n_cols = 3
    n_rows = -(-len(names) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), sharey=True, constrained_layout=True
    )
    for ax, name, column, (d, p) in zip(axes.flat, names, factors.T, ks_stats):
        for mask, label, color in (
            (behavioral, f"behavioral (n={int(behavioral.sum())})", "tab:red"),
            (~behavioral, f"non-behavioral (n={int((~behavioral).sum())})", "tab:blue"),
        ):
            values = np.sort(column[mask])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), color=color, label=label)
        ax.set_xlabel(FACTOR_LABELS.get(name, name))
        ax.set_title(f"KS D={d:.3f}, p={p:.2g}", fontsize="small")
    for ax in axes.flat[: len(names) : n_cols]:
        ax.set_ylabel("empirical CDF")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best", fontsize="small")
    fig.suptitle(title)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


def ks_per_factor(
    names: list[str], factors: np.ndarray, behavioral: np.ndarray
) -> list[tuple[float, float]]:
    """Two-sample KS (D, p) per factor, behavioral vs non-behavioral."""
    from scipy.stats import ks_2samp

    results = []
    for column in factors.T:
        test = ks_2samp(column[behavioral], column[~behavioral])
        results.append((float(test.statistic), float(test.pvalue)))
    return results


def ranking(names: list[str], scores: np.ndarray) -> list[str]:
    """The factor names sorted by descending |score|."""
    return [names[i] for i in np.argsort(-np.abs(scores))]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, factors, qoi = load_sweep(sweep_dir)
    names = list(FACTORS)
    wm_probability = seed_wm_covariate(sweep_dir, rows)
    if wm_probability is not None:
        names.append(SEED_WM)
        factors = np.column_stack([factors, wm_probability])
    rng = np.random.default_rng(args.seed)
    names.append(DUMMY)
    factors = np.column_stack([factors, rng.uniform(size=len(qoi))])
    print(f"{len(qoi)} configurations, QoI column {QOI_COLUMN!r}, factors: {', '.join(names)}")

    # 1. Scatterplots, linear and log10 QoI.
    scatter_grid(
        output_dir / "scatter_qoi",
        names,
        factors,
        qoi,
        f"final tumor mass ({QOI_COLUMN})",
        f"{sweep_dir.name}: QoI vs factors",
    )
    positive = qoi > 0
    if not positive.all():
        print(f"NOTE: {int((~positive).sum())} non-positive QoI values dropped in the log10 grid")
    scatter_grid(
        output_dir / "scatter_log10_qoi",
        names,
        factors[positive],
        np.log10(qoi[positive]),
        f"log10 final tumor mass ({QOI_COLUMN})",
        f"{sweep_dir.name}: log10(QoI) vs factors",
    )

    # 2. Monotone-regression diagnostics on the ranks.
    from scipy.stats import rankdata, spearmanr

    data = np.column_stack([factors, qoi])
    spearman, srrc, prcc, r_squared = rank_correlations(rankdata(data, axis=0))
    spearman_p = np.array([spearmanr(column, qoi).pvalue for column in factors.T])
    srrc_ci, prcc_ci = bootstrap_cis(data, args.n_bootstrap, rng)
    prcc_reliable = r_squared >= 0.7
    reliability = (
        f"rank-regression R^2 = {r_squared:.3f}: "
        + (
            "the SRRC/PRCC ranking is interpretable (R^2 >= 0.7)."
            if prcc_reliable
            else "R^2 < 0.7 -- the SRRC/PRCC ranking is UNRELIABLE and must not be "
            "interpreted; the numbers below are reported but so labelled."
        )
    )
    print(f"\n{reliability}")

    # 3. Regional SA / Monte Carlo filtering.
    behavioral = (qoi >= args.behavioral_min) & (qoi <= args.behavioral_max)
    if not 0 < int(behavioral.sum()) < len(qoi):
        raise ValueError(
            f"behavioral window [{args.behavioral_min:g}, {args.behavioral_max:g}] leaves "
            f"{int(behavioral.sum())} of {len(qoi)} configurations behavioral; both groups "
            "must be non-empty."
        )
    ks_stats = ks_per_factor(names, factors, behavioral)
    bonferroni = 0.05 / len(names)
    print(
        f"\nbehavioral window [{args.behavioral_min:g}, {args.behavioral_max:g}] "
        f"(PLACEHOLDER, requires review): {int(behavioral.sum())}/{len(qoi)} behavioral; "
        f"{len(names)} KS tests, Bonferroni-adjusted significance level "
        f"0.05/{len(names)} = {bonferroni:.2g}"
    )
    cdf_grid(
        output_dir / "regional_cdfs",
        names,
        factors,
        behavioral,
        ks_stats,
        f"{sweep_dir.name}: behavioral filtering, QoI in "
        f"[{args.behavioral_min:g}, {args.behavioral_max:g}] (placeholder window)",
    )

    # KS ranking under alternative windows, so the threshold sensitivity of
    # the regional result is visible.
    window_rows = []
    for quantile in ALTERNATIVE_WINDOW_QUANTILES:
        edge = float(np.quantile(qoi, quantile))
        alt = qoi <= edge
        alt_stats = ks_per_factor(names, factors, alt)
        alt_rank = ranking(names, np.array([d for d, _ in alt_stats]))
        for name, (d, p) in zip(names, alt_stats):
            window_rows.append(
                {
                    "window": f"QoI <= q{int(quantile * 100)} ({edge:.4g})",
                    "n_behavioral": int(alt.sum()),
                    "factor": name,
                    "ks_D": d,
                    "ks_p": p,
                    "rank": alt_rank.index(name) + 1,
                }
            )
    with open(output_dir / "ks_window_sensitivity.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)

    # Tables and summary.
    columns = [
        "factor",
        "spearman_rho",
        "spearman_p",
        "srrc",
        "srrc_ci_low",
        "srrc_ci_high",
        "prcc",
        "prcc_ci_low",
        "prcc_ci_high",
        "ks_D",
        "ks_p",
    ]
    with open(output_dir / "sensitivity.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow(
                {
                    "factor": name,
                    "spearman_rho": spearman[i],
                    "spearman_p": spearman_p[i],
                    "srrc": srrc[i],
                    "srrc_ci_low": srrc_ci[i, 0],
                    "srrc_ci_high": srrc_ci[i, 1],
                    "prcc": prcc[i],
                    "prcc_ci_low": prcc_ci[i, 0],
                    "prcc_ci_high": prcc_ci[i, 1],
                    "ks_D": ks_stats[i][0],
                    "ks_p": ks_stats[i][1],
                }
            )
    summary: dict[str, Any] = {
        "sweep_dir": str(sweep_dir),
        "n_configs": len(qoi),
        "qoi_column": QOI_COLUMN,
        "factors": names,
        "seed_wm_covariate_included": wm_probability is not None,
        "rank_regression_r_squared": r_squared,
        "srrc_prcc_interpretable": bool(prcc_reliable),
        "reliability_note": reliability,
        "behavioral_window": [args.behavioral_min, args.behavioral_max],
        "behavioral_window_is_placeholder": True,
        "n_behavioral": int(behavioral.sum()),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'factor':<26} {'Spearman':>9} {'SRRC [95% CI]':>26} {'PRCC [95% CI]':>26} {'KS D':>7} {'KS p':>9}")
    for i, name in enumerate(names):
        print(
            f"{name:<26} {spearman[i]:>9.3f} "
            f"{srrc[i]:>8.3f} [{srrc_ci[i, 0]:>6.3f}, {srrc_ci[i, 1]:>6.3f}] "
            f"{prcc[i]:>8.3f} [{prcc_ci[i, 0]:>6.3f}, {prcc_ci[i, 1]:>6.3f}] "
            f"{ks_stats[i][0]:>7.3f} {ks_stats[i][1]:>9.2g}"
        )
    print(f"\nranking |Spearman|: {', '.join(ranking(names, spearman))}")
    label = "" if prcc_reliable else " (UNRELIABLE, R^2 < 0.7)"
    print(f"ranking |SRRC|{label}: {', '.join(ranking(names, srrc))}")
    print(f"ranking |PRCC|{label}: {', '.join(ranking(names, prcc))}")
    print(f"ranking KS D (placeholder window): {', '.join(ranking(names, np.array([d for d, _ in ks_stats])))}")
    print(f"\noutputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
