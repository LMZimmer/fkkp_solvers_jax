#!/usr/bin/env python
"""Plot the optimizer trajectories of several scripts/patient_cmaes_fit.py fits
in one figure, in the layout of the usual CMA-ES inspection plot (the cma
package's four panels), one row per fit:

  loss          per generation the best and the median loss of the
                successful members and the best loss so far (the start
                evaluation included), linear axis
  mean          the distribution mean (xmean) per factor in the unit cube
                the optimizer works on; hairlines at the bounds 0 and 1
  principal     the principal axis lengths of the covariance, sqrt(eig(C)),
  axes          sorted (no identity per line), log axis
  std           the standard deviation per coordinate, sigma * sqrt(C_ii),
                per factor, and the step size sigma itself, log axis

The x axis of every panel is the number of function evaluations at the end
of a generation (the start evaluation counts), shared by all rows, so an
unfinished fit ends early. A fit with several restarts is drawn restart
after restart along that axis with a vertical hairline at each restart.

Factor colours follow the factor's group, the same in every row whatever
the seed count (growth: front speed, front width, growth length; seed
shape: seed sigma, seed peak; treatment: rt_alpha, chemo kill rate; one
group per seed position, seed_bbox_x/y/z[_n]); members of a group differ by
line style. The colours are the first eight categorical slots of the
dataviz reference palette in its fixed order.

Input: the fit directories under --fit-root (every subdirectory with
restart_<n>/trace.npz, sorted by name) or the names given with --fits.
Per restart the script reads trace.npz (gen, xmean, sigma, C),
evaluations.csv (generation, loss, success, the u_<factor> column order)
and state.npz (complete, termination, incumbent), and spec.json's cmaes
generations for the budget. Nothing is written into the fit directories.

Output: <output>.pdf and <output>.png (the figure) and <output>.csv (the
table view: per fit the last traced generation, evaluations, best loss
and its evaluation, sigma, axis ratio, and per factor the final mean and
standard deviation).

Run from the project root, e.g.:
  python scripts/visualize_cmaes_trajectory.py \\
      --fit-root /mnt/Drive4/lucas/stupp_patient_fit_respond \\
      --output runs/respond_best_2026-11-09/cmaes_trajectories
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

TRACE_FILE = "trace.npz"
STATE_FILE = "state.npz"
EVALUATIONS_FILE = "evaluations.csv"
UNIT_PREFIX = "u_"

# dataviz reference palette, light mode: categorical slots in their fixed order,
# the text tokens and a recessive gridline gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
LINE_STYLES = ["-", "--", ":", "-."]
LINE_WIDTH = 1.4
SIGMA_STYLE = (0, (7, 2))  # long dashes: sigma, black, apart from the coloured factor lines

# Factor groups in their colour order; the seed positions take the slots after
# the three fixed groups (seed n -> slot 3 + n - 1), so a group keeps its colour
# whether a fit has one seed or several.
FIXED_GROUPS: dict[str, list[str]] = {
    "growth": ["front_speed_mm_per_day", "front_width_mm", "growth_length_mm"],
    "seed shape": ["seed_sigma_mm", "seed_peak_density"],
    "treatment": ["rt_alpha", "chemo_kill_rate"],
}
SEED_POSITION = re.compile(r"^seed_bbox_([xyz])(?:_(\d+))?$")
AXIS_ORDER = {"x": 0, "y": 1, "z": 2}


@dataclass
class Restart:
    index: int
    gen: NDArray            # (G,) generation numbers, 1-based
    evals: NDArray          # (G,) evaluations at the end of each generation (all restarts before included)
    start_evals: int        # evaluations up to and including this restart's start evaluation(s)
    start_best: float       # best loss so far after the start evaluation(s)
    best: NDArray           # (G,) best successful loss of the generation (NaN if none)
    median: NDArray         # (G,) median successful loss of the generation
    best_so_far: NDArray    # (G,) best loss over the fit up to the generation's end
    xmean: NDArray          # (G, N)
    sigma: NDArray          # (G,)
    axes: NDArray           # (G, N) sqrt(eig(C)), ascending
    std: NDArray            # (G, N) sigma * sqrt(diag(C))


@dataclass
class Fit:
    name: str
    factors: list[str]
    restarts: list[Restart]
    status: str
    best_loss: float
    best_eval_id: int


def factor_style(name: str) -> tuple[str, str, str]:
    """(group label, colour, line style) of a factor."""
    for slot, (group, members) in enumerate(FIXED_GROUPS.items()):
        if name in members:
            return group, SERIES[slot], LINE_STYLES[members.index(name)]
    match = SEED_POSITION.match(name)
    if match:
        seed = int(match.group(2) or 1)
        slot = len(FIXED_GROUPS) + seed - 1
        if slot >= len(SERIES):
            raise ValueError(f"factor {name}: more seed positions than the {len(SERIES)} palette slots allow.")
        return f"seed {seed} position", SERIES[slot], LINE_STYLES[AXIS_ORDER[match.group(1)]]
    raise ValueError(f"factor {name} belongs to no known group; add it to FIXED_GROUPS.")


def read_evaluations(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    factors = [f[len(UNIT_PREFIX):] for f in fields if f.startswith(UNIT_PREFIX)]
    return factors, rows


def successful_losses(rows: list[dict[str, str]]) -> tuple[NDArray, NDArray]:
    """(generation, loss) of the successful evaluations with a finite loss."""
    gens, losses = [], []
    for row in rows:
        if row["success"] != "True":
            continue
        loss = float(row["loss"]) if row["loss"] else float("nan")
        if np.isfinite(loss):
            gens.append(int(row["generation"]))
            losses.append(loss)
    return np.asarray(gens, dtype=np.int64), np.asarray(losses, dtype=np.float64)


def load_fit(fit_dir: Path) -> Fit:
    restart_dirs = sorted(
        (d for d in fit_dir.glob("restart_*") if (d / TRACE_FILE).is_file()), key=lambda d: int(d.name.split("_")[1])
    )
    if not restart_dirs:
        raise FileNotFoundError(f"{fit_dir}: no restart_<n>/{TRACE_FILE}.")
    spec_path = fit_dir / "spec.json"
    budgets = json.loads(spec_path.read_text())["cmaes"]["generations"] if spec_path.is_file() else []
    factors: list[str] | None = None
    restarts: list[Restart] = []
    offset = 0
    best_overall = float("inf")
    statuses = []
    best_loss, best_eval_id = float("inf"), -1
    for restart_dir in restart_dirs:
        index = int(restart_dir.name.split("_")[1])
        restart_factors, rows = read_evaluations(restart_dir / EVALUATIONS_FILE)
        if factors is None:
            factors = restart_factors
        elif restart_factors != factors:
            raise ValueError(f"{restart_dir}: factors {restart_factors} differ from the first restart's {factors}.")
        trace = np.load(restart_dir / TRACE_FILE)
        gen = np.asarray(trace["gen"], dtype=np.int64)
        xmean = np.asarray(trace["xmean"], dtype=np.float64)
        if xmean.shape[1] != len(factors):
            raise ValueError(f"{restart_dir}: trace has {xmean.shape[1]} coordinates, evaluations.csv {len(factors)} factors.")
        sigma = np.asarray(trace["sigma"], dtype=np.float64)
        cov = np.asarray(trace["C"], dtype=np.float64)
        all_gens = np.asarray([int(r["generation"]) for r in rows], dtype=np.int64)
        ok_gens, ok_losses = successful_losses(rows)

        start_mask = ok_gens == 0
        start_evals = offset + int((all_gens == 0).sum())
        if start_mask.any():
            best_overall = min(best_overall, float(ok_losses[start_mask].min()))
        start_best = best_overall
        evals = np.empty(gen.size)
        best = np.full(gen.size, np.nan)
        median = np.full(gen.size, np.nan)
        best_so_far = np.empty(gen.size)
        for i, g in enumerate(gen):
            evals[i] = offset + int((all_gens <= g).sum())
            losses = ok_losses[ok_gens == g]
            if losses.size:
                best[i], median[i] = losses.min(), np.median(losses)
                best_overall = min(best_overall, float(best[i]))
            best_so_far[i] = best_overall
        axes = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0.0, None))
        std = sigma[:, None] * np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0.0, None))
        restarts.append(Restart(index, gen, evals, start_evals, start_best, best, median, best_so_far, xmean, sigma, axes, std))
        offset += len(rows)

        state = np.load(restart_dir / STATE_FILE, allow_pickle=True)
        generation = int(state["generation"])
        budget = budgets[index] if index < len(budgets) else (budgets[-1] if budgets else None)
        of_budget = f"/{budget}" if budget is not None else ""
        if bool(state["complete"]):
            statuses.append(f"done ({state['termination']}, {generation}{of_budget} gen)")
        else:
            statuses.append(f"incomplete ({generation}{of_budget} gen)")
        if float(state["incumbent_loss"]) < best_loss:
            best_loss, best_eval_id = float(state["incumbent_loss"]), int(state["incumbent_eval_id"])
    status = "; ".join(statuses) if len(statuses) > 1 else statuses[0]
    return Fit(fit_dir.name, factors or [], restarts, status, best_loss, best_eval_id)


def find_fits(fit_root: Path, names: list[str] | None) -> list[Path]:
    if names:
        dirs = [fit_root / name for name in names]
        missing = [str(d) for d in dirs if not any(d.glob(f"restart_*/{TRACE_FILE}"))]
        if missing:
            raise FileNotFoundError(f"no restart_<n>/{TRACE_FILE} in {missing}.")
        return dirs
    dirs = sorted(d for d in fit_root.iterdir() if d.is_dir() and any(d.glob(f"restart_*/{TRACE_FILE}")))
    if not dirs:
        raise FileNotFoundError(f"no fit directory with restart_<n>/{TRACE_FILE} under {fit_root}.")
    return dirs


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def draw_restart_lines(ax: plt.Axes, fit: Fit) -> None:
    for restart in fit.restarts[1:]:
        ax.axvline(restart.start_evals, color=TEXT_SECONDARY, linewidth=0.6)


def render(fits: list[Fit], output: Path) -> None:
    n_rows = len(fits)
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(19.0, 2.5 * n_rows + 1.6), sharex=True, squeeze=False, constrained_layout=True
    )
    fig.patch.set_facecolor(SURFACE)
    # The loss column carries each fit's title instead (its y label names the quantity).
    titles = {
        1: "distribution mean (unit cube)",
        2: "principal axis lengths  sqrt(eig(C))",
        3: "standard deviations  sigma * sqrt(C_ii)  and sigma",
    }
    for col, title in titles.items():
        axes[0, col].set_title(title, fontsize=11, color=TEXT_PRIMARY, pad=10)
    for row, fit in enumerate(fits):
        ax_loss, ax_mean, ax_axes, ax_std = axes[row]
        for ax in axes[row]:
            style_axis(ax)
            draw_restart_lines(ax, fit)
        for restart in fit.restarts:
            # the best-so-far line starts at the start evaluation
            ax_loss.plot(
                np.concatenate([[restart.start_evals], restart.evals]),
                np.concatenate([[restart.start_best], restart.best_so_far]),
                color=TEXT_PRIMARY, linewidth=LINE_WIDTH + 0.4, drawstyle="steps-post",
            )
            ax_loss.plot(restart.evals, restart.best, color=SERIES[0], linewidth=LINE_WIDTH * 0.8)
            ax_loss.plot(restart.evals, restart.median, color=TEXT_SECONDARY, linewidth=LINE_WIDTH * 0.8, alpha=0.7)
            for k, factor in enumerate(fit.factors):
                _, color, style = factor_style(factor)
                ax_mean.plot(restart.evals, restart.xmean[:, k], color=color, linestyle=style, linewidth=LINE_WIDTH)
                ax_std.plot(restart.evals, restart.std[:, k], color=color, linestyle=style, linewidth=LINE_WIDTH)
            ax_axes.plot(restart.evals, restart.axes, color=TEXT_SECONDARY, linewidth=LINE_WIDTH * 0.8)
            ax_std.plot(restart.evals, restart.sigma, color=TEXT_PRIMARY, linewidth=LINE_WIDTH + 0.6, linestyle=SIGMA_STYLE)
        ax_loss.set_title(
            f"{fit.name}   best {fit.best_loss:.4f} (eval {fit.best_eval_id}), {fit.status}",
            fontsize=9.5, color=TEXT_PRIMARY, loc="left",
        )
        for bound in (0.0, 1.0):
            ax_mean.axhline(bound, color=TEXT_SECONDARY, linewidth=0.6)
        means = np.concatenate([r.xmean.ravel() for r in fit.restarts])
        ax_mean.set_ylim(min(-0.05, means.min() - 0.02), max(1.05, means.max() + 0.02))
        ax_axes.set_yscale("log")
        ax_std.set_yscale("log")
        ax_loss.set_ylabel("loss", fontsize=9, color=TEXT_SECONDARY)
    for ax in axes[-1]:
        ax.set_xlabel("function evaluations", fontsize=9, color=TEXT_SECONDARY)

    loss_handles = [
        Line2D([], [], color=TEXT_PRIMARY, linewidth=LINE_WIDTH + 0.4, label="best so far"),
        Line2D([], [], color=SERIES[0], linewidth=LINE_WIDTH * 0.8, label="generation best"),
        Line2D([], [], color=TEXT_SECONDARY, linewidth=LINE_WIDTH * 0.8, alpha=0.7, label="generation median"),
        Line2D([], [], color=TEXT_PRIMARY, linewidth=LINE_WIDTH + 0.6, linestyle=SIGMA_STYLE, label="sigma (std panel)"),
    ]
    # One legend entry per (colour, line style): a one-seed fit's seed_bbox_x and a
    # multi-seed fit's seed_bbox_x_1 are the same entry.
    entries: dict[tuple[str, str], tuple[str, list[str]]] = {}
    for fit in fits:
        for factor in fit.factors:
            group, color, style = factor_style(factor)
            names = entries.setdefault((color, style), (group, []))[1]
            if factor not in names:
                names.append(factor)
    factor_handles = [
        Line2D([], [], color=color, linestyle=style, linewidth=LINE_WIDTH, label=f"{' / '.join(names)} ({group})")
        for (color, style), (group, names) in sorted(
            entries.items(), key=lambda item: (SERIES.index(item[0][0]), LINE_STYLES.index(item[0][1]))
        )
    ]
    fig.legend(
        handles=loss_handles + factor_handles, loc="outside upper center", ncol=6, fontsize=8.5, frameon=False,
        labelcolor=TEXT_PRIMARY, handlelength=3.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), format="pdf", facecolor=SURFACE)
    fig.savefig(output.with_suffix(".png"), dpi=90, facecolor=SURFACE)
    plt.close(fig)


def write_table(fits: list[Fit], output: Path) -> None:
    factors: list[str] = []
    for fit in fits:
        factors.extend(f for f in fit.factors if f not in factors)
    header = ["fit", "status", "generation", "evaluations", "best_loss", "best_eval_id", "sigma", "axis_ratio"]
    header += [f"mean_{f}" for f in factors] + [f"std_{f}" for f in factors]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for fit in fits:
            last = fit.restarts[-1]
            mean = dict(zip(fit.factors, last.xmean[-1], strict=True))
            std = dict(zip(fit.factors, last.std[-1], strict=True))
            writer.writerow(
                [fit.name, fit.status, int(last.gen[-1]), int(last.evals[-1]), fit.best_loss, fit.best_eval_id,
                 float(last.sigma[-1]), float(last.axes[-1, -1] / last.axes[-1, 0])]
                + [mean.get(f, "") for f in factors] + [std.get(f, "") for f in factors]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-root", required=True, help="directory holding the fit directories")
    parser.add_argument("--fits", nargs="+", default=None, help="fit directory names below --fit-root (default: all)")
    parser.add_argument("--output", required=True, help="output path without suffix (.pdf, .png and .csv are written)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fits = [load_fit(d) for d in find_fits(Path(args.fit_root), args.fits)]
    for fit in fits:
        for factor in fit.factors:
            factor_style(factor)  # raises on an unknown factor before anything is drawn
        print(f"{fit.name}: {fit.status}, best {fit.best_loss:.4f} (eval {fit.best_eval_id}), {len(fit.factors)} factors")
    output = Path(args.output)
    render(fits, output)
    write_table(fits, output)
    print(f"saved {output.with_suffix('.pdf')}, {output.with_suffix('.png')} and {output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
