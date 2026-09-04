"""Helpers shared by the Stupp scripts: JSON-serializable parameter
summaries and the treatment-course figure (3x3 montage of axial slices plus
the total mass over time with the treatment events marked).

``render`` imports matplotlib when called, so matplotlib is needed only by
callers that draw the figure, not by the solvers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CAVITY_COLOR = (210 / 255.0, 43 / 255.0, 43 / 255.0, 1)
SEED_COLOR = (34 / 255.0, 139 / 255.0, 34 / 255.0, 1)


def jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, paths, non-finite floats and nested
    containers to JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    """The params without the volumes (arrays of more than one dimension),
    JSON-serializable."""
    return {
        key: jsonable(value)
        for key, value in params.items()
        if not (isinstance(value, np.ndarray) and value.ndim > 1)
    }


def montage_days(
    resection_time: float, dt: float, rt_times: np.ndarray, stopping_time: float
) -> list[float]:
    """
    Days of the eight recorded montage panels (the seed is the initial
    state): one step before and one day after the resection; three evenly
    spaced days inside the radiotherapy block (first to last fraction,
    endpoints excluded); three evenly spaced days from the last fraction to
    the end of the run, the end included. Passed as the solver's
    snapshot_times, so each frame is recorded at the nearest time step.
    """
    rt_start, rt_end = float(np.min(rt_times)), float(np.max(rt_times))
    return [
        resection_time - dt,
        resection_time + 1.0,
        *np.linspace(rt_start, rt_end, 5)[1:-1],
        *np.linspace(rt_end, stopping_time, 4)[1:],
    ]


def select_panels(
    initial: np.ndarray, frames: np.ndarray, times: np.ndarray, days: list[float]
) -> list[tuple[str, np.ndarray]]:
    """
    The nine montage panels, three per row: the seed, then for each of
    the ``montage_days`` the recorded frame nearest to it.

    Args:
        initial: Initial state (t = 0).
        frames: Recorded frames of the treated run, (n_frames, ...).
        times: Simulation day of each frame (Result.snapshot_times).
        days: The eight montage days the frames were recorded for.

    Returns:
        (title, volume) pairs.
    """
    labels = [" (before resection)", " (after resection)", *[" (RT/TMZ)"] * 3, *[""] * 3]
    panels = [("t = 0 d (seed)", initial)]
    for day, label in zip(days, labels, strict=True):
        k = int(np.argmin(np.abs(times - day)))
        panels.append((f"t = {times[k]:.0f} d{label}", frames[k]))
    return panels


def session_blocks(days: np.ndarray, max_gap: float = 1.0) -> list[tuple[float, float]]:
    """(first, last) day of each run of sessions that are at most max_gap
    days apart, for sorted session days."""
    blocks: list[tuple[float, float]] = []
    for day in days:
        if blocks and day - blocks[-1][1] <= max_gap:
            blocks[-1] = (blocks[-1][0], float(day))
        else:
            blocks.append((float(day), float(day)))
    return blocks


def render(
    outfile_stem: Path,
    header: str,
    panels: list[tuple[str, np.ndarray]],
    background_volume: np.ndarray,
    cavity: np.ndarray | None,
    seed_voxel: tuple[int, int, int],
    z: int,
    threshold: float,
    times: np.ndarray,
    masses: np.ndarray,
    params: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_col = 3
    n_row = int(np.ceil(len(panels) / n_col))
    fig = plt.figure(figsize=(5 * n_col + 0.6, 4 * n_row + 3.6), constrained_layout=True)
    grid = fig.add_gridspec(
        n_row + 1, n_col + 1, height_ratios=[1.0] * n_row + [0.75],
        width_ratios=[1.0] * n_col + [0.05],
    )
    background = np.rot90(background_volume[:, :, z])
    cavity_slice = np.rot90(cavity[:, :, z]) if cavity is not None else None
    # np.rot90 maps array (i, j) to image row (ny - 1 - j), column i.
    seed_col, seed_row = seed_voxel[0], background_volume.shape[1] - 1 - seed_voxel[1]
    image = None
    for index, (title, volume) in enumerate(panels):
        ax = fig.add_subplot(grid[index // n_col, index % n_col])
        ax.imshow(background, cmap="gray", interpolation="none")
        overlay = np.ma.masked_less(np.rot90(volume[:, :, z]), threshold)
        image = ax.imshow(
            overlay, cmap="inferno", alpha=0.90, vmin=0.0, vmax=1.0, interpolation="none"
        )
        if cavity_slice is not None and cavity_slice.any():
            ax.contour(cavity_slice.astype(float), levels=[0.5], colors=[CAVITY_COLOR], linewidths=1.2)
        if seed_voxel[2] == z:
            ax.plot(seed_col, seed_row, "+", color=SEED_COLOR, markersize=10, markeredgewidth=1.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(
            0.02, 0.02, f"mass {volume.sum():.0f}", transform=ax.transAxes,
            color="white", fontsize=10, va="bottom",
        )
        ax.axis("off")
    if image is not None:
        cax = fig.add_subplot(grid[:n_row, n_col])
        fig.colorbar(image, cax=cax, label="cell density")

    ax = fig.add_subplot(grid[n_row, :n_col])
    ax.plot(times, masses, "ko-", markersize=3, label="mass")
    ax.axvline(
        float(params["resection_time"]), color=CAVITY_COLOR, linewidth=1.5, label="resection"
    )
    # Radiotherapy days as lines, split by whether chemotherapy is given the
    # same day; chemotherapy-only days (daily sessions, many of them) as one
    # shaded band per contiguous block of sessions.
    rt_days = np.atleast_1d(np.asarray(params["rt_times"], dtype=np.float64))
    ct_days = np.atleast_1d(np.asarray(params["chemo_times"], dtype=np.float64))
    with_ct = np.isin(rt_days, ct_days)
    for days, label, color in (
        (rt_days[with_ct], "radio- + chemotherapy", "blue"),
        (rt_days[~with_ct], "radiotherapy only", "purple"),
    ):
        for index, t in enumerate(days):
            ax.axvline(
                float(t), color=color, linewidth=0.6, alpha=0.4, label=label if index == 0 else None
            )
    ct_only = np.sort(ct_days[~np.isin(ct_days, rt_days)])
    for index, (start, end) in enumerate(session_blocks(ct_only)):
        ax.axvspan(
            start - 0.5, end + 0.5, color="green", alpha=0.25, linewidth=0,
            label="chemotherapy only" if index == 0 else None,
        )
    ax.set_yscale("log")
    ax.set_xlabel("time [days]", fontsize=12)
    ax.set_ylabel("total mass", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    fig.suptitle(header, horizontalalignment="left", x=0.02, fontsize=15, fontweight="bold")
    fig.savefig(str(outfile_stem) + ".png", dpi=110)
    fig.savefig(str(outfile_stem) + ".pdf", format="pdf")
    plt.close(fig)
