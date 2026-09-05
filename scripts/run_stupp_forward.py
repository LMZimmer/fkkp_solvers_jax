#!/usr/bin/env python
"""Forward-solve one treated tumor growth with fisher_kpp_jax.StuppFKPPSolver.

Reads a JSON config (scripts/stupp_config_example.json by default; see
fisher_kpp_jax.read_config: the StuppFKPPSolver parameters by name, the
tissue probability maps, the labelled segmentation of the resection cavity
and the TOTAL-dose map as NIfTI paths, the time step as steps_per_day, dt
or n_steps), runs the solver and saves the run into its own directory
(fisher_kpp_jax.Result.save):

  <output-dir>/<run-name>/
    config.json                the config of the run, loadable as is
    result.json                success, error, final time, stopping criterion,
                               final stopping quantity, n_steps, dt, wall time,
                               derived values
    initial_cell_density.nii.gz, final_cell_density.nii.gz
                               float32, affine of the wm pbmap
    overview.png (+ .pdf)      unless --no-plot: the 3x3 montage of
                               scripts/run_stupp_example.py (seed, before and
                               after resection; three frames inside the
                               radiotherapy block; three from its end to the
                               end of the run) on the axial slice through the
                               dose map's center of mass, over
                               --background-image or the tissue maps, plus
                               total mass vs time with the treatment events
                               marked

The run directory is created with exist_ok=False and nothing is written
outside it, so many evaluations can run in parallel against the same
output-dir (the default run name carries a UTC timestamp and the pid).
The eight montage frames are the only snapshots recorded, and only for the
plot; with --no-plot nothing is recorded (memory-light for parallel sweeps).

Every parameter comes from the config: edit it, or write one as
scripts/run_stupp_sweep.py does. The horizon is resection_time +
time_after_resection (the run ends that many days after the resection).

Run from the project root, e.g.:
  JAX_PLATFORMS=cpu python scripts/run_stupp_forward.py --output-dir runs/
  python scripts/run_stupp_forward.py --config my_run.json --output-dir runs/ --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Keep XLA from grabbing 75% of a (possibly shared) GPU; must be set before
# jax initializes the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# This machine has more cores than the bundled OpenBLAS's 128-thread build
# limit; cap it so NumPy/SciPy teardown does not emit thread-region warnings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import center_of_mass  # noqa: E402

from fisher_kpp_jax import StuppFKPPSolver, read_config  # noqa: E402
from fisher_kpp_jax.util import montage_days, render, select_panels  # noqa: E402

DEFAULT_CONFIG = str(_ROOT / "scripts" / "stupp_config_example.json")
DEFAULT_THRESHOLD = 0.01  # overlay transparency threshold (cell density)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="run config JSON")
    parser.add_argument("--output-dir", required=True, help="parent of the run directory")
    parser.add_argument(
        "--run-name",
        default=None,
        help="run directory name (default: stupp_<UTC timestamp>_<pid>)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="cell density below which the overlay is transparent",
    )
    parser.add_argument(
        "--background-image",
        default=None,
        help="background image NIfTI for overview.png (default: wm + gm pbmaps)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip overview.png and request no time series",
    )
    return parser.parse_args(argv)


def default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"stupp_{stamp}_{os.getpid()}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or default_run_name()
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    config = read_config(args.config)
    config["verbose"] = True
    solver = StuppFKPPSolver(config)
    if not isinstance(solver, StuppFKPPSolver):
        raise ValueError(f"{args.config} names {type(solver).__name__}, not StuppFKPPSolver.")
    panel_days = None
    if not args.no_plot:
        # The montage frames are the only snapshots; their days need the
        # time step, so the solver is built again with them requested.
        _, dt = solver.resolve_time_stepping()
        panel_days = montage_days(
            float(solver.params["resection_time"]),
            dt,
            solver.params["rt_times"],
            float(solver.params["stopping_time"]),
        )
        solver = StuppFKPPSolver({**config, "snapshot_times": panel_days})
    params = solver.params
    wm, gm = params["white_matter_pbmap"], params["gray_matter_pbmap"]

    print(f"run directory: {run_dir}")
    print(f"tissue: wm={config['white_matter_pbmap']} gm={config['gray_matter_pbmap']}")
    print(f"grid: {wm.shape}, voxel size {params['voxel_size_mm']} mm")
    result = solver.solve(store_result=True, outdir=run_dir)
    if not result.success:
        print(f"FAILED after {result.wall_time_s:.1f} s: {result.error}", file=sys.stderr)
        return 1

    cell_density = result.final_state["cell_density"]
    if not args.no_plot:
        frames = result.time_series["cell_density"]
        times = result.snapshot_times
        masses = frames.sum(axis=(1, 2, 3))
        seed_voxel = tuple(
            int(params[f"gaussian_seed_{axis}_fraction"] * n) for axis, n in zip("xyz", wm.shape)
        )
        if args.background_image is not None:
            background = np.asarray(nib.load(args.background_image).get_fdata(), dtype=np.float64)
            if background.shape != wm.shape:
                raise ValueError(
                    f"--background-image shape {background.shape} differs from the tissue maps {wm.shape}."
                )
        else:
            background = wm + gm
        z = int(round(center_of_mass(params["rt_dose"])[2]))
        panels = select_panels(result.initial_state["cell_density"], frames, times, panel_days)
        header = (
            f"{run_name}: {Path(args.config).name}, axial slice z={z}, "
            f"densities >= {args.threshold:g}\n"
            f"rho {params['rho']:g}, D {params['white_matter_diffusivity']:g}, "
            f"ratio {params['diffusivity_ratio']:g}, resolution "
            f"{params['resolution_factor']:g}, {result.n_steps} steps"
        )
        render(
            run_dir / "overview",
            header,
            panels,
            background,
            params["resection_cavity"],
            seed_voxel,
            z,
            args.threshold,
            times,
            masses,
            params,
        )

    print(
        f"done in {result.wall_time_s:.1f} s: final_time={result.final_time:g}, "
        f"criterion={result.stopping_criterion}, "
        f"final mass={result.final_stopping_quantity:.6g}, "
        f"max density={float(cell_density.max()):.4g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
