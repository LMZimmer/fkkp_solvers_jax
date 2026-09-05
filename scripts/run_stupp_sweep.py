#!/usr/bin/env python
"""Random parameter sweep of the Stupp-protocol forward model.

Samples --n-configs parameter sets uniformly at random (seeded) within the
ranges below, writes one config per configuration and runs each through
scripts/run_stupp_forward.py, so every configuration gets its own folder with
config.json, result.json, the initial and final cell density and the default
overview montage (seed, before/after resection, radiotherapy block, end):

  <output-dir>/<sweep-name>/
    sweep.json                 sampled table, ranges, seed, base config, CLI
    sweep_results.csv          one row per configuration (parameters, seed
                               voxel, success, final mass, wall time), appended
                               as configurations finish
    configs/<config>.json      per-configuration config
    logs/<config>.log          stdout/stderr of the forward run
    <config>/                  output folder of run_stupp_forward.py

Swept parameters and default ranges (uniform):
  rho                      0.0089228 - 0.3449   1/day      (parameter_range.txt)
  white_matter_diffusivity 0.0071209 - 2.1329   mm^2/day   (parameter_range.txt)
  diffusivity_ratio        10.051    - 743.02              (parameter_range.txt, 10^log10 range)
  resection_time           30        - 200      days       (not in the file)
  chemo_kill_rate          6.67e-4   - 2.67e-2  1/day per mg/m^2 (not in the file;
                           the former 0.05 - 2.0 per unit concentration / 75 mg/m^2)
  chemo_decay_rate         1.0       - 20.0     1/day      (not in the file)
  rt_alpha                 0.02      - 0.3      1/Gy       (not in the file)
  rt_alpha_beta_ratio      4.0       - 12.0     Gy         (the linear-quadratic
                           alpha/beta ratio, rt_beta = rt_alpha / it; clinical
                           GBM estimates, Pedicini et al. 2014)
  gaussian_seed_mass       100       - 500                 (not in the file; peak
                           density mass / (4 pi t)^(3/2) = 0.2 - 1.0 at the default
                           seed diffusion time t = 5, above the seed floor 0.1)
  gaussian_seed_scale      0.5       - 2.0                 (not in the file; widens
                           the seed by the factor, the peak stays, the seeded mass
                           becomes gaussian_seed_mass * scale^3)
  seed voxel               uniformly among the cavity voxels (the label of the
                           base config's resection_cavity segmentation) that
                           carry tissue
Ranges are overridable with --range NAME MIN MAX.

Everything else comes from the base config (--config, a config of
fisher_kpp_jax.read_config that must name the tissue maps and the cavity
segmentation): the volumes, the radiotherapy and chemotherapy session
times, the chemotherapy session doses (mg/m^2, copied through unchanged)
and the remaining solver parameters. The radiotherapy and chemotherapy
sessions are shifted with the sampled resection time so the treatment
block keeps its offset after surgery (the base config's
time_after_resection keeps the horizon relative to the resection). The
base config's time step is copied through: the solver raises it to its
explicit-Euler stability estimate itself when the sampled diffusivity needs
it (result.json holds the effective n_steps).

Configurations run as subprocesses, one per GPU slot (--gpus, --jobs-per-gpu);
without --gpus a single worker inherits the environment (e.g.
JAX_PLATFORMS=cpu). Each subprocess writes only into its own folder.

Defaults: 10000 configurations, seed 1, GPUs 1,2,3,5, output under
/mnt/Drive4/lucas/SAILOR/sweep (about 10 h and 60 GB, measured with the
earlier base config's day-200 horizon; the current config runs to day
360). Run from the
project root, detached for the long default run, e.g.:
  nohup python scripts/run_stupp_sweep.py --sweep-name sweep_sub15 > sweep_sub15.log 2>&1 &
  python scripts/run_stupp_sweep.py --output-dir runs/ --n-configs 5 --gpus 1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

import nibabel as nib
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fisher_kpp_jax import read_config, write_config  # noqa: E402

DEFAULT_CONFIG = _ROOT / "scripts" / "stupp_config_example.json"
FORWARD_SCRIPT = _ROOT / "scripts" / "run_stupp_forward.py"
T1C_RELATIVE = Path("skull_stripped") / "t1c_skullstripped.nii.gz"

# (min, max) of the uniformly sampled parameters; see the module docstring.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "rho": (0.0089228, 0.3449),
    "white_matter_diffusivity": (0.0071209, 2.1329),
    "diffusivity_ratio": (10 ** 1.0022, 10 ** 2.871),
    "resection_time": (30.0, 200.0),
    "chemo_kill_rate": (0.05 / 75, 2.0 / 75),  # 1/day per mg/m^2
    "chemo_decay_rate": (1.0, 20.0),
    "rt_alpha": (0.02, 0.3),
    "rt_alpha_beta_ratio": (4.0, 12.0),  # Gy
    "gaussian_seed_mass": (100.0, 500.0),
    "gaussian_seed_scale": (0.5, 2.0),
}

# Defaults sized for an unattended ~10 h run on 4 GPUs (~12 s per
# configuration per GPU on the 1 mm SAILOR grid, ~5.6 MB per configuration).
DEFAULT_OUTPUT_DIR = "/mnt/Drive4/lucas/SAILOR/sweep"
DEFAULT_N_CONFIGS = 10000
DEFAULT_SEED = 1
DEFAULT_GPUS = "1,2,3,5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="base config JSON")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="parent of the sweep directory"
    )
    parser.add_argument(
        "--sweep-name", default=None, help="sweep directory name (default: sweep_<UTC>_<pid>)"
    )
    parser.add_argument("--n-configs", type=int, default=DEFAULT_N_CONFIGS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed of the sampling")
    parser.add_argument(
        "--range",
        nargs=3,
        action="append",
        metavar=("NAME", "MIN", "MAX"),
        default=[],
        help=f"override a sampling range; NAME in {sorted(DEFAULT_RANGES)}",
    )
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPUS,
        help="comma-separated GPU ids to run on in parallel ('' = one worker, current env)",
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument(
        "--background-image",
        default=None,
        help=f"overview background NIfTI (default: <wm session dir>/{T1C_RELATIVE} if present)",
    )
    parser.add_argument("--no-plot", action="store_true", help="pass --no-plot to the forward runs")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the configs and sweep.json, run nothing"
    )
    return parser.parse_args(argv)


def sample_configs(
    rng: np.random.Generator,
    n_configs: int,
    ranges: dict[str, tuple[float, float]],
    cavity_voxels: np.ndarray,
    grid_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Draw n_configs parameter sets: each range uniformly, the seed voxel
    uniformly among cavity_voxels (rows of (i, j, k))."""
    configs = []
    for index in range(n_configs):
        config: dict[str, Any] = {"config": f"config_{index:04d}"}
        for name, (low, high) in ranges.items():
            config[name] = float(rng.uniform(low, high))
        voxel = cavity_voxels[rng.integers(len(cavity_voxels))]
        config["seed_voxel"] = [int(v) for v in voxel]
        for axis, (v, n) in enumerate(zip(voxel, grid_shape)):
            # +0.5 so the solver's int(fraction * N) lands on the voxel.
            config[f"gaussian_seed_{'xyz'[axis]}_fraction"] = (int(v) + 0.5) / n
        configs.append(config)
    return configs


def config_entries(base: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The per-configuration config: the base config with the sampled
    values substituted and the sessions shifted with the resection time."""
    entries = dict(base)
    for key in (
        "rho",
        "white_matter_diffusivity",
        "diffusivity_ratio",
        "gaussian_seed_x_fraction",
        "gaussian_seed_y_fraction",
        "gaussian_seed_z_fraction",
        "resection_time",
        "chemo_kill_rate",
        "chemo_decay_rate",
        "rt_alpha",
        "rt_alpha_beta_ratio",
        "gaussian_seed_mass",
        "gaussian_seed_scale",
    ):
        entries[key] = config[key]
    shift = config["resection_time"] - float(base["resection_time"])
    entries["chemo_times"] = [float(t) + shift for t in base["chemo_times"]]
    # chemo_doses stay as in the base config (one per shifted time).
    entries["rt_times"] = [float(t) + shift for t in base["rt_times"]]
    entries["_sweep"] = (
        f"{config['config']}: sampled values substituted by scripts/run_stupp_sweep.py; "
        f"sessions shifted by {shift:+g} days with the resection time."
    )
    return entries


def run_forward(
    command: list[str], gpu: str | None, log_path: Path
) -> tuple[int, float]:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return code, time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ranges = dict(DEFAULT_RANGES)
    for name, low, high in args.range:
        if name not in ranges:
            raise ValueError(f"--range: unknown parameter {name!r}; expected one of {sorted(ranges)}.")
        ranges[name] = (float(low), float(high))
    sweep_name = args.sweep_name or (
        f"sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    sweep_dir = Path(args.output_dir) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=False)
    (sweep_dir / "configs").mkdir()
    (sweep_dir / "logs").mkdir()

    config_path = Path(args.config).resolve()
    # read_config makes the volume paths absolute, so the per-config
    # configs work from configs/.
    base = read_config(config_path)
    missing = [
        key
        for key in ("white_matter_pbmap", "gray_matter_pbmap", "resection_cavity", "resection_time", "chemo_times", "rt_times")
        if base.get(key) is None
    ]
    if missing:
        raise ValueError(f"the base config lacks {missing}, which the sweep needs.")

    wm = np.asarray(nib.load(base["white_matter_pbmap"]).get_fdata(), dtype=np.float64)
    gm = np.asarray(nib.load(base["gray_matter_pbmap"]).get_fdata(), dtype=np.float64)
    segmentation = np.rint(
        nib.load(base["resection_cavity"]["segmentation"]).get_fdata()
    ).astype(np.int64)
    cavity = segmentation == base["resection_cavity"]["label"]
    # Seeds must lie in brain matter (the solver rejects seeds without tissue).
    cavity_voxels = np.argwhere(cavity & ((wm + gm) > 0))
    if not len(cavity_voxels):
        raise ValueError("no cavity voxel with tissue to seed in.")
    background_image = args.background_image
    if background_image is None:
        candidate = Path(base["white_matter_pbmap"]).parent.parent / T1C_RELATIVE
        background_image = str(candidate) if candidate.is_file() else None

    rng = np.random.default_rng(args.seed)
    configs = sample_configs(rng, args.n_configs, ranges, cavity_voxels, wm.shape)
    for config in configs:
        write_config(config_entries(base, config), sweep_dir / "configs" / f"{config['config']}.json")
    sweep_info = {
        "sweep_name": sweep_name,
        "base_config": str(config_path),
        "cli_args": vars(args),
        "ranges": {name: list(bounds) for name, bounds in ranges.items()},
        "n_cavity_voxels": int(len(cavity_voxels)),
        "background_image": background_image,
        "configs": configs,
    }
    (sweep_dir / "sweep.json").write_text(json.dumps(sweep_info, indent=2) + "\n")
    print(f"sweep directory: {sweep_dir}")
    print(f"{len(configs)} configurations sampled (seed {args.seed}), {len(cavity_voxels)} cavity voxels")
    if args.dry_run:
        return 0

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots: Queue[str | None] = Queue()
    for gpu in gpus or [None]:
        for _ in range(args.jobs_per_gpu if gpus else 1):
            slots.put(gpu)
    lock = threading.Lock()
    results: dict[str, dict[str, Any]] = {}
    columns = [
        "config",
        *ranges,
        "seed_voxel",
        "success",
        "final_time",
        "final_mass",
        "n_steps",
        "wall_time_s",
        "exit_code",
        "error",
    ]
    # Rows are appended as configurations finish, so a killed sweep keeps
    # its partial results (sweep_results.csv is in completion order).
    csv_handle = open(sweep_dir / "sweep_results.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=columns)
    writer.writeheader()

    def worker(config: dict[str, Any]) -> None:
        name = config["config"]
        command = [
            sys.executable,
            str(FORWARD_SCRIPT),
            "--config",
            str(sweep_dir / "configs" / f"{name}.json"),
            "--output-dir",
            str(sweep_dir),
            "--run-name",
            name,
        ]
        if background_image is not None:
            command += ["--background-image", background_image]
        if args.no_plot:
            command.append("--no-plot")
        gpu = slots.get()
        try:
            code, wall = run_forward(command, gpu, sweep_dir / "logs" / f"{name}.log")
        finally:
            slots.put(gpu)
        row: dict[str, Any] = {"exit_code": code, "wall_time_s": wall}
        record_path = sweep_dir / name / "result.json"
        if record_path.is_file():
            with open(record_path, encoding="utf-8") as handle:
                record = json.load(handle)
            row.update(
                success=record.get("success"),
                error=record.get("error"),
                final_time=record.get("final_time"),
                final_mass=record.get("final_stopping_quantity"),
                n_steps=record.get("n_steps"),
            )
        else:
            row.update(success=False, error="no result.json (see log)")
        with lock:
            results[name] = row
            full = {**config, **row, "seed_voxel": " ".join(str(v) for v in config["seed_voxel"])}
            writer.writerow({key: full.get(key, "") for key in columns})
            csv_handle.flush()
            status = "ok" if row.get("success") else "FAILED"
            print(f"  {name}: {status} in {wall:.0f} s ({len(results)}/{len(configs)})", flush=True)

    n_workers = max(1, len(gpus) * args.jobs_per_gpu) if gpus else 1
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(worker, configs))
    total = time.perf_counter() - start
    csv_handle.close()
    failed = [name for name, row in results.items() if not row.get("success")]
    print(f"done: {len(configs) - len(failed)}/{len(configs)} succeeded in {total / 60:.1f} min")
    if failed:
        print("failed: " + ", ".join(sorted(failed)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
