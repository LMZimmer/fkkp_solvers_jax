#!/usr/bin/env python
"""Random parameter sweep of the Stupp-protocol forward model.

Samples --n-configs parameter sets uniformly at random (seeded) within the
ranges below, writes one manifest per configuration and runs each through
scripts/run_stupp_forward.py, so every configuration gets its own folder with
the manifest, run_config.json, final_cell_density.nii.gz and the default
overview montage (seed, before/after resection, radiotherapy block, end):

  <output-dir>/<sweep-name>/
    sweep.json                 sampled table, ranges, seed, base manifest, CLI
    sweep_results.csv          one row per configuration (parameters, seed
                               voxel, success, final mass, wall time), appended
                               as configurations finish
    configs/<config>.json      per-configuration manifest
    logs/<config>.log          stdout/stderr of the forward run
    <config>/                  output folder of run_stupp_forward.py

Swept parameters and default ranges (uniform):
  rho                      0.0089228 - 0.3449   1/day      (parameter_range.txt)
  white_matter_diffusivity 0.0071209 - 2.1329   mm^2/day   (parameter_range.txt)
  diffusivity_ratio        10.051    - 743.02              (parameter_range.txt, 10^log10 range)
  resection_time           30        - 200      days       (not in the file)
  chemo_kill_rate          0.05      - 2.0      1/day      (not in the file)
  chemo_decay_rate         1.0       - 20.0     1/day      (not in the file)
  rt_alpha                 0.02      - 0.3      1/Gy       (not in the file); rt_beta = 0.1 * rt_alpha
  seed voxel               uniformly among the cavity voxels (cavity_label of the
                           base manifest's tumor segmentation) that carry tissue
Ranges are overridable with --range NAME MIN MAX.

Everything else comes from the base manifest (--manifest). The radiotherapy
and chemotherapy sessions are shifted with the sampled resection time so the
treatment block keeps its offset after surgery (the base manifest's
time_after_resection keeps the horizon relative to the resection). The time
step is raised per configuration when the sampled diffusivity needs it:
steps_per_day = max(base, ceil(8 D / h^2 + 1)), h the grid spacing, i.e. the
solver's explicit-Euler diffusion bound.

Configurations run as subprocesses, one per GPU slot (--gpus, --jobs-per-gpu);
without --gpus a single worker inherits the environment (e.g.
JAX_PLATFORMS=cpu). Each subprocess writes only into its own folder.

Defaults: 10000 configurations, seed 1, GPUs 1,2,3,5, output under
/mnt/Drive4/lucas/SAILOR/sweep (about 10 h and 60 GB). Run from the
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

from fisher_kpp_jax.solvers import tissue_paths_from_manifest  # noqa: E402

DEFAULT_MANIFEST = _ROOT / "scripts" / "stupp_manifest_example.json"
FORWARD_SCRIPT = _ROOT / "scripts" / "run_stupp_forward.py"
T1C_RELATIVE = Path("skull_stripped") / "t1c_skullstripped.nii.gz"

# (min, max) of the uniformly sampled parameters; see the module docstring.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "rho": (0.0089228, 0.3449),
    "white_matter_diffusivity": (0.0071209, 2.1329),
    "diffusivity_ratio": (10 ** 1.0022, 10 ** 2.871),
    "resection_time": (30.0, 200.0),
    "chemo_kill_rate": (0.05, 2.0),
    "chemo_decay_rate": (1.0, 20.0),
    "rt_alpha": (0.02, 0.3),
}
BETA_OVER_ALPHA = 0.1

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
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="base manifest JSON")
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
        "--t1c",
        default=None,
        help=f"overview background NIfTI (default: <wm session dir>/{T1C_RELATIVE} if present)",
    )
    parser.add_argument("--no-plot", action="store_true", help="pass --no-plot to the forward runs")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the manifests and sweep.json, run nothing"
    )
    return parser.parse_args(argv)


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


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
        config["rt_beta"] = BETA_OVER_ALPHA * config["rt_alpha"]
        voxel = cavity_voxels[rng.integers(len(cavity_voxels))]
        config["seed_voxel"] = [int(v) for v in voxel]
        for axis, (v, n) in enumerate(zip(voxel, grid_shape)):
            # +0.5 so the solver's int(fraction * N) lands on the voxel.
            config[f"gaussian_seed_{'xyz'[axis]}_fraction"] = (int(v) + 0.5) / n
        configs.append(config)
    return configs


def config_manifest(
    base: dict[str, Any], config: dict[str, Any], grid_spacing_mm: float
) -> dict[str, Any]:
    """The per-configuration manifest: the base manifest with the sampled
    values substituted, sessions shifted with the resection time and the
    time step raised to the diffusion stability bound where needed."""
    manifest = json.loads(json.dumps(base))  # deep copy
    solver = manifest.setdefault("solver", {})
    for key in (
        "rho",
        "white_matter_diffusivity",
        "diffusivity_ratio",
        "gaussian_seed_x_fraction",
        "gaussian_seed_y_fraction",
        "gaussian_seed_z_fraction",
    ):
        solver[key] = config[key]
    resolution = float(solver.get("resolution_factor", 1.0))
    h = grid_spacing_mm / resolution
    base_steps = float(solver.get("steps_per_day", 12))
    solver.pop("dt", None)
    solver.pop("n_steps", None)
    solver["steps_per_day"] = int(
        max(base_steps, np.ceil(8.0 * config["white_matter_diffusivity"] / (h * h) + 1))
    )
    shift = config["resection_time"] - float(base["resection"]["time"])
    manifest["resection"]["time"] = config["resection_time"]
    if "chemotherapy" in manifest:
        section = manifest["chemotherapy"]
        section["times"] = [float(t) + shift for t in section["times"]]
        section["kill_rate"] = config["chemo_kill_rate"]
        section["decay_rate"] = config["chemo_decay_rate"]
    if "radiotherapy" in manifest:
        section = manifest["radiotherapy"]
        section["times"] = [float(t) + shift for t in section["times"]]
        section["alpha"] = config["rt_alpha"]
        section["beta"] = config["rt_beta"]
    manifest["_sweep"] = (
        f"{config['config']}: sampled values substituted by scripts/run_stupp_sweep.py; "
        f"sessions shifted by {shift:+g} days with the resection time; "
        f"steps_per_day {solver['steps_per_day']}."
    )
    return manifest


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

    manifest_path = Path(args.manifest).resolve()
    with open(manifest_path, encoding="utf-8") as handle:
        base = json.load(handle)
    for section in ("resection", "radiotherapy", "chemotherapy"):
        if section not in base:
            raise ValueError(f"the base manifest needs a {section!r} section for the sweep.")
    # Absolute volume paths so the per-config manifests work from configs/.
    base["resection"]["tumor_segmentation"] = str(
        resolve_manifest_path(base["resection"]["tumor_segmentation"], manifest_path)
    )
    base["radiotherapy"]["dose"] = str(resolve_manifest_path(base["radiotherapy"]["dose"], manifest_path))
    tissue = tissue_paths_from_manifest(manifest_path)
    if not tissue:
        raise ValueError("the base manifest needs a 'tissue' section (wm/gm pbmaps).")
    base["tissue"] = {key: str(path) for key, path in tissue.items()}

    wm_img = nib.load(str(tissue["wm"]))
    wm = np.asarray(wm_img.get_fdata(), dtype=np.float64)
    gm = np.asarray(nib.load(str(tissue["gm"])).get_fdata(), dtype=np.float64)
    zooms = tuple(float(v) for v in wm_img.header.get_zooms()[:3])
    segmentation = np.rint(
        nib.load(base["resection"]["tumor_segmentation"]).get_fdata()
    ).astype(np.int64)
    cavity = segmentation == int(base["resection"]["cavity_label"])
    # Seeds must lie in brain matter (the solver rejects seeds without tissue).
    cavity_voxels = np.argwhere(cavity & ((wm + gm) > 0))
    if not len(cavity_voxels):
        raise ValueError("no cavity voxel with tissue to seed in.")
    t1c = args.t1c
    if t1c is None:
        candidate = Path(tissue["wm"]).parent.parent / T1C_RELATIVE
        t1c = str(candidate) if candidate.is_file() else None

    rng = np.random.default_rng(args.seed)
    configs = sample_configs(rng, args.n_configs, ranges, cavity_voxels, wm.shape)
    for config in configs:
        manifest = config_manifest(base, config, min(zooms))
        config["steps_per_day"] = manifest["solver"]["steps_per_day"]
        with open(sweep_dir / "configs" / f"{config['config']}.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    sweep_info = {
        "sweep_name": sweep_name,
        "base_manifest": str(manifest_path),
        "cli_args": vars(args),
        "ranges": {name: list(bounds) for name, bounds in ranges.items()},
        "rt_beta": f"{BETA_OVER_ALPHA} * rt_alpha",
        "n_cavity_voxels": int(len(cavity_voxels)),
        "t1c": t1c,
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
        "rt_beta",
        "seed_voxel",
        "steps_per_day",
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
            "--manifest",
            str(sweep_dir / "configs" / f"{name}.json"),
            "--output-dir",
            str(sweep_dir),
            "--run-name",
            name,
        ]
        if t1c is not None:
            command += ["--t1c", t1c]
        if args.no_plot:
            command.append("--no-plot")
        gpu = slots.get()
        try:
            code, wall = run_forward(command, gpu, sweep_dir / "logs" / f"{name}.log")
        finally:
            slots.put(gpu)
        row: dict[str, Any] = {"exit_code": code, "wall_time_s": wall}
        run_config = sweep_dir / name / "run_config.json"
        if run_config.is_file():
            with open(run_config, encoding="utf-8") as handle:
                info = json.load(handle)
            row.update(
                success=info.get("success"),
                error=info.get("error"),
                final_time=info.get("final_time"),
                final_mass=info.get("final_stopping_quantity"),
                n_steps=info.get("n_steps"),
            )
        else:
            row.update(success=False, error="no run_config.json (see log)")
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
