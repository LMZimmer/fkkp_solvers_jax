"""Solver configs: the JSON form of a solver's parameters.

A config is a dict keyed by solver parameter name plus the entry
``"solver"`` naming the solver class. Every solver holds one as
``solver.config`` (built at construction from the given parameters, the
class defaults filled in), ``Result.save`` writes it next to the results
and ``read_config`` loads it back, so ``solver_from_config(read_config(p))``
reproduces a run. Compared with the parameters the solver validates, a
config differs in three ways:

- Volumes (tissue maps, tensors, treatment maps) are NIfTI paths, absolute
  or relative to the config file; ``read_config`` makes them absolute and
  the solver loads them at construction. A volume that was given as an
  array is recorded as ``VOLUME_IN_MEMORY`` and cannot be reloaded. Solver
  specific entry formats (the resection cavity of ``StuppFKPPSolver``) are
  resolved by the solver class's ``_resolve_config_volume``.
- Derived values (``stopping_time`` of ``StuppFKPPSolver``, a defaulted
  ``volume_threshold``, a ``voxel_size_mm`` read from a NIfTI header) are
  not part of it; ``Result.derived`` reports them.
- JSON has no infinity: the strings "inf" / "-inf" stand for the floats.

Keys starting with '_' are comments and are dropped on load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .base import BaseFKPPSolver

# Config entry naming the solver class.
SOLVER_KEY: str = "solver"

# Config value of a volume that was given as an array instead of a path.
VOLUME_IN_MEMORY: str = "<in-memory>"

# Solver classes by name, filled by ``BaseFKPPSolver.__init_subclass__``.
_SOLVER_REGISTRY: dict[str, type[BaseFKPPSolver]] = {}


def register_solver(cls: type[BaseFKPPSolver]) -> None:
    """Register a solver class under its name for ``solver_class``."""
    name = cls.__name__
    if name in _SOLVER_REGISTRY and _SOLVER_REGISTRY[name] is not cls:
        raise ValueError(f"a solver named {name!r} is already registered.")
    _SOLVER_REGISTRY[name] = cls


def solver_class(name: str) -> type[BaseFKPPSolver]:
    """The solver class registered under name."""
    try:
        return _SOLVER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown solver {name!r}; known: {sorted(_SOLVER_REGISTRY)}."
        ) from None


def _resolve_solver(
    solver: str | type[BaseFKPPSolver] | None, config: Mapping[str, Any], where: str
) -> type[BaseFKPPSolver]:
    """The solver class of a config: the solver argument (a class or a name)
    if given, else the config's "solver" entry, which must agree."""
    named = config.get(SOLVER_KEY)
    if solver is None:
        if named is None:
            raise ValueError(f"{where}: no {SOLVER_KEY!r} entry names the solver class.")
        return solver_class(str(named))
    cls = solver_class(solver) if isinstance(solver, str) else solver
    if named is not None and named != cls.__name__:
        raise ValueError(f"{where}: names solver {named!r}, not {cls.__name__}.")
    return cls


def solver_from_config(config: Mapping[str, Any]) -> BaseFKPPSolver:
    """
    Construct the solver a config names.

    Args:
        config: A config with a "solver" entry (see ``read_config``).

    Returns:
        ``SolverClass(config)`` for the class the "solver" entry names.
    """
    return _resolve_solver(None, config, "config")(config)


def jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, paths, non-finite floats and nested
    containers to JSON-serializable values (inf becomes "inf")."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def _from_json_scalar(value: Any) -> Any:
    """Undo the "inf" / "-inf" spelling of ``jsonable`` (non-volume entries)."""
    if isinstance(value, str) and value in ("inf", "-inf"):
        return float(value)
    return value


def read_config(
    path: str | Path, solver: str | type[BaseFKPPSolver] | None = None
) -> dict[str, Any]:
    """
    Read a solver config from a JSON file without loading its volumes.

    The file holds a JSON object keyed by solver parameter name plus the
    "solver" entry (see the module docstring). Returned are its entries
    without the '_' comment keys, every key checked to be a parameter of
    the solver class, "inf" strings turned into floats and every volume
    entry resolved by the class (NIfTI paths made absolute, a relative path
    counting from the config's directory). The volumes stay paths, so the
    entries can be edited, completed or written back with ``write_config``
    before ``solver_from_config`` loads them; a missing volume file is
    reported when it is loaded.

    Args:
        path: Path of the JSON config.
        solver: The solver class (or its name) the config is for; default
            the class the file's "solver" entry names.

    Returns:
        The config entries, "solver" included.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, Mapping):
        raise ValueError(f"config {config_path}: must be a JSON object.")
    where = f"config {config_path}"
    config = {key: value for key, value in entries.items() if not key.startswith("_")}
    cls = _resolve_solver(solver, config, where)
    unknown = sorted(set(config) - cls.config_keys() - {SOLVER_KEY})
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}; every key must be a "
            f"{cls.__name__} parameter or {SOLVER_KEY!r}."
        )
    base_dir = config_path.resolve().parent
    resolved: dict[str, Any] = {SOLVER_KEY: cls.__name__}
    for key, value in config.items():
        if key == SOLVER_KEY:
            continue
        if key in cls._VOLUME_KEYS:
            resolved[key] = cls._resolve_config_volume(key, value, base_dir, where)
        else:
            resolved[key] = _from_json_scalar(value)
    return resolved


def resolve_config_path(value: Any, base_dir: Path, what: str) -> str:
    """The absolute path of a NIfTI named in a config (a relative path
    counts from base_dir, the config's directory)."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a NIfTI path, got {value!r}.")
    if value == VOLUME_IN_MEMORY:
        raise ValueError(
            f"{what} was an in-memory array when the config was written and "
            "cannot be reloaded; give a NIfTI path."
        )
    volume_path = Path(value)
    if not volume_path.is_absolute():
        volume_path = base_dir / volume_path
    return str(volume_path.resolve())


def write_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """
    Write a config as JSON (indented; numpy values and infinities converted
    by ``jsonable``). Volume paths are written as they are, so a config
    read with ``read_config`` writes absolute paths.

    Args:
        config: The config entries.
        path: Destination file; its directory must exist.

    Returns:
        The path written.
    """
    config_path = Path(path)
    config_path.write_text(json.dumps(jsonable(config), indent=2) + "\n", encoding="utf-8")
    return config_path
