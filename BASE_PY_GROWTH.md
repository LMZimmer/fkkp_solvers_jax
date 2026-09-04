# Why base.py grew in the config refactor

Commit 6e046e1 ("Refactor config logic", 2026-09-04) took `fisher_kpp_jax/base.py`
from 830 to 1314 lines. Nearly all of it follows from the plan's design
(CONFIG_REFACTOR_PLAN.md): steps 1, 2, 3, 5 and 6 each name the base solver as
the place where the logic lives, so `base.py` absorbed loading, saving, config
building and time-step resolution at once.

| Where | Lines | What |
|---|---|---|
| Volume loading (`_load_volumes`, `_load_volume_entry`, `_resolve_voxel_size`, `_resolve_config_volume`, `_config_volume_entry`) | ~105 | step 1, moved here from the Stupp manifest loader and generalized |
| `Result.save` plus new `Result` fields and their docstring | ~110 | steps 2 and 5 |
| `n_steps_from_dt`, `_resolve_time_stepping`, `resolve_time_stepping`, the at-most-one-of validation | ~85 | step 3, moved here from `solvers.py` |
| `solve`, `save`, `_derived_parameters`, `_build_config`, `config_keys`, `get_default_config`, `__init__`, `__init_subclass__` | ~90 | steps 4 to 6 |
| Class docstring, constants, imports, non-finite check | ~60 | |

Two things inflate it beyond the logic itself. About 125 of the new lines inside
functions are docstrings, since each hook and the save layout are documented in
the repository's existing style. And `solvers.py` shrank by only 60 net lines
because its manifest loader was replaced rather than removed, so the moved code
shows up as growth here.

## Possible split

The natural split is to move `Result` and its `save` into their own module, and
the volume loading into `config.py` next to the path resolution it mirrors. That
takes roughly 250 lines out of `base.py` without changing behavior.
