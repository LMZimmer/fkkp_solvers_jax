# `elongate_tensor_along_main_axis_torch`: eigenvalue sum ("trace") is not preserved — drifts by +(2/3)·(factor−1)·λ_max per voxel

**Component:** `TumorGrowthToolkit/FK_DTI/tools.py`,
`elongate_tensor_along_main_axis_torch` (line 112 on current `main` of
m1balcerak/TumorGrowthToolkit).

**Versions used for the numbers below:** TumorGrowthToolkit 0.2 (function on
current `main` verified byte-identical to the copy tested), torch 2.13.0
(CPU), numpy 2.5.2, Python 3.14. The defect is version-independent — it is
in the adjustment algebra, not in any library behavior.

The function's comments say the two non-principal eigenvalues are adjusted "to
keep the sum constant", but the implemented adjustment sequence does not
achieve that: every voxel's tensor trace (mean diffusivity × 3) grows by
`(2/3)·(factor−1)·λ_max` instead of staying constant.

## Where the algebra goes wrong

Let the eigenvalues of a voxel's tensor be `a`, `b`, `m` with `m = λ_max`, and
let `diff = (factor − 1)·m` be the intended increase of the principal
eigenvalue. The code does, in order:

1. `original_sum = a + b + m`
2. `adjustment = diff / 2`; subtract it from the two non-max eigenvalues:
   `e_adjusted = (a − diff/2, b − diff/2, m)` — **the max is still unscaled
   here**, so `e_adjusted_sum = original_sum − diff`.
3. Residual correction: `final_adjustment = (original_sum − e_adjusted_sum)/3
   = diff/3`, added to the two non-max eigenvalues only:
   `e_final = (a − diff/6, b − diff/6, m)`.
   This step assumes the sum discrepancy it measures is a precision residual,
   but at this point the discrepancy is `−diff` *by construction*, because
   the max has not been scaled yet — so step 3 undoes two thirds of the
   compensation from step 2.
4. Scatter the scaled max: `e_final = (a − diff/6, b − diff/6, m + diff)`.

Final sum: `a + b + m − 2·(diff/6) + diff = original_sum + (2/3)·diff`, i.e.

```
trace_after − trace_before = (2/3)·(factor − 1)·λ_max        (per voxel)
```

## Minimal reproducible example (torch path only)

```python
import numpy as np
from TumorGrowthToolkit.FK_DTI import tools

A = np.array([[2.0, 0.5, 0.3],
              [0.5, 1.5, 0.2],
              [0.3, 0.2, 1.0]])          # SPD, trace = 4.5
factor = 1.5

out = tools.elongate_tensor_along_main_axis_torch(A[None, None, None], factor)[0, 0, 0]

lmax = np.linalg.eigvalsh(A.astype(np.float32)).max()   # 2.4016626
print(np.trace(out))                     # 5.3005533  (expected 4.5)
print(np.trace(out) - np.trace(A))       # 0.8005533
print((2 / 3) * (factor - 1) * lmax)     # 0.8005542  <- matches to f32 eps
```

Observed eigenvalues before/after (float32):

| | λ₁ | λ₂ | λ₃ (max) | sum |
|---|---|---|---|---|
| before | 0.9068538 | 1.1914836 | 2.4016626 | 4.5 |
| after | 0.7067154 | 0.9913450 | 3.6024930 | 5.3005533 |

Each non-max eigenvalue drops by only `diff/6 ≈ 0.2001` instead of the
`diff/2 ≈ 0.6004` needed to compensate the max's increase of
`diff ≈ 1.2008`.

## Practical consequence

Any run with `diffusionEllipsoidScaling != 1` silently inflates the mean
diffusivity of every voxel in proportion to that voxel's principal
eigenvalue, on top of the intended anisotropy change. Since the axial
diffusivity field is subsequently normalized by its mean, the net effect is a
spatially varying redistribution of diffusivity toward strongly anisotropic
voxels — not a pure "elongation at constant trace" as the comments state.
The scaling parameter therefore does not mean what the code comments (and
presumably downstream users) assume.

## Suggested fix (behavior change — flagging, not submitting silently)

Apply the compensation *after* (or equivalently, independently of) fixing the
scaled maximum, and drop the residual-correction step whose premise is wrong:

```python
e, v = torch.linalg.eigh(tensor_array)
idx = torch.argmax(e, dim=-1, keepdim=True)
max_e = torch.gather(e, -1, idx)
diff = max_e * (factor - 1)

mask = torch.ones_like(e, dtype=torch.bool)
mask.scatter_(-1, idx, 0)

# subtract diff/2 from each of the two non-max eigenvalues, set max to
# factor * max: the eigenvalue sum is preserved exactly (up to fp rounding)
e_final = torch.where(mask, e - diff / 2, e)
e_final.scatter_(-1, idx, max_e * factor)

tensor_prime = v @ torch.diag_embed(e_final) @ v.transpose(-2, -1)
```

With this version the example above returns trace 4.5 (to float32 rounding).

Two caveats worth an explicit decision by the maintainers:

- **Downstream results change.** Every published/stored result obtained with
  `diffusionEllipsoidScaling != 1` was produced with the current (drifting)
  behavior; fixing the function changes those simulations. A deprecation
  note or a behavior flag may be preferable to a silent fix.
- For large `factor`, subtracting `diff/2` can push a small non-max
  eigenvalue negative (the current code has the same exposure, just at 1/3
  the magnitude); downstream code clamps negative diagonals to 0, but that
  interaction may deserve its own look.
