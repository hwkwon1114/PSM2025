# HLBFGS coarse-resolution sweep

## Purpose

Establish the fastest mesh that still reproduces Putong's smooth global bending
before testing whether numerical load continuation or spatial trajectory order
changes the response.

The physical baseline is `putong_1step.json` with panel parameters
`lx=0.127`, `ly=0.1524`, total thickness 0.6 mm, HLBFGS basin tolerance
`1e-12`, and the archived 0.01 m mesh solution as the field reference.

## Minimizer

Tight HLBFGS enters the deep physical basin directly. Newton-family optimizers
were removed because they either select the wrong shallow basin or add no
material value after HLBFGS has already converged.

```text
-minimizer hlbfgs
-tol 1e-12
```

The exact Hessian remains available only for derivative verification and
final-state stability certification.

## Provisional acceptance criteria

Every coarse result is interpolated to the archived 0.01 m material grid. A
best-fit plane is removed from `u3`, and the equivalent up/down buckling sign is
selected before calculating full-field NRMSE. A mesh passes when:

- it reaches the deep basin (`u3` span at least 10 mm);
- full-field `u3` NRMSE is at most 15%;
- `u3` span and integrated absolute mean-curvature errors are at most 15%;
- energy error is at most 10%;
- mesh-scaled local roughness is at most twice the reference; and

## Result

| Resolution | Faces | Wall time | Field NRMSE | Energy error | Curvature error | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0.080 | 336 | 4.1 s | 9.7% | 10.2% | 11.6% | fail: energy |
| 0.060 | 640 | 12.8 s | 35.2% | 27.4% | 12.0% | fail |
| 0.050 | 960 | 13.4 s | 14.4% | 13.3% | 5.8% | fail: energy |
| 0.040 | 1,440 | 15.5 s | 8.5% | 13.2% | 0.7% | fail: energy |
| **0.035** | **1,904** | **20.2 s** | **5.1%** | **3.0%** | **1.8%** | **pass** |
| 0.030 | 2,720 | 33.8 s | 1.3% | 5.0% | 2.8% | pass |
| 0.025 | 3,840 | 55.2 s | 3.5% | 4.7% | 1.3% | pass |
| 0.020 | 6,000 | 104.1 s | 0.6% | 2.4% | 0.1% | pass |
| 0.015 | 10,560 | 232.2 s | 1.7% | 3.7% | 0.06% | pass |

Resolution 0.035 m is the fastest tested mesh satisfying all provisional
criteria. Resolution 0.03 m is the safer accuracy choice: its field error is
only 1.3% for an additional 14 seconds.

The production defaults are therefore `-res 0.03` and `-tol 1e-12`. Both remain
ordinary command-line options and can be overridden for convergence studies.

The non-monotonic errors at very coarse resolution arise from changes in which
face centroids lie inside the 10 mm toolpath band. They are why maximum
deflection alone is not an adequate mesh criterion.

## Reproduction

```bash
bash scripts/hlbfgs_resolution_sweep.sh

MPLCONFIGDIR=/tmp/psm-hlbfgs-mpl \
/gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  python/analyze_hlbfgs_resolution_sweep.py \
  run/hlbfgs_resolution_sweep \
  --reference run/zigzag_production/pt_1step
```

## Deferred trajectory ablation

After selecting the working resolution, compare exactly three loading schemes
with equal final eigenstrain and the same HLBFGS settings:

1. one-shot full path (`putong_1step.json`);
2. uniform numerical continuation (`putong_uniform10.json`);
3. stripwise trajectory continuation (`putong_traj30.json`).

The discarded face-eventwise full-increment method is not part of this
ablation because it does not reproduce Putong's loading definition.
