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

## Loading-trajectory ablation

The three corrected runs use the same 0.01 m mesh, final eigenstrain, HLBFGS
minimizer, and `tol=1e-12`. This tight tolerance matters: looser tolerances can
stop on the flat plateau and turn the comparison into a solver-artifact study.

| Loading scheme | Cycles | Wall time | Field NRMSE vs one shot | Energy difference | Deflection span | Span difference | `integral(abs(H))` difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-shot full path | 1 | 831 s | 0.00% | 0.00% | 35.222 mm | 0.00% | 0.00% |
| uniform continuation | 10 | 3,461 s | 0.82% | 0.28% | 34.738 mm | 1.37% | 0.15% |
| stripwise trajectory | 30 | 7,590 s | 5.58% | 0.22% | 36.481 mm | 3.57% | <0.01% |

The actual final signed mean-curvature statistics are:

| Loading scheme | Area-weighted mean H (1/m) | Area-weighted RMS H (1/m) | min H (1/m) | max H (1/m) | integral H dA (m) | integral abs(H) dA (m) |
|---|---:|---:|---:|---:|---:|---:|
| one-shot full path | -0.9007 | 2.4927 | -12.5539 | +2.8780 | -0.069769 | 0.094290 |
| uniform continuation | -0.8926 | 2.4911 | -12.5646 | +2.8734 | -0.069141 | 0.094428 |
| stripwise trajectory | -0.8957 | 2.4904 | -12.5883 | +2.8955 | -0.069388 | 0.094293 |

Uniform numerical continuation reaches essentially the same endpoint as the
one-shot solve. Stripwise loading has a measurable path-history effect: it
changes the plane-removed full displacement field by 5.58% and the deflection
span by 3.57%. However, its final energy and integrated absolute mean curvature
remain within 0.22% and 0.01% of the one-shot result. The trajectory therefore
changes where bending is distributed more than it changes the aggregate
energetic or curvature response in this small-growth case.

This is evidence of a modest history-dependent endpoint, not evidence that the
30-cycle trajectory is a better optimizer. It costs about 9.1 times the
one-shot run at this mesh. A trajectory optimization claim would require a
specified target field or objective and a search over alternative strip
orders; this ablation only establishes that such a search can affect the final
field.

The discarded face-eventwise full-increment method remains outside the
ablation because it does not reproduce Putong's loading definition.

Reproduce the comparison with:

```bash
MPLCONFIGDIR=/tmp/psm-trajectory-ablation-mpl \
/gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  python/analyze_trajectory_ablation.py run/zigzag_production
```

The command writes `trajectory_ablation_summary.csv`,
`trajectory_ablation_history.png`, `trajectory_ablation_curvature.png`, and
`trajectory_ablation_curvature_3d.png` under `run/zigzag_production/`.
