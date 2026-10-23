i# Sequential Toolpath Ablation v2 — Corrected Results

## Scope and provenance

This report implements `sequence_ablation_v2_spec.md` and supersedes the numerical conclusions in `sequence_ablation_report.md`.

All solver cases ran serially with `OMP_NUM_THREADS=8`. Inputs, commands, logs, manifests, convergence histories, VTP outputs, executable checksums, source revision/status, and derived tables are stored under `run/sequence_ablation_v2/`. The completed dataset contains the 28 runs in the corrected matrix.

Analysis artifacts:

- `run/sequence_ablation_v2/analysis.json`
- `run/sequence_ablation_v2/case_metrics.csv`
- `run/sequence_ablation_v2/comparisons.csv`
- `run/sequence_ablation_v2/matched_comparisons.csv`

All 28 final states terminated with a recorded converged HLBFGS outcome. The existing fixed-iteration Hessian diagnostic labelled all endpoints `MINIMUM`, but it does not deflate rigid modes or report an eigenpair residual/error bound. Its positive `lambda_min` values are therefore diagnostic, not formal minimum certificates; stability-dependent conclusions remain unproven.

## Validation and numerical floor

The analysis implementation passed two analytic checks:

- proper Kabsch alignment recovered an injected rigid transform with RMS error `2.47e-15 m` and maximum error `3.83e-15 m`;
- a paraboloid with principal curvatures `1` and `3 m^-1` was recovered as `1.000000000000001` and `3.000000000000004 m^-1`.

Five nominally identical baseline runs produced:

- maximum all-pairs aligned RMS: `5.195e-5 m` (`51.95 µm`);
- resulting `5×` geometry threshold: `2.598e-4 m` (`259.75 µm`);
- aligned z-span range: `2.6304–2.7159 mm`;
- energy range: `1.75435e-13–1.75631e-13`.

The panel-relative threshold is approximately `3.97e-5 m` (`39.7 µm`), using `1e-4` of the final panel diagonal. A matched geometry difference must exceed both this threshold and `259.75 µm`; the repeatability threshold is controlling.

## Mesh and tolerance convergence

### Mesh

At `tol=1e-12`, final z-span was:

| `res` | z-span | energy | quadratic principal curvatures (`m^-1`) |
| 0.060 | 2.7096 mm | `1.9487e-13` | `[-0.1918, -0.0612]` |
| 0.030 | 2.7062 mm | `1.7545e-13` | `[-0.2107, -0.0578]` |
| 0.015 | 2.5428 mm | `1.7540e-13` | `[-0.1908, -0.0685]` |

The two finest meshes do not show converged shape or curvature. The apparent agreement between `res=0.060` and `0.030` does not continue at `0.015`. Consequently, none of the optimizer-path or constitutive geometry differences below can be called mesh-certified physical effects.

### Tolerance

At `res=0.03`:

| tolerance | aligned RMS from `1e-12` | z-span | final gradient norm |
|---:|---:|---:|---:|
| `1e-10` | 0.7840 mm | 0.0579 mm | `4.37e-10` |
| `1e-12` | reference | 2.7062 mm | `4.39e-12` |
| `1e-14` | 0.0277 mm | 2.7619 mm | `4.20e-14` |

`1e-10` selects a materially different, nearly flat endpoint and is not adequate for this continuation. The Hessian diagnostic is positive there, but cannot formally certify a separate basin. The `1e-12` versus `1e-14` aligned RMS is below both the repeatability criterion and the panel-relative threshold. `1e-12` is adequate relative to the measured run-to-run floor, although the aligned z-span remains visibly sensitive at the tens-of-micrometres level.

## Warm-start sensitivity

Matched warm/cold comparisons retained identical final target metrics.

| growth per path | aligned RMS | relative RMS | substantial? |
|---:|---:|---:|---|
| `2e-5` | 27.48 µm | `6.93e-5` | No |
| `1e-4` | 22.57 µm | `5.70e-5` | No |

Both differences are below the `5×` repeatability threshold and below `1e-4` of the panel diagonal. The earlier claim that warm starting materially changed the final state is not supported. Warm starting changes runtime and iteration history, but the observed equilibrium differences are numerical-floor effects in these cases.

## Metric-update schemes

The three implemented recurrences pass exact repeated-isotropic closed-form unit tests:

- multiplicative: `(1+e)^(2n)`;
- recursive linearized: `(1+2e)^n`;
- reference-additive linearized: `1+2ne`.

Matched geometric differences from the multiplicative recurrence were:

| growth | comparison | target-metric RMS | aligned RMS | classification |
|---:|---|---:|---:|---|
| `2e-5` | recursive | `1.12e-13` | 3.75 µm | negligible |
| `2e-5` | reference-additive | `3.84e-12` | 6.68 µm | negligible |
| `1e-4` | recursive | `2.79e-12` | 4.04 µm | negligible |
| `1e-4` | reference-additive | `9.60e-11` | 6.77 µm | negligible |
| `5e-4` | recursive | `7.07e-11` | 4.07 µm | negligible |
| `5e-4` | reference-additive | `2.41e-9` | 127.75 µm | below repeatability criterion |

The `5e-4` reference-additive scheme has a distinct target metric by construction and an energy of `1.0160e-10`, versus `1.0331e-10` for multiplicative composition. Its `127.75 µm` geometry difference exceeds the panel-relative threshold but not the `5×` repeatability threshold, so it does not meet the predefined substantial-difference rule. It remains a constitutive target-metric difference whose geometric consequence is not resolved above numerical and mesh uncertainty.

For the corrected alternating `0°/45°`, `ortho=0.8` order test, reversing order changed the target metric by only `2.00e-15` RMS and geometry by `10.47 µm`, below the numerical threshold. In this spatial coverage pattern, order noncommutativity is negligible. The tempting `0°/90°` pair was excluded because orthogonal diagonal growth tensors commute.

## Exact stroke segmentation

All six schedules used actual zero-based `active_strips` selections. Their final target metrics agree with the complete-path schedule to numerical precision: maximum absolute error was `1.36e-20` and was exactly zero for three schedules.

Relative to complete-path ×3:

| schedule | aligned RMS | target-metric maximum | substantial? |
|---|---:|---:|---|
| path-major, solve each stroke | 16.69 µm | `0` | No |
| path-major, solve every 3 strokes | 22.68 µm | `0` | No |
| stroke-major | 21.46 µm | `1.36e-20` | No |
| reversed stroke-major | 22.78 µm | `1.36e-20` | No |
| one final solve | 23.08 µm | `0` | No |

No segmentation result exceeds either required geometry threshold. With load equivalence now established face by face, the observed schedule dependence is numerical noise. Grouping and order do not provide evidence of different equilibria for the three-path low-strain case.

## Corrected conclusions

1. **Repeatability is not zero.** The maximum all-pairs aligned RMS floor reaches `51.95 µm`; claims based on smaller differences are unsupported.
2. **Mesh convergence is the dominant unresolved error.** The finest tested mesh changes z-span by about 6% and shifts the fitted principal-curvature pair relative to `res=0.03`.
3. **Warm versus cold start is negligible** at both tested growth levels under the predefined thresholds.
4. **Recursive linearization is negligible through `5e-4` per path** relative to multiplicative composition in observed geometry.
5. **Reference-additive linearization changes the target metric most at `5e-4`**, but its observed geometry remains below the predefined repeatability criterion and is not mesh-certified.
6. **Exact segmentation, solve cadence, and stroke order are negligible** for the tested low-strain, three-path loading once the final target metric is verified identical.
7. **Loose tolerance can select a materially different endpoint.** `tol=1e-10` is inadequate; the current Hessian diagnostic is positive there but is not a formal certificate.
8. **No branch-changing physical effect is established.** Every candidate either falls below repeatability, lacks mesh persistence, or is explicitly a different constitutive approximation.

## Replacement equilibrium methodology

The ablation above remains a record of the original solver and is not retroactively
reinterpreted as a converged path study. The replacement implementation separates
three levels that were previously conflated:

1. A constitutive path maps one physical toolpath to an exact endpoint
   $(\bar a_\mathrm{top},\bar a_\mathrm{bot})$. Hardening history advances once at
   that endpoint.
2. Numerical continuation interpolates between the path-start and path-end
   positive-definite metrics with a scalar load amplitude $\lambda\in[0,1]$.
   Rejected corrector attempts restore geometry, edge directors, target metrics,
   and pass counts before reducing $\Delta\lambda$. Subdivision therefore cannot
   change the constitutive endpoint.
3. The inner corrector solves static equilibrium at fixed $\lambda$. HLBFGS and
   trust-region Newton-CG are alternative correctors, not alternative physical
   evolution laws. An endpoint is accepted only when the reported residual is
   finite and no larger than the explicit `-equilibrium_grad_tol`; an optimizer
   exit code alone is insufficient.

Free-shell tangent diagnostics now project boundary-compatible rigid motions and
report the smallest Ritz value together with its eigenpair residual. The
classification is `positive` only when the residual-certified lower bound
$\lambda_\mathrm{Ritz}-\|Hv-\lambda_\mathrm{Ritz}v\|$ is positive, `negative`
only when the corresponding upper bound is negative, and `indeterminate`
otherwise. This diagnoses the accepted state; it does not identify a physical
branch or trace an unstable branch.

The deterministic follow-up uses 20 distinct longitudinal, transverse, oblique,
offset, and layer-differential paths. It jointly refines mesh size and continuation
step, compares HLBFGS with the exact-Hessian trust-region corrector, and evaluates
all three metric-update laws on a common discretization. The serial Slurm runner is
`scripts/equilibrium_continuation_study.sbatch`; machine-readable manifests and
per-attempt diagnostics are written below
`run/equilibrium_continuation_study/`.

### Evidence gates

- **Constitutive discrimination:** not established until differences between
  metric-update laws persist under joint mesh/load-step refinement and exceed the
  measured repeatability floor. Solver convergence alone cannot discriminate a
  constitutive law.
- **Surrogate readiness:** not established until all 20 paths pass the residual
  gate, final observables are jointly converged, and branch/stability labels are
  derived from outputs rather than supplied as model inputs. Until then, training
  a low-fidelity surrogate would encode discretization and solver-selection error.
- **Instability evolution:** a negative certified tangent direction marks loss of
  local stability. Following an unstable equilibrium requires an augmented
  residual formulation with arc-length continuation; modeling a measured snap
  requires dynamics and dissipation. Neither behavior is represented by repeatedly
  minimizing static energy.
