# Square-sheet first verification: pre-execution review

**Verdict: APPROVED_WITH_CAVEATS — one fresh attempt-030 solve only.**

**Launch-blocking findings: none for the reviewed collection run.** The treatment-comparison analysis contains a substantive error that must be corrected before interpreting verification. It does not affect the solver or primary curvature gate.

## Scope, authority and identities

Authority: `docs/reports/dome_toolpath_investigation.md:384–424`, particularly the verification protocol, amendment 1 and attempt-030 selection; prior review: `docs/reports/square_bo_pre_execution_review.md`.

Reviewed identities:

| File | SHA256 prefix |
|---|---|
| `calibration/dome/square_attempt.py` | `cb3708762e3b` |
| `calibration/dome/run_square_verification.sbatch` | `18d8425a540d` |
| `scripts/run_dome_continuation.py` | `201830532596` |
| `test/python/test_dome_verification_ceiling.py` | `095ae2685385` |
| `calibration/dome/square_hatch.py` | `6557ef685f1c` |
| `calibration/analysis/dome_continuous_objective.py` | `dda4e7717a86` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942` |
| `calibration/configs/solver/default_adaptive_cutback.json` | `94252b22fd97` |
| Original `calibration/dome/analyze_square_verification.py` | `2560714857a3` |

The analysis script subsequently gained distance percentiles and optional history comparison during review. Those additions were inspected, but their updated hash was not supplied.

Frozen binary: `run/dome_square/frozen/shell`, SHA256  
`522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841`.

Attempt-030 canonical recipe digest:  
`a063f72a2a90656d8dec13b76f3b4764e468e40c894a394a33648a30fb310c32`.

Full supplied hashes are recorded in `docs/reports/square_verification_review_evidence.txt`.

## Execution findings

1. **Tolerance binding is correct.**  
   `scripts/run_dome_continuation.py:19–60` requires both endpoint and every accepted continuation substep to satisfy:
   `0 <= gradient_norm <= gradient_tolerance <= gradient_ceiling`, with positive pivots, accepted termination and full-load history.  
   `calibration/dome/square_attempt.py:87–88,140,179` overrides both solver gradient options to `1e-14` and passes that same ceiling to verification. The four synthetic tests cover strict acceptance and loose endpoint/substep rejection.

2. **Screening interpretation remains unchanged.**  
   The default ceiling remains `5e-14`. Supervisor-supplied reverse-edit checks reproduce the historical `b723c753…` and `f8aff744…` hashes by undoing only the ceiling changes. No objective/curvature change was identified.

3. **Recipe, isolation and safeguards are appropriate.**  
   `calibration/dome/run_square_verification.sbatch:16–24` checks the frozen binary, copies attempt-030’s request and refuses an existing verification directory. The worker validates the canonical recipe digest and refuses an existing solve directory. The command preserves physics, four ordered applications, free boundaries and a fresh rectangle start; continuation warm-starting is internal, not loading the screening endpoint.

   Resources and ceilings are consistent: eight CPUs/32 GB, 3600-second solver timeout, 600-second analysis alarm, outer 4500-second timeout with kill escalation, two-hour allocation. No retry/search loop exists. Actual termination behavior was not exercised.

4. **Do not equate Slurm completion with verification.**  
   Numerical failures intentionally exit zero. Require terminal worker status and accepted artifacts; outer timeout/OOM/interruption can leave `started` status. Report these as unresolved, preserve outputs and do not retry.

## Required before interpreting verification—not solve-launch gates

### Treatment comparison must use a common metric basis

**`calibration/dome/analyze_square_verification.py:38–64` currently compares raw `abar` traces. This is invalid across meshes.**

The native export contains full first fundamental forms in each triangle’s edge basis, not dimensionless material-coordinate perturbations (`src/simulations/Sim_Bilayer_Growth.cpp:1397–1398`). Halving edge length changes their baseline scale by approximately four, even with identical treatment.

Convert each face using:

\[
D_m=[u_2-u_1,\ u_0-u_2],\qquad
C=D_m^{-T}\bar aD_m^{-1}.
\]

Compare `C-I` in common material coordinates; include tensor-component differences rather than relying solely on trace/correlation. The exact edge convention is documented in `src/libshell/GrowthHelper.hpp:713–736,778–786`.

**Clearance:** a cheap two-mesh constant-material-metric fixture must recover identical normalized treatment. Screen-versus-itself cannot detect this defect. Corrected fields may still differ legitimately at stroke edges.

### Complete the frozen diagnostics

- Supply both histories to the new optional comparison. Aggregate hit-events-per-face and maxima are useful but do not replace spatial hit-count comparison. Compare exported `total_hit_count` at common material locations, with topology checked.
- Inspect whole-sheet curvature maps and unexaggerated geometry, including folds, localized bumps and estimator artifacts. Report curvature tails and invalid-area fractions.
- Retain all 32 screening attempts and both screening passes. Selection of attempt30 is consistent with the recorded objective: `0.129739…` versus attempt31’s `0.108202…`, not selection on verification.
- Use only the fine-mesh **15 mm / 0.1 m⁻¹ / 80% connected-area** gate for the primary decision. Sensitivities diagnose discrepancies; they must not become replacement thresholds.

## Shape-analysis caveats

`calibration/dome/analyze_square_verification.py:22–35,80–89` uses the correct proper-rotation Kabsch formula.

The ±79.92 mm sample grid lies inside the specified ±100 mm sheet, so ordinary out-of-mesh interpolation is not expected. Nevertheless, require all **1681** points to be finite rather than silently dropping missing points.

The equal-weight grid RMS and added percentiles are **sampled ROI approximations**, not exact reference-area integrals or whole-sheet maxima. `LinearNDInterpolator` also retriangulates independently of stored faces. For the protocol’s reference-area comparison, use documented material-area quadrature and preferably interpolation on the actual faces. Raw z-ranges are pose-dependent diagnostics, not aligned shape errors.

## Evidence and limitations

Performed: independent static inspection of authority, execution/analysis code, tests, campaign ledger, attempt-030 request/status/history/result, metric-export implementation and reconstructed historical snippets.

Not performed: commands, pytest, solver runs, scheduler checks or independent hash computation; this review environment provided no execution tool. Hash reconstruction and earlier self-check results were supervisor-supplied. The reported self-check reproduced 90.21875% and near-zero self-RMS; its output was not accessible here. Updated pytest/self-check evidence remained pending at review completion.

This is a **model-conditional combined mesh/tolerance refinement check**, not physical validation, a convergence-rate estimate or proof of a global equilibrium. Approval grants no execution permission, retry, BO restart, other-candidate evaluation or scope extension.