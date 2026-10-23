# Independent review — continuation-assisted sequence matrix

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified for the reviewed slice and submission contract.**

Scope: seven fresh primary runs, reuse of accepted `ab_block_final_only_continuation_v1`, then six preselected `ab_block` sensitivity runs **only after all eight primary endpoints pass acceptance**. This review does not authorize execution, retries, extensions, unrelated dome changes or broader experiments.

The quota-failed prior review supplies no approval. The earlier diagnostic review supports only its original scope.

## Authority and identities

Read in full:

- `.pi/goals/active_goal_2026101910270493_mvfeqwh7-k4y0nd.md`
- `docs/reports/sequence_ablation_v2_spec.md`, including **Current revised investigation — continuation-assisted mixed comparisons**

The revised investigation supersedes the historical exact20-solves-versus-one requirement. No historical cumulative budget was applied.

Identity scope is the complete packet:

`run/sequence_order/checks/continuation_design_v2.json`

Principal recorded SHA256 identities:

| Artifact | SHA256 |
|---|---|
| `scripts/run_sequence_order_continuation.py` | `5c6c5cbbe107f3d3cdb2d711082c255794dfcd1e5326620821c4053af57c8750` |
| `scripts/verify_sequence_continuation.py` | `57f6e7d8b843d7e8e4d0b5f6754050dec1f614ff7076443e7b058669a3efbb15` |
| `scripts/run_sequence_continuation.sbatch` | `6267c6c24382f408c4bdfa1f0e9d31138f4ed47aeee438d0a3f92f766fe3d508` |
| `calibration/analysis/analyze_sequence_order.py` | `0750f98d6f63b053d984bdf6962341038399a77f829b571ddcf3e2d840de57ed` |
| `calibration/tests/test_sequence_order_continuation.py` | `ef907b4e66b64e33c32c8067e5bd3caf8e099814df253d7f8165de81d24465d0` |
| `calibration/tests/test_sequence_continuation_verifier.py` | `560b4f680f241b38f9357e3a0e14709ae609dde58524ba14c2c0044d894ddea9` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |

The packet’s three connected native-source identities agree with `run/dome/checks/repair028_identity.sha256`. Relevant source was inspected directly. These are **recorded identities, not independently recomputed hashes**.

The reused baseline retains sequence SHA256 `3e1888be9be10de1d2aaab2612d1ae4c6b7a94b3db54eaaab79e9e3576c9d5a0` and its historical runner/verifier identities; it is not relabeled as having used the revised implementation. Existing checkout changes are not attributable to this review. No source edits were made.

## Checks performed

### Scientific design and matching

The four schedules contain exactly ten A and ten B applications. `scripts/run_sequence_order.py` preserves the first two frozen recipes, profiles, material-frame placement, multiplicative accumulation and face-shared Voce history. It excludes the third baseline toolpath.

The comparison is appropriate for **model-conditional loading-path dependence**, not physical temporal memory. A and B differ in dimensions, location and profile as well as orientation. Their accepted direct controls remain supporting evidence with a different solver cadence.

The analysis distinguishes:

- Changed accumulated targets/history: constitutive or implementation differences, not identical-objective branch evidence.
- Different accepted shapes with equal objectives: numerical loading-path dependence, subject to sensitivity.
- Failed or partial endpoints: unresolved, never scored as equilibria.

The preselected repeat/tight/fine checks and original resolution criterion remain fixed: cadence RMS must exceed both five times the larger cadence-specific repeat discrepancy and `1e-4` of the panel diagonal, with persistence under tightening/refinement. Other orders remain unrefined.

### Native continuation, history and rollback

Inspected `src/simulations/Sim_Bilayer_Growth.cpp:2505–2846`:

- Sequential stage \(k\) interpolates from the preceding accepted target.
- Combined loading accumulates all ordered applications first and interpolates from the initial stress-free target.
- For either layer,
  \[
  \bar A(\lambda)=(1-\lambda)\bar A_{\mathrm{start}}+\lambda\bar A_{\mathrm{end}}.
  \]
  This is a metric homotopy, not fractional replay of growth/hardening.
- Ordered growth and hit-count updates occur outside the trial loop.
- Every trial starts from the last accepted vertex/director DOFs.
- Returned rejection restores those DOFs and accepted-load metrics before reducing the step.
- **Exact endpoint target arrays are installed before the lambda1 solve and acceptance check**, not substituted afterward.
- Completed stages retain accumulated history and geometry.

`src/libshell/GrowthHelper.hpp:796–826` implements the multiplicative congruence update and positive-definiteness check. Convex interpolation preserves SPD between SPD endpoints. No strain multiplication shortcut was introduced.

`src/libshell/NativeHybridShellSolver.hpp:313–362` creates a fresh solver session per invocation and independently checks the selected endpoint. Acceptance requires `gradient_target`, finite energy/gradient, the requested gradient gate and `positive_pivots`. Fixed2000 handoff and repaired linear solves remain enabled; already-converged substeps may stop earlier. Production arithmetic and tolerances are unchanged.

### Stage-wise acceptance

`scripts/verify_sequence_continuation.py:7–58` checks:

- Completed application count and expected physical-stage sequence.
- Matching convergence/proof stream lengths.
- Trial progression from the last **accepted** load.
- Qualification of every accepted substep.
- Exact lambda1 completion and matching endpoint proof for every stage.
- Load reset between stages.
- Application history, summary IDs, intermediate-file presence and final proof consistency.

The verifier’s `len(toolpaths)` counting assumes the reviewed generator’s enabled, single-repeat entries. It is suitable for this fixed matrix, not a general repeated-toolpath verifier.

### Analysis and baseline reuse

`calibration/analysis/analyze_sequence_order.py:60–173` selects the continuation verifier from recorded adaptive-loading options and explicitly reuses only the designated `ab_block` combined baseline.

Cadence comparisons require matching sequence/binary identities, all options except solve mode, initial-VTP digest, complete reference/material fields and exact final targets/history. Native JSON supplies endpoint coordinates. Proper determinant+1, reference-area-weighted alignment preserves material correspondence; energy differences are withheld for unequal objectives.

The reused baseline’s inspected receipt, convergence stream, eleven proofs and result agree:

- Twenty physical applications; one continuation block.
- Eleven accepted optimizer invocations, zero rejected.
- Exact terminal lambda1.
- Terminal gradient `5.902563126326728e-17`.
- Elapsed `1974.195s`.
- Recorded target inspection reports exact layer-target composition and hit-count agreement with the disjoint controls.

Its shorter historical outer timeout does not invalidate an accepted endpoint, but prevents an equal-runtime comparison.

### Resources and stopping

Reviewed submission contract, explicitly confirmed by the parent and incorporated into the current protocol:

| Case | Subprocess ceiling | Explicit Slurm `--time` |
|---|---:|---:|
| Sequential base/repeat/tight | 20h | `21:00:00` |
| Combined base/repeat/tight | 2h | `02:10:00` |
| Sequential fine | 30h | `31:00:00` |
| Combined fine | 3h | `03:10:00` |

All use 8CPU/32GB. Native1800s is a cooperative **per-substep** safeguard, not the case ceiling.

The runner requires a compute allocation, limits thread pools, uses exclusive output creation and records subprocess failure without automatic retry. The parent owns max-two concurrency, stopping new dispatch on failure and primary-before-sensitivity ordering. These are procedural safeguards, not runtime enforcement demonstrated by this review.

## Evidence inspected and limits

- `run/sequence_order/checks/continuation_tests_v2.log`: six tests passed.
- `run/sequence_order/checks/continuation_analysis_tests_v2.log`: fourteen tests reported, thirteen passed and one retained-direct-fixture test skipped.
- Test source covers launch wiring, exclusive output, timeout rejection, stage resets, partial-load/wrong-stage rejection, rollback-log consistency and unqualified intermediate proofs.
- Job4519380 retained fixtures: inspected CSVs show four sequential stages and44 accepted substeps versus11 combined substeps.
- `run/sequence_order/checks/continuation_fixture_analysis_v2/metrics.json`: exact layer targets/history and matching reference/objective checks; runner/verifier/analysis identities agree with the packet.
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: recorded29 passing tests, reused only for the unchanged native identity.

The coarse fixtures exercise stage transitions and history accumulation, but **not native rejected-trial recovery**: rollback evidence here is source inspection plus synthetic verifier tests. Their geometric-curvature valid-area fraction is zero, so they are not curvature-validation or scientific-resolution evidence.

No commands, simulations, builds, installations, fresh tests, hash recomputation, live scheduler checks, library audit or delegation were performed. This was a read-only source/record review.

## Launch-blocking findings

**None under the reviewed identities and explicit submission contract.**

## Nonblocking caveats

1. **Batch time overrides are essential.**  
   `scripts/run_sequence_continuation.sbatch:8` defaults to21h. Parent-confirmed overrides resolve the mismatch; particularly, fine sequential must receive31h. A bare default submission is outside the reviewed fine-run contract.

2. **Preserve and recheck identities.**  
   The runner records current hashes but does not enforce the entire review packet. Parent preflight must compare reviewed identities, including the immutable reused baseline. Consequential changes require reassessment.

3. **Report actual work and retained-output limits.**  
   `equilibrium_solves` counts completed physical-stage blocks, not optimizer invocations. Aggregate accepted/rejected trials and iterations from retained streams. The tiny eleventh increment is a floating-point terminal correction. Intermediate physical-stage VTPs are retained; every accepted substep’s geometry is not exported.

4. **Reporting remains separate from launch acceptance.**  
   The analyzer produces descriptive contrasts, not an automatic sensitivity verdict. Preserve direct controls and all historical failures alongside the continuation report; their absence from the continuation-root matrix must not imply they were never run. Final figures and interpretation were not reviewed here.

5. **No demonstrated resumability or completion guarantee.**  
   Diagnostic exports are not restart checkpoints. A killed wrapper can leave `status=started`, which is not acceptance. The successful combined run does not establish convergence, memory demand or duration for other cases.

## Claim and approval boundaries

Agreement limits the hypothesis for these recipes and numerical settings. Sensitivity-supported disagreement can establish a case-specific failure of combined-target loading to reproduce sequential continuation endpoints. Failed acceptance or unstable sensitivity leaves the comparison unresolved.

Neither outcome establishes physical temporal memory, global optimality, general failure of lumping or robustness of unrefined orders. This approval covers only the reviewed slice and grants no additional execution authority, resource extension or sealed-test release.