# Independent delta review — attempt028 10% ramp

## Verdict: APPROVED_WITH_CAVEATS

**No unresolved scientific or operational launch blockers for one bounded diagnostic attempt.**

Scope: `scripts/test_dome_attempt028_ramp.py` and `scripts/run_dome_attempt028_ramp.sbatch`, using the unchanged repaired binary and archived attempt028 input. Limits remain **1800s native per solve, 2400s total subprocess timeout, 8 CPUs / 32GB / 45 minutes**.

Approval does not authorize execution, another attempt, resource extension, rebuilding, BO restart or changed acceptance thresholds.

## Authority and identities

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`
- `docs/reports/dome_toolpath_investigation.md`, including Current status, Adaptive test result and frozen scientific criteria
- `docs/reports/dome_adaptive028_review.md`
- `docs/reports/dome_linear_repair_review.md`

The current researcher-approved smaller-step test supersedes historical adaptive=false restrictions **for this diagnostic only**.

Inspected `run/dome/checks/ramp028_identity.sha256`:

| Artifact | SHA256 |
|---|---|
| `scripts/test_dome_attempt028_ramp.py` | `8d1c559771dbd64561abde7322357db25ebb141984ea3b32c3b3dd093fd49a3c` |
| `scripts/run_dome_attempt028_ramp.sbatch` | `714240ede5dc9cedda6af47d9b1774330233b26cc0afde59ce03ccda612a8640` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |
| `run/dome/bo_stage3/attempt_028/solve/sequence.json` | `41d546d73d8b285cbf42eb7fffc73e314d423b825fa003aed123aae4dc19f31c` |
| `run/dome/bo_stage3/attempt_028/solve/execution.json` | `9f67061ed0786d5f17ea8139800e559644f5679eaea2daa4a41a90c41e70a6e8` |

These are **manifest identities, not independently recomputed hashes**. Binary/input identities match the retained reviews and prior adaptive receipt. Native-source identity evidence remains `run/dome/checks/repair028_identity.sha256`; unrelated checkout changes are outside scope.

## Checks performed

### Delta and controls

Manual comparison with the prior adaptive runner/batch found only the intended step settings, output/job/log names and purpose text changed.

- `scripts/test_dome_attempt028_ramp.py:17–28` retains the archived command, enables adaptive continuation and adds initial/max step **0.1**. The archived command contains no conflicting initial/max-step flags.
- `src/simulations/Sim_Bilayer_Growth.cpp:1393–1419` accepts these controls: minimum `1/64`, growth `2`, maximum consecutive retries `8`. Successful steps recover toward the 0.1 cap; rejected steps halve. The minimum-step stop can occur before eight retries.
- Fixed2000 warmup configuration, refinement opt-in, tolerances, free boundary and final_only accumulation remain unchanged.
- Input repeats are **4+21+21=46**, with unchanged Attempt209 coefficients and face-shared hardening.

The target remains
\[
A(\lambda)=(1-\lambda)A_0+\lambda A_{46}.
\]
This affine metric homotopy preserves SPD between SPD endpoints. It is **not** proportional engineering strain or fractional physical passes. Intermediate solves do not replay growth/hardening.

### Floating-point endpoint behavior

**Important nonblocking caveat:** `src/simulations/Sim_Bilayer_Growth.cpp:2751–2771` uses strict `acceptedLambda < 1.0` and `min(1.0, acceptedLambda + stepSize)`.

With binary64 arithmetic (`src/libshell/common.hpp:37`), ten uninterrupted additions of 0.1 normally reach **0.9999999999999999**, not exactly1. Consequently, expect an **eleventh native invocation** for the approximately `1.11e-16` remainder.

This is safe endpoint handling, not false completion:

- The loop continues until exact1.
- Exact accumulated metrics are installed before the lambda1 solve.
- Terminal-state export requires lambda≥1 and acceptance.
- The remaining interval may be smaller than the nominal minimum step; that minimum controls failure-driven step reduction, not endpoint clipping.

The extra invocation can consume further warmup/verification time. Do not accept a near-one row as terminal or assume ten native solves suffice. No code change is required for this bounded diagnostic; endpoint-snapping changes would be a separate reviewed delta.

### Execution and repeated-file safety

`scripts/test_dome_attempt028_ramp.py:15–40` preserves allocation checks, fresh-directory refusal, archived-input copying, OpenMP8/thread-limit8, BLAS1 and one direct subprocess with timeout2400. The batch requests the stated resources. Submit from the repository root because the Slurm log path is relative.

`run/dome/repair_attempt028_ramp_v1` was absent when inspected.

Inherited repeated-solve controls remain appropriate:

- Optional fixed-filename trace stays disabled.
- Proof JSONL appends; convergence CSV records each returned attempt.
- `src/simulations/Sim_Bilayer_Growth.cpp:1340–1374` exports native state only for an accepted terminal endpoint, avoiding partial-substep filename collisions.
- Rejection restores accepted state/load before halving.
- No outer retry, campaign write or restart path was introduced.

The1800s native limit is cooperative; the2400s subprocess ceiling is cumulative. Neither promises completion.

## Evidence reused and limits

Directly inspected:

- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 tests passed**,59ms.
- `run/dome/repair_fixture_v3/sequence_result.json`: accepted three-cycle fixture, seven attempts, gradient `4.7416439423536765e-15`.
- Prior adaptive receipt, `sequence_convergence.csv` and native proof JSONL under `run/dome/repair_attempt028_adaptive_v1`.

The prior records confirm full-load rejection followed by accepted half-load equilibrium: gradient `1.004714714246349e-15`, `positive_pivots`, `gradient_target`; the process then timed out at2400s without a completed terminal result.

These support unchanged wiring and observed adaptive recovery—not ramp completion. Starting smaller avoids the known full-load-first expenditure, but ten-plus fresh solver invocations can still exhaust the total limit.

No edits, delegation, builds, simulations, numerical tests, hash recomputation or scheduler queries were performed. Supplied syntax-check success was not rerun. No redundant numerical suite is warranted for this configuration-only delta.

## Launch-blocking findings

**None within the reviewed bounded collection scope.**

## Required interpretation and nonblocking caveats

1. **Collection is not acceptance.**  
   `scripts/test_dome_attempt028_ramp.py:41–52` can record `diagnostic_collected` after a nonzero native return. Timeout records an error and skips proof summarization; inspect retained flushed files.

2. **Require the complete terminal gate before geometry:**
   - Successful sequence completion with all46 ordered cycles.
   - Accepted convergence row at **exact `lambda_trial=1`**, matched by order to terminal proof JSONL.
   - `termination=gradient_target`, finite gradient≤`5e-14`, `curvature=positive_pivots`, accepted=true.
   - Agreement among sequence result, terminal proof, exported native state and final endpoint artifacts.

   Proof rows lack lambda. `equilibrium_solves=1` counts the completed continuation block, not native invocations.

3. **Identity recording is not enforcement.**  
   The runner records actual hashes but does not compare expected values. Verify the bound packet before submission; consequential changes require reassessment.

4. **Scientific claim boundary.**  
   Success establishes model-conditional numerical feasibility along this load path, potentially on a different local branch. It does not isolate the earlier failure’s cause, establish equivalence with the historical BO toolchain, or validate physical parameters. Positive Hessian pivots are not geometric dome curvature.

For partial loading, rejection, timeout, missing proof or endpoint disagreement: **report equilibrium unresolved and stop**. No automatic retry, resource extension or BO observation. Even full numerical acceptance requires the unchanged geometric criteria and subsequent numerical credibility checks before claiming a dome.