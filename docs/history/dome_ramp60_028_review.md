# Independent delta review — attempt028 60-minute ramp

## Verdict: APPROVED_WITH_CAVEATS

**No unresolved scientific or operational launch blockers for ONE fresh, bounded rerun.**

Reviewed limits: **3600s total native-process timeout; 1800s native limit per substep; 8 CPUs / 32GB / 65-minute allocation.** Initial and maximum continuation steps remain **0.1**.

Approval covers this reviewed slice only. It grants no additional execution permission, further extension/retry, BO restart, rebuild or weakened acceptance.

## Authority and scope

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`
- `docs/reports/dome_toolpath_investigation.md`, particularly **Current status / next action** and **Ten-percent ramp result**
- `docs/reports/dome_ramp028_review.md`
- `docs/reports/dome_adaptive028_review.md`
- `docs/reports/dome_linear_repair_review.md`

Current authority explicitly permits restarting from the original flat input because no restartable 90% checkpoint exists. Historical no-rerun restrictions are superseded **only for this one attempt**.

Reviewed new wrapper/batch against their prior-ramp counterparts, archived command/input, retained numerical evidence and relevant continuation/export code. Unrelated checkout changes remain outside scope.

## Current identities

Inspected `run/dome/checks/ramp60_028_identity.sha256`:

| Artifact | SHA256 |
|---|---|
| `scripts/test_dome_attempt028_ramp60.py` | `8fac7b84d2e5fa73f44bac28b60ec2b68bc6f296e249e72e9dd281404e85e517` |
| `scripts/run_dome_attempt028_ramp60.sbatch` | `d49dd4845f9bf0e17d2a9f6f3907f610016ad0099141d7e9a4faf72692a99ec2` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |
| `run/dome/bo_stage3/attempt_028/solve/sequence.json` | `41d546d73d8b285cbf42eb7fffc73e314d423b825fa003aed123aae4dc19f31c` |
| `run/dome/bo_stage3/attempt_028/solve/execution.json` | `9f67061ed0786d5f17ea8139800e559644f5679eaea2daa4a41a90c41e70a6e8` |

These are **manifest identities, not independently recomputed hashes**. Binary/input/command identities match the prior ramp packet; binary/input also match its execution receipt. Numerical-source evidence remains bound through `run/dome/checks/repair028_identity.sha256`.

## Checks performed

### Intended delta and unchanged recipe

Manual comparison found only:
- Total subprocess timeout **2400→3600s**.
- Allocation **45→65 minutes**.
- Fresh output, job/log names, script reference and purpose text.

`scripts/test_dome_attempt028_ramp60.py:17–28` retains adaptive continuation, refinement opt-in, fixed2000 warmup, native1800s, absolute gradient gate `5e-14`, free boundaries and final-only target accumulation. Archived input remains **4+21+21=46 ordered cycles**, with unchanged Attempt209 calibration.

The unchanged affine metric homotopy
\[
A(\lambda)=(1-\lambda)A_0+\lambda A_{46}
\]
preserves SPD between SPD endpoints. It is not proportional engineering strain or fractional physical passes. No new numerical algorithm or tolerance requires another numerical suite.

### Resources and timeout containment

- `scripts/run_dome_attempt028_ramp60.sbatch:2–10`: valid Slurm syntax for account `p33082`, partition `short`, one node/task, eight CPUs,32GB and `01:05:00`.
- The65-minute request is below the documented four-hour `short` ceiling in `docs/reports/boundary_conditions_and_springback_synthesis.md:74`. **Current live partition policy/availability was not queried.**
- `scripts/test_dome_attempt028_ramp60.py:15–40` retains allocation checks, fresh-directory refusal, OpenMP8/thread-limit8 and BLAS1. The new output directory was absent when inspected.
- One direct native subprocess receives `timeout=3600`; this ceiling is cumulative across substeps, not renewed for each native1800s allowance. Python’s timeout path kills/waits for the direct child. The3900s allocation leaves a nominal300s setup/cleanup margin.
- No outer retry or campaign write was introduced. Submit from the repository root because the Slurm output path is relative.

This is a static containment assessment, not runtime-enforcement certification or an exact real-time deadline guarantee.

### Retained evidence

Directly inspected:
- `run/dome/repair_attempt028_ramp_v1/diagnostic_execution.json`: job4398808 timed out after **2400.010s**.
- Its `sequence_convergence.csv` and `bilayer_zigzag_sequence_native_hybrid_proof.jsonl`: nine ordered, accepted steps through `0.89999999999999991`, all `gradient_target`/`positive_pivots`, gradients≤`5e-14`; final gradient **3.9218090780679738e-15**.
- Directory contents: no completed sequence result or exported terminal native state.
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 tests passed**.
- `run/dome/repair_fixture_v3/sequence_result.json`: accepted three-cycle fixture, gradient **4.7416439423536765e-15**.

These support unchanged wiring and partial continuation—not full-load convergence or restartability.

No edits, builds, tests, simulations, hash recomputation or scheduler queries were performed. Supplied Python/Bash syntax-check success was not rerun.

## Launch-blocking findings

**None within the stated bounded collection scope.**

## Nonblocking caveats and required interpretation

1. **Completion may require an eleventh invocation.**  
   `src/simulations/Sim_Bilayer_Growth.cpp:2751–2771` uses strict endpoint comparison. Ten binary64 additions of0.1 can leave `0.9999999999999999`; another native invocation qualifies exact1. Fixed2000 warmup repeats can still exhaust3600s. Do not snap the endpoint or change warmup under this approval.

2. **Collection status is not numerical acceptance.**  
   `scripts/test_dome_attempt028_ramp60.py:41–52` can record `diagnostic_collected` after a nonzero native return. Timeout skips proof summarization; inspect retained flushed files.

3. **Preserve the complete acceptance gate before geometry.**
   - Successful sequence completion and all46 ordered cycles.
   - Accepted CSV row at **exact `lambda_trial=1`**, paired by order with terminal proof.
   - `gradient_target`, finite gradient≤`5e-14`, `positive_pivots`, accepted=true.
   - Agreement among sequence result, proof, exported native state and final endpoint artifacts.

   Proof rows lack lambda. `equilibrium_solves=1` counts the continuation block, not native invocations. Terminal-only state export remains guarded at `src/simulations/Sim_Bilayer_Growth.cpp:1346–1389`.

4. **Identity recording is not enforcement.**  
   The wrapper records hashes without expected-value comparison. Parent should verify the five-entry packet before submission. Consequential changes require reassessment; dynamically loaded libraries were not independently audited.

## Scientific decision and boundaries

This rerun tests whether the unchanged ramp reaches an accepted full-load endpoint within the approved longer ceiling. Prior partial success does not distinguish eventual full-load convergence from branch difficulty or further time exhaustion.

- **Complete accepted endpoint:** audit the gates above, then apply unchanged geometric criteria and verification requirements.
- **Timeout, rejection, partial loading or inconsistent/missing evidence:** retain records, report equilibrium unresolved and stop.

Neither outcome authorizes another extension or BO continuation. Success is model-conditional numerical feasibility, potentially on a different local branch—not physical identification, historical-toolchain equivalence or a dome claim. Positive Hessian pivots are not geometric dome curvature; the frozen80% geometric criterion and numerical-credibility checks remain required.