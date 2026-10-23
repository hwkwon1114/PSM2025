# Continuation BO — independent preexecution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified for the reviewed slice.**

Scope: one fresh campaign at `run/dome/bo_continuation_stage1`, capped at **64 new evaluations**, comprising 24 frozen initial configurations and up to 40 subsequent proposals. Maximum concurrency remains **two 8CPU/32GB solves**, within the proposed **16CPU/64GB/36h** allocation.

This review grants no execution permission, additional evaluations, retries, extensions, recipe changes or verification runs.

## Authority and reviewed identities

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`
- `docs/reports/dome_toolpath_investigation.md`, especially **Current status / next action**, lines 248–258.
- Prior continuation and nine-variable reviews and their retained evidence.

Current authority explicitly supersedes the historical adaptive=false recipe **for this continuation campaign**. The accounting remains \(100-8-28=64\), not a renewed allowance.

Fully inspected:
- `scripts/run_dome_continuation.py`
- `scripts/dome_continuation_campaign.py`
- `scripts/run_dome_continuation_bo.sbatch`
- Both new continuation test files.
- Imported geometry/repeat/offset domains, footprint helper and proposer.
- Existing analyzer/curvature estimator, baseline configuration and frozen initial design.

Also inspected relevant native continuation/export code, seed command/input/history/proofs/result/metrics, previous controller branches, retained native test log and GP fixture.

Recorded identities from `run/dome/checks/continuation_bo_identity.sha256`:

| Artifact | SHA256 |
|---|---|
| `scripts/run_dome_continuation.py` | `b723c753ca3160c28008b87dfde471a1694c785ef0f9cd2a4f5664b6da79248f` |
| `scripts/dome_continuation_campaign.py` | `ccf36b760e68baa3622e1939eb5c29b08449bb98d61474b58ad4042486b74052` |
| `scripts/run_dome_continuation_bo.sbatch` | `5c6ec63b7e175a1c33e9568020108d87a69e4d8f1e3672f850753a334c37a0ea` |
| `run/dome/checks/continuation_lhs.json` | `05cad044c32206808eed6a7697e143103e7b68855622516f469a5e7b84063101` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |

Shared domain/proposer/analyzer manifest identities agree with the earlier reviewed packets. Numerical-source provenance remains in `run/dome/checks/repair028_identity.sha256`.

**These are inspected manifest identities, not independently recomputed hashes.** The proposed campaign output directory was absent from the inspected listing.

## Checks and conclusions

### Scientific relevance and fair information use

The campaign addresses a constructive, **model-conditional toolpath-design** question: whether the authorized numerical continuation recipe can produce accepted equilibria whose geometry passes the frozen screening criterion.

- Attempt209 coefficients, hardening, profiles, widths, rotations, sheet and ordered applications remain fixed.
- The nine geometry/repeat/offset variables retain their approved bounds and complete footprint checks.
- The first configuration is the adapted collaborator baseline under the **new continuation recipe**. Historical baseline scores cannot substitute for its result.
- The remaining initial configurations match the first 23 frozen Stage3 geometries. This is correctly labelled a reused space-filling design, **not an exact LHS or newly optimized design**.
- Historical full-load scores and failure labels do not enter training or exclusions. Repeating their geometries under the changed numerical recipe is valid and consumes the new allowance.
- Exact duplicates are excluded within the current recipe, including the retained seed. Integer coordinates are decoded before feasibility and canonical encoding.

The only retained score is the accepted ramp60 endpoint. Its sequence has **4+21+21=46 cycles**; the inspected receipt matches the proposed mesh, solver, calibration-related settings and continuation controls. Its recorded coverage is **0.38026041666666666**, not qualifying.

BO observations are adaptive training evidence, not independent validation or an equal-budget comparison between optimizers. Changed coverage could reflect layout changes, local equilibrium branches, overlap/hardening interactions or estimator sensitivity—not an identified physical mechanism.

### Mathematics and endpoint acceptance

The unchanged native path interpolates accumulated target metrics:
\[
A(\lambda)=(1-\lambda)A_0+\lambda A_{\mathrm{final}}.
\]
This preserves SPD between SPD endpoints. It is **not proportional engineering strain or fractional physical applications**.

`scripts/run_dome_continuation.py:19–59` checks:
- Completed final-only sequence and expected variable cycle count.
- Ordered accumulation history, positive hit events and terminal-only physical-cycle solve.
- Equal CSV/proof counts and ordered pairing.
- Rollback-aware accepted-load progression.
- Every accepted substep’s `gradient_target`, `positive_pivots` and gradient≤`5e-14`.
- Exact terminal **lambda=1**, plus agreement of terminal proof, sequence result and exported-state proof.

Native source confirms exact final metrics are installed before terminal acceptance/export. Proof rows lack lambda, so ordering and the matched CSV remain essential.

Direct inspection of `run/dome/repair_attempt028_ramp60_v1/` corroborated:
- All 46 accumulation rows.
- Eleven accepted CSV/proof pairs, including the tiny final increment from `0.99999999999999989` to exact 1.
- Final gradient **1.319921482080224e-15**.
- Successful process return after **3228.497s**.

The unchanged geometric estimator retains the fixed ROI, reference-area denominator, coherent-sign scoring, magnitude/ratio thresholds and invalid-fit treatment. Geometry≥80% triggers verification even when director classification disagrees; it does not establish a dome.

### BO, failure accounting and stopping

`scripts/dome_continuation_campaign.py:103–153` implements 32 waves of two reservations. Reservations precede execution; outcomes are retained before subsequent decisions.

Stops include:
- Screening pass.
- 64-attempt exhaustion.
- Two consecutive wholly failed numerical waves.
- Infrastructure/analysis errors.
- Identity drift or inadequate allocation reserve.

Runner-recorded **3600s process timeouts** become unscored, counted failures. A **3900s step timeout**, analysis timeout or proposal exception instead stops the campaign. No partial endpoint receives a shape score.

The unchanged proposer uses CPU-double `SingleTaskGP`, standardized outputs, variance \(10^{-6}\), finite acquisition checks and separated greedy log-EI. This is not joint qEI; the variance is numerical regularization, not measured scientific uncertainty. With all evaluations accepted, GP fits range from 25 to 63 observations.

### Resources and preservation

The batch requests match the proposed allocation. Solve steps request 8CPU/32GB; each analysis replaces its completed solve, so the intended concurrent requests remain within 16CPU/64GB.

Limits are:
- Native: **1800s per substep**.
- Native process: **3600s per candidate**.
- Solve step: **3900s**.
- Analysis: **600s**.
- Proposal: **300s**.
- Before-wave reserve: **4800s**.

Fresh-directory refusal and exclusive record creation preserve old stages. There is no automatic resume/retry path. This is bounded execution logic, **not independently demonstrated runtime enforcement or restartability**.

## Launch-blocking findings

**None for the inspected contents and stated contract.**

## Nonblocking caveats and smallest improvements

1. **Identity checks are bounded, not comprehensive.**  
   `scripts/dome_continuation_campaign.py:27–50,93–105` checks seed recipe flags, binary identity and metrics-to-state identity, then snapshots campaign sources. It does not compare startup files against the review packet or bind loaded libraries. Preserve the reviewed files and perform the ordinary prelaunch packet comparison. Optionally compare the seed sequence hash directly with its receipt and record seed receipt/metrics hashes; no new approval machinery is needed.

2. **Reserve is nominal rather than a completion guarantee.**  
   `scripts/dome_continuation_campaign.py:106` uses exactly \(3900+600+300=4800\) seconds, leaving no explicit allowance for filesystem/scheduler overhead at those maxima. Allocation expiry can therefore interrupt the last wave. A modest overhead margin is optional; interruption must remain unscored and must not trigger replay.

3. **Timeout cleanup and the new classification branch lack direct execution evidence here.**  
   `scripts/dome_continuation_campaign.py:59–67` was inspected statically. The supplied mocks do not exercise real remote-step termination. Optional lightweight mocks should distinguish runner timeout, unclassified failure and step timeout. Existing safeguards and fail-stop behavior make this nonblocking.

4. **Acceptance remains local and recipe-conditional.**  
   Repeated warmups and an eleventh invocation can consume the total cap. Positive Hessian pivots are not positive Gaussian curvature; full-load acceptance is not a global minimum or physical validation. If the matched baseline fails, report that comparison as unresolved.

5. **Sampling distance is not effective treatment distance.**  
   Raw repeat spacing ignores saturation and overlapping face-shared history. The single compatible seed and accepted-only GP may explore inefficiently near numerical failures. Neither warrants importing incompatible scores or changing the approved recipe.

## Evidence limits and approval boundary

**Performed:** read-only source/artifact inspection, recorded-identity comparison, continuation/footprint/curvature reasoning, initial-design comparison and retained seed-history/proof inspection.

**Reused evidence:**
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: 29 tests passed; unchanged numerical implementation only.
- `run/dome/checks/stage3_proposal_fixture.json`: actual unchanged nine-dimensional proposer exercised with 50 accepted observations; not proof of future fit runtime.
- Earlier controller tests: unchanged stopping branches only.

**Supplied attestations, not reviewer reruns:** two verifier tests in 0.004s, three stage tests in 0.805s and successful Bash syntax check.

**Not performed:** commands, tests, builds, simulations, hash recomputation, live scheduler queries, loaded-library audit, full native-state array audit or delegation. Live normal-partition MaxTime=2days is supplied evidence, not independently queried here.

A screening pass requires separately planned spatial/director/scale and numerical verification. Exhaustion or stopping without a pass supports a bounded negative/inconclusive result—not impossibility, global optimality or permission to expand.