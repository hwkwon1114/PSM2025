# Nine-variable replacement BO — independent preexecution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified for the reviewed slice.**

Scope: one fresh campaign through `scripts/run_dome_offset_bo.sbatch`, output `run/dome/bo_stage3`, capped at **92 new evaluations: 24 frozen initial designs plus up to 68 BO proposals**, in pairs. The eight launched Stage2 attempts remain charged against the researcher’s 100-attempt allowance.

This review does not grant execution permission, retries, extensions, recipe changes or verification runs. Screening success is **not a verified dome**.

## Authority and scope inspected

- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — version 3, revision 2457 as read.
- `docs/reports/dome_toolpath_investigation.md:198–208` — **Replacement stage authorization** and **Current status**.
- Supplied researcher authorization accepting shared symmetric side offset **0–118 mm**, unchanged previous bounds, preservation of completed results and cancellation of Stage2.

Completely inspected:
- `scripts/dome_offset_{domain,lhs,propose,campaign}.py`
- `scripts/run_dome_offset_bo.sbatch`
- `calibration/tests/test_dome_offset_{domain,stage}.py`
- `run/dome/checks/stage3_lhs.json`
- `run/dome/checks/stage3_proposal_fixture.json`
- `run/dome/checks/stage3_identity.sha256`
- `run/dome/checks/stage2_replacement.json`

Also inspected imported domains, footprint generation, baseline configuration, unchanged runner, preceding controller/proposer, retained outcome records and relevant native hardening/application code.

Reused unchanged runner/analyzer/estimator/scheduler evidence from:
- `docs/reports/dome_repeat_bo_preexecution_review.md`
- `docs/reports/dome_bo_preexecution_review.md`

Historical permissions and exhausted budgets were not reused. `run/dome/bo_stage3` was absent from the inspected directory listing.

## Recorded identities

From `run/dome/checks/stage3_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `scripts/dome_offset_domain.py` | `1ef5c991c8e1f579318661707eb7e83f9c51df825444bd996a3271fcd1205b67` |
| `scripts/dome_offset_lhs.py` | `ac27d0545fcf8d503a300a9ef02c8abf9cfa457ca84b10501f4f9581c0f0bc9c` |
| `scripts/dome_offset_propose.py` | `b5919173d8b48ff0976483f1733e4a5960d1893e5d60e6acdc724b5eb5dbfac8` |
| `scripts/dome_offset_campaign.py` | `ee185bec7e24b4951ee170318fb2222ec421961e661e15293944ed4ecd6f1e43` |
| `scripts/run_dome_offset_bo.sbatch` | `30812caf5f7e909952e393f9aab49e99cd684fc31e79dcf60169c5b72b98958c` |
| `scripts/dome_bo_repeat_domain.py` | `dd24c797382d2e4e537ca9500eb3626738406f9f7428cefeed8aee9ece06a309` |
| `scripts/dome_bo_domain.py` | `196f9f3930c00dbc62775d59c2ae77d56bb52365af2d1aca808594dc77681265` |
| `scripts/prepare_dome_baseline.py` | `99d4e326297dca894df1ad68d1f3e48fcc1d39a96edab940b018132852f91654` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `calibration/analysis/analyze_dome.py` | `0265c805b1846e959c448eaad379734dcdbaa1b649875f38bfc6a6a63bb08ed4` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942b3c01be28e321b839694300f7ec5419d7a9852ea70c7838ed180` |
| `calibration/configs/dome/baseline_attempt209.json` | `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862` |
| `run/dome/checks/stage3_lhs.json` | `04101326c9c2fe5b928aa03983378578ba5fad165fe42b72eeeb9bb340736fac` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

These are **inspected recorded hashes, not independently recomputed hashes**. Shared identities agree with the previous reviews and retained Stage1/Stage2 identity manifests. Tests, proposal fixture and replacement receipt were inspected but are not hashed in this packet.

## Checks and conclusions

### 1. Domain and shifted feasibility

`scripts/dome_offset_domain.py:22–28` applies shifts `[0,+d]` and `[0,-d]` **before** invoking the retained repeat/geometry validator. It therefore does not accidentally retain the old 64 mm side-span restriction.

The constraints correctly reduce to, in millimetres,
\[
(N_A-2)L_A\tan\alpha_A+8\le200,
\]
\[
(N_S-2)L_S\tan\alpha_S+4+2d\le300.
\]

The remaining dimensions fit because \(L_A+8\le204<300\) and \(L_S+4\le200\). General transformed-footprint validation remains active, with its existing tolerance.

The previous six geometry bounds and integer repeats **1–30** are unchanged. Complete reconstruction checks fixed-recipe compatibility. No overlap prohibition was introduced; coincident B/C paths at \(d=0\) remain separate ordered applications. Calibration, profiles, widths, rotations, A→B→C order and free-boundary final-only execution are preserved.

### 2. Retained evidence and cancellation accounting

`scripts/dome_offset_campaign.py:27–74` carries:

| Source | Accepted | Numerical failures | Interrupted |
|---|---:|---:|---:|
| Declared five historical records and all 48 Stage1 outcomes | 44 | 9 | 0 |
| All eight launched Stage2 attempts | 6 | 0 | 2 |
| **Total** | **50** | **9** | **2** |

All nine coordinates are reconstructed from actual sequence JSON, rather than appending to potentially stale observation arrays. Retained side shifts are ±118 mm, encoded as offset coordinate **1**, not 118 in normalized GP input.

Direct inspection corroborated:
- Six Stage2 observations and their matching primary geometric scores; none screened positive.
- Attempts007/008 have `status: started`, no observation or endpoint JSON, and step logs reporting cancellation.
- The replacement receipt records eight charged attempts and **92 remaining**.

The interrupted records receive neither coverage nor a numerical-failure label. They enter exact-coordinate exclusions only. All six completed Stage2 observations are included without performance selection. The previously declared historical warm-start selection remains unchanged; this is not a census of every earlier sparse diagnostic.

### 3. Frozen initial design and proposal normalization

`scripts/dome_offset_lhs.py:11–53` rounds discrete variables before feasibility and distance calculations. Selection uses canonical normalized physical coordinates, not latent fractional counts or outcome values.

The frozen artifact records:
- Seed **20261020**
- **8192** Latin candidates; **2995** feasible
- **32** greedy starts; **24** selected points
- Minimum normalized separation **1.1546488849225625**
- Offset range **1.902330–116.443159 mm**

This is correctly described as a **constrained LHS-derived, best-found maximin subset**, not an exact Latin design or global optimum. Full shifted-footprint validation is applied to selected points; the controller additionally checks uniqueness and exclusion of retained coordinates.

The controller dispatches the nine-dimensional proposer. Accepted records are encoded once for GP training. Failures and interruptions are excluded from training but retained for novelty checks.

The proposer preserves CPU-double `SingleTaskGP`, standardized outcomes, variance \(10^{-6}\), finite acquisition checks and separated greedy log-EI selection. Original-scale `best_f=y.max()` is consistent with outcome untransformation. The variance is numerical regularization, not measured uncertainty; paired selection is not joint qEI.

### 4. Scientific relevance, overlap and saturation

Moving side treatment inward is relevant to the documented spatial gaps and persistent saddle bands. However, it simultaneously changes placement, overlap and shared hardening history; it does not isolate a physical mechanism.

Native code retains face-shared history and ordered multiplicative updates:
- `src/libshell/ZigZagSequenceGrowth.hpp:653–669`
- `src/simulations/Sim_Bilayer_Growth.cpp:2445–2613`

With the frozen calibration,
\[
q(n)=e^{-0.2231435513142097n}=0.8^n,\qquad
\sum_{n=0}^{R-1}q(n)=5(1-0.8^R).
\]
Thirty otherwise identical hits reach approximately **99.876%** of this scalar-weight asymptote. This is not an additive-strain or curvature prediction. Multiple strip hits and prior middle treatment can saturate overlapping faces sooner.

The retained Stage2 attempt006 provides actual high-repeat evidence: **82 accumulation cycles**, one accepted final-only equilibrium, gradient **4.61695×10⁻¹⁴**, and runner elapsed time **232.71 s**. This supports the unchanged variable-cycle path, but does not certify every inward/coincident configuration or 90-cycle runtime.

### 5. Controller, stops and resources

`scripts/dome_offset_campaign.py:124–174` permits **46 waves × 2 = 92** new reservations. The first twelve waves use consecutive frozen-design slices. The updated test checks exact numbered reservation order, not merely set equality.

Both reservations precede evaluation. Both outcomes are collected when possible. Stops remain:
- Geometric screening pass, including director-disagreement cases requiring verification
- 92-attempt cap
- Two consecutive wholly failed numerical waves
- Infrastructure/analysis error
- Identity drift or insufficient allocation reserve

Existing-output refusal and absence of resume logic prevent automatic replay.

The batch contract matches the proposal: **16 CPUs / 64 GB / 12 h**, two exclusive **8 CPU / 32 GB** solve steps, with analysis replacing its completed solve. Limits remain **600 s native / 1500 s solver / 1700 s step / 600 s analysis / 300 s proposal**. The nominal longest wave path is \(2600<2700\) seconds, within the 45-minute admission reserve before overhead.

These are inspected safeguards, **not a claim of independently demonstrated runtime enforcement**.

## Launch-blocking findings

**None for the reviewed contents, identities and launch contract.**

## Nonblocking caveats

1. **Effective exploration differs from parameter spacing.**  
   `scripts/dome_offset_lhs.py:29–41` and `scripts/dome_offset_propose.py` use raw normalized repeats and offset. Saturated high-repeat configurations may be physically near-equivalent despite large parameter distances. Report this limitation; no sampling change is required.

2. **The GP fixture is smaller than production fits.**  
   `run/dome/checks/stage3_proposal_fixture.json` records 50 accepted observations and two proposals from 5998 feasible candidates. With all new solves accepted, the first production BO fit has **74** observations and the last **140**. Their runtimes were not demonstrated; the 300-second fail-stop remains applicable.

3. **Identity/provenance checks are bounded, not continuous.**  
   `scripts/dome_offset_campaign.py:114–125` snapshots startup files and checks before waves, rather than validating against the review packet or checking after the terminal wave. Preserve the reviewed inputs and perform the parent’s normal prelaunch identity comparison. Stage2 imported observation hashes are not added alongside sequence hashes; adding them would improve auditability without changing results.

4. **Durability is not resumability.**  
   Exclusive writes prevent overwrite but are not transactional checkpoints. The new mocks do not exercise remote-step cleanup after timeout. Interrupted artifacts require assessment, never automatic restart.

## Evidence limits and scientific claim boundaries

**Performed:** read-only source/artifact inspection, algebraic feasibility and saturation checks, retained-status enumeration, cancellation-log inspection, six Stage2 score comparisons and recorded-identity comparison.

**Supplied attestations, not reviewer reruns:** six tests passing in **0.580 s**, Bash syntax success, design job4358438 and actual nine-dimensional GP fixture job4358526. The fixture launched no solver.

**Not performed:** commands, tests, hash recomputation, GP fitting, simulation, scheduling, builds, edits or delegation. Unchanged analyzer evidence was reused; historical geometry was not recomputed.

The target remains a constructive **model-conditional design**, not physical identification, fabrication validation or material recalibration. Adaptive BO observations are training evidence, not independent validation. Baseline and best-score comparisons are descriptive, not equal-budget optimizer comparisons.

The frozen ROI, coherent-sign 80% area criterion, magnitude/ratio thresholds and spatial/director/scale/numerical verification gates remain unchanged. A screening pass requires separately planned verification; exhaustion without a pass supports a bounded negative result, not impossibility or global optimality.