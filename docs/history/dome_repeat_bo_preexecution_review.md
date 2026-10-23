# Stage2 repeat-domain BO — independent preexecution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified for the reviewed slice.**

Scope: one fresh Stage2 campaign through `scripts/run_dome_repeat_bo.sbatch`, output `run/dome/bo_stage2`, maximum **100 new evaluations**, comprising **24 frozen initial designs plus up to 76 BO proposals**, evaluated in pairs. Early stopping may prevent completion.

This review does not authorize execution, retries, extensions, recipe changes or verification runs. A screening pass is **not a verified dome**.

## Authority and reviewed scope

Authority inspected:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — version 3, revision 2359 as read.
- `docs/reports/dome_toolpath_investigation.md:190–204` — Stage2 authorization and Current status.
- Supplied researcher clarification accepting constrained, LHS-derived sampling rather than an exact Latin design, without narrowing bounds.

Reviewed all five new implementation files, `calibration/tests/test_dome_repeat_stage.py`, the frozen design, proposal fixture and identity packet. Inspected imported geometry/footprint helpers, copied Stage1 controller/proposer paths, runner checks, relevant native repeat/hardening/metric-update code, and retained Stage1 records.

Reused `docs/reports/dome_bo_preexecution_review.md` for unchanged runner, analyzer, curvature-estimator and scheduling evidence. Historical permissions and budgets were not reused.

`run/dome/bo_stage2` was absent in the inspected directory listing. Stage1 records corroborate **48 outcomes: 40 accepted and eight failed**, exhaustion without screening success, and attempt028 coverage **54.880208%**. Stage1 is read-only input to Stage2.

## Recorded identities

From `run/dome/checks/stage2_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `scripts/dome_bo_repeat_domain.py` | `dd24c797382d2e4e537ca9500eb3626738406f9f7428cefeed8aee9ece06a309` |
| `scripts/dome_repeat_lhs.py` | `38d3f74af17dfe35e098a167a8e9ede7a61e32b98cf933984400d8f3bbe6cb17` |
| `scripts/dome_repeat_propose.py` | `e147d4948908ed4572c017289641e97d67b08eb947fb2be20838b59357ceff16` |
| `scripts/dome_repeat_campaign.py` | `5bee16bea3f340ea4ea56a22c95c096521bbde01fef7f0452b673b5ceac1e56b` |
| `scripts/run_dome_repeat_bo.sbatch` | `1d2a2f08782e6f3e090787cc286fef2551adb07e8ceb5f0b6bb4c19961af9d99` |
| `scripts/dome_bo_domain.py` | `196f9f3930c00dbc62775d59c2ae77d56bb52365af2d1aca808594dc77681265` |
| `scripts/prepare_dome_baseline.py` | `99d4e326297dca894df1ad68d1f3e48fcc1d39a96edab940b018132852f91654` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `calibration/analysis/analyze_dome.py` | `0265c805b1846e959c448eaad379734dcdbaa1b649875f38bfc6a6a63bb08ed4` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac1942b3c01be28e321b839694300f7ec5419d7a9852ea70c7838ed180` |
| `calibration/configs/dome/baseline_attempt209.json` | `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862` |
| `run/dome/checks/stage2_lhs.json` | `65dc9eb959b3c3903c67aaaa88ab18eeb7c941f4b8e542798deeb8abd3882a18` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

These are **inspected recorded hashes, not independently recomputed hashes**. Shared entries agree with the prior review and Stage1’s `identities.json`. The supplied parent attestation states unchanged hashes match. The Stage2 test and proposal-fixture files were inspected but are not hashed in this packet.

## Checks and conclusions

### Domain, constraints and frozen ordering

`scripts/dome_bo_repeat_domain.py:10–41` correctly:
- Retains the six approved geometry bounds.
- Adds independent integer middle and shared-side repeats in **1–30**.
- Rejects fractional/out-of-range repeats and asymmetric sides.
- Reconstructs the complete candidate for fixed-recipe compatibility.
- Writes repeats as `[middle, side, side]`, giving **3–90 application cycles**.

No overlap exclusion or conditional angle remapping was found.

For the fixed placements, the initial-design constraints correctly reduce to
\[
(N_A-2)L_A\tan\alpha_A+8\le200,\qquad
(N_S-2)L_S\tan\alpha_S+4\le64
\]
in millimetres. Remaining footprint dimensions are already bounded by the approved lengths. Selected points also undergo the general footprint check.

`scripts/dome_repeat_lhs.py:25–53` rounds integers **before feasibility and distance calculations**. It selects by decoded normalized distance only—no coverage, solver outcome or warm-start performance enters selection.

The frozen artifact records seed **20261019**, **8192** Latin candidates, **1770** feasible candidates, **32** greedy starts and **24** selected points, with minimum distance **1.004400531885571**. Integer ties and the lack of exact-Latin/global-optimality guarantees are explicitly disclosed.

`scripts/dome_repeat_campaign.py:113–115` uses consecutive slices `[2*wave:2*wave+2]`. Thus attempt numbering and pair membership preserve frozen order; asynchronous completion cannot reorder subsequent design selection.

### Imports and warm-start normalization

The copied controller imports the repeat-domain module at line 13 and the repeat proposer at line 164. The proposer uses eight-dimensional Sobol points and the repeat-domain encoder. No stale six-dimensional proposal dispatch was found.

`scripts/dome_repeat_campaign.py:27–56` includes the five named historical records and **all 48 Stage1 observations**, not a coverage-selected subset. Stage1’s six-coordinate observation arrays are replaced with eight physical coordinates reconstructed from each actual sequence JSON.

Coordinates remain physical until `encode()` normalizes them once for GP training. Previously fixed repeats therefore become normalized coordinates **1/29** and **2/29**, not incorrectly retained six-dimensional or already-normalized values.

The resulting **53 records / 44 accepted scores / nine unscored failures** agree with supplied checks and inspected Stage1 statuses. Uniform-profile and other sparse changed-repeat runs are excluded by the declared selection. Failures exclude exact attempted coordinates but never become fabricated shape scores.

### GP numerics and training size

`scripts/dome_repeat_propose.py:19–60` preserves the reviewed heuristic:
- \(2^{14}\) scrambled Sobol candidates, decoded and footprint-filtered.
- CPU-double `SingleTaskGP`, standardized outcomes, variance \(10^{-6}\).
- Accepted geometric coverage only.
- Original-scale `best_f=y.max()`, consistent with outcome untransformation.
- Finite acquisition checks and greedy second-point separation of 0.1.

This is finite-pool separated log-EI, **not joint qEI**, physical identification or calibrated uncertainty.

`run/dome/checks/stage2_proposal_fixture.json` records seed **20261031**, **44** training observations, **3564** feasible candidates and two eight-dimensional proposals. The supplied job4355927 result supports this bounded path.

**Production training size is larger:** the first BO fit can contain **68** accepted observations; the last can contain **142**. The fixture does not certify those runtimes. The 300-second proposal timeout makes noncompletion a stop, not permission to extend.

### High-repeat semantics and scientific relevance

Native inspection confirms ordered application and shared face history:
- `src/simulations/Sim_Bilayer_Growth.cpp:2450–2613`
- `src/libshell/ZigZagSequenceGrowth.hpp:653–669`

With the calibrated floor zero and \(\beta=0.2231435513142097\),
\[
q(n)=e^{-\beta n}=0.8^n.
\]
For otherwise identical hits starting from zero history, scalar weights sum to
\[
\sum_{n=0}^{R-1}0.8^n=5(1-0.8^R).
\]
At 30 hits this is approximately **4.99381**, or **99.876%** of its asymptote. This is a scalar-weight illustration—not an additive-strain or curvature prediction.

Multiple strip hits can saturate a face faster than one hit per repeat. Earlier middle treatment also reduces subsequent side increments on overlapping faces. Thus separate repeat controls are scientifically meaningful but **not independent physical effects or proportional dose controls**.

The multiplicative update uses metric congruence with positive incremental stretches and explicit SPD checks (`src/libshell/GrowthHelper.hpp:719–826`). The approved coefficients and nonnegative hardening factors introduce no evident high-repeat singularity. The unchanged runner checks actual cycle count, accumulation history and exactly one accepted final-only equilibrium; it does not assume eight cycles.

### Resources, accounting and stops

Static paths match the requested contract:
- One **16 CPU / 64 GB / 12-hour** batch.
- At most two exclusive **8 CPU / 32 GB** solve steps.
- Analysis replaces its completed solve with a **1 CPU / 4 GB** step.
- Limits: **600s native, 1500s solver process, 1700s solve step, 600s analysis, 300s proposal**.
- Admission reserve: \(300+1700+600=2600<2700\) seconds, before overhead.

Fifty waves cap reservations at 100. Both reservations precede execution; both outcomes are retained. Screening, two consecutive wholly failed waves, infrastructure/analysis errors, identity drift or allocation reserve halt further work. Proposal/identity exceptions terminate rather than retry. Existing-output refusal prevents automatic replay.

These are inspected safeguards, **not independently demonstrated runtime enforcement**.

## Launch-blocking findings

**None for this exact slice.**

## Nonblocking caveats

1. **Saturation limits effective exploration.**  
   `scripts/dome_repeat_lhs.py:29–41` and `scripts/dome_repeat_propose.py:37–53` measure distance in raw normalized repeats, not effective treatment. High-count designs may be physically near-equivalent despite wide parameter spacing. Report this limitation; no bounds or sampling change is required.

2. **Ordering tests are weaker than the implementation.**  
   `calibration/tests/test_dome_repeat_stage.py:51` compares sets, not exact reservation order. Static inspection establishes the current ordering. An optional regression improvement is checking each numbered reservation against the corresponding frozen point.

3. **Runtime and high-count evidence remain limited.**  
   The supplied tests mock execution; no new 90-cycle native run or larger-training-set GP benchmark was performed. Existing variable-cycle evidence and bounded fail-stop behavior support launch without requiring a broad new test campaign.

4. **Identity and provenance checks are not continuous.**  
   `scripts/dome_repeat_campaign.py:95–107` snapshots startup files and checks before waves, rather than validating startup against the review packet or checking after the terminal wave. Warm-start hashes link retained records but do not independently revalidate every historical score. Preserve the reviewed inputs and perform the parent’s normal prelaunch identity check.

5. **Durability is not resumability.**  
   Exclusive creation prevents overwrite, but writes are not transactional checkpoints. Remote-step cleanup after timeout was not exercised here. Interrupted artifacts require assessment; no automatic restart is approved.

## Scientific boundaries and evidence limitations

The target is a constructive **model-conditional dome design**, not material recalibration or proof of a physical mechanism. More coherent curvature could reflect treatment redistribution, saturation/order effects, or estimator/discretization sensitivity.

A geometric screening pass must stop proposals and proceed to separately planned verification. Director disagreement, spatial saddle/fold structure, scale sensitivity or numerical instability cannot be dismissed because coverage improves. The frozen ROI, magnitude/ratio thresholds and 80% coherent-sign criterion remain unchanged.

No pass means a negative or unresolved result for this bounded search—not impossibility or global optimality. Retained observations are adaptive training data, not independent validation. Stage2 versus Stage1 best scores are descriptive comparisons with unequal search domains and effort.

**Performed:** read-only source/artifact inspection, algebraic checks, Stage1 status enumeration and recorded-identity comparison.

**Not performed:** tests, shell commands, hash recomputation, GP fitting, solver execution, scheduling, builds, edits or delegation. Three passing Stage2 tests, Bash syntax success and jobs4355746/4355927 are supplied attestations, not reviewer reruns. Unchanged analyzer evidence was reused rather than rerun.