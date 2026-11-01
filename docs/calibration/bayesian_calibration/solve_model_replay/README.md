# "Does it solve" model: offline replay and stage-2 protocol

Status: replay done (offline, no solver jobs, Run 9 untouched). Stage-2 approved by the researcher (m00905) and Astra-reviewed APPROVED_WITH_CAVEATS (below); launch authorizes only this protocol.
Code: `calibration/bo/replay_solve_model.py`; raw output: `replay.json`, `replay.log` (this folder).
Data: the finished joint Run 6 + Run 10 campaign, `run/joint_run6_run10` (job 5260033, early stop at 48 evaluations,
29 valid, 17 solver, 2 timeout; best attempt 32, J = 1.1955 mm vs baseline 1.2157 mm). Nothing in it was modified.

## Replay result (48 points, small sample, retrospective)

Failure = solver / timeout / registration. Predicting success of each wave's points from earlier waves only
(waves 4-12, n = 36, 25 successes):

| Model | Brier | log loss | AUC |
|---|---|---|---|
| Base rate | 0.263 | 0.720 | - |
| Logistic regression (6 normalized params) | 0.121 | 0.406 | 0.95 |
| GP classifier (Matern-5/2 ARD) | 0.095 | 0.321 | 0.94 |
| Current pipeline's imputed-failure GP (mean score only) | - | - | 0.91 |

Leave-one-out over all 48: logistic Brier 0.087 / AUC 0.98; GPC Brier 0.094 / AUC 0.95.

- Failure is highly predictable from the parameters; the success model is not worse than the imputed GP and is
  scored as probabilities (Brier/log loss are better than the base rate; calibration itself was not shown). The AUC difference 0.94 vs 0.91 is within noise at n = 36.
- Success rate by region: floor > 0.5: 1/13; floor <= 0.5: 28/35; literature box (floor <= 0.10 and
  beta in [0.01, 0.30]): 12/14 (86 %); outside that box: 17/34 (50 %).
- Replayed alternative batches (LogEI on valid points x P(success)): the model's own P(success) is 0.75-0.95 per
  wave vs 0.46-0.82 for the batches actually run, and 33/35 of their nearest later-observed points succeeded
  (nearest distances 0.18-0.9 in normalized units, so this is weak evidence). Whether the alternatives would have found
  lower J was **not** assessed; that needs new solves.
- Limits: 48 points, one campaign, labels partly produced by the same search that steered toward success, GPC
  hyperparameters refit per call, no multiple replicates. This shows the failure model is feasible and predictive,
  not that it improves the calibration.

## Stage-2 protocol (researcher approved "lets do this", m00905, outside goal mvnbhasq-ex2ro1; Astra review: APPROVED_WITH_CAVEATS, no launch blockers)

Question: at equal evaluation count, does a success-aware search reach lower joint J, with fewer failed solves, than the
failure-imputation search of the finished campaign? Two arms in one new box; Run 9 sealed (never read).

- Fixed (same as the finished campaign): objective (equal-weight symmetric RMS, Run 6 + Run 10, failed point imputed
  1.5 x worst valid J in arm B only), corrected convention, `bin/shell` with adaptive cutback, 3600 s per case, 6 parameters,
  q = 4, 15 waves max (12 design + 48 BO = 60), min separation 0.10, early stop from wave 11 (<0.5 % over 4 waves),
  baseline gate (attempt 1 reproduces Run 6 1.18194 / Run 10 1.24824 mm within 1e-3), stop at 8 consecutive failures,
  infrastructure failure stops for review.
- Box B': beta 0.01-0.30 (log), floor 0-0.50 (linear); gtop, gbot, ortho_top, ortho_bottom unchanged. Baseline Attempt 209 is inside B'.
- Design: identical for both arms (seed 2026, verified identical design.json): baseline + 11 LHS points.
- Arm A (`stage2_arm_A.json`, `arm: success`): after the design, LogEI on a GP fitted to valid points only (-log J) +
  log P(success) from a GP classifier on all attempted points (solver/timeout/registration = fail), greedy q = 4 with
  Kriging-believer fantasies over a 2^14 Sobol + 4000 local-perturbation pool. Failures do not enter the objective GP.
- Arm B (`stage2_arm_B.json`, `arm: impute`): the unchanged finished-campaign rule (qLogEI, imputed failures).
- Code: `calibration/bo/run_joint_run6_run10.py` (optional `JOINT_SPEC`; no spec reproduces the finished campaign
  exactly), `run_joint_stage2_{A,B}.sbatch`; outputs `run/joint_stage2/arm_{A,B}`. Two independent jobs, run concurrently.
- Compare: best joint J and per-run RMS, failure fraction, evaluations to reach J <= 1.20 mm. One replicate per arm;
  differences are exploratory unless large; a negative or inconclusive result is valid.
- Not authorized: parameter freezing, Run 9 release, any change to box/rule after launch. Stage 2 is not a replacement
  of the finished campaign; its best point is a candidate only.

## Review record and caveats (Astra `astra-code-reviewer`, openai-codex/gpt-6-astra)

Reviewed: driver `calibration/bo/run_joint_run6_run10.py` SHA256 `af2dd959c2a69429e0631957a614f379f6258aa2e3a7abd9fc5032ad62513d43`
(reverse-applied predecessor = finished-campaign driver SHA `2c4310500a36a5d7ea80e431f469dfea9f0ebe4af501b8fa29eab1f8bebef8a5`),
`stage2_arm_{A,B}.json`, `run_joint_stage2_{A,B}.sbatch`, no-solver smoke `run/joint_stage2/smoke/smoke.json`. Code/specs frozen at launch.
1. Comparison = two complete proposal policies (A: valid-only GP + GPC, pool + Kriging believer; B: joint qLogEI with imputation), not
   the classifier alone. Report best-J at common completed-wave counts through the shorter campaign; final outcomes separately;
   unreached J <= 1.20 mm thresholds are censored; unequal early stopping is reported, not adjusted. One replicate each: exploratory.
2. Classifier warnings are suppressed and kernel/probabilities are not logged; a flat classifier would silently remove discrimination.
   Code is frozen, so diagnostics (probability range, fitted kernel) are recomputed after the run from saved results with the same seeds.
3. One controller per arm; crash recovery not exercised (interrupted evaluations/partial proposals fail closed).
4. The replay supports failure predictability only, not optimization superiority.
