# Run 2 BLIS diagnostic — repair re-review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified.** The command-order repair addresses the pre-solver failure without changing the authorized scientific comparison.

Approval covers only the frozen `a0084` Run 2 flat-start pair, **BLIS unset → BLIS=1**, under `run/diagnostics/run2_blis_a0084_r1/`. It is not permission to resume the paused six-parameter goal, retry an executed scientific arm, calibrate, score scans or release a test.

## Scope, authority and identities

Root: `/gpfs/projects/p33082/PSM2025-zigzag`.

Read:
- `/gpfs/projects/p33082/AGENTS.md` and project `AGENTS.md`.
- `docs/calibration/bayesian_calibration/convention_corrected_investigation.md`, particularly the diagnostic protocol, preparation, review and repair at lines **168–178**.
- `docs/calibration/bayesian_calibration/run2_blis_preexecution_review.md`.
- Current driver, tests, batch script and candidate plan; their preserved predecessors under `run/diagnostics/run2_blis_a0084/`.

The following are **supplied/recorded identities inspected during review, not independently recomputed hashes**:

| Artifact | SHA-256 |
|---|---|
| `calibration/analysis/run2_blis_diagnostic.py` | `d96eb190042ada96ff1dd736852a2c047d4fca36a27e190a7d8362304f122ee0` |
| `calibration/tests/test_run2_blis_diagnostic.py` | `4025d0a33841f7a3ebbaa119aeeb543b21ce86c6879b4594ee3edc3a9e7a4212` |
| `calibration/bo/run2_blis_diagnostic.sbatch` | `288acf43913fd7908e8d30e6e800f7ae77b072d0741299a1417a1c457e67ded8` |
| `run/diagnostics/run2_blis_a0084_r1/plan.json` | `a88e95b8e62cd7d5aa0e048dda0daf94d5a12a22d30f27b75daa708e1b783397` |
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |

The predecessor plan is `0825b5a2a5d912de64d567a05b1817ba6f70ada7ad69c74845df49e283bf765a`. Its driver/test/batch identities begin `fdf7934c76c542fd`, `4530c6b87ec9196e` and `7cb06f8037af1c0c`, respectively.

Both plans record identical remaining input/library identities, including:
- Frozen candidate `run/fresh_lhs_joint4/results/a0084.json`: `1fd2db5d0ac66883…`.
- Original sequence: `f3c75b8c9f7b93d1…`; original flat mesh: `e470b59773859c42…`.
- Reference history: `9ddbf4365a9d5a09…`.
- Initial-state helper: `bfe1a3bb2baa552b…`; runner: `4b93b4513a5d4dbc…`; configuration builder: `8821a2c6d9e07b07…`; adaptive options: `94252b22fd97d165…`.

Full paths, hashes, selected library identities and Python environment are retained in `run/diagnostics/run2_blis_a0084_r1/plan.json:288–339`. Prior evidence is reused only for these unchanged components.

## Checks and findings

### Launch-blocking findings

**None. The former prelaunch operational defect is resolved.**

1. **Repair preserves exact command order.**  
   The preserved implementation at `run/diagnostics/run2_blis_a0084/reviewed_sources/run2_blis_diagnostic.py:88–92` returned the JSON-loaded dictionary. Serialization sorted its keys, whereas `calibration/ewcal/runner.py:85–98` emits arguments in dictionary insertion order.

   The repaired `calibration/analysis/run2_blis_diagnostic.py:88–93` reconstructs the entire plan, compares it with the saved plan, then returns the reconstruction. Dictionary equality checks contents; the command-template list comparison remains order-sensitive. The exact prelaunch argument comparison remains intact at `calibration/analysis/run2_blis_diagnostic.py:161–165`.

2. **Original reviewed template retained.**  
   Manual comparison of `run/diagnostics/run2_blis_a0084/plan.json:2–93` and `run/diagnostics/run2_blis_a0084_r1/plan.json:2–93` found the same ordered command except the sequence pathname. Configuration differences are confined to output/report paths and the three repaired source identities. Frozen parameters, sequence, libraries and Python match.

   This agrees with the retained exact-check attestation in `run/diagnostics/run2_blis_a0084/repair_plan_comparison.log:1`.

3. **Regression evidence now reaches the previously missed boundary.**  
   `calibration/tests/test_run2_blis_diagnostic.py:17–44` prepares and reloads a serialized plan, then compares both regenerated commands against the verified persisted template—not merely against each other.  
   `calibration/tests/test_run2_blis_diagnostic.py:64–95` now exercises real preparation, verification and arm execution with mocked launches for both timeout arms and an unknown failure. It no longer bypasses verification in these cases.

4. **Failure occurred before either scientific arm executed.**  
   `run/diagnostics/run2_blis_a0084/slurm-5655708.out` records the command guard raising before `launch`. Directory inspection found only `unset/sequence.json`, no solver log/result, and no second arm. The new root contains only `plan.json`. These observations support infrastructure repair before execution, not repetition of a consumed pair.

### Unchanged scientific and operational controls

- **Science:** The same frozen six values are passed directly, including `gbot=0`; no legacy codec clipping is introduced. The sequence retains 33 repetitions, corrected signed-angle geometry, identical loading/hardening and flat start. See `calibration/analysis/run2_blis_diagnostic.py:51–76` and `calibration/analysis/run2_initial_state_diagnostic.py:49–80`.
- **Acceptance:** Final-only equilibrium, zero slope feedback, native gradient gate `5e-14`, finite energy and independent positive-pivot verification remain unchanged. Acceptance tracing was checked in `calibration/ewcal/runner.py:169–210`; no solver mathematics or scoring rule changed.
- **Resources/order:** `calibration/bo/run2_blis_diagnostic.sbatch:3–23` retains one node, eight CPUs, 16 GB, 02:30 walltime, disabled core dumps and sequential arms within one core-bound Slurm step. Only its output pathname changed.
- **Threads/matching:** OMP=8, OMP_DYNAMIC=FALSE, OpenBLAS/MKL=1 and removal of per-loop BLIS overrides remain intact. Common environment, hostname, affinity, job and step comparisons remain at `calibration/analysis/run2_blis_diagnostic.py:96–119,147–154`.
- **Caps/failures:** Native 3600-second per-call and controller 3900-second limits remain. Recognized timeout/numerical failures permit the other arm; unknown failures and output/identity errors stop execution. Process-group cleanup remains in `calibration/ewcal/runner.py:100–146`.
- **Preservation:** Exclusive preparation and arm-directory creation prevent solver-output overwrites (`calibration/analysis/run2_blis_diagnostic.py:79–85,155–156`). The batch contains no restart loop.
- **Comparison/access:** Initial geometry, sequence/history, acceptance and terminal load/topology checks remain. Corresponding-vertex RMS and metre-to-millimetre conversion are unchanged at `calibration/analysis/run2_blis_diagnostic.py:220–239`. The execution path performs no scan scoring, calibration or candidate selection; retained target hashing/metadata are provenance checks.

## Evidence and limitations

Inspected `run/diagnostics/run2_blis_a0084/repair_no_solver_test.log:18–36`: **five tests passed in 8.742 seconds**, with both mocked timeout outcomes, unknown-failure stopping and repaired-plan verification. The supplied task and investigation record attest successful `py_compile` and `bash -n`; the retained log does not separately print those commands.

This review used read-only source, plan, log and directory inspection. **No commands/tests were executed by this reviewer**, no hashes independently recomputed, no solver launched, no submission made and no additional scan-score artifacts opened. Native successful execution and live thread/binding behavior remain untested by these fixtures. The lead’s stated immediate pre-submission hash recheck remains necessary for identity continuity.

## Nonblocking caveats and claim boundaries

The prior caveats in `docs/calibration/bayesian_calibration/run2_blis_preexecution_review.md:75–89` remain:

1. Configured BLIS counts and sampled OS threads do not establish simultaneous active BLIS threads, oversubscription or live symbol binding.
2. Native limits apply per continuation call; controller termination can leave incomplete traces. Recorded elapsed time is not isolated factorization cost.
3. Native-frame vertex differences can include rigid drift; they are not scan discrepancy or proof of unique equilibrium.
4. One fixed-order pair remains vulnerable to cache, frequency and shared-node-load effects. A substantial difference supports only a case-specific setting effect; similarity or failure leaves historical timeout causation unresolved.

The original one-shot timeout remains the official outcome. Effective fitted parameters are not physically identified by this diagnostic. Approval is limited to the reviewed slice and identities; it grants no execution permission, additional attempts, budget extension or sealed-test release, and does not certify runtime enforcement.