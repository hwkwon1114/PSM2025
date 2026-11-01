# Run 2 BLIS diagnostic — pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified.**

Approval covers only the frozen `a0084`, flat-start, **BLIS unset → BLIS=1** ordered pair described in `docs/calibration/bayesian_calibration/convention_corrected_investigation.md:168–172`. It does not resume the historical six-parameter goal or authorize calibration, retries, scan scoring, candidate selection, or unseen validation.

## Reviewed identities

Paths are relative to `/gpfs/projects/p33082/PSM2025-zigzag`. These are **inspected recorded identities, not independently recomputed hashes**. Full input/library hashes are in `run/diagnostics/run2_blis_a0084/plan.json`.

| Artifact | SHA-256 prefix |
|---|---|
| `calibration/analysis/run2_blis_diagnostic.py` | `fdf7934c76c542fd` |
| `calibration/bo/run2_blis_diagnostic.sbatch` | `7cb06f8037af1c0c` |
| `calibration/tests/test_run2_blis_diagnostic.py` | `4530c6b87ec9196e` |
| `calibration/analysis/run2_initial_state_diagnostic.py` | `bfe1a3bb2baa552b` |
| `calibration/ewcal/runner.py` | `4b93b4513a5d4dbc` |
| `calibration/bo/run_convention_corrected.py` | `8821a2c6d9e07b07` |
| `shellsim/configs/default_adaptive_cutback.json` | `94252b22fd97d165` |
| `run/diagnostics/run2_blis_a0084/plan.json` | `0825b5a2a5d912de` |
| `bin/shell` | `522633cff2d7ff14` |

The plan pins the exact config, sequence, command template, Python environment and these data identities:

- Frozen candidate: `run/fresh_lhs_joint4/results/a0084.json` — `1fd2db5d0ac66883…`.
- Original sequence: `run/run9_interim/fresh_lhs_joint4_a0084/evaluations/attempt_0001/case_00/sequence.json` — `f3c75b8c9f7b93d1…`.
- Original flat mesh in that directory: `bilayer_zigzag_sequence_cycle_000_initial.vtp` — `e470b59773859c42…`.
- Reference history: `run/diagnostics/run2_initial_state_a0084/flat/evaluations/attempt_0001/case_00/sequence_history.csv` — `9ddbf4365a9d5a09…`.
- Retained target provenance: `run/putong_convention_check/preserved_samples/run9.npz` / `run9.json` — `aa1fc209fa1b1eee…` / `297b0f5b29e44016…`. These are not scored.

Selected resolved libraries are BLIS `9d061aa33f5d685a…`, libgomp `f30347f1b859a299…`, OpenBLAS `60ddaebfbdae101e…`, and TBB `4f266c979da1fb90…`; their exact resolved paths are recorded in the plan.

Authority reviewed: project `AGENTS.md`, the supplied task, and the final diagnostic/preparation sections of the investigation document. Historical approvals were used only as evidence for unchanged components.

## Checks performed

### Scientific comparison and numerics

- `calibration/analysis/run2_blis_diagnostic.py:51–76` bypasses the legacy positive-`gbot` codec and inserts frozen values directly. Both arms retain:
  `gtop=4.420473934455598e-05`, `gbot=0`, `ortho_top=-0.3779292572289705`, `ortho_bottom=0`, `beta=0.010031322617850377`, `floor=0.45599149353802204`.
- Manually compared the prepared sequence with the original sequence: 33 repetitions, 32 strips, −1° angle, 187.5 mm length, 2 mm width, material shift `[0,28.75]`, disabled boundary conditions and unchanged hardening.
- Source tracing confirms \(q(n)=f+(1-f)e^{-\beta n}\), zero bottom growth, final-only equilibrium and zero slope feedback: `src/libshell/ZigZagSequenceGrowth.hpp:653–669` and `src/simulations/Sim_Bilayer_Growth.cpp:2535–2659`. Neither arm imports an alternative starting state.
- Acceptance retains finite energy, gradient ≤`5e-14`, independent verification and positive pivots through `calibration/ewcal/runner.py:169–265`. No new loss, scoring rule or solver mathematics is introduced.
- No scan-scoring/evaluator call occurs in the diagnostic execution path.

### Thread control and resources

- `calibration/bo/run2_blis_diagnostic.sbatch:3–23` requests one node, eight CPUs, 16 GB and 02:30 walltime, disables core dumps, and runs both arms sequentially within one core-bound Slurm step.
- `calibration/analysis/run2_blis_diagnostic.py:95–118,146–154` checks the intended BLIS setting and compares common-environment hash, hostname, affinity, job and step. Per-loop BLIS overrides are removed.
- `calibration/ewcal/runner.py:100–107` inherits the launching process environment; no BLIS-setting override was found in the inspected native source. Build/link evidence supports a real BLIS dependency, with static CHOLMOD and both BLIS/OpenBLAS dynamically present.
- Native TBB parallelism is limited to one during the hybrid solve (`src/libshell/NativeHybridShellSolver.hpp:367`). Potential BLIS/OpenMP contention is deliberately the tested condition, bounded by the allocation and controller—not assumed absent.

### Failure handling and output validity

- Exclusive preparation/arm directories prevent automatic solver reruns or overwriting existing arm outputs.
- `calibration/analysis/run2_blis_diagnostic.py:137–216` permits the second arm after recognized timeout/numerical failure. Unknown failures and output/identity errors raise; batch `set -e` stops the pair.
- Numerical-abort recognition reuses the matching reviewed implementation at `calibration/analysis/run2_initial_state_diagnostic.py:160–189`, requiring SIGABRT and matching terminal native/cycle-33 evidence.
- The 3900-second controller uses monotonic timing and process-group cleanup (`calibration/ewcal/runner.py:100–146`).
- Initial-mesh identity, saved sequence, complete accepted history, finite native positions and topology are checked. The inspected reference history has cycles 1–33, 6065 hit events per cycle and only cycle 33 marked solved.
- `calibration/analysis/run2_blis_diagnostic.py:219–238` requires exact terminal reference/topology/material/target-form agreement before comparing corresponding vertices. The metre-to-millimetre conversion and vertex RMS calculation are correct. Failed arms receive no accepted energy or geometry comparison.

### Evidence and testing limits

Inspected:

- All five fixtures in `calibration/tests/test_run2_blis_diagnostic.py`: frozen configuration/overwrite protection, separate-process BLIS settings, timeout continuation versus unknown-failure stopping, affinity rejection, and matched-load geometry.
- Earlier numerical-classification fixtures and matching review in `docs/calibration/bayesian_calibration/run2_initial_state_astra_rereview.md`.
- `run/diagnostics/run2_blis_a0084/no_solver_checks_harness_excerpt.txt`: lead-attested five-test pass, silent successful compilation/shell syntax checks, matching shell-style environment pair, and plan verification; Slurm 5653638 completed 0:0.
- `run/diagnostics/run2_blis_a0084/linkage_login_snapshot.txt`, `build_cmake/build.ninja:235–275`, and `docs/calibration/bayesian_calibration/convention_current_solver_argon_review.md`.

I performed read-only inspection, not command execution. No solver, job submission, additional scan-score inspection, rebuild, independent hash recomputation or live thread/binding measurement was performed.

## Nonblocking caveats

1. **Configured threads are not measured active threads.**  
   `calibration/analysis/run2_blis_diagnostic.py:95–134` measures parent-process BLIS configuration and samples solver OS threads/RSS/CPU time. It does not establish simultaneous BLIS activity, live BLAS symbol binding or oversubscription. The plan checks selected libraries, not every transitive dependency.

2. **Timing and trace limits.**  
   Native 3600 seconds applies per continuation call, with existing verification grace and non-preemptible work; the controller bounds the overall process. Traces are exported after native returns, so controller termination can leave incomplete traces. Report `solver_seconds` as observed process elapsed time, not isolated factorization time or an exact CPU accounting measure.

3. **Geometry is native-frame geometry.**  
   `calibration/analysis/run2_blis_diagnostic.py:232–237` reports unaligned corresponding-vertex differences, potentially including rigid drift. It is neither scan discrepancy nor proof of identical complete optimizer states, unique equilibrium or physical correctness.

4. **One ordered pair has narrow causal reach.**  
   Ordering, cache state, shared-node load and frequency remain possible influences. A large difference supports a case-specific setting effect; similarity or failure leaves the historical timeout unresolved. Common tracing/buffering and the 3900-second controller differ from the historical one-shot environment. The original timeout remains the official one-shot outcome.

Approval applies only to the reviewed identities and procedure. It grants no additional execution permission, retries, budget extension or test release, and does not certify runtime enforcement.