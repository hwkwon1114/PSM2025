# Scoped runtime-only retry review

## Verdict: APPROVED_WITH_CAVEATS

**No scientific or operational launch blockers identified for this one retry.** Approval covers the reviewed implementation and planned safeguards; it does not itself authorize submission or additional retries.

## Scope and authority

Read:
- `.pi/goals/active_goal_2026101910270493_mvfeqwh7-k4y0nd.md`.
- Current protocol and **Authorized runtime-only restart amendment** in `docs/reports/sequence_ablation_v2_spec.md`.
- Unchanged baseline review, `docs/reports/sequence_order_preexecution_review.md`.
- New retry helper/tests, retained failed execution and native evidence, frozen identities, and reused acceptance verifier.

The amendment permits only restarting failed `ab_block` final-only job4480279 from the common flat start with native limit **1800 seconds**, unchanged model/tolerances/cadence, and separate output:

`run/sequence_order/ab_block_final_only_retry1800`

Running sequential4480278 remains untouched. No other scientific dispatch occurs until this retry passes. Historical protocol sections confer no additional authority.

### Reviewed identities

SHA256 values are supplied/recorded identities, **not independently recomputed by this reviewer**.

| Artifact | SHA256 |
|---|---|
| `scripts/retry_sequence_order_lumped.py` | `cc4a13733ba305ebdb38cc990e2cd0091d1d63730af5b515f5e360b0a0d44a35` |
| `calibration/tests/test_sequence_order_retry.py` | `17187eb40c838ff08dca95d0a16fad5a65cd84f86a803fa6e4299b71393bc446` |
| `scripts/run_sequence_order.py` | `45a835a91ef21b3610570bfaaddde839687d77a6db92d02e74cc5529eb59d9a0` |
| `scripts/run_sequence_order.sbatch` | `aecbafdf15123fa540a2b83c99068b5b47bf80fd7e88852756f2feadf8d842f1` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |
| `run/sequence_order/ab_block_final_only_base/sequence.json` | `3e1888be9be10de1d2aaab2612d1ae4c6b7a94b3db54eaaab79e9e3576c9d5a0` |

Reused identity packet: `run/sequence_order/checks/frozen_design.json`. Existing checkout changes outside this slice are neither reviewed nor approved here.

## Checks actually performed

### Failure and scientific relevance

Inspected `run/sequence_order/ab_block_final_only_base/execution.json`, native proof, application history and convergence CSV:

- Execution failed after **632.9269796162844 seconds**.
- Native termination: `time_cap`; gradient **1.2796230809660088e-12 > 5e-14**.
- Curvature classification: `indefinite`; `accepted=false`.
- Twenty applications were recorded, with one attempted full-load solve at cycle20, `lambda_from=0`, `lambda_trial=1`, acceptance0.
- No final accepted-state artifact was present in the directory listing.

This is an unresolved numerical solve, not an alternative accepted equilibrium or evidence against lumping. Extending runtime without changing acceptance directly addresses that limitation. Another failure remains unscored and requires consultation, not automatic replay.

### Runtime-only implementation

`scripts/retry_sequence_order_lumped.py:13–46`:
- Reads the fixed failed case and checks its failure classification, binary identity and sequence identity.
- Copies all prior options, changing only `-native_hybrid_seconds` from `600` to `1800`.
- Creates output exclusively and copies sequence bytes without regeneration.
- Launches the same binary in the new directory; no checkpoint or partial-state input is supplied.
- Records retry provenance, prior execution hash, helper/verifier identities and command.
- Uses eight OpenMP threads, single-threaded BLAS, and external timeout2070 seconds.
- Checks binary/input stability after execution and calls unchanged `verify()` before marking acceptance.

The original directory is read-only to this helper. The intended retry directory was absent at inspection.

The targeted native binding at `src/simulations/Sim_Bilayer_Growth.cpp:1293–1314` assigns the changed value to the time limit; gradient gates and fixed handoff remain separately configured. No solver mathematics or production dtype changes are introduced.

### Acceptance and evidence reuse

`scripts/run_sequence_order.py:74–118` still requires completed execution, exact cadence, matching proof streams, accepted gradient-target termination, positive pivots, full-load convergence, application history and endpoint artifacts. Thus additional runtime does not weaken acceptance or add equilibrium stages.

`run/sequence_order/checks/retry_unit_tests.log` reports **two tests passed**. Inspection confirms subprocess execution and verification are mocked: tests cover the sole option change, copied sequence identity, exclusive output and timeout rejection. They are wiring evidence—not solver, convergence or restart evidence.

Unchanged numerical/initialization evidence is reused from the baseline review with its stated limitations. This is a fresh solve, not demonstrated checkpoint resumability.

## Launch-blocking findings

**None for the specified retry.**

## Nonblocking caveats and limits

1. **Submission safeguards remain procedural.**  
   The planned DHD-Python invocation with explicit project working directory, **40-minute Slurm allocation,8CPU/32GB**, accommodates the2070-second external timeout with330 seconds remaining. The actual `sbatch --wrap` submission and live allocation were not inspected. The helper does not enforce Slurm memory/wall time, concurrency or campaign stopping.

2. **Preserve identity and normal Python execution.**  
   `scripts/retry_sequence_order_lumped.py:16–22` uses assertions for preflight checks; do not disable them with `-O` or `PYTHONOPTIMIZE`. Frozen identities still require parent preflight confirmation. Explicit exceptions would improve future reuse but are not a launch requirement for the declared invocation.

3. **Analysis adaptation remains outstanding—not launch-blocking.**  
   `calibration/analysis/analyze_sequence_order.py:119–148` currently selects canonical base directories and rejects additional option differences. After acceptance, explicitly select the approved retry path and permit only this declared runtime exception, alongside the existing cadence difference. Preserve all other matching checks and the original failed record. Do not silently replace it or broadly relax option comparison.

4. **Scientific claims remain narrow.**  
   Accepted endpoints may support the model-conditional cadence comparison only after objective/history matching and the frozen numerical-sensitivity assessment. Unequal runtime allowances must be disclosed; this is not an equal-budget efficiency comparison. Neither acceptance nor positive pivots establishes global optimality, physical temporal memory or general failure of lumping. This amendment does not raise limits for other cases or sensitivity runs.

No tests, simulations, builds, installations, hash computations or scheduler operations were executed by this reviewer. No source/data files were edited. Native timing remains cooperative; approval makes no claim of runtime enforcement.