# Run 2 initial-state diagnostic — independent review

## Verdict: `CHANGES_REQUESTED` (`NEEDS_FIX`)

**One launch-blocking finding:** recognized numerical failures can incorrectly stop the pair before the second start. The proposed position-only intervention otherwise preserves the intended mechanical problem in the inspected source.

Read-only review: no edits, jobs, solver execution or delegation.

## Scope and authority

Authority: `docs/calibration/bayesian_calibration/convention_corrected_investigation.md:136–142`, specifically the initialization/timing check, separately authorized diagnostic and preparation sections. Earlier calibration/validation permissions were not reused. The historical six-parameter goal remains paused.

Reviewed:
- `calibration/analysis/run2_initial_state_diagnostic.py`
- `calibration/tests/test_run2_initial_state_diagnostic.py`
- `calibration/bo/run2_initial_state_diagnostic.sbatch`
- Prepared manifest and seed CSV; selected fresh `a0084`; historical Run 2 sequence, command, history and termination evidence.
- Relevant prescribed-state, growth, continuation and export paths in `src/simulations/Sim_Bilayer_Growth.cpp`; connected growth/native-acceptance code.
- `calibration/ewcal/runner.py`, `geometry.py`, `material_frame.py` and configuration construction.

### Reviewed identity anchors

Full dependency hashes and both configurations are recorded in:

`run/diagnostics/run2_initial_state_a0084/inputs/manifest.json`

Its supplied SHA-256 is `4dfce23cdad8235fc01c79f53d40f77fdb3629e01f905e571655388301d4e10d`.

| Artifact | Recorded SHA-256 |
|---|---|
| Diagnostic script | `6c392d7a8c1d07f9202ed6c8df3dfdf89740c28eb95bd5a6d6c19282e2a69e48` |
| Diagnostic tests | `f15d3c055e1309425421a9d63dc5e650234a3da39e47dbb9fce5dacac448414c` |
| Batch script | `6875e3b4e4b40b46f24e705180fcc094f285a664d3d37ac232266cf41c08124f` |
| `bin/shell` | `522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841` |
| Seed native state | `ad6a635b628017395b8b6f5557546528ebd6116e5c9cb8b41cee6ed1c35b9601` |
| Seed displacement CSV | `d5c39af5c5aead8a5b2c1d5aac0420ba139ebfb66f00927d88cdb196ec45a0a3` |
| Selected `a0084.json` | `1fd2db5d0ac66883bb71a8375a5d702eb4d9d5d3cc1d186a55151bc4b02884db` |

These are **inspected recorded identities, not independently recomputed hashes**. Historical source/build linkage evidence was inspected in `docs/calibration/bayesian_calibration/convention_current_build_evidence.txt`; it supports the unchanged binary but is not a prescribed-start integration test.

## Launch-blocking finding

### B1 — Scientific/protocol blocker: numerical-failure handling can censor the comparison

**Locations:**  
`calibration/analysis/run2_initial_state_diagnostic.py:218–225`  
`calibration/ewcal/runner.py:46–68,548–558`

The diagnostic continues after `failure_class='solver'`, but the runner recognizes only the narrowly specified native **time-cap** abort. A non-time-cap equilibrium-corrector abort remains `infrastructure`, triggering the diagnostic’s stop before the seeded start.

Static execution path:

1. Native report ends with, for example, `status=indefinite_curvature`, followed by the final equilibrium-corrector exception.
2. `termination_evidence()` returns `numerical=False` because status is not `time_cap`.
3. `Evaluator` records `failure_class='infrastructure'`.
4. Diagnostic line 225 raises instead of attempting the other start.

This is an exercised project failure mode, not a hypothetical signal interpretation:  
`run/joint_run6_run10/candidates/a0025/evaluations/attempt_0001/case_00/solver.log:9–11` contains the rejected native report and terminal `termination=indefinite_curvature` exception.

**Consequence:** a legitimate numerical outcome can prevent the authorized two-start comparison and misstate its cause.

**Smallest adequate repair:** add diagnostic-local, evidence-based classification before the stopping decision. Recognize documented numerical corrector failures using the final native report and matching terminal exception; preserve raw records. Unknown exits, incomplete signatures, identity failures and interruptions must still stop.

**Required bounded check:** extend the existing no-solver tests to exercise the driver/runner boundary:
- Recognized non-time-cap numerical failure and timeout proceed to the second start.
- Unknown abort/infrastructure failure stops.
- Invalid outcomes retain null diagnostic discrepancy, not the failure penalty.

Refresh reviewed script/test/manifest identities while preserving this frozen packet. No full solver test is needed to demonstrate this repair.

## Checks and conclusions

### Mechanical invariance — supported by source inspection

- The selected parameters agree with the original sequence, including  
  `beta=0.010031322617850377`, `floor=0.45599149353802204`, `gbot=0`, and the original 33-cycle Run 2 operation.
- `make_configs()` compares the generated physical sequence with the original input. The positive-growth codec is deliberately bypassed, preserving exact zero bottom growth.
- `src/simulations/Sim_Bilayer_Growth.cpp:2045–2143,2500–2557` applies the displacement to **current vertex coordinates** after cycle 32, before cycle 33, without resetting reference positions, target forms, history or edge-director DOFs. It adds no persistent displacement constraint.
- Material hit collection depends on material coordinates/connectivity. Current-shape mapping is diagnostic; with slope feedback zero, it does not alter growth increments.
- The update \( \bar a_{\mathrm{new}}=T^\top\bar aT \) preserves positive definiteness for the positive incremental stretches here. Bottom increments vanish at `gbot=0`.
- Final-only adaptive continuation uses the same initial and accumulated endpoint forms in both starts. Intermediate cutbacks may differ **because of initialization**; that is part of the numerical response, not a changed endpoint functional.

Thus this is a position-only initialization comparison, **not** a full-state restart or a comparison of different physical loading histories.

### Seed, scoring and acceptance

- Preparation checks accepted native proof, finite coordinates, matching topology and reference coordinates. Recorded dimensions are 15,251 vertices/30,000 faces; reference disagreement is `7.39e-9 m`, consistent with the float32 initial export.
- The seed’s companion `sequence_result.json` reports gradient `6.5217827107459496e-15`, positive pivots and one independent verification evaluation/Hessian.
- The seed is native model geometry, not scan coordinates. No seed optimization against the Run 2 score appears in this driver.
- The unchanged Run 2 target has 14,468 retained points. Both starts use identical material-reference/shared-footprint scoring.
- Geometry comparison uses corresponding vertices **after each solution’s material-frame leveling**. Array component 2 is internal height, hence public **y**, not public z.
- Native acceptance remains gradient ≤`5e-14` with independent positive-curvature verification. Invalid diagnostic summaries correctly use `objective_mm=null`; raw evaluator penalty values remain bookkeeping only.

### Execution and evidence preservation

- Output isolation and existing-directory rejection prohibit automatic rerun/overwrite. At inspection, the diagnostic directory contained only `inputs/`.
- One node, sequential execution, 8 CPUs/16 GB and 02:40 walltime are consistent with two 3900-second controller caps plus overhead.
- Controller timeout kills the process group and records termination evidence.
- Trace filenames include cycle/attempt suffixes, avoiding cutback overwrites.
- Final native state export occurs only for an accepted terminal endpoint.
- The historical a0084 failure remains a 3600-second **controller timeout**, not an accepted solve or numeric discrepancy.

## Nonblocking caveats and unperformed checks

1. **Timeout semantics:** native 3600 seconds applies per native continuation call, not cumulatively across cutbacks. The 3900-second controller bounds the whole solver process. Trace export occurs only after a native call returns; a blocked operation/controller kill can still leave no trace.
2. **History is not convergence evidence:** `sequence_history.csv` marks the scheduled final solve before it completes. Its `solved=1` row does not establish acceptance.
3. **Timing interpretation:** evaluator seconds include setup, seed exports and scoring. Separate these from native work counters where available. Fixed order and shared-node load remain confounders for small runtime differences.
4. **Threading improvement:** explicitly setting `BLIS_NUM_THREADS=1` in `calibration/bo/run2_initial_state_diagnostic.sbatch:13` would reduce uncertainty from the historically linked BLIS library.
5. **Testing limits:** I inspected the two supplied test implementations and the lead’s pass attestation. Available tools did not permit independent test execution, hash recomputation or full-array reconstruction. The tests do not exercise the compiled prescribed-start mechanism or driver failure policy. The existing post-run imposed-coordinate and target-form checks remain essential; no extra full solve is requested.
6. Physical fixture/face interpretation, model inadequacy, numerical nondeterminism and historical runtime causes remain unresolved.

## Scientific claim boundary

After B1 is repaired, this design can provide **case-specific evidence about these two initializations under this solver policy**:

- Two accepted endpoints: compare energies, leveled geometry and unchanged discrepancies only after load-identity checks pass.
- One accepted endpoint: report the other outcome as censored/nonconverged, without assigning it a shape error.
- Two failures: inconclusive about equilibrium or general initialization sensitivity.
- Any identity mismatch: invalidate the comparison.

A single ordered pair cannot establish failure probabilities, identify the cause of historical timeouts, prove physical parameter identifiability or constitute unseen validation. This review grants no execution permission, retry, budget extension, scope expansion or scientific-test release.