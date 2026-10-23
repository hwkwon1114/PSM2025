# Attempt028 failed-solve diagnostic — independent pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved launch blockers for this one diagnostic collection.** Approval is limited to the reviewed rebuilt executable, archived **stage3 attempt028**, unchanged physical inputs/tolerances, and fresh output directory. It does not establish an identical historical trajectory or authorize BO continuation, retries, solver changes, or dome qualification.

## Scope and authority

Read-only review under `/gpfs/projects/p33082/PSM2025-zigzag`.

Authority inspected:

- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — version 3, revision 2536 as read.
- `docs/reports/dome_toolpath_investigation.md:204–218`, particularly **Authorized single-failure diagnostic**.
- Current supplied researcher approval for one failed-design diagnostic.

Reviewed slice:

- `src/simulations/Sim_Bilayer_Growth.cpp:1289–1341`: native options and new trace export.
- `scripts/diagnose_dome_attempt028.py`: complete runner.
- Connected native trace/failure handling, acceptance, factorization and sequence exception paths.
- Archived attempt028 input, execution receipt, proof, convergence row and solver log.
- Isolated build, retained tests, trace fixture and available original-build provenance.

Unrelated dirty work was inspected only where relevant to this route and was preserved.

### Reviewed identities

Recorded SHA-256 values from `run/dome/checks/diagnostic028_identity.sha256`:

| Artifact | SHA-256 |
|---|---|
| `scripts/diagnose_dome_attempt028.py` | `ce1140f622d33bda073a9b832d896a0609d8f1a450ca29acbfac3d7d461a556a` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `087810e3695d99b1559c74c429b91e1a0edd31ff63b5ac1e6b6dfb9e5c580465` |
| `run/native_shape_refinement/build_4371653/bin/shell` | `89f9ed0a8d904ee7cc24328792f049b5a3468826ba03c79ab49ad3b7d3d1b5a1` |
| `run/dome/bo_stage3/attempt_028/solve/sequence.json` | `41d546d73d8b285cbf42eb7fffc73e314d423b825fa003aed123aae4dc19f31c` |
| `run/dome/bo_stage3/attempt_028/solve/execution.json` | `9f67061ed0786d5f17ea8139800e559644f5679eaea2daa4a41a90c41e70a6e8` |

Original executable identity recorded in the archived execution:
`11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c`.

These manifests were inspected, **not independently rehashed** during this review.

Reviewed proposed command:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/diagnose_dome_attempt028.py
```

Output: `run/dome/diagnostic_attempt028_v1`, absent in the inspected directory listing.

## Checks and conclusions

### Failure trace survives the expected rejection

The export at `src/simulations/Sim_Bilayer_Growth.cpp:1327–1340`:

- Is opt-in and occurs after `solveNativeHybridShell` returns.
- Serializes existing proof and `report.terminal.trace`.
- Makes no additional oracle call or optimizer update.
- Explicitly closes and checks the trace stream before returning to acceptance handling.

`src/libshell/ReferenceHybrid/Core.hpp:680–685` preserves the failed iteration row after catching the numerical stop. The sequence rejection occurs subsequently at `src/simulations/Sim_Bilayer_Growth.cpp:2812–2823`. Thus the expected unaccepted-solve exception/SIGABRT follows a closed trace file.

This does **not** guarantee a trace after an earlier process crash, hard kill, verification exception or filesystem failure.

### Recipe and execution remain bounded

The runner copies the archived command and changes only its executable, cycle-file location and added trace flag. Input copying is byte-preserving.

The archived recipe retains:

- Three ordered toolpaths, repeats **4/21/21**, disabled boundary conditions.
- Calibrated growth/orthotropy and face-shared hardening.
- `final_only`, adaptive continuation false, slope feedback zero, multiplicative updates.
- Fixed handoff **2000**, gradient tolerance **5e−14**, native limit **600 seconds**.
- Eight Hessian threads, no seed escape or checkpointed clamp/release route.

The runner requires an eight-CPU Slurm allocation, refuses existing output, caps threads and uses a **1500-second subprocess timeout**. It contains one subprocess launch and no BO scoring, proposal or retry path. The native time cap remains cooperative, with existing verification overhead; the process and Slurm limits provide outer safeguards.

### Numerical interpretation

The retained proof reports:

- `linear_solve_failed`, acceptance false.
- 2000 warmup and 16 Newton attempts.
- Warmup **143.578 s**, Newton **53.683 s**.
- Verified gradient **1.4485578433238772e−10**, above **5e−14**.
- `positive_pivots`; this is not successful linear-solve certification.

The unchanged CPU-double kernel computes, for the reduced shifted system,

\[
r=\frac{\lVert By+b\rVert_2}{\max(\lVert b\rVert_2,10^{-300})},
\]

and rejects failed/nonfinite solves or \(r>10^{-8}\)
(`src/libshell/ReferenceHybrid/Core.hpp:503–506`).

The exported residual is therefore a **linear-system diagnostic**, not the physical equilibrium gradient or a condition-number estimate. Factorization trial shifts and pivot-gate outcomes are retained. No tolerance change is introduced.

## Launch-blocking findings

**None for the narrowly scoped collection.**

## Nonblocking caveats and claim boundaries

### 1. Toolchain drift prevents an “instrumentation-only identical binary” claim

Available original-build-tree evidence shows:

- `build_cmake/CMakeCache.txt:114–117` and `build_cmake/build.ninja:264–269`: **SuiteSparse 4.4.2 static CHOLMOD**.
- New isolated build cache/log: **SuiteSparse 7.7 shared CHOLMOD**.
- Different global compiler/alignment/linker flags; the new build explicitly sets Eigen alignment to 64 bytes.
- GNU **12.4.0** appears in both compiler records.

The original build tree is generated, mutable evidence—not a complete frozen dependency attestation of the archived executable. Nevertheless, it demonstrates a concrete provenance difference that must not be omitted.

Native solver/Core/factorization hashes in the new `source.sha256` match the prior original-binary identity packet at `run/two_gripper_comparison/checks/final_only_preparation/preexecution_identity.json`. This supports source-level continuity, **not numerical trajectory equivalence**.

**Interpretation:** label the result a fixed-input diagnostic using the rebuilt toolchain. A changed termination may reflect library, compilation, hardware or execution differences—not merely nondeterminism, and not proof that the historical failure disappeared.

### 2. Trace evidence cannot identify every root cause

At `src/libshell/ReferenceHybrid/Core.hpp:501–506`, the trace records residual but not every solve-status/direction-finiteness predicate separately. Nonfinite JSON values may become `null`; missing/null is not zero.

A large finite residual identifies an observed residual-gate violation, but does not alone identify conditioning, assembly error or backend accuracy as its cause. CHOLMOD exceptions can instead surface as `oracle_exception`.

Also, `src/libshell/NativeHybridShellSolver.hpp:339–356` verifies a selected state, potentially `bestX`, after failure. **Proof gradient/curvature need not describe the failed Newton base state.** Use the failed row’s base quantities and factorization trials for that comparison.

### 3. Collection status is deliberately weaker than successful execution

`scripts/diagnose_dome_attempt028.py:36–42` records any returned solver exit and marks collection after parsing the trace. It does not restrict nonzero exits to the anticipated SIGABRT or certify complete sequence output.

Therefore inspect return code, log, proof and terminal row together. `diagnostic_collected` means evidence was collected—not accepted equilibrium, clean process completion or a BO observation.

### 4. One reproduction has a limited decision role

- **Same failure with informative residual:** localizes the numerical rejection in this execution; does not establish the historical root cause or a remedy.
- **Different failure or acceptance:** report nonreproduction and toolchain limitations; do not automatically retry or resume BO.
- **Missing/incomplete trace:** report diagnostic failure and stop.
- **Any outcome:** no inference that failed designs have poor dome geometry, no physical parameter identification, and no permission to relax acceptance.

This is a model-conditional numerical diagnostic supporting the dome investigation, not physical validation or restart evidence.

## Evidence reviewed and limitations

Retained evidence inspected:

- `run/native_shape_refinement/build_4371653/{build.log,environment.txt,source.sha256,source_check.txt,binary.sha256,complete.txt,exit_code.txt}`: isolated offline build, source checks recorded OK, exit zero.
- `run/native_shape_refinement/build_4371653/native_fixtures.log`: **27 passing native tests, 85 ms**.
- `run/dome/trace_fixture_v1/`: actual exported trace, argument log and proof; **seven trace rows**, accepted endpoint, finite Newton residuals.
- Archived attempt028 `execution.json`, `sequence.json`, proof JSONL, `sequence_convergence.csv`, and `solver.log`.
- Original-build identity records and generated compiler/link configuration.

The tiny integration fixture exercises successful export, **not the full failure/SIGABRT route**; preservation on that route was checked by source control flow. The fixture’s job/binary association relies on supplied provenance rather than an independently verified fixture execution receipt.

No commands, builds, tests, simulations, independent checksums or runtime-loader audits were executed during this review. Approval applies only to the stated slice and identities; it grants no execution permission, budget extension or sealed-test release and claims no runtime enforcement.