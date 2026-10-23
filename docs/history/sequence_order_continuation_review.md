# Independent pre-execution review — sequence-order continuation

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers identified.**

Scope is **one fresh `ab_block` adaptive-load diagnostic**, starting flat and accumulating A×10 then B×10 before continuation. Output is restricted to:

`run/sequence_order/ab_block_final_only_continuation_v1`

This is **not adaptive optimizer handoff**, a strict one-solve lumped baseline, or permission to restart any other case.

## Authority and identity scope

Current authority inspected:

- `.pi/goals/active_goal_2026101910270493_mvfeqwh7-k4y0nd.md`
- `docs/reports/sequence_ablation_v2_spec.md`, **Current adaptive-load diagnostic — researcher authorized**
- `run/sequence_order/checks/frozen_design.json`
- `run/sequence_order/ab_block_final_only_base/execution.json` and `sequence.json`

The explicit new selection supersedes the original no-continuation restriction **only for this diagnostic**. Historical dome reviews supply supporting evidence, not authorization or inherited budgets.

### Bound SHA256 identities

| Artifact | SHA256 |
|---|---|
| `scripts/run_sequence_order_continuation.py` | `57c35b4b7c3d80bf066ebffe7c0580c06ef77c44068c77067f11d62e4fed8151` |
| `calibration/tests/test_sequence_order_continuation.py` | `b13bc2324b1caa8e8d433c596c80ae7e808db5d8181b3f4bcdf9afad3fc3ac28` |
| `scripts/run_dome_continuation.py` | `b723c753ca3160c28008b87dfde471a1694c785ef0f9cd2a4f5664b6da79248f` |
| `run/sequence_order/ab_block_final_only_base/sequence.json` | `3e1888be9be10de1d2aaab2612d1ae4c6b7a94b3db54eaaab79e9e3576c9d5a0` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `b7d331d05e614b1ddc44d90f5decd7f755d91434f4202e233f381ab09feb6352` |
| `src/libshell/NativeHybridShellSolver.hpp` | `d71f60390640edf3c1e5335d3cf481e042dfba6eabed5165a2c8d3ecbf6aaf4e` |
| `src/libshell/ReferenceHybrid/Core.hpp` | `f3265448c4d2d210659fab2483d1bf9881852128d1435d7f490e5bc4ae301ebd` |

These are **supplied or inspected manifest identities, not independently recomputed hashes**. Native identities agree across the frozen design and `run/dome/checks/repair028_identity.sha256`. Relevant source was inspected directly; unchanged numerical evidence is reused only within those identities. Unrelated checkout changes are outside scope.

## Checks performed

### Input, model and option matching

The archived input contains twenty enabled, single-repeat applications: A_01–A_10 followed by B_11–B_20. Recipes, inherited profiles, material-frame positions, multiplicative accumulation and face-shared Voce hardening are preserved.

`scripts/run_sequence_order_continuation.py:15–39` changes only:

- Adaptive loading enabled.
- Initial/max step `0.1`; minimum `0.015625`; growth `2`; maximum retries `8`.
- Native allowance `600 → 1800` seconds per substep.

The inspected inherited options retain the 200×300mm sheet, 0.77mm thickness, resolution `.01`, free boundary conditions, zero slope feedback, repaired linear solves, fixed2000 handoff, both gradient gates `5e-14`, and disabled seed escape. Native material remains E=1, ν=.33.

The runner copies sequence bytes, verifies binary/input identities against the archived receipt, refuses an existing output directory, and records the new command/options/provenance. The proposed output directory was absent from the inspected listing.

### Actual continuation and rollback semantics

`src/simulations/Sim_Bilayer_Growth.cpp:2505–2685` accumulates the complete ordered target before the sole terminal physical-cycle continuation block. For each layer,

\[
\bar A(\lambda)=\bar A_0+\lambda(\bar A_{20}-\bar A_0).
\]

The start is the **initial stress-free metric**, not the metric preceding application20. This is an affine metric homotopy—not fractional applications or proportional engineering strain. Convex interpolation preserves SPD between SPD endpoints. The reviewed positive growth factors and multiplicative updates support that endpoint premise; reference curvature is retained.

`src/simulations/Sim_Bilayer_Growth.cpp:2742–2839` confirms:

- Each trial starts from the last accepted vertex/director DOFs.
- Returned rejection restores those DOFs and accepted-load metrics, then halves the step down to the minimum.
- Success advances accepted load and doubles the step up to `.1`.
- Retry count resets after success; `8` limits consecutive retries, not total solves.
- At exact lambda1, the **exact accumulated target arrays are installed before solving and qualifying the endpoint**.
- Growth/hardening is not replayed during substeps. Final accumulated hit counts are restored on completion.

Each invocation creates a fresh native solver session. Fixed2000 remains the handoff rule, although an already-converged substep can terminate earlier. No optimizer-memory transfer across target changes is introduced.

Returned native failures can reach rollback; exceptions or process termination can instead stop the run. This is not blanket exception recovery.

### Full-load acceptance

`scripts/run_dome_continuation.py:19–59` checks:

- Completed20-cycle, final-only result with one completed physical-cycle solve block.
- Accumulation rows1–20 and terminal-only solve cadence.
- Equal, ordered convergence/proof stream lengths.
- Trial loads advancing from the last **accepted** load.
- Every accepted substep satisfying `gradient_target`, `positive_pivots`, and nonnegative gradient ≤ reported tolerance ≤ `5e-14`.
- Exact terminal lambda1.
- Agreement among terminal proof, sequence result and exported-state proof, with a final mesh present.

Native endpoint verification freshly evaluates energy/gradient and checks the gauge-reduced Hessian. Failed/capped optimizer status cannot become acceptance merely because a retained candidate looks satisfactory. Production arithmetic remains double precision; no tolerance weakening is introduced.

Native export occurs only for an accepted terminal full-load endpoint. Partial-load solves therefore do not collide with the exclusive final-state export. `equilibrium_solves=1` counts a continuation block, **not one optimizer invocation**.

### Resources and stopping

`scripts/run_sequence_order_continuation.py:30–67` requires a Slurm allocation with at least8 CPUs, sets OpenMP8/thread-limit8 and BLAS1, runs one direct subprocess with a3600s timeout, and records failure on exceptions. There is no outer retry or campaign dispatch.

The proposed **70min,8CPU,32GB** allocation leaves600s beyond the solver subprocess ceiling. Native1800s timing is cooperative and resets per substep; it is not the total diagnostic limit.

The parent’s explicit-cwd, DHD-Python `sbatch --wrap` submission remains responsible for allocation and one-case scope. No live scheduler or timeout-enforcement test was performed.

## Evidence and limitations

Directly inspected:

- `run/sequence_order/checks/continuation_unit_tests.log`: **2 tests passed in0.130s**.
- Both new tests: option-delta/byte-copy/exclusive-output wiring and fail-closed timeout handling. Solver and verifier are mocked.
- `calibration/tests/test_dome_continuation.py`: retained verifier fixtures cover rejected-trial progression, partial-endpoint rejection and proof-row mismatch.
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 tests passed** for unchanged native implementation.
- `run/dome/repair_attempt028_ramp60_v1/diagnostic_execution.json`, convergence CSV and result: same recorded binary, matched continuation controls, eleven accepted substeps, exact terminal1, gradient `1.319921482080224e-15`, elapsed3228.497s.
- Original600s and1800s lumped proofs: both `time_cap`, indefinite, unaccepted.
- Sequential convergence CSV:13 accepted stages followed by failed stage14.

The retained ramp demonstrates successful repeated-substep wiring on a **different input**, including a tiny eleventh increment caused by floating-point accumulation. It does not establish convergence here or dynamically exercise rejected-step rollback.

No commands, new tests, builds, simulations, installations, hash recomputation, loaded-library audit or delegation were performed.

## Launch-blocking findings

**None for the inspected slice and stated contract.**

## Nonblocking caveats

1. **Preserve reviewed identities.**  
   `scripts/run_sequence_order_continuation.py:25–29,42–45` trusts the archived receipt’s options and records runner/verifier identities rather than enforcing this review’s hashes. Perform the ordinary prelaunch comparison of the reviewed files and source receipt; consequential changes require reassessment.

2. **Report actual substep counts.**  
   `scripts/run_sequence_order_continuation.py:59–60` stores the terminal proof, not aggregate accepted/rejected counts. Those counts remain recoverable from the retained CSV/JSONL and should be reported with final target provenance. Do not describe this as one optimizer solve.

3. **Completion is not guaranteed.**  
   Repeated warmups, rejected trials or the floating-point terminal increment can consume the3600s ceiling. A timeout remains unresolved; it does not justify extension.

4. **No restartability claim.**  
   This general-sequence route has no exercised restart facility. Exported state is diagnostic, not a restart checkpoint. A killed wrapper may leave its manifest at `started`; that is not acceptance.

## Scientific decision and approval boundary

- **Accepted full-load endpoint:** an accepted candidate was reached under this continuation recipe. Audit and retain its target/history provenance and all substeps.
- **Partial load, rejection, timeout or inconsistent evidence:** endpoint unresolved; preserve evidence and stop.

Success would not isolate load stepping from changed cumulative work, repeated warmups or branch selection. It cannot establish a matched cadence contrast because the strict lumped attempts and20-stage sequential endpoint remain unaccepted/incomplete.

No physical sequence-memory, global-minimum, mesh-robustness or general lumping conclusion follows. Approval covers only this diagnostic and grants no execution permission, resource extension, other-case restart or campaign resumption.