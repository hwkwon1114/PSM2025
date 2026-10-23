# Independent linear-repair review

## Verdict: APPROVED_WITH_CAVEATS

**No launch-blocking findings for ONE fixed-input attempt028 repair reproduction.** No code changes are required for this narrow test. This is not approval for BO continuation, retries, broader solver deployment, or changed tolerances.

## Scope and authority

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`
- `docs/reports/dome_toolpath_investigation.md`, particularly **Single-failure diagnostic result** and **Authorized numerical repair investigation**.

The approved question is whether bounded numerical repair satisfies the existing linear-residual gate and removes this reproduction’s rejection without weakening equilibrium acceptance. Calibration, input geometry, target accumulation, mesh, boundary conditions and scientific dome criteria remain fixed.

Reviewed the connected Config → HybridOptions → CLI → Newton/factorization → handoff → final acceptance paths, new tests, retained build/integration evidence, and `scripts/test_dome_attempt028_repair.py`. Unrelated working-tree changes are outside this review.

### Bound identities

The supplied current identity packet is `run/dome/checks/repair028_identity.sha256`:

| Artifact | SHA256 |
|---|---|
| `src/libshell/ReferenceHybrid/Core.hpp` | `f3265448c4d2d210659fab2483d1bf9881852128d1435d7f490e5bc4ae301ebd` |
| `src/libshell/NativeHybridShellSolver.hpp` | `d71f60390640edf3c1e5335d3cf481e042dfba6eabed5165a2c8d3ecbf6aaf4e` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `b7d331d05e614b1ddc44d90f5decd7f755d91434f4202e233f381ab09feb6352` |
| `test/testshell/Test_NativeHybridProduction.cpp` | `84c606a2c3be85316fa657f8c028590e6db9db850c7e325c3570d952807389ce` |
| `scripts/test_dome_attempt028_repair.py` | `ecd93bfd3e4a603032e57f46d3455f0b2d3b584f72e5b593799d3a5824ca188b` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |

Archived input: `run/dome/bo_stage3/attempt_028/solve/sequence.json`, recorded SHA256 `41d546d73d8b285cbf42eb7fffc73e314d423b825fa003aed123aae4dc19f31c`.

Archived command: adjacent `execution.json`, recorded SHA256 `9f67061ed0786d5f17ea8139800e559644f5679eaea2daa4a41a90c41e70a6e8`.

These are **inspected manifest identities, not independently recomputed hashes**. Initial build-directory `source.sha256`/`binary.sha256` are not authoritative for the final repaired slice.

## Checks performed

### Numerical validity and safety

- `src/libshell/ReferenceHybrid/Core.hpp:503–525`: for \(B=A+\lambda I\), the implementation computes \(r=By+b\), solves \(B\delta=-r\), and tests \(y+\delta\). The correction sign and matrix are correct.
- Correction occurs only for a finite inaccurate solve. Only a finite, strictly smaller residual is retained; the unchanged relative-residual threshold remains `1e-8`.
- Rejected returned solves advance through the existing shift schedule and maximum **12 total shift attempts**, including factorization failures. The opt-out path retains the existing `linear_solve_failed` guard.
- `src/libshell/ReferenceHybrid/Core.hpp:533–553`: finite descent and strict energy decrease/Armijo checks remain after linear acceptance.
- `src/libshell/NativeHybridShellSolver.hpp:342–361`: fresh endpoint evaluation and curvature checking remain required. A failed/capped optimizer cannot become accepted merely because `bestX` looks satisfactory.
- Production arithmetic remains double precision. Refinement in the same precision can help but is not guaranteed to improve accuracy; a small shifted-system residual is not a forward-error or conditioning certificate.
- The new symmetric 3D fixture matrix is strictly diagonally dominant with positive diagonal, hence SPD. Its `1e-30` tolerance is explicitly test-only.

### Identity, handoff and execution

- The option defaults false and changes serialized identity only when enabled.
- Both phase configs carry the same bit. Constructor and transition equality guards remain intact; L-BFGS does not consume the option.
- The runner copies the archived command/input, replaces binary/input location, and adds trace plus repair flags. It retains fixed handoff2000, gradient `5e-14`, native600s and process1500s.
- Output creation refuses an existing directory. `run/dome/repair_attempt028_v1` was absent when inspected. No BO write, retry loop or restart path appears in the runner.
- Native timing is cooperative, not a hard600s process ceiling. Existing verification allowances remain; the1500s subprocess timeout fits the proposed30-minute allocation.
- Compute/thread controls are appropriate for the proposed8CPU/32GB run. No new unbounded numerical loop was introduced.
- Both isolated CMake caches identify the same SuiteSparse7.7 location, compiler and Eigen alignment settings. This supports comparison with diagnostic build4371653—not equivalence with the historical4.4.2 BO binary.

### Retained evidence examined

- `run/native_shape_refinement/build_4375711/repair_rebuild_v5.log`: final shell and fixture linking.
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 tests passed,59ms**.
- `run/dome/repair_fixture_v3/argumentparser.log`: repair flag enabled.
- Its trace/result: seven rows, five warmup and two Newton attempts, accepted gradient `4.7416439423536765e-15`, one terminal equilibrium.
- Archived attempt028 command/input and diagnostic execution receipt.

No builds, tests, simulations, hash commands, or scheduler checks were executed by this reviewer.

## Launch-blocking findings

**None for the stated single diagnostic reproduction.**

## Nonblocking caveats and minimal follow-up

1. **Not all backend solve failures retry.**  
   `src/libshell/ReferenceHybrid/CholmodFactorization.hpp:71–76` throws on CHOLMOD solve errors; `src/libshell/ReferenceHybrid/Core.hpp:700` converts exceptions to terminal `oracle_exception`. Those errors bypass shift recovery. This is safe stopping and does not invalidate the observed finite-residual repair case. Narrow claims accordingly; do not add blanket exception recovery merely to match broader wording.

2. **Tests establish bounded path exercise, not repair efficacy.**  
   `test/testshell/Test_NativeHybridProduction.cpp:98–121` checks correction entry and at-most-three trials, but does not require successful recovery, multiple rejected shifts, or explicitly assert retention of the better candidate. The integration fixture’s residuals were already about `1e-15`, so it validates wiring/handoff, not refinement. Optional strengthening: assert rejection progression and retained-residual selection in the existing fixture. These limitations do not block this bounded feasibility test.

3. **Per-solve timing is incomplete.**  
   `src/libshell/ReferenceHybrid/Core.hpp:504–514` overwrites `solve_seconds` across shifts and excludes correction work. Cumulative elapsed/time checks still charge that work. Use cumulative Newton/process time, not this field, for performance interpretation.

4. **Recorded hashes are not enforced comparisons.**  
   `scripts/test_dome_attempt028_repair.py:25–27` records actual binary/input hashes but does not compare them with expected identities. Parent should verify the bound packet and archived input/command immediately before launch. Approval does not certify dynamically loaded library identities or continuous immutability.

5. **Collection is not acceptance or restartability.**  
   `scripts/test_dome_attempt028_repair.py:37–42` marks a parsed trace `diagnostic_collected`, including rejected solves. Before describing a successful reproduction, audit proof, sequence completion, all46 accumulation cycles, single equilibrium and endpoint agreement. Exported trace/state is not restart evidence.

## Interpretation and stopping

The prior residual violation explains that diagnostic rejection; it does not establish its underlying conditioning cause or explain all capped designs.

- **Recovered linear solve, accepted endpoint:** evidence for numerical recovery on this fixed input under the modern toolchain; inspect trace to distinguish correction from shift recovery.
- **Recovered linear solve, later failure/time cap:** local recovery only, not resolved equilibrium.
- **No repair activation or materially different trajectory:** insufficient evidence that the historical failure was repaired.
- **Failure, timeout or missing trace:** retain evidence and stop; no automatic retry, gate relaxation or BO observation.

This is a model-conditional numerical investigation, not physical parameter identification or experimental validation. Positive Hessian pivots are not geometric dome curvature. Even an accepted endpoint does not establish the frozen80% dome criterion, global optimality, mesh robustness or general solver reliability.

Approval applies only to the reviewed single-attempt slice. Execution authorization remains with the parent/researcher.