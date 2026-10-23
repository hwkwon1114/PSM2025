# Independent delta review — longer attempt028 test

## Verdict: APPROVED_WITH_CAVEATS

**No scientific or operational launch blockers found for one fresh fixed-input run:** native limit **1800s**, subprocess timeout **2400s**, allocation **8 CPUs / 32GB / 45 minutes**, output `run/dome/repair_attempt028_long_v1`.

This verdict covers only the reviewed convergence test—not retries, further extensions, BO continuation, recalibration or relaxed acceptance.

## Authority and scope

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`
- `docs/reports/dome_toolpath_investigation.md`, including **Opt-in repair result** and **Current status / next action**
- `docs/reports/dome_linear_repair_review.md`

The current researcher approval explicitly covers this longer test. Historical campaign plans are not used as permission.

Reviewed both new scripts against the previous repair runner, archived command/input, identity packets, retained execution evidence and relevant unchanged timing/acceptance code. Other working-tree changes are outside scope.

### Bound identities

From `run/dome/checks/long028_identity.sha256`:

| Artifact | SHA256 |
|---|---|
| `scripts/test_dome_attempt028_long.py` | `86a63fc34ea42013747c720e5b3ec04d134c6b32f377009b9b36227c4f2a5f04` |
| `scripts/run_dome_attempt028_long.sbatch` | `6b6b233248b09762851041ec3c4ed78f9d4f96d5e620e6a21150b13c88557eb4` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |
| `run/dome/bo_stage3/attempt_028/solve/sequence.json` | `41d546d73d8b285cbf42eb7fffc73e314d423b825fa003aed123aae4dc19f31c` |
| `run/dome/bo_stage3/attempt_028/solve/execution.json` | `9f67061ed0786d5f17ea8139800e559644f5679eaea2daa4a41a90c41e70a6e8` |

These are **inspected manifest identities, not independently recomputed hashes**. Binary/input identities agree with the previous review and repaired-run receipt. Unchanged numerical-source/test evidence is inherited from `run/dome/checks/repair028_identity.sha256` and the prior review.

## Checks performed

### Scope and numerical contract

- `scripts/test_dome_attempt028_long.py:17–25` retains the archived command, substitutes the same repaired binary, copies input bytes, sets native1800s and retains trace/repair flags.
- Input repeats are **4+21+21=46**, with calibrated growth/orthotropy, face-shared hardening and disabled boundary conditions unchanged.
- Archived flags retain `final_only`, adaptive loading disabled, fixed2000 warmup attempts, fixed handoff, no seed escape and gradient gates `5e-14`.
- No checkpoint is loaded. This restarts from the original input, not from the previous endpoint.
- `src/simulations/Sim_Bilayer_Growth.cpp:1296–1313` passes the positive finite time limit independently of the gradient gate.
- `src/libshell/NativeHybridShellSolver.hpp:330–361` retains cooperative stopping and fresh endpoint verification. A capped optimizer cannot become accepted merely because its retained candidate looks satisfactory.

No new numerical equations or arithmetic changes require repeating the unchanged numerical suite.

### Runtime, allocation and dependencies

- `scripts/test_dome_attempt028_long.py:33–37` sets OpenMP8 with thread limit8 and dynamic teams disabled; OpenBLAS/MKL are limited to1. Archived Hessian threads remain8.
- Native1800s leaves600s before the subprocess ceiling; subprocess2400s leaves300s within the2700s allocation. Existing verification allowances are unchanged. Native timing remains cooperative, not a hard1800s process deadline.
- One direct subprocess call, no retry loop, no BO writes and no continuation launcher appear. Exceptions record failure and propagate.
- Output creation uses `exist_ok=False`; the proposed output directory was absent when inspected.
- The batch script requests one task on one compute node and uses the explicit `smcpp_vtk38` Python.
- `run/native_shape_refinement/build_4375711/cmake/CMakeCache.txt` and `build.ninja:294–301` show SuiteSparse7.7 and explicit dependency RPATHs for SuiteSparse, the conda environment and NVHPC libraries. No new dependency installation is introduced.

### Reused evidence and limits

Examined:
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 tests passed,59ms**.
- `run/dome/repair_fixture_v3/argumentparser.log` and trace proof: repair enabled; accepted seven-attempt fixture, gradient `4.7416439423536765e-15`.
- `run/dome/repair_attempt028_v1/diagnostic_execution.json`, solver log and proof header: job4379066,2108 trace rows, return−6, `time_cap`, accepted=false,622.25s total.

The tiny fixture supports wiring, not full-input convergence. The prior numerical review’s limitations remain applicable.

Bash/Python syntax success is supplied evidence; this reviewer did not rerun syntax checks, tests, simulations, hash commands, dynamic-loader checks or scheduler queries.

## Launch-blocking findings

**None for this bounded slice.**

## Nonblocking caveats

1. **Identity recording is not enforcement.**  
   `scripts/test_dome_attempt028_long.py:26–31` records binary/input hashes without comparing expected values. Parent should verify the bound manifest before submission; consequential mismatches require reassessment.

2. **Collection is not acceptance.**  
   `scripts/test_dome_attempt028_long.py:38–43` can report `diagnostic_collected` after solver rejection. Neither that status nor batch exit success establishes equilibrium. Audit full46-cycle completion, exactly one terminal equilibrium, accepted proof and endpoint agreement before geometry analysis.

3. **Dependency and submission limits.**  
   The reviewed toolchain matches the repaired test, **not historical BO SuiteSparse4.4.2**. Actual loaded libraries and CPU compatibility were not independently checked. Submit from the repository root because `scripts/run_dome_attempt028_long.sbatch:10` uses a relative log path.

4. **Timing interpretation.**  
   Use cumulative/native and subprocess elapsed time, not the inherited incomplete `solve_seconds` field. Longer elapsed allowance does not promise an identical trajectory or proportional improvement.

## Scientific interpretation and stopping

Prior row2018 demonstrates **shift recovery**, not correction-only success: residual `6.93e-8 → 2.83e-8` still failed `1e-8`; the shifted solve reached `9.39e-12`. The later gradient `1.2765e-11` remained255.3× above target, with an indefinite endpoint Hessian.

This test distinguishes completion within the larger allowance from continuing numerical difficulty; it cannot prove eventual convergence or diagnose the underlying conditioning mechanism.

- **Accepted endpoint:** verify cycles, single equilibrium and proof first; then apply the existing geometric criteria. Do not add a BO observation.
- **Cap, rejection, timeout or incomplete evidence:** report unresolved convergence and stop. No automatic extension or retry.

Hessian curvature is not surface Gaussian curvature. Even accepted equilibrium does not establish a dome, global optimality, mesh robustness or physical validation. Approval is limited to this reviewed slice and does not itself grant execution permission.