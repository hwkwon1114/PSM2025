# Independent delta review — adaptive attempt028

## Verdict: APPROVED_WITH_CAVEATS

**No unresolved launch blockers found for one bounded adaptive-continuation test.**

Reviewed limits: **1800s native per substep, 2400s total subprocess timeout, 8 CPUs / 32GB / 45-minute allocation**, fresh output `run/dome/repair_attempt028_adaptive_v1`.

This approval covers the reviewed collection slice only—not another attempt, increased resources, BO continuation, solver rebuilding or changed acceptance thresholds.

## Authority and identities

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md`
- `docs/reports/dome_toolpath_investigation.md`, especially **Current status / next action** and **Longer-test result**
- `docs/reports/dome_long028_review.md`
- `docs/reports/dome_linear_repair_review.md`

The current explicit adaptive-test authorization supersedes the historical adaptive=false restriction **for this test only**.

Identities recorded in `run/dome/checks/adaptive028_identity.sha256`:

| Artifact | SHA256 |
|---|---|
| `scripts/test_dome_attempt028_adaptive.py` | `0738e22f2c3cc4095e7604368af29bb92914b52174124759e54d53ad68e0014a` |
| `scripts/run_dome_attempt028_adaptive.sbatch` | `61b6fab93eb170c6fa70141130e4c92c9a4c683925ad48e99d0da923358a66bb` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |
| `run/dome/bo_stage3/attempt_028/solve/sequence.json` | `41d546d73d8b285cbf42eb7fffc73e314d423b825fa003aed123aae4dc19f31c` |
| `run/dome/bo_stage3/attempt_028/solve/execution.json` | `9f67061ed0786d5f17ea8139800e559644f5679eaea2daa4a41a90c41e70a6e8` |

These are **inspected manifest identities, not independently recomputed hashes**. Unchanged numerical-source identities and evidence are inherited from `run/dome/checks/repair028_identity.sha256` and the prior reviews. Unrelated checkout changes are outside scope.

## Checks performed

### Load path, mathematics and restoration

- `src/simulations/Sim_Bilayer_Growth.cpp:2507–2524,2649–2685` accumulates all **4+21+21=46 cycles** before equilibrium. With final_only plus adaptive enabled, interpolation starts from the initial stress-free metrics—not the metrics preceding cycle46.
- The continuation target is
  \[
  A(\lambda)=(1-\lambda)A_0+\lambda A_{46}.
  \]
  This preserves positive definiteness between SPD endpoints. It is an affine **metric homotopy**, not fractional physical passes or proportional engineering strain.
- Ordered multiplicative accumulation, face-shared hardening and fixed reference curvature remain unchanged. Intermediate solves do not replay growth/hardening.
- `src/simulations/Sim_Bilayer_Growth.cpp:2742–2833` starts with step1, halves failures, doubles successful steps up to1, and uses minimum step1/64 and maximum8 consecutive retries. It is **not a fixed10% ramp**.
- Rejection restores the last accepted vertex/director DOFs and load metrics; acceptance advances that state. At lambda1 the exact accumulated metrics are used, avoiding interpolation-rounding changes to the qualified endpoint.
- Each substep creates a fresh native solver/session and objective fingerprint. Optimizer memory is not reused across different targets. Fixed2000 warmup and the native1800s allowance apply independently to each invocation.

### Native failures and repeated-solve outputs

**Returned native rejection does not bypass adaptive retry.** At `src/simulations/Sim_Bilayer_Growth.cpp:2771–2825`, `report.accepted=false` reaches convergence logging, rollback and halving. This includes returned `time_cap` and indefinite-endpoint reports.

Exceptions outside that returned-report path—initialization, independent verification, output failures or process termination—can still terminate the sequence without retry. Engine-caught exceptions become `oracle_exception` reports and can reach continuation if subsequent reporting/verification succeeds. This is not blanket exception recovery.

Repeated-solve output review found:

- Proof JSONL appends one flushed object per returned solve; `scripts/test_dome_attempt028_adaptive.py:41–44` correctly reads it line by line.
- Disabling the optional trace avoids its fixed-filename overwrite refusal.
- **The partial-state-overwrite concern does not apply to this reviewed route.** `src/simulations/Sim_Bilayer_Growth.cpp:1340–1374,2769–2771` exports native state only for an accepted terminal cycle at lambda≥1, and refuses replacement. Accepted partial-load substeps do not export that state.
- Sequence history/convergence streams are opened once; final shape/result writing follows completed continuation. No additional deterministic per-substep filename collision was found.
- Proof rows contain no lambda field. Bind them by order to `sequence_convergence.csv`; do not infer load from the last proof alone.

### Execution safeguards and feasibility

`scripts/test_dome_attempt028_adaptive.py:15–39` requires a Slurm allocation, refuses an existing output directory, copies the archived input, sets OpenMP8/thread-limit8, limits BLAS threads to1 and launches one direct subprocess. No campaign writes or automatic outer retry appear.

The output directory was absent when inspected. The batch requests the stated resources and explicit Python environment; submit from the repository root because its Slurm log path is relative.

The total2400s ceiling leaves300s within the allocation. It does **not** guarantee completion of continuation. The preceding full-load test consumed1818.65s; comparable initial behavior would leave only approximately**581s** for every subsequent substep combined. Do not expand resources to compensate.

## Evidence reused and checks not performed

Directly inspected:
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 tests passed,59ms**.
- `run/dome/repair_fixture_v3/sequence_result.json`: accepted three-cycle fixture, seven attempts, gradient `4.7416439423536765e-15`.
- Prior longer-test receipt, native proof and convergence CSV: job4382385 returned−6, `time_cap`, accepted=false, gradient `1.9910855562594633e-11`, indefinite endpoint.

These support unchanged native behavior and wiring—not adaptive end-to-end rollback exercise or full-input convergence. Tests concerning adaptive **solver handoff** do not establish adaptive **load continuation**.

No edits, builds, tests, simulations, hash recomputation, scheduler queries or dynamic-library checks were performed. Python/Bash syntax success is supplied evidence, not rerun evidence.

## Launch-blocking findings

**None for this bounded collection test.**

## Nonblocking caveats and required interpretation

1. **Collection is not acceptance.**  
   `scripts/test_dome_attempt028_adaptive.py:40–45` may record `diagnostic_collected` after a nonzero native return. A subprocess timeout instead records `diagnostic_error` and skips proof summarization; flushed files remain available for inspection. Neither status establishes equilibrium.

2. **Preserve the terminal gate before any shape interpretation.**  
   Require:
   - Successful complete sequence with46 ordered accumulation cycles.
   - Final accepted convergence row at **lambda_trial=1**, matched to the terminal proof.
   - `gradient_target`, finite gradient≤`5e-14`, and `positive_pivots`.
   - Agreement among sequence result, exported native state/proof and final endpoint artifacts.

   `equilibrium_solves=1` counts the completed physical-cycle continuation block, **not** the number of native substep solves. Do not require exactly one proof row or misreport multiple substeps as one optimizer invocation. The existing analyzer alone does not explicitly audit lambda history.

3. **Identity recording is not enforcement.**  
   `scripts/test_dome_attempt028_adaptive.py:28–30` records hashes without checking expected values. Parent should verify the bound packet before submission; consequential changes require reassessment.

4. **Inherited numerical/toolchain limitations remain.**  
   The repaired build uses the reviewed modern SuiteSparse7.7 toolchain, not the historical BO4.4.2 toolchain. No historical trajectory-equivalence claim is justified. Disabling trace also removes per-iteration evidence for distinguishing correction from shift recovery; proof/CSV retain only solve summaries.

## Scientific decision and stopping

This tests model-conditional numerical feasibility of a different load path with the same final targets. Continuation may select a different local equilibrium branch; success would not isolate the cause of the earlier failure.

- **Complete accepted lambda1 endpoint:** audit the evidence above before applying unchanged dome criteria. No BO observation or campaign restart.
- **Partial loading, rejection, timeout or inconsistent/missing evidence:** report equilibrium **unresolved** and stop. No automatic retry, threshold change or resource extension.

Positive Hessian pivots are not geometric dome curvature. Even accepted equilibrium does not establish the frozen80% geometric criterion, mesh robustness, global optimality or physical validation. This review grants neither execution permission nor runtime-enforcement certification.