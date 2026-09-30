# Absolute-gradient stopping audit

## Current objective: global equilibrium at fixed final eigenstrain

**Researcher decision:** prioritize finding the global minimum of the stated shell energy over its admissible configurations, with final eigenstrain/rest metrics, material parameters, boundary conditions and discretization fixed. Physical trapping, experimental loading-history dependence and their mechanics literature review are deferred. This is the intended numerical objective, not a claim that a global minimum has already been found or that it necessarily describes the experimental forming trajectory.

- Treat all-at-once loading, continuation orders and alternative initializations as **search strategies for the same final energy problem**, not distinct physical predictions, when their final discrete inputs are identical. Account for symmetry or genuinely degenerate minima before requiring one unique shape.
- Compare candidate energies only under the identical final discrete objective and constraints. Check finite energies, residual acceptance and admissibility; assess local stability separately. A small gradient or positive local Hessian is not a global-optimality certificate.
- The safeguarded local restart is a possible efficiency/reliability improvement, **not a mechanism for escaping a well-converged local minimum**. A subsequent bounded search must test multiple basins rather than merely polish one more tightly.
- Report **“lowest-energy solution found”** unless a defensible global-optimality certificate is available. Keep mesh and loading-discretization convergence separate from optimization on a fixed mesh.

No new solver change, search sweep or literature review was launched by recording that decision. A subsequently approved, separate [optimization/JAX decision review](global_energy_optimization_jax_review.md) recommends retaining L-BFGS as comparator, testing structured starts and a native cached/sparse second-order challenger, and gating any JAX port on float64 parity and end-to-end benefit. This is a proposal, not a demonstrated speedup or authorized experiment. The physical-history mechanics review remains deferred. The observations and historical protocols below are preserved.

## Finding

The **original, normalized-stop** finer-mesh failures are explained by inconsistent HLBFGS stopping and sequence acceptance criteria, not by evidence of a TinyAD derivative error. The opt-in absolute-stop recheck below removes that mismatch but still encounters tight-tolerance line-search and iteration-budget failures.

- `src/libshell/HLBFGS_Wrapper.hpp::default_setup`: legacy parameter[5]=tol tests `||g||/max(1,||x||) <= tol`; parameter[6]=1e-16 provides an alternative absolute stop.
- `external/hlbfgs/HLBFGS.cpp`: these are alternative stopping conditions, not simultaneous requirements. Code 2 is the normalized stop.
- `src/simulations/Sim.hpp::MinimizationReport::accepted`: codes 1–4 are eligible only with finite nonnegative `gradientNorm <= equilibrium_grad_tol`. Code 5 is rejected. Legacy `converged()` is weaker and must not be used to certify these sequence endpoints.
- `src/simulations/Sim_Bilayer_Growth.cpp`: the investigated sequence uses `accepted()`, not `converged()`; rejected adaptive trials roll back and reduce the loading increment.
- The production HLBFGS callback computes the analytic `CombinedOperator_Parametric` energy gradient through `engOps`. TinyAD derivatives are used in the alternative Newton/stability/verification paths, not this HLBFGS solve. The manifests disable Newton, stability and final certification.

Refinement changes the number and norm of the coordinate/director variables. A fixed ratio of absolute acceptance to normalized stopping tolerance (here 10) does not guarantee consistent tests across meshes. Failed fine-mesh trials have code 2 with gradients roughly 1.17–1.28e-11 (standard) or 1.20–1.26e-13 (tight), just above their respective absolute gates. Halving the growth step is not a remedy for an optimizer stopping too early.

## Previous endpoint ledger audit

`python/audit_nested_acceptance.py` checks saved CSV rows, result markers and full-load completion of each cycle. This is a ledger audit, **not an independent gradient recomputation at archived configurations**.

| Previous case | Maximum accepted gradient | Gate | Full endpoint |
|---|---:|---:|---|
| Nested g=.005 broad→central standard | 8.453e-12 | 1e-11 | yes, 6 cycles |
| Nested g=.005 central→broad standard | 8.533e-12 | 1e-11 | yes, 6 cycles |
| Nested g=.005 broad→central tight | 8.412e-14 | 1e-13 | yes, 6 cycles |
| Split AB whole g=.005 retry | 7.866e-12 | 1e-11 | yes, 2 cycles |
| Split BA whole g=.005 retry | 8.231e-12 | 1e-11 | yes, 2 cycles |

No accepted-row gate violations were found in these five cases. **Correction to the earlier summary:** the tight broad→central run has one rejected adaptive trial, but subsequently recovers and completes all six cycles at full load. `all_accepted=false` over *all attempted solves* incorrectly excludes this valid completed endpoint. It is not a stability certificate, but it is not a failed trajectory either.

Saved audit: `run/forward_model_diagnostics/absolute_stop_audit/ledger_audit.json`. Failed finer-mesh cases remain excluded; the last original task was still running at this audit.

## Opt-in correction and tests

New `-hlbfgs_absolute_gradient_tol` disables the normalized stop (parameter[5]=0) and sets parameter[6] to the requested absolute target. The default remains unchanged. Both full and reduced wrappers support this setting. In this mode the returned gradient is recomputed analytically at the returned state (free DOFs for reduced solves) before being reported to the unchanged acceptance gate; callback and recomputed norms are logged.

New tests check unchanged legacy parameters, disabled normalized stopping, invalid inputs and a translated quadratic reaching the absolute target. Existing acceptance tests are also selected. A coarse curved-state analytic/TinyAD gradient comparison is a separate build gate; it is not a certificate for grown final states.

Build/test job **6958070 completed** (8 min 3 s; exit 0), with a separate source snapshot and binary under `run/forward_model_diagnostics/absolute_stop_audit/source`. All **15 selected unit tests passed**. The saved coarse curved-state analytic/TinyAD check reports gradient infinity-norm difference **4.066e-20**, relative **2.933e-16**, and completes its verification-only run. These are tests of the stopping integration and a coarse verification state, not certificates for grown endpoints. Evidence: `tests/unit_tests.log`, `tests/tinyad_crosscheck.log`. The historical production binary was not replaced.

## Bounded recheck

Dependent array **6958071** runs only after build/tests succeed. **Concurrency correction:** the earlier note said two tasks; the current submission script and live scheduler report `ArrayTaskThrottle=7`. Eight fresh trajectories: g=.005, two orders, meshes .015/.01, standard/tight absolute acceptance 1e-11/1e-13. Internal absolute stop is half the respective gate. Loading, material, mesh construction, continuation, iteration limit and acceptance thresholds are unchanged. Each task has the existing 24-hour budget. No automatic extension/retry.

These are fresh coarse/fine trajectory rechecks, not restarts from archived VTP endpoints. Cross-mesh footprint discretization and absolute residual scaling remain limitations. Persistence of a shape difference does not alone prove stable physical branches.

## Current partial audit — 2026-09-22 01:46:59 UTC

**Scope:** all eight planned cases; saved ledgers, result markers, immutable copies of completed final VTPs, and scheduler records. No simulations rerun, thresholds changed, or independent endpoint residuals recomputed. The snapshot took under one second, but live jobs are not an atomic scheduler/filesystem snapshot.

**Evidence:** [snapshot ledger](../run/forward_model_diagnostics/absolute_stop_audit/recheck_snapshot_20260922/ledger.json) · [shape comparisons](../run/forward_model_diagnostics/absolute_stop_audit/recheck_snapshot_20260922/analysis/comparison.json) · [reproducible analysis](../python/analyze_absolute_stop_recheck.py).

All eight manifests have the same binary SHA-256 (`c550a744…c8e0d08`), matching sequence hashes, identical fixed command flags, and the same multiset of toolpaths. Only order, mesh and tolerance vary. Attempt keys are unique. All **124 rows** also pass loading-chain checks: each attempt starts from the last accepted loading, rejected attempts do not advance it, step sizes match loading differences, and each preceding cycle reaches full load before the next begins (`analysis/continuation_chain_checks.json`). **89 accepted rows have no residual/code/finite-energy violations**; 35 rejected attempts are retained, not counted as accepted or automatically treated as failed trajectories. All three complete cases have six full-load cycles, their corresponding geometry exports, success markers and no rejected attempts.

| Task | Mesh res | Order | Gate | Status at snapshot | Fully completed cycles | Latest accepted partial cycle |
|---:|---:|---|---:|---|---:|---|
| 0 | .015 | broad→central | 1e-11 | complete, 1h43m | 6/6 | — |
| 1 | .015 | broad→central | 1e-13 | failed, 5h06m | 3/6 | cycle 4: 62.5% |
| 2 | .015 | central→broad | 1e-11 | complete, 5h09m | 6/6 | — |
| 3 | .015 | central→broad | 1e-13 | runner timeout, 23h50m | 3/6 | cycle 4: 3.125% |
| 4 | .010 | broad→central | 1e-11 | complete, 7h19m | 6/6 | — |
| 5 | .010 | broad→central | 1e-13 | running, 20h06m | 3/6 | cycle 4: 56.25% |
| 6 | .010 | central→broad | 1e-11 | running, 15h03m | 5/6 | cycle 6: 25% |
| 7 | .010 | central→broad | 1e-13 | running, 15h03m | 2/6 | cycle 3: 75% |

Progress percentages are loading fractions within a cycle, not estimates of remaining runtime. Max accepted gradients for completed tasks 0/2/4 are respectively **4.937e-12 / 4.998e-12 / 4.993e-12**.

### Failure diagnosis

- **Task 1:** 15 rejected code-1 attempts, each a line-search failure. Final rejected trial: cycle 4, λ=.625→.640625, minimum allowed step **1/64**, recomputed gradient **1.13239e-13 > 1e-13**. The callback and recomputed analytic norm agree to displayed precision. Source tracing confirms that rejection at the minimum step throws; this was not a wall-time or iteration-cap termination. `gradient_norm=0.000000` in the exception is decimal formatting, not a zero residual. Small-residual line-search difficulty is observed; roundoff as its root cause is not established.
- **Task 3:** application timeout after **85,800 s**, recorded as Slurm FAILED/124 rather than scheduler TIMEOUT. Nine completed rejected solves hit the **1,000,000-iteration cap** (reported as 1,000,001); one rejected code-1 solve. The last recorded rejection is **5.4458e-13**, before the subsequent solve was interrupted. Do not label that last completed row the residual of the interrupted state.
- **Running tasks:** task 5 has seven rejected code-1 solves; task 6 has two rejected code-5 solves; task 7 has one rejected code-5 solve. Their already accepted substeps remain valid ledger evidence, but none is a completed endpoint at this snapshot.

Thus the old normalized-stop inconsistency is not the explanation for every new failure. No accepted-row threshold violation was found, and tightening remains expensive or unsuccessful under this fixed recipe. Accepted code-1 states can exceed the internal half-gate target while still passing the unchanged external gate; the two conditions are not equivalent.

### Completed-endpoint comparisons

Only tasks **0, 2 and 4** enter the final-state comparison. Material identities are kept fixed; proper rigid alignment removes translations/rotations, **not reflections, scaling or material relabeling**. Same-mesh metrics use lumped reference-area weights. Cross-mesh positions use piecewise-linear interpolation on the actual material triangulations at equal-area common cell centers; no extrapolation or uncovered samples.

| Comparison | Aligned position RMS | Area-weighted median / 95th percentile | RMS / mean deformation RMS |
|---|---:|---:|---:|
| Coarse order reversal, task 0 vs 2 | **69.279 mm** | 60.853 / 111.770 mm | **86.0%** |
| Broad→central mesh change, task 0 vs 4 | **82.537 mm** | 79.715 / 123.269 mm | **105.0%** |
x
- **Same-mesh loading check:** task 0 vs 2 has identical accumulated hit counts, zero difference in exported `bbar_ref`, and final top/bottom rest-metric maximum relative discrepancy **2.18e-16**. The large material-corresponding shape difference therefore persists at essentially matching exported final metrics on this mesh.
- **Cross-mesh sensitivity:** 100×120 versus 200×240 common-material sampling changes RMS from **82.53658 to 82.53743 mm** (0.00085 mm), far below the observed difference. The hit-count fields disagree over **14.94%** of sampled material area (14.73% on the coarser sampling grid). Thus footprint discretization changes loading as well as the solve mesh. Raw edge-basis metric components are not directly compared across meshes; this is not a cross-mesh tensor equality test.
- Aligned height spans for tasks 0/2/4: **100.88 / 91.26 / 94.58 mm**. The full spatial maps and difference distributions, not one selected trace, support these summaries. VTP positions are exported as float32; analysis in float64 cannot restore lost precision.

### Revised conclusion

**The present results do not establish physical order dependence; numerical robustness remains unresolved.** Keeping the broad→central order and nominal eigenstrain recipe fixed while refining the mesh changes the computed endpoint by **82.54 mm RMS**, compared with **69.28 mm RMS** for reversing order on the coarse mesh. The same-order mesh discrepancy is therefore at least as consequential as the apparent order effect. The nominal cross-mesh recipe is unchanged, but its discretized loading is not identical: hit-count fields differ over **14.94%** of sampled material area.

At fixed coarse mesh and tolerance, the two orders reach different computed shapes despite essentially matching final rest metrics. This is an observation about these numerical trajectories, not evidence by itself of a mesh-independent physical history effect or distinct stable equilibria. Discretization, insufficient convergence and numerical branch/initialization sensitivity remain competing explanations; the current comparisons do not isolate their contributions.

**Mesh sensitivity is demonstrated; tolerance sensitivity is unresolved, not demonstrated**, because no tight-gate trajectory has a complete endpoint in this snapshot. Passing the configured residual gate does not establish solution accuracy, stability or mesh/tolerance convergence. No stability analysis, physical validation or independent repeats were performed. Physical order dependence should remain an unconfirmed hypothesis until numerical and loading-discretization robustness are established.

### Figures and checks

- [All eight endpoint conditions, PNG](../run/forward_model_diagnostics/absolute_stop_audit/recheck_snapshot_20260922/analysis/absolute_stop_endpoint_height.png)
- [Spatial difference distributions and loading progress, PNG](../run/forward_model_diagnostics/absolute_stop_audit/recheck_snapshot_20260922/analysis/absolute_stop_shape_difference_distribution.png)
- [Caption](../run/forward_model_diagnostics/absolute_stop_audit/recheck_snapshot_20260922/analysis/figures.caption.md)

Analysis completed on compute job **7047285** (one CPU, existing `smcpp_vtk38` environment). Rigid-alignment and affine-interpolation self-tests passed. Both PNGs were visually inspected for panel ordering, missing-case labels, common scales, axes, legends and clipping. Scripted text-bound checks passed. PDF companions were exported but **not visually verified**: the PDF rendering attempt found no `pdfinfo`; no dependency was installed. Color-vision simulations were not performed. Earlier analysis attempts encountered a masked-array conversion issue and an out-of-range tick-label check; both were corrected before the successful run, and the earlier numerical artifact is retained in `analysis_attempt_7047259`.

### Decision / bounded follow-up (not launched)

1. Let tasks 5–7 finish within their existing limits; task 6 is needed to complete the standard-gate fine-mesh order pair. Refresh this audit from a new immutable snapshot rather than modifying the old one.
2. **Do not blindly extend tight runs or loosen acceptance.** They do not yet support tolerance robustness, and a larger cap is not evidence of convergence.
3. Prioritize numerical robustness over further order-effect claims: after the remaining results, agree on a controlled same-order mesh/tolerance diagnostic that accounts for mesh-dependent footprint loading and branch/initialization sensitivity, and isolate the tight line-search behavior. Any independent 1e-13 residual recheck needs a full-precision numerical state, not reconstruction from these float32 VTP surfaces. New solves, solver changes, tighter meshing and stability tests require a separately approved bounded plan.

### Subsequent execution-status check — 2026-09-22

The remaining array tasks **5, 6 and 7 all reached the 85,800-second application timeout** (confirmed by their `result.json` records and Slurm exit 124). The eight-case array has now stopped: **3 complete, 1 line-search failure, 4 application timeouts**. No further complete endpoints were added. Tasks 5/6/7 last exported cycles 4/5/2 respectively; their final partial ledgers have not yet been re-audited. The immutable snapshot and figures above still describe their stated earlier capture time; they have not been silently relabeled. This leaves the fine-mesh standard-gate order pair and all complete tight-gate comparisons unavailable.

## Approved focused stall probe — 7054685 / 7054686

After reviewing the previous unsuccessful solver approaches, the researcher approved diagnosing one stalled solve and testing one targeted change. **This is not a rerun of the eight-case sweep or an extension of its jobs.**

### Question and predeclared test

The saved coarse broad→central tight run first rejected a line search on cycle 4, λ=0→0.5, after 135,117 iterations, with residual 1.1968e-13 against the 1e-13 gate. Its historical log only says “Linesearch is failed”; it does not retain the More–Thuente internal stop code or full-precision trial energies. The ordinary progress log prints only 11 significant digits, so apparent energy plateaus there do **not** diagnose floating-point cancellation.

Source inspection also identifies an energy-dependent Zhang–Xu secant correction (`INFO[13]=3`) on top of the M1QN3 scaling (`INFO[3]=1`). Its curvature update uses differences of nearly equal energies. That is a **testable candidate mechanism, not a diagnosed bug**. A diagonal preconditioner, Newton switch, or looser acceptance gate is not being proposed again here.

- **Selection:** reproduce the original coarse tight recipe with unchanged physics, continuation, mesh and residual flags, stopping at the **first line-search failure above the external gate**. This may not reproduce the historical iteration/state bitwise; threading and a fresh trajectory can change the numerical path.
- **Why replay is needed:** archived VTP surfaces are not complete full-precision solver checkpoints with director variables. Do not reconstruct a claimed exact restart from them.
- **Capture:** native float64 optimizer state, last base state/gradient/direction, curvature-memory arrays, More–Thuente reason and trial trace, descent cosine and diagonal range. Eight repeated evaluations at the same state and seven signed directional probes help distinguish oracle reproducibility and line-search resolution limits from slow convergence. Diagonal range is not a Hessian condition-number estimate.
- **One targeted comparison:** from the identical captured state and unchanged in-memory target metrics, compare current energy-corrected updates (`INFO[13]=3`) with ordinary secant updates (`INFO[13]=0`). Both retain M1QN3 scaling and start with **empty optimizer memory**. Only that update selector differs between the two local solves. A baseline improvement could therefore be due to restarting; it would not establish a benefit from removing the energy correction.
- **Budget and acceptance:** 20,000 iterations per local solve (the library may report 20,001 before its existing cap test), **3 hours total** for replay, diagnostics and both local solves; 3h10m scheduler allocation reserves time for postprocessing. Absolute internal target remains **5e-14**, external gate **1e-13**, accepted codes 1–4; capped solves remain rejected. No automatic retries, extensions, or parameter sweep.
- **Readout:** finite recomputed energy/gradient, residual acceptance, complete iteration/evaluation curves, local wall time, and full-precision rigidly aligned position changes plus director changes. One baseline-then-candidate timing pair is exploratory, not a replicated speed benchmark. No Hessian/stability or mesh-convergence claim follows.

### Execution and provenance

Isolated build/test job **7054685 completed** in **8m38s**; all **17 selected tests passed**, including the paired synthetic fixture. The coarse analytic/TinyAD check again reports relative gradient difference **2.933e-16**. Dependent diagnostic job **7054686 completed successfully** in **1h07m17s**. This is diagnostic completion, not a completed six-cycle trajectory. The source is copied from the verified absolute-stop snapshot, with opt-in instrumentation only in its private HLBFGS translation unit. The production and original absolute-stop binaries and existing jobs are untouched; the original binary checksum was rechecked successfully.

[Runner](../python/run_hlbfgs_stall_probe.py) · [Diagnostic harness](../test/diagnostics/HLBFGS_StallProbe.hpp) · [Harness tests](../test/diagnostics/Test_HLBFGS_StallProbe.cpp) · [Build job](../scripts/hlbfgs_stall_probe_build.sbatch) · [Probe job](../scripts/hlbfgs_stall_probe.sbatch).

Artifacts are under `run/forward_model_diagnostics/absolute_stop_audit/stall_probe/`. Build gates include the existing acceptance/stopping unit tests, a synthetic paired-polishing/float64 roundtrip fixture, and the coarse analytic/TinyAD gradient check. The diagnostic binary deliberately exits **86** after the paired test; `diagnostic_status.json` distinguishes that intentional stop from failure and never labels it a completed six-cycle trajectory. If the bounded replay does not reach a qualifying stall, or the time budget interrupts the probe, report it as partial—not as evidence that either update is faster. Captured optimizer arrays alone are not a standalone physical restart; target metrics are retained in process for this local comparison.

### Completed probe: local restart recovered the tight residual

Evidence: [capture](../run/forward_model_diagnostics/absolute_stop_audit/stall_probe/current/broad_to_central_tight_first_stall/capture.json) · [line-search trace](../run/forward_model_diagnostics/absolute_stop_audit/stall_probe/current/broad_to_central_tight_first_stall/failed_line_search.csv) · [paired results](../run/forward_model_diagnostics/absolute_stop_audit/stall_probe/current/broad_to_central_tight_first_stall/probe_analysis.json) · [diagnostic status](../run/forward_model_diagnostics/absolute_stop_audit/stall_probe/current/broad_to_central_tight_first_stall/diagnostic_status.json).

The fresh trajectory first triggered on **cycle 2, λ=.5→1**, not the historical cycle-4 failure. Three prior substeps are accepted in its ledger; the diagnostic stopped inside the fourth solve before a continuation row could be written. Thus this is a new stalled state from the same recipe, **not an exact reproduction of the old failed endpoint**.

- At capture: **116,316 iterations**, residual **1.23709e-13 > 1e-13**. More–Thuente stopped with **code 6**, a bracket/roundoff or invalid-interpolation condition, after **14 of 20** allowed evaluations—not an evaluation-budget exhaustion.
- The bracket collapsed around a step of **2.1588e-11**. At its final trials, energy differences are only a few float64 ulps; eight identical-state evaluations span **2.068e-25 (4 energy ulps)**, with no detected gradient difference. The search direction has descent cosine **3.45e-5**. These observations support a near-stagnant search direction and precision-sensitive line search at this state; they do not measure a Hessian condition number or exclude poor conditioning elsewhere.
- The energy-correction multiplier reached its lower clamp **0.01**, but that alone does not prove the correction caused the failure.

| Local polishing solve | Iterations / evaluations | Recomputed final residual | Local solver time | Gate passed |
|---|---:|---:|---:|---|
| Current energy-corrected update, restarted | **3 / 9** | **4.2132e-14** | 0.0428 s | yes |
| Ordinary secant update, restarted | **3 / 9** | **4.5512e-14** | 0.0405 s | yes |

Both runs used bitwise-identical starting DOFs, the same in-memory target metrics and empty optimizer memory. Both returned code 3 and met even the internal **5e-14** target. Position changes after alignment were **1.04e-9 / 1.50e-9 mm RMS**; their final positions differ by **5.12e-10 mm RMS**, negligible at the scale of the shell. Director RMS changes were **3.55e-15 / 5.86e-15 rad**. These are full-precision numerical comparisons, not measured physical precision.

**Conclusion:** there is **no demonstrated benefit from removing the energy correction** in this one local test. A fresh HLBFGS restart with the existing update recovered the residual in three iterations without a consequential shape change. The restart resets curvature memory, diagonal scaling and initial-step handling together; this test does not isolate which component enabled recovery. The ≈0.04 s times exclude the hour-long loading replay and are not replicated whole-trajectory speedups.

**Next candidate, not implemented or launched:** one safeguarded same-load HLBFGS restart on a qualifying near-gate line-search failure, with independent returned-gradient verification, unchanged gate and a strict retry/iteration cap. Validate it on a bounded full continuation before adopting it. Do not apply this finding automatically to million-iteration cap failures, relax acceptance, or infer stable physical branches. Both three-step curves and raw diagnostic traces are preserved; no new diagnostic figures have been rendered or reviewed yet.
