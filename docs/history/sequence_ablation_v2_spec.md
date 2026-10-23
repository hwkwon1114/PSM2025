# Sequential Toolpath Ablation — Current Investigation and Historical v2 Specification

## Current investigation: perpendicular-toolpath order versus equilibrium cadence

Authority: confirmed goal `.pi/goals/active_goal_2026101910270493_mvfeqwh7-k4y0nd.md`. This section is the current protocol; the original v2 specification below is historical, not current execution authority. No dome campaign is resumed or changed. At preparation, Slurm4467951 was running the independently authorized dome continuation campaign; preserve its code, inputs and outputs.

**Question/decision.** Does postponing equilibrium to the end reproduce the shapes obtained by equilibrating after each physical application? Separate (i) different accumulated targets caused by ordered hardening/composition, (ii) different equilibria of an identical final objective, and (iii) numerical uncertainty. Agreement limits the hypothesis for these recipes; robust disagreement establishes a case-specific failure of lumping, not physical time dependence or a global-minimum claim. Inconclusive numerical acceptance is a valid outcome, not a bad-shape score.

**Recipes, chosen before new outcomes.** Reuse the first two toolpaths of `calibration/configs/dome/baseline_attempt209.json`: A is the centered 0-degree zigzag (Lv140mm, angle9.13deg,N10,width8mm,center_peak inherited with end ratio.4,power2); B is the +118mm-offset 90-degree side zigzag (Lv196mm,angle2.86deg,N8,width4mm,uniform). Only repeat counts and schedule/IDs change. These are existing calibrated recipes, NOT an orientation-only intervention: length, strip count, width, profile and location differ. Their width-inflated envelopes fit the200x300mm sheet: A187.996461x148mm centered; B200x62.750509mm centered at(0,118)mm. Envelopes do not overlap; local hit overlap must still be checked, not invented. This may remove hardening-mediated order effects but does not preclude global elastic interaction or branch selection. Single-path controls establish actual contrasting deformation; do not replace recipes based on mixed-order results. No requirement to find a dome.

Keep thickness.77mm, E=1,nu=.33, flat initial state, no physical clamps or persistent pins, material-frame coverage, slope feedback0, multiplicative updates, native face-shared plastic-hit Voce decay beta.2231435513142097,floor0. Attempt209 growth/orthotropy: top.00028835116227798787,bottom1.4587994001079146e-5,ortho_top-.3989157893255133,ortho_bottom.960306323937336. Preserve inherited profile and explain calibration-transfer limits of widths/profiles. Reference mesh res=.01. Numerical rigid-mode removal is not a physical boundary condition.

**Fixed matrix.** Six schedules: `a_only`=A^10; `b_only`=B^10; `ab_block`=A^10 B^10; `ba_block`=B^10 A^10; `ab_alternate`=(AB)^10; `ba_alternate`=(BA)^10. Each gets `every_cycle` and `final_only`, twelve primary runs (100 sequential equilibrium stages plus six lumped stages). Use identical ordered applications within each pair; each lumped run starts flat and accumulates without intermediate relaxation. No naive strain multiplication, cold resets between sequential passes, optimizer-memory transfer, adaptive partial-loading solves, perturbation/seed escape or extra final solve. All twelve are retained even if targets match; separate lumped runs provide a direct implementation check rather than speculative deduplication. Native solver internal iterations are not physical stages.

**Numerical recipe and sensitivity.** Reuse isolated repaired binary `run/native_shape_refinement/build_4375711/bin/shell`, with refine_linear_solve=true, fixed2000-attempt warmup, native gradient and equilibrium gates5e-14, positive-pivot acceptance, adaptive=false. No C++ edits or binary replacement. Verify current binary/source provenance against the prior repair review before launch. Preselect `ab_block` pair independently of results for three diagnostics: identical repeat; tighter tolerance1e-14 on base mesh; finer mesh res=.0075 at base tolerance. These are six additional runs, not a search or retry of failed cases. Launch diagnostics only after primary cases pass acceptance. They assess the selected contrast, not certify every unrefined pair. Do not treat repeated numerical runs or mesh vertices as independent physical replicates. A changed endpoint under tightening/refinement leaves branch interpretation unresolved. Never borrow the historical HLBFGS numerical floor.

**Measurements.** Validate counts, operation order and every required full-load proof. Compare full-precision final reference coordinates, topology, bbar, material constants, both target metrics, and final hit counts. Report exact equality and relative/absolute target errors; equal history counters alone are insufficient. For cadence pairs, unexpected target differences are an implementation/scientific-confounding failure, not optimizer-path evidence. Use proper determinant+1 reference-area-weighted rigid alignment with fixed material correspondence (no scaling/reflection/remapping), full-sheet RMS, q50/q90/q99 and maximum displacement, normalized RMS by reference panel diagonal, signed displacement maps and curvature distributions. Use native JSON coordinates for endpoint calculations; intermediate VTP shapes and curvature/director fields are labeled by their serialization/estimator limitations. Energies are normalized E=1 and compared as competing minima only for equal objectives. Report area-weighted distributions, not vertex-based confidence intervals. Compare A-only versus B-only shapes without permitting a rotation to relabel material vertices.

Interpret tiny changes descriptively, not as exact equivalence. For selected ab_block, require cadence RMS to exceed both5x the measured identical-repeat RMS (maximum of the two cadence-specific repeat differences) and1e-4 of the reference panel diagonal (~.0361mm), with nonvanishing matched effect under tighter tolerance and finer mesh, before calling it numerically resolved. Inspect curvature and spatial structure as corroboration. A single repeat estimates only a limited repeatability check, not an uncertainty bound. Finer mesh is a sensitivity check, not proof of asymptotic convergence. Other pairs remain unrefined and explicitly qualified. Freeze these criteria before outcome inspection.

**Execution and safeguards.** Heavy commands on Quest compute nodes, per scientific case8CPU/32GB, native600s per equilibrium. External timeout750s per planned stage plus120s setup; Slurm4h30m for20-stage cases,2h30m for10-stage controls,30m forsingle-stage cases. At most two new independent cases concurrently, CPU-based, no GPU request. Check actual available allocation before launch and preserve the other campaign. First execute four single-toolpath controls; if accepted, execute eight mixed cases; then the predeclared six numerical checks. Stop dispatch on any numerical/infrastructure failure and consult, keeping all launched outputs and any concurrently finishing case. No failed-run restart, automatic resource extension, relaxed gates or continuation substitution. No cumulative historical budgets apply.

**Readiness.** Before scientific execution, validate schedule generation and footprint/calibration invariance, fail-closed acceptance/cadence checks, target comparison and rigid alignment. Small correctness fixtures use coarse res=.15,80x100mm sheet, short low-growth paths and two repetitions each,5warmup attempts,20s/native stage,1CPU/4GB, outer120s per invocation; they are not scientific results. Reuse existing curvature analytic fixtures with their documented limits. Independent Astra pre-execution review must bind actual script/input/binary identities and practical resource requests. No absent/stale review is permission. No package installations.

Status: preparation and independent pre-execution review complete. A-only lumped job4475512 completed and independently reverified:10 applications,1 accepted solve, gradient6.560316688121277e-16<5e-14,positive pivots,253.172s runner time. A-only sequential4475511 continued while B controls were dispatched. With one slot freed and all13 identities rechecked, B-only lumped submitted as4475879 (30m,8CPU/32GB). B-only lumped4475879 completed and reverified (248.121s,gradient4.034228823141002e-15,positive pivots); B-only sequential submitted as4476576 (2h30m,8CPU/32GB). A-only sequential4475511 completed with all10 stages accepted (2197.816s runner time); B-only sequential4476576 then completed with all10 stages accepted (2197.813s runner time). All four controls reverified and all13 frozen identities rehashed. Mixed-stage start: ab_block sequential4480278 (4h30m) and ab_block lumped4480279 (30m) submitted, each8CPU/32GB. Other mixed cases remain unlaunched. STOP-DISPATCH event: ab_block lumped4480279 failed after632.927s runner time; native termination=time_cap at the600s cooperative limit, gradient1.2796230809660088e-12 exceeds5e-14, curvature=indefinite,accepted=false. Solver raised `zigzag_sequence: equilibrium corrector failed at cycle 20, lambda=1.000000, gradient_norm=0.000000.` (formatted gradient is misleading; use full-precision proof). No accepted final state exists, so this is an unresolved numerical solve, not evidence of a different minimum or a lumping-induced shape error. ab_block sequential4480278 was left unchanged, but subsequently also failed:13 accepted stages, stage14 native time_cap, gradient3.334008958536199e-13>5e-14, curvature=indefinite,accepted=false. Its partial shapes are not a final20-application endpoint. No sequential restart is authorized. All new scientific dispatch and sensitivity runs stopped; no retry, continuation or gate relaxation performed. Researcher decision required on extending runtime/restarting the failed lumped case versus retaining the incomplete investigation. B-only cadence result in `run/sequence_order/analysis_controls/metrics.json`:exact final targets/history,RMS7.112818871007665e-05mm,max.0002699930911447072mm; both single-path controls show close agreement at tested settings. Accepted A cadence comparison in `run/sequence_order/analysis_a_control/metrics.json`: exact final targets and history; aligned area-weighted RMS.00011823290065392438mm,max.0003120009544013532mm,energy(lumped−sequential)−9.147151429512479e-23. This is close agreement for A at the tested settings, not an exact-equivalence or refined numerical-resolution claim. Preliminary accepted lumped-control comparison in `run/sequence_order/analysis_lumped_controls/`:A hits5241faces,B2858,intersection0. Proper material-correspondence aligned RMS A-vs-B=.7738304596609135mm, confirming contrasting lumped shapes but not a cadence result. The disjoint actual hit sets support the predicted absence of shared-hit hardening between A/B for this recipe. `scripts/run_sequence_order.py` reuses footprint validation and the unchanged native executable; `scripts/run_sequence_order.sbatch` provides case isolation. `calibration/analysis/analyze_sequence_order.py` compares full-precision target fields and matched geometry, reusing existing alignment/curvature helpers. `calibration/tests/test_sequence_order.py` checks recipe/count preservation, matched options, acceptance rejection, rigid alignment and target mismatch handling. Compute job4474689 (1CPU/4GB/10min ceiling) ran the two declared alternating fixtures:4 versus1 native solve, both accepted, exact final targets/history. All6 tests passed; VTK emitted a nonfatal GetData deprecation warning. Evidence: `run/sequence_order/checks/fixture_v1.log` and `run/sequence_order/fixtures_v1/`. The reference binary and three connected C++ source hashes match the recorded repair identities in `docs/reports/dome_linear_repair_review.md`; that old approval does NOT authorize this experiment. Frozen design, options and13 source/binary/helper identities: `run/sequence_order/checks/frozen_design.json`. Shell syntax and git diff whitespace checks passed. The new scope received the independent approval recorded below. Source inspection confirms general sequence mode supports multiple recipes, retains target/hit history, exports final native state and per-solve proofs, and writes intermediate VTPs. The single-recipe checkpointed never-clamped route is deliberately not used. General-sequence diagnostic state is not restartable; interrupted sequences require consultation, not replay. Pre-execution review: `docs/reports/sequence_order_preexecution_review.md`, Astra run5d54b4ad-37bb-4338-88a2-37e02a519c86, APPROVED_WITH_CAVEATS with no launch blockers. Parent read the full verdict and rehashed all13 frozen identities successfully before submission. Recorded limits: controls-first and at most2new jobs are parent-enforced; native600s is cooperative; general sequence is not restartable; comparisons/curvature figures and numerical interpretation remain to be completed; actual hit-set intersection must be checked. Initial-state equivalence rests on inspected common initialization plus matched output, not VTP hashes alone. Existing dome4467951 remains running unchanged. Normal partition is available (13idle nodes at preflight). Next authorized action: consult researcher about the failed lumped solve; preserve and inspect the already-running sequential result when available. Do not dispatch additional cases or restart without approval. The controls-first gate was satisfied; numerical failure now gates further dispatch. Submitted A-only sequential as Slurm4475511 (2h30m) and A-only lumped as Slurm4475512 (30m), each8CPU/32GB. Submission is not completion or acceptance.

While review ran, a separate plotting helper `calibration/analysis/plot_sequence_order.py` was prepared without changing reviewed code. Saved-fixture analysis passed in job4475031; plotting initially rejected a tick outside axis limits, then rendered after fixing that guard in job4475105. PNG inspection found a crowded CDF xlabel/scientific-offset and long missing-state labels; shortened labels and common CDF x limits were applied. Job4475546 rerendered all four fixture figures successfully. The CDF PDF page1 was rendered via Poppler and visually inspected: corrected labels, common axes and missing-case indicators are readable. Other PDF panels and final production figures still require rendered inspection; no color-vision simulation was performed. These are rendering fixtures only, not new solver experiments or production evidence.

---

## Current revised investigation — continuation-assisted mixed comparisons

The confirmed goal tweak supersedes the exact20-solves-versusone requirement. Researcher approved incremental load stepping for all four mixed orders and the increased sequential workload. Preserve20 physical applications per mixed case, fixed optimizer handoff and original acceptance gates. Original direct cases and failures remain historical evidence, not replacements for this revised comparison.

Primary matrix: ab_block,ba_block,ab_alternate,ba_alternate each in every_cycle and final_only modes. Reuse accepted `ab_block_final_only_continuation_v1` after identity/compatibility verification; seven fresh primary runs remain. Retain the four direct single-path controls as supporting evidence, not matched continuation controls. Do not deduplicate other final_only runs speculatively. New outputs must be exclusive and separate from prior attempts.

Native source inspection at Sim_Bilayer_Growth.cpp:2518–2523 and2652–2680 confirms sequential pathStart is the preceding target while adaptive final_only starts at initial stress-free metrics. Both install exact accumulated endpoint metrics atlambda1. No C++ change is planned. Extend only this investigation's isolated Python runner/verifier/analysis, leaving shared dome runners and running campaigns untouched.

Resource plan for review: retain8CPU/32GB and at most2new scientific cases concurrently. Based on1974s for10 substantive combined increments,20application sequential cases may take roughly11hours at similar per-substep cost (an estimate, not a guarantee). Use native1800s per substep, external20hours and Slurm21hours for sequential base/repeat/tight; external2hours and Slurm2h10m for new combined cases. For preselected finer-mesh sensitivity use external30hours/Slurm31hours sequential and external3hours/Slurm3h10m combined, same CPUs/memory, accounting for additional sparse factorization cost. All fit normal partition's48hour maximum. Submission contract: parent explicitly passes `sbatch --time=21:00:00` for base/repeat/tight sequential, `--time=31:00:00` for fine sequential, `--time=02:10:00` for base/repeat/tight combined, and `--time=03:10:00` for fine combined. The batch header's21h is only a default; never use it for fine sequential. Parent enforces max-two concurrency, stop-new-dispatch-on-failure and primary-before-sensitivity. This mapping was confirmed to the independent reviewer in supervisor reply6ade228c-d1da-40b7-8c83-0ff0c8555898. The accepted combined baseline retains its actual1hour external ceiling and is reused only as an accepted endpoint, not an equal-runtime comparison. Stop dispatch on failure; no automatic extension or failed-run retry. These are practical per-command safeguards, not cumulative budgets.

After all eight primary endpoints pass, retain six preselected ab_block sensitivity runs:repeat,tight1e-14,fine res.0075, each cadence. Preserve original RMS criterion (exceed5times measured repeat discrepancy and1e-4panel diagonal, with persistence under tightening/refinement); report tails and curvature and qualify unrefined orders. Freeze updated identities and obtain scoped Astra review before launches. Next: generalize stage-wise continuation acceptance, test full-load/rollback/count invariants with bounded fixtures, and adapt analysis to explicit continuation provenance. Launch status: Astra through codex-auto completed independent review `docs/reports/sequence_continuation_matrix_review.md` (runa3b6e052-c6ac-498e-9ad5-e092a63fc36f), APPROVED_WITH_CAVEATS,no blockers. Parent read verdict, rehashed complete continuation_design_v2.json packet and reverified the immutable reused combined baseline against generated ab_block sequence and stage verifier. Submitted ab_block sequential as4521415 and ba_block sequential as4521416, both explicit21h,8CPU/32GB. Progress: ba_block sequential (4521416) COMPLETED and ACCEPTED after 39385.44s (~10.94h), 220 accepted substeps (0 rejected), all 20 physical stages verified, terminal gradient 8.70e-18, positive pivots. Following verified acceptance and freeing of a concurrency slot, submitted ba_block combined (final_only) as Slurm 4583289 (explicit 02:10:00, 8CPU/32GB). ab_block sequential (4521415) remains running at stage 18/20. Parent enforces max-two,stop-new-dispatch-on-failure,primary-before-sensitivity and explicit time overrides. Caveats retained: rollback dynamics not exercised in coarse fixtures; substep geometry not all exported; actual work counts and numerical sensitivity require final interpretation; no resumability or guaranteed duration/memory. Existing dome4467951 remains untouched. Preparation: generalized the isolated continuation runner to four orders/two cadences and preselected variants; new `scripts/verify_sequence_continuation.py` checks every physical-stage full-load endpoint and accepted/rejected substep history. Existing combined diagnostic remains immutable; its original runner hash is historical, not falsely relabeled as current. Coarse numerical fixture job4519380 (1CPU/4GB,120s subprocess limits) accepted four application-wise blocks/44substeps and combined11substeps; final top/bottom target metrics,reference geometry,bbar/material are exactly equal. Six current tests pass including retained fixture verification and synthetic bad-stage/partial-load/rollback/curvature rejection; logs `run/sequence_order/checks/continuation_tests_v2.log` and `continuation_fixture_v2.log`. Analysis now selects the continuation verifier only when adaptive loading is recorded; explicit primary-result selection/reuse integration and review packet remain pending.

## Prior adaptive-load diagnostic — researcher authorized

After the runtime-only retry failed, the researcher requested adaptive solving and explicitly selected **Adaptive load stepping**, not adaptive optimizer handoff. This authorizes a separate continuation-assisted test of the failed ab_block lumped case; it does not convert it into a strict one-optimization baseline or authorize restarting the sequential case/remaining campaign. The fixed-matrix failures above remain part of the evidence.

Question: can gradual introduction of the same accumulated20-application target reach an accepted full-load equilibrium where direct loading stalled? Success gives an accepted candidate under a different numerical loading path, not a matched cadence comparison, proof of another minimum, or evidence of physical sequence memory. Failure leaves the endpoint unresolved. Keep A/B, ordered accumulation, flat start, mesh, material, fixed2000 handoff, repaired binary,5e-14 gates and positive-pivot requirement unchanged. The full target is accumulated first; interpolate from initial stress-free metrics to that target, restoring last accepted state on rejection. Use existing continuation recipe:initial/max step.1,min step1/64,growth2,max native substep retries8 (existing algorithmic safeguard),native1800s per substep. External3600s total and Slurm70min,8CPU/32GB; one case only. No resource extension/outer retry or additional campaign dispatch on failure. Partial-load states are not scored.

New isolated runner `scripts/run_sequence_order_continuation.py` SHA25657c35b4b7c3d80bf066ebffe7c0580c06ef77c44068c77067f11d62e4fed8151 copies original failed sequence bytes to `run/sequence_order/ab_block_final_only_continuation_v1` (exclusive output). Reuses unchanged `scripts/run_dome_continuation.py::verify(out, cycles)` SHA256b723c753ca3160c28008b87dfde471a1694c785ef0f9cd2a4f5664b6da79248f: exact20 applications, final_only cadence, rollback-aware accepted-load progression, all accepted-substep gradient/curvature gates, matching proof streams and full-load lambda1 final endpoint. It does not call the direct-solve verifier that requires exactly one substep. Record number of accepted/rejected substeps and terminal target provenance; preserve original failures. Existing continuation evidence is supporting only, not approval for this input. New2 mock-only launch tests passed (`run/sequence_order/checks/continuation_unit_tests.log`), test SHA256b13bc2324b1caa8e8d433c596c80ae7e808db5d8181b3f4bcdf9afad3fc3ac28; no new solver fixture or scientific solve yet. Scoped Astra review `docs/reports/sequence_order_continuation_review.md` (run4e5ca80f-f776-43ae-89d1-3a9103d607bc) returned APPROVED_WITH_CAVEATS,no blockers. Parent inspected the full verdict and rehashed all13 original identities plus new runner/test/verifier/input; all matched. Submitted this one diagnostic as Slurm4495662,70min8CPU32GB,explicitcwd/DHD Python. Frozen runner/binary and unrelated dome4467951 remain untouched. Caveats:1800s native limit resets per substep;3600s outer ceiling includes all substeps; count actual accepted/rejected optimizer invocations (equilibrium_solves=1 denotes a physical-cycle block, not one optimizer invocation); exactlambda1 required; no resumability, extension or wider campaign authority. Outcome: Slurm4495662 COMPLETED, runner1974.195s (~32.9min), independently replayed continuation verifier passes. All11 optimizer invocations accepted,0 rejected:lambda .1,.2,...,.9999999999999999,exact1. The eleventh invocation is a floating-point-sized terminal correction (0optimization attempts,verification still performed), not a meaningful extra10% step. Terminal gradient5.902563126326728e-17<5e-14,positive pivots,normalized energy2.7687842759969545e-12. Compute inspection4502137 verified full reference/material/topology equality with controls, exact abar_top/abar_bottom equality to the disjoint-hit composition of A-only/B-only targets, and final hit counts equal to their sum; saved `run/sequence_order/ab_block_final_only_continuation_v1/target_inspection.json`. Thus continuation reached an accepted candidate for the intended combined target where both direct attempts failed. No accepted mixed sequential endpoint exists, so sequence-versus-lumping shape differences remain unidentifiable. The successful path also changes cumulative work and warmups; it is not proof that dynamic step reduction helped (no trials were rejected), nor global optimality. This one authorized diagnostic is complete. No other case launched. Next researcher decision: whether to amend the broader comparison to continuation-assisted solves for sequential and combined targets; that would replace the exact20-solves versusone-solve design and requires an explicit updated contract.

## Authorized runtime-only restart amendment

After inspecting failed lumped job4480279, the researcher selected “Retry with 30-minute solver limit” in the decision dialog: restart only this failed case into a new output directory after scoped review, keep model/start/acceptance unchanged, no extra optimization stages, and resume the remaining planned cases only if it passes. This supersedes the no-retry rule for this one case only; it does not extend the native limit for other cases.

New isolated helper `scripts/retry_sequence_order_lumped.py` (SHA256 cc4a13733ba305ebdb38cc990e2cd0091d1d63730af5b515f5e360b0a0d44a35) preserves the frozen runner and running sequential job4480278. It copies the failed sequence byte-for-byte, changes only native_seconds600→1800, uses external2070s and a planned40min Slurm allocation,8CPU/32GB. Output: `run/sequence_order/ab_block_final_only_retry1800` (must not exist). Original failure retained. No continuation or resume-from-partial-state; one full-load solve from the same flat start. New helper reuses the unchanged `verify()` acceptance contract. Two mock-only wiring tests check sole option change, exclusive output, unchanged sequence and fail-closed timeout recording; evidence `run/sequence_order/checks/retry_unit_tests.log`, test SHA25617187eb40c838ff08dca95d0a16fad5a65cd84f86a803fa6e4299b71393bc446. No solver rerun in these tests. Scoped review `docs/reports/sequence_order_retry_review.md` (run31a52750-4cd8-4444-8d60-8b87c7ff9098) returned APPROVED_WITH_CAVEATS with no blockers. Parent read full verdict, confirmed all13 original identities plus retry/helper/test/input identities, and submitted only the approved lumped retry as Slurm4486577:40min,8CPU/32GB,explicit project cwd,DHD Python,`PYTHONOPTIMIZE` unset. Original sequential4480278 and lumped4480279 outputs unchanged. Review caveats retained: native timing cooperative, resource/concurrency checks procedural, analysis must explicitly disclose runtime exception, no scope beyond this retry. Existing dome4467951 remains untouched. Subsequent discovery: sequential4480278 also failed at stage14; this amendment authorizes no restart of that case. Remaining campaign dispatch stays stopped regardless of the lumped retry until the sequential failure is addressed with the researcher. Retry outcome: Slurm4486577 FAILED after1828.066609s. Native termination=time_cap,accepted=false,curvature=indefinite,gradient1.2611603578607572e-12>5e-14,energy2.7924977433414335e-12,299Newton attempts. Versus the600s attempt gradient1.2796230809660088e-12, threefold runtime improved the terminal gradient by only~1.44%; more time alone did not resolve this case. This is evidence of numerical stagnation under the current solver recipe, not an accepted local minimum or a lumped-shape comparison. No retry endpoint may be scored. Both mixed branches remain incomplete; remaining6 mixed cases and6 sensitivity cases were not launched. No active sequence-order jobs remain. The researcher must choose a bounded solver diagnosis/new recipe or retain an incomplete investigation before further scientific execution. Analysis adaptation for an accepted retry is unnecessary because no retry was accepted; preserve all failed attempts alongside the four accepted controls.

## Historical v2 objective (preserved)


Determine whether differences caused by warm starting, target-metric accumulation, and stroke grouping exceed numerical repeatability and discretization error, and identify regimes where those choices produce substantially different equilibria.

This is a forward-simulation study. No inverse path optimization is performed.

## Frozen baseline

- geometry: `rectangle`
- half extents passed to solver: `lx=0.127 m`, `ly=0.1524 m`
- physical output footprint: approximately `254 x 304.8 mm`
- thickness: `0.6 mm`
- Putong toolpath: `lv=140 mm`, `alpha=9.13 deg`, `n_strips=10`, `width=10 mm`
- baseline growth: top `2e-5` per active stroke, bottom `0`, `ortho=0`
- hardening: none
- minimizer: HLBFGS
- seeded escape: disabled during optimizer-sensitivity comparisons
- final certification: enabled
- thread count: `OMP_NUM_THREADS=8`
- execution: strictly serial; no two solver cases overlap
- storage: `run/sequence_ablation_v2/`, never `/tmp`

Every case records its JSON input, full command, stdout/stderr, executable checksum, source revision/status, environment, wall time, cycle summary, final VTP, and derived metrics.

## Definitions

### Numerical noise floor

Variation among five identical executions with the same binary, mesh, thread count, configuration, and deterministic seed.

### Equivalent geometry

Two point sets are compared after proper Kabsch alignment with determinant `+1`. Report:

- aligned RMS vertex distance;
- aligned maximum vertex distance;
- relative RMS normalized by panel diagonal;
- z-span after alignment;
- intrinsic/local curvature summaries.

### Substantial difference

A difference is substantial only when all conditions hold:

1. aligned RMS exceeds `5 ×` the identical-run RMS noise floor;
2. aligned RMS exceeds `1e-4` of the panel diagonal;
3. the effect persists with the same sign/trend on the two finest meshes;
4. both states converge and are certified minima, or are explicitly classified as different stability outcomes;
5. normalized energy or curvature differences also exceed their repeated-run noise floors.

Differences below the repeatability threshold are numerical noise. Differences above repeatability but failing mesh convergence are discretization-sensitive, not physical evidence.

## Required instrumentation

1. Curvature fits use SI coordinates and return `m^-1` without an extra `1e3` multiplier.
2. Quadratic fits report RMSE and `R²`.
3. Geometry comparisons use proper rigid alignment.
4. Every cycle summary records whether minimization occurred, HLBFGS code, iterations, final gradient norm, convergence, and recomputed energy.
5. Unrelaxed grouped cycles are labelled `pending`, not `final` equilibria.
6. Cold-start reset occurs before mapping/initial-state diagnostics.
7. Target metrics are compared face-by-face before geometric comparisons.
8. Explicit single-stroke schedules select actual strip indices rather than representing inactive strips with zero growth.

## Metric-update schemes

Three distinct schemes are required:

1. **Multiplicative**
   \[
   \bar a_{n+1}=T^T\bar a_nT.
   \]
2. **Recursive linearized**
   \[
   \bar a_{n+1}=\bar a_n+\delta T^T\bar a_n+\bar a_n\delta T.
   \]
3. **Reference-additive linearized**
   \[
   \bar a_{n+1}=\bar a_n+\delta T^T\bar a_0+\bar a_0\delta T.
   \]

Target-metric differences are evaluated before optimization. Geometry comparisons for constitutive schemes start from the same initial current geometry.

## Sequential case matrix

### A. Repeatability

Five identical baseline executions:

- `res=0.03`
- `tol=1e-12`
- 12 paths
- multiplicative metric
- warm start

Purpose: establish geometry, energy, curvature, timing, and target-metric noise floors.

### B. Mesh and tolerance convergence

Mesh sweep at `tol=1e-12`:

- `res={0.06, 0.03, 0.015}`

Tolerance sweep at `res=0.03`:

- `tol={1e-10, 1e-12, 1e-14}`

The baseline `res=0.03`, `tol=1e-12` result is reused from Stage A; it is not rerun concurrently.

### C. Warm-start sensitivity

At `res=0.03`, `tol=1e-12`, 12 paths:

- low strain `2e-5`: warm and cold;
- moderate strain `1e-4`: warm and cold.

Each final state is certified. Cold starts use the same deterministic initial perturbation policy as their matched comparison if a perturbation is required to leave a symmetric flat stationary state.

### D. Metric accumulation

At `res=0.03`, `tol=1e-12`, 12 paths:

- strain `{2e-5, 1e-4, 5e-4}`;
- schemes `{multiplicative, recursive_linearized, reference_additive_linearized}`.

An additional noncommuting case uses `ortho=0.8` with alternating `0 deg/45 deg` path rotations. Run forward and reversed order under multiplicative composition and compare final target metrics before solving. The previously tempting `0 deg/90 deg` pair is excluded because those orthogonal tensors commute.

### E. Exact stroke segmentation

Use the ten explicit straight segments of the Putong zigzag and three repeated paths. Compare:

1. complete path ×3, minimize after each complete path;
2. path-major segments, minimize after every segment;
3. path-major segments, minimize after every three accumulated segments;
4. stroke-major: same stroke ×3, then minimize;
5. reversed stroke-major order;
6. accumulated identical final target metric followed by one solve from a common initial geometry.

Order and minimization cadence are varied independently. Final target metrics and hit counts must match before schedules are treated as equivalent loading.

## Analysis outputs

For every equilibrium:

- convergence status and final gradient;
- energy and energy difference from matched baseline;
- Hessian minimum eigenvalue;
- aligned RMS/max geometry difference;
- z-span;
- quadratic principal curvatures, RMSE, and `R²`;
- area-weighted local mean/Gaussian curvature;
- positive/negative Gaussian-curvature area fractions;
- facewise maximum and RMS target-metric differences;
- iterations, evaluations when available, solver-only time, and total wall time.

## Acceptance criteria

- All cases execute sequentially with no overlapping solver process.
- All inputs and outputs are persistent and traceable to one binary checksum.
- Failed or non-converged cases remain in the dataset but are excluded from equilibrium equivalence claims.
- Curvature units pass an analytic paraboloid check.
- Identical-run geometry comparisons pass Kabsch invariance under an injected rigid transform.
- Metric schemes pass aligned isotropic closed-form checks.
- Segment schedules intended to be load-equivalent have facewise target-metric error below `1e-12` relative.
- Final report classifies each observed difference as numerical noise, discretization-sensitive, optimizer-path-dependent, constitutive, or certified branch-changing.
