# Restart summary — shell optimization

Latest update: parallel trajectory pilot **7654342 completed**, composed pilot **7722208 completed**, and L-BFGS threshold sweep completed (2026-09-28).
Authoritative final report: `docs/nonconvex_solver_final_optimization_report.md`.
Detailed history: `docs/nonconvex_solver_benchmark_plan.md`. This is a navigation/
handoff note, not permission to restart completed fits. **All approved phases are closed.**

**Parallel pilot results (job 7654342):**
- Two four-thread hybrid cells completed and **accepted** in **328 seconds total**
  (5m28s wall clock, 7m52s CPU) on 4 CPUs, well within the 45-minute envelope.
- Hessian time dropped from **125.5s → 32.1s** ((+) start) and **106.2s → 28.1s** ((−) start),
  giving a **3.91× / 3.77× acceleration** on the Hessian oracle (median ~1.63s → ~0.42s).
- Total charged cell wall-clock time dropped:
  - Positive start: **251.8s → 141.7s** (**1.78× speedup**)
  - Negative start: **245.7s → 132.9s** (**1.85× speedup**)
- Negative start reached the **exact same equilibrium basin** to **0.11 µm RMS**
  (0.38 µm max) aligned difference, with relative energy difference **-8.45e-12**.
- Positive start reached an accepted local minimum with slightly lower energy
  (-5.38e-05 relative), 10.3 mm RMS difference, reflecting a nearby stable buckling mode.
- Both passed independent native gradient norm **≤ 5e-14** and positive LDL pivots.
- The Hessian share of total solver time fell from **46.6% → 21.9%**.
  The largest remaining cost is now **Energy / gradient evaluations** (36.4%, ~100s).
- Evidence root: `run/nonconvex_solver_benchmark/parallel_hessian_pilot_7654342/`.

**2026-09-27 solver integration update:** qualification **7651813 passed** in
172/360 allocated seconds on four CPUs. Explicit `hessian_threads: 4` now reaches
the benchmark Newton kernel and is part of its checkpoint/hybrid policy. Two
full-size Newton updates matched serial numerical state exactly; fresh-process
SIGKILL/disk recovery after update one matched uninterrupted parallel execution.
Four toy handoff/progress boundaries, thread/runtime-policy mismatch rejection,
resource-floor and completed-cell barriers, and unavailable-team refusal passed.
Six shell fixture updates total, from a **synthetic** handoff; **no convergence
fits or end-to-end speedup claim**. Production/default serial behavior unchanged.
Evidence: `run/nonconvex_solver_benchmark/parallel_solver_gate_7651813/gate.json`.
The qualification phase is closed; the subsequent pilot is tracked above.

**2026-09-27 kernel update:** user-requested opt-in OpenMP face evaluation is now
implemented in the benchmark local-dihedral kernel; default remains one thread,
production and solver launch policies unchanged. Small checks **7649187** passed
12 free/clamped, flat/curved, 1/2/4-thread cases. One saved full-size ABC state
(**7649255**) gave bitwise serial parity and mean warm Hessian times **1.401 /
0.714 / 0.381s** at 1/2/4 threads (two calls each; 4-thread kernel speedup **3.68×**).
No fits or end-to-end speedup demonstrated. Both bounded check jobs completed;
see the plan's “Opt-in parallel local-dihedral Hessian” section for evidence,
resource accounting, memory cost and remaining qualification limits.

## Goal and boundaries

Reduce time to an accepted local equilibrium of a fixed final-growth shell
objective. Production remains unchanged; all new solver options are isolated
benchmarks. No global-optimality, uniqueness, physical-calibration or mesh-
convergence claim. Two sign-paired starts are dependent probes, not replications.

Acceptance is unchanged: `gradient_target`, independently recomputed native
gradient norm **≤5e-14**, and **positive unshifted restricted LDL pivots**.
The looser 1e-13 residual-candidate gate cannot promote capped/failed fits.
No retries, cap resets, extensions or reopening completed cells without approval.

## Current problem

User recipe: `run/zigzag_abc_preview/sequence_ortho_minus_0p5.json`.
Original +0.5 recipe preserved as `sequence.json` in that directory.

- A ×2: central, 140 mm segment height, 9.13°, 10 strips, 8 mm width,
  0° placement; inherited center-peak profile (end ratio 0.4, power 2).
- B ×3 / C ×3: 200 mm segment height, 2.86°, 8 strips, 4 mm width,
  rotated 90°, shifted in material coordinates by v=+120 / −120 mm;
  uniform profile.
- Base top growth **0.0015**, bottom zero; **ortho −0.5 everywhere**.
  Native directional increments are q*g*(1+ortho) and q*g*(1−ortho):
  0.5*q*g and 1.5*q*g, not negative growth.
- Voce decay: beta=0.223143551, floor=0, shared per-face plastic-hit history.
- Accumulate all eight passes in order, with profiles/hardening and
  multiplicative metric updates; optimize **only the final target**, not each pass.
- Carried-over sheet **254 × 304.8 mm**, thickness **0.6 mm**, E=1, nu=0.33.
- Same coarse mesh: **5427 vertices, 10560 triangles, 15986 edges, 32267 DOFs**;
  nominal spacing 3.81 mm. The 4 mm strips are barely resolved; spatial accuracy
  is not established. B/C footprint clearance to sheet edges is only 0.425 mm.
- Starts: opposite y-axis cylindrical bends, maximum displacement **1.8 mm**,
  consistent directors; not different loading cases.

No-fit target gate **7624619** passed in **130/600 one-CPU seconds**:
8450 ordered hits on 3001 faces, max six hits/face; all hits/profiles/history and
all eight cycle metrics independently checked. Maximum relative tensor mismatch
3.836e-16. Peak principal natural-length growth **0.83254%**. Six-start native/AD
and Hessian/HVP checks passed. This does not establish physical calibration.

Bundle: `run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/bundle`.
Identity: `9f79ed808b6dd6ccca171f5c72538837fe6400653f0f6b4632dcc407e5e259f0`.
Geometry preview: `run/zigzag_abc_preview/geometry/zigzag_abc_footprints.png`.

## Results on this objective

Fixed hybrid: **2000 live L-BFGS attempts**, then local-dihedral CHOLMOD Newton.
Handoff preserves current state and cumulative counters; it does not restart a
failed/capped/accepted endpoint. Shared ceilings: 100000 attempts, 200000
native evaluations, 10000 Hessians. Original cell envelope 1200s, optimizer
1140s, watchdog 1190s. Stop earlier when accepted.

| Start | Exact-Hessian baseline | Guarded-Hessian candidate | Newton Hessians, both |
|---|---:|---:|---:|
| Positive | 251.808 s | 288.408 s | 77 |
| Negative | 245.714 s | 253.477 s | 65 |

**All four fits accepted.** Same-start final coordinate files are byte-identical.
Energies: **2.763029800569301e-11 / 2.7630298005558528e-11**.
Independent gradients: **2.2901687569889483e-14 / 4.459770647471914e-15**;
positive restricted pivots. These endpoint values match baseline and guarded runs.

- Baseline **7629263**: **539/2700 allocated one-CPU seconds**, including
  integration/reporting. Hessian **46.57%**, energy/gradient **22.84%**, numerical
  factorization **9.42%**, symbolic+solve **8.31%**. Hessian median ~1.63s/call.
- Guarded pilot **7642652**: **824/3000 seconds**, including build/checks/reporting.
  **Zero Hessian reuse attempts**. Every eligible cached reduced model was
  invalidated by a changed restricted DOF selection (**48 / 46 refreshes**).
  Thus no Hessian evaluations were saved; there was no end-to-end improvement.
- Do not attribute all timing increases to guard overhead. Positive L-BFGS
  handoff time changed **50.552→80.990s**, before guarded Newton began;
  negative changed **71.992→68.745s**. Different execution conditions/builds
  and larger checkpoints are possible contributors.

## What was optimized / tested

Earlier kernel work: cached sparse assembly saved ~4%; local-dihedral derivatives
saved ~31% of Hessian assembly time; a separate stretching AD calculation gave
no useful benefit. These improvements do **not** exhaust kernel optimization.
Parallel assembly subsequently passed the bounded kernel checks summarized above;
bounded solver/restart qualification also passed (7651813), while full-trajectory
parallel evaluation and other alternatives remain open.

### Symbolic reuse

No-fit qualification **7635939**: **190/600 seconds**, passed. Opt-in
`Config::reuseNewtonSymbolic`, false by default. Exact CSC structure and exact
restricted DOF order must match. Numeric factorization remains fresh. Symbolic
state is process-local; restart reanalyzes. Warm checks cost ~0.0012–0.0017s
versus ~0.223–0.256s fresh symbolic setup on three saved shell states; all
nine shifted diagnostic solves matched fresh solves exactly.

But symbolic analysis was only **7.55%** of baseline charged time, and matrix/
gauge changes are frequent. No scientific symbolic-reuse pilot has been run.

### Guarded Hessian reuse

No-fit qualification **7637624**: **171/600 seconds**, passed. Opt-in
`Config::guardedHessianReuse`, false by default:

1. Only a fresh, unshifted, full accepted Newton step seeds reuse.
2. Reuse at most once, and only with identical restricted DOF ordering.
3. Evaluate current native energy/gradient and numerically factor every attempt.
4. A rejected stale full-step trial is charged, restores the current state and
   clears the model; the next budgeted attempt refreshes. Numerical failures
   remain terminal. Fresh-model line search is unchanged.
5. Persist matrix, restriction and reuse age together; reject age resets and
   cross-recipe resumes. Clear terminal caches; final certification stays fresh.

Eigen/CHOLMOD toy guards, caps, disk restore and symbolic-option composition
passed. Synthetic full-size cached checkpoint was **51,317,092 bytes**, with
**2.671s save/load/restore**. Its overhead must be included in performance.

Integration pilot kept **symbolic reuse OFF** to isolate Hessian reuse.
Actual worker handoff tests passed at before-target, after-target, after-commit
and progress boundaries. All four tested toy pauses happened with cache empty;
cache-live restoration was exercised separately in engine disk/synthetic tests.
Do not claim full-shell cache-live SIGKILL or arbitrary mid-step recovery.

## Main finding / next decision

`ShellBenchmarkProblem::freeDofs` rebuilds a QR-pivoted rigid-body gauge from the
current geometry. The selected coordinate DOFs moved at every eligible reuse
opportunity in this pilot. Caching the **reduced** Hessian under an exact-map guard
therefore provided no reuse.

Possible next designs (NOT authorized or tested): cache the full-coordinate
Hessian and re-restrict it consistently, or qualify a stable gauge with rank/
conditioning safeguards. Do not simply disable the guard. Both alter numerical
behavior and need fresh qualification, identity/restart tests and an approved
budget before fits. Symbolic reuse can eventually be combined, but is not the
main cost. Mesh convergence and tolerance-efficiency studies remain deferred
at the user's request, not completed.

## Why we changed problems

The earlier six-cycle benchmark used top growth **0.005 per hit**, versus
**0.00012** in the original nested zigzag examples (41.7×). Its local accumulated
isotropic growth reached **3.03775%**. Its two accepted hybrid runs took ~17.6
minutes/sample and 356–371 Newton updates. That is a different objective, not
an estimate of runtime for the original mild-growth examples. Current improvement
to ~4.1 baseline minutes came mainly from fewer Newton updates, not faster kernels.

## Evidence and implementation

Run root: `run/nonconvex_solver_benchmark/`.
- `profiled_zigzag_gate_7624619/{gate.json,target_verification.json,bundle/}`
- `profiled_zigzag_pilot_7629263/{status.json,protocol.json,screen/,audit/}`
- `symbolic_reuse_gate_7635939/{gate.json,completion.json,tests.log}`
- `guarded_hessian_gate_7637624/{guarded_gate.json,completion.json,tests.log}`
- `guarded_hessian_pilot_7642652/{launch_gate.json,status.json,protocol.json,screen/,audit/comparison.json}`

Key code:
- `test/diagnostics/{NonconvexBenchmark.hpp,BenchmarkNewtonFactorization.hpp,ShellBenchmarkProblem.hpp}`
- `test/diagnostics/{PrepareProfiledZigzag.cpp,QualifySymbolicReuse.cpp,QualifyGuardedHessian.cpp}`
- `test/diagnostics/{ConvergenceBenchmark.cpp,HybridBenchmarkTransition.hpp,QualifyHybridShell.cpp}`
- `python/{qualify_profiled_zigzag.py,qualify_symbolic_reuse.py,run_profiled_zigzag_pilot.py}`
- `python/{report_profiled_zigzag_pilot.py,report_guarded_hessian_pilot.py}`
- launchers under `scripts/nonconvex_benchmark/`.

Actual guarded worker: `guarded_hessian_pilot_7642652/convergence_worker`.
SHA256 `c72a412995b2517c5c6b27cedc2d01ef3c31136edae0f2340cdaabcf0ea9e8dc`;
runtime `b167cdb2bbe447d746d826b2fa61718413f598e79ba69bee31205852733b272a`.
Baseline worker: `convergence_prepare_7592200/convergence_worker`, SHA256
`abb52ac5be1f502bc245862b2d7f5f096ad9ff9f6ae631931aaa80634d41f04b`.
Never substitute qualification-binary identities for these scientific workers.

## Restart checklist

Read this note and the authoritative plan. Inspect saved evidence before proposing
changes. Preserve all failed/capped/completed trajectories, frozen sources and
checkpoints; do not clean or overwrite them. Production unchanged. Current job
status is recorded at the top; check scheduler/application evidence before any
continuation. Closed historical budgets authorize no further fits. New designs,
retries or extensions need approval. The older guarded-comparison figures still
have pending visual review; no figure-quality certification follows from job success.
