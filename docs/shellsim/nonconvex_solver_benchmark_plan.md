# Nonconvex shell solver benchmark — protocol and implementation

Status: both screens complete; matched pilot **7386943 ended with four time caps
and zero accepted minima**. Its saved-timing audit identifies the current bottlenecks.
Production inputs and operational gates
verified in 7294450 (**26/26 tests**). Post-screen maintenance passed 29/29 tests
in 7363968; the latest v4 candidate passed **31/31 tests** in 7380235.
Production solver paths are unchanged. The original development/diagnosis
allocation used **28m16s / 30 minutes**. The separately approved early-Cholesky
allocation used **14m28s / 15 minutes**.
Screen **7296605 completed** on qnode0025 in **1h49m58s**, exit 0.
All **24/24 fits reached `time_cap`; 0/24 met acceptance**. All endpoint curvature
checks were indefinite. Execution completion is not optimizer convergence. This is the
broader benchmark following the [verified Hessian/Newton pilot](hessian_convergence_check.md),
not a continuation or rerun of its completed fits.

## Objective

Reliable, faster minimization of the SAME final discrete shell energy. Separate:
1. Local reliability/speed: time to the common residual/admissibility checks.
2. Basin search: lowest accepted energy found versus cumulative compute across
   identical structured starts. No global-optimality claim without certification.

The successful Newton pilot involved two closely related, nearly converged
cycle-2 states. It is insufficient to select a production solver. Its measured
times and older L-BFGS restart times used different execution contexts and are
NOT a matched speed comparison.

## Initial candidate set

| Candidate | Implemented development configuration | Current readiness |
|---|---|---|
| Native-kernel HLBFGS adapter | Existing 10-pair M1QN3 and energy correction; raw variables/energy; native absolute residual | Original-kernel trajectory parity across memory wraps and safe-boundary continuation tested; wrapper safety/verification differences disclosed below |
| Nonlinear CG | Standalone PR+ with descent restart and strong Wolfe; explicit variable/energy scales | Analytical fixtures, descent/Wolfe, failure handling and continuation tested; not HLBFGS's PCG switch |
| Regularized sparse Newton | H once/base state, recorded diagonal shift, sparse solve and original-energy Armijo | Indefinite/off-diagonal-curvature fixtures, derivative checks and continuation tested |
| Trust-region Newton-CG | Cached H, Steihaug negative curvature, diagonal metric and bounded inner solve | Rejection rollback, cache reuse/restoration, curvature and cap handling tested; no repeated element-Hessian HVP assembly |

L-BFGS-to-Newton hybrid is a subsequent candidate, not an extra arm in the initial
screen. Choose its switching rule on development fixtures before a separate
confirmation, not retrospectively on screening outcomes. A near-gate L-BFGS
memory restart is also an explicit recipe variant, never disguised as resumption.
No JAX port, installation, Adam or high-dimensional derivative-free sweep is
included in this initial benchmark.

## Preconditions — before scientific timing

- Same native float64 energy and constraints, fixed mesh/material/target tensors;
  verified analytic/TinyAD derivatives and finite initial oracles per comparator.
- Audit scaling and globalization per method. Preserve the unchanged native
  L-BFGS baseline; any explicit nondimensionalization is recorded as part of a
  challenger's recipe. Comparing different preconditioners/scales compares whole
  solver configurations, not an isolated algorithmic effect. Uniform iteration
  caps or untreated default step sizes do not establish fairness.
- Use manufactured nonconvex fixtures and the existing cycle-2 diagnostic states
  for development checks only. Do not relaunch completed historical fits or use
  the performance panel to tune individual methods. Freeze parameters after a
  separately bounded, equally documented development process.
- Prove non-descent rejection, line-search/TR rollback, indefinite-Hessian
  handling, finite checks, cap accounting, and native-gradient acceptance.
  Regularizing the STEP MODEL never changes the energy being minimized or the
  original Hessian used for end diagnostics.
- Exercise atomic interruption/restoration for EACH solver on small fixtures.
  Save full physical state, target tensors, constraints, source/config identities,
  cumulative budgets and best/latest/finished-case status. L-BFGS requires its
  curvature memory/scaling and first-step state; CG its previous gradient and
  direction; Newton/TR their shifts/radii and inner-solve/globalization state.
  Checkpoint only at a documented safe boundary or preserve line-search state.
  Test continuation and identity-mismatch rejection. Reloading positions alone,
  or empty-memory L-BFGS after interruption, is not verified continuation.
- One pinned CPU per timed run; explicitly limit OpenMP, BLAS AND TBB. Same
  build/precision and recorded hardware. Include oracle, Hessian, preconditioner,
  factorization, line-search and end-diagnostic costs; record build/setup apart.

Core numerical barriers now have targeted regression coverage; operational
resumption and scientific-screen gates below remain open. The initial
build/development-check ceiling is **30 CPU-minutes** (coding itself is not
compute usage), with no production sweep if a prerequisite fails. Record per-method
development cases/settings and freeze them before the screen. If more compute
is needed to complete implementation validation, ask rather than silently
extending this allowance.

## Proposed first-screen ceiling — execution remains gated

**Four configurations x six starts x one coarse final objective = 24 new fits.**

Use the complete six-cycle broad-to-central g=0.005 recipe at the existing
32,267-DOF coarse mesh, no hardening. Construct and independently verify its
FINAL target tensors without replaying optimization. The existing cycle-2
archive is not this final objective and must not be mislabeled. Save immutable
full-precision starts and targets before any method runs.

Six deterministic sign-paired starts: positive/negative cylindrical bending
about each plate axis and positive/negative twist. Proposed common amplitude:
three sheet thicknesses; exact geometry/director construction and admissibility
checks must be fixed in the input manifest before launch. These are new
fixed-final-load solves, not reruns of previously completed continuation fits.
No solver gets a private start or extra perturbations. The starts are dependent
probes of one objective, not six independent loading problems.

Proposed per-fit cap: **300 seconds**, one CPU, including verification/checkpoint
I/O and end diagnostics. Reserve up to 30 seconds of this for final checks;
stop optimization by 270 seconds, earlier if sufficient. If diagnostics exhaust
the remaining allowance, report unresolved rather than overrunning the cap.
Total screening ceiling: **120 CPU-minutes**, excluding separately bounded
implementation/build tests. No automatic retries, cap resets or extensions.
Checkpoint continuation, once tested, must respect each original cumulative cap.
Run in a predeclared interleaved order; do not pool incomparable node/thread timings.

Use the current tight native residual gate **1e-13**, internal target **5e-14**,
unchanged across methods. Keep normalized/Hessian diagnostics as additional
reported quantities; do not silently substitute an uncalibrated stopping norm.
Common endpoint checks: independently recomputed native energy/gradient,
constraint satisfaction, finite/nondegenerate geometry, and unmodified-Hessian
curvature diagnostics. Report residual acceptance separately from positive,
negative or unresolved curvature. Negative-curvature stationary states are not
supported minima. Self-contact is not silently added to or removed from the
stated model; any intersection flags are disclosed consistently.

A five-minute cap is a screening recipe, not a claim of full convergence or best
achievable performance. If everything caps, report that outcome; do not rank
underconverged energies as accepted optima or automatically spend more compute.

## Outputs and decision rules

For every planned cell, retain all failures and termination reasons, counters,
resource use, native-gradient and energy curves, and selected/final checkpoints.
Keep lowest-energy residual-accepted state separately from latest resumable
state; preserve its curvature qualification. No selecting a low-energy state
that fails acceptance. Plot complete matched-start outcomes, not just successes:

- acceptance count, caps/failures and unresolved-curvature count;
- paired time to verified acceptance; unsuccessful runs remain censored/failures,
  not dropped from the timing summary;
- lowest accepted energy found versus cumulative compute, separately per solver;
- full per-start endpoint energies, residuals and curvature status;
- proper rigid-aligned shape comparisons when objective/mesh match, with no
  reflection or material relabeling. Local residuals are not true shape errors.

No winner based only on fewest iterations or best single run. Report speed,
acceptance and energy together; an empirical lowest observed energy is not a
known global reference. The same basin-search start set applies to all methods.

## Confirmation before production adoption

Freeze shortlisted recipes, then evaluate on distinct held-out loading/growth
configurations and finer meshes. Do not use those outcomes to retune methods.
Recipe/trajectory, not individual vertices or time steps, is the grouping unit.
The one-objective screen cannot establish reliability over all configurations.
Specify that confirmation budget separately. Solver agreement does not resolve
the previously observed mesh/loading-field sensitivity: verify target-field
consistency before interpreting cross-mesh changes as equilibrium error.
Only then decide whether a solver
or hybrid should replace production L-BFGS. All current production settings and
prior evidence remain untouched.

## Implementation record

New isolated files:
- `test/diagnostics/NonconvexBenchmark.hpp`: step-wise native-kernel L-BFGS,
  nonlinear PR+ CG, shifted sparse Newton, cached-Hessian Steihaug TR-CG;
  explicit state, counters, candidate selection and checkpoint API.
- `test/diagnostics/ShellBenchmarkProblem.hpp`: native energy/gradient and TinyAD
  Hessian adapter with physical constraints and current-state coordinate gauge.
- `test/testshell/Test_NonconvexBenchmark.cpp`: analytical, corruption/failure,
  continuation and small-shell integration regression tests.
- `scripts/benchmark_solver_tests.sbatch`: isolated source/build, development
  evidence and selected regression tests only; does not submit the screen.

No production solver or external HLBFGS source was edited. The L-BFGS adapter
uses existing native update and More–Thuente kernels, memory=10, M1QN3 and energy
correction=3. Its healthy finite trajectory MUST match the original HLBFGS across
memory wraparound. The benchmark wrapper adds common initial/final native-gradient
verification, strict counters, nonfinite/curvature guards and failure rollback.
These wrapper-level safety/stopping semantics are explicit differences from the
monolithic production entry point, not a claim of identical failure return states.

PR+ uses c2=0.1 strong Wolfe with descent restart; the native L-BFGS kernel keeps
c2=0.9 and 20 line evaluations. Newton tries the unshifted scaled Hessian first,
then a diagonal shift starting at 1e-6 times the largest absolute scaled-Hessian
entry (floor 1e-12), multiplying by 10, at most 12 attempts; original-energy
Armijo governs acceptance. Off-diagonal curvature must set a meaningful scale
even when the diagonal vanishes. TR-CG uses
the cached original scaled Hessian, absolute-diagonal metric with a positive
floor, relative inner forcing 0.1 (not the old dimensional sqrt-gradient forcing),
and max 250 inner iterations. Radius updates and rejections preserve the base
state/Hessian cache. All controls are serialized, not hidden tunable defaults.
These are initial development configurations, not yet screening-qualified recipes.

Checkpoint generations contain full optimizer state (including L-BFGS histories,
CG directions, TR radius and cached rejected-base sparse Hessian), counters,
curves and best residual-accepted state. Publication is atomic/no-clobber via a
same-filesystem link after file fsync, followed by parent-directory fsync.
Active-step snapshots/reentry are rejected. Payload checksums detect accidental
corruption; they are not cryptographic authentication. Restore requires identical
recipe/implementation/problem/scales and a mandatory external resource-floor ledger; completed
cells and reductions of spent counters are rejected. Explicit safe-boundary
pause/resume is the scope of the unit tests.

**Gates outstanding at the end of the original solver-core phase (resolved or
explicitly bounded by the launch checks below):** immutable six-cycle final-objective
bundle and six exact starts; a driver that provides independently verified
content identities (including a real build hash in `Config::implementationId`,
not the development version label); a durable external resource/completed-cell ledger accounting
for checkpoint I/O and work lost after the latest checkpoint; tested process
interruption/hard watchdog and directory-publication durability under interruption;
actual-shell/full sparse-cache checkpoint cost checks; common final curvature/admissibility scoring and complete
per-cell reporting. A safe-boundary state-file test does not by itself establish
crash-safe operational resumption. Do not launch the scientific screen merely
because unit tests pass.

### Implemented regression barriers

- Known convex quadratic and nonconvex double-well minima for all four methods.
- Original HLBFGS trajectory/gradient/evaluation-count parity for 25 accepted
  iterations, past two complete 10-pair memory wraps.
- Strong-Wolfe inequalities and descent; indefinite-Hessian regularization and
  original-energy evaluation; TR negative curvature, exact rollback and one
  cached Hessian across rejected trials and resume.
- Nonfinite oracle/overflow, plateau, line-search failure, evaluation/time/attempt
  caps; independent final gradient verification counted as oracle work.
- Native stopping, not a scaled-gradient shortcut; stationary saddles explicitly
  remain stationary points, not certified minima.
- All-method checkpoint continuation against uninterrupted traces; memory,
  direction/radius/cached-Hessian preservation; identity/config/scaling mismatch,
  corrupted payload, overwrite refusal, completed-cell and spent-budget barriers.
- Analytic/TinyAD gradient and finite-difference HVP checks on free and clamped
  small bilayers; all four adapters preserve physical constraints; gauge derived
  from the explicit base geometry rather than a lingering trial state.

Development tests may be rerun after implementation fixes within the approved
cumulative development budget, preserving each failure/snapshot. This is not
authorization to retry numerical failures or completed cells in the scientific
screen.

### Development validation ledger

- **7276772**: isolated first build and execution, exit 0; **18/18 tests passed**
  (12 new tests plus six existing projector/Steihaug regressions). Slurm elapsed
  4m46s, TotalCPU 4m03.021s, peak RSS 1,887,696 KiB. Source snapshot, SHA-256 list,
  environment, binary hash, complete log, XML and checkpoint fixtures are under
  `run/nonconvex_solver_benchmark/development_7276772/`.
- Subsequent source review added explicit active-step checkpoint barriers,
  directory fsync, build-identity comparison, failed-verification candidate
  invalidation, off-diagonal shift scaling and removal of redundant geometry
  restoration after successful oracle calls. Added tests cover these, a
  32,267-variable/full-ten-pair synthetic L-BFGS checkpoint, and inner/Hessian
  caps.
- **7279057**: second isolated build and execution, exit 0; **25/25 tests passed**
  (19 new tests plus six existing regressions). Slurm elapsed **5m44s**, TotalCPU
  **5m03.668s**, peak RSS **1,965,096 KiB**; final test execution 1.556s. All three
  current numerical source files and the submission script were byte-compared
  against this tested snapshot and matched. Artifacts, including local dependency
  SHA-256 hashes, are under `run/nonconvex_solver_benchmark/development_7279057/`.
- Total Slurm allocated single-CPU job time: **10m30s**; summed TotalCPU:
  **9m06.689s**, within the 30 CPU-minute development ceiling. No scientific
  screening budget has been consumed.
- The full-dimension synthetic L-BFGS fixture saved/restored **32,267 variables
  and all ten history pairs after 12 steps**: 18,858,138-byte checkpoint,
  **1.38668s save-plus-load** on qnode0379. Saved vectors/history round-tripped
  exactly; subsequent continuation passed its 1e-13 state-difference bound and
  exact evaluation-counter comparison. This is NOT a physical shell checkpoint
  or an upper bound for full sparse-Hessian cache/bundle I/O.
- Machine-readable audit: `run/nonconvex_solver_benchmark/development_summary.json`.
  This was a targeted Release regression run, not the entire repository suite,
  a sanitizer run, a hard-kill recovery test, or a solver performance comparison.

The first build emitted an ambiguous-else warning in a new test assertion; braces
were added. Other warnings came from existing Triangle/libigl/Eigen code; no
upstream source was changed to silence them. No production-shell optimization,
solver speed comparison or global-basin screening was run in that core phase.

## Scientific launch gates — 7290186 / 7294450

User authorized the four-method benchmark. Build 7290186 failed before numerical
execution: operation growth fields are per-strip vectors, not scalar doubles.
The guard was corrected to check every entry. The failed source and original
runner are preserved under `preflight_repair_7294450/`; the original failed log
and source manifest remain under `development_7290186/`. That isolated build tree
was then updated in place with the preserved two-file revision and reused.
Queued repair 7292852 was cancelled before starting so its script could be
updated; it consumed zero allocated seconds. No scientific fit was retried.
After all later work and explicit cleanup approval, only that failed build's
generated `development_7290186/source/build/` (11.1 MiB logical) was removed;
source, binary, logs, input and provenance remained. The precise scope and
retained hashes are in `development_7290186/pruned_build_record.json`.

Repair/preflight **7294450 completed**, exit 0, 3m48s, peak RSS 2,793,936 KiB:
- **26 tests passed**, including a new guard against substituting a looser-gate
  historical candidate for the converged endpoint when energies tie by roundoff.
- Six-cycle targets constructed without optimization; independently checked as
  isotropic metric products, including the archived verified two-cycle prefix.
  Hit counts: **4966, 4196, 2905, 1640, 885, 517**; actual area **0.0774192 m²**.
- Six fixed, sign-paired starts: exact cylindrical embeddings about y/x with
  1.8 mm boundary sag, and twist with 1.8 mm corner amplitude. Directors are
  initialized from analytic surface normals at reference edge midpoints using
  the native projected-angle constructor. No energy-dependent start selection.
- All six native/TinyAD gradient differences <=**2.167e-20**; finite-difference
  HVP relative errors <=**9.608e-9**. Initial energies, constraints and geometry
  were checked; Hessians have 1,739,471 stored entries each.
- Actual-shell checkpoint round trips passed for all four profiles. The TR
  reduced sparse cache held **1,739,033 entries**, a **47,660,153-byte** checkpoint;
  save-plus-load **3.115 s**. This check did not optimize a screening start.
- Each method passed process-level SIGKILL/restoration after a durable safe
  pause, matching uninterrupted fixture states/counters. Wrong-build identity,
  completed-cell and watchdog-cap reuse were rejected. Suite recovery was
  tested with fake workers, not extra scientific fits.

Verified gate and all input/restart evidence:
`run/nonconvex_solver_benchmark/preflight_repair_7294450/preflight/`.
Binary SHA-256:
`57c9d9df359beca65ecf2bcf9e5bf8c478a16aa0c7f9f4ac96c47cf8094ff392`.
Bundle manifest SHA-256:
`7591e5c69abe16d076cf913806673a1b7f50672c08c11c0a81a579787ab17281`.
Current driver/worker sources match the tested revision.

### Frozen execution and acceptance policy

- One pinned CPU on **qnode0025**, the same hardware as preflight; OpenMP/BLAS
  threads and in-process TBB concurrency are one. Four methods are interleaved
  in rotating order within the six fixed starts.
- The original **300 s/cell** and **7200 s total** ceilings remain. Optimization
  stops by **265 s** (slightly earlier than the proposed 270 s), the worker has
  a **295 s hard watchdog**, and the remainder covers supervisory I/O. All
  checkpoint, verification and recovery costs count. The Slurm allocation also
  bounds setup and optional reporting. No automatic retries or extensions.
- The benchmark reports native residual attainment and original-Hessian status
  separately. Its accepted-minimum tally/energy curve requires `gradient_target`,
  an independently recomputed gradient <=5e-14, and positive unshifted restricted
  LDL pivots. Negative/unresolved curvature, capped or failed fits are not promoted.
  Earlier 1e-13-gate candidates remain in checkpoints but do not replace a
  verified converged endpoint. This is numerical local evidence, not certification.
- Safe pauses can be explicitly resumed under the original remaining allowance.
  **Unclassified mid-step crashes are terminal**, not silently replayed. A suite
  resume charges an unknown interrupted cell its full original 300 s allowance,
  preserves its partial checkpoints, and skips it and every completed cell.
  Thus recovery is conservative, not a promise to continue every kind of crash.
- New driver: `python/run_nonconvex_screen.py`; worker/input generator:
  `test/testshell/Test_NonconvexScreen.cpp`; submission:
  `scripts/nonconvex_screen.sbatch`. Full-panel plots and all available proper
  rigid-aligned position comparisons are generated by
  `python/plot_nonconvex_screen.py` if allocation time remains. Rendering/visual
  review is distinct from numerical acceptance; failed rendering never reruns fits.

### Isolated CHOLMOD integration — separate 15 CPU-minute authorization

Researcher authorized a new **900-second, one-CPU** build/qualification phase,
not a matched optimization pilot. V6 adds opt-in `newtonBackend=cholmod` and a
required runtime dependency identity; Eigen remains the default. CHOLMOD is
compiled/linked only into the isolated test executable via generated build-tree
CMake settings. Production CMake and solvers are unchanged.

The adapter fixes an external Eigen-AMD permutation per base, maps every numeric
CSC entry exactly (including diagonals), rejects pattern changes, and restores
the original coordinate ordering on solves. No shift is inferred from rounded
diagonal differences. Existing shift sequence, original-energy line search,
native stopping criteria and curvature acceptance remain unchanged. Numeric
factor state is temporary and is not added to boundary checkpoints.

Qualification: existing regressions plus mapping/parity, integrated Newton,
backend/dependency mismatch and boundary-resume tests; all-six shell derivative
and checkpoint preflight; all-method safe-pause SIGKILL/recovery, watchdog and
completed-cell/ledger barriers; full-size cycle-2 unshifted linear gate using the
previously hashed fixture. Near-gate bounds remain linear residual <=1e-8 and
relative direction/archived-correction differences <=1e-5, as in the previous
near-equilibrium check. This does not replace native gradient stopping or claim
physical convergence from a linear fixture.

The qualification launcher fingerprints the worker and resolved ELF library
closure, hashes dependencies before each real worker launch, and checks loader
environment/cache and resolution before/after preflight. Its identity is part
of checkpoint configuration. Simulated changed-library digests and changed
checkpoint identities must be rejected without modifying installed libraries.
The generated gate is qualification-only and is blocked by the scientific runner.
No full fits, installations or silent retries. Whole-process watchdog 870s plus
5s kill grace, with parent signal handling to reap separately grouped workers.

**Qualified: 7405668 on qnode0004.** The full opt-in worker built and passed
**36 regression tests**; the optional frozen near test was explicitly skipped in
that invocation and then passed separately. All-six derivative checks passed
(native/AD gradient difference <=2.1664e-20, FD HVP relative error <=9.6079e-9),
as did full-size checkpoint roundtrips, all-four-method safe-pause SIGKILL/recovery,
wrong-build/completed-cell/watchdog barriers, parent cleanup, and mocked suite/
pilot ledgers. The Newton recovery fixtures' saved recipes were independently
checked to select CHOLMOD with the qualified dependency identity.

The full-worker runtime fingerprint covers **31 resolved libraries**, including
the linked OpenMP/BLAS runtimes. Hash/environment/cache checks and simulated
changed-dependency rejection passed. The one-thread full-worker combination is
qualified, not arbitrary multithreaded operation or adversarial/dlopen scenarios.
Binary SHA256: `6d9e10bd92efcc2e59fb9c5823eb3bac726c2c7ab8ece695b09d136ec1794a91`.

Full-size cycle-2 **zero-shift** gate:

| Backend | Relative linear residual | Relative correction difference from Eigen | Relative archived physical-correction difference |
|---|---:|---:|---:|
| Eigen | 2.0716e-9 | 0 | 1.5730e-9 |
| CHOLMOD | 4.5605e-10 | 2.7935e-8 | 2.6756e-8 |

Both passed the existing <=1e-8 linear residual and predeclared <=1e-5
near-correction comparison bounds. The latter is deliberately distinct from
normal shifted-matrix comparison tolerances; no native physical stopping
criterion was relaxed. This is still a frozen linear check, not a post-step
native-gradient convergence certificate.

Default-path smoke job **7406627** compiled the facade without the CHOLMOD macro,
headers or libraries, solved an SPD fixture, rejected unavailable CHOLMOD, and
verified no CHOLMOD dynamic link. The whole default worker was not rebuilt in
this additional check; its Eigen-default behavior was covered by the main suite.

Artifacts: `run/nonconvex_solver_benchmark/integration_7405668/{integration_gate.json,
summary.json,runtime_manifest.json,near_gate.json,preflight/}` and the isolated
`development_7405668/source/` build. The stock scientific runner rejects the
qualification-only gate, and the qualification launcher itself rejects fit
requests. Production code is unchanged; no scientific fit or matched pilot ran.
The adapter's extra remapping/pattern-check cost and whole-trajectory convergence
remain unmeasured; do not substitute the earlier standalone speedup for them.

Budget: build/qualification **718s** main job, conservatively **722s** including
extern; default check **31s**, conservatively **33s** including extern. Total
**755s = 12m35s /15 minutes**, actual CPU **11m03.947s**. Phase closed, no retries
or cap extensions. A matched pilot requires a separately authorized backend-aware
launch rather than bypassing this qualification gate.

### Matched v6 CHOLMOD pilot — separate 30 CPU-minute authorization

Researcher authorized a new **1800-second, one-CPU matched pilot** comparing
native L-BFGS against CHOLMOD-accelerated shifted Newton under the exact same
fixed final eigenstrain objective, starts, and native convergence criteria.
Production code, solvers, and physical stopping tolerances remain unchanged.
Eigen remains default in the facade; CHOLMOD is explicitly selected via
opt-in configuration with verified runtime dependency fingerprinting.

Protocol:
- **Starts:** `cylinder_y_plus` and `cylinder_y_minus` (identical objective, mesh, and perturbed configurations).
- **Cells (4):**
  1. `cylinder_y_plus`: `native_lbfgs` (concurrent baseline)
  2. `cylinder_y_plus`: `sparse_newton` (`newtonBackend="cholmod"`)
  3. `cylinder_y_minus`: `sparse_newton` (`newtonBackend="cholmod"`)
  4. `cylinder_y_minus`: `native_lbfgs` (concurrent baseline)
- **Caps:** 300s per cell (265s optimizer deadline, 295s worker timeout).
- **Stopping & acceptance:** Internal target 5e-14, independent native verification <= 5e-14, positive restricted unshifted LDL pivots. Looser residual candidate (1e-13) retained separately and cannot promote a capped/failed fit.
- **Safety & recovery:** Full runtime dependency verification (31 libraries, hash `a00e101bd0dfef629aef8f87eb1a41a1b555f191024aaeab09d69de761173803`), verified safe pause/resume, watchdog cap, unclassified interruptions charged full 300s and never rerun.
- **Preflight & near gate:** All-six start derivative checks, 4-method safe pause/recovery tests, and the unshifted cycle-2 near-equilibrium gate (`linear_residual <= 1e-8`, `relative_archived_correction_difference <= 1e-5`) must pass before fits launch.
- **Hardware controls:** Single CPU core (`quest10` node), single thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), `CHOLMOD_USE_GPU=0`.
- **Scripts:** `scripts/cholmod_matched_pilot.sbatch`, `python/run_cholmod_pilot.py`.

#### Completed: Job 7494613 on qnode0156 (19m24s wall-clock, 18m26.6s CPU)

Prerequisites: dependency fingerprint (31 libraries, hash `a00e101bd0dfef629aef8f87eb1a41a1b555f191024aaeab09d69de761173803`),
6-start derivative preflight (AD vs native diff <= 2.17e-20, HVP error <= 9.61e-9),
4-method safe pause/SIGKILL/recovery, watchdog, completed-cell guards, and the
unshifted cycle-2 linear gate (`linear_residual=4.56e-10 <= 1e-8`, `relative_archived_correction_difference=2.68e-8 <= 1e-5`)
all passed in 37.6s of setup before fits launched.

**Four completed cells** (`screen/results.csv`, `screen/ledger.json`):

| Cell | Start | Method (backend) | Status | Attempts (committed) | Energy | Native grad norm | Restricted curvature |
|---|---|---|---|---:|---:|---:|---|
| 0 | `cylinder_y_plus` | `native_lbfgs` | `time_cap` | 6287 (6286) | **1.4177e-9** | 4.92e-9 | indefinite |
| 1 | `cylinder_y_plus` | `sparse_newton` (`cholmod`) | `time_cap` | 75 (74) | **1.4693e-9** | 4.01e-8 | **positive_pivots** |
| 2 | `cylinder_y_minus` | `sparse_newton` (`cholmod`) | `time_cap` | 69 (68) | **1.4343e-9** | 1.96e-8 | **positive_pivots** |
| 3 | `cylinder_y_minus` | `native_lbfgs` | `time_cap` | 5683 (5682) | **1.4215e-9** | 5.89e-9 | indefinite |

**Key findings & comparison against earlier Eigen LLT pilot 7386943:**
1. **Factorization speedup realized in full trajectory:**
   - Numerical factorization time collapsed from **335.84s (61.1% of charged Newton time)** in pilot 7386943 down to **96.33s (17.4%)** in 7494613.
   - Median factorization time dropped from ~3.0s/step to **~0.62–0.68s/step**, a **~4.5× per-step factorization speedup**.
2. **Step count almost doubled:**
   - In the same 265s optimizer window, Newton took **75 and 69 attempts** (vs 40 and 38 with Eigen LLT), an **+85% increase in committed steps**.
3. **Energy gap closed by ~90%:**
   - On `cylinder_y_plus`: Newton dropped from 1.7999e-9 (Eigen) to **1.4693e-9** (CHOLMOD).
   - On `cylinder_y_minus`: Newton dropped from 1.7864e-9 (Eigen) to **1.4343e-9** (CHOLMOD), coming within **0.9%** of L-BFGS's concurrent 1.4215e-9!
4. **Restricted curvature qualification:**
   - Both Newton endpoints transitioned from `indefinite` in the earlier pilot to **`positive_pivots`** on the restricted Hessian.
   - For the first time along the trajectory, unshifted Newton steps succeeded (**7 unshifted steps** on `cylinder_y_plus`, **11 unshifted steps** on `cylinder_y_minus`, vs 0 previously).
5. **New bottleneck uncovered:**
   - With factorization down to 17.4%, **model construction now dominates runtime at 66.1% (365.98s total)**.
   - Specifically, element-level TinyAD Hessian evaluation + assembly (`hessian_oracle_seconds`) accounts for **336.70s (92.0% of model time)**, taking a median of **2.30–2.43s per step**.
6. **Convergence conclusion:**
   - Zero accepted minima (target is 5e-14; all 4 cells hit `time_cap` with residuals ~5e-9 for L-BFGS, ~2–4e-8 for Newton).
   - CHOLMOD resolves the factorization bottleneck and allows Newton to enter the lower-energy positive-curvature basin, but further trajectory progress in fixed wall-clock time is now bounded by the forward Hessian oracle evaluation cost.

Evidence: `run/nonconvex_solver_benchmark/cholmod_pilot_7494613/{screen/results.csv, timing_audit.json, run_status.json, runtime_manifest.json, near_gate.json, scheduler_accounting.psv}`.

### Single-core direct Hessian assembly cache — phase 1 (qualification pending)

Goal: reduce per-iteration Hessian construction while leaving the production
`src/libshell/TinyADHessian_Bilayer.hpp`, native stopping gates and CHOLMOD
factorization unchanged. The shell benchmark still calls the original triplet
assembler by default. An explicit `hessianAssembly=cached_csc` benchmark choice
is included in the serialized recipe/checkpoint identity, and the shell adapter
uses it only when compiled with `NONCONVEX_TEST_CACHED_TINYAD`. The qualification
gate cannot launch fits; any cached-optimizer trajectory needs separate approval.
A separately named, benchmark-only
`test/diagnostics/CachedTinyADHessian_Bilayer.hpp` holds a verbatim copy of the
face-energy/gradient/HVP kernel (enforced by
`python/verify_cached_tinyad_source.py`) and substitutes **only** direct CSC
assembly. For each mesh instance it builds the 21×21 per-face contribution map
into sorted CSC offsets once, keeps the pattern, resets numeric values and
accumulates Hessian entries in face order. Mesh instance, connectivity and
constraint-mask changes are rejected; changes in deformation and rest targets
are evaluated afresh. Pattern state is transient, not in optimizer checkpoints.
No science fit or production change is included in this phase.

The exploratory flat-state probe
`test/diagnostics/profile_tinyad_breakdown.cpp` (research scratch, not a
qualified optimizer) observed **~2.10 s** triplet versus **~2.00 s** cached
on a single CPU in its last three sequential runs; timing is not a controlled
matched benchmark on archived states. Its zero matrix-difference norm at one
flat state is **not a bitwise-equality certificate** for curved states or other
objectives. Its 4.65 million triplet *entries* are reserved in one vector,
not 4.65 million independent allocations. The gain is much smaller than the
earlier speculative estimate; AD element arithmetic remains the main cost.
Earlier one-off compile/profile jobs 7504719–7506009 consumed **422 allocated
single-CPU seconds (7m02s)** before this qualification protocol, including
failed toolchain/include/link attempts; that spent allocation is preserved
separately and does not qualify the cache or authorize fresh fits.

Predeclared qualification gates: original energy/gradient/HVP source equality;
small free/clamped shell states including geometry/director perturbations and
changed rest targets with exact CSC index equality and relative Hessian/HVP
agreement <=1e-12, including actual opt-in shell-adapter calls and checkpoint
recipe mismatch rejection; reject a different mesh; all six full-size starts on the
original objective with the same CSC structure, finite entries and relative
Frobenius/max coefficient differences <=1e-12. The existing derivative,
checkpoint, pause/recovery and ledger regressions must pass. This threshold
checks **cache parity**, not the original independent native stopping target;
no gradient gate is loosened. The six-start preflight gate is marked
`qualification_only` and explicitly blocks scientific fits. No speedup claim
beyond this single-core probe until matched curved-state timings pass.

Researcher authorized one 1-core **20-minute** isolated build/regression/
six-state qualification job, **7516445** on qnode0012. It consumed **623s
(10m23s)** allocated, with 9m25.351s CPU and exit code 1. The build passed
**38/38 active tests** (1 unrelated near-gate test skipped), including small
free/clamped and recipe-identity cache tests. The first full-size derivative
worker failed after 5.51s with `missing runtime dependency identity`: the
launcher selected CHOLMOD but omitted `--dependency-identity`. The request,
worker log and failure are preserved in
`cache_qualification_7516445/preflight/derivatives/`. **No six-start parity
result, qualified gate or scientific fit** was produced. The binary SHA256 is
`65c76a6482bd36d366b8510e676c18cbcd9d55e9b9d726f351d5994e82888886`.
Do not restart the completed build/tests, claim qualification, or rerun fits.

The researcher explicitly approved a **preflight-only continuation**, job
**7517700** on qnode0003. It reused the existing saved worker and regressions,
fingerprinted the new binary and 31 resolved runtime libraries, supplied the
required CHOLMOD dependency identity
`b810df0e4f8993bf888919854cd741c9aab585edbbe154336a5b70eee18302d6`,
and verified that dependency before each real worker. A separate preflight root
preserves the original failure. The one-core continuation completed in **73s**
(55.189 CPU-seconds), with exit code 0. The scheduler rounded its requested
9m20s to a 10m time limit; an independent 540s watchdog plus 5s kill grace
kept the planned cumulative cap bounded. Phase cost: **623+73=696s = 11m36s
/20 minutes** of allocated one-CPU time (CPU **620.540s**), phase closed.
No scientific fit or repeated build/test ran.

**Qualification for the declared parity/recovery scope passed.** On all six
inspected starts of the same final objective, the cached adapter's 1,739,471
CSC entries had identical sparse index arrays to the production triplet
assembler; measured relative Frobenius and maximum coefficient differences
were **0** on each state (zero numerical difference in these six tests, not a
universal bitwise certificate). The cached map built once across all six
states. Native/AD gradient differences were <=2.17e-20; max independent
finite-difference HVP relative error was 9.608e-9. Four-method safe-pause/
SIGKILL/resume, parent cleanup, watchdog, identity barriers and mocked suite
ledgers passed. Saved sparse-Newton recovery recipe selects `cholmod` plus
`cached_csc` with the new dependency identity. Gate:
`run/nonconvex_solver_benchmark/cache_qualification_7516445/continuation_7517700/continuation_gate.json`;
original failure is retained separately. Gate is `qualification_only` and
cannot start a scientific screen. Full-size matched curved-state timing,
trajectory progress and held-out confirmation remain unmeasured; the earlier
~0.1s flat-probe timing is provisional. Opt-in cache remains benchmark-only;
production and default triplet assembly are unchanged.

**Predeclared frozen-state timing (researcher approved a bounded one-core
comparison; new one-attempt cap 8 single-CPU minutes, no fits).** Use a separate
isolated executable compiled with `-O3 -DNDEBUG -march=native -std=c++23` on
the assigned compute node, the byte-checked v7 snapshot's original
`TinyADHessian_Bilayer` and qualified cached header in the *same binary*.
Lock material and targets to the verified identical pilot/preflight bundle
(SHA256 manifest `7591e5c69abe16d076cf913806673a1b7f50672c08c11c0a81a579787ab17281`).
Select states by predetermined start/end identity, never by speed or error:
(1) initial cylinder-y+ `cylinder_y_plus.f64` SHA256
`4eb0bd8d66ce217e973b77d6529380603b055061eea5f50769f6585fdfaf3aca`,
(2) capped Newton cylinder-y+ selected/final SHA256
`7032c08cdb2ff9bcd631b6b0311276433a473a485be4fcf2550eb1e8664bc2e1`,
(3) capped Newton cylinder-y-minus selected/final SHA256
`d7ae525b31bf5b19f43ed7898cb3f130dcc72747780cd38a70a6023d17e31661`.
The two capped endpoints are from the prior job 7494613, **not accepted minima**.
Before measuring, verify fixture/source identities and for each state independently
check finite, matching CSC indices, relative full-Hessian Frobenius and maximum
coefficient differences <=1e-12, and only one cache-pattern construction.
Record first-call times separately (including the first cache build); then
predeclare balanced warm sequence `ABBA BAAB`, with `A=triplets` and
`B=cached_csc`, four calls per method/state, same mesh/state/compiler/core.
Report each state's actual first-call and warm medians/ranges, not the best
case; timing distributions are technical repeats on three related states, not
independent physics replicates. Do not extrapolate a solver speedup, accepted
energy or global result from these kernel-only timings. Preserve failure and
stop without retry if a build, parity or identity check fails; no optimizer step
or scientific test is authorized. This timing phase is distinct from the
**closed** 20-minute cache qualification phase.

**Timing attempt 7518799 stopped before compiling or evaluating a Hessian:**
the new standalone profiler lacked the frozen cache header's include directory
in its compile command (`fatal error: CachedTinyADHessian_Bilayer.hpp: No such
file or directory`). Input hashes were checked; **22s one-CPU allocation
(3.228 CPU-seconds) / 480s** charged, exit 1. No curved-state timing, optimizer
step or fit occurred. Preserve the attempt in
`run/nonconvex_solver_benchmark/cache_curved_timing_7518799/`.
A bounded compile-path correction in
`scripts/cached_curved_timing_continuation.sbatch` is prepared but **not
submitted**. It will reuse the locked inputs and original profiler source,
add only `-I<snapshot>/test/diagnostics`, and stop under a 420s scheduler cap
plus a 390s watchdog (22+420=442/480s maximum declared allocation). Because
the predeclared protocol said one attempt/no retry, request the researcher's
explicit permission for this reuse-only continuation before submission; do
not silently reset the budget or run an optimization fit.

**Approved continuation 7519129 completed:** after source/API, include/link
path, shell syntax and input-identity review, the corrected standalone build
and all three states completed on one CPU (qnode0127). Build log and runtime
stderr were empty; exit 0. Original 22s + continuation 104s = **126/480s**
allocated; phase closed, no fits. Binary and input hashes rechecked afterward.
All three states had identical CSC indices and zero measured coefficient/
Frobenius difference; the pattern built once. Warm medians (triplets → cache):

| Saved state | Triplets (s) | Cached (s) | Time reduction |
|---|---:|---:|---:|
| cylinder-y+ start | 2.075868 | 1.984024 | 4.42% |
| cylinder-y+ capped endpoint | 2.065744 | 1.982261 | 4.04% |
| cylinder-y-minus capped endpoint | 2.070440 | 1.983237 | 4.21% |

Four calls/method/state in predeclared balanced order; raw times are preserved.
The first cache call, including pattern construction, took 2.176455s versus
2.076071s for triplets. Later states reused the pattern, so their first calls
are **not cold-cache measurements**. This supports a modest repeated-Hessian
kernel improvement, not measured optimizer acceleration or convergence.
Evidence: `run/nonconvex_solver_benchmark/cache_curved_timing_7518799/continuation_7519129/analysis.json`
and `scheduler_accounting.psv`. No further compute is launched.

### Pure-stretch AD stencil split — completed diagnostic (job 7538408)

Researcher approved a **15-minute, one-CPU cumulative cap**; this single job
completed in **161s allocated (137.623 CPU-s), exit 0**. Build and runtime
stderr were empty, inputs and executable hashes rechecked, no retry or optimizer
fit. Phase closed at **161/900s**. Candidate and profiler are isolated in
`test/diagnostics/SplitCachedTinyADHessian_Bilayer.hpp` and
`test/diagnostics/ProfileSplitCachedHessian.cpp`; cached unsplit oracle,
production oracle and production solver source remain unchanged.

The current per-layer energy is `stretch + bend + mixed`. Only `stretch`
depends exclusively on the own face's nine vertex coordinates. Keep `bend`
and **all mixed stretch–bend terms**, including opposite top/bottom coupling
signs and separate target metrics, in the 21-variable branch. Do not assume
mixed terms cancel for unequal layer targets. Compute pure stretching with
9-variable AD and scatter its derivatives into the first nine local stencil
slots, adding them to the 21-variable bending-plus-mixed derivatives. Do not
compute the full energy and then subtract stretching: that retains its AD
cost and adds cancellation. Preserve material constants, topology, constraint
masking, target forms, face order and cached CSC assembly.

Hypothesis: smaller stretching derivative objects reduce repeated Hessian
cost. Competing explanation: bending/dihedrals and mixed terms still require
most 21-variable work, and duplicated metric computation/dispatch can erase
or reverse the saving. This is a limited diagnostic, not a promised major
speedup. Keep the qualified cache and production oracle unchanged; put the
candidate in an isolated diagnostic implementation, not a solver option yet.

Qualification before timing: small free/clamped flat and curved meshes,
unequal layer targets and nonzero rest curvature; compare energy, gradient,
full Hessian and deterministic HVPs to the unsplit oracle. Require finite
values, identical sparse indices and relative Hessian/HVP agreement <=1e-12
for nonzero reference norms; record absolute discrepancies and use explicit
absolute gates near zero rather than dividing by zero. Fix energy/gradient
absolute and relative gates in the test source before submission, taking
existing derivative checks as the baseline. Include a coupling-sensitive
case whose parity test demonstrably fails when mixed terms are omitted.
These are arithmetic regroupings: do not demand or claim bitwise equality.

Only if all checks pass, compare **unsplit cached CSC versus split cached
CSC**, in the same binary/core/compiler, on the three already locked curved
states used by 7519129 (start and both capped endpoints). Retain the triplet
oracle for parity, not as the timing baseline: otherwise the existing cache
benefit would be misattributed to the split. Use first-call accounting and
four calls per method/state in `ABBA BAAB` order; preserve all measurements,
medians and ranges. No timing-driven state selection or test tuning. Record
source, input, binary and resolved dependency hashes. Use one bounded job
including compilation/checks/timing; stop on failure without automatic retry.
Report any regression or inconclusive saving, and leave production unchanged
regardless of the diagnostic result.

**Result:** full derivative gates passed on four small free/clamped flat/curved
cases with unequal top/bottom metrics, imposed nonzero rest curvature and
coupling-sensitive energy witnesses, plus three locked full-size states.
Maximum energy absolute discrepancy vs native energy: **2.74e-22**; maximum
split/unsplit AD-gradient relative discrepancy **6.71e-13**; maximum Hessian
Frobenius relative discrepancy **7.08e-17** (max coefficient **4.72e-16**);
maximum HVP relative discrepancy **5.12e-16**. On each small case, omitting
mixed coupling changed energy by **~7.0e-11**, far beyond the parity gate, so
this check would detect dropping those terms. CSC patterns matched, cache
reuse passed; all parity gates preceded timing.

**No useful timing gain:** compared cached split against cached unsplit in the
same single-core binary, including fresh pattern construction and four warm
calls/method/state in ABBA BAAB order. Warm median changes (positive = faster)
were **+0.071%, -0.008%, +0.365%** across the three saved states. This is
inconsistent and smaller than the observed call-to-call variation. Cold first
assembly was **15–23 ms slower** for split. This does **not** support a meaningful
speedup; do not adopt or further optimize this stencil split. It leaves the
cache's separate ~4% repeated assembly result unchanged. No full-trajectory
effect, convergence or basin outcome was measured.

Analysis and all raw calls: `run/nonconvex_solver_benchmark/split_cached_hessian_7538408/analysis.json`
and `profile.jsonl`; source/input/binary manifests and scheduler accounting are
in the same directory. For a larger next opportunity, focus diagnosis on the
remaining 21-variable bending/dihedral AD cost or bounded face-level
parallelization; these are proposals only and need separate design/approval.

### Local-dihedral AD-width diagnostic — smoke failed, full stage blocked

Status: researcher approved a **fresh cumulative 15-minute one-core cap** and
asked for a smoke/unit test followed by **explicit confirmation before the
full-size run**. Prepare two separate jobs: at most 360s allocation for
compile+four small free/clamped smoke cases, then (only if explicitly confirmed)
at most 540s for frozen full-size parity/timing with the **same compiled and
hashed binary**. Combined worst-case allocation <=900s; no implicit retry,
unused first-stage time does not raise the second-stage cap. Do not use closed
split/cache budgets, run optimization fits, or edit production code.

The exploratory flat-state scratch breakdown spent ~0.98s of ~2.1s on
three signed dihedrals, but that instrumented duplicate is not a qualified
curved-state measurement. Candidate in
`test/diagnostics/LocalDihedralCachedTinyADHessian_Bilayer.hpp` computes each
interior `atan2` dihedral's gradient/Hessian with 12 active vertex coordinates
(9 own, 3 opposite) and lifts those exact derivatives into the existing
21-DOF element AD for unchanged heights, bending, and bilayer mixed coupling.
No angle is replaced by finite differences. The opposite-vertex index is
face-edge specific; boundary dihedrals remain zero. Signed orientation,
rest targets, material, constraints and CSC accumulation are not changed.
Opposing hypothesis: repeating the own-face normal and scattering 21x21
Hessians three times can erase any narrower-AD saving; a negative result is
expected to be reported, not tuned away.

Qualification precedes timing: one frozen executable contains the original
TinyAD oracle, previously qualified unsplit cached control and local-dihedral
candidate. Compile with portable `-march=nocona -mtune=haswell` to reuse the
same binary across allocated nodes; compare both methods in that same binary,
not absolute times from differently compiled earlier jobs. Smoke-only mode
checks small free/clamped flat/perturbed curved states with unequal layer
targets and nonzero reference curvature and writes a binary/source-hashed gate.
No full-size calculation is performed in smoke mode. After researcher
confirmation, full-only mode verifies the preserved smoke gate, executable,
input and resolved-library identities **without rerunning the small checks**.
All three predeclared full-size states and byte-verified rest/constraint bundle
are the same as job 7519129. Compare candidate energy to the native objective with
absolute/relative gate `1e-17 + 1e-12*|E|`; compare native gradient with
`1e-18 + 1e-7*||g||`, and candidate/unsplit AD gradients, full CSC indices,
finite Hessian coefficients and deterministic HVPs at <=1e-12 relative
(nonzero reference). Check one cache-pattern build per mesh. Fail closed on
any mismatch. No physical convergence or scientific acceptance is tested.
Only after all gates, time fresh-cache first calls and four repeated cached
assemblies per method/state in balanced `ABBA BAAB` order. **A is the already
cached unsplit control; B is the local-dihedral candidate**: the triplet
baseline is not appropriate for crediting an additional speedup. Report each
state's medians, raw ranges, cold cost and any slowdown without independent-
replicate claims. No scientific retries, fits or production adoption is authorized by this
diagnostic. Full-run submission requires a report and explicit new confirmation
after the smoke result, even if the smoke gate passes.

**Smoke attempt 7543030 (qnode0120): failed.** Isolated C++ build and frozen
input/header identity checks passed, but the executable segfaulted with exit
`139:0` before producing its first small-derivative record. The build and
runtime stderr files are empty; `smoke.jsonl` is empty and there is **no
smoke gate**. Exact fault location has **not** been established. Scheduler
charged **62s allocated / 40.381 CPU-s** against the newly approved 900s cap.
No full-size data, timings, optimizer steps, or fits were evaluated. The
full-size script is **not submitted and must not run**. Preserved failure:
`run/nonconvex_solver_benchmark/local_dihedral_smoke_7543030/{failure_summary.json,job.log,binary.sha256,source.sha256,scheduler_accounting.psv}`.
Do not rerun automatically, loosen parity gates, or silently transfer the
remaining cap to a revised implementation. The researcher then approved one
**fault-localizing small-only diagnostic** using at most 180s of the original
remaining one-CPU budget: `scripts/local_dihedral_sanitize.sbatch`. It compiles
the preserved failing candidate/profiler with `-O1 -g -fsanitize=address,undefined`
and frame pointers, and calls `--small-only` under a 150s watchdog. This is a
diagnostic build, not a substitute for the failed original release smoke gate.
No full-size states, optimizer steps or fits; a new release smoke and separate
full-run confirmation would be needed even if the sanitizer returns normally.

**Sanitizer job 7543292 timed out before running:** the 150s watchdog stopped
compilation of the same frozen candidate with `-O1 -g` plus ASan/UBSan;
Slurm charged **154s allocated / 133.825 CPU-s**, exit `124:0`. No sanitizer
binary, stack trace, small derivative result, full-size evaluation or fit.
The build log contains a compiler variable-tracking-limit note only, not a
root-cause diagnosis. Original smoke + diagnostic cost **62+154=216/900s**;
phase remains paused. Failure preserved at
`run/nonconvex_solver_benchmark/local_dihedral_sanitize_7543292/`.
Avoid claiming sanitizer validation or rerunning automatically. A cheaper
possible fault-localization (not submitted) is to use the **already compiled,
non-stripped release binary** with the system's available `libSegFault.so`
on one core to collect a stack trace from small-only mode, without a rebuild.
The researcher approved **one** release-binary stack-trace diagnostic using
at most **120s of the remaining original cap**, `scripts/local_dihedral_stacktrace.sbatch`.
It reuses the frozen, non-stripped binary and preloads the system
`libSegFault.so` for small-only fault localization under a 95s watchdog,
with core dumps disabled; it does not change or qualify the release source.
Job **7543615** reproduced the crash in **11s allocated / 0.987 CPU-s**;
its stack trace maps the fault to the *native* lower-layer
`ComputeCombined_Parametric::processOneTriangle`, called by `native.eval`
**before the candidate dihedral kernel is reached**. At executable offset
`+0x6c1ea`, `vmovdqa` reads a 32-byte-aligned AVX vector from an address with
16-byte alignment (`RCX % 32 = 16`, zero RDX), with a general-protection trap.
Frozen library CMake flags include `-march=native -mtune=native`; our new
standalone profiler was compiled `-march=nocona -mtune=haswell`. This is strong
evidence of a cross-translation-unit Eigen alignment ABI mismatch, **not**
evidence that the new dihedral derivatives are correct or incorrect. Saved
trace: `run/nonconvex_solver_benchmark/local_dihedral_trace_7543615/{small.stderr,trace_analysis.json}`.
Cumulative local-dihedral charge is **62+154+11=227/900s**; no small parity
gate, full-size state or fit. A possible repair is a fresh *release* smoke
built with the frozen library's `-O3 -march=native -mtune=native`, with input
and source unchanged. A reuse-only repair launcher is prepared at
`scripts/local_dihedral_abi_smoke.sbatch`: **120s one-core allocation cap**,
95s watchdog, same frozen candidate/harness/inputs and verified CPU model,
matching the frozen library's Eigen build flags. Even if fully spent,
**227+120+540=887/900s** worst-case including the separately confirmable
full stage. This changes the prior standalone compile recipe and requires
explicit approval; neither a new smoke nor the full-size stage has been
submitted. Until a passing release small-smoke gate, full remains blocked.

**Researcher-approved matching-ABI release smoke 7544213 passed.** On qnode0143
(same Intel Xeon Gold 6230R model as the frozen build), reused the identical
source, targets and four-case test at `-O3 -march=native -mtune=native`.
All small free/clamped flat/curved energy, native/AD gradient, CSC Hessian and
HVP gates passed, with one pattern construction. Maximum absolute energy
error **3.21e-24**, maximum native gradient absolute difference **8.74e-21**;
relative CSC/HVP differences were 0 for these four cases and maximum AD
gradient relative difference was **3.20e-18**. The compiler/runtime stderr
were empty. Gate:
`run/nonconvex_solver_benchmark/local_dihedral_abi_smoke_7544213/smoke_gate.json`;
worker SHA256 `7365ca176ff924ed5be58bccf98eb4c33945e00e0d91795ce6eebb8b0c05deb1`.
This supports the native-alignment diagnosis and qualifies **only the small
smoke**, not full-size parity or a speedup. Job used **66s allocated /
43.304 CPU-s**, exit 0. Phase cumulative **227+66=293/900s**. The already
prepared full-only script has a **540s one-core cap**; worst case
293+540=833/900s. It reuses the exact compiled worker, checks CPU model,
source/input/runtime hashes and smoke gate, and does **not rerun completed
small tests**. It remains **unsubmitted** pending the separately required
researcher confirmation. No optimizer steps/fits or production changes.

**First approved full-only submission 7544367 failed at its launcher:** the
script used the old `local_dihedral_smoke_7544213` directory rather than
`local_dihedral_abi_smoke_7544213`. The attempted protocol copy failed before
loading the qualified gate, worker binary or any full-size state. Exit 1,
**9s allocated / 0.750 CPU-s**; failure preserved at
`run/nonconvex_solver_benchmark/local_dihedral_full_7544367/`.
Cumulative phase charge **293+9=302/900s**. A path-only continuation is
prepared in `scripts/local_dihedral_full_continuation.sbatch`, checks the
specific earlier failure and otherwise reuses the exact smoke binary, inputs,
small gate, runtime-identity checks and unchanged <=540s full-only cap.
Worst case including continuation **302+540=842/900s**. It is **not
submitted**: ask before continuing after the predeclared no-retry failure.
No scientific fit or full-size candidate result existed at that point.

**Researcher-approved full-only continuation 7544454 completed** on qnode0143,
using the exact previously smoke-qualified worker (SHA256
`7365ca176ff924ed5be58bccf98eb4c33945e00e0d91795ce6eebb8b0c05deb1`),
the same Xeon 6230R model, matched resolved-library hashes and loader
environment, and locked bundle/start/endpoint inputs. It did not rerun small
checks, invoke an optimizer or evaluate a scientific fit. All three full-size
saved states passed finite energy/native gradient checks, identical CSC
indices, **zero measured full-Hessian coefficient/Frobenius difference** and
zero measured HVP difference between the cached unsplit control and the
local-dihedral candidate; one cached pattern per fixed mesh/state. Maximum
absolute energy difference vs native: **2.71e-22**; maximum absolute gradient
difference vs native: **2.58e-20**; maximum candidate/unsplit gradient relative
difference: **4.06e-15**. These are numerical checks on three states, not a
universal bitwise-equivalence certificate.

Same binary/core/compiler, four balanced warm calls per method/state
(`ABBA BAAB`, raw times and separate fresh-cache calls retained):

| Saved state | Cached unsplit median (s) [range] | Local-dihedral median (s) [range] | Reduction |
|---|---:|---:|---:|
| cylinder-y+ start | 2.312262 [2.307679–2.319141] | 1.602741 [1.602093–1.606724] | 30.69% |
| capped cylinder-y+ | 2.315768 [2.310676–2.320237] | 1.602329 [1.598576–1.604495] | 30.81% |
| capped cylinder-y-minus | 2.310128 [2.309122–2.321870] | 1.603484 [1.600384–1.613623] | 30.59% |

Fresh-cache first assemblies were also **0.703–0.730s faster** for the
candidate, though these first-call times may include runtime warm-up. This
supports a substantial **single-core Hessian-kernel** benefit relative to the
already-cached unsplit method; it does not measure whole-trajectory throughput,
accepted convergence, energy-basin selection, or an optimizer speedup. The
three states are diagnostics from one objective, not independent tasks.
Success artifacts: `run/nonconvex_solver_benchmark/local_dihedral_full_cont_7544454/{analysis.json,profile.jsonl,runtime_manifest.json,scheduler_accounting.psv}`.

The completed job used **106s allocated / 98.908 CPU-s**, exit 0. The
cumulative local-dihedral phase used **302+106=408/900s** allocated
(CPU **318.155s**); phase closed, unused cap does not authorize more trials.
Isolated candidate remains diagnostic only; qualified CHOLMOD worker,
benchmark default and production are unchanged. A cache/dihedral solver
integration with restart-identity/regression gates and any matched optimization
pilot require distinct plans and approval.

### Separate opt-in local-dihedral solver integration (staged; no fits)

The next benchmark-only implementation accepts `hessianAssembly=local_dihedral_csc`
only in builds compiled with `NONCONVEX_TEST_LOCAL_DIHEDRAL`; the existing
`triplets` default, production source, `cached_csc` baseline, CHOLMOD backend,
and all physical stopping criteria remain unchanged. The source snapshot must
contain the previously smoke-tested local-dihedral diagnostic header. Checkpoint
recipes record the selected assembly and reject another assembly, binary, runtime
library identity, or mesh/constraint mismatch on restart. No cache internals
are checkpointed. A six-start derivative/CSC and operational-recovery gate is
qualification-only and explicitly rejects scientific fits.

Stage 1: at most **14 one-CPU allocation minutes**, 16 GiB, isolated source
snapshot/build, core regressions and small free/clamped curved/target-change
energy, native gradient, CSC Hessian, HVP, adapter/cache and restart identity
checks. On failure stop, report artifacts and charge the allocation; no
automatic retry. Only after reviewing the smoke result and obtaining separate
confirmation may stage 2 run: at most **6 one-CPU allocation minutes**, 16 GiB,
reuse the *exact hashed smoke worker* with verified dependency closure, six
full-size starts, CSC parity and safe-pause/restart fixtures; no build and no
fits. The two caps total at most **20 one-CPU minutes**; no unused time is
carried over to unrelated work. A later matched optimization pilot would require
another authorization and budget. Three related prior timing states and the six
starts are not independent replicates; 30.59–30.81% is kernel time only.

**Stage 1 completed, job 7546843.** The isolated one-CPU build and small smoke
finished with exit code 0 in **459 allocated seconds** (of the 840-second smoke
cap; 16 GiB requested, batch MaxRSS 3042740K). Of 41 selected regressions,
40 passed and one optional frozen-near fixture was skipped. Both new
`BenchmarkLocalDihedral` tests passed: free/clamped flat and curved small states
with unequal bilayer rest metrics/nonzero target curvature, energy/gradient/
Hessian/HVP parity and adapter matching, target-change cache reuse and wrong-mesh
rejection, plus cross-assembly and wrong-implementation checkpoint rejection.
The exact candidate header retains diagnostic SHA256
`9c2d1d46d4b2881263ab7edd4f371bde792b891077f16c848facd21cc972533f`;
smoke worker SHA256 is
`d994c9f63f784a595bfa113ac98fe147786bd6e18535a80020a5caa73102a0d1`.
See `run/nonconvex_solver_benchmark/development_7546843/{build_and_tests.log,test_binary.sha256,source_sha256.txt,tests/tests.xml,exit_code.txt}`.
**Stage 2:** first submission **7547633** failed at Conda activation before
creating an output directory, driver or worker: `SYS_SYSROOT: unbound variable`.
It charged **8 allocated one-CPU seconds**, with zero parity checks/fits. After
explicit continuation approval, the launcher-only fix deferred `set -u` until
after activation and reduced the allocation cap to **352 seconds** (the original
360s less 8s); it reused the byte-identical smoke worker and unchanged driver.
Continuation **7547943 completed in 74 allocated one-CPU seconds** (49.542 CPU-s).
Six full-size starts passed independent native-energy/gradient and finite-
difference HVP gates; local-dihedral CSC indices and values matched the native
TinyAD Hessian with measured relative Frobenius difference **0** on each start.
The safe-pause/restart, completed/capped-cell guards, parent cleanup, and
cross-assembly recipe rejection passed; runtime dependency fingerprint is
`c5cd9a8dbe7c5e0eda6d81e77791a89dda5af76444d55c9f1bb7b2586c805417`.
The gate remains **qualification-only**, scientifically fitted cells **0**:
`run/nonconvex_solver_benchmark/local_dihedral_solver_gate_7547943/qualification_gate.json`.
Full-size gate charge is **82/360s**, and total solver-integration charge is
**459+8+74=541/1200s**; the phase is closed. No optimizer convergence,
whole-trajectory speedup, or accepted minimum is established; further fits
require a separate matched-pilot protocol and authorization.

### Proposed matched whole-trajectory assembly pilot (no fit launched)

This is a *new* exploratory fixed-recipe comparison, not an extension of the
qualification-only gate. Hold the full six-cycle objective, mesh, input bundle,
starts, production native energy/gradient, one CPU, CHOLMOD dependency identity,
physical tolerances and optimizer defaults fixed. On each of the frozen
`cylinder_y_plus` and `cylinder_y_minus` starts compare native L-BFGS with the
unchanged triplet option, CHOLMOD sparse Newton with `cached_csc`, and CHOLMOD
sparse Newton with `local_dihedral_csc`. Interleave recipe order across the two
starts; six cells, each **300 allocated seconds maximum** (worker watchdog at
295s, optimization deadline 265s), one cumulative **2100 allocated single-CPU
second / 35-minute** pilot cap including preparation, checks, failures and
output. This cap is a proposed separate authorization, not a carried-over
qualification allowance. Do not reuse earlier completed pilot fits as matched
controls or run cells outside the declared six.

Before pilot submission, a *separate* one-CPU toy smoke is capped at **180
allocated seconds**; it checks all three recipe/assembly identities, bounded
restart and fail-closed mismatch using the qualified frozen binary, with no
full-size optimization. Stop and report on failure; no automatic retries. Report
its result and obtain explicit confirmation **after** the smoke before starting
any scientific fits. Recheck the six-start qualification gate, bundle hash,
binary hash and runtime dependency closure before each pilot worker; do not
mutate/promote the qualification-only gate. Preserve every started cell's
recipe, ledger, native verification and final/selected checkpoints. Only
explicitly verified safe-pause continuation within the original cumulative
cell and suite cap is eligible later; no automatic resume, rerun of completed
or capped cells, recipe change or cap reset. Numerical failures are terminal.

The primary comparison is energy and native gradient versus *charged wall-clock*
time within each matched start, with Hessian count, phase times, pivots,
line-search outcomes, final and selected states and capped-cell status exposed.
Accepted minima require actual `gradient_target`, independent native gradient
≤5e-14 and positive unshifted restricted LDL pivots; a 1e-13 earlier residual
candidate is reported separately and never promotes a capped/failed cell.
Fixed time/recipe without new validation data is an approved-*if separately
confirmed* exploratory exception, not best-achievable tuning or global
optimization. These two starts belong to one objective and are not independent
replicates. Pilot results cannot justify production adoption or a claim about
all basins; any further run needs a new decision and budget.

**Toy pilot smoke 7550736 completed**: 26 allocated one-CPU seconds against
the separate 180s smoke cap, exit 0. The *same hashed worker* and verified
runtime closure passed all three recipe choices on a toy problem, three
safe-pause/resumes, completed-cell refusal and both directions of CSC-assembly
mismatch rejection. The stored six-cell `proposed_protocol.json` interleaves
start/recipe order and explicitly says `scientific_fits_authorized=false`.
Evidence: `run/nonconvex_solver_benchmark/matched_pilot_smoke_7550736/{smoke_gate.json,proposed_protocol.json,job.log,source_sha256.txt}`.
**No full-size optimization fit ran.** The smoke phase is closed; unused time
does not extend any other phase. Request explicit approval of the separately
proposed 2100-second pilot before its scientific launch. The toy-only smoke
checks recipe and process safety, not the yet-to-be-implemented six-cell pilot
ledger; that runner must itself pass a no-fit dry-run/gate check before launch.

**Researcher approved the proposed six-fit pilot after the toy smoke** (`yes`).
Operational split of the original 2100s ceiling: a new, one-CPU ledger dry-run
has at most **180 allocated seconds** and must execute **zero scientific workers**;
only if it passes may the full fit job be submitted with a reduced **1920s
(32-minute) one-CPU cap**. Thus the two allocations cannot exceed 2100s, even
if the dry-run uses less than its allowance. The exact dry-run-tested driver is
reused for the fit job; no scientific cell starts before verified binary,
loader/dependency, bundle, qualification-only gate and six-cell recipe checks.
A failed dry-run stops the phase; no automatic retry. The full runner is
one-shot: it atomically records a cell as running before launch, refuses an
existing output root and never automatically reruns any cell. Review scheduler
and ledger outcomes before interpreting the six-cell comparison.

**Ledger dry-run 7552414 failed before its first fake worker**, exit 1, **30
allocated one-CPU seconds**. Its `pilot_protocol.json` lacked the concrete
`bundle` path required by the six-cell executor (`KeyError: 'bundle'`); the
preserved `dry_run_screen/ledger.json` shows zero cells and zero scientific
fits. There is **no passing dry-run gate and no pilot job**. The driver now
adds `bundle=str(bundle)` after checking the frozen bundle identity; a local
fake-worker unit of all six request/ledger entries passed without launching a
worker. This corrected working source is **not** the failed job's frozen code.
The pilot phase is paused; no automatic retry. If the researcher explicitly
approves a correction-only continuation, the next no-fit dry-run may request
**at most 150s** (the original 180s less the charged 30s), and the still
unlaunched scientific stage remains capped at **1920s**, so the original
2100s cumulative ceiling cannot reset. Full fits still require a passing
corrected dry-run with the exact subsequent pilot driver.

**Explicitly approved correction-only continuation 7553328 passed**, exit 0,
17 allocated one-CPU seconds: six intercepted requests, zero real workers,
and completed-root reopening rejected. The corrected driver SHA256 is
`908c658afed6b5d2e1aa7bdff47ffa8fe4afedf2478ae323cc518bdeb03e171c`.
Preparation charged **30+17=47s**. Slurm displayed a minute-rounded three-minute
limit for the requested 150s continuation; actual usage was 17s, within the
remaining allowance. No further preparation is authorized by unused time.
**Scientific pilot 7553341 submitted**, six declared cells only, 1920s one-CPU
allocation ceiling; it reuses the exact passing driver from
`matched_pilot_dryrun_7553328/`. Results pending; no speedup or acceptance claim.
Evidence will be `run/nonconvex_solver_benchmark/matched_assembly_pilot_7553341/`.
Preparation plus maximum fit allocation is **47+1920=1967s**, within 2100s.

**Pilot 7553341 completed**, exit 0, **1632 allocated one-CPU seconds**;
preparation plus fits charged **47+1632=1679/2100s**, phase closed. Six cells
finished, zero accepted minima. L-BFGS hit 10,000 attempts on both starts;
Newton hit time caps. Recorded optimizer durations were approximately 265–267s
and charged cell durations 270–272s: these are comparable fixed-budget recipes,
not equal convergence, and L-BFGS's attempt cap must not be silently increased.

Analysis-only **7556591** used 28 allocated one-CPU seconds, separately charged,
with no numerical fits. Final checkpoint hashes match the ledger. All six
trajectories are plotted (no sampling/smoothing) in
`matched_assembly_pilot_7553341/audit/matched_solver_progress.png`; source CSVs,
phase totals, medians, counters and provenance are in that directory.

| Start | Cached Newton committed updates | Local Newton committed updates | Median Hessian, cached → local | Final energy reduction vs cached |
|---|---:|---:|---:|---:|
| Positive | 85 | 114 | 2.253 → 1.535 s | 2.72% |
| Negative | 87 | 114 | 2.229 → 1.538 s | 0.79% |

Thus the local kernel's 31.0–31.9% lower observed per-call median translates to
31.0–34.1% more committed Newton updates within the measured capped runs, not a
measured time-to-convergence speedup. The compared trajectories visit different
states, so their Hessian medians are descriptive, unlike earlier frozen-state
kernel parity/timing. Local Newton's final energy is 2.22% / 0.89% below L-BFGS,
but its final native gradients (3.61e-9 / 4.59e-9) remain far above 5e-14 and
are not uniformly better than L-BFGS (2.96e-9 / 2.73e-9).

The curves show L-BFGS ahead early, with local Newton reaching lower energy
late. No plateau is established: between the last observation at/before 132.5s
and termination, energies fell 21.0% / 17.8% for local Newton, 22.3% / 19.9%
for cached Newton and 4.60% / 4.18% for L-BFGS. These percentages describe
progress over the second half, not proof that further time guarantees a minimum.

Remaining bottlenecks: local Newton spent 353.68s in recorded Hessian calls
across its two cells (~65.3% of charged cell time), 90.11s factorizing and 43.47s
in symbolic analysis. L-BFGS spent 456.28s (~84.5%) evaluating energy/gradient,
versus 34.29s computing directions. Model timers include Hessian/gauge/reduction/
symmetry; do not double-count nested totals. Unattributed time includes partial
terminal timers and verification/I/O, not measured I/O alone.

**Next decision, not launched:** profile element-level native energy/gradient
cost for L-BFGS, or assess a benchmark-only local-Hessian element parallelism
smoke for Newton (thread safety, parity, actual charged-core cost). Symbolic
reuse alone has a smaller ceiling (~8% of local Newton charged time here).
A hybrid L-BFGS→Newton protocol is suggested by early/late curve behavior but
remains an untested algorithmic change requiring a separate declared switch
rule and matched controls. Do not extend capped fits or claim global optimality.
PNG reviewed; PDF rendering and grayscale/CVD checks remain pending.

### Convergence-focused comparison — protocol preparation and toy smoke only

Researcher approved protocol preparation and a small hybrid smoke, **not longer
shell runs**. Compare native L-BFGS, CHOLMOD/local-dihedral Newton and a one-way
L-BFGS→Newton hybrid on the same two prescribed starts and unchanged objective,
scaling, material, constraints and native stopping target. Report separately
(1) accepted residual/curvature status, (2) time to acceptance and (3) energies
among accepted endpoints. Capped residual candidates cannot win a minimum
comparison. Positive restricted pivots are a numerical local-curvature check,
not a proof of global optimality. No held-out/generalization claim is possible.

Proposed hybrid switch: after **2000 cumulative L-BFGS attempts**, at the next
safe boundary, if and only if the state remains `ready`; earlier accepted
convergence ends the fit, while caps/failures terminate it rather than trigger
Newton. This is a fixed exploratory recipe, not a threshold optimized on test
results. Newton starts at the **current**, not best-residual, state. Retain the
original phase checkpoint unchanged; create a new explicitly identified phase
recipe with fresh Newton memory, carrying the objective state, native gradient,
selected candidate, complete trace, elapsed time and all consumed counters.
Total time/evaluation/attempt limits apply across both phases, never reset.
Any new overall attempt ceilings and longer wall budgets must be specified and
approved before comparative fitting, including L-BFGS's prior 10,000-attempt
ceiling; do not silently relax historical recipes or continue capped cells.

First smoke: standalone benchmark transition helper with the existing engine
on a Rosenbrock toy, **300s one-CPU cap including compile**, no shell evaluation
or scientific fit. Exercise switch at 6 attempts (shortened test fixture only),
pre/post-switch checkpoint restoration, resource conservation, fresh working
memory and terminal/wrong-recipe/problem refusals. Matching frozen compiler
flags avoid the earlier ABI mismatch. Production code and qualified worker
remain unchanged. Stop on failure; no automatic retry. This toy uses Eigen,
not the final CHOLMOD shell hybrid, and is only an initial unit gate. Shell
integration must additionally qualify atomic hybrid phase/recipe identity,
process interruption during switch, topology/constraint and dependency checks,
and a no-fit full-size adapter gate before any longer convergence experiment.

**Initial toy smoke 7587331 passed**, exit 0, **47 allocated one-CPU seconds**
of the 300s cap; phase closed. It compiled a standalone helper against frozen
engine headers/HLBFGS library with matching compiler flags, then verified
six-attempt phase transition, resource preservation, fresh Newton working
memory, pre/post-switch disk checkpoint restoration, and rejection of terminal
states and wrong phase/problem identities. Evidence:
`run/nonconvex_solver_benchmark/hybrid_smoke_7587331/smoke_gate.json` and
`source_sha256.txt`. **Zero scientific fits**. Production Hessian and qualified
worker hashes remain unchanged. This is **not** process-kill recovery, an
atomic multi-phase journal, CHOLMOD hybrid shell qualification or evidence of
improved convergence. A caller-level fixed hybrid recipe must enforce matched
phase tolerances/objective and cumulative budgets and survive interruption
between phase checkpoints; the standalone helper is not a fit launcher.

Next recommended authorization is a separately capped **15-minute one-CPU
benchmark-only hybrid integration/operational smoke**, with no comparative fits.
If that qualifies, propose the longer six-cell convergence budget and explicit
new attempt ceilings for researcher approval; none are approved here.

**Hybrid integration qualification authorized:** one new **900-second one-CPU**
allocation, no optimization fits. Build a separate hashed worker from frozen
shell/engine libraries and matching compiler/link flags; do not mutate the
qualified solver. Test CHOLMOD toy handoff with durable journals at four
boundaries: before target checkpoint, after target publication/before phase
commit, after phase commit, and after three Newton updates. Kill with SIGKILL
only after durable pause evidence; restore without changing policy or source
checkpoint. Reject changed switch/runtime identity, exhausted budgets and
completed-run reopening. A full-size synthetic switch checkpoint at the
2000-attempt threshold exercises current-shell native gradient and Hessian
parity but executes **zero shell optimizer steps**; counters are labeled
synthetic. Preserve failures and stop without automatic retries. This qualifies
controlled boundary recovery, not arbitrary mid-step recovery or convergence.

**Qualification 7588808 completed**, exit 0, **170 allocated one-CPU seconds**
(143.271 CPU-s; MaxRSS 3076252K), within 900s; phase closed. Four CHOLMOD toy
SIGKILL/restorations passed at the declared durable boundaries, preserving
source checkpoint bytes and matching uninterrupted final states/counters.
Changed switch policies, wrong runtime identities, exhausted elapsed budgets
and completed-run resumes were rejected. The full-size **32,267-DOF synthetic
handoff** preserved native energy/gradient and had relative Hessian difference
**0** versus native TinyAD. It executed **zero shell optimizer steps** and
**zero scientific fits**. Source/frozen-library hashes were checked after the job.
Worker SHA256:
`cc8026c30d60a87b45339b8c30d990a933af84a947d546e740e0c7fa08cddce0`;
new runtime dependency digest (not interchangeable with the prior worker):
`ab286055bb21e510cb83a7e7e50c266a47848357155704e526b0342202fa1aad`.
Evidence: `run/nonconvex_solver_benchmark/hybrid_shell_gate_7588808/qualification_gate.json`.

Limits: only durable-boundary pauses are eligible; no arbitrary crash recovery,
full-shell process-kill continuation, convergence superiority or production
adoption established. This is still a qualification worker, not a longer-run
fit dispatcher. A scientific runner must freeze the full hybrid recipe and
cumulative counters in its journal, perform final independent native-gradient
and restricted-curvature checks, and pass its own no-fit launch/ledger gate.

**Proposed next budget, not authorized:** six fresh fits (three recipes × two
original starts), **20 minutes per cell**, plus at most **10 one-CPU minutes**
for runner build/qualification, overhead and reporting: **130 one-CPU minutes
cumulative**. Within each cell reserve time for checkpoints and final native/
curvature verification; both phases share the same cell deadline. Proposed
new attempt ceiling 100,000 and evaluation ceiling 200,000 for *all* recipes,
with Hessian cap 10,000, so native L-BFGS is not stopped by its old 10,000-attempt
pilot ceiling. These are explicit new recipes, not extensions of past capped
fits. Keep hybrid switch fixed at 2000 attempts and never switch from a capped,
failed or accepted state. Time budgets and those enlarged ceilings require
researcher approval before implementation/launch; no promise of reaching 5e-14
or acceptance is implied by a longer allocation.

**Researcher approved the 130-minute cumulative budget and the new ceilings.**
Operational split: one **600s / 10-minute** one-CPU preparation allocation and,
only after its no-fit gate passes, one **7200s / 120-minute** one-CPU scientific
allocation. Six new cells, fixed interleaving: positive start L-BFGS, Newton,
hybrid; negative start hybrid, Newton, L-BFGS. Each cell has at most 1200 charged
seconds, an 1140s cumulative optimizer deadline and a 1190s worker watchdog;
remaining time is reserved for checkpoint and independent native/curvature
verification. Hybrid elapsed/attempt/evaluation counters span both phases;
no resets. Attempts 100,000, evaluations 200,000, Hessians 10,000 for every
recipe, switch fixed at 2000 only from live `ready` state. No recipe selection
or automatic extension from interim results. Capped/failed fits never promoted.

The actual longer-run worker is separately compiled and tested with all three
recipes, toy durable-pause/restart before/after hybrid transition and during
Newton, mismatch/completed-cell refusal, full-size synthetic no-step parity,
and an intercepted six-cell ledger. Only the exact hashed passing worker and
driver may be used for fits; runtime libraries and frozen bundle rechecked
before every real process. A failed preparation stops this phase without retry.
Scientific launcher is one-shot, refuses existing cells, and stops on numerical
or infrastructure failure; normal configured caps may advance to the next
predeclared cell. No mid-step recovery is authorized. The old pilot and gates
stay untouched. Numerical lower energies without accepted residual/curvature
remain unfinished trajectory evidence, not more accurate local minima.

**Preparation 7592200 passed**, exit 0, **255 allocated one-CPU seconds** of
600s. The actual worker passed pure L-BFGS and pure Newton progress recovery,
four hybrid handoff/progress recovery boundaries, policy/completed-cell
refusals, full-size synthetic no-step handoff parity and the intercepted six-cell
ledger/reopening check. Zero scientific fits. Source/library and driver hashes
were rechecked against its gate before scientific submission. Worker SHA256
`abb52ac5be1f502bc245862b2d7f5f096ad9ff9f6ae631931aaa80634d41f04b`;
runtime digest
`8746c7591fbf5ad092f7ff53b8712115675a66a14def02aec3107064fe89ec00`;
driver SHA256
`99c9bd0e30d6b3c08608aca24262860803ea77110460e70618cfbf3774ab5c7d`.
Evidence: `run/nonconvex_solver_benchmark/convergence_prepare_7592200/gate.json`.

**Convergence pilot 7592522 submitted**, six fresh cells, at most 7200 allocated
one-CPU seconds, same frozen inputs and exact prepared worker/driver. Preparation
plus maximum scientific allocation is **255+7200=7455/7800s**, without reclaiming
unused preparation time for fits. Authoritative cell ledger:
`run/nonconvex_solver_benchmark/convergence_pilot_7592522/screen/ledger.json`.

**Completed 7592522**, exit 0, 6727 allocated one-CPU seconds; total preparation
plus fits **255+6727=6982/7800s**, phase closed. All six cells finished. Both
hybrid runs reached `gradient_target` with independently recomputed native
norms **7.6596e-15 / 1.0101e-14** and positive unshifted restricted pivots;
energies **1.1063513010514394e-9 / 1.1063513010514392e-9**. These are the first
two accepted minima in this benchmark. Charged cell times were 1092.45s and
1019.75s. Both pure Newton and both L-BFGS cells time capped without acceptance.
No capped cell was promoted or extended; accepted endpoints are not a global
minimum certificate. The full long-run timing/curve audit is recorded below.

**Accepted-shape comparison 7608228**, analysis only, 18 allocated one-CPU
seconds, no fits: all 5427 corresponding vertices, 15986 edges and 10560 faces
were compared. Selected and final endpoint arrays are byte-identical per run;
checkpoint hashes matched the ledger and the immutable input bundle was
verified. Proper rigid Kabsch alignment (minus→plus; determinant +1) permits
translation/rotation only, no reflections, scaling, permutations or deformation.
Raw vertex RMS difference **8.571 mm** falls to **4.115 nm RMS**, **12.113 nm
maximum**; reference-area-weighted alignment gives 4.052 nm weighted RMS.
Relative RMS to the reference diagonal is 1.037e-8. Stored director-angle
RMS difference is **4.279e-10 rad** (max 4.362e-9); edge-length relative RMS
is 1.417e-10 and aligned face-normal RMS is 4.674e-6 degrees. The energy
absolute difference is 2.068e-25. Thus these two accepted endpoints agree to
very high numerical accuracy as the **same local shell state modulo rigid
motion**, not merely equal energy. This does not establish uniqueness across
other starts, exact equality or global optimality. No new oracle evaluation
was performed; acceptance is inherited from saved independent verification.
Evidence: `convergence_pilot_7592522/shape_comparison/comparison.json`, full
vertex/edge residual CSVs; reproducible script
`python/compare_accepted_hybrid_shapes.py`. No new figure generated in that check.

### Long-run performance and shape plots — analysis only

Read-only audit **7617511** used 42 one-CPU seconds; layout correction/export
**7617915** used 29s. No fits or native oracle calls. Final checkpoint hashes
matched the ledger, final arrays matched checkpoint states, and bundle hashes
were verified. Every declared case, matched mesh point and finite energy/gradient
trace observation is retained; no smoothing/downsampling or independent-replicate
claims. Recommended figures and numeric tables are in
`convergence_pilot_7592522/performance_audit/reviewed/`; parent exports are the
preserved initial layout. Source: `python/audit_convergence_benchmark.py`.

Four plots: `convergence_progress` (energy/residual against recorded optimizer
time), `convergence_timing` (exclusive charged-time categories),
`convergence_final_shapes` (all six rigid-aligned full midsurfaces, common scales
and physical aspect), and `convergence_shape_differences` (all material vertices
relative to the same-start accepted hybrid). PNG/PDF and captions supplied.
Final PNGs and progress/timing grayscale reviewed; PDF rendering and CVD checks
remain unverified. Visual overlaps in surfaces do not constitute a self-contact
or physical-admissibility check. Capped shapes are unfinished, not certified
alternative minima. Full vertex residuals are in CSVs, not just selected views.

**Observed hybrid bottlenecks**, summed across its two dependent starts:

| Exclusive category | Seconds across two runs | Share of charged time |
|---|---:|---:|
| Hessian oracle | 1200.20 | 56.82% |
| Other model construction | 165.44 | 7.83% |
| Numerical factorization | 249.97 | 11.83% |
| Symbolic analysis + solve | 246.80 | 11.68% |
| Native energy/gradient | 175.55 | 8.31% |
| L-BFGS direction | 12.47 | 0.59% |
| Unattributed residual | 61.76 | 2.92% |

The mean charged hybrid time is **17.60 minutes**. Its initial 2000 L-BFGS
attempts took **78.02 / 79.80s**; thereafter it required **371 / 356 Newton
updates**, with **144 / 136 backtracked steps**. Only **14 / 11** Newton
updates used nonzero shifts; **357 / 345** were unshifted. Thus repeated
indefinite-factor retries are not the dominant cost of the successful hybrid,
and positive curvature alone did not produce rapid whole-trajectory convergence.
The Hessian median was **1.668 / 1.656s per call**. The code constructs a fresh
factor object each outer Newton step; symbolic analysis alone cost **224.36s
across both fits (~10.62%)**. Pattern/gauge identity must be checked before
any reuse; equal nnz alone is insufficient. Timers are nested: subtract the
Hessian from its parent model timer rather than double-counting both.
Unattributed time includes checkpoints/verification/incomplete timers, not a
measured I/O total. Native L-BFGS separately spent **84.57%** evaluating
energy/gradient; it did not converge within its cell budget.

The first recorded gradient crossing of **1e-10** occurs at **1058.64 / 997.33s**,
and **5e-14** at **1083.17 / 1010.85s**: just **24.53 / 13.52s** later. The
strict final residual requirement is therefore not the bulk of the observed
17–18 minute cost; loosening it is not the recommended acceleration.

**Proposed priorities, not run:** (1) preserve symbolic analysis and restriction/
assembly mappings across outer steps only when exact pattern/gauge guards pass;
(2) test Hessian element parallelism at 1/2/4 cores or a guarded lagged-Hessian/
factorization strategy, initially on small smoke fixtures; (3) for related
production samples, consider explicitly budgeted warm starts rather than two
cold starts for every sample. Hessian lagging changes the numerical recipe and
must preserve native stopping and fresh final curvature verification; additional
cores trade wall time against charged core-seconds. No production adoption or
speed claim follows from these proposals.

Amdahl illustration only, holding iteration counts/other costs fixed: halving
Hessian cost would reduce mean charged time from **17.60 to 12.60 minutes**;
a fourfold Hessian speedup would yield **10.10 minutes**. Neither is measured,
and threading is not assumed to scale ideally. Getting well below this needs
fewer expensive Newton/model updates or better initialization, not kernel work
alone. A future trial must compare time to *accepted* minima, not just iteration
throughput, and check held-out starts/objectives rather than tune indefinitely
on these two related trajectories.

### Representativeness check: growth magnitude versus original zigzags

User questioned whether the large benchmark deformation reflects stronger growth
than the original experiments. Read-only JSON/CSV/source inspection found:

- Original `run/solver_characteristics/shrinking_crown/` recipes
  `crown_nested_broad_to_central/sequence.json` and
  `crown_nested_central_to_broad/sequence.json` both prescribe top growth
  **0.00012 per hit (0.012%)**, bottom zero, for six toolpaths.
- `run/forward_model_diagnostics/current/nested_baseline/` uses the same
  0.00012 per-hit growth and 0.6 mm thickness; its final summary records
  **5.314 mm raw maximum displacement**, with U3 from **−5.305 to +3.673 mm**.
  Its mesh parameter is 0.03 rather than the benchmark's 0.015; this is not a
  controlled growth-only deformation comparison. Raw displacement includes
  rigid motion and is not a Kabsch-aligned deformation metric.
- The benchmark uses **0.005 per hit (0.5%)**, **41.67 times** the original
  nominal growth. The actual input bundle's `hits.csv` contains 15109 hits,
  all top 0.005 and bottom zero, with up to six hits on a face. Its isotropic
  multiplicative target therefore reaches **(1.005)^6−1 = 3.03775%** local
  accumulated top-layer natural-length growth. This is not a uniform sheet
  growth, nor a time rate. Bundle:
  `local_dihedral_solver_gate_7547943/preflight/bundle/`.
- Later `nested_recheck_g0p005_broad_to_central_standard/sequence.json` and
  `nested_absstop_g0p005_broad_to_central_res0p015_standard/sequence.json`
  are parsed-JSON identical to the frozen six-cycle benchmark input sequence.
  Thus the benchmark matches a *later high-growth stress test*, not the original
  low-growth recipe. The later high-growth recheck already recorded large raw
  displacement: **252.05 mm broad-to-central / 267.18 mm central-to-broad**.
  These are saved original-frame diagnostics, not aligned comparison metrics;
  their historical acceptance is not the benchmark's new curvature certificate.
- Loading protocol also differs: earlier nested runs warm-start successive
  growth increments; the benchmark solves the complete final target from a
  shallow prescribed start. Same final growth need not select the same local
  state. A matched continuation/target comparison was not run here.
- The archived multi-zigzag sensitivity command points to
  `scripts/examples/multi_zigzag_orthogonal_patches.json`, which is absent at
  that path; no matching JSON was found within the bounded depth-five search.
  Actual per-patch multi-zigzag growth was **not verified**; logged command-line
  growth defaults must not be substituted for missing patch definitions.

**Interpretation:** the user's concern is supported for the original nested
workload. Large shape changes cannot be attributed solely to improved Newton
convergence. The 17–18 minute timing is high-growth benchmark evidence, not an
established cost per original low-growth sample. Whether 0.005 is physically
excessive requires calibration; low residual/positive pivots establish local
numerical equilibrium, not calibrated physical realism or contact admissibility.
Proposed next comparison should use the intended original growth and loading
protocol with unchanged numerical acceptance, rather than silently lower the
load in this closed benchmark. No new fits, plots or oracle evaluations here.

### New ABC profiled/hardened objective — qualification only

User supplied an eight-pass recipe A×2, B×3, C×3, then requested **ortho −0.5
on every strip** and a single solve after all target updates. Original/revised
JSON and geometry preview are preserved under `run/zigzag_abc_preview/`.
Shared face hit history and multiplicative anisotropic metric composition are
retained in order, but **no intermediate equilibrium** is requested. The
254 × 304.8 mm sheet, 0.6 mm thickness and 32,267-DOF mesh are carried over.
These remain model choices, not parameters supplied in that JSON.

No-fit gate **7624619 passed**, **130/600 allocated one-CPU seconds**, preparation
phase closed. Evidence: `profiled_zigzag_gate_7624619/{gate.json,
target_verification.json,derivatives/preflight.json,bundle/}` under the benchmark
run root. Native target builder `test/diagnostics/PrepareProfiledZigzag.cpp`
uses the frozen native parser, hit mapper, hardening and material-metric update;
these three native headers were hash-equal to current source before launch.
Driver `python/qualify_profiled_zigzag.py` independently reconstructs every
hit/angle/profile/history value and all eight cycle metrics in Python.
Frozen builder/driver, compile/link commands, runtime hashes and input are saved.
No production kernel or solver was changed.

- **8450 ordered hits**, **3001 hit faces**, maximum **6 hits/face**. Path A has
  1597 hits per repeat; B and C each have 876 hits per repeat.
- Maximum relative discrepancy over all eight target tensors **3.836e-16**;
  bottom-layer roundoff-only relative change **5.356e-16**.
- Maximum accumulated principal natural-length growth **0.83254%**, not a
  uniform sheet strain; positive-definite material metrics throughout the native
  update. This is lower than the previous objective's 3.03775% maximum but does
  not itself predict optimizer difficulty or establish physical calibration.
- Six-start native/AD gradient errors ≤**2.30e-20**; finite-difference HVP errors
  ≤**9.563e-9**. Local-dihedral versus original Hessian had **zero measured
  difference** and identical CSC structure for all six starts.
- Initial checkpoint roundtrips passed. Hybrid durable-boundary recovery is
  inherited from the previously qualified worker; no arbitrary-crash or new
  full-shell interruption test is claimed. **Zero optimizer steps/fits**.

Bundle identity:
`9f79ed808b6dd6ccca171f5c72538837fe6400653f0f6b4632dcc407e5e259f0`.
The 4 mm strips are narrow relative to this mesh: centroid hit sampling and
cross-language agreement do not establish spatial convergence. Geometry preview
shows only **0.425 mm** clearance to the sheet boundary for B/C. Contact and
experimental adequacy remain unvalidated.

**Next phase approved (45 one-CPU minutes):** two hybrid solves from the unchanged
±y-cylinder starts, fixed 2000-attempt switch, final-target-only optimization,
original per-cell attempt/evaluation/Hessian ceilings and 20-minute charged
cell envelope, native gradient ≤5e-14 plus positive unshifted restricted pivots
for acceptance. The user approved a fresh **45 one-CPU-minute** ceiling including
launch checks and reporting (no retries/extensions). The driver reserves 150s
for reporting and 30s for finalization, refusing another cell unless its full
1200s envelope fits. Before scientific fits, exercise two-cell dry-ledger and
reopening refusal, exhausted-budget skips, simulated-worker failure stopping,
exact frozen-worker synthetic shell parity and durable toy restart. Only the
suite's fixed six-cell completion count is generalized to the declared cell
count; the frozen solver binary and recipes are unchanged. Preserve all trajectories and
exclusive timers. Diagnose this recipe before proposing a larger three-method
comparison; do not claim a method ranking from two hybrid starts. Execution
request and current status: `run/zigzag_abc_preview/optimization_request.md`.

Submitted **7629263** using
`scripts/nonconvex_benchmark/profiled_zigzag_pilot.sbatch`; evidence root
`run/nonconvex_solver_benchmark/profiled_zigzag_pilot_7629263/`.
**Completed in 539/2700 allocated one-CPU seconds**, two accepted hybrid minima;
phase closed. Positive/negative starts charged **251.808 / 245.714s**, with
**77 / 65 Newton Hessians** after the unchanged 2000-attempt warm-up. Native
residuals **2.29017e-14 / 4.45977e-15**, both `gradient_target` and positive
restricted pivots. Exclusive aggregate costs: Hessian **46.57%**, native
energy/gradient **22.84%**, numerical factorization **9.42%**, symbolic+solve
**8.31%**, other model **5.40%**, direction **1.93%**, unattributed **5.52%**.
Hessian median **1.629 / 1.630s**: lower whole-solve time came mainly from fewer
Newton updates, not faster Hessian kernels. Drivers:
`python/run_profiled_zigzag_pilot.py`, `python/report_profiled_zigzag_pilot.py`.
Numerical report and figures saved under `audit/`; visual review remains pending.
This validates local equilibria on one mesh, not mesh convergence.

### Opt-in Newton symbolic reuse — no-fit qualification

User approved a fresh **600 one-CPU-second implementation/qualification compute
ceiling**, no scientific fits. Candidate changes are limited to benchmark
headers `BenchmarkNewtonFactorization.hpp` and `NonconvexBenchmark.hpp`.
`Config::reuseNewtonSymbolic` defaults false; false preserves the existing JSON
recipe representation. True is explicit in checkpoint identity and rejected for
non-Newton methods. The cache is process-local, not serialized: restoration must
perform a fresh symbolic analysis, retaining all optimizer/resource state.

The guard compares matrix dimension, **complete compressed CSC column/row
indices**, and the **exact restricted DOF sequence**. A changed gauge selection,
ordering or sparsity pattern rebuilds the factorization object. Equal nnz alone
cannot authorize reuse. Fresh Hessian construction, shift search, numerical
factorization, residual and physical acceptance checks remain unchanged.

Submitted no-fit qualification **7635939** via
`scripts/nonconvex_benchmark/symbolic_reuse_qualification.sbatch`; isolated copied
headers/source, matching-ABI flags, library hashes and logs go to
`run/nonconvex_solver_benchmark/symbolic_reuse_gate_7635939/`.
Test source `test/diagnostics/QualifySymbolicReuse.cpp` exercises both Eigen and
CHOLMOD guards, value changes, same-nnz pattern changes, restricted DOF reorder,
dimension changes, recovery after indefinite numerical factors, toy trajectory
parity, checkpoint identity refusal and cold-cache disk restoration. Saved full-
size positive-start initial/handoff/accepted states from **7629263** provide
three fixed-state shifted linear-system parity/timing checks, three repetitions
each. These are diagnostic shifts to ensure positive definiteness, not shell
Newton steps, accepted directions or end-to-end speedup evidence. Input saved
checkpoint checksums and target identity are checked by the native test.

**Passed, 190/600 allocated one-CPU seconds**, phase closed; zero scientific
fits and zero shell optimizer steps. Both GTest cases passed (one umbrella
qualification exercises all guard/toy/saved-state checks, plus the existing
accepted-endpoint policy test). Production Hessian, shell solver and HLBFGS
source hashes were unchanged before/after. Candidate binary SHA256:
`17e9ef7ca418f0d11cd1ef4b8cf60a63be3eec57b59dbdb1bcd1e1567d75b02b`;
runtime identity:
`c1b77755524ed640ea9444859f9a4e4c4f669ba2747809dfb8e935d4a11b30d7`.

For all nine full-size shifted diagnostic solves, cached versus freshly analyzed
solutions had **zero measured difference**, linear residuals **3.82–4.02e-16**.
Fresh symbolic setup took **0.223–0.256s**; the six warm matching-pattern checks
cost **0.00118–0.00172s**. Cold cache analysis remained **0.232–0.261s**.
All three saved states had different reduced patterns (nnz **1739033, 1739031,
1738957**), and each correctly triggered reanalysis. These are saved-state
kernel observations, not nine independent replications or whole-solve speedups.

Read-only inspection of the previously completed **7629263** trace found
`model_nnz` changed at **49/76** adjacent positive-start Newton transitions and
**43/64** negative-start transitions. Equal nnz would still not guarantee equal
CSC/gauge, so this suggests frequent invalidation rather than an assured large
cache hit rate. Symbolic analysis consumed **20.342 / 17.209s** of those
**251.808 / 245.714s** fits: even eliminating all symbolic work would save only
about **7.55% aggregate**, holding all other work/trajectories fixed; actual reuse
must save less when patterns change. No actual trajectory cache hit rate or
end-to-end speedup has been measured.

The option is implemented/qualified directly in the benchmark engine. Historical
convergence binaries remain unchanged; no new scientific launch worker or CLI
has been qualified with this option. A separately approved integration/pilot is
required before scientific use. Given the modest upper bound and frequent
pattern changes, prioritize the remaining Hessian cost before a costly full
rerun solely to demonstrate symbolic caching.

### Guarded Hessian reuse — approved no-fit qualification

User confirmed a **fresh 600 one-CPU-second qualification ceiling**, separate
from the closed symbolic-reuse phase; no scientific fits. Candidate adds
`Config::guardedHessianReuse` (default false) to the benchmark engine only.
The opt-in is explicit in checkpoint identity; false preserves legacy config
and state JSON. This is a modified-Newton recipe, not an exact-Newton speedup.

Predeclared conservative policy:
- Only a **fresh, unshifted, full accepted step** seeds a reusable Hessian.
- At most **one further update** may use that reduced Hessian. Current native
  energy/gradient and a fresh numerical factorization are still used.
- Before reuse, exact restricted DOF ordering must match; otherwise rebuild at
  the current state. Guard time remains inside model timing.
- A stale model receives one full-step Armijo trial. Rejection restores the
  current state, clears the model, records a rejected/charged attempt and
  requests a fresh Hessian on the **next budgeted attempt**. It does not retry
  a failed fit or reset counters. Numerical/oracle failures remain terminal.
- Fresh-model backtracking and shift search are unchanged. Backtracked or
  shifted accepted steps do not seed reuse. Terminal states clear the cache;
  final certification must independently recompute the native gradient and a
  fresh unshifted Hessian, as before.
- Persist cached matrix, restricted map and reuse age together. Loading a
  cache-valid guarded checkpoint with missing/reset age is refused. Restoring
  must preserve the remaining reuse allowance and all resource counters.

Tests: `test/diagnostics/QualifyGuardedHessian.cpp` includes the previous symbolic
qualification unchanged, adds Eigen/CHOLMOD toy checks for all guards, rejected
attempt accounting, numerical failure, iteration/Hessian caps, disk restore,
age/recipe refusal, fresh final toy oracle and composition with symbolic reuse.
A **synthetic** full-size cached-model save/load/restore checks storage and
resource state at the saved accepted shell coordinates; no shell optimizer
steps. Launcher `scripts/nonconvex_benchmark/guarded_hessian_qualification.sbatch`
and shared driver `python/qualify_symbolic_reuse.py --guarded` freeze sources,
retain matching ABI/runtime checks and verify unchanged production kernels.

Submitted no-fit qualification **7637624**; evidence root
`run/nonconvex_solver_benchmark/guarded_hessian_gate_7637624/`.
**Passed, 171/600 allocated one-CPU seconds**, qualification phase closed.
Three GTest cases passed: prior accepted-endpoint policy, symbolic-reuse
regression umbrella, and guarded-Hessian umbrella. Both Eigen and CHOLMOD
passed every listed guard, cap and restoration test. The smooth toy reached
`gradient_target` in seven attempts with four Hessian calls; this demonstrates
the bounded reuse policy, not shell performance. The full-size synthetic cached
model (1,738,957 stored entries) survived disk save/load/restore exactly, with
reuse age and counters intact: **51,317,092 bytes**, **2.671s** for the combined
save/load/restore. This overhead must be counted in any later timing comparison.
No shell optimizer steps or scientific fits occurred. Production Hessian, shell
solver and HLBFGS source hashes were unchanged before/after.

Candidate binary SHA256:
`6cefdd3bd84265595d4fe695c93ecfe47be4c33d53a563653509566d68858097`;
runtime identity:
`0d99ef854629fec23b39b6c7175328a1a8b9663cebc2716d58fbed2232e66913`.
Evidence: `guarded_gate.json`, `gate.json`, `completion.json`, `tests.log`, frozen
source and synthetic checkpoint in the job directory above. No performance
result or new scientific convergence-launch worker is yet qualified. Future
reports must distinguish rejected stale attempts from committed Newton updates.
Actual benefit depends on cache eligibility, moving gauge, extra evaluations
and checkpoint overhead; a changed trajectory or basin is possible despite
unchanged final acceptance.

**Next phase authorized: 50 one-CPU minutes total.** Integration plus two
guarded hybrid solves on the unchanged ABC bundle and ±y-cylinder starts, with
symbolic reuse **off** to isolate the guarded-Hessian change. Preserve the
earlier two accepted exact-Hessian baseline fits; do not rerun/reopen them.
The ceiling covers frozen-worker integration/build/restart checks, two original
20-minute cell envelopes and saved-history reporting. Reserve 210s for reporting
and 30s for finalization; skip unstarted cells if a full envelope cannot fit.
Require unchanged final native/curvature checks and compare energy/aligned shape
as well as time-to-acceptance; do not call a different basin a like-for-like
speedup. No retries or automatic extensions.

Integration uses byte-identical numerical headers from **7637624**, changing
only handoff/request plumbing: the Newton-phase guarded flag is recorded in
both phase policies, validated on resume, and allowed to differ from native
L-BFGS at the fixed 2000-attempt handoff. Other shared phase controls remain
identical. Symbolic reuse requests are rejected. Worker finalization clears
terminal guarded caches, including outer watchdog/time-limit boundaries.
Before fits, the actual executable must pass full-size synthetic handoff parity,
continuous versus restored toy paths at all four durable boundaries, wrong-
policy and completed-cell refusals, plus dry-ledger/budget/failure-stop tests.
No arbitrary mid-step crash recovery is claimed.

Launcher: `scripts/nonconvex_benchmark/guarded_hessian_pilot.sbatch`.
Driver: `python/run_profiled_zigzag_pilot.py --guarded`.
Reporting: `python/report_guarded_hessian_pilot.py` plus shared profile/audit
helpers. Numeric audit separately counts Newton attempts, committed updates,
accepted reuse, stale rejections and restriction refreshes. All available
baseline/candidate curves and meshes are retained; missing/failed cells stay
explicit. Compilation and reporting failures preserve partial evidence.

Submitted **7642652**; evidence root
`run/nonconvex_solver_benchmark/guarded_hessian_pilot_7642652/`.
**Completed in 824/3000 allocated one-CPU seconds; phase closed.** Both guarded
cells reached `gradient_target`, independently passed the native 5e-14 gate and
positive restricted pivots. Reports are complete but visual review remains
pending (`audit/comparison.json`, PNG/PDF comparisons).

Crucial negative result: **zero Hessian reuse attempts in either run**. Every
eligible cached model was invalidated by the changed restricted DOF selection:
**48 / 46 refreshes**. The restriction is selected anew from a QR-pivoted
rigid-mode basis at each geometry (`ShellBenchmarkProblem::freeDofs`); the
conservative guard correctly refused to reuse the old reduced-coordinate model.
Hessian counts therefore remained **77 / 65**, exactly the baseline counts.
Final coordinate files are **byte-identical** to their same-start baseline
(hash-checked); energies and gradients also exactly match the saved baseline.

Charged times were **288.408 / 253.477s**, versus baseline **251.808 / 245.714s**
(+14.53% / +3.16%). Do not attribute the entire difference to guard overhead:
the positive-start L-BFGS handoff alone moved from **50.552 to 80.990s**, before
guarded Newton began (negative: **71.992 to 68.745s**). Different execution
conditions/code builds and checkpoint overhead are possible contributors.
No end-to-end improvement was demonstrated. The candidate's aggregate Hessian
share remained **43.17%**; native energy/gradient **25.25%**.

This does not show that all Hessian reuse is ineffective; it shows that this
one-extra-update cache of the **reduced** Hessian is incompatible with the
observed moving gauge often enough to prevent any reuse. Any next approach
must address coordinate consistency (e.g. cache a full-coordinate Hessian and
re-restrict it, or separately qualify a stable gauge), not simply remove the
safety guard. Such changes and further fits require a new approved protocol.

### Opt-in parallel local-dihedral Hessian — bounded kernel checks, 2026-09-27

User requested parallelization after reviewing the guarded-reuse result. Implemented
in `test/diagnostics/LocalDihedralCachedTinyADHessian_Bilayer.hpp`, benchmark only:
constructor fourth argument selects positive assembly thread count (default **1**).
OpenMP evaluates independent faces into private face slots; serial face/r/c CSC
scatter preserves the original addition order. Mesh updates, pattern validation,
and rest-curvature snapshot happen outside the worker region. Worker exceptions
are captured per face and rethrown after joining. No atomics, global OpenMP setting
changes, production edits, solver-policy changes or automatic parallel enablement.
Parallel scratch costs 441 doubles/face (~35.5 MiB at 10560 faces). Kernel instances
remain non-reentrant, like their mesh/pattern cache. Builds without OpenMP reject
requests above one thread; that fallback has not been separately compiled here.

`ProfileLocalDihedralCachedHessian.cpp --parallel-small-only`: job **7649187**,
completed, 51s allocation with four CPUs. Twelve cases (free/clamped × flat/curved
× 1/2/4 threads) passed energy/native-gradient and Hessian/HVP checks. Parallel
CSC values matched serial **bitwise**, including repeated calls; actual team sizes
matched requests and cache-build count remained one. Invalid nonpositive thread
counts were rejected. This is not a sanitizer/race-detector qualification.

`--parallel-state bundle state`: job **7649255**, completed, 59s allocation with
four CPUs. Read-only ABC `profiled_zigzag_gate_7624619/bundle/cylinder_y_plus.f64`,
10560 faces / 32267 DOFs. All cold/warm matrices matched serial bitwise. Two warm
calls per thread count in balanced order 1,2,4,4,2,1:

| Threads | Mean warm assembly | Kernel speedup vs 1 | Threads × wall seconds (proxy) |
|---|---:|---:|---:|
| 1 | 1.400729 s | 1.000× | 1.400729 |
| 2 | 0.713904 s | 1.962× | 1.427809 |
| 4 | 0.380775 s | 3.679× | 1.523101 |

Cold calls including pattern initialization: 1.541262 / 0.883804 / 0.534236s.
The last column is not measured CPU time or scheduler billing: both check jobs
reserved four CPUs throughout, including compilation and serial checks (440
allocated core-seconds total). One state and two warm calls are only preliminary
kernel timing, not an end-to-end speedup, mesh-scaling result or broad qualification.
No solver fits, completed-cell restarts, or historical-budget extensions occurred.
Further solver integration/restart qualification and a matched trajectory pilot
remain separate decisions. No timing-based pass threshold was used.

Evidence roots: `run/nonconvex_solver_benchmark/parallel_hessian_smoke_7649187/`
and `parallel_hessian_smoke_7649255/`: frozen candidate source, build/job logs,
source/library/binary hashes, environment, runtime library list, arguments and
`checks.jsonl`. Full-state inputs were hash-checked before/after. Reproducer:
`scripts/nonconvex_benchmark/parallel_hessian_smoke.sh`; uses the existing frozen
native library on matching quest10 hardware to preserve the Eigen ABI.

### Parallel Hessian solver integration — qualification passed, 2026-09-27

Following the user's approval of solver/restart qualification (not a trajectory
pilot), wire `hessian_threads` into the benchmark worker, local-dihedral adapter,
Newton phase policy, and checkpoint identity. Default one-thread serialization
is preserved. Refuse nonpositive/noninteger/over-allocation requests, incompatible
assembly policies, and a runtime OpenMP team smaller than the declared request.
No production changes; guarded-Hessian and symbolic reuse stay off.

Declared qualification envelope: **one four-CPU, six-minute allocation** (at most
1440 allocated core-seconds); driver stops at 320s with 40s finalization reserve,
refuses to start checks after 255s, worker watchdogs 15s (toy) / 60s (shell), no
automatic retries or extensions. Compile the actual convergence worker dispatch
with an additional qualification test in the same executable and frozen ABI-
matched libraries. Preserve source/runtime/input identities and all failures.

Checks: thread-policy checkpoint mismatch and resource-floor rejection; unchanged
serial defaults; serial/parallel toy output; SIGKILL/recovery at before-target,
after-target, after-commit and progress durable boundaries; completed-cell refusal.
Then compare two full-size CHOLMOD Newton updates from the same **synthetic** ABC
positive-cylinder handoff at one/four threads, and exercise fresh-process disk
recovery after the first four-thread update. Six total full-size updates across
these checks, not convergence fits or a recreation of 2000 live L-BFGS attempts.
Require exact numerical-state agreement after excluding only timing and explicit
thread-policy telemetry; retain all native/Hessian parity checks. Also require
failure rather than silent serial fallback when the requested team is unavailable.
Arbitrary mid-step recovery, other starts, reuse-option composition, sanitizers,
full convergence and end-to-end speedup remain outside this gate.

Implementation/runner: `test/diagnostics/QualifyParallelHessian.cpp`,
`python/qualify_parallel_hessian_solver.py`, and
`scripts/nonconvex_benchmark/parallel_hessian_solver_qualification.sh`.
Prelaunch Python AST, shell syntax and `git diff --check` checks passed.

**Job 7651813 completed; qualification passed.** Scheduler allocation: **172/360s
at four CPUs** (688 allocated core-seconds); driver 155.279s, reported total CPU
147.051s. No retries, six full-size fixture updates, **zero convergence fits**.
All four toy durable-boundary SIGKILL/resume checks passed. Full-size serial and
four-thread numerical states matched exactly after two Newton updates, excluding
only elapsed/timing fields, recipe encoding and thread-count telemetry. A fresh
process resumed the four-thread checkpoint after update one and exactly matched
the uninterrupted final numerical state; the source checkpoint hash was unchanged
and the elapsed floor was preserved. Both shell parity reports gave relative
Hessian error **0.0**, 32267 DOFs, and the requested actual team sizes.

Checkpoint thread-policy changes 4→1/2, invalid counts/assembly, over-allocation,
runtime-policy changes, resource counter resets/exhaustion and completed-cell
reopening were rejected. Constraining OpenMP to one worker while requesting four
failed explicitly (`requested Hessian OpenMP team unavailable`) rather than
silently falling back. Expected negative-test failures remain in worker logs;
they are not failed scientific trajectories. Build emitted pre-existing style/
unused-variable warnings, but no compilation errors. Source/library hashes and
production-file hashes verified unchanged after the gate.

Evidence: `run/nonconvex_solver_benchmark/parallel_solver_gate_7651813/`
(`gate.json`, `status.json`, `protocol.json`, `source_check.txt`, runtime manifest,
29 `process_*/` request/log/result records, shell parity/checkpoints, frozen source).
Qualified worker SHA256:
`b668179d815515ed68ff30bfc5c8fe526f32ea7de8a2c5bcac06841dbc82a393`.
Runtime identity:
`40b722283c17e8041bd1a9c78446e34cfc98a060f3ab0f88d63008cccffd3494`.

The qualified request option is `hessian_threads: 4`; native L-BFGS stays serial,
and the explicit Newton phase/scene uses four threads. The worker requires an
adequate allocation and OpenMP team limit. The driver still does **not** launch
any convergence fits. No trajectory pilot, general convergence/restart proof,
production promotion or whole-solve speedup is implied by this gate.

### Four-thread Hessian trajectory pilot — completed 2026-09-27 (job 7654342)

Submitted **7654342**; completed in **328 seconds (5m28s wall clock, 7m52s CPU)**
out of 45-minute (2700s) allocation at four CPUs on `qnode0020` (Intel Xeon Gold 6230R).
Evidence root: `run/nonconvex_solver_benchmark/parallel_hessian_pilot_7654342/`.
All launch-gate checks passed (synthetic shell parity relative error 0.0, toy durable
resume, budget skips, failure stops, thread mismatch refusal).

Both four-thread hybrid cells completed and **accepted**:
- Final independent native gradient norm **≤ 5e-14** (actual: 1.026e-15 / 2.069e-17).
- Recomputed unshifted restricted LDL factorizations confirmed **positive pivots**.
- Final energies: **2.762881e-11 / 2.763030e-11** (relative difference to baseline:
  -5.38e-05 for (+) start, -8.45e-12 for (−) start).

#### Trajectory & Timing Comparison (Baseline 7629263 vs 4-Thread Candidate 7654342)

| Cell | Start | Status | Charged (s) | Newton Hessians | Hessian Time (s) | Median Hessian (s) |
|---|---|---|---:|---:|---:|---:|
| Baseline | Positive | Accepted | 251.81 | 77 | 125.52 | 1.629 |
| 4-Thread | Positive | Accepted | **141.70** | 75 | **32.12** | **0.420** |
| Baseline | Negative | Accepted | 245.71 | 65 | 106.19 | 1.630 |
| 4-Thread | Negative | Accepted | **132.89** | 66 | **28.14** | **0.420** |

#### Measured Speedup & Efficiency

- **Hessian oracle speedup**: **3.91×** ((+) start) and **3.77×** ((−) start).
  Median Hessian evaluation time dropped from **1.63 s → 0.42 s**, matching the
  preliminary single-state measurement (0.38s).
- **Wall-clock solve speedup**: **1.78×** ((+) start, 251.8s → 141.7s) and
  **1.85×** ((−) start, 245.7s → 132.9s). Total charged wall time for both cells
  dropped from 497.5s → 274.6s (1.81× overall).
- **Core-second accounting**: Because four CPUs were reserved for the entire job
  (including serial L-BFGS, I/O and reporting), allocated core-seconds increased
  from 497.5 core-s (baseline, 1 CPU) to ~1098 core-s (4-thread cells, 4 CPUs),
  a ratio of **~2.21×**. Total measured CPU time reported by Slurm was 471.7s (7m52s).
- **Bottleneck shift**: The Hessian share of charged time dropped from **46.6% → 21.9%**.
  The largest remaining contributor is now **Energy / gradient evaluations**
  (36.4% of charged time, ~100s total across 2000 L-BFGS steps and line searches).

#### Shape & Basin Analysis

- **Negative start**: Reached the exact same numerical equilibrium as baseline.
  RMS aligned vertex distance is **0.000107 mm (0.11 µm)**; maximum distance
  **0.000378 mm (0.38 µm)**. Final energy difference is **-8.45e-12** relative.
- **Positive start**: Reached an accepted local minimum with slightly lower energy
  (2.76288e-11 vs 2.76303e-11, relative difference -5.38e-05). RMS aligned vertex
  distance is **10.32 mm** (max 29.41 mm), reflecting a slightly different stable
  wrinkling/buckling mode on this coarse mesh (the initial 2000 L-BFGS steps on
  different CPU nodes accumulated slight floating-point differences of 0.37 mm
  at the handoff boundary, landing Newton in an adjacent accepted basin).
- Visual review: 3D rendered shapes (`parallel_vs_exact_shapes.png`,
  `profiled_hybrid_final_shapes.png`), timing breakdowns (`parallel_vs_exact_timing.png`),
  and progress curves (`parallel_vs_exact_progress.png`) were inspected and confirmed.
  (A Matplotlib `Axes3D.set_box_aspect` in-place array mutation bug affecting
  multi-subplot aspect ratios was identified, patched with `tuple(span)`, and
  all artifacts re-rendered cleanly).

Evidence root: `run/nonconvex_solver_benchmark/parallel_hessian_pilot_7654342/`.
All phases authorized for this pilot are now complete and closed.

### Existing-backend check — separate 15 CPU-minute authorization

Researcher approved a new **900-second, one-CPU** development budget to check
existing sparse-Cholesky backends and compare one available backend on the same
saved matrices. No installations, optimization fits or tolerance changes.
Discovery found Quest's `suite-sparse/7.7.0-gcc-12.4.0` system module, although the
earlier checked Conda CHOLMOD paths were absent. Its existing CHOLMOD library
resolves to NVHPC BLAS/LAPACK and runtime libraries. The experiment links those
existing absolute paths; it does not load modules or change persistent settings.

Candidate: CPU-only CHOLMOD supernodal LLT with one thread, natural internal
ordering, no postordering, no diagonal bounding (`dbound=0`), and early return on
nonpositive pivots. Both candidate and external-Eigen control receive the same
Eigen-AMD-permuted matrix; current internal-AMD Eigen LLT is the third control.
Eight prerequisite checks cover dense-reference diagonal extraction/solve,
negative, zero, tiny-positive, NaN and infinite pivots, failed-factor recovery,
strict floors and multiple rectangular supernodes. Squared LL diagonals must
exceed the existing strict floor; no pivot regularization is silently accepted.

Same two indexed matrices as ordering job 7393667: initial cylinder-y+ and cached
late cylinder-y− Newton iteration 37. Per matrix order: current Eigen / external
Eigen / CHOLMOD, then reverse, two repeats each. Same zero-first geometric shifts,
at most 12 attempts, solve residual <=1e-8, negative directional derivative,
equal selected shift and relative correction difference <=1e-8 required. Native
checkpoint checksum is checked before each matrix experiment. Ordering,
permutation, symbolic analysis, factorization, solve and validation costs count
in total kernel time. CHOLMOD stored numeric slots include supernode padding;
they must not be mislabeled as exact nonzeros or peak resident memory.

A single isolated 15-minute batch has a whole-process-group 870-second watchdog
and five-second kill grace. Build, unit checks, dependency hashes and profiling
share the budget. No numerical retry or automatic second backend. Production
and benchmark optimizer code remain untouched; only standalone diagnostics are
added. Full-size unshifted near-equilibrium and operational integration gates
are outside this two-matrix check. Build 7403440 stopped at link time (46 allocated
seconds): Conda's sysrooted linker did not resolve already-installed transitive
BLAS/Fortran libraries. No tests or numerical cells ran. Its source/logs remain
under `backend_7403440/`. Build continuation adds explicit `-rpath-link` directories
from the existing library's `ldd` result; no library is installed/replaced.
The continuation requests only 14 minutes, with an 810-second watchdog plus
five-second grace, within the original remaining 854-second budget.

**Completed kernel check: 7403818 on qnode0021**, one pinned CPU. Existing
SuiteSparse 7.7.0 / **CHOLMOD 5.2.1** linked successfully after the linker-path
repair. All **8 prerequisite checks** and **12 paired comparisons** passed.
No optimizer integration or numerical-cell retry was performed.

| Saved matrix | Current Eigen, s | External-AMD Eigen, s | CHOLMOD, s | Current Eigen / CHOLMOD |
|---|---:|---:|---:|---:|
| Initial cylinder-y+ | 4.383 | 4.410 | 0.718 | **6.10×** |
| Late cylinder-y−, iteration 37 | 5.426 | 5.445 | 0.709 | **7.66×** |

Each entry is the mean of the two retained repetitions. The interval is setup
through verification, excluding input parsing, compilation, output formatting
and factor teardown. Controls are paired within this job; absolute times should
not be compared directly with the earlier job on qnode0024. All modes include
common empty-wrapper/context construction. No warm-up samples were discarded.

Successful factorization time fell **3.946 → 0.290s** initially and **4.980 →
0.334s** later. With CHOLMOD, failed-shift work was 0.136 / 0.112s, while external
ordering/permutation together cost 0.201 / 0.181s. Those costs are included above,
not hidden by reporting the successful factor alone. Selected shifts matched
(0.30114487025276365 / 0.0003052707801841114; seven / four attempted shifts).
Maximum relative correction difference was **3.795e-12**; maximum linear residual
across all backends **1.184e-11**, and within CHOLMOD **1.912e-12**, below 1e-8.
All backends produced finite, descending directions under the unchanged gate.

CHOLMOD stored 9.603M / 10.374M numeric slots versus Eigen's 7.844M / 8.408M
coefficients. Supernode padding and compressed index storage differ: these are
not comparable exact-nonzero or peak-memory measurements. Memory was not profiled.

**Decision:** shortlist CHOLMOD for isolated benchmark integration, not production
adoption. This establishes a substantial saved-matrix kernel gain, not a
whole-trajectory speedup or better convergence/basin selection. Two states of
one objective/mesh are not held-out confirmation. Full-size unshifted
near-equilibrium accuracy, integration, dependency-aware checkpoint identities,
operational recovery and a new matched pilot remain separate gates; none were
silently launched with the unused budget. Hessian construction remains outside
this measurement.

Evidence: `run/nonconvex_solver_benchmark/backend_7403818/summary.json`, raw
`profile_0.jsonl` / `profile_1.jsonl`, unit output, staged source, and hashes.
Analysis job **7404257** verified four source files, the binary, two inputs,
24 linked libraries and 399 headers. Native checkpoint checksums were also
verified before profiling. The all-observation figure is
`figures/sparse_backend_kernel_timing.png`; PNG and grayscale were visually
reviewed with no observed clipping/text overlap. PDF export exists but its visual
review is **blocked**: the rendering stage lacked `pdfinfo`; no available local
alternative was found and no installation was attempted. CVD simulation was
unavailable. Numeric analysis and PNG review are complete; PDF review remains
partial. Caption, layout checks and review notes accompany the figure.

Budget: failed build 7403440 **46s**, successful build/profile 7403818 **100s**,
analysis/rendering job 7404257 **37s** conservatively including extern lifecycle
(30s main job; rendering-only exit 2). Total **183s = 3m03s /15 minutes**, actual
CPU **1m50.037s**. The phase is closed; no numerical comparison was rerun.

### Ordering and finer profiling — new 15 CPU-minute authorization

Researcher approved a separate 15 CPU-minute development allocation: finer timers,
fill/ordering inspection and one saved-matrix candidate; no fits, installations or
physical-tolerance changes. V5 instrumentation leaves the optimizer on AMD/LLT.
It splits Hessian-oracle, gauge, reduced-assembly and symmetry time; logs each
shift's numerical time/outcome and successful factor storage/work proxy; and adds
energy/gradient-oracle and L-BFGS direction timings. Oracle time still includes
shell geometry restoration and derivative evaluation, not their internal split.

The only experimental candidate is deterministic geometric nested dissection
using reference-plane coordinates, balanced median cuts, graph-verified one-sided
separators, and AMD within leaves/separators (fixed leaf threshold 256). It is
not wired into the optimizer. Controls: current internally ordered AMD/LLT and
externally permuted AMD with natural-order LLT, to distinguish ordering from
permutation-handling overhead. Per matrix order: current AMD / external AMD / ND,
then the reverse. Timings include ordering and permutation construction.

Predeclared matrices: canonical initial cylinder-y-plus model and the latest
retained cached terminal model from the actual v4 Newton cylinder-y-minus pilot
(`02_cylinder_y_minus_sparse_newton/checkpoint_9.json`). Selection uses indices,
not errors or energy. Same shift grid and pivot gate; residual <=1e-8, equal
selected shifts, relative direction difference <=1e-8 required. This is a frozen
algebra check, not a physical residual relaxation or a claim about convergence.

A single 15-minute, one-CPU job runs an isolated regression/worker build and then
the bounded profile. A whole-process-group 870-second watchdog plus five-second
kill grace leaves room for scheduler accounting; insufficient post-build time
or a gate failure leaves a partial report without retry. New tests cover the ND
permutation/solve, disconnected/degenerate graphs, timer accounting/cache hits
and factor diagnostics. No alternate sparse backend was found at the checked
Conda CHOLMOD/METIS include/library paths; no installation is attempted.
**Completed: 7393667 on qnode0024.** All **33 regression tests passed**, the
instrumented worker compiled, and all 12 predeclared frozen comparisons passed
the residual/equal-shift/direction gates. Maximum relative linear residual was
**1.1841e-11**; maximum correction difference was **2.8505e-12**. Sources, inputs,
dependencies, binaries and raw paired timings are preserved under
`run/nonconvex_solver_benchmark/ordering_7393667/`; `summary.json` records the
aggregate and limits. Both original checkpoint FNV checksums and source/binary/
input SHA256 manifests were independently verified during analysis.

| Frozen state | Current AMD, s | External AMD, s | Geometric ND, s | ND/current time ratio |
|---|---:|---:|---:|---:|
| Initial cylinder-y+ | 2.312 | 2.385 | 4.362 | 1.886 (slower) |
| Cylinder-y−, iteration 37 | 2.793 | 2.953 | 4.078 | 1.460 (slower) |

These are means of two repeats per mode, including ordering, permutation,
symbolic analysis, shifted factors, solves and verification; checkpoint reading
and Hessian construction are excluded. All raw repetitions are retained.

**Decision: do not adopt this ordering candidate; retain AMD.** ND increased
factor nonzeros from **7.844M to 9.561M** initially (+21.9%) and **8.408M to
9.542M** later (+13.5%). Its column-square structural work proxy rose 50.2%
and 25.1%; that proxy is not a hardware-counter FLOP measurement. This rejects
this particular geometric/AMD-block construction, not all nested-dissection
methods. The external-AMD control had identical fill to current AMD and no
observed mean time improvement.

For current AMD, the successful numerical factorization alone cost **2.005s /
2.463s**; all unsuitable shifts together cost only **0.105s / 0.146s** (5.0% /
5.6% of numerical factor time). The late snapshot is from the actual Newton
trajectory, reinforcing that successful factorization—not repeated bad-shift
attempts—is the main remaining numerical-kernel target. A more efficient sparse
factorization backend remains a possible next test; its availability and benefit
are unverified. No further candidate or ordering retune was launched.

The new model-substage and L-BFGS timers are regression-tested, but **no new
full-size trajectory timing breakdown was collected**. No claim is made about
whole-trajectory speed or convergence. Evidence is tabulated; no new figure was
generated. Operational process-recovery preflight was not repeated for v5.

Budget: main job **638s** (639s conservatively including extern lifecycle), failed
reporting-only job **7399127: 20s**, repaired reporting-only job **7399717: 39s**
conservatively including extern. Total **698s = 11m38s / 15 minutes**, CPU
**9m27.637s**. The analysis failure was Python 3.8 lacking `str.removeprefix`;
its source/error were preserved, and only reporting was repaired/repeated.
No numerical comparison or fit was retried. This phase is closed.

### Matched v4 Newton pilot — authorized launch, gates before fits

Researcher requested a small trajectory run following v4 development. New
experiment cells compare v4 sparse Newton and the unchanged native L-BFGS profile
on exactly the original `cylinder_y_plus` and `cylinder_y_minus` starts. Order:
L-BFGS+/Newton+/Newton−/L-BFGS−, on one pinned CPU of qnode0107 using the tested
7380235 binary. This is a development comparison on previously inspected inputs,
not held-out confirmation or an independent-problem sample.

**Ceiling: 30 single-CPU minutes including preparation**, with a nine-minute
preparation alarm, a guarded total deadline and worker cleanup on parent budget
exceptions. Each fit retains the original 300-second allowance (265-second
optimizer deadline; 295-second worker watchdog); the four-cell ledger has a
1200-second global cap. No automatic retries, cap extensions, reopening of old
cells, production-solver changes or physical-tolerance changes.

Before any fits: refresh all-six-start derivative/input checks, all-method
process restart tests, watchdog/identity/completed-cell barriers, and both 24-cell
and four-cell mocked ledger tests. A new fixture checks parent-exception cleanup
of its worker process. Reconstructed objective/start manifests must byte-match
the original bundle. Separately check the archived **cycle-2 near-equilibrium**
Hessian with the same LLT helper: both backends must take the unshifted path,
meet the existing 1e-8 relative linear residual gate, and agree with one another
and the archived physical correction within 1e-5 relative norm. This relaxed
*correction-comparison* tolerance addresses conditioning; it does not change the
physical gradient gate or any optimizer setting. The derived matrix fixture is
explicitly not an optimizer-resume checkpoint. Failure of any gate blocks fits.

The same original-energy/residual/curvature acceptance policy applies. Curves,
checkpoint histories, every cap/failure and endpoint scores are retained. No
figures are generated automatically with the old six-start plotting layout.
Sources: `scripts/newton_matched_pilot.sbatch`, `python/run_newton_pilot.py`,
`python/run_nonconvex_screen.py`, and `python/prepare_near_equilibrium_matrix.py`.
**Completed: 7386943 on qnode0107**, exit 0, allocated 19m45s (18m37.905s CPU).
All four fits reached their time caps; none met acceptance, and all endpoint
curvature checks were indefinite. All refreshed preflight and near-equilibrium
gates passed before fits; preparation took **68.73s**. Near-equilibrium LDLT/LLT both used zero
shift; maximum relative linear residual **2.072e-9**, LLT/LDLT correction
difference **3.008e-10**, and maximum difference from the archived physical
correction **1.574e-9**. These are frozen linear-system checks, not another
near-equilibrium shell optimization. Parent-exception cleanup and four-cell
ledger/no-reopening fixtures also passed. Original objective/start manifests
matched exactly and the input bundle was made read-only.

Live status, gates, source/dependency hashes and outputs:
`run/nonconvex_solver_benchmark/newton_pilot_7386943/`; fit ledger:
`screen/ledger.json`. L-BFGS endpoint energies were **1.3982e-9 / 1.4010e-9**
for the +/− starts; Newton reached **1.7999e-9 / 1.7864e-9**. Residuals were
3.17e-9–3.71e-9 across all four endpoints, still far above 5e-14. These are
unfinished trajectories, not accepted minima. The previous 15-minute development
phase remains closed at 14m28s and is not reset by this separate authorization.

#### Actual trajectory bottlenecks — read-only audit 7389175

`python/audit_newton_pilot_timings.py` inspected all four terminal JSON histories,
without fits or oracle calls. Job 7389175 completed in 19 allocated seconds
(4.328 CPU seconds); evidence is `newton_pilot_7386943/timing_audit.json`.
Pilot plus this audit consumed 20m04s of allocated one-CPU time, below the
30-minute pilot/analysis ceiling.

Across both Newton fits (549.65 charged seconds):

| Recorded phase | Seconds | Share of charged time |
|---|---:|---:|
| Numerical factorization / shift search | 335.84 | 61.10% |
| Model construction | 174.39 | 31.73% |
| Symbolic analysis | 11.51 | 2.09% |
| Triangular solves | 2.96 | 0.54% |
| Not separately attributed | 24.95 | 4.54% |

Model construction includes Hessian evaluation, coordinate-gauge selection,
scaling/reduced sparse assembly and symmetry checks—not just element AD.
Unattributed time includes initialization, evaluations/line search, I/O and
bookkeeping, endpoint verification, and unfinished stages whose timers were not
recorded. For example, the + run's final interrupted model build has no recorded
model timer. The fractions retain all charged time, not only successful steps.
L-BFGS has no corresponding phase timers; its zero recorded phase totals mean
missing instrumentation, not zero cost.

Newton committed **39 / 37 steps**, of which **36 / 35** used full Armijo steps.
All 76 completed directions needed positive shifts; neither fit reached the
unshifted local Newton regime. Recorded shift searches took 4–7 factorizations
(including aborted trials). Median shifts were about 0.0302 in scaled model
units. Native residuals ended near 3.17e-9 and energy still dropped 12.9–13.6%
from the point nearest half-time to termination. Thus remaining difficulty is
both per-step cost and progress per step, not predominantly Armijo backtracking.

Next priorities, not executed changes: split numerical factorization timing by
shift/outcome and model construction by substage; inspect fill/reordering costs;
then compare a faster SPD factorization backend or a well-preconditioned
iterative shifted solve. Failed-shift avoidance remains conditional on those
per-trial timings. In the earlier three frozen LLT profiles (7380355), unsuitable
shifts consumed only **2.16–4.80%** of numerical factorization time; the successful
SPD factorization dominated. Those are not the later Newton trajectory states,
but they caution against assuming warm-shift heuristics offer another major gain.
Hessian assembly optimization/reuse and an L-BFGS-to-Newton
hybrid are plausible alternatives but alter costs/recipes and require testing.
Eliminating the measured factorization time alone would cap overall time speedup
at about 2.57×; halving it would yield about 1.44× under unchanged trajectories.
These are accounting bounds, not performance forecasts. No solver edits, fits,
retries or cap extensions were performed for this audit.

### Early-aborting Cholesky development — new 15 CPU-minute budget

Researcher explicitly approved a **new 15 single-CPU-minute development budget**
for Newton shift-search improvements, regressions and frozen-matrix checks only.
The previous phase remains recorded at 28m16s; this is not permission to restart
or extend scientific fits. Production solvers and physical tolerances stay fixed.

Candidate v4 replaces the per-shift LDLT numerical factorization with Eigen LLT,
which aborts at a nonpositive pivot. It retains one symbolic analysis per base,
the same unshifted-first/geometric shift sequence, strict pivot floor (checked
using squared LLT diagonals), independent linear residual/descent checks and
original-energy Armijo acceptance. No warm-shift heuristic is introduced.
The factorization helper is shared between the engine and frozen profiler.
Near a pivot threshold floating-point LLT/LDLT decisions need not be bit-identical;
regressions and frozen comparisons check actual directions, shifts and residuals.

Predeclared validation: fresh isolated build, existing targeted suite plus
nonpositive/singular/tiny-positive/nonfinite pivot and same-pattern recovery
checks (10-minute allocation cap). Then a four-minute frozen profiling cap on
three error-independently chosen models: the canonical initial cylinder-y-plus
model and the earliest/latest retained cached matrices in that same first TR
cell (`checkpoint_1.json`, `checkpoint_3.json`). Both backends use symbolic reuse;
order is LDLT/LLT then LLT/LDLT per matrix. This diagnoses one trajectory, not
independent problems, and includes no shell-oracle calls or optimizer advancement.
**Completed:** build/regression job **7380235**, exit 0, **31/31 targeted tests**,
10m55s on qnode0107 (10m11.786s CPU, peak RSS 2,860,988 KiB). The worker compiled;
full six-start operational preflight was not rerun. Existing native-trajectory,
small-shell derivative/constraint, original-energy acceptance, caps, rollback,
checkpoint and continuation tests passed alongside the new LLT checks. Current
numerical sources byte-match the isolated tested snapshot.

Frozen profile **7380355** completed, exit 0, 3m33s on qnode0107 (3m22.578s CPU).
Both backends used the same shared factorization helper, one symbolic analysis,
identical matrices/right-hand sides, and one pinned CPU. All 12 predeclared runs
completed; no optimizer was advanced.

| Saved base iteration | LDLT mean seconds (two runs) | LLT mean seconds (two runs) | Kernel ratio |
|---:|---:|---:|---:|
| 0 | 16.533 | 2.545 | 6.50× |
| 11 | 35.798 | 5.661 | 6.32× |
| 35 | 24.282 | 3.793 | 6.40× |

Every search still **attempted seven shifts**, with identical selected shifts
between backends. LDLT completed the unsuitable indefinite factorizations; LLT
aborted them at nonpositive pivots and completed the final positive-definite
factorization. Maximum relative direction difference was **1.303e-15**, and
maximum independently calculated shifted-system relative residual **4.061e-15**.
Timing variation is visible (e.g. iteration-11 LDLT 30.680–40.916s); ratios are
small-sample descriptive kernel results, not confidence intervals or claimed
whole-trajectory speedups. No warm-shift heuristic or tolerance relaxation was
needed for this gain. Unshifted positive-definite and tiny-positive-pivot cases
were tested on synthetic matrices; these three production-size matrices all
required shifts, so they do not establish full-size near-equilibrium parity.

Evidence: `run/nonconvex_solver_benchmark/development_7380235/` and
`run/nonconvex_solver_benchmark/factor_profile_7380355/` (source/binary/input and
profiler dependency hashes, environment, XML, and all per-shift timings).
Shared implementation: `test/diagnostics/ShiftedNewtonFactorization.hpp`.

**Budget:** requested build limit was 10 minutes but Slurm allowed 10m55s without
an explicit extension. The combined allocation was monitored directly during
profiling, with a 220-second profiling stop threshold reserved to protect the
15-minute total; profiling finished normally at 213 seconds, so no signal was
sent. Actual new allocation **655 + 213 = 868 seconds (14m28s)**; actual CPU
13m34.364s. No test or numerical failure was retried. Future strict allocations
should use an in-process watchdog rather than treating Slurm time limits as exact.
The next separately authorized step is a trajectory-level comparison and updated
operational preflight with a new build identity; this work does not establish
convergence or global optimality and does not reopen any completed scientific cell.

### Post-screen maintenance — 29 tests passed; bounded profiling complete

The researcher approved addressing factorization overhead and investigating TR
scaling, not another optimization screen. The v3 development core now reuses one
symbolic analysis across the existing Newton shift sequence at each base state.
Explicit structural diagonal entries protect zero-diagonal/off-diagonal Hessians.
Shift order, acceptance, physical objective, residual gates and budgets are
unchanged; shift warm-starting is deliberately deferred because it changes the
recipe. New timing/count diagnostics separate model construction, symbolic
analysis, numerical factorization and solves. TR adds metric ranges/floor counts,
dual-gradient and step norms without additional HVPs or changed radii.

Added regression checks compare against fresh-factorization shift search,
exercise failed-shift rollback, and verify scaled TR diagnostic units/counts.
A read-only algebra audit of the archived first initial TR model is included;
it neither evaluates the shell oracle nor advances an optimizer. Builds use a
new isolated tree, preserving all screened binaries and checkpoints. Validation
allocation is capped at **nine single-CPU minutes**, within the original
30-minute development allowance even charging the 25-second history audit.
Production code is unchanged. Validation **7363968 completed** in 6m48s on
qnode0008, exit 0, **29/29 tests passed**, including compilation of the screen
worker. The complete six-start operational preflight was not rerun. Sources,
dependency/binary hashes, XML and logs are archived in
`run/nonconvex_solver_benchmark/development_7363968/`.

#### Frozen-matrix timing: a modest improvement, not the main fix

Profile **7364795 completed** in 1m10s on qnode0002, exit 0. It loaded one archived
32,261-variable reduced initial Hessian and gradient, made no shell-oracle calls,
and advanced no optimizer. The two paths were run old/new, then new/old on one
pinned CPU. Each required **seven numerical factorizations**, selecting the same
shift **0.30114487025276365** and exactly matching computed directions.

- Old: **9.9299, 9.8929 s**; seven symbolic analyses per search.
- Reuse: **9.3930, 9.3970 s**; one symbolic analysis per search.
- Mean kernel time fell **5.21%**. This is one frozen-state microbenchmark, not
  a trajectory speedup or a hardware-independent estimate. Most cost remains
  in repeated numerical factorization; symbolic reuse alone cannot resolve that.

Profiler: `test/diagnostics/ProfileFrozenFactorization.cpp`; archived evidence:
`run/nonconvex_solver_benchmark/factor_profile_7364795/`. The pending node
restriction was relaxed to the same `quest10` hardware class; both paths still
ran on the same pinned CPU. Slurm rounded the requested 2m30s to 3m, so the active
allocation cap was **reduced to 2m** to protect the cumulative development budget.
The scheduler override is preserved. No budget was extended.

#### Trust-region finding: higher-order model error, not initial floor clipping

The algebra audit used only the first cylinder-y-plus initial model; it is a
diagnostic case, not a representative sample. No diagonal entries were floored.
The unconstrained first CG step had metric norm **0.010929**, inside radius 0.1;
positions/directors contributed **72.4%/27.6%** of its direction's metric norm
squared. Thus neither floor clipping nor an initially too-small radius explains
this case.

The actual saved first trial then encountered negative curvature on its second
CG iteration and went to the radius-0.1 boundary. The model predicted a decrease
**6.8034e-8** from energy **3.4751e-9**, whereas trial energy was **0.04897**.
The safeguard correctly rejected it. Four rejections reduced the radius **256×**
to **0.000390625**, where actual/predicted decrease was **0.8499** and the step
was accepted.

For the last three of these trials, the base state, Hessian and first CG direction
were identical; only radius changed by 4×. Actual energy minus quadratic-model
energy was **5.7516e-7, 2.2221e-9, 8.6366e-12**, decreasing by **258.84× and
257.29×**—close to fourth-order scaling (4^4=256), not the scaling expected from
a dominant gradient/Hessian error. This supports higher-order geometric terms
limiting the useful quadratic-model step size. Membrane-versus-bending energy
contributions were not separately measured, and later trajectory states were
not checked by this local audit.

Next decision: test a safeguarded way to reduce numerical shift-search work
(e.g. early-aborting positive-definiteness checks or explicit warm-shift recipes)
and a globalization/preconditioning strategy suited to this nonlinear model
error. Such changes need further authorization/budget and held-out confirmation;
no radius, tolerance or scientific fit has been changed or restarted here.

### Completed-screen diagnosis — saved histories, no reruns

An exploratory read-only audit of all 24 terminal histories ran as **7359652**
(25 allocated seconds, 14.174 CPU seconds). Script:
`python/audit_nonconvex_screen.py`; numerical evidence:
`run/nonconvex_solver_benchmark/screen_7296605/history_diagnosis.json`.
The six starts are matched perturbations of one objective, not independent problems.

- L-BFGS accepted 6,898–7,567 updates per fit; nonlinear CG 5,710–6,234.
  L-BFGS residuals ended at 3.08e-9–6.46e-9, versus the 5e-14 target.
  Its energy still decreased 5.9–7.2% between the recorded point nearest half
  the optimization allowance and termination. Nonlinear CG decreased 3.0–3.9%.
  These are unfinished trajectories, not the previous near-roundoff line-search stall.
- Sparse Newton accepted only **8–10 updates** per fit. Every one of its 57
  completed steps used a positive shift (0.0302–0.3021 in the scaled model;
  median 0.3015), so none was an unshifted local Newton step. Code inspection
  shows the shift search restarts at zero and calls a new `ldlt.compute` for
  every trial shift. Repeated symbolic/numerical factorization is an identifiable
  inefficiency; exact component timing, including I/O, was not instrumented.
- TR-CG accepted **103–117 updates** per fit, with 288 rejected model steps
  across the suite. Of 935 completed inner solves, **903 took one iteration**;
  all stopped at a trust boundary (675) or negative curvature (260), never
  interior residual tolerance. Recorded post-update radii had median 4.88e-5
  versus initial 0.1. Thus it paid for fresh Hessians at accepted states while
  usually taking only a one-direction truncated step.
- The screen used the full six-cycle target from small prescribed deformations,
  not the already nearly equilibrated, positive-curvature states used in the
  successful Newton-polishing pilot. The five-minute allowance did not bring
  any recipe near the required residual gate. Tight stopping alone is not the
  explanation: even the best residual is about 61,500 times the target.

Conclusion: progress/cost and nonconvex globalization bottlenecks, not reported
numerical crashes or established trapping in converged local minima. L-BFGS had
lower terminal energy on every paired start, but no accepted minimum was found.
Poor conditioning is plausible; condition numbers and a unique causal mechanism
were not established. Test success validates checked behavior, not competitive
performance. Before another authorized comparison, profile factorization versus
assembly/I/O, investigate Newton shift reuse/symbolic reuse and TR metric/radius
behavior, and calibrate a convergence-capable protocol without changing the
physical residual requirement. No further fits or budget extensions were run.
Figures exist but have not been visually reviewed in this follow-up.

### Submission record — 7296605

One CPU, 16 GiB, at most two hours after allocation starts; queue time is additional.
Input-manifest hashes were rechecked and the complete numerical bundle made
read-only before submission. Numerical outputs will be under
`run/nonconvex_solver_benchmark/screen_7296605/`; job output is
`run/nonconvex_solver_benchmark/screen_7296605.out`. The per-cell ledger is updated
durably; results/plots must not be interpreted as complete until that ledger is
checked. No winner, speedup or convergence outcome is claimed while queued/running.


