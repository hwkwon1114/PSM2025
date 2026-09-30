# Independent TinyAD convergence check

## Scope and status

Implemented and unit-tested 2026-09-23: **4 tests passed, covering eight
manufactured shell states**. Diagnostic-only small-system reference, not
connected to production stopping. Existing L-BFGS, acceptance gates, and physical
energy are unchanged. No Newton optimization or production trajectory is launched.

The native analytic gradient is checked against TinyAD, not replaced by a newly
"normalized gradient." Normalization is a separate diagnostic construction.

## Frozen bounded verification plan

One build/test job: one CPU, 16 GB, 20 minutes; no automatic retry or trajectory
rerun. The isolated source snapshot and binary hashes, environment, logs and
GoogleTest XML are retained under `run/hessian_convergence_check/job_<id>/`.
Only the four `HessianConvergenceCheck` tests are built/run in the isolated copy.

Eight manufactured plate states: rectangle 0.127 x 0.1524; Triangle relative-area
parameters 0.5 and 0.25 (NOT production mesh resolutions); each with (0) flat
stress-free, (1) flat anisotropic top metric, (2) curved with nonzero directors and
the same metric, (3) curved/grown with one positional edge constraint, modulus 7
and thickness 0.0012. Other states use modulus 1, thickness 0.0006; Poisson ratio
0.33. Metric perturbation is synthetic in face coordinates, not a real toolpath.
No fitted/optimized or saved production endpoints are represented.

For each state:
- Compare analytic vs TinyAD gradient separately for coordinates and directors:
  max absolute error <= 1e-18 + 1e-8 times native block max magnitude.
- Check assembled-Hessian product vs TinyAD element HVP:
  L2 error <= 1e-18 + 1e-10 times assembled product norm.
- Compare analytic energy and gradient central differences in two deterministic
  directions (coordinate-only/director-only), at BOTH steps 1e-6 and 3e-7.
  Energy-derivative error <= 1e-8 Uref + 1e-5 |g dot v| (the floor allows FD
  truncation near zero); Hessian-product error <= 1e-16 + 1e-5 ||Hv||.
- Evaluate the diagnostic without changing mesh state. Flat stress-free fixtures
  must have resolved positive curvature on the admissible nonrigid subspace;
  grown/curved fixtures may legitimately be indefinite or unresolved.

Additional algebra tests: known quadratic correction/decrement, invariance under
positive energy multiplication and coordinate rescaling, constrained/rigid mode
removal, rejection of indefinite/singular/asymmetric/invalid inputs and size cap.

## Diagnostic definition and limits

For q = S y and an orthonormal admissible nonrigid basis Z in y coordinates:

    A = Z^T S^T H S Z / Uref
    b = Z^T S^T g / Uref
    A z = -b
    p = S Z z
    eta = sqrt(-b^T z)
    predicted energy decrease = Uref * eta^2 / 2

The reference fixtures use coordinate scale L=sqrt(reference area), director
scale 1 rad, Uref=Young * thickness * reference area. These are transparent test
scales, NOT a calibrated universal stopping convention; they may underweight
bending and need subsequent accuracy validation. The reference implementation
uses a dense reduced eigensolve and refuses systems larger than 512 DOFs. It must
not be used by densifying the 32k-DOF production system.

Curvature is unresolved when its minimum eigenvalue is <= 1e-12 times the maximum
absolute eigenvalue, unless sufficiently negative to label indefinite. No shift
or pseudoinverse masks negative or weak curvature. The relative reduced solve
residual must be <= 1e-8. Diagnostics include extreme eigenvalues, decrement,
predicted energy reduction, and coordinate/director correction sizes. There is
no pass/fail convergence threshold for the physical model at this stage.

Positive curvature here is a local restricted Hessian observation; the energy
error estimate is a local quadratic approximation, not a global certificate.
Rigid rotations are projected at the CURRENT geometry; off equilibrium their
Cartesian Hessian action need not vanish. The coordinate-metric gauge is explicit.

## Files

- `test/diagnostics/HessianConvergenceCheck.hpp`: bounded algebra reference.
- `test/testshell/Test_HessianConvergenceCheck.cpp`: derivative/diagnostic fixtures.
- `scripts/hessian_convergence_check.sbatch`: isolated build and bounded tests.

## Execution record

1. **7177897**, qnode0161: isolated configuration succeeded; compilation failed
   after 3m20s because test-local variable `top` shadowed the mesh-layer enum.
   No numerical tests ran. Original log, source version, hashes and exit code
   remain saved.
2. Renamed only the two test-local energy operators. The exact patch and corrected
   source hash are saved as `compile_fix.patch` and `compile_fix.sha256`; the
   original `source_sha256.txt` describes the pre-fix snapshot. No numerical gates
   or test cases changed.
3. **7178275**, qnode0115: completed the build and first numerical execution,
   1m12s, exit 0. Requested continuation was capped at 16 minutes, keeping cumulative
   allocation within the original 20-minute budget. **All four tests passed**
   (GoogleTest XML: 0 failures/errors; test execution 0.11 s). Neither numerical
   cases nor optimization trajectories were rerun. The second node compiled the
   new test TU and reused first-node library objects; native compiler flags and
   both node identities are retained. This is not a performance comparison.

Evidence root: `run/hessian_convergence_check/job_7177897/`.
- `tests.xml`, `tests.log`: complete first numerical execution.
- `summary.json`: all 16 block-parity rows, 32 FD rows and 8 diagnostic rows.
- `compile_fix_and_tests.log`, `test_binary.sha256`: successful build/test record.
- `build_and_tests.log`, `exit_code.txt`, `Test_HessianConvergenceCheck.compile_failure.cpp`:
  retained failed compilation.

Results on 61- and 100-DOF fixtures:
- Maximum analytic/TinyAD block-relative gradient discrepancy on the six loaded
  states: **1.21e-13** (coordinates and directors both checked). At stress-free
  states gradients are roundoff-sized, so only the absolute-plus-relative gate
  is meaningful; do not interpret their relative errors.
- Maximum analytic-gradient finite-difference vs TinyAD Hessian-product relative
  discrepancy across all 32 checks: **7.25e-9**. Both predeclared steps passed.
- Both stress-free states had resolved positive restricted curvature; normalized
  decrements were 1.03e-16 and 1.63e-16.
- All six synthetic loaded states were **indefinite**. The diagnostic correctly
  withheld a Newton decrement/correction rather than hiding negative curvature.
  These states were never optimized: this says nothing about the stability of
  any existing L-BFGS endpoint.
- Algebra tests passed for known corrections, energy/coordinate scaling,
  constraint/rigid-mode handling, and refusal of invalid or unresolved inputs.

Shell syntax, `git diff --check`, and byte equality of both current test files
with the successfully compiled snapshot were checked. Build warnings include
third-party Triangle/Eigen code and a nonfatal GoogleTest dangling-else warning;
no compiler errors remained. This is derivative/algebra verification, not
production endpoint convergence validation or a calibrated stopping criterion.

## Next barrier

After these tests: extend derivative checks to full-precision frozen loaded
states, then design the sparse production diagnostic and calibrate its scales.
Archived float32 VTP surfaces alone do not contain the full director/target state
required for this. A Newton solver comparison remains a later bounded task.

## Phase 2: sparse diagnostic and frozen production states (approved)

### Frozen scope and budget

One bounded job: **1 CPU, 16 GB, 30 minutes total build/verification cap**. No
optimization, continuation trajectory, failed-case replay, or production restart.
Do not automatically rerun numerical failures or extend the cap. Compile-only
corrections, if necessary before numerical execution, must preserve failed
artifacts and stay within the original cumulative budget.

Two preselected saved states from the earlier stall probe: `captured_x.f64` and
`energy_corrected_final_x.f64` (both include all directors). These are the stalled
and accepted cold-memory-polished states at the SAME cycle-2 final target, not
final endpoints of the six-cycle experiment. Selection is diagnostic, not a
representative solver comparison. No ordinary-secant comparator is rerun.

The original float64 target tensors were not archived. Reconstruct the regular
mesh and first two material-coordinate growth updates using the same library
headers as the original snapshot. The saved accepted prefix (cycle 1 at .5/1,
cycle 2 at .5), fixed adaptive settings and first-stall control flow imply that
capture occurred at cycle 2 lambda=1. This interpretation is NOT sufficient by
itself: the recreated state must reproduce the saved energy AND full gradient.
No VTP geometry, float32 target fields, or fitted target metrics are used.

Preflight checks archived command, recipe SHA256, prefix ledger, state sizes and
capture metadata. Stop before either Hessian unless BOTH states satisfy:
- native saved-energy relative discrepancy <= 1e-10;
- native-vs-saved full-gradient L2 discrepancy <= 1e-18 + 1e-5 ||g_saved||;
- native/TinyAD coordinate AND director block max discrepancy <=
  1e-18 + 1e-7 times native block max. Near stationarity the absolute floor
  matters; do not report near-zero relative cancellation as a derivative bug.

For each verified state assemble TinyAD H **once**, save CSC arrays, and check its
products against native-gradient central differences at 1e-6 and 3e-7 in two
fixed directions. Coordinate direction is smooth in reference coordinates;
director direction is deterministic sinusoidal. Same FD gates as phase 1:
HVP L2 error <= 1e-16 + 1e-5 ||Hv||, energy-derivative error <= 1e-8 Uref +
1e-5 |g dot v|. Both steps/directions must pass. No step is applied to the state.
Preserve reconstructed rest geometry, connectivity/edge ordering, constraint
mask, material parameters and target tensors with explicit float64 layouts.
This is an energy-state archive, NOT a tested optimizer-resume checkpoint.

### Sparse formulation: explicit gauge change

`SparseHessianConvergenceCheck.hpp` uses coordinate-gauge elimination to preserve
sparsity rather than forming a dense orthogonal complement. Pivoted QR of the
scaled rigid-mode basis selects exactly as many coordinate conditions as there
are admissible rigid modes. Full rank is required: these conditions remove pose
locally, not additional deformation freedoms. Existing physical constraints are
retained. The selected coordinate indices are reported.

On the remaining coordinates, factor S^T H S / Uref using sparse LDLT, without a
shift or pseudoinverse. A sufficiently negative pivot reports `indefinite`;
a failed factorization or a pivot <= 1e-12 max|diag(A)| is unresolved. With
resolved positive pivots, require relative linear residual <= 1e-8, then report
decrement, predicted energy reduction and pose-removed position/director
correction sizes. **LDL pivots are not Hessian eigenvalues.** Positive pivots are
numerical local curvature evidence in the specified gauge, not global or
convergence certification. Away from stationarity this coordinate gauge can
differ from phase 1's orthogonal gauge; no equivalence is silently assumed.

Unit barriers before frozen-state execution: agreement with the dense reference
in the SAME coordinate gauge; agreement with the orthogonal gauge only on an
exactly rigid-invariant synthetic quadratic; energy/scale invariance, physical
constraints, rigid-mode rank, and indefinite/singular-input rejection. Existing
small derivative/algebra fixtures are regression tests, not new optimization.

Reference scales remain explicit, provisional: L=sqrt(actual reference mesh
area), director scale 1 rad, Uref=Young*h*area. No physical stopping threshold is
calibrated or changed. All old L-BFGS acceptance gates remain authoritative.

### Files and execution status

- `test/diagnostics/SparseHessianConvergenceCheck.hpp`
- `test/testshell/Test_SparseHessianConvergenceCheck.cpp`
- `python/validate_frozen_hessian_inputs.py`
- `scripts/frozen_hessian_check.sbatch`

**Job 7180549 completed on qnode0109, exit 0, 6m25s total, peak RSS approximately
2.73 GiB.** No compilation fixes, numerical retries, optimizations or trajectory
reruns were needed. Six unit/regression tests and the production test covering
both frozen states passed. Archived library/TinyAD source comparison and input
preflight passed. Python syntax, shell syntax and `git diff --check` passed;
at phase-2 completion, the then-current sparse header/test source matched the
compiled snapshot byte for byte (subsequent pilot tests extend that source).

Evidence root: `run/hessian_convergence_check/frozen_7180549/`.
- `input_preflight.json`, `input/`, `source_sha256.txt`, `test_binary.sha256`:
  input, archived-recipe and generating-code provenance.
- `unit/tests.xml`, `production/tests.xml`, associated logs: actual test outcomes.
- `production/*_restoration.json`: independent native saved-state checks and
  TinyAD parity, separately for each variable block.
- `production/*_diagnostic.json`: full FD checks, gauge indices, pivots,
  solve residuals, estimates and timings.
- `production/frozen_state_manifest.json`, reference geometry, topology, full
  target tensors, mask, and original `input/*_x.f64`: verified full-precision
  energy-state inputs. Optimizer history/resumption is NOT implemented here.
- `production/*_H_{values,inner,outer}.*`: cached full sparse Hessians (CSC),
  each 32,267 x 32,267 with 1,739,471 stored entries.
- `production/*_correction.f64`, `*_pose_removed_correction.f64`: estimated
  corrections, saved but NEVER applied.
- `production/artifacts.sha256`, `summary.json`: artifact integrity and aggregate
  results; raw records remain authoritative.

### Production-state results

Both full native gradient vectors reproduced their saved vectors **exactly**.
Relative energy discrepancies were 2.54e-16 and 3.81e-16 (rounding differences).
Native/TinyAD full-gradient L2 differences were 2.63e-20 and 2.66e-20; maximum
coordinate-component discrepancies were <=1.10e-21, director discrepancies
<=1.06e-25. Near-zero relative errors should not be compared directly with the
large-gradient manufactured fixtures. All eight production FD checks passed;
maximum relative Hessian-product discrepancy was **2.51e-8**.

| Diagnostic | Captured stall | Earlier accepted local polish |
|---|---:|---:|
| Native gradient norm | 1.237085e-13 | 4.213200e-14 |
| Sparse curvature status | positive LDL pivots | positive LDL pivots |
| Minimum LDL pivot (NOT eigenvalue) | 7.672334e-7 | 7.672334e-7 |
| Relative reduced linear residual | 2.06e-9 | 5.69e-9 |
| Normalized Newton decrement | 6.855578e-8 | 6.855426e-8 |
| Predicted energy decrease (model units) | 1.091586e-19 | 1.091538e-19 |
| Predicted decrease / current energy | 2.674590e-10 | 2.674471e-10 |
| Pose-removed predicted vertex RMS correction | 1.525907 micrometres | 1.525907 micrometres |
| Predicted director RMS correction | 1.659334e-7 rad | 1.659334e-7 rad |
| One Hessian assembly, 1 CPU | 2.11 s | 2.09 s |
| Sparse diagnostic, including gauge/factor/solve | 3.51 s | 3.51 s |

Six rigid modes were removed in each case. Gauge pivot selection differed in
one coordinate between these nearby states; both selected sets and raw as well
as pose-removed corrections are preserved. The close physical correction
estimates are observations, not a general gauge-invariance proof away from
stationarity. Both state vectors remained unchanged.

**Interpretation:** the analytic derivatives are verified on these real loaded
states, and assembling once plus a sparse solve is feasible at this mesh size.
The previous polish reduced the raw gradient norm by about 2.94x, while the
stiffness-aware remaining-correction estimate changed very little. Both should
be logged. Positive LDL pivots are local numerical evidence, not an independent
full eigenspectrum or global-optimality certificate. The 1.53-micrometre value
is a quadratic-model prediction, NOT a measured true solution error. No trial
Newton step or energy decrease has been tested yet, and no universal stopping
threshold is established.

### Next decision

The derivative and sparse-diagnostic barriers passed for these two states. A
bounded, matched-start Newton-polishing comparison is now a reasonable NEXT
experiment, but was not part of this execution. Before broader trajectories:
verify actual versus predicted step/energy changes, preserve original physical
acceptance gates, and separately validate a resumable solver checkpoint.

## Phase 3: bounded Newton-polishing pilot (approved)

### Frozen protocol

Two starts only: the same captured stall and earlier accepted local polish from
phase 2, same frozen cycle-2 lambda=1 energy/targets/constraints. No L-BFGS rerun,
no production-trajectory replay, no global search, no stiffness regularization.
This tests whether actual Newton steps realize the prior quadratic predictions;
it is NOT a matched wall-time solver benchmark or a claim of best performance.

One CPU, 16 GB, **30-minute total build/execution budget**. At most **3 Newton
steps per state**, each with at most **12** original-energy/gradient trial
evaluations (alpha = 1, 1/2, ..., 1/2048). No failed/capped case retries, recipe
changes or budget extensions. Compile-only correction before numerical execution
may use the original remaining budget with preserved failure artifacts.

Restore exact archived geometry/topology and float64 target tensors, and verify
saved energy/full gradients again. Archive source/input hashes and compare
library/TinyAD sources with phase 2. Every direction uses the native analytic
gradient (also cross-checked against TinyAD), freshly assembled/cached TinyAD H,
and the same sparse coordinate gauge, pivot and linear-residual safeguards.

A direction must have resolved positive pivots and g dot p < 0. Armijo uses the
original energy, coefficient 1e-4, and additionally requires strict energy
decrease to reject rounded plateaus. Nonfinite or degenerate-triangle oracles
(minimum current/reference area ratio <=1e-8) stop the case, without automatic
retry. This guard is numerical admissibility, not a self-contact certificate.
Unresolved curvature, invalid directions and exhausted line searches stop.

Internal native-gradient target remains **5e-14**; physical gate remains **1e-13**.
At least one diagnostic step is deliberately attempted even for the already
accepted starting state, to test the Hessian prediction. This is an explicit
pilot-only exception to immediate stopping, NOT a new production stopping rule.
After that, stop on the internal target or cap. Recompute final native energy
and gradient independently. A passed residual gate is reported separately from
termination status; GoogleTest success alone does not mean Newton converged.

Save every trial position/director vector and gradient, every accepted iterate,
every direction, alpha, energy/gradient curves, predicted/actual decrease and
linear-solve diagnostics. The selected/final checkpoint is the last strictly
energy-decreasing accepted state; no later overwrite of earlier checkpoints.
These short pilot archives are not yet a tested resumable optimizer checkpoint.
Before long or broader comparisons, resumability and a matched baseline protocol
remain separate requirements.

Line-search unit barriers cover a full Newton step, successful backtracking,
non-descent rejection, 12-trial plateau exhaustion with unchanged returned state,
and immediate nonfinite-oracle termination. Seven unit/regression tests precede
the two-state pilot. No numerical pilot outcome has yet been observed.

Files: `test/diagnostics/NewtonPolishProbe.hpp`, appended pilot tests in
`test/testshell/Test_SparseHessianConvergenceCheck.cpp`,
`scripts/newton_polish_probe.sbatch`. The frozen-diagnostic build script also
copies the newly required header; its original archived execution is preserved.

### Phase-3 execution record

**Job 7243315 completed 2026-09-24 on qnode0011, exit 0, 7m30s total**, peak RSS
approximately 2.78 GiB. Prior artifact hashes and library-source parity passed.
Seven unit/regression tests passed, followed by the production pilot test
covering both starts. No build corrections, numerical retries or cap extensions
were needed. The pilot's complete two-state test took 9.41 s excluding build.
Current pilot header/test sources match the compiled snapshot byte for byte.

Evidence: `run/hessian_convergence_check/newton_7243315/`.
- `archive/`: copied full-precision inputs and previous diagnostic records;
  original archives are unchanged.
- `source_and_input_sha256.txt`, `test_binary.sha256`, `environment.txt`:
  exact source/input/runtime provenance.
- `unit/tests.xml`, `polish/tests.xml`: seven unit checks and pilot execution.
- `polish/polish_summary.json`, `*_polish.json`: all outcomes, trials and curves.
- `polish/*_step_*`, `*_direction_*`, `*_trial_*`, `*_final_*`: every initial,
  accepted and final state/gradient, every direction and trial state/gradient.
  Checkpoint selection is the last accepted strictly decreasing state.
- `polish/artifacts.sha256`: archived pilot-output hashes.

### Phase-3 results

Both cases accepted **one full Newton step (alpha=1)**, with one trial and no
backtracking. Both then stopped at the predeclared internal native-gradient
target; neither consumed the three-step cap. Independent final energy/gradient
recomputation passed the internal target and unchanged physical residual gate.

| Quantity | Captured stall start | Earlier accepted polish start |
|---|---:|---:|
| Initial native gradient norm | 1.237085e-13 | 4.213200e-14 |
| Recomputed final native gradient norm | 4.155926e-14 | 4.155925e-14 |
| Predicted energy decrease | 1.091586e-19 | 1.091538e-19 |
| Actual original-energy decrease | 1.091439e-19 | 1.091395e-19 |
| Actual / predicted decrease | 0.99986496 | 0.99986913 |
| Relative decrease from starting energy | 2.674229e-10 | 2.674121e-10 |
| Rigid-mode-projected vertex RMS step | 1.525907 micrometres | 1.525907 micrometres |
| Director RMS step | 1.659334e-7 rad | 1.659334e-7 rad |
| Assembly + sparse direction | 4.54 s | 4.32 s |
| Per-state pilot time | 4.75 s | 4.52 s |

Per-state times include state loading/checking, derivatives, assembly, direction,
line search and state-output writes, but exclude shared mesh setup and build.
These timings use one CPU on qnode0011, not a matched solver benchmark. The
vertex measure removes infinitesimal rigid components at the base geometry;
it is a step-size diagnostic, not finite-pose registration or true solution error.

**Finding:** the original-energy decrease realized **99.986%** of the quadratic
forecast, supporting the local Hessian/correction calculation on this frozen
loaded problem. A full Newton step is feasible and lowers energy while meeting
the existing gate. The absolute/relative energy improvement is tiny: about
1.09e-19 model units, or 2.67e-10 of the starting energy. The earlier accepted
start was deliberately stepped once; its native gradient norm improved only
about 1.4%, while the captured stall's norm improved about 2.98x.

**Limits:** these two starts are closely related states from one partial-load
problem, not independent cases, final six-cycle endpoints or global-search
results. No post-step Hessian/decrement was measured: stopping used the original
native-gradient rule. This does not establish a universal normalized threshold,
superiority to L-BFGS, full-trajectory speed, or robustness far from a minimum.
The production solver/acceptance rules remain unchanged. Broader comparisons
still require a bounded matched-start protocol and tested resumability.

### Phase-3 figure

`python/plot_newton_polish.py` plots both predeclared starts and all accepted
steps, without smoothing or inferential intervals. Raw trials remain available
in JSON; zero trials were rejected here. CSV and caption accompany the figure.
Delivered figure: `run/hessian_convergence_check/newton_7243315/figures/newton_polish_convergence.png`
(and PDF, caption, CSV, provenance, exact generating script/style in the same
directory). Render job 7243950 completed. PNG inspected: no clipping/overlap,
both starts/thresholds visible, distinct markers/line styles. The right panel's
fractional tick clutter is cosmetic; its +/-0.04-step display offsets are
explicitly disclosed. An integer-tick refinement is prepared in the current
plot script, but render job 7244090 failed after its interactive client timed
out while queued; no revised image was produced and no automatic resubmission
was made. The original reviewed figure and its exact script snapshot remain
the deliverables. PDF rendering and CVD simulations are not verified.
