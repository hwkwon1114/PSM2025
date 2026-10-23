# Square-sheet BO stage: independent pre-execution reviews (goal mvipfud5-fm48ba)

Reviewer: astra-code-reviewer / openai-codex/gpt-6-astra. Static reviews; no execution permission beyond the stated scope.

## Round 1 - CHANGES_REQUESTED (8bdbaf4f)
# Square-sheet BO pre-execution review

**Verdict: CHANGES_REQUESTED**

The scientific design is suitable for a bounded, model-conditional dome search. Four implementation issues block launch, including a relative-path error that would prevent the submitted solver commands from reading their recipes.

**Report destination:** `/gpfs/projects/p33082/PSM2025-zigzag/docs/reports/square_bo_pre_execution_review.md`

This is the complete report for persistence; no write-capable tool was available.

## Scope, authority and evidence

Authority: `docs/reports/dome_toolpath_investigation.md:384–408`, particularly the frozen square-sheet policy. Earlier rectangular campaigns were treated as history, not authorization or compatible training data.

Reviewed source identities supplied by the requester:

| File | Supplied SHA256 prefix |
|---|---|
| `calibration/dome/square_hatch.py` | `6557ef68` |
| `calibration/dome/square_attempt.py` | `311ffb24` |
| `calibration/dome/square_campaign.py` | `ce4b40ff` |
| `calibration/dome/run_square_campaign.sbatch` | `bf20ece7` |
| `calibration/analysis/dome_continuous_objective.py` | `dda4e771` |
| `scripts/dome_gp_surrogate.py` | `7e2bd725` |
| `test/python/test_square_pipeline.py` | `4f1cc95d` |

Also inspected:

- `calibration/analysis/dome_curvature.py`
- Imported geometry/treatment helpers in `calibration/dome/crosshatch_generator.py`
- `scripts/run_dome_continuation.py`
- Relevant footprint, strip, profile and hardening implementation in `src/libshell/ZigZagGrowth.hpp`, `src/libshell/ZigZagSequenceGrowth.hpp` and `src/simulations/Sim_Bilayer_Growth.cpp`
- Existing objective and GP test sources
- `run/dome/checks/continuation_bo_identity.sha256`

The supplied solver identity is:

`72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a`

The historical continuation identity packet records that same binary hash. **Hashes were not independently recomputed.** No commands, tests, solver runs or scheduler checks were performed. The reported 18 passing tests are supplied evidence; their source was inspected, not their execution independently reproduced.

## Ranked launch-blocking findings

### B1 — Operational: submitted relative paths make the cycle file unreadable

**Locations:** `calibration/dome/run_square_campaign.sbatch:19`; `calibration/dome/square_attempt.py:77–85,98–99`.

The batch script supplies relative root `run/dome_square/campaign`. The attempt runner preserves the relative attempt directory, constructs a relative `-cycle_file` path containing the entire attempt path, then changes the solver working directory to that attempt’s `solve` directory.

Consequently, the solver searches beneath:

`<attempt>/solve/run/dome_square/campaign/.../solve/sequence.json`

rather than the file just written. `src/libshell/ZigZagSequenceGrowth.hpp:327–332` opens the supplied filename directly. With current failure handling, these configuration failures become failed observations and can terminate the campaign after eight attempts.

**Smallest repair:** Resolve `attempt_dir` to an absolute path before constructing `out`, commands or file references. The reused continuation runner already resolves its output directory.

**Clearance:** A lightweight regression using a **relative** attempt directory must establish that the cycle-file argument resolves from the solver’s actual working directory. No new solver run is necessary to demonstrate this repair. Existing real-solver fixtures use absolute temporary paths and miss this defect.

### B2 — Scientific and operational: infrastructure/analysis failures become numerical training labels

**Locations:** `calibration/dome/square_campaign.py:221–237,267–271`; `calibration/dome/square_attempt.py:104–110`.

The controller ignores launcher return codes. Missing status, or a status left at `started`, becomes `state=failed, z=0, y=None`.

Examples include:

- `srun` failure or node loss;
- OOM killing the worker;
- an analysis exception after successful native verification;
- an unexpected worker crash.

The runner also catches `FileNotFoundError`, malformed-artifact `ValueError`/`KeyError`, and every solver `CalledProcessError` as ordinary solve failures.

This violates the frozen immediate infrastructure/analysis stop. It also teaches the feasibility classifier that scheduler or analysis failures identify numerically difficult recipes. Such events wrongly contribute to the two-all-failed-waves stop, and an isolated event permits further waves.

**Smallest repair:**

1. Retain launcher return codes and require a complete terminal worker status.
2. Distinguish recognized numerical nonacceptance/declared solver timeout from infrastructure, configuration, artifact-integrity and analysis errors.
3. Preserve every reserved attempt and available sibling result, but stop before another wave on the latter errors. Do not assign them numerical-feasibility labels.
4. Require finite objective values before accepting scored records.

**Clearance:** Mocked cases covering nonzero launcher exit, absent/started status, analysis exception and genuine numerical failure must demonstrate the distinct behavior. The genuine numerical failure remains `y=None, z=0`; infrastructure/analysis errors stop for the researcher.

### B3 — Scientific/provenance: identity and recipe checks do not enforce the stated freeze

**Locations:** `calibration/dome/square_campaign.py:142–156,225–226,251–271`; `calibration/dome/square_attempt.py:86–101`.

`freeze_identity()` runs only once. Subsequent worker processes can load modified runner/objective code or a replaced binary without the campaign noticing. The worker’s binary comparison detects changes *during that individual solve*, not changes between waves, and collection does not compare its binary hash with the campaign’s frozen hash.

Recipe provenance also has a gap: the worker copies `request['recipe_sha256']` into its status. Collection compares that copied value with the same request field. The written-file check compares against serialization of the current request recipe, not against the reserved canonical digest. Thus a stale digest accompanying an altered recipe can pass these checks.

**Smallest repair:**

- Recheck the frozen identities before each wave and after collection; stop on mismatch.
- Recompute the canonical recipe digest from the actual recipe before execution and compare it with the reservation.
- Validate returned binary identity against the frozen campaign identity.
- Include the batch launcher in the reviewed identity set.

**Clearance:** Lightweight mutation tests must reject a changed recipe with an unchanged digest and a changed execution identity between waves. Continuous filesystem monitoring is unnecessary.

### B4 — Operational: the frozen phase ceilings are missing

**Locations:** `calibration/dome/square_campaign.py:195–205,258–269`; `calibration/dome/square_attempt.py:99,110`; authority at `docs/reports/dome_toolpath_investigation.md:390,402`.

Only the solver subprocess has a timeout. Proposal fitting, analysis and launcher waits are unbounded within the 36-hour allocation. The 4800-second reserve therefore does not bound a wave as the policy assumes; a stuck launcher or analysis can consume the remaining allocation and strand reservations.

There is also a distinction to resolve: the frozen numerics specify **3600 seconds for the solver process**, while compute policy specifies **3900 seconds for the solve step**, plus 600-second analysis and 300-second proposal ceilings. The implementation instead gives the solver itself 3900 seconds.

**Smallest repair:** Implement the already-declared bounded proposal, solve-step and analysis phases, including cleanup of this campaign’s timed-out steps. Preserve the documented 3600-process/3900-step distinction, or explicitly reconcile that discrepancy before launch. Do not silently treat a solver timeout as an entire attempt/step bound.

**Clearance:** Mocked timeout paths must demonstrate bounded waits, retained outcomes and correct failure classification. No production timing experiment is required.

## Answers to the scientific and implementation questions

### 1. Parameterization and simulator treatment

**Scientifically sensible for exploratory screening; not exhaustive for 96 candidates.**

The coordinates affect actual treatment:

- Pitch and span alter strip membership, overlap and strip count.
- Length changes extent, inclination and longitudinal profile placement.
- Repeats change face-shared hardening history.
- End ratio changes top growth, not bottom growth or hit membership.

With the frozen beta, the per-hit factor is \(0.8^h\). Repeated treatment saturates; six repeats are not six times the effective dose. A-first ordering also changes the treatment subsequently available to B. These are coupled layout interventions, not isolated causal effects.

The width-inflated footprint agrees with the inspected simulator placement check:

\[
S_{\rm realised}=(n-2)L_v\tan(\alpha)+8,\qquad
L_{\rm inflated}=L_v+8.
\]

At fixed rotations 0/90 and zero shifts, the Python helper checks the same inflated rectangular envelope as C++. Its \(10^{-9}\) mm tolerance corresponds to the inspected \(10^{-12}\) m simulator tolerance. The final check uses the emitted, rounded angle. **Recipes returned by `decode` satisfy this footprint check; some boundary inputs can legitimately raise.**

Distinct unit points intentionally alias through integer repeats and strip counts. Realized spans can lie outside the *requested* `[96,196]` interval because of rounding, while remaining within the sheet. This is not an invalid recipe. Deduplication uses realized coordinates, addressing these aliases; the six-decimal key can conservatively merge extremely close designs.

The 8 mm versus calibrated 2 mm width and top-only profile extrapolation remain material-transfer limitations, not reasons to forbid this explicitly authorized model search.

### 2. Square ROI, objective and gate

The square call is correct: `roi_half=(0.08,0.08)` selects face centroids in the central square and retains invalid faces in the reference-area denominator. Connected components are restricted to one coherent sign and to the ROI; convex and concave regions are not pooled.

The objective’s mathematical definition is nondecreasing under **uniform scaling of a fixed curvature field and validity mask**: each ramped strength increases, its smoothed quantile increases, and the exact positive/negative Gaussian-curvature flux ratio is scale-free. The evaluator uses exact hinges, not the separate softplus diagnostic helper. The finite-iteration quantile implementation is not a universal numerical proof of this property, but no new ROI-specific mathematical defect was identified.

`j_combined` is used for search; `qualified` comes from the connected gate and causes a screening stop. This correctly separates acquisition from acceptance.

**The gate alone cannot exclude folds or correlated estimator error.** The 15 mm neighborhood can smooth treatment-scale variation at 6–16 mm pitches. However, an ROI-boundary neighborhood reaches at most 95 mm from the center, leaving 5 mm before the physical sheet edge: it is not inherently support-truncated by that edge.

Maps, unexaggerated geometry, scale sensitivity and the separately reviewed finer-mesh verification remain necessary before calling a result a smooth dome.

### 3. Failure learning and acquisition

For correctly classified numerical outcomes, the implementation is sound:

- Failed solves have no objective and are excluded from regression.
- Accepted full-load equilibria receive `z=1`, regardless of dome success.
- The classifier uses all numerical outcomes.
- Regression and acquisition use the same realized recipe coordinates.
- Scaling covariates using all attempted points is not outcome leakage.
- Candidate keys exclude previous attempts and within-batch duplicates.
- EI’s sign and maximization convention are correct.

`EI(J) × P(accepted)` is a reasonable constrained-acquisition approximation, not calibrated physical uncertainty or a joint batch optimum. The single-class Beta fallback avoids assigning zero exploration probability everywhere.

No active scored-indexing defect was found. Seeds are deterministic for fixed software and record order; the implementation counts **BO waves from zero**, which should be made explicit in the policy wording.

Nonfinite CEI is not explicitly rejected before `argmax`; adding a finite-acquisition check is advisable. The principal failure-learning blocker is B2.

### 4. Provenance, stopping and interrupted reservations

Reservation precedes launching the wave. Normal collection retains the entire wave before checking success. The default cap, primary-gate stop, two-all-numerical-failure-waves stop and pre-wave reserve check are present.

Infrastructure stops, changed-identity stops and phase ceilings are incomplete as described above.

`reconcile()` records uncollected reservations as interrupted and stops on the first re-entry. It does not replay them, and their files remain available. **This is fail-stop recovery, not demonstrated resumability.**

Nonblocking under the reviewed single-submission/no-restart contract: the interruption stop is not persistent. A later invocation finds those attempts already in the ledger and can proceed. A durable stop marker or explicit interrupted-record check is the smallest strengthening before any future restart workflow. An accepted status existing before interrupted collection must not be mistaken for a demonstrated numerical failure during manual recovery.

### 5. Launch resources and isolation

The requested allocation and step arithmetic are consistent: 32 CPUs/128 GB permit four exact 8-CPU/32-GB steps. Native Hessian threads are eight, BLAS threads are limited, and the solver receives OpenMP limits. The controller waits for the wave; it does not launch a fifth solver.

The repaired binary path is explicit; the newer `bin/shell` is not selected. Native verification checks full load, terminal proof, positive pivots, gradient tolerance, cycles and continuation history. The 1800-second native ceiling is **per native solve/substep**, not the entire adaptive continuation.

Attempt and solve directories refuse overwrite. No reviewed code targets the unrelated frozen jobs. The resource defects are bounded-wait enforcement and failure handling, not the basic allocation shape.

### 6. Other nonblocking limitations

- `decode_treatment_field()` recognizes `crosshatch_*` IDs when filling directional `along_A/B` and `across_A/B` outputs. With `square_A/B`, those directional arrays remain zero. Hits and top/bottom traces remain useful, but those directional outputs must not be used as verification evidence without repair.
- Summed growth traces are diagnostics, not the complete multiplicatively accumulated target metric.
- The surrogate docstring still describes feasibility as convergence **and** dome success. The square controller correctly uses convergence only; update the misleading documentation.
- The supplied tests cover useful analytic shapes and nominal controller paths, but not the blocking relative-path, launcher-error, analysis-error or between-wave identity cases.
- No new scientific estimand, comparator campaign, recalibration, cadence study or C++ rewrite is needed to clear these findings.

## Claim boundaries and disposition

The target is a **model-conditional free-boundary equilibrium shape**, not physical fabrication accuracy, identification of treatment mechanisms or proof of a global optimum. The anchors are within-stage references; rectangular observations are neither matched comparators nor imported training data.

A negative campaign is informative only about the attempted domain and numerical contract. Failed equilibria say nothing about their unobserved shape objective. A screening pass is not final verification.

**Clear B1–B4 and reassess the changed control paths before launch.** No expensive rerun is required merely to establish these repairs. This review grants no execution permission, additional candidates, restart or verification-stage release.
## Round 2 - CHANGES_REQUESTED, two controller blockers (f90b271d)
# Square-sheet BO pre-execution re-review — round 2

**Verdict: CHANGES_REQUESTED**

**Two small controller fixes remain launch-blocking.** Round-1 B1 and B3 are repaired; B2 and B4 are only partially repaired. The authorized solver switch introduces **no additional identified launch blocker for bounded screening**, subject to the limitations below.

Report destination: `docs/reports/square_bo_pre_execution_review.md`

## Ranked launch-blocking findings

### 1. Operational — proposal fitting remains outside the timeout  
**Round-1 B4 remains partially open.**

**Location:** `calibration/dome/square_campaign.py:206–228`, particularly `sur.fit(...)` at line 219 and `signal.alarm(...)` at line 224.

The 300-second alarm starts **after** fitting both GPs. Consequently, the expensive optimization is unbounded by the declared proposal ceiling. `scripts/dome_gp_surrogate.py:115–127` performs regression and classification optimization with multiple restarts. A slow fit can consume the allocation reserve before another wave is launched.

**Smallest repair:** Arm the existing proposal alarm before proposal preparation/fitting, and cancel it in a `finally` covering the entire phase—not only `select_batch()`.

**Clearance:** A lightweight test should substitute a slow fitting function and establish that the proposal times out before reserving or launching another wave. No solver execution is needed.

### 2. Operational — unsuccessful launcher completion can bypass the infrastructure stop  
**Round-1 B2 remains partially open.**

**Location:** `calibration/dome/square_campaign.py:264–295`.

`collect()` records `launcher_exit` but does not reject a nonzero exit or `'timeout'` when `status.json` already contains `accepted` or `failed`. A worker can publish its terminal status before a launcher/step teardown failure or hang. The controller then treats that attempt as clean numerical completion and can launch subsequent waves despite the frozen infrastructure-stop rule.

**Smallest repair:** Require successful launcher completion before admitting an observation to training. A nonzero exit or launcher timeout must produce `state='error', z=None, y=None`, retaining the worker’s artifacts for diagnosis. Continue collecting available sibling results before stopping.

**Clearance:** Extend the fake-runner tests to cover:
- terminal `accepted` status followed by nonzero exit;
- terminal `failed` status followed by nonzero exit;
- terminal status followed by a launcher timeout.

Each must retain the wave and stop without another launch.

## Round-1 disposition

| Finding | Current disposition |
|---|---|
| **B1: relative paths** | **Resolved.** Campaign and attempt directories are resolved absolutely; `-cycle_file` is absolute. The stub-solver regression actually opens it from the solver working directory. |
| **B2: failure classification** | **Mostly resolved; blocker 2 remains.** Configuration, artifact and analysis failures become errors; recognized numerical failures/timeouts remain unscored feasibility observations. Missing/started status stops the campaign. Accepted objective values are checked for finiteness. |
| **B3: recipe/execution identity** | **Resolved for the declared bounded-check contract.** Canonical request and written-recipe digests are recomputed; returned binary identity is checked; identities are checked before waves and after collection; launcher and solver config are included. |
| **B4: phase ceilings** | **Mostly resolved; blocker 1 remains.** Solver timeout is now 3600 seconds; analysis has a 600-second alarm; launcher waits have a finite outer deadline. GP fitting remains outside the proposal alarm. |

Previously listed caveats were also checked:

- **Interrupted-attempt stop:** repaired. Reconciliation writes an error record with no training labels; subsequent entry stops.
- **Nonfinite acquisition:** repaired by an explicit finite-CEI check.
- **Directional treatment arrays:** repaired at `calibration/dome/crosshatch_generator.py:198`, which now recognizes `square_*` IDs.
- **Feasibility docstring:** repaired; it now describes equilibrium acceptance rather than dome success.
- **BO seed numbering:** explicitly documented in code as BO waves counted from zero. Matching clarification in the protocol would be helpful.
- **Growth traces, mesh/estimator limitations and calibration transfer:** remain interpretation limitations, not defects requiring launch repairs.

## Solver-switch assessment

The supplied researcher decision and amendment at `docs/reports/dome_toolpath_investigation.md:410–412` explicitly supersede the old binary/continuation recipe. Historical solver-audit rejection is not, by itself, a prohibition on this newly authorized, bounded route.

### End-to-end consistency

The inspected implementation is consistent:

- `square_attempt.solver_options()` loads `default_adaptive_cutback.json`, then explicitly overrides tolerance and Hessian threads.
- The configuration requests full load first, maximum 40 Hessians, fixed 2000-attempt warmup, Newton-stall/indefinite-curvature cutback, minimum increment `0.015625`, eight retries and native state export.
- “Adaptive cutback” here means **outer load continuation**; `-native_hybrid_handoff fixed` is consistent with that policy.
- `src/simulations/Sim_Bilayer_Growth.cpp:2780–2868` restores accepted geometry and load after rejection and halves the actual attempted increment.
- `time_cap`, `newton_stagnation`, `indefinite_curvature` and Hessian-cap outcomes cannot themselves become accepted endpoints. Native acceptance still requires `gradient_target`, freshly evaluated finite energy/gradient, the requested gradient threshold and positive restricted pivots.
- `src/libshell/NativeHybridShellSolver.hpp:367–409` reserves one Hessian for independent endpoint checking under the configured total cap.
- Exhausted continuation produces the runner’s recognized `equilibrium corrector failed` marker. Intermediate rejected substeps do not independently label the whole attempt failed.
- `scripts/run_dome_continuation.py:19–59` permits rejected substeps while tracking only accepted load advances. It checks proof-row agreement, accepted-substep qualification, endpoint proof agreement and exact terminal load one.
- Additional convergence-CSV columns are compatible with `csv.DictReader`; they do not shift positional parsing.
- Native state export is enabled and restricted to an accepted terminal endpoint. Canonical state/proof artifacts agree with the verifier’s filenames.
- Production binary and cycle-file paths are absolute.

For this lambda-one-first, power-of-two cutback policy, no near-one floating-point incompatibility with the exact endpoint verifier was identified.

### Time and hardware limitations

The **1800-second native ceiling is cooperative and per subsolve**, not a hard whole-attempt cap. A factorization can overrun it before returning; endpoint checking also has limited additional native allowance. The separate **3600-second solver-process timeout** is therefore important and is present.

Both batch script and runner set `BLIS_NUM_THREADS=1`, alongside OpenBLAS/MKL limits. The runner also bounds OpenMP threads and disables dynamic teams. These address the identified oversubscription route, but do **not** prove the earlier hang’s cause or guarantee runtime.

The 5000-face screening route is reasonably contained by these safeguards. Failure to converge within them remains a valid unscored outcome. I would not require a new comparative benchmark merely to clear the two controller defects.

This assessment does **not** establish solver speedup, universal robustness, or qualification of the 20000-face verification execution.

## Scientific and resource checks

The investigation remains a bounded **model-conditional shape search**, not physical identification:

- The square ROI and connected, coherent-sign gate are preserved.
- Invalid curvature estimates remain in the denominator.
- The search objective is separate from qualification.
- Numerical failures do not enter objective regression.
- No rectangular observations are imported as compatible training data.
- Anchors are reference designs, not independently randomized physical comparators.
- A screening pass requires mesh/scale checks and visual rejection of folds or estimator artifacts before a dome claim.
- A negative result concerns this attempted domain and numerical recipe—not impossibility.

The resource arithmetic is consistent: four exact 8-CPU/32-GB steps fit the 32-CPU/128-GB allocation. The controller waits for the wave rather than launching a fifth solver.

The current outer launcher deadline is **4500 seconds** (`3900 + 600`), encompassing solve and analysis. Documentation should distinguish that combined deadline from the 3600-second solver timeout and the nominal 3900-second solve-step allowance. This wording issue is nonblocking once the missing proposal bound is repaired.

No reviewed path targets unrelated frozen jobs.

## Nonblocking residual risks

1. **Runtime evidence remains limited.** The reported 25 passing pipeline tests include useful coarse real-solver fixtures, not production-mesh runtime qualification. Actual Slurm timeout propagation was not exercised by this review.
2. **Historical benchmark evidence is not reused as qualification.** Inspected benchmark summary entries reference older binaries; current cutback CSVs include incomplete/time-capped histories. They cannot establish performance or acceptance for the frozen campaign binary.
3. **Signals are not universal hard deadlines.** Python alarm delivery can be delayed inside native calls. The worker has an outer launcher timeout; no claim of strict instantaneous phase interruption is made.
4. **Verification remains separate.** The fine-mesh options request `1e-14`, but the shared verifier’s ceiling is hard-coded to `5e-14`. Before verification release, bind its check to the expected stricter tolerance.
5. **Identity checks are bounded, not continuous.** Shared libraries and the complete Python environment are not comprehensively bound. Native state is diagnostic output, not demonstrated restart capability.
6. Width/profile transfer, repeat saturation, coupled family order, local-equilibrium selection and curvature smoothing remain scientific limitations.

## Reviewed identities, authority and evidence

Authority: `docs/reports/dome_toolpath_investigation.md:384–412`, including amendment 1. Earlier plans were treated as historical context.

Supplied SHA256 prefixes for inspected files:

| File | Prefix |
|---|---|
| `calibration/dome/square_hatch.py` | `6557ef68` |
| `calibration/dome/square_attempt.py` | `f8aff744` |
| `calibration/dome/square_campaign.py` | `59bd87cc` |
| `calibration/dome/run_square_campaign.sbatch` | `bf20ece7` |
| `calibration/analysis/dome_continuous_objective.py` | `dda4e771` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac` |
| `scripts/dome_gp_surrogate.py` | `449cbe2e` |
| `scripts/run_dome_continuation.py` | `b723c753` |
| `test/python/test_square_pipeline.py` | `001f8343` |
| `calibration/configs/solver/default_adaptive_cutback.json` | `94252b22` |

Supplied frozen executable:

`run/dome_square/frozen/shell`

SHA256:

`522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841`

Additional static inspection covered the treatment decoder, continuation tests, native cutback/endpoint implementation and relevant native test sources. The round-1 report was read from the supplied subagent artifact.

**No commands, tests, solver runs or scheduler checks were performed. Hashes, binary/source correspondence and read-only permissions were not independently verified.** Reported test passes are supplied evidence; test implementation was inspected.

## Disposition

Repair the two controller paths and reassess their focused regressions. No C++ change, recalibration, new benchmark campaign or additional scientific evaluations are needed to clear these findings.

This review grants no execution permission, candidate extension, restart or verification-stage release.
## Round 3 - APPROVED_WITH_CAVEATS (62a5d607)
# Square-sheet dome BO — round-3 pre-execution review

**Verdict: APPROVED_WITH_CAVEATS**

Goal: `mvipfud5-fm48ba`. **Both round-2 launch blockers are resolved in the inspected code. No new launch blocker was identified in this narrow slice.**

## Ranked findings

### 1. Resolved — proposal timeout covers fitting and selection

`calibration/dome/square_campaign.py:219–228` now arms SIGALRM **before** `make_surrogate()` and `fit()`. The `finally` cancels it on success or exception through batch selection.

`ProposalTimeout` propagates out before `run()` reaches reservation or launch. The inspected surrogate implementation has no exception handler that would swallow it.

`test/python/test_square_pipeline.py:308–332` exercises a sleeping fit with a one-second proposal timeout, checks `ProposalTimeout`, bounded elapsed time and unchanged ledger.

### 2. Resolved — abnormal launcher completion cannot become a training observation

`calibration/dome/square_campaign.py:277–281` converts terminal `accepted`/`failed` results with nonzero launcher exit or `'timeout'` into:

- `state='error'`;
- `failure_type='LauncherNonzeroExit'` or `'LauncherTimeout'`;
- `z=None`, `y=None`.

The worker artifacts remain available. `run()` collects the wave’s sibling results, then the error-first stopping check prevents another wave (`calibration/dome/square_campaign.py:303–338`).

**Clean numerical failure remains valid:** `calibration/dome/square_attempt.py:129–145` records recognized numerical rejection and solver-process timeout as `failed`; `main()` exits **0** for both `accepted` and `failed`, and **1** for `error` (`:179–181`). A solver subprocess’s numerical nonzero exit is therefore not confused with a launcher failure.

**Interrupted path remains intact:** `reconcile()` directly appends `error/interrupted` records and raises; it does **not** call `collect()` (`calibration/dome/square_campaign.py:190–200`). The optional `collect(code=None)` behavior is preserved, while live wave collection supplies actual launcher codes.

`test/python/test_square_pipeline.py:292–306` covers all four combinations: accepted/failed status followed by nonzero exit/timeout, with four retained records and refusal of another wave.

### 3. Optional — strengthen regression assertions

At `test/python/test_square_pipeline.py:292–332`:

- For late-hang cases, explicitly assert that `status.json` contains the expected terminal state. Otherwise, slow startup could exercise the existing missing-status timeout path instead.
- For slow fitting, assert `camp.reserved_dirs() == []`; unchanged ledger alone does not prove absence of reservations.

These are **nonblocking test improvements**. Static control-flow inspection establishes the intended behavior.

## Scope, authority and identities

Authority read: `docs/reports/dome_toolpath_investigation.md:384–412`, including amendment 1.

Round-2 evidence reused:
`/home/pxl1051/.pi/agent/sessions/--gpfs-projects-p33082-PSM2025-zigzag--/subagent-artifacts/f90b271d-3630-43ef-b61f-211a43b89fef_astra-code-reviewer_output.md`.

Supplied SHA256 identities:

| File | Prefix |
|---|---|
| `calibration/dome/square_campaign.py` | `7d34edc5` |
| `test/python/test_square_pipeline.py` | `372a6b6e` |
| `calibration/dome/square_hatch.py` | `6557ef68` |
| `calibration/dome/square_attempt.py` | `f8aff744` |
| `calibration/dome/run_square_campaign.sbatch` | `bf20ece7` |
| `calibration/analysis/dome_continuous_objective.py` | `dda4e771` |
| `calibration/analysis/dome_curvature.py` | `85b0e6ac` |
| `scripts/dome_gp_surrogate.py` | `449cbe2e` |
| `scripts/run_dome_continuation.py` | `b723c753` |
| `calibration/configs/solver/default_adaptive_cutback.json` | `94252b22` |

Frozen executable: `run/dome_square/frozen/shell`, supplied SHA256
`522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841`.

Current controller, tests, worker, surrogate and batch script were inspected. The implementation matches the described repairs. **Hashes and an exact round-to-round byte diff were not independently verified.** Unchanged scientific and solver findings are reused from round 2, not newly requalified.

## Evidence limits and residual risks

- **Static review only:** no commands, tests, solver runs or scheduler checks were performed. The reported 22 tests plus slow-fit test, and earlier real-solver fixtures, remain supplied evidence.
- SIGALRM can be delayed inside native calls; this is not proof of a strict hard deadline or actual Slurm termination behavior.
- Round-2 limitations remain, particularly production-mesh runtime uncertainty and the stricter verification-tolerance check required before verification release.
- The campaign remains a **model-conditional shape search**, not physical identification. Objective improvement is not dome qualification; screening success still requires the frozen verification and visual checks. A negative result does not establish impossibility.

**Approval applies only to the reviewed screening-controller repairs. It grants no execution permission, scope extension, restart, or verification-stage release.**