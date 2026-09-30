# Numerical Review of the Sequence Ablations

## Verdict

**REQUEST CHANGES.** The experiments are useful pilot runs, but they do not yet support the report's claims about physical path dependence, numerical noise, optimization speed, or equivalence of update schemes. Two reported formulas/units are wrong, timing was confounded, the noise floor was never measured, and raw inputs/results were stored only under `/tmp` and are no longer available.

## Scope reviewed

- `reports/sequence_ablation_report.md`
- `src/libshell/GrowthHelper.hpp`
- `src/simulations/Sim_Bilayer_Growth.cpp`
- execution records for the warm/cold, metric-update, and segmentation runs

The original run directories and generated JSON files referenced in the report were under `/tmp`. They are no longer present, so the CSV/VTP/log evidence cannot now be independently reprocessed.

## Findings

### CRITICAL 1 — Reported curvature values are scaled by 1000 incorrectly

The analysis fit used `z`, `material_u`, and `material_v` in metres. In

\[
z=c+k_u u+\tfrac12\kappa_{uu}u^2+\kappa_{uv}uv+\tfrac12\kappa_{vv}v^2,
\]

the fitted quadratic coefficients already have units of `m^-1`. The analysis command multiplied those eigenvalues by `1e3` before labelling them `m^-1`.

Consequently, the values in `reports/sequence_ablation_report.md:83-86,152-155,203-204` are 1000 times too large. Examples:

| Reported | Dimensionally corrected |
|---|---:|
| warm `[-209.6, -57.8] m^-1` | `[-0.2096, -0.0578] m^-1` |
| cold `[-218.6, -54.2] m^-1` | `[-0.2186, -0.0542] m^-1` |
| segmented path-major `[-43.46, -37.64] m^-1` | `[-0.04346, -0.03764] m^-1` |
| segmented stroke-major `[-42.13, -37.23] m^-1` | `[-0.04213, -0.03723] m^-1` |

Any Gaussian curvature computed from those scaled principal curvatures is too large by `1e6`.

**Required correction:** remove the `1e3` curvature multiplier, regenerate every curvature table/figure, and add a unit test using a known paraboloid.

### CRITICAL 2 — The report analyzes a different additive recurrence than the code implements

`src/libshell/GrowthHelper.hpp:801-810` implements a recursive linearized update:

\[
\bar a_{n+1}=\bar a_n+\delta T^T\bar a_n+\bar a_n\delta T.
\]

For aligned isotropic increments `T=(1+e)I`, this gives

\[
\bar a_n^{\mathrm{recursive}}=(1+2e)^n\bar a_0,
\]

not

\[
(1+2ne)\bar a_0,
\]

as stated in `reports/sequence_ablation_report.md:126-134`.

For `e=2e-5`, `n=12`:

- multiplicative: `(1+e)^(2n) = 1.000480110416`
- implemented recursive linearized: `(1+2e)^n = 1.000480105614`
- additive relative to the original metric: `1+2ne = 1.000480000000`

The multiplicative/implemented difference is approximately `4.8e-9`; the multiplicative/original-additive difference is approximately `1.10e-7`. The report used the latter theoretical difference while the numerical experiment used the former update. Therefore the experiment did not test the additive model described in the report or necessarily the additive interpretation posed by the user.

**Required correction:** define three distinct schemes explicitly—multiplicative, recursive Euler-linearized, and reference-additive—and compare their target metrics before invoking the optimizer.

### HIGH 1 — Runtime comparisons are invalid because cases ran concurrently

The warm and cold runs were launched concurrently; the additive run was launched while both were active. The three segmentation schedules were also launched concurrently. All cases competed for CPU, memory bandwidth, and filesystem I/O. `OMP_NUM_THREADS` was not fixed in the run commands.

The 12-path warm run took `216.61 s`, while an earlier nominally comparable 12-path run completed in `83.28 s`. That spread is larger than the claimed warm/cold effect and is consistent with uncontrolled resource contention and/or binary/config differences.

Therefore these statements are unsupported:

- `reports/sequence_ablation_report.md:76`: warm starting reduced runtime by 19.8%.
- `reports/sequence_ablation_report.md:212`: grouping reduced runtime by 9.9%.
- `reports/sequence_ablation_report.md:218-220`: summary speedup claims.

**Required correction:** run cases serially on the same reserved node, fix thread count and affinity, randomize case order, repeat at least five times, and report median plus dispersion. Record optimizer iterations and function/gradient evaluations separately from total wall time.

### HIGH 2 — No repeated identical control establishes the numerical noise floor

Each condition was run once. There is no repeated run of the exact same binary/configuration/threading/seed combination. Thus the observed `3.72–20.3 micrometre` RMS differences cannot be classified relative to solver nondeterminism, parallel reduction order, meshing variability, or stopping sensitivity.

`reports/sequence_ablation_report.md:223` says the effects are not pure floating-point noise, but the experiments do not establish that. They establish differences between distinct algorithms/schedules, not their significance relative to a measured numerical repeatability floor.

**Required correction:** run at least five identical controls first. Define a practical equivalence threshold from aligned geometry, energy, gradient, and target-metric variability.

### HIGH 3 — Geometry comparisons remove translation but not rotation

`reports/sequence_ablation_report.md:78-81` compares point sets after centroid removal only. The shell has rigid modes; two physically identical states can differ by rigid rotation. Centring does not remove this component and can inflate RMS and maximum pointwise differences.

**Required correction:** use a proper Kabsch/Procrustes alignment with determinant `+1`, and separately test mirror-equivalent branches if the loading is symmetric. Report aligned RMS, maximum error, and an area-weighted surface distance.

### HIGH 4 — No mesh, tolerance, or branch-convergence study supports micrometre conclusions

The geometry constructor uses `edgeLength = 2*Lx*res` (`src/simulations/Sim_Bilayer_Growth.cpp:3639-3647`), giving a nominal edge scale near `7.62 mm` for `Lx=0.127 m`, `res=0.03`. The claimed differences are micrometre-scale, roughly three orders of magnitude below the element scale.

A tight HLBFGS gradient tolerance does not establish spatial discretization convergence. Neither warm/cold final state was exact-Hessian certified, as acknowledged in `reports/sequence_ablation_report.md:94`.

**Required correction:** repeat on at least three meshes and at `1e-10`, `1e-12`, and `1e-14` tolerances; record final gradient and Hessian minimum eigenvalue. Only call a difference physical if it persists under refinement and exceeds the repeated-run noise floor.

### HIGH 5 — The segmentation experiment does not implement the requested segmentation

The requested experiment was a five-zigzag path split into ten single-line or half-line segments. The experiment instead used the existing Putong builder's ten strips and activated one strip per operation. The report acknowledges this at `reports/sequence_ablation_report.md:189`.

This is a related pilot, not the requested test. It also does not test reverse or randomized stroke orders.

**Required correction:** represent explicit segment endpoints and run the exact requested ten-segment path. Compare forward, reverse, stroke-major, path-major, and randomized permutations.

### HIGH 6 — Grouping did not reduce computation relative to the actual whole-path baseline

The grouped stroke-major case required 10 minimizations and `108.01 s`; the existing whole-path schedule required 3 minimizations and `55.88 s` (`reports/sequence_ablation_report.md:193-197`). Grouping only improved over an artificial 30-minimization segmented baseline. It was approximately 93% slower than the whole-path baseline.

The conclusion that grouping decreases iteration cost is therefore context-dependent and should not be presented as an acceleration of the existing whole-path sequence.

**Required correction:** state the baseline explicitly. If segmentation is physically required, compare grouped versus fully relaxed segmentation. If it is not required, the whole-path implementation remains faster in this experiment.

### HIGH 7 — Load equivalence and commutativity were not verified

Before comparing optimizer outputs, the final top/bottom target metrics should have been compared face by face. That was done for multiplicative versus additive cases, but not for whole-path, path-major, and stroke-major schedules.

Moreover, the segmentation test used `ortho=0`. Each active growth tensor is isotropic, so metric increments commute trivially. This does not answer whether differently oriented anisotropic strokes commute.

**Required correction:** first prove equal final `abar_top`/`abar_bot` and hit histories for schedules intended to be equivalent. Then include an anisotropic (`ortho != 0`) rotated-path case where noncommutativity can be measured directly.

### HIGH 8 — Per-cycle convergence is not recorded or enforced

The sequence loop invokes `minimizeEnergy`/`minimizeEnergyReduced`, then unconditionally serializes the geometry and reads energy from the operator cache (`src/simulations/Sim_Bilayer_Growth.cpp:1782-1812`). Unlike another solver path in the same file, it does not record `lastMinimization.code`, iteration count, gradient norm, or a convergence flag. A line-search failure or `max_iter` termination therefore follows the same reporting path as a converged equilibrium.

**Required correction:** persist termination code, iterations, final gradient, and convergence for every actual solve; recompute energy at the serialized state; fail or explicitly mark non-converged states.

### MEDIUM 1 — The explanation of path dependence is mechanically inaccurate

`reports/sequence_ablation_report.md:210` says later growth increments are applied on the geometry changed by prior minimization. In this implementation, hit selection and growth directions are defined in persistent material coordinates (`src/simulations/Sim_Bilayer_Growth.cpp:1599-1611`), and target metrics are updated directly (`:1688-1743`). The current geometry is the next optimizer initial guess; it does not modify the material-frame increment in these flat-panel runs.

For commuting isotropic increments, schedule differences arise from intermediate relaxation, local-minimizer basin selection, and stopping behavior—not from recomputing growth on the deformed geometry.

### MEDIUM 2 — Grouped-cycle output does not identify equilibrium states robustly

`-sequence_minimize_every N` can write cycle VTP/CSV rows for increments where no minimization occurred. The current code now writes `NaN` energy for skipped cycles (`src/simulations/Sim_Bilayer_Growth.cpp:1779-1812`), but the CSV schema has no `minimized_this_cycle`, iteration-count, gradient-norm, or convergence column (`:1152-1167`). The reported grouping runs were executed before the stale-energy reporting was changed to `NaN`.

**Required correction:** add explicit equilibrium/convergence metadata to every cycle row and exclude unrelaxed states from equilibrium analyses.

### MEDIUM 3 — Cold-start mapping diagnostics show the wrong starting geometry

The mapping diagnostic is written at the start of the repeat (`src/simulations/Sim_Bilayer_Growth.cpp:1657-1666`), but the cold reset occurs only immediately before minimization (`:1775-1778`). From cycle two onward, the cold-run mapping file therefore shows the preceding converged geometry rather than the flat/rest geometry actually passed to HLBFGS.

**Required correction:** move the cold reset to the beginning of each repeat, before mapping diagnostics and any initial-state output.

### MEDIUM 4 — Global quadratic curvature fits lack fit diagnostics

The report uses a global unweighted quadratic fit without reporting residuals, `R^2`, spatial weighting, excluded boundary regions, or comparison with the VTP local `gauss` and `mean` fields. A single quadratic can hide localized cylinders, edge curling, or mixed-curvature regions.

**Required correction:** report fit RMSE and `R^2`, area-weighted local principal-curvature distributions, positive/negative Gaussian-curvature area fractions, and sensitivity to excluding boundary elements.

### MEDIUM 5 — Raw artifacts and provenance were not preserved

The report points to `/tmp` directories (`reports/sequence_ablation_report.md:248-256`), but those directories and JSON files are now absent. The report lacks a git revision, binary checksum, compiler/build identity, thread count, environment, random seed, and exact command manifest.

**Required correction:** store configs, logs, summaries, analysis tables, and compact final VTPs under a persistent run directory with a machine-readable manifest.

## What the current experiments actually establish

1. The code can execute warm/cold, multiplicative/recursive-linearized, and grouped-relaxation variants.
2. Distinct algorithms produced distinct single-run outputs.
3. At `2e-5` per path, multiplicative and recursive-linearized target metrics were close, as expected from their second-order difference.
4. Intermediate relaxation schedule can perturb the final local-solver result.
5. The experiments do **not** establish whether the geometric differences exceed numerical repeatability/discretization error.
6. They do **not** establish a strain/path-count threshold where differences become substantial.
7. They do **not** establish that any compared state belongs to a distinct certified physical branch.

## Corrected experiment matrix

### Stage A — Numerical repeatability

Run one frozen configuration 5–10 times, serially, with fixed threads and seed. Record:

- binary checksum and git revision;
- iterations, energy/gradient evaluations, line-search failures;
- final energy and normalized gradient;
- Hessian minimum eigenvalue;
- Kabsch-aligned RMS/max geometry error;
- facewise target-metric error.

This defines the noise floor.

### Stage B — Mesh and tolerance convergence

Use at least three spatial resolutions and tolerances `1e-10`, `1e-12`, `1e-14`. Require the ablation effect to converge faster than the discretization error.

### Stage C — Warm-start sensitivity

For each accumulated target state, solve from:

- previous equilibrium;
- fixed flat state plus the same deterministic perturbation;
- several controlled random perturbations.

Certify each final state. Compare energy and aligned geometry; do not use wall time from concurrent runs.

### Stage D — Metric-update mechanics

Compare target metrics before solving for:

1. exact multiplicative composition;
2. recursive Euler-linearized composition;
3. reference-additive accumulation.

Sweep eigenstrain `{2e-5, 1e-4, 5e-4, 3e-3}` and path count `{3, 6, 12, 24}`. Include aligned isotropic and rotated anisotropic increments. For noncommuting increments, compare forward and reverse order directly at the target-metric level.

### Stage E — Exact segmentation

Implement the requested explicit ten-segment path. Compare:

- complete path ×3 with three relaxations;
- every segment relaxed;
- same segment ×3 then relaxed;
- forward/reverse/random segment order;
- one final solve from an identical initial geometry after confirming equal final target metrics.

The last case separates constitutive/load-order differences from continuation-basin effects.

## Approval recommendation

**REQUEST CHANGES.** Preserve the pilot results as exploratory observations, correct the curvature units and additive recurrence immediately, and do not use the current timing or micrometre differences as validation claims. Re-run the corrected matrix before drawing conclusions about physical path dependence or computational acceleration.
