# Sequential Toolpath Ablation v2 — Experiment Specification

## Objective

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
