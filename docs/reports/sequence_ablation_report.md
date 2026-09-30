# Sequence-Path Ablation Report

## Question

This report tests three questions for recurring zigzag toolpaths:

1. Does warm-starting each energy minimization from the previous converged geometry matter?
2. Is the accumulated target metric update better represented by multiplicative metric composition or by an additive small-strain approximation?
3. Does changing the order and grouping of segmented strokes change the final state, and can grouping reduce computation?

The tests use forward simulations only. No inverse path optimizer or target-shape objective was used.

## Common test case

The baseline was the Putong-style rectangular panel and toolpath, with the low-strain loading retained from the preceding experiments:

- thickness: `h_total = 0.6 mm`
- panel input: `lx = 0.127 m`, `ly = 0.1524 m`
- mesh resolution: `res = 0.03`
- zigzag length: `140 mm`
- angle: `9.13 deg`
- strips: `10`
- strip width: `10 mm`
- top eigenstrain per active strip hit: `2e-5`
- bottom eigenstrain: `0`
- orthotropy: `0`
- HLBFGS tolerance: `1e-12`
- seeded branch escape: disabled for the ablations

The resulting solver coordinates span approximately `254 x 305 mm`; this is the established coordinate convention of this geometry setup.

All comparisons use the same mesh, material model, target loading, tolerance, and solver. The only changed variable is the ablation under test.

## Mechanics and implementation model

The sequence code keeps the reference geometry and reference curvature fixed. It stores the accumulated top and bottom target first fundamental forms in the rest configuration. Each material hit contributes an incremental growth stretch tensor.

For a triangle, the code constructs the material-edge matrix `Dm`, forms the material-space growth tensor `G`, pulls it into the triangle edge basis as

\[
T = D_m^{-1} G D_m,
\]

and updates the target metric as

\[
\bar a_{n+1} = T^T \bar a_n T.
\]

This is the finite-kinematics metric update implemented in `src/libshell/GrowthHelper.hpp:702-731,769-811`.

During minimization, the target metric is fixed. The solver varies the current geometry and recomputes the current first and second fundamental forms:

\[
a(X), \qquad b(X).
\]

The converged current geometry is then used as the next warm-start geometry. The sequence loop and per-hit target updates are in `src/simulations/Sim_Bilayer_Growth.cpp:1567-1776`.

## 1. Warm-start ablation

### Cases

- **Warm:** `-sequence_warm_start true`
- **Cold:** `-sequence_warm_start false`

Both cases used 12 identical Putong-style paths. The cold case reset only the current geometry to the fixed rest geometry before each minimization; it retained the accumulated target metric.

### Results

| Case | Energy minimizations | Runtime | Final z-span |
|---|---:|---:|---:|
| Warm start | 12 | 216.61 s | 2.692 mm |
| Cold start | 12 | 270.07 s | 2.799 mm |

Warm starting reduced runtime by approximately 19.8%.

After removing rigid translation from the final point sets:

- RMS geometry difference: `20.3 micrometers`
- maximum pointwise difference: `89.1 micrometers`

Global quadratic-fit principal curvatures were:

- warm: `[-209.6, -57.8] m^-1`
- cold: `[-218.6, -54.2] m^-1`

### Interpretation

**Evidence:** Warm-starting changes both runtime and the final numerical state in this 12-path sequence.

**Inference:** Warm-starting is not merely a computational cache. In a nonlinear shell problem, the previous equilibrium is a physically meaningful continuation point and can influence which nearby basin the local minimizer follows.

**Limit:** These two ablations were HLBFGS-converged but not Hessian-certified. The observed difference does not prove that warm and cold starts found distinct physical minima; it could include basin selection, finite stopping, or numerical path effects. A stronger branch claim requires exact-Hessian certification of both final states.

## 2. Multiplicative versus additive metric update

### Mechanics basis

The target metric is a squared-length object. If the incremental growth deformation is

\[
F_g = I + \delta F,
\]

then the exact metric transformation is

\[
\bar a_{n+1} = F_g^T \bar a_n F_g.
\]

Expanding to first order gives

\[
\bar a_{n+1}
\approx
\bar a_n + \delta F^T\bar a_n + \bar a_n\delta F.
\]

The additive form omits the second-order term

\[
\delta F^T\bar a_n\delta F.
\]

Thus additive updating is a valid first-order small-strain approximation, while the multiplicative congruence is the mechanically consistent finite-increment update. Repeated aligned isotropic growth illustrates the difference:

\[
\bar a_n^{\mathrm{mult}}=(1+e)^{2n}\bar a_0,
\qquad
\bar a_n^{\mathrm{add}}=(1+2ne)\bar a_0.
\]

For `e=2e-5` and `n=12`, the relative metric difference is only `1.10e-7`.

### Cases

- **Multiplicative:** `-metric_update multiplicative`
- **Additive linearized:** `-metric_update additive_linearized`

The additive branch uses

\[
\Delta\bar a
= \delta T^T\bar a + \bar a\delta T,
\]

not a naive constant matrix addition. The production default remains multiplicative.

### Results

| Update | Runtime | Final z-span | Principal curvatures |
|---|---:|---:|---:|
| Multiplicative | 216.61 s | 2.692 mm | `[-209.6, -57.8] m^-1` |
| Additive linearized | 216.96 s | 2.698 mm | `[-208.4, -59.2] m^-1` |

Final geometry difference:

- RMS: `3.72 micrometers`
- maximum: `20.2 micrometers`

Stored target-metric difference:

- maximum component difference: `1.09e-12`
- relative difference: `9.60e-9`

### Interpretation

**Evidence:** At the tested eigenstrain and 12 paths, the additive and multiplicative results are nearly identical. The observed geometric difference is micrometer-scale relative to a roughly 250--305 mm panel.

**Inference:** The current low-strain experiment is in the regime where the first-order additive approximation is numerically adequate for this particular sequence. This does not make it the preferred constitutive update.

**Mechanics limit:** The discrepancy grows with increment size and accumulated path count. Since the omitted terms are second order in the incremental deformation, increasing eigenstrain or applying many more increments can produce substantial target-metric drift. The additive branch can also become non-positive-definite at sufficiently large increments; the implementation checks positive definiteness after each update.

## 3. Stroke segmentation and ordering

### Construction

The canonical Putong zigzag builder creates 10 strip segments. To isolate the effect of stroke ordering, the same 10-strip geometry was represented as 10 separate operations. Each operation activated only one strip's eigenstrain list entry; inactive strips contributed zero growth.

Three repeated-toolpath schedules were run:

1. **Whole-path baseline:** all 10 strips, repeated three times; minimize after each complete path.
2. **Segmented path-major:** strips 1--10, then strips 1--10 again, then strips 1--10 again; minimize after every segment.
3. **Segmented stroke-major:** strip 1 three times, minimize; strip 2 three times, minimize; through strip 10.

The stroke-major case used the new `-sequence_minimize_every 3` control. Non-minimizing repeats still updated the target metrics; they deferred the nonlinear solve until the group boundary.

This test isolates the existing 10 generated strips. It does not split a single centerline at its midpoint.

### Results

| Schedule | Physical cycles | Energy minimizations | Runtime | Final z-span |
|---|---:|---:|---:|---:|
| Whole-path baseline | 3 | 3 | 55.88 s | 0.787 mm |
| Segmented path-major | 30 | 30 | 119.92 s | 0.757 mm |
| Segmented stroke-major | 30 | 10 | 108.01 s | 0.731 mm |

Segmented path-major versus segmented stroke-major:

- RMS geometry difference: `3.78 micrometers`
- maximum pointwise difference: `18.9 micrometers`
- path-major curvatures: `[-43.46, -37.64] m^-1`
- stroke-major curvatures: `[-42.13, -37.23] m^-1`

### Interpretation

**Evidence:** The two segmented orders do not produce exactly the same state. They differ by several micrometers, with approximately 3.5% difference in fitted z-span and small curvature differences.

**Inference:** The order is not mathematically interchangeable once each group is allowed to relax independently. The reason is that the state after each minimization changes the geometry on which later growth increments are applied, and the nonlinear solve is path dependent.

**Computation:** Grouping three repeated strokes reduced the number of minimizations from 30 to 10. Runtime decreased from 119.92 s to 108.01 s, approximately 9.9% for this coarse case. The runtime saving is smaller than 3x because each cycle still performs hit collection, target-metric updates, diagnostics, and bookkeeping.

**Practical conclusion:** Grouping is a valid acceleration approximation when micrometer-scale state differences are acceptable. It should not be assumed exactly equivalent near buckling, branch transitions, high strain, coarse tolerances, or strongly noncommuting path sequences.

## Overall conclusions

1. **Warm starts matter.** In the tested 12-path sequence they saved about 20% runtime and changed the final state by up to 89 micrometers pointwise.
2. **Multiplicative metric composition is mechanically preferred.** At `2e-5` per path, the additive linearized approximation was effectively indistinguishable, but its error scales upward with strain and accumulated increments.
3. **Stroke grouping changes the result.** The segmented order comparison produced small but measurable differences. Grouping reduced the number of solves by 3x and runtime by about 10% in this test.
4. **Substantial differences are most likely near nonlinear sensitivity points.** The present low-strain cases do not show large divergence, but the mechanics and warm-start evidence indicate that larger differences can occur when the target metric increment is no longer small, when the accumulated state approaches a bifurcation, or when competing equilibrium branches are close in energy.

The current evidence establishes that the effects are not pure floating-point noise, but it does not define a universal threshold for when they become large. That threshold requires a controlled sweep over eigenstrain, path count, tolerance, and proximity to branch transitions, with final-state Hessian certification.

## Reproducibility controls added

The sequence solver now supports:

```text
-sequence_warm_start true|false
-metric_update multiplicative|additive_linearized
-sequence_minimize_every N
```

Defaults preserve the original behavior:

```text
-sequence_warm_start true
-metric_update multiplicative
-sequence_minimize_every 1
```

Relevant implementation files:

- `src/libshell/GrowthHelper.hpp`
- `src/simulations/Sim_Bilayer_Growth.cpp`

Representative output directories:

```text
/tmp/ablate_warm/
/tmp/ablate_cold/
/tmp/ablate_additive/
/tmp/seg_path_major/
/tmp/seg_stroke_major/
```
