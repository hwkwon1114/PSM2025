# Independent Review: Directional Mechanics and Angle Distortion

## Verdict: CHANGES_REQUESTED

**The decision to exclude these mechanisms from the current paper is sound.** Preserve them as exploratory hypotheses for future English-wheel modeling, not as validated contact mechanisms or production recipes.

The local angle-distortion tensor derivation is correct. However, the investigation contains consequential implementation defects, inconsistent strain/curvature conventions, and benchmark provenance errors. These block scientific reliance on the reported quantitative conclusions and future deployment of the directional extensions until corrected.

This verdict does **not** require reopening the current paper, authorize new experiments, or invalidate unrelated baseline calibration results.

---

## 1. Reviewed scope, identities, and authority

### Authority and objective

The supplied researcher directive is:

> “for the current paper i think i wont use this idea but it is useful in the longrun for simulating english wheel dynamics.”

The review therefore assesses mathematical validity, exploratory evidence, implementation integrity, and requirements for future work. No approved prospective experimental protocol or authorization to resume the investigation was supplied.

Applicable project instructions were supplied from:

- `/gpfs/projects/p33082/AGENTS.md`
- `/gpfs/projects/p33082/PSM2025-zigzag/AGENTS.md`

Historical reports and production recommendations are evidence to inspect, not execution authority. No historical cumulative compute allowance was imposed.

### Reviewed identities

The working tree was unchanged against reviewer-launch HEAD **`b93f4772f019`**, both at initial and final inspection.

Reviewed:

- The three supplied directional mechanics reports.
- Both supplied coupon benchmark reports.
- `src/libshell/ZigZagGrowth.hpp`
- `src/libshell/ZigZagSequenceGrowth.hpp`
- Relevant sequence-processing and geometry sections of `src/simulations/Sim_Bilayer_Growth.cpp`
- Both supplied Python benchmark scripts.
- `test/testshell/Test_ZigZagSequenceMetric.cpp`
- Supporting metric-update, geometry, configuration-accessor, and bilayer-energy code.
- Saved benchmark summaries, selected result/convergence/stability records, argument logs, and VTP headers.
- Historical `build_cmake/Testing/Temporary/LastTest.log`.

**The angle-distortion model is a design proposal, not an implemented feature:** no `angle_distortion` implementation was found under `src/`.

### Checks actually performed

Static control/data-flow review; independent tensor and laminate derivations; comparison of reported results with saved metadata; examination of test coverage and output precision.

No commands, builds, simulations, sanitizer runs, numerical derivative checks, or experimental measurements were executed. Saved artifacts were not independently recomputed, and their generating binary/source hashes were not established.

---

## 2. Launch-blocking findings

These findings block **future scientific reliance or directional-feature deployment**, not archival retention of the hypotheses or the decision to defer them.

### B1 — Executed coupon geometry does not match the stated benchmark

**Locations:**  
`python/benchmark_directional_baseline.py:125–145`;  
`python/benchmark_directional_prototype.py:135–145`;  
`src/simulations/Sim_Bilayer_Growth.cpp:4726–4734`;  
`src/libshell/Geometry.hpp:478–497,542–545`;  
`docs/reports/directional_path_dependence_synthesis_and_scaling.md:38–44`.

Both scripts pass `-lx 0.127 -ly 0.1524`. The selected rectangle constructor interprets these as **half-extents**.

Saved VTP headers confirm:

- `material_u`: −0.127 to +0.127 m
- `material_v`: −0.1524 to +0.1524 m

Thus the executed coupon was **254 × 304.8 mm**, not **127 × 152.4 mm**.

Additional discrepancies:

- Saved meshes have **1,435 vertices and 2,720 faces**, not the reported mesh description.
- The selected geometry is a structured right-triangle mesh, not an equilateral Delaunay mesh.
- `res=0.03` is used through `edgeLength = 2*Lx*res`; it is not a 0.03 m edge-length input. Here the nominal spacing is 7.62 mm.
- The simulation sets **`E=1`**, with `nu=0.33` (`Sim_Bilayer_Growth.cpp:203–211`), rather than explicitly using 69 GPa. Uniform energy normalization can preserve unloaded equilibrium shapes, but it changes the interpretation of energies and absolute gradient tolerances.

**Clearance:** Correct the benchmark specification and provenance. Either relabel historical results with their actual geometry or, under new authorization, rerun the intended geometry using the correct half-extents. Export and assert material bounds, mesh counts, thickness, material normalization, and actual spacing.

### B2 — Model B uses a headless orientation as a directed travel vector

**Location:** `src/simulations/Sim_Bilayer_Growth.cpp:2521–2524`.

The code constructs:

```cpp
const double travel_sign = op.reverse_traversal ? -1.0 : 1.0;
const Eigen::Vector2d t_vec(std::cos(hit.angle_rad), std::sin(hit.angle_rad));
```

But `hit.angle_rad` has already been normalized modulo π. The forward zigzag alternates upward and downward strokes; normalization removes that alternation.

For example, a downward forward stroke has a negative vertical travel component, but its normalized director can point upward. Multiplying all strokes by one operation-level sign cannot recover their physical tangents.

The prototype can therefore break symmetry while implementing the **wrong directional slope field**.

**Clearance:** Carry an explicit material-frame unit travel vector derived from ordered endpoints, rotate it with placement, and reverse it with traversal. Test upward, downward, oblique, rotated, and reversed strokes before reassessing Model B results.

### B3 — Reversal corrupts physical strip identity

**Locations:**  
`src/libshell/ZigZagGrowth.hpp:281–287,644,834`;  
`src/simulations/Sim_Bilayer_Growth.cpp:2361–2377,2527–2534`.

Reversal reverses the strip vector. Collectors then assign `hit.strip_idx = k`, using the **new traversal position**. Downstream code uses this index for:

- `active_strips`, documented as physical strip selection;
- `op.ortho_top` and `op.ortho_bottom`, stored in original physical-strip order.

The strip’s scalar growth travels with its reversed object, while orthotropy is fetched from a differently indexed array. Nonuniform per-strip inputs consequently become mismatched. Selecting physical strip 0 can instead select the opposite-end strip.

The homogeneous, all-strip prototype inputs mask this defect.

**Clearance:** Separate immutable physical strip ID from traversal index. Add reversal tests with distinct per-strip growth and layer orthotropy, plus single-strip selection.

### B4 — Benchmark comparisons mix incompatible metrics and nonobjective shape descriptors

**Locations:**  
`python/benchmark_directional_baseline.py:218–219`;  
`python/benchmark_directional_prototype.py:105–126,255`;  
the comparison tables in both benchmark reports and the synthesis.

The baseline transforms edge-coordinate metrics into the material Cartesian frame. The prototype compares raw `[a11,a12,a22]` edge components.

Consequently:

- The baseline and prototype metric numbers have different bases and dimensional interpretations.
- The prototype vector norm also omits the second off-diagonal contribution required by a symmetric tensor’s Frobenius norm.
- The metric columns cannot be used as a common mechanism-strength comparison.

The prototype estimates “principal curvature” from the Hessian of a quadratic fitted in unaligned global XY coordinates. This is only a small-slope approximation, not a general principal-curvature calculation. Its global Z-span is likewise sensitive to rigid rotation.

The reported Model A results illustrate the problem: essentially identical aligned RMS differences coexist with Z-span differences of **8.526 mm** versus **0.150 mm** between cadences. These descriptors do not establish physical curvature or height changes.

**Clearance:** Use one material-frame metric definition, consistent tensor norms, checked vertex/face correspondence, and preferably material-area weighting. Compute signed curvature in a documented material-attached frame or from discrete fundamental forms. Report fit residuals and rigid-transformation invariance tests.

### B5 — The shear-to-curvature interpretation is not consistent with the implemented bilayer model

**Location:** `docs/reports/directional_angle_distortion_orthotropic_review.md:55–82,135–136`.

The report conflates:

1. A forward-minus-reverse **stretch-tensor difference**;
2. Single-pass **engineering shear strain**;
3. Tensorial versus engineering twisting curvature;
4. Surface values of a linear through-thickness strain profile versus piecewise-constant bilayer eigenstrains.

These distinctions change factors of two and, for the bilayer reduction, factors of 3/2. The proposed direct inversion for δ is therefore not established.

**Clearance:** Adopt explicit strain and curvature conventions, derive the layer reduction from the actual energy, and verify it with an analytic small-strain twist patch test. Correct the inference formula and uncertainty treatment before calibration.

### B6 — Saved evidence is not safely reusable or sufficient to support the broad conclusions

**Locations:**  
`python/benchmark_directional_baseline.py:108–118,159–174`;  
`python/benchmark_directional_prototype.py:166–169,221–224`;  
`docs/reports/directional_path_dependence_synthesis_and_scaling.md:113–133`.

The baseline overwrites the requested configuration before accepting an old completed result, without matching configuration, binary, source, mesh, or tolerance identity. This can associate new inputs with stale outputs.

Both scripts permit fallback selection of VTP files rather than requiring the exact accepted final artifact. The prototype does not validate the result manifest’s completion/acceptance before analysis.

The following claims are also unsupported by the inspected evidence:

- 96–99% grouped-cadence accuracy;
- 100% Model A accuracy;
- approximately one-minute runtime for 239 paths;
- a hybrid recipe having the highest physical fidelity;
- the cited 51.95 μm value being a transferable numerical noise floor.

**Clearance:** Use immutable run identities and manifests, exact final-artifact selection, and explicit acceptance checks. Remove unsupported accuracy/runtime claims or label them as untested hypotheses. Future scaling validation requires a separately approved protocol.

### B7 — Directional regression tests and input contracts are incomplete

**Locations:**  
`src/libshell/ZigZagSequenceGrowth.hpp:299–324`;  
`src/simulations/Sim_Bilayer_Growth.cpp:1416–1417,2516–2524`;  
`src/libshell/ZigZagGrowth.hpp:512,637–639,829–830,971`.

No targeted tests were found for directional drift, reversal, slope feedback, or angle distortion in the searched test sources.

Specific gaps include:

- No explicit finite/range validation for the drift coefficient or slope-feedback coefficient.
- No local conditioning/finite-result check around the slope reconstruction.
- Clipping can hide invalid arithmetic and changes the proposed linear model outside its unclipped regime.
- `DirectionalDrift` is accepted by the shared profile parser but silently ignored by the legacy `apply` and `applyMaterialCoordinates` paths.

**Clearance:** Define supported interfaces and coefficient domains; reject unsupported modes rather than silently ignoring them. Add targeted unit/integration tests, including disabled-feature equivalence, malformed inputs, degeneracy, and finite/SPD outputs.

---

## 3. Independent mathematical assessment

### 3.1 Local nematic symmetry: correct, with an important boundary

For

\[
G(\theta)=R(\theta)L R(\theta)^T,
\qquad L=\operatorname{diag}(1+g_1,1+g_2),
\]

\[
R(\theta+\pi)=-R(\theta)
\quad\Longrightarrow\quad
G(\theta+\pi)=G(\theta).
\]

This is an exact statement about **reversing one stroke’s tangent while preserving its location, parameters, and update order**.

It does not prove that reversing an entire sequence leaves the final metric unchanged. Multiplicative updates associated with different orientations generally do not commute. Hardening and overlap assignment can introduce additional order dependence.

The baseline script reverses strip order but does not set `reverse_traversal` on its operations. It therefore primarily tests **order reversal**, not an independently implemented endpoint-reversal operation.

The saved baseline maximum material-metric difference is **8.48×10⁻⁸**, not approximately 10⁻²⁰. This is compatible with a small order effect; it is not proof of machine-exact schedule invariance.

### 3.2 Angle-distortion discrepancy: correct in the nominal local frame

Write \(o=\mathrm{ortho}\), \(\ell_i=1+g_i\), and

\[
g_1-g_2=2go.
\]

Then

\[
G_{12}(\delta)=\frac{g_1-g_2}{2}\sin(2\delta),
\]

so

\[
\boxed{
\Delta G
=
G(+\delta)-G(-\delta)
=
2go\sin(2\delta)
\begin{pmatrix}0&1\\1&0\end{pmatrix}.
}
\]

This derivation is correct.

Qualifications:

- It is a stretch difference, not directly a metric increment.
- It is off-diagonal in the **nominal path-aligned frame**. In global coordinates, use \(R(\theta)\Delta G R(\theta)^T\); its diagonal entries generally do not vanish.
- Each rotated tensor retains the same determinant and eigenvalues. Relative to the unrotated tensor, individual diagonal components change at order \(\delta^2\).
- Anisotropy is necessary but not sufficient for a nonzero discrepancy: \(g=0\), \(\delta=0\), and other zeros of \(\sin 2\delta\) also eliminate it.

For an initially Euclidean material metric, \(a=G^TG=G^2\). Hence

\[
\Delta a=(2+g_1+g_2)\Delta G=2(1+g)\Delta G,
\]

and the Green-strain difference is

\[
\Delta E=\tfrac12\Delta a=(1+g)\Delta G.
\]

These exact relations should replace ambiguous use of “growth metric” for \(G\).

### 3.3 Lie-algebra structure: correct and useful

With

\[
J=
\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\qquad R(\delta)=e^{\delta J},
\]

\[
G(\delta)=e^{\delta J}Le^{-\delta J},
\qquad
\frac{dG}{d\delta}=[J,G].
\]

At zero,

\[
[J,L]=(g_1-g_2)
\begin{pmatrix}0&1\\1&0\end{pmatrix},
\]

so

\[
G(+\delta)-G(-\delta)=2\delta[J,L]+O(\delta^3).
\]

Although \(J\) is skew-symmetric, its commutator with symmetric \(L\) is symmetric, as required. This gives a straightforward analytic derivative and verification target.

### 3.4 Engineering shear and twisting curvature

For small strain, define engineering shear by

\[
\gamma_{12}=2\epsilon_{12}.
\]

For one positive-angle pass,

\[
\gamma_{12}^{+}\approx 2G_{12}(+\delta)
=2go\sin(2\delta).
\]

Thus the report’s expression can represent **single-pass positive-direction engineering shear**. The forward-minus-reverse engineering-shear difference is twice that expression.

Now define tensorial curvature \(K\) by

\[
\epsilon(z)=\epsilon_0+zK.
\]

For a linear strain profile specified at \(z=\pm h/2\),

\[
\boxed{
K_{12}=\frac{\gamma_{\rm top}-\gamma_{\rm bot}}{2h}.
}
\]

If “twisting curvature” instead means engineering twist \(2K_{12}\), division by \(h\) is correct. Normal/sign conventions may reverse the sign.

But the code represents two finite layers, not merely two surface samples. For equal-stiffness, equal-thickness layers with piecewise-constant eigenstrains, the unconstrained small-strain natural curvature satisfies

\[
K^*
=\frac{12}{h^3}\int_{-h/2}^{h/2}z\,\epsilon^*(z)\,dz
=\frac{3}{2h}
(\epsilon^*_{\rm top}-\epsilon^*_{\rm bot}),
\]

giving

\[
|K_{12}^*|
=\frac{3}{4h}|\gamma^*_{\rm top}-\gamma^*_{\rm bot}|.
\]

The sign must be reconciled with the repository’s second-fundamental-form convention. The opposite-sign layer mixed-energy coefficients in `src/libshell/TinyADHessian_Bilayer.hpp:77–80,259–266` make this a necessary implementation-level check.

Finally, natural curvature is not necessarily the realized curvature of a finite incompatible shell: membrane compatibility, boundaries, and equilibrium branches matter.

### 3.5 Accumulation and cancellation

Forward reinforcement and reverse cancellation are sensible **first-order predictions for the odd shear component**, under matched loading and fixed parameters.

They do not establish exact cancellation of multiplicative tensor updates, nor guarantee an untwisted final shell. Unequal hardening, overlapping orientations, normal growth, constraints, and branch selection can leave residual twist.

### 3.6 Other formulation corrections

`docs/reports/directional_growth_mechanics_formulation.md:81–83` contains a parity error:

\[
(-t)\otimes(-m)=t\otimes m,
\]

and, if \(v=vt\) with positive speed, \(\operatorname{sgn}(t\cdot v)\) does not reverse. The proposed expression is therefore not odd as written. A separately defined material/machine polarity or history variable is required.

Also:

- \(\nabla z\) is dimensionless when \(z,u,v\) have length units; the unnormalized slope coefficient is dimensionless.
- \(t^Tb\,t\) is even under \(t\mapsto-t\). Curvature feedback alone does not break reversal symmetry.
- Frictional transverse shear \(\gamma_{13}\) is not the in-plane shear \(\gamma_{12}\) responsible for the proposed twist coupling.
- The implemented Model B is clipped slope feedback; it does not implement the report’s curvature-feedback term.

---

## 4. Contact-mechanics plausibility

| Proposed mechanism | Assessment | Discriminating evidence |
|---|---|---|
| Spindle compliance/caster deflection | Plausible **if** machine geometry supplies a tangential-force-to-yaw coupling. A symmetric spindle need not yaw merely because longitudinal friction reverses. | Measure wheel yaw and lateral displacement versus travel sign and load; vary spindle stiffness, alignment, and wheel mounting. |
| Contact-patch slip asymmetry | Plausible with lateral creepage, camber, misalignment, or asymmetric contact. Longitudinal friction alone does not establish in-plane lateral shear. | Record force/torque, wheel and sheet speeds, lubrication, and near-contact displacement fields; test whether yaw/shear changes with these controls. |
| Crystallographic rolling texture | Plausible source of anisotropic/noncoaxial flow, but not by itself proof of travel-odd behavior. Ordinary orthotropy retains 180° symmetry. | Test sheet orientations relative to mill rolling direction, characterize texture/r-values, and separate texture effects from reversal-sensitive history or machine asymmetry. |

The contact mechanisms are **hypotheses**, not deductions from the observed shape difference.

Similarly, Model A is an imposed longitudinal growth gradient, not a resolved frictional shear model. Steady rolling does not automatically imply monotonically increasing strain along a long stroke.

Model B additionally needs a defined machine-relative frame. Using global \(z\) on a freely rotating specimen can change the growth prescription under an arbitrary rigid-body rotation. A scalar height gradient is physically meaningful only relative to a specified tool/contact orientation.

A large simulated response at \(\gamma=25\) establishes sensitivity of this chosen model—not physical plausibility, parameter calibration, or a demonstrated mechanical bifurcation.

---

## 5. Computational architecture and numerical integrity

### Valid advantages of the angle formulation

With a prescribed, state-independent δ and fixed ordered hits:

- Target metrics can be assembled without intermediate equilibria.
- A final-only equilibrium solve is architecturally compatible.
- Rotations preserve positive principal stretches.
- Constitutive derivatives with respect to δ are analytic.
- Additional constitutive work is modest relative to nonlinear equilibrium.

These are useful reasons to retain the concept.

### Necessary limitations

- **Metric cadence independence is not equilibrium branch independence.** Identical target metrics can yield different local minima under different continuation histories.
- Cadence independence disappears if δ depends on evolving force, geometry, yaw, or plastic history.
- “Zero intermediate solves” means **one final solve**, not zero total solves.
- The angle feature has not been benchmarked for 239 paths. Runtime cannot be inferred from coupon solve counts.
- End-to-end differentiability is not guaranteed by an analytic rotation. Hit selection, clipping, solver branching, and singular equilibrium Hessians remain obstacles.
- Integration is not a two-line change: schema, directed travel, layer-specific angles, diagnostics, derivatives, and tests are all needed.

### Numerical safety

The multiplicative update in `GrowthHelper.hpp:740–826` correctly uses congruence, checks positive incremental stretches, rejects degenerate material triangles, symmetrizes the result, and checks positive definiteness. These are valuable safeguards.

However:

- Finite-value checks should be explicit before clipping and metric assembly.
- Slope reconstruction needs conditioning and residual checks.
- Large feedback can produce extreme growth or extensive clipping; report both.
- `countsTowardHistory` uses nominal hit growth, not slope-scaled effective growth (`Sim_Bilayer_Growth.cpp:2570–2572` vicinity). With `plastic_hits`, a slope-clipped zero-growth hit can still harden the face. This must be specified and tested.
- Model B with cold resets, curved initial states, or prescribed deformation needs explicit semantics. “Final-only implies zero feedback” holds only for the relevant flat, unchanged initial state.

### Backward compatibility and memory safety

Static inspection supports disabled-feature compatibility:

- `reverse_traversal=false` avoids reversal.
- Uniform profiles bypass drift.
- Zero slope feedback bypasses slope reconstruction.
- Directional drift with coefficient zero produces a unit profile for finite valid inputs.

But **bit-for-bit historical compatibility was not demonstrated** by this review.

No dangling-pointer defect was identified in the new slope loop. `getVertices()` returns an Eigen reference view (`DCSConfigurations.hpp:340–358`); the backing mesh remains alive and is not resized during that loop. Bounds-checked orthotropy access avoids silent out-of-range access, though it does not fix the semantic indexing defect.

This is not sanitizer evidence. Generic public geometry APIs also retain assumptions about nonempty coordinate arrays and three-column face connectivity.

---

## 6. What the saved evidence actually establishes

The saved JSON summaries reproduce the quoted shape differences:

| Case | Saved aligned RMS | Interpretation |
|---|---:|---|
| Baseline final solve | 0.00535 μm | Near-null output comparison for the executed schedule and mesh |
| Baseline pass-wise | 0.01970 μm | Same limitation |
| Model A final solve | 134.424 μm | Exploratory directional response on the larger-than-reported coupon |
| Model A pass-wise | 134.421 μm | Similar pairwise response; not proof of identical cadence-selected states |
| Model B pass-wise | 7,566.411 μm | Response of the implemented, directionally incorrect slope prescription |

Evidence paths:

- `run/benchmark_directional_baseline/baseline_directional_summary.json`
- `run/benchmark_directional_prototype/prototype_directional_summary.json`

Selected result records support actual solver acceptance: the inspected baseline final solve and Model B reverse run report gradient targets of \(5\times10^{-14}\), accepted solves, and `positive_pivots`.

However:

- The inspected `sequence_stability.csv` reports `not_evaluated`. Do not equate this with a separately performed physical stability/bifurcation analysis.
- Saved VTP `Points` are **Float32**. The inspected baseline file lacks `X_current`, so its analysis falls back to those coordinates. Converting them to double does not recover lost precision.
- Nanometer-scale differences at these coordinate magnitudes are close to output quantization scales. They cannot establish double-precision solver accuracy.
- Neither benchmark script actually passes `-sequence_solve_mode final_only`; they obtain one final equilibrium through cadence settings under `every_cycle`. This supports the solve-count observation, not direct exercise of the named final-only route.
- No experimental scans, uncertainty estimates, or calibrated directional coefficients were inspected.

Historical `LastTest.log` records 128 tests scheduled and 127 passed. The four inspected `ZigZagSequenceMetric` tests cover pullback, orthogonal-pair behavior, and isotropic accumulation—not directional extensions. Their historical passage is useful but not a version-matched directional regression certificate.

The Python files are exploratory benchmark drivers, not assertion-based regression tests.

---

## 7. Required future validation and experimental protocol

These are recommendations for a **newly approved future protocol**, not execution instructions.

### Stage A — Repair verification before physical inference

1. Correct directed travel and physical strip identity.
2. Add exact algebraic tests for δ=0, isotropic growth, reversal, rotation covariance, layer separation, and analytic derivatives.
3. Verify the bilayer twist coefficient and sign against a small-strain analytic case.
4. Assert actual coupon dimensions and mesh identity.
5. Test rigid-transform invariance of postprocessing.
6. Compare full-precision states, target metrics, realized curvature fields, and solver acceptance.
7. Exercise defaults versus explicit zero flags.
8. Add invalid-input, degenerate-triangle, clipping, and effective-hardening tests.

### Stage B — Separate traversal direction from operation order

Begin with straight, nonoverlapping strokes before returning to zigzags.

Use matched FF, FR, RF, and RR schedules, with independent replicated coupons and randomized run order. Separately vary:

- Endpoint reversal at fixed strip order;
- Strip order at fixed local travel direction;
- Both together.

Control alloy/temper, thickness, initial flatness, mill direction, force or gap, speed, lubrication, wheel geometry, supports, and repositioning. Determine replication from measured variability and a prespecified smallest relevant effect; two schedule examples are not sufficient replication.

### Stage C — Measure twist and its candidate causes

Use registered laser profilometry or comparable 3D metrology before forming and after selected passes. Retain material fiducials and signed surface normals.

Measure:

- Full curvature tensor in material coordinates, especially signed \(K_{12}\);
- Corner heights as secondary descriptors;
- In-plane strain/shear, preferably through DIC or a fiducial grid;
- Thickness/profile changes along strokes;
- Machine-relative wheel yaw, load, torque, and speed.

Repeat measurements after removal/repositioning to quantify metrology and handling uncertainty. Account for gravity and support conditions.

Vary spindle stiffness/alignment, lubrication, and sheet orientation to discriminate compliance, slip, and texture hypotheses.

### Stage D — Prespecify interpretation under each outcome

- **Odd twist tracks measured yaw and changes with spindle stiffness:** supports compliance-mediated angle distortion.
- **Effect tracks lubrication/creepage with negligible yaw:** favors slip-related mechanisms.
- **Effect follows sheet orientation:** supports texture coupling, but travel-odd behavior still requires explanation.
- **Effect depends strongly on intermediate geometry or pass history:** a fixed-δ model is incomplete.
- **Endpoint gradients dominate:** investigate drift rather than identifying yaw from shape alone.
- **No repeatable effect above uncertainty:** bound the effective directional parameter; do not force a nonzero calibration.
- **Several mechanisms predict the same scans:** report an effective model-conditional parameter, not a uniquely identified physical mechanism.

### Stage E — Identification and held-out prediction

For small δ, shear scales approximately with \(g\,o\,\delta\). Final shape alone may therefore identify only a product of parameters.

The reported `ortho_top≈−0.46` is a prior model-calibration result, not an independently established physical contact property. Its uncertainty, transferability, and omitted-mechanism bias must propagate into δ inference.

Fit a constrained small-angle model on designated calibration coupons and evaluate held-out loads, orientations, and schedules. Include baseline, drift-only, angle-only, and appropriately defined state-feedback comparators. Do not tune them on the held-out outcomes.

Future unattended execution should record immutable source/binary/configuration identities and exercise interruption/restart if restart capability is claimed. Existing completed-output skipping is not resumability evidence.

---

## 8. Strategic recommendation and claim boundaries

**Maintain the present paper boundary.** Adding δ now would introduce an unvalidated degree of freedom into a calibration whose physical identification is already conditional on the chosen model.

Preserve the angle formulation because it offers a compact, differentiable way to represent an experimentally testable odd shear contribution. Label it:

> A phenomenological directional extension motivated by possible contact asymmetries; not yet calibrated or physically identified.

Safe conclusions are:

1. The baseline local orthotropic increment is nematic.
2. Direction-dependent scalar modulation or an explicitly signed axis offset can break that local symmetry.
3. State-independent directional increments can be assembled before one final equilibrium solve.
4. The current prototypes show numerical sensitivity, subject to the defects and provenance limits above.

Unsafe conclusions are:

- Exact invariance of all reversed schedules;
- Established frictional, caster, or texture causation;
- Correct quantitative twist prediction from the current formula;
- Unique identification of δ from panel orthotropy and coupon shape;
- Validated 239-path accuracy or runtime;
- Production readiness of the directional code.

**Residual risk remains even after the listed fixes:** nonunique physical mechanisms, model-dependent parameter identification, incompatible natural curvature, equilibrium multistability, and imperfect transfer from coupons to full English-wheel dynamics.

Approval of any corrected slice would still require reassessment and would not itself grant execution permission, additional resources, or held-out-data release.