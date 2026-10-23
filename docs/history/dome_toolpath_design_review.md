# Calibrated dome investigation — independent design review

## Verdict: `CHANGES_REQUESTED`

**The investigation is scientifically relevant and the proposed implementation route is appropriately small.** One scoring-contract issue must be resolved before scientific execution: distinguish a dome in the **surface geometry** from positive Gaussian curvature in the shell’s **director-dependent curvature field**.

This is a **design review**, not implementation certification or launch approval. No files were changed, simulations run, packages installed, or work delegated.

## Reviewed scope and authority

Authority supplied and inspected:

- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — objective version 3, revision 60 as read.
- `docs/reports/dome_toolpath_investigation.md` — current draft protocol.
- The task’s explicit researcher decisions: collaborator JSON, Attempt209 coefficients, iterative layout search, and 200 × 300 × 0.77 mm sheet with minimal side-path adaptation.

Inputs inspected:

- `calibration/data/processed/zigzag_doublecurv.json`.
- `run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00/{sequence,command}.json`.
- `run/run9_transfer_expanded/evaluations/attempt_0209/case_00/{sequence,command}.json`.
- `run/run9_transfer_expanded/evaluations/attempt_9901/case_00/sequence.json`, resolving the Run9 command’s path alias at the inspected-content level.

Implementation inspected:

- `src/libshell/ZigZagGrowth.hpp`.
- `src/libshell/ZigZagSequenceGrowth.hpp`.
- Relevant portions of `src/libshell/GrowthHelper.hpp`, `ComputeCurvatures.{hpp,cpp}`, `ExtendedTriangleInfo.hpp`, and `WriteVTK.hpp`.
- General sequence accumulation, native acceptance/export, final-only execution and checkpoint restrictions in `src/simulations/Sim_Bilayer_Growth.cpp`.
- `test/testshell/Test_ZigZagSequenceMetric.cpp`.

Supporting records: `docs/reports/absolute_gradient_stopping_audit.md` and the completed four-condition section of `docs/reports/boundary_conditions_and_springback_synthesis.md`. Historical permissions and cumulative budgets were not imported.

The working tree already contained tracked and untracked work, including changes to `Sim_Bilayer_Growth.cpp`. Review applies to the contents inspected, **not a newly hashed source/binary freeze**. No executable identity was certified.

## Scientific assessment

The question is well connected to the objective: can spatial/directional redistribution of a fixed prescribed-growth law produce sustained double curvature rather than cylindrical bending?

The proposed baseline, fixed central region, area weighting, magnitude floor, anisotropy ratio, sensitivity reporting and preservation of unsuccessful trials are appropriate. The decisions under different outcomes should remain:

- **Verified qualifying dome:** constructive success for this model and recipe.
- **Improvement without qualification:** evidence guiding another documented, within-scope layout change.
- **Accepted equilibrium that fails the criterion:** negative result for that candidate.
- **Failed equilibrium, inconsistent curvature estimators or mesh-sensitive qualification:** unresolved—not evidence that no dome exists.

Competing explanations include genuine geometric double curvature, director/geometry disagreement, footprint aliasing, spatially localized bending, and numerical branch selection. None requires recalibration or solver research to recognize.

The target is a **model-conditional simulated shape**. Attempt209 supplies fitted effective growth parameters; this investigation does not physically identify contact mechanics, validate width/profile transfer, or establish manufacturing performance.

## Launch-blocking design finding

### B1 — Give geometric curvature an explicit acceptance role

**Locations:** `docs/reports/dome_toolpath_investigation.md:25–27`; `src/libshell/ComputeCurvatures.cpp:14–34`; `src/libshell/ExtendedTriangleInfo.hpp:81–84,229–242`; `src/simulations/Sim_Bilayer_Growth.cpp:1890–1901`.

The exported fields are mathematically legitimate shell curvatures:

\[
S=a^{-1}b,\qquad H=\tfrac12\operatorname{tr}S,\qquad K=\det S,
\qquad k_\pm=H\pm\sqrt{H^2-K}.
\]

However, `b` depends on edge directors through
`alpha = 0.5*theta + sign*phi`. It is **not determined solely by the displayed vertex surface**. A valid signed principal-curvature calculation from these fields therefore does not, by itself, establish that the visible surface is a dome.

The draft requests a geometric cross-check but does not yet specify how disagreement affects qualification. It also leaves implementation room to pool positive-convex and negative-concave patches into one “same-sign” area fraction.

**Concrete consequence:** an analyzer could pass a director-curvature field, mixed-sign lobes, or a result whose geometric estimate is inconclusive, while reporting a smooth geometric dome.

**Smallest adequate repair before searching:**

1. State whether geometry-derived principal curvature is the primary acceptance measurement, or require geometric confirmation of the director-based qualification. **Disagreement about qualification must produce “unresolved,” not success.** Numerical equality between estimators is unnecessary.
2. Freeze the geometric estimator’s neighborhood/scale, normal convention and validity rules using analytic fixtures, not candidate outcomes.
3. Make sign coherence explicit. For each global sign \(s\in\{-1,+1\}\), compute the qualifying reference-area fraction using
   \[
   sk_1>0,\quad sk_2>0,\quad
   \min(|k_1|,|k_2|)\ge0.1\ {\rm m}^{-1},\quad
   \frac{\min(|k_1|,|k_2|)}{\max(|k_1|,|k_2|)}\ge0.1.
   \]
   Require one sign’s fraction to reach 80%; do **not** add the two fractions. Invalid estimates must remain in the fixed region’s denominator and cannot contribute qualifying area.
4. Retain the stated spatial/unexaggerated-shape veto against folds, isolated bumps and cell-scale noise.

This is a protocol clarification and analysis implementation requirement, not a request to change physics or acceptance thresholds.

**Operational blockers found:** none inherent in the proposed design. Execution identities, safeguards and tests remain future pre-execution review requirements.

## Source-level findings and nonblocking caveats

### 1. Calibration identity and transfer

Both inspected Attempt209 sequences agree on:

| Parameter | Value |
|---|---:|
| `gtop` | 0.00028835116227798787 |
| `gbot` | 1.4587994001079146e-05 |
| `ortho_top` | -0.3989157893255133 |
| `ortho_bottom` | 0.960306323937336 |
| `beta` | 0.2231435513142097 |

They use uniform profiles, 2 mm widths, floor zero and face-shared `plastic_hits`. Run6 has 29 repeats; Run9 has 33. Both commands use final-only, the same sheet/thickness, native gradient gate \(5\times10^{-14}\), 2000 warmup attempts and eight Hessian threads. Their archived commands enable adaptive continuation and request 3600 seconds; these are **not** the new proposed execution settings.

The Run9 `attempt_9901` sequence inspected has the same content as its Attempt209 sequence. Byte/hash equality was not independently computed here.

The collaborator JSON instead contains legacy growth and shared-orthotropy values. A derived baseline must replace them with the scalar Attempt209 coefficients and remove `ortho` when supplying both layer-specific fields; the parser rejects mixed representations.

**Transfer limitations are already correctly recognized:**

- A inherits `center_peak`, multiplying **top growth only** by  
  \(0.4+0.6\sin^2(\pi s)\); bottom growth is unchanged. This changes the local layer mismatch, not merely a common treatment amplitude.
- Widths 8/4 mm versus calibrated 2 mm expand treated area without a calibrated width-dependent contact law.
- Preserve these collaborator features as fixed assumptions. Neither quietly normalize them away nor describe their transfer as experimentally validated.

Negative `ortho_top` means the top-layer increment along the strip is smaller than its transverse increment. Toolpath orientation alone therefore cannot identify the weaker bending direction.

### 2. Proposed footprint adaptation is valid

`ZigZagGrowth.hpp` constructs centerline span

\[
W=(N-2)L_v\tan\alpha
\]

and checks the transformed width-inflated bounding rectangle against material bounds.

For the stated 0°/90° placements, approximate bounds in millimetres are:

| Path | \(u\) bounds | \(v\) bounds |
|---|---:|---:|
| A, unchanged | ±94 | ±74 |
| Original B | ±102 | 88.03 to 151.97 |
| Adapted B: Lv=196, shift=118 | ±100 | 86.62 to 149.38 |
| Adapted C | ±100 | -149.38 to -86.62 |

Thus the original sides fail the approved sheet; the adaptation fits. The sides **touch** the \(u\) boundaries rather than having positive clearance. The placement checker includes a small numerical tolerance.

This is a minimally adapted calibrated collaborator-layout baseline, not an exact reproduction of the original input. Every candidate still needs width-aware validation. Increasing strip count at fixed angle widens the footprint; densification at fixed span requires compensating angle changes.

Continuous footprint validity does not establish adequate discrete face coverage. Hits are selected by material-space face centroids, so mesh refinement can change treatment near band boundaries.

### 3. Material-frame targets do not depend on equilibrium cadence under this recipe

The relevant update is

\[
G=R\,\mathrm{diag}(1+g_1,1+g_2)R^\top,\quad
T=D_m^{-1}GD_m,\quad
\bar a_{\rm new}=T^\top\bar a_{\rm old}T.
\]

With nonsingular material triangles and positive incremental stretches, this preserves positive definiteness; the implementation symmetrizes and checks the result.

In `Sim_Bilayer_Growth.cpp:2396–2608`, hit selection, directions, profile values and hardening depend on material coordinates and ordered history. Current geometry affects the growth magnitude only through optional slope feedback. Therefore, with **`sequence_slope_feedback=0`**, identical mesh, ordered hits and multiplicative updates give the same final targets/history for final-only and every-cycle execution.

The equilibrium shape can still depend on cadence through initialization and branch selection. Mapping visualizations onto the current surface do not make the constitutive update geometry-dependent.

Here \(q_n=0.8^n\). Repeats saturate rather than depositing proportionally increasing strain. Changing overlap, direction or order can change accumulated targets; cadence independence must not be generalized to recipe-order independence.

### 4. General three-toolpath final-only execution is the smallest route

`Sim_Bilayer_Growth.cpp:1048–1059` explicitly restricts `sequence_never_clamped` to one recipe. Do not enable it for A/B/C or remove that guard merely to obtain familiar outputs.

The general sequence loop supports all three recipes. With final-only and adaptive continuation disabled, it accumulates all repeats and solves once at the complete target. The baseline has **eight cycles: A×2, B×3, C×3**.

Existing outputs suffice for screening and analysis:

- `sequence_history.csv`;
- convergence records and native proof;
- `sequence_result.json`;
- final VTP with accumulated metrics/history and mean/Gaussian curvature;
- native endpoint JSON with full-precision positions, topology, material coordinates, targets and selected DOFs.

Important distinctions:

- `sequence_history.csv` writes its solve-intent flag before acceptance; it is not an acceptance certificate.
- Native state export explicitly says **diagnostic, not a restart checkpoint**.
- General three-toolpath checkpoint/resume is not supported by the inspected route.
- A fresh, bounded final-only run does not require checkpoint generalization. If interrupted, retain the failure; do not claim resumability or silently restart.

### 5. Curvature numerics and geometry measurement

For nondegenerate triangles, `a` is SPD and `b` symmetric, so \(a^{-1}b\) has real eigenvalues despite generally being nonsymmetric. The signed reconstruction is valid. Units are \(H,k:\mathrm{m}^{-1}\), \(K:\mathrm{m}^{-2}\).

Use a scale-aware roundoff allowance for \(H^2-K\); materially negative or nonfinite values must be flagged, not silently clipped. Near a cylinder, stable recovery of the smaller root using the product \(k_1k_2=K\) avoids avoidable cancellation.

A local quadratic fit should use a Euclidean tangent chart with its first and second fundamental forms, or a full parametric surface fit. A height Hessian in reference \((u,v)\) alone is not generally physical curvature after deformation. Keep normals consistently oriented and report conditioning/invalid support.

`WriteVTK.hpp` stores scalar fields as doubles, but its point allocation does not request double precision. Use native endpoint JSON positions for geometric fits and small shape differences; casting VTP positions to float64 cannot recover lost precision.

The scalar \(H,K\) fields contain **no principal directions**. Direction-guided refinement needs geometric directions or reconstructed shape operators, mapped back to material coordinates. Avoid relying on legacy `computeDir` without separate verification: its projection code is global-axis dependent and contains an apparent `ProjPointX`/`ProjPointY` assignment error.

### 6. Existing four-condition evidence has limited reuse

The completed record reports very small cadence differences for Run6/9: approximately \(4.35\times10^{-5}\) and \(5.28\times10^{-7}\) mm RMS. That supports using final-only as a practical screening strategy, not equivalence for this three-path, wider-band, profiled layout.

A final-only dome can be reported specifically as such. Any claim that per-cycle forming gives the same dome requires the proposed matched cadence verification with equal targets/history. Neither positive native pivots nor these previous comparisons establish global optimality.

## Recommended minimal implementation and verification

1. **Resolve B1 and freeze the protocol.** Keep the existing ROI and primary thresholds. Record sensitivity analyses separately.
2. **Generate a new derived JSON without modifying the source.** Preserve profiles, widths and A/B/C order; replace calibration fields; apply only the approved initial side adaptation. Validate finite parameters, footprint bounds, repeat counts and exact intended differences.
3. **Use a thin isolated runner and analyzer; avoid C++ changes initially.** Explicitly fix final-only, adaptive=false, multiplicative updates, slope feedback zero, no prescribed deformation/seed escape, disabled BCs, and native state export. Refuse existing output directories.
4. **Exercise changed paths before production:** generator/parser checks; analytic plane, sphere/dome, cylinder and saddle curvature fixtures under rigid transformations; invalid-field and mixed-sign rejection; and a small three-recipe final-only integration check demonstrating ordered accumulation and one accepted full-load solve. Existing metric tests support algebra but do not cover this whole path.
5. **At launch review, bind actual identities and evidence.** Verify cycles=8, one terminal solve, complete native acceptance, matching endpoint artifacts, finite/SPD targets and expected history. Preserve the dirty tree and old binaries/results.
6. **Run the baseline first, then documented sparse refinements.** Keep all trials. Predeclare how candidate improvement is judged; do not move the region or thresholds.
7. **Verify the selected candidate** with tighter supported tolerance and finer mesh, retaining RMS/tail/max shape comparisons and curvature coverage. Compare hit fields and material-frame metrics across meshes; raw edge-basis metric components are not directly comparable. Treat changed qualification as unresolved.

The proposed compute-node safeguards and external process ceiling are sensible. Native timing is cooperative; this review does not establish runtime enforcement. No historical cumulative quota applies.

## Checks performed and limitations

Performed: read-only authority/provenance comparison, source-level mathematics and control-flow inspection, manual footprint calculation, review of existing test coverage and supporting conclusions, and working-tree delta inspection.

Not performed: code execution, tests, simulation, compilation, independent hashes, endpoint recomputation, figure inspection, or scheduler/resource verification. No dome-specific implementation or result existed to certify.