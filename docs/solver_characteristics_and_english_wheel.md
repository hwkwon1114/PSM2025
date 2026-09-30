# Solver characteristics and English-wheel validation

## Completed study: observed results

The joint study (Slurm 5490113) completed in 1:52:51; the constitutive study (5490142) completed in 21:52. All nine mesh/load-step cases and all three constitutive cases accepted 20 physical paths. These are equilibrium-residual results, not physical validation.

At half-path amplitude steps, the existing analyzer reports:

| mesh parameter | rigid-aligned height span (mm) | quadratic fitted principal curvatures (1/m) |
|---|---:|---|
| 0.06 | 18.248 | -1.55517, -0.000841 |
| 0.03 | 18.341 | -1.59162, -0.007535 |
| 0.015 | 11.206 | -1.30465, -0.037151 |

Mesh convergence is not demonstrated: medium-to-fine height span changes by approximately 39%. On the finest mesh, full/half/quarter-path steps give height spans 11.191/11.206/11.421 mm. The earlier conclusion based on near-equal final energies was insufficient; geometry must also converge. Global quadratic curvatures are shape descriptors, not the complete local curvature field. Strongly unequal fitted curvatures do not demonstrate a balanced crown.

At mesh parameter 0.03 and half steps, multiplicative/recursive-linearized/reference-additive models give height spans 18.322/18.315/18.319 mm. This does not resolve constitutive discrimination above numerical uncertainty. The independent multiplicative joint run gives 18.341 mm, illustrating why repeated runs are required before attributing micrometre-scale differences to constitutive physics.

HLBFGS completed the coarse 20-path half-step case in 97.8 seconds. The trust-region batch timed out after 2:05:09; its CSV contains only two accepted substeps completing physical path one. This establishes a performance problem in that configuration, not correctness of a full trust-region trajectory or a different physical branch.

## Separate experimental questions

1. **Path dependence:** compare forward/reverse histories, measure final top/bottom target metrics, and include endpoint-only equilibrium controls. Different final target metrics mean different constitutive loads; they are not evidence of multiple equilibria at identical load.
2. **Numerical repeatability:** repeat identical runs at fixed thread count and compare single-thread runs separately. Floating-point parallel reduction variability is not an intentionally stochastic optimization algorithm.
3. **Prescribed geometry:** impose a current displacement field while retaining reference geometry, target metrics, directors/history conventions and constraints. Compare immediate next-path loading with re-equilibration before next-path loading. This tests initial-configuration sensitivity, not experimentally established accuracy. A maintained displacement constraint and a stress-free reference reset are different models.
4. **Crown coverage:** compare nested broad-to-central zigzags, reversed nesting, and constant footprint, with cross-wheeling as a separate control. Record coverage and integrated inelastic dose; differing dose distributions are intentional here, not matched-load path-dependence evidence.

## Verified English-wheel sources

### Eastwood / Mike Phillips: direct qualitative crown procedure

Source: [Form a Crown Using an English Wheel — Tech Tip From Eastwood](https://www.youtube.com/watch?v=GjzI7aAjno8), Eastwood Company, 2016-06-17, 6:21. The video description and auto-generated transcript were retrieved directly.

The transcript explicitly describes avoiding the outer edge, wheeling an initial broad box, then successively smaller boxes so the center receives more work. It contrasts this with repeatedly wheeling the same region. It also describes 90-degree cross-wheeling to even curvature in both directions and staggering stroke endpoints to avoid a ridge.

This verifies the user's described technique. It does not provide registered 3D scans, a calibrated pressure/gap history, plastic strain fields, or a quantitative deformation-error target. Treat it as qualitative validation of the coverage recipe only. The description lists machine dimensions and capability, not sufficient specimen/process data for simulation calibration.

### Rossi and Nicholas (2018): measured robotic English-wheel probes

Source: [Modelling A Complex Fabrication System: New design tools for doubly curved metal surfaces fabricated using the English Wheel](https://papers.cumincad.org/data/works/att/ecaade2018_298.pdf), eCAADe 36, pp. 811–820. Full PDF text was retrieved.

Reported setup: 25 by 25 cm, 1.5 mm aluminium sheets; Dinosaurier English wheel; ABB IRB1600 robot; Kinect point-cloud feedback. Tracking spacing is varied from 5 to 35 mm. The paper reports that leaving an unworked border is important for positive Gaussian curvature and that repeated/crossed passes alter the shape. Its feedback loop scans after a pass and adapts the next tool motion to the actual nonflat sheet.

Important distinction: adapting robot contact motion to a measured surface is not equivalent to resetting the constitutive state to that surface. The paper explicitly discusses elastic and plastic deformation and nonlinear accumulation.

Figures provide experimental shapes and tracking patterns; the retrieved paper does not include a downloadable, registered per-pass displacement dataset, force/gap history, or complete material calibration sufficient for a direct numerical accuracy benchmark. Its augmented 320-tensor ML dataset is not 320 independently measured panels. Pixel classification performance is not forming geometry error.

### Northwestern process metrology: promising data lead

Source: [Ping Guo group publications](https://pingguo.mech.northwestern.edu/publications/). The retrieved institutional text describes using Vicon to monitor sheet position, orientation and deformation during English wheeling, for error compensation and subsequent toolpaths. This supports the relevance of process-resolved measurements. No raw measurement download was established from the retrieved page.

### Practitioner boundary and mixed-process caution

Source: [How to use an English wheel for metal shaping](https://www.streetmachine.com.au/features/how-to-use-an-english-wheel-for-metal-shaping), Chad Atkinson / Street Machine. The page describes a 1 mm sheet, outer border, tight M-like tracking and matching anvil contour. Its worked example also uses mallet/shot-bag stretching, tuck shrinking and planishing. It is not a pure English-wheel dataset and must not be used as one.

## Physical validation boundary

No quantitatively usable public raw English-wheel dataset has yet been established by this search. This is a search outcome, not a claim that none exists. Contacting the cited experimental groups or obtaining per-pass scans is preferable to extracting heights from perspective video frames.

To test accurate next-pass prediction, obtain: initial reference geometry, material and thickness, wheel radii and gap/force, contact path and pass order, fixture/handling conditions, registered unloaded 3D scans before and after each pass, measurement uncertainty, and repeated specimens. For a geometry-only restart, residual stress and plastic/internal variables remain unidentified. A simulation can test sensitivity to that missing state, but cannot prove accuracy without independent measurements.

## Output conventions

Report full vector displacement U = x_current - X_reference, all three components in metres, with original reference/current coordinates and mesh connectivity in VTP. Provide CSV exports and 3D equal-scale figures; display-unit conversion to millimetres is labeled. Rigid-aligned shape comparisons are separate from raw displacement. Toolpath figures must identify actual centerlines versus face coverage and show the plate boundary and path order.

### Additional verified bibliographic lead (full text not retrieved)

Huang, Suarez, Kang, Ehmann and Cao (2023), *Robot forming: Automated English wheel as an avenue for flexibility and repeatability*, Manufacturing Letters 35, 342–349, [DOI 10.1016/j.mfglet.2023.08.104](https://doi.org/10.1016/j.mfglet.2023.08.104). Title, authors, pages and DOI were verified through Crossref. Publisher and ResearchGate retrieval returned HTTP 403; the text-mining endpoint returned HTTP 400. No measured values are attributed to this article here without its full text. This is a particularly relevant source/data-contact lead because it addresses the exact process, unlike generic incremental sheet forming or automotive wheel manufacture.

## Full-field refinement result

The full-field export reveals an orthogonal material-axis change, not merely a lower amplitude. At load step 0.5, material-frame fitted `(k_uu, k_vv)` are approximately `(-0.0023, -1.531)`, `(-0.0088, -1.567)`, and `(-1.294, -0.0391)` m⁻¹ for resolutions 0.06, 0.03, and 0.015. Dominant bending directions are 89.6°, 89.8°, and 178.0° from material u. This is consistent with mesh-dependent equilibrium-branch selection; it is not evidence of a converged crown.

The plate flags `-lx 0.127 -ly 0.1524` specify half-extents, so the actual plate is **254 × 304.8 mm**, not 127 × 152.4 mm. The `-res` value is a meshing control parameter, not a physical element length.

See `run/solver_characteristics/figures/mesh_comparison_three_meshes.png` and `cross_mesh_shape_metrics.csv`. Per-case CSV files contain every vertex's material coordinates, reference/current xyz, raw displacement components and magnitude, plus separately labeled rigid-aligned displacement.

## Separate experiment execution status

Prepared 28 cases: 10 path-order/cadence controls, 8 identical-input HLBFGS repetitions (5 at eight threads, 3 at one thread), 3 prescribed-shape controls, and 7 shrinking/constant/cross-wheeled coverage controls. The shrinking recipe uses the actual 254 × 304.8 mm plate, beginning with a 220 × 272 mm centerline footprint and ending at 40 × 52 mm. These are uncalibrated model inputs, not measured wheel parameters.

The compute-node build and all four study jobs completed with exit code 0. All 28 cases passed their configured acceptance checks. The analytic prescribed-shape execution path has now run successfully, both directly and with re-equilibration. The displacement CSV loader separately passed 17 parser checks.

The solver jobs are dependency-chained so no two study families run simultaneously:

| Job | Work | Dependency |
|---|---|---|
| 5561645 | Single-process C++ build | Completed |
| 5561732 | Prescribed deformation controls | Successful build |
| 5561734 | Matched-target path order and cadence | Previous job finishes |
| 5561735 | Identical-input repeatability | Previous job finishes |
| 5561736 | Shrinking crown controls | Previous job finishes |

All batch analysis and visualization commands completed. The numerical analyzer was rerun after all 28 cases finished; diagnostics below use `run/solver_characteristics/characteristics_analysis.json`.

## Completed experiment diagnosis

All shape RMS differences below compare corresponding material vertices after proper rigid alignment; they are not differences in scalar height span.

- **Repeatability:** contrasts against the first replica give 0.0036–0.0192 mm RMS at eight threads and 0.0084–0.0094 mm at one thread, with bitwise-identical final metrics. Single-thread runs are not identical either; attributing this solely to OpenMP is unsupported.
- **Cadence:** every-path versus endpoint-only equilibration gives 3.140 mm RMS forward and 3.159 mm reverse, with identical target metrics. Every-two-path differences are smaller (0.0415 / 0.0317 mm). This establishes substantial numerical history/initialization sensitivity, not physical plastic path dependence.
- **Order:** additive forward/reverse gives 0.0495 mm RMS at matching metrics. Multiplicative gives 0.0370 mm RMS and only 6.37e-14 relative metric difference. Contrary to the design's expectation, this parameter set does not meaningfully separate constitutive noncommutativity.
- **Prescribed shape:** direct next path differs from baseline by 0.0323 mm RMS; re-equilibration before the next path differs by 3.127 mm, at identical final metrics. This is consistent with different equilibrium basins; Hessian stability and tighter convergence comparisons are needed before certifying distinct minima.
- **Shrinking recipe:** broad-to-central gives 8.461 mm height span; reverse gives 5.397 mm and 2.619 mm RMS shape difference at essentially identical metrics (1.15e-16 relative). This reinforces strong numerical path sensitivity.
- **Crown quality:** nested fitted principal curvatures are (-0.6867,-0.0313) m^-1, a magnitude ratio of about 22. Cross-wheeling gives (-0.5773,-0.0367), ratio about 16, but also changes footprints. These are highly unequal double curvatures, not balanced dome formation. Constant broad/mid/central spans are 14.279/7.461/2.014 mm; central-only fit R²=0.660 makes global quadratic curvature a poor descriptor there. Coverage controls change total loading and cannot establish equal-work efficiency.

Decision: investigate multiple initial states at a frozen final target, comparing energies, free residuals, and smallest Hessian modes before changing the physical model or training a surrogate. Small order effects need replicated paired runs; large cadence/prescription effects are already far above the observed repeatability differences. No physical accuracy claim follows from these tests.
