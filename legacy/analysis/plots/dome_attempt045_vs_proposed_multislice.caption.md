# Figure: Multi-Slice Elevation and Curvature Comparison

**File:** `dome_attempt045_vs_proposed_multislice.pdf` / `dome_attempt045_vs_proposed_multislice.png`  
**Datasets:**
- Attempt 045 Baseline (3 discrete toolpaths, Stage 1 BO best compromise): `run/dome/bo_continuation_stage1/attempt_045/`
- Proposed Pitch-Graded Concentric Crosshatch: `run/dome/proposed_matched_analysis_v1/`

### Description:
Quantitative comparison of out-of-plane displacement $z$ and Gaussian curvature $K$ between the 3-toolpath Bayesian optimization baseline (Attempt 045) and the proposed pitch-graded concentric cross-hatch design (`proposed_matched_solve`).

- **(a) Longitudinal Centerline ($y = 0$ mm):** The proposed crosshatch raises peak dome deflection from 4.03 mm to 7.50 mm (+86% amplitude gain) while establishing a smooth, symmetric spherical profile across the 200 mm sheet width.
- **(b) Transverse Centerline ($x = 0$ mm):** Along the 300 mm length, the 3-toolpath baseline suffers from boundary flattening and edge curling, whereas the proposed crosshatch maintains continuous positive arching across the full length.
- **(c) Off-Center Slice ($y = +50$ mm):** Located in the critical transition zone between toolpaths. Attempt 045 drops below $z = 0$, reflecting edge flaring and anti-clastic saddle warping. In contrast, the proposed design preserves an elevated dome profile ($z > 3.5$ mm).
- **(d) Gaussian Curvature Profile ($x = 0$ mm):** Highlights the root mechanics breakthrough. The discrete toolpaths in Attempt 045 cause Gaussian curvature to oscillate violently, plunging into severe negative saddle valleys ($K = -0.423\ \mathrm{m}^{-2}$ at $y = \pm 75$ mm) in the unprinted gaps between strips. The pitch-graded concentric crosshatch completely eliminates the saddle valleys, maintaining strictly positive Gaussian curvature ($K > +0.17\ \mathrm{m}^{-2}$) throughout the entire central ROI.
