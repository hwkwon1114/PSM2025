# REPLICATE-HW1 — Forward prediction of collaborator experiment "Hyunwoo Run 1" (24 passes) with Attempt 209

**Status:** COMPLETE 2026-11-04 (results below). APPROVED 2026-11-04 by the researcher, as drafted. Thickness 0.61 mm (A2) and front-view back-face frame (A1) confirmed. **Drafted:** 2026-11-04.

## Question

If the 24-pass wheeling toolpath in `shape_design/data/Wheeling_Toolpath_24_Passes_Consolidated.csv` is applied in the frozen Attempt 209 model, does the predicted shape match the scanned specimen `shape_design/data/Hyunwoo_Run_1_24_Passes.ply`?

This is a **blind forward prediction**. Nothing is fitted to this scan. Researcher instructions (2026-11-04): use the Attempt 209 calibration. For a turned-over (back-face) pass, swap the growth factors.

## Decision driven

- **Shape reproduced** (sign pattern and amplitude within a factor of 2): Attempt 209 transfers to this sheet and toolpath. The project can use it for design on this setup.
- **Sign pattern reproduced, amplitude off:** the model captures the mechanism but not the magnitude, for example because of the thickness or pitch extrapolation. Recalibration on this specimen would be a separate card.
- **Sign pattern not reproduced:** the back-face representation or the transfer fails. Report it; do not tune.

## Inputs (read-only)

**Toolpath.** The CSV has 24 passes in 4 rounds. Each round runs R1 F, R2 F, R3 F, R3 R, R2 R, R1 R.

| Region | Face | CSV x (mm) | Legs (inclined) |
|---|---|---|---|
| Region 1 | front | 10–60 | 12 (10) |
| Region 2 | front | 150–210 | 14 (12) |
| Region 3 | back | 70–150 | 18 (16) |

- All passes span y = 10–114 mm. The pitch is 5 mm per inclined leg, so α = atan(5/104) = 2.7525°.
- A reverse (R) pass retraces the forward segments in the opposite order and direction (checked).

**Scan.** Geomagic PLY, 429,465 vertices, single surface. The PCA in-plane extent is 220.7 × 124.5 mm and the height range is about 3.4 mm. Fourteen debris points lie more than 5 mm off the plane.

**Model.** Frozen Attempt 209 parameters and solver configuration, loaded and checked by `shape_design.family.load_frozen()`:
- `gtop=2.8835e-4`, `gbot=1.4588e-5`, `ortho_top=-0.39892`, `ortho_bottom=0.96031`
- Voce hit-count hardening, β = 0.22314, face-shared, plastic hits
- native hybrid, final-only equilibrium, 5e-14 gradient gate

## Formulas

- **F1. Frame.** u = x − 110 and v = y − 62 (mm). The sheet is 220 × 124 mm and centred, as in `shape_design/family.py`.
- **F2. Pass → solver operation.** Each pass is one `zigzag` operation with `lv = 104`, `n_strips` = number of legs, `|α| = 2.7525°`, `center_uv` = bounding-box centre, and `width_mm = 2` (the calibrated band).
  - The sign of α, the rotation (0° or 180°) and `reverse_traversal` are chosen so that the C++ `buildZigZagStrips` copy regenerates the CSV legs **exactly**, with the same order and direction.
  - Region 1 needs a 180° rotation because its zigzag phase is opposite (it starts at the bottom).
- **F3. Back face.** Swap `gtop ↔ gbot` and `ortho_top ↔ ortho_bottom`. Geometry is unchanged under assumption A1.
- **F4. Comparison.** The scan is levelled by its PCA plane and debris points with |h| > 5 mm are removed. The prediction is the full-precision solver state.
  - The scan is rigidly aligned to the prediction by point-to-surface ICP for each of the 4 proper orientations of the rectangle (identity, 180° about the normal, and turned over about either in-plane axis).
  - **Primary:** two-sided symmetric RMS distance in the best orientation. The other three orientations are reported too.
  - **Secondary:** the width profile (mean height across the central |v| ≤ 40 mm, in 5 mm bins along u), the profile peak-to-valley ratio prediction/scan, and the per-region curvature sign (quadratic fit of each region's profile).
  - **Implementation details (added 2026-11-04 after verifier review; no change of metric):**
    - Debris triangles, meaning any triangle with a debris vertex, are removed from the scan surface used for backward distances.
    - Forward RMS is computed over one scan point per 2 mm cell. Backward RMS is computed over the prediction vertices, uniformly weighted.
    - The profile bins span u ∈ [−105, 105] mm, giving 42 bins and avoiding the free edges. Within each bin, height is linearly interpolated on a fixed 1 mm (u, v) grid and then averaged, so scan and prediction are sampled identically.
    - ICP is symmetric and rigid only: forward point-to-plane and backward point-to-point, with monotone step acceptance. Iteration count and stopping status are recorded.

## Assumptions (A1, A2 confirmed by researcher 2026-11-04)

- **A1.** Back-face rows in the CSV are written in the same front-view frame as front rows.
  - If they are instead in the turned-over machine frame, flipping about the 124 mm axis changes nothing: the Region 3 zigzag with an even number of legs is mirror-symmetric in u.
  - Flipping about the 220 mm axis would reverse the Region 3 zigzag phase (v → −v).
- **A2.** The sheet is 220 × 124 × **0.61 mm**: the thickness of the double-curvature design study. The scan confirms 220 × 124. Attempt 209 was calibrated at 0.77 mm, so a thinner sheet is an extrapolation.
- **A3.** The 2 mm calibrated band applies to the 5 mm-pitch path. Only about 38 % of each region is hit, against near-full coverage at the Run 1 pitch of 2.27 mm. This is an extrapolation.

## Known-answer checks (before any solve)

- **K1.** Every pass is regenerated exactly from the C++ strip formula: max |error| = 0 mm (pass, done).
- **K2.** Per-face hits are predicted **directly from the CSV legs**, independently of F2. After the solve they must equal the solver's `total_hit_count` on every face.
- **K3.** Sequence JSON: 24 toolpaths. Front passes carry the frozen parameters and back passes the swapped ones. Defaults and hardening equal the Attempt 209 source.
- **K4.** Comparison fixtures:
  - A predicted surface compared with itself, after a known rigid motion and orientation, gives RMS ≈ 0 and recovers the orientation.
  - A height-scaled copy gives the right profile ratio.

## Acceptance (after the solve)

- `runner.check_output` passes: gradient target and positive pivots.
- The state, VTP and Python grid correspond within the same tolerances as `shape_design/run_case.py`.
- K2 shows 0 mismatched faces.

## Stopping rule

There is one solve: 2 mm mesh (the calibrated resolution), 3600 s limit. If it fails or times out, report it; do not retry or relax tolerances without approval. A 1 mm mesh check is optional and needs separate approval. No parameter changes and no fitting to the scan.

## Compute plan

- One Slurm job: normal partition, 8 CPUs, 16 GB, wall time 2 h, `BLIS_NUM_THREADS=1` per project policy.
- The solve is expected to take 10–40 min, based on the 13,640-face design-study pilots.
- The analysis is solver-free.
- A `verifier` review is required before the solve.

## Code and outputs

- `shape_design/replicate_hw1.py`: CSV parser, F2 fit, sequence, independent hit prediction.
- `shape_design/run_hw1.py`: one solve with checks.
- `shape_design/compare_hw1.py`: F4.
- `shape_design/tests/test_replicate_hw1.py`: K1, K3, K4.
- Outputs go to `run/shape_design/hw_run1/`.

## Results (2026-11-04)

**Run.** Slurm 7249407; outputs in `run/shape_design/hw_run1/solve_mesh2/`. The verifier first blocked the run (B1: profile bins; B2: debris triangles), both were fixed, and the re-review gave APPROVED_WITH_CAVEATS.

**Solve.** Accepted in 138 s:
- gradient-target termination with positive pivots; gradient norm 2.9e-16 against a 5e-14 tolerance;
- 24 cycles;
- K2 hit check: 0 of 13,640 faces mismatched (max 32 hits; 29.6 % of faces hit);
- state, VTP and grid correspondence within tolerance.

**Comparison** (`comparison/comparison.json`; rigid alignment only):

| Orientation | Symmetric RMS (mm) |
|---|---|
| turn_over_u_axis (best) | 0.372 |
| turn_over_v_axis | 0.383 |
| identity | 1.921 |
| rot180_normal | 1.917 |

- **Orientation.** The scan was recorded from the other face; this is a scanner pose, not a model result. The two turned-over poses differ by only 3 %, so which end of the sheet is Region 1 is not resolved by the nearly symmetric shape. ICP reached the 100-iteration cap for `identity` and `turn_over_v_axis`; the best pose stopped on a rejected step after 56 iterations.
- **Best pose.** Forward RMS is 0.373 mm (p95 0.77 mm) and backward RMS 0.370 mm.
- **Width profile.** Peak-to-valley is 2.79 mm predicted against 3.20 mm scanned, a ratio of 0.87. The profile RMS difference is 0.31 mm.
- **Region curvature (2a, mm⁻¹).** The signs match in all three regions: front regions concave down and the back region concave up in the aligned frame.

| Region | Predicted | Scan | Predicted / scan |
|---|---|---|---|
| Region 1 | −1.46e-3 | −2.37e-3 | 0.62 |
| Region 3 | +1.49e-3 | +2.37e-3 | 0.63 |
| Region 2 | −1.46e-3 | −2.59e-3 | 0.57 |

**Outcome per the decision rule: "Shape reproduced"** (sign pattern matches; amplitude ratio 0.87, within a factor of 2).

**Qualification.**
- Local bending inside each region is under-predicted by about 40 %.
- The peak-to-valley ratio is closer to 1 mainly because the predicted crests stay high out to the free edges (x = ±100 mm). The scan instead falls off toward the edges, with crests near the region centres (x ≈ ±75 mm).
- This is one specimen and one prediction. It extrapolates thickness (0.61 vs 0.77 mm), pitch (5 vs 2.27 mm) and the back-face representation.

**Figure.** `comparison/hw_run1_shape_comparison.{pdf,png}` with a caption, produced by `shape_design/plot_hw1_comparison.py`. The PNG was inspected.
