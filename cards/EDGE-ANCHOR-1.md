# EDGE-ANCHOR-1 — Scan placement from physical sheet edges

**Status:** COMPLETE — edge anchor adopted for future campaigns (DECISIONS.md, 2026-11-01). APPROVED 2026-11-01 by the researcher ("sure"), with the gates as proposed. Pilot (Run 1 only) done 2026-11-01; full pass: `calibration/analysis/scan_edge_anchor_full.py` → `run/diagnostics/scan_edge_anchor/`.
**Owner:** researcher. **Drafted:** 2026-11-01.

## Question

The double-sided symmetric RMS is retained unchanged. Does anchoring the scan's in-plane origin and yaw to its **physical sheet edges** change the scores or rankings of saved simulations, compared with the current anchor? The current anchor is the 0.5/99.5 % raw-extent quantile midpoint plus a long-edge yaw set during preprocessing, followed by the point-centroid zeroing in `score_material_frame`.

Principle (researcher, 2026-11-01): the scan and the simulation start from the same flat sheet, so edge misalignment caused by deformation is real error and must stay in the loss. The pose is **not** fitted per candidate. The scored domain must not change with the pose; earlier rotation fits biased the score by dropping the high-error edge points.

## Decision driven

- **Negligible** (all gates G1–G3 pass on every scan): keep the current anchor. Record that it is equivalent to the edge anchor within tolerance. No metric change.
- **Material** (any gate fails): adopt the edge-anchored scan frame (variant A, or B if A and B differ) for **future** campaigns only. Rescore completed campaigns in a new output folder; never overwrite them. Running jobs (e.g. Slurm 6653633 `run1_clean`) are not changed.

## Formulas

- **F1. Edge samples.** Raw STL vertices are mapped by the recorded target transform `(p − origin) @ basis_columns − shift_local_mm`, with a gross z gate of |z − median| ≤ 20 mm. Keep the largest 8-connected occupied component on a 0.5 mm grid (≥2 points per cell, holes filled). For each 1 mm strip along each side, within the central 80 % of the side (corners excluded), take the 5th-most-extreme point across the side.
- **F2. Edge lines and frame.** Fit Theil–Sen lines: x = a + b·y for the long sides, y = c + d·x for the ends. The centre c_e is the intersection of the two midlines. The yaw θ_e is the mean of atan(−b̄) (long sides) and atan(d̄) (ends).
- **F3. Variant A (scan edge-anchored).** Replace the scan centring `t − mean(t)` with `(t − [c_e, mean z]) @ Rz(θ_e)`. Everything else is as in `score_material_frame`: the simulation keeps its Kabsch material frame, both are levelled, and the score is the same double-sided symmetric RMS on the same 14,504 points with a fail-closed footprint.
- **F4. Variant B (like-for-like gauge).** As A, and the simulation is additionally re-anchored by F2 applied to its exact material-boundary vertices (|u| = 100 or |v| = 150 mm, central 80 %).
- **F5. Paired change.** ΔJ = J_variant − J_current per saved evaluation. Rank agreement is the number of discordant pairs (Kendall).

## Known-answer checks

- **K1. Synthetic rectangle** (~40 points/mm², interior hole) with known shift and yaw: recover the centre within 0.05 mm and yaw within 0.01°; the centroid must be visibly biased. *Pilot: PASS (centroid bias 12.1 mm).*
- **K2. Target reproduction.** Re-running the recorded crop/bin step on the transformed raw STL reproduces the frozen target exactly. *Pilot: PASS, 14,504 cells, max |diff| 0.*
- **K3. Production identity.** The variant scorer with the current anchor equals `score_material_frame` to within 1e-12 mm, and equals every stored objective to within 1e-6 mm. *Pilot: PASS for all 10 evaluations.*
- **K4 (full run).** Rescoring a synthetic simulation identical to the scan gives J = 0 under A and B.

## Metrics and gates (per scan)

- **G1.** Edge-derived centre offset |c_e| ≤ 0.5 mm and |θ_e| ≤ 0.2°.
- **G2.** |ΔJ|/J ≤ 1 % for every saved valid evaluation with J < 2 mm.
- **G3.** Zero discordant pairs among the saved valid evaluations of each campaign, plus Attempt 209.
- **Always report:** the long-side and end yaw separately, the fitted width and length, and the per-side offset profiles.

These tolerances are proposed and need the researcher's approval. They are not tuned on the pilot.

## Data and information access

- Raw scans `calibration/data/raw/run{6,9,10}.stl` and frozen targets `run/putong_convention_check/preserved_samples/run{6,9,10}.npz`, hash-checked.
- Saved valid evaluations (`prediction_mm.vtp` plus native state JSON):
  - `run/convention_corrected/run{6,9,10}` (Attempt 209)
  - `run/run1_clean_24lhs` (snapshot at analysis time)
  - the joint campaigns already scored with the shared-footprint metric
- Run 2 has already been seen; it is used here only as a placement check, not for validation.
- Read-only on all inputs. Outputs go to `run/diagnostics/scan_edge_anchor/`.

## Stopping rule

One pass over the listed scans and evaluations. If K2–K4 fail, stop and report. No iteration on edge-fit settings to change a conclusion. A sensitivity table (rank 1/5/20, side fraction 0.6/0.8/0.9) is reported alongside, not selected from.

## Compute plan

Solver-free. One `srun-here` CPU job per scan (≈5 min each, 4 CPU, 32 GB). No shell solves; no verifier needed for bounded CPU fixtures of this size.

## Code

- `calibration/analysis/scan_edge_anchor_check.py`: F1–F5 and K1–K3. The pilot currently runs Run 1 only and must be extended to Runs 2–3, K4 and the sensitivity table.
- `calibration/analysis/plot_scan_edge_anchor.py`: figure.

## Pilot results (Run 1, 2026-11-01; `run/diagnostics/scan_edge_anchor_pilot/`)

- Edge-derived centre (Δx, Δz) = (+0.000, −0.121) mm. Yaw −0.046°: long sides +0.003°, ends −0.096°.
- The fitted width is 199.46 mm and the length 298.98 mm. Both edges of each pair lie about 0.25–0.6 mm inside the nominal sheet, which is consistent with projection foreshortening plus the sampling rank.
- 10 saved valid Run 1 evaluations were rescored: Attempt 209 plus the 9 valid `run1_clean` evaluations to date. The largest |ΔJ| was 3.9 µm (largest relative change 0.58 %). There were 0 / 45 discordant pairs. A and B agree to 0.1 µm: the simulation's edge frame coincides with its Kabsch frame to within 0.02 mm and 0.006°.
- **G1–G3 pass for Run 1.** Runs 2 and 3 have not been checked. Run 2 had the largest residual preprocessing tilt (0.53°), so it is the case most likely to differ.
- Figure: `run/diagnostics/scan_edge_anchor_pilot/scan_edge_anchor_comparison.{pdf,png}` with a caption.

## Full-pass results (2026-11-01; Slurm 6755614, 18 min; `run/diagnostics/scan_edge_anchor/`)

**Known-answer checks**
- K1 PASS.
- K2 PASS for all three scans: the targets are reproduced exactly from the raw STL.
- K4 PASS. The synthetic scan is the simulated surface itself, displaced by (1.0, −1.5) mm and 0.4°. The edge fit recovered (0.986, −1.510) mm and 0.402°. J is 0.0014 mm under both A and B, against 0.0384 mm with the current anchor.
- K3: 1036 of 1037 case-evaluations reproduce production and the stored score. One record (`joint_stage2/arm_A` a0020, Run 3) differs from its stored score by 1.8 nm, which exceeds the 1e-6 mm tolerance. It is excluded from the gates and reported here.

**Edge frames** (centre offset; mean yaw; long-side / end yaw)

| Scan | Centre offset (mm) | Mean yaw | Long-side / end yaw |
|---|---|---|---|
| Run 1 | (+0.00, −0.12) | −0.046° | +0.003° / −0.096° |
| Run 2 | (−0.38, −0.17) | +0.028° | −0.33° / +0.38° |
| Run 3 | (−0.09, +0.19) | +0.052° | +0.015° / +0.089° |

- Across the 9 sensitivity settings, the centres vary by at most 0.34 mm and the yaws stay within ±0.16°.
- Run 2's outline is not rectangular in projection: its length is 292.6 mm and its sides are non-parallel.

**Score changes, variant A** (case-evaluations with J < 2 mm)

| Scan | n | Max \|ΔJ\| | Median ΔJ |
|---|---|---|---|
| Run 1 | 354 | 3.9 µm | +3.7 µm |
| Run 2 | 110 | 1.7 µm | −1.2 µm |
| Run 3 | 399 | 4.8 µm | +2.4 µm |

- The largest change at any J is 60 µm, on a 28.6 mm Run 2 fit.
- B matches A for good fits and differs only for badly deformed fits (J > 5 mm).

**Gates as written**
- G1 passes for all scans.
- G2 fails for Run 3: 171 fits at J = 0.16–0.28 mm change by more than 1 %. The largest relative change is 3.0 %, and the absolute change is ≤ 4.8 µm.
- G3 fails for every scan, with 1–4 discordant pairs per campaign. Every inversion is between candidates whose current scores differ by ≤ 1.14 µm.
- **Per the card's rule the outcome is "material"** (a gate failed), which would mean adopting the edge anchor for future campaigns. The failures are relative/near-tie effects of a few µm, not a placement error. Whether to act on the literal gate outcome is the researcher's decision; the gates were not relaxed.

Figure: `run/diagnostics/scan_edge_anchor/scan_edge_anchor_three_runs.{pdf,png}` with a caption.
