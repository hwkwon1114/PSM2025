# Square-sheet BO stage — result (goal `mvipfud5-fm48ba`)

Authority/protocol: `docs/reports/dome_toolpath_investigation.md` ("Square-sheet BO stage" and later sections). Reviews: `square_bo_pre_execution_review.md` (screening), `square_verification_pre_execution_review.md` (verification).

## Outcome
- **Search:** 32 of up to 96 attempts were run (2 anchors, 22 initial-design points, 8 BO points; job 4828522, 4 simultaneous 8-CPU solves). All 32 equilibria were accepted; there were zero failed/error solves, so feasibility learning saw no negatives. The search stopped by the frozen rule after the wave in which attempts 30 and 31 passed the screening gate. 64 attempts were not used; stopping early is the frozen policy, not a budget renewal.
- **Screening passes (4 mm mesh, 5e-14):** attempt 30 (90.22 % connected ROI area, J = 0.1297) and attempt 31 (90.81 %, J = 0.1082). Only 2 of 32 attempts reached 80 %; the median coverage was 29 %; anchors (uniform tiling) gave 2–3 %.
- **Verification of attempt 30 (selected on the frozen objective, before verification): NOT VERIFIED.** Fresh solve on the 2 mm mesh (20000 faces), gradient tolerance 1e-14, same recipe/binary/physics (job 4836328, accepted, 787 s, same binary sha 522633cf). Primary gate on the fine mesh (15 mm, 0.1 m^-1): **43.98 % connected area < 80 %** (screening 90.22 %).
- Attempt 31 was **not** verified (outside the single-candidate verification protocol; needs researcher direction).
- Conclusion: in this search domain a screening dome pass did not survive mesh refinement. The shape is still smooth and dome-like (≈4–5 mm plane-removed height range over a 200 mm sheet, no folds), but the fine mesh resolves stroke-scale curvature banding that drops the curvature below the 0.1 m^-1 floor in bands. This is a model-conditional, combined mesh/tolerance refinement check, not physical validation, not a proof of impossibility, and no global-optimum claim.

## Verification diagnostics (`run/dome_square/verification/attempt_030/comparison.json`, `maps.png`; diagnostics only, no re-tuning)
| radius / floor | screen connected % | fine connected % |
|---|---|---|
| 10 mm / 0.05 | 82.2 | 64.0 |
| 10 mm / 0.1 | 75.1 | 60.0 |
| 15 mm / 0.05 | 92.3 | 73.5 |
| **15 mm / 0.1 (primary)** | **90.2** | **44.0** |
| 15 mm / 0.2 | 72.0 | 19.8 |
| 20 mm / 0.05 | 92.2 | 92.7 |
| 20 mm / 0.1 | 89.4 | 84.6 |
| 20 mm / 0.2 | 74.6 | 31.9 |

- Shape (reference-area weighted, fine-mesh faces; coarse positions by barycentric interpolation on the coarse mesh's actual faces; area-weighted rigid Kabsch alignment; `whole_sheet_diagnostics.json`): ROI RMS 0.57 mm (p50 0.38, p90 0.99, p99 1.27, max 1.34 mm; 12800 faces); whole sheet RMS 0.86 mm (p50 0.57, p90 1.48, p99 1.91, max 2.04 mm; 20000 faces); plane-removed height range ≈4.0 mm (screen) vs 4.1 mm (fine) over the 200 mm sheet. This supersedes the earlier equal-weight 41x41 grid estimate (0.60 mm; sampled approximation).
- Treatment comparability (face metric converted to material basis C−I, nearest-centroid at 81² ROI points): means agree within ≈0.5 % (top), higher moments differ; per-component RMS difference 24–49 % of screening RMS (stroke-edge discretisation); total hit count identical at 83 % of points, max difference 6 hits. The same recipe thus gives a comparable but not identical treatment field on the two meshes.
- Whole-sheet opposite-sign area at 15 mm / 0.1: 20.4 % (screen) vs 13.5 % (fine). Map inspection: both meshes show one smooth central dome; the fine mesh gate loss is horizontal bands (and small saddle patches) where curvature is below the floor; edge saddles outside the ROI in both. At the 20 mm scale the fine mesh passes (84.6 %) — the curvature-scale sensitivity is the main discrepancy; the primary frozen gate decides, so no verified dome is claimed.
- Operational note: job 4836319 was cancelled by the agent after ~14 s (no result; directory preserved as `verification/attempt_030_cancelled_job4836319`). Analysis checks: `test/python/test_square_verification_analysis.py` (mesh-independent metric basis) and `test_dome_verification_ceiling.py` pass (5 tests); screen-vs-screen self-check reproduces 90.22 %.

## Whole-sheet inspection (retained outputs; `calibration/dome/whole_sheet_diagnostics.py`, `whole_sheet_diagnostics.json`, `whole_sheet_maps.png`, `unexaggerated_geometry.png`; the earlier `maps_superseded_incomplete_mask.png` omitted the weaker/stronger ratio and is superseded)
Qualification mask = `dome_continuous_objective.coherent_masks` (floor 0.1 m^-1 AND weaker/stronger >= 0.1; invalid faces counted). Area fractions at 15 mm / 0.1 (reference area):

| region | mesh | qualifying one sign | weak / not qualifying | saddle (floor and ratio) | invalid |
|---|---|---|---|---|---|
| whole sheet | 4 mm | 63.0 % | 16.5 % | 20.4 % | 0 |
| whole sheet | 2 mm | 55.8 % | 30.8 % | 13.5 % | 0 |
| central ROI | 4 mm | 90.2 % | 7.6 % | 2.2 % | 0 |
| central ROI | 2 mm | 66.8 % total same sign (44.0 % largest connected component) | 27.5 % | 5.7 % | 0 |

- Curvature distributions (area-weighted; |k_weaker| / |k_stronger| in m^-1) — ROI 4 mm: weaker p10/p50/p90 = 0.12/0.24/0.33, stronger 0.37/0.62/0.85, max 1.18; ROI 2 mm: weaker 0.035/0.20/0.49 (max 0.81), stronger 0.43/0.65/0.80, max 1.01; whole sheet stronger p99 1.70 (4 mm) and 1.58 (2 mm). On the fine mesh the weaker curvature is strongly modulated at the stroke pitch (stripes visible in `whole_sheet_maps.png`, right column): the dome sign is preserved everywhere in the ROI (negative-mean-curvature area 100 % in the ROI), but alternating bands of low weaker curvature fail the 0.1 floor and sever connectivity; saddle patches are small and located in those low-curvature bands plus the outer edge saddles (also present on the screening mesh).
- Fold/artifact checks on the actual faces (both meshes): maximum adjacent-face normal angle 0.49° (4 mm) and 0.28° (2 mm), 0 faces flipped relative to the mean normal, deformed/reference face-area ratio 0.9999–1.0010, so no fold, crease, isolated bump or inverted triangle; the shape is a smooth, very shallow dome (≈4 mm plane-removed height over 200 mm, i.e. 2 % of the span; the equal-axis views in `unexaggerated_geometry.png` are therefore nearly flat-looking, as expected for these curvatures of 0.2-0.6 m^-1).
- The fine-mesh disagreement is therefore a curvature-magnitude/connectivity-gate discrepancy under stroke-scale banding, not a geometric artifact. Whether the bands are physical (2 mm elements resolve 8 mm-wide strokes at 6-16 mm pitch; 4 mm elements smooth them) or discretisation-dependent was not separated further (no third mesh was authorized).
- Caveat: the common-point treatment comparison above uses a nearest-centroid 81x81 ROI grid (diagnostic); the shape comparison now uses reference-area weights.

## Decisions left to the researcher
1. Verify attempt 31 (second screening pass, whole-sheet opposite-sign 6 %) with the same frozen protocol, and/or
2. Resume BO with the fine mesh (or a mesh-converged, or larger-scale curvature) as the evaluation, or accept 20 mm scale; any of these changes the frozen policy and is not authorized here.

## All attempted candidates (ledger `run/dome_square/campaign/ledger.jsonl`; recipes in `.../attempts/attempt_NNN/request.json`)
| # | wave | source | state | ROI connected % (15mm/0.1) | qualified | J_combined | whole-sheet opposite-sign % | realized (pA,pB,SA,SB,LvA,LvB,rA,rB,e) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | anchor | accepted | 2.1 | False | 0.0000 | 0.0 | 8, 8, 192, 192, 192, 192, 3, 3, 1 |
| 2 | 0 | anchor | accepted | 2.8 | False | 0.0000 | 0.1 | 8, 8, 192, 192, 192, 192, 3, 3, 0.4 |
| 3 | 0 | design | accepted | 53.4 | False | 0.0017 | 47.3 | 10.5, 6.52, 103, 145, 107, 135, 6, 4, 0.523 |
| 4 | 0 | design | accepted | 5.7 | False | 0.0000 | 8.4 | 13.2, 15.9, 193, 183, 190, 180, 4, 1, 0.999 |
| 5 | 1 | design | accepted | 21.8 | False | 0.0000 | 31.0 | 14.3, 14.6, 93.6, 95.9, 155, 146, 1, 4, 0.999 |
| 6 | 1 | design | accepted | 49.3 | False | 0.0000 | 22.7 | 8.15, 13.4, 195, 142, 99.7, 98.8, 1, 4, 0.565 |
| 7 | 1 | design | accepted | 2.4 | False | 0.0000 | 7.4 | 6.88, 6.61, 187, 114, 189, 178, 5, 5, 0.93 |
| 8 | 1 | design | accepted | 56.0 | False | 0.0062 | 7.9 | 13.4, 10, 115, 198, 181, 179, 1, 2, 0.426 |
| 9 | 2 | design | accepted | 8.2 | False | 0.0000 | 3.5 | 7.2, 6.18, 130, 101, 99.4, 186, 1, 2, 0.964 |
| 10 | 2 | design | accepted | 2.2 | False | 0.0000 | 18.0 | 8.03, 15.5, 96.3, 194, 186, 124, 6, 5, 0.76 |
| 11 | 2 | design | accepted | 9.6 | False | 0.0000 | 20.1 | 15.1, 15.3, 189, 99.6, 96.8, 177, 4, 6, 0.529 |
| 12 | 2 | design | accepted | 46.5 | False | 0.0000 | 55.8 | 7.28, 13.9, 124, 106, 98.4, 108, 4, 1, 0.905 |
| 13 | 3 | design | accepted | 25.2 | False | 0.0000 | 24.2 | 8.38, 8.24, 192, 148, 163, 122, 5, 1, 0.442 |
| 14 | 3 | design | accepted | 44.5 | False | 0.0000 | 20.1 | 6.65, 11.8, 114, 102, 185, 183, 1, 6, 0.452 |
| 15 | 3 | design | accepted | 59.1 | False | 0.0141 | 19.0 | 10.6, 8.53, 189, 187, 182, 110, 1, 4, 0.952 |
| 16 | 3 | design | accepted | 20.8 | False | 0.0121 | 12.8 | 12.9, 15.6, 137, 180, 98.2, 172, 2, 4, 0.958 |
| 17 | 4 | design | accepted | 11.1 | False | 0.0000 | 14.3 | 12.1, 6.07, 93, 105, 191, 138, 5, 1, 0.711 |
| 18 | 4 | design | accepted | 42.4 | False | 0.0104 | 29.6 | 6.23, 8.18, 176, 180, 96, 162, 6, 2, 0.805 |
| 19 | 4 | design | accepted | 58.1 | False | 0.0000 | 51.2 | 15.1, 6.23, 98.8, 195, 173, 96.5, 1, 5, 0.647 |
| 20 | 4 | design | accepted | 33.3 | False | 0.0000 | 43.1 | 14.6, 16, 111, 152, 102, 127, 5, 1, 0.498 |
| 21 | 5 | design | accepted | 71.9 | False | 0.0485 | 18.3 | 6.1, 15.6, 154, 149, 167, 152, 1, 1, 0.528 |
| 22 | 5 | design | accepted | 15.5 | False | 0.0037 | 9.7 | 12.3, 15.4, 168, 116, 179, 99.5, 4, 2, 0.903 |
| 23 | 5 | design | accepted | 3.8 | False | 0.0000 | 11.3 | 12.7, 13.1, 186, 192, 177, 137, 6, 6, 0.465 |
| 24 | 5 | design | accepted | 57.7 | False | 0.0031 | 51.5 | 6.29, 6.37, 96, 154, 113, 106, 1, 3, 0.91 |
| 25 | 6 | bo | accepted | 11.9 | False | 0.0000 | 32.6 | 8.84, 15.7, 141, 102, 166, 110, 3, 4, 0.631 |
| 26 | 6 | bo | accepted | 27.0 | False | 0.0000 | 35.0 | 9.97, 15.1, 158, 98.8, 168, 117, 3, 5, 0.514 |
| 27 | 6 | bo | accepted | 64.0 | False | 0.0300 | 6.5 | 9.06, 14.7, 171, 96.3, 166, 159, 1, 5, 0.474 |
| 28 | 6 | bo | accepted | 16.3 | False | 0.0139 | 11.2 | 15.9, 14.8, 151, 96.7, 168, 188, 2, 1, 0.963 |
| 29 | 7 | bo | accepted | 36.7 | False | 0.0000 | 54.0 | 12.5, 9.62, 158, 171, 103, 101, 1, 6, 0.419 |
| 30 | 7 | bo | accepted | 90.2 | True | 0.1297 | 20.4 | 7.3, 12.3, 161, 192, 104, 124, 1, 3, 0.58 |
| 31 | 7 | bo | accepted | 90.8 | True | 0.1082 | 6.0 | 9.97, 10.9, 158, 182, 112, 168, 1, 1, 0.891 |
| 32 | 7 | bo | accepted | 31.8 | False | 0.0417 | 20.1 | 11.9, 15.5, 162, 194, 102, 146, 1, 5, 0.786 |
## Analysis-code identities (post-run diagnostics; edited after the Astra review, reviewed only as static additions, not as launch gates)
Retained outputs only; no solver rerun. `analyze_square_verification.py` (metric-basis fix + hit-count comparison, tested by `test_square_verification_analysis.py`), `whole_sheet_diagnostics.py` (primary whole-sheet/fold/shape diagnostics; supersedes the equal-weight grid shape number and `legacy/dome_bo/plot_square_verification.py`, whose mask omitted the ratio condition). Identity hashes of these post-run files are not recorded for the time of the report; the later study manifest `run/dome_square/mesh_study/checks/identity_manifest.sha256` hashes the current versions.
