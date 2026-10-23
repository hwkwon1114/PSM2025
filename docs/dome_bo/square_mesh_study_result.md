# Square-sheet mesh/tolerance convergence study: result (2026-10-22)

Protocol, criteria and review: `dome_toolpath_investigation.md` ("Square-sheet mesh/tolerance convergence study") and `square_mesh_study_pre_execution_review.md`. Analysis: `calibration/dome/mesh_convergence.py` on retained states (no new solves); numbers in `run/dome_square/mesh_study/mesh_convergence_result.json`, per-cell metrics in `run/dome_square/mesh_study/cell_metrics/`.

## Outcome

**Outcome (ii) of the frozen rules: no mesh pair meets C1-C3 through 1 mm. Reported as non-converged; no finer mesh was run; no mesh or tolerance is adopted.** All 18 cells (17 fresh solves plus the reused attempt 30 at 2 mm) finished `accepted` at gradient tolerance 1e-14; no solver failures.

| Pair | C1 ROI shape RMS <= 0.10 mm | C2 gate change <= 5 pp, no 80 % flip | C3 ranking (rho >= 0.9, same top) | All met |
|---|---|---|---|---|
| 4 -> 2 mm | fails: attempt 30 | fails: attempts 30, 32 | rho 0.66, top 31 -> 31: fails | no |
| 2 -> 1 mm | passes all six | fails: attempt 30 | rho 0.83, top 31 -> 31: fails | no |

## Connected qualifying fraction by mesh (frozen gate: 15 mm, 0.1 1/m floor, ratio 0.1; pass >= 80 %)

| Attempt | 4 mm | 2 mm | 1 mm | ROI shape RMS 4->2 / 2->1 (mm) |
|---|---|---|---|---|
| 30 | 0.902 | 0.440 | 0.674 | 0.573 / 0.002 |
| 31 | 0.908 | 0.893 | 0.895 | 0.007 / 0.001 |
| 21 | 0.719 | 0.707 | 0.710 | 0.004 / 0.001 |
| 32 | 0.318 | 0.247 | 0.276 | 0.009 / 0.003 |
| 27 | 0.640 | 0.648 | 0.652 | 0.003 / 0.002 |
| 19 (control) | 0.581 | 0.585 | 0.586 | 0.009 / 0.006 |

## What this supports and does not support

- For five of six candidates, shape is mesh-insensitive already at 4 mm (ROI RMS <= 0.009 mm to 2 mm, <= 0.006 mm 2 -> 1 mm) and the gate moves by <= 1.6 pp except attempt 32 (-7.1 pp, then +2.8 pp).
- Attempt 30, the candidate that triggered the study, is the only one that differs: 4 -> 2 mm changes the shape (0.57 mm ROI RMS, p99 1.27 mm), and the gate is non-monotone (0.90, 0.44, 0.67). Between 2 and 1 mm its shape agrees to 0.002 mm while the gate still moves 23 pp, so for that candidate the thresholded connected-fraction is highly sensitive to a very small shape change near the threshold. Whether the 4 -> 2 mm change is a different equilibrium (branch) or a resolution effect was not isolated.
- Attempt 30 never reaches 80 % at 2 or 1 mm; the screening pass at 4 mm (0.902) is not reproduced on any finer mesh. Attempt 31 (0.89) is the top candidate at every mesh but stays below 80 %, so no candidate passes the primary gate at 2 or 1 mm.
- Not shown: asymptotic convergence order (three meshes), tolerance invariance (tolerance fixed at 1e-14), branch uniqueness, any physical dome claim.

## Limits and carried review caveats

- 1 mm solves took 2208-4054 s (2 mm 171-787 s, 4 mm 34-38 s). A text search of the 1 mm proof histories found no time-cap termination; this was not a full structured inspection.
- Whole-sheet p99 shape distance and hit-count totals (review request) were not computed or saved; they are reported-not-gating.
- C3 has no tie or constant-ranking problem here (rho defined at both pairs).
- Calibration mesh resolution, estimator choice and the 1800 s -> 14400 s watchdog change for 1 mm remain unisolated effects. The study was designed after the attempt-30 discrepancy (not blinded).
- Identity check (`sha256sum -c checks/identity_manifest.sha256` from the project root, 2026-10-22): `calibration/dome/mesh_convergence.py` (edited after the review, used for the analysis; the reviewed version 7a6acfb66e0b was not retained) and `calibration/dome/run_mesh_study.sbatch` differ from the manifest. The solve-side files `mesh_study.py`, `square_attempt.py`, `whole_sheet_diagnostics.py` and the frozen binary match. The report and investigation-record hashes change with every edit.
- `scripts/run_dome_continuation.py` was edited after the solves (options and binary only; `verify()` unchanged), so its hash differs from `checks/identity_manifest.sha256`.

## Decision left to the researcher

Choose the working mesh/gate definition for any new search, or request further diagnostics. The evidence shows a screening-mesh (4 mm) optimum can be a mesh artefact for the specific candidate sitting near the gate; 2 mm agrees with 1 mm in shape for all six candidates but not in the gate for attempt 30. Options: (a) adopt 2 mm with a gate margin or a smoother gate statistic; (b) examine attempt 30 at 2 and 1 mm for branch versus estimator sensitivity (retained states only); (c) run a finer or tolerance-varied check. None was started or authorized.

## Final-state maps (added 2026-10-22; `calibration/dome/plot_mesh_study.py`, retained states, no solves)

Figures in `run/dome_square/mesh_study/figures/`: `mesh_study_attempt30_maps.png` (height, weaker-curvature and qualification maps at 4/2/1 mm, centre sections and differences to 1 mm), `mesh_study_attempt31_maps.png`, `mesh_study_candidates_height.png` (all six candidates; per-row colour scale).

- **Attempt 30 at 4 mm is a different dome orientation, not a small refinement.** Along the v = 0 section the 4 mm state drops 1.3 mm over half the sheet, whereas the 2 mm and 1 mm states drop 3.3 mm (the u = 0 section is the reverse). The dome is elongated along u at 4 mm and along v at 2 and 1 mm; the 2 mm and 1 mm sections agree to ~0.01 mm. This is the 0.57 mm ROI shape change at 4 -> 2 mm. The other five candidates keep the same orientation at all three meshes; their centre-section peak heights fall monotonically with refinement by <= 0.07 mm (read from the plots).
- **The 4 mm mesh does not resolve the treatment-line curvature bands.** At 2 and 1 mm the weaker-curvature map shows stripes of near-zero curvature between treated lines; at 4 mm the field is smooth. The connected qualifying region at 2 and 1 mm is cut by these stripes, so the gate fraction depends on whether narrow bridges between stripes stay above the 0.1 1/m floor. This is how attempt 30 can have almost identical 2 and 1 mm shape (0.002 mm RMS) but gate 0.44 versus 0.67.
- Not isolated: whether the orientation switch at 4 mm is a distinct equilibrium reached from the same flat start (branch) or a resolution-dependent bifurcation. The maps show the difference, not its cause.

### Metric sensitivity check (added 2026-10-22; retained states, `run/dome_square/mesh_study/checks/floor_scan.py`)

The shape converges but the gate statistic (largest connected qualifying area fraction) does not, because it is a threshold-and-connectivity statistic. Evidence:
- Attempt 30, gate fraction vs curvature floor (4 / 2 / 1 mm): floor 0.06 -> 0.92 / 0.73 / 0.73; 0.08 -> 0.91 / 0.70 / 0.71; **0.10 (frozen) -> 0.90 / 0.44 / 0.67**; 0.12 -> 0.89 / 0.34 / 0.34; 0.15 -> 0.86 / 0.31 / 0.31. At 2 and 1 mm the two meshes agree at every floor except the frozen 0.10, which sits on a cliff where narrow bridges between the stripe islands switch on or off. About 11 % of the ROI area has weaker curvature within 0.08-0.125 1/m at 2 and 1 mm (4 % at 4 mm).
- Attempt 31 shows the same cliff elsewhere: floor 0.12 -> 0.85 / 0.42 / 0.84 (4 / 2 / 1 mm).
- Curvature radius changes attempt 30's gate non-monotonically (10 / 15 / 20 mm at 2 mm: 0.60 / 0.44 / 0.85).
- Non-connected measures are stable at 2 -> 1 mm: total qualifying fraction (r = 15 mm) for attempt 30 is 0.67 / 0.67, and `j_combined` is 0.03 / 0.03; for the other five candidates both change by <= 0.01 (attempt 32 total fraction 0.59 / 0.58).
- Reading: the dome shape is mesh-converged at 2 mm for all six candidates in the sense of C1; the connected-fraction gate is not a continuous functional of the shape. The 4 mm orientation change of attempt 30 is a separate effect (see above). Not shown: that a different gate would rank candidates the same way, or that these measures are the scientifically right target; both would need a researcher decision.

### Curvature stripes follow the toolpath (added 2026-10-22; `checks/path_overlay.py`, `checks/footprint_fraction.py`, `figures/path_overlay.png`)

The near-zero-curvature islands at 2 and 1 mm lie on the treated strokes (attempt 30: bands where family-B strokes overlap with 6-8 hits; attempt 31: stripes between family-A strokes). The strokes are 8 mm wide at 6-16 mm pitch, so a mask "near the path" removes most of the ROI: share of the ROI within 4 / 2 / 1 mm of a stroke centre line (2 mm mesh) is 0.76 / 0.49 / 0.28 (attempt 30), 0.80 / 0.52 / 0.30 (31), 0.92 / 0.66 / 0.39 (21), 0.66 / 0.40 / 0.22 (32), 0.78 / 0.50 / 0.29 (27), 0.61 / 0.43 / 0.25 (19). Retained area would differ by candidate. No masked or filtered metric has been evaluated yet; the choice (mask vs larger curvature scale vs curvature of a low-order surface fit) is awaiting the researcher.

### Larger-scale curvature: radius scan and smooth-surface fit (added 2026-10-22; retained states, no solves)

Code: `legacy/dome_bo/radius_scan.py` (jobs 4945325; estimator radius 25 and 30 mm, 15/20 mm from the cell metrics) and `dome_bo/surface_fit_scan.py` (job 4945681; total-degree-d Legendre fit of x, y, z over material (u, v), area-weighted least squares, principal curvatures from the fitted fundamental forms; existing objective and gate unchanged). Outputs: `run/dome_square/mesh_study/radius_scan/`, `surface_fit/`. Selected after seeing the 15 mm result (not blinded).

Gate fraction / J at 4, 2, 1 mm:

| Attempt | surface fit d = 8 | surface fit d = 10 | radius 30 mm |
|---|---|---|---|
| 30 | 0.91 / 0.93 / 0.93 ; J .139 / .101 / .104 | 0.92 / 0.93 / 0.94 ; .139 / .101 / .104 | 0.88 / 0.94 / 0.94 ; .122 / .118 / .120 |
| 31 | 1.00 x3 ; .114 / .117 / .117 | 1.00 x3 ; .114 / .117 / .118 | 1.00 x3 ; .122 / .125 / .126 |
| 21 | 0.97 x3 ; .092 / .093 / .093 | 0.94 / 0.96 / 0.96 ; .086 / .087 / .088 | 0.96 / 0.97 / 0.97 ; .085 / .087 / .088 |
| 32 | 0.90 / 0.87 / 0.87 ; .080 / .076 / .075 | 0.88 / 0.87 / 0.87 ; .076 / .073 / .072 | 0.84 x3 ; .068 / .066 / .066 |
| 27 | 0.76 x3 ; .046 / .047 / .047 | 0.71 x3 ; .034 / .035 / .036 | 0.74 / 0.75 / 0.75 ; .055 / .056 / .056 |
| 19 (control) | 0.54 / 0.53 / 0.53 ; 0 | 0.54 x3 ; 0 | 0.52 x3 ; 0 |

- Fit residual RMS (all vertices) is 0.003-0.011 mm at d = 8-10 (0.016-0.038 mm at d = 4, which loses shape; the dome height is ~4 mm).
- At these scales 2 -> 1 mm changes gate by <= 0.02 and J by <= 0.005 for every candidate, and Spearman rho = 1.0 for gate and J (surface d = 8, 10; radius 25, 30 mm; 4 -> 2 mm rho 0.94-1.0). Attempt 31 is top at 2 and 1 mm. Attempt 30's J falls 4 -> 2 mm in the surface fit (.139 -> .101) but not at radius 30 (.122 -> .118), consistent with the 4 mm orientation change seen in the maps.
- The connected-fraction cliff is not removed in general: surface d = 12 gives attempt 32 gate 0.88 / 0.44 / 0.43 (total fraction 0.85 / 0.84 stable), and the radius-20 mm gate gives 0.76 / 0.38 / 0.38. Total fraction and J remained stable in all cases.
- Scale dependence: J and the gate level depend on the chosen scale (attempt 27 J: .087, .046, .034, .030 for d = 4, 8, 10, 12; attempt 30 versus 31 order flips at d = 4). Under these large-scale metrics attempts 30, 31, 21 and 32 exceed 0.80 on the 2 and 1 mm meshes (not under the 15 mm metric), so the 80 % threshold, set at 15 mm, no longer has the same meaning. No physical dome claim; whether stripe-scale curvature matters for the intended use is a researcher decision.
