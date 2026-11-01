# Gripper-side / toolpath-coordinate audit

## Active correction investigation
Researcher approved moving the fixture clearance to the bottom in current coordinates and correcting the affected transfer predictions. Authorized offsets: Run9 +28.75 mm, Run10 +22.5 mm; Run6 unchanged. Preserve old results and use separate outputs. Frozen Attempt-209 deformation parameters; no refit, optimizer search, face selection by error, or new clamp model. Scientific decision: determine whether bottom-margin placement changes transfer agreement once physical scan orientation is established. Two corrected forward predictions are supporting evidence, not fresh independent held-out validation after inspecting previous results.

Current blocker: scan-to-physical-edge correspondence is unverified. Both oriented metadata files explicitly require photo/toolpath checking; Run9 further says face/start corner are not independently identified from STL. Moving the path alone does not verify that correspondence. Obtain researcher-confirmed physical edge/start/face annotation before simulation. Then prepare separate corrected configs, verify unchanged parameters and evaluation settings against archived identities, complete independent Astra pre-execution review, run one frozen prediction per corrected case under existing per-case resource limits, and report matched geometric metrics plus spatial residuals. Stop on invalid solves or unresolved orientation; do not try alternative flips to minimize scan errors. No corrected solve has been launched.

Read-only scientific audit; no scan, recipe, parameter or result changed.

## Confirmed
- Current manuscript toolpath plots use saved sequence material offsets: Run6 0, Run9 -28.75 mm, Run10 -22.5 mm. Nominal longitudinal path bounds: Run6 [-65,65], Run9 [-122.5,65], Run10 [-110,65] mm. All leave 85 mm to the +150 mm edge; clearances to the -150 mm edge are 85,27.5,40 mm respectively.
- `src/libshell/ZigZagGrowth.hpp:700–701` uses material shift directly (pattern-frame shift would be rotated). Negative offsets are not just a plotting choice.
- Legacy `legacy/analysis/plot_toolpaths_on_sheet.py:53–61` explicitly distinguishes the physical bottom fixture datum from the scan frame, stating the fixture margin is at the TOP of the scan frame (+65 to +150). Later boundary-condition illustration code also places the fixture at positive y. Later proposed clamp layouts are NOT independent evidence of the original experiment.
- `legacy/analysis/flip_run9_face.py:33–57` applies (x,y,z)->(x,-y,-z), composed into scan basis and shift. Its provenance says the researcher requested the turnover but physical face/start corner are not independently identified from STL.
- `calibration/analysis/prepare_transfer_run.py` and `legacy/analysis/prepare_run10.py` apply the same x-turnover to Run10, followed by edge yaw. They use shift=(top-bottom)/2; their 'bottom clearance 85 mm' description is physically ambiguous because the final plotted/model positive-y clearance is 85 mm.
- Original frozen Run6/9/10 comparisons have disabled fixture constraints. Boundary-side interpretation still matters for asymmetric toolpath/scan registration; lack of a clamp does not eliminate placement error risk.

## Interpretation
There is a documented coordinate-frame explanation for the apparent reversal: a physical edge called bottom can appear at +150 after the y-sign-changing turnover. The new figure agrees with the saved simulation offsets; this is not an isolated plot-sign bug. However, the archive explicitly lacks independent physical face/start-corner identification. Therefore neither physical correctness nor a wrong validation simulation has been established.

If the gripped edge is definitively y=-150 IN THE FINAL SCAN/MODEL FRAME and has 85 mm longitudinal clearance, Run9/10 offsets should be +28.75/+22.5 mm rather than the saved negative values. If -150 refers to the pre-turnover frame, the saved negative offsets may be consistent. A global coordinate change must transform scan, model, toolpath and gripper together, including z under x-turnover; changing the toolpath alone is not a harmless relabeling.

## Needed decision/evidence
Researcher: specify whether y=-150 refers to experimental/pre-turnover or final plotted/model coordinates. Supply a photo, marked scan, or laboratory coordinate sketch linking gripped edge, first path segment/start corner, and worked face. Do not use lower residual as the criterion for selecting the physical orientation. Hold final validation claims until frame linkage is resolved. No rerun authorized or performed by this audit.
