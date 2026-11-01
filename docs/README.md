# Attempt 209 documentation

## Calibration and paper

- [Attempt 209 manuscript draft](calibration/bayesian_calibration/attempt209_manuscript_draft.tex) — methods, equations and results for transcription.
- [Current calibration record](calibration/bayesian_calibration/convention_corrected_investigation.md) — selected fixed-hardening fit, corrected forward evaluations, conventions and limitations.
- [Calibration reconstruction and verification](calibration/bayesian_calibration/calibration_draft_verification.md) — evidence for the historical search and selected parameters.
- [Attempt 209 publication figures](../calibration/paper_figures/attempt209_publication_clean/README.md) — seven figure sets, captions and reproduction instructions.

The original Run 1 search used a historical scoring convention; the corrected Run 1–3 shape figures use the revised shared-footprint metric. Runs 2 and 3 were previously inspected and are **exploratory transfers**, not blind validation. Public Run 1/2/3 labels map to historical identifiers via `calibration/case_labels.py`.

## Exploratory shape design

- [Attempt 209 double-curvature family](shape_design/attempt209_double_curvature.md) — sheet dimensions, toolpaths, solver checks, PCA and mesh-comparison limitations.
- [PCA and mesh-comparison figure](../run/shape_design/mesh_comparison_figures/family_pca_mesh_comparison.pdf) — descriptive model results; no physical validation is implied.

## Solver and mechanics

- [Bilayer zigzag workflow](shellsim/bilayer_zigzag_workflow.md) and [sequence format](shellsim/zigzag_sequence_format.md).
- [Native equilibrium solver](shellsim/solver.md) and [shell simulation usage](../shellsim/README.md).
- Shared solver settings: `shellsim/configs/default_adaptive_cutback.json`. New or changed scientific code requires the applicable pre-execution review.

## Cleanup record

On researcher approval, 95 older history, dome-study and optimization-diagnostics files were deleted. Their exact paths and pre-deletion hashes are retained in `run/document_cleanup/proposed_deletions.txt` and `run/document_cleanup/proposed_deletions.sha256`; this inventory is not a backup. The reorganization manifest `history/reorganization_2026-10-22_moves.json` remains. Existing research papers, manuscript drafts, solver documentation, code, data and result artifacts were excluded from that deletion.
