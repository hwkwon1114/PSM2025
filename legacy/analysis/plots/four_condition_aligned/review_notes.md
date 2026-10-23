# Aligned stage views

Generated from unchanged retained four-condition data using `calibration/analysis/plot_four_conditions.py --analysis run/two_gripper_comparison/analysis_four_conditions --output calibration/analysis/plots/four_condition_aligned --align-stages` (compute job4476881).

Every stage is aligned independently to its case's clamped state with reference-area-weighted proper rigid least squares using all corresponding vertices. No scale, reflection, cropping of data or deformation. Clamped surface/pads remain in their original frame. Pin markers move with fitted vertices; they denote historical constrained DOFs, not new constraints in rotated axes. Common cameras/colour scales and inherited5x height exaggeration retained. Transform matrices/translations and RMS-to-clamped distances saved in provenance.json. The existing full-field pairwise contrasts are unchanged and use their own comparator fits, not this common stage-display transform.

Input hashes checked; rotations checked orthogonal with determinant+1. Run6/Run9 delivered PDF page1 rendered with existing DHD Poppler and visually inspected: labels, panels, colourbars visible, no evident clipping/overlap. No colour-vision simulation performed. Original PDFs and running BO files untouched.

RMS distance to clamped after alignment: Run6 released/free/never/final-only approximately0.7182mm; Run9 approximately0.5041mm. These quantify springback relative to the clamped state, not differences between final-condition pairs. Use four_condition_full_field.pdf for the latter. Source data are two deterministic cases, not independent mesh-vertex samples.
