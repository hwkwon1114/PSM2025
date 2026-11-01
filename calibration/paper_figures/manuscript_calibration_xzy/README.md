# Current figure convention: x width, z length, y height

Use this directory for the current manuscript figure set. Prior draft directories retain historical axes and are not current deliverables.

- Sheet: x in [-100,100] mm, z in [-85,215] mm; y is height.
- Toolpaths: reference plane x–z; bottom clearance z in [-85,0]; right start at z=0, initial travel +z.
- Residual maps: x–z coordinates, Δy = Analytical minus Experiment.
- Cross-sections: x–y at fixed z = -35,65,165 mm. These correspond to the same three spatial slices as the earlier draft, reordered after the in-plane rotation.
- Overlay: x–y projection along longitudinal z; height exaggerated 12×, explicitly labelled.
- Saved field arrays carry coordinate_order=['x','z','y']; first two columns are planar coordinates, last is height. Height residual key is dy. Low-level interpolation retains its old variable names but figure labels/data metadata use the new convention.

## Scientific status — important
Deformation figures use existing saved predictions and existing accepted scan samples, transformed together by x=-old_x, y=old_z, z=65-old_y (mm). This retains sheet-edge alignment and does not rebin, refit, or rerun the solver. Geometric errors are invariant under this common rigid reorientation; it is not a test of a different physical trajectory.

Toolpath figure depicts the requested right-start, +z traversal. Rotating the historical trajectories moves their start to the opposite longitudinal endpoint and reverses travel direction; therefore the historical deformation results are NOT yet verified predictions of the newly specified start/direction. Do not infer equivalence for history-dependent simulation from sheet symmetry alone. No corrected-placement/direction solve has been launched.

## Checks
Regenerated using calibration/analysis/calibration_paper_figures.py and calibration/analysis/calibration_toolpaths.py. Input hashes checked unchanged by scripts. Rendered and inspected residual, section and overlay PDFs in check_* directories: correct x/z planar labels, y height labels, Δy residual labels; no observed clipping or overlap. Toolpath layout previously inspected with the same new convention. Objective history is coordinate-independent. No new uncertainty analysis or journal-specific certification.
