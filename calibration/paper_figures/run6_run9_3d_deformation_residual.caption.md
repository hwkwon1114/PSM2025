# 3D Spatial Comparison: Attempt 209 (global optimal expanded BO calibration on Run 6 and validated transfer to Run 9)

Top row: Run 6; bottom row: Run 9.
- **(a, d) 3D Isometric Residual Surface**: Deformed 3D point cloud colored continuously by signed height residual $\Delta z = z_{\text{model}} - z_{\text{scan}}$ (elev=25°, azim=-55°). Shows the 3D saddle/arch curvature and how errors distribute over the crest, flanks, and corners.
- **(b, e) 3D End-On Profile View**: Viewing down the longitudinal toolpath axis (elev=10°, azim=-15°). Highlights vertical arch conformity and transverse edge discrepancy.
- **(c, f) 3D Physical Shape Overlay**: Direct spatial overlay of experimental scan points (orange circles, #D55E00) and the model simulation surface (semi-transparent blue mesh, #0072B2), showing the true geometric agreement in 3D space.

## Shared Color Scale
- $\Delta z$ ranges symmetrically from -3.54 mm (blue, model lower) to +3.54 mm (red, model higher), with white indicating exact vertical agreement.

*Reproduction*: `python calibration/analysis/run6_run9_3d_comparison.py --attempt 209`.
