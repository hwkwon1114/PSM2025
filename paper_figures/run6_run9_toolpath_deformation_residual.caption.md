# Attempt 153 (optimal expanded BO calibration on Run 6 and validated transfer to Run 9)

Top row: Run 6; bottom row: Run 9.
- **(a, d) Toolpath and experimental surface**: Nominal CSV toolpath overlaid on experimental scan height ($z$ coordinate); stars indicate nominal start positions.
- **(b, e) Full XZ deformation projection**: Direct comparison of all experimental scan cells (orange circles, #D55E00) and model prediction vertices (blue crosses, #0072B2), not selected cross-sections.
- **(c, f) Signed height residual**: $\Delta z = z_{\text{model}} - z_{\text{scan}}$ evaluated at matching $XY$ positions via piecewise-linear interpolation on actual projected model facets without extrapolation. Red indicates model higher than experiment; blue indicates model lower; gray indicates unsupported regions outside the projected model footprint.

## Quantitative Residual Summary
- **Run 6** (14504 points, 14504 supported):
  - Height residual RMS: 0.6547 mm (mean: 0.0720 mm, 95th percentile $|\Delta z|$: 1.2318 mm)
  - BO symmetric surface distance metric: 1.185083 mm
- **Run 9** (14468 points, 14468 supported):
  - Height residual RMS: 1.0670 mm (mean: 0.0641 mm, 95th percentile $|\Delta z|$: 1.9835 mm)
  - Transfer symmetric surface distance metric: 1.378669 mm

## Parameters
- $g_{\text{top}}$: 0.0002707846657674896
- $g_{\text{bot}}$: 3.4596472597085075e-05
- $\text{ortho}_{\text{top}}$: -0.6245194974886215
- $\text{ortho}_{\text{bot}}$: 0.24900349954284007

*Reproduction*: `python calibration/analysis/run6_run9_six_panel.py --attempt 153`.
