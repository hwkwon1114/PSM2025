# Zigzag initial design: 500 samples

This directory contains a deterministic, symmetry-reduced initial design for
zigzag English-wheel trajectories on the Putong panel (254 x 304.8 mm).

## Design composition

- 150 local paths: rotated bounding-box footprint below 5% of panel area
- 250 medium paths: footprint from 5% through 80%
- 100 near-full paths: footprint at or above 80%
- Strip counts from 3 through 14 are balanced across the 500 samples
- Centers use a first-quadrant representative of the panel's left/right and
  top/bottom reflection symmetry
- Membrane growth, signed curvature growth, band width, orientation, footprint,
  aspect ratio, and orthotropy are varied

`coverage_fraction` is the area of the rotated, band-expanded bounding box
divided by panel area. It is a scale descriptor, not the exact swept-area union.

## Files

- `zigzag_samples_500.csv`: flat table for analysis and visualization
- `zigzag_samples_500.jsonl`: one complete v1 solver configuration per line
- `zigzag_samples_500_summary.json`: bounds, category counts, and diagnostics
- `representative_*.json`: one directly runnable solver input per scale class
- `zigzag_samples_gallery.png`: decoded path examples
- `zigzag_parameter_coverage.png`: marginal design coverage
- `zigzag_scale_aspect_scatter.png`: scale/aspect coverage by class

## Reproduce

From the repository root:

```bash
python -m python.active_sampling.zigzag_action \
  --count 500 \
  --seed 20250820 \
  --output-dir notebooks/zigzag_action_outputs
```

The design is an initial candidate set; the full 500 simulations have not yet
been run. The three representative JSON files were accepted by the current C++
parser and completed coarse-mesh smoke simulations.
