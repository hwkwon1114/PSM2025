# Initial-sampling metric benchmark

These files define seven equal-budget, nested 500-sample initial designs. They
are solver-ready manifests, not completed simulations.

## Designs

| File prefix | Selection rule | Uses a trajectory distance? |
|---|---|---|
| `design_random` | Stratified random feasible sampling | No |
| `design_lhs` | Latin-hypercube coordinate stratification | No |
| `design_maximin_action_euclidean` | Farthest-first in standardized action coordinates | Yes |
| `design_maximin_d2_ordered_l2` | Farthest-first using ordered, reflection-invariant input distance | Yes |
| `design_maximin_d2_frechet` | Farthest-first using the D2 quotient of discrete Fréchet | Yes |
| `design_maximin_d2_dtw` | Farthest-first using the D2 quotient of constrained DTW | Yes |
| `design_spatial_information` | Greedy information gain from final top/bottom treatment-metric fields | No; uses the final spatial state |

Every design has 150 local, 250 medium, and 100 near-full paths, with strip
counts balanced from 3 through 14. The ordering keeps prefixes approximately
balanced, so the first 25, 50, 100, 200, and 500 records define nested learning
curves without requiring new simulations.

The maximin designs use the same independently generated 1,500-path feasible
reference ensemble. This isolates the effect of changing the distance. It does
not imply that maximin mathematically requires a finite pool; continuous or
on-demand proposal optimization can replace the reference ensemble later.

The D2 action contains identity, horizontal reflection, vertical reflection,
and 180-degree rotation. These operations transform spatial coordinates but do
not reverse path time. DTW is a useful dissimilarity but is not a mathematical
metric because it need not satisfy the triangle inequality.

## Diagnostics

- `initial_design_benchmark_summary.json`: balance, timing, and native-metric
  maximin diagnostics
- `maximin_insertion_diagnostics.csv`: nearest-earlier-point distance as the
  design grows
- `maximin_insertion_distances.png`: visualization of those insertion distances

Distances have different units and normalizations. Their numerical magnitudes
must not be compared across metric families; compare designs using downstream
neural-operator error on one frozen test set.

The spatial-information design rasterizes the final top and bottom target
metrics on a fixed material-coordinate grid, balances their six symmetric
tensor channels, constructs a positive-definite RBF kernel, and greedily
maximizes log-determinant information gain under the same scale/strip schedule
as every baseline. Intermediate equilibrium states and traversal timing are not
features. Band overlaps are accumulated into the final target metric. Geometric
coverage is still a raster approximation: each stored grid cell averages a
16-by-16 subgrid. The summary records a convergence diagnostic against a
32-by-32 subgrid on a fixed 150-candidate subset; this measures raster error but
does not replace a future comparison with simulator-emitted face metrics.

This is an input-space prior, not yet a learned response kernel. Once a pilot
set of simulated output-curvature fields exists, fit a response-aware kernel on
their reduced coefficients and compare its selections on the frozen test set.
The benchmark summary reports log-determinant spatial information gain under
the same reference-fitted channel scales, bandwidth, and noise level. Random,
maximin, and spatial-information designs share one reference candidate pool and
are directly comparable. LHS is generated independently, so its information
gain is explicitly labeled as an independent-pool diagnostic and must not be
used for an apples-to-apples selector ranking. No information-gain diagnostic
is a substitute for the downstream neural-operator learning curve.

## Reproduce

Use a Python environment containing NumPy, SciPy with `scipy.stats.qmc`, and
Matplotlib:

```bash
python -m python.active_sampling.initial_design_benchmark \
  --output-dir notebooks/initial_design_benchmark_outputs \
  --reference-size 1500 \
  --trajectory-points 16 \
  --seed 20260820
```

The environment used for the checked outputs was:

```bash
MPLCONFIGDIR=/tmp/psm-initial-design-mpl \
/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python \
  -m python.active_sampling.initial_design_benchmark \
  --output-dir notebooks/initial_design_benchmark_outputs \
  --reference-size 1500 --trajectory-points 16 --seed 20260820
```
