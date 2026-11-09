# Exploratory Attempt 209 section shape-distance pilot

This is a preliminary code-based adaptation, **not a verified reproduction of the published method**, not a new calibration objective, and not evidence that the full surface or a design ranking is accurate. No solver or new registration was run.

## Source and method

Demisse, Aouada and Ottersten, *Deformation Based Curved Shape Representation*, IEEE TPAMI 40(6), 1338–1351 (2018), https://ieeexplore.ieee.org/document/7938364/. The paper PDF was inaccessible in this environment (the retrieved response was HTML), so the equations have not been checked against the full paper. The pilot translates the planar segment transformation in `Classes/Tools.m:Trans_Mat`, normalization in `Tools.m:normal_points`, open-curve representation and summation in `Classes/Curve.m:curve_rep` and `minus`, and SE distance in `Classes/Lie.m:geo_dist_SE`, from the authors' repository https://github.com/DemisseG/Curved_shape_representation at commit `44104ba0880ed9fb5f158ae303c4aa90ee21a169`. Unlike their demo, this fixes correspondence at equal physical x, does not optimize rotation/reflection, and does not perform dynamic-programming resampling. It uses 101 points and gives the authors' rotation and translation terms equal weights. Isotropic scale and translation are removed by centering and normalization of each entire (x,y) curve. This is not normalization of height alone. No original MATLAB comparison was available, and the SVD translation has not been independently verified against MATLAB.

## Input and sampling

`calibration/paper_figures/convention_corrected/calibration_transfer_sections.npz`, SHA-256 `e24806cfea84ee8135d0339840fbc48e88d8c31796ec49681217f15efedf6902`. Existing independent scan/model interpolants, no new smoothing or pose fitting; scan triangles with edges longer than 5 mm excluded. The three z sections were fixed before this analysis. For each pair use the longest contiguous jointly supported interval (388 original samples), then interpolate 101 equally spaced x values on that interval. No bridging of unsupported gaps. The 51-point sensitivity is nested on the same domain. Historical run6/run9/run10 correspond to public Runs 1/2/3 here, not the alternative numbering of the HW1 trend prototype. Run 1 was calibrated; Runs 2/3 were previously inspected and are exploratory transfers.

| Case | z (mm) | x support (mm) | distance, 101 points | distance, 51 points | height correlation | model/scan peak-to-valley |
|---|---:|---|---:|---:|---:|---:|
| Run 1 | −35 | −96.5 to 97 | 0.000327 | 0.000604 | 0.900 | 0.785 |
| Run 1 | 65 | −97 to 96.5 | 0.000290 | 0.000517 | 0.999 | 1.044 |
| Run 1 | 165 | −96.5 to 97 | 0.000364 | 0.000651 | 0.889 | 0.745 |
| Run 2 | −35 | −96.5 to 97 | 0.000353 | 0.000646 | 0.978 | 0.722 |
| Run 2 | 65 | −96.5 to 97 | 0.000695 | 0.001265 | 0.993 | 0.766 |
| Run 2 | 165 | −96.5 to 97 | 0.000968 | 0.001750 | 0.990 | 0.625 |
| Run 3 | −35 | −96.5 to 97 | 0.000561 | 0.001036 | 0.991 | 0.553 |
| Run 3 | 65 | −96.5 to 97 | 0.000883 | 0.001634 | 0.997 | 0.623 |
| Run 3 | 165 | −96.5 to 97 | 0.000837 | 0.001563 | 0.997 | 0.624 |

Correlations are descriptive spatial profile correlations, not independent replicate statistics. Peak-to-valley values are sampled; full values and mean height differences are in `scratch/curve_metric_pilot/results.json`. These numbers describe only the sections and chosen support. The similar-shaped Run 3 profiles are substantially lower amplitude; correlation alone hides this. Distance has no established practical threshold. Its approximately 1.8× change when sampling is reduced to 51 points makes the raw values unsuitable as an interpretable manuscript metric without a carefully justified fixed sampling convention and sensitivity study. Differences in support and the relative x/y scales also affect interpretation. The section comparisons alone cannot establish full-sheet agreement or suitability for toolpath ranking.

## Known-answer checks

On a synthetic parabola, identity = 0; translation = 5.41e−15; isotropic scaling = 0; rotation without alignment = 0.01088; inverted bending = 0.01362; doubling height alone = 0.00664; symmetry difference = 0. These test selected behaviors, not correctness against the original MATLAB code or the full paper. Rotation is deliberately not removed. No uncertainty or physical acceptance threshold is implied.

Run from project root: `/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python calibration/analysis/attempt209_curve_metric_pilot.py`. The result file is exploratory scratch output and can be regenerated. No figure was changed.
