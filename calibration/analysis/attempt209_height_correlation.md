# Attempt 209: full supported-scan height correlation

Exploratory descriptive statistics at the frozen Attempt 209 parameters, using saved corrected-convention scans and model surfaces. At each retained scan point (x,z), model height y is interpolated on the native projected model triangle; points outside model support are excluded. Equal weight per scan point, not area weights. No new solve or registration. Public runs 1/2/3 are historical run6/run9/run10. Run 1 was used for calibration; Runs 2 and 3 were previously inspected and are exploratory, not blind validation. One specimen per case; spatial samples are not independent replicates.

| Case | Pearson r | Model / scan height SD | Vertical RMSE (mm) | Median absolute Δy (mm) | 95th percentile absolute Δy (mm) | Supported / total |
|---|---:|---:|---:|---:|---:|---:|
| Run 1 | 0.9365 | 0.956 | 0.632 | 0.402 | 1.187 | 14504 / 14504 |
| Run 2 | 0.9834 | 0.744 | 0.689 | 0.451 | 1.327 | 14468 / 14468 |
| Run 3 | 0.9844 | 0.648 | 0.726 | 0.576 | 1.392 | 14504 / 14504 |

Δy is model minus scan at matching in-plane positions. Height SD is the population SD over supported scan points, in the saved material-reference frame; the ratio compares spatial height variation, not peak-to-valley range. Vertical RMSE is not the manuscript's symmetric Euclidean surface RMS. Correlation is invariant to constant offsets and positive rescaling and does not measure magnitude agreement. No uncertainty interval or sufficiency threshold is asserted. Scan density affects these point-weighted statistics; they are not explicitly area-weighted. The existing figure `calibration/paper_figures/attempt209_publication/height_residuals.pdf` includes signed residual maps and empirical cumulative distributions of absolute vertical differences; it is not a histogram. These statistics and the figure use saved frame conventions; the near-zero average residual must not be interpreted as an independently estimated bias.

Reproduction: `srun-here --cpus 1 --mem 6G --time 00:05:00 python calibration/analysis/attempt209_height_correlation.py`. The script prints data hashes and all values, checks a known-answer positive rescale plus offset example and invokes existing interpolation fixtures. Executed successfully as Slurm 7371782; no figure regenerated.
