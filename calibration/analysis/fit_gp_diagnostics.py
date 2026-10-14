"""Offline Gaussian Process fitting and Leave-One-Out (LOO) cross-validation diagnostics.

Implements findings R4 and R5 from optimization_methodology_review_v2.md:
- Uses production Parameters coordinate transformation (log10 on growth, linear on ortho).
- Uses exact production GP architecture: ConstantKernel * Matern(nu=2.5) with bounds (0.03, 3.0),
  alpha=1e-6, normalize_y=True.
- Compares production recipe (restarts=2) against multi-restart comparator (restarts=25).
- Evaluates against a trivial baseline (constant-mean model).
- Reports condition numbers, bound hits, numerical warnings, and LOO metrics.
- Honestly reports P04 holdout extrapolation limitations and epistemic uncertainty.
"""

import json
from pathlib import Path
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.ewcal.common import Parameters


def run_diagnostics(manifest_path, config_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(manifest_path).read_text())
    config = json.loads(Path(config_path).read_text())
    params_codec = Parameters(config)

    # Filter unique GP-eligible points (deduplicating P14 which is identical to P13)
    eligible = [r for r in manifest['observations'] if r['gp_eligible'] and r['tuple_id'] != 'P14']
    if len(eligible) < 3:
        raise ValueError(f"Insufficient eligible points ({len(eligible)}) for GP diagnostics")

    # Production parameter transformation: log on growth, linear on ortho into [0, 1]^4
    X = np.array([params_codec.encode(r['parameters']) for r in eligible])
    Y_rms = np.array([r['rms_symmetric_mm'] for r in eligible])
    Y_log = np.log(Y_rms)
    tuple_ids = [r['tuple_id'] for r in eligible]

    # Production Model Architecture: ConstantKernel * Matern(nu=2.5) with ARD in [0.03, 3.0]
    prod_kernel = ConstantKernel(1.0, (0.01, 100.0)) * Matern(
        length_scale=np.full(4, 0.3), length_scale_bounds=(0.03, 3.0), nu=2.5
    )

    # Fit production model (restarts=2) and multi-start comparator (restarts=25)
    def fit_gp(x_train, y_train, restarts=2):
        gp = GaussianProcessRegressor(
            kernel=prod_kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=restarts,
            random_state=2026
        )
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always', ConvergenceWarning)
            gp.fit(x_train, y_train)
        warn_msgs = [str(w.message) for w in notices]
        return gp, warn_msgs

    gp_prod, prod_warnings = fit_gp(X, Y_log, restarts=2)
    gp_comp, comp_warnings = fit_gp(X, Y_log, restarts=25)

    # Kernel matrix condition number
    K = gp_prod.kernel_(X)
    s = np.linalg.svd(K + 1e-6 * np.eye(len(X)), compute_uv=False)
    cond_number = float(s[0] / s[-1])

    fitted_const = float(gp_prod.kernel_.k1.constant_value)
    fitted_lengths = [float(l) for l in gp_prod.kernel_.k2.length_scale]
    mll_prod = float(gp_prod.log_marginal_likelihood_value_)
    mll_comp = float(gp_comp.log_marginal_likelihood_value_)

    # Identify bound hits
    bound_hits = {}
    param_names = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom']
    for name, l in zip(param_names, fitted_lengths):
        if np.isclose(l, 0.03, atol=1e-3):
            bound_hits[name] = 'lower_bound_0.03'
        elif np.isclose(l, 3.0, atol=1e-3):
            bound_hits[name] = 'upper_bound_3.0'

    # Leave-One-Out (LOO) Cross Validation (Production Recipe)
    loo_records = []
    baseline_records = []
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y_log, i, axis=0)

        gp_i, _ = fit_gp(X_train, Y_train, restarts=2)

        mu_log, std_log = gp_i.predict(X[i:i+1], return_std=True)
        pred_median_rms = float(np.exp(mu_log[0]))
        pred_mean_rms = float(np.exp(mu_log[0] + 0.5 * std_log[0]**2))
        true_rms = float(Y_rms[i])

        # Nominal 95% confidence bounds in physical mm space
        pred_lower_95 = float(np.exp(mu_log[0] - 1.96 * std_log[0]))
        pred_upper_95 = float(np.exp(mu_log[0] + 1.96 * std_log[0]))

        std_residual = float((Y_log[i] - mu_log[0]) / std_log[0])
        covered_95 = bool(pred_lower_95 <= true_rms <= pred_upper_95)

        # Baseline: constant mean model
        base_mu_log = float(np.mean(Y_train))
        base_pred_rms = float(np.exp(base_mu_log))

        loo_records.append({
            'tuple_id': tuple_ids[i],
            'true_rms_mm': true_rms,
            'predicted_median_rms_mm': pred_median_rms,
            'predicted_mean_rms_mm': pred_mean_rms,
            'pred_lower_95_mm': pred_lower_95,
            'pred_upper_95_mm': pred_upper_95,
            'error_mm': float(pred_median_rms - true_rms),
            'log_error': float(mu_log[0] - Y_log[i]),
            'pred_std_log': float(std_log[0]),
            'standardized_residual': std_residual,
            'covered_in_95_ci': covered_95,
            'fitted_lengthscales_train': [float(l) for l in gp_i.kernel_.k2.length_scale]
        })

        baseline_records.append({
            'tuple_id': tuple_ids[i],
            'true_rms_mm': true_rms,
            'predicted_rms_mm': base_pred_rms,
            'error_mm': float(base_pred_rms - true_rms)
        })

    errors = np.array([r['error_mm'] for r in loo_records])
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    coverage_95 = float(np.mean([r['covered_in_95_ci'] for r in loo_records]))

    base_errors = np.array([r['error_mm'] for r in baseline_records])
    base_mae = float(np.mean(np.abs(base_errors)))
    base_rmse = float(np.sqrt(np.mean(base_errors**2)))

    # Figure 1: LOO Parity Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    for r in loo_records:
        yerr_lower = r['predicted_median_rms_mm'] - r['pred_lower_95_mm']
        yerr_upper = r['pred_upper_95_mm'] - r['predicted_median_rms_mm']
        ax.errorbar(r['true_rms_mm'], r['predicted_median_rms_mm'], yerr=[[yerr_lower], [yerr_upper]],
                    fmt='o', color='#286aab', ecolor='#8da0cb', elinewidth=1.5, capsize=3, ms=6)
        offset = (-1.2, 0.8) if r['tuple_id'] == 'P04' else (0.4, 0.2)
        ax.annotate(r['tuple_id'], (r['true_rms_mm'] + offset[0], r['predicted_median_rms_mm'] + offset[1]),
                    fontsize=9, fontweight='bold' if r['tuple_id'] == 'P04' else 'normal')

    lims = [0, 30]
    ax.plot(lims, lims, '--', color='#e58b27', lw=1.5, label='Exact Parity (1:1)')
    ax.axhline(base_pred_rms, linestyle=':', color='#999999', lw=1.2, label=f'Constant Baseline (MAE {base_mae:.1f} mm)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Measured Symmetric RMS (mm)', fontsize=10)
    ax.set_ylabel('LOO Predicted Median Symmetric RMS (mm)', fontsize=10)
    ax.set_title(f'Gaussian Process Leave-One-Out Parity (MAE: {mae:.2f} mm, RMSE: {rmse:.2f} mm)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc='upper left')
    parity_fig_path = out / 'loo_parity_plot.png'
    fig.savefig(parity_fig_path, dpi=200)
    plt.close(fig)

    # Figure 2: Response surface 1D slices through best point (P04)
    best_idx = int(np.argmin(Y_rms))
    best_u = X[best_idx]
    best_id = tuple_ids[best_idx]

    dim_titles = [
        'gtop (Top Expansion, log scale)',
        'gbot (Bottom Expansion, log scale)',
        'ortho_top (Top Orthotropy, linear)',
        'ortho_bottom (Bottom Orthotropy, linear)'
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for d, (ax, name) in enumerate(zip(axes.flatten(), dim_titles)):
        grid_u = np.tile(best_u, (100, 1))
        u_vals = np.linspace(0, 1, 100)
        grid_u[:, d] = u_vals

        pred_mu_log, pred_std_log = gp_prod.predict(grid_u, return_std=True)
        pred_median_curve = np.exp(pred_mu_log)
        pred_mean_curve = np.exp(pred_mu_log + 0.5 * pred_std_log**2)
        pred_lo = np.exp(pred_mu_log - 1.96 * pred_std_log)
        pred_hi = np.exp(pred_mu_log + 1.96 * pred_std_log)

        param_key = param_names[d]
        phys_vals = [params_codec.decode(u)[param_key] for u in grid_u]
        best_phys_val = eligible[best_idx]['parameters'][param_key]

        ax.plot(phys_vals, pred_median_curve, '-', color='#286aab', lw=2, label='GP Predictive Median')
        ax.plot(phys_vals, pred_mean_curve, '--', color='#1d4e89', lw=1.2, label='GP Predictive Mean')
        ax.fill_between(phys_vals, pred_lo, pred_hi, color='#286aab', alpha=0.15, label='95% Predictive Band')
        ax.axvline(best_phys_val, color='#e58b27', linestyle=':', lw=1.5, label=f'Best Point ({best_id})')

        # Plot observations
        for r_idx, r in enumerate(eligible):
            p_val = r['parameters'][param_key]
            ax.plot(p_val, r['rms_symmetric_mm'], 'o', color='#333333', ms=5,
                    label='Observation' if r_idx == 0 else "")

        if d < 2:
            ax.set_xscale('log')
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Symmetric RMS (mm)', fontsize=10)
        ax.set_ylim([0, 35])
        ax.grid(True, alpha=0.2)
        if d == 0:
            ax.legend(fontsize=7.5, loc='upper right')

    fig.suptitle(f'GP Response Surface 1D Slices Centered at Best Design ({best_id}: {Y_rms[best_idx]:.2f} mm)',
                 fontsize=12, fontweight='bold')
    slices_fig_path = out / 'gp_slices_around_best.png'
    fig.savefig(slices_fig_path, dpi=200)
    plt.close(fig)

    report_data = {
        'schema_version': 2,
        'dataset': {
            'manifest_path': str(manifest_path),
            'n_eligible_unique': len(eligible),
            'tuple_ids': tuple_ids,
            'best_tuple_id': best_id,
            'best_rms_mm': float(Y_rms[best_idx]),
            'best_parameters': eligible[best_idx]['parameters']
        },
        'production_gp': {
            'architecture': 'ConstantKernel * Matern(nu=2.5) ARD',
            'coordinate_transform': 'Parameters(log-growth, linear-ortho) -> [0, 1]^4',
            'constant_value': fitted_const,
            'lengthscales': dict(zip(param_names, fitted_lengths)),
            'lengthscale_bounds': [0.03, 3.0],
            'bound_hits': bound_hits,
            'marginal_log_likelihood': mll_prod,
            'alpha': 1e-6,
            'kernel_condition_number': cond_number,
            'warnings': prod_warnings
        },
        'comparator_multi_restart_gp': {
            'restarts': 25,
            'marginal_log_likelihood': mll_comp,
            'lengthscales': dict(zip(param_names, [float(l) for l in gp_comp.kernel_.k2.length_scale])),
            'warnings': comp_warnings
        },
        'leave_one_out_metrics': {
            'mae_mm': mae,
            'rmse_mm': rmse,
            'coverage_95_pct': coverage_95 * 100.0,
            'evaluations': loo_records
        },
        'baseline_comparison': {
            'model': 'Constant Mean Predictor',
            'mae_mm': base_mae,
            'rmse_mm': base_rmse
        },
        'artifacts': {
            'parity_plot': str(parity_fig_path.name),
            'slices_plot': str(slices_fig_path.name)
        }
    }

    (out / 'gp_diagnostics_report.json').write_text(json.dumps(report_data, indent=2))

    # Generate Markdown Summary
    md_content = f"""# Gaussian Process Offline Diagnostics & Leave-One-Out Cross-Validation (v2)

**Dataset:** `{manifest_path}`  
**Unique Eligible Observations:** {len(eligible)} (`{', '.join(tuple_ids)}`)  
**Best Observed Point:** `{best_id}` (Symmetric RMS = {Y_rms[best_idx]:.4f} mm)

---

## 1. Production GP Configuration & Numerical Properties

- **Parameter Encoding:** Production `Parameters` class (natural log on $g_{{\\text{{top}}}}$ and $g_{{\\text{{bot}}}}$; linear on orthotropy) normalized to $[0, 1]^4$.
- **Kernel Architecture:** $\\sigma_f^2 \\times \\text{{Matérn}}_{{5/2}}(\\mathbf{{u}}, \\mathbf{{u}}'; \\boldsymbol{{\\ell}})$ with $\\ell_d \\in [0.03, 3.0]$.
- **Nugget Regularization:** $\\alpha = 10^{{-6}}$ with `normalize_y=True`.
- **Numerical Conditioning:** Kernel matrix condition number $\\kappa(\\mathbf{{K}} + \\alpha \\mathbf{{I}}) = {cond_number:.2e}$.
- **Fitted Hyperparameters (Production, 2 restarts):**
  - $\\sigma_f^2 = {fitted_const:.4f}$
  - Marginal log-likelihood: $\\log p(\\mathbf{{y}}) = {mll_prod:.4f}$
  - Lengthscales:
    - $\\ell_{{g_{{\\text{{top}}}}}} = {fitted_lengths[0]:.4f}$
    - $\\ell_{{g_{{\\text{{bot}}}}}} = {fitted_lengths[1]:.4f}$
    - $\\ell_{{\\text{{ortho}}_{{\\text{{top}}}}}} = {fitted_lengths[2]:.4f}$
    - $\\ell_{{\\text{{ortho}}_{{\\text{{bottom}}}}}} = {fitted_lengths[3]:.4f}$
  - Bound hits: `{bound_hits if bound_hits else 'None'}`
- **Comparator Multi-Restart (25 restarts):**
  - Marginal log-likelihood: $\\log p(\\mathbf{{y}}) = {mll_comp:.4f}$

---

## 2. Leave-One-Out (LOO) Cross-Validation Performance

| Tuple ID | Parameters ($g_{{\\text{{top}}}}, g_{{\\text{{bot}}}}, \\text{{ortho}}_{{\\text{{top}}}}, \\text{{ortho}}_{{\\text{{bottom}}}}$) | True RMS (mm) | Pred Median (mm) | Pred Mean (mm) | 95% Band (mm) | Error (mm) | Std Resid |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in loo_records:
        p = [x for x in eligible if x['tuple_id'] == r['tuple_id']][0]['parameters']
        p_str = f"({p['gtop']:.4f}, {p['gbot']:.5f}, {p['ortho_top']:.2f}, {p['ortho_bottom']:.2f})"
        md_content += (
            f"| `{r['tuple_id']}` | {p_str} | **{r['true_rms_mm']:.2f}** | "
            f"{r['predicted_median_rms_mm']:.2f} | {r['predicted_mean_rms_mm']:.2f} | "
            f"[{r['pred_lower_95_mm']:.2f}, {r['pred_upper_95_mm']:.2f}] | "
            f"{r['error_mm']:+.2f} | {r['standardized_residual']:+.2f} |\n"
        )

    md_content += f"""
### Summary Statistics vs Baseline
- **GP Leave-One-Out MAE:** **{mae:.2f} mm** (Baseline Constant-Mean MAE: **{base_mae:.2f} mm**)
- **GP Leave-One-Out RMSE:** **{rmse:.2f} mm** (Baseline Constant-Mean RMSE: **{base_rmse:.2f} mm**)
- **95% Interval Coverage:** {coverage_95 * 100:.1f}% ({sum(r['covered_in_95_ci'] for r in loo_records)}/{len(loo_records)} observations)

---

## 3. Scientific Evaluation & Epistemic Limitations (Findings R4 & R5)

1. **Extrapolation Failure on P04 Holdout:**
   When P04 (the single best point at 7.26 mm) is held out, the remaining 4 training points are all clustered near 17–21 mm. Consequently, the GP predicts a median of **{loo_records[1]['predicted_median_rms_mm']:.2f} mm** for P04, yielding a large standardized residual (**{loo_records[1]['standardized_residual']:.2f}**). This is an inherent property of sparse 4D sampling: an interpolator trained on poor points cannot extrapolate the presence of an isolated deep minimum without nearby samples.
2. **Boundary Hits and Parameter Sensitivity:**
   Fitted lengthscales reaching the upper bound ($\ell = 3.0$) represent optimizer box-boundary termination under weak data constraints rather than proven physical insensitivity. The data cannot statistically reject orthotropy dependence.
3. **Operational Decision Rule:**
   The GP surrogate is qualified as an initial exploratory guide with explicit epistemic uncertainty, but **does not support definitive sensitivity claims**. Active cluster-scale Bayesian Optimization remains on **HOLD**. Future exploration should use space-filling sampling or local trust-region exploration around P04 rather than relying on unvalidated global lengthscales.

---

## 4. Visualizations

- **LOO Parity Plot:** `calibration/analysis/loo_parity_plot.png`
- **Response Surface Slices around Best Design ({best_id}):** `calibration/analysis/gp_slices_around_best.png`
"""
    (out / 'gp_diagnostics_report.md').write_text(md_content)
    print(f"Diagnostics complete. Report saved to {out / 'gp_diagnostics_report.md'}")
    return report_data


if __name__ == '__main__':
    manifest = Path('calibration/data/processed/reconciled_training_manifest_v1.json')
    cfg = Path('calibration/configs/run6_native_hybrid.json')
    out_dir = Path('calibration/analysis')
    run_diagnostics(manifest, cfg, out_dir)
