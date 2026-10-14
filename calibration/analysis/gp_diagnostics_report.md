# Gaussian Process Offline Diagnostics & Leave-One-Out Cross-Validation (v2)

**Dataset:** `calibration/data/processed/reconciled_training_manifest_v1.json`  
**Unique Eligible Observations:** 5 (`P01, P04, P08, P10, P13`)  
**Best Observed Point:** `P04` (Symmetric RMS = 7.2575 mm)

---

## 1. Production GP Configuration & Numerical Properties

- **Parameter Encoding:** Production `Parameters` class (natural log on $g_{\text{top}}$ and $g_{\text{bot}}$; linear on orthotropy) normalized to $[0, 1]^4$.
- **Kernel Architecture:** $\sigma_f^2 \times \text{Matérn}_{5/2}(\mathbf{u}, \mathbf{u}'; \boldsymbol{\ell})$ with $\ell_d \in [0.03, 3.0]$.
- **Nugget Regularization:** $\alpha = 10^{-6}$ with `normalize_y=True`.
- **Numerical Conditioning:** Kernel matrix condition number $\kappa(\mathbf{K} + \alpha \mathbf{I}) = 9.45e+01$.
- **Fitted Hyperparameters (Production, 2 restarts):**
  - $\sigma_f^2 = 1.5604$
  - Marginal log-likelihood: $\log p(\mathbf{y}) = -5.8364$
  - Lengthscales:
    - $\ell_{g_{\text{top}}} = 0.2813$
    - $\ell_{g_{\text{bot}}} = 3.0000$
    - $\ell_{\text{ortho}_{\text{top}}} = 1.8973$
    - $\ell_{\text{ortho}_{\text{bottom}}} = 3.0000$
  - Bound hits: `{'gbot': 'upper_bound_3.0', 'ortho_bottom': 'upper_bound_3.0'}`
- **Comparator Multi-Restart (25 restarts):**
  - Marginal log-likelihood: $\log p(\mathbf{y}) = -5.8364$

---

## 2. Leave-One-Out (LOO) Cross-Validation Performance

| Tuple ID | Parameters ($g_{\text{top}}, g_{\text{bot}}, \text{ortho}_{\text{top}}, \text{ortho}_{\text{bottom}}$) | True RMS (mm) | Pred Median (mm) | Pred Mean (mm) | 95% Band (mm) | Error (mm) | Std Resid |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `P01` | (0.0030, 0.00010, -0.50, -0.50) | **17.27** | 19.72 | 19.83 | [16.04, 24.24] | +2.45 | -1.26 |
| `P04` | (0.0013, 0.00002, -0.49, -0.45) | **7.26** | 18.88 | 18.92 | [16.51, 21.59] | +11.62 | -13.99 |
| `P08` | (0.0055, 0.00013, 0.49, -0.48) | **18.35** | 21.81 | 22.90 | [11.85, 40.17] | +3.47 | -0.56 |
| `P10` | (0.0029, 0.00021, -0.96, 0.42) | **19.31** | 16.04 | 16.17 | [12.49, 20.59] | -3.27 | +1.45 |
| `P13` | (0.0039, 0.00002, -0.31, -0.78) | **20.78** | 7.46 | 7.54 | [5.55, 10.02] | -13.33 | +6.80 |

### Summary Statistics vs Baseline
- **GP Leave-One-Out MAE:** **6.83 mm** (Baseline Constant-Mean MAE: **5.57 mm**)
- **GP Leave-One-Out RMSE:** **8.26 mm** (Baseline Constant-Mean RMSE: **6.49 mm**)
- **95% Interval Coverage:** 60.0% (3/5 observations)

---

## 3. Scientific Evaluation & Epistemic Limitations (Findings R4 & R5)

1. **Extrapolation Failure on P04 Holdout:**
   When P04 (the single best point at 7.26 mm) is held out, the remaining 4 training points are all clustered near 17–21 mm. Consequently, the GP predicts a median of **18.88 mm** for P04, yielding a large standardized residual (**-13.99**). This is an inherent property of sparse 4D sampling: an interpolator trained on poor points cannot extrapolate the presence of an isolated deep minimum without nearby samples.
2. **Boundary Hits and Parameter Sensitivity:**
   Fitted lengthscales reaching the upper bound ($\ell = 3.0$) represent optimizer box-boundary termination under weak data constraints rather than proven physical insensitivity. The data cannot statistically reject orthotropy dependence.
3. **Operational Decision Rule & Protocol Status:**
   The GP diagnostic evidence is strictly retrospective and exploratory. It does not establish formal surrogate qualification or authorize active acquisition policies. Furthermore, the active offline protocol at verification_summary.md lines 900–910 specified an isotropic Matérn baseline comparator before flexible ARD, which was omitted in this initial retrospective run; this protocol discrepancy remains unresolved for future formal qualification. Active cluster-scale Bayesian Optimization remains strictly on **HOLD**. Future exploration should use space-filling sampling or local trust-region exploration around P04 rather than relying on unvalidated global lengthscales.

---

## 4. Visualizations

- **LOO Parity Plot:** `calibration/analysis/loo_parity_plot.png`
- **Response Surface Slices around Best Design (P04):** `calibration/analysis/gp_slices_around_best.png`
