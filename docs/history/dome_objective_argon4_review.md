# Argon4 Independent Scientific Review: Dome Bayesian Optimization Objective and Global-Cap Qualification

## Review

- **Correct**:
  - **Local geometry estimator (`calibration/analysis/dome_curvature.py:13-52`)**: `geometric_curvature()` uses full-precision native JSON vertex positions, fixed $15\,\text{mm}$ material-coordinate support neighborhoods (`RADIUS = 0.015`, minimum 12 vertices), mesh-winding-consistent SVD tangent charts, condition-number gating ($\sigma_0/\sigma_5 \le 10^6$), and a Cholesky-factored symmetric generalized eigenproblem $\mathbf{II}\,\mathbf{v} = \kappa\,\mathbf{I}\,\mathbf{v}$ rather than a raw coordinate height Hessian. Rigid-pose invariance to $10^{-8}\,\text{m}^{-1}$ and accuracy on analytic plane/sphere/cylinder/saddle fixtures are verified in `calibration/tests/test_dome_curvature.py:21-49`.
  - **Acceptance and false-positive safeguards (`calibration/analysis/dome_curvature.py:55-70`, `calibration/analysis/analyze_dome.py:16-58`, `calibration/analysis/dome_scale_diagnostic.py:1-36`)**: Separate coherent-sign accounting ($s \in \{-1, +1\}$ without pooling convex and concave regions), retention of invalid fits in the fixed ROI denominator ($|u| \le 80\,\text{mm}, |v| \le 120\,\text{mm}$), mandatory agreement with discrete director $(H,K)$ curvature (`analyze_dome.py:51-54`), multi-scale checks ($R \in \{10, 15, 20\}\,\text{mm}$), and unexaggerated 3D/spatial vetoes (`calibration/analysis/plot_dome_diagnostic.py`, `plot_dome_material_curvatures.py`, `plot_dome_principal_directions.py`) already prevent a saddle-split or folded surface from being falsely certified as a verified dome.
  - **Campaign isolation and recipe hygiene (`scripts/dome_continuation_campaign.py:25-46, 70-105`)**: The active continuation campaign (`run/dome/bo_continuation_stage1`) strictly isolates 10%-step adaptive continuation scores from historical single-step full-load scores (`run/dome/bo_stage1`, `run/dome/bo_stage2`, `run/dome/bo_stage3`).
- **Fixed**: None (read-only scientific review; no repository files, jobs, or running campaigns were modified).
- **Finding**: **P1 — BO scalar objective $C_{\max}$ is invariant to opposite-sign (saddle) severity and rewards side-band threshold crossings that simultaneously worsen middle/transition saddle bands**
  - **Location**: `calibration/analysis/dome_curvature.py:64-68`, `scripts/dome_bo_propose.py:35-38`, `scripts/dome_offset_propose.py:35-38`, `scripts/dome_continuation_campaign.py:43, 66`.
  - **Evidence**:
    1. In `dome_curvature.py:65-67`, each ROI triangle $f$ is scored by a binary indicator $I_f(s) = \mathbb{I}(s\kappa_{1,f} \ge 0.1)\,\mathbb{I}(s\kappa_{2,f} \ge 0.1)\,\mathbb{I}(\kappa_{\text{small},f} \ge 0.1\kappa_{\text{large},f})$. Any non-qualifying face receives **exact weight $0$**, whether it is nearly elliptic ($s\kappa_{\text{weak}} = +0.095\,\text{m}^{-1}$) or strongly anticlastic ($s\kappa_{\text{weak}} = -0.45\,\text{m}^{-1}$).
    2. In retained contrast data (`run/dome/baseline_analysis_v1/directions.log` vs. `run/dome/middle_repeat4_analysis_v1/directions.json`), increasing middle repeats from $2 \to 4$ increased primary coverage $C_{-1}$ from **$33.890625\%$ to $41.927083\%$ ($+8.04\,\text{pp}$)** by pushing weak $u$-curvature in the outer side regions ($|v| \ge 86.6\,\text{mm}$) over the $0.1\,\text{m}^{-1}$ step (side sub-threshold area dropped from $60.50\%$ to $17.85\%$), **even though it simultaneously drove opposite-sign saddle area up from $31.16\%$ to $42.95\%$ across the ROI** (middle $|v| \le 74\,\text{mm}$ saddle fraction jumped from $43.96\%$ to $59.30\%$; gap $74 < |v| < 86.6\,\text{mm}$ saddle fraction jumped from $37.81\%$ to $63.85\%$).
    3. In historical Stage 1 best `run/dome/bo_stage1/attempt_028` ($C_{-1} = 54.880208\%$, `run/dome/bo_stage1/attempt_028/analysis/metrics.json`, `run/dome/checks/bo_best_scale.json`, `calibration/analysis/plots/dome_best_material_curvatures/material_curvatures.pdf`), the $54.88\%$ qualifying area is split into three disconnected negative-curvature zones (central patch $|v| \lesssim 40\,\text{mm}$ and two outer side bands $|v| \in [85, 120]\,\text{mm}$) separated by two broad transverse saddle bands ($|v| \in [40, 80]\,\text{mm}$, $k_u < 0, k_v > 0$) covering **$36.0625\%$ of the ROI** at $15\,\text{mm}$ ($39.6979\%$ at $10\,\text{mm}$) vs. $31.1563\%$ in baseline (`run/dome/checks/scale_sensitivity_v1.json`). Lowering the magnitude floor from $0.1$ to $0.05\,\text{m}^{-1}$ increases `attempt_028` coverage by only $+1.85\,\text{pp}$ ($54.88\% \to 56.73\%$), proving that the binding barrier is the $36\%$ opposite-sign saddle band rather than sub-threshold magnitude.
  - **Smallest fix**: For any **next** researcher-authorized BO stage (leaving `run/dome/bo_continuation_stage1` untouched and keeping the frozen $C_{\max} \ge 0.80$ + multi-gate acceptance protocol unchanged), replace the raw binary step indicator inside the BO acquisition target with the bounded, continuous, saddle-penalized signed weak-curvature margin $J_{\lambda}$ defined in Section 5 below, while reporting largest connected qualifying central component $C_{\text{conn}}(s)$ and ROI saddle fraction $S_{\text{ROI}}$ in `analyze_dome.py`.
- **Finding**: **P2 — Hard anisotropy-ratio gate $\kappa_{\text{small}} \ge 0.1\,\kappa_{\text{large}}$ inside the binary BO score creates a cliff that penalizes strengthening orthogonal curvature when $\kappa_{\text{large}} > 1.0\,\text{m}^{-1}$**
  - **Location**: `calibration/analysis/dome_curvature.py:64-65`.
  - **Evidence**: When $\kappa_{\text{large},f} > 1.0\,\text{m}^{-1}$ (which occurs across the lateral middle-strip margins in `attempt_028`, where $|\kappa_{\text{strong}}|$ reaches $1.0\text{–}1.35\,\text{m}^{-1}$; see `calibration/analysis/plots/dome_best_principal_directions/principal_directions.png` panel (b)), the effective threshold on $|\kappa_{\text{weak},f}|$ rises above $0.1\,\text{m}^{-1}$ to $0.1\,\kappa_{\text{large},f}$. Under a binary indicator, a face with $|\kappa_{\text{weak}}| = 0.11\,\text{m}^{-1}$ and $|\kappa_{\text{strong}}| = 0.95\,\text{m}^{-1}$ scores $1$, whereas increasing $|\kappa_{\text{strong}}|$ to $1.15\,\text{m}^{-1}$ flips the face score from $1$ to $0$.
  - **Smallest fix**: Normalizing the continuous positive attainment by $\tau_f = \max(\kappa_{\min}, \rho_{\min}\kappa_{\text{large},f})$ (Section 5) eliminates the discontinuity while preserving the exact frozen qualification boundary $r_f^{(s)} = 1$.
- **Merge verdict**: **OK with notes** (current acceptance protocol is scientifically sound against false dome claims; the current continuation campaign `run/dome/bo_continuation_stage1` must complete or stop under its frozen contract without mid-flight edits; a minimal continuous saddle-penalized BO objective is justified **only** if a subsequent stage is authorized by the researcher).

---

## 1. Executive Verdict on the Fitness of the Current Objective

1. **As a Screening & Qualification Criterion (Preventing False Dome Claims): FIT.**
   The frozen acceptance protocol in `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` and `docs/reports/dome_toolpath_investigation.md:27-35`—requiring coherent-sign coverage $C_{\max} \ge 0.80$ over $|u| \le 80\,\text{mm}, |v| \le 120\,\text{mm}$ with $|\kappa_{\text{weak}}| \ge 0.1\,\text{m}^{-1}$ and $|\kappa_{\text{weak}}|/|\kappa_{\text{strong}}| \ge 0.1$, cross-checked against discrete director $(H,K)$ coverage (`calibration/analysis/analyze_dome.py:51-54`), multi-scale $10/15/20\,\text{mm}$ fits (`calibration/analysis/dome_scale_diagnostic.py`), and unexaggerated 3D/spatial curvature maps—**does not need threshold tuning and is not vulnerable to certifying a cylinder, saddle, or isolated fold as a verified dome**. Parent criticism should not be misread as proving inadequacy of the existing verification gates.

2. **As a Scalar Surrogate Objective for Bayesian Optimization in the Sub-Qualifying Regime ($C_{\max} \approx 5\%\text{–}55\%$): DEMONSTRABLY DEFICIENT.**
   In every BO campaign (`scripts/dome_bo_propose.py:35`, `scripts/dome_offset_propose.py:35`, `scripts/dome_continuation_campaign.py:43, 66`), the Gaussian Process (`SingleTaskGP`) is trained solely on the binary-thresholded area fraction $C_{\max} = \max(C_{-1}, C_{+1})$. Both mathematical structure and retained empirical records prove that in the $30\%\text{–}55\%$ coverage regime:
   - $C_{\max}$ **cannot distinguish** a single connected central cap from **three disconnected same-sign patches separated by broad anticlastic saddle bands** (as occurred in Stage 1 best `attempt_028` at $54.880208\%$).
   - $C_{\max}$ assigns **exact zero penalty** to the magnitude or area of opposite-sign (saddle) curvature within the non-qualifying $1 - C_{\max}$ fraction of the ROI. Consequently, interventions that push sub-threshold elliptic faces in the outer side bands ($|v| \ge 86.6\,\text{mm}$) just above $0.1\,\text{m}^{-1}$ are rewarded with large increases in $C_{\max}$ **even when they simultaneously worsen the anticlastic saddle bands ($k_v > 0, k_u < 0$) across the middle and middle-to-side transition zones** (`middle_repeat4_v1`, `middle_n18_repeat2_v1`, `bo_stage1/attempt_028`).

3. **Distinction Between Objective Limitations, Solver Failures, and Physical/Parameterization Limits:**
   - Improving the BO scalar objective **will not** circumvent native equilibrium solver failures (`time_cap` or `linear_solve_failed`) and **cannot** guarantee that a qualifying smooth dome exists within the frozen 9-parameter Attempt 209 zigzag family.
   - However, if a subsequent BO stage is authorized, replacing the step-function acquisition score with a **bounded continuous weak-curvature attainment minus saddle penalty** ($J_{\lambda}$, Section 5)—while keeping the frozen $80\%$ acceptance gate unchanged—is the **minimal, mathematically rigorous improvement** needed to stop BO from chasing disconnected side-band coverage at the expense of central/transition saddle bands.

---

## 2. Mathematical Analysis: Local Principal-Curvature Ellipticity vs. Global Cap, Material $(u,v)$ Curvatures, and Twist

### 2.1 Local Principal Curvatures vs. Fixed Material-Direction Normal Curvatures $(k_u, k_v)$ and Twist $\tau_{uv}$

Let $\mathbf{r}(u,v) \in \mathbb{R}^3$ denote the deformed midsurface over the reference sheet $(u,v) \in [-100, 100]\,\text{mm} \times [-150, 150]\,\text{mm}$. At each face $f$, `geometric_curvature()` (`calibration/analysis/dome_curvature.py:25-51`) constructs a local Euclidean tangent chart oriented consistently with the mesh normal $\mathbf{n}_f$ and computes the shape operator $\mathbf{S}_f = \mathbf{I}_f^{-1}\mathbf{II}_f$, with principal curvatures $\kappa_{1,f}, \kappa_{2,f}$ and orthonormal principal directions $\mathbf{e}_{1,f}, \mathbf{e}_{2,f}$.

When the material coordinate tangents $\mathbf{J}_f = [\mathbf{r}_u, \mathbf{r}_v]$ are projected into the fitted tangent plane (`calibration/analysis/plot_dome_material_curvatures.py:13-24`), Euler's theorem gives the signed normal curvatures along the deformed material $u$ and $v$ curves:
$$k_{u,f} = \frac{\kappa_{1,f}(\mathbf{e}_{1,f}\cdot\mathbf{r}_u)^2 + \kappa_{2,f}(\mathbf{e}_{2,f}\cdot\mathbf{r}_u)^2}{(\mathbf{e}_{1,f}\cdot\mathbf{r}_u)^2 + (\mathbf{e}_{2,f}\cdot\mathbf{r}_u)^2}, \qquad k_{v,f} = \frac{\kappa_{1,f}(\mathbf{e}_{1,f}\cdot\mathbf{r}_v)^2 + \kappa_{2,f}(\mathbf{e}_{2,f}\cdot\mathbf{r}_v)^2}{(\mathbf{e}_{1,f}\cdot\mathbf{r}_v)^2 + (\mathbf{e}_{2,f}\cdot\mathbf{r}_v)^2}.$$
In an orthonormal tangent frame $(\hat{\mathbf{e}}_u, \hat{\mathbf{e}}_{u_\perp})$ aligned with the material $u$-direction (where $k_{u_\perp} = k_v + \mathcal{O}(\gamma_{uv})$ for small in-plane membrane shear $\gamma_{uv}$), the shape operator takes the matrix form:
$$\mathbf{S}_{\text{mat},f} = \begin{pmatrix} k_{u,f} & \tau_{uv,f} \\ \tau_{uv,f} & k_{u_\perp,f} \end{pmatrix},$$
where $\tau_{uv,f} = (\kappa_{1,f} - \kappa_{2,f})\cos\theta_f\sin\theta_f$ is the geodesic torsion (surface twist) relative to the material axes. The mean and Gaussian curvatures satisfy:
$$H_f = \frac{\kappa_{1,f} + \kappa_{2,f}}{2} = \frac{k_{u,f} + k_{u_\perp,f}}{2}, \qquad K_f = \kappa_{1,f}\kappa_{2,f} = k_{u,f} k_{u_\perp,f} - \tau_{uv,f}^2.$$

From these exact identities, three critical consequences follow for objective design:

1. **Same-sign material normal curvatures ($s k_u > 0, s k_v > 0$) are necessary but NOT sufficient for local dome ellipticity:**
   Because $K_f = k_{u,f} k_{u_\perp,f} - \tau_{uv,f}^2 \le k_{u,f} k_{u_\perp,f}$, a surface with $s k_{u,f} > 0$ and $s k_{v,f} > 0$ is still a **saddle ($K_f < 0$)** whenever twist induced by oblique zigzag angles $\alpha$ satisfies $|\tau_{uv,f}| > \sqrt{k_{u,f} k_{u_\perp,f}}$.
2. **Signed weaker principal curvature $\kappa_{\text{weak},f}^{(s)} := \min(s\kappa_{1,f}, s\kappa_{2,f})$ is both necessary and sufficient to control $(k_u, k_v)$ AND twist $\tau_{uv}$:**
   Since $\kappa_{1,f}, \kappa_{2,f}$ are the extreme eigenvalues of $\mathbf{S}_{\text{mat},f}$, Rayleigh's quotient guarantees for *every* unit tangent direction $\hat{\mathbf{t}}$ (including $\hat{\mathbf{t}}_u$ and $\hat{\mathbf{t}}_v$):
   $$\kappa_{\text{weak},f}^{(s)} = \min(s\kappa_{1,f},\, s\kappa_{2,f}) = s H_f - \sqrt{H_f^2 - K_f} \le \min(s k_{u,f},\, s k_{v,f}).$$
   Any twist $\tau_{uv,f} \ne 0$ strictly increases $\sqrt{H_f^2 - K_f} = \sqrt{\frac{1}{4}(k_{u,f} - k_{u_\perp,f})^2 + \tau_{uv,f}^2}$ and therefore **strictly decreases $\kappa_{\text{weak},f}^{(s)}$**. Consequently, formulating the objective in terms of $\kappa_{\text{weak},f}^{(s)} = \min(s\kappa_{1,f}, s\kappa_{2,f})$ automatically penalizes twist and guarantees $s k_{u,f} \ge \kappa_{\text{weak},f}^{(s)}$ and $s k_{v,f} \ge \kappa_{\text{weak},f}^{(s)}$ without coordinate-axis bias.
3. **Why $(k_u, k_v)$ maps (`calibration/analysis/plots/dome_best_material_curvatures/material_curvatures.pdf`) are indispensable diagnostically:**
   In `bo_stage1/attempt_028`, inspecting `principal_directions.png` and `material_curvatures.png` together reveals that the stronger principal direction is aligned with material $u$ across almost the entire ROI ($k_u \approx -0.4\text{ to }-1.1\,\text{m}^{-1} < 0$), whereas the weaker principal direction is aligned with material $v$. In the bands $|v| \in [40, 80]\,\text{mm}$, $k_v$ flips sign to $+0.30\text{ to }+0.48\,\text{m}^{-1} > 0$, creating the two broad blue $K < 0$ saddle bands in `material_curvatures.png` panel (c). Thus, $(k_u, k_v)$ identifies *which* material bending direction reversed sign ($v$-bending at the longitudinal ends of middle toolpath A), while $\kappa_{\text{weak},f}^{(s)}$ remains the rigorous rotation- and twist-safe scalar quantity for optimization.

### 2.2 Local Ellipticity vs. a Smooth Global Cap

Why does maximizing the area fraction where $\kappa_{\text{weak},f}^{(s)} \ge \tau_f > 0$ fail to guarantee a single smooth global cap when coverage is partial ($C_{\max} < 1$)?

- **100% Coverage Implies a Global Convex/Concave Cap:**
  In a small-to-moderate slope Monge representation $z(u,v)$ over a convex region $\Omega_{\text{ROI}}$, $\kappa_{\text{weak}}^{(s)}(u,v) \ge \kappa_{\min} > 0$ everywhere on $\Omega_{\text{ROI}}$ implies that the Hessian $\nabla^2(s z)$ is uniformly positive-definite ($\nabla^2(s z) \succeq \kappa_{\min}\mathbf{I}_2$). A twice-differentiable function with uniformly positive-definite Hessian on a convex domain is strictly convex, possesses at most one interior extremum, and cannot have interior saddle bands or multiple domes.
- **Partial Coverage ($50\%\text{–}85\%$) Permits Disconnected Same-Sign Islands Separated by Saddle Troughs:**
  By Rolle's theorem along any material line $u = \text{const}$, if $s z(u,v)$ has two separate concave-down (or concave-up) lobes along $v$—or if differential eigenstrain between the middle toolpath A ($|v| \le L_{v,A}/2$) and the $90^\circ$-rotated side toolpaths B/C ($v \approx \pm d$) induces an inflection—the normal curvature $s k_v(u,v) \approx \partial_v^2(s z)$ **must change sign** ($s k_v < 0$, hence $\kappa_{\text{weak}}^{(s)} < 0$, a saddle band $K < 0$) in the transition zone between them.
  - Because the binary indicator $I_f(s) \in \{0, 1\}$ is purely additive over disconnected subsets of $\Omega_{\text{ROI}}$ and places **zero penalty on how negative $\kappa_{\text{weak}}^{(s)}$ becomes inside the transition zone**, a surface with **three disconnected negative-curvature zones** ($v \in [-120, -85]\,\text{mm}$, $v \in [-40, +40]\,\text{mm}$, and $v \in [+85, +120]\,\text{mm}$) separated by two intense saddle bands ($|v| \in [40, 80]\,\text{mm}$) achieves $C_{-1} = 54.880208\%$ in `bo_stage1/attempt_028` (`bo_best_shape_curvature.png`).
  - Even more critically, if an optimizer narrows those two saddle transition bands so that they occupy $\le 18\%$ of $\Omega_{\text{ROI}}$ while concentrating sharp opposite-sign bending ($s k_v \ll 0$) inside them, a multi-lobe corrugated sheet could in principle approach or exceed $80\%$ local indicator coverage while possessing severe anticlastic troughs between the middle and side regions.

---

## 3. Empirical Evidence: Objective Limitations vs. Solver Failures vs. Parameterization Limits

To avoid conflating distinct barriers, Table 1 separates what retained project records **demonstrate empirically** from **theoretical BO risks**, **numerical solver behavior**, and **toolpath parameterization constraints**.

| Retained Run / Stage | Solver Recipe | Coverage $C_{-1}$ ($15\,\text{mm}$) | ROI Saddle Area ($15\,\text{mm}$ / $10\,\text{mm}$) | Key Empirical Lesson |
|---|---|---:|---:|---|
| `baseline_v1` | Full-load (1 solve) | $33.8906\%$ | $31.1563\%$ / $37.3281\%$ | Weak side $u$-curvature ($60.5\%$ below $0.1\,\text{m}^{-1}$) and middle saddle ($44.0\%$). |
| `middle_repeat4_v1` | Full-load (1 solve) | $41.9271\%$ | $42.9531\%$ / — | **Smoking gun**: $+8.04\,\text{pp}$ higher $C_{-1}$ via side $u$-curling, while ROI saddle area worsened by $+11.80\,\text{pp}$ (middle saddle $44.0\% \to 59.3\%$, gap saddle $37.8\% \to 63.9\%$). |
| `middle_n18_repeat2_v1` | Full-load (1 solve) | $52.8177\%$ | $39.7448\%$ / $43.8229\%$ | $+18.93\,\text{pp}$ higher $C_{-1}$ than baseline, but ROI saddle area worsened by $+8.59\,\text{pp}$. |
| `middle_n18_repeat4_v1` | Full-load (1 solve) | $50.8750\%$ | $39.9688\%$ / $47.0781\%$ | Higher central dose degrades $C_{-1}$ and pushes $10\,\text{mm}$ saddle fraction to $47.08\%$. |
| `bo_stage1/attempt_028` | Full-load (1 solve, 6D BO best of 48) | $54.8802\%$ | $36.0625\%$ / $39.6979\%$ | Best 6D BO score ($+20.99\,\text{pp}$ over baseline), yet saddle area is still $+4.91\,\text{pp}$ worse than baseline; $0.05\,\text{m}^{-1}$ floor gives $56.7344\%$ ($+1.85\,\text{pp}$), proving $36.06\%$ saddle band is the binding wall. |
| `bo_stage3/attempt_028` (`repair_attempt028_ramp60_v1`) | 10%-step continuation + linear repair (9D) | $38.0260\%$ (failed full-load) | — | **Solver recipe non-equivalence**: Failed full-load (`linear_solve_failed`); accepted under 60-min continuation at $38.03\%$. Cannot be pooled with full-load scores. |
| `bo_continuation_stage1` (waves 0–9, attempts 001–020) | 10%-step continuation + linear repair (9D) | $5.2552\%\text{–}36.4427\%$ (seed $38.0260\%$) | — | Waves 0–11 (attempts 001–024) are **frozen initial LHS points**, not GP proposals (`scripts/dome_continuation_campaign.py:92-95`). 15 accepted, 3 failed equilibrium in attempts 001–018. |

### 3.1 Demonstrated Evidence vs. Theoretical BO Inefficiency Risk

1. **What is Demonstrated Empirically:**
   - **Saddle-blind ranking distortion is demonstrated fact, not theory.** Across `baseline_v1` $\to$ `middle_repeat4_v1` $\to$ `middle_n18_repeat2_v1` $\to$ `bo_stage1/attempt_028`, maximizing $C_{\max}$ consistently selected configurations with **higher ROI saddle-area fractions than the baseline** ($36.06\%\text{–}42.95\%$ vs. $31.16\%$ at $15\,\text{mm}$; `run/dome/checks/scale_sensitivity_v1.json`, `run/dome/checks/bo_best_scale.json`, `run/dome/middle_repeat4_analysis_v1/directions.json`).
   - **Why this happened physically and mathematically:** Because $C_{\max}$ counts any face with $\kappa_{\text{weak},f}^{(s)} \ge \tau_f$ as $+1$ and all other faces as $0$, increasing central treatment density/repeats bent the sheet more strongly along $u$, pulling the outer side regions ($|v| \ge 86.6\,\text{mm}$) from $0.085\,\text{m}^{-1}$ (just below the $0.1\,\text{m}^{-1}$ cliff) to $0.16\,\text{m}^{-1}$ (above the cliff). That side-region threshold crossing added $\sim 15\text{–}20\%$ of ROI area to $C_{-1}$, completely masking the simultaneous expansion and intensification of the $k_v > 0$ anticlastic saddle bands in $|v| \in [40, 80]\,\text{mm}$.
2. **What is Theoretical / Structural BO Risk (Not the Cause of Stage 3 or Current Continuation Stops):**
   - In 6D Stage 1 (`run/dome/bo_stage1/proposal_23.json`), `SingleTaskGP` with `LogExpectedImprovement` successfully located a reproducible high-$C_{-1}$ basin ($13$ of $32$ GP-proposed points exceeded $50\%$ coverage, peaking at $54.88\%$). Thus, step-function quantization noise across the 3840-triangle ROI did **not** prevent the GP from optimizing $C_{-1}$ in 6D; rather, **the optimum of $C_{-1}$ itself is a saddle-banded non-dome**.
   - In 9D Stage 3 (`run/dome/bo_stage3/proposal_13.json`), the campaign terminated after only 2 BO waves (attempts 025–028) due to the solver stopping rule (`two_full_waves_failed_equilibrium`). And in `run/dome/bo_continuation_stage1`, waves 0–11 (attempts 001–024) are predetermined initial design points (`continuation_lhs.json`), so **zero GP acquisition proposals have been executed yet in the continuation campaign**. Any claim that GP surrogate smoothing of the threshold objective caused the Stage 3 halt or the current continuation scores ($5.26\%\text{–}36.44\%$ on initial LHS points) would be factually incorrect.
   - However, once `bo_continuation_stage1` (or a successor stage) enters its GP acquisition phase in 9D (where repeat counts $1\text{–}30$ and side offsets $d \in [0, 118]\,\text{mm}$ create severe saddle configurations), training the GP on $C_{\max}$ alone with holds a **structural information-loss risk**: two designs with $C_{-1} = 30\%$—one having mild sub-threshold elliptic curvature ($k_v \approx -0.07\,\text{m}^{-1}$) across the transition zone and the other having violent anticlastic buckling ($k_v \approx +0.60\,\text{m}^{-1}$)—are treated as having identical objective value by the GP.

### 3.2 Separating Solver Failures and Parameterization Limitations from Objective Limitations

1. **Solver Non-Acceptance (`time_cap`, `linear_solve_failed`):**
   - Full-load solver failures in Stage 1 (8/48) and Stage 3 (11/28), and continuation failures in `bo_continuation_stage1` (3/18 in attempts 001–018: attempts 003, 007, 015), occur **inside the C++ equilibrium solver** (`bin/shell`) before `analyze_dome.py` or any BO objective is ever evaluated.
   - Changing the geometric post-processing objective has **zero effect** on whether a given toolpath sequence converges in `bin/shell`, though a saddle-penalizing objective may steer future BO proposals away from extreme incompatible repeat ratios (e.g., 1:30 or 29:3 with tiny $d$) that generate severe buckling/saddle bifurcations.
2. **Physical / Parameterization Limitation Under Fixed Attempt 209 Physics:**
   - Under Attempt 209 coefficients (`gtop = 2.8835e-4`, `gbot = 1.4588e-5`, `ortho_top = -0.3989`, `ortho_bottom = +0.9603`, `beta = 0.2231`), the middle zigzag A (rotation $0^\circ$, `center_peak` profile decaying to $0.4\times$ at its longitudinal ends $v = \pm L_{v,A}/2$) produces strong transverse curvature $k_u < 0$ across $|v| \le L_{v,A}/2$.
   - By Gauss's *Theorema Egregium*, an in-plane metric gradient where $\bar{a}_{uu}(v)$ decays sharply near $v = \pm L_{v,A}/2$ while $90^\circ$-rotated side bands B/C impose orthogonal eigenstrains at $v = \pm d$ induces an intrinsic target Gaussian curvature deficit $\Delta \bar{K} \propto -\frac{1}{2}\partial_v^2 \bar{a}_{uu} - \frac{1}{2}\partial_u^2 \bar{a}_{vv}$ that resolves elastically via **anticlastic transverse bending ($k_v > 0, k_u < 0$)** across the transition bands $|v| \in [40, 80]\,\text{mm}$ (`material_curvatures.png` panels (a–c)).
   - **Crucial boundary**: Even with an ideal continuous BO objective,Varying only the 9 geometric/repeat/offset parameters (while keeping `center_peak` on A, $0^\circ/90^\circ$ rotations, band widths $8/4\,\text{mm}$, and Attempt 209 scalars fixed) **may be physically incapable of eliminating those transition saddle bands to reach $80\%$ coherent-sign coverage**. A better BO objective guarantees that if a saddle-canceling overlap ($d$) and repeat balance exists within the 9D domain, BO is guided toward it; if none exists, it provides clean, unimpeachable evidence of the parameterization limit.

---

## 4. Critical Assessment of Parent's Suggestions and Candidate Objective Designs

The parent session suggested adding a **continuous curvature-deficit / saddle penalty** plus **global-cap diagnostics**, while **not** replacing local curvature checks with a global paraboloid fit. Below is an independent critical evaluation of each component and its potential pathological optima.

### 4.1 Global Paraboloid Fit vs. Topological / Spatial Global-Cap Diagnostics

1. **Why a Global Paraboloid Fit Must NOT Replace Local Checks (Endorsed):**
   Fitting a global quadratic height surface $z(u,v) \approx c_0 + c_1 u + c_2 v + \frac{1}{2}\bar{\kappa}_u u^2 + \bar{\kappa}_{uv} u v + \frac{1}{2}\bar{\kappa}_v v^2$ integrates curvature twice ($\nabla^2 z \approx \boldsymbol{\kappa}$), acting as a $k^{-2}$ low-pass filter on spatial wavenumber $k$. As `bo_best_shape_curvature.png` panel (a) illustrates, a shallow sheet with two $40\,\text{mm}$-wide anticlastic saddle bands can still exhibit a small global height residual against a shallow paraboloid, violating the authoritative goal requirement (`active_goal_2026101722293411_mvd9obky-2qqxhb.md`) to distinguish a smooth dome from a saddle or wavy sheet via spatial curvature measurements.
2. **Why a Global Paraboloid Fit Is Also Weak as a Primary Diagnostic (Critique of Parent Suggestion):**
   Even as a supplementary diagnostic, a constant-curvature paraboloid fit over $|u| \le 80\,\text{mm}, |v| \le 120\,\text{mm}$ conflates harmless smooth spatial variation in curvature magnitude (e.g., $\kappa_{\text{weak}}$ varying smoothly between $-0.15\,\text{m}^{-1}$ and $-0.45\,\text{m}^{-1}$ with uniform negative sign) with genuine sign reversals ($K < 0$).
3. **Superior Global-Cap Diagnostics Without Constant-Curvature Assumptions:**
   What actually distinguishes a single coherent global cap from Attempt 028's split 3-band pattern (`material_curvatures.png`) is **topological connectivity and saddle absence** on the mesh dual graph within $\Omega_{\text{ROI}}$:
   - **ROI Saddle-Area Fraction ($S_{\text{ROI}}$)**: Area fraction of $\Omega_{\text{ROI}}$ where $\text{valid}_f \wedge (\kappa_{1,f}\kappa_{2,f} < 0)$ (already computed in `dome_scale_diagnostic.py:29`, currently missing from `analyze_dome.py`).
   - **Largest Connected Qualifying Component Fraction ($C_{\text{conn}}(s)$)**: Area of the largest edge-connected component of qualifying faces in $\Omega_{\text{ROI}}$ (or largest component containing the central core $|u| \le 40\,\text{mm}, |v| \le 40\,\text{mm}$), divided by $A_{\text{ROI}}$.
     - In `bo_stage1/attempt_028`, $C_{-1} = 54.88\%$, but because the two $K < 0$ saddle bands at $|v| \in [40, 80]\,\text{mm}$ completely sever the central patch ($|v| \lesssim 40\,\text{mm}$) from the upper and lower side patches ($|v| \in [85, 120]\,\text{mm}$ across the full width $u \in [-80, 80]\,\text{mm}$; see `bo_best_shape_curvature.png` panel (c)), any single connected qualifying component covers **less than half** of $C_{-1}$ ($\lesssim 25\%$ of $\Omega_{\text{ROI}}$)!
     - By contrast, for any genuine single central dome with $C_{\max} \ge 0.80$, $C_{\text{conn}} \approx C_{\max}$.

### 4.2 Pathological Optima of Naive Continuous Objectives (And How to Prevent Them)

If a continuous objective is used for BO acquisition, naive formulations fail in four predictable ways:

1. **Pathology 1 — Unbounded Magnitude Exploit (Edge Curling Compensating Central Saddle):**
   If the objective averages raw signed weaker curvature $\kappa_{\text{weak},f}^{(s)}$ without an upper saturation cap, a design that produces $|\kappa_{\text{weak}}| = 0.65\,\text{m}^{-1}$ in the outer side bands (as in `bo_best_shape_curvature.png` panel (c) at $|v| > 100\,\text{mm}$) earns $6.5\times$ the credit of a $0.10\,\text{m}^{-1}$ dome face, allowing strong side curling to mask a large central saddle.
   - *Prevention*: **Clip positive attainment at $1.0$** once $\kappa_{\text{weak},f}^{(s)}$ reaches the required threshold $\tau_f = \max(\kappa_{\min}, \rho_{\min}\kappa_{\text{large},f})$. Faces above threshold cannot earn extra credit to subsidize non-qualifying regions.
2. **Pathology 2 — Unbounded Negative Outlier Dominance in GP Regression:**
   If positive attainment is clipped at $+1$ while opposite-sign (saddle) curvature $\kappa_{\text{weak},f}^{(s)} < 0$ is penalized linearly without a lower bound, an extreme localized crease ($\kappa_{\text{weak},f}^{(s)} = -2.5\,\text{m}^{-1}$, i.e., $-25\kappa_{\min}$) on $5\%$ of the sheet produces a huge negative spike that dominates the `SingleTaskGP` marginal likelihood and flattens lengthscales across the moderate-curvature regime.
   - *Prevention*: **Bound the per-face saddle penalty in $[-\lambda, 0]$** via smooth rational saturation or clipping at a fixed curvature scale $\kappa_{\text{sad}}$, keeping the total objective strictly bounded in $[-\lambda, 1]$.
3. **Pathology 3 — Confounding Saddle Penalty with $\kappa_{\text{large},f}$:**
   If the negative branch ($\kappa_{\text{weak},f}^{(s)} < 0$) is divided by $\tau_f = \max(\kappa_{\min}, \rho_{\min}\kappa_{\text{large},f})$ rather than the fixed floor $\kappa_{\min} = 0.1\,\text{m}^{-1}$, then on a saddle face where $s\kappa_1 < 0$ and $s\kappa_2 > 1.0\,\text{m}^{-1}$, increasing the orthogonal bending $s\kappa_2$ increases $\tau_f$ and *reduces* the apparent saddle penalty!
   - *Prevention*: Normalize the positive branch ($\kappa_{\text{weak},f}^{(s)} \ge 0$) by $\tau_f = \max(\kappa_{\min}, \rho_{\min}\kappa_{\text{large},f})$ (so reaching $+1$ is **mathematically equivalent** to passing both frozen thresholds), but normalize the negative branch ($\kappa_{\text{weak},f}^{(s)} < 0$) by the **fixed scale $\kappa_{\min} = 0.1\,\text{m}^{-1}$** (or fixed $\kappa_{\text{sad}}$) so the penalty depends purely on the wrong-sign curvature magnitude.
4. **Pathology 4 — Per-Face Sign Flipping vs. Global Sign Coherence:**
   Maximizing $|\kappa_{\text{weak},f}|$ or choosing $s_f \in \{-1, +1\}$ per face would reward a half-convex, half-concave S-wave.
   - *Prevention*: Evaluate the ROI integral $J_{\lambda}(s)$ separately for global sign $s = -1$ and global sign $s = +1$, and take $J_{\lambda,\max} = \max(J_{\lambda}(-1), J_{\lambda}(+1))$—identical to the global sign rule in `dome_curvature.py:66-68`.

---

## 5. Concrete Minimal Alternative BO Objective Equation (If Next Stage Is Authorized)

### 5.1 Exact Mathematical Formulation

Preserve `geometric_curvature()` (`calibration/analysis/dome_curvature.py:13-52`), the frozen ROI $\Omega_{\text{ROI}} = \{f : |u_f| \le 0.08\,\text{m},\; |v_f| \le 0.12\,\text{m}\}$, reference triangle areas $A_f$, total ROI area $A_{\text{ROI}} = \sum_{f \in \Omega_{\text{ROI}}} A_f$, magnitude floor $\kappa_{\min} = 0.1\,\text{m}^{-1}$, and anisotropy ratio floor $\rho_{\min} = 0.1$ **without modification**.

For each face $f \in \Omega_{\text{ROI}}$ and each global sign $s \in \{-1, +1\}$, define:
1. **Signed weaker principal curvature** for global sign $s$:
   $$\kappa_{\text{weak},f}^{(s)} = \min\!\left(s\kappa_{1,f},\; s\kappa_{2,f}\right), \qquad \kappa_{\text{large},f} = \max\!\left(|\kappa_{1,f}|,\; |\kappa_{2,f}|\right).$$
2. **Required per-face target floor** (combining the frozen magnitude and anisotropy-ratio gates):
   $$\tau_f = \max\!\left(\kappa_{\min},\; \rho_{\min}\,\kappa_{\text{large},f}\right) \ge 0.1\,\text{m}^{-1}.$$
   *(Note the exact equivalence: $\kappa_{\text{weak},f}^{(s)} \ge \tau_f \iff s\kappa_{1,f} \ge \kappa_{\min} \;\wedge\; s\kappa_{2,f} \ge \kappa_{\min} \;\wedge\; \min(|\kappa_{1,f}|, |\kappa_{2,f}|) \ge \rho_{\min}\kappa_{\text{large},f}$, i.e., `eligible & (sign*values > 0).all()` in `dome_curvature.py:65-67`.)*
3. **Bounded continuous per-face score $u_f^{(s)} \in [-\lambda, 1]$**:
   $$u_f^{(s)} = \begin{cases}
   \min\!\left(\dfrac{\kappa_{\text{weak},f}^{(s)}}{\tau_f},\; 1\right), & \text{if } \text{valid}_f \text{ and } \kappa_{\text{weak},f}^{(s)} \ge 0, \\[10pt]
   -\lambda\,\dfrac{\left|\kappa_{\text{weak},f}^{(s)}\right|}{\kappa_{\text{sad}} + \left|\kappa_{\text{weak},f}^{(s)}\right|}, & \text{if } \text{valid}_f \text{ and } \kappa_{\text{weak},f}^{(s)} < 0, \\[10pt]
   0, & \text{if } \neg\text{valid}_f,
   \end{cases}$$
   where $\lambda \ge 0$ is the saddle-penalty weight (recommended default $\lambda = 0.5$) and $\kappa_{\text{sad}} > 0$ is the saddle half-saturation scale (recommended default $\kappa_{\text{sad}} = \kappa_{\min} = 0.1\,\text{m}^{-1}$, so a wrong-sign curvature of $-0.1\,\text{m}^{-1}$ receives half-maximal penalty $-\lambda/2$, and $-0.4\,\text{m}^{-1}$ as in Attempt 028's saddle bands receives $-0.8\lambda$).
4. **Global coherent-sign BO acquisition objective**:
   $$J_{\lambda}(s) = \frac{1}{A_{\text{ROI}}} \sum_{f \in \Omega_{\text{ROI}}} A_f\, u_f^{(s)}, \qquad J_{\lambda,\max} = \max_{s \in \{-1, +1\}} J_{\lambda}(s) \in [-\lambda, 1].$$

### 5.2 Mathematical and Practical Properties of $J_{\lambda,\max}$

- **Exact consistency with the frozen acceptance criterion**:
  - For every face $f$, $u_f^{(s)} \le 1$, with equality $u_f^{(s)} = 1$ **if and only if** face $f$ passes the frozen binary qualification test in `dome_curvature.py:65-67`.
  - Moreover, $u_f^{(s)} \ge \mathbb{I}(\kappa_{\text{weak},f}^{(s)} \ge \tau_f) - \lambda\,\mathbb{I}(\kappa_{\text{weak},f}^{(s)} < 0)$, so a 100% qualifying dome achieves the unique global maximum $J_{\lambda,\max} = 1.0$, and any design with $C_{\max} \ge 0.80$ and zero saddle area satisfies $J_{\lambda,\max} \ge 0.80$.
- **Resolves the `middle_repeat4` / `attempt_028` ranking pathology**:
  - Under $J_{\lambda,\max}$ (with $\lambda = 0.5, \kappa_{\text{sad}} = 0.1\,\text{m}^{-1}$), a sub-threshold elliptic face with $\kappa_{\text{weak},f}^{(-1)} = 0.08\,\text{m}^{-1}$ and $\tau_f = 0.1\,\text{m}^{-1}$ contributes $+0.80$ (instead of $0$), whereas a saddle face in Attempt 028 with wrong-sign curvature $k_v = +0.40\,\text{m}^{-1}$ ($\kappa_{\text{weak},f}^{(-1)} = -0.40\,\text{m}^{-1}$) contributes $-0.5 \times \frac{0.40}{0.10 + 0.40} = -0.40$ (instead of $0$).
  - Consequently, an intervention like `middle_repeat4_v1` that pushes side-region weak curvature from $0.085 \to 0.159\,\text{m}^{-1}$ (a tiny continuous gain from $+0.85 \to +1.00$ on side faces) while expanding opposite-sign saddle area across the middle/gaps by $+11.80\,\text{pp}$ (turning positive/near-zero faces into $-0.3\text{ to }-0.4$ penalties) is **heavily penalized rather than rewarded**.
- **Minimal footprint and zero new simulation cost**:
  - Requires ~12 lines in `calibration/analysis/dome_curvature.py` and `calibration/analysis/analyze_dome.py` to output `continuous_score`, `saddle_area_fraction`, and `largest_connected_qualifying_fraction` alongside the unchanged `geometry` and `director` coverage dictionaries.
  - Because `curvature_fields.npz` and `bilayer_zigzag_sequence_native_hybrid_state.json` are already saved for every accepted historical and continuation run, $J_{\lambda,\max}$ can be computed deterministically for all retained same-recipe runs without re-running a single C++ shell solve.

---

## 6. Limitations, Negative Outcomes, and Researcher Decision Points

### 6.1 Strict Boundaries and Non-Pooling Rules

1. **Do NOT modify `run/dome/bo_continuation_stage1` while running:**
   The active continuation campaign (`job 4467951`, `run/dome/bo_continuation_stage1`, currently at initial-design wave 9, attempts 001–020) enforces SHA256 identity checks on `calibration/analysis/dome_curvature.py`, `calibration/analysis/analyze_dome.py`, `scripts/dome_offset_propose.py`, and `scripts/dome_continuation_campaign.py` before every wave (`scripts/dome_continuation_campaign.py:72-86`). Editing any of those files while the job is active would trigger `RuntimeError('Campaign source identity changed; stop without retry')` and abort the campaign.
2. **Do NOT pool historical full-load scores (`54.88%`) with continuation scores (`38.03%`):**
   As demonstrated by `bo_stage3/attempt_028` (failed full-load `linear_solve_failed`; converged under 60-minute 10%-step continuation to $38.0260\%$ in `run/dome/repair_attempt028_ramp60_analysis_v1/metrics.json`), the single-step full-load recipe and the 10%-step adaptive continuation recipe with linear-solve repair have different convergence domains and may select different equilibrium branches on strongly nonlinear shells. Training a GP on mixed-recipe observations remains scientifically invalid.
3. **Preserve the frozen acceptance threshold ($C_{\max} \ge 0.80$) and history:**
   Even if $J_{\lambda,\max}$ is adopted as the GP acquisition score in a future stage, **screening qualification must still require $C_{\max} \ge 0.80$** under `dome_curvature.coverage()`, agreement with discrete director coverage, multi-scale ($10/15/20\,\text{mm}$) checks, and visual/spatial verification (`C_conn` and $(k_u, k_v, K)$ maps). Replacing the BO acquisition objective does **not** relax dome qualification.
4. **Possibility of a Negative Scientific Outcome:**
   If a continuation stage using $J_{\lambda,\max}$ reduces the ROI saddle-area fraction $S_{\text{ROI}}$ (e.g., from $36\%$ toward $10\%\text{–}15\%$) by moving side offsets $d$ inward or rebalancing repeats, yet the maximum achievable coherent-sign coverage $C_{\max}$ still plateaus well below $80\%$ because diminishing the central/side dose drops $|\kappa_{\text{weak}}|$ below $0.1\,\text{m}^{-1}$, that outcome **must be reported as a genuine physical/parameterization limitation** of the 3-toolpath Attempt 209 zigzag family (specifically, the incompatibility between fixed `center_peak` top-only modulation on A, $0^\circ/90^\circ$ discrete orientations, and Attempt 209 orthotropy), rather than tuned away by lowering $\kappa_{\min}$ or the $80\%$ area requirement.

### 6.2 Choices Requiring Explicit Researcher Direction (For Next-Stage Decision Only)

This review provides scientific analysis for the researcher's next-stage decision and **does not** authorize code edits, pre-execution clearance, or launching a new campaign. Once `run/dome/bo_continuation_stage1` reaches a terminal state (or if the researcher decides to transition stages), the following choices require explicit researcher direction:

1. **Wait vs. Transition Decision on `bo_continuation_stage1`:**
   - Whether to let `run/dome/bo_continuation_stage1` finish its frozen 24-point initial design (waves 0–11) and subsequent GP waves under the current $C_{\max}$ objective, or stop after a designated checkpoint to switch the GP acquisition score to $J_{\lambda,\max}$ in a new isolated stage directory (reusing all completed continuation attempts as compatible same-recipe warm starts by computing $J_{\lambda,\max}$ from their saved `analysis/curvature_fields.npz` files).
2. **Hyperparameters of the Minimal Alternative Objective $J_{\lambda,\max}$ (if adopted):**
   - **Saddle penalty weight $\lambda$**: Recommended $\lambda = 0.5$ (balances eliminating $K < 0$ saddle bands against maintaining $|\kappa_{\text{weak}}| \ge 0.1\,\text{m}^{-1}$; $\lambda = 1.0$ weights saddle suppression equally with full threshold attainment).
   - **Saddle half-saturation scale $\kappa_{\text{sad}}$**: Recommended $\kappa_{\text{sad}} = \kappa_{\min} = 0.1\,\text{m}^{-1}$ (reuses the already-frozen curvature floor so no new physical scale constant is introduced).
3. **Scope Decision if Saddle-Penalized BO Still Plateaus Below $80\%$:**
   - Whether to conclude and report a negative result for the fixed-profile $0^\circ/90^\circ$ zigzag parameterization under Attempt 209, or authorize specific recipe expansions currently out of scope (such as modifying the longitudinal profile on middle path A to taper curvature transitions, or allowing non-orthogonal strip rotations).