# Baseline Directional Symmetry Audit: Forward-Forward vs. Forward-Backward

**Date:** October 2026  
**Simulation Engine:** `bin/shell` (`native_hybrid` exact-Hessian solver, gradient tolerance $5 \times 10^{-14}$)  
**Geometry:** Rectangular coupon $L_x = 0.127\text{ m}$ (127 mm), $L_y = 0.1524\text{ m}$ (152.4 mm), thickness $h = 0.6\text{ mm}$, mesh resolution $\text{res} = 0.03\text{ m}$  
**Toolpath Schedule:** 2-pass zigzag ($Lv = 120\text{ mm}$, $\alpha = 10^\circ$, $N = 6$ strips, width $w = 12\text{ mm}$, top growth $g_{\text{top}} = 0.002$, bottom $g_{\text{bot}} = 0.0$, orthotropy $\text{ortho} = 0.8$)

---

## 1. Executive Summary

In physical English wheeling / roll forming experiments, reversing the traversal direction on a second pass (forward–forward vs. forward–backward) produces an observable macroscopic difference in the deformed sheet shape.

To evaluate whether the current analytical model can capture this effect, we conducted controlled forward simulations on a standardized coupon benchmark comparing:
1. **Schedule A (`fwd_fwd`):** Pass 1 (strokes $0 \to 5$) followed by Pass 2 (strokes $0 \to 5$).
2. **Schedule B (`fwd_rev`):** Pass 1 (strokes $0 \to 5$) followed by Pass 2 (strokes $5 \to 0$, reversed stroke order and reversal of stroke traversal).

Across all temporal relaxation cadences (single final solve, intermediate solve every complete pass, and intermediate solve after every single stroke), the baseline model produces:
- **Top Target Metric Difference:** $\text{RMS} = 9.77 \times 10^{-9}$, $\text{Max} = 8.48 \times 10^{-8}$ (confined exclusively to second-order non-commutative strip-intersection corners).
- **Aligned 3D Shape Difference:** $\text{RMS} = 0.005 - 0.020\ \mu\text{m}$ ($5\text{ to }20\text{ nanometers}$).
- **Principal Curvature Discrepancy:** $\Delta \kappa_1 = 2.9 \times 10^{-8}\ \text{m}^{-1}$ (relative error $< 7 \times 10^{-9}$).
- **Out-of-Plane Z-Span:** Identical to $45.696\text{ mm}$ ($< 0.1\ \mu\text{m}$ difference).

**Conclusion:** Under the current analytical formulation, the simulation is **strictly invariant (to within machine precision / nanometer solver noise)** under toolpath reversal. The current model is mathematically incapable of reproducing the experimental forward vs. backward discrepancy.

---

## 2. Quantitative Results

All cases converged to machine precision with `native_hybrid` (gradient norm $\|\nabla E\| \le 5 \times 10^{-14}$, exact positive-definite Hessian certification).

| Relaxation Cadence | Case Pair | Top Metric Diff Max | Top Metric Diff RMS | Aligned Shape RMS ($\mu$m) | Aligned Shape Max ($\mu$m) | Principal Curvatures $\kappa_1, \kappa_2$ ($\text{m}^{-1}$) | Z-Span (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Solve** (accumulate 12 strokes, 1 solve) | `single_fwd_fwd`<br>`single_fwd_rev` | $8.48 \times 10^{-8}$ | $9.77 \times 10^{-9}$ | **0.005** | 0.018 | $-4.278924, +0.255739$<br>$-4.278924, +0.255739$ | 45.696<br>45.696 |
| **Pass-by-Pass** (solve after each 6-stroke pass) | `inter_pass_fwd_fwd`<br>`inter_pass_fwd_rev` | $8.48 \times 10^{-8}$ | $9.77 \times 10^{-9}$ | **0.020** | 0.063 | $-4.278921, +0.255739$<br>$-4.278919, +0.255739$ | 45.696<br>45.696 |
| **Stroke-by-Stroke** (solve after every 1 stroke) | `inter_strip_fwd_fwd`<br>`inter_strip_fwd_rev` | $8.48 \times 10^{-8}$ | $9.77 \times 10^{-9}$ | **0.008** | 0.029 | $-4.278924, +0.255739$<br>$-4.278923, +0.255739$ | 45.696<br>45.696 |

*Note: For context, the physical repeatability noise floor of the sheet solver is $51.95\ \mu\text{m}$. The observed differences ($0.005 - 0.020\ \mu\text{m}$) are more than $2500\times$ below the noise floor and represent pure floating-point roundoff.*

---

## 3. Mathematical & Physical Root Causes

Why does the analytical model fail to see the physical difference?

### Cause 1: Nematic $\pi$-Symmetry in the Growth Metric Update
In `src/libshell/GrowthHelper.hpp:780-795` and `src/libshell/ZigZagGrowth.hpp:94-101`:
```cpp
// Normalize orientation so opposite directions are equivalent (0..pi)
inline double normalize_angle_pi(double a) {
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a > M_PI) a -= 2.0 * M_PI;
    if (a < 0.0) a += M_PI;
    return a;
}
```
The incremental material growth tensor is constructed as:
$$\mathbf{G}(\theta) = \mathbf{R}(\theta) \begin{pmatrix} 1 + g_1 & 0 \\ 0 & 1 + g_2 \end{pmatrix} \mathbf{R}(\theta)^T$$
When a stroke is traversed from $\mathbf{A} \to \mathbf{B}$ vs $\mathbf{B} \to \mathbf{A}$, the travel angle reverses: $\theta \to \theta + \pi$.
Because $\mathbf{R}(\theta + \pi) = -\mathbf{R}(\theta)$, we have:
$$\mathbf{G}(\theta + \pi) = (-\mathbf{R}) \, \mathbf{L} \, (-\mathbf{R})^T = \mathbf{R} \, \mathbf{L} \, \mathbf{R}^T = \mathbf{G}(\theta)$$
The growth tensor is a **second-order nematic line field** ($\mathbf{t} \otimes \mathbf{t}$), completely symmetric under sign inversion of the travel tangent $\mathbf{t}$. The model has no vector arrow of travel.

### Cause 2: Growth Evaluated Exclusively in Flat Reference Coordinates $(u, v)$
In `src/libshell/ZigZagGrowth.hpp:460-520` and `collectMaterialCoordinateHits`, the distance $d$ between a triangle face centroid $\mathbf{c}_i$ and a toolpath segment is calculated using fixed 2D material coordinates $(u, v)$.
Even when intermediate minimization is enabled (`-sequence_minimize_every 1`), the sheet deforms into 3D (reaching $Z_{\text{span}} \approx 45.7\text{ mm}$), but the growth increment calculation is completely uncoupled from the 3D deformed geometry $\mathbf{x}(\mathbf{u})$:
- The roller contact pressure and contact footprint are assumed constant regardless of sheet slope or curvature.
- The roller entry bite angle is assumed planar.
- No geometric feedback exists between the current deformed configuration $\mathbf{q}_k$ and the eigenstrain increment $\Delta \bar{\mathbf{a}}_{k+1}$.

### Cause 3: Neglect of Frictional Shear Traction and Cyclic Plasticity
Physical rolling applies intense surface shear traction along the rolling direction $\mathbf{t}$, tilting the through-thickness shear strain $\gamma_{13}$. When the roller reverses, this shear reverses, inducing a Bauschinger effect and an asymmetric out-of-plane twist. The current hyperelastic shell energy only stores symmetric 2D fundamental forms ($\bar{\mathbf{a}}, \bar{\mathbf{b}}$).

---

## 4. Requirement for Task 2

To reproduce the experimental forward vs. backward discrepancy, the analytical model must incorporate a mechanism that breaks the nematic $\pi$-symmetry:
1. **Mechanism 1 (Directional Shear / Curvature Tilt):** An odd-parity eigenstrain or reference curvature term proportional to the signed travel vector $\mathbf{t} = (\cos\theta, \sin\theta)$.
2. **Mechanism 2 (Curvature-Dependent Contact Growth):** Coupling the effective growth rate $g_{\text{eff}}$ to the existing 3D deformed curvature along the travel direction ($\kappa_{\mathbf{t}} = \mathbf{t}^T \mathbf{I\!I} \mathbf{t}$).
