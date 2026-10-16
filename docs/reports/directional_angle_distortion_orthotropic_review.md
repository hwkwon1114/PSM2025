# Directional Angle Distortion and Through-Thickness Orthotropic Shear Coupling in English Wheel Dynamics

**Document Type**: Theoretical Review & Long-Term Modeling Framework  
**Author**: Computational Shell Mechanics Investigation Team  
**Date**: March 2025  
**Status**: Recorded for future development; deferred from current manuscript scope  
**Related Documents**:
- `docs/reports/directional_growth_mechanics_formulation.md`
- `docs/reports/directional_path_dependence_synthesis_and_scaling.md`
- `benchmarks/directional_path_coupon/prototype_mechanism_comparison.md`

---

## 1. Executive Summary & Context

Experimental observations in English wheeling and roller leveling indicate that tool traversal direction is a first-order processing parameter: running passes in a **forward–forward** (`fwd_fwd`) schedule generates different out-of-plane shapes (specifically torsional twist, transverse camber, and corner lift) compared to a **forward–backward** (`fwd_rev`) schedule.

Existing continuum shell growth formulations predict zero difference due to strict nematic $\pi$-symmetry ($\mathbf{G}(-\mathbf{t}) \equiv \mathbf{G}(\mathbf{t})$). While along-stroke linear drift (Model A) successfully accounts for longitudinal curling/camber ($\kappa_{11}$), it does not directly capture torsional twisting modes ($\kappa_{12}$).

This review formalizes a complementary physical concept: **Directional Angle Distortion coupled to through-thickness orthotropy**. By allowing rolling direction to introduce a small contact yaw/shear tilt $\pm \delta$ on the top driven surface, the growth tensor generates an in-plane shear strain $\gamma_{12}$ that is directly modulated by the calibrated top orthotropy parameter $\text{ortho}_{\text{top}}$.

### Strategic Scope Decision
* **For the current manuscript**: This mechanism will **not** be included in the active calibration or main results. Retaining the clean, verified baseline eigenstrain model prevents over-parameterization and keeps the paper tightly bounded.
* **For long-term development**: This review preserves the mathematical derivation, contact-mechanics justification, and computational architecture as the theoretical blueprint for next-generation English wheel dynamical digital twins.

---

## 2. Mathematical Formulation

### 2.1 Local Orthotropic Growth Tensor
In the local material chart aligned with nominal toolpath tangent $\mathbf{t} = (\cos\theta, \sin\theta)^T$ and in-plane normal $\mathbf{t}^\perp = (-\sin\theta, \cos\theta)^T$, the diagonal growth tensor is:
$$\mathbf{L} = \begin{pmatrix} 1 + g_1 & 0 \\ 0 & 1 + g_2 \end{pmatrix}$$
where:
$$g_1 = g (1 + \text{ortho}), \qquad g_2 = g (1 - \text{ortho})$$
The difference between longitudinal and transverse stretching is governed directly by the orthotropy factor:
$$g_1 - g_2 = 2 g \cdot \text{ortho}$$

### 2.2 Directional Angle Distortion
Let $\mathbf{v} = v \, \mathbf{t}$ denote the directed velocity vector of the rolling tool. Reversing the traversal direction inverts velocity: $\mathbf{v} \to -\mathbf{v}$.

Under rolling contact friction, lateral wheel compliance, or contact patch asymmetry, the principal axis of plastic deformation skews by an angle $\delta$ relative to the velocity vector. In the global material coordinate frame:
$$\theta_{\text{eff}} = \begin{cases} \theta + \delta & \text{for forward traversal } (+\mathbf{t}) \\ \theta - \delta & \text{for reverse traversal } (-\mathbf{t}) \end{cases}$$

### 2.3 The Shear Discrepancy Tensor
The rotated metric growth increment $\mathbf{G}(\delta)$ is given by:
$$\mathbf{G}(\delta) = \mathbf{R}(\delta) \mathbf{L} \mathbf{R}(\delta)^T$$
Expanding with the 2D rotation matrix $\mathbf{R}(\delta)$:
$$\mathbf{G}(\delta) = \begin{pmatrix} (1+g_1)\cos^2\delta + (1+g_2)\sin^2\delta & \frac{1}{2}(g_1 - g_2)\sin(2\delta) \\ \frac{1}{2}(g_1 - g_2)\sin(2\delta) & (1+g_1)\sin^2\delta + (1+g_2)\cos^2\delta \end{pmatrix}$$

Evaluating the difference between forward and reverse traversal:
$$\Delta \mathbf{G} = \mathbf{G}(+\delta) - \mathbf{G}(-\delta)$$

Since $\cos^2(-\delta) = \cos^2\delta$ and $\sin^2(-\delta) = \sin^2\delta$, the diagonal normal strain terms cancel out identically. The off-diagonal term satisfies $\sin(-2\delta) = -\sin(2\delta)$:

$$\Delta \mathbf{G} = \begin{pmatrix} 0 & (g_1 - g_2)\sin(2\delta) \\ (g_1 - g_2)\sin(2\delta) & 0 \end{pmatrix} = \begin{pmatrix} 0 & 2 g \cdot \text{ortho} \cdot \sin(2\delta) \\ 2 g \cdot \text{ortho} \cdot \sin(2\delta) & 0 \end{pmatrix}$$

### Core Mathematical Properties:
1. **Pure In-Plane Shear**: Angle distortion induces zero net change in longitudinal or transverse area dilation; it induces **pure shear strain** $\gamma_{12} = 2 g \cdot \text{ortho} \cdot \sin(2\delta)$.
2. **Orthotropy Gating**: If $\text{ortho} = 0$ (isotropic growth), $\Delta \mathbf{G} \equiv \mathbf{0}$. The directional asymmetry exists if and only if the process exhibits anisotropic plastic flow.

---

## 3. Coupling with Through-Thickness Asymmetry (`ortho_top` vs `ortho_bottom`)

In our existing bilayer calibration framework (`Run 6` and `Run 9`), the model distinguishes upper and lower contact mechanics:
$$\text{Parameters}: \quad (g_{\text{top}}, g_{\text{bot}}, \text{ortho}_{\text{top}}, \text{ortho}_{\text{bottom}})$$
In calibrated `Run 6`, the upper driven wheel experiences strong orthotropy ($\text{ortho}_{\text{top}} \approx -0.46$), while the anvil wheel contact is essentially isotropic ($\text{ortho}_{\text{bottom}} \approx 0.0$).

### 3.1 Differential Shear and Torsional Curvature
Applying angle distortion $\delta_{\text{top}}$ to the upper contact interface yields:
$$\gamma_{12,\text{top}} = 2 g_{\text{top}} \cdot \text{ortho}_{\text{top}} \cdot \sin(2\delta_{\text{top}})$$
$$\gamma_{12,\text{bot}} = 2 g_{\text{bot}} \cdot \text{ortho}_{\text{bottom}} \cdot \sin(2\delta_{\text{bot}}) \approx 0$$

In Kirchhoff–Love and Koiter shell theory, a through-thickness gradient in shear strain produces **out-of-plane twisting curvature $\kappa_{12}$**:
$$\kappa_{12} \approx \frac{\gamma_{12,\text{top}} - \gamma_{12,\text{bot}}}{h} \approx \frac{2 g_{\text{top}} \cdot \text{ortho}_{\text{top}} \cdot \sin(2\delta_{\text{top}})}{h}$$

### 3.2 Pass Accumulation Dynamics
* **Forward–Forward (`fwd_fwd`)**: Both passes apply shear of the same sign:
  $$\kappa_{12}^{\text{net}} \approx \frac{4 g_{\text{top}} \cdot \text{ortho}_{\text{top}} \cdot \sin(2\delta_{\text{top}})}{h} \neq 0$$
  This produces a persistent **helical twist / saddle warpage** where two diagonally opposite corners lift.
* **Forward–Backward (`fwd_rev`)**: The reverse pass inverts the shear sign:
  $$\kappa_{12}^{\text{net}} \approx \kappa_{12}^{(1)} - \kappa_{12}^{(2)} \approx 0$$
  The sheet remains **untwisted and symmetric**.

---

## 4. Physical Origins in English Wheeling Contact Dynamics

Why should rolling velocity introduce an angle distortion $\delta$?

```
                    Upper Driven Wheel (Rigid / Cambered)
                            [   O   ]  ---> Travel (+v)
                             \  |  /
                              \ | /   Frictional Traction (tau_12)
      ========================= v ========================= Sheet (h)
                               / \
                              /   \   Anvil Reaction
                            [   O   ] Lower Anvil Wheel
```

1. **Spindle Compliance & Dynamic Caster**: An English wheel carriage experiences substantial normal forces ($1$–$10\text{ kN}$). Forward rolling friction exerts a tangential reaction force on the upper wheel hub, causing micro-yaw deflection of the spindle axis.
2. **Contact Patch Slip Asymmetry**: The contact ellipse between a toroidal wheel and a thin sheet contains distinct stick and slip zones. Asymmetric slip boundaries under rolling friction induce a non-coaxial traction vector, tilting the principal stress tensor away from the purely longitudinal direction.
3. **Rolling Texture Interaction**: Sheet metals possess crystallographic rolling texture. When toolpaths traverse the sheet obliquely relative to the sheet's prior mill rolling direction, plastic yielding exhibits non-collinear plastic flow (anomalous shear coupling).

---

## 5. Computational & Algorithmic Advantages

From a computational perspective, the angle distortion formulation possesses ideal numerical properties:

| Feature | Angle Distortion ($\delta$) | Geometric Slope Feedback (Model B) | Full 3D Contact FEA |
| :--- | :--- | :--- | :--- |
| **Cadence Requirement** | **Single Final Solve (`final_only`)** | Intermediate Solves | Step-by-Step Contact Solves |
| **Intermediate Solve Count** | **1 solve** | 6–10 solves | Thousands of time-steps |
| **Runtime for 239 Paths** | $\approx 1\text{ minute}$ | $\approx 6–10\text{ minutes}$ | Hours / Days |
| **Differentiability** | **Analytic (Closed-Form)** | Piecewise Smooth | Non-smooth / Contact Chatter |
| **Deformation Mode Captured**| **Torsional Twist ($\kappa_{12}$)** | Curvature Bifurcation ($\kappa_1$) | Full localized contact stress |

Because $\theta_{\text{eff}} = \theta \pm \delta$ is evaluated directly from the toolpath trajectory in flat material coordinates $(u, v)$, it introduces **zero intermediate solver overhead**. It scales effortlessly to multi-hundred path schedules.

---

## 6. Strategic Recommendations & Future Roadmap

### 6.1 For the Current Manuscript (Clean Scope)
* **Exclude $\delta$ from active calibration**: Do not add $\delta$ to the 4-parameter set $(g_{\text{top}}, g_{\text{bot}}, \text{ortho}_{\text{top}}, \text{ortho}_{\text{bottom}})$ for the upcoming paper.
* **Maintain focus on core contributions**: Keep the paper focused on the core eigenstrain formulation, the 2-parameter/4-parameter calibration on `Run 6`/`Run 9`, and the verified finding that the baseline model operates consistently under a single final solve (`final_only`).

### 6.2 For Future Work & Next-Generation Modeling
1. **Coupon Diagnostic Experiment**: Design a focused 4-coupon experiment specifically measuring torsional twist:
   * Specimen A: 4 passes all forward (`fwd_fwd_fwd_fwd`).
   * Specimen B: 4 passes alternating (`fwd_rev_fwd_rev`).
   * Measure corner deflection via 3D scanning or dial indicators to isolate $\Delta \kappa_{12}$.
2. **Targeted Calibration**: Since $\text{ortho}_{\text{top}}$ is already known from macroscopic panel calibration ($-0.46$), a 1D line search on the coupon data will identify $\delta$ directly via:
   $$\sin(2\delta) \approx \frac{h \cdot \kappa_{12}^{\text{meas}}}{4 g_{\text{top}} \cdot \text{ortho}_{\text{top}}}$$
3. **C++ Integration**: When scheduled, integration requires updating only two lines in `src/simulations/Sim_Bilayer_Growth.cpp`:
   ```cpp
   const double travel_angle_offset = (op.reverse_traversal ? -1.0 : 1.0) * op.angle_distortion_rad;
   const double eff_angle_top = hit.angle_rad + travel_angle_offset;
   ```
