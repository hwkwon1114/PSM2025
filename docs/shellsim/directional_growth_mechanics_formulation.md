# Directional Growth Mechanics: Breaking Nematic $\pi$-Symmetry in Toolpath-Driven Shell Incompatibility

**Author:** PSM2025-zigzag Core Mechanics Team  
**Date:** October 2026  
**Document Identity:** `docs/reports/directional_growth_mechanics_formulation.md`  
**Target:** Task 2 of Goal `mvaamanv-vds7e3` (`derive-directional-formulation`)

---

## 1. Context and Problem Statement

Physical English wheeling involves cyclic, directional rolling contact where an elastoplastic metal sheet is repeatedly compressed and stretched by an anvil wheel moving along prescribed toolpaths. In physical experiments:
- Executing two passes in the **same direction** ($\text{Path}_1 \to \text{Path}_2$, Forward–Forward) produces a distinct final 3D equilibrium shape compared to
- Executing the second pass in the **reversed direction** ($\text{Path}_1 \to -\text{Path}_2$, Forward–Backward).

However, baseline simulations in `Sim_Bilayer_Growth` using `zigzag_sequence` yield an aligned RMS shape difference of only **$0.005 - 0.020\ \mu\text{m}$ (5 to 20 nanometers)**—well below the numerical noise floor of $51.95\ \mu\text{m}$.

This mathematical document rigorously derives:
1. Why standard eigenstrain shell kinematics exhibit an exact **nematic $\pi$-symmetry** that enforces invariance under path reversal;
2. Candidate Model A: **Directional Arc-Length Drift & Frictional Shear Traction**;
3. Candidate Model B: **State-Dependent Curvature and Slope Feedback**;
4. The asymptotic scaling of both mechanisms across multi-pass forming schedules.

---

## 2. The Nematic $\pi$-Symmetry of Standard Metric Updates

### 2.1 The Discrete First Fundamental Form and Edge Basis
Let $\mathcal{M} \subset \mathbb{R}^2$ be the flat material reference domain parametrized by coordinates $\mathbf{u} = (u, v)^T$. In a 2D triangular mesh, each face $f$ has vertices $\mathbf{u}_0, \mathbf{u}_1, \mathbf{u}_2$. The discrete edge basis matrix $\mathbf{D}_m \in \mathbb{R}^{2 \times 2}$ is defined in `GrowthHelper.hpp:780` as:
$$\mathbf{D}_m = \begin{pmatrix} \mathbf{u}_2 - \mathbf{u}_1 & \mathbf{u}_0 - \mathbf{u}_2 \end{pmatrix}$$
The flat reference first fundamental form in this edge basis is $\mathbf{a}_0 = \mathbf{D}_m^T \mathbf{D}_m$.

### 2.2 Standard Eigenstrain Transformation
A toolpath stroke is defined by a directed straight line segment from $\mathbf{A}$ to $\mathbf{B}$ in material space. The unit travel tangent vector is:
$$\mathbf{t} = \frac{\mathbf{B} - \mathbf{A}}{\|\mathbf{B} - \mathbf{A}\|} = \begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix}$$
with orthogonal in-plane normal $\mathbf{m} = (-\sin\theta, \cos\theta)^T$.

In standard orthotropic growth, principal expansions $g_1$ (longitudinal along $\mathbf{t}$) and $g_2$ (transverse along $\mathbf{m}$) form the diagonal stretch tensor $\mathbf{L} = \text{diag}(1 + g_1, 1 + g_2)$. The material-frame deformation gradient is:
$$\mathbf{G}(\theta) = \mathbf{R}(\theta) \, \mathbf{L} \, \mathbf{R}(\theta)^T$$
where $\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

### 2.3 Proof of Exact Nematic Invariance
Consider reversing the traversal direction of the stroke from $\mathbf{B}$ to $\mathbf{A}$. The new travel vector is:
$$\mathbf{t}' = -\mathbf{t} = \begin{pmatrix} \cos(\theta + \pi) \\ \sin(\theta + \pi) \end{pmatrix}$$
The rotation matrix becomes:
$$\mathbf{R}(\theta + \pi) = \begin{pmatrix} \cos(\theta+\pi) & -\sin(\theta+\pi) \\ \sin(\theta+\pi) & \cos(\theta+\pi) \end{pmatrix} = -\mathbf{R}(\theta)$$
Substituting into the growth tensor:
$$\mathbf{G}(\theta + \pi) = (-\mathbf{R}(\theta)) \, \mathbf{L} \, (-\mathbf{R}(\theta))^T = (-1)^2 \, \mathbf{R}(\theta) \, \mathbf{L} \, \mathbf{R}(\theta)^T = \mathbf{G}(\theta)$$

Expanding $\mathbf{G}(\theta)$ dyadically:
$$\mathbf{G}(\theta) = (1 + g_1) \, (\mathbf{t} \otimes \mathbf{t}) + (1 + g_2) \, (\mathbf{m} \otimes \mathbf{m})$$
Because $(-\mathbf{t}) \otimes (-\mathbf{t}) = \mathbf{t} \otimes \mathbf{t}$ and $(-\mathbf{m}) \otimes (-\mathbf{m}) = \mathbf{m} \otimes \mathbf{m}$, $\mathbf{G}$ is a **rank-2 nematic line field**. It possesses strictly even parity under spatial reversal:
$$\mathbf{G}(-\mathbf{t}) \equiv \mathbf{G}(\mathbf{t})$$

Consequently:
$$\mathbf{T} = \mathbf{D}_m^{-1} \mathbf{G} \mathbf{D}_m \implies \bar{\mathbf{a}}_{n+1} = \mathbf{T}^T \bar{\mathbf{a}}_n \mathbf{T}$$
is mathematically identical regardless of the sign of $\mathbf{t}$. **No modification of solver tolerance, mesh resolution, or minimization cadence can break this identity.**

---

## 3. Candidate Model A: Directional Arc-Length Drift and Shear Traction

### 3.1 Physical Mechanism
In real roll forming, contact is non-conservative and driven by frictional contact traction:
$$\boldsymbol{\tau}_{\text{contact}} = \mu \, p_{\text{roller}} \, \mathbf{t}$$
Material is continuously drawn and extruded along the direction of travel $\mathbf{t}$. As the roller rolls along the stroke:
1. **Accumulated plastic flow:** Material near the exit of the stroke experiences higher accumulated plastic strain than material at the initial entry bite.
2. **Through-thickness tilt:** Forward traction tilts the cross-section, inducing an asymmetric transverse shear strain $\gamma_{13}$ that changes sign when the roller travels backwards.

### 3.2 Mathematical Formulation
Let $s \in [0, 1]$ denote the normalized arc-length coordinate of a material point $\mathbf{u}$ projected onto the stroke $\mathbf{A} \to \mathbf{B}$:
$$s(\mathbf{u}, \mathbf{t}) = \frac{(\mathbf{u} - \mathbf{A}) \cdot \mathbf{t}}{\|\mathbf{B} - \mathbf{A}\|}$$
When the stroke is traversed from $\mathbf{A} \to \mathbf{B}$, $s$ increases from $0$ to $1$. When traversed in reverse from $\mathbf{B} \to \mathbf{A}$, the parameter is:
$$s'(\mathbf{u}, -\mathbf{t}) = \frac{(\mathbf{u} - \mathbf{B}) \cdot (-\mathbf{t})}{\|\mathbf{A} - \mathbf{B}\|} = 1 - s(\mathbf{u}, \mathbf{t})$$

We define the **directional drift profile**:
$$g_{\text{top}}(\mathbf{u}, \mathbf{t}) = g_0 \left[ 1 + \eta_{\text{drift}} \left(s - \frac{1}{2}\right) \right]$$
where $\eta_{\text{drift}} \in [-1, 1]$ is the dimensionless forward-drift coefficient.

Furthermore, we introduce an **odd-parity directional shear eigenstrain** $\Delta \bar{\mathbf{a}}_{\text{shear}}$:
$$\Delta \bar{\mathbf{a}}_{\text{shear}}(\mathbf{t}) = \xi_{\text{shear}} \, g_0 \left( \mathbf{t} \otimes \mathbf{m} + \mathbf{m} \otimes \mathbf{t} \right) \cdot \text{sgn}(\mathbf{t} \cdot \mathbf{v})$$
Because $\mathbf{t} \to -\mathbf{t}$ while $\mathbf{m} \to -\mathbf{m}$ flips sign, the product $\mathbf{t} \otimes \mathbf{m}$ changes sign under travel reversal:
$$(-\mathbf{t}) \otimes (-\mathbf{m}) = \mathbf{t} \otimes \mathbf{m}$$
However, using the explicit velocity vector $\mathbf{v} = v \, \mathbf{t}$, the directional traction moment:
$$\bar{\mathbf{b}}_{\text{traction}} = \chi_{\text{traction}} \, (\mathbf{t} \cdot \mathbf{e}_{\text{travel}}) \, \mathbf{t} \otimes \mathbf{t}$$
breaks the parity directly.

### 3.3 Two-Pass Asymptotic Prediction
Consider two identical overlapping strokes over a region.
- **Forward–Forward (`fwd_fwd`):**
  Pass 1: $g^{(1)} = g_0 [1 + \eta (s - 0.5)]$
  Pass 2: $g^{(2)} = g_0 [1 + \eta (s - 0.5)]$
  $$\Sigma g_{\text{fwd\_fwd}}(s) = 2 g_0 \left[ 1 + \eta \left(s - \frac{1}{2}\right) \right]$$
  The along-stroke gradient is $\frac{\partial \Sigma g}{\partial s} = 2 \eta g_0$.
- **Forward–Backward (`fwd_rev`):**
  Pass 1: $g^{(1)} = g_0 [1 + \eta (s - 0.5)]$
  Pass 2: $g^{(2)} = g_0 [1 + \eta ((1-s) - 0.5)] = g_0 [1 - \eta (s - 0.5)]$
  $$\Sigma g_{\text{fwd\_rev}}(s) = g_0 [1 + \eta(s - 0.5)] + g_0 [1 - \eta(s - 0.5)] = 2 g_0$$
  The along-stroke gradient cancels out: $\frac{\partial \Sigma g}{\partial s} \equiv 0$!

**Key Finding:** 
In `fwd_fwd`, the directional drift compounds, producing an end-to-end thickness/curvature gradient $\Delta \kappa \propto 2 \eta g_0$.  
In `fwd_rev`, the reversed pass cancels the first-order gradient, restoring planar balance. This predicts a macroscopic difference in **asymmetric curling and end-lift** between the two schedules.

---

## 4. Candidate Model B: State-Dependent 3D Curvature and Slope Feedback

### 4.1 Physical Mechanism
Physical rolling takes place on a workpiece whose geometry evolves in 3D space:
1. In Pass 1, the flat plate bends and buckles out of plane, creating an out-of-plane displacement field $z(\mathbf{u})$ and curvature field $\mathbf{I\!I}(\mathbf{u})$.
2. In Pass 2, the roller wheel encounters this pre-deformed geometry. If the sheet is sloping upwards into the roller ($\mathbf{t} \cdot \nabla z > 0$), the roller bites into an inclined wedge, generating higher normal pressure and plastic contact depth than if the sheet is sloping downwards away from the roller.

### 4.2 Mathematical Formulation
Let $\mathbf{x}(\mathbf{u}) = (x, y, z)^T$ be the current 3D equilibrium configuration of the shell mid-surface.
1. The **out-of-plane slope along the travel direction** is:
   $$\psi_{\mathbf{t}}(\mathbf{u}) = \mathbf{t} \cdot \nabla z(\mathbf{u}) = t_u \frac{\partial z}{\partial u} + t_v \frac{\partial z}{\partial v}$$
   Notice that:
   $$\psi_{-\mathbf{t}}(\mathbf{u}) = (-\mathbf{t}) \cdot \nabla z(\mathbf{u}) = -\psi_{\mathbf{t}}(\mathbf{u})$$
   **The slope projection is strictly odd in the travel tangent $\mathbf{t}$.**

2. The **normal curvature along the travel direction** is:
   $$\kappa_{\mathbf{t}}(\mathbf{u}) = \mathbf{t}^T \mathbf{I\!I} \, \mathbf{t}$$
   where $\mathbf{I\!I}$ is the second fundamental form of the deformed configuration.

We formulate the **state-dependent effective growth rate**:
$$g_{\text{eff}}(\mathbf{u}, \mathbf{t}, \mathbf{x}) = g_0 \left[ 1 + \gamma_{\text{slope}} \, (\mathbf{t} \cdot \nabla z) + \gamma_{\text{curv}} \, \kappa_{\mathbf{t}} \right]$$
where:
- $\gamma_{\text{slope}} \ge 0$ is the wedge/slope coupling parameter ($[\text{m}^{-1}]$ or dimensionless if normalized by thickness $h$),
- $\gamma_{\text{curv}}$ is the contact bite curvature compliance ($[\text{m}]$).

### 4.3 Multi-Pass Progression
Under Model B:
- **Pass 1:** Starts from flat state $\mathbf{x}_0$ ($z \equiv 0, \nabla z \equiv 0$). Effective growth is uniform: $g_{\text{eff}}^{(1)} = g_0$. The shell equilibrates to 3D state $\mathbf{q}_1$ with non-zero slope $\nabla z_1$ and curvature $\kappa_1$.
- **Pass 2 Forward:** Roller moves along $+\mathbf{t}$:
  $$g_{\text{eff}}^{(2,\text{fwd})} = g_0 \left[ 1 + \gamma_{\text{slope}} \, (\mathbf{t} \cdot \nabla z_1) + \gamma_{\text{curv}} \, \kappa_{\mathbf{t},1} \right]$$
- **Pass 2 Reversed:** Roller moves along $-\mathbf{t}$:
  $$g_{\text{eff}}^{(2,\text{rev})} = g_0 \left[ 1 - \gamma_{\text{slope}} \, (\mathbf{t} \cdot \nabla z_1) + \gamma_{\text{curv}} \, \kappa_{\mathbf{t},1} \right]$$

The discrepancy between the two passes is:
$$\Delta g_{\text{eff}} = g_{\text{eff}}^{(2,\text{fwd})} - g_{\text{eff}}^{(2,\text{rev})} = 2 g_0 \gamma_{\text{slope}} \, (\mathbf{t} \cdot \nabla z_1)$$
This discrepancy is:
1. **Directly proportional to the intermediate deformed slope $\nabla z_1$:** If intermediate relaxation is bypassed and all growth is applied to the flat state, $\nabla z = 0$ and the discrepancy vanishes!
2. **Coupled to intermediate minimization cadence:** Minimizing after Pass 1 establishes $\nabla z_1 \ne 0$, allowing Pass 2 to branch into a distinct deformed state.

---

## 5. Summary Comparison of Candidate Formulations

| Property | Baseline Model | Candidate Model A (Directional Drift) | Candidate Model B (Curvature Feedback) |
| :--- | :---: | :---: | :---: |
| **Parity under $\mathbf{t} \to -\mathbf{t}$** | Strictly even ($\pi$-symmetric) | Odd along arc-length $s$ | Odd in projected slope $\mathbf{t} \cdot \nabla z$ |
| **Depends on 3D Deformed State?** | No (flat $(u, v)$ only) | No (flat $(u, v)$ arc-length) | **Yes** (uses 3D coordinates $\mathbf{x}_k$) |
| **Requires Intermediate Minimization?** | No | No (effective in single solve) | **Yes** (requires deformed intermediate state) |
| **Predicted Physical Effect** | Zero difference ($< 20\text{ nm}$) | Asymmetric end-to-end curl / tilt | Curvature amplitude and warpage bifurcation |
| **Implementation Complexity** | Zero (existing) | Minimal (add $s$-modulation to profile) | Moderate (query mesh vertex $z$ in mapping) |

Both candidate models provide rigorous, physically motivated paths to break the nematic symmetry. In Task 3, we implement and test these models to evaluate their simulated deformation signatures on the coupon benchmark.
