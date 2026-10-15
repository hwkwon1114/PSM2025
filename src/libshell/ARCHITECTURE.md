# Discrete Shell Architecture & Research Guide

This document provides a concise architectural mental map of `src/libshell/` and a practical guide for extending the physics (e.g., temporal, rate-dependent, or history-dependent growth laws).

---

## 1. High-Level System Layers

The codebase is structured into four distinct architectural layers:

```
┌────────────────────────────────────────────────────────┐
│  Layer 4: Kinematics & Growth Laws (ACTIVE RESEARCH)  │
│  - ZigZagGrowth.hpp, GrowthHelper.hpp                  │
│  - Toolpath schedules, footprint projection, growth   │
└──────────────────────────┬─────────────────────────────┘
                           │ Sets target metric a_target
┌──────────────────────────▼─────────────────────────────┐
│  Layer 3: Mechanics Solver & Numerical Optimization    │
│  - NativeHybridShellSolver.hpp, ReferenceHybrid/*      │
│  - L-BFGS warmup + exact Newton with CHOLMOD factorize │
└──────────────────────────┬─────────────────────────────┘
                           │ Evaluates E, grad E, Hess E
┌──────────────────────────▼─────────────────────────────┐
│  Layer 2: Elastic Energy & Fundamental Forms           │
│  - CombinedOperator_Parametric, BendingOperator        │
│  - First (a) & second (b) fundamental forms of shell   │
└──────────────────────────┬─────────────────────────────┘
                           │ Operates on mesh vertices
┌──────────────────────────▼─────────────────────────────┐
│  Layer 1: Discrete Differential Geometry & Topology    │
│  - Mesh.hpp, TopologyData.hpp, TriangleInfo.hpp        │
│  - Dual areas, 1-ring stencils, hinge dihedral angles  │
└────────────────────────────────────────────────────────┘
```

---

## 2. Frozen Engine vs. Active Research Files

You **do not need to understand or modify all 55+ files**. 95% of the codebase is the numerical solver and geometry engine that can remain completely untouched.

| Component | Files | Status | When to Touch |
| :--- | :--- | :---: | :--- |
| **Growth Kinematics** | `src/libshell/ZigZagGrowth.hpp`<br>`src/libshell/GrowthHelper.hpp` | **ACTIVE** | **Any time you modify the growth law, temporal decay, or footprint shape.** |
| **Simulation Driver** | `src/simulations/Sim_Bilayer_Growth.cpp` | **ACTIVE** | **When introducing time steps ($\Delta t$), velocity ($v$), or cycle loops.** |
| **Optimization Solver** | `NativeHybridShellSolver.hpp`<br>`ReferenceHybrid/*` | **FROZEN** | Never. Provides guaranteed machine-precision convergence ($\le 1.5 \times 10^{-14}$). |
| **Linear Algebra** | `SparseDirectFactorization.hpp` | **FROZEN** | Never. Direct Cholesky solver with CHOLMOD. |
| **Shell Energy Physics** | `CombinedOperator_Parametric.*`<br>`BendingOperator_Parametric.*` | **FROZEN** | Never. Discrete shell membrane and bending elasticity. |
| **Mesh Data Structures** | `Mesh.hpp`, `TopologyData.hpp` | **FROZEN** | Never. Half-edge and face-vertex topology. |

---

## 3. How the Growth Kinematics Work

In the English wheel forming process, rolling deformation is modeled as an **eigenstrain growth tensor** $\mathbf{G}$ that modifies the target first fundamental form (metric) $\mathbf{a}_{\text{target}}$ of the bilayer shell:

$$\mathbf{a}_{\text{target}} = \mathbf{G}^T \mathbf{a}_{\text{ref}} \mathbf{G}$$

### The Contact Footprint
In `src/libshell/ZigZagGrowth.hpp`, for each triangular element $i$ with centroid $\mathbf{c}_i = \frac{1}{3}(\mathbf{u}_0 + \mathbf{u}_1 + \mathbf{u}_2)$ in the 2D sheet domain:
1. The perpendicular distance $d = \text{dist}(\mathbf{c}_i, \text{segment})$ to the current toolpath strip is evaluated.
2. If $d \le \frac{1}{2} w$ (where $w$ is the roller track width, default $2\text{ mm}$):
   - The element receives top-layer expansion $g_{\text{top}}$ and bottom-layer expansion $g_{\text{bottom}}$.
   - The expansion is oriented along the roller travel angle $\theta_{\text{pass}}$.
   - An orthotropy ratio $\text{ortho} \in [-1, +1]$ controls longitudinal vs. transverse expansion.

---

## 4. Guide: Implementing Temporal & Rate-Dependent Effects

If you want to investigate whether the analytical model can capture the **temporal aspect** (e.g., rolling speed, time-dependent plastic accumulation, or inter-pass relaxation), here is the exact blueprint.

### Concept 1: Strain-Rate Dependence (Roller Speed $v$)
Plastic deformation in rolling depends on the effective strain rate $\dot{\varepsilon} \approx v / w$:
$$g(v) = g_0 \left[ 1 + C \ln \left( \frac{v}{v_0} \right) \right]$$
- **Where to implement**: In `ZigZagGrowth.hpp`, add `double roller_speed_mps = 0.1` to `struct Params`.
- Multiply `s.gtop` and `s.gbot` by the rate multiplier when inside the contact footprint.

### Concept 2: Inter-Pass Viscoelastic / Creep Relaxation (Time $\Delta t$)
Between successive roller passes, residual stresses partially relax over the elapsed idle time $\Delta t$:
$$g_{\text{retained}}(t) = g_{\text{plastic}} + g_{\text{transient}} \cdot \exp\left(-\frac{\Delta t}{\tau}\right)$$
where $\tau$ is the characteristic relaxation time of the sheet material.
- **Where to implement**: In `Sim_Bilayer_Growth.cpp`, track the cumulative simulation time:
  ```cpp
  double current_time_s = 0.0;
  double dt_pass = strip_length / roller_velocity;
  current_time_s += dt_pass;
  ```
- Pass `current_time_s` or cycle timestamps into `applyZigZagGrowth(...)`.

### Concept 3: History-Dependent Hardening (Pass Accumulation)
Instead of a static Voce decay per cycle, accumulate plastic work locally per element:
```cpp
// In ZigZagGrowth.hpp:
if (d <= half_w) {
    passCountFaces(i) += 1;
    double local_n = passCountFaces(i);
    double hardening = floor + (1.0 - floor) * std::exp(-beta * local_n);
    growthRates_t(i) += gtop_incremental * hardening;
}
```

---

## 5. Adding New Parameters to the CLI & Config JSON

When you add new temporal parameters (e.g. `-roller_speed`, `-relaxation_time`):

1. **Read from CLI in `Sim_Bilayer_Growth.cpp`**:
   ```cpp
   const double roller_speed = parser.parse<double>("-roller_speed", 0.05); // m/s
   const double relaxation_time = parser.parse<double>("-relaxation_time", 10.0); // s
   ```
2. **Add to JSON Config (`calibration/configs/run6.json`)**:
   ```json
   "operation": {
       "roller_speed_mps": 0.05,
       "relaxation_time_s": 10.0
   }
   ```
3. **Pass into `zigzag::applyZigZagGrowth`**:
   Assign the parameters to `zigzag::Params zz` before computing the growth field.

The underlying solver (`native_hybrid`) will automatically solve for the resulting 3D equilibrium state without any modifications.
