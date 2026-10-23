# Prototype Directional Mechanism Comparison: Coupon Benchmark

## Executive Summary

To investigate the experimental discrepancy between forward–forward (`fwd_fwd`) and forward–backward (`fwd_rev`) rolling schedules, two candidate physical mechanisms were formulated, implemented in C++, and evaluated on the minimal 6-strip coupon benchmark ($127 \times 152.4 \times 0.6$ mm, $g_{\text{top}} = 0.002$, $\text{ortho} = 0.8$):

1. **Model A (Directional Arc-Length Drift / Rolling Traction)**: Introduces an along-stroke linear drift $\eta (s - 0.5) \mathbf{t} \otimes \mathbf{t}$ into the target metric, representing frictional shear and material accumulation along the rolling stroke.
2. **Model B (Deformed 3D Slope / Curvature Feedback)**: Scales local growth rate by $1 + \gamma (\mathbf{t} \cdot \nabla z)$, coupling plastic deformation to the intermediate deformed geometry produced by preceding passes.

Both models break the nematic $\pi$-symmetry of the baseline code and produce distinct, physically consistent deformations between forward–forward and forward–backward passes.

---

## Benchmark Results Summary

All simulations used the `native_hybrid` exact-Hessian solver to gradient norm $< 5 \times 10^{-14}$ on Quest compute nodes. Aligned shape differences were computed using the optimal rigid-body Kabsch algorithm.

| Scenario | Model Mechanism | Cadence | Max Target Metric Diff | Aligned Shape RMS ($\mu$m) | Aligned Shape Max ($\mu$m) | $\Delta \kappa_1$ ($\text{m}^{-1}$) | $\Delta Z$ (mm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Audit** | Unmodified Nematic $\pi$-Symmetry | Single Solve | $8.48 \times 10^{-8}$ | **0.005** (5 nm) | 0.018 | $0.000003$ | 0.000 |
| **Baseline Audit** | Unmodified Nematic $\pi$-Symmetry | Pass-by-Pass | $8.48 \times 10^{-8}$ | **0.020** (20 nm) | 0.048 | $0.000003$ | 0.000 |
| **Model A** | Directional Drift ($\eta = 0.40$) | Single Solve | $3.30 \times 10^{-7}$ | **134.42** (0.134 mm) | **341.77** (0.342 mm) | 0.014459 | 8.526 |
| **Model A** | Directional Drift ($\eta = 0.40$) | Pass-by-Pass | $3.30 \times 10^{-7}$ | **134.42** (0.134 mm) | **341.81** (0.342 mm) | 0.000421 | 0.150 |
| **Model B** | Geometric Slope Feedback ($\gamma = 25$) | Pass-by-Pass | $1.18 \times 10^{-5}$ | **7,566.41** (7.57 mm) | **17,624.44** (17.62 mm) | **0.709471** | **10.196** |

---

## Key Physical & Computational Findings

### 1. Model A (Directional Drift) Operates Effectively Under Single-Solve Cadence (`final_only`)
- Because Model A parameterizes the along-stroke gradient via the material arc-length coordinate $s \in [0, 1]$, the target metric $\bar{\mathbf{a}}(u, v)$ explicitly records the path orientation and travel direction.
- Inverting traversal direction ($b \to a$) flips the sign of the gradient along the strip, resulting in a genuine metric difference of $3.30 \times 10^{-7}$ and an aligned shape difference of **$134.42\ \mu\text{m}$ (RMS)** and **$341.8\ \mu\text{m}$ (Max)**.
- **Computational Impact**: Model A requires **zero intermediate solves**. Its directional asymmetry is fully captured even when optimizing all toolpaths at once in a single final solve (`-sequence_solve_mode final_only`).

### 2. Model B (Geometric Slope Feedback) Drives Strong Non-Linear Curvature Bifurcations
- In Model B, pass 1 establishes an initial out-of-plane dome ($Z_{\text{span}} \approx 45.7$ mm).
- When pass 2 traverses this dome, the forward direction climbs from $z=0$ up to $z_{\max}$ and down, whereas reversing traversal reverses the climb/descent sequence relative to the strip sequence.
- Because $\nabla z$ is non-zero, $\mathbf{t} \cdot \nabla z$ modulates the effective plastic strain. Climbing the slope enhances growth, while descending reduces it.
- This creates an enormous curvature discrepancy: $\kappa_1$ changes from $-6.364\ \text{m}^{-1}$ (`fwd_fwd`) to $-5.654\ \text{m}^{-1}$ (`fwd_rev`), yielding a **$7.57$ mm RMS shape difference** and **$10.2$ mm $Z$-span difference**.
- **Computational Impact**: Model B fundamentally couples growth to the deformed intermediate geometry $q_k$. It therefore **requires intermediate equilibrium solves** (e.g. after each major pass or group of passes) to update $\nabla z$.

---

## Physical Consistency Assessment

- Both models capture distinct aspects of real physical wheeling / peening processes:
  - **Model A** reflects the *microscopic/contact-shear level*: the wheel pushes material ahead of the contact patch in the direction of rolling (material pile-up / forward extrusion).
  - **Model B** reflects the *macroscopic/structural level*: rolling over an already-curved sheet alters the local contact curvature, penetration depth, and effective contact patch area.
- In experimental practice, both effects are likely active simultaneously.

---

## Artifact Provenance
- Benchmark script: `python/benchmark_directional_prototype.py`
- Simulation output data: `run/benchmark_directional_prototype/prototype_directional_summary.json`
- C++ implementation:
  - `src/libshell/ZigZagGrowth.hpp` (`TopProfileMode::DirectionalDrift`)
  - `src/libshell/ZigZagSequenceGrowth.hpp` (`directional_drift_coeff`, `reverse_traversal`)
  - `src/simulations/Sim_Bilayer_Growth.cpp` (`-sequence_slope_feedback`, in-plane $Du^{-1} dz$ solver)
