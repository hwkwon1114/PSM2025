# VAL-RUN-2 (REPLICATE-HW3) — Forward prediction of collaborator experiment "Val-Run 2" (30 passes) with Attempt 209

**Status:** COMPLETE 2026-11-09. Adopted as `Val-Run 2` under researcher decision (DECISIONS.md, 2026-11-09). **Drafted:** 2026-11-09.

## Question

If the 30-pass localized wheeling toolpath in `shape_design/data/Run3_toolpath.csv` is applied in the frozen Attempt 209 model, does the predicted shape reproduce the scanned physical specimen `shape_design/data/Run_3.ply`?

This is a **blind forward prediction**. Nothing is fitted to this scan. All 30 passes are single-sided (front face) in the central half-width patch.

## Decision driven

- **Shape reproduced** (symmetric surface RMS $< 0.85\text{ mm}$ and amplitude within factor of 2): Attempt 209 transfers cleanly to localized single-sided wheeling on this $0.61\text{ mm}$ sheet.
- **Sign pattern reproduced, amplitude off:** Captures mechanism but extrapolates thickness or pitch.
- **Sign pattern not reproduced:** Formulation fails on localized high-pass patches.

## Inputs (read-only)

**Toolpath.** `shape_design/data/Run3_toolpath.csv` has 30 passes across a single patch:
- Width span: $x \in [-27.7425, 30.0000]\text{ mm}$ (centered at $u_0 = 1.12875\text{ mm}$, 14 strips).
- Feeding span: $y \in [0.0000, 55.0000]\text{ mm}$ (half-width from centerline to 55 mm, $v_0 = 27.5\text{ mm}$).
- $lv = 55.0\text{ mm}$, pitch $p = 4.8118765\text{ mm}$, inclination angle $\alpha = 5.0000^\circ$.
- Odd passes forward (`reverse_traversal: true`), even passes reverse (`reverse_traversal: false`).

**Scan.** `shape_design/data/Run_3.ply`:
- 427,687 vertices, 852,484 triangle faces.
- In-plane PCA extent: $221.3 \times 126.4\text{ mm}$, height range $\approx 4.3\text{ mm}$ (56 debris points removed).

**Model.** Frozen Attempt 209 parameters:
- $g_{\text{top}} = 2.8835 \times 10^{-4}, g_{\text{bot}} = 1.4588 \times 10^{-5}, \text{ortho}_{\text{top}} = -0.3989, \text{ortho}_{\text{bottom}} = 0.9603$
- Voce hit-count hardening, $\beta = 0.22314$, face-shared, plastic hits.
- Native hybrid backend, final-only equilibrium, $5 \times 10^{-14}$ gradient gate.

## Acceptance & Results (2026-11-09)

- **Solver Execution:** Slurm job `7828880` on compute node `qnode0008` accepted in **90.1 seconds** (only 13 Hessian evaluations, 0 cutbacks, $\|\nabla E\| = 4.09 \times 10^{-14} < 5.0 \times 10^{-14}$).
- **Hit Check:** **0 mismatched faces** out of 13,640 faces against independent CSV leg hit prediction.
- **Surface RMS:** **$0.5299\text{ mm}$ symmetric surface RMS** (best orientation: `turn_over_u_axis`).
- **Profile RMS:** **$0.5240\text{ mm}$ vertical profile RMS** ($\Delta y$).
- **Amplitude:** Profile peak-to-valley ratio prediction/scan = **$0.739$** ($2.58\text{ mm}$ predicted vs $3.49\text{ mm}$ scanned).
- **Curvatures:** 100% sign agreement across all regions.
- **Classification:** Local synclastic dome double-curvature ($K > 0$) within the worked patch, embedded within a predominantly single-curvature cylindrical trough globally (10:1 deflection along length vs width).

## Code and Outputs

- Runner: `shape_design/run_hw3.py`
- Generator: `shape_design/replicate_hw3.py`
- Visualizer: `shape_design/plot_hw3_3d.py`, `shape_design/plot_hw3_comparison.py`
- Outputs in `run/shape_design/hw_run3/solve_mesh2/`

