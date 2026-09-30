# Bilayer Zigzag Trajectory Library

This directory contains canonical, verified input trajectories (`.json`) for the Bilayer SVK shell equilibrium simulator (`bin/shell -sim bilayer_growth -growth_type zigzag_sequence`).

Each trajectory defines sequential rolling passes across top and/or bottom layers, parameterized by toolpath bounds, stroke directions, pitch, growth increments ($\Delta g$), and directional orthotropy ($\text{ortho}$).

---

## Curated Canonical Examples

| File | Numbered Alias | Description | Benchmark Use Case |
| :--- | :--- | :--- | :--- |
| `putong_1step.json` | `01_putong_1step.json` | 1-step top rolling pass ($\Delta g = 0.003$) | Rapid smoke test, CI unit verification, pipeline sanity |
| `putong_traj30.json` | `02_putong_traj30.json` | 30-step sequential trajectory | Full parameter calibration against experimental English wheel scans |
| `zigzag_cont30.json` | `03_zigzag_cont30.json` | 30-cycle continuation study | Standard production multi-cycle benchmark (`scripts/run_zigzag_certified.sbatch`) |
| `zigzag_cont60.json` | `04_zigzag_cont60.json` | 60-cycle high-resolution continuation | Mesh and step-size convergence reference |
| `zigzag_sequence.json` | `05_orthogonal_2cycle_iso.json` | 2-cycle orthogonal toolpaths ($0^\circ, 90^\circ$), isotropic SVK | Multi-pass cross-wheeling wrinkling benchmark |
| `orthogonal_same_side_clamped_dense_centerpeak_ortho_m05.json` | `06_orthogonal_2cycle_ortho.json` | 2-cycle orthogonal toolpaths with negative orthotropy ($\text{ortho} = -0.5$) | Certified orthotropic post-buckling benchmark ($E \approx 3.8206 \times 10^{-11}\text{ J}$) |
| `zigzag_large.json` | `07_zigzag_large.json` | High-strain / deep post-buckling trajectory | Large-deformation stability and solver certification |

---

## How to Run a Trajectory

Run via `bin/shell`:
```bash
./bin/shell \
  -sim bilayer_growth \
  -case custom \
  -geometry rectangle \
  -lx 0.13 -ly 0.16 \
  -res 0.008 \
  -h_total 0.0005 \
  -growth_type zigzag_sequence \
  -cycle_file trajectories/01_putong_1step.json \
  -minimizer hlbfgs \
  -tol 1e-12 \
  -hessian_threads 4 \
  -basename test_run \
  -export_stl true
```

Or submit a cluster batch job:
```bash
sbatch scripts/run_zigzag_certified.sbatch trajectories/03_zigzag_cont30.json 0.008 prod
```

---

## Historical & Parameter-Sweep Variants

Additional one-off variations (e.g. radial patterns, opposite-side passes, square domains, and reverse toolpaths) generated during earlier parametric studies are preserved under `trajectories/archive/`.
