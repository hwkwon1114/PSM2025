# PSM2025-zigzag: Optimal Anisotropic Growth Calibration in Bilayer Forming

This repository contains the C++ discrete differential geometry shell solver, Bayesian optimization calibration framework, and validation data accompanying:

> **Optimal Anisotropic Growth Calibration in Bilayer Zigzag Forming**

---

## Overview

We model the plastic deformation induced by an English wheel / sequential roller toolpath as an anisotropic bilayer eigenstrain shell problem. The repository provides:
1. **High-Performance C++ Mechanics Solver**: A discrete differential geometry shell formulation using a native hybrid L-BFGS warmup + exact Newton optimizer with positive-definite Cholesky factorizations (CHOLMOD/SuiteSparse) terminating at machine precision ($1.5 \times 10^{-14}$ gradient norm).
2. **Bayesian Optimization Framework**: A 4-parameter anisotropic growth law ($g_{\text{top}}, g_{\text{bottom}}, \text{ortho}_{\text{top}}, \text{ortho}_{\text{bottom}}$) calibrated against 3D optical scans of formed sheet metal (Run 6) with held-out generalization to different toolpaths (Run 9).
3. **Rigorous Geometric Loss**: Material-frame rigid alignment ($SO(3)$ Kabsch) and discrete shell dual-area Voronoi quadrature weighting ($L_{\text{sym}}$).
4. **Reproducibility Suite**: One-click scripts and interactive Jupyter notebooks reproducing all paper figures and quantitative benchmarks.

---

## Installation & Environment Setup

### 1. Conda Environment
Create and activate the environment specified in `environment.yml`:
```bash
conda env create -f environment.yml
conda activate psm2025-zigzag
```

### 2. C++ Compiler & Libraries
Ensure a C++17 compliant compiler (`gcc` $\ge 9$ or `clang` $\ge 11$), CMake ($\ge 3.24$), Ninja, Eigen ($\ge 3.4$), and SuiteSparse are installed.

---

## Building the C++ Shell Solver

Configure and build the native shell solver using CMake and Ninja:
```bash
cmake -B build_cmake -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake -j 8
```
This generates:
- `./bin/testshell`: C++ unit and regression test runner.
- `./bin/sim_bilayer_growth`: Production CLI for bilayer sequence simulations.

### Running C++ Unit Tests
Run the complete C++ test suite (99 passing tests across 36 suites):
```bash
./bin/testshell
```

---

## Running the Python Verification Suite

Run the 56 automated unit and regression tests covering geometry processing, $SO(3)$ alignment, dual-area quadrature, and Bayesian optimization workflows:
```bash
python -m unittest discover -s calibration/tests/ -p "test_*.py" -v
```

---

## One-Click Paper Figure Reproduction

To reproduce the paper's primary figures and quantitative tables from the optimal calibrated parameters (Attempt 209), run:
```bash
./scripts/reproduce_paper_figures.sh --out-dir paper_figures --force
```
Or directly using Python:
```bash
python scripts/reproduce_paper_figures.py --out-dir paper_figures --force
```

### Generated Deliverables in `paper_figures/`:
1. **2D Six-Panel Toolpath & Residual Figure**:
   - `run6_run9_toolpath_deformation_residual.png` (High-res raster)
   - `run6_run9_toolpath_deformation_residual.pdf` (Publication vector)
   - Displays the wheeling path, continuous out-of-plane $z(x, y)$ deflection, and signed height residuals $\Delta z = z_{\text{sim}} - z_{\text{scan}}$ for both Run 6 and Run 9.
2. **3D Isometric & Profile Comparison Figure**:
   - `run6_run9_3d_deformation_residual.png` / `.pdf`
   - Displays true 3D spatial curvature, saddle warpage, and end-on midline profile overlays.
3. **Structured Provenance**:
   - `provenance.json` & `caption.md` recording exact parameter values, input hashes, and metric breakdowns.

---

## Interactive Jupyter Demonstration

An interactive demonstration notebook is provided in `notebooks/`:
```bash
code notebooks/demo_zigzag_calibration.ipynb
```
The notebook contains:
- Interactive walkthrough of the anisotropic bilayer formulation.
- Step-by-step evaluation of the optimal parameters.
- Inline visualization of the 2D toolpaths and 3D deformation shapes.
- Pre-rendered outputs ready for immediate review.

---

## Optimal Calibrated Parameters & Quantitative Results

Calibrated on **Run 6** (29 passes, $130 \times 130\text{ mm}$ sheet) across 156 Bayesian optimization evaluations and independently evaluated on **Run 9** (33 passes):

| Parameter | Optimal Value (Attempt 209) | Physical Role |
| :--- | :--- | :--- |
| **$g_{\text{top}}$** | $2.8835 \times 10^{-4}$ | Top layer longitudinal growth |
| **$g_{\text{bottom}}$** | $1.4588 \times 10^{-5}$ | Bottom layer growth |
| **$\text{ortho}_{\text{top}}$** | $-0.3989$ | Directional anisotropy |
| **$\text{ortho}_{\text{bottom}}$** | $+0.9603$ | Transverse Poisson coupling |

### Quantitative Error Metrics:
- **Run 6 (Training Sheet)**:
  - Symmetric Dual-Area Weighted RMS: **$1.1823\text{ mm}$**
  - Single-Sided Forward RMS: **$0.6497\text{ mm}$**
  - Single-Sided Backward RMS: **$1.5406\text{ mm}$**
- **Run 9 (Independent Validation Sheet)**:
  - Symmetric Dual-Area Weighted RMS: **$1.3725\text{ mm}$**
  - Single-Sided Forward RMS: **$1.0582\text{ mm}$**
  - Single-Sided Backward RMS: **$1.6272\text{ mm}$**

---

## Repository Directory Structure

```text
PSM2025-zigzag/
├── CMakeLists.txt              # Top-level C++ build configuration
├── environment.yml             # Conda environment definition
├── README.md                   # This document
├── src/
│   ├── libshell/               # C++ discrete shell equilibrium solver & hybrid Newton
│   └── simulations/            # Bilayer growth simulation executable
├── test/
│   └── testshell/              # C++ unit tests
├── calibration/
│   ├── analysis/               # Figure generation & post-processing CLI tools
│   ├── configs/                # Configuration specifications (Run 6 and Run 9)
│   ├── data/raw/               # Raw experimental scan STLs and toolpath CSVs
│   ├── ewcal/                  # Core Python evaluation & geometry libraries
│   └── tests/                  # Python calibration unit & integration tests
├── scripts/
│   ├── reproduce_paper_figures.sh  # One-click reproduction script
│   └── reproduce_paper_figures.py  # Python reproduction CLI
└── notebooks/
    └── demo_zigzag_calibration.ipynb # Interactive demonstration notebook
```

---

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
