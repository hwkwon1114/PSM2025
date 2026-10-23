# PSM2025-zigzag: Optimal Anisotropic Growth Calibration in Bilayer Forming

This repository contains the C++ discrete differential geometry shell solver, Bayesian optimization calibration framework, and validation data accompanying:

> **Optimal Anisotropic Growth Calibration in Bilayer Zigzag Forming**

---

## Overview

We model the plastic deformation induced by an English wheel / sequential roller toolpath as an anisotropic bilayer eigenstrain shell problem. The repository provides:
1. **High-Performance C++ Mechanics Solver**: A discrete differential geometry shell formulation using the native hybrid L-BFGS warmup + exact Newton optimizer with positive-definite Cholesky factorizations (CHOLMOD/SuiteSparse) terminating at a gradient-norm gate (5e-14 in the shared configuration `shellsim/configs/default_adaptive_cutback.json`). Legacy solver branches have been pruned to streamline future temporal growth modeling.
2. **Bayesian Optimization Framework**: A 4-parameter anisotropic growth law ($g_{\text{top}}, g_{\text{bottom}}, \text{ortho}_{\text{top}}, \text{ortho}_{\text{bottom}}$) calibrated against 3D optical scans of formed sheet metal (Run 6) with held-out generalization to different toolpaths (Run 9).
3. **Rigorous Geometric Loss**: Material-frame rigid alignment ($SO(3)$ Kabsch) and discrete shell dual-area Voronoi quadrature weighting ($L_{\text{sym}}$).
4. **Reproducibility Suite**: One-click scripts and interactive Jupyter notebooks reproducing all paper figures and quantitative benchmarks.

---

## Repository Layout

Three parts:

```text
PSM2025-zigzag/
├── calibration/    1. Calibrating the growth/hardening model by Bayesian optimization (Run 6, Run 9)
│   ├── ewcal/          forward model, geometry, loss and BO library
│   ├── bo/             campaign drivers (Run 6 BO, transfer to Run 9)
│   ├── analysis/       post-processing and figure scripts
│   ├── configs/        Run 6/7/9/10 configurations (solver settings are shared, see shellsim/)
│   ├── data/           experimental scans and toolpaths
│   ├── paper_figures/  one-click reproduction of the paper figures
│   ├── notebooks/      demonstration notebook
│   └── tests/
├── dome_bo/        2. Finding a good dome toolpath by Bayesian optimization (square sheet)
│   └── tests/          campaign, objective, mesh-study and verification code
├── shellsim/       3. Using the solver: toolpath -> JSON -> simulation, in a notebook
│   ├── toolpath.py, simulate.py
│   ├── configs/default_adaptive_cutback.json   the one solver configuration
│   └── examples/       notebook, ready-made trajectories, SVG toolpath input
├── src/, test/, cmake/, CMakeLists.txt   C++ shell solver and its tests
├── docs/           reports and records (start at docs/README.md)
├── legacy/         earlier studies kept for reproduction (rectangular-sheet dome stages, ablations, benchmarks)
└── environment.yml
```

Build the solver (below), then see `shellsim/README.md` to run your own toolpath; `docs/README.md` indexes the reports.

---

---

## Installation & Environment Setup

### 1. Conda Environment
Create and activate the environment specified in `environment.yml`:
```bash
conda env create -f environment.yml
conda activate psm2025-zigzag
```

### 2. C++ Compiler & Libraries
Ensure a C++23 compliant compiler (`gcc` $\ge 13$ or `clang` $\ge 16$), CMake ($\ge 3.24$), Ninja, Eigen ($\ge 3.4$), SuiteSparse, and `nlohmann_json` are installed.

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
Run the complete C++ test suite:
```bash
./bin/testshell
```

---

## Running the Python Verification Suite

Run the automated unit and regression tests covering geometry processing, $SO(3)$ alignment, dual-area quadrature, and Bayesian optimization workflows:
```bash
python -m unittest discover -s calibration/tests/ -p "test_*.py" -v
```

---

## Reproducing the calibration figures

```bash
python calibration/paper_figures/reproduce_paper_figures.py --force   # or ./calibration/paper_figures/reproduce_paper_figures.sh --force
```
This regenerates the Run 6/Run 9 toolpath/deformation/residual figures (2D and 3D, PNG and PDF) from the Attempt 209 parameters into `calibration/paper_figures/`, with `provenance.json` and `caption.md` (parameters, input hashes, metrics). `calibration/notebooks/demo_zigzag_calibration.ipynb` walks through the same evaluation with pre-rendered outputs.

---

## Optimal Calibrated Parameters & Quantitative Results

Calibrated on **Run 6** (29 passes, $200 \times 300\text{ mm}$ sheet) across 156 Bayesian optimization evaluations and independently evaluated on **Run 9** (33 passes):

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

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
