# PSM2025-zigzag: Optimal Anisotropic Growth Calibration in Bilayer Forming

This repository contains the C++ discrete differential geometry shell solver, Bayesian optimization calibration framework, and validation data accompanying:

> **Optimal Anisotropic Growth Calibration in Bilayer Zigzag Forming**

---

## Overview

We model the plastic deformation induced by an English wheel / sequential roller toolpath as an anisotropic bilayer eigenstrain shell problem. The repository provides:
1. **High-Performance C++ Mechanics Solver**: A discrete differential geometry shell formulation using the native hybrid L-BFGS warmup + exact Newton optimizer with positive-definite Cholesky factorizations (CHOLMOD/SuiteSparse) terminating at a gradient-norm gate (5e-14 in the shared configuration `shellsim/configs/default_adaptive_cutback.json`). Legacy solver branches have been pruned to streamline future temporal growth modeling.
2. **Bayesian Optimization Framework**: The selected paper calibration is Run 1, Attempt 209, with fixed hardening and four layer-growth/orthotropy parameters. Later joint and pairwise searches are retained as separate investigations.
3. **Geometric Loss**: A fixed material-reference frame and a symmetric, shared-footprint surface discrepancy; the scan is not rigidly fitted to each prediction.
4. **Research Records**: The selected Attempt 209 calibration and figure provenance are described in [`docs/calibration/bayesian_calibration/convention_corrected_investigation.md`](docs/calibration/bayesian_calibration/convention_corrected_investigation.md). Some older documentation was deleted at the researcher's direction; see [`docs/README.md`](docs/README.md).

---

## Repository Layout

Three parts:

```text
PSM2025-zigzag/
├── calibration/    1. Calibration, exploratory searches and publication figures
│   ├── ewcal/          shared forward model, geometry, loss and BO library
│   ├── bo/             joint controller, campaign configurations and historical drivers
│   ├── analysis/       post-processing and figure scripts, including historical analyses
│   ├── configs/        configurations with historical case identifiers (see case_labels.py)
│   ├── data/           experimental scans and toolpaths
│   ├── paper_figures/  selected Attempt 209 figures and historical figure bundles
│   ├── notebooks/      demonstration notebook
│   └── tests/
├── shape_design/   Frozen Attempt 209 exploratory shapes and collaborator exports
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

## Current calibration and figures

**Selected paper figures:** [`calibration/paper_figures/attempt209_publication_clean/`](calibration/paper_figures/attempt209_publication_clean/README.md). These use frozen Attempt 209 parameters: historical optimization/GP scores and later corrected-convention shape evaluations are explicitly distinguished. Runs 2 and 3 are exploratory transfers, not blind validation. See the [investigation record](docs/calibration/bayesian_calibration/convention_corrected_investigation.md) for the decision, numerical values and public Run 1/2/3 mapping.

Reproduce all seven 16 pt figures without simulations, into a new directory:
```bash
python calibration/analysis/reproduce_attempt209_publication.py --out-dir run/attempt209_paper_reproduction
```
Existing outputs are refused. The older `reproduce_paper_figures.py` and demonstration notebook reproduce a historical pilot, not this delivered bundle.

**Manuscript draft:** `docs/calibration/bayesian_calibration/attempt209_manuscript_draft.tex` matches the selected Attempt 209 figure set and distinguishes the historical optimization objective from later shape evaluation. The researcher must check editorial notes and references before submission. `calibration_manuscript_draft.tex` and `calibration/paper_figures/current_joint_calibration/` describe a different joint fit and remain historical.

Later scalar, Pareto and pairwise searches remain under `calibration/bo/` and `run/`; they do not replace the paper selection. Archived source paths are compatibility links to `legacy/` snapshots. Changes to reviewed code require applicable authorization and re-review before scientific execution.

The separate [shape-design protocol](docs/shape_design/attempt209_double_curvature.md) describes exploratory 0.61 mm sheet predictions and collaborator exports. These extrapolate beyond calibration and are not machine-ready programs.

---

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
