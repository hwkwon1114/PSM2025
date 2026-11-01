# PSM2025: Discrete Differential Geometry Shell Solver for Bilayer Forming

A high-performance C++ discrete differential geometry (DDG) thin-shell mechanics solver for simulating sheet deformation induced by sequential roller / English wheel toolpaths via anisotropic bilayer eigenstrain growth.

---

## Features

- **DDG Thin-Shell Formulation**: Models combined membrane stretching and bending under anisotropic eigenstrain growth fields.
- **Native Hybrid Optimizer**: Combines L-BFGS warmup with an exact Newton corrector using positive-definite Cholesky factorization (SuiteSparse/CHOLMOD) and adaptive step cutback.
- **Strain Hardening**: Natively tracks repeated roller passes on triangular faces using an empirical Voce-decay model:
  $$q(n) = f + (1-f)e^{-\beta n}$$
- **Python Interface (`shellsim`)**: High-level Python API to define toolpaths, generate sequence configurations, run the solver, and inspect deformed 3D geometries.

---

## Installation & Build

### 1. Environment Setup

Create and activate the environment using Conda:
```bash
conda env create -f environment.yml
conda activate psm2025-zigzag
```

Prerequisites:
- C++23 compliant compiler (`gcc` $\ge 13$ or `clang` $\ge 16$)
- CMake ($\ge 3.24$) and Ninja
- SuiteSparse, Eigen ($\ge 3.4$), and `nlohmann_json`

### 2. Building the Solver

Build the C++ binaries using CMake and Ninja:
```bash
cmake -B build_cmake -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake -j 8
```

This compiles:
- `./bin/shell`: The core bilayer shell simulation binary.
- `./bin/testshell`: C++ unit and regression test suite.

### 3. Verify the Build
Run the C++ unit test suite:
```bash
./bin/testshell
```

---

## Running Simulations

### Option A: Via Python (`shellsim`)

The Python module `shellsim` provides an easy interface to script toolpaths, run the solver, and inspect results.

```python
from shellsim import toolpath, simulate

# 1. Define a zigzag toolpath sequence
seq = toolpath.sequence([
    ('A', toolpath.zigzag(
        lv_mm=140,       # Track length (mm)
        alpha_deg=9.13,   # Strip angle (deg)
        n_strips=10,      # Number of parallel passes
        width_mm=10,      # Strip width (mm)
        gtop=0.003,       # Top-layer growth strain
        gbot=0.0          # Bottom-layer growth strain
    ))
])

# 2. Run simulation on a 200 x 200 x 0.77 mm sheet
state = simulate.run(
    seq,
    casefolder='run/example_case',
    lx=0.1,          # Half-width in x (meters) -> 200 mm total
    ly=0.1,          # Half-length in y (meters) -> 200 mm total
    res=0.01,        # Mesh resolution (meters) -> 10 mm edge length
    h_total=0.00077  # Total sheet thickness (meters)
)

# 3. Extract deformed coordinates and height field
x, y, z = simulate.height(state)
print(f"Deformed sheet vertical displacement range: {y.ptp():.3f} mm")
```

See `shellsim/examples/notebooks/simulate_toolpath.ipynb` for an interactive end-to-end demonstration.

---

### Option B: Direct C++ Command Line (`bin/shell`)

Before running `bin/shell`, set the BLIS single-thread environment to prevent thread contention with OpenMP:

```bash
export BLIS_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT
```

Run the solver directly with a sequence file:

```bash
./bin/shell \
  -sim bilayer_growth \
  -case custom \
  -geometry rectangle \
  -lx 0.1 -ly 0.15 \
  -res 0.01 \
  -h_total 0.00077 \
  -cycle_file path/to/sequence.json \
  -sequence_solve_mode final_only \
  -equilibrium_solver native_hybrid \
  -equilibrium_grad_tol 5e-14 \
  -native_hybrid_warmup_attempts 2000 \
  -native_hybrid_grad_tol 5e-14 \
  -native_hybrid_seconds 3600 \
  -sequence_adaptive true \
  -hessian_threads 8 \
  -export_stl false
```

#### Key Solver Arguments:
| Argument | Description | Default / Recommended |
|---|---|---|
| `-sim` | Simulation type | `bilayer_growth` |
| `-geometry` | Mesh geometry | `rectangle` or `disk` |
| `-lx`, `-ly` | Half-dimensions of rectangle (meters) | e.g. `0.1`, `0.15` ($200 \times 300\text{ mm}$) |
| `-h_total` | Sheet thickness (meters) | e.g. `0.00077` ($0.77\text{ mm}$) |
| `-res` | Target mesh edge length (meters) | e.g. `0.01` ($10\text{ mm}$) |
| `-cycle_file` | Path to toolpath definition JSON | `sequence.json` |
| `-sequence_solve_mode` | Equilibrium solve cadence | `final_only` (fast, lumped) or `every_cycle` |
| `-equilibrium_solver` | Non-linear solver algorithm | `native_hybrid` (L-BFGS + Newton) |
| `-equilibrium_grad_tol` | Newton gradient norm stopping tolerance | `5e-14` |
| `-hessian_threads` | OpenMP threads for Hessian assembly | `8` (adjust for CPU) |

Shared default configuration settings are stored in `shellsim/configs/default_adaptive_cutback.json`.

---

## Toolpath Input Format (`sequence.json`)

The solver reads toolpath geometries and material growth parameters from a structured JSON file:

```json
{
  "schema_version": 1,
  "units": {
    "length": "mm",
    "angle": "deg",
    "growth": "engineering_strain"
  },
  "hardening": {
    "model": "voce_decay",
    "history_variable": "hit_count",
    "beta": 0.223,
    "floor": 0.0,
    "scope": "face_shared",
    "count_rule": "plastic_hits"
  },
  "boundary_conditions": {
    "enabled": false
  },
  "toolpaths": [
    {
      "id": "pass_1",
      "repeat": 1,
      "operation": {
        "type": "zigzag",
        "lv_mm": 130.0,
        "alpha_deg": -1.0,
        "n_strips": 68,
        "width_mm": 2.0,
        "shift_uv_mm": [0.0, 0.0],
        "rotation_deg": 0.0,
        "gtop": 0.000288,
        "gbot": 1.458e-05,
        "ortho_top": -0.399,
        "ortho_bottom": 0.960
      }
    }
  ]
}
```

### Parameter Definitions:
- **`hardening`**:
  - `beta` ($\beta$): Hardening decay rate per pass.
  - `floor` ($f$): Hardening floor factor ($0 \le f \le 1$). If $f=1$, growth does not decay with repeated passes.
- **`operation`**:
  - `lv_mm`: Strip length in millimeters.
  - `alpha_deg`: Angle of the strip raster relative to the transverse axis.
  - `n_strips`: Number of raster strips.
  - `width_mm`: Spacing between adjacent strip centers.
  - `gtop` / `gbot`: Eigenstrain amplitude in top / bottom layer.
  - `ortho_top` / `ortho_bottom`: Layer orthotropy factor $\in [-1, 1]$.

---

## Coordinate Conventions

- **Internal Solver Coordinates**: $(u, h, v)$, where $(u, v)$ is the in-plane reference coordinates and $h$ is out-of-plane height.
- **Public / Experimental Sheet Axes**:
  - **Width ($x$)**: $x = u \in [-L_x, L_x]$
  - **Length / Feed ($z$)**: $z = v + 65\text{ mm}$
  - **Height ($y$)**: $y = h$ (vertical displacement)

---

## Solver Outputs

Simulations write the following artifacts into the output directory:
- `bilayer_zigzag_sequence_cycle_XXX_final.vtp`: VTK PolyData mesh containing final 3D node positions, face normals, and hit counts (openable in ParaView).
- `bilayer_zigzag_sequence_native_hybrid_state.json`: Solver convergence diagnostics, iteration counts, final energy, and gradient norms.
- `bilayer_zigzag_sequence_summary.csv`: Cycle-by-cycle tracking of energy, maximum face hits, and elapsed time.

---

## Repository Layout

```text
PSM2025-zigzag/
├── src/                C++ discrete differential geometry thin-shell solver
├── test/               C++ unit and regression tests
├── bin/                Compiled binaries (shell, testshell)
├── shellsim/           Python wrapper API, configs, and interactive notebooks
├── shape_design/       Double-curvature parametric toolpaths & export scripts
├── CMakeLists.txt      CMake build configuration
└── environment.yml     Conda environment definition
```

---

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
