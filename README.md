# PSM_2025: Non-Convex Bilayer Shell Equilibrium Solver & Sequential Toolpath Forming

Code accompanying the 2025 Surface Morphing paper, adapted from Prof. Wim M. van Rees.

## Documentation & Guides

- **Core Mechanics & Simulation**: [`docs/bilayer_zigzag_workflow.md`](docs/bilayer_zigzag_workflow.md)
- **JSON Sequence Toolpath Format**: [`docs/zigzag_sequence_format.md`](docs/zigzag_sequence_format.md)
- **Canonical Trajectory Suite**: [`trajectories/README.md`](trajectories/README.md)
- **Benchmark Reports & Literature**: [`docs/reports/`](docs/reports/) (indexed in [`docs/README.md`](docs/README.md))

## Build & Test Instructions

### Quick Build (CMake & Ninja)
```bash
source ~/miniforge/etc/profile.d/conda.sh && conda activate smcpp_vtk38
cmake -B build_cmake -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake -j 4
```

### Running Tests
```bash
# C++ Shell Solver Tests (93 unit tests)
./bin/testshell

# Python Calibration Test Suite (12 unit & CLI regression tests)
python -m unittest discover -s calibration/tests/ -p "test_*.py" -v
```

See [wiki](https://github.com/PutongK/PSM2025/wiki) for additional background.

