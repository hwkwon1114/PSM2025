# PSM 2025 shell solver

This repository contains the C++ discrete-shell code used for the 2025 Surface
Morphing work. It is adapted from Wim M. van Rees's earlier shell solver.

Start with the [technical overview](docs/TECHNICAL_OVERVIEW.md). It documents
the architecture, mechanics, target-metric/eigenstrain representation,
optimization settings, simulations, file formats, tests, and a section-by-section
review checklist.

The historical [project wiki](https://github.com/PutongK/PSM2025/wiki) is useful
background; the technical overview describes the behavior of this workspace.

## Quick start

The full environment and CMake hints are recorded in [`env_settings`](env_settings).
After configuring the dependencies:

```bash
cmake -S . -B build_cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake --parallel 2
ctest --test-dir build_cmake --output-on-failure
```

Run simulations in separate working directories because outputs use fixed relative
names and some logs are appended:

```bash
mkdir -p run/example
cd run/example
../../bin/shell -sim monolayer_growth -case basic_disk \
  -growthcase spherical -R 1 -res 24 -h 0.01 \
  -initmode flat -maxstages 1 -maxiterations 20
```
