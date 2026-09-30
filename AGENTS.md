# Project context

- C/C++ project built with CMake; the active generated build tree is `build_cmake/`.
- Core implementation is under `src/libshell/` and `src/simulations/`; tests are under `test/testshell/`.
- Python experiment and analysis tooling is under `python/`; SLURM and utility scripts are under `scripts/`.
- `run/`, `bin/`, `lib/`, and `build_cmake/` contain generated outputs or build products, not canonical source.
- Historical project activity identified `scripts/run_zigzag_certified.sbatch` as the most frequently consulted workflow entry point. Treat that as a navigation hint, not authoritative current state.
