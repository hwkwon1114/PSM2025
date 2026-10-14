# Project context

## Compute limits — current policy

- Researcher-directed cleanup supersedes historical cumulative verification/fixture budgets. There is no active project-wide seconds, core-seconds, token, cost or "remaining allowance" requirement unless the researcher explicitly sets one for the current task.
- Do not carry limits from old reports, review packets, session notes or paused goal snapshots into new work. Historical runtime/accounting remains evidence, not an execution quota or a reason to request budget renewal.
- Keep sensible per-command timeouts, Slurm resource/wall-time requests and agreed scientific evaluation/stopping rules. These are not cumulative maintenance budgets. This cleanup does not authorize new experiments, recipe changes, unreviewed expensive execution or resuming paused goals.

- C/C++ project built with CMake; the active generated build tree is `build_cmake/`.
- Core implementation is under `src/libshell/` and `src/simulations/`; tests are under `test/testshell/`.
- Python experiment and analysis tooling is under `python/`; SLURM and utility scripts are under `scripts/`.
- `run/`, `bin/`, `lib/`, and `build_cmake/` contain generated outputs or build products, not canonical source.
- Historical project activity identified `scripts/run_zigzag_certified.sbatch` as the most frequently consulted workflow entry point. Treat that as a navigation hint, not authoritative current state.
