# Nonconvex benchmark launchers

## Current workflow

1. `scripts/benchmark_solver_tests.sbatch`: isolated build and regression harness.
   Current local-dihedral smoke is already completed as **7546843**; do not rerun it.
2. `scripts/local_dihedral_solver_full_qualification.sbatch`: reuse-only full-size
   gate, not a fit. **7547633 failed during Conda activation before any worker**;
   the explicitly approved launcher-only correction **7547943 passed** in 74s
   with the unchanged smoke worker. This qualification phase is closed.

3. `scripts/nonconvex_benchmark/matched_pilot_toy_smoke.sbatch`:
   **7550736 passed** (26s); three toy recipes/restarts, zero scientific fits.
   Its smoke phase is closed; do not resubmit it.

The six-cell pilot was approved after the toy smoke, but its first no-fit
ledger dry-run **7552414 failed** (missing bundle field; 30 seconds, zero fits).
The explicitly approved correction-only continuation **7553328 passed** in 17s
(preparation total 47s). Fit job **7553341 submitted**, capped at 1920s, using
the exact passing driver. Results pending. Do not resubmit the completed
integration gates, toy smoke, dry-runs or any started/completed pilot cell. Full protocol, tolerances and
budgets: `docs/nonconvex_solver_benchmark_plan.md`.

## New convergence comparison

- `convergence_prepare.sbatch`: **7592200 passed** in 255 one-CPU seconds;
  actual-worker toy recovery, synthetic shell check and six-cell fake ledger.
- `convergence_pilot.sbatch`: **7592522 submitted** under a separate explicit
  authorization; 7200s one-CPU job cap, six fresh cells, unchanged native
  acceptance, new shared ceilings of 100,000 attempts / 200,000 evaluations.
  Hybrid switch fixed at 2000 live L-BFGS attempts. Results pending.
- Do not resubmit completed preparation or any started cell. Frozen worker and
  driver are taken from the passing preparation, not later working-tree edits.

## Historical launchers

`archive/` contains 19 superseded pilot/diagnostic launchers, byte-preserved.
They are evidence of earlier protocols, **not a suggested execution sequence**.
Top-level compatibility symlinks preserve old references and script self-copy
paths. `archive_manifest.json` records their paths and hashes. No frozen source
snapshot in `run/` was modified; no failed/completed cell is reopened by cleanup.

Do not merge these scripts into one parameterized runner without testing each
recipe: their budgets, input identities and recovery policies differ.
