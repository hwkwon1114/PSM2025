# Project research notes

One collection for explanations, literature reviews, experiment plans and findings.
The former `knowledge/`, `reports/`, and `docs/reports/` prose is consolidated here;
these labels did not warrant separate collections.

## Start here

- [Model and workflow](Home.md)
- [Experiment index](Experiment%20index.md) — findings and historical job IDs
- [Open questions](Open%20questions.md)
- [Current solver investigation](nonconvex_solver_benchmark_plan.md) — authoritative history
- [Final solver optimization report](nonconvex_solver_final_optimization_report.md) — final parallel benchmarks, gauge analysis, and hybrid threshold results
- [Solver restart summary](nonconvex_solver_restart_summary.md) — compact handoff after job 7642652
- [Mechanics literature review](sequential_eigenstrain_mechanics_literature_review.md)
- [Optimization literature review](global_energy_optimization_jax_review.md)
- [Paper collection and provenance](papers/README.md)
- [Executable notebook](notebooks/path_representation_space.ipynb)
- [Sequence format and usage](../README_ZigZag_Sequence.md)

## Current experiment decision

Latest: ABC profiled/hardened target, ortho −0.5, optimized only after all eight
passes. Baseline and guarded-Hessian pilots both accepted both starts; guarded
reuse saved **zero Hessians**, because changing rigid-gauge DOFs invalidated all
eligible reduced caches. All phases are closed. See the restart summary for
current evidence, limits and next design options. Historical development follows.

Local-dihedral **solver integration small smoke 7546843 passed**: 40 tests,
one optional skip, 459 allocated one-CPU seconds. Full-size solver qualification
**7547633 failed before its worker** during Conda activation (8 allocated seconds).
An explicitly approved launcher-only continuation **7547943** passed six-start
full-size parity, restart and identity checks in 74 allocated seconds. Its gate
is qualification-only: **no optimizer fits** or production adoption. A
three-recipe toy-only pilot smoke **7550736 passed**; its proposed 35-minute
whole-trajectory pilot was separately approved, but its first ledger dry-run
**7552414 failed before any worker** (missing bundle field, 30s charged, zero
fits). Approved corrected dry-run **7553328 passed** in 17s, bringing preparation
to 47s. Six-fit pilot **7553341 completed** in 1632 one-CPU seconds: zero
accepted minima. Local-dihedral Newton completed 31–34% more updates than
cached Newton and ended lower in energy on both starts, but gradients remain
far above target. See the solver plan for the timing/curve audit (7556591). Prior diagnostic kernel
savings of 30.59–30.81% are not whole-trajectory speedups or convergence evidence.
See the solver plan for protocols and the experiment index for history.

Hybrid qualification **7588808 passed**: controlled CHOLMOD toy interruption/
recovery and a synthetic full-size shell handoff, **zero shell optimizer steps**.
The longer six-cell comparison was approved with explicit new ceilings.
Actual-worker preparation **7592200 passed** (255s, no fits); convergence pilot
**7592522 completed**: both hybrid starts reached accepted minima; pure methods
time capped. The accepted shapes agree after proper rigid alignment to **4.12 nm
RMS / 12.11 nm maximum** (analysis 7608228), with matching director variables.
This supports the same numerical local state, not global optimality. Reviewed
six-run progress, timing and shape PNGs are in
`run/nonconvex_solver_benchmark/convergence_pilot_7592522/performance_audit/reviewed/`.
The hybrid's principal measured cost is Hessian construction (56.8%); see the
solver plan for the full audit and untested acceleration proposals.

## What stays separate

- `docs/`: interpretations and supporting documents; no distinction between a
  “knowledge note” and a “report” solely because of its name.
- `docs/papers/`: source PDFs and bibliographic provenance, not new findings.
- `docs/notebooks/`: executable notebook and its existing companion outputs.
- `run/`: immutable raw experiment evidence, checkpoints and frozen builds.
- `python/`, `scripts/`, `src/`, `test/`: implementation and execution tooling.

Old top-level names are compatibility symlinks, **not duplicate copies**. They
remain because frozen scripts, paper provenance, notebooks and bookmarks use
those paths. `docs/reports/` contains only compatibility aliases. New notes go
straight into `docs/`. No archived run or frozen launcher was rewritten.

`reorganization_manifest.json` records original paths, canonical paths and
pre-relocation hashes; all 193 relocated files were checked byte-for-byte, also
through their historical aliases. Later documented edits are not expected to
match the relocation snapshot. Notebook cells and PDFs were not re-executed or
re-rendered. Relative links in existing prose retain their legacy paths; this is
not a complete rendered-link or notebook-portability qualification.

`.omp/config.yml` remains an active tool-specific configuration, not research
notes. Project instructions have one source at root `AGENTS.md`, reached by a
compatibility alias from `.omp/AGENTS.md`. No tool configuration was disabled.

With explicit approval, only `development_7290186/source/build/` (a failed
compile's disposable CMake intermediates) was removed; its logs, frozen source,
binary and manifests remain. See `run/nonconvex_solver_benchmark/development_7290186/pruned_build_record.json`.
No failed-run evidence or successful/qualified build was deleted. Further
retention review must precede any other deletion; superficially similar runs
are not duplicates.
