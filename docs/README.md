# PSM2025 Documentation & Research Reports

Central documentation hub for simulation mechanics, solver optimization, sequence toolpath design, and research findings.

## Primary Documentation & Guides

- **Core Mechanics & Simulation**: [`bilayer_zigzag_workflow.md`](bilayer_zigzag_workflow.md) — metric tensors, director DOFs, and state variables
- **JSON Sequence Format**: [`zigzag_sequence_format.md`](zigzag_sequence_format.md) — toolpath recipe schema, repetition, and boundary conditions
- **Trajectory Dataset Catalog**: [`../trajectories/README.md`](../trajectories/README.md) — canonical benchmarks `01_*` through `07_*`
- **Publication Figures**: [`figures/`](figures/) — deformation states, activation patterns, and non-convex equilibria
- **Paper Bibliography & Manifest**: [`papers/README.md`](papers/README.md) — literature provenance and active sampling papers

## Technical Reports & Investigations (`reports/`)

Historical benchmark investigations, ablation studies, and literature reviews are curated in [`reports/`](reports/):

### Solver Optimization & Benchmarks
- [Final Solver Optimization Report](reports/nonconvex_solver_final_optimization_report.md) — OpenMP parallel TinyAD Hessian, RobustCurvatureGate, and 2.07× speedup
- [Non-Convex Solver Benchmark Plan](reports/nonconvex_solver_benchmark_plan.md) — multi-stage architecture and qualification protocols
- [Solver Restart Summary](reports/nonconvex_solver_restart_summary.md) — compact handoff after job 7642652
- [Absolute Gradient Stopping Audit](reports/absolute_gradient_stopping_audit.md) — certification criteria ($\|\nabla E\| \le 5\times 10^{-12}\text{ N}$)
- [Hessian Convergence Check](reports/hessian_convergence_check.md) — gradient machine precision and Hessian scaling
- [HLBFGS Resolution Sweep](reports/hlbfgs_resolution_sweep.md) — mesh discretization sensitivity

### Mechanics & Literature Reviews
- [Sequential Eigenstrain Mechanics Review](reports/sequential_eigenstrain_mechanics_literature_review.md) — plate theory and residual eigenstrain
- [Optimization & JAX Decision Memo](reports/global_energy_optimization_jax_review.md) — optimizer architectures and GPU/JAX trade-offs
- [English Wheel Model Choices](reports/english_wheel_model_choices.md) — mechanical comparisons with rolling/forming literature
- [Solver Characteristics on Forming Paths](reports/solver_characteristics_and_english_wheel.md) — performance across English wheel passes

### Sequence & Trajectory Studies
- [Sequence Ablation v2 Report](reports/sequence_ablation_v2_report.md) & [Spec](reports/sequence_ablation_v2_spec.md) — toolpath ordering and eigenstrain accumulation
- [Sequence Ablation Numerical Review](reports/sequence_ablation_numerical_review.md) — numerical audit of path dependence
- [Zigzag Sensitivity Report](reports/zigzag_sensitivity_report.md) — parameter sensitivity with audited exact-Newton corrections
- [Trajectory Metric Benchmark](reports/initial_sampling_metric_benchmark.md) & [Multifidelity Map](reports/trajectory_aware_multifidelity_benchmark_map.md) — active trajectory learning

