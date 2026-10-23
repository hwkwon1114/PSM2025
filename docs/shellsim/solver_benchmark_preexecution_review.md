# Pre-Execution Review: Solver Adaptive Cutback Benchmark

## 1. Objective and Scientific Scope
This benchmark qualifies the **adaptive bisection cutback solver policy** against the baseline **conservative continuation solver policy** across a three-tiered spectrum of structural deformation regimes:
1. **Tier 1 (Simple / Mild)**: `simple_a_1cycle` (1-cycle Toolpath A, gentle curvature, unconstrained free sheet, 30,000 faces). Evaluates single-step convergence at $\lambda = 1.0$ vs. 10 incremental substeps ($\Delta \lambda = 0.1$), quantifying wall-clock runtime speedup and verifying machine-precision shape agreement.
2. **Tier 2 (Moderate / Non-Overlapping)**: `moderate_nonoverlapping_20cycle` (20-cycle $A^{10} B^{10}$ with 12.6 mm spatial clearance gap). Evaluates whether moderate coupled deformation admits direct convergence or mild cutback.
3. **Tier 3 (Complex / Overlapping)**: `complex_overlapping_20cycle` (20-cycle $A^{10} B^{10}$ with 1,848 co-hit faces and Voce strain hardening decay). Evaluates bisection cutback down to $\Delta \lambda \le 0.125$ upon full-step rejection and verifies recovery to identical equilibrium minimum.

## 2. Policy Definitions and Timing Scope
- The benchmark evaluates two **complete operational solver policies**:
  - **Policy A (Adaptive Cutback Deployment)**: Initial step $\lambda = 1.0$, min step $0.015625$, max step $1.0$, growth $2.0$, max retries $8$, per-substep ceiling $600\text{ s}$. Designed as a responsive production default that cuts back quickly if a large step stalls.
  - **Policy B (Conservative Continuation Deployment)**: Initial step $0.1$, min step $0.015625$, max step $0.1$, growth $2.0$, max retries $8$, per-substep ceiling $1800\text{ s}$. Established robust recipe ensuring small load increments.
- Timing comparisons are reported as descriptive operational wall-clock execution times under each complete policy. Where steps converge within minutes, the timeout ceiling is non-binding.
- **Launch Topology**: Production cases are dispatched as **one case and policy per 4-hour SLURM job** on Quest compute nodes (`sbatch scripts/run_solver_benchmark.sbatch <case> <policy>`) for resource isolation and parallel node scheduling, rather than a single shared job.

## 3. Shape Equivalence Metrics, Tolerances, and Stopping Rules
- **Stopping Rule (Implemented C++ Engine)**:
  - If a trial substep hits the substep ceiling (`time_cap`) or fails line search/regularization, the trial is rejected (`solveConverged = false`), triggering bisection rollback: `stepSize = max(continuation_min_step, 0.5 * stepSize)`.
  - The retry counter increments on rejection and resets to 0 upon each accepted substep.
  - Run halts as non-converged only if `retries > max_retries (8)` or step cannot be reduced below `continuation_min_step (0.015625)`.
- **Pre-Scoring Constitutive Identity Gate**:
  - Prior to shape evaluation, endpoints must verify identical mesh topology (vertices, faces, UVs), identical constitutive targets ($\bar{a}_{\text{top}}, \bar{a}_{\text{bot}}$ to machine precision), and matched pass counts.
- **Alignment and Metric Definitions**:
  - Evaluated offline using double-precision native hybrid states (`bilayer_zigzag_sequence_native_hybrid_state.json`) via `analyze_sequence_order` / `analyze_two_gripper_release.align`.
  - Area-weighted rigid Procrustes alignment (Kabsch SVD).
  - *Tier 1 (Simple)*: Strict numerical equivalence: Procrustes RMS displacement $< 1.0 \times 10^{-6}\text{ mm}$ (1 nm), max displacement $< 1.0 \times 10^{-5}\text{ mm}$ (10 nm), $|\Delta \kappa_1| < 10^{-5}\text{ m}^{-1}$.
  - *Tiers 2 & 3 (Moderate/Complex)*: Procrustes RMS displacement $< 36.1\ \mu\text{m}$ ($10^{-4} \times \text{diagonal}$), max displacement $< 100\ \mu\text{m}$.
  - *Principal Curvature $\kappa_1$*: Extracted via paraboloid surface fit over the central valid interior region ($|u| \le 40\text{ mm}, |v| \le 40\text{ mm}$) to exclude free edge curl effects. $|\Delta \kappa_1| = |\kappa_1^{\text{cutback}} - \kappa_1^{\text{continuation}}|$.

## 4. Code and Configuration Identities
- **Binary**: `run/native_shape_refinement/build_4375711/bin/shell` (SHA256 verified)
- **Runner**: `scripts/run_solver_benchmark.py`
- **SLURM Launcher**: `scripts/run_solver_benchmark.sbatch`
- **Test Matrix**: `calibration/configs/solver/benchmark_cases.json`
- **Cutback Config**: `calibration/configs/solver/default_adaptive_cutback.json`
  - `-sequence_adaptive true`, `-sequence_initial_step 1.0`, `-sequence_max_step 1.0`, `-sequence_min_step 0.015625`, `-sequence_step_growth 2.0`, `-sequence_max_retries 8`, `-native_hybrid_seconds 600`
- **Continuation Config**: `calibration/configs/solver/conservative_continuation.json`
  - `-sequence_adaptive true`, `-sequence_initial_step 0.1`, `-sequence_max_step 0.1`, `-sequence_min_step 0.015625`, `-sequence_step_growth 2.0`, `-sequence_max_retries 8`, `-native_hybrid_seconds 1800`

## 3. Resource Safeguards and Execution Policy
- **Partition**: Quest `normal` partition.
- **Node Allocation**: 1 node, 8 OpenMP threads (`-c 8`), 24 GB RAM (`--mem=24G`).
- **Wall-Time Limit**: 4 hours per SLURM job.
- **Solver Step Limits**: 600 seconds per substep ceiling for cutback; 1800 seconds for continuation.
- **Output Isolation**: Unique directories per case under `run/solver_benchmark/matrix/<case>_<policy>/`.
- **Integrity Gate**: Pre-run artifact clearance to comply with C++ `_native_hybrid_state.json` immutability assertion; post-run verification via `scripts/verify_sequence_continuation.py` checking gradient norm ($\le 5\times 10^{-14}$) and positive definiteness.
