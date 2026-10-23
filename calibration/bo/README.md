# Calibration BO drivers

Finished campaigns, kept for reproduction. Later stages import earlier ones (`run_run6_bo_refinement.py` is the shared controller library), so the files form one chain rather than separate scripts.

| Stage | Script | Role |
|---|---|---|
| 1 | `run_run6_bo_refinement.py` | Shared BO controller and the corrected240 campaign |
| 2 | `run_run6_expanded_bo.py` (+ `.sbatch`) | Expanded-bounds BO on Run 6, warm-started from 120 evaluations |
| 3 | `run_run6_expanded_bo_continue20.py`, `..._recovery.py` (+ `.sbatch`) | Continuation to 240 evaluations and the authorized recovery of 205–240 |
| 4 | `run_run6_bo_edge.py` (+ `.sbatch`) | Refined-edge success/separation-aware BO |
| 5 | `run_run9_transfer_expanded.py` (+ `.sbatch`) | Frozen Run 6 optimum evaluated on Run 9 |
| 6 | `run_transfer.py run7\|run10` (+ `.sbatch`, `RUN=run7\|run10`) | Frozen Attempt 209 evaluated once on Run 7 or Run 10 |

Prepare a transfer sheet first: `python calibration/analysis/prepare_transfer_run.py run7|run10`; plot it afterwards with `calibration/analysis/plot_transfer.py run7|run10`. Solver settings come from `shellsim/configs/default_adaptive_cutback.json` through the run configs.
