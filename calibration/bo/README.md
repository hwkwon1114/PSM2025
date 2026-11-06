# Calibration BO drivers

## Research workflows and paper selection

- The paper uses fixed-hardening Attempt 209; figures and reproduction instructions are in `calibration/paper_figures/attempt209_publication_clean/README.md`.
- Later joint-search controller: `calibration/bo/joint_calibration.py`, with scalar and Pareto specifications. `pairwise_floor1.py` runs the matched-pair wrapper.
- `fresh_lhs_joint4.json`, `multiobjective_joint4.json` and `pairwise_floor1.json` describe completed investigations; none supersedes the paper selection.
- Shared forward evaluation and metric: `calibration/ewcal/runner.py` and `calibration/ewcal/material_frame.py`.
- Current scientific state, results, limits and public-to-historical case mapping: `docs/calibration/bayesian_calibration/convention_corrected_investigation.md` and `calibration/case_labels.py`.

Completed campaigns are not launch permission. Any new scientific execution requires applicable authorization and review of the actual code/configuration identities. `calibration_manuscript_draft.tex` and `current_joint_calibration/` describe a different joint fit, not the selected Attempt 209 paper.

## Archived campaigns

Original pilot, joint-stage and interim sources, launchers and configurations are preserved in `legacy/calibration_bo/`. Their former `calibration/bo/` paths are symbolic links to the pre-cleanup bytes, preserving old imports and their hashes at the time of this cleanup; they are not additional current workflows. Earlier reviews remain tied to the versions they recorded, which can differ from these later snapshots. Historical figure drafts are similarly preserved in `legacy/analysis/`, and the original versions of updated test fixtures are in `legacy/tests/*_pre_cleanup.py.txt`. The archived controller `run_joint_run6_run10.py` remains distinct from the refactored current controller and is needed to inspect historical stage behavior.

The following finished pilot campaigns are kept for their recorded evidence. Later stages import earlier ones (`run_run6_bo_refinement.py` is their shared controller library), so they cannot be treated as interchangeable copies:

| Stage | Script | Role |
|---|---|---|
| 1 | `run_run6_bo_refinement.py` | Shared BO controller and the corrected240 campaign |
| 2 | `run_run6_expanded_bo.py` (+ `.sbatch`) | Expanded-bounds BO on Run 6, warm-started from 120 evaluations |
| 3 | `run_run6_expanded_bo_continue20.py`, `..._recovery.py` (+ `.sbatch`) | Continuation to 240 evaluations and the authorized recovery of 205–240 |
| 4 | `run_run6_bo_edge.py` (+ `.sbatch`) | Refined-edge success/separation-aware BO |
| 5 | `run_run9_transfer_expanded.py` (+ `.sbatch`) | Frozen Run 6 optimum evaluated on Run 9 |
| 6 | `run_transfer.py run7\|run10` (+ `.sbatch`, `RUN=run7\|run10`) | Frozen Attempt 209 evaluated once on Run 7 or Run 10 |

Prepare a transfer sheet first: `python calibration/analysis/prepare_transfer_run.py run7|run10`; plotting scripts are kept outside the repository (local only). Solver settings come from `shellsim/configs/default_adaptive_cutback.json` through the run configs.
