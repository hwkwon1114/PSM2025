import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[2]

file_list = [
    "AGENTS.md",
    "calibration/ewcal/__init__.py",
    "calibration/ewcal/__main__.py",
    "calibration/ewcal/batch_bayes.py",
    "calibration/ewcal/batch_integrity.py",
    "calibration/ewcal/bayes.py",
    "calibration/ewcal/common.py",
    "calibration/ewcal/geometry.py",
    "calibration/ewcal/lhs.py",
    "calibration/ewcal/lhs_preflight.py",
    "calibration/ewcal/optimize.py",
    "calibration/ewcal/prepare.py",
    "calibration/ewcal/progress.py",
    "calibration/ewcal/report.py",
    "calibration/ewcal/runner.py",
    "calibration/tests/test_parallel_lhs.py",
    "calibration/tests/test_batch_integrity.py",
    "calibration/tests/test_geometry.py",
    "calibration/tests/test_batch_bayes.py",
    "calibration/tests/test_bayes.py",
    "calibration/tests/test_native_forward.py",
    "calibration/tests/test_batch_v16.py",
    "calibration/tests/mock_solver.py",
    "calibration/tests/test_cli_solver.py",
    "calibration/tests/test_evaluator_reassessment.py",
    "calibration/tests/test_pipeline.py",
    "scripts/run_stage2_qlogei_batch.py",
    "scripts/run_stage2_qlogei_batch.sbatch",
    "calibration/configs/run6_stage2_bayesian_optimization.json",
    "run/optimization_review/historical_import_v16.json",
    "run/stage2_bayesian_optimization/bayes_state.json",
    "calibration/data/processed/reconciled_training_manifest_v1.json",
    "run/optimization_review/v15_fixtures/predecessor_state.json",
    "run/optimization_review/v15_fixtures/predecessor_manifest.json",
    "run/stage2_bayesian_optimization/stage2_cluster_protocol.md",
    "run/optimization_review/saved_endpoint_scoring_comparison.json",
    "run/optimization_review/preexecution_review_qlogei_batch_v16_sol_codex_auto.md",
    "run/sparse_direct_build/verification_summary.md",
]

file_hashes = {}
for rel in file_list:
    p = base / rel
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        file_hashes[rel] = h
    else:
        print(f"WARNING: missing {rel}")

packet = {
    "schema_version": 1,
    "stage": "v17 fresh pre-execution reassessment",
    "repo": "/gpfs/projects/p33082/PSM2025-zigzag",
    "branch": "main",
    "head": "8665c1610f632c27b6a2ce6f6a0fc00305e9e47c",
    "working_tree": "dirty shared parent tree; no child writes",
    "objective": "Four-parameter Run6 model-conditional symmetric surface calibration; resolved review blockers F1 and F2 for q=4 qLogEI controller.",
    "authority": "Researcher authorized minimal fixes for F1/F2 and explicit icp_iterations=60 alignment; Sol review through codex-auto requested.",
    "protocol": "run/stage2_bayesian_optimization/stage2_cluster_protocol.md",
    "record": "run/sparse_direct_build/verification_summary.md",
    "prior_review": "run/optimization_review/preexecution_review_qlogei_batch_v16_sol_codex_auto.md",
    "files": file_hashes,
    "blocker_resolutions": {
        "F1": {
            "finding": "Scoring discrepancy between historical 60 ICP iterations and production 30 ICP iterations; P04 qualified at iter 47.",
            "evidence": "run/optimization_review/saved_endpoint_scoring_comparison.json compares all 12 endpoints under 30 vs 60 iterations against reconciled_training_manifest_v1.json. At 60 iters, all 12 endpoints reproduce historical scores to < 1e-8 mm RMS and match qualification. Under 30 iters, P04 hit iteration_limit (7.261 mm) and was disqualified.",
            "resolution": "Researcher explicitly decided to align production scoring by setting icp_iterations to 60. Updated calibration/configs/run6_stage2_bayesian_optimization.json. Historical bytes preserved unchanged."
        },
        "F2": {
            "finding": "Stopping metadata (status, streak) validated after recovery checkpoint/JSONL commit, risking mutation on invalid inputs.",
            "evidence": "calibration/tests/test_batch_v16.py: test_stopping_metadata_rejected_before_writes and test_valid_stopping_metadata_positive_controls confirm zero writes/launches on invalid status/streak while preserving valid positive controls.",
            "resolution": "Updated validate_campaign_ledger in calibration/ewcal/batch_bayes.py to enforce valid status and nonnegative integer streak before committing recovery state."
        }
    },
    "review_questions": [
        "Confirm whether blocker F1 (scoring recipe discrepancy and per-observation provenance) is satisfactorily resolved by the 12-endpoint saved replay in saved_endpoint_scoring_comparison.json and researcher-authorized icp_iterations=60 configuration.",
        "Confirm whether blocker F2 (startup stopping metadata validation before mutation/commit) is satisfactorily resolved by validate_campaign_ledger checks and test_batch_v16 tests.",
        "Assess whether any launch-blocking findings remain for submitting the bounded 5-wave q=4 qLogEI campaign (32 total evaluation ceiling) under scripts/run_stage2_qlogei_batch.sbatch."
    ],
    "non_goals": [
        "Reviewer read-only: no edits, tests, GP fits, or solver jobs.",
        "No recipe/acceptance changes outside the authorized icp_iterations=60 alignment.",
        "No physical identification or global optimality claims."
    ]
}

out_p = base / "run/optimization_review/review_packet_v17.json"
out_p.write_text(json.dumps(packet, indent=2))
print(f"Wrote v17 review packet to {out_p}")
