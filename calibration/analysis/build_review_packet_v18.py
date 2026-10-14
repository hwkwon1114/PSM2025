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
    "run/optimization_review/saved_endpoint_replay_provenance.json",
    "run/optimization_review/preexecution_review_qlogei_batch_v17_sol_codex_auto.md",
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
    "stage": "v18 fresh pre-execution reassessment",
    "repo": "/gpfs/projects/p33082/PSM2025-zigzag",
    "branch": "main",
    "head": "8665c1610f632c27b6a2ce6f6a0fc00305e9e47c",
    "working_tree": "dirty shared parent tree; no child writes",
    "objective": "Four-parameter Run6 model-conditional symmetric surface calibration; resolved review blockers B1 and B2 from v17 Sol review.",
    "authority": "Researcher authorized minimal fixes for F1/F2 and explicit icp_iterations=60 alignment; Sol review through codex-auto requested.",
    "protocol": "run/stage2_bayesian_optimization/stage2_cluster_protocol.md",
    "record": "run/sparse_direct_build/verification_summary.md",
    "prior_review": "run/optimization_review/preexecution_review_qlogei_batch_v17_sol_codex_auto.md",
    "files": file_hashes,
    "blocker_resolutions": {
        "B1": {
            "finding": "batch_integrity.py:151-159 strictly required new_config == old_config without admitting the authorized 30->60 registration iteration amendment, causing rejection of current production config.",
            "evidence": "calibration/ewcal/batch_integrity.py now narrowly admits the 30->60 amendment alongside the 31->32 evaluation ceiling amendment while strictly enforcing equality for all other scientific parameters. calibration/tests/test_batch_v16.py: test_real_identity_import_and_bound_restart passes (all 10 tests in suite OK in 1.95s), verifying valid import and rejection of unauthorized modifications.",
            "resolution": "Updated batch_integrity.py and refreshed historical_import_v16.json."
        },
        "B2": {
            "finding": "Replay provenance, gauge treatment of raw meshes, precision disclosures, and rationale for importing fixed-sequence predecessor endpoints into the adaptive campaign were incomplete.",
            "evidence": "run/optimization_review/saved_endpoint_replay_provenance.json binds complete replay command, python/library versions, input file digests, per-endpoint gauge/rejection reasons, numerical precision disclosures (P04 within 5.56e-9 mm; max discrepancy 1.29e-5 mm on rejected P06; 100% qualification agreement), and physical justification that converged static equilibrium shapes from fixed stepping satisfy the identical shell governing laws as adaptive stepping.",
            "resolution": "Created saved_endpoint_replay_provenance.json and corrected summary records."
        }
    },
    "review_questions": [
        "Confirm whether blocker B1 (narrow admission of the authorized 30->60 registration amendment in batch_integrity.py with strict rejection of unauthorized modifications) is satisfactorily resolved.",
        "Confirm whether blocker B2 (comprehensive frozen replay provenance, numerical precision disclosures, and physical justification for importing fixed-sequence predecessor equilibrium endpoints into the adaptive campaign) is satisfactorily resolved in saved_endpoint_replay_provenance.json and updated records.",
        "Assess whether any launch blockers remain for submitting the bounded 5-wave q=4 qLogEI campaign (32 total evaluation ceiling) under scripts/run_stage2_qlogei_batch.sbatch."
    ],
    "non_goals": [
        "Reviewer read-only: no edits, tests, GP fits, or solver jobs.",
        "No recipe/acceptance changes outside the authorized icp_iterations=60 alignment.",
        "No physical identification or global optimality claims."
    ]
}

out_p = base / "run/optimization_review/review_packet_v18.json"
out_p.write_text(json.dumps(packet, indent=2))
print(f"Wrote v18 review packet to {out_p}")
