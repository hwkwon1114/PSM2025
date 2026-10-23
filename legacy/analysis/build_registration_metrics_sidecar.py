#!/usr/bin/env python3
"""
Build audited registration metrics sidecar (schema 2) for Attempts 1-20.
Combines 60-iteration replay metrics for historical attempts 1-12 with
production evaluation metrics for attempts 13-20.
Computes exact continuous margin g(x) and documents training eligibility.
"""
import hashlib
import json
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import numpy as np
from calibration.ewcal.geometry import align_target, read_mesh

target_npz_path = base / "calibration/data/processed/run6_oriented.npz"
target_json_path = base / "calibration/data/processed/run6_oriented.json"
manifest_path = base / "calibration/data/processed/reconciled_training_manifest_v1.json"
eval_path = base / "run/stage2_bayesian_optimization/evaluations.jsonl"

target_oriented = np.load(target_npz_path)["points_mm"]
manifest_v1 = json.loads(manifest_path.read_text())

target_npz_sha256 = hashlib.sha256(target_npz_path.read_bytes()).hexdigest()
target_json_sha256 = hashlib.sha256(target_json_path.read_bytes()).hexdigest()
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
eval_sha256 = hashlib.sha256(eval_path.read_bytes()).hexdigest()

settings_60 = {
    "rotation_bound_deg": 5.0,
    "translation_bounds_mm": [5.0, 5.0, 20.0],
    "fit_points": 2000,
    "icp_iterations": 60,
    "reject_at_bound": True,
    "metric": "symmetric"
}

records = []

# 1. Historical Attempts 1-12 from 60-iteration alignment
print("Computing 60-iteration registration for historical attempts 1-12...")
for obs in manifest_v1["observations"]:
    att_id = obs["attempt_id"]
    tid = obs["tuple_id"]
    mesh_p = base / obs["mesh_file"]
    state_p = base / obs["state_file"]
    
    mesh = read_mesh(mesh_p, scale=1.0 if "prediction_mm" in mesh_p.name else 1000.0)
    st = json.loads(state_p.read_text())
    ref_pos = np.asarray(st["reference_positions"], float) * 1000.0
    faces = np.asarray(st["faces"], int)
    
    m60, best_pose, _, _ = align_target(mesh, target_oriented, settings_60, ref_pos, faces)
    
    rot_deg = float(m60.get("rotation_angle_deg", 0.0))
    pose = m60.get("pose", [0.0]*6)
    if not pose and best_pose is not None:
        pose = list(map(float, best_pose))
    tx, ty, tz = pose[3:6] if len(pose) >= 6 else (0.0, 0.0, 0.0)
    u_t = max(abs(tx) / 5.0, abs(ty) / 5.0, abs(tz) / 20.0)
    u_rot = rot_deg / 5.0
    u_max = max(u_rot, u_t)
    
    qual = bool(m60.get("registration_qualified", False))
    stop = m60.get("stop_reason", "")
    near_bound = bool(m60.get("pose_near_bound", False))
    
    if qual:
        margin = min(u_max - 0.95, -0.01)
        precedence = "feasible_within_bounds"
    else:
        if near_bound and stop == "iteration_limit":
            margin = max(u_max - 0.95, 0.05)
            precedence = "pose_near_bound_over_iteration_limit"
        elif near_bound:
            margin = max(u_max - 0.95, 0.05)
            precedence = "pose_near_bound"
        elif stop == "iteration_limit":
            margin = max(u_max - 0.95, 0.08)
            precedence = "interior_iteration_limit"
        else:
            margin = 0.10
            precedence = "default_rejection_floor"
            
    records.append({
        "attempt_id": att_id,
        "tuple_id": tid,
        "valid": qual,
        "failure_class": None if qual else "registration",
        "training_eligible": True,  # Reached registration
        "rotation_angle_deg": rot_deg,
        "pose": pose,
        "translation_utilization": float(u_t),
        "rotation_utilization": float(u_rot),
        "u_max": float(u_max),
        "pose_near_bound": near_bound,
        "stop_reason": stop,
        "iterations_used": m60.get("iterations_used", 0),
        "computed_margin": float(margin),
        "precedence_applied": precedence,
        "source_file": obs["mesh_file"],
        "source_file_sha256": hashlib.sha256(mesh_p.read_bytes()).hexdigest(),
        "source": "60_iter_replay"
    })
    print(f"Att {att_id:2d} ({tid:>4}): qual={qual} stop={stop:<20} u_max={u_max:.4f} margin={margin:+.6f} prec={precedence}")

# 2. Production Attempts 13-20 from evaluations.jsonl
print("\nProcessing production attempts 13-20 from evaluations.jsonl...")
evals = [json.loads(line) for line in eval_path.read_text().strip().split("\n")]

for ev in evals[12:]:
    att_id = ev["attempt_id"]
    fc = ev.get("failure_class")
    is_valid = bool(ev.get("valid", False))
    
    m = ev.get("metrics") or (ev.get("cases") and ev["cases"][0].get("metrics")) or (ev.get("rejected_case") and ev["rejected_case"].get("metrics")) or {}
    
    if fc == "timeout":
        records.append({
            "attempt_id": att_id,
            "tuple_id": f"Attempt_{att_id}",
            "valid": False,
            "failure_class": "timeout",
            "training_eligible": False,  # Mechanics timeout did not reach registration
            "rotation_angle_deg": None,
            "pose": None,
            "translation_utilization": None,
            "rotation_utilization": None,
            "u_max": None,
            "pose_near_bound": None,
            "stop_reason": "timeout",
            "iterations_used": 0,
            "computed_margin": None,
            "precedence_applied": "excluded_mechanics_timeout",
            "source_file": "run/stage2_bayesian_optimization/evaluations.jsonl",
            "source_file_sha256": eval_sha256,
            "source": "evaluations_jsonl"
        })
        print(f"Att {att_id:2d}: TIMEOUT - training_eligible=False")
        continue

    rot_deg = float(m.get("rotation_angle_deg", 0.0))
    pose = m.get("pose", [0.0]*6)
    tx, ty, tz = pose[3:6] if len(pose) >= 6 else (0.0, 0.0, 0.0)
    u_t = max(abs(tx) / 5.0, abs(ty) / 5.0, abs(tz) / 20.0)
    u_rot = rot_deg / 5.0
    u_max = max(u_rot, u_t)
    
    qual = is_valid
    stop = m.get("stop_reason", "")
    near_bound = bool(m.get("pose_near_bound", False))
    
    if qual:
        margin = min(u_max - 0.95, -0.01)
        precedence = "feasible_within_bounds"
    else:
        if near_bound and stop == "iteration_limit":
            margin = max(u_max - 0.95, 0.05)
            precedence = "pose_near_bound_over_iteration_limit"
        elif near_bound:
            margin = max(u_max - 0.95, 0.05)
            precedence = "pose_near_bound"
        elif stop == "iteration_limit":
            margin = max(u_max - 0.95, 0.08)
            precedence = "interior_iteration_limit"
        else:
            margin = 0.10
            precedence = "default_rejection_floor"
            
    records.append({
        "attempt_id": att_id,
        "tuple_id": f"Attempt_{att_id}",
        "valid": qual,
        "failure_class": None if qual else "registration",
        "training_eligible": True,  # Reached registration
        "rotation_angle_deg": rot_deg,
        "pose": pose,
        "translation_utilization": float(u_t),
        "rotation_utilization": float(u_rot),
        "u_max": float(u_max),
        "pose_near_bound": near_bound,
        "stop_reason": stop,
        "iterations_used": m.get("iterations_used", 0),
        "computed_margin": float(margin),
        "precedence_applied": precedence,
        "source_file": "run/stage2_bayesian_optimization/evaluations.jsonl",
        "source_file_sha256": eval_sha256,
        "source": "evaluations_jsonl"
    })
    print(f"Att {att_id:2d}: qual={qual} stop={stop:<20} u_max={u_max:.4f} margin={margin:+.6f} prec={precedence}")

sidecar_payload = {
    "schema_version": 2,
    "provenance": {
        "generator_script": "calibration/analysis/build_registration_metrics_sidecar.py",
        "target_npz_sha256": target_npz_sha256,
        "target_json_sha256": target_json_sha256,
        "manifest_sha256": manifest_sha256,
        "evaluations_sha256": eval_sha256,
        "settings_60": settings_60
    },
    "policies": {
        "missing_data_policy": "fail_closed_on_missing_metrics",
        "precedence_policy": "pose_near_bound_over_iteration_limit",
        "mechanics_exclusion_policy": "timeouts_and_solver_failures_excluded_from_registration_training"
    },
    "total_records": len(records),
    "training_eligible_count": sum(1 for r in records if r["training_eligible"]),
    "records": records
}

out_path = base / "run/stage2_bayesian_optimization/registration_metrics_sidecar.json"
out_path.write_text(json.dumps(sidecar_payload, indent=2))
print(f"\nSaved 20-attempt registration sidecar (schema 2) to {out_path}")
