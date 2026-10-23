"""Seed Stage 2 Bayesian Optimization state from reconciled training manifest.

Initializes run/stage2_bayesian_optimization/bayes_state.json with verified
historical observations so bayesian_search begins proposing Expected Improvement
candidates immediately without repeating initial space-filling simulations.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import sklearn

from calibration.ewcal.common import Parameters
from calibration.ewcal.runner import Evaluator
from calibration.ewcal.bayes import settings_for, save_json

ROOT = Path(__file__).resolve().parents[2]


def seed_state():
    cfg_path = ROOT / "calibration/configs/run6_stage2_bayesian_optimization.json"
    manifest_path = ROOT / "calibration/data/processed/reconciled_training_manifest_v1.json"

    cfg = json.loads(cfg_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)

    params = Parameters(cfg)
    evaluator = Evaluator(cfg)
    settings = settings_for(cfg)
    seed = int(cfg["optimization"]["seed"])

    signature = hashlib.sha256(
        json.dumps(
            {
                "evaluator": evaluator.identity,
                "settings": settings,
                "seed": seed,
                "sklearn": sklearn.__version__,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    obs = manifest["observations"]
    records = []
    reservations = []
    completed = []
    seen_coords = []

    bounds = {k: v["bounds"] for k, v in cfg["parameters"].items()}

    for o in obs:
        p_dict = o["parameters"]
        # Check domain membership dynamically against configured parameter bounds
        if not all(bounds[k][0] <= p_dict[k] <= bounds[k][1] for k in bounds):
            continue

        p_norm = params.encode(p_dict)

        # Deduplicate identical coordinates (e.g. duplicate P14 reproducing P13)
        if any(np.allclose(p_norm, sc, atol=1e-4) for sc in seen_coords):
            continue
        seen_coords.append(p_norm)

        attempt_id = len(records) + 1
        is_valid = bool(o.get("gp_eligible", False))
        pub_dir = eval_dir / f"attempt_{attempt_id:04d}"
        pub_dir.mkdir(exist_ok=True)

        rec = {
            "attempt_id": attempt_id,
            "tuple_id": o["tuple_id"],
            "parameters": p_dict,
            "normalized_parameters": p_norm.tolist(),
            "valid": is_valid,
            "publication_dir": str(pub_dir),
            "state_file": o.get("state_file"),
            "mesh_file": o.get("mesh_file"),
        }

        if is_valid:
            rec["objective_mm"] = float(o["rms_symmetric_mm"])
            rec["metrics"] = {
                "rms_symmetric_mm": float(o["rms_symmetric_mm"]),
                "rms_target_to_sim_mm": float(o.get("rms_target_to_sim_mm", o["rms_symmetric_mm"])),
                "rms_sim_to_target_weighted_mm": float(o.get("rms_sim_to_target_weighted_mm", o["rms_symmetric_mm"])),
                "registration_qualified": True,
            }
        else:
            reason = o.get("ineligibility_reason", "")
            if "pose near bound" in reason:
                rec["failure_class"] = "registration"
            elif "iteration limit" in reason:
                rec["failure_class"] = "solver"
            else:
                rec["failure_class"] = "registration"
            rec["error"] = reason

        # Write result.json in publication directory for durable provenance
        res_file = pub_dir / "result.json"
        res_file.write_text(json.dumps(rec, indent=2))

        reservation = {
            "attempt_id": attempt_id,
            "evaluator_identity": evaluator.identity,
            "x": p_norm.tolist(),
            "publication_dir": str(pub_dir),
            "result_file": str(res_file),
            "info": {"acquisition": "seeded_history", "tuple_id": o["tuple_id"]},
            "status": "completed",
            "created_at": time.time(),
        }

        records.append(rec)
        reservations.append(reservation)
        completed.append(attempt_id)

    design = [r["normalized_parameters"] for r in records[:5]]

    state = {
        "schema_version": 1,
        "signature": signature,
        "observations": records,
        "reserved_attempts": reservations,
        "completed_attempts": completed,
        "pending": None,
        "design": design,
        "proposals": [],
        "status": "running",
    }

    state_path = out_dir / "bayes_state.json"
    save_json(state_path, state)
    print(f"Saved seeded state to {state_path}")
    n_valid = sum(rec["valid"] for rec in records)
    print(f"Total seeded observations: {len(records)} ({n_valid} valid, {len(records)-n_valid} failures)")

    valid_records = [r for r in records if r["valid"]]
    best = min(valid_records, key=lambda r: r["objective_mm"]) if valid_records else None
    summary = {
        "method": "Gaussian process / expected improvement on log RMS",
        "status": state["status"],
        "completed_attempts": len(records),
        "successful_evaluations": len(valid_records),
        "failed_evaluations": len(records) - len(valid_records),
        "requested_total_budget": cfg["optimization"]["max_evaluations"],
        "fresh_evaluations_this_invocation": 0,
        "best": best,
        "near_bound_parameters": [],
        "note": "State initialized from verified reconciled historical training manifest.",
    }
    summary_path = out_dir / "bayes_summary.json"
    save_json(summary_path, summary)
    print(f"Saved initial summary to {summary_path}")
    print(f"Current best: {best['tuple_id']} (RMS = {best['objective_mm']:.4f} mm)")


if __name__ == "__main__":
    seed_state()
