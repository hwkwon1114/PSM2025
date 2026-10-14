"""Reconcile all 12 candidate simulation endpoints across campaigns into authoritative manifest v1.
Uses unified calibration.ewcal.geometry.align_target against calibration/data/processed/run6_oriented.npz.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

from calibration.ewcal.geometry import align_target, read_mesh


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    target_oriented = np.load(base / "calibration/data/processed/run6_oriented.npz")["points_mm"]

    m = json.loads(Path("run/run6_point_cloud_view/rescore_input_manifest_v2.json").read_text())
    rows = list(m["rows"])

    # Also add P14 from completion_substepping
    rows.append({
        "tuple_id": "P14",
        "directory": str(base / "run/lhs_calibration_run6_completion_substepping/trials/0003/evaluations/20261008-221053-e804baf010")
    })

    settings = {
        "rotation_bound_deg": 5.0,
        "translation_bounds_mm": [5.0, 5.0, 20.0],
        "fit_points": 2000,
        "icp_iterations": 60,
        "reject_at_bound": True,
        "metric": "symmetric"
    }

    bounds = {
        "gtop": [0.001, 0.01],
        "gbot": [0.00001, 0.0003],
        "ortho_top": [-1.0, 0.5],
        "ortho_bottom": [-1.0, 0.5]
    }

    print(f"Reconciling {len(rows)} simulation endpoints...")
    reconciled = []
    for r in rows:
        tid = r["tuple_id"]
        d = Path(r["directory"])
        case_d = d if (d / "bilayer_zigzag_sequence_native_hybrid_state.json").exists() else d / "case_00"
        state_p = case_d / "bilayer_zigzag_sequence_native_hybrid_state.json"
        mesh_p = case_d / "prediction_mm.vtp"
        if not mesh_p.exists():
            mesh_p = case_d / "bilayer_zigzag_sequence_cycle_029_run6_r029_final.vtp"
        req_p = d / "request.json"

        req = json.loads(req_p.read_text()) if req_p.exists() else {}
        params = req.get("parameters", {
            "gtop": float(r.get("gtop", 0.003)),
            "gbot": float(r.get("gbot", 0.0001)),
            "ortho_top": float(r.get("ortho_top", -0.5)),
            "ortho_bottom": float(r.get("ortho_bottom", -0.5))
        })

        in_domain = (
            bounds["gtop"][0] <= params["gtop"] <= bounds["gtop"][1] and
            bounds["gbot"][0] <= params["gbot"] <= bounds["gbot"][1] and
            bounds["ortho_top"][0] <= params["ortho_top"] <= bounds["ortho_top"][1] and
            bounds["ortho_bottom"][0] <= params["ortho_bottom"] <= bounds["ortho_bottom"][1]
        )

        st = json.loads(state_p.read_text())
        ref_pos_mm = np.asarray(st["reference_positions"], float) * 1000.0
        faces = np.asarray(st["faces"], int)
        mesh = read_mesh(mesh_p, scale=1.0 if "prediction_mm" in mesh_p.name else 1000.0)

        metrics, aligned, dist, _ = align_target(mesh, target_oriented, settings, ref_pos_mm, faces)

        qual = metrics["registration_qualified"]
        eligible = in_domain and qual

        rec = {
            "tuple_id": tid,
            "directory": str(d.relative_to(base)),
            "request_json": str(req_p.relative_to(base)) if req_p.exists() else None,
            "state_file": str(state_p.relative_to(base)),
            "state_sha256": sha256_file(state_p),
            "mesh_file": str(mesh_p.relative_to(base)),
            "mesh_sha256": sha256_file(mesh_p),
            "parameters": params,
            "in_domain": in_domain,
            "registration_qualified": qual,
            "outer_converged": metrics["outer_converged"],
            "stop_reason": metrics["stop_reason"],
            "pose_near_bound": metrics["pose_near_bound"],
            "rotation_angle_deg": metrics["rotation_angle_deg"],
            "rms_symmetric_mm": metrics["rms_symmetric_mm"],
            "rms_target_to_sim_mm": metrics["rms_target_to_sim_mm"],
            "rms_sim_to_target_weighted_mm": metrics["rms_sim_to_target_weighted_mm"],
            "gp_eligible": eligible,
            "ineligibility_reason": None if eligible else (
                "out_of_domain" if not in_domain else
                "pose_near_bound" if metrics["pose_near_bound"] else
                "registration_unconverged"
            )
        }
        reconciled.append(rec)

    out_json = base / "calibration/data/processed/reconciled_training_manifest_v1.json"
    out_json.write_text(json.dumps({"schema_version": 1, "observations": reconciled}, indent=2))
    print(f"\nSaved reconciled manifest to {out_json}")


if __name__ == "__main__":
    main()
