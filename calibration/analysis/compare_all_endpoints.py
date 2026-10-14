import json
import time
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import numpy as np
from calibration.ewcal.geometry import align_target, read_mesh

target_oriented = np.load(base / "calibration/data/processed/run6_oriented.npz")["points_mm"]
manifest_v1 = json.loads((base / "calibration/data/processed/reconciled_training_manifest_v1.json").read_text())

settings_30 = {
    "rotation_bound_deg": 5.0,
    "translation_bounds_mm": [5.0, 5.0, 20.0],
    "fit_points": 2000,
    "icp_iterations": 30,
    "reject_at_bound": True,
    "metric": "symmetric"
}
settings_60 = dict(settings_30, icp_iterations=60)

print(f"{'Tuple':<6} | {'Hist RMS':<10} {'Hist Qual':<10} | {'30 RMS':<10} {'30 Qual':<10} {'30 Stop':<20} | {'60 RMS':<10} {'60 Qual':<10} {'60 Stop':<20}")
print("-" * 95)

results = []
for obs in manifest_v1["observations"]:
    tid = obs["tuple_id"]
    hist_rms = obs.get("rms_symmetric_mm")
    hist_qual = obs.get("registration_qualified")
    
    mesh_p = base / obs["mesh_file"]
    state_p = base / obs["state_file"]
    
    mesh = read_mesh(mesh_p, scale=1.0 if "prediction_mm" in mesh_p.name else 1000.0)
    st = json.loads(state_p.read_text())
    ref_pos = np.asarray(st["reference_positions"], float) * 1000.0
    faces = np.asarray(st["faces"], int)
    
    m30, _, _, _ = align_target(mesh, target_oriented, settings_30, ref_pos, faces)
    m60, _, _, _ = align_target(mesh, target_oriented, settings_60, ref_pos, faces)
    
    rms30 = m30.get("rms_symmetric_mm", -1.0)
    qual30 = m30.get("registration_qualified", False)
    stop30 = m30.get("stop_reason", "")
    used30 = m30.get("iterations_used", 0)
    
    rms60 = m60.get("rms_symmetric_mm", -1.0)
    qual60 = m60.get("registration_qualified", False)
    stop60 = m60.get("stop_reason", "")
    used60 = m60.get("iterations_used", 0)
    
    hist_rms_str = f"{hist_rms:.4f}" if hist_rms is not None else "None"
    rms30_str = f"{rms30:.4f}" if rms30 is not None else "None"
    rms60_str = f"{rms60:.4f}" if rms60 is not None else "None"
    print(f"{tid:<6} | {hist_rms_str:<10} {str(hist_qual):<10} | {rms30_str:<10} {str(qual30):<10} {stop30+'('+str(used30)+')':<20} | {rms60_str:<10} {str(qual60):<10} {stop60+'('+str(used60)+')':<20}")
    results.append({
        "tuple_id": tid,
        "hist_rms": hist_rms, "hist_qual": hist_qual,
        "rms_30": rms30, "qual_30": qual30, "stop_30": stop30, "used_30": used30,
        "rms_60": rms60, "qual_60": qual60, "stop_60": stop60, "used_60": used60,
    })

out_p = base / "run/optimization_review/saved_endpoint_scoring_comparison.json"
out_p.write_text(json.dumps(results, indent=2))
print(f"\nWrote full comparison to {out_p}")
