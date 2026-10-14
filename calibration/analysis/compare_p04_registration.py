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
p = base / "run/lhs_calibration_run6_v2/evaluations/20261007-213830-4d0933e1f2/case_00"

mesh = read_mesh(p / "prediction_mm.vtp", scale=1.0)
st = json.loads((p / "bilayer_zigzag_sequence_native_hybrid_state.json").read_text())
ref_pos = np.asarray(st["reference_positions"], float) * 1000.0
faces = np.asarray(st["faces"], int)

settings_30 = {
    "rotation_bound_deg": 5.0,
    "translation_bounds_mm": [5.0, 5.0, 20.0],
    "fit_points": 2000,
    "icp_iterations": 30,
    "reject_at_bound": True,
    "metric": "symmetric"
}

t0 = time.time()
m30, _, _, _ = align_target(mesh, target_oriented, settings_30, ref_pos, faces)
dt30 = time.time() - t0

settings_60 = dict(settings_30, icp_iterations=60)
t0 = time.time()
m60, _, _, _ = align_target(mesh, target_oriented, settings_60, ref_pos, faces)
dt60 = time.time() - t0

print("30 iter: dt={:.2f}s, qual={}, conv={}, stop={}, used={}, rms={:.6f}".format(
    dt30, m30.get("registration_qualified"), m30.get("outer_converged"),
    m30.get("stop_reason"), m30.get("iterations_used"), m30.get("rms_symmetric_mm", -1.0)
))
print("60 iter: dt={:.2f}s, qual={}, conv={}, stop={}, used={}, rms={:.6f}".format(
    dt60, m60.get("registration_qualified"), m60.get("outer_converged"),
    m60.get("stop_reason"), m60.get("iterations_used"), m60.get("rms_symmetric_mm", -1.0)
))
