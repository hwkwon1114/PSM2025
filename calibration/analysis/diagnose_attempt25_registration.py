"""Two authorized saved-endpoint registration calls; never modify campaign artifacts."""
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

import numpy as np
import scipy
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from calibration.ewcal.geometry import align_target, points, read_mesh, reference_dual_areas

CASE = "run/stage2_bayesian_optimization/evaluations/attempt_0025/case_00/"
OUTPUT = BASE / "run/optimization_review/attempt25_registration_diagnostic.json"
PINNED = {
    "calibration/ewcal/geometry.py": "f7aebfee26552c2f661fa0acd9796a3540fb23ab3327bf85533e660da9196e43",
    "calibration/configs/run6_stage2_bayesian_optimization.json": "7d2e38aaa06f66be8d570e28d26a2fa0c4a108352ef08a8e65d6c4d73fbdc549",
    "calibration/data/processed/run6_oriented.npz": "f2faaddbc5492b6482dd93a68f1cd9699a5ac17f740b577df5f1a8b7392f812e",
    "run/stage2_bayesian_optimization/evaluations/attempt_0025/result.json": "d112d6d3d7b8c6fe93b6a12ed5a1f09e82edaa77051b1c8fa71d248291cf1f83",
    CASE + "prediction_mm.vtp": "1c4f8b94b56ae48b4b783e7891d7f162505bb01546d5e9d585cc600559fafbd0",
    CASE + "bilayer_zigzag_sequence_native_hybrid_state.json": "5972a5308d7c405ac9762a331a909593a2f4dacf7e928d2e7f644a566bb4a401",
    CASE + "sequence_result.json": "65c566f98f10d1e59495e5f8ad654cc42a2cb72fd55f8291cd0cd0cd967995f1",
    "run/stage2_bayesian_optimization/bayes_state.json": "9a372c10b18f43f69786af844cc6e504887a76ffdd5d4a4c76316fba627912c6",
    "run/stage2_bayesian_optimization/evaluations.jsonl": "8ccfa082230fcf4d600b8ecdb7fdf0dc149bb952af47f2d0fed20facd5de9d69",
}


def verify_inputs():
    for name, expected in PINNED.items():
        actual = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        assert actual == expected, f"Input changed: {name}"


assert sys.flags.optimize == 0, "Assertions must be enabled"
assert os.environ.get("SLURM_JOB_ID"), "Run only in the approved compute allocation"
assert not OUTPUT.exists(), "Preserve prior diagnostic; do not overwrite or rerun"
verify_inputs()
config = json.loads((BASE / "calibration/configs/run6_stage2_bayesian_optimization.json").read_text())
settings = config["registration"]
assert settings["icp_iterations"] == 60 and settings["metric"] == "symmetric"
record = json.loads((BASE / "run/stage2_bayesian_optimization/evaluations/attempt_0025/result.json").read_text())
historical = record["rejected_case"]["metrics"]
assert record["attempt_id"] == 25 and not record["valid"]
assert historical["stop_reason"] == "iteration_limit" and not historical["pose_near_bound"]
mesh = read_mesh(BASE / (CASE + "prediction_mm.vtp"), scale=1.0)
state = json.loads((BASE / (CASE + "bilayer_zigzag_sequence_native_hybrid_state.json")).read_text())
assert state["schema_version"] == 1 and state["units"] in ("metres", "m")
ref_m = np.asarray(state["reference_positions"], dtype=float)
faces = np.asarray(state["faces"])
assert ref_m.shape == points(mesh).shape and np.isfinite(ref_m).all()
assert faces.ndim == 2 and faces.shape[1] == 3 and np.issubdtype(faces.dtype, np.integer)
assert faces.size and np.all(faces >= 0) and np.all(faces < len(ref_m))
mesh_cells = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)
assert np.all(mesh_cells[:, 0] == 3) and np.array_equal(mesh_cells[:, 1:], faces)
ref_mm = ref_m * 1000.0
area = float(reference_dual_areas(ref_mm, faces).sum())
assert abs(area - config["expected_reference_area_mm2"]) / config["expected_reference_area_mm2"] <= config["area_tolerance_rel"]
with np.load(BASE / "calibration/data/processed/run6_oriented.npz", allow_pickle=False) as data:
    target = data["points_mm"].copy()
assert target.ndim == 2 and target.shape[1] == 3 and np.isfinite(target).all()

report = {
    "schema_version": 1, "status": "running_control", "attempt_id": 25,
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ["SLURM_JOB_ID"], "hostname": platform.node(),
    "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
    "vtk": vtk.vtkVersion.GetVTKVersion(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_sha256": PINNED, "registration_settings": settings,
    "baseline_absolute_tolerance": 1e-6, "target_rms_mm": 3.0,
    "reference_area_mm2": area, "target_points": len(target),
    "parameters": record["parameters"], "historical_metrics": historical,
    "claim_limit": "Diagnostic only; no campaign relabeling or scientific recipe adoption",
}
with OUTPUT.open("x") as file:
    json.dump(report, file, indent=2, allow_nan=False)

started = time.monotonic()
for iterations in (60, 240):
    step_start = time.monotonic()
    metrics, _, _, _ = align_target(mesh, target, dict(settings, icp_iterations=iterations), ref_mm, faces)
    report[f"metrics_{iterations}"] = metrics
    report[f"elapsed_seconds_{iterations}"] = time.monotonic() - step_start
    report["status"] = "control_measured" if iterations == 60 else "diagnostic_measured"
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if iterations == 60:
        for key in ("rms_symmetric_mm", "rms_target_to_sim_mm", "rms_sim_to_target_weighted_mm", "pose"):
            assert np.allclose(metrics[key], historical[key], rtol=0, atol=1e-6), f"Control mismatch: {key}"
        for key in ("registration_qualified", "registration_converged", "pose_near_bound", "stop_reason", "iterations_used"):
            assert metrics[key] == historical[key], f"Control mismatch: {key}"
        report["control_matches"] = True
    print(json.dumps({"icp_limit": iterations, "rms_symmetric_mm": metrics["rms_symmetric_mm"],
                      "qualified": metrics["registration_qualified"], "stop": metrics["stop_reason"],
                      "iterations_used": metrics["iterations_used"]}), flush=True)

verify_inputs()
report["inputs_unchanged"] = True
report["status"] = "complete"
report["elapsed_seconds_total"] = time.monotonic() - started
report["diagnostic_meets_target"] = bool(report["metrics_240"]["registration_qualified"] and report["metrics_240"]["rms_symmetric_mm"] <= 3.0)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(f"Saved {OUTPUT}", flush=True)
