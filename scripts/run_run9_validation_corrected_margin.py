#!/usr/bin/env python3
"""
Evaluate Run 9 forward validation with the corrected 85.0 mm bottom margin
(shift_uv_mm: [0, 28.75]) using the frozen calibrated parameters from Run 6 Attempt 153.
"""

import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from calibration.ewcal.common import Parameters, save_json
from calibration.ewcal.runner import Evaluator

CONFIG_PATH = BASE / "calibration/configs/run9.json"
RUN6_SUMMARY_PATH = BASE / "run/run6_expanded_bo/bayes_summary.json"
OUT_DIR = BASE / "run/run9_transfer_expanded"

BASELINE_ATTEMPT83_RUN9_MM = 2.1951052062325785
BASELINE_UNSHIFTED_RUN9_MM = 1.378669094159767


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen parameters from Run 6 Attempt 153
    run6_summary = json.loads(RUN6_SUMMARY_PATH.read_text())
    best = run6_summary["best"]
    parameters = best["parameters"]
    attempt_id = best["attempt_id"]
    print(f"Loaded frozen parameters from Run 6 Attempt {attempt_id}:")
    for k, v in parameters.items():
        print(f"  {k}: {v}")

    # 2. Load Run 9 corrected configuration
    config = json.loads(CONFIG_PATH.read_text())
    codec = Parameters(config)
    norm_params = codec.encode(parameters).tolist()

    # 3. Initialize Evaluator and run
    evaluator = Evaluator(config)
    print(f"\nEvaluating Run 9 forward simulation with shift_uv_mm: {config['cases'][0]['operation']['shift_uv_mm']}...")
    record = evaluator.evaluate(norm_params, use_cache=False, attempt_id=9901)

    if not record.get("valid"):
        raise RuntimeError(f"Run 9 validation failed: {record.get('failure_class')} - {record.get('error')}")

    metrics = record["cases"][0]["metrics"]
    rms_sym = metrics["rms_symmetric_mm"]
    rms_fwd = metrics["rms_target_to_sim_mm"]
    rms_rev = metrics["rms_sim_to_target_weighted_mm"]

    print("\n" + "=" * 70)
    print("RUN 9 VALIDATION WITH CORRECTED 85.0 MM BOTTOM MARGIN COMPLETED")
    print("=" * 70)
    print(f"Symmetric RMS           : {rms_sym:.6f} mm")
    print(f"Scan -> Surface RMS     : {rms_fwd:.6f} mm")
    print(f"Surface -> Scan Area-RMS: {rms_rev:.6f} mm")

    delta_unshifted = rms_sym - BASELINE_UNSHIFTED_RUN9_MM
    delta_att83 = rms_sym - BASELINE_ATTEMPT83_RUN9_MM

    print(f"\nComparison vs. Previous Unshifted Configuration (1.3787 mm):")
    print(f"  Delta: {delta_unshifted:+.6f} mm ({delta_unshifted / BASELINE_UNSHIFTED_RUN9_MM * 100:+.2f}%)")
    print(f"Comparison vs. Legacy Attempt 83 Baseline (2.1951 mm):")
    print(f"  Delta: {delta_att83:+.6f} mm ({delta_att83 / BASELINE_ATTEMPT83_RUN9_MM * 100:+.2f}%)")

    summary = {
        "run6_best_attempt_id": attempt_id,
        "run6_best_objective_mm": best["objective_mm"],
        "parameters": parameters,
        "normalized_parameters": norm_params,
        "shift_uv_mm": config["cases"][0]["operation"]["shift_uv_mm"],
        "bottom_margin_mm": 85.0,
        "top_margin_mm": 27.5,
        "rms_symmetric_mm": rms_sym,
        "metrics": metrics,
        "comparison": {
            "previous_unshifted_symmetric_rms_mm": BASELINE_UNSHIFTED_RUN9_MM,
            "delta_vs_unshifted_mm": delta_unshifted,
            "delta_vs_unshifted_pct": delta_unshifted / BASELINE_UNSHIFTED_RUN9_MM * 100,
            "baseline_attempt83_mm": BASELINE_ATTEMPT83_RUN9_MM,
            "delta_vs_attempt83_mm": delta_att83,
            "delta_vs_attempt83_pct": delta_att83 / BASELINE_ATTEMPT83_RUN9_MM * 100,
        },
    }

    out_file = OUT_DIR / "run9_transfer_expanded_summary.json"
    save_json(out_file, summary)
    print(f"\nSaved summary to: {out_file}")


if __name__ == "__main__":
    main()
