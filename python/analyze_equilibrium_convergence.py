#!/usr/bin/env python3
"""Analyze the serial 20-path joint mesh/load-step equilibrium study."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analyze_sequence_ablation_v2 import (
    areas,
    cell_array,
    final_vtp,
    load_vtp,
    point_array,
    proper_kabsch,
    quadratic_fit,
)
from run_equilibrium_continuation_study import cases as study_cases

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "run" / "equilibrium_continuation_study"


def read_attempts(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def analyze_case(marker: Path) -> tuple[dict, np.ndarray | None]:
    case_dir = marker.parent
    result = json.loads(marker.read_text())
    manifest = json.loads((case_dir / "manifest.json").read_text())
    spec = manifest["case"]
    record = {"tier": spec["tier"], "case": spec["name"],
              "res": spec["res"], "step": spec["step"],
              "metric": spec["metric"], "solver": spec["solver"],
              "tol": spec["tol"], "return_code": result.get("return_code"),
              "wall_seconds": result.get("wall_seconds")}
    if result.get("return_code") != 0:
        return record, None

    attempts = read_attempts(case_dir / "sequence_convergence.csv")
    solved = [row for row in attempts
              if row["minimization_performed"] == "1"
              and row.get("state_kind", "physical") == "physical"]
    rejected = [row for row in solved if row["equilibrium_accepted"] != "1"]
    accepted = [row for row in solved if row["equilibrium_accepted"] == "1"]
    completed_paths = {
        int(row["executed_cycle_index"]) for row in accepted
        if abs(float(row["lambda_trial"]) - 1.0) <= 1.0e-14
    }
    record.update({
        "physical_paths": len(completed_paths),
        "corrector_attempts": len(solved),
        "rejected_attempts": len(rejected),
        "all_attempted_paths_accepted": len(completed_paths) == manifest["path_count"],
        "max_accepted_gradient_norm": max(
            (finite_float(row["final_gradient_norm"]) or 0.0 for row in accepted), default=None),
        "iterations_total": sum(int(row["iterations"]) for row in solved),
        "evaluations_total": sum(int(row["evaluations"]) for row in solved),
    })

    data, points, triangles = load_vtp(final_vtp(case_dir, result))
    area = areas(points, triangles)
    weights = area / area.sum()
    mean = cell_array(data, "mean")
    gaussian = cell_array(data, "gauss")
    material = np.column_stack((point_array(data, "material_u"),
                                point_array(data, "material_v"),
                                np.zeros(points.shape[0])))
    aligned, _, _ = proper_kabsch(material, points)
    record.update({
        "z_span_m": float(np.ptp(aligned[:, 2])),
        "mean_curvature_area_weighted_m_inv": float(weights @ mean),
        "gaussian_curvature_area_weighted_m_inv2": float(weights @ gaussian),
        "gaussian_positive_area_fraction": float(weights[gaussian > 0.0].sum()),
        "gaussian_negative_area_fraction": float(weights[gaussian < 0.0].sum()),
        **quadratic_fit(points),
        "vtp": str(final_vtp(case_dir, result).relative_to(ROOT)),
    })
    return record, points


def main() -> None:
    records: list[dict] = []
    points_by_case: dict[str, np.ndarray] = {}
    for marker in sorted(RUN_ROOT.glob("*/*/result.json")):
        record, points = analyze_case(marker)
        records.append(record)
        if points is not None:
            points_by_case[record["case"]] = points
    if not records:
        raise SystemExit(f"no results under {RUN_ROOT}")

    comparisons = []
    for left_index, left in enumerate(records):
        if left["case"] not in points_by_case:
            continue
        for right in records[left_index + 1:]:
            if right["case"] not in points_by_case:
                continue
            comparable = (
                left["res"] == right["res"] and
                (left["tier"] == right["tier"] or
                 {left["tier"], right["tier"]}
                    <= {"joint", "corrector", "constitutive"})
            )
            if not comparable:
                continue
            left_points = points_by_case[left["case"]]
            right_points = points_by_case[right["case"]]
            if left_points.shape != right_points.shape:
                continue
            _, rms, maximum = proper_kabsch(left_points, right_points)
            comparisons.append({"left": left["case"], "right": right["case"],
                                "aligned_rms_m": rms, "aligned_max_m": maximum})

    joint = sorted((record for record in records if record["tier"] == "joint"),
                   key=lambda row: (row["res"], row["step"]))
    expected_cases = {case["name"] for case in study_cases()}
    observed_cases = {record["case"] for record in records}
    gates = {
        "all_cases_completed": (
            observed_cases == expected_cases and
            all(record.get("return_code") == 0 for record in records)
        ),
        "all_paths_equilibrated": (
            observed_cases == expected_cases and
            all(record.get("all_attempted_paths_accepted", False)
                for record in records)
        ),
        "joint_mesh_step_grid_complete": len(joint) == 9,
        "constitutive_discrimination_ready": False,
        "surrogate_ready": False,
    }
    gates["constitutive_discrimination_ready"] = (
        gates["all_cases_completed"] and gates["all_paths_equilibrated"]
        and gates["joint_mesh_step_grid_complete"]
    )
    # Numerical convergence is necessary but not sufficient for a surrogate:
    # physical calibration/validation data and branch labels are also required.
    gates["surrogate_ready"] = False

    output = {"cases": records, "aligned_shape_comparisons": comparisons,
              "gates": gates,
              "gate_notes": {
                  "constitutive_discrimination_ready":
                      "True only after every joint mesh/load-step case completes and all 20 paths meet the explicit residual criterion.",
                  "surrogate_ready":
                      "False until numerical gates pass and independent process-resolved experimental calibration/validation data exist."
              }}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    with (RUN_ROOT / "cases.csv").open("w", newline="") as stream:
        fields = sorted({key for record in records for key in record})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with (RUN_ROOT / "aligned_shape_comparisons.csv").open("w", newline="") as stream:
        fields = ["left", "right", "aligned_rms_m", "aligned_max_m"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
