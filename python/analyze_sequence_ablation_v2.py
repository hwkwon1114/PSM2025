#!/usr/bin/env python3
"""Analyze persistent sequence-ablation v2 outputs with SI-consistent metrics."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "run" / "sequence_ablation_v2"


def load_vtp(path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    current = data.GetPointData().GetArray("X_current")
    points = vtk_to_numpy(
        current if current is not None else data.GetPoints().GetData()
    ).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    return data, points, triangles


def proper_kabsch(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, float, float]:
    ref0 = reference - reference.mean(axis=0)
    mov0 = moving - moving.mean(axis=0)
    u, _, vt = np.linalg.svd(mov0.T @ ref0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = mov0 @ rotation + reference.mean(axis=0)
    distances = np.linalg.norm(aligned - reference, axis=1)
    return aligned, float(np.sqrt(np.mean(distances**2))), float(distances.max())


def quadratic_fit(points: np.ndarray) -> dict:
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt[2, 2] < 0.0:
        vt[2, :] *= -1.0
    local = centered @ vt.T
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    design = np.column_stack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    coef, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    predicted = design @ coef
    residual = z - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(residual @ residual)
    ss_tot = float(np.sum((z - z.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    slopes = coef[3:5]
    first = np.array([[1 + slopes[0]**2, slopes[0]*slopes[1]],
                      [slopes[0]*slopes[1], 1 + slopes[1]**2]])
    second = np.array([[2*coef[0], coef[1]], [coef[1], 2*coef[2]]]) / math.sqrt(1 + slopes @ slopes)
    curvatures = np.linalg.eigvals(np.linalg.solve(first, second)).real
    return {"principal_curvatures_m_inv": sorted(map(float, curvatures)),
            "quadratic_rmse_m": rmse, "quadratic_r2": float(r2)}


def cell_array(data, name: str) -> np.ndarray:
    array = data.GetCellData().GetArray(name)
    if array is None:
        raise RuntimeError(f"missing VTP cell array {name}")
    return vtk_to_numpy(array).astype(float)


def point_array(data, name: str) -> np.ndarray:
    array = data.GetPointData().GetArray(name)
    if array is None:
        raise RuntimeError(f"missing VTP point array {name}")
    return vtk_to_numpy(array).astype(float)


def areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                         points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1)


def read_last_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def certification(log: str) -> dict:
    matches = re.findall(
        r"\[certify_final\].*?lam_min=([+\-0-9.eE]+).*?\b(MINIMUM|saddle|INDEFINITE)\b",
        log,
        flags=re.IGNORECASE,
    )
    if not matches:
        values = re.findall(r"lam_min=([+\-0-9.eE]+)", log)
        return {"hessian_min_eigenvalue": float(values[-1]) if values else None,
                "certification": "UNKNOWN"}
    value, verdict = matches[-1]
    return {"hessian_min_eigenvalue": float(value), "certification": verdict.upper()}


def final_vtp(case_dir: Path, result: dict) -> Path:
    named = result.get("final_vtp")
    if named and (case_dir / named).is_file():
        return case_dir / named
    candidates = sorted(p for p in case_dir.glob("*.vtp")
                        if "mapping" not in p.name and "pending" not in p.name)
    if not candidates:
        raise RuntimeError(f"no final VTP in {case_dir}")
    return candidates[-1]


def metric_vector(data) -> np.ndarray:
    names = ("abar_top_11", "abar_top_12", "abar_top_22",
             "abar_bot_11", "abar_bot_12", "abar_bot_22")
    return np.column_stack([cell_array(data, name) for name in names])


def self_test() -> dict:
    rng = np.random.default_rng(123)
    points = rng.normal(size=(200, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    moved = points @ q + np.array([2.0, -1.0, 4.0])
    _, rms, maximum = proper_kabsch(points, moved)
    if rms > 1e-12 or maximum > 1e-11:
        raise RuntimeError("Kabsch rigid-transform invariance check failed")
    x, y = np.meshgrid(np.linspace(-0.02, 0.02, 31), np.linspace(-0.02, 0.02, 33))
    cloud = np.column_stack((x.ravel(), y.ravel(), (1.5*x*x + 0.5*y*y).ravel()))
    fit = quadratic_fit(cloud)
    got = np.array(fit["principal_curvatures_m_inv"])
    if not np.allclose(got, [1.0, 3.0], rtol=3e-3, atol=3e-3):
        raise RuntimeError(f"SI paraboloid curvature check failed: {got}")
    return {"kabsch_rms_m": rms, "kabsch_max_m": maximum,
            "paraboloid_curvatures_m_inv": got.tolist()}


def analyze() -> None:
    checks = self_test()
    records, loaded = [], {}
    for marker in sorted(RUN_ROOT.glob("*/*/result.json")):
        case_dir = marker.parent
        result = json.loads(marker.read_text())
        manifest = json.loads((case_dir / "manifest.json").read_text())
        if result.get("return_code") != 0:
            records.append({"stage": manifest["case"]["stage"], "case": manifest["case"]["name"],
                            "return_code": result.get("return_code")})
            continue
        path = final_vtp(case_dir, result)
        data, points, triangles = load_vtp(path)
        area = areas(points, triangles)
        mean = cell_array(data, "mean")
        gaussian = cell_array(data, "gauss")
        weights = area / area.sum()
        convergence = read_last_csv(case_dir / "sequence_convergence.csv")
        summary_files = sorted(case_dir.glob("*_summary.csv"))
        summary = read_last_csv(summary_files[-1]) if summary_files else {}
        log = (case_dir / "run.log").read_text(errors="replace")
        material_frame = np.column_stack((
            point_array(data, "material_u"),
            point_array(data, "material_v"),
            np.zeros(points.shape[0]),
        ))
        aligned_material, _, _ = proper_kabsch(material_frame, points)
        cert = certification(log)
        record = {
            "stage": manifest["case"]["stage"], "case": manifest["case"]["name"],
            "res": manifest["case"]["res"], "tol": manifest["case"]["tol"],
            "warm": manifest["case"]["warm"], "metric_scheme": manifest["case"]["metric"],
            "wall_seconds": result["wall_seconds"], "converged": int(convergence.get("converged", 0)),
            "hlbfgs_code": convergence.get("hlbfgs_code"),
            "iterations": convergence.get("iterations"),
            "final_gradient_norm": convergence.get("final_gradient_norm"),
            "energy": convergence.get("recomputed_energy", summary.get("total_energy")),
            "z_span_m": float(np.ptp(aligned_material[:, 2])),
            "mean_curvature_area_weighted_m_inv": float(weights @ mean),
            "gaussian_curvature_area_weighted_m_inv2": float(weights @ gaussian),
            "gaussian_positive_area_fraction": float(weights[gaussian > 0].sum()),
            "gaussian_negative_area_fraction": float(weights[gaussian < 0].sum()),
            **quadratic_fit(points), **cert, "vtp": str(path.relative_to(ROOT)),
        }
        records.append(record)
        if record["converged"] and cert["certification"] == "MINIMUM":
            loaded[record["case"]] = (data, points, metric_vector(data))
    baseline_name = "repeat_01"
    comparisons = []
    if baseline_name in loaded:
        base_data, base_points, base_metric = loaded[baseline_name]
        diagonal = float(np.linalg.norm(np.ptp(base_points, axis=0)))
        for name, (_, points, metric) in loaded.items():
            if points.shape != base_points.shape or metric.shape != base_metric.shape:
                continue
            _, rms, maximum = proper_kabsch(base_points, points)
            delta = metric - base_metric
            comparisons.append({"reference": baseline_name, "case": name,
                                "aligned_rms_m": rms, "aligned_max_m": maximum,
                                "relative_rms_panel_diagonal": rms / diagonal,
                                "metric_rms": float(np.sqrt(np.mean(delta**2))),
                                "metric_max": float(np.max(np.abs(delta)))})
    repeat_names = sorted(name for name in loaded if name.startswith("repeat_"))
    repeat_rms = []
    for index, first in enumerate(repeat_names):
        for second in repeat_names[index + 1:]:
            first_points = loaded[first][1]
            second_points = loaded[second][1]
            if first_points.shape == second_points.shape:
                repeat_rms.append(proper_kabsch(first_points, second_points)[1])
    noise_floor = max(repeat_rms, default=0.0)
    for item in comparisons:
        item["exceeds_5x_repeatability"] = item["aligned_rms_m"] > 5.0 * noise_floor
        item["exceeds_relative_1e4"] = item["relative_rms_panel_diagonal"] > 1e-4

    output = {"self_tests": checks, "identical_run_rms_noise_floor_m": noise_floor,
              "cases": records, "comparisons_to_repeat_01": comparisons}
    matched_pairs = []
    pairs = [
        ("repeat_01", "cold_g2e5"),
        ("warm_g1e4", "cold_g1e4"),
        ("repeat_01", "metric_recursive_linearized_2e5"),
        ("repeat_01", "metric_reference_additive_linearized_2e5"),
        ("metric_multiplicative_1e4", "metric_recursive_linearized_1e4"),
        ("metric_multiplicative_1e4", "metric_reference_additive_linearized_1e4"),
        ("metric_multiplicative_5e4", "metric_recursive_linearized_5e4"),
        ("metric_multiplicative_5e4", "metric_reference_additive_linearized_5e4"),
        ("noncommuting45_forward", "noncommuting45_reverse"),
        ("repeat_01", "tol_1e10"),
        ("repeat_01", "tol_1e14"),
    ]
    pairs.extend(("seg_complete_path", name) for name in (
        "seg_path_each", "seg_path_every3", "seg_stroke_major",
        "seg_stroke_reverse", "seg_single_solve"))
    for reference, name in pairs:
        if reference not in loaded or name not in loaded:
            continue
        _, ref_points, ref_metric = loaded[reference]
        _, points, metric = loaded[name]
        if points.shape != ref_points.shape or metric.shape != ref_metric.shape:
            continue
        _, rms, maximum = proper_kabsch(ref_points, points)
        delta = metric - ref_metric
        diagonal = float(np.linalg.norm(np.ptp(ref_points, axis=0)))
        matched_pairs.append({
            "reference": reference, "case": name,
            "aligned_rms_m": rms, "aligned_max_m": maximum,
            "relative_rms_panel_diagonal": rms / diagonal,
            "metric_rms": float(np.sqrt(np.mean(delta**2))),
            "metric_max": float(np.max(np.abs(delta))),
            "exceeds_5x_repeatability": rms > 5.0 * noise_floor,
            "exceeds_relative_1e4": rms / diagonal > 1e-4,
        })

    output["matched_comparisons"] = matched_pairs
    columns = sorted({key for row in matched_pairs for key in row})
    with (RUN_ROOT / "matched_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(matched_pairs)
    (RUN_ROOT / "analysis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    columns = sorted({key for row in records for key in row})
    with (RUN_ROOT / "case_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(records)
    columns = sorted({key for row in comparisons for key in row})
    with (RUN_ROOT / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps({"cases": len(records), "comparisons": len(comparisons),
                      "noise_floor_m": noise_floor, "self_tests": checks}, indent=2))


if __name__ == "__main__":
    analyze()
