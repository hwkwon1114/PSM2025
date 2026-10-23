#!/usr/bin/env python3
"""
Benchmark directional path dependence on a 2-to-4 path coupon under the baseline model.

Investigates:
1. Schedule A: Forward-Forward (fwd_fwd) - Pass 1 (strips 0..N-1) then Pass 2 (strips 0..N-1)
2. Schedule B: Forward-Backward (fwd_rev) - Pass 1 (strips 0..N-1) then Pass 2 (strips N-1..0)
3. Cadence comparison:
   - Single final solve (no intermediate relaxation)
   - Step-by-step intermediate solve (minimization after each pass)
4. Target metric tensor comparison (face-by-face)
5. Kabsch-aligned 3D shape comparison and curvature extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
SHELL = ROOT / "bin" / "shell"
OUT_DIR = ROOT / "run" / "benchmark_directional_baseline"


def make_operation(
    growth: float = 0.002,
    ortho: float = 0.8,
    active: list[int] | None = None,
    rotation: float = 0.0,
    n_strips: int = 6,
    lv_mm: float = 120.0,
    width_mm: float = 12.0,
) -> dict:
    op = {
        "type": "zigzag",
        "lv_mm": lv_mm,
        "alpha_deg": 10.0,
        "n_strips": n_strips,
        "width_mm": width_mm,
        "rotation_deg": rotation,
        "gtop": growth,
        "gbot": 0.0,
        "ortho": ortho,
    }
    if active is not None:
        op["active_strips"] = active
    return op


def make_config(toolpaths: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": {"mode": "uniform"}},
        "toolpaths": toolpaths,
    }


def generate_schedules(n_strips: int = 6, growth: float = 0.002, ortho: float = 0.8) -> dict[str, dict]:
    """Generate 2-pass schedules: forward-forward vs forward-backward."""
    # Pass 1: forward strips 0..n_strips-1
    pass1 = [
        {"id": f"p1_s{s}", "enabled": True, "repeat": 1, "operation": make_operation(growth, ortho, active=[s], n_strips=n_strips)}
        for s in range(n_strips)
    ]
    # Pass 2 Forward: forward strips 0..n_strips-1
    pass2_fwd = [
        {"id": f"p2_fwd_s{s}", "enabled": True, "repeat": 1, "operation": make_operation(growth, ortho, active=[s], n_strips=n_strips)}
        for s in range(n_strips)
    ]
    # Pass 2 Backward: reverse strips n_strips-1..0
    pass2_rev = [
        {"id": f"p2_rev_s{s}", "enabled": True, "repeat": 1, "operation": make_operation(growth, ortho, active=[s], n_strips=n_strips)}
        for s in reversed(range(n_strips))
    ]

    # Full monolithic passes (each pass is 1 toolpath hitting all strips)
    mono_pass1 = {"id": "pass1", "enabled": True, "repeat": 1, "operation": make_operation(growth, ortho, n_strips=n_strips)}
    mono_pass2 = {"id": "pass2", "enabled": True, "repeat": 1, "operation": make_operation(growth, ortho, n_strips=n_strips)}

    return {
        "fwd_fwd_segmented": make_config(pass1 + pass2_fwd),
        "fwd_rev_segmented": make_config(pass1 + pass2_rev),
        "fwd_fwd_monolithic": make_config([mono_pass1, mono_pass2]),
    }


def run_simulation(
    case_name: str,
    config_dict: dict,
    minimize_every: int,
    res: float = 0.03,
    tol: float = 1e-12,
) -> Path:
    case_dir = OUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "sequence.json"
    config_path.write_text(json.dumps(config_dict, indent=2) + "\n")

    result_json = case_dir / "sequence_result.json"
    if result_json.exists():
        try:
            with result_json.open() as f:
                meta = json.load(f)
                if meta.get("completed", False):
                    final_mesh = meta.get("final_mesh")
                    if final_mesh and (case_dir / final_mesh).exists():
                        print(f"SKIP {case_name}: already completed successfully ({final_mesh})", flush=True)
                        return case_dir / final_mesh
        except Exception:
            pass

    cmd = [
        str(SHELL),
        "-sim", "bilayer_growth",
        "-case", "custom",
        "-geometry", "rectangle",
        "-lx", "0.127",
        "-ly", "0.1524",
        "-res", str(res),
        "-h_total", "0.0006",
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(config_path.resolve()),
        "-sequence_warm_start", "true",
        "-metric_update", "multiplicative",
        "-sequence_minimize_every", str(minimize_every),
        "-sequence_adaptive", "false",
        "-equilibrium_solver", "native_hybrid",
        "-enable_passE", "false",
        "-nsteps", "1",
        "-tol", str(tol),
        "-max_iter", "50000",
        "-certify_final", "true",
        "-seed_escape", "false",
        "-basename", case_name,
        "-export_stl", "false",
    ]

    log_path = case_dir / "run.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"

    print(f"Running {case_name} (minimize_every={minimize_every})...", flush=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=case_dir, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Simulation failed for {case_name}, see {log_path}")

    if result_json.exists():
        with result_json.open() as f:
            meta = json.load(f)
            final_mesh = meta.get("final_mesh")
            if final_mesh and (case_dir / final_mesh).exists():
                return case_dir / final_mesh

    # Fallback search
    final_candidates = sorted(case_dir.glob("*_final.vtp"))
    if final_candidates:
        return final_candidates[-1]
    vtps = sorted(case_dir.glob("*.vtp"))
    if vtps:
        return vtps[-1]
    raise FileNotFoundError(f"No VTP found in {case_dir}")


def extract_vtp_data(vtp_path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    data = reader.GetOutput()

    # Points (deformed configuration)
    curr_arr = data.GetPointData().GetArray("X_current")
    points = vtk_to_numpy(curr_arr if curr_arr is not None else data.GetPoints().GetData()).astype(float)

    # Triangles
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)

    # Material coords
    u_arr = data.GetPointData().GetArray("material_u")
    v_arr = data.GetPointData().GetArray("material_v")
    material = np.column_stack((vtk_to_numpy(u_arr).astype(float), vtk_to_numpy(v_arr).astype(float)))

    # Edge basis and target metrics
    n_faces = len(triangles)
    edge_bases = np.empty((n_faces, 2, 2), dtype=float)
    material_area = np.empty(n_faces, dtype=float)
    for face, (i0, i1, i2) in enumerate(triangles):
        Dm = np.column_stack((material[i2] - material[i1], material[i0] - material[i2]))
        edge_bases[face] = Dm
        material_area[face] = 0.5 * abs(np.linalg.det(Dm))

    def get_cell(name: str):
        arr = data.GetCellData().GetArray(name)
        if arr is None:
            return None
        return vtk_to_numpy(arr).astype(float)

    material_metrics = {}
    for layer in ("top", "bot"):
        a11 = get_cell(f"abar_{layer}_11")
        a12 = get_cell(f"abar_{layer}_12")
        a22 = get_cell(f"abar_{layer}_22")
        if a11 is not None and a12 is not None and a22 is not None:
            metrics = np.empty((n_faces, 2, 2), dtype=float)
            for f in range(n_faces):
                edge_metric = np.array([[a11[f], a12[f]], [a12[f], a22[f]]])
                inv_Dm = np.linalg.inv(edge_bases[f])
                metrics[f] = inv_Dm.T @ edge_metric @ inv_Dm
            material_metrics[layer] = metrics

    return {
        "points": points,
        "triangles": triangles,
        "material": material,
        "material_area": material_area,
        "material_metrics": material_metrics,
        "hits_total": get_cell("total_hit_count"),
    }


def proper_kabsch_alignment(ref_pts: np.ndarray, mov_pts: np.ndarray):
    ref0 = ref_pts - ref_pts.mean(axis=0)
    mov0 = mov_pts - mov_pts.mean(axis=0)
    u, _, vt = np.linalg.svd(mov0.T @ ref0)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    aligned = mov0 @ rot + ref_pts.mean(axis=0)
    dist = np.linalg.norm(aligned - ref_pts, axis=1)
    rms = float(np.sqrt(np.mean(dist**2)))
    max_d = float(np.max(dist))
    return aligned, rms, max_d


def fit_quadratic(points: np.ndarray) -> dict:
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt[2, 2] < 0.0:
        vt[2, :] *= -1.0
    local = centered @ vt.T
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    design = np.column_stack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    coef, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    pred = design @ coef
    residual = z - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    slopes = coef[3:5]
    first = np.array([[1 + slopes[0]**2, slopes[0]*slopes[1]],
                      [slopes[0]*slopes[1], 1 + slopes[1]**2]])
    second = np.array([[2*coef[0], coef[1]], [coef[1], 2*coef[2]]]) / math.sqrt(1 + slopes @ slopes)
    curvatures = np.linalg.eigvals(np.linalg.solve(first, second)).real
    return {
        "principal_curvatures": sorted(map(float, curvatures)),
        "rmse": rmse,
        "z_span": float(z.max() - z.min()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", type=float, default=0.03, help="Mesh resolution in meters")
    parser.add_argument("--tol", type=float, default=1e-12, help="Solver tolerance")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedules = generate_schedules(n_strips=6, growth=0.002, ortho=0.8)

    cases_to_run = [
        # Single solve: accumulate both passes, then solve once
        ("single_fwd_fwd", schedules["fwd_fwd_segmented"], 12),
        ("single_fwd_rev", schedules["fwd_rev_segmented"], 12),
        # Intermediate solve: minimize after each complete pass (every 6 strips)
        ("inter_pass_fwd_fwd", schedules["fwd_fwd_segmented"], 6),
        ("inter_pass_fwd_rev", schedules["fwd_rev_segmented"], 6),
        # Intermediate solve: minimize after each individual strip
        ("inter_strip_fwd_fwd", schedules["fwd_fwd_segmented"], 1),
        ("inter_strip_fwd_rev", schedules["fwd_rev_segmented"], 1),
    ]

    results = {}
    for case_name, cfg, minimize_every in cases_to_run:
        vtp_path = run_simulation(case_name, cfg, minimize_every, res=args.res, tol=args.tol)
        data = extract_vtp_data(vtp_path)
        qfit = fit_quadratic(data["points"])
        results[case_name] = {
            "vtp": str(vtp_path),
            "data": data,
            "qfit": qfit,
        }

    # Now compute comparisons:
    # 1. Target metric differences between fwd_fwd and fwd_rev
    comparisons = []
    pairs = [
        ("single_fwd_fwd", "single_fwd_rev", "Single Solve (12 accumulated strokes)"),
        ("inter_strip_fwd_fwd", "inter_strip_fwd_rev", "Intermediate Solve (every 1 stroke)"),
        ("inter_pass_fwd_fwd", "inter_pass_fwd_rev", "Intermediate Solve (every 1 pass = 6 strokes)"),
    ]

    print("\n" + "="*80)
    print("BASELINE COMPARISONS: FORWARD-FORWARD vs FORWARD-BACKWARD (CURRENT MODEL)")
    print("="*80)

    summary_records = []
    for c1, c2, desc in pairs:
        d1 = results[c1]["data"]
        d2 = results[c2]["data"]
        q1 = results[c1]["qfit"]
        q2 = results[c2]["qfit"]

        # Target metric difference
        top1 = d1["material_metrics"]["top"]
        top2 = d2["material_metrics"]["top"]
        metric_diff = np.linalg.norm(top1 - top2, axis=(1, 2))
        max_metric_diff = float(np.max(metric_diff))
        rms_metric_diff = float(np.sqrt(np.mean(metric_diff**2)))

        # Aligned shape difference
        _, rms_geom, max_geom = proper_kabsch_alignment(d1["points"], d2["points"])

        print(f"\n--- {desc} ---")
        print(f"Comparing {c1} vs {c2}:")
        print(f"  Target Metric Top ||a1 - a2|| : max = {max_metric_diff:.4e}, rms = {rms_metric_diff:.4e}")
        print(f"  Aligned Shape Difference      : max = {max_geom*1e6:.3f} um, rms = {rms_geom*1e6:.3f} um")
        print(f"  Curvatures {c1} : {q1['principal_curvatures']}, z_span = {q1['z_span']*1e3:.3f} mm")
        print(f"  Curvatures {c2} : {q2['principal_curvatures']}, z_span = {q2['z_span']*1e3:.3f} mm")

        summary_records.append({
            "description": desc,
            "case_fwd": c1,
            "case_rev": c2,
            "max_metric_diff": max_metric_diff,
            "rms_metric_diff": rms_metric_diff,
            "aligned_rms_um": rms_geom * 1e6,
            "aligned_max_um": max_geom * 1e6,
            "k1_fwd": q1["principal_curvatures"][0],
            "k2_fwd": q1["principal_curvatures"][1],
            "k1_rev": q2["principal_curvatures"][0],
            "k2_rev": q2["principal_curvatures"][1],
            "z_span_fwd_mm": q1["z_span"] * 1e3,
            "z_span_rev_mm": q2["z_span"] * 1e3,
        })

    # Save summary json
    summary_path = OUT_DIR / "baseline_directional_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
