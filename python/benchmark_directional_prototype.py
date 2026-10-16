#!/usr/bin/env python3
"""
Benchmark candidate directional extensions on the coupon benchmark:
1. Model A: Directional Arc-Length Drift (frictional shear / along-stroke gradient)
2. Model B: State-Dependent 3D Slope Feedback (geometric coupling to intermediate deformed state)

Evaluates:
- Target metric difference (max, RMS)
- Kabsch-aligned 3D shape difference (RMS, Max in um and mm)
- Curvature tensors and principal curvatures (k1, k2)
- Out-of-plane z-span and asymmetric curling / twist
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "bin" / "shell"
OUT_DIR = ROOT / "run" / "benchmark_directional_prototype"


def make_operation(
    growth: float = 0.002,
    ortho: float = 0.8,
    active: list[int] | None = None,
    rotation: float = 0.0,
    n_strips: int = 6,
    lv_mm: float = 120.0,
    width_mm: float = 12.0,
    reverse_traversal: bool = False,
    profile_mode: str = "uniform",
    drift_coeff: float = 0.0,
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
        "reverse_traversal": reverse_traversal,
        "profile": {
            "mode": profile_mode,
            "directional_drift_coeff": drift_coeff,
        },
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


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Align P onto Q via optimal rigid rotation and translation."""
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    p_centered = P - centroid_P
    q_centered = Q - centroid_Q
    H = p_centered.T @ q_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    diag = np.eye(3)
    if d < 0:
        diag[2, 2] = -1.0
    R = Vt.T @ diag @ U.T
    p_aligned = (p_centered @ R.T) + centroid_Q
    diffs = np.linalg.norm(p_aligned - Q, axis=1)
    rms_diff = float(np.sqrt(np.mean(diffs**2)))
    max_diff = float(np.max(diffs))
    return p_aligned, rms_diff, max_diff


def read_vtp_points(path: Path) -> np.ndarray:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    return vtk_to_numpy(poly.GetPoints().GetData())


def estimate_quadric_curvature(points: np.ndarray) -> tuple[float, float, float]:
    p = points - np.mean(points, axis=0)
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    A = np.column_stack([x**2, x * y, y**2, x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c, _, _, _ = coeffs
    H = np.array([[2 * a, b], [b, 2 * c]])
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0]), float(eigvals[1]), float(np.max(z) - np.min(z))


def parse_final_mapping_metrics(vtp_path: Path) -> np.ndarray:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    cd = reader.GetOutput().GetCellData()
    a11 = vtk_to_numpy(cd.GetArray("abar_top_11"))
    a12 = vtk_to_numpy(cd.GetArray("abar_top_12"))
    a22 = vtk_to_numpy(cd.GetArray("abar_top_22"))
    return np.column_stack([a11, a12, a22])


def run_case(name: str, config: dict, minimize_every: int, slope_feedback: float = 0.0) -> Path:
    case_dir = OUT_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    json_path = case_dir / "cycle.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        str(SHELL),
        "-sim", "bilayer_growth",
        "-case", "custom",
        "-geometry", "rectangle",
        "-lx", "0.127",
        "-ly", "0.1524",
        "-res", "0.03",
        "-h_total", "0.0006",
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(json_path.resolve()),
        "-sequence_warm_start", "true",
        "-metric_update", "multiplicative",
        "-sequence_minimize_every", str(minimize_every),
        "-sequence_adaptive", "false",
        "-equilibrium_solver", "native_hybrid",
        "-enable_passE", "false",
        "-nsteps", "1",
        "-tol", "5e-14",
        "-max_iter", "50000",
        "-certify_final", "true",
        "-seed_escape", "false",
        "-basename", name,
        "-export_stl", "false",
        "-hessian_threads", "4",
        "-sequence_slope_feedback", str(slope_feedback),
    ]

    print(f"Running {name} (minimize_every={minimize_every}, slope_feedback={slope_feedback})...", flush=True)
    with open(case_dir / "stdout.log", "w") as log_f:
        res = subprocess.run(cmd, cwd=case_dir, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Simulation {name} failed: see {case_dir}/stdout.log")
    return case_dir


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    growth = 0.002
    ortho = 0.8
    drift_coeff = 0.40  # 40% along-stroke linear drift
    slope_feedback = 25.0  # geometric slope feedback

    # Define test pairs:
    # 1. Model A (Directional Drift) - Single solve
    op_drift_fwd = make_operation(growth=growth, ortho=ortho, profile_mode="directional_drift", drift_coeff=drift_coeff, reverse_traversal=False)
    op_drift_rev = make_operation(growth=growth, ortho=ortho, profile_mode="directional_drift", drift_coeff=drift_coeff, reverse_traversal=True)

    cfg_drift_fwd_fwd = make_config([
        {"id": "pass1_fwd", "repeat": 1, "operation": op_drift_fwd},
        {"id": "pass2_fwd", "repeat": 1, "operation": op_drift_fwd},
    ])
    cfg_drift_fwd_rev = make_config([
        {"id": "pass1_fwd", "repeat": 1, "operation": op_drift_fwd},
        {"id": "pass2_rev", "repeat": 1, "operation": op_drift_rev},
    ])

    # 2. Model B (State-dependent slope feedback) - Intermediate solve after pass 1
    op_uniform_fwd = make_operation(growth=growth, ortho=ortho, profile_mode="uniform", reverse_traversal=False)
    op_uniform_rev = make_operation(growth=growth, ortho=ortho, profile_mode="uniform", reverse_traversal=True)

    cfg_slope_fwd_fwd = make_config([
        {"id": "pass1_fwd", "repeat": 1, "operation": op_uniform_fwd},
        {"id": "pass2_fwd", "repeat": 1, "operation": op_uniform_fwd},
    ])
    cfg_slope_fwd_rev = make_config([
        {"id": "pass1_fwd", "repeat": 1, "operation": op_uniform_fwd},
        {"id": "pass2_rev", "repeat": 1, "operation": op_uniform_rev},
    ])

    cases = [
        # Model A - Single Solve
        ("modelA_single_fwd_fwd", cfg_drift_fwd_fwd, 999, 0.0),
        ("modelA_single_fwd_rev", cfg_drift_fwd_rev, 999, 0.0),
        # Model A - Intermediate Pass-by-Pass Solve
        ("modelA_inter_fwd_fwd", cfg_drift_fwd_fwd, 1, 0.0),
        ("modelA_inter_fwd_rev", cfg_drift_fwd_rev, 1, 0.0),
        # Model B - State Slope Feedback (Pass-by-Pass)
        ("modelB_inter_fwd_fwd", cfg_slope_fwd_fwd, 1, slope_feedback),
        ("modelB_inter_fwd_rev", cfg_slope_fwd_rev, 1, slope_feedback),
    ]

    results = {}
    for name, cfg, min_every, slope_fb in cases:
        case_dir = run_case(name, cfg, min_every, slope_fb)
        vtp_candidates = sorted(case_dir.glob("*_final.vtp"))
        if not vtp_candidates:
            vtp_candidates = sorted(case_dir.glob("*.vtp"))
        final_vtp = vtp_candidates[-1]
        pts = read_vtp_points(final_vtp)
        k1, k2, z_span = estimate_quadric_curvature(pts)
        metrics = parse_final_mapping_metrics(final_vtp)

        results[name] = {
            "vtp": str(final_vtp),
            "points": pts,
            "k1": k1,
            "k2": k2,
            "z_span_mm": z_span * 1e3,
            "metrics": metrics,
        }

    # Compare pairs
    comparisons = [
        ("Model A: Directional Drift (Single Solve)", "modelA_single_fwd_fwd", "modelA_single_fwd_rev"),
        ("Model A: Directional Drift (Pass-by-Pass)", "modelA_inter_fwd_fwd", "modelA_inter_fwd_rev"),
        ("Model B: Geometric Slope Feedback (Pass-by-Pass)", "modelB_inter_fwd_fwd", "modelB_inter_fwd_rev"),
    ]

    summary_list = []
    print("\n" + "=" * 90)
    print("DIRECTIONAL PROTOTYPE BENCHMARK COMPARISONS")
    print("=" * 90)

    for desc, c_fwd, c_rev in comparisons:
        d_fwd = results[c_fwd]
        d_rev = results[c_rev]

        # Metric differences
        m_diff = np.linalg.norm(d_fwd["metrics"] - d_rev["metrics"], axis=1)
        max_m_diff = float(np.max(m_diff))
        rms_m_diff = float(np.sqrt(np.mean(m_diff**2)))

        # Aligned shape differences
        _, rms_shape, max_shape = kabsch_align(d_fwd["points"], d_rev["points"])
        rms_um = rms_shape * 1e6
        max_um = max_shape * 1e6

        dk1 = abs(d_fwd["k1"] - d_rev["k1"])
        dk2 = abs(d_fwd["k2"] - d_rev["k2"])
        dz = abs(d_fwd["z_span_mm"] - d_rev["z_span_mm"])

        summary = {
            "description": desc,
            "case_fwd": c_fwd,
            "case_rev": c_rev,
            "max_metric_diff": max_m_diff,
            "rms_metric_diff": rms_m_diff,
            "aligned_rms_um": rms_um,
            "aligned_max_um": max_um,
            "k1_fwd": d_fwd["k1"],
            "k2_fwd": d_fwd["k2"],
            "k1_rev": d_rev["k1"],
            "k2_rev": d_rev["k2"],
            "delta_k1": dk1,
            "delta_k2": dk2,
            "z_span_fwd_mm": d_fwd["z_span_mm"],
            "z_span_rev_mm": d_rev["z_span_mm"],
            "delta_z_mm": dz,
        }
        summary_list.append(summary)

        print(f"\n[{desc}]")
        print(f"  Fwd vs Rev Metric Diff : Max = {max_m_diff:.6e}, RMS = {rms_m_diff:.6e}")
        print(f"  Aligned Shape Diff     : RMS = {rms_um:.2f} um ({rms_um/1e3:.4f} mm), Max = {max_um:.2f} um ({max_um/1e3:.4f} mm)")
        print(f"  Fwd Principal Curvatures: k1 = {d_fwd['k1']:.6f}, k2 = {d_fwd['k2']:.6f} m^-1, Z-span = {d_fwd['z_span_mm']:.3f} mm")
        print(f"  Rev Principal Curvatures: k1 = {d_rev['k1']:.6f}, k2 = {d_rev['k2']:.6f} m^-1, Z-span = {d_rev['z_span_mm']:.3f} mm")
        print(f"  Delta Curvature         : |dk1| = {dk1:.6f} m^-1, |dk2| = {dk2:.6f} m^-1, |dZ| = {dz:.3f} mm")

    out_summary_path = OUT_DIR / "prototype_directional_summary.json"
    with open(out_summary_path, "w") as f:
        json.dump(summary_list, f, indent=2)
    print(f"\nSaved summary to {out_summary_path}")


if __name__ == "__main__":
    main()
