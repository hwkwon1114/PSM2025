"""Test true 1-step simultaneous solve from flat vs sequential solve."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def fit_quadratic(current: np.ndarray):
    x = current[:, 0] - np.mean(current[:, 0])
    y = current[:, 1] - np.mean(current[:, 1])
    z = current[:, 2]
    design = np.column_stack((np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y))
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    hessian = np.array([[coeffs[3], coeffs[4]], [coeffs[4], coeffs[5]]])
    principal, directions = np.linalg.eigh(hessian)
    direction_angles = np.mod(np.degrees(np.arctan2(directions[1], directions[0])), 180.0)
    weaker = max(float(np.min(np.abs(principal))), 1e-15)
    return {
        "k1": float(principal[0]),
        "k2": float(principal[1]),
        "k1_dir_deg": float(direction_angles[0]),
        "k2_dir_deg": float(direction_angles[1]),
        "K": float(np.prod(principal)),
        "anisotropy": float(np.max(np.abs(principal)) / weaker),
        "z_span_mm": float(1000.0 * np.ptp(z)),
    }


def evaluate_vtp(vtp_path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    cell_data = data.GetCellData()
    gaussian = vtk_to_numpy(cell_data.GetArray("gauss")).astype(float)
    
    area = 0.5 * np.linalg.norm(
        np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        ),
        axis=1,
    )
    total_area = float(np.sum(area))
    pos_area = float(np.sum(area[gaussian > 0.0]) / total_area)
    
    fit = fit_quadratic(points)
    abs_k = sorted((abs(fit["k1"]), abs(fit["k2"])))
    balance = abs_k[0] / max(abs_k[1], 1e-15)
    return {
        **fit,
        "curvature_balance": balance,
        "positive_K_area": pos_area,
    }


def main():
    repo_root = Path.cwd()
    out_dir = repo_root / "run/same_side_dome_search/test_true_1step_vs_sequential"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 1-Step Simultaneous Config
    cfg_1step = {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {
            "start_mode": "left_bottom_up",
            "profile": {"mode": "center_peak", "top_end_ratio": 0.2, "top_profile_power": 2.0}
        },
        "toolpaths": [
            {
                "id": "simultaneous_0deg_and_90deg",
                "repeat": 1,
                "operation": {
                    "type": "zigzag",
                    "lv_mm": 200.0,
                    "alpha_deg": 3.18,
                    "n_strips": 20,
                    "width_mm": 10.0,
                    "center_uv_mm": [0.0, 0.0],
                    "rotation_deg": 0.0,
                    "gtop": 0.003,
                    "gbot": 0.0,
                    "ortho": -0.5
                }
            }
        ]
    }
    
    print("Executing 1-step test...")


if __name__ == "__main__":
    main()
