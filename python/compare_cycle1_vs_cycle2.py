"""Directly compare 1-path (Cycle 1) vs 2-path (Cycle 2) shapes when bbar = 0."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    return {
        "k1": float(principal[0]),
        "k2": float(principal[1]),
        "k1_dir_deg": float(direction_angles[0]),
        "k2_dir_deg": float(direction_angles[1]),
        "z_span_mm": float(1000.0 * np.ptp(z)),
        "min_z_mm": float(1000.0 * np.min(z)),
        "max_z_mm": float(1000.0 * np.max(z)),
    }


def load_vtp(path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    u3 = vtk_to_numpy(data.GetPointData().GetArray("U3_from_cycle0")).astype(float)
    mu = vtk_to_numpy(data.GetPointData().GetArray("material_u")).astype(float)
    mv = vtk_to_numpy(data.GetPointData().GetArray("material_v")).astype(float)
    return points, triangles, u3, mu, mv


def main():
    root = Path("run/same_side_dome_search/compare_cycle1_cycle2_test")
    c1_vtp = sorted(root.glob("*cycle_001*_final.vtp"))[0]
    c2_vtp = sorted(root.glob("*cycle_002*_final.vtp"))[0]

    c1_pts, c1_tri, c1_u3, c1_mu, c1_mv = load_vtp(c1_vtp)
    c2_pts, c2_tri, c2_u3, c2_mu, c2_mv = load_vtp(c2_vtp)

    c1_fit = fit_quadratic(c1_pts)
    c2_fit = fit_quadratic(c2_pts)

    print("--- Cycle 1 (1 path: 0° only) ---")
    print(json.dumps(c1_fit, indent=2))
    print("\n--- Cycle 2 (2 paths: 0° + 90°) ---")
    print(json.dumps(c2_fit, indent=2))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # Cross sections along U
    tol_v = 8.0
    c1_mask_u = np.abs(1000.0 * c1_mv) < tol_v
    ord1_u = np.argsort(c1_mu[c1_mask_u])
    c2_mask_u = np.abs(1000.0 * c2_mv) < tol_v
    ord2_u = np.argsort(c2_mu[c2_mask_u])

    axes[0].plot(
        1000.0 * c1_mu[c1_mask_u][ord1_u],
        1000.0 * c1_u3[c1_mask_u][ord1_u],
        "b-o",
        label=f"Cycle 1 (1 path): span={c1_fit['z_span_mm']:.1f}mm",
    )
    axes[0].plot(
        1000.0 * c2_mu[c2_mask_u][ord2_u],
        1000.0 * c2_u3[c2_mask_u][ord2_u],
        "r-s",
        label=f"Cycle 2 (2 paths): span={c2_fit['z_span_mm']:.1f}mm",
    )
    axes[0].set_title("Midline Profile along U (Bending Axis)")
    axes[0].set_xlabel("Material U (mm)")
    axes[0].set_ylabel("$U_3$ (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Cross sections along V
    tol_u = 8.0
    c1_mask_v = np.abs(1000.0 * c1_mu) < tol_u
    ord1_v = np.argsort(c1_mv[c1_mask_v])
    c2_mask_v = np.abs(1000.0 * c2_mu) < tol_u
    ord2_v = np.argsort(c2_mv[c2_mask_v])

    axes[1].plot(
        1000.0 * c1_mv[c1_mask_v][ord1_v],
        1000.0 * c1_u3[c1_mask_v][ord1_v],
        "b-o",
        label="Cycle 1 (along V, U≈0)",
    )
    axes[1].plot(
        1000.0 * c2_mv[c2_mask_v][ord2_v],
        1000.0 * c2_u3[c2_mask_v][ord2_v],
        "r-s",
        label="Cycle 2 (along V, U≈0)",
    )
    axes[1].set_title("Midline Profile along V (Transverse Axis)")
    axes[1].set_xlabel("Material V (mm)")
    axes[1].set_ylabel("$U_3$ (mm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Ratio / shape comparison
    axes[2].plot(
        1000.0 * c1_mu[c1_mask_u][ord1_u],
        1000.0 * c1_u3[c1_mask_u][ord1_u] / max(abs(c1_fit['min_z_mm']), 1e-6),
        "b-o",
        label="Cycle 1 normalized",
    )
    axes[2].plot(
        1000.0 * c2_mu[c2_mask_u][ord2_u],
        1000.0 * c2_u3[c2_mask_u][ord2_u] / max(abs(c2_fit['min_z_mm']), 1e-6),
        "r--s",
        label="Cycle 2 normalized",
    )
    axes[2].set_title("Normalized Shape Profile ($U_3 / U_{3,max}$)")
    axes[2].set_xlabel("Material U (mm)")
    axes[2].set_ylabel("Normalized $U_3$")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Comparison of 1-Path (Cycle 1) vs 2-Path (Cycle 2) Shape under Fixed bbar = 0", fontsize=13, fontweight="bold")
    out_png = root / "c1_vs_c2_shape_comparison.png"
    fig.savefig(out_png, dpi=180)
    out_svg = root / "c1_vs_c2_shape_comparison.svg"
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved comparison plot to {out_png} and {out_svg}")

if __name__ == "__main__":
    main()
