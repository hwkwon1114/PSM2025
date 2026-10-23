"""Analyze and visualize Putong's multi_zigzag overlapping orthogonal run."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
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
        "min_z_mm": float(1000.0 * np.min(z)),
        "max_z_mm": float(1000.0 * np.max(z)),
    }


def main():
    root = Path("run/same_side_dome_search/test_putong_multi_zigzag")
    vtp_path = root / "bilayer_multi_zigzag_final_01.vtp"

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    
    # Check point data fields
    point_data = data.GetPointData()
    mat_u_arr = point_data.GetArray("material_u")
    mat_v_arr = point_data.GetArray("material_v")
    if mat_u_arr is not None and mat_v_arr is not None:
        mat_u = vtk_to_numpy(mat_u_arr).astype(float)
        mat_v = vtk_to_numpy(mat_v_arr).astype(float)
    else:
        # Fallback to undeformed XY
        mat_u = points[:, 0]
        mat_v = points[:, 1]

    u3 = points[:, 2] - np.mean(points[:, 2])

    fit = fit_quadratic(points)
    print("--- Putong multi_zigzag Quadratic Fit ---")
    print(json.dumps(fit, indent=2))

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)

    # 1. 3D view
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    norm = colors.Normalize(vmin=1000.0 * np.min(u3), vmax=1000.0 * np.max(u3))
    surf = ax1.plot_trisurf(
        1000.0 * points[:, 0],
        1000.0 * points[:, 1],
        1000.0 * points[:, 2],
        triangles=triangles,
        cmap="coolwarm",
        norm=norm,
        edgecolor="none",
        alpha=0.95,
    )
    ax1.set_title(f"Putong multi_zigzag\n3D Surface (k=({fit['k1']:.2f}, {fit['k2']:.2f}))", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.view_init(elev=28, azim=-55)

    # 2. Elevation heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    tcf = ax2.tripcolor(
        1000.0 * mat_u,
        1000.0 * mat_v,
        triangles,
        1000.0 * u3,
        cmap="coolwarm",
        shading="gouraud",
    )
    ax2.set_aspect("equal")
    ax2.set_title("Putong multi_zigzag\n$U_3$ Elevation Field (mm)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("U (mm)")
    ax2.set_ylabel("V (mm)")
    fig.colorbar(tcf, ax=ax2, shrink=0.8, label="$U_3$ (mm)")

    # 3. Midline Cross-Sections
    ax3 = fig.add_subplot(1, 3, 3)
    tol = 8.0
    m_u = np.abs(1000.0 * mat_v) < tol
    ord_u = np.argsort(mat_u[m_u])
    m_v = np.abs(1000.0 * mat_u) < tol
    ord_v = np.argsort(mat_v[m_v])

    ax3.plot(1000.0 * mat_u[m_u][ord_u], 1000.0 * u3[m_u][ord_u], "b-o", label=f"Along U (k = {fit['k1']:.2f} m⁻¹)")
    ax3.plot(1000.0 * mat_v[m_v][ord_v], 1000.0 * u3[m_v][ord_v], "r--s", label=f"Along V (k = {fit['k2']:.2f} m⁻¹)")
    ax3.set_title("Midline Profiles along U and V", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Coordinate (mm)")
    ax3.set_ylabel("$U_3$ (mm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.suptitle("Putong's multi_zigzag Mode: 2 Orthogonal Zigzag Patches (0° and 90°)", fontsize=14, fontweight="bold")
    out_png = root / "multi_zigzag_shape_analysis.png"
    fig.savefig(out_png, dpi=180)
    out_svg = root / "multi_zigzag_shape_analysis.svg"
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved analysis plot to {out_png} and {out_svg}")


if __name__ == "__main__":
    main()
