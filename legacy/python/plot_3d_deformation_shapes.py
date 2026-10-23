#!/usr/bin/env python3
"""Render 3D deformation shapes from VTP simulation results."""

import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_vtp_mesh(vtp_path):
    """Load points, triangles, and U3 displacement from a VTP file."""
    if not Path(vtp_path).exists():
        raise FileNotFoundError(f"Missing {vtp_path}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()

    pts = vtk_to_numpy(poly.GetPoints().GetData())
    
    # Extract U3 array if present, otherwise default to Z displacement or zeros
    u3_arr = poly.GetPointData().GetArray("U3_from_cycle0")
    if u3_arr is not None:
        u3 = vtk_to_numpy(u3_arr)
    else:
        u_arr = poly.GetPointData().GetArray("U_from_cycle0")
        if u_arr is not None:
            u3 = vtk_to_numpy(u_arr)[:, 2]
        else:
            u3 = pts[:, 2]

    # Extract triangles
    conn = vtk_to_numpy(poly.GetPolys().GetData())
    triangles = []
    i = 0
    while i < len(conn):
        n = conn[i]
        triangles.append(conn[i+1:i+1+n])
        i += 1 + n
    triangles = np.array(triangles)

    return pts, triangles, u3


def main():
    base_dir = Path("run/orthotropic_study_matrix")
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select key simulation cases
    c0_file = base_dir / "ortho_2cycle/res_0.0250_tol_1.0e-12/bilayer_zigzag_sequence_cycle_000_initial.vtp"
    c1_file = base_dir / "ortho_2cycle/res_0.0250_tol_1.0e-12/bilayer_zigzag_sequence_cycle_001_top_dense_centerpeak_0deg_m05_r001_final.vtp"
    c2_file = base_dir / "ortho_2cycle/res_0.0250_tol_1.0e-12/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"

    tol8_file = base_dir / "ortho_2cycle/res_0.0250_tol_1.0e-08/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
    tol10_file = base_dir / "ortho_2cycle/res_0.0150_tol_1.0e-10/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
    tol12_fine = base_dir / "ortho_2cycle/res_0.0150_tol_1.0e-12/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"

    panels = [
        {"title": "(a) Cycle 0: Initial Reference Mesh", "path": c0_file, "desc": "Flat undeformed state ($U_3 = 0$ mm)"},
        {"title": "(b) Cycle 1: 0° Pass (Longitudinal)", "path": c1_file, "desc": "Cylindrical trough ($U_3 \\in [-25, 12]$ mm)"},
        {"title": "(c) Cycle 2: 90° Pass (Orthogonal)", "path": c2_file, "desc": "Anticlastic saddle ($U_3 \\in [-36, 24]$ mm)"},
        {"title": "(d) Saddle Trap (tol = 1e-8, res = 0.025)", "path": tol8_file, "desc": "Unbuckled plateau ($U_3 \\leq 0.13$ mm)"},
        {"title": "(e) Partial Transition (tol = 1e-10, res = 0.015)", "path": tol10_file, "desc": "Premature halt on slope ($U_3 \\leq 10.3$ mm)"},
        {"title": "(f) Full Equilibrium (tol = 1e-12, res = 0.015)", "path": tol12_fine, "desc": "Converged minimum ($U_3 \\in [-36, 23]$ mm)"},
    ]

    # Global color limits in mm for consistent visual scaling across post-buckling states
    vmin, vmax = -38.0, 26.0

    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 10), dpi=200)

    # Common viewing angle
    elev, azim = 26, -58

    for idx, p in enumerate(panels, 1):
        pts, tris, u3 = load_vtp_mesh(p["path"])
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        
        # Scale to millimeters
        x = pts[:, 0] * 1000.0
        y = pts[:, 1] * 1000.0
        z = pts[:, 2] * 1000.0
        u3_mm = u3 * 1000.0

        # Plot 3D surface
        surf = ax.plot_trisurf(
            x, y, z,
            triangles=tris,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0.15,
            edgecolor=(0.1, 0.1, 0.1, 0.4),
            alpha=0.95,
            antialiased=True,
        )

        ax.set_title(p["title"], fontsize=11, fontweight="bold", pad=8)
        ax.text2D(0.05, 0.92, p["desc"], transform=ax.transAxes, fontsize=9, 
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

        ax.set_xlabel("X (mm)", fontsize=8, labelpad=2)
        ax.set_ylabel("Y (mm)", fontsize=8, labelpad=2)
        ax.set_zlabel("Z (mm)", fontsize=8, labelpad=2)
        
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)
        
        # Consistent axis limits
        ax.set_xlim([-80, 80])
        ax.set_ylim([-90, 90])
        ax.set_zlim([-45, 35])

    # Shared colorbar
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.05, top=0.94, wspace=0.10, hspace=0.15)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(surf, cax=cbar_ax)
    cbar.set_label("Out-of-Plane Deflection $U_3$ (mm)", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    out_png = fig_dir / "orthotropic_3d_deformation_shapes.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Generated 3D deformation shapes figure: {out_png}")


if __name__ == "__main__":
    main()
