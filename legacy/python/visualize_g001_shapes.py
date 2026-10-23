"""Visualize deformed 3D surfaces and cross-sections for g=0.001 forward vs reverse."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_vtp(path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    point_data = data.GetPointData()
    u3 = vtk_to_numpy(point_data.GetArray("U3_from_cycle0")).astype(float)
    mat_u = vtk_to_numpy(point_data.GetArray("material_u")).astype(float)
    mat_v = vtk_to_numpy(point_data.GetArray("material_v")).astype(float)
    return points, triangles, u3, mat_u, mat_v


def main():
    root = Path("run/same_side_dome_search")
    fwd_vtp = root / "free_dense_centerpeak_m05_g001_r030" / "bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_g001_r001_final.vtp"
    rev_vtp = root / "free_dense_centerpeak_m05_g001_reverse_r030" / "bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_0deg_m05_g001_r001_final.vtp"

    fwd_pts, fwd_tri, fwd_u3, fwd_mu, fwd_mv = load_vtp(fwd_vtp)
    rev_pts, rev_tri, rev_u3, rev_mu, rev_mv = load_vtp(rev_vtp)

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)

    # 1. 3D view Forward
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    norm_fwd = colors.Normalize(vmin=1000.0 * np.min(fwd_u3), vmax=1000.0 * np.max(fwd_u3))
    surf1 = ax1.plot_trisurf(
        1000.0 * fwd_pts[:, 0],
        1000.0 * fwd_pts[:, 1],
        1000.0 * fwd_pts[:, 2],
        triangles=fwd_tri,
        cmap="coolwarm",
        norm=norm_fwd,
        edgecolor="none",
        alpha=0.95,
    )
    ax1.set_title("Forward (0° -> 90°)\n3D Deformed Surface (mm)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.view_init(elev=28, azim=-55)

    # 2. Top-down U3 contour Forward
    ax2 = fig.add_subplot(2, 3, 2)
    tcf1 = ax2.tripcolor(
        1000.0 * fwd_mu,
        1000.0 * fwd_mv,
        fwd_tri,
        1000.0 * fwd_u3,
        cmap="coolwarm",
        shading="gouraud",
    )
    ax2.set_aspect("equal")
    ax2.set_title("Forward (0° -> 90°)\n$U_3$ Elevation Field (mm)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Material U (mm)")
    ax2.set_ylabel("Material V (mm)")
    fig.colorbar(tcf1, ax=ax2, shrink=0.8, label="$U_3$ (mm)")

    # 3. Midline Cross-Sections Forward vs Reverse
    ax3 = fig.add_subplot(2, 3, 3)
    # Midline along U (V ~ 0)
    tol_v = 8.0
    fwd_mask_u = np.abs(1000.0 * fwd_mv) < tol_v
    order_fwd_u = np.argsort(fwd_mu[fwd_mask_u])
    ax3.plot(
        1000.0 * fwd_mu[fwd_mask_u][order_fwd_u],
        1000.0 * fwd_u3[fwd_mask_u][order_fwd_u],
        "b-o",
        markersize=4,
        label="Forward (along U, V≈0)",
    )
    # Midline along V (U ~ 0)
    tol_u = 8.0
    fwd_mask_v = np.abs(1000.0 * fwd_mu) < tol_u
    order_fwd_v = np.argsort(fwd_mv[fwd_mask_v])
    ax3.plot(
        1000.0 * fwd_mv[fwd_mask_v][order_fwd_v],
        1000.0 * fwd_u3[fwd_mask_v][order_fwd_v],
        "b--s",
        markersize=4,
        label="Forward (along V, U≈0)",
    )

    rev_mask_u = np.abs(1000.0 * rev_mv) < tol_v
    order_rev_u = np.argsort(rev_mu[rev_mask_u])
    ax3.plot(
        1000.0 * rev_mu[rev_mask_u][order_rev_u],
        1000.0 * rev_u3[rev_mask_u][order_rev_u],
        "r-o",
        markersize=4,
        label="Reverse (along U, V≈0)",
    )
    rev_mask_v = np.abs(1000.0 * rev_mu) < tol_u
    order_rev_v = np.argsort(rev_mv[rev_mask_v])
    ax3.plot(
        1000.0 * rev_mv[rev_mask_v][order_rev_v],
        1000.0 * rev_u3[rev_mask_v][order_rev_v],
        "r--s",
        markersize=4,
        label="Reverse (along V, U≈0)",
    )
    ax3.set_title("Midline Profiles: Bending Anisotropy", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Material Coordinate (mm)")
    ax3.set_ylabel("$U_3$ (mm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc="upper right")

    # 4. 3D view Reverse
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    norm_rev = colors.Normalize(vmin=1000.0 * np.min(rev_u3), vmax=1000.0 * np.max(rev_u3))
    surf2 = ax4.plot_trisurf(
        1000.0 * rev_pts[:, 0],
        1000.0 * rev_pts[:, 1],
        1000.0 * rev_pts[:, 2],
        triangles=rev_tri,
        cmap="coolwarm",
        norm=norm_rev,
        edgecolor="none",
        alpha=0.95,
    )
    ax4.set_title("Reverse (90° -> 0°)\n3D Deformed Surface (mm)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    ax4.set_zlabel("Z (mm)")
    ax4.view_init(elev=28, azim=-55)

    # 5. Top-down U3 contour Reverse
    ax5 = fig.add_subplot(2, 3, 5)
    tcf2 = ax5.tripcolor(
        1000.0 * rev_mu,
        1000.0 * rev_mv,
        rev_tri,
        1000.0 * rev_u3,
        cmap="coolwarm",
        shading="gouraud",
    )
    ax5.set_aspect("equal")
    ax5.set_title("Reverse (90° -> 0°)\n$U_3$ Elevation Field (mm)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Material U (mm)")
    ax5.set_ylabel("Material V (mm)")
    fig.colorbar(tcf2, ax=ax5, shrink=0.8, label="$U_3$ (mm)")

    # 6. Curvature Summary Box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary_text = (
        "Deformation & Mechanics Summary (gtop = 0.001)\n"
        "--------------------------------------------------\n\n"
        "Forward (0° -> 90°):\n"
        "  • Dominant Curvature (k1): -3.00 m⁻¹\n"
        "  • Transverse Curvature (k2): +0.00 m⁻¹\n"
        "  • Dominant Cylinder Axis: 0.02°\n"
        "  • Max |U3| span: 25.9 mm (52x sheet thickness)\n"
        "  • Equilibrium Energy: 9.020e-12 J\n\n"
        "Reverse (90° -> 0°):\n"
        "  • Dominant Curvature (k1): -3.56 m⁻¹\n"
        "  • Transverse Curvature (k2): +0.05 m⁻¹\n"
        "  • Dominant Cylinder Axis: 90.18°\n"
        "  • Max |U3| span: 42.3 mm (85x sheet thickness)\n"
        "  • Equilibrium Energy: 8.424e-12 J\n\n"
        "Kinematic Takeaway:\n"
        "  • Target metrics abar are identical (diff < 7e-13)\n"
        "  • The 90° swap in dominant cylinder orientation\n"
        "    is purely due to path-dependent basin selection."
    )
    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="lightgray"),
    )

    fig.suptitle(
        "Sequential Zigzag Bilayer Growth: Forward (0°->90°) vs Reverse (90°->0°) at $g_{top} = 0.001$",
        fontsize=15,
        fontweight="bold",
    )

    out_path = root / "g001_forward_vs_reverse_comparison.png"
    fig.savefig(out_path, dpi=200)
    out_svg = root / "g001_forward_vs_reverse_comparison.svg"
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved visualization to {out_path} and {out_svg}")

if __name__ == "__main__":
    main()
