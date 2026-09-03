"""Visualize the side-by-side comparison between the Trapped Cylinder and the True Dome."""

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
    cyl_vtp = root / "free_dense_centerpeak_m05_r030" / "bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
    dome_vtp = root / "clamped_dense_centerpeak_m05_r030" / "bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"

    cyl_pts, cyl_tri, cyl_u3, cyl_mu, cyl_mv = load_vtp(cyl_vtp)
    dome_pts, dome_tri, dome_u3, dome_mu, dome_mv = load_vtp(dome_vtp)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)

    # 1. 3D view: Trapped Cylinder (Free Boundary)
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    norm_cyl = colors.Normalize(vmin=1000.0 * np.min(cyl_u3), vmax=1000.0 * np.max(cyl_u3))
    surf1 = ax1.plot_trisurf(
        1000.0 * cyl_pts[:, 0],
        1000.0 * cyl_pts[:, 1],
        1000.0 * cyl_pts[:, 2],
        triangles=cyl_tri,
        cmap="coolwarm",
        norm=norm_cyl,
        edgecolor="none",
        alpha=0.95,
    )
    ax1.set_title("A. Trapped Cylinder (Free Boundary)\n3D Deformed Surface", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.view_init(elev=28, azim=-55)

    # 2. Elevation contour: Trapped Cylinder
    ax2 = fig.add_subplot(2, 3, 2)
    tcf1 = ax2.tripcolor(
        1000.0 * cyl_mu,
        1000.0 * cyl_mv,
        cyl_tri,
        1000.0 * cyl_u3,
        cmap="coolwarm",
        shading="gouraud",
    )
    ax2.set_aspect("equal")
    ax2.set_title("Trapped Cylinder\n$U_3$ Elevation Field (mm)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Material U (mm)")
    ax2.set_ylabel("Material V (mm)")
    fig.colorbar(tcf1, ax=ax2, shrink=0.8, label="$U_3$ (mm)")

    # 3. Midline Cross-Sections: Cylinder vs Dome
    ax3 = fig.add_subplot(2, 3, 3)
    tol = 8.0
    # Cylinder profiles
    cyl_m_u = np.abs(1000.0 * cyl_mv) < tol
    ord_cyl_u = np.argsort(cyl_mu[cyl_m_u])
    cyl_m_v = np.abs(1000.0 * cyl_mu) < tol
    ord_cyl_v = np.argsort(cyl_mv[cyl_m_v])

    ax3.plot(
        1000.0 * cyl_mu[cyl_m_u][ord_cyl_u],
        1000.0 * cyl_u3[cyl_m_u][ord_cyl_u],
        "b-o",
        markersize=4,
        label="Cylinder: along U (k1 = -12.0 m⁻¹)",
    )
    ax3.plot(
        1000.0 * cyl_mv[cyl_m_v][ord_cyl_v],
        1000.0 * cyl_u3[cyl_m_v][ord_cyl_v],
        "b--s",
        markersize=4,
        label="Cylinder: along V (k2 ≈ 0.0 m⁻¹)",
    )

    # Dome profiles
    dome_m_u = np.abs(1000.0 * dome_mv) < tol
    ord_dome_u = np.argsort(dome_mu[dome_m_u])
    dome_m_v = np.abs(1000.0 * dome_mu) < tol
    ord_dome_v = np.argsort(dome_mv[dome_m_v])

    ax3.plot(
        1000.0 * dome_mu[dome_m_u][ord_dome_u],
        1000.0 * dome_u3[dome_m_u][ord_dome_u],
        "r-o",
        markersize=4,
        label="Dome: along U (k1 = -0.50 m⁻¹)",
    )
    ax3.plot(
        1000.0 * dome_mv[dome_m_v][ord_dome_v],
        1000.0 * dome_u3[dome_m_v][ord_dome_v],
        "r--s",
        markersize=4,
        label="Dome: along V (k2 = -0.39 m⁻¹)",
    )
    ax3.set_title("Midline Profiles: Cylinder vs Dome", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Material Coordinate (mm)")
    ax3.set_ylabel("$U_3$ (mm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8.5, loc="upper right")

    # 4. 3D view: True Dome (Clamped Fixture / Symmetric Constraint)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    norm_dome = colors.Normalize(vmin=1000.0 * np.min(dome_u3), vmax=1000.0 * np.max(dome_u3))
    surf2 = ax4.plot_trisurf(
        1000.0 * dome_pts[:, 0],
        1000.0 * dome_pts[:, 1],
        1000.0 * dome_pts[:, 2],
        triangles=dome_tri,
        cmap="coolwarm",
        norm=norm_dome,
        edgecolor="none",
        alpha=0.95,
    )
    ax4.set_title("B. True 2D Dome (Edge-Clamped Forming)\n3D Deformed Surface", fontsize=12, fontweight="bold")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    ax4.set_zlabel("Z (mm)")
    ax4.view_init(elev=28, azim=-55)

    # 5. Elevation contour: True Dome (Concentric circular contours)
    ax5 = fig.add_subplot(2, 3, 5)
    tcf2 = ax5.tripcolor(
        1000.0 * dome_mu,
        1000.0 * dome_mv,
        dome_tri,
        1000.0 * dome_u3,
        cmap="coolwarm",
        shading="gouraud",
    )
    ax5.set_aspect("equal")
    ax5.set_title("True 2D Dome\n$U_3$ Elevation Field (Concentric)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Material U (mm)")
    ax5.set_ylabel("Material V (mm)")
    fig.colorbar(tcf2, ax=ax5, shrink=0.8, label="$U_3$ (mm)")

    # 6. Mechanics & Summary Box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary_text = (
        "Mechanics Comparison: Cylinder vs Dome\n"
        "--------------------------------------------------\n\n"
        "A. Trapped Cylinder (Free Boundary):\n"
        "  • k1 = -11.98 m⁻¹, k2 = +0.04 m⁻¹\n"
        "  • Gaussian Curvature K ≈ 0.00 m⁻² (1D)\n"
        "  • Curvature Balance: 0.003 (Pure Cylinder)\n"
        "  • Why: Free edges curl in Cycle 1, locking\n"
        "    into an isometric cylindrical well.\n\n"
        "B. True 2D Dome (Clamped Forming):\n"
        "  • k1 = -0.50 m⁻¹, k2 = -0.39 m⁻¹\n"
        "  • Gaussian Curvature K = +0.195 m⁻² (2D Dome!)\n"
        "  • Curvature Balance: 0.793 (80% Isotropic Dome)\n"
        "  • Center Peak Ratio: 99.1% (Smooth Dome Apex)\n"
        "  • Why: Clamping edges suppresses 1D curling,\n"
        "    forcing the 2D eigenstrain into positive\n"
        "    Gaussian curvature (dome bifurcation)."
    )
    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10.0,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="lightgray"),
    )

    fig.suptitle(
        "How the Dome is Formed: Free Trapped Cylinder vs. Clamped Double-Curved Dome",
        fontsize=15,
        fontweight="bold",
    )

    out_png = root / "dome_vs_cylinder_comparison.png"
    fig.savefig(out_png, dpi=200)
    out_svg = root / "dome_vs_cylinder_comparison.svg"
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved comparison to {out_png} and {out_svg}")


if __name__ == "__main__":
    main()
