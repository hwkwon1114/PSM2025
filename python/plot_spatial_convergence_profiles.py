#!/usr/bin/env python3
"""Plot spatial centerline profiles and cross-sections demonstrating mesh convergence."""

import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})


def load_vtp(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    u3 = vtk_to_numpy(poly.GetPointData().GetArray("U3_from_cycle0"))
    return pts, u3


def extract_centerline(pts, u3, axis=0, tol=0.003):
    """Extract points near axis (axis=0 for X=0 -> profile along Y; axis=1 for Y=0 -> profile along X)."""
    mask = np.abs(pts[:, axis]) <= tol
    sub_pts = pts[mask]
    sub_u3 = u3[mask]
    other_axis = 1 - axis
    sort_idx = np.argsort(sub_pts[:, other_axis])
    coord = sub_pts[sort_idx, other_axis] * 1000.0  # mm
    disp = sub_u3[sort_idx] * 1000.0  # mm
    return coord, disp


def main():
    base_dir = Path("run/orthotropic_study_matrix")
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [0.035, 0.030, 0.025, 0.020, 0.015]
    colors_res = {
        0.035: "#9467bd",
        0.030: "#2ca02c",
        0.025: "#1f77b4",
        0.020: "#ff7f0e",
        0.015: "#d62728",
    }
    markers_res = {
        0.035: "s",
        0.030: "D",
        0.025: "^",
        0.020: "v",
        0.015: "o",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # (a) Transverse profile U3(X) at Y=0 across resolutions
    ax1 = axes[0, 0]
    for res in resolutions:
        vtp_path = base_dir / f"ortho_2cycle/res_{res:.4f}_tol_1.0e-12/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
        pts, u3 = load_vtp(vtp_path)
        x_mm, u3_mm = extract_centerline(pts, u3, axis=1, tol=0.0035)
        ax1.plot(
            x_mm, u3_mm,
            label=f"res = {res*1000:.0f} mm",
            color=colors_res[res],
            marker=markers_res[res],
            markevery=max(1, len(x_mm)//12),
            alpha=0.85,
        )
    ax1.set_xlabel("Transverse Coordinate $X$ (mm)")
    ax1.set_ylabel("Out-of-Plane Deflection $U_3$ (mm)")
    ax1.set_title("(a) Transverse Profile $U_3(X, Y=0)$ across Mesh Resolutions")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower center", frameon=True)

    # (b) Longitudinal profile U3(Y) at X=0 across resolutions
    ax2 = axes[0, 1]
    for res in resolutions:
        vtp_path = base_dir / f"ortho_2cycle/res_{res:.4f}_tol_1.0e-12/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
        pts, u3 = load_vtp(vtp_path)
        y_mm, u3_mm = extract_centerline(pts, u3, axis=0, tol=0.0035)
        ax2.plot(
            y_mm, u3_mm,
            label=f"res = {res*1000:.0f} mm",
            color=colors_res[res],
            marker=markers_res[res],
            markevery=max(1, len(y_mm)//12),
            alpha=0.85,
        )
    ax2.set_xlabel("Longitudinal Coordinate $Y$ (mm)")
    ax2.set_ylabel("Out-of-Plane Deflection $U_3$ (mm)")
    ax2.set_title("(b) Longitudinal Profile $U_3(X=0, Y)$ across Mesh Resolutions")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower center", frameon=True)

    # (c) Peak Deflections & Deflection Span vs Mesh Size h
    ax3 = axes[1, 0]
    h_vals = np.array(resolutions) * 1000.0  # mm
    max_u3s = []
    min_u3s = []
    spans = []
    for res in resolutions:
        vtp_path = base_dir / f"ortho_2cycle/res_{res:.4f}_tol_1.0e-12/bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
        pts, u3 = load_vtp(vtp_path)
        max_u3s.append(u3.max() * 1000.0)
        min_u3s.append(u3.min() * 1000.0)
        spans.append((u3.max() - u3.min()) * 1000.0)

    ax3.plot(h_vals, spans, "o-", color="#d62728", label=r"Total Span $\Delta U_3$", linewidth=2.0)
    ax3.plot(h_vals, max_u3s, "s--", color="#1f77b4", label=r"Apex Peak $\max(U_3)$")
    ax3.plot(h_vals, min_u3s, "^--", color="#2ca02c", label=r"Edge Trough $\min(U_3)$")
    ax3.set_xlabel("Nominal Mesh Element Size $h$ (mm) [Refined $\\leftarrow$]")
    ax3.set_ylabel("Deflection Extrema (mm)")
    ax3.set_title("(c) Asymptotic Deflection Extrema vs Mesh Element Size")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="center right", frameon=True)

    # (d) Buckling bifurcation profile across tolerances at fine resolution (res=0.015)
    ax4 = axes[1, 1]
    tol_cases = [
        ("tol = 1e-8 (Saddle Trap)", "res_0.0150_tol_1.0e-08", "#7f7f7f", ":"),
        ("tol = 1e-10 (Partial Buckling)", "res_0.0150_tol_1.0e-10", "#ff7f0e", "--"),
        ("tol = 1e-12 (Post-Buckling Eq.)", "res_0.0150_tol_1.0e-12", "#1f77b4", "-"),
    ]
    for label, dir_name, col, ls in tol_cases:
        vtp_path = base_dir / "ortho_2cycle" / dir_name / "bilayer_zigzag_sequence_cycle_002_top_dense_centerpeak_90deg_m05_r001_final.vtp"
        pts, u3 = load_vtp(vtp_path)
        x_mm, u3_mm = extract_centerline(pts, u3, axis=1, tol=0.0035)
        ax4.plot(x_mm, u3_mm, label=label, color=col, linestyle=ls, linewidth=2.0)

    ax4.set_xlabel("Transverse Coordinate $X$ (mm)")
    ax4.set_ylabel("Out-of-Plane Deflection $U_3$ (mm)")
    ax4.set_title("(d) Tolerance Bifurcation Profile $U_3(X, Y=0)$ at $res = 15$ mm")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower center", frameon=True)

    out_png = fig_dir / "orthotropic_spatial_cross_sections.png"
    fig.savefig(out_png, dpi=300)
    plt.close()
    print(f"Generated spatial cross-section convergence figure: {out_png}")


if __name__ == "__main__":
    main()
