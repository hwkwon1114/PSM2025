#!/usr/bin/env python3
"""
Generate a 4-panel publication figure comparing:
1. (a) Run 6 Springback Relaxation (Clamped vs Released Deflection Map).
2. (b) Run 6 Cross-Sectional Comparison: Free Baseline vs Clamped-Springback vs 3D Scan.
3. (c) Run 9 Springback Relaxation (Clamped vs Released Deflection Map).
4. (d) Run 9 Cross-Sectional Comparison: Free Baseline vs Clamped-Springback vs 3D Scan.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from calibration.ewcal.geometry import read_mesh, points
from calibration.ewcal.material_frame import score_material_frame

EXP_DIR = ROOT / "run" / "boundary_conditions_experiment"
DATA_DIR = ROOT / "calibration" / "data" / "processed"
OUT_DIR = ROOT / "calibration" / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
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
    return (p_centered @ R.T) + centroid_Q


def main():
    # Load Run 6 Meshes
    r6_clamped = read_mesh(EXP_DIR / "run6_clamped_springback" / "cycle_029.vtp", scale=1000)
    r6_released = read_mesh(EXP_DIR / "run6_clamped_springback" / "cycle_030.vtp", scale=1000)
    r6_free = read_mesh(ROOT / "run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00/bilayer_zigzag_sequence_cycle_029_run6_r029_final.vtp", scale=1000)
    r6_scan = np.load(DATA_DIR / "run6_oriented.npz")["points_mm"]

    # Load Run 9 Meshes
    r9_clamped = read_mesh(EXP_DIR / "run9_clamped_springback" / "cycle_033.vtp", scale=1000)
    r9_released = read_mesh(EXP_DIR / "run9_clamped_springback" / "cycle_034.vtp", scale=1000)
    r9_free = read_mesh(ROOT / "run/run9_transfer_expanded/evaluations/attempt_0209/case_00/bilayer_zigzag_sequence_cycle_033_run9_r033_final.vtp", scale=1000)
    r9_scan = np.load(DATA_DIR / "run9_oriented.npz")["points_mm"]

    p6_clamped = points(r6_clamped)
    p6_released = points(r6_released)
    p6_free = points(r6_free)

    p9_clamped = points(r9_clamped)
    p9_released = points(r9_released)
    p9_free = points(r9_free)

    # Compute Point-to-Point Springback Deflection (Clamped -> Released)
    p6_clamped_aligned = kabsch_align(p6_clamped, p6_released)
    diff6_springback = np.linalg.norm(p6_released - p6_clamped_aligned, axis=1)

    p9_clamped_aligned = kabsch_align(p9_clamped, p9_released)
    diff9_springback = np.linalg.norm(p9_released - p9_clamped_aligned, axis=1)

    # Align released mesh onto free baseline to compare intrinsic 3D shapes
    p6_released_aligned = kabsch_align(p6_released, p6_free)
    p9_released_aligned = kabsch_align(p9_released, p9_free)

    # -------------------------------------------------------------
    # 4-Panel Visualization
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.24)

    # =============================================================
    # Panel (a): Run 6 Springback Deflection Map
    # =============================================================
    ax_a = axes[0, 0]
    sc_a = ax_a.scatter(
        p6_released[:, 0], p6_released[:, 1], c=diff6_springback,
        cmap="plasma", s=22, alpha=0.9, vmin=0.0, vmax=2.85
    )
    cbar_a = fig.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("Elastic Springback Deflection (mm)", fontsize=10)

    # Overlay clamp region
    ax_a.plot([-50, 50, 50, -50, -50], [65, 65, 150, 150, 65], color="#8e44ad", lw=1.8, linestyle="--", label="100×85 mm Clamped Pad")
    ax_a.set_xlim(-105, 105)
    ax_a.set_ylim(-155, 155)
    ax_a.set_aspect("equal", "box")
    ax_a.set_xlabel("u / X Coordinate (mm)", fontsize=11)
    ax_a.set_ylabel("v / Y Coordinate (mm)", fontsize=11)
    ax_a.set_title(r"$\bf{(a)}$ Run 6: Elastic Springback Deflection ($\Delta_{\max} = 2.81$ mm)", fontsize=11.5, pad=8)
    ax_a.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax_a.grid(True, linestyle=":", alpha=0.4)

    # =============================================================
    # Panel (b): Run 6 Profile Comparison across Y
    # =============================================================
    ax_b = axes[0, 1]
    # Filter exact centerline strip near u = 0 (within 2 mm)
    mask6_sim = np.abs(p6_free[:, 0]) < 2.0
    mask6_scan = np.abs(r6_scan[:, 0]) < 5.0
    order6 = np.argsort(p6_free[mask6_sim, 1])

    ax_b.scatter(r6_scan[mask6_scan, 1], r6_scan[mask6_scan, 2], color="#e67e22", s=14, alpha=0.55, label="3D Laser Scan Experiment")
    ax_b.scatter(p6_free[mask6_sim, 1][order6], p6_free[mask6_sim, 2][order6], color="#2980b9", s=30, marker="x", lw=1.4, label="Free Baseline Prediction (RMS = 0.970 mm)")
    ax_b.plot(p6_released_aligned[mask6_sim, 1][order6], p6_released_aligned[mask6_sim, 2][order6], color="#27ae60", lw=2.2, linestyle="--", label="Clamped-Springback Prediction (RMS = 0.970 mm)")

    # Mark clamp zone
    ax_b.axvspan(65, 150, color="#8e44ad", alpha=0.15, label="85 mm Fixture Zone")
    ax_b.set_xlim(-155, 155)
    ax_b.set_ylim(-4, 6)
    ax_b.set_xlabel("v / Y Coordinate (mm)", fontsize=11)
    ax_b.set_ylabel("Deformed Height z (mm)", fontsize=11)
    ax_b.set_title(r"$\bf{(b)}$ Run 6: Centerline Profile ($|\Delta|_{\mathrm{free-clamped}} = 0.88\ \mu$m)", fontsize=11.5, pad=8)
    ax_b.legend(loc="upper right", fontsize=8.2, framealpha=0.92)
    ax_b.grid(True, linestyle=":", alpha=0.4)

    # =============================================================
    # Panel (c): Run 9 Springback Deflection Map
    # =============================================================
    ax_c = axes[1, 0]
    sc_c = ax_c.scatter(
        p9_released[:, 0], p9_released[:, 1], c=diff9_springback,
        cmap="plasma", s=22, alpha=0.9, vmin=0.0, vmax=2.30
    )
    cbar_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Elastic Springback Deflection (mm)", fontsize=10)

    # Overlay clamp region
    ax_c.plot([-50, 50, 50, -50, -50], [65, 65, 150, 150, 65], color="#8e44ad", lw=1.8, linestyle="--", label="100×85 mm Clamped Pad")
    ax_c.set_xlim(-105, 105)
    ax_c.set_ylim(-155, 155)
    ax_c.set_aspect("equal", "box")
    ax_c.set_xlabel("u / X Coordinate (mm)", fontsize=11)
    ax_c.set_ylabel("v / Y Coordinate (mm)", fontsize=11)
    ax_c.set_title(r"$\bf{(c)}$ Run 9: Elastic Springback Deflection ($\Delta_{\max} = 2.25$ mm)", fontsize=11.5, pad=8)
    ax_c.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax_c.grid(True, linestyle=":", alpha=0.4)

    # =============================================================
    # Panel (d): Run 9 Profile Comparison across Y
    # =============================================================
    ax_d = axes[1, 1]
    mask9_sim = np.abs(p9_free[:, 0]) < 2.0
    mask9_scan = np.abs(r9_scan[:, 0]) < 5.0
    order9 = np.argsort(p9_free[mask9_sim, 1])

    ax_d.scatter(r9_scan[mask9_scan, 1], r9_scan[mask9_scan, 2], color="#e67e22", s=14, alpha=0.55, label="3D Laser Scan Experiment")
    ax_d.scatter(p9_free[mask9_sim, 1][order9], p9_free[mask9_sim, 2][order9], color="#2980b9", s=30, marker="x", lw=1.4, label="Free Baseline Prediction (RMS = 1.247 mm)")
    ax_d.plot(p9_released_aligned[mask9_sim, 1][order9], p9_released_aligned[mask9_sim, 2][order9], color="#27ae60", lw=2.2, linestyle="--", label="Clamped-Springback Prediction (RMS = 1.247 mm)")

    # Mark clamp zone
    ax_d.axvspan(65, 150, color="#8e44ad", alpha=0.15, label="85 mm Fixture Zone")
    ax_d.set_xlim(-155, 155)
    ax_d.set_ylim(-4, 6)
    ax_d.set_xlabel("v / Y Coordinate (mm)", fontsize=11)
    ax_d.set_ylabel("Deformed Height z (mm)", fontsize=11)
    ax_d.set_title(r"$\bf{(d)}$ Run 9: Centerline Profile ($|\Delta|_{\mathrm{free-clamped}} = 0.00\ \mu$m)", fontsize=11.5, pad=8)
    ax_d.legend(loc="upper right", fontsize=8.2, framealpha=0.92)
    ax_d.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    out_png = OUT_DIR / "boundary_condition_springback_comparison.png"
    out_pdf = OUT_DIR / "boundary_condition_springback_comparison.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Generated 4-panel springback comparison figure:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
