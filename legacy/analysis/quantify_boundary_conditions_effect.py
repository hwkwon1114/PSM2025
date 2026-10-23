#!/usr/bin/env python3
"""
Quantify the effect of persistent 85 mm fixture zone clamping followed by
elastic springback on Run 6 and Run 9 under calibrated Attempt 209 parameters.

Uses production material-reference scoring against experimental 3D scans.
"""

from pathlib import Path
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from calibration.ewcal.geometry import read_mesh, points
from calibration.ewcal.material_frame import score_material_frame

EXP_DIR = ROOT / "run" / "boundary_conditions_experiment"
DATA_DIR = ROOT / "calibration" / "data" / "processed"


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


def analyze_run(
    case_name: str,
    clamped_vtp: Path,
    released_vtp: Path,
    free_vtp: Path,
    state_json: Path,
    scan_npz: Path,
) -> dict:
    mesh_clamped = read_mesh(clamped_vtp, scale=1000)
    mesh_released = read_mesh(released_vtp, scale=1000)
    mesh_free = read_mesh(free_vtp, scale=1000)

    pts_clamped = points(mesh_clamped)
    pts_released = points(mesh_released)
    pts_free = points(mesh_free)

    # 1. Springback: Clamped -> Released
    _, rms_springback, max_springback = kabsch_align(pts_clamped, pts_released)

    # 2. Comparison vs. Free Baseline: Released vs. Free
    _, rms_vs_free, max_vs_free = kabsch_align(pts_released, pts_free)

    # 3. Curvatures (on millimeter coordinates, curvature in mm^-1, converted to m^-1)
    k1_clamp, k2_clamp, z_clamp = estimate_quadric_curvature(pts_clamped / 1e3)
    k1_rel, k2_rel, z_rel = estimate_quadric_curvature(pts_released / 1e3)
    k1_free, k2_free, z_free = estimate_quadric_curvature(pts_free / 1e3)

    # 4. Production Material-Reference Scoring against 3D Scans
    scan_data = np.load(scan_npz)
    scan_pts_mm = scan_data["points_mm"]

    st_data = json.loads(state_json.read_text())
    st_ref_mm = np.asarray(st_data["reference_positions"], dtype=float) * 1000.0
    st_faces = np.asarray(st_data["faces"], dtype=int)

    metrics_free, _, _, _, _ = score_material_frame(mesh_free, scan_pts_mm, st_ref_mm, st_faces)
    metrics_released, _, _, _, _ = score_material_frame(mesh_released, scan_pts_mm, st_ref_mm, st_faces)
    metrics_clamped, _, _, _, _ = score_material_frame(mesh_clamped, scan_pts_mm, st_ref_mm, st_faces)

    return {
        "case_name": case_name,
        "num_vertices": len(pts_released),
        "springback_relaxation": {
            "rms_diff_um": rms_springback * 1e3,  # pts in mm -> * 1e3 = um
            "max_diff_um": max_springback * 1e3,
            "rms_diff_mm": rms_springback,
            "max_diff_mm": max_springback,
        },
        "deviation_from_free_baseline": {
            "aligned_rms_um": rms_vs_free * 1e3,
            "aligned_max_um": max_vs_free * 1e3,
            "aligned_rms_mm": rms_vs_free,
            "aligned_max_mm": max_vs_free,
        },
        "curvatures": {
            "clamped": {"k1": k1_clamp, "k2": k2_clamp, "z_span_mm": z_clamp * 1e3},
            "released_springback": {"k1": k1_rel, "k2": k2_rel, "z_span_mm": z_rel * 1e3},
            "free_baseline": {"k1": k1_free, "k2": k2_free, "z_span_mm": z_free * 1e3},
            "delta_k1_released_vs_free": abs(k1_rel - k1_free),
            "delta_k2_released_vs_free": abs(k2_rel - k2_free),
            "delta_z_span_released_vs_free_mm": abs(z_rel - z_free) * 1e3,
        },
        "scan_alignment": {
            "free_baseline": {
                "rms_symmetric_mm": float(metrics_free["rms_symmetric_mm"]),
                "rms_target_to_sim_mm": float(metrics_free["rms_target_to_sim_mm"]),
                "rms_sim_to_target_weighted_mm": float(metrics_free["rms_sim_to_target_weighted_mm"]),
            },
            "clamped_springback": {
                "rms_symmetric_mm": float(metrics_released["rms_symmetric_mm"]),
                "rms_target_to_sim_mm": float(metrics_released["rms_target_to_sim_mm"]),
                "rms_sim_to_target_weighted_mm": float(metrics_released["rms_sim_to_target_weighted_mm"]),
            },
            "in_situ_clamped": {
                "rms_symmetric_mm": float(metrics_clamped["rms_symmetric_mm"]),
                "rms_target_to_sim_mm": float(metrics_clamped["rms_target_to_sim_mm"]),
                "rms_sim_to_target_weighted_mm": float(metrics_clamped["rms_sim_to_target_weighted_mm"]),
            },
            "delta_symmetric_rms_mm": float(metrics_released["rms_symmetric_mm"] - metrics_free["rms_symmetric_mm"]),
            "delta_symmetric_rms_pct": float((metrics_released["rms_symmetric_mm"] - metrics_free["rms_symmetric_mm"]) / metrics_free["rms_symmetric_mm"] * 100),
        },
    }


def main():
    run6_clamped = EXP_DIR / "run6_clamped_springback" / "cycle_029.vtp"
    run6_released = EXP_DIR / "run6_clamped_springback" / "cycle_030.vtp"
    run6_free = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "evaluations" / "attempt_0209" / "case_00" / "bilayer_zigzag_sequence_cycle_029_run6_r029_final.vtp"
    run6_state = EXP_DIR / "run6_clamped_springback" / "bilayer_zigzag_sequence_BC_native_hybrid_state.json"
    run6_scan = DATA_DIR / "run6_oriented.npz"

    run9_clamped = EXP_DIR / "run9_clamped_springback" / "cycle_033.vtp"
    run9_released = EXP_DIR / "run9_clamped_springback" / "cycle_034.vtp"
    run9_free = ROOT / "run" / "run9_transfer_expanded" / "evaluations" / "attempt_0209" / "case_00" / "bilayer_zigzag_sequence_cycle_033_run9_r033_final.vtp"
    run9_state = EXP_DIR / "run9_clamped_springback" / "bilayer_zigzag_sequence_BC_native_hybrid_state.json"
    run9_scan = DATA_DIR / "run9_oriented.npz"

    print("Analyzing Run 6 boundary condition effects...")
    res6 = analyze_run("run6", run6_clamped, run6_released, run6_free, run6_state, run6_scan)

    print("Analyzing Run 9 boundary condition effects...")
    res9 = analyze_run("run9", run9_clamped, run9_released, run9_free, run9_state, run9_scan)

    summary = {
        "run6": res6,
        "run9": res9,
    }

    out_file = EXP_DIR / "boundary_conditions_quantification.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("BOUNDARY CONDITIONS QUANTIFICATION SUMMARY")
    print("=" * 80)

    for case, d in [("Run 6 (Lv = 130 mm, 29 passes)", res6), ("Run 9 (Lv = 187.5 mm, 33 passes)", res9)]:
        sb = d["springback_relaxation"]
        dev = d["deviation_from_free_baseline"]
        c = d["curvatures"]
        scan = d["scan_alignment"]

        print(f"\n[{case}]")
        print(f"  Internal Springback Relaxation (Clamped -> Released):")
        print(f"    RMS Deflection = {sb['rms_diff_um']:.2f} um ({sb['rms_diff_mm']:.4f} mm), Max = {sb['max_diff_mm']:.4f} mm")
        print(f"  Deviation from Free Baseline (Released Springback vs Free):")
        print(f"    Aligned 3D RMS = {dev['aligned_rms_um']:.2f} um ({dev['aligned_rms_mm']:.4f} mm), Max = {dev['aligned_max_mm']:.4f} mm")
        print(f"  Curvatures:")
        print(f"    Free Baseline       : k1 = {c['free_baseline']['k1']:.5f}, k2 = {c['free_baseline']['k2']:.5f} m^-1, Z = {c['free_baseline']['z_span_mm']:.2f} mm")
        print(f"    Clamped-Springback  : k1 = {c['released_springback']['k1']:.5f}, k2 = {c['released_springback']['k2']:.5f} m^-1, Z = {c['released_springback']['z_span_mm']:.2f} mm")
        print(f"    Delta Curvature     : |dk1| = {c['delta_k1_released_vs_free']:.5f} m^-1, |dZ| = {c['delta_z_span_released_vs_free_mm']:.3f} mm")
        print(f"  Experimental 3D Scan Alignment:")
        print(f"    Free Baseline Symmetric RMS       : {scan['free_baseline']['rms_symmetric_mm']:.4f} mm")
        print(f"    Clamped-Springback Symmetric RMS  : {scan['clamped_springback']['rms_symmetric_mm']:.4f} mm")
        print(f"    Delta vs Free                     : {scan['delta_symmetric_rms_mm']:+.4f} mm ({scan['delta_symmetric_rms_pct']:+.2f}%)")

    print(f"\nSaved quantification metrics to: {out_file}")


if __name__ == "__main__":
    main()
