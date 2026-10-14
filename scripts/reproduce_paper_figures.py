#!/usr/bin/env python3
"""
scripts/reproduce_paper_figures.py
==================================

One-click publication reproduction script for:
"Optimal Anisotropic Growth Calibration in Bilayer Zigzag Forming"

Reproduces:
1. 2D Toolpath, Deformation Projection, and Signed Height Residuals (6-panel figure)
2. 3D Spatial Comparison: Isometric Surface Residuals and Profile Overlays
3. Comprehensive Quantitative Error Metrics (Dual-area weighted symmetric loss)

Usage:
  python scripts/reproduce_paper_figures.py [--attempt 153] [--out-dir paper_figures] [--force]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def run_figure_generation(attempt: int, out_dir: Path, force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable

    print("=" * 70)
    print("PSM2025-zigzag: One-Click Paper Figure Reproduction Pipeline")
    print(f"Target Attempt:  {attempt} (Global Optimal Parameters)")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)

    # 1. Generate 2D Six-Panel Toolpath & Deformation Figure
    print("\n[1/2] Generating 2D Toolpath & Height Residual Six-Panel Figure...")
    cmd_6panel = [
        python_exe,
        str(REPO_ROOT / "calibration/analysis/run6_run9_six_panel.py"),
        "--attempt", str(attempt),
        "--out-dir", str(out_dir),
    ]
    if force:
        cmd_6panel.append("--force")
    
    res_6p = subprocess.run(cmd_6panel, capture_output=True, text=True)
    if res_6p.returncode != 0:
        print(f"ERROR generating six-panel figure:\n{res_6p.stderr}", file=sys.stderr)
        sys.exit(res_6p.returncode)
    print("  -> Successfully generated run6_run9_toolpath_deformation_residual.png / .pdf")

    # 2. Generate 3D Comparison Figure
    print("\n[2/2] Generating 3D Isometric & Profile Comparison Figure...")
    cmd_3d = [
        python_exe,
        str(REPO_ROOT / "calibration/analysis/run6_run9_3d_comparison.py"),
        "--attempt", str(attempt),
        "--out-dir", str(out_dir),
    ]
    res_3d = subprocess.run(cmd_3d, capture_output=True, text=True)
    if res_3d.returncode != 0:
        print(f"ERROR generating 3D comparison figure:\n{res_3d.stderr}", file=sys.stderr)
        sys.exit(res_3d.returncode)
    print("  -> Successfully generated run6_run9_3d_deformation_residual.png / .pdf")

    # 3. Verify Generated Outputs
    expected_files = [
        out_dir / "run6_run9_toolpath_deformation_residual.png",
        out_dir / "run6_run9_toolpath_deformation_residual.pdf",
        out_dir / "run6_run9_3d_deformation_residual.png",
        out_dir / "run6_run9_3d_deformation_residual.pdf",
        out_dir / "provenance.json",
        out_dir / "run6_run9_toolpath_deformation_residual.caption.md",
        out_dir / "run6_run9_3d_deformation_residual.caption.md",
    ]

    missing = [str(f) for f in expected_files if not f.exists() or f.stat().st_size == 0]
    if missing:
        print(f"ERROR: Missing expected output files: {missing}", file=sys.stderr)
        sys.exit(1)

    # 4. Print Quantitative Reproduction Summary
    r6_res_file = REPO_ROOT / "run/run6_expanded_bo/evaluations/attempt_0153/result.json"
    r9_res_file = REPO_ROOT / "run/run9_transfer_expanded/evaluations/attempt_9901/result.json"
    
    if r6_res_file.exists():
        with open(r6_res_file) as f:
            r6_data = json.load(f)
        p = r6_data.get("parameters", {})
        r6_m = r6_data["cases"][0]["metrics"]
        
        print("\n" + "=" * 70)
        print("REPRODUCTION VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Calibrated Optimal Parameters (Attempt {attempt}):")
        print(f"  Growth Top (g_top):          {float(p.get('gtop', 0)):.6e}")
        print(f"  Growth Bottom (g_bottom):    {float(p.get('gbot', 0)):.6e}")
        print(f"  Orthotropy Top:              {float(p.get('ortho_top', 0)):.6f}")
        print(f"  Orthotropy Bottom:           {float(p.get('ortho_bottom', 0)):.6f}")
        print("-" * 70)
        print("Quantitative Metric Breakdown:")
        print(f"  Run 6 (Training / Calibration Target):")
        print(f"    - Symmetric Dual-Area Weighted RMS:  {float(r6_m.get('rms_symmetric_mm', 0)):.4f} mm")
        print(f"    - Single-Sided Forward RMS:          {float(r6_m.get('rms_target_to_sim_mm', 0)):.4f} mm")
        print(f"    - Single-Sided Backward RMS:         {float(r6_m.get('rms_sim_to_target_weighted_mm', 0)):.4f} mm")

        if r9_res_file.exists():
            with open(r9_res_file) as f:
                r9_data = json.load(f)
            r9_m = r9_data["cases"][0]["metrics"]
            print(f"  Run 9 (Held-Out Independent Validation):")
            print(f"    - Symmetric Dual-Area Weighted RMS:  {float(r9_m.get('rms_symmetric_mm', 0)):.4f} mm")
            print(f"    - Single-Sided Forward RMS:          {float(r9_m.get('rms_target_to_sim_mm', 0)):.4f} mm")
            print(f"    - Single-Sided Backward RMS:         {float(r9_m.get('rms_sim_to_target_weighted_mm', 0)):.4f} mm")
            
        print("=" * 70)
        print("ALL PAPER FIGURES AND ARTIFACTS VERIFIED SUCCESSFULLY.")
        print(f"Artifacts located at: {out_dir.resolve()}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="One-click publication reproduction script for PSM2025-zigzag."
    )
    parser.add_argument("--attempt", type=int, default=153, help="Attempt preset (default: 153)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "paper_figures",
        help="Output directory (default: paper_figures/)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    run_figure_generation(attempt=args.attempt, out_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
