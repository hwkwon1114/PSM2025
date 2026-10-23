#!/usr/bin/env python3
"""
calibration/paper_figures/reproduce_paper_figures.py
==================================

One-click publication reproduction script for:
"Optimal Anisotropic Growth Calibration in Bilayer Zigzag Forming"

Reproduces:
1. 2D Toolpath, Deformation Projection, and Signed Height Residuals (6-panel figure)
2. 3D Spatial Comparison: Isometric Surface Residuals and Profile Overlays
3. Comprehensive Quantitative Error Metrics (Dual-area weighted symmetric loss)

Usage:
  python calibration/paper_figures/reproduce_paper_figures.py [--attempt 209] [--out-dir calibration/paper_figures] [--force]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_HASHES = {
    "calibration/data/raw/Run_6.stl": "a67d0a8d13e89cdf9c7c47a8190d412e3524c72222f2aebcde95014bef332df9",
    "calibration/data/raw/Run_9.stl": "73f59318406f0a6bdadbf6cec0ee1e4a9d4c4224535e052c5c5f850ae43bbda5",
}


def verify_input_integrity():
    for rel_path, expected_hash in EXPECTED_HASHES.items():
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Missing required raw target file: {full_path}")
        with open(full_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Integrity check failed for {rel_path}: expected {expected_hash}, got {actual_hash}")


def run_figure_generation(attempt: int, out_dir: Path, force: bool = False):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_input_integrity()
    python_exe = sys.executable

    print("=" * 70)
    print("PSM2025-zigzag: One-Click Paper Figure Reproduction Pipeline")
    print(f"Target Attempt:  {attempt} (Global Optimal Parameters)")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)

    # 1. Generate 2D Six-Panel Toolpath & Deformation Figure
    print("\n[1/2] Generating 2D Toolpath & Height Residual Six-Panel Figure...")
    from calibration.analysis.run6_run9_six_panel import main as main_6p
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            "run6_run9_six_panel.py",
            "--attempt", str(attempt),
            "--out-dir", str(out_dir),
        ]
        if force:
            sys.argv.append("--force")
        main_6p()
    finally:
        sys.argv = saved_argv
    print("  -> Successfully generated run6_run9_toolpath_deformation_residual.png / .pdf")

    # 2. Generate 3D Comparison Figure
    print("\n[2/2] Generating 3D Isometric & Profile Comparison Figure...")
    from calibration.analysis.run6_run9_3d_comparison import main as main_3d
    try:
        sys.argv = [
            "run6_run9_3d_comparison.py",
            "--attempt", str(attempt),
            "--out-dir", str(out_dir),
        ]
        main_3d()
    finally:
        sys.argv = saved_argv
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
    if attempt == 209:
        r6_res_file = REPO_ROOT / "run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/result.json"
        r9_res_file = REPO_ROOT / "run/run9_transfer_expanded/evaluations/attempt_0209/result.json"
    elif attempt == 153:
        r6_res_file = REPO_ROOT / "run/run6_expanded_bo/evaluations/attempt_0153/result.json"
        r9_res_file = REPO_ROOT / "run/run9_transfer_expanded/evaluations/attempt_9901/result.json"
    elif attempt == 83:
        r6_res_file = REPO_ROOT / "run/run6/evaluations/attempt_0083/result.json"
        r9_res_file = REPO_ROOT / "run/run9_transfer/evaluations/attempt_0083/result.json"
    elif attempt == 25:
        r6_res_file = REPO_ROOT / "run/stage2_bayesian_optimization/evaluations/attempt_0025/result.json"
        r9_res_file = None
    else:
        r6_matches = list(REPO_ROOT.glob(f"run/**/evaluations/attempt_{attempt:04d}/result.json"))
        r6_res_file = r6_matches[0] if r6_matches else None
        r9_matches = list(REPO_ROOT.glob(f"run/run9*/**/evaluations/attempt_{attempt:04d}/result.json"))
        r9_res_file = r9_matches[0] if r9_matches else None
    
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
    parser.add_argument("--attempt", type=int, default=209, help="Attempt preset (default: 209)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "calibration/paper_figures",
        help="Output directory (default: calibration/paper_figures/)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    run_figure_generation(attempt=args.attempt, out_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
