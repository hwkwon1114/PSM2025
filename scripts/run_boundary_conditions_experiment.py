#!/usr/bin/env python3
"""
Execute clamped growth and elastic springback simulations for Run 6 and Run 9
using the calibrated Attempt 209 parameter set and centered fixture clamp pad.
"""

from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "bin" / "shell"
OUT_BASE = ROOT / "run" / "boundary_conditions_experiment"
CONFIG_DIR = ROOT / "calibration" / "configs" / "boundary_conditions"


def run_case(case_name: str, cycle_json: Path) -> Path:
    case_dir = OUT_BASE / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SHELL),
        "-sim", "bilayer_growth",
        "-case", "custom",
        "-geometry", "rectangle",
        "-lx", "0.1",
        "-ly", "0.15",
        "-res", "0.01",
        "-h_total", "0.00077",
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(cycle_json.resolve()),
        "-sequence_warm_start", "true",
        "-metric_update", "multiplicative",
        "-sequence_solve_mode", "final_only",
        "-sequence_minimize_every", "1",
        "-sequence_adaptive", "true",
        "-equilibrium_solver", "native_hybrid",
        "-enable_passE", "false",
        "-nsteps", "1",
        "-tol", "3e-12",
        "-native_hybrid_grad_tol", "5e-14",
        "-equilibrium_grad_tol", "5e-14",
        "-max_iter", "50000",
        "-native_hybrid_warmup_attempts", "2000",
        "-native_hybrid_seconds", "3600",
        "-native_hybrid_export_state", "true",
        "-certify_final", "true",
        "-seed_escape", "false",
        "-basename", case_name,
        "-export_stl", "true",
        "-hessian_threads", "8",
    ]

    print(f"[{time.strftime('%X')}] Starting simulation: {case_name}...")
    print(f"  Working dir: {case_dir}")
    print(f"  Cycle config: {cycle_json}")

    log_path = case_dir / "solver.log"
    with open(log_path, "w") as log_f:
        res = subprocess.run(cmd, cwd=case_dir, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    if res.returncode != 0:
        raise RuntimeError(f"Simulation {case_name} failed with code {res.returncode}. See {log_path}")

    print(f"[{time.strftime('%X')}] Simulation {case_name} completed successfully!")
    return case_dir


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    cases = [
        ("run6_clamped_springback", CONFIG_DIR / "run6_clamped_springback_cycle.json"),
        ("run9_clamped_springback", CONFIG_DIR / "run9_clamped_springback_cycle.json"),
    ]

    results = {}
    for case_name, cycle_json in cases:
        t0 = time.time()
        case_dir = run_case(case_name, cycle_json)
        dur = time.time() - t0

        vtp_files = sorted(case_dir.glob("*.vtp"))
        results[case_name] = {
            "status": "completed",
            "directory": str(case_dir),
            "duration_seconds": dur,
            "vtp_files": [f.name for f in vtp_files],
        }

    summary_file = OUT_BASE / "simulation_execution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL BOUNDARY CONDITION SIMULATIONS COMPLETED")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v['duration_seconds']:.1f}s, {len(v['vtp_files'])} VTP meshes exported.")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
