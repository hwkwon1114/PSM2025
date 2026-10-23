#!/usr/bin/env python3
"""Run a resolution and tolerance study for negative-orthotropic zigzag toolpaths."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SHELL_BIN = Path("./bin/shell").resolve()

DEFAULT_TOOLPATHS = [
    ("putong_1step", Path("trajectories/putong_1step_ortho_m05.json")),
    ("ortho_2cycle", Path("trajectories/06_orthogonal_2cycle_ortho.json")),
    ("zigzag_3pass", Path("trajectories/08_zigzag_3pass_ortho_m05.json")),
]

DEFAULT_RESOLUTIONS = [0.035, 0.030, 0.025, 0.020, 0.015]
DEFAULT_TOLERANCES = [1e-8, 1e-10, 1e-12]


def parse_solver_output(log_text: str):
    """Extract convergence and performance metrics from solver stdout."""
    res = {
        "converged": False,
        "energy": None,
        "final_eps": None,
        "iterations": None,
        "eval_time": None,
        "mesh_time": None,
    }

    if "Convergence :" in log_text or "HLBFGS return value = 2" in log_text:
        res["converged"] = True

    m_energy = re.search(r"Energy went from [\d.eE+-]+ to ([\d.eE+-]+), using epsilon = ([\d.eE+-]+), final eps = ([\d.eE+-]+)", log_text)
    if m_energy:
        res["energy"] = float(m_energy.group(1))
        res["final_eps"] = float(m_energy.group(3))

    m_eval = re.search(r"\[compute energy / gradient\]:.*?([\d.]+) \([\d.]+\) s\t\((\d+) samples\)", log_text)
    if m_eval:
        res["eval_time"] = float(m_eval.group(1))
        res["iterations"] = int(m_eval.group(2))

    return res


def read_summary_csv(csv_path: Path):
    """Read the last cycle row from bilayer_zigzag_sequence_summary.csv."""
    if not csv_path.exists():
        return {}
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            return rows[-1]
    return {}


def run_single_simulation(
    bin_path: Path,
    toolpath_file: Path,
    res: float,
    tol: float,
    out_dir: Path,
    threads: int = 4,
    timeout: int = 1800,
):
    """Run a single shell simulation with specified res and tol."""
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = f"run_res{res:.4f}_tol{tol:.1e}"
    log_file = out_dir / f"{basename}.log"
    summary_file = out_dir / "bilayer_zigzag_sequence_summary.csv"

    cmd = [
        str(bin_path),
        "-sim", "bilayer_growth",
        "-case", "custom",
        "-geometry", "rectangle",
        "-lx", "0.127",
        "-ly", "0.1524",
        "-res", str(res),
        "-h_total", "0.0006",
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(toolpath_file),
        "-minimizer", "hlbfgs",
        "-tol", str(tol),
        "-hessian_threads", str(threads),
        "-basename", str(out_dir / basename),
        "-export_stl", "false",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(out_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.perf_counter() - t0
        log_text = proc.stdout
        with open(str(log_file), "w") as f:
            f.write(log_text)
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        with open(str(log_file), "w") as f:
            f.write(f"TIMEOUT expired after {timeout} seconds\n")
        returncode = -9
        log_text = ""

    parsed = parse_solver_output(log_text)
    summary_data = read_summary_csv(summary_file)

    # Extract deflection and energy from summary if available
    max_disp = float(summary_data.get("max_displacement", 0.0)) if "max_displacement" in summary_data else None
    min_u3 = float(summary_data.get("min_U3", 0.0)) if "min_U3" in summary_data else None
    max_u3 = float(summary_data.get("max_U3", 0.0)) if "max_U3" in summary_data else None
    summary_energy = float(summary_data.get("total_energy", 0.0)) if "total_energy" in summary_data else None

    energy = summary_energy if summary_energy is not None else parsed.get("energy")

    return {
        "toolpath": toolpath_file.stem,
        "res": res,
        "tol": tol,
        "wall_time_s": round(elapsed, 2),
        "returncode": returncode,
        "converged": parsed["converged"],
        "iterations": parsed.get("iterations"),
        "energy": energy,
        "max_displacement": max_disp,
        "min_U3": min_u3,
        "max_U3": max_u3,
        "u3_span": (max_u3 - min_u3) if (max_u3 is not None and min_u3 is not None) else None,
        "log_file": str(log_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Resolution and Tolerance Study with Negative Orthotropy")
    parser.add_argument("--toolpaths", nargs="+", default=["putong_1step", "ortho_2cycle"], help="Toolpath keys to run")
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.035, 0.030, 0.025, 0.020], help="Resolutions (m)")
    parser.add_argument("--tolerances", nargs="+", type=float, default=[1e-8, 1e-10, 1e-12], help="Tolerances")
    parser.add_argument("--output_dir", type=str, default="run/orthotropic_study", help="Output directory")
    parser.add_argument("--threads", type=int, default=4, help="Hessian assembly OpenMP threads")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-run timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that have already completed")
    args = parser.parse_args()

    out_base = Path(args.output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    toolpath_map = dict(DEFAULT_TOOLPATHS)
    selected_tps = [(k, toolpath_map[k]) for k in args.toolpaths if k in toolpath_map]

    total_runs = len(selected_tps) * len(args.resolutions) * len(args.tolerances)
    print(f"=== Resolution and Tolerance Study (Negative Orthotropy) ===")
    print(f"Toolpaths: {[k for k, _ in selected_tps]}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Tolerances: {args.tolerances}")
    print(f"Total simulations to execute: {total_runs}")
    print(f"Output directory: {out_base}")
    print(f"Parallel threads: {args.threads}")

    if args.dry_run:
        print("[Dry Run] Exiting without execution.")
        return

    csv_out = out_base / "study_results_summary.csv"
    existing_results = {}
    if csv_out.exists():
        with open(csv_out, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("toolpath"), float(row.get("res")), float(row.get("tol")))
                existing_results[key] = row

    results = []
    run_idx = 0

    for tp_name, tp_path in selected_tps:
        for res in sorted(args.resolutions, reverse=True):
            for tol in sorted(args.tolerances, reverse=True):
                run_idx += 1
                case_dir = out_base / tp_name / f"res_{res:.4f}_tol_{tol:.1e}"
                key = (tp_path.stem, float(res), float(tol))
                
                # Check if we can skip
                summary_file = case_dir / "bilayer_zigzag_sequence_summary.csv"
                if args.skip_existing and key in existing_results and summary_file.exists():
                    print(f"[{run_idx}/{total_runs}] Skipping existing {tp_name} res={res:.4f} tol={tol:.1e}", flush=True)
                    results.append(existing_results[key])
                    continue

                print(f"[{run_idx}/{total_runs}] Running {tp_name} res={res:.4f} tol={tol:.1e}...", flush=True)
                
                res_dict = run_single_simulation(
                    bin_path=SHELL_BIN,
                    toolpath_file=tp_path.resolve(),
                    res=res,
                    tol=tol,
                    out_dir=case_dir,
                    threads=args.threads,
                    timeout=args.timeout,
                )
                results.append(res_dict)
                existing_results[key] = res_dict
                print(f"    -> Status: {'OK' if res_dict['converged'] else 'FAIL'}, Time: {res_dict['wall_time_s']}s, Energy: {res_dict['energy']}, U3 Span: {res_dict['u3_span']}", flush=True)

                # Write incremental CSV
                with open(csv_out, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)

    print(f"\nStudy completed. Results saved to {csv_out}")


if __name__ == "__main__":
    main()
