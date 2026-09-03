#!/usr/bin/env python3
"""Run the 20-distinct-path equilibrium/continuation study strictly serially."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "bin" / "shell"
RUN_ROOT = ROOT / "run" / "equilibrium_continuation_study"

# id, rotation_deg, center_uv_mm, strips, width_mm, gtop, gbot, ortho
PATHS = (
    ("P01_center_long", 0, (0, 0), 10, 10, 1.5e-4, 0, 0),
    ("P02_center_trans", 90, (0, 0), 10, 10, 1.5e-4, 0, 0),
    ("P03_diag_pos", 45, (0, 0), 8, 12, 1.2e-4, 0, 0.2),
    ("P04_diag_neg", -45, (0, 0), 8, 12, 1.2e-4, 0, 0.2),
    ("P05_offset_east", 0, (35, 0), 8, 10, 1.0e-4, 0, 0),
    ("P06_offset_west", 0, (-35, 0), 8, 10, 1.0e-4, 0, 0),
    ("P07_offset_north", 90, (0, 45), 8, 10, 1.0e-4, 0, 0),
    ("P08_offset_south", 90, (0, -45), 8, 10, 1.0e-4, 0, 0),
    ("P09_oblique_p30", 30, (0, 0), 10, 8, 1.2e-4, 0, 0.1),
    ("P10_oblique_m30", -30, (0, 0), 10, 8, 1.2e-4, 0, 0.1),
    ("P11_oblique_p60", 60, (0, 0), 10, 8, 1.2e-4, 0, 0.1),
    ("P12_oblique_m60", -60, (0, 0), 10, 8, 1.2e-4, 0, 0.1),
    ("P13_eccentric_ne", 45, (25, 30), 6, 10, 1.0e-4, 0, 0),
    ("P14_eccentric_sw", 45, (-25, -30), 6, 10, 1.0e-4, 0, 0),
    ("P15_eccentric_nw", -45, (-25, 30), 6, 10, 1.0e-4, 0, 0),
    ("P16_eccentric_se", -45, (25, -30), 6, 10, 1.0e-4, 0, 0),
    ("P17_perim_long_e", 0, (60, 0), 6, 8, 8.0e-5, 0, -0.2),
    ("P18_perim_long_w", 0, (-60, 0), 6, 8, 8.0e-5, 0, -0.2),
    ("P19_anticlastic_bot", 90, (0, 0), 10, 12, 0, 6.0e-5, 0.3),
    ("P20_final_crown", 0, (0, 0), 12, 10, 1.5e-4, 0, 0),
)


def sequence_config() -> dict:
    toolpaths = []
    for path_id, angle, center, strips, width, gtop, gbot, ortho in PATHS:
        toolpaths.append({
            "id": path_id,
            "enabled": True,
            "repeat": 1,
            "operation": {
                "type": "zigzag", "lv_mm": 140.0, "alpha_deg": 9.13,
                "n_strips": strips, "width_mm": width,
                "center_uv_mm": list(center), "rotation_deg": angle,
                "gtop": gtop, "gbot": gbot, "ortho": ortho,
            },
        })
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": {"mode": "uniform"}},
        "toolpaths": toolpaths,
    }


def cases() -> list[dict]:
    result = []
    # Joint load-step/mesh refinement for the preferred multiplicative model.
    for res in (0.06, 0.03, 0.015):
        for step in (1.0, 0.5, 0.25):
            result.append({"name": f"joint_r{res:g}_s{step:g}", "tier": "joint",
                           "res": res, "step": step, "solver": "hlbfgs",
                           "metric": "multiplicative", "tol": 1e-12})
    # Corrector comparison on a common coarse discretization.
    for solver in ("hlbfgs", "trust_region"):
        result.append({"name": f"corrector_{solver}", "tier": "corrector",
                       "res": 0.06, "step": 0.5, "solver": solver,
                       "metric": "multiplicative", "tol": 1e-12})
    # Constitutive discrimination at common mesh/path resolution.
    for metric in ("multiplicative", "recursive_linearized", "reference_additive_linearized"):
        result.append({"name": f"constitutive_{metric}", "tier": "constitutive",
                       "res": 0.03, "step": 0.5, "solver": "hlbfgs",
                       "metric": metric, "tol": 1e-12})
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False).stdout.strip()


def summarize(case_dir: Path) -> dict:
    with (case_dir / "sequence_convergence.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    attempted = [r for r in rows if r["minimization_performed"] == "1"]
    accepted = [r for r in attempted if r["equilibrium_accepted"] == "1"]
    rejected = [r for r in attempted if r["equilibrium_accepted"] == "0"]
    cycles = {int(r["executed_cycle_index"]) for r in accepted}
    return {
        "attempts": len(attempted), "accepted_substeps": len(accepted),
        "rejected_substeps": len(rejected), "accepted_paths": len(cycles),
        "all_twenty_paths_accepted": cycles == set(range(1, 21)),
        "max_gradient_norm": max((float(r["final_gradient_norm"]) for r in accepted), default=None),
        "final_energy": float(accepted[-1]["recomputed_energy"]) if accepted else None,
    }


def run_case(case: dict, timeout: int, force: bool) -> None:
    case_dir = RUN_ROOT / case["tier"] / case["name"]
    marker = case_dir / "result.json"
    if force and case_dir.exists():
        shutil.rmtree(case_dir)
    desired_sequence = json.dumps(sequence_config(), indent=2) + "\n"
    desired_sequence_sha256 = hashlib.sha256(
        desired_sequence.encode("utf-8")).hexdigest()
    desired_binary_sha256 = sha256(SHELL)
    sequence_path = case_dir / "sequence.json"
    command = [
        str(SHELL), "-sim", "bilayer_growth", "-case", "custom",
        "-geometry", "rectangle", "-lx", "0.127", "-ly", "0.1524",
        "-res", str(case["res"]), "-h_total", "0.0006",
        "-growth_type", "zigzag_sequence", "-cycle_file", str(sequence_path.resolve()),
        "-sequence_warm_start", "true", "-metric_update", case["metric"],
        "-sequence_minimize_every", "1", "-sequence_adaptive", "true",
        "-sequence_initial_step", str(case["step"]),
        "-sequence_max_step", str(case["step"]), "-sequence_min_step", "0.015625",
        "-equilibrium_solver", case["solver"], "-equilibrium_grad_tol", str(10 * case["tol"]),
        "-tol", str(case["tol"]), "-max_iter", "50000", "-sequence_stability", "false",
        "-enable_passE", "false", "-nsteps", "1", "-seed_escape", "false",
        "-basename", case["name"], "-export_stl", "false",
    ]
    if marker.exists() and (case_dir / "manifest.json").exists():
        cached_result = json.loads(marker.read_text())
        cached_manifest = json.loads(
            (case_dir / "manifest.json").read_text())
        cache_matches = (
            cached_result.get("return_code") == 0
            and cached_result.get("convergence", {}).get(
                "all_twenty_paths_accepted") is True
            and cached_manifest.get("case") == case
            and cached_manifest.get("path_count") == len(PATHS)
            and cached_manifest.get("command") == command
            and cached_manifest.get("binary_sha256")
                == desired_binary_sha256
            and cached_manifest.get("sequence_sha256")
                == desired_sequence_sha256
        )
        if cache_matches:
            print(f"SKIP {case['name']} (matching verified result)", flush=True)
            return
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    sequence_path.write_text(desired_sequence)
    manifest = {
        "case": case, "path_count": len(PATHS), "command": command,
        "sequence_sha256": desired_sequence_sha256,
        "binary_sha256": desired_binary_sha256,
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"), "host": platform.node(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    print(f"RUN {case['name']}", flush=True)
    started = time.monotonic()
    with (case_dir / "run.log").open("w") as log:
        try:
            completed = subprocess.run(command, cwd=case_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return_code, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            return_code, timed_out = 124, True
    result = {"return_code": return_code, "timed_out": timed_out,
              "wall_seconds": time.monotonic() - started,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    if return_code == 0:
        result["convergence"] = summarize(case_dir)
        final_files = sorted(
            path for path in case_dir.glob("*.vtp")
            if "mapping" not in path.name and "pending" not in path.name)
        result["final_vtp"] = (
            final_files[-1].name if final_files else None)
    marker.write_text(json.dumps(result, indent=2) + "\n")
    if return_code != 0:
        raise RuntimeError(f"{case['name']} failed with return code {return_code}")
    if not result["convergence"]["all_twenty_paths_accepted"]:
        raise RuntimeError(f"{case['name']} did not accept all 20 paths")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("joint", "corrector", "constitutive", "all"), default="all")
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    selected = [case for case in cases()
                if (args.tier == "all" or case["tier"] == args.tier)
                and (args.case is None or case["name"] == args.case)]
    if not selected:
        raise SystemExit("no matching cases")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RUN_ROOT / ".study.lock"
    with lock_path.open("w") as study_lock:
        fcntl.flock(study_lock, fcntl.LOCK_EX)
        for case in selected:
            run_case(case, args.timeout, args.force)


if __name__ == "__main__":
    main()
