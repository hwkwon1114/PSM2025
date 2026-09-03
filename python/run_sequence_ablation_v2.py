#!/usr/bin/env python3
"""Generate and execute the sequence-ablation v2 cases strictly serially."""
from __future__ import annotations

import argparse
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
RUN_ROOT = ROOT / "run" / "sequence_ablation_v2"
SHELL = ROOT / "bin" / "shell"
STAGES = ("repeatability", "mesh_tolerance", "warm_start", "metric_update", "segmentation")


def operation(growth: float, rotation: float = 0.0, ortho: float = 0.0,
              active: list[int] | None = None) -> dict:
    op = {
        "type": "zigzag", "lv_mm": 140.0, "alpha_deg": 9.13,
        "n_strips": 10, "width_mm": 10.0, "rotation_deg": rotation,
        "gtop": growth, "gbot": 0.0, "ortho": ortho,
    }
    if active is not None:
        op["active_strips"] = active
    return op


def config(ops: list[tuple[str, int, dict]]) -> dict:
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": {"mode": "uniform"}},
        "toolpaths": [
            {"id": name, "enabled": True, "repeat": repeat, "operation": op}
            for name, repeat, op in ops
        ],
    }


def standard_case(name: str, stage: str, growth: float = 2e-5,
                  res: float = 0.03, tol: float = 1e-12,
                  warm: bool = True, metric: str = "multiplicative") -> dict:
    return {"name": name, "stage": stage, "res": res, "tol": tol,
            "warm": warm, "metric": metric, "minimize_every": 1,
            "config": config([("putong", 12, operation(growth))])}


def cases() -> list[dict]:
    result: list[dict] = []
    for i in range(1, 6):
        result.append(standard_case(f"repeat_{i:02d}", "repeatability"))
    result += [
        standard_case("mesh_res_006", "mesh_tolerance", res=0.06),
        standard_case("mesh_res_0015", "mesh_tolerance", res=0.015),
        standard_case("tol_1e10", "mesh_tolerance", tol=1e-10),
        standard_case("tol_1e14", "mesh_tolerance", tol=1e-14),
        standard_case("cold_g2e5", "warm_start", warm=False),
        standard_case("warm_g1e4", "warm_start", growth=1e-4),
        standard_case("cold_g1e4", "warm_start", growth=1e-4, warm=False),
    ]
    for growth, label in ((2e-5, "2e5"), (1e-4, "1e4"), (5e-4, "5e4")):
        for metric in ("multiplicative", "recursive_linearized", "reference_additive_linearized"):
            if growth == 2e-5 and metric == "multiplicative":
                continue
            result.append(standard_case(f"metric_{metric}_{label}", "metric_update",
                                        growth=growth, metric=metric))
    alternating = [(f"rot_{i:02d}", 1, operation(1e-4, 0.0 if i % 2 == 0 else 45.0, 0.8))
                   for i in range(12)]
    for name, seq in (("noncommuting45_forward", alternating),
                      ("noncommuting45_reverse", list(reversed(alternating)))):
        result.append({"name": name, "stage": "metric_update", "res": 0.03,
                       "tol": 1e-12, "warm": True, "metric": "multiplicative",
                       "minimize_every": 12, "config": config(seq)})

    path_major = [(f"path_{p}_strip_{s}", 1, operation(2e-5, active=[s]))
                  for p in range(3) for s in range(10)]
    stroke_major = [(f"strip_{s}", 3, operation(2e-5, active=[s])) for s in range(10)]
    reverse_stroke = [(f"strip_{s}", 3, operation(2e-5, active=[s])) for s in reversed(range(10))]
    schedules = (
        ("seg_complete_path", [("complete", 3, operation(2e-5))], 1),
        ("seg_path_each", path_major, 1),
        ("seg_path_every3", path_major, 3),
        ("seg_stroke_major", stroke_major, 3),
        ("seg_stroke_reverse", reverse_stroke, 3),
        ("seg_single_solve", path_major, 30),
    )
    for name, ops, cadence in schedules:
        result.append({"name": name, "stage": "segmentation", "res": 0.03,
                       "tol": 1e-12, "warm": True, "metric": "multiplicative",
                       "minimize_every": cadence, "config": config(ops)})
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


def run_case(case: dict, timeout: int, force: bool = False) -> None:
    case_dir = RUN_ROOT / case["stage"] / case["name"]
    if force and case_dir.exists():
        shutil.rmtree(case_dir)
    marker = case_dir / "result.json"
    if marker.exists() and json.loads(marker.read_text()).get("return_code") == 0:
        print(f"SKIP {case['name']} (successful result exists)", flush=True)
        return
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "sequence.json"
    config_path.write_text(json.dumps(case["config"], indent=2) + "\n")
    command = [
        str(SHELL), "-sim", "bilayer_growth", "-case", "custom",
        "-geometry", "rectangle", "-lx", "0.127", "-ly", "0.1524",
        "-res", str(case["res"]), "-h_total", "0.0006",
        "-growth_type", "zigzag_sequence", "-cycle_file", str(config_path.resolve()),
        "-sequence_warm_start", str(case["warm"]).lower(),
        "-metric_update", case["metric"],
        "-sequence_minimize_every", str(case["minimize_every"]),
        "-sequence_adaptive", "false",
        "-equilibrium_solver", "hlbfgs",
        "-enable_passE", "false", "-nsteps", "1", "-tol", str(case["tol"]),
        "-max_iter", "50000", "-certify_final", "true",
        "-seed_escape", "false", "-basename", case["name"], "-export_stl", "false",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    manifest = {
        "case": {k: v for k, v in case.items() if k != "config"},
        "command": command, "command_text": " ".join(command),
        "binary_sha256": sha256(SHELL), "git_revision": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"), "omp_num_threads": "8",
        "host": platform.node(), "platform": platform.platform(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"RUN {case['name']}", flush=True)
    started = time.monotonic()
    with (case_dir / "run.log").open("w") as log:
        try:
            completed = subprocess.run(command, cwd=case_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return_code, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            return_code, timed_out = 124, True
    final_files = sorted(str(p.relative_to(case_dir)) for p in case_dir.glob("*.vtp")
                         if "mapping" not in p.name and "pending" not in p.name)
    result = {"return_code": return_code, "timed_out": timed_out,
              "wall_seconds": time.monotonic() - started,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "final_vtp": final_files[-1] if final_files else None}
    marker.write_text(json.dumps(result, indent=2) + "\n")
    if return_code != 0:
        raise RuntimeError(f"{case['name']} failed with return code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    selected = [c for c in cases() if (args.stage == "all" or c["stage"] == args.stage)
                and (args.case is None or c["name"] == args.case)]
    if not selected:
        raise SystemExit("no matching cases")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "reports" / "sequence_ablation_v2_spec.md", RUN_ROOT / "spec.md")
    for case in selected:
        run_case(case, args.timeout, args.force)


if __name__ == "__main__":
    main()
