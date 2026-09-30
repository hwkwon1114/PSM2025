#!/usr/bin/env python3
"""Forward-model diagnostics for the recurring zigzag sequence.

Three finite, strictly serial families:

  nested     the existing crown_nested_broad_to_central recipe, byte-identical,
             run by both implementations (pinned upstream and current).
  diversity  five single-cycle sequence cases that probe distinct eigenstrain
             geometries (isotropic top-only, equal-layer central, opposing-layer
             wave, anisotropic rotated pair), run by both implementations.
  frozen     current-only. The nested recipe plus one appended zero-growth probe
             cycle. -prescribed_at_cycle imposes a seed shape at the boundary
             after the last growing cycle, so the probe cycle re-solves the SAME
             frozen target metrics from four different initial states.

The diversity toolpaths are dense serpentines with pitch equal to strip width.
Inclined strips and endpoint caps can overlap; the actual hit-count and metric
fields, not nominal bounding boxes, determine the applied loading.

Upstream (b165fc52) has no -sequence_*, -metric_update, -equilibrium_* or
-prescribed_* flags and its sequence loader rejects unknown JSON keys, so the
shared cases use only flags and keys that exist in both revisions and never
use active_strips. Nothing is passed to a binary that cannot read it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "run" / "forward_model_diagnostics"
UPSTREAM_ROOT = RUN_ROOT / "upstream_source"
UPSTREAM_PIN = "b165fc52c24f0f29a32edcbc6c4759a0851579fe"

NESTED_SEQUENCE = (ROOT / "run" / "solver_characteristics" / "shrinking_crown" /
                   "crown_nested_broad_to_central" / "sequence.json")
MATERIAL_BASELINE = (ROOT / "run" / "solver_characteristics" /
                     "prescribed_deformation" / "prescribed_crown_reequilibrated" /
                     "bilayer_zigzag_sequence_prescribed_reeq_after_cycle_004"
                     "_imposed_vertices.csv")

# Plate and mesh are fixed for every case: half-lengths in metres.
LX, LY, RES, H_TOTAL = 0.127, 0.1524, 0.03, 0.0006
# Per-hit top-layer engineering strain of the reused nested recipe.
GROWTH = 1.2e-4
# Seed amplitude convention already used by the prescribed_deformation family.
SEED_AMPLITUDE = 0.004
FAMILIES = ("nested", "diversity", "frozen", "crosswheel")
IMPLEMENTATIONS = {
    "current": {"binary": ROOT / "bin" / "shell", "repo": ROOT, "pin": None},
    "upstream": {"binary": UPSTREAM_ROOT / "bin" / "shell",
                 "repo": UPSTREAM_ROOT, "pin": UPSTREAM_PIN},
}
RETCODE_RE = re.compile(r"HLBFGS return value = (-?\d+)")
GNORM_RE = re.compile(r"final eps = ([0-9.eE+-]+)")
CYCLE_VTP_RE = re.compile(r"_cycle_(\d+)_.*_final\.vtp$")


def serpentine(tid: str, lv_mm: float, width_mm: float, span_mm: float,
               gtop, gbot=0.0, ortho: float = 0.0,
               rotation_deg: float = 0.0) -> dict:
    """Contiguous serpentine toolpath spanning span_mm laterally."""
    n_strips = 2 + max(1, math.ceil(span_mm / width_mm))
    op = {
        "type": "zigzag",
        "lv_mm": lv_mm,
        "alpha_deg": math.degrees(math.atan(width_mm / lv_mm)),
        "n_strips": n_strips,
        "width_mm": width_mm,
        "center_uv_mm": [0.0, 0.0],
        "rotation_deg": rotation_deg,
        "gtop": gtop(n_strips) if callable(gtop) else gtop,
        "gbot": gbot(n_strips) if callable(gbot) else gbot,
        "ortho": ortho,
    }
    return {"id": tid, "enabled": True, "repeat": 1, "operation": op}


def sequence(toolpaths: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": {"mode": "uniform"}},
        "toolpaths": toolpaths,
    }


def alternating(value: float, phase: int):
    return lambda n: [value if i % 2 == phase else 0.0 for i in range(n)]


def nested_toolpaths() -> list[dict]:
    if not NESTED_SEQUENCE.is_file():
        raise SystemExit(f"missing reused nested recipe: {NESTED_SEQUENCE}")
    return json.loads(NESTED_SEQUENCE.read_text())["toolpaths"]


def diversity_cases() -> list[dict]:
    """Single-cycle eigenstrain-geometry probes; both implementations."""
    return [
        {"name": "iso_top_full", "family": "diversity",
         "note": "isotropic (ortho=0) top-only growth over a broad serpentine; "
                 "diagnoses cylinder versus two-directional response without assuming a dome",
         "sequence": sequence([serpentine("F1_iso_top_full", 288.0, 12.0, 240.0,
                                          GROWTH)])},
        {"name": "equal_layer_central", "family": "diversity",
         "note": "gtop = gbot over a central 110 x 110 mm serpentine: pure "
                 "membrane load with no bending drive, so any out-of-plane "
                 "response is a buckling branch selected by the solver",
         "sequence": sequence([serpentine("C1_equal_layer_central", 100.0, 10.0,
                                          100.0, GROWTH, gbot=GROWTH)])},
        {"name": "opposing_layer_wave", "family": "diversity",
         "note": "opposing-layer treatment: alternate strips grow top-only or "
                 "bottom-only (ortho=1, uniaxial along the strip), giving a "
                 "48 mm-period corrugation drive",
         "sequence": sequence([serpentine("W1_opposing_wave", 276.0, 24.0, 216.0,
                                          alternating(GROWTH, 0),
                                          gbot=alternating(GROWTH, 1),
                                          ortho=1.0)])},
        {"name": "aniso_rot000", "family": "diversity",
         "note": "uniaxial (ortho=1) top-only growth on a 210 x 210 mm square "
                 "footprint, pattern rotation 0 deg",
         "sequence": sequence([serpentine("A1_aniso_rot000", 200.0, 10.0, 200.0,
                                          GROWTH, ortho=1.0)])},
        {"name": "aniso_rot090", "family": "diversity",
         "note": "rotated counterpart of aniso_rot000; nominal footprint is "
                 "square but the discrete hit field and rectangular plate are "
                 "not rotation-invariant",
         "sequence": sequence([serpentine("A1_aniso_rot090", 200.0, 10.0, 200.0,
                                          GROWTH, ortho=1.0, rotation_deg=90.0)])},
    ]


def crosswheel_cases():
    from run_solver_characteristics import crown_path, CROWN_SQUARE_HALF_EXTENTS_MM
    schedules = {
        'single': (0, 0, 0, 0, 0, 0),
        'alternating': (0, 90, 0, 90, 0, 90),
        'blocked': (0, 0, 0, 90, 90, 90),
    }
    return [
        dict(name=f'crosswheel_{name}', family='crosswheel', res=0.015,
             implementations=('current',),
             note='Centered square nested paths; isotropic top-only growth; '
                  'only direction schedule differs. Actual hit maps determine loading.',
             sequence=sequence([
                 crown_path(i, f'X{i+1}_rot{angle:03d}',
                            ladder=CROWN_SQUARE_HALF_EXTENTS_MM,
                            stagger=False, rotation_deg=angle)
                 for i, angle in enumerate(angles)]))
        for name, angles in schedules.items()]


def frozen_cases() -> list[dict]:
    """Current-only: identical frozen final metrics, four initial states."""
    nested = nested_toolpaths()
    growing_cycles = sum(int(t.get("repeat", 1)) for t in nested)
    probe = json.loads(json.dumps(nested[-1]))
    probe["id"] = "Z_zero_growth_probe"
    probe["repeat"] = 1
    probe["operation"]["gtop"] = 0.0
    probe["operation"]["gbot"] = 0.0
    seq = sequence(nested + [probe])
    cases = [{"name": "frozen_control", "family": "frozen", "seed": None,
              "implementations": ("current",), "sequence": seq,
              "note": "unperturbed control: the zero-growth probe cycle re-solves "
                      "from the warm state reached after cycle "
                      f"{growing_cycles}"}]
    for seed in ("flat", "crown", "cyl_u", "cyl_v"):
        cases.append({
            "name": f"frozen_{seed}", "family": "frozen", "seed": seed,
            "implementations": ("current",), "sequence": seq,
            "prescribed": {"at_cycle": growing_cycles, "tag": f"frozen_{seed}"},
            "note": f"'{seed}' seed imposed from rest at the boundary after cycle "
                    f"{growing_cycles}; the zero-growth probe cycle then solves the "
                    "identical frozen target metrics from that initial state",
        })
    return cases


def cases() -> list[dict]:
    nested = [{"name": "nested_baseline", "family": "nested",
               "note": "byte-identical reuse of "
                       f"{NESTED_SEQUENCE.relative_to(ROOT)}",
               "sequence": json.loads(NESTED_SEQUENCE.read_text())}]
    return nested + diversity_cases() + frozen_cases() + crosswheel_cases()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False).stdout.strip()


def material_baseline(path: Path) -> list[tuple[float, float]]:
    """Per-vertex (material_u, material_v) in mesh vertex order, validated."""
    rows: dict[int, tuple[float, float]] = {}
    with path.open() as stream:
        header = next(stream).strip().split(",")
        if header[:3] != ["vertex_index", "material_u", "material_v"]:
            raise SystemExit(f"{path}: unexpected header {header[:3]}")
        for line in stream:
            if not line.strip():
                continue
            fields = line.split(",")
            index = int(fields[0])
            if index in rows:
                raise SystemExit(f"{path}: duplicate vertex index {index}")
            rows[index] = (float(fields[1]), float(fields[2]))
    if sorted(rows) != list(range(len(rows))):
        raise SystemExit(f"{path}: vertex indices are not 0..n-1")
    coords = [rows[i] for i in range(len(rows))]
    box = (min(u for u, _ in coords), max(u for u, _ in coords),
           min(v for _, v in coords), max(v for _, v in coords))
    if max(abs(box[0] + LX), abs(box[1] - LX),
           abs(box[2] + LY), abs(box[3] - LY)) > 1e-9:
        raise SystemExit(
            f"{path}: material bounding box {box} does not match the "
            f"lx={LX}, ly={LY} plate of this study")
    return coords


def write_seed_csv(path: Path, seed: str, amplitude: float,
                   coords: list[tuple[float, float]]) -> None:
    """Full per-vertex displacement from rest; only dz is nonzero."""
    lines = ["vertex_index,dx,dy,dz"]
    for index, (u, v) in enumerate(coords):
        cu = math.cos(0.5 * math.pi * u / LX)
        cv = math.cos(0.5 * math.pi * v / LY)
        dz = {"flat": 0.0, "crown": amplitude * cu * cv,
              "cyl_u": amplitude * cu, "cyl_v": amplitude * cv}[seed]
        lines.append(f"{index},0,0,{dz!r}")
    path.write_text("\n".join(lines) + "\n")


def build_command(binary: Path, case: dict, case_dir: Path,
                  implementation: str, args: argparse.Namespace) -> list[str]:
    command = [
        str(binary), "-sim", "bilayer_growth", "-case", "custom",
        "-geometry", "rectangle", "-lx", str(LX), "-ly", str(LY),
        "-res", str(case.get("res", RES)), "-h_total", str(H_TOTAL),
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str((case_dir / "sequence.json").resolve()),
        "-enable_passE", "false", "-nsteps", "1", "-tol", str(case.get("tol", 1e-12)),
        "-max_iter", str(case.get("max_iter", 50000)), "-basename", case["name"], "-export_stl", "false",
    ]
    if implementation == "upstream":
        return command
    # -sequence_adaptive false is the upstream-equivalent algorithm: one full
    # load step per cycle. True subdivides the load path, which upstream cannot
    # do, so it is no longer a parity comparison.
    command += [
        "-sequence_warm_start", "true", "-metric_update", "multiplicative",
        "-sequence_minimize_every", str(case.get("minimize_every", 1)),
        "-sequence_adaptive", "true" if args.sequence_adaptive else "false",
        "-sequence_stability", "false", "-equilibrium_solver", "hlbfgs",
        "-equilibrium_grad_tol", repr(args.grad_tol),
        "-certify_final", "false", "-seed_escape", "false",
    ]
    if "hlbfgs_absolute_gradient_tol" in case:
        command += ["-hlbfgs_absolute_gradient_tol", str(case["hlbfgs_absolute_gradient_tol"])]
    for flag in ("sequence_initial_step", "sequence_max_step", "sequence_min_step",
                 "sequence_step_growth", "sequence_max_retries"):
        if flag in case:
            command += ["-" + flag, str(case[flag])]
    prescribed = case.get("prescribed")
    if prescribed:
        command += [
            "-prescribed_at_cycle", str(prescribed["at_cycle"]),
            "-prescribed_disp_csv", str((case_dir / "seed.csv").resolve()),
            "-prescribed_reference", "rest",
            "-prescribed_reequilibrate", "false",
            "-prescribed_tag", prescribed["tag"],
        ]
    return command


def solver_evidence(case_dir: Path, implementation: str) -> dict:
    """Only what the binaries actually report. Upstream has no acceptance test."""
    log = (case_dir / "run.log").read_text(errors="replace")
    convergence = case_dir / "sequence_convergence.csv"
    return {
        "hlbfgs_return_codes": [int(m) for m in RETCODE_RE.findall(log)],
        "hlbfgs_final_gradient_norms": [float(m) for m in GNORM_RE.findall(log)],
        "convergence_csv": convergence.name if convergence.is_file() else None,
        "residual_acceptance_enforced": implementation == "current",
        "evidence_note":
            "current gates every cycle on -equilibrium_grad_tol and logs "
            "sequence_convergence.csv; upstream b165fc52 has no acceptance test "
            "and no residual column, so exit code 0 and written VTP files do NOT "
            "certify an accepted equilibrium -- the per-solve HLBFGS return code "
            "and 'final eps' gradient norm scraped from run.log are the only "
            "residual evidence it produces",
    }


def final_vtp(case_dir: Path) -> tuple[str | None, list[str]]:
    cycles = [(int(m.group(1)), p.name) for p in case_dir.glob("*.vtp")
              if (m := CYCLE_VTP_RE.search(p.name))]
    imposed = sorted(p.name for p in case_dir.glob("*_imposed.vtp"))
    return (max(cycles)[1] if cycles else None), imposed


def run_case(case: dict, implementation: str, args: argparse.Namespace,
             coords: list[tuple[float, float]] | None) -> int:
    spec = IMPLEMENTATIONS[implementation]
    binary = spec["binary"]
    if not binary.is_file():
        raise SystemExit(f"{implementation}: missing binary {binary}")
    observed = git_value(spec["repo"], "rev-parse", "HEAD")
    if spec["pin"] and observed != spec["pin"]:
        raise SystemExit(f"{implementation}: {spec['repo']} is at {observed}, "
                         f"expected pinned {spec['pin']}")

    case_dir = RUN_ROOT / implementation / case["name"]
    if args.force and case_dir.exists():
        shutil.rmtree(case_dir)
    marker = case_dir / "result.json"
    if marker.is_file():
        raise SystemExit(
            f"{case_dir}: existing result preserved; use --force explicitly to rerun")
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "sequence.json").write_text(
        json.dumps(case["sequence"], indent=2) + "\n")
    if case.get("seed"):
        write_seed_csv(case_dir / "seed.csv", case["seed"],
                       args.seed_amplitude, coords)

    command = build_command(binary, case, case_dir, implementation, args)
    manifest = {
        "case": case["name"],
        "family": case["family"],
        "implementation": implementation,
        "seed": case.get("seed"),
        "note": case["note"],
        "plate": {"lx": LX, "ly": LY, "res": case.get("res", RES), "h_total": H_TOTAL},
        "growth_per_hit": case.get("growth_per_hit", GROWTH),
        "seed_amplitude_m": args.seed_amplitude if case.get("seed") else None,
        "material_baseline": (str(args.material_baseline)
                              if case.get("seed") else None),
        "material_baseline_sha256": (sha256(args.material_baseline)
                                     if case.get("seed") else None),
        "command": command,
        "command_text": " ".join(command),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "expected_revision": spec["pin"],
        "observed_revision": observed,
        "git_status": git_value(spec["repo"], "status", "--short"),
        "sequence_sha256": sha256(case_dir / "sequence.json"),
        "seed_csv_sha256": (sha256(case_dir / "seed.csv")
                            if case.get("seed") else None),
        "sequence_based": True,
        "omp_num_threads": str(args.threads),
        "timeout_s": args.timeout,
        "host": platform.node(),
        "platform": platform.platform(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"RUN {implementation}/{case['name']}", flush=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    started = time.monotonic()
    with (case_dir / "run.log").open("w") as log:
        try:
            completed = subprocess.run(command, cwd=case_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=False)
            return_code, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            return_code, timed_out = 124, True
    vtp, imposed = final_vtp(case_dir)
    if return_code == 0 and vtp is None:
        return_code = 1
    result = {
        "case": case["name"],
        "family": case["family"],
        "implementation": implementation,
        "seed": case.get("seed"),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "final_vtp": vtp,
        "imposed_vtp": imposed,
        "solver_evidence": solver_evidence(case_dir, implementation),
    }
    marker.write_text(json.dumps(result, indent=2) + "\n")
    status = "OK" if return_code == 0 else f"FAIL rc={return_code}"
    print(f"{status} {implementation}/{case['name']} "
          f"{result['wall_seconds']:.1f}s final_vtp={vtp}", flush=True)
    return return_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", action="append",
                        choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--case", action="append")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-case wall-clock limit in seconds")
    parser.add_argument("--threads", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--seed-amplitude", type=float, default=SEED_AMPLITUDE)
    parser.add_argument("--grad-tol", type=float, default=1e-11,
                        help="current-only -equilibrium_grad_tol")
    parser.add_argument("--material-baseline", type=Path,
                        default=MATERIAL_BASELINE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--sequence-adaptive", action="store_true",
                        help="current-only: subdivide the per-cycle load path; "
                             "off by default so the algorithm matches upstream")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="abort the batch at the first failing case instead "
                             "of running every case and reporting at the end")
    args = parser.parse_args()

    implementations = args.implementation or sorted(IMPLEMENTATIONS)
    selected = [c for c in cases()
                if (not args.family or c["family"] in args.family)
                and (not args.case or c["name"] in args.case)]
    plan = [(c, i) for c in selected for i in implementations
            if i in c.get("implementations", ("current", "upstream"))]
    if args.case:
        missing = set(args.case) - {c["name"] for c in selected}
        if missing:
            raise SystemExit(f"unknown case(s): {sorted(missing)}")
    unsupported = [(c["name"], i) for c in selected for i in implementations
                   if i not in c.get("implementations", ("current", "upstream"))]
    for name, impl in unsupported:
        print(f"UNSUPPORTED {impl}/{name}: the frozen family is current-only "
              "because its seeded cases need -prescribed_* flags that upstream "
              "b165fc52 does not have; its control stays in the same family so "
              "the seed comparison is like-for-like", flush=True)
    if args.list:
        for case, impl in plan:
            print(f"{impl:9s} {case['family']:10s} {case['name']}")
        return
    if not plan:
        raise SystemExit("no matching cases")
    if any(c.get("seed") for c, _ in plan):
        if not args.material_baseline.is_file():
            raise SystemExit(f"missing material baseline: {args.material_baseline}")
        coords = material_baseline(args.material_baseline)
        print(f"seeds use {len(coords)} vertices from "
              f"{args.material_baseline}", flush=True)
    else:
        coords = None

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []
    for case, impl in plan:
        if run_case(case, impl, args, coords) != 0:
            failures.append(f"{impl}/{case['name']}")
            if args.stop_on_failure:
                raise SystemExit(f"failed: {failures[-1]}")
    if failures:
        raise SystemExit(f"failed cases: {', '.join(failures)}")


if __name__ == "__main__":
    main()
