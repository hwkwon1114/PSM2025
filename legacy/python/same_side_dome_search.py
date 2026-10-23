"""Generate, run, and evaluate same-side orthogonal zigzag dome searches."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

CYCLES = (2, 4, 8, 12)
TOTAL_GROWTH_PER_DIRECTION = (0.0005, 0.001, 0.002, 0.003)
PROFILES = ("uniform", "center_peak")


def stage1_cases() -> list[dict[str, object]]:
    cases = []
    for task_id, (cycles, total_growth, profile) in enumerate(
        itertools.product(CYCLES, TOTAL_GROWTH_PER_DIRECTION, PROFILES)
    ):
        growth_tag = f"{int(round(1e6 * total_growth)):04d}u"
        cases.append(
            {
                "task_id": task_id,
                "cycles": cycles,
                "total_growth_per_direction": total_growth,
                "profile": profile,
                "case": f"c{cycles:02d}_g{growth_tag}_{profile}",
            }
        )
    return cases


def sequence_config(case: dict[str, object]) -> dict[str, object]:
    cycles = int(case["cycles"])
    per_cycle_growth = float(case["total_growth_per_direction"]) / (cycles / 2)
    profile_name = str(case["profile"])
    profile = {"mode": "uniform"}
    if profile_name == "center_peak":
        profile = {
            "mode": "center_peak",
            "top_end_ratio": 0.2,
            "top_profile_power": 2.0,
        }
    toolpaths = []
    for index in range(cycles):
        rotation = 0.0 if index % 2 == 0 else 90.0
        toolpaths.append(
            {
                "id": f"cycle{index + 1:02d}_top_{int(rotation)}deg",
                "repeat": 1,
                "operation": {
                    "type": "zigzag",
                    "lv_mm": 140.0,
                    "alpha_deg": 9.13,
                    "n_strips": 10,
                    "width_mm": 10.0,
                    "center_uv_mm": [0.0, 0.0],
                    "rotation_deg": rotation,
                    "gtop": per_cycle_growth,
                    "gbot": 0.0,
                    "ortho": 0.0,
                },
            }
        )
    return {
        "schema_version": 1,
        "units": {
            "length": "mm",
            "angle": "deg",
            "growth": "engineering_strain",
        },
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up", "profile": profile},
        "toolpaths": toolpaths,
    }


def prepare_stage1(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "stage1",
        "objective": "clear same-side dome",
        "resolution": 0.03,
        "tolerance": 1e-12,
        "cases": stage1_cases(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(manifest['cases'])} cases in {output_root}")


def last_match(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def run_task(
    repo_root: Path,
    output_root: Path,
    task_id: int,
    resolution: float,
    tolerance: float,
    timeout_seconds: int,
) -> None:
    cases = stage1_cases()
    if task_id < 0 or task_id >= len(cases):
        raise SystemExit(f"task_id must be in [0, {len(cases) - 1}]")
    case = cases[task_id]
    case_dir = output_root / str(case["case"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        previous_final = case_dir / str(previous.get("final_file", ""))
        if (
            previous.get("status") == 0
            and str(previous.get("verdict", "")).startswith("MINIMUM")
            and previous_final.is_file()
        ):
            print(f"SKIP {case['case']}: completed result already exists")
            return

    config_path = case_dir / "sequence.json"
    config_path.write_text(
        json.dumps(sequence_config(case), indent=2) + "\n", encoding="utf-8"
    )
    log_path = case_dir / "run.log"
    command = [
        str(repo_root / "bin" / "shell"),
        "-sim", "bilayer_growth",
        "-case", "custom",
        "-geometry", "rectangle",
        "-lx", "0.13",
        "-ly", "0.16",
        "-res", str(resolution),
        "-h_total", "0.0005",
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(config_path.resolve()),
        "-enable_passE", "false",
        "-nsteps", "1",
        "-tol", str(tolerance),
        "-minimizer", "hlbfgs",
        "-max_iter", "50000",
        "-certify_final", "true",
        "-basename", str(case["case"]),
        "-export_stl", "false",
    ]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(
                command,
                cwd=case_dir,
                env=os.environ,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True
    wall_seconds = time.monotonic() - started
    log = log_path.read_text(encoding="utf-8", errors="replace")
    energy = last_match(r"certify_final\] esc=\d+\s+E=([0-9.e+-]+)", log)
    gradient = last_match(r"\|g\|=([0-9.e+-]+)", log)
    lambda_min = last_match(r"lam_min=([+-][0-9.e-]+)", log)
    verdict = last_match(r"certify_final.*?-> ([^\n]+)", log).strip()
    final_candidates = sorted(
        case_dir.glob(
            f"bilayer_zigzag_sequence_cycle_{int(case['cycles']):03d}_*_final.vtp"
        )
    )
    final_file = final_candidates[-1].name if final_candidates else ""
    status = return_code
    if timed_out:
        verdict = "TIMEOUT"
    elif return_code == 0 and verdict.startswith("MINIMUM") and final_file:
        status = 0
    elif return_code == 0:
        status = 3
        verdict = verdict or "FAILED"
    result = {
        **case,
        "resolution": resolution,
        "tolerance": tolerance,
        "per_cycle_growth": float(case["total_growth_per_direction"])
        / (int(case["cycles"]) / 2),
        "status": status,
        "return_code": return_code,
        "wall_seconds": wall_seconds,
        "energy": float(energy) if energy else None,
        "gradient": float(gradient) if gradient else None,
        "lambda_min": float(lambda_min) if lambda_min else None,
        "verdict": verdict,
        "final_file": final_file,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE {case['case']} status={status} wall={wall_seconds:.1f}s "
        f"energy={energy or '-'} lambda_min={lambda_min or '-'} verdict={verdict or '-'}"
    )
    if status != 0:
        raise SystemExit(status)


def fit_global_quadratic(current: np.ndarray) -> dict[str, float]:
    x = current[:, 0] - np.mean(current[:, 0])
    y = current[:, 1] - np.mean(current[:, 1])
    z = current[:, 2]
    design = np.column_stack(
        (np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y)
    )
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ coefficients
    hessian = np.array(
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]]
    )
    principal, directions = np.linalg.eigh(hessian)
    direction_angles = np.mod(
        np.degrees(np.arctan2(directions[1], directions[0])), 180.0
    )
    variance = float(np.sum((z - np.mean(z)) ** 2))
    weaker = max(float(np.min(np.abs(principal))), 1e-15)
    return {
        "global_k1_m-1": float(principal[0]),
        "global_k2_m-1": float(principal[1]),
        "global_projected_k1_direction_deg": float(direction_angles[0]),
        "global_projected_k2_direction_deg": float(direction_angles[1]),
        "global_K_m-2": float(np.prod(principal)),
        "global_anisotropy": float(np.max(np.abs(principal)) / weaker),
        "fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "fit_r2": float(1.0 - np.sum(residual**2) / max(variance, 1e-30)),
    }


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate_case(case_dir: Path) -> dict[str, object] | None:
    result_path = case_dir / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != 0 or not str(result.get("verdict", "")).startswith("MINIMUM"):
        return None
    final_path = case_dir / str(result["final_file"])
    if not final_path.is_file():
        return None
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(final_path))
    reader.Update()
    data = reader.GetOutput()
    current = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = (
        vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    )
    cell_data = data.GetCellData()
    gaussian = vtk_to_numpy(cell_data.GetArray("gauss")).astype(float)
    mean = vtk_to_numpy(cell_data.GetArray("mean")).astype(float)
    area = 0.5 * np.linalg.norm(
        np.cross(
            current[triangles[:, 1]] - current[triangles[:, 0]],
            current[triangles[:, 2]] - current[triangles[:, 0]],
        ),
        axis=1,
    )
    total_area = float(np.sum(area))
    root = np.sqrt(np.maximum(mean**2 - gaussian, 0.0))
    local_k1 = mean + root
    local_k2 = mean - root
    fit = fit_global_quadratic(current)
    abs_principal = sorted((abs(fit["global_k1_m-1"]), abs(fit["global_k2_m-1"])))
    balance = abs_principal[0] / max(abs_principal[1], 1e-15)
    positive_area = float(np.sum(area[gaussian > 0.0]) / total_area)
    negative_area = float(np.sum(area[gaussian < 0.0]) / total_area)
    double_area = float(
        np.sum(area[(np.abs(local_k1) > 0.05) & (np.abs(local_k2) > 0.05)])
        / total_area
    )
    same_sign = fit["global_K_m-2"] > 0.0
    point_data = data.GetPointData()
    u3 = vtk_to_numpy(point_data.GetArray("U3_from_cycle0")).astype(float)
    material_u = vtk_to_numpy(point_data.GetArray("material_u")).astype(float)
    material_v = vtk_to_numpy(point_data.GetArray("material_v")).astype(float)
    center_index = int(np.argmin(material_u**2 + material_v**2))
    peak_u3 = float(np.max(u3))
    if abs(float(np.min(u3))) > abs(peak_u3):
        peak_u3 = float(np.min(u3))
    center_peak_ratio = float(u3[center_index] / peak_u3) if peak_u3 else 0.0
    z_span = float(np.ptp(u3))
    score = (
        0.25 * clamp(balance / 0.5)
        + 0.15 * clamp((positive_area - 0.30) / 0.35)
        + 0.20 * clamp((fit["fit_r2"] - 0.65) / 0.25)
        + 0.15 * clamp(abs_principal[0] / 0.5)
        + 0.10 * clamp(double_area / 0.9)
        + 0.15 * clamp(center_peak_ratio)
    )
    if not same_sign:
        score *= 0.25
    strict_local_pass = bool(
        same_sign
        and balance >= 0.4
        and abs_principal[0] >= 0.25
        and positive_area >= 0.65
        and fit["fit_r2"] >= 0.85
        and double_area >= 0.8
    )
    macro_dome_pass = bool(
        same_sign
        and balance >= 0.6
        and abs_principal[0] >= 0.25
        and fit["fit_r2"] >= 0.75
        and double_area >= 0.9
        and z_span >= 0.005
        and center_peak_ratio >= 0.8
    )
    return {
        **result,
        "faces": len(triangles),
        "z_span_m": z_span,
        "mean_abs_K_m-2": float(np.sum(area * np.abs(gaussian)) / total_area),
        "positive_K_area_fraction": positive_area,
        "negative_K_area_fraction": negative_area,
        "double_curvature_area_fraction": double_area,
        "curvature_balance": balance,
        "center_peak_ratio": center_peak_ratio,
        "dome_score": score,
        "pass": macro_dome_pass,
        "macro_dome_pass": macro_dome_pass,
        "strict_local_pass": strict_local_pass,
        **fit,
    }


def write_analysis(rows: list[dict[str, object]], output_root: Path) -> None:
    fields = tuple(rows[0])
    with (output_root / "dome_search_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    (output_root / "best_candidate.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )


def plot_top_candidates(rows: list[dict[str, object]], output_root: Path) -> None:
    selected = rows[: min(8, len(rows))]
    geometries = []
    global_abs_u3 = 0.0
    for row in selected:
        final_path = output_root / str(row["case"]) / str(row["final_file"])
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(final_path))
        reader.Update()
        data = reader.GetOutput()
        current = 1000.0 * vtk_to_numpy(data.GetPoints().GetData()).astype(float)
        current -= np.mean(current, axis=0)
        triangles = (
            vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
        )
        u3 = 1000.0 * vtk_to_numpy(
            data.GetPointData().GetArray("U3_from_cycle0")
        ).astype(float)
        global_abs_u3 = max(global_abs_u3, float(np.max(np.abs(u3))))
        geometries.append((current, triangles, u3))
    norm = colors.TwoSlopeNorm(vmin=-global_abs_u3, vcenter=0.0, vmax=global_abs_u3)
    figure = plt.figure(figsize=(16.0, 8.5))
    for index, (row, geometry) in enumerate(zip(selected, geometries), start=1):
        current, triangles, u3 = geometry
        axis = figure.add_subplot(2, 4, index, projection="3d")
        surface = axis.plot_trisurf(
            current[:, 0],
            current[:, 1],
            current[:, 2],
            triangles=triangles,
            cmap="coolwarm",
            norm=norm,
            linewidth=0.02,
            antialiased=True,
            shade=False,
        )
        surface.set_array(np.mean(u3[triangles], axis=1))
        extent = np.maximum(np.ptp(current, axis=0), 1e-9)
        axis.set_box_aspect(extent)
        axis.view_init(elev=25.0, azim=-55.0)
        axis.set_xticks(())
        axis.set_yticks(())
        axis.set_zticks(())
        axis.set_title(
            f"{row['case']}  score={row['dome_score']:.2f}\n"
            f"k=({row['global_k1_m-1']:+.2f},{row['global_k2_m-1']:+.2f}) "
            f"posK={100 * row['positive_K_area_fraction']:.0f}%"
        )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    color_axis = figure.add_axes((0.92, 0.18, 0.014, 0.62))
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_label("$U_3$ from cycle 0 (mm)")
    figure.suptitle("Highest-scoring same-side orthogonal dome candidates", fontsize=16)
    figure.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.88, wspace=0.02)
    figure.savefig(output_root / "dome_search_top_candidates.png", dpi=220)
    plt.close(figure)


def analyze(output_root: Path) -> None:
    rows = [
        row
        for case_dir in sorted(output_root.glob("c*_g*"))
        if case_dir.is_dir() and (row := evaluate_case(case_dir)) is not None
    ]
    if not rows:
        raise SystemExit(f"No certified completed cases found under {output_root}")
    rows.sort(key=lambda row: (not bool(row["pass"]), -float(row["dome_score"])))
    write_analysis(rows, output_root)
    plot_top_candidates(rows, output_root)
    passed = sum(bool(row["pass"]) for row in rows)
    best = rows[0]
    print(
        f"Analyzed {len(rows)}/{len(stage1_cases())} certified cases; "
        f"passes={passed}; best={best['case']} score={best['dome_score']:.3f} "
        f"k=({best['global_k1_m-1']:+.3f},{best['global_k2_m-1']:+.3f}) "
        f"positive_K_area={best['positive_K_area_fraction']:.3f} "
        f"R2={best['fit_r2']:.3f}"
    )


def evaluate_and_write(case_dir: Path) -> None:
    row = evaluate_case(case_dir)
    if row is None:
        raise SystemExit(f"No certified completed result found in {case_dir}")
    (case_dir / "evaluation.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-stage1")
    prepare_parser.add_argument("output_root", type=Path)

    run_parser = subparsers.add_parser("run-task")
    run_parser.add_argument("task_id", type=int)
    run_parser.add_argument("output_root", type=Path)
    run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--resolution", type=float, default=0.03)
    run_parser.add_argument("--tolerance", type=float, default=1e-12)
    run_parser.add_argument("--timeout", type=int, default=14000)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("output_root", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate-case")
    evaluate_parser.add_argument("case_dir", type=Path)


    args = parser.parse_args()
    if args.command == "prepare-stage1":
        prepare_stage1(args.output_root)
    elif args.command == "run-task":
        run_task(
            args.repo_root.resolve(),
            args.output_root.resolve(),
            args.task_id,
            args.resolution,
            args.tolerance,
            args.timeout,
        )
    elif args.command == "evaluate-case":
        evaluate_and_write(args.case_dir)
    else:
        analyze(args.output_root)


if __name__ == "__main__":
    main()
