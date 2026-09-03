"""Controlled (m, d) eigenstrain ablation study for multi-pass English wheeling.

Decouples membrane dilation m = (gtop + gbot)/2 from bending gradient d = (gtop - gbot)/2.
Uses exact multiplicative log-stretch compounding so total deposited metric is identical
across 2, 4, 8, and 12 passes to machine precision.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# -----------------------------------------------------------------------------
# Parameter Space
# -----------------------------------------------------------------------------
D_TOTAL = (0.0005, 0.001, 0.002)          # Bending differential d_tot
LAMBDA_RATIO = (0.0, 0.5, 1.0, 1.5, 2.0)   # lambda = m_tot / d_tot
CYCLES = (2, 4, 8, 12)                     # Pass divisions (alternating 0° and 90°)
ORTHOTROPY = (-0.5, 0.0, 0.5)              # Footprint orthotropy


def compute_step_growth(g_tot: float, ortho_tot: float, n_passes: int) -> tuple[float, float]:
    """Compute exact per-pass (g_step, ortho_step) preserving total stretch via log-stretches."""
    if n_passes <= 1:
        return g_tot, ortho_tot
    
    lam1_tot = 1.0 + g_tot * (1.0 + ortho_tot)
    lam2_tot = 1.0 + g_tot * (1.0 - ortho_tot)
    
    if lam1_tot <= 0.0 or lam2_tot <= 0.0:
        raise ValueError(f"Nonpositive total stretch: lam1={lam1_tot}, lam2={lam2_tot}")
    
    lam1_step = np.exp(np.log(lam1_tot) / n_passes)
    lam2_step = np.exp(np.log(lam2_tot) / n_passes)
    
    g1_step = lam1_step - 1.0
    g2_step = lam2_step - 1.0
    
    g_step = (g1_step + g2_step) / 2.0
    ortho_step = (g1_step - g2_step) / (2.0 * g_step) if abs(g_step) > 1e-15 else 0.0
    
    return float(g_step), float(ortho_step)


def ablation_cases(subset: str = "core") -> list[dict[str, object]]:
    cases = []
    task_id = 0

    if subset == "core":
        # Focused core sweep: d=0.001, spanning all lambda and cycles, plus ortho variation
        d_vals = (0.0005, 0.001, 0.002)
        lam_vals = (0.0, 0.5, 1.0, 1.5, 2.0)
        cycle_vals = (2, 4, 8, 12)
        ortho_vals = (-0.5, 0.0)
    elif subset == "quick":
        # Fast diagnostic subset
        d_vals = (0.001,)
        lam_vals = (0.0, 1.0, 2.0)
        cycle_vals = (2, 4, 8)
        ortho_vals = (-0.5, 0.0)
    else:
        # Full comprehensive grid
        d_vals = D_TOTAL
        lam_vals = LAMBDA_RATIO
        cycle_vals = CYCLES
        ortho_vals = ORTHOTROPY

    for d_tot, lam, n_cycles, ortho in itertools.product(d_vals, lam_vals, cycle_vals, ortho_vals):
        m_tot = lam * d_tot
        gtop_tot = m_tot + d_tot
        gbot_tot = m_tot - d_tot
        
        d_tag = f"{int(round(1e6 * d_tot)):04d}u"
        lam_tag = f"L{int(round(10 * lam)):02d}"
        ortho_tag = "m05" if ortho == -0.5 else ("p05" if ortho == 0.5 else "iso")
        case_name = f"c{n_cycles:02d}_d{d_tag}_{lam_tag}_{ortho_tag}"

        cases.append({
            "task_id": task_id,
            "case": case_name,
            "d_tot": d_tot,
            "m_tot": m_tot,
            "lambda_ratio": lam,
            "gtop_tot": gtop_tot,
            "gbot_tot": gbot_tot,
            "cycles": n_cycles,
            "orthotropy": ortho,
        })
        task_id += 1

    return cases


def build_sequence_json(case: dict[str, object], reverse: bool = False) -> dict[str, object]:
    cycles = int(case["cycles"])
    n_passes_per_dir = cycles // 2
    
    gtop_step, ortho_top_step = compute_step_growth(
        float(case["gtop_tot"]), float(case["orthotropy"]), n_passes_per_dir
    )
    gbot_step, ortho_bot_step = compute_step_growth(
        float(case["gbot_tot"]), float(case["orthotropy"]), n_passes_per_dir
    )

    toolpaths = []
    for idx in range(cycles):
        if not reverse:
            rot = 0.0 if idx % 2 == 0 else 90.0
        else:
            rot = 90.0 if idx % 2 == 0 else 0.0

        toolpaths.append({
            "id": f"cycle{idx + 1:02d}_rot_{int(rot)}deg",
            "repeat": 1,
            "operation": {
                "type": "zigzag",
                "lv_mm": 200.0,
                "alpha_deg": 3.18,
                "n_strips": 20,
                "width_mm": 10.0,
                "center_uv_mm": [0.0, 0.0],
                "rotation_deg": rot,
                "gtop": gtop_step,
                "gbot": gbot_step,
                "ortho": ortho_top_step,
            }
        })

    return {
        "schema_version": 1,
        "units": {
            "length": "mm",
            "angle": "deg",
            "growth": "engineering_strain"
        },
        "hardening": {"model": "none"},
        "defaults": {
            "start_mode": "left_bottom_up",
            "profile": {
                "mode": "uniform"
            }
        },
        "toolpaths": toolpaths
    }


def fit_quadratic(current: np.ndarray) -> dict[str, float]:
    x = current[:, 0] - np.mean(current[:, 0])
    y = current[:, 1] - np.mean(current[:, 1])
    z = current[:, 2]
    design = np.column_stack((np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y))
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ coeffs
    hessian = np.array([[coeffs[3], coeffs[4]], [coeffs[4], coeffs[5]]])
    principal, directions = np.linalg.eigh(hessian)
    direction_angles = np.mod(np.degrees(np.arctan2(directions[1], directions[0])), 180.0)
    variance = float(np.sum((z - np.mean(z)) ** 2))
    weaker = max(float(np.min(np.abs(principal))), 1e-15)
    return {
        "k1": float(principal[0]),
        "k2": float(principal[1]),
        "k1_dir_deg": float(direction_angles[0]),
        "k2_dir_deg": float(direction_angles[1]),
        "K": float(np.prod(principal)),
        "anisotropy": float(np.max(np.abs(principal)) / weaker),
        "fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "fit_r2": float(1.0 - np.sum(residual**2) / max(variance, 1e-30)),
        "z_span_m": float(np.ptp(z)),
    }


def evaluate_vtp(vtp_path: Path) -> dict[str, object]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    
    cell_data = data.GetCellData()
    gaussian = vtk_to_numpy(cell_data.GetArray("gauss")).astype(float)
    mean = vtk_to_numpy(cell_data.GetArray("mean")).astype(float)
    
    area = 0.5 * np.linalg.norm(
        np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        ),
        axis=1,
    )
    total_area = float(np.sum(area))
    pos_area = float(np.sum(area[gaussian > 0.0]) / total_area)
    neg_area = float(np.sum(area[gaussian < 0.0]) / total_area)
    
    fit = fit_quadratic(points)
    abs_k = sorted((abs(fit["k1"]), abs(fit["k2"])))
    balance = abs_k[0] / max(abs_k[1], 1e-15)
    
    is_dome = (
        fit["K"] > 0.0
        and balance >= 0.5
        and abs_k[0] >= 0.1
        and pos_area >= 0.35
        and fit["fit_r2"] >= 0.70
    )
    
    return {
        **fit,
        "curvature_balance": balance,
        "positive_K_area": pos_area,
        "negative_K_area": neg_area,
        "is_dome": bool(is_dome),
    }


def run_case(
    repo_root: Path,
    case_dir: Path,
    case: dict[str, object],
    reverse: bool = False,
    resolution: float = 0.03,
    tolerance: float = 1e-12,
    timeout_sec: int = 3600,
    max_iter: int = 50000,
) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = case_dir / "sequence.json"
    cfg_path.write_text(json.dumps(build_sequence_json(case, reverse=reverse), indent=2) + "\n", encoding="utf-8")
    
    log_path = case_dir / "run.log"
    cmd = [
        str(repo_root / "bin" / "shell"),
        "-sim", "bilayer_growth",
        "-case", "custom",
        "-geometry", "rectangle",
        "-lx", "0.13",
        "-ly", "0.16",
        "-res", str(resolution),
        "-h_total", "0.0005",
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(cfg_path.resolve()),
        "-enable_passE", "false",
        "-nsteps", "1",
        "-tol", str(tolerance),
        "-minimizer", "hlbfgs",
        "-max_iter", str(max_iter),
        "-certify_final", "true",
        "-basename", str(case["case"]),
        "-export_stl", "false",
    ]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(
            cmd,
            cwd=case_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ret = proc.wait(timeout=timeout_sec)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            ret = 124
            timed_out = True
    wall = time.monotonic() - started
    
    log = log_path.read_text(encoding="utf-8", errors="replace")
    
    matches_e = re.findall(r"certify_final\] esc=\d+\s+E=([0-9.e+-]+)", log)
    energy = float(matches_e[-1]) if matches_e else None
    
    matches_lam = re.findall(r"lam_min=([+-][0-9.e-]+)", log)
    lam_min = float(matches_lam[-1]) if matches_lam else None
    
    matches_v = re.findall(r"certify_final.*?-> ([^\n]+)", log)
    verdict = matches_v[-1].strip() if matches_v else ("TIMEOUT" if timed_out else "FAILED")
    
    final_file = None
    metrics = {}
    if ret == 0 and not timed_out:
        final_candidates = sorted(case_dir.glob(f"*cycle_{int(case['cycles']):03d}*_final.vtp"))
        if final_candidates:
            final_file = final_candidates[-1]
            metrics = evaluate_vtp(final_file)
        else:
            verdict = "MISSING_FINAL_FILE"
    
    res = {
        **case,
        "reverse": reverse,
        "return_code": ret,
        "wall_seconds": wall,
        "energy": energy,
        "lambda_min": lam_min,
        "verdict": verdict,
        "final_file": final_file.name if final_file else "",
        **metrics,
    }
    
    (case_dir / "result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def plot_ablation_results(rows: list[dict[str, object]], out_dir: Path) -> None:
    valid = [r for r in rows if r.get("return_code") == 0 and "K" in r]
    if not valid:
        print("No valid completed cases to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # 1. Curvature Balance vs Lambda Ratio by Pass Count
    cycles_set = sorted({int(r["cycles"]) for r in valid})
    markers = {2: "o", 4: "s", 8: "^", 12: "D"}
    colors_c = {2: "navy", 4: "teal", 8: "darkorange", 12: "crimson"}

    for n in cycles_set:
        sub = [r for r in valid if int(r["cycles"]) == n and r.get("orthotropy") == -0.5]
        if not sub:
            sub = [r for r in valid if int(r["cycles"]) == n]
        sub = sorted(sub, key=lambda x: float(x["lambda_ratio"]))
        lams = [float(x["lambda_ratio"]) for x in sub]
        bals = [float(x["curvature_balance"]) for x in sub]
        axes[0].plot(
            lams,
            bals,
            marker=markers.get(n, "o"),
            color=colors_c.get(n, "black"),
            linewidth=2,
            markersize=7,
            label=f"N = {n} passes",
        )

    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Dome Threshold (0.5)")
    axes[0].set_title("Curvature Balance vs. Membrane Ratio ($\lambda = m/d$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("$\lambda = m/d$  (0: Pure Bending, 1: Top-only, >1: Membrane-Dominated)")
    axes[0].set_ylabel("Curvature Balance  $\min(|\kappa_1|,|\kappa_2|) / \max(|\kappa_1|,|\kappa_2|)$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Gaussian Curvature K vs Lambda Ratio
    for n in cycles_set:
        sub = [r for r in valid if int(r["cycles"]) == n and r.get("orthotropy") == -0.5]
        if not sub:
            sub = [r for r in valid if int(r["cycles"]) == n]
        sub = sorted(sub, key=lambda x: float(x["lambda_ratio"]))
        lams = [float(x["lambda_ratio"]) for x in sub]
        ks = [float(x["K"]) for x in sub]
        axes[1].plot(
            lams,
            ks,
            marker=markers.get(n, "o"),
            color=colors_c.get(n, "black"),
            linewidth=2,
            markersize=7,
            label=f"N = {n} passes",
        )

    axes[1].axhline(0.0, color="gray", linestyle="--", alpha=0.7, label="K = 0 (1D Cylinder)")
    axes[1].set_title("Gaussian Curvature $K$ vs. Membrane Ratio ($\lambda = m/d$)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("$\lambda = m/d$  (0: Pure Bending, 1: Top-only, >1: Membrane-Dominated)")
    axes[1].set_ylabel("Gaussian Curvature $K$ ($\mathrm{m}^{-2}$)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Decoupled (m, d) Eigenstrain Ablation: Transition from Cylinder to Dome", fontsize=13, fontweight="bold")
    out_png = out_dir / "ablation_regime_map.png"
    fig.savefig(out_png, dpi=200)
    out_svg = out_dir / "ablation_regime_map.svg"
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved regime maps to {out_png} and {out_svg}")

def _worker_entry(item: tuple[Path, Path, dict[str, object], float, float, int, int]) -> dict[str, object]:
    repo_root, c_dir, case, res, tol, timeout_sec, max_iter = item
    return run_case(repo_root, c_dir, case, resolution=res, tolerance=tol, timeout_sec=timeout_sec, max_iter=max_iter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subset", choices=("quick", "core", "full"), default="quick")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--res", type=float, default=0.03)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-iter", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = ablation_cases(subset=args.subset)
    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"Starting eigenstrain (m, d) ablation sweep: {len(cases)} cases in {out_dir} with {args.workers} workers (timeout={args.timeout}s)")
    
    import concurrent.futures
    
    items = [
        (args.repo_root.resolve(), out_dir / str(c["case"]), c, args.res, args.tol, args.timeout, args.max_iter)
        for c in cases
    ]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_case = {executor.submit(_worker_entry, item): item[2] for item in items}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_case), start=1):
            case = future_to_case[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"[{idx}/{len(cases)}] COMPLETED {row['case']} wall={row['wall_seconds']:.1f}s k1={row.get('k1', 0):+.2f} k2={row.get('k2', 0):+.2f} K={row.get('K', 0):+.3f} dome={row.get('is_dome', False)}")
            except Exception as e:
                print(f"[{idx}/{len(cases)}] FAILED {case['case']}: {e}")
    if not rows:
        print("No cases completed.")
        return
    rows = sorted(rows, key=lambda x: int(x.get("task_id", 0)))
    summary_path = out_dir / "ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as s:
        writer = csv.DictWriter(s, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary CSV to {summary_path}")
    plot_ablation_results(rows, out_dir)

if __name__ == "__main__":
    main()
