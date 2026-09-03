"""Aggregate completed ablation runs, generate summary CSV and regime plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_vtp(path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    point_data = data.GetPointData()
    u3 = vtk_to_numpy(point_data.GetArray("U3_from_cycle0")).astype(float)
    mat_u = vtk_to_numpy(point_data.GetArray("material_u")).astype(float)
    mat_v = vtk_to_numpy(point_data.GetArray("material_v")).astype(float)
    return points, triangles, u3, mat_u, mat_v


def main():
    root = Path("run/ablation_md_quick")
    result_files = sorted(root.glob("*/result.json"))

    rows = []
    for rf in result_files:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
            if data.get("return_code") == 0 and data.get("final_file"):
                rows.append(data)
        except Exception as e:
            print(f"Error reading {rf}: {e}")

    rows = sorted(rows, key=lambda x: (float(x.get("lambda_ratio", 0)), int(x.get("cycles", 0)), str(x.get("orthotropy"))))

    print(f"Loaded {len(rows)} valid completed cases:")
    for r in rows:
        print(f"  {r['case']}: lam={r['lambda_ratio']}, N={r['cycles']}, ortho={r['orthotropy']} -> k1={r.get('k1', 0):+.2f}, k2={r.get('k2', 0):+.2f}, K={r.get('K', 0):+.3f}, z_span={r.get('z_span_m', 0)*1000:.1f}mm, balance={r.get('curvature_balance', 0):.3f}")

    if not rows:
        return

    # Write summary CSV
    summary_csv = root / "ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as s:
        writer = csv.DictWriter(s, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote summary CSV to {summary_csv}")

    # Plot regime map
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # 1. Primary curvature k1 vs cycles by lambda
    lam_set = sorted({float(r["lambda_ratio"]) for r in rows})
    colors_lam = {0.0: "navy", 1.0: "teal", 2.0: "crimson"}
    markers_lam = {0.0: "o", 1.0: "s", 2.0: "^"}

    for lam in lam_set:
        sub = [r for r in rows if float(r["lambda_ratio"]) == lam and r.get("orthotropy") == -0.5]
        if not sub:
            sub = [r for r in rows if float(r["lambda_ratio"]) == lam]
        sub = sorted(sub, key=lambda x: int(x["cycles"]))
        cycs = [int(x["cycles"]) for x in sub]
        k1s = [abs(float(x["k1"])) for x in sub]
        axes[0].plot(cycs, k1s, marker=markers_lam.get(lam, "o"), color=colors_lam.get(lam, "black"), linewidth=2, markersize=8, label=f"$\lambda = m/d = {lam:.1f}$")

    axes[0].set_title("Primary Curvature $|\kappa_1|$ vs. Pass Count ($N$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Pass Count $N_{\mathrm{cycles}}$")
    axes[0].set_ylabel("Primary Curvature $|\kappa_1|$ ($\mathrm{m}^{-1}$)")
    axes[0].set_xticks([2, 4, 8])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Curvature balance vs cycles
    for lam in lam_set:
        sub = [r for r in rows if float(r["lambda_ratio"]) == lam and r.get("orthotropy") == -0.5]
        if not sub:
            sub = [r for r in rows if float(r["lambda_ratio"]) == lam]
        sub = sorted(sub, key=lambda x: int(x["cycles"]))
        cycs = [int(x["cycles"]) for x in sub]
        bals = [float(x["curvature_balance"]) for x in sub]
        axes[1].plot(cycs, bals, marker=markers_lam.get(lam, "o"), color=colors_lam.get(lam, "black"), linewidth=2, markersize=8, label=f"$\lambda = m/d = {lam:.1f}$")

    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Dome Threshold (0.5)")
    axes[1].set_title("Curvature Balance ($\min|\kappa|/\max|\kappa|$) vs. $N$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Pass Count $N_{\mathrm{cycles}}$")
    axes[1].set_ylabel("Curvature Balance Ratio")
    axes[1].set_xticks([2, 4, 8])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Vertical z-span vs cycles
    for lam in lam_set:
        sub = [r for r in rows if float(r["lambda_ratio"]) == lam and r.get("orthotropy") == -0.5]
        if not sub:
            sub = [r for r in rows if float(r["lambda_ratio"]) == lam]
        sub = sorted(sub, key=lambda x: int(x["cycles"]))
        cycs = [int(x["cycles"]) for x in sub]
        z_spans = [float(x["z_span_m"]) * 1000.0 for x in sub]
        axes[2].plot(cycs, z_spans, marker=markers_lam.get(lam, "o"), color=colors_lam.get(lam, "black"), linewidth=2, markersize=8, label=f"$\lambda = m/d = {lam:.1f}$")

    axes[2].set_title("Total Vertical Deflection Span vs. $N$", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Pass Count $N_{\mathrm{cycles}}$")
    axes[2].set_ylabel("Deflection Span $\Delta z$ (mm)")
    axes[2].set_xticks([2, 4, 8])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Controlled (m, d) Eigenstrain Ablation: Scaling Trends across Multi-Pass Cycles", fontsize=13, fontweight="bold")
    out_png = root / "ablation_trends_analysis.png"
    out_svg = root / "ablation_trends_analysis.svg"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved trend analysis to {out_png} and {out_svg}")


if __name__ == "__main__":
    main()
