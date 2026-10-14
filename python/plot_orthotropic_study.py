#!/usr/bin/env python3
"""Plot results from the negative orthotropy resolution and tolerance study."""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("run/orthotropic_study_matrix/study_results_summary.csv")
OUT_DIR = Path("run/orthotropic_study_matrix/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} not found.")
    exit(1)

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

# Group by toolpath
toolpaths = sorted(list(set(r["toolpath"] for r in rows)))

plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

colors = {
    "putong_1step_ortho_m05": "#1f77b4", 
    "putong_1step": "#1f77b4",
    "06_orthogonal_2cycle_ortho": "#ff7f0e",
    "ortho_2cycle": "#ff7f0e"
}
markers = {0.035: "o", 0.030: "s", 0.025: "^", 0.020: "D", 0.015: "v"}
labels = {
    "putong_1step_ortho_m05": "Single-Pass (10-strip)",
    "putong_1step": "Single-Pass (10-strip)",
    "06_orthogonal_2cycle_ortho": "2-Cycle Orthogonal (20-strip)",
    "ortho_2cycle": "2-Cycle Orthogonal (20-strip)",
}

# Subplot 1: Deflection vs Tolerance (log x)
ax1 = axes[0, 0]
for tp in toolpaths:
    tp_rows = [r for r in rows if r["toolpath"] == tp]
    resolutions = sorted(list(set(float(r["res"]) for r in tp_rows)), reverse=True)
    for res in resolutions:
        pts = sorted([r for r in tp_rows if float(r["res"]) == res], key=lambda x: float(x["tol"]))
        tols = [float(p["tol"]) for p in pts]
        u3_mm = [float(p["u3_span"]) * 1000.0 for p in pts]
        ax1.plot(
            tols, u3_mm,
            marker=markers.get(res, "o"),
            color=colors.get(tp, "k"),
            linestyle="-" if tp == "06_orthogonal_2cycle_ortho" else "--",
            label=f"{labels.get(tp, tp)} (res={res:.3f}m)",
        )
ax1.set_xscale("log")
ax1.set_xlabel("Solver Gradient Tolerance (|g|)")
ax1.set_ylabel("Out-of-Plane Deflection Span (mm)")
ax1.set_title("(a) Deflection vs Solver Tolerance (Buckling Transition)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", fontsize=8, frameon=True)

# Subplot 2: Energy vs Tolerance
ax2 = axes[0, 1]
for tp in toolpaths:
    tp_rows = [r for r in rows if r["toolpath"] == tp]
    resolutions = sorted(list(set(float(r["res"]) for r in tp_rows)), reverse=True)
    for res in resolutions:
        pts = sorted([r for r in tp_rows if float(r["res"]) == res], key=lambda x: float(x["tol"]))
        tols = [float(p["tol"]) for p in pts]
        energies = [float(p["energy"]) * 1e11 for p in pts]
        ax2.plot(
            tols, energies,
            marker=markers.get(res, "o"),
            color=colors.get(tp, "k"),
            linestyle="-" if tp == "06_orthogonal_2cycle_ortho" else "--",
        )
ax2.set_xscale("log")
ax2.set_xlabel("Solver Gradient Tolerance (|g|)")
ax2.set_ylabel(r"Total Elastic Energy ($\times 10^{-11}$ J)")
ax2.set_title("(b) Energy Minimization vs Solver Tolerance")
ax2.grid(True, linestyle=":", alpha=0.6)

# Subplot 3: Deflection vs Mesh Resolution (at tol=1e-12)
ax3 = axes[1, 0]
for tp in toolpaths:
    tp_rows = [r for r in rows if r["toolpath"] == tp and float(r["tol"]) == 1e-12]
    pts = sorted(tp_rows, key=lambda x: float(x["res"]))
    res_vals = [float(p["res"]) for p in pts]
    u3_mm = [float(p["u3_span"]) * 1000.0 for p in pts]
    ax3.plot(
        res_vals, u3_mm,
        marker="o",
        color=colors.get(tp, "k"),
        label=labels.get(tp, tp),
    )
ax3.set_xlabel("Mesh Resolution parameter (m) [Coarser ->]")
ax3.set_ylabel("Out-of-Plane Deflection Span (mm)")
ax3.set_title("(c) Mesh Convergence at tol = 1e-12")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="best", frameon=True)

# Subplot 4: Wall-Clock Time vs Mesh Resolution (at tol=1e-12)
ax4 = axes[1, 1]
for tp in toolpaths:
    tp_rows = [r for r in rows if r["toolpath"] == tp and float(r["tol"]) == 1e-12]
    pts = sorted(tp_rows, key=lambda x: float(x["res"]))
    res_vals = [float(p["res"]) for p in pts]
    times = [float(p["wall_time_s"]) for p in pts]
    ax4.plot(
        res_vals, times,
        marker="s",
        color=colors.get(tp, "k"),
        label=labels.get(tp, tp),
    )
ax4.set_xlabel("Mesh Resolution parameter (m) [Coarser ->]")
ax4.set_ylabel("Simulation Wall-Clock Time (s)")
ax4.set_title("(d) Computational Cost vs Mesh Resolution (4 Threads)")
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.legend(loc="best", frameon=True)

out_fig = OUT_DIR / "orthotropic_resolution_tolerance_study.png"
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"Saved figure to {out_fig}")
