#!/usr/bin/env python3
"""
Fit a Gaussian Process to all 209 Bayesian Optimization evaluations from Run 6
(combining the 160-point campaign and the 20-point recovery campaign).

Generates:
1. 1D sensitivity slices around the all-time best point (Attempt 209) with 95% confidence bands (±2σ).
2. 2D response contour surfaces for parameter interactions with observed sample points overlaid.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "calibration" / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN6_LOG_1 = ROOT / "run" / "run6_expanded_bo" / "evaluations.jsonl"
RUN6_LOG_2 = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "evaluations.jsonl"
BEST_SUMMARY = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "bayes_summary.json"


def load_all_evaluations():
    records = []
    seen_attempts = set()

    for path in [RUN6_LOG_1, RUN6_LOG_2]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                att_id = d.get("attempt_id")
                if att_id in seen_attempts:
                    continue
                seen_attempts.add(att_id)
                records.append(d)

    # Sort by attempt_id
    records.sort(key=lambda r: r.get("attempt_id", 0))

    X_list = []
    y_list = []
    valid_records = []

    for r in records:
        if not r.get("valid", False):
            continue
        p = r.get("parameters", {})
        # Extract [gtop, gbot, ortho_top, ortho_bottom]
        x = [
            float(p["gtop"]),
            float(p["gbot"]),
            float(p["ortho_top"]),
            float(p["ortho_bottom"]),
        ]
        obj = float(r.get("objective_mm", 999.0))
        if obj < 50.0:  # exclude numerical blowup / failure penalties
            X_list.append(x)
            y_list.append(obj)
            valid_records.append(r)

    return np.array(X_list), np.array(y_list), records, valid_records


def main():
    X, y, all_records, valid_records = load_all_evaluations()
    print(f"Total evaluations parsed: {len(all_records)}")
    print(f"Valid shell solves with finite objective: {len(valid_records)}")

    # Load best parameters from Attempt 209
    best_info = json.loads(BEST_SUMMARY.read_text())["best"]
    best_params = best_info["parameters"]
    best_x = np.array([
        best_params["gtop"],
        best_params["gbot"],
        best_params["ortho_top"],
        best_params["ortho_bottom"],
    ])
    best_y = best_info["objective_mm"]
    print(f"All-time best (Attempt {best_info['attempt_id']}): {best_y:.6f} mm")
    print(f"Best parameters: {best_params}")

    # Parameter bounds
    param_names = [r"$g_{\mathrm{top}}$", r"$g_{\mathrm{bot}}$", r"$\mathrm{ortho}_{\mathrm{top}}$", r"$\mathrm{ortho}_{\mathrm{bottom}}$"]
    param_keys = ["gtop", "gbot", "ortho_top", "ortho_bottom"]
    bounds = [
        (5e-5, 0.01),      # gtop
        (1e-5, 0.001),     # gbot
        (-1.0, 1.0),       # ortho_top
        (-1.0, 1.0),       # ortho_bottom
    ]

    # Normalize X to [0, 1]^4 for numerical stability in GP fitting
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    X_norm = (X - lb) / (ub - lb)
    best_x_norm = (best_x - lb) / (ub - lb)

    # Fit Gaussian Process with Matern 5/2 anisotropic kernel + noise
    kernel = ConstantKernel(constant_value=np.var(y), constant_value_bounds=(1e-4, 1e2)) * \
             Matern(length_scale=np.ones(4) * 0.3, length_scale_bounds=(0.02, 5.0), nu=2.5) + \
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

    print("\nFitting Gaussian Process to all 188 valid evaluations...")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X_norm, y)
    print(f"Optimized GP Kernel:\n  {gp.kernel_}")

    # -------------------------------------------------------------
    # Figure: 2x2 1D Slices + 2D Contour Map
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # 1D Slices across all 4 parameters passing through Attempt 209
    slice_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for i, ax in enumerate(slice_axes):
        p_name = param_names[i]
        p_lb, p_ub = bounds[i]
        n_pts = 150

        # Create 1D sweep grid
        sweep = np.linspace(p_lb, p_ub, n_pts)
        X_sweep = np.tile(best_x, (n_pts, 1))
        X_sweep[:, i] = sweep

        # Normalize for GP prediction
        X_sweep_norm = (X_sweep - lb) / (ub - lb)
        y_mean, y_std = gp.predict(X_sweep_norm, return_std=True)

        # Plot 95% confidence interval (±2σ)
        ax.fill_between(
            sweep, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
            color="#3498db", alpha=0.25, label=r"GP 95% CI ($\pm 1.96\sigma$)"
        )
        # Plot mean prediction
        ax.plot(sweep, y_mean, color="#2980b9", lw=2.0, label="GP Mean Prediction")

        # Scatter nearby observed points (within 0.15 normalized distance in other coordinates)
        other_dims = [j for j in range(4) if j != i]
        dists = np.linalg.norm(X_norm[:, other_dims] - best_x_norm[other_dims], axis=1)
        nearby_mask = dists < 0.25

        ax.scatter(
            X[nearby_mask, i], y[nearby_mask],
            color="#7f8c8d", alpha=0.6, s=25, edgecolors="none", label="Nearby BO Samples"
        )

        # Mark Attempt 209 (Best)
        ax.scatter(
            best_x[i], best_y,
            color="#e74c3c", s=110, zorder=5, marker="*", edgecolor="black", lw=1.2,
            label=f"Attempt 209 ({best_y:.4f} mm)"
        )

        ax.axvline(best_x[i], color="#e74c3c", linestyle=":", lw=1.2, alpha=0.8)
        ax.set_xlabel(f"{p_name}", fontsize=12)
        ax.set_ylabel("Symmetric RMS Error (mm)", fontsize=11)
        ax.set_title(f"Sensitivity: {p_name}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Focus 1D slices on the physically informative region where samples cluster
        if i == 0:
            ax.set_xlim(5e-5, 0.0008)
        elif i == 1:
            ax.set_xlim(1e-5, 0.00015)
        else:
            ax.set_xlim(p_lb, p_ub)
        ax.set_ylim(1.0, 3.0)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # 2D Contour Surfaces for key parameter pairs (Focused on Active Optimum)
    # -------------------------------------------------------------
    # Surface 1: (gtop, ortho_top)
    ax_cont1 = fig.add_subplot(gs[0, 2])
    n_grid = 80
    gtop_grid = np.linspace(0.0001, 0.0008, n_grid)
    ortho_top_grid = np.linspace(-1.0, 0.5, n_grid)
    G1, O1 = np.meshgrid(gtop_grid, ortho_top_grid)

    X_grid1 = np.tile(best_x, (n_grid * n_grid, 1))
    X_grid1[:, 0] = G1.ravel()
    X_grid1[:, 2] = O1.ravel()
    X_grid1_norm = (X_grid1 - lb) / (ub - lb)
    Z1 = gp.predict(X_grid1_norm).reshape(n_grid, n_grid)

    cs1 = ax_cont1.contourf(G1 * 1e4, O1, Z1, levels=np.linspace(1.15, 2.5, 28), cmap="viridis_r", extend="both")
    cbar1 = fig.colorbar(cs1, ax=ax_cont1, fraction=0.046, pad=0.04)
    cbar1.set_label("RMS Error (mm)", fontsize=10)

    # Filter scatter points to this window
    mask1 = (X[:, 0] <= 0.0008) & (X[:, 2] <= 0.5)
    ax_cont1.scatter(X[mask1, 0] * 1e4, X[mask1, 2], color="white", edgecolors="#2c3e50", s=25, alpha=0.75, label=f"BO Samples ({np.sum(mask1)})")
    ax_cont1.scatter(best_x[0] * 1e4, best_x[2], color="#e74c3c", s=150, marker="*", edgecolor="black", zorder=6, label="Attempt 209")
    ax_cont1.set_xlabel(r"$g_{\mathrm{top}} \times 10^4$", fontsize=11)
    ax_cont1.set_ylabel(r"$\mathrm{ortho}_{\mathrm{top}}$", fontsize=11)
    ax_cont1.set_title(r"Response Basin: $g_{\mathrm{top}}$ vs $\mathrm{ortho}_{\mathrm{top}}$", fontsize=11, fontweight="bold")
    ax_cont1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Surface 2: (gtop, gbot)
    ax_cont2 = fig.add_subplot(gs[1, 2])
    gbot_grid = np.linspace(1e-5, 0.00012, n_grid)
    G_top2, G_bot2 = np.meshgrid(gtop_grid, gbot_grid)

    X_grid2 = np.tile(best_x, (n_grid * n_grid, 1))
    X_grid2[:, 0] = G_top2.ravel()
    X_grid2[:, 1] = G_bot2.ravel()
    X_grid2_norm = (X_grid2 - lb) / (ub - lb)
    Z2 = gp.predict(X_grid2_norm).reshape(n_grid, n_grid)

    cs2 = ax_cont2.contourf(G_top2 * 1e4, G_bot2 * 1e5, Z2, levels=np.linspace(1.15, 2.5, 28), cmap="viridis_r", extend="both")
    cbar2 = fig.colorbar(cs2, ax=ax_cont2, fraction=0.046, pad=0.04)
    cbar2.set_label("RMS Error (mm)", fontsize=10)

    mask2 = (X[:, 0] <= 0.0008) & (X[:, 1] <= 0.00012)
    ax_cont2.scatter(X[mask2, 0] * 1e4, X[mask2, 1] * 1e5, color="white", edgecolors="#2c3e50", s=25, alpha=0.75, label=f"BO Samples ({np.sum(mask2)})")
    ax_cont2.scatter(best_x[0] * 1e4, best_x[1] * 1e5, color="#e74c3c", s=150, marker="*", edgecolor="black", zorder=6, label="Attempt 209")
    ax_cont2.set_xlabel(r"$g_{\mathrm{top}} \times 10^4$", fontsize=11)
    ax_cont2.set_ylabel(r"$g_{\mathrm{bot}} \times 10^5$", fontsize=11)
    ax_cont2.set_title(r"Response Basin: $g_{\mathrm{top}}$ vs $g_{\mathrm{bot}}$", fontsize=11, fontweight="bold")
    ax_cont2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        f"Gaussian Process Surrogate across all 209 BO Evaluations (Run 6 Calibration)\n"
        f"Global Minimum: Attempt 209 (RMS = {best_y:.4f} mm, g_top = {best_params['gtop']:.5f}, ortho_top = {best_params['ortho_top']:.4f})",
        fontsize=14, fontweight="bold", y=0.98
    )

    out_png = OUT_DIR / "run6_all_209_evaluations_gp_fit.png"
    out_pdf = OUT_DIR / "run6_all_209_evaluations_gp_fit.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSuccessfully generated comprehensive GP fit plots:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
