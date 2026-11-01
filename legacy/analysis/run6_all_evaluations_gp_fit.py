#!/usr/bin/env python3
"""
Exact BoTorch SingleTaskGP + qLogExpectedImprovement plot matching calibration/ewcal/batch_bayes.py.

Surrogate Model:
- Inputs: Normalized parameters in [0, 1]^4
- Target: Y = -log(RMS_mm)
- Kernel: BoTorch SingleTaskGP with ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))
- Acquisition: LogExpectedImprovement(model=gp, best_f=max(Y))

6-Panel Structure:
1. (a) BoTorch GP Sensitivity: g_top (log scale, with 95% CI and 212 BO points).
2. (b) BoTorch GP Sensitivity: g_bot (log scale, with 95% CI and 212 BO points).
3. (c) BoTorch GP Sensitivity: ortho_top (regular linear scale, with 95% CI and 212 BO points).
4. (d) BoTorch GP Sensitivity: ortho_bottom (regular linear scale, with 95% CI and 212 BO points).
5. (e) 2D Response Basin: g_top vs ortho_top (BoTorch Predicted RMS in mm).
6. (f) 2D Acquisition Surface: g_top vs ortho_top (BoTorch qLogEI Acquisition Function).
"""

from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "evaluations.jsonl"
BEST_SUMMARY = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "bayes_summary.json"
OUT_DIR = ROOT / "calibration" / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    lines = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
    valid_obs = [r for r in lines if r.get("valid") and r.get("objective_mm", 999.0) < 50.0]
    print(f"Loaded {len(valid_obs)} valid observations from {LOG_PATH}")

    # Best attempt info
    best_info = json.loads(BEST_SUMMARY.read_text())["best"]
    best_params = best_info["parameters"]
    best_norm = best_info.get("normalized_parameters")
    best_id = best_info["attempt_id"]
    best_rms = best_info["objective_mm"]
    print(f"All-time best: Attempt #{best_id}, RMS = {best_rms:.4f} mm")

    bounds_phys = [
        (5e-5, 0.01),      # gtop
        (1e-5, 0.001),     # gbot
        (-1.0, 1.0),       # ortho_top
        (-1.0, 1.0),       # ortho_bottom
    ]
    param_names = [r"$g_{\mathrm{top}}$", r"$g_{\mathrm{bot}}$", r"$\mathrm{ortho}_{\mathrm{top}}$", r"$\mathrm{ortho}_{\mathrm{bottom}}$"]

    if best_norm is None:
        best_norm = [
            (best_params["gtop"] - bounds_phys[0][0]) / (bounds_phys[0][1] - bounds_phys[0][0]),
            (best_params["gbot"] - bounds_phys[1][0]) / (bounds_phys[1][1] - bounds_phys[1][0]),
            (best_params["ortho_top"] - bounds_phys[2][0]) / (bounds_phys[2][1] - bounds_phys[2][0]),
            (best_params["ortho_bottom"] - bounds_phys[3][0]) / (bounds_phys[3][1] - bounds_phys[3][0]),
        ]

    # Assemble training data: Y = -log(RMS) matching batch_bayes.py
    X_train_list = []
    Y_train_list = []
    phys_pts = []

    for r in valid_obs:
        p = r["parameters"]
        norm_x = r.get("normalized_parameters")
        if norm_x is None:
            norm_x = [
                (p["gtop"] - bounds_phys[0][0]) / (bounds_phys[0][1] - bounds_phys[0][0]),
                (p["gbot"] - bounds_phys[1][0]) / (bounds_phys[1][1] - bounds_phys[1][0]),
                (p["ortho_top"] - bounds_phys[2][0]) / (bounds_phys[2][1] - bounds_phys[2][0]),
                (p["ortho_bottom"] - bounds_phys[3][0]) / (bounds_phys[3][1] - bounds_phys[3][0]),
            ]
        X_train_list.append(norm_x)
        Y_train_list.append(-np.log(r["objective_mm"]))
        phys_pts.append([p["gtop"], p["gbot"], p["ortho_top"], p["ortho_bottom"], r["objective_mm"]])

    train_X = torch.tensor(X_train_list, dtype=torch.double)
    train_Y = torch.tensor(Y_train_list, dtype=torch.double).unsqueeze(-1)
    phys_pts = np.array(phys_pts)

    torch.set_num_threads(4)
    print(f"Fitting BoTorch SingleTaskGP (nu=2.5, ard_num_dims=4) to {train_X.shape[0]} points...")
    d = train_X.shape[-1]
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    gp = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 60}})
    print("BoTorch SingleTaskGP successfully optimized!")

    best_f = train_Y.max()
    print(f"Max train_Y: {best_f.item():.4f} (Min RMS: {np.exp(-best_f.item()):.4f} mm)")

    # BoTorch LogEI acquisition function
    logEI_acq = LogExpectedImprovement(model=gp, best_f=best_f)

    # -------------------------------------------------------------
    # 6-Panel Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.30)

    # Panels (a) to (d): 1D Sensitivity Slices
    slice_map = [
        (axes[0, 0], 0, r"$\bf{(a)}$ BoTorch Sensitivity: $g_{\mathrm{top}}$"),
        (axes[0, 1], 1, r"$\bf{(b)}$ BoTorch Sensitivity: $g_{\mathrm{bot}}$"),
        (axes[1, 0], 2, r"$\bf{(c)}$ BoTorch Sensitivity: $\mathrm{ortho}_{\mathrm{top}}$"),
        (axes[1, 1], 3, r"$\bf{(d)}$ BoTorch Sensitivity: $\mathrm{ortho}_{\mathrm{bottom}}$"),
    ]

    best_norm_t = torch.tensor(best_norm, dtype=torch.double)
    n_pts = 160

    for ax, i, title in slice_map:
        p_name = param_names[i]
        p_lb, p_ub = bounds_phys[i]

        s_norm = np.linspace(0.0, 1.0, n_pts)
        X_test = best_norm_t.repeat(n_pts, 1)
        X_test[:, i] = torch.tensor(s_norm, dtype=torch.double)

        with torch.no_grad():
            posterior = gp.posterior(X_test)
            mu = posterior.mean.squeeze(-1).numpy()
            std = np.sqrt(posterior.variance.squeeze(-1).numpy())

        # Back-transform from Y = -log(RMS) to RMS (mm)
        rms_pred = np.exp(-mu)
        rms_lower = np.exp(-(mu + 1.96 * std))
        rms_upper = np.exp(-(mu - 1.96 * std))

        s_phys = p_lb + s_norm * (p_ub - p_lb)

        # Plot 95% Credible Interval
        ax.fill_between(
            s_phys, rms_lower, rms_upper,
            color="#3498db", alpha=0.25, label=r"BoTorch 95% CI ($\pm 1.96\sigma$)"
        )
        # Plot Posterior Median / Mean
        ax.plot(s_phys, rms_pred, color="#2980b9", lw=2.2, label="BoTorch Posterior Mean")

        # Scatter all BO solves
        ax.scatter(
            phys_pts[:, i], phys_pts[:, 4],
            color="#7f8c8d", alpha=0.55, s=20, edgecolors="none", label=f"BO Solves (N={len(phys_pts)})"
        )

        # Mark Attempt 209
        best_val = best_params[["gtop", "gbot", "ortho_top", "ortho_bottom"][i]]
        ax.scatter(
            best_val, best_rms,
            color="#e74c3c", s=140, zorder=6, marker="*", edgecolor="black", lw=1.2,
            label=f"Attempt #{best_id} ({best_rms:.3f} mm)"
        )
        ax.axvline(best_val, color="#e74c3c", linestyle=":", lw=1.2, alpha=0.8)

        ax.set_yscale("log")
        ax.set_ylim(0.9, 60.0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel(f"{p_name}", fontsize=11.5)
        ax.set_ylabel("RMS Error (mm, Log Scale)", fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

        if i in (0, 1):
            ax.set_xscale("log")
            ax.set_xlim(p_lb, p_ub)
        else:
            ax.set_xlim(p_lb, p_ub)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # 2D Grid Setup for (gtop, ortho_top)
    # -------------------------------------------------------------
    n_grid = 70
    gtop_norm = np.linspace(0.0, 0.6, n_grid)  # up to ~0.006
    otop_norm = np.linspace(0.0, 1.0, n_grid)
    G1_n, O1_n = np.meshgrid(gtop_norm, otop_norm)

    X_grid = best_norm_t.repeat(n_grid * n_grid, 1)
    X_grid[:, 0] = torch.tensor(G1_n.ravel(), dtype=torch.double)
    X_grid[:, 2] = torch.tensor(O1_n.ravel(), dtype=torch.double)

    with torch.no_grad():
        post = gp.posterior(X_grid)
        mu_grid = post.mean.squeeze(-1).numpy().reshape(n_grid, n_grid)
        rms_grid = np.exp(-mu_grid)

        # Evaluate exact LogEI acquisition values
        acq_vals = logEI_acq(X_grid.unsqueeze(1)).numpy().reshape(n_grid, n_grid)

    gtop_phys = bounds_phys[0][0] + G1_n * (bounds_phys[0][1] - bounds_phys[0][0])
    otop_phys = bounds_phys[2][0] + O1_n * (bounds_phys[2][1] - bounds_phys[2][0])

    # -------------------------------------------------------------
    # Panel (e): 2D Response Basin: g_top vs ortho_top (Predicted RMS)
    # -------------------------------------------------------------
    ax_e = axes[0, 2]
    levels_rms = [1.18, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 35.0]
    cs_e = ax_e.contourf(
        gtop_phys, otop_phys, rms_grid,
        levels=levels_rms, norm=LogNorm(vmin=1.18, vmax=35.0), cmap="viridis_r", extend="both"
    )
    cbar_e = fig.colorbar(cs_e, ax=ax_e, fraction=0.046, pad=0.04)
    cbar_e.set_label("BoTorch Predicted RMS (mm)", fontsize=10)
    cbar_e.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Overlay contour lines with labels
    cs_lines = ax_e.contour(gtop_phys, otop_phys, rms_grid, levels=[1.3, 2.0, 5.0], colors="black", linewidths=0.7, alpha=0.5)
    ax_e.clabel(cs_lines, inline=True, fontsize=8, fmt="%.1f mm")

    ax_e.scatter(phys_pts[:, 0], phys_pts[:, 2], color="white", edgecolors="#2c3e50", s=22, alpha=0.8, label=f"BO Solves (N={len(phys_pts)})")
    ax_e.scatter(best_params["gtop"], best_params["ortho_top"], color="#e74c3c", s=150, marker="*", edgecolor="black", zorder=6, label=f"Attempt #{best_id}")
    ax_e.set_xscale("log")
    ax_e.set_xlim(bounds_phys[0][0], 0.006)
    ax_e.set_ylim(-1.0, 1.0)
    ax_e.set_xlabel(r"$g_{\mathrm{top}}$ (Log Scale)", fontsize=11)
    ax_e.set_ylabel(r"$\mathrm{ortho}_{\mathrm{top}}$", fontsize=11)
    ax_e.set_title(r"$\bf{(e)}$ BoTorch Response Basin (RMS)", fontsize=12, pad=8)
    ax_e.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel (f): 2D Acquisition Surface: g_top vs ortho_top (qLogEI)
    # -------------------------------------------------------------
    ax_f = axes[1, 2]
    # In BoTorch, LogEI values are in log space (e.g. -15 to -3)
    acq_clean = np.clip(acq_vals, -18.0, 0.0)

    cs_f = ax_f.contourf(
        gtop_phys, otop_phys, acq_clean,
        levels=25, cmap="magma", extend="both"
    )
    cbar_f = fig.colorbar(cs_f, ax=ax_f, fraction=0.046, pad=0.04)
    cbar_f.set_label("qLogEI Acquisition Value", fontsize=10)

    ax_f.scatter(phys_pts[:, 0], phys_pts[:, 2], color="cyan", edgecolors="#2c3e50", s=22, alpha=0.8, label=f"BO Solves (N={len(phys_pts)})")
    ax_f.scatter(best_params["gtop"], best_params["ortho_top"], color="#e74c3c", s=150, marker="*", edgecolor="white", lw=1.2, zorder=6, label=f"Attempt #{best_id}")
    ax_f.set_xscale("log")
    ax_f.set_xlim(bounds_phys[0][0], 0.006)
    ax_f.set_ylim(-1.0, 1.0)
    ax_f.set_xlabel(r"$g_{\mathrm{top}}$ (Log Scale)", fontsize=11)
    ax_f.set_ylabel(r"$\mathrm{ortho}_{\mathrm{top}}$", fontsize=11)
    ax_f.set_title(r"$\bf{(f)}$ BoTorch qLogEI Acquisition Surface", fontsize=12, pad=8)
    ax_f.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    out_png = OUT_DIR / "run6_all_209_evaluations_gp_fit.png"
    out_pdf = OUT_DIR / "run6_all_209_evaluations_gp_fit.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSuccessfully generated exact BoTorch qLogEI-matched plots:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
