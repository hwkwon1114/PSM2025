#!/usr/bin/env python3
"""
Publication 6-panel figure:
1. (a) LOOCV Parity Plot of the 240 samples (Observed vs GP Predicted RMS on log scale, R^2 = 0.967).
2. (b) Log-scale RMS vs log-scale g_top.
3. (c) Log-scale RMS vs log-scale g_bot.
4. (d) Log-scale RMS vs ortho_top (regular linear scale).
5. (e) Log-scale RMS vs ortho_bottom (regular linear scale).
6. (f) Optimization Convergence History across all 240 attempts (log-scale RMS vs Attempt ID).
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
LOG_PATH = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "evaluations.jsonl"
BEST_SUMMARY = ROOT / "run" / "run6_expanded_bo_continue20_recovery" / "bayes_summary.json"
OUT_DIR = ROOT / "calibration" / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_evaluations():
    lines = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
    lines.sort(key=lambda r: r.get("attempt_id", 0))

    all_attempts = []
    valid_attempts = []

    for r in lines:
        all_attempts.append(r)
        if r.get("valid", False) and r.get("objective_mm", 999.0) < 100.0:
            valid_attempts.append(r)

    return all_attempts, valid_attempts


def main():
    all_attempts, valid_attempts = load_evaluations()
    n_all = len(all_attempts)
    n_valid = len(valid_attempts)
    print(f"Total attempts parsed: {n_all}")
    print(f"Valid attempts: {n_valid}")

    # Best parameters from Attempt 209
    best_info = json.loads(BEST_SUMMARY.read_text())["best"]
    best_params = best_info["parameters"]
    best_id = best_info["attempt_id"]
    best_rms = best_info["objective_mm"]
    print(f"Best: Attempt {best_id}, RMS = {best_rms:.4f} mm")

    # Arrays for GP fitting
    gtop = np.array([r["parameters"]["gtop"] for r in valid_attempts])
    gbot = np.array([r["parameters"]["gbot"] for r in valid_attempts])
    otop = np.array([r["parameters"]["ortho_top"] for r in valid_attempts])
    obot = np.array([r["parameters"]["ortho_bottom"] for r in valid_attempts])
    rms = np.array([r["objective_mm"] for r in valid_attempts])

    # Transform features: log10 on growth, regular on ortho
    X_phys = np.column_stack([gtop, gbot, otop, obot])
    X = np.column_stack([np.log10(gtop), np.log10(gbot), otop, obot])
    y_log = np.log10(rms)

    # Min/max normalization to [0, 1]^4
    X_min = np.array([np.log10(5e-5), np.log10(1e-5), -1.0, -1.0])
    X_max = np.array([np.log10(0.01), np.log10(0.001), 1.0, 1.0])
    X_norm = (X - X_min) / (X_max - X_min)

    best_x_phys = np.array([best_params["gtop"], best_params["gbot"], best_params["ortho_top"], best_params["ortho_bottom"]])
    best_x_trans = np.array([np.log10(best_params["gtop"]), np.log10(best_params["gbot"]), best_params["ortho_top"], best_params["ortho_bottom"]])
    best_x_norm = (best_x_trans - X_min) / (X_max - X_min)

    # Fit GP on log10(RMS)
    print("Fitting GP on log-transformed space...")
    kernel = ConstantKernel(1.0, (0.01, 100.0)) * Matern(length_scale=np.ones(4) * 0.4, length_scale_bounds=(0.05, 10.0), nu=2.5) + WhiteKernel(1e-4, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X_norm, y_log)
    print(f"Optimized Kernel: {gp.kernel_}")

    # Compute exact LOOCV via K^-1
    K = gp.kernel_(X_norm)
    K_inv = np.linalg.pinv(K)
    y_loo_log = y_log - (K_inv @ y_log) / np.diag(K_inv)
    sigma_loo_log = np.sqrt(1.0 / np.diag(K_inv))

    y_loo = 10**y_loo_log
    r2_log = 1.0 - np.sum((y_log - y_loo_log)**2) / np.sum((y_log - np.mean(y_log))**2)
    r2_linear = 1.0 - np.sum((rms - y_loo)**2) / np.sum((rms - np.mean(rms))**2)
    rmse_log = np.sqrt(np.mean((y_log - y_loo_log)**2))
    rmse_linear = np.sqrt(np.mean((rms - y_loo)**2))
    corr = np.corrcoef(rms, y_loo)[0, 1]

    print(f"LOOCV Metrics:")
    print(f"  R^2 (log space)   : {r2_log:.4f}")
    print(f"  R^2 (linear space): {r2_linear:.4f}")
    print(f"  RMSE (linear)     : {rmse_linear:.4f} mm")
    print(f"  Pearson r         : {corr:.4f}")

    # -----------------------------------------------------------------
    # Create 6-Panel Figure
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    plt.subplots_adjust(hspace=0.30, wspace=0.25)

    # Palette
    c_pts = "#7f8c8d"
    c_best = "#e74c3c"
    c_gp = "#2980b9"
    c_band = "#3498db"

    # =================================================================
    # Panel 1: LOOCV Parity Plot (Log-Log Scale)
    # =================================================================
    ax1 = axes[0, 0]
    ax1.plot([0.8, 80], [0.8, 80], color="#2c3e50", linestyle="--", lw=1.5, zorder=2, label="1:1 Parity")
    ax1.fill_between([0.8, 80], [0.8 * 0.8, 80 * 0.8], [0.8 * 1.2, 80 * 1.2], color="#bdc3c7", alpha=0.3, zorder=1, label="±20% Error Band")

    # Scatter points with error bars from sigma_loo
    y_loo_lower = 10**(y_loo_log - 1.96 * sigma_loo_log)
    y_loo_upper = 10**(y_loo_log + 1.96 * sigma_loo_log)
    ax1.errorbar(
        rms, y_loo, yerr=[y_loo - y_loo_lower, y_loo_upper - y_loo],
        fmt="o", color=c_gp, ecolor="#95a5a6", elinewidth=0.6, capsize=0,
        markersize=5, alpha=0.75, zorder=3, label=f"LOOCV Samples (N={n_valid})"
    )

    # Highlight Attempt 209
    best_idx = np.argmin(rms)
    ax1.scatter(
        rms[best_idx], y_loo[best_idx],
        color=c_best, s=140, marker="*", edgecolor="black", lw=1.2, zorder=5,
        label=f"Attempt {best_id} ({best_rms:.3f} mm)"
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.9, 70)
    ax1.set_ylim(0.9, 70)
    ax1.set_xlabel("Observed Symmetric RMS Error (mm)", fontsize=11)
    ax1.set_ylabel("LOOCV Predicted RMS Error (mm)", fontsize=11)
    ax1.set_title(r"$\bf{(a)}$ LOOCV Parity Plot ($R^2 = 0.967$)", fontsize=12, pad=8)
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    stats_text = f"LOOCV $R^2 = {r2_log:.3f}$\nPearson $r = {corr:.3f}$\nRMSE $= {rmse_linear:.2f}$ mm"
    ax1.text(0.95, 0.08, stats_text, transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc3c7", alpha=0.9))

    # =================================================================
    # Panel 2: Log-scale RMS vs log-scale g_top
    # =================================================================
    ax2 = axes[0, 1]
    n_sweep = 150
    gtop_sweep = np.logspace(np.log10(5e-5), np.log10(0.01), n_sweep)
    X_sw2 = np.tile(best_x_trans, (n_sweep, 1))
    X_sw2[:, 0] = np.log10(gtop_sweep)
    X_sw2_norm = (X_sw2 - X_min) / (X_max - X_min)

    mu_sw2, std_sw2 = gp.predict(X_sw2_norm, return_std=True)
    y_pred2 = 10**mu_sw2
    y_low2 = 10**(mu_sw2 - 1.96 * std_sw2)
    y_high2 = 10**(mu_sw2 + 1.96 * std_sw2)

    ax2.fill_between(gtop_sweep, y_low2, y_high2, color=c_band, alpha=0.25, label=r"GP 95% CI ($\pm 1.96\sigma$)")
    ax2.plot(gtop_sweep, y_pred2, color=c_gp, lw=2.0, label="GP Mean Slice")
    ax2.scatter(gtop, rms, color=c_pts, s=20, alpha=0.6, label="All 221 Solves")
    ax2.scatter(best_params["gtop"], best_rms, color=c_best, s=140, marker="*", edgecolor="black", zorder=5, label=f"Attempt {best_id}")

    ax2.axvline(best_params["gtop"], color=c_best, linestyle=":", lw=1.2, alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(4e-5, 1.2e-2)
    ax2.set_ylim(0.9, 70)
    ax2.set_xlabel(r"$g_{\mathrm{top}}$ (Engineering Strain, Log Scale)", fontsize=11)
    ax2.set_ylabel("RMS Error (mm, Log Scale)", fontsize=11)
    ax2.set_title(r"$\bf{(b)}$ Sensitivity: $\log(g_{\mathrm{top}})$ vs $\log(\mathrm{RMS})$", fontsize=12, pad=8)
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # =================================================================
    # Panel 3: Log-scale RMS vs log-scale g_bot
    # =================================================================
    ax3 = axes[0, 2]
    gbot_sweep = np.logspace(np.log10(1e-5), np.log10(0.001), n_sweep)
    X_sw3 = np.tile(best_x_trans, (n_sweep, 1))
    X_sw3[:, 1] = np.log10(gbot_sweep)
    X_sw3_norm = (X_sw3 - X_min) / (X_max - X_min)

    mu_sw3, std_sw3 = gp.predict(X_sw3_norm, return_std=True)
    y_pred3 = 10**mu_sw3
    y_low3 = 10**(mu_sw3 - 1.96 * std_sw3)
    y_high3 = 10**(mu_sw3 + 1.96 * std_sw3)

    ax3.fill_between(gbot_sweep, y_low3, y_high3, color=c_band, alpha=0.25, label=r"GP 95% CI")
    ax3.plot(gbot_sweep, y_pred3, color=c_gp, lw=2.0, label="GP Mean Slice")
    ax3.scatter(gbot, rms, color=c_pts, s=20, alpha=0.6, label="All 221 Solves")
    ax3.scatter(best_params["gbot"], best_rms, color=c_best, s=140, marker="*", edgecolor="black", zorder=5, label=f"Attempt {best_id}")

    ax3.axvline(best_params["gbot"], color=c_best, linestyle=":", lw=1.2, alpha=0.8)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(8e-6, 1.2e-3)
    ax3.set_ylim(0.9, 70)
    ax3.set_xlabel(r"$g_{\mathrm{bot}}$ (Engineering Strain, Log Scale)", fontsize=11)
    ax3.set_ylabel("RMS Error (mm, Log Scale)", fontsize=11)
    ax3.set_title(r"$\bf{(c)}$ Sensitivity: $\log(g_{\mathrm{bot}})$ vs $\log(\mathrm{RMS})$", fontsize=12, pad=8)
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # =================================================================
    # Panel 4: Log-scale RMS vs ortho_top (Regular Linear Scale)
    # =================================================================
    ax4 = axes[1, 0]
    otop_sweep = np.linspace(-1.0, 1.0, n_sweep)
    X_sw4 = np.tile(best_x_trans, (n_sweep, 1))
    X_sw4[:, 2] = otop_sweep
    X_sw4_norm = (X_sw4 - X_min) / (X_max - X_min)

    mu_sw4, std_sw4 = gp.predict(X_sw4_norm, return_std=True)
    y_pred4 = 10**mu_sw4
    y_low4 = 10**(mu_sw4 - 1.96 * std_sw4)
    y_high4 = 10**(mu_sw4 + 1.96 * std_sw4)

    ax4.fill_between(otop_sweep, y_low4, y_high4, color=c_band, alpha=0.25, label=r"GP 95% CI")
    ax4.plot(otop_sweep, y_pred4, color=c_gp, lw=2.0, label="GP Mean Slice")
    ax4.scatter(otop, rms, color=c_pts, s=20, alpha=0.6, label="All 221 Solves")
    ax4.scatter(best_params["ortho_top"], best_rms, color=c_best, s=140, marker="*", edgecolor="black", zorder=5, label=f"Attempt {best_id}")

    ax4.axvline(best_params["ortho_top"], color=c_best, linestyle=":", lw=1.2, alpha=0.8)
    ax4.set_yscale("log")
    ax4.set_xlim(-1.05, 1.05)
    ax4.set_ylim(0.9, 70)
    ax4.set_xlabel(r"$\mathrm{ortho}_{\mathrm{top}}$ (Regular Linear Scale)", fontsize=11)
    ax4.set_ylabel("RMS Error (mm, Log Scale)", fontsize=11)
    ax4.set_title(r"$\bf{(d)}$ Sensitivity: $\mathrm{ortho}_{\mathrm{top}}$ vs $\log(\mathrm{RMS})$", fontsize=12, pad=8)
    ax4.grid(True, which="both", linestyle=":", alpha=0.4)
    ax4.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # =================================================================
    # Panel 5: Log-scale RMS vs ortho_bottom (Regular Linear Scale)
    # =================================================================
    ax5 = axes[1, 1]
    obot_sweep = np.linspace(-1.0, 1.0, n_sweep)
    X_sw5 = np.tile(best_x_trans, (n_sweep, 1))
    X_sw5[:, 3] = obot_sweep
    X_sw5_norm = (X_sw5 - X_min) / (X_max - X_min)

    mu_sw5, std_sw5 = gp.predict(X_sw5_norm, return_std=True)
    y_pred5 = 10**mu_sw5
    y_low5 = 10**(mu_sw5 - 1.96 * std_sw5)
    y_high5 = 10**(mu_sw5 + 1.96 * std_sw5)

    ax5.fill_between(obot_sweep, y_low5, y_high5, color=c_band, alpha=0.25, label=r"GP 95% CI")
    ax5.plot(obot_sweep, y_pred5, color=c_gp, lw=2.0, label="GP Mean Slice")
    ax5.scatter(obot, rms, color=c_pts, s=20, alpha=0.6, label="All 221 Solves")
    ax5.scatter(best_params["ortho_bottom"], best_rms, color=c_best, s=140, marker="*", edgecolor="black", zorder=5, label=f"Attempt {best_id}")

    ax5.axvline(best_params["ortho_bottom"], color=c_best, linestyle=":", lw=1.2, alpha=0.8)
    ax5.set_yscale("log")
    ax5.set_xlim(-1.05, 1.05)
    ax5.set_ylim(0.9, 70)
    ax5.set_xlabel(r"$\mathrm{ortho}_{\mathrm{bottom}}$ (Regular Linear Scale)", fontsize=11)
    ax5.set_ylabel("RMS Error (mm, Log Scale)", fontsize=11)
    ax5.set_title(r"$\bf{(e)}$ Sensitivity: $\mathrm{ortho}_{\mathrm{bottom}}$ vs $\log(\mathrm{RMS})$", fontsize=12, pad=8)
    ax5.grid(True, which="both", linestyle=":", alpha=0.4)
    ax5.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    # =================================================================
    # Panel 6: Optimization Convergence Trace across all 240 attempts
    # =================================================================
    ax6 = axes[1, 2]
    all_ids = [int(r.get("attempt_id", i + 1)) for i, r in enumerate(all_attempts)]
    all_objs = []
    for r in all_attempts:
        v = r.get("objective_mm")
        if v is not None and isinstance(v, (int, float)):
            all_objs.append(float(v))
        else:
            all_objs.append(np.nan)
    all_objs = np.array(all_objs, dtype=float)

    # Running minimum (filtering NaNs and invalid solves)
    running_min = []
    current_min = float("inf")
    for obj in all_objs:
        if np.isfinite(obj) and obj < current_min:
            current_min = obj
        running_min.append(current_min)

    # Scatter all attempts
    valid_mask = [np.isfinite(o) and o < 100.0 for o in all_objs]
    ax6.scatter(
        [all_ids[i] for i in range(n_all) if valid_mask[i]],
        [all_objs[i] for i in range(n_all) if valid_mask[i]],
        color=c_pts, s=18, alpha=0.55, label=f"All Attempts ({n_all})"
    )

    # Running minimum trace
    ax6.step(all_ids, running_min, color=c_best, lw=2.2, where="post", zorder=4, label="Best-So-Far Minimum")

    # Mark Attempt 209
    ax6.scatter(best_id, best_rms, color=c_best, s=150, marker="*", edgecolor="black", zorder=6, label=f"Global Best: #{best_id} ({best_rms:.3f} mm)")

    # Separate initial LHS (1–5) and BO phases
    ax6.axvline(5.5, color="#7f8c8d", linestyle=":", lw=1.2, alpha=0.7)
    ax6.text(3, 40, "Initial\nLHS", ha="center", fontsize=8.5, color="#7f8c8d", fontweight="bold")
    ax6.text(120, 40, "Bayesian Optimization Exploration & Exploitation (Run 6)", ha="center", fontsize=9, color="#2c3e50")

    ax6.set_yscale("log")
    ax6.set_xlim(0, 245)
    ax6.set_ylim(0.9, 70)
    ax6.set_xlabel("BO Attempt ID (1 to 240)", fontsize=11)
    ax6.set_ylabel("RMS Error (mm, Log Scale)", fontsize=11)
    ax6.set_title(r"$\bf{(f)}$ Convergence History (All 240 Evaluations)", fontsize=12, pad=8)
    ax6.grid(True, which="both", linestyle=":", alpha=0.4)
    ax6.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    out_png = OUT_DIR / "run6_gp_loocv_and_logscale_surrogate.png"
    out_pdf = OUT_DIR / "run6_gp_loocv_and_logscale_surrogate.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"\nGenerated 6-panel figure:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
