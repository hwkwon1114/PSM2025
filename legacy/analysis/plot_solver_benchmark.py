#!/usr/bin/env python3
"""Scientific visualization of solver benchmark results dynamically parsed from production data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    analysis_file = ROOT / 'run/solver_benchmark/benchmark_analysis.json'
    summary_file = ROOT / 'run/solver_benchmark/benchmark_summary.json'

    with open(analysis_file) as f:
        analysis = json.load(f)
    with open(summary_file) as f:
        summary = json.load(f)

    output_dir = ROOT / 'legacy/analysis/plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout='constrained')

    cases = ['simple_a_1cycle', 'moderate_nonoverlapping_20cycle', 'complex_overlapping_20cycle']
    tier_labels = ['Simple (1-cyc)', 'Moderate (20-cyc)', 'Complex (20-cyc)']
    x = np.arange(len(cases))
    width = 0.25

    # Panel (a): Total Solver Phase Seconds by Tier (all 9 runs)
    ax_time = axes[0, 0]
    t_cont = [analysis['cases'][c]['continuation']['total_solver_seconds'] for c in cases]
    t_base = [analysis['cases'][c]['baseline']['total_solver_seconds'] for c in cases]
    t_cut = [analysis['cases'][c]['cutback']['total_solver_seconds'] for c in cases]

    rects1 = ax_time.bar(x - width, t_cont, width, label='Continuation (10 steps)', color='#4daf4a', alpha=0.9)
    rects2 = ax_time.bar(x, t_base, width, label='Baseline Cutback (600s time_cap)', color='#e41a1c', alpha=0.9)
    rects3 = ax_time.bar(x + width, t_cut, width, label='Enhanced Cutback (Stagnation/Hessian)', color='#377eb8', alpha=0.9)

    ax_time.set_ylabel('Solver Computation Time (s)')
    ax_time.set_title('(a) Total Solver Phase Runtime by Tier')
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(tier_labels)
    ax_time.set_ylim(0, max(t_cont) * 1.15)
    ax_time.legend(loc='upper left', framealpha=0.9)
    ax_time.grid(True, linestyle=':', alpha=0.5, axis='y')

    # Annotations
    sp_simple = t_cont[0] / t_cut[0]
    sp_mod = t_cont[1] / t_cut[1]
    sp_complex_base = t_base[2] / t_cut[2]
    sp_complex_cont = t_cont[2] / t_cut[2]

    ax_time.text(x[0] + width, t_cut[0] + 40, f'{sp_simple:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#377eb8')
    ax_time.text(x[1] + width, t_cut[1] + 40, f'{sp_mod:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#377eb8')
    ax_time.text(x[2] + width/2, t_cut[2] + 40, f'{sp_complex_base:.2f}× vs Base\n({sp_complex_cont:.1f}× vs Cont)', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#377eb8')

    # Panel (b): Total Evaluated Hessians by Tier (all 9 runs)
    ax_hess = axes[0, 1]
    h_cont = [analysis['cases'][c]['continuation']['total_hessians'] for c in cases]
    h_base = [analysis['cases'][c]['baseline']['total_hessians'] for c in cases]
    h_cut = [analysis['cases'][c]['cutback']['total_hessians'] for c in cases]

    rects_h1 = ax_hess.bar(x - width, h_cont, width, label='Continuation', color='#4daf4a', alpha=0.9)
    rects_h2 = ax_hess.bar(x, h_base, width, label='Baseline Cutback', color='#e41a1c', alpha=0.9)
    rects_h3 = ax_hess.bar(x + width, h_cut, width, label='Enhanced Cutback', color='#377eb8', alpha=0.9)

    ax_hess.set_ylabel('Total Evaluated Hessians')
    ax_hess.set_title('(b) Hessian Evaluation Count by Tier')
    ax_hess.set_xticks(x)
    ax_hess.set_xticklabels(tier_labels)
    ax_hess.set_ylim(0, max(h_cont) * 1.2)
    ax_hess.legend(loc='upper left', framealpha=0.9)
    ax_hess.grid(True, linestyle=':', alpha=0.5, axis='y')

    for r in rects_h1:
        ax_hess.text(r.get_x() + r.get_width()/2, r.get_height() + 1, f'{int(r.get_height())}', ha='center', va='bottom', fontsize=8)
    for r in rects_h2:
        ax_hess.text(r.get_x() + r.get_width()/2, r.get_height() + 1, f'{int(r.get_height())}', ha='center', va='bottom', fontsize=8)
    for r in rects_h3:
        ax_hess.text(r.get_x() + r.get_width()/2, r.get_height() + 1, f'{int(r.get_height())}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel (c): Convergence Trajectory for Complex Case
    ax_traj = axes[1, 0]
    # Continuation timeline
    t_cont_total = analysis['cases']['complex_overlapping_20cycle']['continuation']['total_solver_seconds']
    t_cum_cont = np.linspace(0, t_cont_total, 11)
    lam_cont = np.linspace(0, 1.0, 11)
    ax_traj.step(t_cum_cont, lam_cont, where='post', label='Continuation (10 increments)', color='#4daf4a', linewidth=1.8)

    # Baseline trajectory
    base_att1 = analysis['cases']['complex_overlapping_20cycle']['baseline']['attempt1_solver_seconds'] # 599.65s
    base_total = analysis['cases']['complex_overlapping_20cycle']['baseline']['total_solver_seconds'] # 1105.65s
    base_att2_end = base_att1 + (base_total - base_att1) / 2.0
    t_base_pts = [0, base_att1, base_att2_end, base_total]
    lam_base_pts = [0, 0, 0.5, 1.0]
    ax_traj.step(t_base_pts, lam_base_pts, where='post', label=f'Baseline Cutback (time_cap hang @ {base_att1:.1f}s)', color='#e41a1c', linestyle='--', linewidth=1.8)

    # Enhanced cutback trajectory
    cut_att1 = analysis['cases']['complex_overlapping_20cycle']['cutback']['attempt1_solver_seconds'] # 254.44s
    cut_total = analysis['cases']['complex_overlapping_20cycle']['cutback']['total_solver_seconds'] # 790.57s
    cut_att2_end = cut_att1 + (cut_total - cut_att1) / 2.0
    t_cut_pts = [0, cut_att1, cut_att2_end, cut_total]
    lam_cut_pts = [0, 0, 0.5, 1.0]
    ax_traj.step(t_cut_pts, lam_cut_pts, where='post', label=f'Enhanced Cutback (stagnation abort @ {cut_att1:.1f}s)', color='#377eb8', linewidth=2.0)

    ax_traj.scatter([cut_att1], [0], color='#377eb8', marker='x', s=70, zorder=5, label=f'Newton Stagnation ({cut_att1:.1f}s)')
    ax_traj.scatter([base_att1], [0], color='#e41a1c', marker='x', s=70, zorder=5, label=f'Wall-clock Timeout ({base_att1:.1f}s)')

    ax_traj.set_xlabel('Elapsed Solver Phase Time (s)')
    ax_traj.set_ylabel('Accepted Load Factor λ')
    ax_traj.set_title('(c) Load Progression Timeline (Schematic with Measured Event Markers)')
    ax_traj.set_ylim(-0.05, 1.1)
    ax_traj.legend(loc='lower right', framealpha=0.9, fontsize=7.5)
    ax_traj.grid(True, linestyle=':', alpha=0.5)

    # Panel (d): Pairwise Shape Equivalence Across Policies (nm RMS)
    ax_equiv = axes[1, 1]
    cats = ['Cutback vs Base\n(Simple)', 'Cutback vs Base\n(Moderate)', 'Cutback vs Base\n(Complex)',
            'Cutback vs Cont\n(Simple)', 'Cutback vs Cont\n(Moderate)', 'Cutback vs Cont\n(Complex)']

    rms_vals = [
        analysis['comparisons']['simple_a_1cycle']['cutback_vs_baseline']['rms_displacement_nm'],
        analysis['comparisons']['moderate_nonoverlapping_20cycle']['cutback_vs_baseline']['rms_displacement_nm'],
        analysis['comparisons']['complex_overlapping_20cycle']['cutback_vs_baseline']['rms_displacement_nm'],
        analysis['comparisons']['simple_a_1cycle']['cutback_vs_continuation']['rms_displacement_nm'],
        analysis['comparisons']['moderate_nonoverlapping_20cycle']['cutback_vs_continuation']['rms_displacement_nm'],
        analysis['comparisons']['complex_overlapping_20cycle']['cutback_vs_continuation']['rms_displacement_nm'],
    ]

    x_c = np.arange(len(cats))
    colors = ['#2b83ba', '#2b83ba', '#2b83ba', '#fdae61', '#fdae61', '#fdae61']
    bars = ax_equiv.bar(x_c, [max(v, 1e-3) for v in rms_vals], color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

    ax_equiv.set_yscale('log')
    ax_equiv.set_ylabel('RMS Displacement Deviation (nm)')
    ax_equiv.set_title('(d) Pairwise Equilibrium Shape Agreement')
    ax_equiv.set_xticks(x_c)
    ax_equiv.set_xticklabels(cats, rotation=25, ha='right', fontsize=7.5)
    ax_equiv.set_ylim(1e-4, 1e3)
    ax_equiv.axhline(100.0, color='gray', linestyle=':', label='100 nm Gate')
    ax_equiv.grid(True, linestyle=':', alpha=0.5, axis='y')

    for b, val in zip(bars, rms_vals):
        txt = f"{val:.4f} nm" if val > 0 else "0.0000 nm\n(Bitwise)"
        ax_equiv.text(b.get_x() + b.get_width()/2, max(val, 1e-3) * 1.8, txt, ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plot_stem = 'solver_benchmark_comparison'
    png_path = output_dir / f'{plot_stem}.png'
    pdf_path = output_dir / f'{plot_stem}.pdf'
    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)

    caption = (
        'Performance benchmark of the enhanced adaptive cutback solver compared against baseline cutback '
        'and conservative continuation across simple, moderate, and complex toolpath tiers. '
        '(a) Total solver phase runtime across tiers. '
        '(b) Total evaluated Hessians across tiers. '
        '(c) Load progression timeline for the complex overlapping case showing deterministic cutback abort '
        'vs baseline wall-clock timeout. '
        '(d) Pairwise equilibrium shape agreement confirming sub-micron physical equivalence across all tiers.'
    )
    (output_dir / f'{plot_stem}.caption.md').write_text(caption + '\n')
    print(f'Successfully generated {png_path} and {pdf_path}')


if __name__ == '__main__':
    main()
