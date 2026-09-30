#!/usr/bin/env python3
"""
Generate publication-quality diagnostic figure assessing the numerical accuracy
of coordinate-stable gauge hints and guarded factorization reuse against
fresh greedy QR baseline on the Bilayer SVK shell equilibrium benchmark.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parent.parent
    bench_dir = root / 'run/nonconvex_solver_benchmark/numerical_gauge_accuracy'
    report_file = bench_dir / 'numerical_accuracy_report.json'

    with open(report_file) as f:
        data = json.load(f)

    # Style configuration
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    # Okabe-Ito colors
    c_qr = '#0072B2'       # Blue
    c_hint = '#D55E00'     # Vermillion
    c_guarded = '#009E73'  # Green
    c_neutral = '#56B4E9'  # Sky blue

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_energy = fig.add_subplot(gs[0, 0])
    ax_grad = fig.add_subplot(gs[0, 1])
    ax_cond = fig.add_subplot(gs[0, 2])
    ax_ecdf = fig.add_subplot(gs[1, 0])
    ax_single = fig.add_subplot(gs[1, 1])
    ax_intrinsic = fig.add_subplot(gs[1, 2])

    traj = data['trajectories']
    h_qr = traj['qr']['history']
    h_hint = traj['hint']['history']

    steps_qr = [s['step'] for s in h_qr]
    e_qr = [s['energy'] for s in h_qr]
    g_qr = [s['gradient_norm'] for s in h_qr]

    steps_hint = [s['step'] for s in h_hint]
    e_hint = [s['energy'] for s in h_hint]
    g_hint = [s['gradient_norm'] for s in h_hint]
    cond_hint = [s['gauge_condition_number'] for s in h_hint]

    # Panel 1: Energy Descent
    ax_energy.plot(steps_qr, e_qr, color=c_qr, label=f'Fresh QR ({len(steps_qr)} steps)', linestyle='-')
    ax_energy.plot(steps_hint, e_hint, color=c_hint, label=f'Stable Hint ({len(steps_hint)} steps)', linestyle='--')
    ax_energy.set_xlabel('Newton Iteration')
    ax_energy.set_ylabel('Total Elastic Energy [J]')
    ax_energy.set_title('(a) Equilibrium Energy Descent')
    ax_energy.grid(True)
    ax_energy.legend(loc='upper right')

    # Panel 2: Gradient Norm Convergence
    ax_grad.semilogy(steps_qr, g_qr, color=c_qr, label='Fresh QR', linestyle='-')
    ax_grad.semilogy(steps_hint, g_hint, color=c_hint, label='Stable Hint', linestyle='--')
    ax_grad.axhline(5e-14, color='black', linestyle=':', linewidth=1.2, label='Target Tol (5e-14)')
    ax_grad.set_xlabel('Newton Iteration')
    ax_grad.set_ylabel(r'$\|\nabla E(x)\|$ [N]')
    ax_grad.set_title('(b) Gradient Norm Convergence')
    ax_grad.grid(True, which='both')
    ax_grad.legend(loc='lower left')

    # Panel 3: Gauge Submatrix Condition Number
    ax_cond.plot(steps_hint, cond_hint, color=c_hint, marker='o', markersize=3, label=r'$\kappa(R_{\mathrm{gauge}})$')
    ax_cond.axhline(100.0, color='red', linestyle='--', linewidth=1.2, label='Safeguard Threshold (100)')
    ax_cond.set_xlabel('Newton Iteration')
    ax_cond.set_ylabel(r'Condition Number $\kappa$')
    ax_cond.set_title(r'(c) Gauge Condition Stability ($\kappa \in [4.05, 4.08]$)')
    ax_cond.set_ylim(0, 10)
    ax_cond.grid(True)
    ax_cond.legend(loc='upper right')

    # Panel 4: ECDF of Procrustes Vertex Discrepancy
    nV = 5427
    x_qr = np.fromfile(bench_dir / 'x_final_qr.f64', dtype=np.float64)
    x_hint = np.fromfile(bench_dir / 'x_final_hint.f64', dtype=np.float64)
    v_qr = x_qr[:nV*3].reshape((nV, 3), order='F')
    v_hint = x_hint[:nV*3].reshape((nV, 3), order='F')

    centroid_qr = v_qr.mean(axis=0)
    centroid_hint = v_hint.mean(axis=0)
    P = v_qr - centroid_qr
    Q = v_hint - centroid_hint

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_aligned = P @ R.T
    v_diff_nm = np.linalg.norm(P_aligned - Q, axis=1) * 1e9  # in nanometers

    sorted_diff = np.sort(v_diff_nm)
    ecdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)

    ax_ecdf.plot(sorted_diff, ecdf, color=c_hint, linewidth=2.0)
    rms_nm = np.sqrt(np.mean(v_diff_nm**2))
    max_nm = np.max(v_diff_nm)
    ax_ecdf.axvline(rms_nm, color='black', linestyle='--', label=f'RMS = {rms_nm:.1f} nm')
    ax_ecdf.axvline(max_nm, color='gray', linestyle=':', label=f'Max = {max_nm:.1f} nm')
    ax_ecdf.set_xlabel('Procrustes Vertex Error [nm]')
    ax_ecdf.set_ylabel('Empirical CDF')
    ax_ecdf.set_title('(d) Converged Shape Positional Error (ECDF)')
    ax_ecdf.grid(True)
    ax_ecdf.legend(loc='lower right')

    # Panel 5: Single-Step Displacement Decomposition
    single_comp = data['single_step_linear_solve']['comparison']
    rigid_nm = single_comp['rigid_component_norm'] * 1e3
    phys_nm = single_comp['non_rigid_component_norm'] * 1e3
    raw_nm = single_comp['raw_diff_norm'] * 1e3

    bars = ax_single.bar(['Rigid Shift\n(Pure Gauge)', 'Non-Rigid\n(Physical Strain)', 'Raw Step Diff\n$\|\Delta x_{\mathrm{qr}} - \Delta x_{\mathrm{hint}}\|$'],
                         [rigid_nm, phys_nm, raw_nm], color=[c_neutral, c_hint, c_qr], width=0.55)
    ax_single.set_ylabel(r'Displacement Norm [$10^{-3}$]')
    ax_single.set_title('(e) Step 1 Displacement Decomposition')
    ax_single.grid(axis='y')
    for bar in bars:
        yval = bar.get_height()
        ax_single.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
    ax_single.set_ylim(0, max(rigid_nm, phys_nm, raw_nm) * 1.25)

    # Panel 6: Intrinsic Gauge-Invariant Quantities
    inv = data['intrinsic_invariance']
    theta_qr = x_qr[nV*3:]
    theta_hint = x_hint[nV*3:]
    theta_rms = np.sqrt(np.mean((theta_qr - theta_hint)**2))

    labels = [
        r'Rel. Energy' + '\n' + r'$\Delta E^* / E^*$',
        'Edge Length\nRMS Error [m]',
        'Face Area\nRMS Error [m$^2$]',
        'Director Angle\nRMS Error [rad]'
    ]
    vals = [
        traj['hint']['relative_energy_diff_vs_qr'],
        inv['edge_length_rms_diff'],
        inv['face_area_rms_diff'],
        theta_rms
    ]

    x_pos = np.arange(len(labels))
    bars6 = ax_intrinsic.bar(x_pos, vals, color='#CC79A7', width=0.55)
    ax_intrinsic.set_yscale('log')
    ax_intrinsic.set_xticks(x_pos)
    ax_intrinsic.set_xticklabels(labels, fontsize=9)
    ax_intrinsic.set_ylabel('Absolute / Relative Difference')
    ax_intrinsic.set_title('(f) Intrinsic Metric Parity (Hint vs QR)')
    ax_intrinsic.grid(axis='y', which='both')
    for bar in bars6:
        yval = bar.get_height()
        ax_intrinsic.text(bar.get_x() + bar.get_width()/2.0, yval * 2.0, f'{yval:.1e}', ha='center', va='bottom', fontsize=9)
    ax_intrinsic.set_ylim(1e-16, 1e-6)

    fig.suptitle('Numerical Accuracy and Gauge Invariance Assessment: Coordinate-Stable Gauge vs Fresh QR Baseline', fontsize=15, y=1.02)

    plot_path = bench_dir / 'accuracy_metrics.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] Saved publication-quality diagnostic figure to {plot_path}")

if __name__ == '__main__':
    main()
