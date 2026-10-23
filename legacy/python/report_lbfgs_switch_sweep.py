#!/usr/bin/env python3
"""Generate publication figures and summary tables for the L-BFGS switch sweep."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    screen_dir = root / 'screen'
    audit_dir = root / 'audit'
    audit_dir.mkdir(exist_ok=True)

    ledger = json.loads((screen_dir / 'ledger.json').read_text())
    cells = ledger['cells']

    records = []
    traces = {}
    for c in cells:
        idx = c['index']
        res = c.get('result', {})
        sw = c.get('switch_attempts')
        start = c.get('start')
        charged = c.get('charged_seconds', 0)
        accepted = res.get('accepted', False)
        hessians = res.get('hessians', 0)
        evals = res.get('evaluations', 0)
        energy = res.get('energy', float('nan'))
        grad = res.get('gradient_norm', float('nan'))

        # Read trace from checkpoint
        cp_path = Path(res.get('checkpoint', ''))
        lbfgs_time = 0.0
        newton_time = 0.0
        shifted = 0
        if cp_path.exists():
            cp_data = json.loads(cp_path.read_text())
            trace = cp_data.get('payload', {}).get('state', {}).get('trace', [])
            valid = [[v['elapsed'], v['energy'], v['gradient_norm']] for v in trace if 'energy' in v and 'gradient_norm' in v]
            traces[idx] = np.array(valid)
            newton_steps = [v for v in trace if 'hessian_oracle_seconds' in v]
            if newton_steps:
                first_newton_t = newton_steps[0]['elapsed']
                final_t = trace[-1]['elapsed']
                lbfgs_time = first_newton_t
                newton_time = final_t - first_newton_t
                shifted = sum(1 for v in newton_steps if v.get('shift', 0.0) > 0.0)
            else:
                lbfgs_time = trace[-1]['elapsed'] if trace else 0.0

        records.append({
            'index': idx,
            'start': start,
            'switch_attempts': sw,
            'charged_seconds': charged,
            'lbfgs_seconds': lbfgs_time,
            'newton_seconds': newton_time,
            'newton_steps': hessians,
            'shifted_updates': shifted,
            'evaluations': evals,
            'accepted': accepted,
            'energy': energy,
            'gradient_norm': grad
        })

    # Save CSV
    csv_path = audit_dir / 'switch_sweep_results.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f'Saved results to {csv_path}')

    # Visualizations
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')

        starts = ['cylinder_y_plus', 'cylinder_y_minus']
        colors = {'cylinder_y_plus': '#0072B2', 'cylinder_y_minus': '#D55E00'}
        markers = {'cylinder_y_plus': 'o', 'cylinder_y_minus': 's'}

        # 1. Total Wall Time vs Switch Attempts
        ax = axs[0]
        for s in starts:
            sub = [r for r in records if r['start'] == s]
            sub.sort(key=lambda r: r['switch_attempts'])
            sws = [r['switch_attempts'] for r in sub]
            times = [r['charged_seconds'] for r in sub]
            label = 'Positive start' if s == 'cylinder_y_plus' else 'Negative start'
            ax.plot(sws, times, marker=markers[s], color=colors[s], lw=2, ms=7, label=label)
        ax.set_xlabel('L-BFGS Switch Attempts')
        ax.set_ylabel('Cell Wall Time (s)')
        ax.set_title('Total Solve Time vs. Switch Point')
        ax.grid(True, ls=':', alpha=0.6)
        ax.legend(frameon=False)

        # 2. Breakdown: L-BFGS vs Newton Time
        ax = axs[1]
        width = 60
        sub_plus = sorted([r for r in records if r['start'] == 'cylinder_y_plus'], key=lambda r: r['switch_attempts'])
        sws = np.array([r['switch_attempts'] for r in sub_plus])
        lbfgs_t = np.array([r['lbfgs_seconds'] for r in sub_plus])
        newton_t = np.array([r['newton_seconds'] for r in sub_plus])

        ax.bar(sws - width/2, lbfgs_t, width=width, color='#56B4E9', label='L-BFGS phase')
        ax.bar(sws - width/2, newton_t, bottom=lbfgs_t, width=width, color='#E69F00', label='Newton phase')
        ax.set_xlabel('L-BFGS Switch Attempts')
        ax.set_ylabel('Time (s)')
        ax.set_title('Phase Breakdown (Positive Start)')
        ax.set_xticks(sws)
        ax.grid(True, ls=':', alpha=0.6)
        ax.legend(frameon=False)

        # 3. Newton Iteration Count vs Switch Attempts
        ax = axs[2]
        for s in starts:
            sub = [r for r in records if r['start'] == s]
            sub.sort(key=lambda r: r['switch_attempts'])
            sws = [r['switch_attempts'] for r in sub]
            n_steps = [r['newton_steps'] for r in sub]
            label = 'Positive start' if s == 'cylinder_y_plus' else 'Negative start'
            ax.plot(sws, n_steps, marker=markers[s], color=colors[s], lw=2, ms=7, label=label)
        ax.set_xlabel('L-BFGS Switch Attempts')
        ax.set_ylabel('Newton Iterations to $\le 5\\times 10^{-14}$ N')
        ax.set_title('Newton Iteration Count')
        ax.grid(True, ls=':', alpha=0.6)
        ax.legend(frameon=False)

        fig.savefig(audit_dir / 'switch_sweep_analysis.png', dpi=300)
        fig.savefig(audit_dir / 'switch_sweep_analysis.pdf')
        plt.close(fig)
        print(f'Saved figures to {audit_dir / "switch_sweep_analysis.png"}')

    # Progress curves figure
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
        palette = {200: '#009E73', 500: '#E69F00', 1000: '#56B4E9', 2000: '#CC79A7'}

        for s_idx, s in enumerate(starts):
            ax = axs[s_idx]
            sub = [r for r in records if r['start'] == s]
            sub.sort(key=lambda r: r['switch_attempts'])
            for r in sub:
                sw = r['switch_attempts']
                d = traces.get(r['index'])
                if d is not None and len(d):
                    ax.plot(d[:, 0], d[:, 2], color=palette.get(sw, 'gray'), lw=1.5, label=f'Switch @ {sw}')
            ax.axhline(5e-14, color='black', ls=':', label='Target ($5\\times 10^{-14}$ N)')
            ax.set_yscale('log')
            ax.set_xlabel('Optimizer Wall Time (s)')
            ax.set_ylabel('Native Gradient Norm (N)')
            ax.set_title('Positive Start' if s == 'cylinder_y_plus' else 'Negative Start')
            ax.grid(True, ls=':', alpha=0.6)
            ax.legend(frameon=False, fontsize=9)

        fig.savefig(audit_dir / 'switch_sweep_progress.png', dpi=300)
        fig.savefig(audit_dir / 'switch_sweep_progress.pdf')
        plt.close(fig)
        print(f'Saved figures to {audit_dir / "switch_sweep_progress.png"}')


if __name__ == '__main__':
    main()
