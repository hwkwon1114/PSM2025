#!/usr/bin/env python3
"""Show frozen pairwise fits on all three previously seen cases; no solver runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from report_pairwise_floor1 import BASE, OUT, digest, load_pair

CASES = ('run6', 'run9', 'run10')
PAIR_LABELS = ('Run 1 + 2', 'Run 2 + 3', 'Run 1 + 3')
PAIRS = ('12', '23', '13')
STEM = 'pairwise_case_error_comparison'


def main():
    outputs = [OUT / f'{STEM}.{ext}' for ext in ('png', 'pdf', 'caption.md', 'json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Refusing to overwrite an existing case-comparison output')
    groups = [load_pair(BASE, pair) for pair in PAIRS]
    rows = []
    for group in groups:
        if group['transfer']['status'] != 'valid':
            raise ValueError(f"{group['pair']}: transfer is not valid; do not fabricate a score")
        record = json.loads((BASE / f"pair_{group['pair']}" / 'transfer_result.json').read_text())
        transfer_case = record['cases'][0]
        for case in CASES:
            excluded = case == transfer_case['id']
            metric = transfer_case['metrics'] if excluded else group['best']['case_metrics'][case]
            rows.append({'pair': group['pair'], 'case': case, 'excluded': excluded,
                         'rms_mm': float(metric['rms_symmetric_mm'] if excluded else metric['rms_mm']),
                         'scan_to_sim_p95_mm': float(metric['p95_target_to_sim_mm']),
                         'sim_to_scan_p95_mm': float(metric['p95_sim_to_target_mm'])})

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.05), sharey=True, sharex=True, layout='constrained')
    for i, (ax, case) in enumerate(zip(axes, CASES), start=1):
        for y, (pair, row) in enumerate(zip(PAIRS, [r for r in rows if r['case'] == case])):
            ax.scatter(row['rms_mm'], y, s=55, marker='o', linewidths=1.3,
                       edgecolors='#0072B2', facecolors='white' if row['excluded'] else '#0072B2', zorder=3)
            ax.scatter(row['scan_to_sim_p95_mm'], y - 0.09, s=42, marker='x', color='#333333', zorder=2)
            ax.scatter(row['sim_to_scan_p95_mm'], y + 0.09, s=42, marker='+', color='#333333', zorder=2)
        ax.set_title(f'Run {i}', fontsize=11)
        ax.set_xlabel('Surface distance (mm)')
        ax.set_yticks(range(3), PAIR_LABELS)
        ax.set_ylim(2.5, -0.5)
        ax.set_xlim(left=0)
        ax.grid(axis='x', color='#dddddd', linewidth=0.7)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [
        Line2D([], [], marker='o', color='#0072B2', markerfacecolor='#0072B2', linestyle='None', label='RMS, calibration'),
        Line2D([], [], marker='o', color='#0072B2', markerfacecolor='white', linestyle='None', label='RMS, excluded run'),
        Line2D([], [], marker='x', color='#333333', linestyle='None', label='Scan→model p95'),
        Line2D([], [], marker='+', color='#333333', linestyle='None', label='Model→scan p95'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False, fontsize=9)
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    plt.close(fig)
    outputs[2].write_text(
        '# Frozen pairwise fit errors on all three cases\n\n'
        'Each row is the minimum valid joint-RMS fit of the indicated calibration pair; each column is a public run. '
        'Filled circles show the two calibration-case symmetric surface RMS errors, and open circles show the '
        'one excluded-case exploratory transfer after parameters were frozen. Cross and plus show the 95th percentile '
        'of scan-to-model and model-to-scan unsigned distances, respectively, with small vertical offsets only for '
        'legibility. The p95 values are unweighted quantiles of the respective sampled distances; the reverse RMS '
        'uses reference-area weighting. Spatial distances are dependent. All runs were previously seen, so excluded '
        'cases are not blind validation. No error bar or '
        'independent-sample uncertainty is implied. See pairwise_provenance.json and pairwise_results.csv for identities, '
        'selection rules, candidate failures, and other observations.\n')
    outputs[3].write_text(json.dumps({'rows': rows, 'analysis_script_sha256': digest(Path(__file__)),
                                       'input_sha256': {g['pair']: g['hashes'] for g in groups}},
                                      indent=2, allow_nan=False) + '\n')
    print('\n'.join(str(p) for p in outputs))


if __name__ == '__main__':
    main()
