#!/usr/bin/env python3
"""Figure for card EDGE-ANCHOR-1 (full pass) from saved outputs of scan_edge_anchor_full.py; no recomputation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'run/diagnostics/scan_edge_anchor'
STYLE = Path('/home/pxl1051/.pi/agent/skills/scientific-visualization/assets/publication.mplstyle')
Z0 = 65.0
HALF = {'left': 100., 'right': 100., 'bottom': 150., 'top': 150.}
SIGN = {'left': -1, 'right': 1, 'bottom': -1, 'top': 1}
SIDE = {'left': ('#0072B2', 'o', 'x = −100 edge'), 'right': ('#D55E00', 's', 'x = +100 edge'),
        'bottom': ('#009E73', '^', 'z = −85 end'), 'top': ('#CC79A7', 'D', 'z = +215 end')}
SCANS = ('run6', 'run9', 'run10')


def main():
    s = json.loads((D / 'summary.json').read_text())
    rows = [r for r in json.loads((D / 'case_scores.json').read_text()) if 'error' not in r]
    with plt.style.context(str(STYLE)) if STYLE.exists() else plt.style.context('default'):
        fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.4), layout='constrained', sharey='row')
        for j, sid in enumerate(SCANS):
            sc = s['scans'][sid]; fr = sc['edge_frame']; b = np.load(D / f'boundary_{sid}.npz')
            a = ax[0, j]
            for k in ('left', 'right', 'bottom', 'top'):
                t, v = b[f'side_{k}_0'], b[f'side_{k}_1']; col, mk, lab = SIDE[k]
                sh = Z0 if k in ('left', 'right') else 0.
                a.plot(t + sh, SIGN[k] * v - HALF[k], mk, ms=1.2, c=col, alpha=.4, mew=0, rasterized=True)
                ln = fr['lines'][k]; tt = np.array([t.min(), t.max()])
                a.plot(tt + sh, SIGN[k] * (ln['intercept'] + ln['slope'] * tt) - HALF[k], c=col, lw=1.3, label=lab)
            a.axhline(0, c='k', lw=.6, ls='--')
            a.set_title(f"({'abc'[j]}) {sc['name']}", loc='left')
            a.text(.03, .04, f"centre ({fr['centre_mm'][0]:+.2f}, {fr['centre_mm'][1]:+.2f}) mm\nyaw {fr['yaw_deg']:+.3f}°",
                   transform=a.transAxes, fontsize=7, va='bottom')
            a.set_xlabel('Position along edge (mm)')
            r = [x for x in rows if x['scan'] == sid]
            cur = np.array([x['current'] for x in r])
            a2 = ax[1, j]
            jj = np.linspace(cur.min() * .9, cur.max() * 1.1, 200) if len(cur) else np.array([.5, 20])
            a2.fill_between(jj, -10 * jj, 10 * jj, color='0.9', lw=0, label='±1 % of J')
            a2.axhline(0, c='k', lw=.6)
            a2.plot(cur, 1e3 * (np.array([x['A'] for x in r]) - cur), 'o', ms=2.2, c='#0072B2', alpha=.7, mew=0,
                    label='A: scan edge-anchored')
            a2.plot(cur, 1e3 * (np.array([x['B'] for x in r]) - cur), 'x', ms=2.6, c='#D55E00', alpha=.7, mew=.6,
                    label='B: both edge-anchored')
            g = s['gates'][sid]
            a2.set_title(f"({'def'[j]}) n = {g['n']}", loc='left')
            a2.set(xscale='log', xlabel='Current symmetric RMS (mm)')
        ax[0, 0].set_ylabel('Outward edge offset (mm)'); ax[1, 0].set_ylabel('Change in RMS (µm)')
        allv = np.concatenate([1e3 * (np.array([x[v] for x in rows]) - np.array([x['current'] for x in rows])) for v in 'AB'])
        lo, hi = float(allv.min()), float(allv.max())   # no clipping
        pad = .15 * (hi - lo); ax[1, 0].set_ylim(lo - pad, hi + pad)
        h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='outside upper center', ncol=4, frameon=False, fontsize=7)
        ax[1, 2].legend(fontsize=6.5, frameon=False, loc='upper right')
        for ext in ('pdf', 'png'):
            fig.savefig(D / f'scan_edge_anchor_three_runs.{ext}', dpi=300)
    print('wrote', D / 'scan_edge_anchor_three_runs.pdf')


if __name__ == '__main__':
    main()
