#!/usr/bin/env python3
"""Figure for the EDGE-ANCHOR-1 pilot from saved outputs of scan_edge_anchor_check.py (no recomputation)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'run/diagnostics/scan_edge_anchor_pilot'
STYLE = Path('/home/pxl1051/.pi/agent/skills/scientific-visualization/assets/publication.mplstyle')
Z0 = 65.0                                   # public z = internal v + 65 mm
HALF = {'left': 100., 'right': 100., 'bottom': 150., 'top': 150.}
SIGN = {'left': -1, 'right': 1, 'bottom': -1, 'top': 1}
SIDE_STYLE = {'left': ('#0072B2', 'o'), 'right': ('#D55E00', 's'), 'bottom': ('#009E73', '^'), 'top': ('#CC79A7', 'D')}
SIDE_NAME = {'left': 'x = −100 edge', 'right': 'x = +100 edge', 'bottom': 'z = −85 end', 'top': 'z = +215 end'}


def main():
    s = json.loads((D / 'summary.json').read_text())
    b = np.load(D / 'scan_boundary.npz')
    fr = s['scan_edge_frame']
    with plt.style.context(str(STYLE)) if STYLE.exists() else plt.style.context('default'):
        fig = plt.figure(figsize=(7.2, 6.4), layout='constrained')
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.0])
        ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1:]); ax_c = fig.add_subplot(gs[1, 1])
        ax_l = fig.add_subplot(gs[1, 0]); ax_l.axis('off')
        ax_d = fig.add_subplot(gs[1, 2])

        # (a) plan view of the raw scan outline, nominal sheet and fitted edges
        o = b['outline_sample']
        ax_a.scatter(o[:, 0], o[:, 1] + Z0, s=0.2, c='0.75', lw=0, rasterized=True, label='Raw scan (subsample)')
        ax_a.plot([-100, 100, 100, -100, -100], np.array([-150, -150, 150, 150, -150]) + Z0, 'k--', lw=0.8,
                  label='Nominal sheet, current frame')
        ax_a.plot([-98, 98, 98, -98, -98], np.array([-148, -148, 148, 148, -148]) + Z0, ':', c='0.35', lw=0.8,
                  label='Scored scan domain')
        for k, ln in fr['lines'].items():
            col = SIDE_STYLE[k][0]
            if k in ('left', 'right'):
                y = np.array([-150, 150]); ax_a.plot(ln['intercept'] + ln['slope'] * y, y + Z0, c=col, lw=1.2)
            else:
                x = np.array([-100, 100]); ax_a.plot(x, ln['intercept'] + ln['slope'] * x + Z0, c=col, lw=1.2)
        cx, cy = s['current_scan_anchor']['centre_mm']
        ax_a.plot(cx, cy + Z0, 'k+', ms=8,
                  label='Current anchor')
        ax_a.plot(fr['centre_mm'][0], fr['centre_mm'][1] + Z0, 'x', c='#D55E00', ms=7, label='Edge-derived centre')
        ax_a.set(xlabel='x (mm)', ylabel='z (mm)', aspect='equal', xlim=(-112, 112), ylim=(-97, 227))
        ax_l.legend(*ax_a.get_legend_handles_labels(), loc='upper left', fontsize=7, frameon=False, markerscale=1.0,
                    title='(a) legend', title_fontsize=7, alignment='left')
        ax_a.set_title('(a) Scan outline', loc='left')

        # (b) outward offset of each raw edge from the nominal edge of the current frame
        for k in ('left', 'right', 'bottom', 'top'):
            t, v = b[f'side_{k}_0'], b[f'side_{k}_1']
            along = t + Z0 if k in ('left', 'right') else t
            off = SIGN[k] * v - HALF[k]
            col, mk = SIDE_STYLE[k]
            ax_b.plot(along, off, mk, ms=1.6, c=col, alpha=.45, mew=0, rasterized=True)
            ln = fr['lines'][k]; tt = np.array([t.min(), t.max()])
            ax_b.plot(tt + (Z0 if k in ('left', 'right') else 0), SIGN[k] * (ln['intercept'] + ln['slope'] * tt) - HALF[k],
                      c=col, lw=1.4, label=SIDE_NAME[k])
        ax_b.axhline(0, c='k', lw=.6, ls='--')
        ax_b.set(xlabel='Position along edge (mm; x for ends, z for sides)', ylabel='Outward edge offset (mm)')
        ax_b.legend(ncol=2, fontsize=7, frameon=False, loc='lower center')
        ax_b.set_title('(b) Raw edge position vs nominal sheet', loc='left')

        # (c) paired score change
        res = s['results']
        cur = np.array([r['current']['rms_symmetric_mm'] for r in res])
        dA = np.array([r['A']['rms_symmetric_mm'] for r in res]) - cur
        dB = np.array([r['B']['rms_symmetric_mm'] for r in res]) - cur
        ax_c.axhline(0, c='k', lw=.6)
        ax_c.plot(cur, 1e3 * dA, 'o', c='#0072B2', ms=4, label='A: scan edge-anchored')
        ax_c.plot(cur, 1e3 * dB, 'x', c='#D55E00', ms=5, label='B: both edge-anchored')
        for r, x, y in zip(res, cur, dA):
            if r['label'].startswith('Attempt') or r['label'].endswith('a0014'):
                ax_c.annotate('Att. 209' if r['label'].startswith('Attempt') else 'a0014', (x, 1e3 * y),
                              xytext=(16, -14) if r['label'].startswith('Attempt') else (16, 4),
                              textcoords='offset points', fontsize=7, arrowprops=dict(arrowstyle='-', lw=.5, color='0.3'))
        ax_c.set(xscale='log', xlabel='Current symmetric RMS (mm)', ylabel='Change in RMS (µm)')
        ax_c.legend(fontsize=7, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.62))
        ax_c.set_title('(c) Score change, saved Run 1 fits', loc='left')

        # (d) pointwise change in scan-to-model distance, Attempt 209
        p = np.load(D / 'pointwise_corrected.npz')
        delta = 1e3 * (p['d_A'] - p['d_current'])
        lim = float(np.max(np.abs(delta)))
        im = ax_d.scatter(p['xy_current'][:, 0], p['xy_current'][:, 1] + Z0, c=delta, s=0.6, lw=0, cmap='RdBu_r',
                          norm=TwoSlopeNorm(0, -lim, lim), rasterized=True)
        ax_d.set(xlabel='x (mm)', ylabel='z (mm)', aspect='equal', xlim=(-100, 100), ylim=(-85, 215))
        fig.colorbar(im, ax=ax_d, label='Δ scan-to-model distance (µm)', shrink=.9)
        ax_d.set_title('(d) A − current, Att. 209', loc='left')

        for ext in ('pdf', 'png'):
            fig.savefig(D / f'scan_edge_anchor_comparison.{ext}', dpi=300)
    n = len(res)
    (D / 'scan_edge_anchor_comparison.caption.md').write_text(
        'Run 1 scan placement from physical sheet edges versus the current scan anchor (pilot for card EDGE-ANCHOR-1). '
        '(a) Plan view (public x–z) of a 60,000-point random subsample of the raw scan component, the nominal 200×300 mm '
        'sheet centred on the current anchor (dashed), the scored 196×296 mm scan domain (dotted), Theil–Sen lines fitted '
        'to each raw edge over its central 80% (corners excluded), the current anchor (+) and the edge-derived centre (×). '
        f"Edge-derived centre offset: Δx = {fr['centre_mm'][0]:+.3f} mm, Δz = {fr['centre_mm'][1]:+.3f} mm; yaw "
        f"{fr['yaw_deg']:+.3f}° (long sides {fr['yaw_long_deg']:+.3f}°, ends {fr['yaw_short_deg']:+.3f}°); projected "
        f"edge-to-edge width {fr['width_mm']:.2f} mm and length {fr['length_mm']:.2f} mm. "
        '(b) Outward offset of the raw boundary (5th-most-extreme point per 1 mm strip) from the nominal edge, with the '
        'fitted lines; negative = inside the nominal sheet. A common inward offset on opposite edges reflects projection '
        'foreshortening of the curved sheet and the boundary-sampling rank, not a shift; half the difference between '
        'opposite edges is the centre shift, the slope is yaw. '
        f'(c) Change in double-sided symmetric RMS for {n} saved valid Run 1 simulations (Attempt 209 corrected-convention '
        'forward solve plus all valid run1_clean evaluations to date), A: scan anchored at the edge-derived centre and yaw, '
        'simulation in its unchanged material frame; B: simulation additionally anchored by the same edge functional '
        'applied to its exact boundary vertices. The scored point set (all 14,504 scan points, fail-closed footprint) is '
        'identical in all variants. Unchanged-anchor rescoring reproduced every stored objective. '
        '(d) Per-point change in scan-to-model distance for Attempt 209 under A, at scan positions. '
        'Single scan, solver-free rescoring of saved meshes; the edge fit depends on the boundary sampling rule and the '
        'central-80% side window. Generated by calibration/analysis/scan_edge_anchor_check.py and plot_scan_edge_anchor.py.\n')
    print('wrote', D / 'scan_edge_anchor_comparison.pdf')


if __name__ == '__main__':
    main()
