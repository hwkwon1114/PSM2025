#!/usr/bin/env python3
"""Rank the 32 screening attempts (4 mm, retained states) by the global surface-fit J (degree D) and plot them.
Usage: rank_deformations.py OUT_DIR [DEGREE=8]"""
import glob
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import sag_objective as so          # noqa: E402

CAMP = ROOT / 'run/dome_square/campaign'


def main(out_dir, d=8):
    led = {r['attempt']: r for r in map(json.loads, (CAMP / 'ledger.jsonl').read_text().splitlines())}
    rows = []
    for a in sorted(led):
        if led[a]['state'] != 'accepted':
            continue
        sp = glob.glob(str(CAMP / f'attempts/attempt_{a:03d}/solve/*native_hybrid_state.json'))
        pos, uv, faces, av, resid, m = so.fit_state(sp[0], d)
        A = np.c_[uv, np.ones(len(uv))]
        z = pos[:, 2] - A @ np.linalg.lstsq(A, pos[:, 2], rcond=None)[0]
        rows.append(dict(a=a, j_fit=m['j_combined'], gate_fit=m['largest_connected_fraction'], tot_fit=m['total_coherent_fraction'],
                         j_old=led[a]['metrics']['j_combined'], gate_old=led[a]['largest_connected_fraction'],
                         rms=float(np.sqrt((av[:, None] * resid ** 2).sum() / av.sum() / 3) * 1e3),
                         height=float(np.ptp(z) * 1e3), uv=uv * 1e3, faces=faces, z=z * 1e3))
    rows.sort(key=lambda r: -r['j_fit'])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rho = spearmanr([r['j_fit'] for r in rows], [r['j_old'] for r in rows])[0]
    print(f'n={len(rows)} Spearman(J_fit, J_15mm)={rho:.2f}')
    print('rank attempt J_fit J_15mm gate_fit gate_15mm height_mm fit_rms_mm')
    for i, r in enumerate(rows, 1):
        print(i, r['a'], '%.3f %.3f %.2f %.2f %.2f %.4f' % (r['j_fit'], r['j_old'], r['gate_fit'], r['gate_old'], r['height'], r['rms']))
    allz = np.concatenate([r['z'] for r in rows])
    lo, hi = np.percentile(allz, [1, 99])
    n = len(rows); cols = 8; nr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(2.3 * cols, 2.5 * nr), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (i, r) in zip(axes.ravel(), enumerate(rows, 1)):
        pc = PolyCollection(r['uv'][r['faces']][:, :, :2], array=r['z'][r['faces']].mean(1), cmap='viridis', edgecolors='none')
        pc.set_clim(lo, hi); ax.add_collection(pc)
        ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_aspect('equal'); ax.axis('on')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'#{i} att {r["a"]}\nJ {r["j_fit"]:.3f} (15mm {r["j_old"]:.3f})', fontsize=7)
    fig.colorbar(pc, ax=axes, shrink=0.4, label='Height, plane removed (mm)')
    fig.savefig(out / 'attempts_ranked_by_surface_fit_J.png', dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    ax[0].scatter([r['j_old'] for r in rows], [r['j_fit'] for r in rows], s=18)
    for r in rows[:6]:
        ax[0].annotate(str(r['a']), (r['j_old'], r['j_fit']), fontsize=7)
    ax[0].set_xlabel('J, 15 mm estimator'); ax[0].set_ylabel(f'J, surface fit d={d}')
    ax[1].scatter([r['height'] for r in rows], [r['j_fit'] for r in rows], s=18)
    ax[1].set_xlabel('Height range (mm)'); ax[1].set_ylabel(f'J, surface fit d={d}')
    fig.savefig(out / 'surface_fit_J_scatter.png', dpi=120)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
