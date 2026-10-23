"""Single-face toy: how toolpath order changes the accumulated target metric abar (top/bottom).

Uses the repo update rule (GrowthHelper::updateAbarWithMaterialGrowthIncrement, multiplicative):
  G = R diag(1+g1,1+g2) R^T ; abar <- G^T abar G  (Dm = I, so T = G; flat material frame)
and Voce hardening q(N) = qinf + (1-qinf) exp(-beta N), N = prior hits on the face (shared top/bottom).
bbar is NOT updated by the sequence (kept at its original value), so only abar_top/abar_bot depend on order.
Constants: Run 6 attempt-209 calibration. Path angles are illustrative (A=0 deg, B=ANG_B).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = dict(gtop=2.8835116e-4, gbot=1.4588e-5, ot=-0.3989158, ob=0.9603063, beta=0.2231435513)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def inc(g, o, ang):
    G = rot(ang) @ np.diag([1 + g * (1 + o), 1 + g * (1 - o)]) @ rot(ang).T
    return G


def run(order, k, ang_b, harden, iso):
    """order: string of path labels, each applied k times. returns abar_top, abar_bot histories."""
    at, ab = np.eye(2), np.eye(2)
    n = 0
    hist = [(at.copy(), ab.copy())]
    for lab in order:
        for _ in range(k):
            q = np.exp(-P['beta'] * n) if harden else 1.0
            ang = 0.0 if lab == 'A' else ang_b
            ot, ob = (0.0, 0.0) if iso else (P['ot'], P['ob'])
            Gt, Gb = inc(q * P['gtop'], ot, ang), inc(q * P['gbot'], ob, ang)
            at, ab = Gt.T @ at @ Gt, Gb.T @ ab @ Gb
            n += 1
            hist.append((at.copy(), ab.copy()))
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=6, help='repeats per path (A x k then B x k)')
    ap.add_argument('--angle-b', type=float, default=45.0)
    ap.add_argument('--out', type=Path, default=Path('plots/order_abar_toy'))
    a = ap.parse_args()
    ang = np.deg2rad(a.angle_b)
    cb = {'AB': '#0072B2', 'BA': '#D55E00'}
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), layout='constrained')

    # (a) final abar_top components, calibrated case with hardening, AB vs BA
    comp = [('a11', (0, 0)), ('a12', (0, 1)), ('a22', (1, 1))]
    fin = {o: run(o, a.k, ang, True, False)[-1] for o in ('AB', 'BA')}
    x = np.arange(3); w = 0.38
    for j, o in enumerate(('AB', 'BA')):
        v = [(fin[o][0][i] - np.eye(2)[i]) * 1e4 for _, i in comp]
        v = [fin[o][0][i] * 1e4 - (1e4 if i[0] == i[1] else 0) for _, i in comp]
        ax[0].bar(x + (j - .5) * w, v, w, color=cb[o], label=f'{o[0]} then {o[1]}')
    ax[0].set_xticks(x, ['xx (a11)', 'xy (a12)', 'yy (a22)']); ax[0].axhline(0, c='k', lw=.5)
    ax[0].set(title='(a) accumulated top-layer strain state', ylabel='Accumulated change in target metric\n(top layer, ×10⁻⁴)')
    ax[0].legend(frameon=False)

    # (b) total effective growth delivered by each path (sum of q*g_top over its k hits)
    q = np.exp(-P['beta'] * np.arange(2 * a.k)) * P['gtop'] * 1e4
    tot = {'AB': (q[:a.k].sum(), q[a.k:].sum()), 'BA': (q[a.k:].sum(), q[:a.k].sum())}  # (A, B)
    flat = a.k * P['gtop'] * 1e4
    for j, o in enumerate(('AB', 'BA')):
        ax[1].bar(np.arange(2) + (j - .5) * w, tot[o], w, color=cb[o], label=f'{o[0]} then {o[1]}')
    ax[1].axhline(flat, c='k', ls='--', lw=.8)
    ax[1].set_ylim(0, flat * 1.25); ax[1].text(1.45, flat * 1.03, 'no hardening', ha='right', fontsize=7)
    ax[1].set_xticks([0, 1], ['path A (0°)', f'path B ({a.angle_b:g}°)'])
    ax[1].set(title='(b) growth each path actually delivers', ylabel='Total effective growth from the path\n(after hardening, ×10⁻⁴)')
    ax[1].legend(frameon=False)

    # (c) final order gap, relative to the size of the accumulated change
    var = [('isotropic\nno hardening', False, True), ('orthotropic\nno hardening', False, False),
           ('isotropic\nhardening', True, True), ('orthotropic\nhardening', True, False)]
    rel = []
    for name, h, iso in var:
        x1, x2 = run('AB', a.k, ang, h, iso)[-1][0], run('BA', a.k, ang, h, iso)[-1][0]
        rel.append(max(np.linalg.norm(x1 - x2) / np.linalg.norm(x1 - np.eye(2)), 1e-16))
    ax[2].bar(range(4), rel, color=['#999999', '#009E73', '#CC79A7', '#000000'])
    ax[2].set_yscale('log'); ax[2].set_ylim(1e-12, 10)
    ax[2].set_xticks(range(4), [v[0] for v in var], fontsize=6.5)
    ax[2].set(title='(c) how much the final state depends on order', ylabel='Order sensitivity\n(fraction of total change that differs\nbetween A→B and B→A; 1 = entirely)')
    for i, r in enumerate(rel):
        ax[2].text(i, 3e-12 if r < 1e-12 else r * 1.6, '≈0' if r < 1e-12 else f'{r:.1e}', ha='center', fontsize=7)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for e in ('png', 'pdf'):
        fig.savefig(f'{a.out}.{e}', dpi=300, facecolor='white')
    for name, h, iso in var:
        h1, h2 = run('AB', a.k, ang, h, iso)[-1], run('BA', a.k, ang, h, iso)[-1]
        print(f'{name.replace(chr(10),' '):30s} final ||dabar_top||={np.linalg.norm(h1[0]-h2[0]):.3e}  ||dabar_bot||={np.linalg.norm(h1[1]-h2[1]):.3e}  ||d(top-bot)||={np.linalg.norm((h1[0]-h1[1])-(h2[0]-h2[1])):.3e}')


if __name__ == '__main__':
    main()
