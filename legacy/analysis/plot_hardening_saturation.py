"""Voce-decay hardening saturation: per-hit effectiveness q(N)=exp(-beta N) and cumulative growth fraction vs hit number on a face."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
beta, gtop = 0.2231435513, 2.8835116e-4   # run6 attempt-209 / run10 recipe
n = np.arange(1, 51)
q = np.exp(-beta * (n - 1))
cum = np.cumsum(q) / (1 / (1 - np.exp(-beta)))
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'pdf.fonttype': 42, 'axes.linewidth': .6})
fig, ax = plt.subplots(1, 2, figsize=(8, 3.3), layout='constrained')
ax[0].semilogy(n, q, 'o-', ms=3, lw=1, c='#0072B2')
for t, c in ((.1, '#D55E00'), (.01, '#009E73')):
    k = int(np.argmax(q < t)) + 1
    ax[0].axhline(t, c=c, ls=':', lw=.8); ax[0].plot(k, q[k - 1], 'o', c=c)
    ax[0].annotate(f'q<{t:g} at hit {k}', (k, q[k - 1]), xytext=(k + 4, q[k - 1] * 3), fontsize=7, color=c)
ax[0].set(title='(a) effectiveness of each hit, q = 0.8$^{\mathrm{hit}-1}$', xlabel='hit number on the face', ylabel='q', ylim=(1e-5, 2))
ax[1].plot(n, cum * 100, c='#0072B2', lw=1.2)
for f in (.5, .9, .99):
    k = int(np.argmax(cum >= f)) + 1
    ax[1].plot(k, f * 100, 'o', c='k', ms=3); ax[1].annotate(f'{f*100:.0f}% at hit {k}', (k, f * 100), xytext=(k + 2, f * 100 - 8), fontsize=7)
ax[1].axvline(50, c='gray', ls='--', lw=.7); ax[1].text(49, 20, 'Run 10: 50 repeats', rotation=90, ha='right', fontsize=7, color='gray')
ax[1].set(title='(b) cumulative growth, % of saturation 5g', xlabel='hit number on the face', ylabel='% of saturation', ylim=(0, 105))
fig.suptitle(f'Voce decay, beta=0.2231, floor=0 (saturation = 5·g_top = {5*gtop:.2e})', fontsize=8)
for e in ('png', 'pdf'): fig.savefig(f'plots/hardening_saturation.{e}', dpi=300, facecolor='white')
