"""Figure for the gbot-importance GP diagnostic (reads run/joint_stage3_gbot0/gp_gbot_importance.json; no refitting)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / 'run/joint_stage3_gbot0/gp_gbot_importance.json').read_text())
out = ROOT / 'docs/calibration/bayesian_calibration/figures'
out.mkdir(parents=True, exist_ok=True)
plt.style.use('/home/pxl1051/.pi/agent/skills/scientific-visualization/assets/publication.mplstyle')
LAB = {'gtop': r'$g_\mathrm{top}$', 'gbot': r'$g_\mathrm{bot}$', 'ortho_top': r'$o_\mathrm{top}$',
       'ortho_bottom': r'$o_\mathrm{bot}$', 'beta': r'$\beta$', 'floor': 'floor'}
GREY, BLUE, ORANGE = '0.55', '#0072B2', '#D55E00'

fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)

# (a) CV error by input set (basin J <= 1 mm)
sets = [('all6', 'All six'), ('no_gbot', r'No $g_\mathrm{bot}$'), ('no_obot', r'No $o_\mathrm{bot}$'),
        ('no_gbot_no_obot', r'No $g_\mathrm{bot}$, $o_\mathrm{bot}$'), ('no_beta', r'No $\beta$'), ('no_floor', 'No floor')]
m = [d['cv'][f'{k}/basin']['rmse_logJ'] for k, _ in sets]
s = [d['cv'][f'{k}/basin']['sd'] for k, _ in sets]
col = [GREY, BLUE, BLUE, BLUE, ORANGE, ORANGE]
y = np.arange(len(sets))[::-1]
ax[0, 0].barh(y, m, xerr=s, color=col, height=0.6, capsize=2)
ax[0, 0].set_yticks(y, [l for _, l in sets])
ax[0, 0].axvline(m[0], color='k', lw=0.6, ls=':')
ax[0, 0].set_xlabel(r'CV RMSE of $\ln J$')
ax[0, 0].set_title('(a) GP prediction without each input', loc='left')

# (b) local total-effect indices
names = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor']
x = np.arange(len(names)); w = 0.38
for i, (sub, lab, c) in enumerate([('all', 'All valid (n=%d)' % d['n_valid'], '0.7'),
                                   ('basin', r'$J\leq1$ mm (n=%d)' % d['n_basin_J_le_1mm'], BLUE)]):
    te = d[f'total_effect_local_{sub}']
    ax[0, 1].bar(x + (i - 0.5) * w, [te[n] for n in names], w, color=c, label=lab)
ax[0, 1].set_xticks(x, [LAB[n] for n in names])
ax[0, 1].set_ylabel('Total-effect index')
ax[0, 1].legend(frameon=False, loc='upper left')
ax[0, 1].set_title('(b) Local sensitivity near best point', loc='left')

# (c) GP profile along gbot at the best point
p = d['profile_gbot_at_best_basin']
g = np.array([q['gbot'] for q in p]); mu = np.array([q['J_mm'] for q in p])
lo = np.array([q['lo95'] for q in p]); hi = np.array([q['hi95'] for q in p])
gx = np.where(g == 0, 3e-7, g)
ax[1, 0].fill_between(gx, lo, hi, color=BLUE, alpha=0.2, lw=0, label='Mean ± 2 SD')
ax[1, 0].plot(gx, mu, 'o-', color=BLUE, ms=3, label='GP mean')
ax[1, 0].set_xscale('log')
ax[1, 0].set_xticks([3e-7, 1e-6, 1e-5, 1e-4], ['0', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$'])
ax[1, 0].set_xlabel(r'$g_\mathrm{bot}$ (others fixed at best point)')
ax[1, 0].set_ylabel('Predicted joint RMS $J$ (mm)')
ax[1, 0].legend(frameon=False, loc='upper left')
ax[1, 0].set_title(r'(c) GP profile along $g_\mathrm{bot}$', loc='left')

# (d) direct paired solves gbot -> 0
pr = d['paired_gbot0']
x = np.arange(len(pr)); w = 0.26
for i, (k, lab, c) in enumerate([('dr6', 'Run 6', ORANGE), ('dr10', 'Run 10', '#009E73'), ('dJ', 'Joint $J$', 'k')]):
    ax[1, 1].bar(x + (i - 1) * w, [q[k] * 1e3 for q in pr], w, color=c, label=lab)
ax[1, 1].axhline(0, color='k', lw=0.6)
ax[1, 1].set_xticks(x, ['A%d' % q['armA'] for q in pr])
ax[1, 1].set_xlabel(r'Arm A point re-solved with $g_\mathrm{bot}=0$')
ax[1, 1].set_ylabel(r'Change in RMS ($\mu$m)')
ax[1, 1].set_ylim(-42, 24)
ax[1, 1].legend(frameon=False, ncol=3, loc='upper center')
ax[1, 1].set_title(r'(d) Direct solves, $g_\mathrm{bot}\to0$', loc='left')

for ext in ('pdf', 'png'):
    fig.savefig(out / f'gbot_importance_gp_diagnostic.{ext}', dpi=300)
print(out / 'gbot_importance_gp_diagnostic.pdf')
