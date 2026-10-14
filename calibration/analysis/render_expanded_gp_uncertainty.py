"""Render final loss-GP uncertainty diagnostic figure from saved npz/json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'run/optimization_review/run6_expanded_bo/gp_uncertainty'
CONFIG = ROOT / 'calibration/configs/run6.json'
import hashlib
assert not any((OUT / f'gp_uncertainty.{ext}').exists() for ext in ('png', 'pdf', 'caption.md')), 'Preserve existing figures'

data = np.load(OUT / 'gp_uncertainty.npz')
report = json.loads((OUT / 'gp_uncertainty.json').read_text())
cfg = json.loads(CONFIG.read_text())
for name, expected in report['data']['input_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f'Input drift: {name}'
LABELS = [r'$g_{\mathrm{top}}$', r'$g_{\mathrm{bottom}}$', r'$o_{\mathrm{top}}$', r'$o_{\mathrm{bottom}}$']
NAMES = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom']

actual = data['observed_rms']
loo_rms = np.exp(-data['loo_log_mean'])
global_post = data['domain_posterior']
local_post = data['local_posterior']
profile_post = data['profile_posterior']
profile_x = data['profile_x']
best = report['best']


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        artists = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            labels = []
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    artists.extend([tick.label1, tick.label2])
                    if tick.label1.get_visible() and tick.label1.get_text():
                        labels.append(tick.label1)
            for i, label in enumerate(labels):
                for other in labels[i+1:]:
                    assert not label.get_window_extent(renderer).overlaps(other.get_window_extent(renderer)), 'Overlapping ticks'
        if ax.get_legend() is not None:
            artists.append(ax.get_legend())
        for artist in artists:
            if not artist.get_visible():
                continue
            bb = artist.get_window_extent(renderer)
            assert bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.x1+1 and bb.y1 <= fig.bbox.y1+1, 'Clipped text'

with plt.rc_context({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), layout='constrained')
    
    # 1. Fixed-hyperparameter LOO
    ax = axes.flat[0]
    ax.scatter(actual, loo_rms, s=16, color='#0072B2', alpha=0.75, edgecolors='none', label='147 real losses')
    lo = min(actual.min(), loo_rms.min()) * 0.8
    hi = max(actual.max(), loo_rms.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], '--', color='#333333', linewidth=0.9, label='Identity (1:1)')
    ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi),
           xlabel='Observed RMS (mm)', ylabel='LOO median RMS (mm)',
           title='Fixed-hyperparameter LOO')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=7.5, loc='upper left')

    # 2. Cumulative distribution of latent SD
    ax = axes.flat[1]
    for p, label, style in [(global_post, 'Full domain (Sobol 4096)', '-'),
                            (local_post, 'Near best (±0.1 box, 4096)', '--')]:
        ax.plot(np.sort(p[:, 3]), np.arange(1, len(p) + 1) / len(p), style, label=label, linewidth=1.2)
    ax.set(xlabel='Latent SD of log RMS', ylabel='Empirical CDF',
           title='Latent uncertainty distribution')
    ax.legend(fontsize=7.5, loc='lower right')

    # Four conditional slices; other coordinates fixed at Attempt 153.
    for dim, ax in enumerate(list(axes.flat)[2:]):
        p = profile_post[dim]
        lo_b, hi_b = cfg['parameters'][NAMES[dim]]['bounds']
        u = profile_x[dim][:, dim]
        val = np.exp(np.log(lo_b) + u * np.log(hi_b / lo_b)) if dim < 2 else lo_b + u * (hi_b - lo_b)
        ax.fill_between(val, p[:, 1], p[:, 2], color='#0072B2', alpha=0.2, label='95% latent band')
        ax.plot(val, p[:, 0], color='#0072B2', linewidth=1.2, label='GP median')
        best_p = report['data']['best_parameters'][NAMES[dim]]
        ax.plot(best_p, best['observed_rms_mm'], '*', color='#D55E00', markersize=8, label='Observed Attempt 153')
        ax.set(xlabel=LABELS[dim], ylabel='Symmetric RMS (mm)', yscale='log',
               title='Conditional profile')
        if dim < 2:
            ax.set_xscale('log')
        if dim == 0:
            ax.legend(fontsize=7, loc='upper right')

    check_layout(fig)
    fig.savefig(OUT / 'gp_uncertainty.png', dpi=180)
    fig.savefig(OUT / 'gp_uncertainty.pdf')
    plt.close(fig)

(OUT / 'gp_uncertainty.caption.md').write_text(
    'Final expanded-bounds loss-GP diagnostic. 147 real RMS losses from 160 adaptive attempts on Run 6; 13 timeouts supply no loss targets.\n'
    'LOO uses fixed hyperparameters estimated from all 147 observations, not independent validation.\n'
    'CDF: 4096 seed-2026 Sobol locations each in the full normalized domain and clipped ±0.1 box around Attempt 153.\n'
    'Profiles fix the other three parameters at 153; star is the selected best diagnostic, not a representative example.\n'
    'Lines show RMS posterior medians; shading is pointwise 95% latent-function bands transformed from Gaussian -log RMS.\n'
    'Bands condition on fitted hyperparameters and omit observation noise, hyperparameter, alignment and physics-model uncertainty.\n'
    'Growth axes use production log normalization. No GP acquisition, mechanics, target change or campaign write.\n'
    'Exploratory desktop diagnostic (11.5 × 6.8 inches), PNG preview and vector PDF. Same production Matérn5/2 ARD GP on -log RMS, seed 2026. Model estimates are not observed improvements, parameter posteriors or global/stationary-optimum certificates. Completion uncertainty is not mapped. Source: calibration/analysis/check_expanded_gp_uncertainty.py and calibration/analysis/render_expanded_gp_uncertainty.py. Numeric data, fitted hyperparameters, warnings, source/input hashes and versions: gp_uncertainty.json and gp_uncertainty.npz.\n'
)
print('RENDER_COMPLETE')
