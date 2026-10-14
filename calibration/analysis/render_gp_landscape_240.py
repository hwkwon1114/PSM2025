"""Render saved GP posterior arrays only; never fit a model or change campaign data."""
import ast
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'run/optimization_review'
assert os.environ.get('SLURM_JOB_ID')
r = json.loads((OUT / 'gp_landscape_240.json').read_text())
assert r['status'] == 'complete' and len(r['training']) == 13
assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in r['input_sha256'].items())
assert not any((OUT / f'gp_landscape_240_{stem}.{ext}').exists() for stem in ('pairs', 'profiles') for ext in ('png', 'pdf'))
assert not (OUT / 'gp_landscape_240.caption.md').exists()
KEYS = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom']
LABELS = [r'$g_{\mathrm{top}}$', r'$g_{\mathrm{bottom}}$', r'$o_{\mathrm{top}}$', r'$o_{\mathrm{bottom}}$']


def growth_axis(axis, dim):
    ticks = [1e-3, 3e-3, 1e-2] if dim == 0 else [1e-5, 1e-4, 3e-4]
    def label(value, position):
        exponent = int(np.floor(np.log10(value)))
        coefficient = int(round(value / 10.**exponent))
        return rf'$10^{{{exponent}}}$' if coefficient == 1 else rf'${coefficient}\times10^{{{exponent}}}$'
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(label))
    axis.set_minor_formatter(NullFormatter())


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for ax in fig.axes:
        artists = [ax.xaxis.label, ax.yaxis.label, ax.title]
        # Matplotlib generates extra log ticks outside view; these are not drawn.
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    artists.extend([tick.label1, tick.label2])
            labels = [t.label1 for t in axis.get_major_ticks()
                      if min(limits) <= t.get_loc() <= max(limits) and t.label1.get_visible() and t.label1.get_text()]
            for i, label in enumerate(labels):
                for other in labels[i + 1:]:
                    assert not label.get_window_extent(renderer).overlaps(other.get_window_extent(renderer)), 'Overlapping tick labels'
        for a in artists:
            if not a.get_visible() or not a.get_text(): continue
            b = a.get_window_extent(renderer)
            assert b.x0 >= box.x0 - 2 and b.y0 >= box.y0 - 2 and b.x1 <= box.x1 + 2 and b.y1 <= box.y1 + 2, ('Clipped text', a.get_text())
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'gp_landscape_240_{stem}.{ext}', dpi=180, bbox_inches=None)
    plt.close(fig)


with plt.rc_context({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
    pairs, profiles, best = r['pairs'], r['profiles'], r['best']
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout='constrained')
    vmax = max(np.max(p['median_mm']) for p in pairs)
    vmin = min(np.min(p['median_mm']) for p in pairs)
    smax = max(np.max(p['sd_log_rms']) for p in pairs)
    for n, p in enumerate(pairs):
        i, j = p['dims']; row, col = divmod(n, 2)
        for k, (field, title, cmap, low, high) in enumerate([('median_mm', 'Median RMS', 'viridis_r', vmin, vmax),
                                                             ('sd_log_rms', 'Latent SD', 'cividis', 0, smax)]):
            ax = axes[row, col * 2 + k]
            z = ax.pcolormesh(p['x'], p['y'], p[field], cmap=cmap, vmin=low, vmax=high, shading='auto', rasterized=True)
            ax.plot(best['parameters'][KEYS[i]], best['parameters'][KEYS[j]], '*', color='#D55E00', ms=9, mec='black', mew=.5, clip_on=False)
            ax.set(xlabel=LABELS[i], ylabel=LABELS[j], title=title,
                   xlim=(p['x'][0], p['x'][-1]), ylim=(p['y'][0], p['y'][-1]))
            if i < 2:
                ax.set_xscale('log'); growth_axis(ax.xaxis, i)
            if j < 2:
                ax.set_yscale('log'); growth_axis(ax.yaxis, j)
            cb = fig.colorbar(z, ax=ax, fraction=.046, pad=.03)
            cb.set_label('RMS (mm)' if k == 0 else 'SD of log RMS')
    save(fig, 'pairs')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    for ax, p in zip(axes.flat, profiles):
        d = p['dim']
        ax.fill_between(p['x'], p['lower95_mm'], p['upper95_mm'], color='#0072B2', alpha=.18, label='95% latent band')
        ax.plot(p['x'], p['median_mm'], color='#0072B2', label='GP median')
        ax.plot(best['parameters'][KEYS[d]], best['objective_mm'], '*', color='#D55E00', ms=10, label='Attempt 25', clip_on=False)
        ax.plot([o['parameters'][KEYS[d]] for o in r['training']], [.025] * len(r['training']), '|', color='.35',
                transform=ax.get_xaxis_transform(), label='Training projections')
        ax.set(xlabel=LABELS[d], ylabel='Symmetric RMS (mm)', yscale='log', xlim=(p['x'][0],p['x'][-1]))
        if d < 2:
            ax.set_xscale('log'); growth_axis(ax.xaxis, d)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'profiles')

# Preserve the already inspected caption literal from the exact fit-generating source.
source = ROOT / 'calibration/analysis/plot_gp_landscape_240.py'
node = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.value.id == 'caption_path' and n.func.attr == 'write_text')
caption = ast.literal_eval(node.args[0])
(OUT / 'gp_landscape_240.caption.md').write_text(caption + '\nRendering: calibration/analysis/render_gp_landscape_240.py from saved posterior arrays, without refitting. Original fit job9536391 completed numerical predictions but its overly broad off-view-tick extent check stopped before exports; renderer now checks only actually drawn ticks. Both generating sources and logs preserved.\n')
print('RENDER_COMPLETE; no model fitting', flush=True)
