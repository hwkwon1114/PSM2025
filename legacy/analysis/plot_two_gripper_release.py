#!/usr/bin/env python3
"""Full-field paired-release differences; no selected vertices or fitted uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.text import Text
import numpy as np

# Stable run identities; panels also identify each run in text.
STYLES = {'run6': {'color': '#0072B2', 'linestyle': '-'},
          'run9': {'color': '#D55E00', 'linestyle': '--'}}
# Adapted from scientific-visualization/assets/publication.mplstyle.
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
         'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'axes.spines.top': False, 'axes.spines.right': False,
         'axes.linewidth': .6, 'legend.fontsize': 7, 'legend.frameon': False,
         'pdf.fonttype': 42, 'savefig.facecolor': 'white'}


def audit_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    # Locators retain labels beyond the view limits; Axis.draw does not draw them.
    undrawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < low - 1e-12 or tick.get_loc() > high + 1e-12:
                    undrawn.update((tick.label1, tick.label2))
    for text in fig.findobj(Text):
        if text in undrawn or not text.get_visible() or not text.get_text().strip():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < box.x0 - 1 or b.y0 < box.y0 - 1 or b.x1 > box.x1 + 1 or b.y1 > box.y1 + 1:
            raise ValueError('Text outside figure: ' + text.get_text())
    for ax in fig.axes:
        for text in ax.texts:
            b = text.get_window_extent(renderer)
            a = ax.get_window_extent(renderer)
            if b.x0 < a.x0 - 1 or b.y0 < a.y0 - 1 or b.x1 > a.x1 + 1 or b.y1 > a.y1 + 1:
                raise ValueError('Annotation outside axes: ' + text.get_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New figure directory')
    args = parser.parse_args()
    metrics_path = args.input / 'paired_release_metrics.json'
    report = json.loads(metrics_path.read_text())
    cases = {c['case']: c for c in report['cases']}
    if set(cases) != {'run6', 'run9'}:
        raise ValueError('Both complete production cases required; no partial figure')
    arrays = {}
    paths = [metrics_path]
    for name in ('run6', 'run9'):
        path = args.input / (name + '_full_field.npz')
        paths.append(path)
        with np.load(path) as data:
            arrays[name] = {k: data[k].copy() for k in ('distance_mm', 'weights', 'material_uv_mm')}
        a = arrays[name]
        if not all(np.isfinite(v).all() for v in a.values()) or np.any(a['distance_mm'] < 0):
            raise ValueError('Invalid field')
        if not np.isclose(a['weights'].sum(), 1) or np.any(a['weights'] <= 0):
            raise ValueError('Invalid reference weights')
        if len(a['distance_mm']) != cases[name]['vertices']:
            raise ValueError('Vertex-count mismatch')
    targets = [args.output / ('two_gripper_release_comparison.' + suffix)
               for suffix in ('png', 'pdf', 'caption.md', 'provenance.json')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Figure target already exists; refusing overwrite')
    args.output.mkdir(parents=True, exist_ok=True)
    maximum = max(a['distance_mm'].max() for a in arrays.values()) * 1000
    norm = Normalize(0, max(maximum, 1e-12))
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), constrained_layout=True)
        for col, name in enumerate(('run6', 'run9')):
            a, metrics = arrays[name], cases[name]
            distance = a['distance_mm'] * 1000
            order = np.argsort(distance)
            ax = axes[0, col]
            ax.step(np.r_[0, distance[order]], np.r_[0, np.cumsum(a['weights'][order])],
                    where='post', **STYLES[name])
            ax.set(xlim=(0, norm.vmax * 1.04), ylim=(0, 1.02),
                   xlabel='Aligned displacement (µm)', ylabel='Cumulative reference area')
            ax.set_title('(%s) Run %s distribution' % ('ab'[col], name[-1]), loc='left')
            ax.text(.97, .08, 'RMS: %.3g µm\nMax: %.3g µm' %
                    (metrics['area_weighted_rms_difference_mm'] * 1000, distance.max()),
                    transform=ax.transAxes, ha='right', va='bottom')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
            ax = axes[1, col]
            uv = a['material_uv_mm']
            field = ax.scatter(uv[:, 0], uv[:, 1], c=distance, norm=norm, cmap='cividis',
                               marker='s', s=2, linewidths=0, rasterized=True)
            ax.set(xlim=(-100, 100), ylim=(-150, 150), xlabel='Material x (mm)', ylabel='Material y (mm)')
            ax.set_aspect('equal')
            ax.set_title('(%s) Run %s displacement' % ('cd'[col], name[-1]), loc='left')
        bar = fig.colorbar(field, ax=list(axes[1]), shrink=.85, pad=.03)
        bar.set_label('Aligned displacement (µm)')
        audit_layout(fig)
        for suffix in ('png', 'pdf'):
            fig.savefig(args.output / ('two_gripper_release_comparison.' + suffix), dpi=300)
        plt.close(fig)
    caption = ('Paired-release shape differences for the researcher-specified two-gripper setup. '
               '(a,b) Cumulative reference-area distributions and area-weighted RMS/maxima; '
               '(c,d) displacement magnitude at every mesh vertex in material coordinates, with a shared color scale. '
               'Putong release is aligned to free release by one proper area-weighted rigid transformation '
               'using identical material vertices, without reflection, rescaling, smoothing or vertex exclusion. '
               'Both releases start from each run\'s identical clamped state after 29/33 forming repeats. '
               'Distances use full-precision checkpoint coordinates, not reduced-precision VTP files. '
               'Two deterministic cases, not independent spatial replicates; no confidence interval or physical '
               'generalization is implied. Blue solid: Run 6; vermilion dashed: Run 9.\n')
    (args.output / 'two_gripper_release_comparison.caption.md').write_text(caption)
    provenance = {'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [Path(__file__)]},
                  'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'size_inches': [7.2, 6.5],
                  'dpi': 300, 'styles': STYLES, 'colormap': 'cividis', 'field_units': 'micrometres',
                  'color_limits': [0, norm.vmax], 'subsampling': 'none', 'uncertainty': 'none',
                  'layout_check': 'all text within figure; annotations within axes',
                  'visual_review': 'pending; PNG and PDF must both be inspected'}
    (args.output / 'two_gripper_release_comparison.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
