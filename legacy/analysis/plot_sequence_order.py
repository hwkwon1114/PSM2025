#!/usr/bin/env python3
"""Matched all-case figures; missing cases stay visible. No simulation or selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from analyze_two_gripper_release import align, area_weights, digest

LABELS = {'a_only': 'A × 10', 'b_only': 'B × 10', 'ab_block': 'A × 10 → B × 10',
          'ba_block': 'B × 10 → A × 10', 'ab_alternate': '(A → B) × 10',
          'ba_alternate': '(B → A) × 10'}


def render(source, output):
    metrics = json.loads((source / 'metrics.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    inputs = {str(source / 'metrics.json'): digest(source / 'metrics.json')}
    fields, pairs = {}, {}
    for name in LABELS:
        for mode in ('every_cycle', 'final_only'):
            key = f'{name}_{mode}_base'
            path = source / (key + '.npz')
            if path.exists():
                inputs[str(path)] = digest(path)
                with np.load(path) as loaded:
                    fields[key] = {k: loaded[k] for k in loaded.files}
        path = source / ('cadence_' + name + '.npz')
        if path.exists():
            inputs[str(path)] = digest(path)
            with np.load(path) as loaded:
                pairs[name] = {k: loaded[k] for k in loaded.files}
    fixture = any(c.get('execution', {}).get('fixture') for c in metrics['cases'].values())
    labels = {k: (v.replace('10', '2') + ' [fixture]' if fixture else v) for k, v in LABELS.items()}
    style = {'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
             'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42}

    def missing(ax):
        ax.text(.5, .5, 'Not available', ha='center', va='center', transform=ax.transAxes)

    def save(fig, stem, caption):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Batch export guard: page clipping. Visual inspection still required.
        for ax in fig.axes:
            ticks = [t for t, x in zip(ax.get_xticklabels(), ax.get_xticks()) if min(ax.get_xlim()) <= x <= max(ax.get_xlim())]
            ticks += [t for t, y in zip(ax.get_yticklabels(), ax.get_yticks()) if min(ax.get_ylim()) <= y <= max(ax.get_ylim())]
            for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts, *ticks]:
                if not text.get_visible() or not text.get_text():
                    continue
                b = text.get_window_extent(renderer)
                if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                    raise ValueError('Text clipped: ' + text.get_text())
        for ext in ('png', 'pdf'):
            fig.savefig(output / (stem + '.' + ext), dpi=250)
        (output / (stem + '.caption.md')).write_text(caption + '\n')
        plt.close(fig)

    common = ('All six prespecified schedules shown, including missing/failed cases as gaps. '
              'Each is a deterministic model case; vertices are spatially dependent, not replicates. '
              'No confidence intervals or global-optimality claims. ')
    if fixture:
        common = 'CORRECTNESS FIXTURE ONLY: two applications per toolpath, not scientific evidence. ' + common
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), layout='constrained', sharex=True, sharey=True)
        for ax, (name, label) in zip(axes.flat, labels.items()):
            ax.set_title(label)
            if name in pairs:
                p = pairs[name]
                order = np.argsort(p['distance_mm'])
                ax.step(np.r_[0, p['distance_mm'][order]], np.r_[0, np.cumsum(p['weights'][order])],
                        where='post', color='#0072B2')
            else:
                missing(ax)
            ax.set(xlabel='Distance (mm)', ylabel='Cumulative area fraction', ylim=(0, 1.02))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        save(fig, 'sequence_cadence_distributions', 'Full-sheet cadence differences. ' + common +
             'Proper reference-area-weighted rigid alignment in fixed material correspondence; no reflection or scaling. '
             'Distances are lumped versus sequential endpoints, weighted by one third of adjacent reference-face areas.')

        maximum = max((np.max(abs(p['delta_mm'][:, 2])) for p in pairs.values()), default=0)
        norm = Normalize(-maximum if maximum else -1, maximum if maximum else 1)
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.3), layout='constrained')
        artist = None
        for ax, (name, label) in zip(axes.flat, labels.items()):
            ax.set_title(label)
            key = name + '_every_cycle_base'
            if name in pairs and key in fields:
                f, p = fields[key], pairs[name]
                artist = ax.tripcolor(f['uv'][:, 0]*1000, f['uv'][:, 1]*1000, f['faces'],
                                      p['delta_mm'][:, 2], shading='gouraud', cmap='RdBu_r', norm=norm,
                                      rasterized=True)
            else:
                missing(ax)
            ax.set(xlabel='Material u (mm)', ylabel='Material v (mm)', aspect='equal')
        if artist is not None:
            fig.colorbar(artist, ax=axes, label='Aligned Δz (mm)', shrink=.7)
        save(fig, 'sequence_cadence_signed_maps', 'Signed cadence differences. ' + common +
             'Lumped minus sequential z component after proper rigid alignment to the sequential endpoint; '
             'z is that endpoint’s solver-frame axis, not a local surface normal. Shared symmetric limits; '
             'vertex fields interpolated within triangles for display only. The CDF retains full vector distances.')

        for kind in ('height', 'curvature'):
            values = {}
            for key, f in fields.items():
                if kind == 'height':
                    reference = np.column_stack((f['uv'], np.zeros(len(f['uv']))))
                    weights = area_weights(reference, f['faces'])
                    aligned, _, _ = align(f['positions'], reference, weights)
                    values[key] = aligned[:, 2]*1000
                else:
                    values[key] = np.ma.array(f['geometric_principal_m_inverse'].mean(axis=1),
                                              mask=~f['geometric_valid'])
            maximum = max((float(np.max(abs(v))) for v in values.values() if np.ma.count(v)), default=0)
            norm = Normalize(-maximum if maximum else -1, maximum if maximum else 1)
            fig, axes = plt.subplots(6, 2, figsize=(6.5, 13), layout='constrained')
            artist = None
            for row, (name, label) in enumerate(labels.items()):
                for col, mode in enumerate(('every_cycle', 'final_only')):
                    ax = axes[row, col]
                    ax.set_title(label + (' · Sequential' if col == 0 else ' · Lumped'))
                    key = f'{name}_{mode}_base'
                    if key in fields:
                        f = fields[key]
                        args = {'facecolors': values[key]} if kind == 'curvature' else {}
                        vertex = [] if kind == 'curvature' else [values[key]]
                        artist = ax.tripcolor(f['uv'][:, 0]*1000, f['uv'][:, 1]*1000, f['faces'], *vertex,
                                              **args, cmap='RdBu_r', norm=norm, rasterized=True)
                    else:
                        missing(ax)
                    ax.set(xlabel='Material u (mm)', ylabel='Material v (mm)', aspect='equal')
            if artist is not None:
                fig.colorbar(artist, ax=axes, shrink=.3, label='Height (mm)' if kind == 'height' else 'Geometric mean curvature (m⁻¹)')
            detail = ('Each endpoint independently rigid-aligned to the flat material-coordinate sheet; no scaling or reflection. '
                      'Color encodes height, not a deformation exaggeration.' if kind == 'height' else
                      'Mean of geometric principal curvatures from the frozen15mm material-neighborhood fit. '
                      'Invalid faces masked; valid-area fractions in metrics.json. Not director-based curvature or a stability certificate.')
            save(fig, 'sequence_endpoint_' + kind, 'Matched endpoint fields. ' + common + detail + ' Shared limits across all accepted panels.')
    (output / 'provenance.json').write_text(json.dumps(dict(input_sha256=inputs,
        script_sha256=digest(Path(__file__)), matplotlib=matplotlib.__version__, numpy=np.__version__,
        fixture=fixture, rendered_inspection='pending', color_vision_simulation='not performed'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.analysis, args.output)
