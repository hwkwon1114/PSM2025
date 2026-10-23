#!/usr/bin/env python3
"""Full-field forming/release contrasts and matched phase views; no solver."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from analyze_two_gripper_release import digest
from plot_two_gripper_stages import load_case, setup_panel, shape_panel, STYLE


def save(fig, output, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_visible() and t.get_text()]
        for text in texts:
            bb = text.get_window_extent(renderer)
            if not (ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)):
                raise ValueError('Text outside panel: ' + text.get_text())
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                    raise ValueError('Overlapping panel annotations')
    for ext in ('png', 'pdf'):
        fig.savefig(output / (stem + '.' + ext), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    metrics_path = args.analysis / 'forming_restraint_metrics.json'
    metrics = json.loads(metrics_path.read_text())
    sources[str(metrics_path)] = digest(metrics_path)
    data = {}
    cases = {}
    for name in ('run6', 'run9'):
        path = args.analysis / (name + '_full_field.npz')
        sources[str(path)] = digest(path)
        with np.load(path) as archive:
            data[name] = dict(archive)
        cases[name] = load_case(name, sources)
        cases[name]['shapes']['never'] = data[name]['never_clamped_mm']
    for case in metrics['cases']:
        for source, expected in case['input_sha256'].items():
            if digest(Path(source)) != expected:
                raise ValueError('Analysis input changed: ' + source)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
        vmax = max(float(data[n][key].max()) for n in data
                   for key in ('distance_mm', 'forming_distance_mm')) * 1000
        vmax = max(vmax, np.finfo(float).eps)
        for row, name in enumerate(('run6', 'run9')):
            a = data[name]
            for key, label, color, ls in [('distance_mm', 'Pinned − free', '#0072B2', '-'),
                                          ('forming_distance_mm', 'Never − released', '#D55E00', '--')]:
                order = np.argsort(a[key])
                axes[row, 0].step(a[key][order] * 1000, np.cumsum(a['weights'][order]),
                                  where='post', label=label, color=color, linestyle=ls)
            axes[row, 0].set(xlabel='Aligned difference (µm)', ylabel='Reference area fraction',
                            ylim=(0, 1), title='Run ' + name[-1] + ': distributions')
            axes[row, 0].legend(fontsize=8)
            for col, key, title in [(1, 'distance_mm', 'Pinned − free'),
                                     (2, 'forming_distance_mm', 'Never − released')]:
                uv = a['material_uv_mm']
                artist = axes[row, col].tripcolor(uv[:, 0], uv[:, 1], a['faces'], a[key] * 1000,
                    shading='gouraud', cmap='cividis', vmin=0, vmax=vmax, rasterized=True)
                axes[row, col].set(xlabel='Material x (mm)', ylabel='Material y (mm)', title=title)
                axes[row, col].set_aspect('equal')
        fig.colorbar(artist, ax=axes[:, 1:].ravel().tolist(), label='Aligned difference magnitude (µm)', shrink=.8)
        save(fig, args.output, 'forming_and_release_full_field')
        height = max(8., max(float(np.max(np.abs(q[:, 2]))) for c in cases.values() for q in c['shapes'].values()))
        norm = TwoSlopeNorm(vmin=-height, vcenter=0, vmax=height)
        for name, case in cases.items():
            fig = plt.figure(figsize=(10, 12), constrained_layout=True)
            gs = fig.add_gridspec(3, 2)
            setup_panel(fig.add_subplot(gs[0, 0]), case, name)
            axes3d = []
            for cell, branch, title in [(gs[0, 1], 'clamped', '(b) Clamped equilibrium'),
                    (gs[1, 0], 'putong', '(c) Released: pinned'), (gs[1, 1], 'free', '(d) Released: free'),
                    (gs[2, 0], 'never', '(e) Never clamped')]:
                ax = fig.add_subplot(cell, projection='3d', computed_zorder=False)
                shape_panel(ax, case, branch, title, norm)
                ax.set_zlim(-height - 1, height + 1)
                ax.set_box_aspect((220, 320, (2 * height + 2) * 5))
                if branch == 'never':
                    ax.texts[0].set_text('No pads or persistent pins during forming')
                axes3d.append(ax)
            fig.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), ax=axes3d,
                         orientation='horizontal', fraction=.03, pad=.04,
                         label='Height z (mm; original solver frame)')
            save(fig, args.output, name + '_forming_and_release_stages')
    caption = ('Matched forming-restraint and release-constraint comparisons for Run 6 and Run 9. '
        'Full-field panels show area-weighted cumulative distributions and all material-vertex distances '
        'after separate proper area-weighted rigid alignment to clamped-then-free release; no scaling or reflection. '
        'Maps interpolate vertex magnitudes over the unchanged triangles and share one scale. '
        'The cases are deterministic; spatial vertices are not independent replicates and no confidence intervals are implied. '
        'Stage figures show original solver coordinates, identical cameras and height scales, with height exaggerated ×5; '
        'axis values remain actual millimetres. The never-clamped panel starts from the same reference sheet, not a released checkpoint. '
        'All final states and surface triangles are included. Apparent pose differences in stage panels are not intrinsic shape differences. '
        'See metrics for full-field norms, accepted-repeat evidence and exact target/history checks.\n')
    (args.output / 'forming_and_release.caption.md').write_text(caption)
    for filename in ('plot_never_clamped.py', 'plot_two_gripper_stages.py', 'plot_two_gripper_release.py'):
        path = Path(__file__).with_name(filename)
        sources[str(path)] = digest(path)
    (args.output / 'provenance.json').write_text(json.dumps(dict(sources_sha256=sources,
        numpy=np.__version__, matplotlib=matplotlib.__version__, dpi=220,
        visual_review='pending', exclusions='none', independent_cases=2), indent=2) + '\n')


if __name__ == '__main__':
    main()
