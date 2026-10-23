#!/usr/bin/env python3
"""Three full-field contrasts and all four conditions; preserve earlier figures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from analyze_two_gripper_release import digest, align
from plot_two_gripper_stages import load_case, setup_panel, shape_panel, STYLE
from plot_never_clamped import save

CONTRASTS = [('distance_mm', 'Pinned − free', '#0072B2', '-'),
             ('forming_distance_mm', 'Never − released', '#D55E00', '--'),
             ('cadence_distance_mm', 'Final-only − per-cycle', '#009E73', '-.')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--align-stages', action='store_true', help='Rigidly align stages to the clamped reference')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.analysis / 'four_condition_metrics.json'
    metrics = json.loads(path.read_text())
    sources = {str(path): digest(path)}
    for case in metrics['cases']:
        for source, expected in case['input_sha256'].items():
            if digest(Path(source)) != expected:
                raise ValueError('Analysis input changed: ' + source)
    data, cases = {}, {}
    for name in ('run6', 'run9'):
        path = args.analysis / (name + '_full_field.npz')
        sources[str(path)] = digest(path)
        with np.load(path) as archive:
            data[name] = dict(archive)
        cases[name] = load_case(name, sources)
        cases[name]['shapes'].update(never=data[name]['never_clamped_mm'],
                                     final_only=data[name]['final_only_mm'])
    alignment = {}
    if args.align_stages:
        for name, case in cases.items():
            reference = case['shapes']['clamped']
            weights = data[name]['weights']
            assert np.isclose(weights.sum(), 1) and np.all(weights > 0)
            alignment[name] = {}
            for branch, points in case['shapes'].items():
                fitted, rotation, translation = align(points, reference, weights)
                assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                assert np.isclose(np.linalg.det(rotation), 1)
                # Keep the clamped reference/pad coordinates exactly unchanged.
                if branch == 'clamped':
                    fitted = points.copy()
                case['shapes'][branch] = fitted
                alignment[name][branch] = dict(rotation=rotation.tolist(), translation_mm=translation.tolist(),
                    rms_to_clamped_mm=float(np.sqrt(weights @ np.sum((fitted-reference)**2, axis=1))))
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
        vmax = max(float(data[name][key].max()) * 1000 for name in data for key, _, _, _ in CONTRASTS)
        vmax = max(vmax, np.finfo(float).eps)
        for row, name in enumerate(('run6', 'run9')):
            a = data[name]
            for col, (key, label, color, ls) in enumerate(CONTRASTS, 1):
                order = np.argsort(a[key])
                axes[row, 0].step(a[key][order] * 1000, np.cumsum(a['weights'][order]),
                                  where='post', label=label, color=color, linestyle=ls)
                uv = a['material_uv_mm']
                artist = axes[row, col].tripcolor(uv[:, 0], uv[:, 1], a['faces'], a[key] * 1000,
                    shading='gouraud', cmap='cividis', vmin=0, vmax=vmax, rasterized=True)
                axes[row, col].set(xlabel='Material x (mm)', ylabel='Material y (mm)', title=label)
                axes[row, col].set_aspect('equal')
            axes[row, 0].set(xlabel='Aligned difference (µm)', ylabel='Reference area fraction',
                             ylim=(0, 1), title='Run ' + name[-1] + ': distributions')
            axes[row, 0].legend(fontsize=8)
        fig.colorbar(artist, ax=axes[:, 1:].ravel().tolist(), label='Aligned difference magnitude (µm)', shrink=.8)
        save(fig, args.output, 'four_condition_full_field')
        height = max(8., max(float(np.max(np.abs(q[:, 2]))) for c in cases.values() for q in c['shapes'].values()))
        norm = TwoSlopeNorm(vmin=-height, vcenter=0, vmax=height)
        for name, case in cases.items():
            fig = plt.figure(figsize=(10, 12), constrained_layout=True)
            gs = fig.add_gridspec(3, 2)
            setup_panel(fig.add_subplot(gs[0, 0]), case, name)
            axes3d = []
            for cell, branch, title in [(gs[0, 1], 'clamped', '(b) Clamped equilibrium'),
                    (gs[1, 0], 'putong', '(c) Released: pinned'), (gs[1, 1], 'free', '(d) Released: free'),
                    (gs[2, 0], 'never', '(e) Never-clamped per-cycle'),
                    (gs[2, 1], 'final_only', '(f) Never-clamped final-only')]:
                ax = fig.add_subplot(cell, projection='3d', computed_zorder=False)
                shape_panel(ax, case, branch, title, norm)
                ax.set_zlim(-height - 1, height + 1)
                ax.set_box_aspect((220, 320, (2 * height + 2) * 5))
                if branch in ('never', 'final_only'):
                    ax.texts[0].set_text('No pads or persistent pins during forming')
                axes3d.append(ax)
            fig.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), ax=axes3d,
                         orientation='horizontal', fraction=.03, pad=.04,
                         label='Height z (mm; clamped reference frame)' if args.align_stages else 'Height z (mm; original solver frame)')
            save(fig, args.output, name + '_four_condition_stages')
    (args.output / 'four_conditions.caption.md').write_text(
        'Four-condition comparison for Run 6 and Run 9. The full-field figure shows reference-area-weighted '
        'cumulative distributions and all material-vertex distances after proper rigid alignment, without scale/reflection: '
        'pinned versus free release, never-clamped per-cycle versus clamped-then-free, and final-only versus never-clamped '
        'per-cycle. Each contrast uses its own comparator and fitted rigid transform. Maps interpolate vertex magnitudes '
        'over unchanged triangles and share one scale; small contrasts can therefore look nearly uniform. '
        'Two deterministic cases, not independent vertex replicates; no confidence intervals or numerical error bounds are implied. ' +
        ('Stage views include setup, intermediate clamped state and all four final conditions after reference-area-weighted '
         'proper rigid alignment to the clamped state, using all corresponding vertices; no scale or reflection. '
         'This removes pose differences, not shape differences. Constraint markers follow the transformed vertices. ' if args.align_stages else
         'Stage views include setup, intermediate clamped state and all four final conditions in original solver coordinates, ') +
        'with common cameras/height scales and ×5 vertical exaggeration; axes remain actual millimetres. '
        'Unaligned pose differences are not intrinsic shape differences. All vertices/triangles and both cases are included.\n')
    for filename in ('plot_four_conditions.py', 'plot_never_clamped.py', 'plot_two_gripper_stages.py', 'plot_two_gripper_release.py'):
        path = Path(__file__).with_name(filename)
        sources[str(path)] = digest(path)
    (args.output / 'provenance.json').write_text(json.dumps(dict(sources_sha256=sources,
        numpy=np.__version__, matplotlib=matplotlib.__version__, dpi=220,
        visual_review='pending', exclusions='none', independent_cases=2,
        contrast_styles=CONTRASTS, stage_alignment=alignment,
        stage_reference='clamped' if args.align_stages else 'original solver frame'), indent=2) + '\n')


if __name__ == '__main__':
    main()
