#!/usr/bin/env python3
"""Plot every accepted full-pass state, with shared true-scale geometry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from analyze_sequence_ablation_v2 import load_vtp, point_array, proper_kabsch
from visualize_solver_characteristics import place_strips, CYCLE_RE
from run_negative_orthotropy_passes import NAME

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'run/forward_model_diagnostics/current'/NAME)
    args = parser.parse_args()
    directory = args.run.resolve()
    seq = json.loads((directory/'sequence.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    with (directory/'sequence_convergence.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    accepted = {int(r['executed_cycle_index']) for r in rows
                if r['equilibrium_accepted'] == '1' and float(r['lambda_trial']) == 1.0}
    frames, sources = {}, []
    for path in sorted(directory.glob('*_final.vtp')):
        match = CYCLE_RE.search(path.name)
        if not match or int(match.group(1)) not in accepted:
            continue
        data, xyz, triangles = load_vtp(path)
        uv = np.column_stack([point_array(data, key) for key in ('material_u', 'material_v')])
        aligned, _, _ = proper_kabsch(np.column_stack([uv, np.zeros(len(uv))]), xyz)
        assert np.isfinite(aligned).all()
        frames[int(match.group(1))] = (aligned*1000, triangles)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not frames:
        raise SystemExit('No accepted full-pass states available; no deformation figure generated.')
    out = directory/'deformation_figures'
    out.mkdir(exist_ok=False)
    lx, ly = manifest['plate']['lx']*1000, manifest['plate']['ly']*1000
    all_xyz = np.vstack([xyz for xyz, _ in frames.values()])
    zlim = max(float(np.abs(all_xyz[:, 2]).max()), 1e-6)
    xlim = max(lx, float(np.abs(all_xyz[:, 0]).max()))
    ylim = max(ly, float(np.abs(all_xyz[:, 1]).max()))
    norm = Normalize(-zlim, zlim)
    n = len(seq['toolpaths'])
    style = {'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42,
             'figure.facecolor': 'white', 'axes.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(5*n, 8), layout='constrained')
        grid = fig.add_gridspec(2, n, height_ratios=(1, 1.25))
        bottom = []
        for col, path in enumerate(seq['toolpaths']):
            cycle = col+1
            op = path['operation']
            ax = fig.add_subplot(grid[0, col])
            ax.add_patch(Rectangle((-lx, -ly), 2*lx, 2*ly, fc='none', ec='.5', lw=.8))
            placement = place_strips(op, (-lx/1000, lx/1000, -ly/1000, ly/1000))
            for strip in placement['strips']:
                a, b = np.asarray(strip['a_uv_m'])*1000, np.asarray(strip['b_uv_m'])*1000
                ax.plot([a[0], b[0]], [a[1], b[1]], color='#0072B2', lw=1.6)
            ax.set(xlim=(-lx*1.06, lx*1.06), ylim=(-ly*1.06, ly*1.06),
                   xlabel='Material u (mm)', ylabel='Material v (mm)',
                   title=f'Pass {cycle}: {op["rotation_deg"]:g}°')
            ax.set_aspect('equal')
            ax = fig.add_subplot(grid[1, col], projection='3d')
            bottom.append(ax)
            if cycle not in frames:
                ax.text2D(.5, .5, 'No accepted full-pass state', transform=ax.transAxes, ha='center')
                ax.set_axis_off()
                continue
            xyz, triangles = frames[cycle]
            colors = plt.get_cmap('RdBu_r')(norm(xyz[triangles, 2].mean(axis=1)))
            mesh = Poly3DCollection(xyz[triangles], facecolors=colors, edgecolors='none', rasterized=True)
            ax.add_collection3d(mesh)
            ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim), zlim=(-zlim, zlim),
                   xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)', title=f'After pass {cycle}')
            ax.set_box_aspect((2*xlim, 2*ylim, 2*zlim))
            ax.view_init(elev=25, azim=-60)
            ax.tick_params(labelsize=8, pad=1)
        fig.colorbar(ScalarMappable(norm=norm, cmap='RdBu_r'), ax=bottom,
                     location='bottom', shrink=.6, pad=.08, label='Aligned height (mm)')
        fig.canvas.draw()
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'zigzag_pass_deformation.{ext}', dpi=180)
        plt.close(fig)
    caption = (
        'Cumulative deformation after successive negative-orthotropy zigzag passes. '
        'Top row: centerline of the current pass in material coordinates (not the strip footprint). '
        'Bottom row: accepted full-load deformed plate after that pass, including all preceding growth. '
        'All planned passes are shown; missing accepted endpoints are explicitly labeled. '
        'One deterministic trajectory, not independent samples. Top growth 0.005 per hit, bottom growth zero, '
        'orthotropy -0.5, no hardening. Each state is rigidly aligned to the flat material-coordinate mesh '
        'by proper Kabsch alignment; no deformation magnification. Common camera, axis limits, physical aspect '
        'and color scale; face color is mean aligned vertex height. Absolute residual acceptance 1e-11. '
        'These plots do not establish stability or quantify curvature directions.\n')
    (out/'zigzag_pass_deformation.caption.md').write_text(caption)
    (out/'provenance.json').write_text(json.dumps(dict(
        source_run=str(directory), sources=sources, accepted_cycles=sorted(frames),
        planned_cycles=n, height_limits_mm=[-zlim, zlim], python=platform.python_version(),
        numpy=np.__version__, matplotlib=matplotlib.__version__,
        script=str(Path(__file__).resolve()), size_inches=[5*n, 8], dpi=180,
        visual_review='Pending; export success is not visual inspection.'), indent=2)+'\n')
    print(out, flush=True)


if __name__ == '__main__':
    main()
