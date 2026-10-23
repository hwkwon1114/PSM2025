#!/usr/bin/env python3
"""Saved Run 6/9 setup, final clamped equilibrium, and both release endpoints.

No solver invocation. Geometry uses original solver coordinates; no rigid fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from plot_two_gripper_release import STYLE, STYLES

ROOT = Path(__file__).resolve().parents[2]
PAD_COLOR = '#CC79A7'
GAUGE_COLOR = '#009E73'
VERTICAL_EXAGGERATION = 5


def toolpath(operation):
    # Matches buildZigZagStrips in src/libshell/ZigZagGrowth.hpp.
    n = operation['n_strips']
    length = operation['lv_mm']
    pitch = length * np.tan(np.deg2rad(operation['alpha_deg']))
    left = -(n - 2) * pitch / 2
    pts = [[left, -length / 2], [left, length / 2]]
    for j in range(1, n - 1):
        pts.append([left + j * pitch, (-1 if j % 2 else 1) * length / 2])
    pts.append([pts[-1][0], -pts[-1][1]])
    assert len(pts) == n + 1 and operation['rotation_deg'] == 0
    return np.asarray(pts) + operation['shift_uv_mm']


def load_case(name, sources):
    folder = ROOT / 'run/two_gripper_comparison' / name
    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    config = read(ROOT / 'legacy/configs/boundary_conditions' / (name + '_two_gripper_cycle.json'))
    parent_envelope = read(folder / 'paired_clamped_checkpoint.json')
    parent = parent_envelope['payload']
    ends = {b: read(folder / ('release_' + b + '.json'))['payload'] for b in ('putong', 'free')}
    uv = np.asarray(parent['identity']['uv']) * 1000
    nv = len(uv)
    def xyz(state):
        return np.asarray(state, float).reshape(-1)[:3 * nv].reshape(3, nv).T * 1000
    shapes = {'clamped': xyz(parent['state'])}
    masks = {'clamped': np.asarray(parent['identity']['vertex_mask'], bool)}
    assert parent['proof']['accepted']
    for branch, end in ends.items():
        assert end['proof']['accepted'] and end['parent_checksum'] == parent_envelope['checksum']
        assert end['target_history_unchanged'] and end['fixed_coordinates_unchanged']
        shapes[branch] = xyz(end['state'])
        masks[branch] = np.asarray(end['vertex_mask'], bool)
        assert np.array_equal(shapes[branch][masks[branch]], shapes['clamped'][masks[branch]])
    assert ends['putong']['start_state_checksum'] == ends['free']['start_state_checksum']
    assert masks['putong'].sum() == 6 and masks['free'].sum() == 0
    op = config['toolpaths'][0]['operation']
    path = toolpath(op)
    assert np.isclose(path[:, 1].max(), 65)
    assert np.all(np.abs(path[:, 0]) <= 100) and path[:, 1].min() >= -150
    pad_vertices = np.zeros(nv, bool)
    for region in config['boundary_conditions']['regions']:
        center, size = np.array(region['center_uv_mm']), np.array(region['size_uv_mm'])
        assert region['rotation_deg'] == 0 and np.array_equal(size, [50, 20])
        assert center[1] + size[1] / 2 == 150
        pad_vertices |= np.all(np.abs(uv - center) <= size / 2 + 1e-9, axis=1)
    assert np.array_equal(masks['clamped'], np.repeat(pad_vertices[:, None], 3, axis=1))
    reference = xyz(parent['identity']['rest'])
    assert np.array_equal(shapes['clamped'][masks['clamped']], reference[masks['clamped']])
    assert all(np.isfinite(q).all() for q in shapes.values())
    return dict(config=config, path=path, shapes=shapes, masks=masks,
                faces=np.asarray(parent['identity']['faces']), repeats=parent['cycle'])


def setup_panel(ax, case, name):
    ax.add_patch(Rectangle((-100, -150), 200, 300, fc='#FAFAFA', ec='#333333', lw=.9))
    path = case['path']
    ax.plot(path[:, 0], path[:, 1], color=STYLES[name]['color'], lw=.55)
    ax.plot(path[0, 0], path[0, 1], marker='o', ms=4, color='black')
    for region in case['config']['boundary_conditions']['regions']:
        x, y = region['center_uv_mm']
        ax.add_patch(Rectangle((x - 25, y - 10), 50, 20, fc=PAD_COLOR, ec='#663355', hatch='///', lw=.9))
        ax.text(x, 114, '(%+d, 140)' % x, ha='center', va='center', fontsize=8)
    ax.axhline(65, ls=':', color='#666666', lw=.7)
    ax.annotate('', xy=(90, 150), xytext=(90, 65), arrowprops={'arrowstyle': '<->', 'lw': .8})
    ax.text(104, 98, '85 mm', rotation=90, ha='center', va='center', fontsize=8)
    ax.set(xlim=(-112, 112), ylim=(-160, 174), xlabel='Material x (mm)', ylabel='Material y (mm)')
    ax.set_aspect('equal')
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-150, -100, -50, 0, 65, 130, 150])
    ax.set_title('(a) Run %s: toolpath + BC' % name[-1], loc='left')
    ax.text(0, 162, 'Pad tops: y = 150 mm', ha='center', fontsize=8)


def shape_panel(ax, case, branch, title, norm):
    q = case['shapes'][branch]
    surface = ax.plot_trisurf(q[:, 0], q[:, 1], q[:, 2], triangles=case['faces'],
                             cmap='coolwarm', norm=norm, linewidth=0, antialiased=False,
                             shade=False, rasterized=True, zorder=1)
    if branch == 'clamped':
        for region in case['config']['boundary_conditions']['regions']:
            x, y = region['center_uv_mm']
            corners = [[x-25,y-10,0], [x+25,y-10,0], [x+25,y+10,0], [x-25,y+10,0]]
            ax.add_collection3d(Poly3DCollection([corners], facecolors=PAD_COLOR,
                                                 edgecolors='#663355', linewidths=1.2, zorder=3))
        note = '%d repeats; both pads fixed xyz' % case['repeats']
    elif branch == 'putong':
        for count, marker, label in [(3, 'o', 'A: xyz'), (2, 's', 'B: yz'), (1, '^', 'C: z')]:
            ids = np.flatnonzero(case['masks'][branch].sum(axis=1) == count)
            assert len(ids) == 1
            point = q[ids[0]]
            ax.scatter(*point, color=GAUGE_COLOR, marker=marker, s=26, edgecolors='black', linewidths=.5, label=label, depthshade=False, zorder=5)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.02), ncol=3, columnspacing=.5, handletextpad=.1, fontsize=8)
        note = 'Pads removed; six coordinates retained'
    else:
        note = 'Pads removed; no fixed vertices'
    ax.text2D(.02, .88, note, transform=ax.transAxes, fontsize=8)
    ax.text2D(.02, .03, 'Height ×5; z labels are actual mm', transform=ax.transAxes, fontsize=8)
    ax.set(xlim=(-110, 110), ylim=(-160, 160), zlim=(-9, 9), xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
    ax.set_xticks([-100, 0, 100]); ax.set_yticks([-150, 0, 150]); ax.set_zticks([-8, 0, 8])
    ax.set_box_aspect((220, 320, 18 * VERTICAL_EXAGGERATION))
    ax.view_init(elev=24, azim=-60)
    ax.set_title(title, loc='left')
    ax.grid(False)
    return surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    cases = {name: load_case(name, sources) for name in ('run6', 'run9')}
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
    assert all(np.max(np.abs(q[:, 2])) <= 8 for c in cases.values() for q in c['shapes'].values())
    for name, case in cases.items():
        with plt.style.context(dict(STYLE, **{'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10})):
            fig = plt.figure(figsize=(9, 8.5), constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            setup_panel(fig.add_subplot(gs[0, 0]), case, name)
            axes = []
            for cell, branch, title in [(gs[0,1], 'clamped', '(b) Final clamped equilibrium'),
                                        (gs[1,0], 'putong', '(c) Released: 3–2–1 pinning'),
                                        (gs[1,1], 'free', '(d) Released: fully free')]:
                ax = fig.add_subplot(cell, projection='3d', computed_zorder=False)
                shape_panel(ax, case, branch, title, norm)
                axes.append(ax)
            fig.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), ax=axes,
                         orientation='horizontal', fraction=.045, pad=.04, shrink=.75,
                         label='Height z (mm; original solver frame)')
            fig.canvas.draw()
            # In-panel notes must stay inside their axes; titles/ticks are reviewed visually.
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                for text in ax.texts:
                    bb = text.get_window_extent(renderer)
                    assert ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1), text.get_text()
            stem = name + '_two_gripper_stages'
            for extension in ('png', 'pdf'):
                fig.savefig(args.output / (stem + '.' + extension), dpi=220)
            plt.close(fig)
        caption = ('Run %s: toolpath, clamped forming and release. (a) Centerline reconstructed from the exact '
                   'simulation recipe; dot marks its start. Hatched purple pads are 50 × 20 mm, centered '
                   'at (−50,140) and (+50,140) mm. The 85 mm dimension is from the upper path turnaround '
                   'to the sheet edge, not the pad gap (65 mm). (b) Accepted equilibrium after %d complete '
                   'forming repeats with all pad vertex positions fixed. (c,d) Separate accepted release '
                   'equilibria from the identical clamped state, without additional growth; (c) retains '
                   'A:xyz/B:yz/C:z at release-time values, (d) has no fixed vertices. All 30,000 mesh '
                   'triangles are rendered; no smoothing or selected examples. All 3D views use identical '
                   'camera, limits and height colors across runs. Height is exaggerated ×5 geometrically '
                   'for visibility, while z ticks/color values remain actual millimetres. Original solver '
                   'coordinates are shown, without rigid alignment: apparent pose differences are not '
                   'intrinsic shape differences. These are final phase equilibria, not intermediate optimizer '
                   'iterations. See the separate aligned full-field comparison for quantitative release '
                   'differences. Independent final scientific review remains pending.\n') % (name[-1], case['repeats'])
        (args.output / (stem + '.caption.md')).write_text(caption)
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {'sources_sha256': sources, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                  'figure_size_inches': [9, 8.5], 'dpi': 220, 'vertical_exaggeration': 5,
                  'coordinate_frame': 'original solver; no rigid alignment', 'colormap': 'coolwarm',
                  'height_limits_mm': [-8, 8], 'view': {'elevation': 24, 'azimuth': -60},
                  'selection': 'both runs, every surface triangle, final accepted phase equilibria',
                  'pad_color': PAD_COLOR, 'gauge_color': GAUGE_COLOR, 'run_styles': STYLES,
                  'checks': 'accepted endpoints; matching parent/start; exact fixed values; full pad mask; 6/0 release DOFs; finite geometry; path bounds',
                  'visual_review': 'pending'}
    (args.output / 'two_gripper_stages.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
