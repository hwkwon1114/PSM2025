"""Publication figures for the Hyunwoo Run 1 blind prediction (card REPLICATE-HW1), Attempt 209 style.

Writes new files only (refuses to overwrite) into calibration/paper_figures/attempt209_publication/:
  hw_run1_toolpath, hw_run1_shape_comparison, hw_run1_shape_3d  (.pdf, .png, .caption.md)
  hw_run1_provenance.json
Typography follows the folder: 14 in canvas with uniform 20 pt text (about 10 pt at 7 in print width).
Public frame: width x = u, length z = v + 65 mm, height y. Nothing is fitted; the scan placement is the
saved rigid alignment from shape_design/compare_hw1.py.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.common import digest, save_json
from calibration.analysis import publication_palette as palette
palette.register()
from shape_design import compare_hw1 as cmp, replicate_hw1 as rep
from shape_design.plot_hw1_comparison import grid_heights

OUT = ROOT / 'calibration/paper_figures/attempt209_publication'
SOLVE = ROOT / 'run/shape_design/hw_run1/solve_mesh2'
Z0 = 65.0
FONT = 20
STYLE = {'font.size': FONT, 'axes.titlesize': FONT, 'axes.labelsize': FONT, 'xtick.labelsize': FONT,
         'ytick.labelsize': FONT, 'legend.fontsize': FONT, 'pdf.fonttype': 42, 'axes.spines.top': False,
         'axes.spines.right': False, 'savefig.facecolor': 'white'}
BLUE, ORANGE = palette.BLUE, palette.ORANGE
FACE_COLOR = {'front': BLUE, 'back': ORANGE}
REGION_ORDER = ('Region 1', 'Region 3', 'Region 2')     # left to right on the sheet


def save(fig, name, caption):
    """Uniform-font and on-page checks (as attempt209_publication.save), then write without overwriting."""
    paths = [OUT / f'{name}.{e}' for e in ('pdf', 'png', 'caption.md')]
    if any(p.exists() for p in paths):
        raise FileExistsError(f'Refusing to overwrite {name}')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    for ax in fig.axes:
        texts.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        axes = [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]
        if hasattr(ax, 'zaxis'):
            axes.append((ax.zaxis, ax.get_zlim()))
            texts.append(ax.zaxis.label)
        for axis, limits in axes:
            lo, hi = sorted(limits)
            for tick in axis.majorTicks:
                if lo <= tick.get_loc() <= hi:
                    texts.extend([tick.label1, tick.label2])
        if ax.get_legend():
            texts.extend(ax.get_legend().get_texts())
    for text in texts:
        if text.get_visible() and text.get_text():
            if not np.isclose(text.get_fontsize(), FONT):
                raise ValueError(f'{name}: non-uniform font: {text.get_text()!r} = {text.get_fontsize()}')
            b = text.get_window_extent(renderer).extents
            if np.isfinite(b).all() and (b[0] < -1 or b[1] < -1 or b[2] > fig.bbox.width + 1
                                         or b[3] > fig.bbox.height + 1):
                raise ValueError(f'{name}: text outside page: {text.get_text()!r}')
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    paths[2].write_text(caption + '\n')


def sheet_axes(ax):
    ax.add_patch(Rectangle((-110, 3), 220, 124, facecolor='#F0F0F0', edgecolor='#999999', lw=1, zorder=0))
    ax.set(xlim=(-112, 112), ylim=(1, 129), aspect='equal', xlabel='Width $x$ (mm)', ylabel='Length $z$ (mm)')
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([13, 65, 117])


def toolpath_figure(passes):
    fig = plt.figure(figsize=(14, 10.5), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 0.9])
    ax = top.subplots()
    ax.add_patch(Rectangle((-110, 3), 220, 124, facecolor='#F0F0F0', edgecolor='#999999', lw=1, zorder=0))
    ax.set(xlim=(-112, 112), ylim=(-22, 158), aspect='equal', xlabel='Width $x$ (mm)', ylabel='Length $z$ (mm)')
    ax.set_xticks([-100, -50, 0, 40, 100])
    ax.set_yticks([13, 65, 117])
    for p in passes[:6]:                                  # rounds repeat exactly; draw round 1 once
        if p['direction'] != 'F':
            continue
        legs = p['legs'] - rep.ORIGIN
        xy = np.vstack([legs[:, 0], legs[-1:, 1]])
        ax.plot(xy[:, 0], xy[:, 1] + Z0, color=FACE_COLOR[p['face']], lw=1.6, zorder=3)
        ax.plot(*(xy[0] + [0, Z0]), 'o', color='black', ms=9, zorder=5)
    for name in REGION_ORDER:
        p = next(p for p in passes if p['region'] == name)
        legs = p['legs'] - rep.ORIGIN
        lo, hi = legs[:, :, 0].min(), legs[:, :, 0].max()
        ax.annotate('', xy=(hi, 122), xytext=(lo, 122), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax.text((lo + hi) / 2, 132, f"{name}\n{p['face']} face", ha='center', va='bottom',
                color=FACE_COLOR[p['face']])
    ax.annotate('', xy=(-50, -6), xytext=(-40, -6), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(-45, -10, '10 mm gap', ha='center', va='top')
    ax.text(40, -10, 'shared stroke', ha='center', va='top')
    ax.plot([40, 40], [-7, 3], color='black', lw=1.5)
    ax.set_title('(a) Wheel centrelines, one round', pad=8)

    ax = bottom.subplots()
    rows = {name: i for i, name in enumerate(REGION_ORDER)}
    for p in passes:
        marker = '>' if p['direction'] == 'F' else '<'
        ax.plot(p['pass'], rows[p['region']], marker=marker, ms=15, ls='', color=FACE_COLOR[p['face']])
    for r in range(1, 4):
        ax.axvline(6 * r + 0.5, color='#999999', lw=1, ls=':')
    for r in range(4):
        ax.text(6 * r + 3.5, 2.75, f'Round {r + 1}', ha='center', va='center')
    ax.set(xlim=(0.3, 24.7), ylim=(3.1, -0.5), xlabel='Pass number', title='(b) Pass order')
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_yticks(range(3), REGION_ORDER)
    ax.legend(handles=[Line2D([], [], marker='>', ls='', color='#555555', ms=15, label='Forward'),
                       Line2D([], [], marker='<', ls='', color='#555555', ms=15, label='Reverse'),
                       Line2D([], [], marker='s', ls='', color=BLUE, ms=13, label='Front face'),
                       Line2D([], [], marker='s', ls='', color=ORANGE, ms=13, label='Back face')],
              loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    return fig


def height_fields(shape, scan):
    """Heights on a 1 mm grid relative to the common mean plane. The scan is thinned to its point nearest
    each 0.5 mm cell centre (actual scan points) to keep the triangulation fast."""
    U, V, h_pred = grid_heights(shape['aligned_mm'])
    _, _, h_scan = grid_heights(cmp.subsample(scan, 0.5))
    both = np.isfinite(h_pred) & np.isfinite(h_scan)
    h_pred = np.where(both, h_pred - h_pred[both].mean(), np.nan)
    h_scan = np.where(both, h_scan - h_scan[both].mean(), np.nan)
    return U, V, h_scan, h_pred


def comparison_figure(fields, prof):
    U, V, h_scan, h_pred = fields
    diff = h_scan - h_pred
    fig = plt.figure(figsize=(14, 10), layout='constrained')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    lim = float(np.nanmax(np.abs([h_scan, h_pred])))
    dl = float(np.nanmax(np.abs(diff)))
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, field, title in zip(axes[:2], (h_scan, h_pred), ('(a) Scan', '(b) Prediction')):
        im = ax.pcolormesh(U, V + Z0, field, cmap=palette.DIVERGING, vmin=-lim, vmax=lim, shading='auto', rasterized=True)
        sheet_axes(ax)
        ax.set_title(title)
    imd = axes[2].pcolormesh(U, V + Z0, diff, cmap=palette.DIVERGING, vmin=-dl, vmax=dl, shading='auto', rasterized=True)
    sheet_axes(axes[2])
    axes[2].set_title('(c) Scan − prediction')
    for ax in axes[1:]:
        ax.set_ylabel('')
    fig.colorbar(im, ax=axes[:2], label='Height $y$ (mm)', orientation='horizontal', shrink=0.5, aspect=25)
    fig.colorbar(imd, ax=axes[2], label=r'$\Delta y$ (mm)', orientation='horizontal', shrink=0.9, aspect=12,
                 ticks=[-1, 0, 1])
    ax = fig.add_subplot(gs[1, :])
    faces = {'Region 1': 'front', 'Region 3': 'back', 'Region 2': 'front'}
    for name, (lo, hi) in cmp.REGIONS_U.items():
        ax.axvspan(lo, hi, color=FACE_COLOR[faces[name]], alpha=0.08, lw=0)
        ax.text((lo + hi) / 2, 0.96, f'{name}, {faces[name]}', transform=ax.get_xaxis_transform(),
                ha='center', va='top')
    ax.plot(prof['u_mm'], prof['profile_scan_mm'], 'o-', color=ORANGE, ms=6, lw=1.5, label='Scan')
    ax.plot(prof['u_mm'], prof['profile_pred_mm'], 's--', color=BLUE, ms=6, lw=1.5, label='Prediction')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set(xlim=(-110, 110), ylim=(-2.4, 2.4), xlabel='Width $x$ (mm)', ylabel='Mean height $y$ (mm)',
           title=r'(d) Width profile, $|z - 65| \leq 40$ mm')
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.legend(loc='lower left', frameon=False)
    return fig, float(np.sqrt(np.nanmean(diff ** 2)))


def shape3d_figure(fields, exag):
    U, V, h_scan, h_pred = (f[::2, ::2] for f in fields)      # 2 mm display grid
    diff = h_scan - h_pred
    lim = float(np.nanmax(np.abs([h_scan, h_pred])))
    dl = float(np.nanmax(np.abs(diff)))
    hn, dn = colors.Normalize(-lim, lim), colors.Normalize(-dl, dl)
    fig = plt.figure(figsize=(14, 12))
    spec = [(h_scan, h_scan, palette.DIVERGING, hn, '(a) Scan'),
            (h_pred, h_pred, palette.DIVERGING, hn, '(b) Prediction'),
            (h_pred, diff, palette.DIVERGING, dn, '(c) Prediction coloured by scan − prediction')]
    for k, (height, col, cmap, norm, title) in enumerate(spec):
        bottom = 0.67 - 0.31 * k
        ax = fig.add_axes([-0.06, bottom, 0.9, 0.31], projection='3d', computed_zorder=False)
        ax.plot_surface(U, V + Z0, height, facecolors=plt.get_cmap(cmap)(norm(np.nan_to_num(col))),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, rasterized=True)
        ax.set_box_aspect((220, 124, 2 * lim * exag), zoom=1.2)
        ax.set(xlim=(-110, 110), ylim=(3, 127), zlim=(-lim, lim))
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([13, 117]); ax.set_zticks([])
        ax.set_xlabel('Width $x$ (mm)', labelpad=16)
        ax.set_ylabel('Length $z$ (mm)', labelpad=16)
        ax.view_init(elev=28, azim=-60)
        ax.text2D(0.16, 0.88, title, transform=ax.transAxes)
        cax = fig.add_axes([0.74, bottom + 0.05, 0.02, 0.2])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                     label=r'$\Delta y$ (mm)' if k == 2 else 'Height $y$ (mm)')
    fig.text(0.86, 0.99, f'Vertical exaggeration ×{exag:g}', ha='right', va='top')
    return fig, float(np.sqrt(np.nanmean(diff ** 2)))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--exaggeration', type=float, default=10.0)
    ap.add_argument('--out', type=Path, default=OUT, help='staging directory for drafts')
    ap.add_argument('--only', nargs='+', choices=('toolpath', 'comparison', '3d'),
                    default=['toolpath', 'comparison', '3d'])
    a = ap.parse_args()
    OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((SOLVE / 'result.json').read_text())
    summary = json.loads((SOLVE / 'comparison/comparison.json').read_text())
    if result['status'] != 'accepted' or result['hit_check']['mismatched_faces'] != 0:
        raise SystemExit('Solve not accepted')
    if summary['inputs']['shape_npz_sha256'] != digest(SOLVE / 'shape.npz'):
        raise ValueError('shape.npz changed since comparison')
    shape = dict(np.load(SOLVE / 'shape.npz'))
    prof = dict(np.load(SOLVE / 'comparison/profiles.npz'))
    scan = prof['scan_in_pred_frame_mm']
    passes = rep.read_passes()
    best = summary['orientations'][summary['best_orientation']]
    model = ('frozen Attempt 209 parameters (calibrated on Run 1, 0.77 mm); sheet 220 × 124 × 0.61 mm; '
             'back-face passes swap the worked/unworked layer growth and orthotropy; one 2 mm-mesh equilibrium '
             'after all 24 passes')
    with plt.rc_context(STYLE):
        if 'toolpath' in a.only:
            save(toolpath_figure(passes), 'hw_run1_toolpath',
                 '# Hyunwoo Run 1 toolpath (24 passes)\n\n'
                 'Collaborator toolpath `Wheeling_Toolpath_24_Passes_Consolidated.csv` in the public frame (width '
                 'x = CSV x − 110 mm; length z = CSV y + 3 mm). The 220 × 124 mm sheet (grey) is divided across '
                 'its width into three regions, each worked by its own zig-zag of 104 mm strokes (z = 13–117 mm) on '
                 'a 5 mm pitch, closed by a straight stroke at each end: Region 1, x = −100 to −50 mm (12 strokes), '
                 'and Region 2, x = 40 to 100 mm (14 strokes), are worked on the front face; Region 3, x = −40 to '
                 '40 mm (18 strokes), is worked on the back face after turning the sheet over (CSV coordinates are '
                 'given in the front-view frame). There is a 10 mm gap between the centrelines of Regions 1 and 3 '
                 '(x = −50 to −40 mm, never worked), whereas Regions 3 and 2 share the stroke at x = 40 mm, which is '
                 'therefore worked from both faces. (a) Centrelines of one round; black dots mark where each forward '
                 'pass starts and arrows its stepping direction. Reverse passes retrace the same strokes in the '
                 'opposite order. (b) Pass order: each round runs Regions 1, 2, 3 forward and then 3, 2, 1 in '
                 'reverse; the four rounds are identical (24 passes). Blue: front face; orange: back face.')
        fields = height_fields(shape, scan) if {'comparison', '3d'} & set(a.only) else None
        if 'comparison' in a.only:
            fig, rms2d = comparison_figure(fields, prof)
            save(fig, 'hw_run1_shape_comparison',
             '# Hyunwoo Run 1: scanned shape vs blind Attempt 209 prediction\n\n'
             f'Prediction: {model}; nothing fitted to the scan. The scan is placed by rigid alignment only '
             f'(best of the four proper orientations of the rectangle). (a, b) Height y relative to the common '
             f'mean plane on a shared colour scale, linearly interpolated to a 1 mm grid (scan thinned to one measured point per 0.5 mm cell). (c) Vertical '
             f'difference scan − prediction (RMS {rms2d:.2f} mm). (d) Mean height over |z − 65| ≤ 40 mm in '
             f'5 mm bins along x (mean removed); shading marks the worked regions and faces. Symmetric surface '
             f'RMS {best["symmetric_rms_mm"]:.2f} mm; profile peak-to-valley {summary["profile_ptv_pred_mm"]:.2f} '
             f'mm predicted vs {summary["profile_ptv_scan_mm"]:.2f} mm scanned (ratio '
             f'{summary["profile_ptv_ratio_pred_over_scan"]:.2f}); the bending sign matches in all three '
             f'regions, while the regional profile curvature is about 40 % lower in the prediction. One '
             f'specimen and one prediction; thickness, 5 mm pitch and back-face working extrapolate the '
             f'Run 1 calibration.')
        if '3d' in a.only:
            fig, rms3d = shape3d_figure(fields, a.exaggeration)
            save(fig, 'hw_run1_shape_3d',
             '# Hyunwoo Run 1: three-dimensional view of scanned and predicted shapes\n\n'
             f'Same data and alignment as `hw_run1_shape_comparison`. Heights relative to the common mean '
             f'plane, shown on a 2 mm grid, drawn with vertical exaggeration ×{a.exaggeration:g}; read height values from the colour bars. '
             f'(a) Scan; (b) blind prediction ({model}); (a) and (b) share one colour scale. (c) Predicted '
             f'surface coloured by the vertical difference scan − prediction (grid RMS {rms3d:.2f} mm).')
    prov = OUT / 'hw_run1_provenance.json'
    record = json.loads(prov.read_text()) if prov.exists() else {}
    inputs = [SOLVE / 'shape.npz', SOLVE / 'result.json', SOLVE / 'comparison/comparison.json',
              SOLVE / 'comparison/profiles.npz', rep.CSV_PATH, rep.SCAN_PATH]
    record.update({'card': 'cards/REPLICATE-HW1.md', 'script': str(Path(__file__).resolve().relative_to(ROOT)),
                   'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
                   'canvas_font_pt': FONT, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__})
    record.pop('script_sha256', None)
    canvas = {'toolpath': [14, 10.5], 'comparison': [14, 10], '3d': [14, 12]}
    names = {'toolpath': 'hw_run1_toolpath', 'comparison': 'hw_run1_shape_comparison', '3d': 'hw_run1_shape_3d'}
    figures = record.setdefault('figures', {})
    for key in a.only:
        figures[names[key]] = {'script_sha256': digest(Path(__file__)), 'canvas_in': canvas[key]}
        if key == '3d':
            figures[names[key]]['vertical_exaggeration'] = a.exaggeration
    record.pop('canvas_in', None)
    save_json(prov, record)
    print('written', OUT)


if __name__ == '__main__':
    main()
