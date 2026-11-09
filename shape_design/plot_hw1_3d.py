"""3D comparison for card REPLICATE-HW1: scanned vs Attempt 209-predicted shape (Hyunwoo Run 1, 24 passes).

Usage: python shape_design/plot_hw1_3d.py [--solve run/shape_design/hw_run1/solve_mesh2] [--exaggeration 10]
Uses the saved rigid alignment from compare_hw1 (nothing refitted). Public frame: x width, z length (v + 65 mm),
y height. Heights are relative to the common mean plane and drawn with a stated vertical exaggeration.
Writes <solve>/comparison/hw_run1_shape_3d.{pdf,png,caption.md} and hw_run1_shape_3d_interactive.html.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm, colors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.common import digest
from shape_design.plot_hw1_comparison import Z_OFFSET, grid_heights

STEM = 'hw_run1_shape_3d'


def fields(solve, step):
    prof = np.load(solve / 'comparison/profiles.npz')
    pred = np.load(solve / 'shape.npz')['aligned_mm']
    U, V, h_pred = grid_heights(pred, step)
    _, _, h_scan = grid_heights(prof['scan_in_pred_frame_mm'], step)
    both = np.isfinite(h_pred) & np.isfinite(h_scan)
    h_pred = np.where(both, h_pred - h_pred[both].mean(), np.nan)
    h_scan = np.where(both, h_scan - h_scan[both].mean(), np.nan)
    return U, V + Z_OFFSET, h_scan, h_pred


def static_figure(X, Z, h_scan, h_pred, exag, out):
    diff = h_scan - h_pred
    lim = float(np.nanmax(np.abs([h_scan, h_pred])))
    dl = float(np.nanmax(np.abs(diff)))
    hnorm = colors.Normalize(-lim, lim)
    dnorm = colors.Normalize(-dl, dl)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(6.0, 7.2))
    panels = [(h_scan, h_scan, 'RdBu_r', hnorm, '(a) Scan'),
              (h_pred, h_pred, 'RdBu_r', hnorm, '(b) Prediction'),
              (h_pred, diff, 'PuOr_r', dnorm, '(c) Prediction, coloured by scan − prediction')]
    for k, (height, colour, cmap, norm, title) in enumerate(panels):
        bottom = 0.675 - 0.33 * k
        ax = fig.add_axes([-0.04, bottom - 0.01, 0.9, 0.32], projection='3d', computed_zorder=False)
        face = plt.get_cmap(cmap)(norm(np.nan_to_num(colour)))
        ax.plot_surface(X, Z, height, facecolors=face, rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False, rasterized=True)
        ax.set_box_aspect((220, 124, 2 * lim * exag), zoom=1.25)
        ax.set(xlim=(-110, 110), ylim=(3, 127), zlim=(-lim, lim), xlabel='x (mm)', ylabel='z (mm)')
        ax.set_zlabel('y (mm)', labelpad=-2)
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([13, 65, 117]); ax.set_zticks([-1.5, 0, 1.5])
        ax.view_init(elev=28, azim=-60)
        ax.text2D(0.08, 0.86, title, transform=ax.transAxes, fontsize=8)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([0.87, bottom + 0.06, 0.018, 0.2])
        fig.colorbar(mappable, cax=cax, label='Δy (mm)' if k == 2 else 'Height y (mm)')
    fig.text(0.86, 0.985, f'Vertical exaggeration ×{exag:g}', ha='right', va='top', fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    return float(np.sqrt(np.nanmean(diff ** 2)))


def interactive(X, Z, h_scan, h_pred, exag, out):
    lim = float(np.nanmax(np.abs([h_scan, h_pred])))
    kw = dict(x=X, y=Z, cmin=-lim, cmax=lim, colorscale='RdBu_r')
    fig = go.Figure([
        go.Surface(z=h_pred, name='Prediction', showscale=True,
                   colorbar=dict(title='Height y (mm)'), **kw),
        go.Surface(z=h_scan, name='Scan', showscale=False, opacity=1.0, visible=False, **kw),
        go.Surface(z=h_scan, name='Scan (grey overlay)', showscale=False, opacity=0.55, visible=False,
                   x=X, y=Z, colorscale=[[0, '#555555'], [1, '#555555']]),
    ])
    show = lambda a, b, c: [{'visible': [a, b, c]}]
    fig.update_layout(
        title=f'Hyunwoo Run 1: Attempt 209 prediction vs scan (vertical exaggeration ×{exag:g})',
        scene=dict(xaxis_title='x (mm)', yaxis_title='z (mm)', zaxis_title='y (mm)',
                   aspectmode='manual', aspectratio=dict(x=1, y=124 / 220, z=2 * lim * exag / 220),
                   camera=dict(eye=dict(x=-1.1, y=-1.5, z=0.9))),
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.08, xanchor='left', buttons=[
            dict(label='Prediction', method='update', args=show(True, False, False)),
            dict(label='Scan', method='update', args=show(False, True, False)),
            dict(label='Overlay', method='update', args=show(True, False, True))])],
        margin=dict(l=0, r=0, t=60, b=0))
    fig.write_html(out / f'{STEM}_interactive.html', include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solve', type=Path, default=ROOT / 'run/shape_design/hw_run1/solve_mesh2')
    ap.add_argument('--exaggeration', type=float, default=10.0)
    a = ap.parse_args()
    out = a.solve / 'comparison'
    summary = json.loads((out / 'comparison.json').read_text())
    if summary['inputs']['shape_npz_sha256'] != digest(a.solve / 'shape.npz'):
        raise ValueError('shape.npz changed since comparison')
    X, Z, h_scan, h_pred = fields(a.solve, 2.0)
    rms = static_figure(X, Z, h_scan, h_pred, a.exaggeration, out)
    interactive(X, Z, h_scan, h_pred, a.exaggeration, out)
    best = summary['orientations'][summary['best_orientation']]
    (out / f'{STEM}.caption.md').write_text(
        f"""**Hyunwoo Run 1 (24 passes): 3D view of the scanned and Attempt 209-predicted shapes.** Card REPLICATE-HW1.
Heights y are relative to the common mean plane, linearly interpolated to a 2 mm grid, and drawn with
**vertical exaggeration ×{a.exaggeration:g}**. (a) Scan, placed by the saved rigid alignment
(`{summary['best_orientation']}`; nothing fitted). (b) Blind prediction (frozen Attempt 209, 0.61 mm sheet,
back-face passes with swapped layer parameters, accepted 2 mm solve). (a) and (b) share one colour scale.
(c) Predicted surface coloured by the vertical difference scan − prediction (grid RMS {rms:.3f} mm).
Symmetric surface RMS {best['symmetric_rms_mm']:.3f} mm. Public frame: x width, z length (v + 65 mm), y height.
Single specimen, single prediction. Interactive version: `{STEM}_interactive.html` (buttons: prediction, scan,
overlay with the scan in translucent grey). Script {digest(Path(__file__))[:12]}.
""")
    print(out / f'{STEM}.png', rms)


if __name__ == '__main__':
    main()
