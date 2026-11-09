"""3D visualization for Run 3: scanned shape Run_3.ply and Attempt 209 prediction.

Usage:
  python shape_design/plot_hw3_3d.py [--solve run/shape_design/hw_run3/solve_mesh2] [--exaggeration 8]
Public frame: x width, z length (v + 65 mm), y height.
Writes:
  <solve>/comparison/hw_run3_shape_3d.png
  <solve>/comparison/hw_run3_shape_3d.pdf
  <solve>/comparison/hw_run3_shape_3d_interactive.html
  <solve>/comparison/run3_scan_only_3d.png
  <solve>/comparison/run3_scan_only_interactive.html
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
from shape_design.plot_hw3_comparison import Z_OFFSET, grid_heights

STEM = 'hw_run3_shape_3d'


def fields(solve, step=1.5):
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
    fig = plt.figure(figsize=(6.2, 7.5))
    panels = [
        (h_scan, h_scan, 'RdBu_r', hnorm, '(a) Physical Scan (Run_3.ply)'),
        (h_pred, h_pred, 'RdBu_r', hnorm, '(b) Simulation Prediction (Attempt 209)'),
        (h_pred, diff, 'PuOr_r', dnorm, '(c) Prediction colored by residual (Scan − Prediction)')
    ]
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

    # Also make a dedicated single 3D plot of Run_3.ply
    fig_scan = plt.figure(figsize=(7.0, 5.0))
    ax_scan = fig_scan.add_subplot(111, projection='3d', computed_zorder=False)
    face_scan = plt.get_cmap('viridis')(norm(np.nan_to_num(h_scan)))
    ax_scan.plot_surface(X, Z, h_scan, facecolors=face_scan, rstride=1, cstride=1, linewidth=0,
                         antialiased=True, shade=True, rasterized=True)
    ax_scan.set_box_aspect((220, 124, 2 * lim * exag), zoom=1.15)
    ax_scan.set(xlim=(-110, 110), ylim=(3, 127), zlim=(-lim, lim),
                xlabel='x: width (mm)', ylabel='z: length/feed (mm)')
    ax_scan.set_zlabel('y: height (mm)', labelpad=4)
    ax_scan.view_init(elev=32, azim=-55)
    ax_scan.set_title(f'Run_3.ply Scanned 3D Surface (Vertical Exaggeration ×{exag:g})', fontsize=10, pad=12)
    mappable_scan = cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = fig_scan.colorbar(mappable_scan, ax=ax_scan, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Height y (mm)')
    for ext in ('pdf', 'png'):
        fig_scan.savefig(out / f'run3_scan_only_3d.{ext}', dpi=300)
    plt.close(fig_scan)

    return float(np.sqrt(np.nanmean(diff ** 2)))


def interactive(X, Z, h_scan, h_pred, exag, out):
    lim = float(np.nanmax(np.abs([h_scan, h_pred])))
    kw = dict(x=X, y=Z, cmin=-lim, cmax=lim, colorscale='RdBu_r')

    # Interactive 1: Comparison between Scan and Prediction
    fig = go.Figure([
        go.Surface(z=h_scan, name='Scan (Run_3.ply)', showscale=True,
                   colorbar=dict(title='Height y (mm)'), **kw),
        go.Surface(z=h_pred, name='Prediction', showscale=False, opacity=1.0, visible=False, **kw),
        go.Surface(z=h_pred, name='Prediction (grey overlay)', showscale=False, opacity=0.55, visible=False,
                   x=X, y=Z, colorscale=[[0, '#555555'], [1, '#555555']]),
    ])
    show = lambda a, b, c: [{'visible': [a, b, c]}]
    fig.update_layout(
        title=f'Run 3: Scanned Surface (Run_3.ply) vs Attempt 209 Prediction (Vertical exaggeration ×{exag:g})',
        scene=dict(xaxis_title='x (width, mm)', yaxis_title='z (length, mm)', zaxis_title='y (height, mm)',
                   aspectmode='manual', aspectratio=dict(x=1, y=124 / 220, z=2 * lim * exag / 220),
                   camera=dict(eye=dict(x=-1.2, y=-1.4, z=0.9))),
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.08, xanchor='left', buttons=[
            dict(label='Scan (Run_3.ply)', method='update', args=show(True, False, False)),
            dict(label='Simulation Prediction', method='update', args=show(False, True, False)),
            dict(label='Overlay (Scan + Prediction)', method='update', args=show(True, False, True))])],
        margin=dict(l=0, r=0, t=60, b=0))
    fig.write_html(out / f'{STEM}_interactive.html', include_plotlyjs=True)

    # Interactive 2: Dedicated Scan-only 3D view
    fig_scan = go.Figure([
        go.Surface(z=h_scan, name='Run_3.ply', showscale=True,
                   colorbar=dict(title='Height y (mm)'),
                   x=X, y=Z, colorscale='Viridis', cmin=-lim, cmax=lim)
    ])
    fig_scan.update_layout(
        title=f'Run_3.ply Physical 3D Scanned Surface (Vertical exaggeration ×{exag:g})',
        scene=dict(xaxis_title='x: width (mm)', yaxis_title='z: length (mm)', zaxis_title='y: height (mm)',
                   aspectmode='manual', aspectratio=dict(x=1, y=124 / 220, z=2 * lim * exag / 220),
                   camera=dict(eye=dict(x=-1.2, y=-1.4, z=0.9))),
        margin=dict(l=0, r=0, t=50, b=0))
    fig_scan.write_html(out / 'run3_scan_only_interactive.html', include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solve', type=Path, default=ROOT / 'run/shape_design/hw_run3/solve_mesh2')
    ap.add_argument('--exaggeration', type=float, default=8.0)
    a = ap.parse_args()
    out = a.solve / 'comparison'
    summary = json.loads((out / 'comparison.json').read_text())
    X, Z, h_scan, h_pred = fields(a.solve, 1.5)
    rms = static_figure(X, Z, h_scan, h_pred, a.exaggeration, out)
    interactive(X, Z, h_scan, h_pred, a.exaggeration, out)
    print(f'Generated 3D visualizations in {out}:')
    print(' ', out / f'{STEM}.png')
    print(' ', out / 'run3_scan_only_3d.png')
    print(' ', out / f'{STEM}_interactive.html')
    print(' ', out / 'run3_scan_only_interactive.html')


if __name__ == '__main__':
    main()
