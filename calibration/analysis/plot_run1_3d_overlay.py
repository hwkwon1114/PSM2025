"""Overlay saved Run 1 scan and model in 3-D; no solver or Run 2 access.

Fixed-frame model vs the previously recorded +7° post-hoc yaw counterfactual.
Static views and a self-contained interactive HTML use the same input surfaces.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.diagnose_run1_twist import load_case
from calibration.analysis.plot_run1_mismatch import counterfactual
from calibration.ewcal.geometry import read_mesh, points, shared_footprint_faces

SOURCE = ROOT / 'run/diagnostics/run1_twist_20261027/results.json'
OUT = ROOT / 'calibration/paper_figures/diagnostics'
STEM = 'run1_scan_model_3d_overlay'
SCAN, MODEL = '#0072B2', '#D55E00'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_segments(p, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    # NaN separator keeps disconnected boundaries and any holes separate.
    boundary = p[edges[counts == 1]]
    return np.concatenate([boundary, np.full((len(boundary), 1, 3), np.nan)], axis=1).reshape(-1, 3)


def rotate(p, degrees, offset):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    q = p.copy()
    q[:, 0] = c*p[:, 0] - s*p[:, 1]
    q[:, 1] = s*p[:, 0] + c*p[:, 1]
    q[:, 2] += offset
    return q


def grid_lines(p, faces, extent):
    """Piecewise-linear wires over each existing triangulated surface; no extrapolation."""
    tri = mtri.Triangulation(p[:, 0], p[:, 1], faces)
    interp = mtri.LinearTriInterpolator(tri, p[:, 2])
    xend, vend = extent
    lines = []
    for x in np.linspace(-xend, xend, 9):
        v = np.linspace(-vend, vend, 151)
        xx = np.full_like(v, x)
        lines.append(np.column_stack((xx, v, interp(xx, v).filled(np.nan))))
    for v in np.linspace(-vend, vend, 13):
        x = np.linspace(-xend, xend, 101)
        vv = np.full_like(x, v)
        lines.append(np.column_stack((x, vv, interp(x, vv).filled(np.nan))))
    return lines


def plot_wire(ax, lines, color, dashed=False):
    for line in lines:
        ax.plot(line[:, 0], line[:, 1] + 65, line[:, 2],
                color=color, linewidth=0.85, alpha=0.9,
                linestyle='--' if dashed else '-')


def plot_boundary(ax, p, faces, color, dashed=False):
    e = edge_segments(p, faces)
    ax.plot(e[:, 0], e[:, 1] + 65, e[:, 2], color=color, linewidth=1.3,
            linestyle='--' if dashed else '-')


def interactive_wire(lines):
    nan = np.full((1, 3), np.nan)
    q = np.concatenate([np.concatenate((line, nan), axis=0) for line in lines])
    return q[:, 0], q[:, 1] + 65, q[:, 2]


def main():
    for suffix in ('pdf', 'png', 'html'):
        p = OUT / f'{STEM}.{suffix}'
        if p.exists() or p.is_symlink():
            raise FileExistsError(f'Output already exists: {p}')
    rec = json.loads(SOURCE.read_text())
    assert rec['best_joint_attempt'] == 84
    ctrl = rec['controls']['fresh_best_joint']
    folder = ROOT / ctrl['path']
    comparison_path, mesh_path = folder / 'comparison.npz', folder / 'prediction_mm.vtp'
    assert sha(comparison_path) == ctrl['comparison_sha256']
    assert sha(mesh_path) == ctrl['yaw']['mesh_sha256']
    target, _, _ = load_case(folder)
    model_mesh = read_mesh(mesh_path)
    model = points(model_mesh)
    model_faces = vtk_to_numpy(model_mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    scan_faces = shared_footprint_faces(target[:, :2])

    common, _, residual7, angles, rms, i0, i7 = counterfactual(target, model_mesh, ctrl['yaw'])
    theta, offset = float(angles[i7]), float(ctrl['yaw']['posthoc_offset_mm'])
    assert theta == 7.0 and int(common.sum()) == 8623
    rotated = rotate(model, theta, offset)
    rotated_tri = mtri.Triangulation(rotated[:, 0], rotated[:, 1], model_faces)
    rotated_height = mtri.LinearTriInterpolator(rotated_tri, rotated[:, 2])
    check = rotated_height(target[common, 0], target[common, 1]).filled(np.nan)
    assert np.isfinite(check).all()
    assert np.allclose(target[common, 2] - check, residual7[common], atol=1e-8, rtol=1e-8)
    assert np.isclose(np.sqrt(np.mean((target[common, 2] - check)**2)), rms[i7], atol=1e-10)

    scan_wires = grid_lines(target, scan_faces, (94, 144))
    model_wires = grid_lines(model, model_faces, (96, 146))
    rotated_wires = [rotate(line, theta, offset) for line in model_wires]
    heights = np.concatenate((target[:, 2], model[:, 2], rotated[:, 2]))
    zlo, zhi = np.floor(heights.min()) - 1, np.ceil(heights.max()) + 1
    xrange, yrange, zrange = (-125, 125), (-110, 240), (float(zlo), float(zhi))
    dz = zhi - zlo
    OUT.mkdir(parents=True, exist_ok=True)

    # Equal data scales in x–z, one near-plan panel at true scale; oblique
    # panels exaggerate the height axis by 8x, without changing numeric labels.
    style = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
             'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 10,
             'pdf.fonttype': 42}
    views = [('Oblique · height ×8', 23, -65, 8),
             ('Opposite side · height ×8', 23, 115, 8),
             ('Near plan · true scale', 72, -90, 1)]
    with plt.style.context(style):
        fig = plt.figure(figsize=(11, 13), layout='constrained')
        # Reserve room for 3-D axes labels, which can extend beyond their boxes.
        fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.92))
        axes = fig.subplots(3, 2, subplot_kw={'projection': '3d'})
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0], [0], color=SCAN, lw=2, label='Scan (solid)'),
                            Line2D([0], [0], color=MODEL, lw=2, ls='--', label='Model (dashed)')],
                   loc='outside upper center', ncol=2, frameon=False)
        for row, (name, elev, azim, stretch) in enumerate(views):
            for col, (title, shape, wires) in enumerate((('Fixed frame', model, model_wires),
                                                          ('+7° post hoc', rotated, rotated_wires))):
                ax = axes[row, col]
                ax.plot_trisurf(target[:, 0], target[:, 1] + 65, target[:, 2],
                                triangles=scan_faces, color=SCAN, alpha=0.12,
                                linewidth=0, shade=False, rasterized=True)
                ax.plot_trisurf(shape[:, 0], shape[:, 1] + 65, shape[:, 2],
                                triangles=model_faces, color=MODEL, alpha=0.12,
                                linewidth=0, shade=False, rasterized=True)
                plot_wire(ax, scan_wires, SCAN)
                plot_wire(ax, wires, MODEL, dashed=True)
                plot_boundary(ax, target, scan_faces, SCAN)
                plot_boundary(ax, shape, model_faces, MODEL, dashed=True)
                ax.set_title(f'{title} — {name}', pad=3)
                ax.set(xlim=xrange, ylim=yrange, zlim=zrange,
                       xlabel='Width x (mm)', ylabel='Length z (mm)', zlabel='Height y (mm)')
                ax.set_xticks([-100, 0, 100])
                ax.set_yticks([-85, 65, 215])
                if stretch == 1:
                    # At true scale the 10 mm height axis is very short; its
                    # ticks would overlap and add nothing to a near-plan view.
                    ax.set_zticks([])
                    ax.set_zlabel('')
                    ax.zaxis.line.set_linewidth(0)
                else:
                    ax.set_zticks([round(zlo + 1), 0, round(zhi - 1)])
                ax.tick_params(pad=1)
                ax.view_init(elev=elev, azim=azim)
                ax.set_box_aspect((xrange[1]-xrange[0], yrange[1]-yrange[0], dz*stretch), zoom=1.18)
                ax.grid(False)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.pane.set_alpha(0.0)
        fig.savefig(OUT / f'{STEM}.pdf', dpi=220)
        fig.savefig(OUT / f'{STEM}.png', dpi=180)
        plt.close(fig)

    # Two separate rotatable scenes; WebGL geometry is the original triangulated
    # scan and saved model, not the coarse wireframe samples.
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=('Fixed frame: saved model', '+7° post hoc + height offset'))
    for col, (shape, wires) in enumerate(((model, model_wires), (rotated, rotated_wires)), 1):
        for label, surf, faces, color in (('Scan', target, scan_faces, SCAN),
                                          ('Model', shape, model_faces, MODEL)):
            fig.add_trace(go.Mesh3d(x=surf[:, 0], y=surf[:, 1]+65, z=surf[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    color=color, opacity=0.16, name=label,
                                    showlegend=col == 1, legendgroup=label,
                                    hoverinfo='name'), row=1, col=col)
        for label, lines, color in (('Scan', scan_wires, SCAN), ('Model', wires, MODEL)):
            x, y, z = interactive_wire(lines)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=label,
                                       legendgroup=label, showlegend=False,
                                       line={'color': color, 'width': 3,
                                             'dash': 'dash' if label == 'Model' else 'solid'},
                                       hoverinfo='skip'), row=1, col=col)
        # Boundaries use color; interior wires also distinguish line styles.
        for label, surf, faces, color in (('Scan edge', target, scan_faces, SCAN),
                                          ('Model edge', shape, model_faces, MODEL)):
            e = edge_segments(surf, faces)
            fig.add_trace(go.Scatter3d(x=e[:, 0], y=e[:, 1]+65, z=e[:, 2],
                                       mode='lines', name=label, showlegend=False,
                                       line={'color': color, 'width': 4}, hoverinfo='skip'), row=1, col=col)
    scene = {'xaxis': {'title': 'Width x (mm)', 'range': list(xrange)},
             'yaxis': {'title': 'Length z (mm)', 'range': list(yrange)},
             'zaxis': {'title': 'Height y (mm)', 'range': list(zrange)},
             'aspectmode': 'manual',
             'aspectratio': {'x': xrange[1]-xrange[0], 'y': yrange[1]-yrange[0], 'z': float(dz*8)},
             'camera': {'eye': {'x': 1.4, 'y': -1.7, 'z': 0.9}}}
    fig.update_layout(scene=scene, scene2=scene, width=1400, height=720,
                      title={'text': 'Run 1 overlay — height exaggerated ×8; rotate each view by dragging', 'x': 0.5},
                      legend={'orientation': 'h', 'y': 0.97, 'x': 0.42},
                      margin={'l': 0, 'r': 0, 'b': 0, 't': 100})
    html = fig.to_html(full_html=True, include_plotlyjs=True, config={'responsive': True})
    with (OUT / f'{STEM}.html').open('x', encoding='utf-8') as f:
        f.write(html)
    print(json.dumps({'script_sha256': sha(Path(__file__)), 'source_sha256': sha(SOURCE),
                      'comparison_sha256': sha(comparison_path), 'mesh_sha256': sha(mesh_path),
                      'attempt': 84, 'scan_points': len(target), 'model_vertices': len(model),
                      'fixed_support_points': int(common.sum()), 'counterfactual_yaw_deg': theta,
                      'counterfactual_height_offset_mm': offset,
                      'centered_rms_mm': [float(rms[i0]), float(rms[i7])],
                      'display_limits': {'x_mm': xrange, 'z_mm': yrange, 'height_y_mm': zrange},
                      'outputs': [str(OUT / f'{STEM}.{e}') for e in ('pdf', 'png', 'html')]}, indent=2))


if __name__ == '__main__':
    main()
