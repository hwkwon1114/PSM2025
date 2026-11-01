"""Visualize the saved Run 1 shape mismatch and post-hoc yaw diagnostic.

Uses the fixed a0084 snapshot and saved Run 1 data only. This does not change
registration, objective, campaign files, or solver settings, and does not load
Run 2. The counterfactual is explicitly not a new fit or the 3-D objective.
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
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.diagnose_run1_twist import load_case
from calibration.ewcal.geometry import points, read_mesh, shared_footprint_faces

SOURCE = ROOT / 'run/diagnostics/run1_twist_20261027/results.json'
OUT = ROOT / 'calibration/paper_figures/diagnostics'
STEM = 'run1_shape_and_yaw_diagnostic'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counterfactual(target, mesh, yaw):
    """Reproduce stored fixed-support, height-only diagnostic before plotting."""
    p = points(mesh)
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    tri = mtri.Triangulation(p[:, 0], p[:, 1], faces)
    interp = mtri.LinearTriInterpolator(tri, p[:, 2])
    xy = target[:, :2]
    angles = np.asarray(yaw['angle_grid_deg'])
    heights = []
    for theta in np.deg2rad(angles):
        c, s = np.cos(theta), np.sin(theta)
        # Inverse query for a positive in-plane rotation of the model mesh.
        heights.append(interp(c * xy[:, 0] + s * xy[:, 1],
                              -s * xy[:, 0] + c * xy[:, 1]).filled(np.nan))
    heights = np.asarray(heights)
    common = ((np.abs(xy[:, 0]) <= 75) & (np.abs(xy[:, 1]) <= 115)
              & np.all(np.isfinite(heights), axis=0))
    assert common.sum() == yaw['common_points'] and len(common) == yaw['all_scan_points']
    residuals = target[common, 2][None, :] - heights[:, common]
    centered = residuals - residuals.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1))
    assert np.allclose(rms, yaw['height_rms_centered_by_angle_mm'], rtol=1e-10, atol=1e-10)
    i0 = np.flatnonzero(angles == 0).item()
    ibest = np.argmin(rms)
    assert angles[ibest] == yaw['posthoc_min_yaw_deg']
    z = np.full(len(target), np.nan)
    z[common] = centered[i0]
    best = np.full(len(target), np.nan)
    best[common] = centered[ibest]
    return common, z, best, angles, rms, i0, ibest


def main():
    record = json.loads(SOURCE.read_text())
    selected = record['controls']['fresh_best_joint']
    assert record['best_joint_attempt'] == 84
    folder = ROOT / selected['path']
    comparison_path = folder / 'comparison.npz'
    mesh_path = folder / 'prediction_mm.vtp'
    assert sha(comparison_path) == selected['comparison_sha256']
    assert sha(mesh_path) == selected['yaw']['mesh_sha256']
    target, closest, distance = load_case(folder)
    mesh = read_mesh(mesh_path)
    common, res0, resbest, angles, curve, i0, ibest = counterfactual(target, mesh, selected['yaw'])

    # Restrict plotted triangles to the actual measured/shared footprint;
    # interior panels have identical support at every rotation angle.
    faces = shared_footprint_faces(target[:, :2])
    xy = target[:, :2]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1] + 65, faces)
    interior_faces = faces[np.all(common[faces], axis=1)]
    interior_tri = mtri.Triangulation(xy[:, 0], xy[:, 1] + 65, interior_faces)
    residual = target[:, 2] - closest[:, 2]
    height_lo = np.floor(min(target[:, 2].min(), closest[:, 2].min()))
    height_hi = np.ceil(max(target[:, 2].max(), closest[:, 2].max()))
    lim = np.ceil(2 * max(np.max(np.abs(residual)), np.nanmax(np.abs(res0)),
                          np.nanmax(np.abs(resbest)))) / 2
    heights = matplotlib.colors.Normalize(height_lo, height_hi)
    errors = matplotlib.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    metric = next(r['run1_rms_mm'] for r in record['records'] if r['attempt'] == 84)

    # Match the project's existing calibration-figure typography.
    style = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
             'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False}
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [OUT / f'{STEM}.{ext}' for ext in ('pdf', 'png')]
    if any(p.exists() or p.is_symlink() for p in outputs):
        raise FileExistsError('Diagnostic figure already exists; choose a new destination')
    with plt.style.context(style):
        fig = plt.figure(figsize=(10.0, 7.7), layout='constrained')
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.055])
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
        hbar, ebar = fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])
        common_kw = dict(shading='gouraud', rasterized=True)
        img_h = axes[0].tripcolor(tri, target[:, 2], cmap='viridis', norm=heights, **common_kw)
        axes[1].tripcolor(tri, closest[:, 2], cmap='viridis', norm=heights, **common_kw)
        img_e = axes[2].tripcolor(tri, residual, cmap='RdBu_r', norm=errors, **common_kw)
        axes[3].tripcolor(interior_tri, np.nan_to_num(res0), cmap='RdBu_r', norm=errors, **common_kw)
        axes[4].tripcolor(interior_tri, np.nan_to_num(resbest), cmap='RdBu_r', norm=errors, **common_kw)
        for ax, title in zip(axes[:5], ('(a) Scan height', '(b) Closest model height',
                                        '(c) Closest-point Δy', '(d) Zero yaw: centered',
                                        '(e) +7° post hoc: centered')):
            ax.set_title(title, loc='left', pad=5)
            ax.set(xlim=(-100, 100), ylim=(-85, 215), xlabel='Width x (mm)',
                   ylabel='Length z (mm)')
            ax.set_xticks([-100, 0, 100])
            ax.set_yticks([-85, 65, 215])
            ax.set_aspect('equal')
        box = dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2)
        axes[2].text(0.04, 0.04, f'3-D symmetric RMS\n{metric:.3f} mm',
                     transform=axes[2].transAxes, fontsize=8, bbox=box, va='bottom')
        for ax, value in ((axes[3], curve[i0]), (axes[4], curve[ibest])):
            ax.text(0.04, 0.04, f'Centered RMS\n{value:.3f} mm',
                    transform=ax.transAxes, fontsize=8, bbox=box, va='bottom')
        fig.colorbar(img_h, cax=hbar, label='Height y (mm)')
        fig.colorbar(img_e, cax=ebar, label='Height residual Δy (mm)')

        ax = axes[5]
        ax.set_title('(f) Post-hoc orientation check', loc='left', pad=5)
        ax.plot(angles, curve, color='#333333', linewidth=1.8)
        ax.scatter([angles[i0], angles[ibest]], [curve[i0], curve[ibest]],
                   color=['#0072B2', '#D55E00'], s=35, zorder=4)
        ax.axvline(0, color='#0072B2', linewidth=1, linestyle=':')
        ax.axvline(angles[ibest], color='#D55E00', linewidth=1, linestyle='--')
        ax.set(xlim=(angles.min(), angles.max()), ylim=(0, np.ceil(curve.max()*10)/10),
               xlabel='Model yaw (degrees)', ylabel='Centered vertical RMS (mm)')
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.grid(axis='y', color='0.88', linewidth=0.6)
        fig.savefig(outputs[0], dpi=300)
        fig.savefig(outputs[1], dpi=220)
        plt.close(fig)

    print(json.dumps({'figure_pdf': str(outputs[0]), 'figure_png': str(outputs[1]),
                      'diagnostic_source_sha256': sha(SOURCE),
                      'comparison_sha256': sha(comparison_path), 'mesh_sha256': sha(mesh_path),
                      'script_sha256': sha(Path(__file__)), 'candidate': 'a0084',
                      'run1_symmetric_rms_mm': metric, 'scan_points': len(target),
                      'fixed_interior_points': int(common.sum()),
                      'yaw_zero_centered_rms_mm': float(curve[i0]),
                      'yaw_selected_deg': float(angles[ibest]),
                      'yaw_selected_centered_rms_mm': float(curve[ibest]),
                      'full_sample_scan_to_model_rms_mm': float(np.sqrt(np.mean(distance**2))),
                      'height_scale_mm': [height_lo, height_hi], 'residual_limit_mm': lim}, indent=2))


if __name__ == '__main__':
    main()
