"""Figure for card REPLICATE-HW1: scanned vs Attempt 209-predicted shape of Hyunwoo Run 1 (24 passes).

Usage: python shape_design/plot_hw1_comparison.py [--solve run/shape_design/hw_run1/solve_mesh2]
Public frame: x = width (u), z = length/feed (v + 65 mm), y = height. Writes into <solve>/comparison/.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.common import digest
from shape_design import compare_hw1 as cmp
from shape_design.plot_style import format_sheet_ax

Z_OFFSET = 65.0
STEM = 'hw_run1_shape_comparison'
REGION_FACE = {'Region 1': 'front', 'Region 3': 'back', 'Region 2': 'front'}


def grid_heights(p, step=1.0):
    u = np.arange(-109.5, 110, step)
    v = np.arange(-61.5, 62, step)
    U, V = np.meshgrid(u, v, indexing='xy')
    return U, V, LinearNDInterpolator(p[:, :2], p[:, 2])(U, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solve', type=Path, default=ROOT / 'run/shape_design/hw_run1/solve_mesh2')
    a = ap.parse_args()
    out = a.solve / 'comparison'
    summary = json.loads((out / 'comparison.json').read_text())
    if summary['inputs']['shape_npz_sha256'] != digest(a.solve / 'shape.npz'):
        raise ValueError('shape.npz changed since comparison')
    prof = np.load(out / 'profiles.npz')
    pred = np.load(a.solve / 'shape.npz')['aligned_mm']
    scan = prof['scan_in_pred_frame_mm']

    U, V, h_pred = grid_heights(pred)
    _, _, h_scan = grid_heights(scan)
    both = np.isfinite(h_pred) & np.isfinite(h_scan)
    h_pred = np.where(both, h_pred - h_pred[both].mean(), np.nan)
    h_scan = np.where(both, h_scan - h_scan[both].mean(), np.nan)
    diff = h_scan - h_pred
    vertical_rms = float(np.sqrt(np.nanmean(diff ** 2)))

    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.2, 4.8), layout='constrained')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    lim = float(np.nanmax(np.abs([h_scan, h_pred])))
    Z = V + Z_OFFSET
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, field, title in zip(axes[:2], (h_scan, h_pred), ('(a) Scan', '(b) Prediction')):
        im = ax.pcolormesh(U, Z, field, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto', rasterized=True)
        format_sheet_ax(ax, title)
        ax.set_xlabel('x (mm)')
    axes[0].set_ylabel('z (mm)')
    fig.colorbar(im, ax=axes[:2], label='Height y (mm)', orientation='horizontal', shrink=0.6, aspect=30)
    dl = float(np.nanmax(np.abs(diff)))
    im = axes[2].pcolormesh(U, Z, diff, cmap='PuOr_r', vmin=-dl, vmax=dl, shading='auto', rasterized=True)
    format_sheet_ax(axes[2], '(c) Scan − prediction')
    axes[2].set_xlabel('x (mm)')
    fig.colorbar(im, ax=axes[2], label='Δy (mm)', orientation='horizontal', shrink=0.9, aspect=15)

    ax = fig.add_subplot(gs[1, :])
    for name, (lo, hi) in cmp.REGIONS_U.items():
        ax.axvspan(lo, hi, color='#0072B2' if REGION_FACE[name] == 'front' else '#D55E00', alpha=0.08, lw=0)
        ax.text((lo + hi) / 2, 0.97, f'{name}\n{REGION_FACE[name]} face', transform=ax.get_xaxis_transform(),
                ha='center', va='top')
    ax.plot(prof['u_mm'], prof['profile_scan_mm'], 'o-', color='black', ms=3, lw=1, label='Scan')
    ax.plot(prof['u_mm'], prof['profile_pred_mm'], 's--', color='#009E73', ms=3, lw=1, label='Prediction')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set(xlim=(-110, 110), xlabel='x (mm)', ylabel='Mean height y (mm)', title='(d) Width profile, |z − 65| ≤ 40 mm')
    ax.set_ylim(top=ax.get_ylim()[1] + 0.9)
    ax.legend(loc='lower left', frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{STEM}.{ext}', dpi=300)

    best = summary['orientations'][summary['best_orientation']]
    cap = f"""**Hyunwoo Run 1 (24 passes): scanned shape vs blind Attempt 209 prediction.** Card REPLICATE-HW1.
Sheet 220 × 124 × 0.61 mm; frozen Attempt 209 parameters; back-face (Region 3) passes swap the worked/unworked
layer parameters; one 2 mm-mesh equilibrium after all 24 passes (accepted: gradient target, positive pivots,
0/13,640 hit-count mismatches). The scan is placed by rigid alignment only (best of four proper orientations,
`{summary['best_orientation']}`); nothing is fitted to it. (a, b) Height y relative to the mean plane, on a shared
colour scale, linearly interpolated to a 1 mm grid. (c) Vertical difference scan − prediction (RMS {vertical_rms:.3f} mm).
(d) Mean height over |z − 65| ≤ 40 mm in 5 mm bins along x (mean removed); shading marks the worked regions.
Symmetric surface RMS {best['symmetric_rms_mm']:.3f} mm (forward {best['forward_rms_mm']:.3f}, backward
{best['backward_rms_mm']:.3f}); profile peak-to-valley prediction/scan = {summary['profile_ptv_ratio_pred_over_scan']:.3f}
({summary['profile_ptv_pred_mm']:.2f} vs {summary['profile_ptv_scan_mm']:.2f} mm). Public frame: x width, z length
(v + 65 mm), y height. Single specimen, single prediction; not a statistical validation.
Inputs: shape.npz {summary['inputs']['shape_npz_sha256'][:12]}, scan {summary['inputs']['scan_sha256'][:12]};
script {digest(Path(__file__))[:12]}.
"""
    (out / f'{STEM}.caption.md').write_text(cap)
    print(out / f'{STEM}.png', 'vertical_rms', vertical_rms)


if __name__ == '__main__':
    main()
