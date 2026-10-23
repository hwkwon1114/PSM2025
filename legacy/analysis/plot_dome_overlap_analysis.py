#!/usr/bin/env python3
"""Plot analyzing why 3-toolpath BO designs max out at ~55% dome coverage and saddle mechanics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
BO_DIR = ROOT / 'run/dome/bo_continuation_stage1'
OUT_DIR = ROOT / 'legacy/analysis/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_candidate(attempt_id):
    att_dir = BO_DIR / f'attempt_{attempt_id:03d}'
    obs = json.loads((att_dir / 'observation.json').read_text())
    seq = json.loads((att_dir / 'sequence.json').read_text())
    curv = np.load(att_dir / 'analysis/curvature_fields.npz')
    return obs, seq, curv


def main():
    attempts = [1, 20, 45]
    titles = [
        '(a) Attempt 001: Separated Strips (Cov = 33.9%)',
        '(b) Attempt 020: Concentrated Center Cross (Cov = 9.8%, 78.6% Saddle)',
        '(c) Attempt 045: Best BO Compromise (Cov = 55.1%, 53.9% Saddle)'
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), layout='constrained')

    for col, att_id in enumerate(attempts):
        obs, seq, curv = load_candidate(att_id)
        uv = curv['uv'] * 1000.0 # mm
        pos = curv['positions'] * 1000.0 # mm
        faces = curv['faces']
        geo = curv['geometric']
        k1 = geo[:, 0]
        k2 = geo[:, 1]
        K = k1 * k2 # Gaussian curvature

        face_uv = np.mean(uv[faces], axis=1)

        # 1. 3D Shape (row 0)
        ax3d = fig.add_subplot(3, 3, col + 1, projection='3d')
        axes[0, col].remove() # replace 2D with 3D
        u_grid = np.linspace(-100, 100, 81)
        v_grid = np.linspace(-150, 150, 101)
        Ug, Vg = np.meshgrid(u_grid, v_grid)
        Zg = griddata((pos[:, 0], pos[:, 1]), pos[:, 2], (Ug, Vg), method='linear')
        ax3d.plot_surface(Ug, Vg, Zg, cmap='viridis', rstride=2, cstride=2, edgecolor='none', alpha=0.9)
        ax3d.set_title(titles[col], fontsize=10, pad=5)
        ax3d.set_xlabel('x (mm)')
        ax3d.set_ylabel('y (mm)')
        ax3d.set_zlabel('z (mm)')
        ax3d.view_init(elev=28, azim=-55)

        # 2. Gaussian Curvature Sign Map (row 1)
        # Classify: Synclastic (Dome, K > 0 and same sign) = +1, Flat = 0, Saddle (K < 0) = -1
        curv_class = np.zeros(len(faces))
        syn_mask = (K > 0) & (np.abs(k1) >= 0.1) & (np.abs(k2) >= 0.1)
        sad_mask = (K < -1e-4)
        curv_class[syn_mask] = 1.0
        curv_class[sad_mask] = -1.0

        ax_curv = axes[1, col]
        tc = ax_curv.tripcolor(uv[:, 0], uv[:, 1], faces, facecolors=curv_class,
                               cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1), rasterized=True)
        ax_curv.set_title(f'Attempt {att_id:03d}: Curvature Classes\n(Blue=Saddle, Red=Dome)')
        ax_curv.set_xlabel('Material u (mm)')
        ax_curv.set_ylabel('Material v (mm)')
        ax_curv.set_aspect('equal')

        # 3. Toolpath Footprint Layout (row 2)
        ax_tool = axes[2, col]
        ax_tool.set_xlim(-105, 105)
        ax_tool.set_ylim(-155, 155)
        ax_tool.set_aspect('equal')
        ax_tool.set_xlabel('Material u (mm)')
        ax_tool.set_ylabel('Material v (mm)')
        ax_tool.set_title(f'Attempt {att_id:03d}: Toolpath Footprint')
        ax_tool.add_patch(Rectangle((-100, -150), 200, 300, fill=False, edgecolor='gray', linestyle=':'))

        # Draw toolpaths
        colors = ['#1b9e77', '#d95f02', '#7570b3']
        labels = ['A (Middle, 0°)', 'B (Top, 90°)', 'C (Bot, 90°)']
        for p_idx, path in enumerate(seq['toolpaths']):
            op = path['operation']
            lv = op['lv_mm']
            w = op['width_mm']
            n = op['n_strips']
            alpha = op['alpha_deg']
            rot = op['rotation_deg']
            shift = op['shift_uv_mm']

            step = lv * np.tan(np.radians(alpha))
            span_v = (n - 1) * step + w
            span_u = lv + w

            if rot == 90:
                span_u, span_v = span_v, span_u

            rect = Rectangle((shift[0] - span_u/2, shift[1] - span_v/2), span_u, span_v,
                             fill=True, alpha=0.35, facecolor=colors[p_idx], edgecolor=colors[p_idx],
                             label=labels[p_idx])
            ax_tool.add_patch(rect)
        if col == 0:
            ax_tool.legend(loc='lower left', fontsize=7, framealpha=0.85)

    plot_stem = 'dome_bo_overlap_mechanics_comparison'
    fig.savefig(OUT_DIR / f'{plot_stem}.png', dpi=200)
    fig.savefig(OUT_DIR / f'{plot_stem}.pdf')
    plt.close(fig)
    print(f'Successfully generated {OUT_DIR / plot_stem}.png and .pdf')


if __name__ == '__main__':
    main()
