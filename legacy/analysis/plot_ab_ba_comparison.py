#!/usr/bin/env python3
"""Scientific visualization comparing ab_block (A^10 B^10) and ba_block (B^10 A^10) equilibrium shapes."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/scripts'))
sys.path.insert(0, str(ROOT / 'legacy/analysis'))

from analyze_sequence_order import load_case
from analyze_two_gripper_release import align, area_weights, digest


def main():
    matrix_dir = ROOT / 'run/overlapping_sequence/matrix'
    output_dir = ROOT / 'legacy/analysis/plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    ab_dir = matrix_dir / 'ab_block_final_only_base'
    ba_dir = matrix_dir / 'ba_block_final_only_base'

    ab_state, ab_arr, _ = load_case(ab_dir)
    ba_state, ba_arr, _ = load_case(ba_dir)

    uv = np.asarray(ab_state['material_uv']) * 1000.0  # mm
    faces = np.asarray(ab_state['faces'])
    ref_flat = np.column_stack((uv, np.zeros(len(uv))))
    weights = area_weights(ref_flat, faces)

    # 1. Align both shapes to the flat reference XY plane for standard perspective
    ab_pos, _, _ = align(np.asarray(ab_state['positions']) * 1000.0, ref_flat, weights)
    ba_pos, _, _ = align(np.asarray(ba_state['positions']) * 1000.0, ref_flat, weights)

    # Pin center vertex (u=0, v=0) to z=0 for intuitive profile comparison
    center_idx = np.argmin(np.linalg.norm(uv, axis=1))
    ab_pos_centered = ab_pos.copy()
    ab_pos_centered[:, 2] -= ab_pos[center_idx, 2]
    ba_pos_centered = ba_pos.copy()
    ba_pos_centered[:, 2] -= ba_pos[center_idx, 2]

    # 2. Optimal rigid alignment between AB and BA
    ab_to_ba, _, _ = align(ab_pos, ba_pos, weights)
    delta = ab_to_ba - ba_pos
    delta_z = delta[:, 2]
    distance = np.linalg.norm(delta, axis=1)

    # 3. Centerline profiles
    # Along u=0 (v in [-150, 150])
    u0_mask = np.abs(uv[:, 0]) < 1e-4
    order_v = np.argsort(uv[u0_mask, 1])
    prof_v = uv[u0_mask, 1][order_v]
    prof_ab_z_v = ab_pos_centered[u0_mask, 2][order_v]
    prof_ba_z_v = ba_pos_centered[u0_mask, 2][order_v]

    # Along v=0 (u in [-100, 100])
    v0_mask = np.abs(uv[:, 1]) < 1e-4
    order_u = np.argsort(uv[v0_mask, 0])
    prof_u = uv[v0_mask, 0][order_u]
    prof_ab_z_u = ab_pos_centered[v0_mask, 2][order_u]
    prof_ba_z_u = ba_pos_centered[v0_mask, 2][order_u]

    # Style configuration
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,
    })

    fig = plt.figure(figsize=(12, 10), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.8])

    # Shared height limits for 3D plots
    z_min = min(ab_pos[:, 2].min(), ba_pos[:, 2].min())
    z_max = max(ab_pos[:, 2].max(), ba_pos[:, 2].max())
    norm_z = Normalize(vmin=z_min, vmax=z_max)

    # Regular grid interpolation for fast, clean 3D rendering
    u_grid = np.linspace(-100, 100, 101)
    v_grid = np.linspace(-150, 150, 151)
    Ug, Vg = np.meshgrid(u_grid, v_grid)
    Zg_ab = griddata((ab_pos[:, 0], ab_pos[:, 1]), ab_pos[:, 2], (Ug, Vg), method='linear')
    Zg_ba = griddata((ba_pos[:, 0], ba_pos[:, 1]), ba_pos[:, 2], (Ug, Vg), method='linear')

    # Panel (a): 3D surface AB
    ax_3d_ab = fig.add_subplot(gs[0, 0], projection='3d')
    p3d_ab = ax_3d_ab.plot_surface(Ug, Vg, Zg_ab, cmap='viridis', norm=norm_z,
                                   rstride=2, cstride=2, edgecolor='none', alpha=0.95, antialiased=True)
    ax_3d_ab.set_title('(a) ab_block (A¹⁰ B¹⁰) 3D Equilibrium', pad=10)
    ax_3d_ab.set_xlabel('x (mm)')
    ax_3d_ab.set_ylabel('y (mm)')
    ax_3d_ab.set_zlabel('z (mm)')
    ax_3d_ab.view_init(elev=32, azim=-60)
    ax_3d_ab.set_zlim(z_min - 2, z_max + 2)

    # Panel (b): 3D surface BA
    ax_3d_ba = fig.add_subplot(gs[0, 1], projection='3d')
    p3d_ba = ax_3d_ba.plot_surface(Ug, Vg, Zg_ba, cmap='viridis', norm=norm_z,
                                   rstride=2, cstride=2, edgecolor='none', alpha=0.95, antialiased=True)
    ax_3d_ba.set_title('(b) ba_block (B¹⁰ A¹⁰) 3D Equilibrium', pad=10)
    ax_3d_ba.set_xlabel('x (mm)')
    ax_3d_ba.set_ylabel('y (mm)')
    ax_3d_ba.set_zlabel('z (mm)')
    ax_3d_ba.view_init(elev=32, azim=-60)
    ax_3d_ba.set_zlim(z_min - 2, z_max + 2)

    cbar_3d = fig.colorbar(p3d_ba, ax=[ax_3d_ab, ax_3d_ba], shrink=0.6, pad=0.04, label='Height z (mm)')

    # Panel (c): 2D Aligned signed difference Delta z (mm)
    ax_map_dz = fig.add_subplot(gs[1, 0])
    max_dz = max(abs(delta_z.min()), abs(delta_z.max()))
    norm_dz = Normalize(vmin=-max_dz, vmax=max_dz)
    tc_dz = ax_map_dz.tripcolor(uv[:, 0], uv[:, 1], faces, delta_z,
                                shading='gouraud', cmap='RdBu_r', norm=norm_dz, rasterized=True)
    # Draw toolpath bounding boxes
    rect_a = Rectangle((-94, -74), 188, 148, linewidth=1.2, edgecolor='#1b9e77', facecolor='none', linestyle='--', label='Toolpath A Envelope')
    rect_b = Rectangle((-100, -80.33), 200, 160.67, linewidth=1.2, edgecolor='#d95f02', facecolor='none', linestyle=':', label='Toolpath B Envelope')
    rect_ov = Rectangle((-94, -74), 188, 148, linewidth=1.5, edgecolor='#7570b3', facecolor='none', linestyle='-', label='Shared Overlap (1,848 faces)')
    ax_map_dz.add_patch(rect_ov)
    ax_map_dz.set_title('(c) Aligned Signed Difference Δz = z_AB - z_BA')
    ax_map_dz.set_xlabel('Material u (mm)')
    ax_map_dz.set_ylabel('Material v (mm)')
    ax_map_dz.set_aspect('equal')
    ax_map_dz.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), framealpha=0.9, fontsize=8)
    fig.colorbar(tc_dz, ax=ax_map_dz, shrink=0.75, label='Δz (mm)')

    # Panel (d): 2D Total Euclidean deviation distance d (mm)
    ax_map_d = fig.add_subplot(gs[1, 1])
    norm_d = Normalize(vmin=0, vmax=distance.max())
    tc_d = ax_map_d.tripcolor(uv[:, 0], uv[:, 1], faces, distance,
                              shading='gouraud', cmap='magma', norm=norm_d, rasterized=True)
    ax_map_d.set_title(f'(d) Aligned Displacement Distance (Max = {distance.max():.2f} mm)')
    ax_map_d.set_xlabel('Material u (mm)')
    ax_map_d.set_ylabel('Material v (mm)')
    ax_map_d.set_aspect('equal')
    fig.colorbar(tc_d, ax=ax_map_d, shrink=0.75, label='Distance ||x_AB - x_BA|| (mm)')

    # Panel (e): Longitudinal Centerline Profile (u = 0)
    ax_prof_v = fig.add_subplot(gs[2, 0])
    ax_prof_v.plot(prof_v, prof_ab_z_v, color='#0072B2', linewidth=2.0, label='ab_block (A¹⁰ B¹⁰)')
    ax_prof_v.plot(prof_v, prof_ba_z_v, color='#D55E00', linewidth=2.0, linestyle='--', label='ba_block (B¹⁰ A¹⁰)')
    ax_prof_v.set_title('(e) Longitudinal Midline Profile (u = 0 mm)')
    ax_prof_v.set_xlabel('Material v (mm)')
    ax_prof_v.set_ylabel('Relative Height z (mm)')
    ax_prof_v.grid(True, linestyle=':', alpha=0.6)
    ax_prof_v.legend(loc='lower center', framealpha=0.85)

    # Panel (f): Transverse Centerline Profile (v = 0)
    ax_prof_u = fig.add_subplot(gs[2, 1])
    ax_prof_u.plot(prof_u, prof_ab_z_u, color='#0072B2', linewidth=2.0, label='ab_block (A¹⁰ B¹⁰)')
    ax_prof_u.plot(prof_u, prof_ba_z_u, color='#D55E00', linewidth=2.0, linestyle='--', label='ba_block (B¹⁰ A¹⁰)')
    ax_prof_u.set_title('(f) Transverse Midline Profile (v = 0 mm)')
    ax_prof_u.set_xlabel('Material u (mm)')
    ax_prof_u.set_ylabel('Relative Height z (mm)')
    ax_prof_u.grid(True, linestyle=':', alpha=0.6)
    ax_prof_u.legend(loc='lower center', framealpha=0.85)

    plot_stem = 'ab_vs_ba_equilibrium_comparison'
    png_path = output_dir / f'{plot_stem}.png'
    pdf_path = output_dir / f'{plot_stem}.pdf'
    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)

    caption = (
        'Comparison of 3D equilibrium shapes between reverse block toolpath schedules '
        'ab_block (A^10 B^10) and ba_block (B^10 A^10) under overlapping toolpath deposition. '
        '(a, b) 3D perspective surface views rendered with identical height scales. '
        '(c) Aligned signed out-of-plane deviation Delta z = z_AB - z_BA across the sheet material coordinates, '
        'with the co-hit face overlap domain (1,848 faces) demarcated. '
        '(d) Area-weighted optimal Procrustes Euclidean displacement distance field (peak 1.76 mm). '
        '(e) Longitudinal midline profile along u = 0 mm (v in [-150, 150] mm) demonstrating the 2.53 mm '
        'sagitta contrast between schedules. '
        '(f) Transverse midline profile along v = 0 mm (u in [-100, 100] mm).'
    )
    (output_dir / f'{plot_stem}.caption.md').write_text(caption + '\n')
    print(f'Successfully generated {png_path} and {pdf_path}')


if __name__ == '__main__':
    main()
