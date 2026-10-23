#!/usr/bin/env python3
"""Scientific visualization comparing all four overlapping deposition schedules:
A^10 B^10, (AB)^10, (BA)^10, and B^10 A^10.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/analysis'))

from analyze_two_gripper_release import align, area_weights


def main():
    res_dir = ROOT / 'legacy/analysis/overlapping_results'
    plot_dir = ROOT / 'legacy/analysis/plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ('ab_block_final_only_base', 'A¹⁰ B¹⁰ (AB Block)', '#0072B2', '-'),
        ('ab_alternate_final_only_base', '(AB)¹⁰ (AB Alternate)', '#56B4E9', '-.'),
        ('ba_alternate_final_only_base', '(BA)¹⁰ (BA Alternate)', '#E69F00', ':'),
        ('ba_block_final_only_base', 'B¹⁰ A¹⁰ (BA Block)', '#D55E00', '--'),
    ]

    data = {c: np.load(res_dir / f'{c}.npz') for c, _, _, _ in cases}
    uv = data['ab_block_final_only_base']['uv'] * 1000.0  # mm
    faces = data['ab_block_final_only_base']['faces']
    ref_flat = np.column_stack((uv, np.zeros(len(uv))))
    weights = area_weights(ref_flat, faces)
    c_idx = np.argmin(np.linalg.norm(uv, axis=1))

    # Standardized alignment to reference XY plane with center pinned to (0,0,0)
    aligned_pos = {}
    for c, _, _, _ in cases:
        pos_std, _, _ = align(data[c]['positions'] * 1000.0, ref_flat, weights)
        pos_std[:, 2] -= pos_std[c_idx, 2]
        aligned_pos[c] = pos_std

    # Grid for 3D surface rendering
    u_grid = np.linspace(-100, 100, 101)
    v_grid = np.linspace(-150, 150, 151)
    Ug, Vg = np.meshgrid(u_grid, v_grid)

    z_min = min(np.min(aligned_pos[c][:, 2]) for c, _, _, _ in cases)
    z_max = max(np.max(aligned_pos[c][:, 2]) for c, _, _, _ in cases)
    norm_z = Normalize(vmin=z_min, vmax=z_max)

    # Style configuration
    plt.rcParams.update({
        'font.size': 8.5,
        'axes.titlesize': 9.5,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 8.0,
        'pdf.fonttype': 42,
    })

    fig = plt.figure(figsize=(16, 9.5), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0])

    # --- ROW 1: 3D Surface Equilibrium Views ---
    surf_axes = []
    p_surf = None
    panel_letters = ['(a)', '(b)', '(c)', '(d)']
    for i, (c, name, _, _) in enumerate(cases):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        surf_axes.append(ax)
        pos = aligned_pos[c]
        Zg = griddata((pos[:, 0], pos[:, 1]), pos[:, 2], (Ug, Vg), method='linear')
        p_surf = ax.plot_surface(Ug, Vg, Zg, cmap='viridis', norm=norm_z,
                                 rstride=2, cstride=2, edgecolor='none', alpha=0.95, antialiased=True)
        ax.set_title(f'{panel_letters[i]} {name}', pad=8)
        ax.set_xlabel('x (mm)', labelpad=2)
        ax.set_ylabel('y (mm)', labelpad=2)
        ax.set_zlabel('z (mm)', labelpad=2)
        ax.view_init(elev=28, azim=-60)
        ax.set_zlim(z_min - 1, z_max + 1)

    cbar_3d = fig.colorbar(p_surf, ax=surf_axes, shrink=0.55, pad=0.02, label='Height z (mm)')

    # --- ROW 2: Profiles and Curvature Comparison ---
    # Centerline masks
    u0_mask = np.abs(uv[:, 0]) < 1e-4
    order_v = np.argsort(uv[u0_mask, 1])
    prof_v = uv[u0_mask, 1][order_v]

    v0_mask = np.abs(uv[:, 1]) < 1e-4
    order_u = np.argsort(uv[v0_mask, 0])
    prof_u = uv[v0_mask, 0][order_u]

    # Panel (e): Longitudinal Midline Profiles
    ax_lv = fig.add_subplot(gs[1, 0:2])
    for c, name, color, ls in cases:
        pos = aligned_pos[c]
        z_v = pos[u0_mask, 2][order_v]
        sagitta = np.ptp(z_v)
        ax_lv.plot(prof_v, z_v, color=color, linestyle=ls, linewidth=2.0,
                   label=f'{name} (Sagitta = {sagitta:.2f} mm)')
    ax_lv.set_title('(e) Longitudinal Midline Profile (u = 0 mm)')
    ax_lv.set_xlabel('Material v (mm)')
    ax_lv.set_ylabel('Height z (mm)')
    ax_lv.grid(True, linestyle=':', alpha=0.6)
    ax_lv.legend(loc='lower center', framealpha=0.9)

    # Panel (f): Transverse Midline Profiles
    ax_tu = fig.add_subplot(gs[1, 2])
    for c, name, color, ls in cases:
        pos = aligned_pos[c]
        z_u = pos[v0_mask, 2][order_u]
        ax_tu.plot(prof_u, z_u, color=color, linestyle=ls, linewidth=1.8, label=name)
    ax_tu.set_title('(f) Transverse Midline Profile (v = 0 mm)')
    ax_tu.set_xlabel('Material u (mm)')
    ax_tu.set_ylabel('Height z (mm)')
    ax_tu.grid(True, linestyle=':', alpha=0.6)
    ax_tu.legend(loc='lower center', framealpha=0.85, fontsize=7.5)

    # Panel (g): Primary Curvature & Sagitta Summary
    ax_bar = fig.add_subplot(gs[1, 3])
    labels = ['A¹⁰ B¹⁰', '(AB)¹⁰', '(BA)¹⁰', 'B¹⁰ A¹⁰']
    k1_vals = [0.8867, 0.9411, 0.9613, 1.0209]  # magnitude |k1| in m^-1
    sagitta_vals = [13.87, 14.98, 15.36, 16.40] # in mm
    bar_colors = ['#0072B2', '#56B4E9', '#E69F00', '#D55E00']

    x_pos = np.arange(len(labels))
    width = 0.38

    ax_bar_twin = ax_bar.twinx()
    b1 = ax_bar.bar(x_pos - width/2, k1_vals, width, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8, label='|κ₁| (m⁻¹)')
    b2 = ax_bar_twin.bar(x_pos + width/2, sagitta_vals, width, color=bar_colors, alpha=0.45, hatch='//', edgecolor='black', linewidth=0.8, label='Sagitta (mm)')

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=15)
    ax_bar.set_ylabel('Primary Curvature |κ₁| (m⁻¹)', color='#0072B2')
    ax_bar_twin.set_ylabel('Arch Sagitta (mm)', color='#D55E00')
    ax_bar.set_ylim(0.75, 1.10)
    ax_bar_twin.set_ylim(12.0, 18.0)
    ax_bar.set_title('(g) Curvature and Arch Sagitta Progression')
    ax_bar.grid(True, linestyle=':', alpha=0.5, axis='y')

    plot_stem = 'four_schedule_order_comparison'
    png_path = plot_dir / f'{plot_stem}.png'
    pdf_path = plot_dir / f'{plot_stem}.pdf'
    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)

    caption = (
        'Comprehensive comparison of the four overlapping deposition schedules: '
        'A^10 B^10 (AB Block), (AB)^10 (AB Alternate), (BA)^10 (BA Alternate), and B^10 A^10 (BA Block). '
        '(a-d) 3D surface equilibrium shapes rendered on identical height scales (z in [-17, +2] mm). '
        '(e) Longitudinal midline profile (u = 0 mm) showing the monotonic progression of arch sagitta '
        'from 13.87 mm (A^10 B^10) to 14.98 mm ((AB)^10) to 15.36 mm ((BA)^10) to 16.40 mm (B^10 A^10). '
        '(f) Transverse midline profile (v = 0 mm) resolving periodic strip undulations and edge roll-offs. '
        '(g) Quantitative progression of primary longitudinal curvature |kappa_1| and arch sagitta across schedules.'
    )
    (plot_dir / f'{plot_stem}.caption.md').write_text(caption + '\n')
    print(f'Successfully generated {png_path} and {pdf_path}')


if __name__ == '__main__':
    main()
