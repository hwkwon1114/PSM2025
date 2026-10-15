"""Consolidated 3D comparison tool: 3D isometric residual, 3D end-on profile residual, and 3D shape overlay.

Visualizes true 3D spatial curvature, saddle warpage, and height residual fields for Run 6 and Run 9.
Supports arbitrary evaluation cases or known attempt presets (default: Attempt 153).
"""
import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import (
    BASE as BASE_DIR,
    load_evaluation_data,
    pin,
    resolve_case_paths,
)
from calibration.ewcal.common import digest, save_json


def render_3d_figure(rows, out_dir, desc, attempt_id, pins):
    """Render 3D isometric residuals, profile views, and physical shape overlays."""
    # Common residual scale
    residual_limit = max(np.nanmax(np.abs(r['dz'])) for r in rows)
    residual_norm = TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit)

    style_rc = {
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'pdf.fonttype': 42,
        'axes.linewidth': 0.6,
        'savefig.facecolor': 'white',
    }

    with plt.rc_context(style_rc):
        fig = plt.figure(figsize=(15.0, 8.2))
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1.0, 1.0, 0.04, 1.05], wspace=0.25, hspace=0.22)

        # Row 0: Run 6
        r6 = rows[0]
        scan6, q6, dz6 = r6['scan'], r6['prediction'], r6['dz']
        faces6 = r6['faces']

        r9 = rows[1]
        scan9, q9, dz9 = r9['scan'], r9['prediction'], r9['dz']
        faces9 = r9['faces']

        # Aspect ratios: x is ~200mm, y is length (130 or 187.5), z is ~8mm
        aspect6 = [1.0, 1.3, 0.35]
        aspect9 = [1.0, 1.8, 0.35]

        # Panel (a): Run6 Isometric residual
        ax0 = fig.add_subplot(gs[0, 0], projection='3d')
        sc_res = ax0.scatter(
            scan6[:, 0], scan6[:, 1], scan6[:, 2],
            c=dz6, cmap='RdBu_r', norm=residual_norm, s=2, rasterized=True
        )
        ax0.set_box_aspect(aspect6)
        ax0.view_init(elev=25, azim=-55)
        ax0.set(title='(a) Run6: 3D residual (isometric)', xlabel='x (mm)', ylabel='y (mm)')
        ax0.set_zlabel('z (mm)', labelpad=1)

        # Panel (b): Run6 End-on profile residual
        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax1.scatter(
            scan6[:, 0], scan6[:, 1], scan6[:, 2],
            c=dz6, cmap='RdBu_r', norm=residual_norm, s=2, rasterized=True
        )
        ax1.set_box_aspect(aspect6)
        ax1.view_init(elev=10, azim=-15)
        ax1.set(title='(b) Run6: 3D residual (end-on)', xlabel='x (mm)', ylabel='y (mm)')
        ax1.set_zlabel('z (mm)', labelpad=1)

        # Colorbar column
        cax = fig.add_subplot(gs[:, 2])
        fig.colorbar(
            sc_res, cax=cax,
            label='Δz: prediction − experiment (mm)',
            ticks=np.linspace(-residual_limit, residual_limit, 5),
            format='%.1f'
        )

        # Panel (c): Run6 3D shape overlay
        ax2 = fig.add_subplot(gs[0, 3], projection='3d')
        h_exp6 = ax2.scatter(
            scan6[:, 0], scan6[:, 1], scan6[:, 2],
            color='#D55E00', s=2, alpha=0.5, label='Experiment', rasterized=True
        )
        ax2.plot_trisurf(
            q6[:, 0], q6[:, 1], q6[:, 2],
            triangles=faces6, color='#0072B2', alpha=0.35, edgecolor='none'
        )
        h_pred6 = mpatches.Patch(color='#0072B2', alpha=0.5, label='Prediction')
        ax2.set_box_aspect(aspect6)
        ax2.view_init(elev=25, azim=-55)
        ax2.set(title='(c) Run6: 3D shape overlay', xlabel='x (mm)', ylabel='y (mm)')
        ax2.set_zlabel('z (mm)', labelpad=1)
        ax2.legend(handles=[h_exp6, h_pred6], loc='upper right', framealpha=0.9, fontsize=8)

        # Panel (d): Run9 Isometric residual
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        ax3.scatter(
            scan9[:, 0], scan9[:, 1], scan9[:, 2],
            c=dz9, cmap='RdBu_r', norm=residual_norm, s=2, rasterized=True
        )
        ax3.set_box_aspect(aspect9)
        ax3.view_init(elev=25, azim=-55)
        ax3.set(title='(d) Run9: 3D residual (isometric)', xlabel='x (mm)', ylabel='y (mm)')
        ax3.set_zlabel('z (mm)', labelpad=1)

        # Panel (e): Run9 End-on profile residual
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        ax4.scatter(
            scan9[:, 0], scan9[:, 1], scan9[:, 2],
            c=dz9, cmap='RdBu_r', norm=residual_norm, s=2, rasterized=True
        )
        ax4.set_box_aspect(aspect9)
        ax4.view_init(elev=10, azim=-15)
        ax4.set(title='(e) Run9: 3D residual (end-on)', xlabel='x (mm)', ylabel='y (mm)')
        ax4.set_zlabel('z (mm)', labelpad=1)

        # Panel (f): Run9 3D shape overlay
        ax5 = fig.add_subplot(gs[1, 3], projection='3d')
        h_exp9 = ax5.scatter(
            scan9[:, 0], scan9[:, 1], scan9[:, 2],
            color='#D55E00', s=2, alpha=0.5, label='Experiment', rasterized=True
        )
        ax5.plot_trisurf(
            q9[:, 0], q9[:, 1], q9[:, 2],
            triangles=faces9, color='#0072B2', alpha=0.35, edgecolor='none'
        )
        h_pred9 = mpatches.Patch(color='#0072B2', alpha=0.5, label='Prediction')
        ax5.set_box_aspect(aspect9)
        ax5.view_init(elev=25, azim=-55)
        ax5.set(title='(f) Run9: 3D shape overlay', xlabel='x (mm)', ylabel='y (mm)')
        ax5.set_zlabel('z (mm)', labelpad=1)
        ax5.legend(handles=[h_exp9, h_pred9], loc='upper right', framealpha=0.9, fontsize=8)

        for ext in ('png', 'pdf'):
            fig.savefig(out_dir / f'run6_run9_3d_deformation_residual.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Save 3D provenance
    provenance = {
        'status': 'rendered',
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        'description': f'3D views: {desc}',
        'attempt_id': attempt_id,
        'input_sha256': pins,
        'script_sha256': digest(Path(__file__)),
        'statistics': [r['statistics'] for r in rows],
        'residual_color_limits_mm': [-float(residual_limit), float(residual_limit)],
        'views': {
            'isometric': {'elev_deg': 25, 'azim_deg': -55},
            'end_on_profile': {'elev_deg': 10, 'azim_deg': -15},
            'shape_overlay': {'elev_deg': 25, 'azim_deg': -55},
        },
        'aspect_ratios': {
            'run6': aspect6,
            'run9': aspect9,
        },
    }
    save_json(out_dir / 'provenance_3d.json', provenance)

    caption = f"""# 3D Spatial Comparison: {desc}

Top row: Run 6; bottom row: Run 9.
- **(a, d) 3D Isometric Residual Surface**: Deformed 3D point cloud colored continuously by signed height residual $\\Delta z = z_{{\\text{{model}}}} - z_{{\\text{{scan}}}}$ (elev=25°, azim=-55°). Shows the 3D saddle/arch curvature and how errors distribute over the crest, flanks, and corners.
- **(b, e) 3D End-On Profile View**: Viewing down the longitudinal toolpath axis (elev=10°, azim=-15°). Highlights vertical arch conformity and transverse edge discrepancy.
- **(c, f) 3D Physical Shape Overlay**: Direct spatial overlay of experimental scan points (orange circles, #D55E00) and the model simulation surface (semi-transparent blue mesh, #0072B2), showing the true geometric agreement in 3D space.

## Shared Color Scale
- $\\Delta z$ ranges symmetrically from -{residual_limit:.2f} mm (blue, model lower) to +{residual_limit:.2f} mm (red, model higher), with white indicating exact vertical agreement.

*Reproduction*: `python calibration/analysis/run6_run9_3d_comparison.py --attempt {attempt_id if attempt_id else '209'}`.
"""
    (out_dir / 'run6_run9_3d_deformation_residual.caption.md').write_text(caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=209, help='Attempt preset (209=global optimal BO, 153=previous best, 83=stage2 best, 25=baseline)')
    parser.add_argument('--run6-case', type=Path, help='Direct path to Run 6 case directory (overrides --attempt)')
    parser.add_argument('--run9-case', type=Path, help='Direct path to Run 9 case directory (overrides --attempt)')
    parser.add_argument('--out-dir', type=Path, help='Output directory (default: run/optimization_review/run6_run9_six_panel_attempt{attempt})')
    args = parser.parse_args()

    r6_case, r9_case, desc, attempt_id = resolve_case_paths(args, BASE_DIR)

    if args.out_dir:
        out_dir = args.out_dir if args.out_dir.is_absolute() else BASE_DIR / args.out_dir
    else:
        out_dir = BASE_DIR / f'run/optimization_review/run6_run9_six_panel_attempt{attempt_id if attempt_id else "custom"}'
    out_dir.mkdir(parents=True, exist_ok=True)

    pins = {}
    print(f"Loading Run 6 from: {r6_case}")
    row6 = load_evaluation_data('run6', r6_case, pins, BASE_DIR)
    print(f"Loading Run 9 from: {r9_case}")
    row9 = load_evaluation_data('run9', r9_case, pins, BASE_DIR)

    rows = [row6, row9]
    print(f"Rendering 3D comparison in: {out_dir}")
    render_3d_figure(rows, out_dir, desc, attempt_id, pins)

    print("\n3D Rendering complete!")
    print(f"PNG preview: {out_dir / 'run6_run9_3d_deformation_residual.png'}")
    print(f"Vector PDF:  {out_dir / 'run6_run9_3d_deformation_residual.pdf'}")
    print(f"Caption:     {out_dir / 'run6_run9_3d_deformation_residual.caption.md'}")
    print(f"Provenance:  {out_dir / 'provenance_3d.json'}")


if __name__ == '__main__':
    main()
