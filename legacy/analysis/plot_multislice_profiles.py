"""Publication-grade multi-slice profile comparison: Calibrated Model vs. Scan STL.

Extracts transverse and longitudinal cross-sections at key stations across the plate,
comparing the calibrated FE shell predictions with experimental scan targets.
Exports vector PDF, high-resolution PNG, structured provenance, and external caption.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import Bbox
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import (
    BASE as BASE_DIR,
    heights_at_xy,
    load_evaluation_data,
    pin,
    resolve_case_paths,
)
from calibration.ewcal.common import digest, save_json


def extract_transverse_slice(case_data, y_target, x_eval, scan_tri_interp, band_halfwidth=1.5):
    """Evaluate model and scan along line y = y_target, and gather discrete scan points."""
    q, faces, scan = case_data['prediction'], case_data['faces'], case_data['scan']
    pts = np.column_stack([x_eval, np.full_like(x_eval, y_target)])

    # Model height
    h_model, support, _ = heights_at_xy(q, faces, pts)

    # Continuous scan height
    h_scan = scan_tri_interp(x_eval, np.full_like(x_eval, y_target)).filled(np.nan)

    # Discrete nearby scan points
    mask = np.abs(scan[:, 1] - y_target) <= band_halfwidth
    pts_scan = scan[mask]

    dz = h_model - h_scan
    valid_dz = dz[np.isfinite(dz)]
    rms = float(np.sqrt(np.mean(valid_dz ** 2))) if len(valid_dz) > 0 else float('nan')
    max_err = float(np.max(np.abs(valid_dz))) if len(valid_dz) > 0 else float('nan')

    return {
        'y_target': float(y_target),
        'x_eval': x_eval,
        'h_model': h_model,
        'h_scan': h_scan,
        'dz': dz,
        'scan_pts': pts_scan,
        'rms_mm': rms,
        'max_err_mm': max_err,
    }


def extract_longitudinal_slice(case_data, x_target, y_eval, scan_tri_interp, band_halfwidth=1.5):
    """Evaluate model and scan along line x = x_target, and gather discrete scan points."""
    q, faces, scan = case_data['prediction'], case_data['faces'], case_data['scan']
    pts = np.column_stack([np.full_like(y_eval, x_target), y_eval])

    h_model, support, _ = heights_at_xy(q, faces, pts)
    h_scan = scan_tri_interp(np.full_like(y_eval, x_target), y_eval).filled(np.nan)

    mask = np.abs(scan[:, 0] - x_target) <= band_halfwidth
    pts_scan = scan[mask]

    dz = h_model - h_scan
    valid_dz = dz[np.isfinite(dz)]
    rms = float(np.sqrt(np.mean(valid_dz ** 2))) if len(valid_dz) > 0 else float('nan')
    max_err = float(np.max(np.abs(valid_dz))) if len(valid_dz) > 0 else float('nan')

    return {
        'x_target': float(x_target),
        'y_eval': y_eval,
        'h_model': h_model,
        'h_scan': h_scan,
        'dz': dz,
        'scan_pts': pts_scan,
        'rms_mm': rms,
        'max_err_mm': max_err,
    }


def render_multislice_figure(rows, out_dir, desc, attempt_id, pins):
    """Plot publication-quality multi-slice comparison figure."""
    y_stations = [-100.0, -50.0, 0.0, 50.0, 100.0]
    station_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    x_eval = np.linspace(-95, 95, 191)
    y_eval = np.linspace(-145, 145, 291)

    # Process slices for each case
    case_slices = []
    for r in rows:
        tri_scan = mtri.Triangulation(r['scan'][:, 0], r['scan'][:, 1])
        scan_interp = mtri.LinearTriInterpolator(tri_scan, r['scan'][:, 2])

        t_slices = [extract_transverse_slice(r, y_s, x_eval, scan_interp) for y_s in y_stations]
        l_slice = extract_longitudinal_slice(r, 0.0, y_eval, scan_interp)
        case_slices.append({'transverse': t_slices, 'longitudinal': l_slice})

    # Global color limits for residual maps
    res_lim = max(np.nanmax(np.abs(r['dz'])) for r in rows)
    norm_res = TwoSlopeNorm(vmin=-res_lim, vcenter=0, vmax=res_lim)

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
        fig, axes = plt.subplots(
            2, 4,
            figsize=(12.0, 6.8),
            layout='constrained',
            gridspec_kw={'width_ratios': [0.9, 1.25, 1.1, 1.25]},
        )

        im_res = None
        for i, (r, cs) in enumerate(zip(rows, case_slices)):
            name_cap = r['name'].capitalize()
            t_slices = cs['transverse']
            l_slice = cs['longitudinal']

            # Col 0: Residual map with slice stations indicated
            ax0 = axes[i, 0]
            ax0.set_facecolor('#EAEAEA')
            im_res = ax0.scatter(
                r['scan'][r['support'], 0], r['scan'][r['support'], 1],
                c=r['dz'][r['support']], s=2, cmap='RdBu_r', norm=norm_res, rasterized=True
            )
            # Plot slice station indicator lines
            for y_s, col in zip(y_stations, station_colors):
                ax0.axhline(y_s, color=col, lw=1.2, ls='--', alpha=0.9)
            # Longitudinal centerline indicator
            ax0.axvline(0.0, color='#111111', lw=1.0, ls=':', alpha=0.8)

            ax0.set(
                title=f'({chr(97 + i*4)}) {name_cap} slice stations',
                xlabel='x (mm)', ylabel='y (mm)',
                xlim=(-105, 105), ylim=(-155, 155),
                aspect='equal',
            )
            ax0.set_xticks([-100, 0, 100])
            ax0.set_yticks([-150, -100, -50, 0, 50, 100, 150])

            # Col 1: Transverse Profile Overlays z(x) at selected stations
            ax1 = axes[i, 1]
            for s_idx, (sl, col) in enumerate(zip(t_slices, station_colors)):
                y_val = int(sl['y_target'])
                lbl_scan = f'Scan (y={y_val:+d})' if i == 0 and s_idx == 2 else None
                lbl_mod = f'Model (y={y_val:+d})' if i == 0 and s_idx == 2 else None

                # Model profile
                ax1.plot(sl['x_eval'], sl['h_model'], color=col, lw=1.3, ls='-', label=lbl_mod)
                # Discrete scan data points
                ax1.scatter(sl['scan_pts'][:, 0], sl['scan_pts'][:, 2], color=col, s=4, alpha=0.45, marker='o', rasterized=True)

            ax1.set(
                title=f'({chr(98 + i*4)}) Transverse profiles z(x)',
                xlabel='x (mm)', ylabel='z (mm)',
                xlim=(-100, 100), ylim=(-5.2, 3.2),
            )
            ax1.set_xticks([-100, -50, 0, 50, 100])
            ax1.grid(True, ls=':', lw=0.4, alpha=0.6)
            if i == 0:
                # Custom legend entry
                p_mod = plt.Line2D([0], [0], color='#333333', lw=1.5, ls='-', label='Model prediction')
                p_scn = plt.Line2D([0], [0], color='#333333', marker='o', ms=3, ls='', alpha=0.6, label='Scan STL points')
                ax1.legend(handles=[p_mod, p_scn], loc='lower right', framealpha=0.9)

            # Col 2: Transverse Residuals Δz(x)
            ax2 = axes[i, 2]
            ax2.axhspan(-0.25, 0.25, color='#e6f4ea', alpha=0.9, zorder=1)
            ax2.axhline(0.0, color='#666666', lw=0.6, ls='-', zorder=2)
            for sl, col in zip(t_slices, station_colors):
                y_val = int(sl['y_target'])
                ax2.plot(sl['x_eval'], sl['dz'], color=col, lw=1.2, label=f'y={y_val:+d} mm')

            ax2.set(
                title=f'({chr(99 + i*4)}) Height residual Δz(x)',
                xlabel='x (mm)', ylabel='Δz: model − scan (mm)',
                xlim=(-100, 100), ylim=(-1.4, 1.4),
            )
            ax2.set_xticks([-100, -50, 0, 50, 100])
            ax2.grid(True, ls=':', lw=0.4, alpha=0.6)
            if i == 0:
                ax2.legend(loc='lower left', framealpha=0.9, ncol=2, columnspacing=0.8, handlelength=1.2)

            # Col 3: Longitudinal Centerline Profile z(y) at x=0 & Residual
            ax3 = axes[i, 3]
            # Primary axis: height profile
            ax3.plot(l_slice['y_eval'], l_slice['h_model'], color='#0072B2', lw=1.6, ls='-', label='Model (x=0)')
            ax3.scatter(l_slice['scan_pts'][:, 1], l_slice['scan_pts'][:, 2], color='#D55E00', s=5, alpha=0.6, marker='o', label='Scan (x=0)', rasterized=True)
            # Secondary line: residual along x=0
            ax3.plot(l_slice['y_eval'], l_slice['dz'], color='#10b981', lw=1.2, ls='--', label='Δz residual')
            ax3.axhline(0.0, color='#999999', lw=0.5, ls=':')

            ax3.set(
                title=f'({chr(100 + i*4)}) Centerline profile at x=0',
                xlabel='y (mm)', ylabel='z and Δz (mm)',
                xlim=(-150, 150), ylim=(-1.8, 3.5),
            )
            ax3.set_xticks([-150, -100, -50, 0, 50, 100, 150])
            ax3.grid(True, ls=':', lw=0.4, alpha=0.6)
            ax3.legend(loc='lower right', framealpha=0.9)

        # Colorbar for col 0
        fig.colorbar(
            im_res, ax=axes[:, 0].tolist(), label='Δz: model − scan (mm)', shrink=0.7,
            ticks=np.linspace(-res_lim, res_lim, 5), format='%.1f'
        )

        # Verification of text extent
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        all_boxes = []
        for ax in fig.axes:
            texts = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
            if ax.get_legend():
                texts.extend(ax.get_legend().get_texts())
            for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                for tick in axis.get_major_ticks():
                    if min(limits) <= tick.get_loc() <= max(limits):
                        texts.extend([tick.label1, tick.label2])
            for text in texts:
                if not text.get_visible() or not text.get_text():
                    continue
                box = text.get_window_extent(renderer)
                assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, f"Clipped text: {text.get_text()}"
                all_boxes.append((text.get_text(), box))

        for ext in ('png', 'pdf'):
            fig.savefig(out_dir / f'run6_run9_multislice_profiles.{ext}', dpi=300)
        plt.close(fig)

    # Save Provenance
    provenance = {
        'status': 'rendered',
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        'description': desc,
        'attempt_id': attempt_id,
        'input_sha256': pins,
        'script_sha256': digest(Path(__file__)),
        'transverse_stations_y_mm': y_stations,
        'longitudinal_stations_x_mm': [0.0],
        'figures': [
            str(out_dir / 'run6_run9_multislice_profiles.png'),
            str(out_dir / 'run6_run9_multislice_profiles.pdf'),
        ],
        'tolerance_band_mm': [-0.25, 0.25],
    }
    save_json(out_dir / 'run6_run9_multislice_profiles.provenance.json', provenance)

    caption = f"""# Multi-Slice Cross-Section Comparison: Calibrated Model vs. Scan STL

Companion to six-panel deformation analysis ({desc}).
- **(a, e) Slice station map**: Residual error fields $\\Delta z$ with indicated transverse cut stations ($y = -100, -50, 0, +50, +100$ mm) and longitudinal centerline ($x=0$).
- **(b, f) Transverse profiles $z(x)$**: Physical cross-sectional curves of calibrated FE shell model (solid lines) compared against discrete experimental scan points (markers) across plate width ($x \\in [-100, 100]$ mm).
- **(c, g) Height residual curves $\\Delta z(x)$**: Local error curves $\\Delta z = z_{{\\text{{model}}}} - z_{{\\text{{scan}}}}$ along each station. Shaded green band indicates $\\pm 0.25$ mm tolerance envelope.
- **(d, h) Longitudinal centerline profile $z(y)$**: Longitudinal profile along the symmetry plane ($x=0$), showing model curve (blue), experimental scan points (orange), and residual trace (green).
"""
    (out_dir / 'run6_run9_multislice_profiles.caption.md').write_text(caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=209)
    parser.add_argument('--run6-case', type=Path)
    parser.add_argument('--run9-case', type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()

    r6_case, r9_case, desc, attempt_id = resolve_case_paths(args, BASE)
    out_dir = args.out_dir if args.out_dir else BASE / 'legacy/analysis/plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    pins = {}
    print(f"Loading Run 6: {r6_case}")
    d6 = load_evaluation_data('run6', r6_case, pins, BASE)
    print(f"Loading Run 9: {r9_case}")
    d9 = load_evaluation_data('run9', r9_case, pins, BASE)

    print("Generating multi-slice comparison figure...")
    render_multislice_figure([d6, d9], out_dir, desc, attempt_id, pins)
    print(f"Generated: {out_dir / 'run6_run9_multislice_profiles.pdf'}")
    print(f"Generated: {out_dir / 'run6_run9_multislice_profiles.png'}")


if __name__ == '__main__':
    main()
