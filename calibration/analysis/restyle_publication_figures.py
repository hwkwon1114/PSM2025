#!/usr/bin/env python3
"""Restyle Attempt 209 publication figures with enlarged, publication-consistent fonts.

Ensures that when all figures are scaled to a standard manuscript text width (7.0 in / 178 mm),
the effective printed font size is uniformly ~10 pt across all seven figures:
- 14.0 in wide figures use 20 pt canvas font (effective print size: 10.0 pt).
- 18.0 in wide 3-D figure uses 26 pt canvas font (effective print size: 10.1 pt).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis import attempt209_publication as pub
from calibration.analysis import attempt209_horizontal_overlays as overlays
from calibration.ewcal.common import digest

STYLE_20PT = {
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.facecolor': 'white'
}

STYLE_26PT_3D = {
    'font.size': 26,
    'axes.titlesize': 26,
    'axes.labelsize': 26,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 26,
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.facecolor': 'white'
}


def render_all(out_dir, source_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pub.OUT = out_dir

    previous = json.loads((source_dir / 'provenance.json').read_text())
    summary = json.loads(pub.SUMMARY.read_text())
    config = json.loads(pub.CONFIG.read_text())
    chosen = summary['best']['parameters']
    bounds = {key: config['parameters'][key]['bounds'] for key in pub.PARAMS}
    records = sorted((json.loads(line) for line in pub.HISTORY.read_text().splitlines() if line.strip()),
                     key=lambda record: record['attempt_id'])
    pins = {}
    rows = [pub.shapes.load(run, pins) for run in pub.RUNS]

    # 1-5: Five non-overlay figures with 20pt
    print('Rendering 5 figures with 20 pt (effective print size 10.0 pt)...')
    with plt.rc_context(STYLE_20PT):
        # 1. Clean single-plot incumbent history under the revised metric
        clean_eval_path = ROOT / 'run/run1_clean_24lhs/evaluations.jsonl'
        clean_records = [json.loads(line) for line in clean_eval_path.read_text().splitlines() if line.strip()]
        attempts, incumbent = [], []
        best = float('inf')
        for r in clean_records:
            if r['valid']:
                best = min(best, r['objective_mm'])
            if np.isfinite(best):
                attempts.append(r['attempt'])
                incumbent.append(best)

        fig_hist, ax_hist = plt.subplots(figsize=(14, 6), layout='constrained')
        ax_hist.step(attempts, incumbent, color=pub.BLUE, lw=3.0, where='post')
        ax_hist.set(yscale='log', xlabel='Evaluation', ylabel='Best shape error RMS (mm)', xlim=(1, len(clean_records)))
        ax_hist.set_yticks([0.6, 1.0, 2.0, 5.0, 10.0, 20.0])
        ax_hist.get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax_hist.grid(axis='y', color='#e5e5e5', lw=0.6)

        # Mark just the best value
        best_idx = np.argmin(incumbent)
        best_eval = attempts[best_idx]
        ax_hist.scatter([best_eval], [best], color=pub.BLUE, s=140, zorder=5)
        ax_hist.text(best_eval + 1.2, best * 1.15, f'{best:.3f} mm', color=pub.BLUE, fontweight='bold', va='bottom')

        pub.save(fig_hist, 'optimization_history',
                 '# Optimization history\n\n'
                 'Running incumbent of the shape error RMS (mm) under the revised shared-footprint '
                 'double-sided surface metric as a function of evaluation count. The curve traces progressive '
                 f'refinement to an optimal {best:.3f} mm fit, marked at evaluation {best_eval}.')

        pub.sensitivity_figure(records, chosen, bounds)
        pub.toolpaths(rows)
        pub.residuals(rows)
        pub.sections_figure(rows)

    # 6: Horizontal x-y overlay with 20pt
    print('Rendering shape_overlays_xy with 20 pt (effective print size 10.0 pt)...')
    prov_indices = previous['overlay']['sample_indices']
    indices = [np.array(prov_indices[run], dtype=int) for run in pub.RUNS]
    lines = [overlays.grid_lines(row['prediction'], row['faces']) for row in rows]
    ylim = (-6.0, 4.0)

    legend = [Line2D([], [], color=pub.ORANGE, marker='o', ls='', ms=6, label='Experiment (sampled)'),
              Line2D([], [], color=pub.BLUE, lw=1.8, label='Simulation grid')]

    with plt.rc_context(STYLE_20PT):
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2), sharey=True, layout='constrained')
        fig.get_layout_engine().set(w_pad=.16, h_pad=.12, wspace=.16)
        for i, (ax, row, idx, wires, name) in enumerate(zip(axes, rows, indices, lines, pub.NAMES)):
            p = row['scan'][idx]
            ax.scatter(p[:, 0], p[:, 2], s=3, color=pub.ORANGE, alpha=.35, rasterized=True)
            for line in wires:
                ax.plot(line[:, 0], line[:, 2], color=pub.BLUE, lw=.75, alpha=.8)
            ax.set(xlim=(-105, 105), ylim=ylim, xticks=[-100, 0, 100],
                   xlabel='Width $x$ (mm)', title=f'({chr(97+i)}) {name}')
            if i == 0:
                ax.set_ylabel('Height $y$ (mm)')
            ax.grid(color='#e7e7e7', lw=.5)
        fig.legend(handles=legend, loc='outside upper center', ncol=2, frameon=False)
        pub.save(fig, 'shape_overlays_xy', (source_dir / 'shape_overlays_xy.caption.md').read_text().strip())

    # 7: Horizontal 3-D overlay with 26pt on 18x7.5 in
    print('Rendering shape_overlays_3d with 26 pt (effective print size 10.1 pt)...')
    with plt.rc_context(STYLE_26PT_3D):
        fig = plt.figure(figsize=(18.0, 7.5))
        fig.subplots_adjust(left=.045, right=.89, top=.84, bottom=.15, wspace=.20)
        axes = fig.subplots(1, 3, subplot_kw={'projection': '3d'})
        for i, (ax, row, idx, wires, name) in enumerate(zip(axes, rows, indices, lines, pub.NAMES)):
            p = row['scan'][idx]
            ax.scatter(p[:, 0], p[:, 1] + 65, p[:, 2], s=2, color=pub.ORANGE,
                       alpha=.55, depthshade=False, rasterized=True)
            for line in wires:
                ax.plot(line[:, 0], line[:, 1] + 65, line[:, 2], color=pub.BLUE, lw=.75)
            ax.set(xlim=(-105, 105), ylim=(-90, 220), zlim=ylim,
                   xticks=[-100, 0, 100], yticks=[0, 150], zticks=[-6, 0, 4],
                   xlabel='Width $x$ (mm)', ylabel='Length $z$ (mm)',
                   zlabel='Height $y$ (mm)', title=f'({chr(97+i)}) {name}')
            ax.view_init(elev=23, azim=-65)
            ax.set_box_aspect((210, 310, (ylim[1] - ylim[0]) * 8), zoom=.90)
            ax.tick_params(pad=2, labelsize=26)
            ax.xaxis.labelpad = 18
            ax.yaxis.labelpad = 22
            ax.zaxis.labelpad = 18
            ax.grid(False)
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False)
        pub.save(fig, 'shape_overlays_3d', (source_dir / 'shape_overlays_3d.caption.md').read_text().strip())

    # Write provenance record
    previous['publication_font_revision'] = {
        'generator': str(Path(__file__).relative_to(ROOT)),
        'generator_sha256': digest(Path(__file__)),
        'target_manuscript_text_width_in': 7.0,
        'effective_manuscript_font_size_pt': '10.0-10.1 pt',
        'canvas_specifications': {
            'optimization_history': {'canvas_in': [14.0, 6.0], 'canvas_font_pt': 20, 'effective_pt': 10.0},
            'toolpaths': {'canvas_in': [14.0, 8.5], 'canvas_font_pt': 20, 'effective_pt': 10.0},
            'shape_overlays_xy': {'canvas_in': [14.0, 5.2], 'canvas_font_pt': 20, 'effective_pt': 10.0},
            'height_residuals': {'canvas_in': [14.0, 12.0], 'canvas_font_pt': 20, 'effective_pt': 10.0},
            'fixed_sections': {'canvas_in': [14.0, 13.0], 'canvas_font_pt': 20, 'effective_pt': 10.0},
            'historical_gp_sensitivity': {'canvas_in': [14.0, 11.0], 'canvas_font_pt': 20, 'effective_pt': 10.0},
            'shape_overlays_3d': {'canvas_in': [18.0, 7.5], 'canvas_font_pt': 26, 'effective_pt': 10.1},
        }
    }
    (out_dir / 'provenance.json').write_text(json.dumps(previous, indent=2, allow_nan=False) + '\n')
    (out_dir / 'README.md').write_text(
        '# Attempt 209 Publication Figures (Enlarged Consistent Typography)\n\n'
        'All seven publication figures with enlarged, consistent typography.\n'
        'When placed into a standard manuscript page (7.0 in / 178 mm width), '
        'every figure renders at a uniform 10.0–10.1 pt effective font size.\n'
    )
    print(f'Successfully rendered all 7 figures to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'calibration/paper_figures/attempt209_publication')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'calibration/paper_figures/attempt209_publication')
    args = parser.parse_args()
    render_all(args.out_dir, args.source_dir)


if __name__ == '__main__':
    main()
