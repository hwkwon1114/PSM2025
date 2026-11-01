#!/usr/bin/env python3
"""Project the saved pairwise shape overlaps onto public width x and height y."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.plot_pairwise_shape_overlays import (
    BASE, OUT, CASES, PAIR_ORDER, FIT_LABELS, SCAN_COLOR, MODEL_COLOR,
    digest, load_pair, load_panel,
)

STEM = 'pairwise_shape_overlays_xy'
SOURCE = OUT / 'pairwise_shape_overlays_3x3_v2.json'


def main():
    targets = [OUT / f'{STEM}.{ext}' for ext in ('png', 'pdf', 'caption.md', 'json')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Refusing to overwrite an existing x–y figure')
    source = json.loads(SOURCE.read_text())
    groups = [load_pair(BASE, pair) for pair in PAIR_ORDER]
    reference = {}
    panels = [[load_panel(group, case, reference) for case in CASES] for group in groups]
    indices = {case: np.asarray(source['sample_indices'][case], dtype=int) for case in CASES}
    for i, row in enumerate(panels):
        if groups[i]['hashes'] != source['selection_and_results_sha256'][PAIR_ORDER[i]]:
            raise ValueError('Selection or result has changed since the 3-D figure')
        for j, panel in enumerate(row):
            previous = source['panels'][i][j]
            if panel['inputs'] != previous['inputs'] or panel['rms_mm'] != previous['rms_mm']:
                raise ValueError(f'Input or score changed: {panel["pair"]}/{panel["case"]}')
            sample = indices[panel['case']]
            if len(sample) != len(set(sample)) or np.any(sample < 0) or np.any(sample >= len(panel['scan'])):
                raise ValueError('Invalid saved sample indices')
    xlim = source['limits']['x_width_mm']
    ylim = source['limits']['y_height_mm']
    style = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
             'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 9,
             'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), sharex=True, sharey=True)
        fig.subplots_adjust(left=.105, right=.985, top=.86, bottom=.08, hspace=.44, wspace=.17)
        fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=SCAN_COLOR, markersize=5,
                                   label='Experiment (sampled points)'),
                            Line2D([], [], color=MODEL_COLOR, linewidth=1.2, label='Simulation (mesh grid)')],
                   loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False)
        fig.text(.5, .925, 'Projection across all lengths z · axis scales unequal · RMS is full 3-D surface metric',
                 ha='center', fontsize=9)
        for i, row in enumerate(panels):
            for j, panel in enumerate(row):
                ax = axes[i, j]
                scan = panel['scan'][indices[panel['case']]]
                ax.scatter(scan[:, 0], scan[:, 2], color=SCAN_COLOR, s=2, alpha=.30,
                           rasterized=True, linewidths=0)
                for line in panel['wires']:
                    ax.plot(line[:, 0], line[:, 2], color=MODEL_COLOR, linewidth=.55, alpha=.72)
                label = 'excluded check' if panel['transfer'] else 'calibration'
                ax.set_title(f'Run {j+1} · {label}\nSurface RMS = {panel["rms_mm"]:.3f} mm', pad=5)
                ax.set(xlim=xlim, ylim=ylim)
                ax.set_xticks([-100, -50, 0, 50, 100])
                ax.set_yticks([-6, -3, 0, 3])
                ax.grid(color='#e8e8e8', linewidth=.5)
                ax.set_axisbelow(True)
                if j == 0:
                    ax.set_ylabel('Height y (mm)')
                if i == 2:
                    ax.set_xlabel('Width x (mm)')
            box = axes[i, 0].get_position()
            fig.text(.018, (box.y0 + box.y1)/2, FIT_LABELS[PAIR_ORDER[i]], rotation=90,
                     va='center', ha='center', fontsize=10, fontweight='bold')
        fig.savefig(targets[0], dpi=200)
        fig.savefig(targets[1], dpi=200)
        plt.close(fig)
    targets[2].write_text(
        '# Width–height projections of the pairwise calibrated shapes\n\n'
        'Rows are frozen fits to Runs 1+2, 2+3 and 1+3; columns are Runs 1, 2 and 3. '
        'The excluded-run panels are exploratory checks, not blind validation. Orange points '
        'are experimental scan points; blue lines are the saved simulation mesh grid. This is an '
        'orthographic x–y projection across the entire length z, not a section at fixed z; '
        'overlapping points can belong to different lengths and are not pointwise matches. '
        'The x and y axes use unequal display scales. All panels share the same limits. '
        'The 2,500 points per case are the identical fixed-seed sample used in the 3-D companion '
        'figure, not selected by error; model grid lines are piecewise-linear on saved facets and '
        'masked outside the mesh. Titles give the full three-dimensional shared-footprint symmetric '
        'surface RMS in mm, not an error of this 2-D projection or the displayed point sample. '
        'No new registration, solver or refitting was performed. Data are spatially dependent. '
        'Full directional error distributions appear in pairwise_case_error_comparison.\n')
    targets[3].write_text(json.dumps({
        'script_sha256': digest(Path(__file__)), 'companion_provenance_sha256': digest(SOURCE),
        'input_hashes': {p['pair'] + '/' + p['case']: p['inputs'] for row in panels for p in row},
        'sample_indices': source['sample_indices'], 'xlim_mm': xlim, 'ylim_mm': ylim,
        'projection': 'public width x versus height y; all length z collapsed',
        'surface_rms_mm': {p['pair'] + '/' + p['case']: p['rms_mm'] for row in panels for p in row},
    }, indent=2, allow_nan=False) + '\n')
    print('\n'.join(str(p) for p in targets))


if __name__ == '__main__':
    main()
