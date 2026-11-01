#!/usr/bin/env python3
"""Re-render Attempt 209 overlays horizontally from the identical saved samples."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis import convention_corrected_figures as shapes
from calibration.analysis import attempt209_publication as publication
from calibration.analysis.attempt209_publication import OUT, RUNS, NAMES, BLUE, ORANGE, save
from calibration.analysis.plot_pairwise_shape_overlays import grid_lines
from calibration.ewcal.common import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=OUT)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args(argv)
    provenance = json.loads((args.source_dir / 'provenance.json').read_text())
    staging = args.out_dir or Path(tempfile.mkdtemp(prefix='attempt209_horizontal_', dir=ROOT / 'run'))
    if any(staging.glob('shape_overlays_*.*')):
        raise FileExistsError(f'Refusing to overwrite overlays in {staging}')
    staging.mkdir(parents=True, exist_ok=True)
    publication.OUT = staging
    pins = {}
    rows = [shapes.load(run, pins) for run in RUNS]
    for path, sha in pins.items():
        if provenance['input_sha256'].get(path) != sha:
            raise ValueError(f'Saved input identity changed: {path}')
    if any(row['frozen'] != provenance['frozen_parameters'] for row in rows):
        raise ValueError('Frozen parameters changed')
    indices = [np.array(provenance['overlay']['sample_indices'][run], dtype=int) for run in RUNS]
    for row, idx in zip(rows, indices):
        if len(idx) != len(np.unique(idx)) or np.any(idx < 0) or np.any(idx >= len(row['scan'])):
            raise ValueError('Invalid saved scan sample')
    lines = [grid_lines(row['prediction'], row['faces']) for row in rows]
    ylim = provenance['overlay']['height_limits_mm']
    style = {'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
             'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16,
             'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
    legend = [Line2D([], [], color=ORANGE, marker='o', ls='', ms=5, label='Experiment (sampled)'),
              Line2D([], [], color=BLUE, lw=1.4, label='Simulation grid')]
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), sharey=True, layout='constrained')
        fig.get_layout_engine().set(w_pad=.16, h_pad=.12, wspace=.16)
        for i, (ax, row, idx, wires, name) in enumerate(zip(axes, rows, indices, lines, NAMES)):
            p = row['scan'][idx]
            ax.scatter(p[:, 0], p[:, 2], s=3, color=ORANGE, alpha=.35, rasterized=True)
            for line in wires:
                ax.plot(line[:, 0], line[:, 2], color=BLUE, lw=.75, alpha=.8)
            ax.set(xlim=(-105, 105), ylim=ylim, xticks=[-100, 0, 100],
                   xlabel='Width $x$ (mm)', title=f'({chr(97+i)}) {name}')
            if i == 0:
                ax.set_ylabel('Height $y$ (mm)')
            ax.grid(color='#e7e7e7', lw=.5)
        fig.legend(handles=legend, loc='outside upper center', ncol=2, frameon=False)
        save(fig, 'shape_overlays_xy',
             '# Width–height projections\n\nRun 1 calibration and Runs 2–3 exploratory transfers using frozen Attempt 209 parameters in the corrected convention. Points are the same 2,500 scan points per case selected uniformly without replacement in the original overlay, using seed 20261029. Lines are model grids interpolated on saved native facets and masked outside the mesh. The x–y view projects all lengths z together; it is not a fixed-z section, and overlapping points need not be corresponding material positions. Axes have unequal display scales. All panels share axis limits. No new solve or registration.')
        fig = plt.figure(figsize=(18.0, 7.0))
        fig.subplots_adjust(left=.045, right=.89, top=.84, bottom=.15, wspace=.20)
        axes = fig.subplots(1, 3, subplot_kw={'projection': '3d'})
        for i, (ax, row, idx, wires, name) in enumerate(zip(axes, rows, indices, lines, NAMES)):
            p = row['scan'][idx]
            ax.scatter(p[:, 0], p[:, 1] + 65, p[:, 2], s=2, color=ORANGE,
                       alpha=.55, depthshade=False, rasterized=True)
            for line in wires:
                ax.plot(line[:, 0], line[:, 1] + 65, line[:, 2], color=BLUE, lw=.75)
            ax.set(xlim=(-105, 105), ylim=(-90, 220), zlim=ylim,
                   xticks=[-100, 0, 100], yticks=[0, 150], zticks=[-6, 0, 4],
                   xlabel='Width $x$ (mm)', ylabel='Length $z$ (mm)',
                   zlabel='Height $y$ (mm)', title=f'({chr(97+i)}) {name}')
            ax.view_init(elev=23, azim=-65)
            ax.set_box_aspect((210, 310, (ylim[1]-ylim[0])*8), zoom=.90)
            ax.tick_params(pad=2, labelsize=16)
            ax.xaxis.labelpad = 15
            ax.yaxis.labelpad = 18
            ax.zaxis.labelpad = 15
            ax.grid(False)
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False)
        save(fig, 'shape_overlays_3d',
             '# Three-dimensional shape overlays\n\nCorrected-convention saved surfaces using frozen Attempt 209 parameters. Run 1 was used for calibration; Runs 2 and 3 are previously seen exploratory transfers. Orange points are the same fixed-seed sample as in the x–y view; blue lines are piecewise-linear grid lines on the saved simulation mesh. The height axis is displayed 8× relative to width and length; numerical ticks remain in mm. All panels use the same camera and axis limits. No new solve or alignment.')
    provenance['horizontal_overlay_revision'] = {
        'script': str(Path(__file__).relative_to(ROOT)),
        'script_sha256': digest(Path(__file__)),
        'layout': 'one row of three runs; all visible text 16 pt',
        'sample_indices_reused': True,
        'xy_size_inches': [14.0, 5.0], '3d_size_inches': [18.0, 7.0],
    }
    (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    print('Staged horizontal x–y and 3-D overlays in', staging)


if __name__ == '__main__':
    main()
