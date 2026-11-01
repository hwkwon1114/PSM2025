"""One-run full-field design draft; existing Attempt 209 data, no new alignment/solve."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import load_evaluation_data


def main():
    out = BASE / 'calibration/paper_figures/redesign_run6'
    out.mkdir(exist_ok=True)
    stem = out / 'run6_calibration_fields'
    pins = {}
    case = BASE / 'run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00'
    row = load_evaluation_data('run6', case, pins)
    scan, model, dz = row['scan'], row['prediction'], row['dz']
    assert np.isfinite(scan).all() and np.isfinite(model).all()
    assert row['support'].all(), 'Draft expects full scan support; show missing points if changed.'
    # Check the recomputed residual summaries against the existing figure record.
    reference = json.loads((BASE / 'calibration/paper_figures/provenance.json').read_text())
    expected = reference['statistics'][0]
    for key in ('rms_dz_mm', 'mean_signed_dz_mm', 'p95_absolute_dz_mm'):
        assert np.isclose(row['statistics'][key], expected[key], rtol=1e-10)
    height = Normalize(-5, 3)
    error = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    assert min(scan[:, 2].min(), model[:, 2].min()) >= height.vmin
    assert max(scan[:, 2].max(), model[:, 2].max()) <= height.vmax
    assert np.abs(dz).max() <= error.vmax
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
             'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
             'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
             'pdf.fonttype': 42, 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.2, 4.45))
        gs = fig.add_gridspec(2, 3, left=.085, right=.975, bottom=.17, top=.91,
                              height_ratios=[1, .045], hspace=.48, wspace=.17)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        kw = dict(s=2.5, marker='s', linewidths=0, rasterized=True)
        measured = axes[0].scatter(scan[:, 0], scan[:, 1], c=scan[:, 2], cmap='cividis', norm=height, **kw)
        axes[1].tripcolor(model[:, 0], model[:, 1], row['faces'], model[:, 2],
                         shading='gouraud', cmap='cividis', norm=height, rasterized=True)
        residual = axes[2].scatter(scan[:, 0], scan[:, 1], c=dz, cmap='RdBu_r', norm=error, **kw)
        for i, (ax, title) in enumerate(zip(axes, ['Measured shape', 'Predicted shape', 'Height error'])):
            ax.set_title(f'({chr(97+i)})  {title}', loc='left', pad=10)
            ax.set(xlim=(-103, 103), ylim=(-153, 153), aspect='equal', xlabel='$x$ (mm)',
                   xticks=[-100, 0, 100], yticks=[-150, -75, 0, 75, 150])
            ax.tick_params(direction='out', length=3)
            for spine in ax.spines.values():
                spine.set_color('#9A9A9A')
            if i == 0:
                ax.set_ylabel('$y$ (mm)')
            else:
                ax.tick_params(labelleft=False)
        cb = fig.colorbar(measured, cax=fig.add_subplot(gs[1, :2]), orientation='horizontal', ticks=[-5,-3,-1,1,3])
        cb.set_label('Height, $z$ (mm)', labelpad=4)
        cb = fig.colorbar(residual, cax=fig.add_subplot(gs[1, 2]), orientation='horizontal', ticks=[-2.5,0,2.5])
        cb.set_label('$z_{\\mathrm{pred}} - z_{\\mathrm{meas}}$ (mm)', labelpad=4)
        for ext in ('pdf', 'png'):
            fig.savefig(stem.with_suffix('.' + ext), dpi=350)
        plt.close(fig)
    for path, sha in pins.items():
        assert hashlib.sha256((BASE / path).read_bytes()).hexdigest() == sha
    record = {'source_hashes': pins, 'statistics': row['statistics'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
              'figure_inches': [7.2, 4.45], 'dpi': 350,
              'height_limits_mm': [-5,3], 'residual_limits_mm': [-2.5,2.5],
              'scope': 'Run 6 calibration design draft; not held-out performance; no new solve/alignment',
              'encoding': 'All accepted scan points as squares; native model facets with linear vertex colors. No data smoothing or exclusion. Limits provisional for one-run draft.'}
    stem.with_suffix('.provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    stem.with_suffix('.caption.md').write_text(
        '# Run 6 calibration: full-field agreement (design draft)\n\n'
        '(a) Measured sheet height; (b) calibrated prediction (Attempt 209); '
        '(c) signed vertical error, prediction minus measurement. All coordinates are in mm. '
        'Existing material-frame alignment is unchanged. All 14,504 accepted scan points are shown; '
        'model height at each scan XY is interpolated on the native projected triangles, '
        'with all points supported and no extrapolation. The prediction map uses all 15,251 vertices '
        'and native faces; color interpolation is linear, not a smoothing fit. '
        'Height scales are shared; error limits are symmetric and no values are clipped. '
        'Limits are provisional for this single-run draft and must be shared across Runs 6, 9, 10 later.\n\n'
        'One calibration specimen, not independent validation. Signed-height RMS: 0.651 mm; '
        '95th percentile absolute height error: 1.229 mm; mean signed error: +0.073 mm. '
        'These pointwise summaries have no independence-based uncertainty intervals. '
        'The recorded symmetric dual-area geometric RMS is 1.182 mm: a different metric, '
        'not the RMS of panel (c). Existing preprocessing/accepted-point filtering is inherited.\n\n'
        'Regenerate: `python calibration/analysis/run6_calibration_redesign.py`. '
        'Input hashes, full metrics and versions are in the adjacent provenance JSON. '
        'Original figures and Run 7 data are preserved.\n')
    print(stem)
    print(json.dumps({k: row['statistics'][k] for k in ('rms_dz_mm','p95_absolute_dz_mm','supported_points')}))


if __name__ == '__main__':
    main()
