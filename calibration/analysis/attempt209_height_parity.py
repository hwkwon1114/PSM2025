#!/usr/bin/env python3
"""Full supported-scan height parity, saved Attempt 209 comparisons only."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.convention_corrected_figures import load
from calibration.analysis.attempt209_height_correlation import summarize

OUT = ROOT / 'calibration/paper_figures/attempt209_publication/attempt209_height_correlation_percent_final_11pt'
STYLE = Path('/home/pxl1051/.pi/agent/skills/scientific-visualization/assets/publication.mplstyle')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f'height_parity.{ext}' for ext in ('pdf', 'png')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Prototype outputs exist')
    pins, rows = {}, []
    for name in ('run6', 'run9', 'run10'):
        row = load(name, pins)
        mask = row['support'] & np.isfinite(row['dz'])
        experiment = row['scan'][mask, 2]
        model = experiment + row['dz'][mask]
        rows.append((name, experiment, model, summarize(experiment, model), int((~mask).sum())))
    maxabs = max(max(np.abs(exp).max(), np.abs(mod).max()) for _, exp, mod, _, _ in rows)
    limit = float(np.ceil(maxabs))
    edges = np.linspace(-limit, limit, int(2*limit/.25)+1)
    grids = [np.histogram2d(exp, mod, bins=[edges, edges])[0] / len(exp) for _, exp, mod, _, _ in rows]
    assert all(np.isclose(g.sum(), 1) for g in grids)
    vmax = max(float(g.max()) for g in grids)
    with plt.style.context(str(STYLE)):
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                             'xtick.labelsize': 11, 'ytick.labelsize': 11, 'pdf.fonttype': 42})
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharex=True, sharey=True, layout='constrained')
        for i, (ax, (_, exp, mod, stats, _), grid) in enumerate(zip(axes, rows, grids)):
            image = ax.pcolormesh(edges, edges, np.ma.masked_equal(grid.T, 0),
                                  vmin=0, vmax=vmax, cmap='Blues', shading='flat', rasterized=True)
            slope, intercept = np.polyfit(exp, mod, 1)
            xfit = np.array([exp.min(), exp.max()])
            ax.plot(xfit, slope*xfit + intercept, color='#d55e00', linestyle='--', linewidth=1.3,
                    label='Linear fit')
            stats['linear_fit_slope'] = float(slope)
            stats['linear_fit_intercept_mm'] = float(intercept)
            ax.set(xlim=(-limit,limit), ylim=(-limit,limit), aspect='equal',
                   xlabel='Experimental height y (mm)', title=f'({chr(97+i)}) Run {i+1}')
            ax.text(.04, .96, f"r = {stats['pearson_r']:.3f}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=11,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
            if i == 0:
                ax.set_ylabel('Model height y (mm)')
        axes[0].legend(loc='lower right', frameon=False, fontsize=11)
        colorbar = fig.colorbar(image, ax=axes, orientation='horizontal', shrink=.67, pad=.13,
                                label='Scan points per bin (%)')
        colorbar.set_ticks(np.arange(0, vmax + 1e-12, .01))
        colorbar.formatter = PercentFormatter(xmax=1, decimals=0, symbol='')
        colorbar.update_ticks()
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in axes:
            text=ax.texts[0].get_window_extent(renderer)
            if not ax.bbox.contains(text.x0,text.y0) or not ax.bbox.contains(text.x1,text.y1):
                raise ValueError('Annotation outside axes')
        fig.savefig(targets[0]); fig.savefig(targets[1], dpi=250)
        plt.close(fig)
    provenance = dict(input_sha256=pins, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      axis_limit_mm=limit, bin_width_mm=.25, bins=edges.tolist(), vmax_fraction=vmax,
                      matplotlib=matplotlib.__version__, numpy=np.__version__,
                      cases={name:{**stats,'unsupported':missing} for name, _, _, stats, missing in rows})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUT/'height_parity.caption.md').write_text(
        '# Height comparison at supported scan points\n\n'
        'Experimental height and model height at the same retained scan x–z coordinates. '
        'The model is interpolated on its native projected triangular facets; unsupported points are excluded. '
        'Darker blue indicates a higher percentage of supported scan points with that combination of '
        'experimental and model heights. Each bin covers 0.25 × 0.25 mm, independently normalized '
        'per case and shown on the same color scale; white cells contain no points. '
        'Dashed line: ordinary least-squares fit of model height on experimental height, '
        'drawn over the observed experimental height range for each case; it is not an equal-height line. '
        'Pearson r describes spatial height correlation and does not quantify absolute agreement. '
        'Axes share limits and equal physical scales. Runs 1–3 here are historical run6, run9 and run10; '
        'Run 3 here is not the collaborator HW1 called Run 3 in the separate shape-trends figure. '
        'Run 1 was used for calibration; '
        'Runs 2 and 3 are previously inspected exploratory transfers. One specimen per case; '
        'spatial points are dependent and equally point-weighted, not independent replicates. '
        'Saved material-reference frames were used; no new registration or solve.\n')
    print(json.dumps({'axis_limit_mm':limit,'vmax_fraction':vmax,'cases':provenance['cases']},indent=2))

if __name__ == '__main__':
    main()
