#!/usr/bin/env python3
"""Signed height-error histogram for saved corrected-convention cases."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.convention_corrected_figures import load

OUT = ROOT / 'calibration/paper_figures/attempt209_publication/attempt209_signed_error_11pt'
STYLE = Path('/home/pxl1051/.pi/agent/skills/scientific-visualization/assets/publication.mplstyle')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f'signed_height_error_density.{ext}' for ext in ('pdf', 'png')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Prototype already exists; preserve existing output')
    pins, rows = {}, []
    for name in ('run6', 'run9', 'run10'):
        row = load(name, pins)
        good = row['support'] & np.isfinite(row['dz'])
        rows.append((name, row['dz'][good], int((~good).sum())))
    # Identical, zero-centred 0.2 mm bins cover every observation.
    maxabs = max(float(np.max(np.abs(v))) for _, v, _ in rows)
    halfbins = int(np.ceil(maxabs / 0.2 - 0.5))
    edges = (np.arange(-halfbins, halfbins + 2) - 0.5) * 0.2
    assert all(d.min() >= edges[0] and d.max() <= edges[-1] for _, d, _ in rows)
    stats = {}
    with plt.style.context(str(STYLE)):
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                             'xtick.labelsize': 11, 'ytick.labelsize': 11, 'pdf.fonttype': 42})
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True, sharey=True, layout='constrained')
        for i, (ax, (name, d, unsupported)) in enumerate(zip(axes, rows)):
            density, _ = np.histogram(d, bins=edges, density=True)
            assert np.isclose(np.sum(density*np.diff(edges)), 1.)
            ax.stairs(density, edges, fill=True, color='#0072B2', alpha=.32, linewidth=0)
            ax.stairs(density, edges, color='#0072B2', linewidth=1.1)
            ax.axvline(0, color='#333333', linestyle='--', linewidth=.8)
            ax.set_title(f'({chr(97+i)}) Run {i+1}', loc='left')
            ax.set_xlabel('Height error Δy (mm)')
            ax.set_xlim(edges[0], edges[-1])
            ax.grid(axis='y', color='#dedede', linewidth=.5)
            stats[name] = dict(n=len(d), unsupported=unsupported,
                               q05_mm=float(np.quantile(d,.05)), median_mm=float(np.median(d)),
                               q95_mm=float(np.quantile(d,.95)), min_mm=float(d.min()), max_mm=float(d.max()))
        axes[0].set_ylabel('Probability density (mm⁻¹)')
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        bounds=fig.bbox
        checked = [a for ax in axes for a in (ax.title, ax.xaxis.label, ax.yaxis.label)]
        for artist in checked:
            if artist.get_visible() and artist.get_text().strip():
                box=artist.get_window_extent(renderer)
                if box.x0 < bounds.x0-1 or box.y0 < bounds.y0-1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1:
                    raise ValueError('Text outside canvas: '+artist.get_text())
        fig.savefig(targets[0]); fig.savefig(targets[1], dpi=250)
        plt.close(fig)
    provenance=dict(description='Per-case normalized histogram of model minus scan height at supported scan coordinates',
                    bin_width_mm=.2, bin_edges_mm=edges.tolist(), stats=stats, input_sha256=pins,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    matplotlib=matplotlib.__version__, numpy=np.__version__)
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (OUT/'signed_height_error_density.caption.md').write_text(
        '# Signed height-error distributions\n\n'
        'Signed height difference Δy = model − experiment at retained scan x–z positions; '
        'model height is interpolated on native projected triangular facets. Histograms use all supported '
        'scan points with equal point weights and identical 0.2 mm bins, normalized independently to '
        'unit area. Positive values mean the model is higher. Dashed line: zero. '
        'Runs 1–3 here are historical run6, run9 and run10; Run 3 here is not the collaborator HW1 '
        'called Run 3 in the separate shape-trends figure. Run 1 was used for calibration; '
        'Runs 2 and 3 are previously inspected exploratory transfers. '
        'Each case is one specimen; spatial samples are dependent, not independent replicates. '
        'These vertical differences are not the symmetric Euclidean surface-distance objective. '
        'Saved material-reference frames were used; no new registration or solve. '
        'Density depends on bin width and scan sampling density; no uncertainty is implied.\n')
    print(json.dumps({'bin_edges':[float(edges[0]),float(edges[-1])], 'stats':stats},indent=2))

if __name__ == '__main__':
    main()
