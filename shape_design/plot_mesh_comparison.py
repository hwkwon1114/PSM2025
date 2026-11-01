"""Plot saved turned-over family PCA and endpoint mesh comparisons; no solver runs."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'run/shape_design/mesh_comparison/results.json'
PCA = ROOT / 'run/shape_design/mesh_comparison/pca.npz'
OUT = ROOT / 'run/shape_design/mesh_comparison_figures'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    data = json.loads(RESULTS.read_text())
    # Verify every saved simulation record used by the analysis, not only the summary.
    for row in data['coarse']:
        for path, expected in row['inputs'].items():
            if digest(ROOT / path) != expected:
                raise ValueError(f'Changed input: {path}')
    for endpoint in data['endpoints'].values():
        for path, expected in endpoint['fine_inputs'].items():
            if digest(ROOT / path) != expected:
                raise ValueError(f'Changed input: {path}')
    with np.load(PCA, allow_pickle=False) as archive:
        ref, components, scores, ks = (archive[key].copy() for key in ('reference_mm', 'components', 'scores', 'k'))
    assert np.array_equal(ks, np.arange(-4, 5))
    assert np.allclose(scores[:, 0], [row['pc1'] for row in data['coarse']])
    # Plot true mesh connectivity from a saved full-precision shape (no resampling).
    with np.load(ROOT / 'run/shape_design/pilot/flip_k-4/shape.npz', allow_pickle=False) as shape:
        assert np.allclose(shape['reference_mm'], ref, atol=1e-9, rtol=0)
        faces = shape['faces'].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    style = {'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
             'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        tri = mtri.Triangulation(ref[:, 0], ref[:, 1] + 65, faces)
        limit = float(np.max(np.abs(components[0])))
        c = ax[0, 0].tripcolor(tri, components[0], shading='gouraud', cmap='RdBu_r',
                               vmin=-limit, vmax=limit, rasterized=True)
        ax[0, 0].set_aspect('equal'); ax[0, 0].set(xlabel='Width x (mm)', ylabel='Length z (mm)',
            title='(a) First height-field mode')
        fig.colorbar(c, ax=ax[0, 0], label='PC1 loading (unit norm)', shrink=.82)
        ax[0, 1].plot(ks, scores[:, 0], 'o-', color='#0072B2', label='2 mm')
        fine_scores = [data['endpoints'][str(k)]['fine_pc1'] for k in (-4, 4)]
        ax[0, 1].plot([-4, 4], fine_scores, 's', mfc='none', color='#D55E00', ms=8, label='1 mm endpoints')
        ax[0, 1].set(xlabel='Coverage parameter k', ylabel='PC1 score (mm)', title='(b) Family scores')
        ax[0, 1].set_xticks(ks); ax[0, 1].legend(frameon=False)
        coarse_k = np.array([r['fit']['K_fit_per_mm2'] for r in data['coarse']]) * 1e6
        ax[1, 0].plot(ks, coarse_k, 'o-', color='#0072B2', label='2 mm')
        fine_k = [data['endpoints'][str(k)]['fine_fit']['K_fit_per_mm2'] * 1e6 for k in (-4, 4)]
        ax[1, 0].plot([-4, 4], fine_k, 's', mfc='none', color='#D55E00', ms=8, label='1 mm endpoints')
        ax[1, 0].axhline(0, color='#777777', lw=.8)
        ax[1, 0].set(xlabel='Coverage parameter k', ylabel=r'Quadratic $K_{fit}$ (m$^{-2}$)',
                     title='(c) Fitted Gaussian curvature')
        ax[1, 0].set_xticks(ks)
        points = [data['endpoints'][str(k)] for k in (-4, 4)]
        x = np.arange(2)
        ax[1, 1].plot(x, [p['height_difference_rms_mm'] for p in points], 'o', color='#0072B2', label='RMS')
        ax[1, 1].plot(x, [p['height_difference_p95_abs_mm'] for p in points], '^', color='#D55E00', label='95th percentile')
        ax[1, 1].plot(x, [p['height_difference_max_abs_mm'] for p in points], 'x', color='#222222', label='Maximum')
        ax[1, 1].set_xticks(x, ['k = −4', 'k = +4'])
        ax[1, 1].set_ylim(bottom=0)
        ax[1, 1].set(ylabel='Absolute height difference (mm)', title='(d) Endpoint mesh differences')
        ax[1, 1].legend(frameon=False, fontsize=9)
        for a in (ax[0, 1], ax[1, 0], ax[1, 1]): a.grid(axis='y', color='#dddddd', lw=.6)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT / f'family_pca_mesh_comparison.{ext}', dpi=240)
        plt.close(fig)
    (OUT / 'family_pca_mesh_comparison.caption.md').write_text(
        'Turned-over Attempt 209 exploratory family. (a) First unit-norm PCA loading of the nine 2 mm height fields. '
        '(b) Scores of the nine coarse shapes and projections of the two 1 mm endpoint shapes onto the same coarse PCA basis. '
        '(c) Gaussian curvature of the global quadratic fit on the treated rectangle. '
        '(d) RMS, unweighted 95th percentile and maximum of absolute 1 mm minus 2 mm mean-removed height differences '
        'at nested coarse vertices, after separately aligning each shape to its flat reference. '
        f"PC1 explains {100*data['pca_explained_ratio'][0]:.2f}% of this nine-shape model sample. "
        'The PC1 sign is oriented to correlate positively with k, and scores are descriptive, not independent replicates. '
        'Only two endpoints and two mesh sizes were checked; this does not prove convergence or physical validation. '
        'A one-face pilot failed and is not part of this turned-over PCA. Public coordinates are x=width, z=length, y=height.\n')
    (OUT / 'provenance.json').write_text(json.dumps({'script_sha256': digest(Path(__file__)),
        'results_sha256': digest(RESULTS), 'pca_sha256': digest(PCA), 'mesh_mm': [2, 1],
        'figure_size_inches': [10, 7], 'n_coarse': len(ks), 'n_fine': 2}, indent=2) + '\n')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
