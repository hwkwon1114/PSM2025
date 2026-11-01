"""Render the nine saved, accepted 2 mm turned-over shapes and local-curvature signs."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_design.analysis import height, local_curvature
from shape_design.plot_style import curvature_cmap_and_norm, curvature_legend_handles, format_sheet_ax

RESULTS = ROOT / 'run/shape_design/mesh_comparison/results.json'
OUT = ROOT / 'run/shape_design/family_maps'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    data = json.loads(RESULTS.read_text())
    cases = []
    for row in data['coarse']:
        k = row['k']
        for filename, expected in row['inputs'].items():
            if sha(ROOT / filename) != expected:
                raise ValueError(f'Changed input: {filename}')
        stage = 'pilot' if k in (-4, 0, 4) else 'sweep_flip'
        folder = ROOT / 'run/shape_design' / stage / f'flip_k{k:+d}'
        record = json.loads((folder / 'result.json').read_text())
        if record['status'] != 'accepted' or record['hit_check']['mismatched_faces'] != 0:
            raise ValueError(f'Invalid result: {folder}')
        with np.load(folder / 'shape.npz', allow_pickle=False) as d:
            ref, pos, faces = (d[n].copy() for n in ('reference_mm', 'aligned_mm', 'faces'))
        h = height(pos)
        loc = local_curvature(ref, pos, faces)
        cases.append((k, ref, faces, h, loc))
    assert [case[0] for case in cases] == list(range(-4, 5))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42}):
        for kind in ('height', 'local_curvature_sign'):
            fig, axes = plt.subplots(3, 3, figsize=(10.8, 7.8), sharex=True, sharey=True, layout='constrained')
            lim = max(np.abs(row[3]).max() for row in cases)
            cmap, norm = curvature_cmap_and_norm()
            for ax, (k, ref, faces, h, loc) in zip(axes.flat, cases):
                tri = mtri.Triangulation(ref[:, 0], ref[:, 1] + 65, faces)
                if kind == 'height':
                    artist = ax.tripcolor(tri, h, shading='gouraud', cmap='RdBu_r',
                                         norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), rasterized=True)
                else:
                    sign = np.where(loc['valid'], np.sign(loc['gauss_per_m2']), 0)
                    ax.tripcolor(tri, facecolors=sign, shading='flat', cmap=cmap, norm=norm, rasterized=True)
                format_sheet_ax(ax, title=f'k = {k:+d}')
                summary[str(k)] = {key: loc[key] for key in ('positive_area_fraction','negative_area_fraction','invalid_area_fraction','median_gauss_per_m2')}
            for ax in axes[-1]: ax.set_xlabel('Width x (mm)')
            for ax in axes[:, 0]: ax.set_ylabel('Length z (mm)')
            if kind == 'height':
                fig.colorbar(artist, ax=axes, shrink=.75, label='Mean-removed height y (mm)')
            else:
                fig.legend(handles=curvature_legend_handles(),
                           loc='outside lower center', ncol=3, frameon=False)
            for ext in ('pdf','png'): fig.savefig(OUT / f'family_{kind}.{ext}', dpi=220)
            plt.close(fig)
    (OUT / 'results.json').write_text(json.dumps({'cases':summary,'results_sha256':sha(RESULTS),
        'script_sha256':sha(Path(__file__)), 'local_radius_mm':15},indent=2)+'\n')
    (OUT / 'caption.md').write_text('Nine deterministic turned-over 2 mm model shapes, k = −4 to +4, ordered left to right then top to bottom. Height is whole-sheet mean-removed after full-grid rigid alignment, with one shared symmetric scale. Gaussian-curvature sign uses 15 mm local material-neighborhood fits; grey denotes zero or invalid. Dashed rectangle is the fixed treated ROI. Area fractions are reference-area weighted within the ROI; invalid faces stay in the denominator. No physical replication, experimental validation or mesh convergence is implied. The separate one-face k = −4 pilot failed and is not included in this turned-over-only family. Public axes are width x, length z, height y.\n')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
