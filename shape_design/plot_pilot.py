"""Summarize accepted Attempt 209 pilot shapes; no solver or new optimization."""
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
from shape_design.analysis import height, local_curvature, quadratic_fit
from shape_design.plot_style import curvature_cmap_and_norm, curvature_legend_handles, format_sheet_ax

BASE = ROOT / 'run/shape_design/pilot'
OUT = ROOT / 'run/shape_design/pilot_analysis'
CASES = [('flip', -4), ('flip', 0), ('flip', 4),
         ('oneface', -4), ('oneface', 0), ('oneface', 4)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for variant, k in CASES:
        directory = BASE / f'{variant}_k{k:+d}'
        result_path = directory / 'result.json'
        result = json.loads(result_path.read_text())
        row = {'variant': variant, 'k': k, 'status': result['status'],
               'result_sha256': sha(result_path), 'elapsed_s': result['elapsed_s']}
        if result['status'] == 'accepted':
            assert result['hit_check']['mismatched_faces'] == 0
            path = directory / 'shape.npz'
            with np.load(path) as data:
                ref = data['reference_mm'].copy()
                aligned = data['aligned_mm'].copy()
                faces = data['faces'].copy()
            h = height(aligned)
            q = quadratic_fit(ref, h)
            local = local_curvature(ref, aligned, faces)
            row.update(shape_sha256=sha(path), height_range_mm=float(np.ptp(h)), quadratic=q,
                       local={key: local[key] for key in ('positive_area_fraction', 'negative_area_fraction',
                                                         'invalid_area_fraction', 'median_gauss_per_m2')})
            row['_ref'], row['_faces'], row['_height'] = ref, faces, h
            row['_K'], row['_valid'] = local['gauss_per_m2'], local['valid']
        else:
            row['error'] = result.get('error')
        records.append(row)

    style = {'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
             'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42}
    with plt.rc_context(style):
        for kind in ('height', 'curvature_sign'):
            fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.7), layout='constrained', sharex=True, sharey=True)
            if kind == 'height':
                limit = max(np.abs(r['_height']).max() for r in records if r['status'] == 'accepted')
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            else:
                cmap, norm = curvature_cmap_and_norm()
            for ax, r in zip(axes.flat, records):
                tag = 'Both faces' if r['variant'] == 'flip' else 'One face'
                format_sheet_ax(ax, title=f"{tag}, k = {r['k']:+d}")
                if r['status'] != 'accepted':
                    ax.set_facecolor('#f3f3f3')
                    ax.text(.5, .5, 'Solver failed\n(no shape)', transform=ax.transAxes,
                            ha='center', va='center', color='#555555')
                    continue
                ref, faces = r['_ref'], r['_faces']
                tri = mtri.Triangulation(ref[:, 0], ref[:, 1] + 65, faces)
                if kind == 'height':
                    mappable = ax.tripcolor(tri, r['_height'], cmap='RdBu_r', norm=norm, shading='gouraud', rasterized=True)
                else:
                    sign = np.where(r['_valid'], np.sign(r['_K']), 0)
                    ax.tripcolor(tri, facecolors=sign, cmap=cmap, norm=norm, shading='flat', rasterized=True)
            for ax in axes[-1]: ax.set_xlabel('Width x (mm)')
            for ax in axes[:, 0]: ax.set_ylabel('Length z (mm)')
            if kind == 'height':
                fig.colorbar(mappable, ax=axes, shrink=.82, label='Mean-removed height y (mm)')
            else:
                fig.legend(handles=curvature_legend_handles(),
                           loc='outside lower center', ncol=3, frameon=False)
            for ext in ('pdf', 'png'):
                fig.savefig(OUT / f'pilot_{kind}.{ext}', dpi=220)
            plt.close(fig)

    serial = [{k: v for k, v in row.items() if not k.startswith('_')} for row in records]
    (OUT / 'results.json').write_text(json.dumps({'cases': serial, 'script_sha256': sha(Path(__file__)),
        'local_radius_mm': 15, 'roi_mm': {'u': [-100, 100], 'v': [-52, 52]},
        'height_alignment': 'full-grid rigid Kabsch fit to flat reference; mean removed',
        'public_mapping': '(x,y,z)=(u,height,v+65)', 'mesh_mm': 2}, indent=2) + '\n')
    (OUT / 'caption.md').write_text('''# Attempt 209 double-curvature pilot\n\nFive accepted shapes from six attempted frozen-parameter simulations; the one-face k = −4 case failed and has no accepted shape. Rows: both-face and one-face treatment; columns: k = −4, 0, +4. Dashed rectangle is the pre-specified treated ROI. Height maps share a single symmetric colour scale and show Kabsch-aligned, whole-sheet mean-removed height. Curvature-sign maps show the sign of local Gaussian curvature K from 15 mm material-neighborhood fits; grey marks exact zero or an invalid fit. The plotted field covers the whole sheet; reported fractions use reference triangle areas within the ROI, with invalid faces in the denominator. Public coordinates are x = width, z = length and y = height. The 2 mm mesh, new sheet thickness, crosswise passes, and back-face model are extrapolations; no mesh convergence or experimental validation is established. Samples are five deterministic simulated shapes, not independent replicates.\n''')
    print(json.dumps(serial, indent=2))


if __name__ == '__main__':
    main()
