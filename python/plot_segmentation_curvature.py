#!/usr/bin/env python3
"""Compare final high-growth principal curvatures on identical material faces."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_sequence_ablation_v2 import load_vtp, point_array, cell_array, areas

ROOT = Path(__file__).resolve().parents[1] / 'run/forward_model_diagnostics'
LABELS = ['All at once', 'All at once: repeat', 'Pairs', 'Individual strips']


def main():
    out = ROOT / 'segmentation_curvature'
    out.mkdir(exist_ok=True)
    fields, diagnostics = [], []
    for suffix in ('all', 'all_repeat', 'pairs', 'strips'):
        directory = ROOT / 'current' / f'segmentation_high_{suffix}'
        result = json.loads((directory / 'result.json').read_text())
        if result['return_code'] != 0:
            raise RuntimeError(f'Unsuccessful case: {directory}')
        data, _, faces = load_vtp(directory / result['final_vtp'])
        material = np.column_stack([point_array(data, n) for n in ('material_u', 'material_v')])
        if not fields:
            uv, tri = material, faces
            weights = areas(np.column_stack([uv, np.zeros(len(uv))]), tri)
            weights /= weights.sum()
        else:
            np.testing.assert_array_equal(faces, tri)
            np.testing.assert_array_equal(material, uv)
        h, k = cell_array(data, 'mean'), cell_array(data, 'gauss')
        disc = h*h-k
        if not np.all(np.isfinite(disc)) or np.any(disc < -1e-10*np.maximum(1, h*h+np.abs(k))):
            raise RuntimeError(f'Invalid principal-curvature discriminant: {directory}')
        radius = np.sqrt(np.maximum(disc, 0))
        fields.append(np.array([h+radius, h-radius]))
        diagnostics.append(dict(case=directory.name,minimum_discriminant=float(disc.min())))
    fields = np.array(fields)
    delta = fields-fields[0]
    for i, record in enumerate(diagnostics):
        record['area_weighted_curvature_rms_m_inv'] = np.sqrt((fields[i]**2) @ weights).tolist()
        record['area_weighted_difference_rms_m_inv'] = np.sqrt((delta[i]**2) @ weights).tolist()
        record['max_abs_difference_m_inv'] = np.max(np.abs(delta[i]),axis=1).tolist()
    metadata = dict(cases=diagnostics,definition='k_plus=H+sqrt(H^2-K), k_minus=H-sqrt(H^2-K); algebraically ordered, not magnitude-ordered or fixed material directions',weighting='reference material triangle area',smoothing='none',units='1/m')
    (out/'comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')

    fig, axes = plt.subplots(4,4,figsize=(14,14),layout='constrained')
    names = [r'$k_+$', r'$k_-$']
    for row in range(4):
        component = row % 2
        values = fields[:,component] if row < 2 else delta[:,component]
        limit = max(float(np.max(np.abs(values))), 1e-12)
        for col in range(4):
            ax = axes[row,col]
            image = ax.tripcolor(uv[:,0]*1000,uv[:,1]*1000,tri,facecolors=values[col],cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlabel('x (mm)')
            if col == 0:
                ax.set_ylabel(('Curvature ' if row < 2 else 'Difference ') + names[component]+'\ny (mm)')
            rms = np.sqrt(weights @ values[col]**2)
            ax.set_title(f'{LABELS[col]}\narea-weighted RMS = {rms:.3f} /m')
        fig.colorbar(image,ax=list(axes[row]),label='1/m',shrink=.9)
    fig.suptitle('High-growth segmentation: principal curvature and differences\nTop: raw per-face curvature. Bottom: case minus all-at-once. One shared color scale per row.',fontsize=15)
    for ext in ('png','pdf'):
        fig.savefig(out/f'principal_curvature_maps.{ext}',dpi=180)
    plt.close(fig)

    # Transverse bins suppress the distracting triangle texture without changing
    # the raw maps. These are material-area-weighted band averages, not curvature
    # evaluated on a centerline or curvature of an averaged surface.
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    centroid = uv[tri].mean(axis=1)
    colors = ['black','0.55','tab:orange','tab:blue']
    for dim in range(2):
        edges = np.linspace(uv[:,dim].min(),uv[:,dim].max(),33)
        bins = np.clip(np.searchsorted(edges,centroid[:,dim],side='right')-1,0,31)
        den = np.bincount(bins,weights=weights,minlength=32)
        centers = (edges[:-1]+edges[1:])*500
        for component in range(2):
            ax=axes[component,dim]
            for i in range(4):
                values=np.bincount(bins,weights=weights*fields[i,component],minlength=32)
                average=np.divide(values,den,out=np.full(32,np.nan),where=den>0)
                ax.plot(centers,average,label=LABELS[i],color=colors[i],linestyle='--' if i==1 else '-',linewidth=1.8)
                ax.set_xlabel(f'{"xy"[dim]} (mm)')
            ax.set_ylabel(names[component]+' (1/m)')
            ax.set_title('Area-weighted band average across '+('y' if dim==0 else 'x'))
            ax.grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle('Same broad curvature pattern; compare amplitude and distribution\nBand averages of signed principal values—not fixed-direction bending curvatures.',fontsize=14)
    for ext in ('png','pdf'):
        fig.savefig(out/f'principal_curvature_profiles.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(metadata,indent=2))
    print(f'Figures written to {out}')


if __name__ == '__main__':
    main()
