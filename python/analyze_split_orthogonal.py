#!/usr/bin/env python3
"""Compare completed split-half runs and matched crown implementations."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_sequence_ablation_v2 import load_vtp,point_array,proper_kabsch,metric_vector

ROOT=Path(__file__).resolve().parents[1]/'run/forward_model_diagnostics'


def load(directory):
    r=json.loads((directory/'result.json').read_text())
    assert r['return_code']==0
    d,x,t=load_vtp(directory/r['final_vtp'])
    uv=np.column_stack([point_array(d,n) for n in ('material_u','material_v')])
    y,rms,_=proper_kabsch(np.column_stack([uv,np.zeros(len(uv))]),x)
    record=dict(case=directory.name,height_mm=float(np.ptp(y[:,2])*1000),deformation_rms_mm=float(rms*1000))
    convergence=directory/'sequence_convergence.csv'
    if convergence.exists():
        with convergence.open() as f: rows=list(csv.DictReader(f))
        record.update(solves=len(rows),all_accepted=all(row['equilibrium_accepted']=='1' for row in rows),max_gradient=max(float(row['final_gradient_norm']) for row in rows))
    return dict(record=record,x=x,y=y,uv=uv,tri=t,metric=metric_vector(d))


def compare(a,b):
    np.testing.assert_array_equal(a['tri'],b['tri'])
    np.testing.assert_array_equal(a['uv'],b['uv'])
    aligned,rms,maximum=proper_kabsch(a['x'],b['x'])
    return dict(left=a['record']['case'],right=b['record']['case'],rms_mm=float(rms*1000),max_mm=float(maximum*1000),metric_relative_difference=float(np.max(np.abs(a['metric']-b['metric']))/max(np.max(np.abs(a['metric'])),np.max(np.abs(b['metric'])))))


def main():
    out=ROOT/'split_orthogonal_comparison';out.mkdir(exist_ok=True)
    names=[f'split_orthogonal_{order}_{schedule}' for order in ('AB','BA') for schedule in ('whole','pairs','strips')]
    runs=[load(ROOT/'current'/name) for name in names]
    comparisons=[compare(runs[i],runs[i+3]) for i in range(3)]
    comparisons += [compare(runs[base],runs[base+j]) for base in (0,3) for j in (1,2)]
    fig,axes=plt.subplots(2,3,figsize=(13,9),layout='constrained')
    limit=max(np.max(np.abs(r['y'][:,2])) for r in runs)*1000
    for ax,r in zip(axes.flat,runs):
        image=ax.tripcolor(r['uv'][:,0]*1000,r['uv'][:,1]*1000,r['tri'],r['y'][:,2]*1000,shading='gouraud',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.axvline(0,color='0.4',ls='--',lw=.7);ax.set_aspect('equal')
        ax.set(title=r['record']['case'].replace('split_orthogonal_','')+f" | span {r['record']['height_mm']:.1f} mm",xlabel='material u (mm)',ylabel='material v (mm)')
    fig.colorbar(image,ax=list(axes.flat),label='Aligned out-of-plane displacement (mm)',shrink=.8)
    fig.suptitle('Split-half orthogonal paths: same color scale for all six final shapes\nA = vertical path on left; B = horizontal path on right. Top: A→B; bottom: B→A.')
    fig.savefig(out/'final_shapes.png',dpi=170);plt.close(fig)
    crown=[load(ROOT/impl/'collaborator_crown_iter200000') for impl in ('upstream','current')]
    crown_comparison=compare(*crown)
    crown_comparison.update(left_implementation='upstream',right_implementation='current')
    report=dict(cases=[r['record'] for r in runs],comparisons=comparisons,crown_cases=[r['record'] for r in crown],crown_comparison=crown_comparison)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
