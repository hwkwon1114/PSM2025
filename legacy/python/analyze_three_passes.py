#!/usr/bin/env python3
"""Compare successful three-pass schedules, excluding failed partial states."""
import json
import csv
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_split_orthogonal import ROOT, load, compare


def main():
    out=ROOT/'three_pass_comparison';out.mkdir(exist_ok=True)
    labels=['Strip ×3, solve','Pair ×3, solve','A, solve; B, solve (×3)','A+B, solve (×3)']
    names=['strip_triplets','pair_triplets','each_path','each_AB']
    runs=[load(ROOT/'current'/('split_three_'+n)) for n in names]
    for r in runs:
        with (ROOT/'current'/r['record']['case']/'sequence_convergence.csv').open() as f:
            rows=list(csv.DictReader(f))
        solves=[v for v in rows if v['minimization_performed']=='1']
        r['record'].update(solves=len(solves),all_accepted=all(v['equilibrium_accepted']=='1' for v in solves),max_gradient=max(float(v['final_gradient_norm']) for v in solves))
    report=dict(cases=[r['record'] for r in runs],comparisons=[compare(a,b) for a,b in combinations(runs,2)],excluded='all_at_once: original failed; million-iteration retry not yet complete')
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,4,figsize=(15,5),layout='constrained')
    limit=max(np.max(np.abs(r['y'][:,2])) for r in runs)*1000
    for ax,r,label in zip(axes,runs,labels):
        im=ax.tripcolor(r['uv'][:,0]*1000,r['uv'][:,1]*1000,r['tri'],r['y'][:,2]*1000,shading='gouraud',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_aspect('equal');ax.axvline(0,color='0.4',ls='--',lw=.6)
        ax.set(title=label+f"\nheight span {r['record']['height_mm']:.1f} mm",xlabel='material u (mm)',ylabel='material v (mm)')
    fig.colorbar(im,ax=list(axes),label='Aligned out-of-plane displacement (mm)',shrink=.8)
    fig.suptitle('Three applications per strip at g=0.0015: completed schedules only\nSame color scale; all-at-once failed and is not shown as a completed result.')
    fig.savefig(out/'final_shapes.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
