#!/usr/bin/env python3
"""Show saved equilibria during second-half loading; never interpolate cycles."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_sequence_ablation_v2 import load_vtp,point_array,proper_kabsch
from visualize_solver_characteristics import CYCLE_RE

ROOT=Path(__file__).resolve().parents[1]/'run/forward_model_diagnostics'


def main():
    out=ROOT/'split_orthogonal_comparison'
    all_runs=[]
    for order in ('AB','BA'):
        for schedule in ('whole','pairs','strips'):
            directory=ROOT/'current'/f'split_orthogonal_{order}_{schedule}'
            paths=json.loads((directory/'sequence.json').read_text())['toolpaths']
            first_id=paths[0]['id'].split('_s')[0]
            boundary=sum(p['id'].split('_s')[0]==first_id for p in paths)
            with (directory/'sequence_convergence.csv').open() as f: accepted=list(csv.DictReader(f))
            assert len(accepted)==len(paths) and all(r['equilibrium_accepted']=='1' for r in accepted)
            saved={}
            for filename in directory.glob('*_final.vtp'):
                match=CYCLE_RE.search(filename.name)
                if match: saved[int(match.group(1))]=filename
            assert set(saved)==set(range(1,len(paths)+1))
            total=len(paths)-boundary
            offsets=[0,None,None,None,1] if total==1 else [0,round(total*.25),round(total*.5),round(total*.75),total]
            frames=[]
            for offset in offsets:
                if offset is None:
                    frames.append(None);continue
                cycle=boundary+offset
                d,x,t=load_vtp(saved[cycle])
                uv=np.column_stack([point_array(d,n) for n in ('material_u','material_v')])
                y,rms,_=proper_kabsch(np.column_stack([uv,np.zeros(len(uv))]),x)
                strips=sum(len(p['operation']['active_strips']) for p in paths[boundary:cycle])
                frames.append(dict(cycle=cycle,second_strips=strips,x=x,y=y,uv=uv,tri=t,height_mm=float(np.ptp(y[:,2])*1000)))
            _,change,maximum=proper_kabsch(frames[0]['x'],frames[-1]['x'])
            all_runs.append(dict(order=order,schedule=schedule,boundary=boundary,frames=frames,second_half_shape_change_rms_mm=float(change*1000),second_half_shape_change_max_mm=float(maximum*1000)))
    limit=max(np.max(np.abs(f['y'][:,2]))*1000 for r in all_runs for f in r['frames'] if f is not None)
    fig,axes=plt.subplots(6,5,figsize=(15,19),layout='constrained')
    for row,run in enumerate(all_runs):
        for col,frame in enumerate(run['frames']):
            ax=axes[row,col]
            if frame is None:
                ax.text(.5,.5,'No intermediate solve\n(whole second path\napplied in one step)',ha='center',va='center',transform=ax.transAxes,color='0.4');ax.axis('off');continue
            y,uv,t=frame['y'],frame['uv'],frame['tri']
            image=ax.tripcolor(uv[:,0]*1000,uv[:,1]*1000,t,y[:,2]*1000,shading='gouraud',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.axvline(0,color='0.4',ls='--',lw=.6);ax.set_aspect('equal')
            ax.set_title(f"Cycle {frame['cycle']}: {frame['second_strips']} second-half strips\nheight span {frame['height_mm']:.1f} mm",fontsize=9)
            if col==0: ax.set_ylabel(run['order']+' / '+run['schedule']+'\nmaterial v (mm)')
            if row==5: ax.set_xlabel('material u (mm)')
    fig.colorbar(image,ax=list(axes.flat),label='Out-of-plane displacement after rigid alignment (mm)',shrink=.65)
    fig.suptitle('First half finished → loading the second half → final shape\nA: vertical path on left. B: horizontal path on right. Same color scale throughout.\nEach panel is a saved accepted equilibrium; intermediate strip counts differ between orders.',fontsize=14)
    fig.savefig(out/'second_half_progression.png',dpi=170);fig.savefig(out/'second_half_progression.pdf');plt.close(fig)
    summary=[]
    for r in all_runs:
        record={k:v for k,v in r.items() if k!='frames'}
        record['frames']=[None if f is None else {k:f[k] for k in ('cycle','second_strips','height_mm')} for f in r['frames']]
        summary.append(record)
    (out/'second_half_progression.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
