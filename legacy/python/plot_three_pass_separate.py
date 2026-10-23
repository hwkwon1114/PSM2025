#!/usr/bin/env python3
"""Separate actual-input and saved-equilibrium figures for two three-pass schedules."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from analyze_split_orthogonal import ROOT
from analyze_sequence_ablation_v2 import load_vtp,point_array,proper_kabsch
from visualize_solver_characteristics import place_strips,capsule_polygon,CYCLE_RE


def main():
 out=ROOT/'three_pass_comparison'
 for suffix,cycles in [('strip_triplets',[6,12,24,42,66,90]),('all_at_once_iter1000000',[6])]:
  directory=ROOT/'current'/('split_three_'+suffix)
  seq=json.loads((directory/'sequence.json').read_text());m=json.loads((directory/'manifest.json').read_text());opts=dict(zip(m['command'][1::2],m['command'][2::2]))
  saved={int(match.group(1)):p for p in directory.glob('*_final.vtp') if (match:=CYCLE_RE.search(p.name))}
  with (directory/'sequence_convergence.csv').open() as f: records=list(csv.DictReader(f))
  fig,axes=plt.subplots(2,4,figsize=(16,10),layout='constrained');axes=list(axes.flat)
  unique={}
  for p in seq['toolpaths']:
   op=dict(p['operation']);op.pop('active_strips',None);unique[json.dumps(op,sort_keys=True)]=op
  ax=axes[0];ax.add_patch(Rectangle((-127,-152.4),254,304.8,fc='#fafafa',ec='black'))
  for op,color in zip(unique.values(),['#2466ae','#d37720']):
   for s in place_strips(op,(-.127,.127,-.1524,.1524))['strips']:
    a,b=s['a_uv_m'],s['b_uv_m'];ax.add_patch(Polygon(capsule_polygon(a,b,.006)*1000,fc=color,ec='none',alpha=.2));ax.plot([a[0]*1000,b[0]*1000],[a[1]*1000,b[1]*1000],color=color,lw=1)
  ax.set_title('Shared footprint: A left, B right\nEach strip applied three times');ax.set_aspect('equal');ax.set(xlabel='x (mm)',ylabel='y (mm)')
  frames=[]
  for cycle in cycles:
   assert any(int(r['executed_cycle_index'])==cycle and r['equilibrium_accepted']=='1' for r in records)
   d,x,t=load_vtp(saved[cycle]);uv=np.column_stack([point_array(d,n) for n in ('material_u','material_v')]);y,_,_=proper_kabsch(np.column_stack([uv,np.zeros(len(uv))]),x);frames.append((cycle,uv,y,t))
  for ax,(cycle,uv,y,t) in zip(axes[1:],frames):
   im=ax.tripcolor(uv[:,0]*1000,uv[:,1]*1000,t,y[:,2]*1000,shading='gouraud',cmap='RdBu_r',vmin=-115,vmax=115);ax.set_aspect('equal');ax.set(xlabel='x (mm)',ylabel='y (mm)')
   if suffix=='strip_triplets':
    completed=cycle//3;label=f'A strips 1–{completed}' if completed<=8 else f'All A + B strips 1–{completed-8}'
   else:label='All A+B applications complete'
   ax.set_title(f'{label}\nAccepted solve; span {np.ptp(y[:,2])*1000:.1f} mm',fontsize=10)
  for ax in axes[1+len(frames):]:ax.axis('off')
  if suffix.startswith('all_at_once'):
   axes[2].text(0,.9,'Loading history (no solves):\n\nA → B → A → B → A → B\n\nGeometry stays at the initial flat state\nuntil the single final optimization.\n\nNo saved within-solver geometry frames;\nno interpolated intermediate shapes.',va='top',fontsize=12)
  else:axes[-1].text(0,.9,'A₁ ×3 → solve; A₂ ×3 → solve; …\nthen B₁ ×3 → solve; …\n\n30 solves total; selected accepted\nintermediate configurations shown.',va='top',fontsize=12)
  fig.colorbar(im,ax=axes,label='Rigid-aligned height (mm); same scale in both figures',shrink=.6)
  fig.suptitle(('Strip-triplets: incremental loading and equilibration' if suffix=='strip_triplets' else 'All-at-once: accumulate three passes, then one solve')+'\n'+f"gtop=0.0015/hit; gbot=0; ortho=0; no hardening | res=0.015; HLBFGS; tol={opts['-tol']}; acceptance={opts['-equilibrium_grad_tol']}; max_iter={opts['-max_iter']}",fontsize=13)
  for ext in ('png','pdf'):fig.savefig(out/f'{suffix}_progression.{ext}',dpi=160)
  plt.close(fig)
  print(out/f'{suffix}_progression.png')


if __name__=='__main__':main()
