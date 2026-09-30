#!/usr/bin/env python3
"""Plot actual saved three-pass toolpaths and their optimization schedules."""
import copy
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
from matplotlib.lines import Line2D
import numpy as np
from analyze_split_orthogonal import ROOT
from visualize_solver_characteristics import place_strips,capsule_polygon


def main():
    out=ROOT/'three_pass_comparison';out.mkdir(exist_ok=True)
    suffixes=['strip_triplets','pair_triplets','each_path','each_AB','all_at_once']
    signatures=[]
    for suffix in suffixes:
        directory=ROOT/'current'/('split_three_'+suffix)
        seq=json.loads((directory/'sequence.json').read_text())
        assert seq['hardening']['model']=='none'
        counter=Counter()
        for path in seq['toolpaths']:
            op=copy.deepcopy(path['operation']);active=op.pop('active_strips')
            assert op['gtop']==.0015 and op['gbot']==0 and op['ortho']==0
            for strip in active: counter[(json.dumps(op,sort_keys=True),strip)]+=path['repeat']
        signatures.append(counter)
    assert all(c==signatures[0] for c in signatures) and set(signatures[0].values())=={3}
    # Recover full paths from unique operations, not the old proposal geometry.
    operations=[json.loads(s) for s in sorted({k[0] for k in signatures[0]})]
    operations.sort(key=lambda op:op['center_uv_mm'][0])
    manifest=json.loads((ROOT/'current/split_three_each_path/manifest.json').read_text())
    lx,ly=manifest['plate']['lx'],manifest['plate']['ly']
    fig,(ax,text)=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1,1.05]},layout='constrained')
    ax.add_patch(Rectangle((-lx*1000,-ly*1000),2*lx*1000,2*ly*1000,facecolor='#fafafa',edgecolor='black',lw=1.5))
    ax.axvline(0,color='0.5',ls='--',lw=.8)
    for op,color,name in zip(operations,['#2266b4','#dc7620'],['A: vertical / left','B: horizontal / right']):
        placement=place_strips(op,(-lx,lx,-ly,ly))
        for strip in placement['strips']:
            a,b=strip['a_uv_m'],strip['b_uv_m']
            ax.add_patch(Polygon(capsule_polygon(a,b,placement['half_width_m'])*1000,facecolor=color,edgecolor='none',alpha=.18))
            ax.plot([a[0]*1000,b[0]*1000],[a[1]*1000,b[1]*1000],color=color,lw=1)
            mid=(a+b)*500;direction=(b-a)/np.linalg.norm(b-a)
            ax.annotate('',xy=mid+4*direction,xytext=mid-4*direction,arrowprops=dict(arrowstyle='-|>',color=color,lw=1))
        ax.scatter(*(placement['strips'][0]['a_uv_m']*1000),c='#239b56',edgecolors='black',s=40,zorder=5)
        ax.scatter(*(placement['strips'][-1]['b_uv_m']*1000),c=color,edgecolors='black',marker='s',s=35,zorder=5)
        ax.text(op['center_uv_mm'][0],160,name,ha='center',color=color,fontsize=10,weight='bold')
    ax.set(xlim=(-137,137),ylim=(-162,172),xlabel='x (mm)',ylabel='y (mm)');ax.set_aspect('equal')
    ax.legend(handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#239b56',label='Start'),Line2D([],[],marker='s',color='none',markerfacecolor='0.4',label='End')],loc='lower center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
    text.axis('off')
    rows=[('Same spatial footprint in all five schedules','Each of the 30 strips is applied three times.\nRepeated traversals lie on the same lines; no offsets.\nShading: 12 mm growth footprint, not a hit-count map.\nSheet: 254 × 304.8 mm; central unworked gap: 15 mm.'),('1. Individual strip ×3, then optimize — 30 solves','A₁ ×3 → O; A₂ ×3 → O; …; finish A, then B.'),('2. Pair of strips ×3, then optimize — 15 solves','(A₁+A₂) ×3 → O; (A₃+A₄) ×3 → O; …; then B.'),('3. Optimize after each whole path — 6 solves','[A → O → B → O] ×3'),('4. Optimize after each A+B — 3 solves','[A → B → O] ×3'),('5. All growth before one optimization — 1 solve','[A → B] ×3 → O\nOriginal solve hit its iteration cap; retry submitted.')]
    positions=[.96,.73,.57,.41,.28,.15]
    for (title,body),y in zip(rows,positions):
        text.text(0,y,title,transform=text.transAxes,fontsize=11,weight='bold',va='top')
        text.text(0,y-.045,body,transform=text.transAxes,fontsize=10,va='top',linespacing=1.6)
    fig.suptitle('Toolpath used in the three-pass comparison\nGrowth per hit: top 0.0015, bottom 0; isotropic; no hardening. O = optimize.',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(out/f'toolpath_and_schedules.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Verified all five saved sequences: identical geometry and three applications per strip.')
    print(out/'toolpath_and_schedules.png')


if __name__=='__main__':main()
