#!/usr/bin/env python3
"""Generate an unsubmitted two-half orthogonal zigzag proposal and preview."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import numpy as np
from run_forward_model_diagnostics import sequence
from visualize_solver_characteristics import place_strips, capsule_polygon


def main():
    out = Path(__file__).resolve().parents[1] / 'run/forward_model_diagnostics/split_orthogonal_proposal'
    out.mkdir(exist_ok=True)
    paths = []
    for name, length, span, count, angle, center in [
        ('A_left_vertical',280.,100.,8,0.,[-63.5,0.]),
        ('B_right_horizontal',100.,280.,22,90.,[63.5,0.]),
    ]:
        paths.append(dict(id=name,enabled=True,repeat=1,operation=dict(
            type='zigzag',lv_mm=length,alpha_deg=math.degrees(math.atan(span/(count-2)/length)),
            n_strips=count,width_mm=12.,center_uv_mm=center,shift_uv_mm=[0.,0.],
            shift_frame='material',rotation_deg=angle,gtop=.0012,gbot=0.,ortho=0.)))
    proposal = sequence(paths)
    (out/'sequence.json').write_text(json.dumps(proposal,indent=2)+'\n')
    bbox=(-.127,.127,-.1524,.1524)
    fig, ax = plt.subplots(figsize=(9,10),layout='constrained')
    ax.add_patch(Rectangle((-127,-152.4),254,304.8,facecolor='#fafafa',edgecolor='black',linewidth=1.8))
    ax.axvline(0,color='0.4',linestyle='--',linewidth=1)
    checks=[]
    for i,(path,color) in enumerate(zip(paths,['#2266b4','#dc7620'])):
        placement=place_strips(path['operation'],bbox)
        endpoints=np.array([[s['a_uv_m'],s['b_uv_m']] for s in placement['strips']])*1000
        lower=endpoints.min(axis=(0,1))-6
        upper=endpoints.max(axis=(0,1))+6
        assert np.all(lower >= [-127,-152.4]) and np.all(upper <= [127,152.4])
        assert upper[0] < 0 if i==0 else lower[0] > 0
        length=float(np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1).sum())
        checks.append(dict(path=path['id'],capsule_bbox_mm=[lower.tolist(),upper.tolist()],centerline_length_mm=length))
        for strip in placement['strips']:
            a,b=strip['a_uv_m'],strip['b_uv_m']
            ax.add_patch(Polygon(capsule_polygon(a,b,.006)*1000,facecolor=color,edgecolor='none',alpha=.18))
            ax.plot([a[0]*1000,b[0]*1000],[a[1]*1000,b[1]*1000],color=color,linewidth=1.2)
            mid=(a+b)/2*1000
            direction=(b-a)/np.linalg.norm(b-a)
            ax.annotate('',xy=mid+4*direction,xytext=mid-4*direction,arrowprops=dict(arrowstyle='-|>',color=color,lw=1.2))
        start,end=endpoints[0,0],endpoints[-1,1]
        ax.scatter(*start,s=60,c='#239b56',edgecolors='black',zorder=5)
        ax.scatter(*end,s=45,c=color,marker='s',edgecolors='black',zorder=5)
        ax.text((-63.5 if i==0 else 63.5),160,('A: left / vertical' if i==0 else 'B: right / horizontal'),ha='center',color=color,weight='bold',fontsize=12)
    gap=checks[1]['capsule_bbox_mm'][0][0]-checks[0]['capsule_bbox_mm'][1][0]
    assert gap > 0
    report=dict(status='proposal only; no solver submitted',sheet_mm=[254,304.8],strip_width_mm=12,central_unworked_gap_mm=gap,paths=checks,loading='top-only isotropic 0.0012 per hit; no hardening; A then equilibrate, B then equilibrate',caution='Different strip counts accommodate the half-sheet aspect ratio. Equal per-hit strain does not guarantee equal area-integrated loading.')
    (out/'layout_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    ax.set(xlim=(-136,136),ylim=(-163,170),xlabel='material u (mm)',ylabel='material v (mm)')
    ax.set_aspect('equal')
    ax.legend(handles=[Line2D([],[],color='#2266b4',label='A: 8 long strips'),Line2D([],[],color='#dc7620',label='B: 22 short strips'),Line2D([],[],marker='o',color='none',markerfacecolor='#239b56',markeredgecolor='black',label='Start'),Line2D([],[],marker='s',color='none',markerfacecolor='0.5',label='End')],loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2,frameon=False)
    fig.suptitle('PROPOSED: orthogonal zigzags on separate sheet halves\nShading = 12 mm footprint; arrows = traversal direction\nA → equilibrate → B → equilibrate. Gap: 15 mm. Not yet simulated.',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(out/f'toolpath_preview.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print(out/'toolpath_preview.png')


if __name__=='__main__':
    main()
