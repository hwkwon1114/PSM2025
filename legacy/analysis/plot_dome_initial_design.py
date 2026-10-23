"""All24 frozen stage2 initial toolpaths, in scheduled order; not outcomes."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
from dome_bo_repeat_domain import sequence
base=json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
design=json.loads((ROOT/'run/dome/checks/stage2_lhs.json').read_text())
out=ROOT/'run/dome/checks/stage2_initial_toolpaths.png'
if out.exists(): raise FileExistsError(out)
fig,axes=plt.subplots(4,6,figsize=(16,15),layout='constrained')
for index,(ax,params) in enumerate(zip(axes.flat,design['parameters']),1):
    for path,color,style in zip(sequence(base,params)['toolpaths'],['#0072B2','#D55E00','#009E73'],['-','--','-.']):
        op=path['operation']; length=op['lv_mm']; n=op['n_strips']; pitch=length*np.tan(np.deg2rad(op['alpha_deg'])); left=-(n-2)*pitch/2
        points=[[left,-length/2],[left,length/2]]
        points.extend([[left+j*pitch,(-1 if j%2 else 1)*length/2] for j in range(1,n-1)])
        points.append([points[-1][0],-points[-1][1]])
        a=np.deg2rad(op['rotation_deg']); rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        points=np.array(points)@rotation.T+np.array(op['center_uv_mm'])+np.array(op['shift_uv_mm'])
        polygons=[]
        for start,end in zip(points[:-1],points[1:]):
            d=end-start; normal=np.array([-d[1],d[0]])/np.linalg.norm(d)*op['width_mm']/2
            polygons.append([start+normal,end+normal,end-normal,start-normal])
        ax.add_collection(PolyCollection(polygons,facecolors=color,alpha=.16,edgecolors='none'))
        ax.plot(*points.T,color=color,linestyle=style,lw=.65)
    ax.add_patch(Rectangle((-100,-150),200,300,fill=False,color='black',lw=.7))
    ax.set(xlim=(-105,105),ylim=(-155,155),aspect='equal',xticks=[-100,0,100],yticks=[-150,0,150],
           title=f'{index:02d} | repeats {params[6]} / {params[7]} / {params[7]}')
    ax.tick_params(labelsize=7)
    ax.title.set_fontsize(9)
    if index>18: ax.set_xlabel('u (mm)',fontsize=8)
    if (index-1)%6==0: ax.set_ylabel('v (mm)',fontsize=8)
fig.savefig(out,dpi=160)
plt.close(fig)
out.with_suffix('.caption.md').write_text('All24 frozen constrained LHS-derived initial designs, in scheduled attempt order; no outcome selection or omissions. Each panel shows the200×300mm sheet with matching aspect and axes. Blue solid: middle A; orange dashed: upper B; green dash-dot: lower C. Shading indicates rectangular treatment widths, not strain magnitude; intersections are allowed. Panel repeats are A/B/C whole-path applications; traces are drawn once, not repeatedly overplotted. These selected feasible points are not an exact Latin hypercube. Source run/dome/checks/stage2_lhs.json, seed20261019. No simulation result is displayed.\n')
