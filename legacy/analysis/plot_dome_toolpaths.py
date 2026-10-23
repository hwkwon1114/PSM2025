"""Plot actual zigzag centerlines and rectangular treatment bands in material mm."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'run/dome/bo_stage1/attempt_028/sequence.json'
config=json.loads(source.read_text())
out=ROOT/'run/dome/bo_stage1/attempt_028/analysis/bo_best_toolpaths.png'
if out.exists(): raise FileExistsError(out)
fig,ax=plt.subplots(figsize=(6,7),layout='constrained')
for path,color,style,name in zip(config['toolpaths'],['#0072B2','#D55E00','#009E73'],['-','--','-.'],['A: middle','B: upper side','C: lower side']):
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
    ax.plot(*points.T,color=color,linestyle=style,lw=1,label=f"{name} — {path['repeat']} applications")
    ax.plot(*points[0],marker='o',color=color,markersize=4)
ax.add_patch(Rectangle((-100,-150),200,300,fill=False,color='black',lw=1))
ax.set(xlim=(-110,110),ylim=(-160,160),xlabel='Material u (mm)',ylabel='Material v (mm)',aspect='equal')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),frameon=False,fontsize=10)
fig.savefig(out,dpi=180)
plt.close(fig)
out.with_suffix('.caption.md').write_text('Best-observed BO design (attempt028), selected by geometric coverage, not a verified dome. Exact centerline construction follows buildZigZagStrips in src/libshell/ZigZagGrowth.hpp; angles relative to local vertical before rotation. Shading shows geometric rectangular treatment bands, not mesh hit assignment or accumulated strain. Dots mark starts. Sheet200×300mm. All48 BO candidates used A×2, B×3, C×3 (eight whole-path applications); all strains precede one free equilibrium. Input: '+str(source)+'\n')
