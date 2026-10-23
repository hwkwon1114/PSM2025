"""Plot saved principal directions in material coordinates; no solver or refitting."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fields',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.fields) as a:
        uv,pos,faces,k,directions,valid=[a[n] for n in ('uv','positions','faces','geometric','directions','valid')]
    centers=uv[faces].mean(axis=1)
    order=np.argsort(abs(k),axis=1)
    values=np.take_along_axis(k,order,axis=1)
    physical=np.take_along_axis(directions,order[:,:,None],axis=1)
    # Pull spatial tangent directions back through each triangle's deformation map.
    U=np.stack((uv[faces[:,1]]-uv[faces[:,0]],uv[faces[:,2]]-uv[faces[:,0]]),axis=2)
    P=np.stack((pos[faces[:,1]]-pos[faces[:,0]],pos[faces[:,2]]-pos[faces[:,0]]),axis=2)
    J=P@np.linalg.inv(U)
    material=np.linalg.pinv(J)@physical.transpose(0,2,1)
    material=material.transpose(0,2,1)
    norms=np.linalg.norm(material,axis=2)
    good=valid & np.isfinite(material).all(axis=(1,2)) & (norms>1e-12).all(axis=1)
    material=material/np.maximum(norms[:,:,None],1e-300)
    # Display-only omission: near-isotropy or weak/strong absolute-magnitude ties.
    gap=abs(values[:,1])-abs(values[:,0])
    good &= gap >= np.maximum(.05,.1*np.max(abs(k),axis=1))
    grid=np.array([(u,v) for v in np.arange(-.14,.141,.02) for u in np.arange(-.09,.091,.02)])
    _,ids=cKDTree(centers).query(grid)
    ids=np.unique(ids)
    selected=ids[good[ids]]
    limit=float(np.nanmax(abs(k)))
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,2,figsize=(8,5.7),layout='constrained')
        for j,ax in enumerate(axes):
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#888888')
            artist=ax.tripcolor(uv[:,0]*1000,uv[:,1]*1000,faces,
                facecolors=np.ma.masked_invalid(values[:,j]),cmap=cmap,vmin=-limit,vmax=limit,rasterized=True)
            c=centers[selected]*1000;delta=material[selected,j]*5
            ax.add_collection(LineCollection(np.stack((c-delta,c+delta),axis=1),colors='white',linewidths=2.3))
            ax.add_collection(LineCollection(np.stack((c-delta,c+delta),axis=1),colors='black',linewidths=.9))
            ax.add_patch(Rectangle((-80,-120),160,240,fill=False,ls='--',lw=1,color='black'))
            ax.set(xlim=(-102,102),ylim=(-152,152),aspect='equal',xlabel='Material u (mm)',ylabel='Material v (mm)',
                   title=['(a) Weaker principal curvature','(b) Stronger principal curvature'][j])
            ax.set_xticks([-100,0,100]);ax.set_yticks([-150,-100,-50,0,50,100,150])
        fig.colorbar(artist,ax=axes,label='Signed principal curvature (m$^{-1}$)',shrink=.85)
        fig.savefig(args.output/'principal_directions.pdf')
        fig.savefig(args.output/'principal_directions.png',dpi=220)
        plt.close(fig)
    report=dict(source=str(args.fields.resolve()),sha256=hashlib.sha256(args.fields.read_bytes()).hexdigest(),
                sampled_faces=len(ids),displayed_direction_pairs=len(selected),omitted_pairs=len(ids)-len(selected),
                display_gap_threshold='abs(k_strong)-abs(k_weak) >= max(0.05 m^-1, 0.1*max(abs(k)))',
                grid_spacing_mm=20,segment_length_mm=10)
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'principal_directions.caption.md').write_text(
        'Principal curvature directions of the retained stage1 attempt028 candidate (54.880208% coverage; selected as best stage1 score, not representative or verified dome). '
        'Colours show signed weaker/stronger principal curvatures with one shared full-range scale. '
        'Black unoriented segments show corresponding tangent axes pulled back to material coordinates using each deformed triangle Jacobian; '
        'they are not toolpath directions, and need not appear orthogonal after this coordinate mapping. '
        'Segments have constant10mm display length, independent of curvature magnitude; sampled on a fixed20mm material grid. '
        'Directions with absolute-curvature gap below max(0.05 m^-1,10% of larger magnitude) are omitted as display-ambiguous; '
        'all face curvature colours remain, and neither the original score nor its criteria are changed. '
        'Dash box: fixed central scoring region. Red/blue is curvature sign relative to the mesh normal, not success/failure. '
        'Saved15mm local geometry fits reused without refitting. No mesh convergence or direction uncertainty certificate is implied.\n')


if __name__=='__main__': main()
