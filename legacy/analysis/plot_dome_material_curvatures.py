"""Fixed material-direction normal curvatures from saved local principal fits."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def material_curvatures(pos, uv, faces, k, directions):
    # Tangents of the piecewise-linear deformation map, not ambient x/y axes.
    U=np.stack((uv[faces[:,1]]-uv[faces[:,0]],uv[faces[:,2]]-uv[faces[:,0]]),axis=2)
    P=np.stack((pos[faces[:,1]]-pos[faces[:,0]],pos[faces[:,2]]-pos[faces[:,0]]),axis=2)
    J=P@np.linalg.inv(U)
    # Project material tangents into the same fitted tangent plane as the saved
    # principal axes. Euler's normal-curvature formula then applies.
    coefficients=directions@J
    norm2=np.sum(coefficients**2,axis=1)
    result=np.sum(k[:,:,None]*coefficients**2,axis=1)/norm2
    result[norm2<=1e-24]=np.nan
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fields',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.fields) as a:
        pos,uv,faces,k,directions,valid=[a[n] for n in ('positions','uv','faces','geometric','directions','valid')]
    normal=material_curvatures(pos,uv,faces,k,directions)
    normal[~valid]=np.nan
    gaussian=np.prod(k,axis=1);gaussian[~valid]=np.nan
    limit=np.nanmax(abs(normal));klimit=np.nanmax(abs(gaussian))
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,3,figsize=(11,5.5),layout='constrained')
        for ax,field,title,label,lim in zip(axes,[normal[:,0],normal[:,1],gaussian],
            ['(a) Along material u','(b) Along material v','(c) Gaussian curvature'],
            ['Normal curvature (m$^{-1}$)']*2+['Gaussian curvature (m$^{-2}$)'],[limit,limit,klimit]):
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#888888')
            artist=ax.tripcolor(uv[:,0]*1000,uv[:,1]*1000,faces,facecolors=np.ma.masked_invalid(field),
                cmap=cmap,vmin=-lim,vmax=lim,rasterized=True)
            ax.add_patch(Rectangle((-80,-120),160,240,fill=False,ls='--',lw=.9,color='black'))
            ax.set(aspect='equal',xlabel='Material u (mm)',ylabel='Material v (mm)',title=title)
            ax.set_xticks([-100,0,100]);ax.set_yticks([-150,-100,-50,0,50,100,150])
            if ax is axes[1]:fig.colorbar(artist,ax=axes[:2],label=label,shrink=.8)
            elif ax is axes[2]:fig.colorbar(artist,ax=ax,label=label,shrink=.8)
        fig.savefig(args.output/'material_curvatures.pdf')
        fig.savefig(args.output/'material_curvatures.png',dpi=220)
        plt.close(fig)
    np.savez_compressed(args.output/'material_curvatures.npz',normal_curvatures=normal,gaussian=gaussian)
    (args.output/'material_curvatures.caption.md').write_text(
        'Fixed material-direction curvature diagnostic for stage1 attempt028, selected as best of that stage (54.88% coverage), not a verified dome. '
        '(a,b) Signed normal curvature along the deformed images of the reference u and v coordinate directions. '
        'These are sheet-wide material directions, not locally ranked principal axes or fixed ambient spatial axes. '
        'Deformation tangents from each triangle are projected into the saved local fitted tangent plane; '
        'Euler formula k(t)=sum_i k_i (e_i dot t)^2 / sum_i (e_i dot t)^2 is evaluated using both principal axes. '
        'The two maps share a symmetric colour scale; red/blue mean opposite bending signs relative to the mesh normal. '
        '(c) Gaussian curvature from the saved principal-curvature product, with its own units and scale; positive is elliptic, negative saddle. '
        'Generally K is not k_u*k_v: the material directions need not be orthogonal or principal, and twist matters. '
        'All faces shown; invalid fits gray. Dashed box is the unchanged central scoring region. '
        'Saved15mm fit reused, no new fit, smoothing, solver run or acceptance-criterion change. No mesh-convergence or uncertainty claim.\n')
    (args.output/'provenance.json').write_text(json.dumps(dict(source=str(args.fields.resolve()),
        source_sha256=hashlib.sha256(args.fields.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':main()
