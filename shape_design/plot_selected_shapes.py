"""Saved k=±4 pilot shapes for illustrated toolpath workbooks; no solver."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_design.analysis import height

OUT = ROOT / 'run/shape_design/pilot_toolpath_csv/selected_shape_images'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    provenance = {}
    for k in (-4, 4):
        base = ROOT / 'run/shape_design/pilot' / f'flip_k{k:+d}'
        result = json.loads((base/'result.json').read_text())
        if result['status'] != 'accepted' or result['hit_check']['mismatched_faces'] != 0:
            raise ValueError(base)
        file = base/'shape.npz'
        with np.load(file) as data:
            ref, pos, faces = [data[name].copy() for name in ('reference_mm','aligned_mm','faces')]
        h = height(pos)
        lim = float(np.ceil(np.max(np.abs(h))*10)/10)
        norm = TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
        fig = plt.figure(figsize=(11, 5))
        fig.subplots_adjust(left=.07,right=.88,bottom=.15,top=.87,wspace=.32)
        ax = fig.add_subplot(121,projection='3d')
        ax.plot_trisurf(pos[:,0],pos[:,1]+65,h,triangles=faces,cmap='RdBu_r',norm=norm,
                        linewidth=0,antialiased=False,shade=False,rasterized=True)
        ax.set(xlim=(-110,110),ylim=(3,127),zlim=(-lim,lim),xlabel='Width x (mm)',
               ylabel='Length z (mm)')
        ax.set_zticks([])  # Height is quantified by the common colorbar; avoid compressed 3-D ticks.
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([20, 65, 110])
        ax.set_box_aspect((220,124,2*lim*10)); ax.view_init(elev=25,azim=-65)
        ax.set_title('3-D view · height exaggerated 10×')
        ax = fig.add_subplot(122)
        tri = mtri.Triangulation(ref[:,0],ref[:,1]+65,faces)
        im = ax.tripcolor(tri,h,shading='gouraud',cmap='RdBu_r',norm=norm,rasterized=True)
        ax.set(xlim=(-110,110),ylim=(3,127),xlabel='Width x (mm)',ylabel='Length z (mm)',title='Mean-centered height')
        ax.set_aspect('equal')
        fig.colorbar(im,ax=ax,label='Height y (mm)',fraction=.045,pad=.03)
        fig.suptitle(f'k = {k:+d} · saved 2 mm pilot simulation',fontsize=13)
        fig.canvas.draw()
        for a in fig.axes:
            for text in (a.title,a.xaxis.label,a.yaxis.label,*([a.zaxis.label] if hasattr(a,'zaxis') else [])):
                if text.get_visible() and text.get_text():
                    b=text.get_window_extent(fig.canvas.get_renderer())
                    if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                        raise ValueError(f'text outside page: {text.get_text()}')
        fig.savefig(OUT/f'flip_k{k:+d}_shape.png',dpi=180)
        plt.close(fig)
        provenance[str(k)]={'shape':str(file.relative_to(ROOT)), 'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                            'height_range_mm':float(np.ptp(h)),'accepted_status':result['status']}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__ == '__main__':
    main()
