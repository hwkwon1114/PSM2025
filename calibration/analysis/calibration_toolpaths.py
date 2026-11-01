"""Researcher-corrected x-z plan view; archived solve outputs remain unchanged."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from calibration.ewcal.common import digest
OUT=BASE/'calibration/paper_figures/manuscript_calibration_xzy'
CASES=[('run6','run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00','Calibration'),('run9','run/run9_transfer_expanded/evaluations/attempt_0209/case_00','Validation'),('run10','run/run10_transfer/evaluations/attempt_9902/case_00','Validation')]

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    pins={};records=[];arrays={}
    with plt.rc_context({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(7.2,4.8))
        fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.85,wspace=.32)
        for i,(name,case,role) in enumerate(CASES):
            seqpath=BASE/case/'sequence.json';csvpath=BASE/f'calibration/data/raw/{name}_zigzag_path.csv'
            for p in (seqpath,csvpath): pins[str(p.relative_to(BASE))]=digest(p)
            seq=json.loads(seqpath.read_text());tool=seq['toolpaths'][0];op=tool['operation']
            assert op.get('rotation_deg',0)==0 and op['shift_frame']=='material'
            with csvpath.open() as f:
                p=np.array([[float(r['x_mm']),float(r['y_mm'])] for r in csv.DictReader(f)])
            # New physical convention: x width, z longitudinal; start right at z=0.
            # This replaces the archived longitudinal offsets, not the archived solves.
            p[:,0]=-(p[:,0]-p[:,0].mean())
            p[:,1]-=p[0,1]
            assert np.isfinite(p).all() and (np.abs(p[:,0])<=100).all()
            assert np.isclose(p[:,1].min(),0) and np.isclose(p[:,1].max(),op['lv_mm'])
            assert p[0,0]>0 and np.isclose(p[0,1],0) and p[1,1]>p[0,1]
            arrays[name+'_path_mm']=p
            ax=axes[i]
            ax.add_patch(Rectangle((-100,-85),200,300,facecolor='#f5f5f5',edgecolor='#555555',lw=.8,zorder=0))
            ax.add_patch(Rectangle((-100,-85),200,85,facecolor='#e3e3e3',edgecolor='none',zorder=1))
            ax.axhline(0,color='#888888',lw=.6,ls='--',zorder=1)
            ax.text(0,-42.5,'85 mm clearance',ha='center',va='center',fontsize=8,color='#444444')
            ax.plot(p[:,0],p[:,1],color='#0072B2',lw=.55,solid_capstyle='round',zorder=2)
            ax.plot(*p[0],marker='o',color='#D55E00',ms=4,zorder=4)
            # Direction arrow on first nonzero segment; no displaced schematic path.
            j=np.flatnonzero(np.linalg.norm(p-p[0],axis=1)>1e-8)[0]
            v=p[j]-p[0]
            ax.annotate('',xy=p[0]+.65*v,xytext=p[0]+.35*v,arrowprops={'arrowstyle':'->','color':'#D55E00','lw':1.2},zorder=5)
            ax.set(xlim=(-105,105),ylim=(-90,220),aspect='equal',xticks=[-100,0,100],yticks=[-85,0,100,215],xlabel='$x$ (mm)')
            if i==0:ax.set_ylabel('$z$ (mm)')
            else:ax.tick_params(labelleft=False)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.tick_params(length=3)
            ax.set_title(f'({chr(97+i)}) Run {name[3:]} · {role}',pad=10)
            ax.text(.5,-.19,f"$L={op['lv_mm']:g}$ mm · $\\alpha={op['alpha_deg']:g}^\\circ$\n{op['n_strips']} strips · {tool['repeat']} cycles",ha='center',va='top',transform=ax.transAxes,fontsize=8)
            records.append({'run':name,'role':role,'archived_operation_not_updated':op,'plotted_coordinates':'x,z (mm); y=0 reference sheet','placement':'x=-(raw_x-mean(raw_x)); z=raw_y-raw_y_start','cycles':tool['repeat'],'csv_points':len(p),'path_bounds_mm':[p.min(axis=0).tolist(),p.max(axis=0).tolist()]})
        fig.legend(handles=[Line2D([],[],color='#0072B2',lw=.8,label='Toolpath centreline'),Line2D([],[],color='#D55E00',marker='o',ms=4,lw=1,label='Start / travel direction')],loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False,fontsize=8)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        texts=list(fig.legends[0].get_texts())
        for ax in axes:texts.extend([ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts])
        for t in texts:
            if not t.get_text():continue
            b=t.get_window_extent(renderer)
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
        for ext in ('pdf','png'):fig.savefig(OUT/f'run6_run9_run10_toolpaths.{ext}',dpi=400)
        plt.close(fig)
    assert all(digest(BASE/p)==h for p,h in pins.items())
    np.savez(OUT/'run6_run9_run10_toolpaths.npz',**arrays)
    (OUT/'run6_run9_run10_toolpaths.provenance.json').write_text(json.dumps({'inputs_sha256':pins,'script_sha256':digest(Path(__file__)),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'cases':records},indent=2)+'\n')
    (OUT/'run6_run9_run10_toolpaths.caption.md').write_text('Zigzag toolpaths on the undeformed 200 × 300 mm sheet. (a) Run 6 calibration; (b,c) Runs 9 and 10 designated validation cases, using the same deformation parameters. Blue lines show path centrelines; orange markers/arrows identify the start and initial travel direction. The centreline is shown once; cycle counts report prescribed repetitions, not overplotted trajectories. Line thickness is graphical and does not represent the model’s 2 mm effective track width. All panels use equal physical aspect and identical limits. Researcher-corrected convention: x is width (−100 to 100 mm), z is length (−85 to 215 mm), and y is height. The reference sheet is at y=0. Archived CSV x coordinates are centered and reversed so paths start on the right; longitudinal coordinates begin at z=0 and initially increase. The shaded band −85≤z≤0 is the 85 mm toolpath clearance, not a specified clamp footprint. Path ranges are 0–130, 0–187.5 and 0–175 mm. This replaces the earlier negative-offset placement for Runs 9/10; it does NOT depict the unchanged archived simulation placement. No corrected simulations are represented. L and alpha are the archived zigzag length and angle settings. Sheet outline is the undeformed reference rectangle; neither deformation nor gripper geometry is depicted. No fitting, simulation or optimization was performed. Validation is the researcher-designated role, not a claim of independently held-out model development.\n')
    print(OUT/'run6_run9_run10_toolpaths.pdf')

if __name__=='__main__':main()
