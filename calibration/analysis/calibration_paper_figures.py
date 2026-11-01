"""Saved-data manuscript figures; no fitting, simulation or new alignment."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, LightSource, to_rgb
from matplotlib.lines import Line2D
import numpy as np
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from calibration.analysis.run6_run9_six_panel import load_evaluation_data, heights_at_xy
from calibration.ewcal.common import digest
OUT = BASE/'calibration/paper_figures/manuscript_calibration_xzy'
CASES = [('run6','run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00'),
         ('run9','run/run9_transfer_expanded/evaluations/attempt_0209/case_00'),
         ('run10','run/run10_transfer/evaluations/attempt_9902/case_00')]
LABELS = ['Run 6 · Calibration','Run 9 · Transfer','Run 10 · Transfer']
ORANGE, BLUE = '#D55E00','#0072B2'


def save(fig,stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Catch text escaping the physical page; manual review checks inter-panel collisions.
    texts=list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label])
        texts.extend(ax.texts)
        if ax.get_legend() is not None:
            texts.extend(ax.get_legend().get_texts())
    for text in texts:
        if text.get_visible() and text.get_text():
            bb=text.get_window_extent(renderer)
            if not np.isfinite(bb.extents).all():
                continue  # Non-rendered out-of-range colorbar ticks.
            assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1, (stem,text.get_text(),bb)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=350)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pins={}
    rows=[load_evaluation_data(name,BASE/path,pins) for name,path in CASES]
    assert all(r['parameters']==rows[0]['parameters'] for r in rows)
    # Helper arrays are packed (x,z,y): planar coordinates first, height last.
    for r in rows:
        for key in ('scan','prediction'):
            r[key] = r[key] * np.array([-1.,-1.,1.]) + np.array([0.,65.,0.])
    records=[]
    for r in rows:
        records.append(r['statistics'])
        np.savez(OUT/f"{r['name']}_fields.npz",scan=r['scan'],prediction=r['prediction'],faces=r['faces'],dy=r['dz'],support=r['support'],coordinate_order=np.array(['x','z','y']))
    limit=np.ceil(max(np.max(np.abs(r['dz'][r['support']])) for r in rows)*2)/2
    with plt.rc_context({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig=plt.figure(figsize=(7.2,5.8),layout='constrained')
        gs=fig.add_gridspec(2,3,height_ratios=[2.2,1])
        mapaxes=[]
        for i,r in enumerate(rows):
            ax=fig.add_subplot(gs[0,i]);mapaxes.append(ax)
            p=r['scan'];s=r['support']
            ax.set_facecolor('#eeeeee')
            artist=ax.scatter(p[s,0],p[s,1],c=r['dz'][s],s=2,marker='s',linewidths=0,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),rasterized=True)
            if (~s).any(): ax.scatter(p[~s,0],p[~s,1],s=3,color='black',marker='x',linewidths=.4,rasterized=True)
            ax.set(xlim=(-103,103),ylim=(-88,218),aspect='equal',xlabel='$x$ (mm)',xticks=[-100,0,100],yticks=[-85,0,100,215])
            ax.set_title(f'({chr(97+i)}) {LABELS[i]}',loc='left')
            if i==0: ax.set_ylabel('$z$ (mm)')
        fig.colorbar(artist,ax=mapaxes,shrink=.8,pad=.025,label=r'$\Delta y$ (mm)')
        ax=fig.add_subplot(gs[1,:])
        for r,col,ls in zip(rows,[BLUE,ORANGE,'#333333'],['-','--',':']):
            err=np.sort(np.abs(r['dz'][r['support']]))
            ax.step(err,np.arange(1,len(err)+1)/len(err),where='post',color=col,ls=ls,label=r['name'].replace('run','Run '))
        ax.set(xlabel='Absolute height difference (mm)',ylabel='Fraction of points',xlim=(0,limit),ylim=(0,1.02),yticks=[0,.5,1])
        ax.set_title('(d) Full supported-scan distributions',loc='left')
        ax.legend(frameon=False,loc='lower right',ncol=3)
        ax.grid(color='#e5e5e5',lw=.5)
        save(fig,'run6_run9_run10_residuals')
        # Whole-sheet x-z projections; common camera and physical axis spans.
        fig=plt.figure(figsize=(7.2,3.2))
        zlo=np.floor(min(min(r['scan'][:,2].min(),r['prediction'][:,2].min()) for r in rows))
        zhi=np.ceil(max(max(r['scan'][:,2].max(),r['prediction'][:,2].max()) for r in rows))
        for i,r in enumerate(rows):
            ax=fig.add_subplot(1,3,i+1,projection='3d',computed_zorder=False)
            p=r['scan'];tri=mtri.Triangulation(p[:,0],p[:,1]);xy=p[tri.triangles,:2]
            tri.set_mask(np.linalg.norm(xy-np.roll(xy,1,axis=1),axis=2).max(axis=1)>5)
            displayed=p[tri.get_masked_triangles()].copy();displayed[:,:,2]*=12
            normals=np.cross(displayed[:,1]-displayed[:,0],displayed[:,2]-displayed[:,0])
            normals/=np.linalg.norm(normals,axis=1)[:,None]
            intensity=LightSource(azdeg=315,altdeg=35).shade_normals(normals)
            colors=np.array(to_rgb('#E6AA70'))[None,:]*(.45+.55*intensity[:,None])
            surface=ax.plot_trisurf(tri,p[:,2],shade=False,linewidth=0,antialiased=False,rasterized=True,zorder=1)
            surface.set_facecolors(colors)
            grid=[np.column_stack([np.linspace(-98,98,101),np.full(101,y)]) for y in np.linspace(-83,213,13)]
            grid += [np.column_stack([np.full(151,x),np.linspace(-83,213,151)]) for x in np.linspace(-98,98,11)]
            for uv in grid:
                z,_,_=heights_at_xy(r['prediction'],r['faces'],uv)
                ax.plot(uv[:,0],uv[:,1],z,color=BLUE,lw=.6,zorder=2)
            ax.set(xlim=(-105,105),ylim=(-90,220),zlim=(zlo,zhi),xticks=[-100,0,100],yticks=[],zticks=np.arange(zlo,zhi+1,2))
            ax.set_box_aspect((210,310,(zhi-zlo)*12),zoom=1.35)
            ax.set_proj_type('ortho');ax.view_init(elev=0,azim=-90);ax.grid(False)
            ax.set_xlabel('$x$ (mm)',labelpad=5);ax.set_zlabel('$y$ (mm)',labelpad=3)
            ax.tick_params(pad=0,labelsize=7)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.pane.fill=False;axis.pane.set_edgecolor('white')
            ax.set_title(f'({chr(97+i)}) {LABELS[i]}',loc='center',pad=3)
        fig.subplots_adjust(left=.035,right=.985,bottom=.12,top=.77,wspace=.08)
        fig.text(.06,.94,'x–y projection · height scale ×12',fontsize=10)
        fig.legend(handles=[Line2D([],[],color='#C48D56',lw=5,label='Experiment'),Line2D([],[],color=BLUE,lw=1,label='Analytical')],loc='upper right',bbox_to_anchor=(.98,.99),frameon=False,ncol=2,fontsize=8)
        save(fig,'run6_run9_run10_xy_overlay')
        (OUT/'run6_run9_run10_xy_overlay.caption.md').write_text('Whole-sheet x–y projections of Run 6 calibration and Runs 9/10 frozen-parameter transfers, viewed along the longitudinal z direction. Experiment is a directionally shaded piecewise-linear scan triangulation; Analytical is a wire grid interpolated on native model facets. Scan triangles with XZ edges >5 mm are excluded. Both datasets use the same 12× vertical exaggeration and shared physical axis limits; tick values remain in mm. Brightness encodes lighting, not error. All z positions overlap: these are not individual sections. Wires deliberately render over the experimental surface, so wire visibility does not imply positive residual. No new registration, smoothing fit or simulation. Fixed-z sections and residual maps provide quantitative comparisons.\n')
        # Fixed y sections, with no error-based selection or new registration.
        fig,axes=plt.subplots(3,3,figsize=(7.2,6.6),sharex=True,sharey=True,layout='constrained')
        x=np.linspace(-100,100,401);ys=[-35,65,165];section_arrays={'x_mm':x,'z_mm':np.array(ys)}
        zvalues=[]
        for i,r in enumerate(rows):
            p=r['scan'];tri=mtri.Triangulation(p[:,0],p[:,1]);xy=p[tri.triangles,:2]
            tri.set_mask(np.linalg.norm(xy-np.roll(xy,1,axis=1),axis=2).max(axis=1)>5)
            interp=mtri.LinearTriInterpolator(tri,p[:,2])
            for j,y in enumerate(ys):
                obs=interp(x,np.full(len(x),y)).filled(np.nan)
                pred,_,_=heights_at_xy(r['prediction'],r['faces'],np.column_stack([x,np.full(len(x),y)]))
                assert (np.isfinite(obs)&np.isfinite(pred)).any()
                zvalues.extend([obs,pred]);section_arrays[f"{r['name']}_measured_{y}"]=obs;section_arrays[f"{r['name']}_analytical_{y}"]=pred
                ax=axes[i,j]
                ax.plot(x,obs,color=ORANGE,lw=1.2,marker='o',ms=2,markevery=32)
                ax.plot(x,pred,color=BLUE,lw=1.2,ls='--')
                ax.set_title(f'({chr(97+3*i+j)}) $z={y}$ mm',loc='left')
                ax.grid(axis='y',color='#e5e5e5',lw=.5)
                ax.set(xticks=[-100,0,100],xlim=(-100,100))
                if j==0: ax.set_ylabel(LABELS[i]+'\n$y$ (mm)')
                if i==2: ax.set_xlabel('$x$ (mm)')
        lo=np.floor(min(np.nanmin(v) for v in zvalues));hi=np.ceil(max(np.nanmax(v) for v in zvalues))
        axes[0,0].set_ylim(lo,hi)
        axes[0,1].legend(handles=[Line2D([],[],color=ORANGE,marker='o',ms=3,label='Experiment'),Line2D([],[],color=BLUE,ls='--',label='Analytical')],loc='lower center',frameon=False,fontsize=8)
        save(fig,'run6_run9_run10_sections')
        np.savez(OUT/'sections.npz',**section_arrays)
        # Plot retrospective rescored records separately from the current search.
        hp=BASE/'run/run6_expanded_bo_continue20_recovery/evaluations.jsonl';pins[str(hp.relative_to(BASE))]=digest(hp)
        hist=[json.loads(line) for line in hp.read_text().splitlines() if line.strip()]
        assert len({r['attempt_id'] for r in hist})==len(hist)
        hist.sort(key=lambda r:r['attempt_id'])
        assert [r['attempt_id'] for r in hist]==list(range(1,241))
        for r in hist:
            if r['valid']:
                m=r.get('metrics') or r['cases'][0]['metrics']
                assert m['scoring_method']=='material_reference'
                assert np.isclose(r['objective_mm'],m['rms_symmetric_mm'])
        fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained',gridspec_kw={'width_ratios':[1,2]})
        for ax,subset,title in [(axes[0],hist[:80],'(a) Rescored prior records'),(axes[1],hist[80:],'(b) Expanded search and continuation')]:
            valid=[r for r in subset if r['valid']];invalid=[r for r in subset if not r['valid']]
            ax.scatter([r['attempt_id'] for r in valid],[r['objective_mm'] for r in valid],s=10,color='#888888',alpha=.7,label='Valid evaluation')
            ax.scatter([r['attempt_id'] for r in invalid],[.025]*len(invalid),transform=ax.get_xaxis_transform(),marker='x',s=18,color=ORANGE,label='Invalid record',clip_on=False)
            ax.set(yscale='log',xlabel='Archived attempt ID',ylabel='Symmetric RMS (mm)')
            ax.set_title(title,loc='left');ax.grid(axis='y',color='#e5e5e5',lw=.5)
        active=hist[80:];best=[];v=np.inf
        for r in active:
            if r['valid']: v=min(v,r['objective_mm'])
            best.append(v)
        axes[1].step([r['attempt_id'] for r in active],best,where='post',color=BLUE,lw=1.5,label='Best current-search value')
        selected=next(r for r in hist if r['attempt_id']==209)
        assert np.isclose(selected['objective_mm'],min(r['objective_mm'] for r in active if r['valid']))
        axes[1].scatter([209],[selected['objective_mm']],marker='*',s=70,color=BLUE,zorder=4)
        axes[1].annotate('Selected: 1.182 mm',xy=(209,selected['objective_mm']),xytext=(130,2.6),fontsize=8,arrowprops={'arrowstyle':'-','color':BLUE,'lw':.6})
        axes[1].legend(frameon=False,fontsize=7,loc='upper right')
        save(fig,'run6_objective_history')
        with (OUT/'objective_history.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['attempt_id','valid','training_only','objective_mm','failure_class'])
            for r in hist: writer.writerow([r['attempt_id'],r['valid'],r.get('training_only',False),r.get('objective_mm') if r['valid'] else '',r.get('failure_class','')])
    assert all(digest(BASE/p)==h for p,h in pins.items())
    (OUT/'provenance.json').write_text(json.dumps({'input_sha256':pins,'script_sha256':digest(Path(__file__)),'statistics':records,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'residual_limit_mm':limit,'section_z_mm':ys,'coordinate_conversion':'x=-old_x; y=old_z; z=65-old_y (mm); arrays packed x,z,y','status':'rigidly converted historical results; not new corrected-placement solves','history_counts':{'prior_records':80,'prior_valid':sum(r['valid'] for r in hist[:80]),'current_records':160,'current_valid':sum(r['valid'] for r in active)}},indent=2)+'\n')
    (OUT/'run6_run9_run10_residuals.caption.md').write_text('Signed height differences for Run 6 calibration and Runs 9/10 frozen-parameter transfer. All retained scan points supported by the projected native model facets are plotted; Δy = Analytical − Experiment. Shared symmetric limits include all supported values without clipping. Grey indicates unmeasured locations within the plotted window; black crosses identify scan points outside model support, if any. Native-facet interpolation, no smoothing or new alignment. (d) Empirical distributions of absolute height differences, separately normalized per specimen; spatially correlated points are not independent replicates. These are height differences, not the bidirectional geometric optimization objective. Coverage counts and full field arrays accompany the figure. Runs 9/10 are not claimed to be independently held-out. Run 7 omitted by researcher choice; no all-experiment performance claim.\n')
    (OUT/'run6_run9_run10_sections.caption.md').write_text('Fixed-location sections across Run 6 calibration and Runs 9/10 frozen-parameter transfers. Columns: z = −35, 65, 165 mm, selected without inspecting errors. Experiment: orange solid lines and markers; Analytical: blue dashed numerical shell-model predictions (not closed-form solutions). Experimental triangles with any XZ edge exceeding 5 mm are excluded. Sections sampled every 0.5 mm; experimental markers denote interpolated values, not raw samples. Native model facets supply analytical heights. Unsupported coordinates remain gaps. All panels share physical axis limits; vertical and horizontal display scales differ. No new solve, registration, smoothing fit or parameter adjustment. One specimen per row; no confidence bands or independent-validation claim.\n')
    (OUT/'run6_objective_history.caption.md').write_text('Run 6 archived objective records. (a) Attempts 1–80 are retrospective rescored, training-only prior records, not a contemporaneous optimization trajectory under the final objective. (b) Attempts 81–240 comprise the expanded search and continuation; blue staircase is best valid current-search value, without an incumbent inherited from training-only records. All valid records were checked against material-reference symmetric RMS. Invalid records are crosses in a separate axis-relative bottom strip, not penalty-valued observations. Logarithmic ordinate. Attempt IDs are archived ordering, not wall-clock completion order (batch evaluations may run concurrently), and exclude unrecorded infrastructure retries. Selected attempt 209: 1.1823136 mm. This is one adaptive search, not evidence of convergence or global optimality. Ledger counts: 70 valid/10 invalid prior records and 151 valid/9 invalid current-search records. An audit of original historical proposal decisions remains pending.\n')
    print(OUT)

if __name__=='__main__': main()
