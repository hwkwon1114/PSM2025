#!/usr/bin/env python3
"""Saved-checkpoint-only report; includes partial/failed cells, no oracle calls."""
import argparse,csv,json,os,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_convergence_benchmark import read,sha,exclusive,export,align,CATS,COLORS,HATCH


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;out=root/'audit';out.mkdir(exist_ok=True)
    ledger=read(root/'screen/ledger.json');protocol=read(root/'protocol.json');bundle=Path(protocol['request_base']['bundle'])
    assert sha(bundle/'manifest.json')==protocol['request_base']['identity']
    for name,h in read(bundle/'manifest.json')['sha256'].items():assert sha(bundle/name)==h
    meta=read(bundle/'metadata.json');nv=meta['nv'];nd=meta['nd']
    records=[];curves={};shapes={};hashes={str(root/'screen/ledger.json'):sha(root/'screen/ledger.json')}
    for cell in ledger['cells']:
        r=dict(index=cell['index'],start=cell['start'],status=cell['status'],charged_seconds=cell.get('charged_seconds',0),accepted=False)
        if 'result' not in cell:
            r['missing_result']=True;records.append(r);continue
        result=cell['result'];cp=Path(result['checkpoint']);assert sha(cp)==cell['checkpoint_sha256'];hashes[str(cp)]=sha(cp)
        payload=read(cp)['payload'];state=payload['state'];trace=state['trace']
        if protocol['request_base'].get('guarded_hessian',False) and result['phase']=='newton':
            assert payload['config']['guardedHessianReuse'] and not payload['config'].get('reuseNewtonSymbolic',False)
        if protocol['request_base'].get('hessian_threads',1)>1 and result['phase']=='newton':
            assert payload['config']['hessianThreads']==protocol['request_base']['hessian_threads']
            assert not payload['config'].get('guardedHessianReuse',False)
            assert payload['config'].get('reuseNewtonSymbolic',False)==protocol['request_base'].get('reuse_symbolic',False)
            assert all(v.get('hessian_threads')==4 for v in trace if 'hessian_oracle_seconds' in v)
        if result['accepted']:assert result['status']=='gradient_target' and result['selected_gradient_norm']<=5e-14 and result['curvature']=='positive_pivots'
        valid=[v for v in trace if 'energy' in v and 'gradient_norm' in v]
        data=np.array([[v['elapsed'],v['energy'],v['gradient_norm']] for v in valid])
        if len(data):
            assert np.isfinite(data).all() and (data[:,1:]>0).all() and (np.diff(data[:,0])>=0).all()
            curves[cell['index']]=data
        with (out/('trace_%02d.csv'%cell['index'])).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['optimizer_elapsed_s','energy','native_gradient_norm']);w.writerows(data)
        qpath=Path(cell['out'])/'final_x.f64';q=np.fromfile(qpath,dtype='<f8');assert len(q)==nd and np.array_equal(q,np.array(state['x']))
        if result['accepted']:assert np.array_equal(q,np.fromfile(Path(cell['out'])/'selected_x.f64',dtype='<f8'))
        shapes[cell['index']]=q[:3*nv].reshape(nv,3,order='F');hashes[str(qpath)]=sha(qpath)
        phases={}
        for key in sorted({k for v in trace for k in v if k.endswith('_seconds')}):
            vals=[v[key] for v in trace if key in v];phases[key]=dict(seconds=sum(vals),median=statistics.median(vals),count=len(vals))
        steps=[v for v in trace if 'shift' in v]
        r.update(accepted=result['accepted'],energy=result['energy'],gradient_norm=result['gradient_norm'],
            hessian_threads=payload['config'].get('hessianThreads',1),
            selected_gradient_norm=result.get('selected_gradient_norm'),curvature=result.get('curvature'),attempts=result['attempts'],
            evaluations=result['evaluations'],hessians=result['hessians'],phase=result['phase'],phases=phases,
            exclusive_seconds=exclusive(trace,cell['charged_seconds']),newton_attempts=len(steps),
            newton_updates=sum(bool(v.get('accepted',False)) for v in steps),
            reused_model_attempts=sum(bool(v.get('hessian_reused',False)) for v in trace),
            accepted_reused_updates=sum(bool(v.get('hessian_reused',False)) and bool(v.get('accepted',False)) for v in trace),
            rejected_stale_trials=sum(v.get('hessian_refresh_reason')=='stale_full_step_rejected' for v in trace),
            restriction_refreshes=sum(v.get('hessian_refresh_reason')=='age_or_restricted_dofs_changed' for v in trace),
            shifted_updates=sum(v['shift']>0 and bool(v.get('accepted',False)) for v in steps),backtracked_updates=sum(v.get('alpha',1)<1 and bool(v.get('accepted',False)) for v in steps),
            handoff_seconds=next((v['elapsed'] for v in trace if v.get('iterations')==2000),None),
            gradient_crossings={str(t):next((v['elapsed'] for v in valid if v['gradient_norm']<=t),None) for t in [1e-8,1e-10,1e-12,5e-14]})
        records.append(r)
        del state,trace
    available=[r for r in records if 'exclusive_seconds' in r]
    charged=sum(r['charged_seconds'] for r in available)
    totals={k:sum(r['exclusive_seconds'][k] for r in available) for k in CATS}
    audit=dict(cells=records,planned_cells=protocol['cells'],aggregate=dict(charged_seconds=charged,exclusive_seconds=totals,
        fractions={k:v/charged if charged else None for k,v in totals.items()}),source_hashes=hashes,
        script_sha256=sha(__file__),job=os.getenv('SLURM_JOB_ID'),review='pending',
        caveats=['Two dependent starts of one objective, no independent-replicate uncertainty or general method ranking.',
          'All available traces and full endpoint meshes retained. Failed/unstarted cells remain listed.',
          'Unattributed charged time is a residual, not measured I/O. Hessian subtracted from inclusive model timer.',
          'Curves use recorded optimizer time; bars use charged cell time.',
          'Capped endpoints are not certified minima. Contact and mesh adequacy unverified.'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if available:
        with plt.rc_context({'font.size':11,'pdf.fonttype':42}):
            fig,axs=plt.subplots(3,1,figsize=(8,9),layout='constrained')
            styles={0:('#0072B2','-','Positive start'),1:('#D55E00','--','Negative start')}
            for r in available:
                i=r['index'];color,ls,label=styles[i];d=curves.get(i)
                if d is None:continue
                for ax,col in zip(axs[:2],[1,2]):
                    ax.plot(d[:,0]/60,d[:,col],color=color,ls=ls,label=label)
                    ax.plot(d[-1,0]/60,d[-1,col],marker='^' if r['accepted'] else 'o',mfc=color if r['accepted'] else 'white',mec=color,ls='none')
                    if r['handoff_seconds'] is not None:ax.axvline(r['handoff_seconds']/60,color=color,ls=':',lw=.8)
                left=0
                for cat,c,h in zip(CATS,COLORS,HATCH):
                    val=r['exclusive_seconds'][cat]/60;axs[2].barh(i,val,left=left,color=c,hatch=h,edgecolor='white',height=.55,label=cat if r==available[0] else None);left+=val
            axs[0].set(ylabel='Energy',xlabel='Recorded optimizer time (min)');axs[0].legend(frameon=False)
            axs[1].set(yscale='log',ylabel='Native gradient norm',xlabel='Recorded optimizer time (min)');axs[1].axhline(5e-14,color='black',ls=':',label='Target: 5×10⁻¹⁴');axs[1].legend(frameon=False)
            axs[2].set(yticks=[r['index'] for r in available],yticklabels=[styles[r['index']][2] for r in available],xlabel='Charged cell time (min)')
            axs[2].invert_yaxis();axs[2].legend(loc='upper left',bbox_to_anchor=(0,-.25),ncol=3,frameon=False,fontsize=9)
            export(fig,out,'profiled_hybrid_progress_timing')
    if shapes:
        ref=np.fromfile(bundle/'reference.f64',dtype='<f8').reshape(nv,3,order='F')
        faces=np.fromfile(bundle/'faces.i32',dtype='<i4').reshape(meta['nf'],3,order='F')
        anchor=align(shapes[min(shapes)],ref);aligned={i:align(x,anchor)*1000 for i,x in shapes.items()}
        allx=np.vstack(list(aligned.values()));lo=allx.min(axis=0);hi=allx.max(axis=0);span=np.maximum(hi-lo,1.);center=(lo+hi)/2
        with plt.rc_context({'font.size':10,'pdf.fonttype':42}):
            fig=plt.figure(figsize=(11,5));fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.87,wspace=.12)
            for i in range(2):
                ax=fig.add_subplot(1,2,i+1,projection='3d')
                if i not in aligned:ax.set_title(('Positive' if i==0 else 'Negative')+' start: unavailable');continue
                x=aligned[i];r=next(r for r in records if r['index']==i)
                ax.plot_trisurf(x[:,0],x[:,1],x[:,2],triangles=faces,color='#79a9be',linewidth=0,antialiased=False,rasterized=True)
                ax.set(xlim=(center[0]-.55*span[0],center[0]+.55*span[0]),ylim=(center[1]-.55*span[1],center[1]+.55*span[1]),zlim=(center[2]-.55*span[2],center[2]+.55*span[2]),xlabel='x (mm)',ylabel='y (mm)')
                ax.set_box_aspect(tuple(span));ax.view_init(elev=25,azim=-60)
                ax.set_title(('Positive' if i==0 else 'Negative')+' start\n'+('Accepted' if r['accepted'] else r['status']))
                if i==1:ax.set_zlabel('z (mm)')
            export(fig,out,'profiled_hybrid_final_shapes')
        if len(aligned)==2:
            d=np.linalg.norm(aligned[0]-aligned[1],axis=1)
            audit['aligned_endpoint_difference_mm']=dict(rms=float(np.sqrt(np.mean(d*d))),max=float(d.max()))
            (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'caption.md').write_text('Two hybrid trajectories for the complete ABC target, ortho −0.5, with Voce hardening. '
        'No intermediate equilibrium solves. Dotted vertical lines mark the fixed 2000-attempt handoff, where reached. '
        'Filled triangles indicate accepted endpoints; open circles denote unfinished endpoints. All saved finite observations are shown without smoothing. '
        'Timing categories are exclusive; unattributed time includes verification, checkpointing and uninstrumented work, not a measured I/O total. '
        'Meshes use every face, matched vertex IDs, proper rigid alignment only and common physical scale/aspect. '
        'Two starts are not independent replications. Geometry images do not validate contact or mesh convergence. '
        'Partial/missing outcomes are retained in audit.json; PDF/PNG render review is pending.\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
