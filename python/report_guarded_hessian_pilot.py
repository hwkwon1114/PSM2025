#!/usr/bin/env python3
"""Saved-results comparison against preserved exact-Hessian baseline; no solves."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_convergence_benchmark import read,sha,align,export,CATS,COLORS,HATCH
import report_profiled_zigzag_pilot as candidate_report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    candidate_report.main()
    out=root/'audit';protocol=read(root/'protocol.json');baseline=Path(protocol['baseline'])
    oldprotocol=read(baseline/'protocol.json');assert oldprotocol['request_base']['identity']==protocol['request_base']['identity']
    bundle=Path(protocol['request_base']['bundle']);meta=read(bundle/'metadata.json');nv=meta['nv']
    ref=np.fromfile(bundle/'reference.f64',dtype='<f8').reshape(nv,3,order='F')
    faces=np.fromfile(bundle/'faces.i32',dtype='<i4').reshape(meta['nf'],3,order='F')
    curves={};shapes={};records={};provenance={};pairs=[];configs={};endpoint_hashes={}
    parallel=protocol['request_base'].get('hessian_threads',1)>1
    symbolic=protocol['request_base'].get('reuse_symbolic',False)
    candidate='composed' if (parallel and symbolic) else ('parallel' if parallel else 'guarded')
    cand_label='Composed (4T + symbolic)' if (parallel and symbolic) else ('Four threads' if parallel else 'Guarded reuse')
    starts=['cylinder_y_plus','cylinder_y_minus'];labels={'exact':'Serial baseline',candidate:cand_label}
    styles={'exact':('#0072B2','-'),candidate:('#009E73' if (parallel and symbolic) else '#D55E00','--')}
    residuals={}
    for name,source in [('exact',baseline),(candidate,root)]:
        audit=read(source/'audit/audit.json');record_map={r['start']:r for r in audit['cells']}
        ledger=read(source/'screen/ledger.json');provenance[str(source/'screen/ledger.json')]=sha(source/'screen/ledger.json')
        for c in ledger['cells']:
            key=(name,c['start']);records[key]=record_map[c['start']]
            if 'result' not in c:continue
            result=c['result'];cp=Path(result['checkpoint']);assert sha(cp)==c['checkpoint_sha256'];provenance[str(cp)]=sha(cp)
            payload=read(cp)['payload'];state=payload['state'];configs[key]=payload['config']
            if name==candidate and result['phase']=='newton':
                assert payload['config'].get('guardedHessianReuse',False)==protocol['request_base'].get('guarded_hessian',False)
                assert payload['config'].get('reuseNewtonSymbolic',False)==symbolic
                assert payload['config'].get('hessianThreads',1)==(4 if parallel else 1)
            if name=='exact':assert not payload['config'].get('guardedHessianReuse',False)
            assert result['accepted']==records[key]['accepted']
            if result['accepted']:assert result['status']=='gradient_target' and result['selected_gradient_norm']<=5e-14 and result['curvature']=='positive_pivots'
            trace=[v for v in state['trace'] if 'energy' in v and 'gradient_norm' in v]
            curves[key]=np.array([[v['elapsed'],v['energy'],v['gradient_norm']] for v in trace])
            assert np.isfinite(curves[key]).all() and np.all(np.diff(curves[key][:,0])>=0)
            xfile=Path(c['out'])/'final_x.f64';q=np.fromfile(xfile,dtype='<f8');assert np.array_equal(q,np.array(state['x']))
            shapes[key]=q[:3*nv].reshape(nv,3,order='F');provenance[str(xfile)]=sha(xfile);endpoint_hashes[key]=sha(xfile)
    for start in starts:
        a=records.get(('exact',start));b=records.get((candidate,start));row=dict(start=start,baseline=a,**{candidate:b})
        if a and b and 'energy' in a and 'energy' in b:
            if parallel:
                row['same_terminal_phase']=a['phase']==b['phase']
                if row['same_terminal_phase']:
                    keep=lambda cfg:{k:v for k,v in cfg.items() if k not in ('implementationId','dependencyIdentity','hessianThreads','reuseNewtonSymbolic')}
                    assert keep(configs['exact',start])==keep(configs[candidate,start]), 'numerical recipe mismatch'
            row['final_state_byte_identical']=endpoint_hashes['exact',start]==endpoint_hashes[candidate,start]
            row['energy_gradient_trace_exact']=np.array_equal(curves['exact',start][:,1:],curves[candidate,start][:,1:])
            ratio=b['charged_seconds']/a['charged_seconds']
            row.update(both_accepted=a['accepted'] and b['accepted'],
                       **{'charged_time_ratio_'+candidate+'_over_exact':ratio},
                       relative_energy_difference=(b['energy']-a['energy'])/abs(a['energy']))
            if parallel:row['allocated_cell_core_seconds_ratio']=4*ratio
            row['vertex_coordinates_exact']=np.array_equal(shapes[candidate,start],shapes['exact',start])
            delta=align(shapes[candidate,start],shapes['exact',start])-shapes['exact',start]
            d=np.linalg.norm(delta,axis=1)*1000;residuals[start]=d
            row['aligned_vertex_difference_mm']=dict(rms=float(np.sqrt(np.mean(d*d))),p95=float(np.quantile(d,.95)),maximum=float(d.max()))
            row['unaligned_vertex_max_difference_mm']=float(np.linalg.norm(shapes[candidate,start]-shapes['exact',start],axis=1).max()*1000)
            with (out/(start+'_'+candidate+'_vs_exact_vertices.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['vertex_id','distance_mm']);w.writerows(enumerate(d))
        pairs.append(row)
    comparison=dict(pairs=pairs,source_hashes=provenance,script_sha256=sha(__file__),styles=styles,review='pending',
        caveats=['Two dependent prescribed starts, not independent replications or a mesh-convergence study.',
        (('Four-thread Hessian evaluation with coordinate-stable symbolic factorization reuse. Baseline fits preserved, not rerun.' if (parallel and symbolic) else
          'Only Hessian face evaluation uses four threads; reuse options off. Baseline fits preserved, not rerun.') if parallel else
         'Only guarded Hessian reuse is enabled; symbolic reuse off. Baseline fits preserved, not rerun.'),
        'Four-thread allocation includes serial work; allocated core-seconds are not measured CPU time.' if parallel else 'Single-CPU allocations.',
        'Time ratios are descriptive; acceptance, energy and shape differences must be considered before interpreting speedup.',
        'Different execution nodes/code builds can influence timing. Runtime library identity is checked by launcher.',
        'All available curves and full meshes shown. Failed/unstarted cells are retained as missing, not zero.',
        'Rigid alignment uses fixed IDs, proper rotation/translation only. No reflection, scaling or contact test.'])
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    with plt.rc_context({'font.size':11,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for col,start in enumerate(starts):
            axes[0,col].set_title('Positive start' if col==0 else 'Negative start')
            for name in ['exact',candidate]:
                key=(name,start);color,ls=styles[name]
                if key not in curves:
                    axes[0,col].text(.05,.08,labels[name]+': unavailable',transform=axes[0,col].transAxes);continue
                d=curves[key];r=records[key]
                for ax,k in [(axes[0,col],1),(axes[1,col],2)]:
                    ax.plot(d[:,0]/60,d[:,k],color=color,ls=ls,label=labels[name])
                    ax.plot(d[-1,0]/60,d[-1,k],marker='^' if r['accepted'] else 'o',mfc=color if r['accepted'] else 'white',mec=color)
                    if r.get('handoff_seconds') is not None:ax.axvline(r['handoff_seconds']/60,color=color,ls=':',lw=.8)
            axes[0,col].set_ylabel('Energy');axes[0,col].legend(frameon=False)
            axes[1,col].set(yscale='log',ylabel='Native gradient norm',xlabel='Recorded optimizer time (min)')
            axes[1,col].axhline(5e-14,color='black',ls=':',lw=1)
        export(fig,out,candidate+'_vs_exact_progress')
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained');yt=[];ys=[]
        for sign,start in enumerate(starts):
            for method,name in enumerate(['exact',candidate]):
                y=2*sign+method;yt.append(labels[name]+(' (+)' if sign==0 else ' (−)'));ys.append(y)
                r=records.get((name,start),{})
                if 'exclusive_seconds' not in r:
                    ax.text(.02,y,'Unavailable: '+r.get('status','unstarted'),transform=ax.get_yaxis_transform());continue
                left=0
                for category,color,hatch in zip(CATS,COLORS,HATCH):
                    v=r['exclusive_seconds'][category]/60
                    ax.barh(y,v,left=left,height=.6,color=color,hatch=hatch,edgecolor='white',label=category if y==0 else None);left+=v
        ax.set(yticks=ys,yticklabels=yt,xlabel='Charged cell time (min)');ax.invert_yaxis()
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),ncol=3,frameon=False,fontsize=9)
        export(fig,out,candidate+'_vs_exact_timing')
        if residuals:
            fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
            for ax,start in zip(axes,starts):
                ax.set_title('Positive start' if start==starts[0] else 'Negative start')
                if start in residuals:
                    d=np.sort(residuals[start]);ax.step(d,np.arange(1,len(d)+1)/len(d),where='post',color='#0072B2')
                else:ax.text(.05,.5,'Unavailable',transform=ax.transAxes)
                ax.set(xlabel='Aligned vertex distance (mm)',ylabel='Fraction of vertices',ylim=(0,1))
                ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
            export(fig,out,candidate+'_vs_exact_vertex_ecdf')
    if shapes:
        anchor=align(shapes['exact',starts[0]],ref);aligned={k:align(v,anchor)*1000 for k,v in shapes.items()}
        points=np.vstack(list(aligned.values()));center=(points.max(0)+points.min(0))/2;span=np.maximum(np.ptp(points,axis=0),1.)
        fig=plt.figure(figsize=(11,9));fig.subplots_adjust(left=.03,right=.97,bottom=.06,top=.94,wspace=.12,hspace=.18)
        for si,start in enumerate(starts):
            for mi,name in enumerate(['exact',candidate]):
                ax=fig.add_subplot(2,2,2*si+mi+1,projection='3d');key=(name,start);r=records.get(key,{})
                ax.set_title(labels[name]+(' (+)' if si==0 else ' (−)')+'\n'+('Accepted' if r.get('accepted') else r.get('status','unstarted')))
                if key not in aligned:continue
                x=aligned[key];ax.plot_trisurf(x[:,0],x[:,1],x[:,2],triangles=faces,color='#79a9be',linewidth=0,antialiased=False,rasterized=True)
                ax.set(xlim=(center[0]-.55*span[0],center[0]+.55*span[0]),ylim=(center[1]-.55*span[1],center[1]+.55*span[1]),zlim=(center[2]-.55*span[2],center[2]+.55*span[2]),xlabel='x (mm)',ylabel='y (mm)')
                if mi==1:ax.set_zlabel('z (mm)')
                ax.set_box_aspect(tuple(span));ax.view_init(elev=25,azim=-60)
        export(fig,out,candidate+'_vs_exact_shapes')
    opening=('Serial baseline versus composed four-thread Hessian assembly and symbolic reuse' if (parallel and symbolic) else
             ('Serial baseline versus four-thread Hessian assembly' if parallel else 'Exact-Hessian baseline versus guarded one-update reuse'))
    (out/'comparison.caption.md').write_text(opening+' on the same fixed ABC growth objective and mesh. '
        'Both sign-paired starts are included; no baseline fit was rerun. All available trace points and all mesh faces are retained. '
        'Filled triangles mark accepted endpoints; open circles mark unfinished endpoints; dotted vertical lines indicate the fixed handoff after 2000 L-BFGS attempts. '
        'The horizontal gradient threshold is 5e-14. Timing bars use exclusive categories and charged cell time; curves use optimizer elapsed time. '
        'All attempts remain in traces. Parallel assembly does not change the declared stopping or acceptance rules. '
        'The full vertex-distance ECDF is spatially dependent, not an independent-replicate uncertainty estimate. '
        'When raw coordinates match exactly, nonzero aligned distances can reflect floating-point rigid-alignment roundoff. '
        'Shapes share physical axes/aspect after proper rigid alignment without scaling/reflection. Full pairwise vertex residuals are supplied as CSV. '
        'Acceptance does not establish global optimality, mesh convergence or physical calibration; two starts are not independent replicates. '
        'PNG/PDF rendering review is pending.\n')
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
