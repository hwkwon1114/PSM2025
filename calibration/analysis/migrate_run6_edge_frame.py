"""Non-destructive fixed scan-edge frame migration; no solver or GP entry points."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'run/optimization_review/run6_edge_alignment'
SNAP=ROOT/'snapshot'
TARGET='calibration/data/processed/run6_edge_aligned.npz'
CONFIG='calibration/configs/run6_bo_edge_aligned.json'
HISTORY='run/run6_bo_edge_aligned/corrected_history.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def snap(p):return SNAP/p

def freeze():
    if ROOT.exists() and any(ROOT.iterdir()):raise FileExistsError('Preserve partial or completed migration')
    ROOT.mkdir(exist_ok=True)
    checkpoint='run/run6_bo_material_frame/bayes_state.json'
    s=read(BASE/checkpoint);assert s['status']=='budget_complete'
    observations=sorted(s['observations'],key=lambda r:r['attempt_id'])
    assert [r['attempt_id'] for r in observations]==list(range(1,81))
    assert sum(r['valid'] for r in observations)==70
    plan=read(BASE/'run/optimization_review/free_translation_plan.json')
    endpoints={r['attempt_id']:r for r in plan['endpoints']}
    sources=[]
    for r in observations:
        if not r['valid']:
            assert r['failure_class']=='timeout'
            # Modern evaluator uses an operational1e6 sentinel; it is NOT a loss observation.
            continue
        if r['attempt_id']<=40:
            e=endpoints[r['attempt_id']]
            entry={k:e[k] for k in ['mesh_file','state_file','sequence_file','scale_to_mm']}
        else:
            case=r['cases'][0];folder=Path(case['directory']).relative_to(BASE)
            entry={'mesh_file':str(folder/'prediction_mm.vtp'),'state_file':str(folder/'bilayer_zigzag_sequence_native_hybrid_state.json'),
                   'sequence_file':str(folder/'sequence_result.json'),'scale_to_mm':1.}
        entry.update(attempt_id=r['attempt_id'],old_rms_mm=r['objective_mm'],parameters=r['parameters'])
        sources.append(entry)
    controls=[checkpoint,'run/run6_bo_material_frame/bayes_summary.json','run/run6_bo_material_frame/corrected_history.json',
              'run/optimization_review/free_translation_plan.json','run/optimization_review/direction_frame_gallery/direction_frame_results.json',
              'calibration/configs/run6_bo_material_frame.json','calibration/data/processed/run6.json','calibration/data/processed/run6_oriented.json',
              'calibration/data/raw/run6_zigzag_path.csv','calibration/ewcal/common.py','calibration/ewcal/geometry.py',
              'calibration/ewcal/material_frame.py','calibration/ewcal/prepare.py','calibration/ewcal/runner.py',
              'calibration/analysis/orient_run9_edges.py','scripts/run_run6_bo_refinement.py','scripts/run_run6_bo_material_frame.sbatch']
    files=set(controls+['calibration/data/raw/run6.stl','calibration/data/processed/run6.npz','calibration/data/processed/run6_oriented.npz'])
    for e in sources:files.update(e[k] for k in ['mesh_file','state_file','sequence_file'])
    pins={p:sha(BASE/p) for p in sorted(files)}
    for p in controls:
        dest=snap(p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/p,dest)
        assert sha(dest)==pins[p]
    for p,h in pins.items():assert sha(BASE/p)==h,p
    save(ROOT/'manifest.json',{'schema_version':1,'status':'frozen','slurm_job':os.environ.get('SLURM_JOB_ID'),
         'original_input_sha256':pins,'snapshot_controls':controls,'sources':sources,'observations':observations,
         'attempts':80,'completed_shapes':70,'timeout_ids':[r['attempt_id'] for r in observations if not r['valid']],
         'no_mechanics_or_gp':True})
    print('Frozen80 attempts /70 completed shapes /10 timeouts;',len(pins),'original files pinned')

def verify():
    m=read(ROOT/'manifest.json')
    for p,h in m['original_input_sha256'].items():
        if sha(BASE/p)!=h:raise RuntimeError('Protected input drift: '+p)
    for p in m['snapshot_controls']:
        if sha(snap(p))!=m['original_input_sha256'][p]:raise RuntimeError('Snapshot drift: '+p)
    return m

def prepare():
    import csv
    import numpy as np
    from calibration.analysis.orient_run9_edges import edge_angles,yaw_matrix,self_test
    from calibration.ewcal.geometry import read_mesh,points
    from calibration.ewcal.prepare import prepare_case
    m=verify()
    destinations=[BASE/TARGET,(BASE/TARGET).with_suffix('.json'),(BASE/TARGET).with_suffix('.png'),BASE/CONFIG,
                  ROOT/'edge_diagnostic.png',ROOT/'edge_diagnostic.pdf',ROOT/'edge_alignment.json']
    if any(p.exists() for p in destinations):raise FileExistsError('Preserve existing/partial edge outputs')
    self_test()
    meta=read(snap('calibration/data/processed/run6.json'))
    t=meta['transform'];basis=np.array(t['basis_columns']);turn=np.diag([1.,-1.,-1.]);old_yaw=yaw_matrix(1.5)
    # Historical oriented metadata records its extra rotation only in prose.
    with np.load(BASE/'calibration/data/processed/run6.npz',allow_pickle=False) as f:base_cloud=f['points_mm'].copy()
    with np.load(BASE/'calibration/data/processed/run6_oriented.npz',allow_pickle=False) as f:old_cloud=f['points_mm'].copy()
    reconstructed=base_cloud@turn@old_yaw.T
    np.testing.assert_allclose(reconstructed,old_cloud,atol=1e-9,rtol=0)
    raw=np.unique(points(read_mesh(BASE/'calibration/data/raw/run6.stl',meta['scan_scale_to_mm'])),axis=0)
    base=(raw-np.array(t['origin_scanner_mm']))@basis-np.array(t['shift_local_mm'])
    before=base@turn@old_yaw.T
    long=edge_angles(before[:,:2]);short=edge_angles(before[:,[1,0]])
    angle=.5*(long['mean_angle_deg']-short['mean_angle_deg'])
    coarse_long=edge_angles(before[:,:2],2.);coarse_short=edge_angles(before[:,[1,0]],2.)
    coarse_angle=.5*(coarse_long['mean_angle_deg']-coarse_short['mean_angle_deg'])
    rotation=yaw_matrix(angle);after=before@rotation.T
    np.testing.assert_allclose(before[:,2],after[:,2],atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(before,axis=1),np.linalg.norm(after,axis=1),atol=1e-10,rtol=0)
    full_basis=basis@turn@old_yaw.T@rotation.T
    np.testing.assert_allclose(full_basis.T@full_basis,np.eye(3),atol=1e-12,rtol=0)
    assert abs(np.linalg.det(full_basis)-1)<1e-12
    long_after=edge_angles(after[:,:2]);short_after=edge_angles(after[:,[1,0]])
    config=read(snap('calibration/configs/run6_bo_material_frame.json'))
    config['cases'][0]['target']=TARGET
    config['cases'][0]['scan_orientation']={'basis_columns':full_basis.tolist(),'flip_z':False,'flip_xy':False}
    config['output_dir']='run/run6_bo_edge_aligned';config['report_dir']='run/run6_bo_edge_aligned/report'
    # Only target/orientation and isolated output paths differ, not the scientific model.
    original=copy.deepcopy(config);prior=read(snap('calibration/configs/run6_bo_material_frame.json'))
    original['cases'][0]=prior['cases'][0]
    original['output_dir']=prior['output_dir'];original['report_dir']=prior['report_dir']
    assert original==prior
    prepared=prepare_case(config,config['cases'][0])
    details={'incremental_z_yaw_deg':angle,'old_post_turnover_z_yaw_deg':1.5,'new_post_turnover_z_yaw_deg':1.5+angle,
             'coarse_incremental_yaw_deg':coarse_angle,'long_before':long,'short_before':short,
             'long_after':long_after,'short_after':short_after,'method':'All four uncropped dense-component occupancy/Theil–Sen boundaries; no curvature or simulation fit.'}
    prepared.update(target_npz=TARGET,orientation_notes=f'Existing x-turnover preserved; total z-yaw {1.5+angle:+.9f}deg from geometric sheet edges; complete basis/shift recorded.',edge_alignment=copy.deepcopy(details))
    # prepare_case writes metadata first; enrich that newly owned output only.
    (BASE/TARGET).with_suffix('.json').write_text(json.dumps(prepared,indent=2)+'\n')
    save(BASE/CONFIG,config)
    with np.load(BASE/TARGET,allow_pickle=False) as f:cloud=f['points_mm'].copy()
    nt=prepared['transform'];aligned=(raw-np.array(nt['origin_scanner_mm']))@np.array(nt['basis_columns'])-np.array(nt['shift_local_mm'])
    height_shift=float(np.median(aligned[:,2]-before[:,2]))
    np.testing.assert_allclose(aligned[:,2]-before[:,2],height_shift,atol=1e-10,rtol=0)
    idx=np.random.default_rng(2026).choice(len(raw),40000,replace=False)
    before_display=before[idx].copy();before_display[:,2]+=height_shift
    with snap('calibration/data/raw/run6_zigzag_path.csv').open() as f:
        path=np.array([[float(r[k]) for k in ['x_mm','y_mm']] for r in csv.DictReader(f)])
    path[:,1]-=65.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    fig,axes=plt.subplots(1,3,figsize=(12,5.4),layout='constrained')
    lo,hi=float(cloud[:,2].min()),float(cloud[:,2].max())
    for ax,p,title in zip(axes,[before_display,aligned[idx],cloud],['Old fixed yaw (uncropped)','Edge-aligned (uncropped)','New prepared target']):
        artist=ax.scatter(p[:,0],p[:,1],c=p[:,2],s=2,cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
        ax.plot(path[:,0],path[:,1],color='#111111',lw=.6,label='Nominal CSV path')
        ax.plot(path[0,0],path[0,1],marker='*',color='#D55E00',ms=7,label='Start')
        ax.plot([-98,98,98,-98,-98],[-148,-148,148,148,-148],color='#D55E00',ls='--',lw=.7,label='Crop')
        ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)',aspect='equal',xlim=(-125,125),ylim=(-175,175))
        ax.set_xticks([-100,0,100]);ax.set_yticks([-150,-100,-50,0,50,100,150])
    axes[-1].legend(loc='lower left',fontsize=7)
    fig.colorbar(artist,ax=axes.tolist(),label='Height (mm)',extend='both',ticks=np.linspace(lo,hi,5),format='%.1f',shrink=.8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(renderer)
            assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,text.get_text()
    for ext in ['png','pdf']:fig.savefig(ROOT/f'edge_diagnostic.{ext}',dpi=250)
    plt.close(fig)
    verify()
    details.update(status='prepared',prepared_metadata=prepared,old_point_count=len(old_cloud),new_point_count=len(cloud),
                   height_shift_mm=height_shift,raw_display_subset=40000,seed=2026,
                   raw_display_outside_limits_after=int(np.sum(np.any(np.abs(aligned[idx,:2])>[125,175],axis=1))),
                   new_target_sha256=sha(BASE/TARGET),new_config_sha256=sha(BASE/CONFIG),
                   script_sha256=sha(Path(__file__)),python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,
                   slurm_job=os.environ.get('SLURM_JOB_ID'))
    save(ROOT/'edge_alignment.json',details)
    print(json.dumps(details,indent=2))

def finalize_prepare():
    """Audit saved outputs and finalize reporting after serialization failure; no reprocessing."""
    import numpy as np
    from calibration.analysis.orient_run9_edges import yaw_matrix
    from calibration.ewcal.geometry import read_mesh,points
    verify()
    meta=read((BASE/TARGET).with_suffix('.json'))
    assert sha(BASE/TARGET)==meta['target_sha256']
    old=read(snap('calibration/data/processed/run6.json'));t=old['transform'];nt=meta['transform']
    raw=np.unique(points(read_mesh(BASE/'calibration/data/raw/run6.stl')),axis=0)
    base=(raw-np.array(t['origin_scanner_mm']))@np.array(t['basis_columns'])-np.array(t['shift_local_mm'])
    before=base@np.diag([1.,-1.,-1.])@yaw_matrix(1.5).T
    after=(raw-np.array(nt['origin_scanner_mm']))@np.array(nt['basis_columns'])-np.array(nt['shift_local_mm'])
    height_shift=float(np.median(after[:,2]-before[:,2]))
    np.testing.assert_allclose(after[:,2]-before[:,2],height_shift,atol=1e-10,rtol=0)
    floor=meta['preprocess']['z_outlier_floor_mm']
    def keep(q):
        height=q[:,2]-np.median(q[:,2]);cut=max(floor,12*np.median(np.abs(height)))
        return np.all(np.abs(q[:,:2])<=[98,148],axis=1)&(np.abs(height)<=cut)
    old_keep,new_keep=keep(base),keep(after)
    idx=np.random.default_rng(2026).choice(len(raw),40000,replace=False)
    report=copy.deepcopy(meta['edge_alignment'])
    report.update(status='prepared_and_audited',prepared_metadata=meta,
                  old_point_count=read(snap('calibration/data/processed/run6_oriented.json'))['accepted_points'],
                  new_point_count=meta['accepted_points'],height_shift_mm=height_shift,
                  raw_display_subset=40000,seed=2026,
                  display_outside_limits_before=int(np.sum(np.any(np.abs(before[idx,:2])>[125,175],axis=1))),
                  display_outside_limits_after=int(np.sum(np.any(np.abs(after[idx,:2])>[125,175],axis=1))),
                  raw_crop_before_cell_mad={'old_retained':int(old_keep.sum()),'new_retained':int(new_keep.sum()),
                                          'intersection':int((old_keep&new_keep).sum()),
                                          'newly_included':int((new_keep&~old_keep).sum()),'newly_excluded':int((old_keep&~new_keep).sum())},
                  target_sha256=sha(BASE/TARGET),config_sha256=sha(BASE/CONFIG),
                  before_serialization_fix_source_sha256=sha(ROOT/'prepare_source_before_serialization_fix.py'),
                  audit_script_sha256=sha(Path(__file__)),python=sys.version,numpy=np.__version__,
                  audit_slurm_job=os.environ.get('SLURM_JOB_ID'),
                  limitation='Zero estimated additional yaw at1/2mm occupancy resolution is not a sub-bin angle certificate; principal change is rotate-before-crop reprocessing, not a newly nonzero yaw.')
    verify()
    save(ROOT/'edge_alignment_complete.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','prepare','finalize-prepare'])
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        sys.path.insert(0,str(BASE))
        prepare() if args.stage=='prepare' else finalize_prepare()
