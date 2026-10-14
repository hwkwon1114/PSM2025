"""Replay saved mechanics under one fixed target policy; NEVER launch solver/GP."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from calibration.analysis.migrate_run6_edge_frame import ROOT,SNAP,TARGET,CONFIG,HISTORY,sha,read,save,verify
import numpy as np
from scipy.spatial import cKDTree
from calibration.ewcal.geometry import read_mesh,points,reference_dual_areas
from calibration.ewcal.material_frame import score_material_frame
from calibration.ewcal.runner import Evaluator,validate_mesh_topology


def validate_saved(mesh,state,sequence):
    if state['schema_version']!=1 or state['units'] not in ('m','metres'):
        raise ValueError('Invalid native schema/units')
    ref=np.asarray(state['reference_positions'],float)*1000
    faces=np.asarray(state['faces'])
    if ref.shape!=points(mesh).shape or not np.isfinite(ref).all():raise ValueError('Invalid reference positions')
    if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer):raise ValueError('Invalid faces')
    if faces.size==0 or faces.min()<0 or faces.max()>=len(ref):raise ValueError('Invalid face indices')
    validate_mesh_topology(mesh,faces)
    native=np.asarray(state['positions'],float)*1000;p=points(mesh)
    if native.shape!=p.shape or not np.isfinite(native).all():raise ValueError('Invalid native deformed positions')
    for a,b in [(0,1),(1,2),(2,0)]:
        if np.max(np.abs(np.linalg.norm(native[faces[:,a]]-native[faces[:,b]],axis=1)-
                         np.linalg.norm(p[faces[:,a]]-p[faces[:,b]],axis=1)))>1e-3:
            raise ValueError('Native/mesh geometry or unit mismatch')
    areas=reference_dual_areas(ref,faces)
    if not np.isfinite(areas).all() or np.any(areas<=0) or abs(areas.sum()-60000)>600:
        raise ValueError('Invalid native reference area')
    if (sequence['schema_version']!=1 or sequence['completed'] is not True or sequence['cycles']!=29 or
        sequence['backend']!='native_hybrid' or sequence['equilibrium_solves']!=1 or len(sequence['solves'])!=1):
        raise ValueError('Incomplete/wrong native solve')
    s=sequence['solves'][0];native=s['native_hybrid']
    if (s['cycle']!=29 or native['accepted'] is not True or native['termination']!='gradient_target' or
        native['curvature']!='positive_pivots' or not np.isfinite(native['gradient_norm']) or
        not 0<=native['gradient_norm']<=5e-14 or s.get('return_code',0)!=0):
        raise ValueError('Native solve not numerically accepted')
    return ref,faces,areas/areas.sum()


def load(entry):
    mesh=read_mesh(BASE/entry['mesh_file'],scale=entry['scale_to_mm'])
    ref,faces,weights=validate_saved(mesh,read(BASE/entry['state_file']),read(BASE/entry['sequence_file']))
    return mesh,ref,faces,weights


def current_bindings(config):
    identity,targets=Evaluator.identity_inputs(config)  # Read-only API, not constructor/evaluate.
    eid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity,targets['run6'],eid


def check_bindings():
    manifest=verify();binding=read(ROOT/'rescore_bindings.json')
    for p,h in binding['input_sha256'].items():
        if sha(BASE/p)!=h:raise RuntimeError('Reviewed migration drift: '+p)
    return manifest,binding


def preflight():
    m=verify();config=read(BASE/CONFIG)
    if config['output_dir']!='run/run6_bo_edge_aligned' or config['report_dir']!='run/run6_bo_edge_aligned/report':
        raise ValueError('Nonisolated target variant')
    prior=read(SNAP/'calibration/configs/run6_bo_material_frame.json')
    normalized=copy.deepcopy(config);normalized['cases']=prior['cases']
    normalized['output_dir']=prior['output_dir'];normalized['report_dir']=prior['report_dir']
    if normalized!=prior:raise ValueError('Unapproved scientific recipe change')
    identity,new_target,eid=current_bindings(config)
    with np.load(BASE/'calibration/data/processed/run6_oriented.npz',allow_pickle=False) as f:old_target=f['points_mm'].copy()
    rows=[]
    for attempt in [2,25,41]:
        entry=next(r for r in m['sources'] if r['attempt_id']==attempt)
        mesh,ref,faces,weights=load(entry)
        old,*_=score_material_frame(mesh,old_target,ref,faces)
        difference=abs(old['rms_symmetric_mm']-entry['old_rms_mm'])
        if difference>1e-6:raise ValueError(f'Old-score fidelity mismatch attempt{attempt}: {difference}')
        new,*_=score_material_frame(mesh,new_target,ref,faces)
        rows.append({'attempt_id':attempt,'old_rms_mm':entry['old_rms_mm'],'old_replay_rms_mm':old['rms_symmetric_mm'],
                     'absolute_fidelity_error_mm':difference,'new_fixture_rms_mm':new['rms_symmetric_mm']})
    pins=dict(m['original_input_sha256'])
    extra=[CONFIG,TARGET,str(Path(TARGET).with_suffix('.json')),
           'calibration/analysis/migrate_run6_edge_frame.py','calibration/analysis/rescore_run6_edge_history.py',
           'calibration/tests/test_edge_history_migration.py',str((ROOT/'manifest.json').relative_to(BASE)),
           str((ROOT/'edge_alignment_complete.json').relative_to(BASE)),str((ROOT/'config_lineage.json').relative_to(BASE)),
           config['solver']['executable']]
    extra+= [str(p.relative_to(BASE)) for p in (BASE/'calibration/ewcal').glob('*.py')]
    for p in extra:
        actual=sha(BASE/p)
        if p in pins and pins[p]!=actual:raise RuntimeError('Protected scientific source drift: '+p)
        pins[p]=actual
    original_exec=read(SNAP/'run/run6_bo_material_frame/corrected_history.json')['identity_inputs']['executable']
    if identity['executable']!=original_exec:raise ValueError('Solver executable changed from frozen lineage')
    verify()
    save(ROOT/'preflight.json',{'status':'passed','old_score_fixtures':rows,'identity':eid,
                              'python':sys.version,'numpy':np.__version__,'slurm_job':os.environ.get('SLURM_JOB_ID')})
    save(ROOT/'rescore_bindings.json',{'input_sha256':pins,'identity_inputs':identity,'evaluator_identity':eid,
                                     'history_count':80,'completed_count':70,'config':config})
    print(json.dumps(rows,indent=2));print('Bound',len(pins),'artifacts; identity',eid)


def migrated_observation(original,metrics,identity,folder):
    row=copy.deepcopy(original)
    row['prior_objective_mm']=row.get('objective_mm')
    row['identity']=identity
    row['source_directory']=row.get('directory')
    if not row['valid']:
        if row['failure_class']!='timeout':raise ValueError('Unverified physical failure classification')
        row['objective_mm']=None  # Historical operational1e6 sentinel never enters loss GP.
        row['training_only']=True
        return row
    if metrics is None or not np.isfinite(metrics['rms_symmetric_mm']):raise ValueError('Missing/nonfinite replay score')
    row.update(objective_mm=metrics['rms_symmetric_mm'],metrics=metrics,scoring_source='fixed_scan_edge_frame',
               training_only=True,rescored_artifact_directory=str(folder))
    for case in row.get('cases',[]):
        case['prior_metrics']=case.get('metrics');case['metrics']=metrics;case['case_rms']=metrics['rms_symmetric_mm']
    return row


def rescore():
    m,binding=check_bindings();config=binding['config']
    identity,target,eid=current_bindings(config)
    if identity!=binding['identity_inputs'] or eid!=binding['evaluator_identity']:raise RuntimeError('Runtime identity drift')
    out=BASE/config['output_dir']
    out.mkdir(exist_ok=False)
    save(out/'migration_started.json',{'bindings_sha256':sha(ROOT/'rescore_bindings.json'),'slurm_job':os.environ.get('SLURM_JOB_ID')})
    originals={r['attempt_id']:r for r in m['observations']};results={};table=[];start=time.monotonic()
    for entry in m['sources']:
        # Already pinned paths are reverified per case; final all-pin verification follows.
        for key in ['mesh_file','state_file','sequence_file']:
            p=entry[key]
            if sha(BASE/p)!=binding['input_sha256'][p]:raise RuntimeError('Saved shape drift: '+p)
        mesh,ref,faces,weights=load(entry)
        metrics,framed,scan,forward,closest=score_material_frame(mesh,target,ref,faces)
        if not np.isfinite(metrics['rms_symmetric_mm']):raise ValueError('Nonfinite accepted shape score')
        q=points(framed);reverse,_=cKDTree(scan).query(q)
        folder=out/'rescored_shapes'/f'attempt_{entry["attempt_id"]:04d}';folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'comparison.npz',target_mm=scan,prediction_mm=q,faces=faces,
                            distance_mm=forward,closest_mm=closest,reverse_distance_mm=reverse,reference_area_weights=weights)
        row=migrated_observation(originals[entry['attempt_id']],metrics,eid,folder)
        results[entry['attempt_id']]=row
        save(folder/'result.json',{'attempt_id':entry['attempt_id'],'parameters':entry['parameters'],'metrics':metrics,
                                 'source':entry,'evaluator_identity':eid})
        table.append({'attempt_id':entry['attempt_id'],'old_rms_mm':entry['old_rms_mm'],
                      'new_rms_mm':metrics['rms_symmetric_mm'],'difference_mm':metrics['rms_symmetric_mm']-entry['old_rms_mm']})
        print('RESCORE',entry['attempt_id'],metrics['rms_symmetric_mm'],flush=True)
    observations=[results.get(r['attempt_id']) or migrated_observation(r,None,eid,None) for r in m['observations']]
    assert [r['attempt_id'] for r in observations]==list(range(1,81)) and len(results)==70
    assert all(r['normalized_parameters']==originals[r['attempt_id']]['normalized_parameters'] and
               r['parameters']==originals[r['attempt_id']]['parameters'] for r in observations)
    assert all(r['objective_mm'] is None for r in observations if not r['valid'])
    check_bindings()
    history={'schema_version':1,'config':config,'evaluator_identity':eid,'identity_inputs':identity,
             'input_sha256':binding['input_sha256'],'observations':observations,
             'source_manifest':str((ROOT/'manifest.json').relative_to(BASE)),
             'scoring_method':'material_reference','target_policy':'fixed_geometric_scan_frame_then_crop_bin',
             'no_new_mechanics_or_gp':True,'not_a_resumed_checkpoint':True}
    save(BASE/HISTORY,history)
    best=min(results.values(),key=lambda r:r['objective_mm'])
    summary={'status':'complete','attempt_count':80,'scored_count':70,'timeout_ids':m['timeout_ids'],
             'best_attempt_id':best['attempt_id'],'best_rms_mm':best['objective_mm'],'best_parameters':best['parameters'],
             'evaluator_identity':eid,'history_sha256':sha(BASE/HISTORY),'score_table':table,
             'seconds':time.monotonic()-start,'slurm_job':os.environ.get('SLURM_JOB_ID'),
             'interpretation':'Old/new target-policy comparison, not same-policy optimization improvement; one calibration specimen.',
             'scope':'No solver, GP, new BO, run9 modification or old campaign overwrite.'}
    save(out/'migration_summary.json',summary)
    save(out/'best.json',{'attempt_id':best['attempt_id'],'parameters':best['parameters'],'objective_mm':best['objective_mm'],
                         'metrics':best['metrics'],'evaluator_identity':eid,'source':'rescored_existing_shapes'})
    print(json.dumps({k:v for k,v in summary.items() if k!='score_table'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','rescore']);args=parser.parse_args()
    preflight() if args.stage=='preflight' else rescore()
