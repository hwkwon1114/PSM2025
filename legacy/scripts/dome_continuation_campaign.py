#!/usr/bin/env python3
"""Dome BO stage: durable attempts, two Slurm steps, no solver retries."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dome_offset_domain import compatible, coordinates, sequence
from prepare_dome_baseline import ROOT

PYTHON = '/gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python'
DHD = '/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python'
MAX_NEW = 64
INITIAL_DESIGN = ROOT/'run/dome/checks/continuation_lhs.json'


def save_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')


def warm_start(base):
    sys.path.insert(0, str(ROOT / 'dome_bo'))
    from run_dome_continuation import verify
    run = ROOT/'run/dome/repair_attempt028_ramp60_v1'
    config = json.loads((run/'sequence.json').read_text())
    if not compatible(base,config):
        raise ValueError('Incompatible continuation seed')
    verify(run,sum(p['repeat'] for p in config['toolpaths']))
    receipt=json.loads((run/'diagnostic_execution.json').read_text())
    command=receipt['command']
    expected={'-sequence_adaptive':'true','-sequence_initial_step':'0.1',
              '-sequence_max_step':'0.1','-native_hybrid_refine_linear_solve':'true',
              '-native_hybrid_warmup_attempts':'2000','-native_hybrid_seconds':'1800'}
    if receipt['returncode']!=0 or any(command[command.index(k)+1]!=v for k,v in expected.items()):
        raise ValueError('Seed recipe mismatch')
    binary=ROOT/'run/native_shape_refinement/build_4375711/bin/shell'
    if receipt['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest():
        raise ValueError('Seed binary mismatch')
    metrics=json.loads((ROOT/'run/dome/repair_attempt028_ramp60_analysis_v1/metrics.json').read_text())
    state=run/'bilayer_zigzag_sequence_native_hybrid_state.json'
    if metrics['native_state_sha256']!=hashlib.sha256(state.read_bytes()).hexdigest():
        raise ValueError('Seed metrics identity mismatch')
    return [dict(parameters=coordinates(config),source=str(run),status='accepted',
                 coverage=max(metrics['geometry']['sign_fractions'].values()),
                 sequence_sha256=hashlib.sha256((run/'sequence.json').read_bytes()).hexdigest())]


def evaluate(attempt):
    # Every attempted point is reserved on disk before a subprocess can start.
    command = ['srun','--exclusive','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=32G',
               PYTHON, str(ROOT/'dome_bo/run_dome_continuation.py'), '--sequence',str(attempt/'sequence.json'),
               '--output',str(attempt/'solve')]
    with (attempt/'step.log').open('x') as log:
        proc = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=3900)
    if proc.returncode:
        solver_log = attempt/'solve/solver.log'
        text = solver_log.read_text() if solver_log.exists() else ''
        receipt_path=attempt/'solve/execution.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        timed_out=receipt.get('failure_type')=='TimeoutExpired'
        if 'equilibrium corrector failed' not in text and not timed_out:
            raise RuntimeError(f'Infrastructure/unclassified failure in {attempt}; no automatic retry')
        return {'status':'failed','reason':'native_equilibrium_not_accepted','returncode':proc.returncode}
    command = ['srun','--exclusive','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G',
               PYTHON,str(ROOT/'legacy/analysis/analyze_dome.py'),str(attempt/'solve'),str(attempt/'analysis')]
    with (attempt/'analysis.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
    metrics = json.loads((attempt/'analysis/metrics.json').read_text())
    return {'status':'accepted','coverage':max(metrics['geometry']['sign_fractions'].values()),
            'screening_pass':metrics['geometry']['qualifies'], 'analysis_status':metrics['status']}


def campaign(output):
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0')) < 16:
        raise ValueError('Requires a 16CPU/64GB batch allocation')
    base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
    records = warm_start(base)
    initial = json.loads(INITIAL_DESIGN.read_text())
    points = initial['parameters']
    if len(points)!=24 or len({tuple(p) for p in points})!=24:
        raise ValueError('Expected24 distinct frozen initial samples')
    for point in points:
        sequence(base,point)
        if any(point==r['parameters'] for r in records):
            raise ValueError('Initial design duplicates retained observation')
    output.mkdir(parents=True,exist_ok=False)
    save_new(output/'warm_start.json',records)
    paths = [ROOT/'run/native_shape_refinement/build_4375711/bin/shell', ROOT/'dome_bo/run_dome_continuation.py', ROOT/'legacy/scripts/dome_continuation_campaign.py',
             ROOT/'legacy/scripts/dome_offset_propose.py', ROOT/'legacy/scripts/dome_offset_domain.py',
             ROOT/'legacy/scripts/dome_bo_repeat_domain.py', ROOT/'legacy/scripts/dome_bo_domain.py',
             ROOT/'legacy/scripts/dome_offset_lhs.py', INITIAL_DESIGN,
             ROOT/'legacy/scripts/prepare_dome_baseline.py', ROOT/'legacy/analysis/analyze_dome.py',
             ROOT/'dome_bo/dome_curvature.py',
             ROOT/'legacy/configs/dome/baseline_attempt209.json']
    identities = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save_new(output/'identities.json',identities)
    failures = 0
    for wave in range(MAX_NEW//2):
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in identities.items()):
            raise RuntimeError('Campaign source identity changed; stop without retry')
        if time.time()+4800 >= float(os.environ.get('DOME_BO_DEADLINE','inf')):
            save_new(output/'stop.json',{'reason':'allocation_reserve'}); return
        request = output/f'proposal_{wave:02d}.json'
        save_new(request,{'records':records,'seed':20261021+wave,'initial':wave<12})
        proposal_path = output/f'proposal_{wave:02d}_result.json'
        if wave<12:
            save_new(proposal_path,{'parameters':initial['parameters'][2*wave:2*wave+2],
                                   'seed':initial['seed'],'method':initial['method']})
        else:
            with (output/f'proposal_{wave:02d}.log').open('x') as log:
                subprocess.run([DHD,str(Path(__file__).resolve()),'propose',str(request),str(proposal_path)],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
        proposal = json.loads(proposal_path.read_text())
        points = proposal['parameters']
        if len(points) != 2 or points[0] == points[1] or any(p == r['parameters'] for p in points for r in records):
            raise ValueError('Proposal must contain two novel distinct configurations')
        attempts = []
        for j,params in enumerate(proposal['parameters']):
            attempt = output/f'attempt_{wave*2+j+1:03d}'
            attempt.mkdir(exist_ok=False)
            save_new(attempt/'reservation.json',{'parameters':params,'wave':wave,'seed':proposal['seed']})
            save_new(attempt/'sequence.json',sequence(base,params))
            attempts.append((attempt,params))
        # Always retain both outcomes, even if one reports infrastructure failure.
        errors = []
        wave_records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(evaluate,a):(a,p) for a,p in attempts}
            for future in concurrent.futures.as_completed(pending):
                attempt,params = pending[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {'status':'error','reason':str(exc)}
                    errors.append(str(exc))
                result.update(parameters=params,source=str(attempt))
                save_new(attempt/'observation.json',result)
                wave_records.append(result)
        records.extend(sorted(wave_records,key=lambda r:r['source']))
        if errors:
            save_new(output/'stop.json',{'reason':'infrastructure_or_analysis_error','errors':errors}); return
        if any(r.get('screening_pass') for r in wave_records):
            save_new(output/'stop.json',{'reason':'screening_pass_requires_verification'}); return
        failures = failures+1 if all(r['status']=='failed' for r in wave_records) else 0
        if failures == 2:
            save_new(output/'stop.json',{'reason':'two_full_waves_failed_equilibrium'}); return
    save_new(output/'stop.json',{'reason':'64_continuation_evaluations_exhausted'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['propose','run'])
    parser.add_argument('path',type=Path)
    parser.add_argument('result',nargs='?',type=Path)
    args = parser.parse_args()
    if args.mode == 'propose':
        from dome_offset_propose import propose
        request = json.loads(args.path.read_text())
        base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
        save_new(args.result,propose(base,request['records'],request['seed'],initial=request['initial']))
    else:
        campaign(args.path.resolve())
