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

from dome_bo_repeat_domain import compatible, coordinates, sequence
from prepare_dome_baseline import ROOT

PYTHON = '/gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python'
DHD = '/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python'
MAX_NEW = 100
INITIAL_DESIGN = ROOT/'run/dome/checks/stage2_lhs.json'


def save_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')


def warm_start(base):
    records = []
    for name in ('baseline', 'middle_n18_repeat2', 'middle_n18_length168', 'middle_n18_sides_n14', 'middle_length196'):
        run = ROOT/f'run/dome/{name}_v1'
        config = json.loads((run/'sequence.json').read_text())
        if not compatible(base, config):
            raise ValueError(f'Incompatible seed {name}')
        execution = json.loads((run/'execution.json').read_text())
        rec = {'parameters':coordinates(config), 'source':str(run),
               'sequence_sha256':hashlib.sha256((run/'sequence.json').read_bytes()).hexdigest(),
               'status':execution['status']}
        if rec['status'] == 'accepted':
            metrics = json.loads((ROOT/f'run/dome/{name}_analysis_v1/metrics.json').read_text())
            rec['coverage'] = max(metrics['geometry']['sign_fractions'].values())
        elif name != 'middle_length196' or rec['status'] != 'failed':
            raise ValueError('Unexpected retained failure')
        records.append(rec)
    for observation in sorted((ROOT/'run/dome/bo_stage1').glob('attempt_*/observation.json')):
        old = json.loads(observation.read_text())
        config_path = observation.parent/'sequence.json'
        config = json.loads(config_path.read_text())
        if not compatible(base,config) or old['status'] not in ('accepted','failed'):
            raise ValueError('Incompatible stage1 warm start')
        rec = dict(old, parameters=coordinates(config),
                   observation_sha256=hashlib.sha256(observation.read_bytes()).hexdigest(),
                   sequence_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        records.append(rec)
    if len(records)!=53:
        raise ValueError('Expected five original plus48 stage1 records')
    return records


def evaluate(attempt):
    # Every attempted point is reserved on disk before a subprocess can start.
    command = ['srun','--exclusive','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=32G',
               PYTHON, str(ROOT/'legacy/scripts/run_dome.py'), '--sequence',str(attempt/'sequence.json'),
               '--output',str(attempt/'solve')]
    with (attempt/'step.log').open('x') as log:
        proc = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1700)
    if proc.returncode:
        solver_log = attempt/'solve/solver.log'
        text = solver_log.read_text() if solver_log.exists() else ''
        if 'equilibrium corrector failed' not in text:
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
    paths = [ROOT/'bin/shell', ROOT/'legacy/scripts/run_dome.py', ROOT/'legacy/scripts/dome_repeat_campaign.py',
             ROOT/'legacy/scripts/dome_repeat_propose.py', ROOT/'legacy/scripts/dome_bo_repeat_domain.py',
             ROOT/'legacy/scripts/dome_bo_domain.py', ROOT/'legacy/scripts/dome_repeat_lhs.py', INITIAL_DESIGN,
             ROOT/'legacy/scripts/prepare_dome_baseline.py', ROOT/'legacy/analysis/analyze_dome.py',
             ROOT/'dome_bo/dome_curvature.py',
             ROOT/'legacy/configs/dome/baseline_attempt209.json']
    identities = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save_new(output/'identities.json',identities)
    failures = 0
    for wave in range(MAX_NEW//2):
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in identities.items()):
            raise RuntimeError('Campaign source identity changed; stop without retry')
        if time.time()+2700 >= float(os.environ.get('DOME_BO_DEADLINE','inf')):
            save_new(output/'stop.json',{'reason':'allocation_reserve'}); return
        request = output/f'proposal_{wave:02d}.json'
        save_new(request,{'records':records,'seed':20261019+wave,'initial':wave<12})
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
    save_new(output/'stop.json',{'reason':'100_new_evaluations_exhausted'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['propose','run'])
    parser.add_argument('path',type=Path)
    parser.add_argument('result',nargs='?',type=Path)
    args = parser.parse_args()
    if args.mode == 'propose':
        from dome_repeat_propose import propose
        request = json.loads(args.path.read_text())
        base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
        save_new(args.result,propose(base,request['records'],request['seed'],initial=request['initial']))
    else:
        campaign(args.path.resolve())
