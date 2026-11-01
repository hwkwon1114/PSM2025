#!/usr/bin/env python3
"""Fresh matched pairwise calibrations, isolated from the running Pareto controller.

Reuse its scalar design, codec, evaluator and acquisition without modifying the
pinned running source. Run only after the applicable independent review.
"""
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPEC_PATH = ROOT / 'calibration/bo/pairwise_floor1.json'
os.environ['JOINT_SPEC'] = str(SPEC_PATH)
from calibration.bo import joint_calibration as joint
from calibration.bo import run_convention_corrected as cc
from calibration.ewcal.runner import Evaluator

PAIRS = {'12': ('run6', 'run9'), '23': ('run9', 'run10'), '13': ('run6', 'run10')}
EXCLUDED = {'12': 'run10', '23': 'run6', '13': 'run9'}
BASE_ROOT = ROOT / 'run/pairwise_floor1'


def configure(pair):
    if pair not in PAIRS:
        raise ValueError(f'Unknown pair: {pair}')
    joint.BASE_REL = f'run/pairwise_floor1/pair_{pair}'
    joint.BASE = ROOT / joint.BASE_REL
    joint.joint_config = lambda values: pair_config(pair, values)
    return joint.BASE


def pair_config(pair, values):
    names = PAIRS[pair]
    configs = [cc.config(name) for name in names]
    first = configs[0]
    for other in configs[1:]:
        for key in set(first) | set(other):
            if key not in ('cases', 'output_dir', 'report_dir') and first.get(key) != other.get(key):
                raise ValueError(f'Pair settings differ: {key}')
    cfg = copy.deepcopy(first)
    cfg['cases'] = [copy.deepcopy(c['cases'][0]) for c in configs]
    if [c['id'] for c in cfg['cases']] != list(names) or any(c.get('weight') != 1 for c in cfg['cases']):
        raise ValueError('Wrong case IDs or non-unit weights')
    if cfg.get('lhs'):
        raise ValueError('LHS positivity gate would reject an exact zero objective')
    cfg['hardening']['beta'] = float(values['beta'])
    cfg['hardening']['floor'] = float(values['floor'])
    return cfg


def plan_paths(pair):
    paths = [Path(__file__), SPEC_PATH, ROOT / 'calibration/bo/joint_calibration.py',
             ROOT / 'calibration/bo/run_convention_corrected.py',
             ROOT / 'calibration/bo/run_pairwise_floor1.sbatch',
             ROOT / 'calibration/tests/test_pairwise_floor1.py',
             ROOT / 'shellsim/configs/default_adaptive_cutback.json', ROOT / 'bin/shell']
    paths.extend((ROOT / 'calibration/ewcal').glob('*.py'))
    # Pin excluded-case inputs too, without scoring them during calibration.
    for case in (*PAIRS[pair], EXCLUDED[pair]):
        target = ROOT / f'run/putong_convention_check/preserved_samples/{case}.npz'
        paths.extend((ROOT / cc.REQUESTS[case], target, target.with_suffix('.json')))
    return sorted({str(p.absolute()) for p in paths})


def prepare(pair):
    base = configure(pair)
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f'Output already exists: {base}')
    cfg = pair_config(pair, joint.decode([0.5] * joint.D))
    identity, _ = Evaluator.identity_inputs(cfg)
    transfer_identity, _ = Evaluator.identity_inputs(cc.config(EXCLUDED[pair]))
    plan = {'pair': pair, 'cases': list(PAIRS[pair]), 'evaluator_identity': identity,
            'transfer_evaluator_identity': transfer_identity,
            'versions': joint.software_versions(), 'files': {p: joint.sha256(p) for p in plan_paths(pair)}}
    base.mkdir(parents=True, exist_ok=True)
    joint.save_json(base / 'plan.json', plan)
    print(f'Prepared {base / "plan.json"}; no solver launched', flush=True)


def verify_plan(pair):
    plan_path = joint.BASE / 'plan.json'
    expected_plan = os.environ.get('PAIR_PLAN_SHA256')
    if not expected_plan or joint.sha256(plan_path) != expected_plan:
        raise RuntimeError('Reviewed plan hash missing or changed')
    plan = joint.jread(plan_path)
    environment = {k: os.environ.get(k, '') for k in ('CONDA_PREFIX', 'LD_LIBRARY_PATH')}
    if (plan['pair'] != pair or plan['cases'] != list(PAIRS[pair]) or plan['versions'] != joint.software_versions()
            or plan['evaluator_identity']['environment'] != environment):
        raise RuntimeError('Pair, software or runtime environment identity changed')
    for path, expected in plan['files'].items():
        if joint.sha256(path) != expected:
            raise RuntimeError(f'Reviewed input changed: {path}')
    if set(plan['files']) != set(plan_paths(pair)):
        raise RuntimeError('Reviewed input list changed')
    return plan


def validate_results(pair, res):
    for attempt, record in res.items():
        proposal = joint.jread(joint.BASE / 'proposals' / f'a{attempt:04d}.json')
        if (record['attempt'] != attempt or record['wave'] != proposal['wave'] or
                record['x'] != proposal['x'] or record['parameters'] != proposal['parameters'] or
                record['parameters'] != joint.decode(record['x'])):
            raise ValueError(f'Attempt {attempt} differs from its proposal or physical parameters')
        if record['valid']:
            case = record['case_rms_mm']
            if set(case) != set(PAIRS[pair]):
                raise ValueError(f'Attempt {attempt} has wrong cases')
            errors = [float(case[k]) for k in PAIRS[pair]]
            if not all(math.isfinite(e) and e >= 0 for e in errors):
                raise ValueError(f'Attempt {attempt} has invalid case errors')
            expected = math.sqrt(sum(e * e for e in errors) / 2)
            if not math.isclose(float(record['objective_mm']), expected, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError(f'Attempt {attempt} has wrong joint objective')
        elif record.get('failure_class') not in ('solver', 'registration', 'timeout', 'infrastructure'):
            raise ValueError(f'Attempt {attempt} has unknown failure')
    return res


def finish(pair, status):
    """Freeze from training results, then attempt the excluded case once."""
    selection_path = joint.BASE / 'selection.json'
    if selection_path.exists() or (joint.BASE / 'transfer').exists():
        raise FileExistsError('Selection or transfer already exists; no automatic rerun')
    joint.summary(status)
    res = validate_results(pair, joint.results())
    valid = [r for r in res.values() if r['valid']]
    if not valid:
        joint.save_json(joint.BASE / 'transfer_unavailable.json', {'reason': 'no_valid_pairwise_candidate'})
        return
    best = min(valid, key=lambda r: (r['objective_mm'], r['attempt']))
    joint.save_json(selection_path, {'pair': pair, 'training_cases': list(PAIRS[pair]),
                                    'excluded_case': EXCLUDED[pair], 'selection_rule': 'lowest_valid_joint_rms_then_attempt',
                                    'attempt': best['attempt'], 'parameters': best['parameters'],
                                    'objective_mm': best['objective_mm'], 'case_rms_mm': best['case_rms_mm'],
                                    'result_sha256': joint.sha256(joint.BASE / 'results' / f"a{best['attempt']:04d}.json"),
                                    'plan_sha256': joint.sha256(joint.BASE / 'plan.json')})
    if int(os.environ['SLURM_JOB_END_TIME']) - time.time() < 3600 + 300:
        joint.save_json(joint.BASE / 'transfer_unavailable.json', {'reason': 'insufficient_allocation_time'})
        return
    verify_plan(pair)
    cfg = cc.config(EXCLUDED[pair])
    pair_cfg = pair_config(pair, best['parameters'])
    for key in set(cfg) | set(pair_cfg):
        if key not in ('cases', 'output_dir', 'report_dir', 'hardening') and cfg.get(key) != pair_cfg.get(key):
            raise ValueError(f'Transfer settings differ: {key}')
    cfg['hardening'] = copy.deepcopy(pair_cfg['hardening'])
    cfg['output_dir'] = joint.BASE_REL + '/transfer'
    cfg['report_dir'] = cfg['output_dir'] + '/report'
    # The controller uses four threads for BO; the solver uses eight like paired cases.
    os.environ['OMP_NUM_THREADS'] = '8'
    check_solver_environment()
    ev = Evaluator(cfg)
    params = {k: best['parameters'][k] for k in cfg['parameters']}
    ev.parameters = SimpleNamespace(decode=lambda _x: dict(params))
    rec = ev.evaluate([0.0] * len(params), use_cache=False, attempt_id=1)
    joint.save_json(joint.BASE / 'transfer_result.json', rec)
    print(f'TRANSFER pair={pair} excluded={EXCLUDED[pair]} valid={rec["valid"]}', flush=True)


def run(pair):
    configure(pair)
    verify_plan(pair)
    # An interrupted or previously completed job requires a deliberate reviewed continuation.
    if any((joint.BASE / p).exists() for p in ('proposals', 'results', 'logs', 'summary.json')):
        raise FileExistsError('Existing campaign artifacts; automatic restart is forbidden')
    for d in ('proposals', 'results', 'logs'):
        (joint.BASE / d).mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', BLIS_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    env.pop('PYTHONOPTIMIZE', None)
    for key in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT'):
        env.pop(key, None)
    for wave in range(1, joint.WAVES + 1):
        verify_plan(pair)
        # Reserve enough time for a paired batch and one frozen transfer case.
        if int(os.environ['SLURM_JOB_END_TIME']) - time.time() < 3 * 3600 + 300:
            finish(pair, 'allocation_limit')
            return
        res = validate_results(pair, joint.results())
        if joint.failure_streak(res) >= joint.MAX_CONSEC_FAIL:
            finish(pair, 'stopped_consecutive_failures')
            return
        ids = joint.make_wave(res, wave)
        if int(os.environ['SLURM_JOB_END_TIME']) - time.time() < 3 * 3600 + 300:
            finish(pair, 'allocation_limit')
            return
        procs = []
        for attempt in ids:
            log = open(joint.BASE / 'logs' / f'a{attempt:04d}.log', 'ab')
            try:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), pair, 'evaluate', str(attempt)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            finally:
                log.close()
            procs.append((attempt, process))
        for _, process in procs:
            process.wait()
        res = validate_results(pair, joint.results())
        missing = [a for a in ids if a not in res]
        if missing or any(res[a]['failure_class'] == 'infrastructure' for a in ids):
            raise RuntimeError(f'Wave {wave}: infrastructure/missing results {missing}; stopping for review')
        if joint.failure_streak(res) >= joint.MAX_CONSEC_FAIL:
            finish(pair, 'stopped_consecutive_failures')
            return
        # Zero is mathematically optimal; do not feed log(0) to the GP.
        if any(r['valid'] and r['objective_mm'] == 0 for r in res.values()):
            finish(pair, 'exact_zero_optimum')
            return
        if joint.stalled(res, wave):
            finish(pair, 'early_stop_no_improvement')
            return
        joint.summary('running')
    finish(pair, 'budget_complete')


def check_solver_environment():
    fixed = {'OMP_NUM_THREADS': '8', 'BLIS_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    if any(os.environ.get(k) != v for k, v in fixed.items()) or any(
            k in os.environ for k in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT')):
        raise RuntimeError('Solver thread environment differs from the reviewed settings')


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in PAIRS:
        raise SystemExit('Usage: pairwise_floor1.py {12,23,13} {prepare,run,evaluate} [attempt]')
    pair, action = sys.argv[1:3]
    if action == 'prepare':
        prepare(pair)
    elif action == 'run':
        run(pair)
    elif action == 'evaluate' and len(sys.argv) == 4:
        configure(pair)
        verify_plan(pair)
        check_solver_environment()
        joint.evaluate(int(sys.argv[3]))
    elif action == 'verify' and len(sys.argv) == 3:
        configure(pair)
        verify_plan(pair)
        print(f'Plan verified for pair {pair}; no solver launched', flush=True)
    else:
        raise SystemExit('Invalid action')


if __name__ == '__main__':
    main()
