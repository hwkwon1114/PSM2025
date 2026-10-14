#!/usr/bin/env python
"""Expanded-bounds success/separation-aware BO on Run 6; warm-started from 120 historical evaluations."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_run6_bo_refinement as controller
from calibration.ewcal import batch_bayes as batch
from calibration.ewcal.common import Parameters, save_json
from calibration.ewcal.runner import Evaluator
from calibration.ewcal.progress import message

BASE = ROOT / 'run/run6_expanded_bo'
CONFIG = ROOT / 'calibration/configs/run6.json'
SOURCE = ROOT / 'run/run6/bayes_state.json'
SOURCE_HIST = ROOT / 'run/run6/corrected_history.json'
REVIEW = ROOT / 'run/optimization_review/run6_expanded_bo'
HISTORY = BASE / 'corrected_history.json'
META = {
    'base_seed': 2026,
    'history_count': 120,
    'new_budget': 40,
    'q': 4,
    'waves': 10,
    'scoring_method': 'material_reference',
    'epoch_start_attempt': 121,
    'proposal_method': 'success_aware_separation',
    'target_policy': 'expanded_edge',
}
sha, read, write_new = controller.sha, controller.read, controller.write_new


def configure(base=None, config=None, history=None):
    controller.CONFIG = config or CONFIG
    controller.BASE = base or BASE
    controller.HISTORY = history or HISTORY
    controller.META, controller.CAMPAIGN_TYPE = META, 'expanded_edge_success_separation'


def select_best(observations):
    valid = [o for o in observations if o.get('valid')]
    if not valid:
        return None
    return min(valid, key=lambda r: (r['objective_mm'], r['attempt_id']))


def prepare():
    if HISTORY.exists() or (BASE / 'bayes_state.json').exists():
        raise FileExistsError('Preserve existing expanded BO history/state')
    source = read(SOURCE)
    source_hist = read(SOURCE_HIST)
    cfg = read(CONFIG)

    old_params = Parameters(source_hist['config'])
    new_params = Parameters(cfg)

    # Re-encode 120 historical observations into the expanded bounding box [0, 1]^4
    records = []
    for r in sorted(source['observations'], key=lambda x: x['attempt_id']):
        rec = copy.deepcopy(r)
        phys = old_params.decode(rec['normalized_parameters'])
        rec['normalized_parameters'] = new_params.encode(phys).tolist()
        batch.validate_observation_record(rec)
        controller.validate_material_science(rec)
        records.append(rec)

    if [r['attempt_id'] for r in records] != list(range(1, 121)) or sum(r['valid'] for r in records) != 108:
        raise ValueError('Need all 120 historical observations, 108 valid losses')

    names = [
        SOURCE,
        CONFIG,
        Path(__file__),
        ROOT / 'scripts/run_run6_expanded_bo.sbatch',
        ROOT / 'scripts/run_run6_bo_refinement.py',
        ROOT / 'calibration/data/processed/run6_edge_aligned_refined.npz',
        ROOT / 'calibration/data/processed/run6_edge_aligned_refined.json',
        ROOT / cfg['solver']['executable'],
        ROOT / 'calibration/analysis/success_aware_bo/model.py',
        ROOT / 'calibration/analysis/success_aware_bo/separation.py',
    ]
    names += list((ROOT / 'calibration/ewcal').glob('*.py'))

    payload = {
        'schema_version': 1,
        'metadata': META,
        'config': cfg,
        'evaluator_identity': Evaluator.identity_for_config(cfg),
        'observations': copy.deepcopy(records),
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in names},
        'limits': 'Seed payload warm-starts 120 Run 6 observations into expanded parameter space [5e-5, 1e-2] x [1e-5, 1e-3].',
    }
    BASE.mkdir(parents=True, exist_ok=True)
    write_new(HISTORY, payload)
    print('HISTORY', sha(HISTORY), flush=True)


def check_review(expected):
    packet_path = REVIEW / 'review_packet.json'
    if sha(packet_path) != expected:
        raise ValueError('Reviewed packet digest mismatch')
    packet = read(packet_path)
    binding = read(REVIEW / 'launch_binding.json')
    if (
        binding['packet_sha256'] != expected
        or binding['report_sha256'] != sha(REVIEW / binding['report'])
        or binding['verdict'] not in ('APPROVED', 'APPROVED_WITH_CAVEATS')
    ):
        raise ValueError('Missing/mismatched independent clearance')
    for name, digest in packet['artifact_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Reviewed artifact changed: {name}')


def proposal(records, seed):
    import numpy as np
    import torch
    from calibration.analysis.success_aware_bo.model import fit_models
    from calibration.analysis.success_aware_bo.separation import propose, admissible

    torch.manual_seed(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        loss, success, best, info = fit_models(records)
        diag = propose(records, loss, success, best, seed=seed)
    diag.update(training=info, fit_seed=seed, outer_warnings=[str(w.message) for w in caught])
    pts = np.asarray(diag['final'])
    if not admissible(pts, [r['normalized_parameters'] for r in records]) or diag['repaired_indices']:
        raise ValueError('Proposal violates unchanged separation/no-random-repair policy')
    return pts, diag


def bounded_proposal(records, seed):
    """Hard child-process cap covers fitting, initialization and optimization."""
    import numpy as np

    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'propose', '--seed', str(seed)]
    log = BASE / f'proposal_{seed}.log'
    with log.open('a') as errors:
        result = subprocess.run(
            command,
            input=json.dumps(records, allow_nan=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=errors,
            timeout=1500,
            check=True,
            cwd=ROOT,
        )
    diag = json.loads(result.stdout)
    return np.asarray(diag['final']), diag


def production_evaluator(cfg):
    """Keep frozen scientific identity but publish exclusively under this epoch's BASE."""
    import calibration.ewcal.runner as original

    spec = importlib.util.spec_from_file_location('calibration.ewcal.edge_runner', original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = module.launch

    def capped_launch(args, cwd, timeout, *extra, **kwargs):
        try:
            return launch(args, cwd, timeout, *extra, **kwargs)
        except module.SolverProcessFailure as error:
            evidence = error.evidence
            if (
                evidence.get('reason') == 'native_time_cap_abort'
                and evidence.get('timeout') is True
                and evidence.get('numerical') is True
            ):
                values = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '-native_hybrid_seconds']
                if timeout != 3600 or len(values) != 1 or float(values[0]) != timeout:
                    raise ValueError('Native/outer mechanics cap mismatch') from error
                evidence['limit_seconds'] = timeout
                module.save_json(cwd / 'termination.json', evidence)
            raise

    module.launch = capped_launch

    class PublicationEvaluator(module.Evaluator):
        @staticmethod
        def identity_inputs(storage_config):
            # Storage location is not a scientific change to the frozen config.
            return module.Evaluator.identity_inputs(cfg)

        def __init__(self):
            storage_config = copy.deepcopy(cfg)
            storage_config['output_dir'] = str(BASE)
            # Redirect before the parent constructor creates/writes its cache.
            super().__init__(storage_config)
            self.config = cfg

    return PublicationEvaluator()


@contextmanager
def durable_stop():
    with batch.ExclusiveLock(BASE / '.campaign.lock'):
        marker = BASE / 'controller_failure.json'
        if marker.exists():
            raise RuntimeError('Campaign stopped by preserved controller_failure.json; no relaunch')
        phase = {'name': 'preflight'}
        try:
            yield phase
        except Exception as error:
            failure = {
                'status': 'stopped_controller_failure',
                'phase': phase['name'],
                'exception_type': type(error).__name__,
                'reason': str(error),
                'traceback': traceback.format_exc(),
                'time': time.time(),
                'accounting': 'Preserved checkpoint/publications; no automatic reconciliation or restart.',
            }
            path = BASE / 'bayes_state.json'
            if path.exists():
                failure['checkpoint_sha256'] = sha(path)
                try:
                    saved, history = read(path), read(HISTORY)
                    controller.validate_ledger(saved, history['evaluator_identity'], 2026, history=history)
                    failure['accounting'] = {
                        'new_charged': saved['charged_attempts'] - META['history_count'],
                        'new_completed': len(saved['observations']) - META['history_count'],
                        'unresolved_attempts': [
                            r['attempt_id']
                            for r in saved['reserved_attempts']
                            if r['attempt_id'] not in saved['completed_attempts']
                        ],
                    }
                except Exception as accounting_error:
                    failure['accounting_error'] = str(accounting_error)
            write_new(marker, failure)
            raise


def run(expected):
    import torch

    configure()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.time()
    with durable_stop() as phase:
        check_review(expected)
        history, cfg = read(HISTORY), read(CONFIG)
        digest = sha(HISTORY)
        controller.verify_inputs(history)
        identity = Evaluator.identity_for_config(cfg)
        if identity != history['evaluator_identity'] or cfg != history['config'] or history['metadata'] != META:
            raise ValueError('Reviewed evaluator/config/history changed')
        phase['name'] = 'recovery'
        path = BASE / 'bayes_state.json'
        state = read(path) if path.exists() else controller.initial_state(history, digest)
        if state['history_sha256'] != digest:
            raise ValueError('Checkpoint history changed')
        state = controller.recover(state, history, identity)
        controller.update_stop(state)
        save_json(path, state)
        batch.sync_evaluations_log(BASE / 'evaluations.jsonl', state['observations'])
        if state['status'] in controller.TERMINAL:
            message('EXPANDED-BO', f'Terminal state {state["status"]}; zero launches')
            return
        evaluator = production_evaluator(cfg)
        validator = lambda s, i, seed, val: controller.validate_ledger(s, i, seed, val, history=history)
        for wave in range(state['waves_reserved'] + 1, META['waves'] + 1):
            phase['name'] = 'wave_input_deadline_check'
            check_review(expected)
            controller.verify_inputs(history)
            if batch.get_remaining_allocation_seconds(start, 43200) < 5700:
                state['status'] = 'deadline_stop'
                break
            next_id = controller.validate_ledger(state, identity, 2026, history=history)
            if len(state['observations']) != next_id - 1:
                raise ValueError('Unresolved reservations prohibit proposals')
            phase['name'] = 'proposal'
            seed = 2026 + META.get('proposal_seed_offset', 0) + wave
            pts, diag = bounded_proposal(state['observations'], seed)
            phase['name'] = 'postproposal_input_deadline_check'
            check_review(expected)
            controller.verify_inputs(history)
            if batch.get_remaining_allocation_seconds(start, 43200) < 3900:
                state['status'] = 'deadline_stop'
                break
            state['wave_diagnostics'].append({'wave_idx': wave, 'seed': seed, 'diagnostics': diag})
            reservations = []
            for idx, pt in enumerate(pts):
                att = next_id + idx
                folder = BASE / 'evaluations' / f'attempt_{att:04d}'
                r = {
                    'attempt_id': att,
                    'x': pt.tolist(),
                    'wave_idx': wave,
                    'proposal_idx': idx,
                    'evaluator_identity': identity,
                    'publication_dir': str(folder),
                    'result_file': str(folder / 'result.json'),
                    'status': 'reserved',
                    'created_at': time.time(),
                }
                state['reserved_attempts'].append(r)
                reservations.append((att, pt, r, idx))
            state['waves_reserved'], state['charged_attempts'] = wave, next_id + 3
            validator(state, identity, 2026, batch.validate_observation_record)
            save_json(path, state)
            message('EXPANDED-BO', f'Wave{wave}/{META["waves"]} attempts{next_id}–{next_id+3}; job={os.environ.get("SLURM_JOB_ID")}')
            phase['name'] = 'mechanics_publication'
            batch._execute_wave(
                evaluator,
                reservations,
                state,
                path,
                BASE / 'evaluations.jsonl',
                BASE,
                2026,
                threading.Lock(),
                ledger_validator=validator,
            )
            controller.update_stop(state)
            save_json(path, state)
            if state['status'] in controller.TERMINAL:
                break
        phase['name'] = 'final_verification'
        check_review(expected)
        controller.verify_inputs(history)
        validator(state, identity, 2026, batch.validate_observation_record)
        save_json(path, state)
        publish_summary(state, digest, expected)


def publish_summary(state, digest, packet_sha=None):
    best = select_best(state['observations'])
    save_json(
        BASE / 'bayes_summary.json',
        {
            'status': state['status'],
            'new_charged': state['charged_attempts'] - META['history_count'],
            'new_completed': len(state['observations']) - META['history_count'],
            'best': best,
            'history_sha256': digest,
            'packet_sha256': packet_sha,
            'baseline_attempt83_mm': 2.7080123244518637,
            'predecessor_best': read(HISTORY).get('predecessor_best'),
            'planned_waves': META['waves'],
            'planned_new_evaluations': META['new_budget'],
            'new_valid': sum(o['valid'] for o in state['observations'][META['history_count']:]),
            'new_timeouts': sum(o.get('failure_class') == 'timeout'
                                for o in state['observations'][META['history_count']:]),
            'limits': 'Adaptive single-specimen Run6 calibration; no claim of stationarity, physical identification or independent generalization.',
        },
    )
    message('EXPANDED-BO', f'{state["status"]}; best={best["objective_mm"]:.9g}mm attempt{best["attempt_id"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'propose'])
    parser.add_argument('--packet-sha256')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'propose':
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if args.seed is None:
            parser.error('propose requires --seed')
        _, diagnostics = proposal(json.load(sys.stdin), args.seed)
        print(json.dumps(diagnostics, allow_nan=False))
    elif not args.packet_sha256:
        parser.error('run requires --packet-sha256')
    else:
        run(args.packet_sha256)
