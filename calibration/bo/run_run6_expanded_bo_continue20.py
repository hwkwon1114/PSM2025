#!/usr/bin/env python
"""Continue the completed expanded Run 6 campaign: 20 q4 waves, no Run 9 access."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_run6_expanded_bo as runner
from calibration.bo import run_run6_bo_refinement as controller
from calibration.ewcal import batch_bayes as batch
from calibration.ewcal.common import Parameters
from calibration.ewcal.runner import Evaluator

SOURCE_BASE = ROOT / 'run/run6_expanded_bo'
BASE = ROOT / 'run/run6_expanded_bo_continue20'
REVIEW = ROOT / 'run/optimization_review/run6_expanded_bo_continue20'
HISTORY = BASE / 'corrected_history.json'
META = dict(runner.META, history_count=160, new_budget=80, waves=20,
            epoch_start_attempt=161, proposal_seed_offset=10)


def validated_source():
    """Check the original completed ledger without changing any original publication."""
    runner.configure()
    history = controller.read(SOURCE_BASE / 'corrected_history.json')
    state = controller.read(SOURCE_BASE / 'bayes_state.json')
    summary = controller.read(SOURCE_BASE / 'bayes_summary.json')
    cfg = controller.read(runner.CONFIG)
    if (history['metadata'] != runner.META or cfg != history['config'] or
            Evaluator.identity_for_config(cfg) != history['evaluator_identity']):
        raise ValueError('Source evaluator/config/history changed')
    # Only the reviewed count parameterization is allowed to change a seed input.
    for name, digest in history['input_sha256'].items():
        if name == 'calibration/bo/run_run6_expanded_bo.py':
            continue
        if controller.sha(ROOT / name) != digest:
            raise ValueError(f'Source scientific input changed: {name}')
    controller.validate_ledger(state, history['evaluator_identity'], 2026, history=history)
    best = runner.select_best(state['observations'])
    if (state['status'] != 'budget_complete' or state['charged_attempts'] != 160 or
            sorted(o['attempt_id'] for o in state['observations']) != list(range(1, 161)) or
            sum(o['valid'] for o in state['observations']) != 147 or
            summary['status'] != 'budget_complete' or summary['new_charged'] != 40 or
            summary['new_completed'] != 40 or summary['best'] != best or
            state['history_sha256'] != controller.sha(SOURCE_BASE / 'corrected_history.json') or
            summary['history_sha256'] != state['history_sha256']):
        raise ValueError('Source must be the completed 160-observation campaign with its true best')
    streak, crossed = batch.reconstruct_proposal_order_streak(
        state['observations'], epoch_start_attempt=121)
    if streak or crossed or (SOURCE_BASE / 'controller_failure.json').exists():
        raise ValueError('Cannot extend a failed/stopped predecessor')
    codec = Parameters(cfg)
    for observation in state['observations']:
        batch.validate_observation_record(observation)
        controller.validate_material_science(observation)
        if not np.allclose(codec.encode(observation['parameters']),
                           observation['normalized_parameters'], rtol=0, atol=1e-12):
            raise ValueError('Source physical/normalized parameter mismatch')
    state = copy.deepcopy(state)
    state['observations'].sort(key=lambda o: o['attempt_id'])
    return history, state, summary


def prepare():
    if BASE.exists():
        raise FileExistsError('Preserve existing continuation directory')
    history, state, summary = validated_source()
    names = [SOURCE_BASE / name for name in
             ('corrected_history.json', 'bayes_state.json', 'bayes_summary.json')]
    names += [Path(__file__), ROOT / 'calibration/bo/run_run6_expanded_bo_continue20.sbatch',
              ROOT / 'calibration/tests/test_expanded_bo_continuation.py',
              ROOT / 'run/stage2_bayesian_optimization/stage2_cluster_protocol.md']
    pins = dict(history['input_sha256'])
    pins['calibration/bo/run_run6_expanded_bo.py'] = controller.sha(ROOT / 'calibration/bo/run_run6_expanded_bo.py')
    pins.update({str(p.relative_to(ROOT)): controller.sha(p) for p in names})
    payload = {
        'schema_version': 1, 'metadata': META, 'config': history['config'],
        'evaluator_identity': history['evaluator_identity'],
        'observations': copy.deepcopy(state['observations']), 'input_sha256': pins,
        'predecessor_best': copy.deepcopy(summary['best']),
        'limits': 'Exact 160 Run6 seeds; 80 new attempts161-240, unchanged recipe; no Run9 or retry.',
    }
    controller.verify_inputs(payload)
    BASE.mkdir(parents=True)
    controller.write_new(HISTORY, payload)
    print('HISTORY', controller.sha(HISTORY), flush=True)


def configure():
    runner.BASE, runner.HISTORY, runner.REVIEW, runner.META = BASE, HISTORY, REVIEW, META
    runner.configure()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--packet-sha256')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif not args.packet_sha256:
        parser.error('run requires --packet-sha256')
    else:
        configure()
        runner.run(args.packet_sha256)
