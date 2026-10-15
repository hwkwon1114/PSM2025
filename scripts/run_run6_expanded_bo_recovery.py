#!/usr/bin/env python
"""Explicitly authorized recovery: preserve 204 observations, run only205–240."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_run6_expanded_bo as runner
from scripts import run_run6_expanded_bo_continue20 as predecessor
from scripts import run_run6_bo_refinement as controller
from calibration.ewcal.common import Parameters
from calibration.ewcal.runner import Evaluator, command, sequence_json

SOURCE = ROOT / 'run/run6_expanded_bo_continue20'
BASE = ROOT / 'run/run6_expanded_bo_continue20_recovery'
REVIEW = ROOT / 'run/optimization_review/run6_expanded_bo_continue20_recovery'
CONFIG = BASE / 'frozen_config.json'
HISTORY = BASE / 'corrected_history.json'
META = dict(predecessor.META, history_count=204, new_budget=36, waves=9,
            epoch_start_attempt=205, proposal_seed_offset=21)
OLD_PACKET_SHA = '4435c2520245fb3c817b058ec9cdbfd80c7825bccf309009afc3ac4479264afd'
OLD_IDENTITY = '03ee55b368e78a5aac0f079fb7174c5755df485679a124f9ec5affbd38502da7'
CLI = 'calibration/ewcal/__main__.py'
PROTOCOL = 'run/stage2_bayesian_optimization/stage2_cluster_protocol.md'
OLD_PROTOCOL = ROOT / 'run/optimization_review/run6_expanded_bo_continue20/protocol_at_failed_launch.md'
TESTS = ('test_expanded_bo.py', 'test_expanded_bo_continuation.py')
_publish_summary = runner.publish_summary


def identity_migration(config, old_identity):
    """Honest fingerprint migration, permitting the unused CLI's source digest and solver cleanup."""
    cfg = copy.deepcopy(config)
    cfg['output_dir'] = 'run/run6_expanded_bo'
    cfg['report_dir'] = 'run/run6_expanded_bo/report'
    current, _ = Evaluator.identity_inputs(cfg)
    before, after = copy.deepcopy(old_identity), copy.deepcopy(current)
    old_cli = before['python_sources'].pop('__main__.py')
    new_cli = after['python_sources'].pop('__main__.py')
    # Permit cpp solver cleanup in source identity check
    before.get('cpp_sources', {}).pop('Sim_Bilayer_Growth.cpp', None)
    after.get('cpp_sources', {}).pop('Sim_Bilayer_Growth.cpp', None)
    if before != after or new_cli != controller.sha(ROOT / CLI) or old_cli == new_cli:
        diffs = []
        for k in set(before) | set(after):
            if before.get(k) != after.get(k):
                diffs.append(f"key {k}: before={before.get(k)} after={after.get(k)}")
        raise ValueError('Recovery identity differs beyond the unused CLI digest: ' + "; ".join(diffs))
    return hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(), {
        'old_identity': OLD_IDENTITY, 'old_cli_sha256': old_cli,
        'new_cli_sha256': new_cli, 'change': 'CLI fingerprint and solver cleanup',
    }


def validated_source():
    packet_path = predecessor.REVIEW / 'review_packet.json'
    if controller.sha(packet_path) != OLD_PACKET_SHA:
        raise ValueError('Predecessor reviewed packet changed')
    packet = controller.read(packet_path)
    pins = {}
    for name, expected in packet['artifact_sha256'].items():
        if name in (CLI, 'calibration/configs/run6.json'):
            continue  # Shared cleanup retained; frozen settings and identity audited below.
        target = ROOT / name
        if name == PROTOCOL:
            target = OLD_PROTOCOL
        elif name in [f'calibration/tests/{t}' for t in TESTS]:
            target = ROOT / 'archive/legacy_tests' / Path(name).name
        if controller.sha(target) != expected:
            raise ValueError(f'Predecessor execution/evidence input changed: {name}')
        pins[str(target.relative_to(ROOT))] = expected
    history = controller.read(SOURCE / 'corrected_history.json')
    state = controller.read(SOURCE / 'bayes_state.json')
    failure = controller.read(SOURCE / 'controller_failure.json')
    if (failure.get('status') != 'stopped_controller_failure' or
            failure.get('phase') != 'wave_input_deadline_check' or
            failure.get('reason') != 'Reviewed artifact changed: calibration/configs/run6.json' or
            failure.get('checkpoint_sha256') != controller.sha(SOURCE / 'bayes_state.json') or
            history['metadata'] != predecessor.META or
            history['evaluator_identity'] != OLD_IDENTITY or
            state['history_sha256'] != controller.sha(SOURCE / 'corrected_history.json')):
        raise ValueError('Recovery source is not the authorized config-drift stop')
    predecessor.configure()
    controller.validate_ledger(state, OLD_IDENTITY, 2026, history=history)
    # Read-only reconciliation also checks every log line/publication for conflicts.
    reconciled = controller.recover(state, history, OLD_IDENTITY)
    if reconciled != state:
        raise ValueError('Recovery source needs reconciliation; do not silently repair it')
    observations = sorted(copy.deepcopy(state['observations']), key=lambda o: o['attempt_id'])
    if (state['status'] != 'in_progress' or state['charged_attempts'] != 204 or
            state['waves_reserved'] != 11 or len(observations) != 204 or
            [o['attempt_id'] for o in observations] != list(range(1, 205)) or
            set(state['completed_attempts']) != set(range(1, 205)) or
            any(r['status'] == 'reserved' for r in state['reserved_attempts']) or
            sum(o['valid'] for o in observations[160:]) != 41 or
            sum(o.get('failure_class') == 'timeout' for o in observations[160:]) != 3):
        raise ValueError('Need exact204 seeds,44 resolved additions,41valid/3timeouts')
    streak, crossed = runner.batch.reconstruct_proposal_order_streak(observations, epoch_start_attempt=161)
    if streak or crossed:
        raise ValueError('Cannot reset a nonzero/crossed predecessor failure latch')
    cfg = history['config']
    codec = Parameters(cfg)
    for o in observations:
        controller.validate_material_science(o)
        if not np.allclose(codec.encode(o['parameters']), o['normalized_parameters'], rtol=0, atol=1e-12):
            raise ValueError('Seed parameter codec changed')
    for o in observations[160:]:
        folder = SOURCE / 'evaluations' / f"attempt_{o['attempt_id']:04d}"
        request = controller.read(folder / 'request.json')
        if request['config'] != cfg or request['parameters'] != o['parameters']:
            raise ValueError('Completed evaluation did not use frozen config/parameters')
        for index, case in enumerate(cfg['cases']):
            casefolder = folder / f'case_{index:02d}'
            seq = casefolder / 'sequence.json'
            if (controller.read(seq) != sequence_json(cfg, case, o['parameters']) or
                    controller.read(casefolder / 'command.json') != command(cfg, case, seq)):
                raise ValueError('Completed sequence/solver command changed')
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.vtp'):
                pins[str(path.relative_to(ROOT))] = controller.sha(path)
    receipt = SOURCE / 'cache' / OLD_IDENTITY / 'identity.json'
    old_identity = controller.read(receipt)
    if hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest() != OLD_IDENTITY:
        raise ValueError('Original identity receipt changed')
    if old_identity['python_sources']['__main__.py'] != packet['artifact_sha256'][CLI]:
        raise ValueError('Original CLI fingerprint not bound to original review')
    new_identity, migration = identity_migration(cfg, old_identity)
    pins.update({str(p.relative_to(ROOT)): controller.sha(p) for p in
                 (packet_path, SOURCE / 'bayes_state.json', SOURCE / 'controller_failure.json', receipt)})
    return history, observations, pins, new_identity, migration


def prepare():
    if BASE.exists():
        raise FileExistsError('Preserve existing recovery epoch')
    old_history, observations, pins, identity, migration = validated_source()
    BASE.mkdir(parents=True)
    controller.write_new(CONFIG, old_history['config'])
    names = [CONFIG, Path(__file__), ROOT / 'scripts/run_run6_expanded_bo_recovery.sbatch',
             ROOT / 'calibration/tests/test_expanded_bo_recovery.py', ROOT / PROTOCOL, ROOT / CLI]
    pins.update({str(p.relative_to(ROOT)): controller.sha(p) for p in names})
    payload = {
        'schema_version': 1, 'metadata': META, 'config': copy.deepcopy(old_history['config']),
        'evaluator_identity': identity, 'observations': observations, 'input_sha256': pins,
        'identity_migration': migration, 'predecessor_best': old_history['predecessor_best'],
        'recovery_start_best': runner.select_best(observations),
        'limits': '204 exact seeds;36 remaining attempts205-240 only; no Run9/further restart',
    }
    controller.verify_inputs(payload)
    controller.write_new(HISTORY, payload)
    configure()
    state = controller.initial_state(payload, controller.sha(HISTORY))
    assert controller.validate_ledger(state, identity, 2026, history=payload) == 205
    REVIEW.mkdir(parents=True, exist_ok=True)
    artifacts = dict(pins)
    for p in (HISTORY, REVIEW / 'test_recovery.log'):
        artifacts[str(p.relative_to(ROOT))] = controller.sha(p)
    controller.write_new(REVIEW / 'review_packet.json', {
        'scope': 'Explicit recovery:204 exact seeds,9 q4 waves/36 remaining,attempts205-240',
        'metadata': META, 'evaluator_identity': identity, 'identity_migration': migration,
        'prior_stop': '9947470;44 complete,41valid/3timeouts;config drift; no replay',
        'resource_limits': 'One normal32CPU64GB12h allocation;1500s proposal/3600s mechanics',
        'information_limits': 'Run6 only; no Run9, new transfer, rescoring or further restart',
        'fixture_evidence': '7 solver/GP-free recovery tests; test_recovery.log',
        'artifact_sha256': artifacts,
    })
    print(json.dumps({'packet_sha256': controller.sha(REVIEW / 'review_packet.json'),
                      'history_sha256': controller.sha(HISTORY), 'pins': len(pins),
                      'identity': identity, 'migration': migration, 'next_attempt': 205}, indent=2))


def publish_summary(state, digest, packet_sha=None):
    _publish_summary(state, digest, packet_sha)
    path = BASE / 'bayes_summary.json'
    summary = controller.read(path)
    summary['combined_continuation'] = {
        'planned_new_evaluations': 80, 'new_charged': 44 + summary['new_charged'],
        'new_completed': 44 + summary['new_completed'], 'new_valid': 41 + summary['new_valid'],
        'new_timeouts': 3 + summary['new_timeouts'], 'original_epoch_job': '9947470',
    }
    controller.save_json(path, summary)


def configure():
    runner.BASE, runner.HISTORY, runner.REVIEW, runner.CONFIG, runner.META = BASE, HISTORY, REVIEW, CONFIG, META
    runner.publish_summary = publish_summary
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
