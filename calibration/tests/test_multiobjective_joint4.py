"""Solver-free checks of the Run 1/3 multiobjective continuation."""
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from calibration.bo import pareto_bayes as pb

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def controller():
    path = ROOT / 'calibration/bo/joint_calibration.py'
    spec = importlib.util.spec_from_file_location('joint_calibration_pareto_fixture', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {'JOINT_SPEC': str(ROOT / 'calibration/bo/multiobjective_joint4.json')}):
        spec.loader.exec_module(module)
    return module


def record(a, wave, e1, e3, *, valid=True, x=None):
    return {'attempt': a, 'wave': wave, 'valid': valid,
            'failure_class': None if valid else 'solver',
            'x': list(x if x is not None else [0.5] * 4),
            'parameters': {}, 'case_rms_mm': {'run6': e1, 'run10': e3}}


def test_pareto_minimization_ties_and_failure_exclusion():
    points = [[1, 4], [2, 2], [4, 1], [3, 3], [2, 2]]
    assert pb.nondominated(points).tolist() == [True, True, True, False, True]
    records = [record(i, 0, *p) for i, p in enumerate(points)]
    records.append(record(6, 0, 0.01, 0.01, valid=False))
    assert [r['attempt'] for r in pb.observed_front(records)] == [0, 1, 2, 4]
    with pytest.raises(ValueError):
        pb.case_errors(records[-1])
    with pytest.raises(ValueError):
        pb.nondominated([[1, float('nan')]])


def test_real_seed_is_paired_unchanged_and_uses_only_run1_run3(controller):
    seed = controller.seed_results()
    assert len(seed) == 88
    assert sum(r['valid'] for r in seed.values()) == 86
    assert {r['original_attempt'] for r in seed.values() if not r['valid']} == {2, 9}
    assert all(r['failure_class'] == 'solver' for r in seed.values() if not r['valid'])
    assert len(pb.observed_front(list(seed.values()))) == 9
    cfg = controller.joint_config(controller.decode([0.5] * 4))
    assert [c['id'] for c in cfg['cases']] == ['run6', 'run10']
    assert all('run9' not in c['target'] for c in cfg['cases'])
    assert cfg['solver']['timeout_seconds'] == 3600
    assert cfg['registration']['metric'] == 'symmetric'


@pytest.mark.parametrize('damage', ['parameters', 'identity'])
def test_seed_tampering_fails_closed(controller, monkeypatch, damage):
    original = controller.jread

    def altered(path):
        result = original(path)
        path = str(path)
        if '/a0001/' in path and path.endswith('/result.json') and damage == 'parameters':
            result['parameters']['gtop'] *= 2
        if '/a0001/' in path and path.endswith('/identity.json') and damage == 'identity':
            result['versions']['numpy'] = 'tampered'
        return result

    monkeypatch.setattr(controller, 'jread', altered)
    with pytest.raises(ValueError, match='Seed scientific identity mismatch'):
        controller.seed_results()


def test_multiobjective_acquisition_uses_separate_valid_case_errors(monkeypatch):
    from calibration.ewcal import batch_bayes
    original = batch_bayes.fit_gp_model
    seen = []

    def spy(observations):
        seen.append([r['objective_mm'] for r in observations])
        return original(observations)

    monkeypatch.setattr(batch_bayes, 'fit_gp_model', spy)
    rng = np.random.default_rng(12)
    records = []
    for i, x in enumerate(rng.uniform(0.05, 0.95, (12, 4))):
        e1 = 0.4 + (x[0] - 0.2) ** 2 + 0.1 * x[1]
        e3 = 0.3 + (x[0] - 0.8) ** 2 + 0.1 * x[2]
        records.append(record(i + 1, 0, e1, e3, x=x))
    records.append(record(13, 0, 99, 99, valid=False, x=[0.01] * 4))
    batch, scores = pb.propose(records, [3, 3], seed=8, batch_size=4, min_sep=0.01,
                               pool_power=5, local_count=8, mc_samples=8, chunk_size=16)
    assert seen == [[r['case_rms_mm'][case] for r in records if r['valid']] for case in pb.CASES]
    assert batch.shape == (4, 4) and np.isfinite(batch).all() and np.isfinite(scores).all()
    assert np.all((batch >= 0) & (batch <= 1))
    all_x = np.array([r['x'] for r in records] + list(batch))
    dist = np.linalg.norm(all_x[:, None, :] - all_x[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    assert np.min(dist[-4:]) >= 0.01


def test_preserved_proposals_are_checked_before_resume(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(controller, 'BASE', tmp_path)
    paths = [tmp_path / 'proposals' / f'a{a:04d}.json' for a in range(1, 5)]
    for a, path in enumerate(paths, 1):
        x = [a / 5, 0.2, 0.4, 0.6]
        controller.save_json(path, {'attempt': a, 'wave': 1, 'x': x,
                                    'parameters': controller.decode(x), 'source': 'greedy_qlogehvi'})
    seed = {-1: record(-1, 0, 1, 1, x=[0, 0, 0, 0])}
    assert controller.make_wave(seed, 1) == [1, 2, 3, 4]
    damaged = json.loads(paths[0].read_text())
    damaged['parameters']['gtop'] *= 2
    controller.save_json(paths[0], damaged)
    with pytest.raises(ValueError, match='Invalid preserved proposal'):
        controller.make_wave(seed, 1)


def test_summary_keeps_seed_and_new_frontier_provenance(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(controller, 'BASE', tmp_path)
    seed_point = dict(record(-1, 0, 1, 2), original_attempt=7, source='seed')
    new_point = record(1, 1, 2, 1)
    monkeypatch.setattr(controller, 'results', lambda: {1: new_point})
    controller.summary('running', {-1: seed_point})
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['seed_valid'] == result['new_valid'] == 1
    assert [(r['source'], r['attempt']) for r in result['frontier']] == [('seed', 7), ('continuation', 1)]


def test_four_batch_plateau_is_not_triggered_by_new_tradeoff(controller):
    seed = {-1: record(-1, 0, 1, 1)}
    res = {4 * (w - 1) + i: record(4 * (w - 1) + i, w, 2 + i / 10, 2)
           for w in range(1, 5) for i in range(1, 5)}
    assert controller.pareto_stalled(seed, res, 4)
    res[16] = record(16, 4, 0.8, 1.2)
    assert not controller.pareto_stalled(seed, res, 4)
    del res[16]
    with pytest.raises(ValueError, match='Incomplete batch'):
        controller.pareto_stalled(seed, res, 4)


def test_allocation_guard_and_plan_identity(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(controller.time, 'time', lambda: 1000.0)
    monkeypatch.setenv('SLURM_JOB_END_TIME', '8500')
    assert controller.enough_slurm_time()
    monkeypatch.setenv('SLURM_JOB_END_TIME', '8499')
    assert not controller.enough_slurm_time()
    monkeypatch.delenv('SLURM_JOB_END_TIME')
    with pytest.raises(RuntimeError, match='SLURM_JOB_END_TIME'):
        controller.enough_slurm_time()

    monkeypatch.setattr(controller, 'BASE', tmp_path)
    source = tmp_path / 'source.txt'
    source.write_text('original')
    monkeypatch.setattr(controller, 'software_versions', lambda: {'fixture': '1'})
    plan = {'files': {str(source): controller.sha256(source)},
            'reference_mm': controller.SPEC['reference_mm'], 'versions': {'fixture': '1'}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    assert controller.verify_plan() == plan
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='Reviewed input changed'):
        controller.verify_plan()
    with pytest.raises(FileExistsError):
        controller.prepare()


def test_expiring_allocation_never_launches_a_candidate(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(controller, 'BASE', tmp_path)
    monkeypatch.setattr(controller, 'seed_results', lambda: {-1: record(-1, 0, 1, 1)})
    monkeypatch.setattr(controller, 'verify_plan', lambda: {})
    time_checks = iter([True, False])
    monkeypatch.setattr(controller, 'enough_slurm_time', lambda: next(time_checks))
    monkeypatch.setattr(controller, 'make_wave', lambda res, wave: [1, 2, 3, 4])
    summaries = []
    monkeypatch.setattr(controller, 'summary', lambda status, seed=None: summaries.append(status))
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *a, **kw: pytest.fail('candidate was launched'))
    controller.run()
    assert summaries == ['allocation_limit']
