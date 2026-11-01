"""Bounded, solver-free checks for fresh three-pair calibrations."""
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.bo import pairwise_floor1 as p
from calibration.ewcal import runner
from calibration.ewcal.geometry import triangle_mesh
from calibration.tests.test_direction_frame import fixture


@pytest.mark.parametrize('pair,ids', [('12', ('run6', 'run9')), ('23', ('run9', 'run10')), ('13', ('run6', 'run10'))])
def test_pair_configuration_and_codec(pair, ids):
    p.configure(pair)
    cfg = p.pair_config(pair, p.joint.decode([0.4, 0.3, 0.2, 1.0]))
    assert tuple(c['id'] for c in cfg['cases']) == ids
    assert p.EXCLUDED[pair] not in ids
    assert all(c['weight'] == 1 for c in cfg['cases'])
    assert cfg['registration'] == {'method': 'material_reference', 'metric': 'symmetric', 'require_state_export': True}
    assert not cfg.get('lhs') and cfg['solver']['timeout_seconds'] == 3600
    assert cfg['hardening']['floor'] == 1.0
    lo, hi = p.joint.decode(np.zeros(4)), p.joint.decode(np.ones(4))
    assert lo['floor'] == 0 and hi['floor'] == 1
    assert lo['beta'] == pytest.approx(0.01) and hi['beta'] == pytest.approx(1.0)
    assert lo['gtop'] == pytest.approx(2e-5) and hi['gtop'] == pytest.approx(5e-4)
    assert hi['gbot'] == hi['ortho_bottom'] == 0
    assert np.allclose(p.joint.encode(hi), np.ones(4))
    with pytest.raises(ValueError):
        p.joint.decode([0.5, 0.5, 0.5, 1.001])


def test_all_three_designs_are_the_same_fresh_lhs(tmp_path, monkeypatch):
    designs = []
    for pair in p.PAIRS:
        p.configure(pair)
        base = tmp_path / pair
        monkeypatch.setattr(p.joint, 'BASE', base)
        designs.append(p.joint.design())
    assert all(np.array_equal(designs[0], x) for x in designs[1:])
    assert designs[0].shape == (16, 4)
    for col in designs[0].T:
        assert sorted(np.floor(16 * col).astype(int)) == list(range(16))
    assert (p.joint.WAVES, p.joint.N_DESIGN_WAVES, p.joint.Q) == (24, 4, 4)


@pytest.fixture
def synthetic_evaluator(tmp_path, monkeypatch):
    def make(cfg, positions, ref, faces, targets, values):
        cfg['output_dir'] = str(tmp_path)
        monkeypatch.setattr(runner.Evaluator, 'identity_inputs', staticmethod(lambda _: ({'fixture': True}, targets)))
        monkeypatch.setattr(runner, 'sequence_json', lambda *a: {})
        monkeypatch.setattr(runner, 'command', lambda *a: ['fake'])
        def fake_launch(args, folder, *unused):
            state = {'schema_version': 1, 'units': 'metres', 'reference_positions': (ref / 1000).tolist(), 'faces': faces.tolist()}
            (folder / 'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps(state))
        monkeypatch.setattr(runner, 'launch', fake_launch)
        monkeypatch.setattr(runner, 'check_output', lambda *a: ({}, triangle_mesh(positions, faces)))
        monkeypatch.setattr(runner, 'write_mesh', lambda *a: None)
        ev = runner.Evaluator(cfg)
        ev.parameters = SimpleNamespace(decode=lambda _: {k: values[k] for k in cfg['parameters']})
        return ev
    return make


@pytest.mark.parametrize('errors,expected', [((0.0, 0.0), 0.0), ((3.0, 4.0), math.sqrt(12.5))])
def test_real_evaluator_accepts_zero_and_aggregates_two_case_scores(synthetic_evaluator, monkeypatch, errors, expected):
    """Exercise actual evaluator control/aggregation; replace only the expensive forward solve and score."""
    values = p.joint.decode([0.3, 0.4, 0.5, 0.6])
    cfg = p.pair_config('12', values)
    positions, ref, faces = fixture()
    targets = {'run6': np.full((10, 3), errors[0]), 'run9': np.full((10, 3), errors[1])}
    ev = synthetic_evaluator(cfg, positions, ref, faces, targets, values)
    def fake_score(mesh, target, reference, faces):
        e = float(target[0, 0])
        return {'rms_mm': e, 'rms_symmetric_mm': e}, mesh, target, np.zeros(len(target)), target
    monkeypatch.setattr(runner, 'score_material_frame', fake_score)
    rec = ev.evaluate([0] * len(cfg['parameters']), use_cache=False, attempt_id=1)
    assert rec['valid'] and rec['objective_mm'] == pytest.approx(expected)
    assert [r['case_rms'] for r in rec['cases']] == list(errors)


@pytest.mark.parametrize('pair', ['12', '23', '13'])
def test_real_metric_and_evaluator_identical_surface_zero(pair, synthetic_evaluator):
    """Actual scorer and evaluator together; only the solver is replaced by a saved synthetic mesh."""
    from calibration.ewcal.geometry import interpolate_mesh_z
    values = p.joint.decode([0.3, 0.4, 0.5, 0.6])
    cfg = p.pair_config(pair, values)
    positions, ref, faces = fixture()
    sx, sy = np.meshgrid(np.linspace(-85, 85, 18), np.linspace(-135, 135, 28))
    xy = np.column_stack([sx.ravel(), sy.ravel()])
    scan = np.column_stack([xy, interpolate_mesh_z(positions, faces, xy)]) + [12.0, -34.0, 5.5]
    targets = {name: scan.copy() for name in p.PAIRS[pair]}
    ev = synthetic_evaluator(cfg, positions, ref, faces, targets, values)
    rec = ev.evaluate([0] * len(cfg['parameters']), use_cache=False, attempt_id=1)
    assert rec['valid'] and rec['objective_mm'] < 1e-12
    for case in rec['cases']:
        assert case['metrics']['rms_target_to_sim_mm'] < 1e-12
        assert case['metrics']['rms_sim_to_target_weighted_mm'] < 1e-12
        assert case['case_rms'] < 1e-12


def saved_result(base, attempt, errors, valid=True, failure_class=None):
    x = [0.1, 0.2, 0.3, 0.4]
    params = p.joint.decode(x)
    proposal = {'attempt': attempt, 'wave': 1, 'x': x, 'parameters': params}
    p.joint.save_json(base / 'proposals' / f'a{attempt:04d}.json', proposal)
    rec = dict(proposal, valid=valid, failure_class=failure_class,
               objective_mm=math.sqrt(sum(e * e for e in errors) / 2),
               case_rms_mm={'run6': errors[0], 'run9': errors[1]})
    p.joint.save_json(base / 'results' / f'a{attempt:04d}.json', rec)
    return rec


def test_validate_result_rejects_wrong_pair_and_fictitious_objective(tmp_path, monkeypatch):
    p.configure('12')
    monkeypatch.setattr(p.joint, 'BASE', tmp_path)
    r = saved_result(tmp_path, 1, (3, 4))
    assert p.validate_results('12', {1: r})
    with pytest.raises(ValueError, match='wrong joint objective'):
        p.validate_results('12', {1: dict(r, objective_mm=1)})
    with pytest.raises(ValueError, match='wrong cases'):
        p.validate_results('12', {1: dict(r, case_rms_mm={'run6': 3, 'run10': 4})})
    with pytest.raises(ValueError, match='invalid case errors'):
        p.validate_results('12', {1: dict(r, case_rms_mm={'run6': -3, 'run9': 4})})
    failed = dict(r, valid=False, failure_class='solver', objective_mm=999)
    assert p.validate_results('12', {1: failed})[1]['failure_class'] == 'solver'


def test_freeze_before_one_excluded_case_and_no_retry(tmp_path, monkeypatch):
    p.configure('12')
    monkeypatch.setattr(p.joint, 'BASE', tmp_path)
    monkeypatch.setattr(p.joint, 'BASE_REL', str(tmp_path))
    monkeypatch.setattr(p, 'verify_plan', lambda pair: None)
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(int(time.time()) + 5000))
    for key in ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    for key in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT'):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / 'plan.json').write_text('{}')
    saved_result(tmp_path, 1, (3, 4))
    best = saved_result(tmp_path, 2, (1, 2))
    seen = []
    class FakeEvaluator:
        def __init__(self, cfg):
            selection = json.loads((tmp_path / 'selection.json').read_text())
            assert selection['attempt'] == 2 and selection['excluded_case'] == 'run10'
            assert [c['id'] for c in cfg['cases']] == ['run10']
            assert cfg['hardening']['floor'] == best['parameters']['floor']
            self.cfg = cfg
        def evaluate(self, x, **kwargs):
            seen.append((self.parameters.decode(x), kwargs))
            return {'valid': False, 'failure_class': 'timeout', 'objective_mm': 999}
    monkeypatch.setattr(p, 'Evaluator', FakeEvaluator)
    p.finish('12', 'budget_complete')
    assert len(seen) == 1 and seen[0][1] == {'use_cache': False, 'attempt_id': 1}
    assert json.loads((tmp_path / 'transfer_result.json').read_text())['failure_class'] == 'timeout'
    with pytest.raises(FileExistsError, match='no automatic rerun'):
        p.finish('12', 'budget_complete')
    assert len(seen) == 1


def test_exact_zero_stops_before_gp(tmp_path, monkeypatch):
    p.configure('12')
    monkeypatch.setattr(p, 'configure', lambda _: tmp_path)
    monkeypatch.setattr(p.joint, 'BASE', tmp_path)
    monkeypatch.setattr(p, 'verify_plan', lambda _: None)
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(int(time.time()) + 20000))
    statuses = []
    monkeypatch.setattr(p, 'finish', lambda pair, status: statuses.append(status))
    monkeypatch.setattr(p.joint, 'propose', lambda *a: pytest.fail('zero must not be passed to log GP'))
    class FakeProcess:
        def __init__(self, args, **kwargs):
            attempt = int(args[-1])
            proposal = p.joint.jread(tmp_path / 'proposals' / f'a{attempt:04d}.json')
            p.joint.save_json(tmp_path / 'results' / f'a{attempt:04d}.json', {
                **proposal, 'valid': True, 'failure_class': None, 'objective_mm': 0.0,
                'case_rms_mm': {'run6': 0.0, 'run9': 0.0}})
        def wait(self):
            return 0
    monkeypatch.setattr(p.subprocess, 'Popen', FakeProcess)
    p.run('12')
    assert statuses == ['exact_zero_optimum']


def test_plan_hash_and_file_changes_fail_closed(tmp_path, monkeypatch):
    p.configure('12')
    monkeypatch.setattr(p.joint, 'BASE', tmp_path)
    source = tmp_path / 'input.txt'
    source.write_text('reviewed')
    monkeypatch.setattr(p, 'plan_paths', lambda _: [str(source)])
    monkeypatch.setattr(p.joint, 'software_versions', lambda: {'fixture': '1'})
    environment = {k: p.os.environ.get(k, '') for k in ('CONDA_PREFIX', 'LD_LIBRARY_PATH')}
    plan = {'pair': '12', 'cases': ['run6', 'run9'], 'versions': {'fixture': '1'},
            'evaluator_identity': {'environment': environment}, 'files': {str(source): p.joint.sha256(source)}}
    p.joint.save_json(tmp_path / 'plan.json', plan)
    monkeypatch.setenv('PAIR_PLAN_SHA256', p.joint.sha256(tmp_path / 'plan.json'))
    assert p.verify_plan('12') == plan
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='Reviewed input changed'):
        p.verify_plan('12')
    source.write_text('reviewed')
    monkeypatch.setenv('PAIR_PLAN_SHA256', 'incorrect')
    with pytest.raises(RuntimeError, match='plan hash'):
        p.verify_plan('12')


def test_solver_thread_guard(monkeypatch):
    for key in ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    for key in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT'):
        monkeypatch.delenv(key, raising=False)
    p.check_solver_environment()
    monkeypatch.setenv('BLIS_JC_NT', '2')
    with pytest.raises(RuntimeError, match='thread environment'):
        p.check_solver_environment()


def test_allocation_guard_fails_closed_and_retains_results(tmp_path, monkeypatch):
    p.configure('23')
    monkeypatch.setattr(p, 'configure', lambda _: tmp_path)
    monkeypatch.setattr(p.joint, 'BASE', tmp_path)
    monkeypatch.setattr(p, 'verify_plan', lambda _: None)
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(int(time.time()) + 10000))
    statuses = []
    monkeypatch.setattr(p, 'finish', lambda pair, status: statuses.append(status))
    monkeypatch.setattr(p.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    p.run('23')
    assert statuses == ['allocation_limit']
