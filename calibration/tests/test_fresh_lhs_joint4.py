"""Fresh four-parameter design and controller checks; no shell simulations."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def joint(tmp_path, monkeypatch):
    monkeypatch.setenv('JOINT_SPEC', str(ROOT / 'calibration/bo/fresh_lhs_joint4.json'))
    import calibration.bo.joint_calibration as module
    module = importlib.reload(module)
    monkeypatch.setattr(module, 'BASE', tmp_path)
    yield module
    monkeypatch.delenv('JOINT_SPEC')
    importlib.reload(module)


def test_fresh_lhs_is_four_dimensional_stratified_and_unseeded_by_history(joint):
    assert joint.D == 4
    assert (joint.Q, joint.N_DESIGN_WAVES, joint.WAVES) == (4, 4, 24)
    assert (joint.EARLY_STOP_FROM_WAVE, joint.EARLY_WINDOW) == (12, 4)
    d = joint.design()
    assert d.shape == (16, 4) and np.all((d >= 0) & (d <= 1))
    for col in d.T:
        assert sorted(np.floor(col * 16).astype(int)) == list(range(16))
    distances = np.linalg.norm(d[:, None] - d[None], axis=-1) + np.eye(16) * 9
    assert distances.min() >= joint.MIN_SEP
    assert np.array_equal(d, joint.design())
    assert json.loads((joint.BASE / 'design.json').read_text())['seed'] >= 20261026
    lo, hi = joint.decode(np.zeros(4)), joint.decode(np.ones(4))
    assert lo['gtop'] == pytest.approx(2e-5) and hi['gtop'] == pytest.approx(5e-4)
    assert lo['beta'] == pytest.approx(0.01) and hi['beta'] == pytest.approx(1.0)
    assert lo['floor'] == 0 and hi['floor'] == pytest.approx(0.5)
    assert lo['gbot'] == hi['gbot'] == 0 and lo['ortho_bottom'] == hi['ortho_bottom'] == 0


def test_fresh_evaluate_passes_exact_values_to_both_sequence_inputs(joint, monkeypatch):
    from calibration.ewcal.runner import sequence_json

    x = [0.0, 0.3, 0.4, 0.2]  # gtop = 2e-5, below the old Evaluator codec bound
    joint.save_json(joint.BASE / 'proposals/a0001.json', {'attempt': 1, 'wave': 1, 'x': x})
    seen = []

    class FakeEvaluator:
        def __init__(self, cfg):
            self.cfg = cfg

        def evaluate(self, x4, **kwargs):
            p = self.parameters.decode(x4)
            assert p['gtop'] == pytest.approx(2e-5) and p['gbot'] == 0
            for case in self.cfg['cases']:
                seq = sequence_json(self.cfg, case, p)
                op = seq['toolpaths'][0]['operation']
                seen.append((case['id'], op['gtop'], op['gbot'], seq['hardening']))
            return {'valid': True, 'cases': [
                {'id': c['id'], 'metrics': {'rms_mm': 0.5}, 'case_rms': 0.5}
                for c in self.cfg['cases']], 'objective_mm': 0.5, 'seconds': 0,
                'directory': str(joint.BASE)}

    monkeypatch.setattr(joint, 'Evaluator', FakeEvaluator)
    joint.evaluate(1)
    assert [r[0] for r in seen] == ['run6', 'run10']
    assert all(r[1] == pytest.approx(2e-5) and r[2] == 0 for r in seen)
    assert all(r[3]['beta'] == pytest.approx(joint.decode(x)['beta']) for r in seen)
    assert all(r[3]['floor'] == pytest.approx(0.1) for r in seen)


def test_fresh_controller_finishes_design_without_baseline_gate(joint, monkeypatch):
    monkeypatch.setattr(joint, 'WAVES', 4)

    class FakeProcess:
        def __init__(self, args, **kwargs):
            attempt = int(args[-1])
            proposal = joint.jread(joint.BASE / 'proposals' / f'a{attempt:04d}.json')
            joint.save_json(joint.BASE / 'results' / f'a{attempt:04d}.json', {
                **proposal, 'valid': True, 'failure_class': None, 'objective_mm': 5.0,
                'case_rms_mm': {'run6': 5.0, 'run10': 5.0}})

        def wait(self):
            return 0

    monkeypatch.setattr(joint.subprocess, 'Popen', FakeProcess)
    joint.run()
    result = joint.jread(joint.BASE / 'summary.json')
    assert result['status'] == 'budget_complete' and result['evaluations'] == result['valid'] == 16
    assert result['first_design_attempt']['attempt'] == 1 and 'baseline_attempt1' not in result
    assert len(joint.results()) == 16


def test_success_aware_gp_uses_only_valid_new_evaluations(joint, monkeypatch):
    import calibration.ewcal.batch_bayes as batch

    d = joint.design()
    records = {}
    for i, x in enumerate(d, 1):
        valid = i % 4 != 0
        records[i] = {'attempt': i, 'wave': (i - 1) // 4 + 1, 'x': x.tolist(),
                      'valid': valid, 'failure_class': None if valid else 'solver',
                      'objective_mm': 0.8 + 0.4 * float(np.dot(x, x)) if valid else 999.0}
    original = batch.fit_gp_model
    observed = []

    def spy(rows):
        observed.extend(rows)
        return original(rows)

    monkeypatch.setattr(batch, 'fit_gp_model', spy)
    proposal, repaired = joint.propose(records, 5)
    assert len(observed) == 12 and all(r['objective_mm'] < 999 for r in observed)
    assert proposal.shape == (4, 4) and repaired == 0
    taken = [np.asarray(r['x']) for r in records.values()]
    for i, p in enumerate(proposal):
        assert all(np.linalg.norm(p - q) >= joint.MIN_SEP for q in taken + list(proposal[:i]))

    # Compare against the preserved controller under the identical fresh specification.
    spec = importlib.util.spec_from_file_location('archived_joint_fixture', ROOT / 'calibration/bo/run_joint_run6_run10.py')
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    monkeypatch.setattr(archived, 'BASE', joint.BASE.parent / 'archived')
    assert np.array_equal(d, archived.design())
    saved_design = json.loads((ROOT / 'run/fresh_lhs_joint4/design.json').read_text())['x']
    assert np.array_equal(d, np.asarray(saved_design))
    for x in d:
        assert archived.decode(x) == joint.decode(x)
        assert np.array_equal(archived.encode(archived.decode(x)), joint.encode(joint.decode(x)))
    assert archived.joint_config(joint.decode(d[0])) == joint.joint_config(joint.decode(d[0]))
    observed.clear()
    old_proposal, old_repaired = archived.propose(records, 5)
    assert len(observed) == 12 and old_repaired == repaired
    assert np.allclose(old_proposal, proposal, atol=1e-14, rtol=0)


def test_fresh_controller_rejects_warm_start_and_historical_specs(tmp_path):
    spec = json.loads((ROOT / 'calibration/bo/fresh_lhs_joint4.json').read_text())
    spec['warm_start'] = 'run/joint_stage4_reduced4'
    path = tmp_path / 'warm.json'
    path.write_text(json.dumps(spec))
    for spec_path in (path, ROOT / 'calibration/bo/stage4_reduced4.json'):
        env = dict(os.environ, JOINT_SPEC=str(spec_path))
        result = subprocess.run([sys.executable, '-c', 'import calibration.bo.joint_calibration'],
                                cwd=ROOT, env=env, text=True, capture_output=True)
        assert result.returncode != 0
        assert 'Expected the fresh four-parameter success-aware specification' in result.stderr


def test_completed_campaign_is_not_relaunched_or_rewritten(joint):
    path = joint.BASE / 'summary.json'
    joint.save_json(path, {'status': 'early_stop_no_improvement'})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='Completed campaign is preserved'):
        joint.run()
    assert path.read_bytes() == before
    assert sorted(p.name for p in joint.BASE.iterdir()) == ['summary.json']


def test_summary_cli_reads_without_changing_saved_result(tmp_path):
    spec = json.loads((ROOT / 'calibration/bo/fresh_lhs_joint4.json').read_text())
    base = tmp_path / 'campaign'
    base.mkdir()
    spec['base'] = str(base)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    summary = base / 'summary.json'
    summary.write_text('{"status": "budget_complete", "best": 0.5}\n')
    before = summary.read_bytes()
    env = dict(os.environ, JOINT_SPEC=str(path))
    result = subprocess.run([sys.executable, str(ROOT / 'calibration/bo/joint_calibration.py'), 'summary'],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['status'] == 'budget_complete'
    assert summary.read_bytes() == before
