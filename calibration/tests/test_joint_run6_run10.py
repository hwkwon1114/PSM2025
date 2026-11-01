import json
import numpy as np
import pytest
from calibration.bo import run_joint_run6_run10 as J
from calibration.ewcal.runner import sequence_json


def test_codec_roundtrip_and_baseline_in_box():
    x = J.encode(J.BASELINE)
    assert x.shape == (6,) and x[5] == 0 and np.all((0 <= x) & (x <= 1))
    for k, v in J.decode(x).items():
        assert v == pytest.approx(J.BASELINE[k], rel=1e-12, abs=1e-15)
    with pytest.raises(ValueError):
        J.decode(np.r_[x[:5], 1.1])


def test_joint_config_equal_weights_shared_settings_and_hardening_reaches_solver_input():
    a = J.joint_config({'beta': 0.5, 'floor': 0.3})
    b = J.joint_config({'beta': 0.2, 'floor': 0.0})
    assert [(c['id'], c['weight']) for c in a['cases']] == [('run6', 1), ('run10', 1)]
    assert a['registration'] == {'method': 'material_reference', 'metric': 'symmetric', 'require_state_export': True}
    p = {n: J.BASELINE[n] for n in J.NAMES[:4]}
    s = sequence_json(a, a['cases'][1], p)
    assert (s['hardening']['beta'], s['hardening']['floor']) == (0.5, 0.3)
    assert sequence_json(b, b['cases'][1], p)['hardening']['beta'] == 0.2
    assert a['solver'] == b['solver'] and a['cases'] == b['cases']
    for c in a['cases']:
        assert sequence_json(a, c, p)['toolpaths'][0]['operation']['gtop'] == p['gtop']


def test_design_baseline_first_and_separated(tmp_path, monkeypatch):
    monkeypatch.setattr(J, 'BASE', tmp_path)
    d = J.design()
    assert d.shape == (12, 6) and np.allclose(d[0], J.encode(J.BASELINE))
    dist = np.linalg.norm(d[:, None] - d[None], axis=-1) + np.eye(12) * 9
    assert dist.min() >= J.MIN_SEP


def test_gp_proposal_is_admissible_and_stop_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(J, 'BASE', tmp_path)
    d = J.design()
    res = {}
    for i, x in enumerate(d, 1):
        ok = i != 5
        res[i] = dict(attempt=i, wave=(i - 1) // 4 + 1, x=list(x), valid=ok, failure_class=None if ok else 'solver',
                      objective_mm=1.2 + 0.1 * np.sin(5 * x.sum()) if ok else 1e6)
    batch, _ = J.propose(res, 4)
    assert batch.shape == (4, 6) and np.all((0 <= batch) & (batch <= 1))
    pts = [np.array(r['x']) for r in res.values()] + list(batch)
    dist = np.linalg.norm(np.array(pts)[:, None] - np.array(pts)[None], axis=-1) + np.eye(len(pts)) * 9
    assert dist.min() >= J.MIN_SEP - 1e-12
    flat = {i: dict(valid=True, objective_mm=1.0, wave=(i - 1) // 4 + 1) for i in range(1, 45)}
    assert J.stalled(flat, 11) and not J.stalled(flat, 10)
    flat[40]['objective_mm'] = 0.9
    assert not J.stalled(flat, 11)


def test_failure_streak_detected_across_wave_boundaries():
    # attempt 1 ok, 2-9 fail, 10-12 ok: the last-eight window at wave 3 misses the streak
    res = {i: dict(valid=not (2 <= i <= 9)) for i in range(1, 13)}
    assert J.failure_streak(res) == 8
    assert J.failure_streak({i: dict(valid=not (2 <= i <= 8)) for i in range(1, 13)}) == 7
    assert J.failure_streak({}) == 0


def test_native_corrector_abort_is_solver_failure_not_infrastructure(tmp_path):
    d = tmp_path / 'case_00'; d.mkdir()
    (d / 'solver.log').write_text('terminate called\n  what():  zigzag_sequence: equilibrium corrector failed at cycle 29')
    base = dict(valid=False, failure_class='infrastructure', error='Solver exit code -6: x', directory=str(tmp_path))
    assert J.classify(base)['failure_class'] == 'solver'
    (d / 'solver.log').write_text('segfault')
    assert J.classify(base)['failure_class'] == 'infrastructure'
    assert J.classify(dict(base, error='Missing required state export'))['failure_class'] == 'infrastructure'
