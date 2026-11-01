"""Stage-3 gbot->0 driver mode: codec reaches gbot=0, warm start re-encodes Arm A, forced first wave, passthrough evaluation."""
import importlib, json, os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def J(tmp_path, monkeypatch):
    monkeypatch.setenv('JOINT_SPEC', os.path.join(ROOT, 'calibration/bo/stage3_gbot0.json'))
    import calibration.bo.run_joint_run6_run10 as m
    m = importlib.reload(m)
    monkeypatch.setattr(m, 'BASE', tmp_path)
    yield m
    monkeypatch.delenv('JOINT_SPEC')
    importlib.reload(m)


def test_codec_includes_gbot_zero_and_roundtrips(J):
    p = dict(J.BASELINE)
    for g in (0.0, 1e-5, 1.4588e-5, 1e-3):
        p['gbot'] = g
        x = J.encode(p)
        assert J.decode(x)['gbot'] == pytest.approx(g, rel=1e-10, abs=1e-15)
    assert J.encode(dict(p, gbot=0.0))[1] == 0.0 and J.encode(dict(p, gbot=1e-3))[1] == pytest.approx(1.0)
    assert J.encode(dict(p, gbot=1e-5))[1] == pytest.approx(np.log(2) / np.log(101))
    with pytest.raises(ValueError):
        J.encode(dict(p, gbot=-1e-6))
    assert (J.WAVES, J.N_DESIGN_WAVES, J.EARLY_STOP_FROM_WAVE, J.EARLY_WINDOW) == (6, 1, 4, 3)


def test_warm_start_and_forced_first_wave(J):
    w = J.warm()
    a = json.load(open(os.path.join(ROOT, 'run/joint_stage2/arm_A/summary.json')))
    assert len(w) == a['evaluations'] and all(r['wave'] == 0 and r['attempt'] < 0 for r in w)
    for r in w:
        for k, v in J.decode(r['x']).items():
            assert v == pytest.approx(r['parameters'][k], rel=1e-9, abs=1e-14)
    d = J.design()
    src = {-r['attempt']: r['parameters'] for r in w}
    assert d.shape == (4, 6)
    for x, att in zip(d, [26, 43, 30, 25]):
        v = J.decode(x)
        assert v['gbot'] == 0.0
        for k in ('gtop', 'ortho_top', 'ortho_bottom', 'beta', 'floor'):
            assert v[k] == pytest.approx(src[att][k], rel=1e-10, abs=1e-14)
    # stall rule counts the warm-start best (Arm A a0026)
    assert J.best_through({}, 0) == pytest.approx(a['best']['objective_mm'])


def test_proposal_admissible_against_warm_and_own(J):
    d = J.design()
    res = {i + 1: dict(attempt=i + 1, wave=1, x=list(x), valid=True, failure_class=None,
                       objective_mm=0.6 + 0.01 * i) for i, x in enumerate(d)}
    batch, _ = J.propose(res, 2)
    pts = [np.array(r['x']) for r in list(res.values()) + J.warm()]
    for b in batch:
        assert all(np.linalg.norm(b - p) >= J.MIN_SEP - 1e-12 for p in pts)


def test_gbot_zero_reaches_solver_input(J):
    from calibration.ewcal.runner import sequence_json
    v = J.decode(J.design()[0])
    cfg = J.joint_config(v)
    s = sequence_json(cfg, cfg['cases'][0], {k: v[k] for k in cfg['parameters']})
    op = s['toolpaths'][0]['operation']
    assert op['gbot'] == 0.0 and op['gtop'] == pytest.approx(v['gtop'])
    assert (s['hardening']['beta'], s['hardening']['floor']) == (v['beta'], v['floor'])
