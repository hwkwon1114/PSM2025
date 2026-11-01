"""Stage-4 reduced 4-parameter mode: 4-D codec with fixed gbot/ortho_bottom, filtered warm start, forced wave, 4-D proposals."""
import importlib, json, os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def J(tmp_path, monkeypatch):
    monkeypatch.setenv('JOINT_SPEC', os.path.join(ROOT, 'calibration/bo/stage4_reduced4.json'))
    import calibration.bo.run_joint_run6_run10 as m
    m = importlib.reload(m)
    monkeypatch.setattr(m, 'BASE', tmp_path)
    yield m
    monkeypatch.delenv('JOINT_SPEC')
    importlib.reload(m)


def test_codec_4d(J):
    assert J.D == 4 and [J.NAMES[i] for i in J.ACTIVE] == ['gtop', 'ortho_top', 'beta', 'floor']
    p = dict(gtop=1.6e-4, gbot=7e-6, ortho_top=-0.6, ortho_bottom=0.9, beta=0.7, floor=0.2)
    x = J.encode(p)
    assert x.shape == (4,)
    d = J.decode(x)
    assert d['gbot'] == 0.0 and d['ortho_bottom'] == 0.0
    for k in ('gtop', 'ortho_top', 'beta', 'floor'):
        assert d[k] == pytest.approx(p[k], rel=1e-12)
    assert J.decode(np.ones(4))['beta'] == pytest.approx(1.0) and J.decode(np.zeros(4))['beta'] == pytest.approx(0.01)
    assert J.decode(np.ones(4))['floor'] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        J.encode(dict(p, beta=1.01))
    with pytest.raises(ValueError):
        J.decode(np.zeros(6))
    assert (J.WAVES, J.N_DESIGN_WAVES, J.EARLY_STOP_FROM_WAVE, J.EARLY_WINDOW, J.Q * J.WAVES) == (24, 1, 12, 4, 96)


def test_warm_filter_design_and_proposal(J):
    w = J.warm()
    assert sorted(-r['attempt'] for r in w) == [1, 2, 3, 4, 5, 9, 13, 16, 18, 22]
    assert all(r['parameters']['gbot'] == 0.0 and len(r['x']) == 4 for r in w)
    d = J.design()
    src = [json.load(open(os.path.join(ROOT, f'run/joint_stage3_gbot0/results/a{a:04d}.json'))) for a in (19, 24, 17, 14)]
    for x, s in zip(d, src):
        v = J.decode(x)
        assert v['gbot'] == 0.0
        for k in ('gtop', 'ortho_top', 'beta', 'floor'):
            assert v[k] == pytest.approx(s['parameters'][k], rel=1e-12)
    batch, _ = J.propose({}, 2)
    assert batch.shape == (4, 4)
    taken = [np.array(r['x']) for r in w]
    for i, b in enumerate(batch):
        assert all(np.linalg.norm(b - t) >= J.MIN_SEP for t in taken + list(batch[:i]))
