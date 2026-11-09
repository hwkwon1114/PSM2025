"""Unit tests for clean Run 2 and Run 3 Bayesian Optimization calibration (BO-RUN2-RUN3)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo.clean_calibration_runner import CalibrationContext


@pytest.mark.parametrize('spec_file,expected_case,expected_label,expected_alpha', [
    ('calibration/bo/run2_clean_24lhs.json', 'run9', 'Run 2', -1.0),
    ('calibration/bo/run3_clean_24lhs.json', 'run10', 'Run 3', -1.5),
])
def test_context_and_case_config(spec_file, expected_case, expected_label, expected_alpha):
    ctx = CalibrationContext(ROOT / spec_file)
    assert ctx.case == expected_case
    assert ctx.case_label == expected_label

    # K1: Roundtrip encode/decode
    rng = np.random.default_rng(123)
    u_orig = rng.uniform(0.05, 0.95, size=(10, 4))
    for u in u_orig:
        params = ctx.decode(u)
        assert 5e-5 <= params['gtop'] <= 0.01
        assert 1e-5 <= params['gbot'] <= 0.001
        assert -1.0 <= params['ortho_top'] <= 1.0
        assert -1.0 <= params['ortho_bottom'] <= 1.0
        assert np.isclose(params['beta'], 0.2231435513142097)
        assert np.isclose(params['floor'], 0.0)

        u_recon = ctx.encode(params)
        assert np.allclose(u, u_recon, atol=1e-7)

    # K2: Case config structure and operation
    sample_params = ctx.decode([0.3, 0.2, 0.1, 0.4])
    cfg = ctx.case_config(sample_params)
    assert len(cfg['cases']) == 1
    assert cfg['cases'][0]['id'] == expected_case
    assert np.isclose(cfg['cases'][0]['operation']['alpha_deg'], expected_alpha)
    assert np.isclose(cfg['cases'][0]['operation']['gtop'], sample_params['gtop'])
    assert np.isclose(cfg['cases'][0]['operation']['gbot'], sample_params['gbot'])
    assert np.isclose(cfg['cases'][0]['operation']['ortho_top'], sample_params['ortho_top'])
    assert np.isclose(cfg['cases'][0]['operation']['ortho_bottom'], sample_params['ortho_bottom'])
    assert np.isclose(cfg['hardening']['beta'], 0.2231435513142097)
    assert np.isclose(cfg['hardening']['floor'], 0.0)


def test_k3_deterministic_lhs():
    ctx2 = CalibrationContext(ROOT / 'calibration/bo/run2_clean_24lhs.json')
    ctx3 = CalibrationContext(ROOT / 'calibration/bo/run3_clean_24lhs.json')
    pts2 = ctx2.lhs_design(seed=20261030, n_points=24)
    pts3 = ctx3.lhs_design(seed=20261030, n_points=24)

    assert pts2.shape == (24, 4)
    assert np.allclose(pts2, pts3)

    for i in range(len(pts2)):
        for j in range(i + 1, len(pts2)):
            dist = np.linalg.norm(pts2[i] - pts2[j])
            assert dist >= ctx2.min_sep
