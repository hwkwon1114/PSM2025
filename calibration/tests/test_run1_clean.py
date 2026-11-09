"""Unit tests for clean Run 1 Bayesian Optimization calibration."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import calibration.bo.run1_clean_calibration as r1c


def test_decode_encode_roundtrip():
    rng = np.random.default_rng(42)
    u_orig = rng.uniform(0.05, 0.95, size=(10, 4))
    for u in u_orig:
        params = r1c.decode(u)
        assert params['gtop'] >= 5e-5 and params['gtop'] <= 0.01
        assert params['gbot'] >= 1e-5 and params['gbot'] <= 0.001
        assert params['ortho_top'] >= -1.0 and params['ortho_top'] <= 1.0
        assert params['ortho_bottom'] >= -1.0 and params['ortho_bottom'] <= 1.0
        assert np.isclose(params['beta'], 0.2231435513142097)
        assert np.isclose(params['floor'], 0.0)

        u_recon = r1c.encode(params)
        assert np.allclose(u, u_recon, atol=1e-7)


def test_case_config_structure():
    params = r1c.decode([0.5, 0.5, 0.5, 0.5])
    cfg = r1c.case_config(params)
    assert len(cfg['cases']) == 1
    assert cfg['cases'][0]['id'] == 'run6'
    assert cfg['cases'][0]['repeat'] == 29
    assert cfg['cases'][0]['operation']['alpha_deg'] == -1
    assert np.isclose(cfg['cases'][0]['operation']['gtop'], params['gtop'])
    assert np.isclose(cfg['cases'][0]['operation']['gbot'], params['gbot'])
    assert np.isclose(cfg['cases'][0]['operation']['ortho_top'], params['ortho_top'])
    assert np.isclose(cfg['cases'][0]['operation']['ortho_bottom'], params['ortho_bottom'])
    assert np.isclose(cfg['hardening']['beta'], 0.2231435513142097)
    assert np.isclose(cfg['hardening']['floor'], 0.0)


def test_lhs_design_24_points():
    pts = r1c.lhs_design(seed=20261030, n_points=24)
    assert pts.shape == (24, 4)
    assert np.all(pts >= 0.0) and np.all(pts <= 1.0)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[j])
            assert dist >= r1c.MIN_SEP, f"LHS points {i} and {j} too close: {dist} < {r1c.MIN_SEP}"


def test_case_mapping():
    assert r1c.PUBLIC_NAMES['run6'] == 'Run 1'
    assert r1c.PUBLIC_NAMES['run9'] == 'Run 2'
    assert r1c.PUBLIC_NAMES['run10'] == 'Run 3'
    assert r1c.HISTORICAL_NAMES['Run 1'] == 'run6'
    assert r1c.HISTORICAL_NAMES['Run 2'] == 'run9'
    assert r1c.HISTORICAL_NAMES['Run 3'] == 'run10'


def test_failure_classification():
    assert r1c.classify({'valid': True})['failure_class'] is None
    assert r1c.classify({'valid': False, 'error': 'Solver timed out'})['failure_class'] == 'timeout'
    assert r1c.classify({'valid': False, 'error': 'indefinite curvature in Newton'})['failure_class'] == 'solver'
    assert r1c.classify({'valid': False, 'error': 'Registration failed'})['failure_class'] == 'registration'
