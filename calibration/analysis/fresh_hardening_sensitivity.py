"""Exploratory hardening sensitivity from saved Run 1+3 solves; no solver or Run 2."""
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPEC = ROOT / 'calibration/bo/fresh_lhs_joint4.json'
os.environ['JOINT_SPEC'] = str(SPEC)
from calibration.bo import run_joint_run6_run10 as joint

OUT = ROOT / 'run/diagnostics/fresh_hardening_sensitivity_20261027.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(X, y):
    kernel = ConstantKernel(1.0, (0.1, 10)) * Matern(np.ones(4), (0.05, 20), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True,
                                  n_restarts_optimizer=1, random_state=2026)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        gp.fit(X, np.log(y))
    return gp, [str(item.message) for item in w]


def selected(r):
    return {'attempt': r['attempt'], 'wave': r['wave'], 'J_mm': r['objective_mm'],
            'Run1_mm': r['case_rms_mm']['run6'], 'Run3_mm': r['case_rms_mm']['run10'],
            'parameters': {k: r['parameters'][k] for k in ('gtop', 'ortho_top', 'beta', 'floor')}}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = list(joint.results().values())
    summary = json.loads((joint.BASE / 'summary.json').read_text())
    valid = [r for r in rows if r['valid']]
    assert len(rows) == summary['evaluations'] == 88 and len(valid) == summary['valid'] == 86
    assert sum(r['failure_class'] == 'solver' for r in rows) == 2
    assert not any(r['failure_class'] == 'infrastructure' for r in rows)
    for r in valid:
        assert np.allclose(joint.encode(r['parameters']), r['x'], atol=1e-10, rtol=0)
        assert r['objective_mm'] > 0
    best = min(valid, key=lambda r: r['objective_mm'])
    assert best['attempt'] == summary['best']['attempt'] == 84
    X = np.array([r['x'] for r in valid], float)
    J = np.array([r['objective_mm'] for r in valid], float)
    waves = np.array([r['wave'] for r in valid])
    assert len(np.unique(X, axis=0)) == len(X)
    gp, fitted_warnings = fit(X, J)
    early = waves <= 16
    gp_early, early_warnings = fit(X[early], J[early])
    pred, sd = gp_early.predict(X[~early], return_std=True)
    delta = pred - np.log(J[~early])
    check = {'train_valid_waves_1_through_16': int(early.sum()),
             'later_valid_waves_17_through_22': int((~early).sum()),
             'log_J_rmse': float(np.sqrt(np.mean(delta**2))),
             'nominal_95_percent_coverage': float(np.mean(np.abs(delta) <= 1.96*sd)),
             'interval_type': 'pointwise latent predictive interval, fitted hyperparameters',
             'limit': 'valid-only adaptive later observations; not independent confirmation'}
    x0 = joint.encode(best['parameters'])
    points = {'beta': [0.01, 0.02, 0.05, 0.1, 0.2, 0.49],
              'floor': [0.0, 0.1, 0.2, 0.3, 0.4, 0.45599149353802204, 0.5]}
    slices = {}
    for name, values in points.items():
        queries = []
        for value in values:
            params = dict(best['parameters'], **{name: value})
            x = joint.encode(params)
            mu, std = gp.predict(x[None], return_std=True)
            queries.append({'value': value, 'predicted_median_J_mm': float(np.exp(mu[0])),
                            'lower_95_mm': float(np.exp(mu[0] - 1.96*std[0])),
                            'upper_95_mm': float(np.exp(mu[0] + 1.96*std[0])),
                            'nearest_valid_4d_normalized_distance': float(np.min(np.linalg.norm(X - x, axis=1)))})
        slices[name] = queries
    near = [r for r in valid if r['objective_mm'] <= 1.05 * best['objective_mm']]
    near_ranges = {name: [float(min(r['parameters'][name] for r in near)),
                          float(max(r['parameters'][name] for r in near))]
                   for name in ('gtop', 'ortho_top', 'beta', 'floor')}
    b, f = best['parameters']['beta'], best['parameters']['floor']
    factor = lambda beta, floor, n: floor + (1-floor)*np.exp(-beta*n)
    law = {'previous_hits': 50, 'at_best': float(factor(b, f, 50)),
           'beta_doubled_others_fixed': float(factor(2*b, f, 50)),
           'floor_minus_0.1_others_fixed': float(factor(b, f-0.1, 50)),
           'floor_plus_0.04_others_fixed': float(factor(b, f+0.04, 50))}
    out = {'snapshot': {'summary_status': summary['status'], 'attempts': len(rows),
                        'valid': len(valid), 'numerical_solver_failures': 2,
                        'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                            [SPEC, joint.BASE / 'summary.json', *sorted((joint.BASE / 'results').glob('a*.json'))]}},
           'selected': selected(best), 'top_ten': [selected(r) for r in sorted(valid, key=lambda r:r['objective_mm'])[:10]],
           'within_5_percent': {'count': len(near), 'threshold_mm': float(1.05*best['objective_mm']),
                                'parameter_min_max': near_ranges,
                                'definition': 'descriptive only; not statistical equivalence'},
           'gp': {'kernel': str(gp.kernel_), 'warnings': fitted_warnings,
                  'early_kernel': str(gp_early.kernel_), 'early_warnings': early_warnings,
                  'forward_check': check, 'slices_at_best': slices},
           'hardening_factor_example': law,
           'script_sha256': digest(Path(__file__))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k,v in out.items() if k != 'snapshot'}, indent=2))
    print('written:', OUT)


if __name__ == '__main__':
    main()
