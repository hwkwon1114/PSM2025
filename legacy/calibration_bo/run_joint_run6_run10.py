#!/usr/bin/env python
"""Joint Run6+Run10 calibration: 4 deformation + 2 hardening parameters, equal-weight symmetric RMS.

Protocol: docs/calibration/bayesian_calibration/convention_corrected_investigation.md
(section "Joint Run 6 + Run 10 calibration"). Run 9 is never read here.
Usage: run_joint_run6_run10.py run | evaluate <attempt> | summary
"""
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_convention_corrected as cc
from calibration.ewcal.common import Parameters, save_json
from calibration.ewcal.runner import Evaluator

# Optional stage-2 spec (JSON): {"base": rel dir, "arm": "impute"|"success", "beta": [lo, hi], "floor": [lo, hi]}.
# No spec -> the original campaign (run/joint_run6_run10, box beta 0.02-1.0, floor 0-1, failure imputation).
SPEC = json.loads(Path(os.environ['JOINT_SPEC']).read_text()) if os.environ.get('JOINT_SPEC') else {}
ARM = SPEC.get('arm', 'impute')
BASE_REL = SPEC.get('base', 'run/joint_run6_run10')
BASE = ROOT / BASE_REL
NAMES = ('gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor')
LOG = np.array([True, True, False, False, True, False])  # logarithmic coordinates
LO = np.array([5e-5, 1e-5, -1.0, -1.0, 0.02, 0.0])
HI = np.array([1e-2, 1e-3, 1.0, 1.0, 1.0, 1.0])
if SPEC:
    LO[4], HI[4] = SPEC['beta']
    LO[5], HI[5] = SPEC['floor']
    if 'gtop' in SPEC:
        LO[0], HI[0] = SPEC['gtop']
# Stage-3 (gbot -> 0): gbot coordinate log(gbot + s) so gbot = 0 is the lower bound; warm start from a finished
# campaign in the same box; first wave = given warm-start points with gbot set to 0.
SHIFT = np.zeros(6)
if 'gbot_shift' in SPEC:
    SHIFT[1] = SPEC['gbot_shift']
    LO[1], HI[1] = SPEC['gbot']
# Stage-4 reduced model: only SPEC['active'] are optimized (normalized x has len(active) entries); SPEC['fixed']
# pins the rest. With gbot = 0 the bottom eigenstrain is exactly zero, so ortho_bottom has no mechanical effect.
ACTIVE = [NAMES.index(n) for n in SPEC.get('active', NAMES)]
FIXED = SPEC.get('fixed', {})
D = len(ACTIVE)
assert set(FIXED) | {NAMES[i] for i in ACTIVE} == set(NAMES) and not set(FIXED) & {NAMES[i] for i in ACTIVE}


def _t(v):
    return np.where(LOG, np.log(np.where(LOG, v + SHIFT, 1.0)), v)
BASELINE = {'gtop': 0.00028835116227798787, 'gbot': 1.4587994001079146e-05,
            'ortho_top': -0.3989157893255133, 'ortho_bottom': 0.960306323937336,
            'beta': 0.2231435513142097, 'floor': 0.0}      # Attempt 209, beta/floor as fixed so far
BASELINE_CASE_RMS = {'run6': 0.6258310360506094, 'run10': 0.7191718391707898}  # convention_corrected shared-footprint surface RMS
BASELINE_TOL_MM = 1e-3
Q, WAVES, N_DESIGN_WAVES = 4, 15, 3                      # 12 design + 48 BO = 60 evaluations
MIN_SEP, SEED = 0.10, 2026
EARLY_STOP_FROM_WAVE, EARLY_WINDOW, EARLY_REL = 11, 4, 0.005
if 'initial_from_warm' in SPEC:
    WAVES, N_DESIGN_WAVES = SPEC['waves'], 1
    EARLY_STOP_FROM_WAVE, EARLY_WINDOW = SPEC['early_from'], SPEC['early_window']
if SPEC.get('fresh_lhs'):
    if 'warm_start' in SPEC or 'initial_from_warm' in SPEC:
        raise ValueError('A fresh LHS cannot have a warm start')
    WAVES, N_DESIGN_WAVES = SPEC['waves'], SPEC['design_waves']
    EARLY_STOP_FROM_WAVE, EARLY_WINDOW = SPEC['early_from'], SPEC['early_window']
    SEED = SPEC['seed']
MAX_CONSEC_FAIL = 8


def encode(values):
    values = dict(values, **FIXED)
    v = np.array([values[n] for n in NAMES], float)
    lo, hi, v = _t(LO), _t(HI), _t(v)
    x = ((v - lo) / (hi - lo))[ACTIVE]
    if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError('Parameters outside bounds')
    return np.clip(x, 0, 1)


def decode(x):
    x = np.asarray(x, float)
    if x.shape != (D,) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError(f'Expected {D} finite normalized parameters in [0,1]')
    lo, hi = _t(LO), _t(HI)
    full = np.zeros(6)
    full[ACTIVE] = x
    v = lo + full * (hi - lo)
    v = np.where(LOG, np.maximum(np.exp(v) - SHIFT, 0.0), v)
    out = dict(zip(NAMES, map(float, v)))
    out.update({k: float(val) for k, val in FIXED.items()})
    return out


def joint_config(values):
    """Both cases, one shared parameter set, identical settings; only the case blocks differ."""
    c6, c10 = cc.config('run6'), cc.config('run10')
    for key in set(c6) | set(c10):
        if key not in ('cases', 'output_dir', 'report_dir'):
            assert c6.get(key) == c10.get(key), f'Run6/Run10 setting differs: {key}'
    cfg = copy.deepcopy(c6)
    cfg['cases'] = [copy.deepcopy(c6['cases'][0]), copy.deepcopy(c10['cases'][0])]
    assert [c['id'] for c in cfg['cases']] == ['run6', 'run10'] and all(c['weight'] == 1 for c in cfg['cases'])
    cfg['hardening']['beta'], cfg['hardening']['floor'] = float(values['beta']), float(values['floor'])
    return cfg


def jread(path):
    return json.loads(Path(path).read_text())


def warm():
    """Finished-campaign records (same box except gbot extension), re-encoded in this codec, wave 0, attempt -id."""
    if 'warm_start' not in SPEC:
        return []
    out = []
    for p in sorted((ROOT / SPEC['warm_start'] / 'results').glob('a*.json')):
        r = classify(jread(p))
        if 'warm_require' in SPEC and any(r['parameters'][k] != v for k, v in SPEC['warm_require'].items()):
            continue
        out.append(dict(r, attempt=-r['attempt'], wave=0, x=encode(r['parameters']).tolist()))
    return out


def results():
    out = {}
    for p in sorted((BASE / 'results').glob('a*.json')):
        r = jread(p)
        out[r['attempt']] = classify(r)
    return out


def classify(r):
    """Evaluator labels a native solver abort 'infrastructure'. A native 'equilibrium corrector failed' abort
    is a numerical solver failure under the protocol (enters the GP as a failed point), so relabel it on read;
    the stored result files stay untouched. Any other infrastructure failure still stops the campaign."""
    if r['valid'] or r['failure_class'] != 'infrastructure' or not str(r.get('error')).startswith('Solver exit code'):
        return r
    for log in sorted(Path(r['directory']).glob('case_*/solver.log')):
        if 'zigzag_sequence: equilibrium corrector failed' in log.read_text(errors='replace')[-4000:]:
            return dict(r, failure_class='solver', reclassified_from='infrastructure')
    return r


def design():
    """Reproduce the configured design, including a wholly fresh LHS when requested."""
    path = BASE / 'design.json'
    if path.exists():
        return np.array(jread(path)['x'])
    if 'initial_from_warm' in SPEC:
        src = {a: jread(ROOT / SPEC['warm_start'] / 'results' / f'a{a:04d}.json') for a in SPEC['initial_from_warm']}
        pts = np.array([encode(dict(src[a]['parameters'], gbot=0.0)) for a in SPEC['initial_from_warm']])
        save_json(path, {'initial_from_warm': SPEC['initial_from_warm'], 'gbot': 0.0, 'x': pts.tolist()})
        return pts
    from scipy.stats import qmc
    fresh = SPEC.get('fresh_lhs', False)
    base = None if fresh else encode(BASELINE)
    for k in range(200):
        n = Q * N_DESIGN_WAVES - (0 if fresh else 1)
        pts = qmc.LatinHypercube(d=D, seed=SEED + k, optimization='random-cd').random(n)
        allp = pts if fresh else np.vstack([base, pts])
        d = np.linalg.norm(allp[:, None] - allp[None], axis=-1) + np.eye(len(allp)) * 9
        if d.min() >= MIN_SEP:
            save_json(path, {'seed': SEED + k, 'x': allp.tolist()})
            return allp
    raise RuntimeError('Could not build design satisfying separation')


def admissible(pt, others):
    return all(np.linalg.norm(pt - o) >= MIN_SEP for o in others)


def propose(res, wave):
    """qLogEI over -log(J); failures enter the GP only as imputed 1.5x worst valid J (heuristic)."""
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from calibration.ewcal.batch_bayes import fit_gp_model
    recs = list(res.values()) + warm()
    valid = [r for r in recs if r['valid']]
    if len(valid) < 2:
        raise RuntimeError('Need at least 2 valid observations')
    worst = max(r['objective_mm'] for r in valid)
    train = [dict(normalized_parameters=r['x'], objective_mm=r['objective_mm'] if r['valid'] else 1.5 * worst)
             for r in recs if r['valid'] or r['failure_class'] in ('solver', 'registration', 'timeout')]
    torch.manual_seed(SEED + wave); rng = np.random.default_rng(SEED + wave)
    if ARM == 'success':
        return propose_success(recs, wave)
    gp, best_f = fit_gp_model(train)
    acq = qLogExpectedImprovement(model=gp, best_f=best_f)
    bounds = torch.stack([torch.zeros(D, dtype=torch.double), torch.ones(D, dtype=torch.double)])
    cand, _ = optimize_acqf(acq, bounds=bounds, q=Q, num_restarts=10, raw_samples=512)
    taken = [np.array(r['x']) for r in recs]
    batch, repaired = [], 0
    for c in np.clip(cand.detach().cpu().numpy(), 0, 1):
        if not admissible(c, taken + batch):
            repaired += 1
            for _ in range(10000):
                c = rng.uniform(0, 1, D)
                if admissible(c, taken + batch):
                    break
            else:
                raise RuntimeError('No admissible replacement candidate')
        batch.append(c)
    return np.array(batch), repaired


def propose_success(recs, wave):
    """Arm 'success': LogEI on a valid-only GP + log P(success) (GP classifier on all attempted points), greedy q with
    Kriging-believer fantasies, same separation rule. Failures never enter the objective GP."""
    import warnings
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from calibration.ewcal.batch_bayes import fit_gp_model
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    warnings.filterwarnings('ignore')
    att = [r for r in recs if r['valid'] or r['failure_class'] in ('solver', 'registration', 'timeout')]
    X, y = np.array([r['x'] for r in att]), np.array([int(r['valid']) for r in att])
    valid = [r for r in att if r['valid']]
    gp, best_f = fit_gp_model([dict(normalized_parameters=r['x'], objective_mm=r['objective_mm']) for r in valid])
    rng = np.random.default_rng(SEED + wave)
    pool = qmc.Sobol(D, seed=SEED + wave).random(2 ** 14)
    top = np.array([r['x'] for r in sorted(valid, key=lambda r: r['objective_mm'])[:3]])
    local = np.clip(top[rng.integers(len(top), size=4000)] + rng.normal(0, 0.05, (4000, D)), 0, 1)
    pool = np.vstack([pool, local])
    if y.min() == y.max():
        logp = np.zeros(len(pool))
    else:
        k = ConstantKernel(1.0, (0.1, 10)) * Matern(length_scale=np.ones(D), length_scale_bounds=(0.1, 20), nu=2.5)
        gpc = GaussianProcessClassifier(k, n_restarts_optimizer=2, random_state=SEED + wave).fit(X, y)
        logp = np.log(np.clip(gpc.predict_proba(pool)[:, 1], 1e-3, 1))
    P = torch.tensor(pool, dtype=torch.double).unsqueeze(1)
    taken, batch, cur = [x for x in X], [], gp
    for _ in range(Q):
        with torch.no_grad():
            score = LogExpectedImprovement(cur, best_f=best_f)(P).numpy() + logp
        for i in np.argsort(-score):
            if admissible(pool[i], taken + batch):
                break
        else:
            raise RuntimeError('No admissible candidate in pool')
        batch.append(pool[i])
        with torch.no_grad():
            mu = cur.posterior(torch.tensor(pool[i:i + 1], dtype=torch.double)).mean
        cur = cur.condition_on_observations(torch.tensor(pool[i:i + 1], dtype=torch.double), mu)
    return np.array(batch), 0


def make_wave(res, wave):
    ids = list(range(Q * (wave - 1) + 1, Q * wave + 1))
    paths = [BASE / 'proposals' / f'a{a:04d}.json' for a in ids]
    if all(p.exists() for p in paths):
        return ids
    assert not any(p.exists() for p in paths), 'partial wave proposal'
    if wave <= N_DESIGN_WAVES:
        xs, info = design()[Q * (wave - 1):Q * wave], {'source': 'design'}
    else:
        xs, rep = propose(res, wave)
        info = {'source': 'qlogei', 'repaired': rep}
    for a, p, x in zip(ids, paths, xs):
        save_json(p, {'attempt': a, 'wave': wave, 'x': list(map(float, x)), 'parameters': decode(x), **info})
    return ids


def evaluate(att):
    prop = jread(BASE / 'proposals' / f'a{att:04d}.json')
    vals = decode(prop['x'])
    cfg = joint_config(vals)
    cfg['output_dir'] = f'{BASE_REL}/candidates/a{att:04d}'
    cfg['report_dir'] = cfg['output_dir'] + '/report'
    ev = Evaluator(cfg)
    p4 = {k: vals[k] for k in cfg['parameters']}
    if 'gbot_shift' in SPEC:   # the Evaluator's 4-D log codec cannot represent gbot = 0; pass exact values through
        ev.parameters = SimpleNamespace(decode=lambda _x: dict(p4))
        x4 = [0.0] * 4
    else:
        x4 = Parameters(cfg).encode(p4).tolist()
    rec = ev.evaluate(x4, use_cache=False, attempt_id=1)
    m = {c['id']: c['metrics'] for c in rec['cases']}
    keep = ('rms_mm', 'rms_symmetric_mm', 'p95_target_to_sim_mm', 'p95_sim_to_target_mm',
            'max_target_to_sim_mm', 'max_sim_to_target_mm')
    save_json(BASE / 'results' / f'a{att:04d}.json', {
        'attempt': att, 'wave': prop['wave'], 'x': prop['x'], 'parameters': vals, 'valid': bool(rec['valid']),
        'failure_class': rec.get('failure_class'), 'error': rec.get('error'),
        'objective_mm': rec['objective_mm'],
        'case_rms_mm': {c['id']: c['case_rms'] for c in rec['cases']},
        'case_metrics': {k: {n: v.get(n) for n in keep} for k, v in m.items()},
        'seconds': rec['seconds'], 'directory': rec['directory']})
    return rec


def run():
    for d in ('proposals', 'results', 'logs'):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', BLIS_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    env.pop('PYTHONOPTIMIZE', None)
    for wave in range(1, WAVES + 1):
        res = results()
        if failure_streak(res) >= MAX_CONSEC_FAIL:      # latched across restarts
            summary('stopped_consecutive_failures'); return
        ids = make_wave(res, wave)
        todo = [a for a in ids if a not in res]
        for a in todo:
            d = BASE / f'candidates/a{a:04d}/evaluations'
            if d.exists() and any(d.iterdir()):
                raise RuntimeError(f'Attempt {a} was interrupted; decide resume policy manually')
        procs = []
        for a in todo:
            log = open(BASE / 'logs' / f'a{a:04d}.log', 'ab')
            procs.append((a, subprocess.Popen([sys.executable, '-u', str(Path(__file__)), 'evaluate', str(a)],
                                              env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)))
        for a, p in procs:
            p.wait()
        res = results()
        missing = [a for a in ids if a not in res]
        if missing or any(res[a]['failure_class'] == 'infrastructure' for a in ids):
            raise RuntimeError(f'Wave {wave}: infrastructure/missing results {missing}; stopping for review')
        if wave == 1 and 'initial_from_warm' not in SPEC and not SPEC.get('fresh_lhs'):
            r1 = res[1]
            bad = {k: (r1['case_rms_mm'][k], v) for k, v in BASELINE_CASE_RMS.items()
                   if not r1['valid'] or abs(r1['case_rms_mm'][k] - v) > BASELINE_TOL_MM}
            if bad:
                raise RuntimeError(f'Baseline reproduction failed (new, old): {bad}')
        if failure_streak(res) >= MAX_CONSEC_FAIL:
            summary('stopped_consecutive_failures'); return
        if stalled(res, wave):
            summary('early_stop_no_improvement'); return
    summary('budget_complete')


def failure_streak(res):
    """Longest run of consecutive failed attempts (by attempt id) anywhere in the history."""
    longest = run = 0
    for a in sorted(res):
        run = 0 if res[a]['valid'] else run + 1
        longest = max(longest, run)
    return longest


def best_through(res, wave):
    v = [r['objective_mm'] for r in list(res.values()) + warm() if r['valid'] and r['wave'] <= wave]
    return min(v) if v else math.inf


def stalled(res, wave):
    if wave < EARLY_STOP_FROM_WAVE:
        return False
    now, before = best_through(res, wave), best_through(res, wave - EARLY_WINDOW)
    return (before - now) < EARLY_REL * before


def summary(status=None):
    res = results()
    valid = [r for r in res.values() if r['valid']]
    best = min(valid, key=lambda r: r['objective_mm']) if valid else None
    first_key = 'first_design_attempt' if SPEC.get('fresh_lhs') else 'baseline_attempt1'
    save_json(BASE / 'summary.json', {'status': status, 'evaluations': len(res), 'valid': len(valid),
              'failures': {k: sum(1 for r in res.values() if r['failure_class'] == k)
                           for k in ('solver', 'registration', 'timeout', 'infrastructure')},
              'best': best, first_key: res.get(1)})
    print('SUMMARY', status, len(res), best and best['objective_mm'], flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'evaluate':
        evaluate(int(sys.argv[2]))
    elif sys.argv[1] == 'summary':
        summary('manual')
    else:
        raise SystemExit(2)
