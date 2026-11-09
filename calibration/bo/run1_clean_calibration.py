#!/usr/bin/env python3
"""Clean Run 1 Bayesian Optimization calibration on the revised surface metric.

Replicates the Attempt 209 parameterization (4 active parameters: gtop, gbot,
ortho_top, ortho_bottom with fixed hardening beta=0.223, floor=0) directly on
the revised shared-footprint double-sided surface metric.

Uses a fresh 24-point Latin Hypercube Sample (6 waves of 4) followed by
success-aware Log-EI Bayesian Optimization in batches of 4.
"""
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_convention_corrected as cc
from calibration.ewcal.common import save_json
from calibration.ewcal.runner import Evaluator

SPEC_PATH = ROOT / 'calibration/bo/run1_clean_24lhs.json'
SPEC = json.loads(SPEC_PATH.read_text())

BASE_REL = SPEC['base']
BASE = ROOT / BASE_REL
ACTIVE_NAMES = tuple(SPEC['active'])
D = len(ACTIVE_NAMES)
Q = 4
N_DESIGN_WAVES = SPEC.get('design_waves', 6)
TOTAL_WAVES = SPEC.get('waves', 31)
SEED = SPEC['seed']
MIN_SEP = SPEC.get('min_separation', 0.05)
EARLY_FROM = SPEC.get('early_from', 16)
EARLY_WINDOW = SPEC.get('early_window', 6)
EARLY_REL = SPEC.get('early_rel', 0.005)
MAX_CONSEC_FAIL = 8
FAILURES = ('solver', 'registration', 'timeout', 'infrastructure')

PUBLIC_NAMES = {'run6': 'Run 1', 'run9': 'Run 2', 'run10': 'Run 3'}
HISTORICAL_NAMES = {'Run 1': 'run6', 'Run 2': 'run9', 'Run 3': 'run10'}

# Bounds
BOUNDS = {
    'gtop': tuple(SPEC['gtop']),
    'gbot': tuple(SPEC['gbot']),
    'ortho_top': tuple(SPEC['ortho_top']),
    'ortho_bottom': tuple(SPEC['ortho_bottom']),
}
FIXED = SPEC['fixed']


def decode(u):
    """Map normalized u in [0, 1]^4 to physical dictionary."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    gtop_lo, gtop_hi = BOUNDS['gtop']
    gbot_lo, gbot_hi = BOUNDS['gbot']
    ot_lo, ot_hi = BOUNDS['ortho_top']
    ob_lo, ob_hi = BOUNDS['ortho_bottom']

    gtop = float(np.exp(np.log(gtop_lo) + u[0] * (np.log(gtop_hi) - np.log(gtop_lo))))
    gbot = float(np.exp(np.log(gbot_lo) + u[1] * (np.log(gbot_hi) - np.log(gbot_lo))))
    ortho_top = float(ot_lo + u[2] * (ot_hi - ot_lo))
    ortho_bottom = float(ob_lo + u[3] * (ob_hi - ob_lo))

    out = {
        'gtop': gtop,
        'gbot': gbot,
        'ortho_top': ortho_top,
        'ortho_bottom': ortho_bottom,
    }
    out.update(FIXED)
    return out


def encode(p):
    """Map physical parameters to normalized u in [0, 1]^4."""
    gtop_lo, gtop_hi = BOUNDS['gtop']
    gbot_lo, gbot_hi = BOUNDS['gbot']
    ot_lo, ot_hi = BOUNDS['ortho_top']
    ob_lo, ob_hi = BOUNDS['ortho_bottom']

    u0 = (np.log(p['gtop']) - np.log(gtop_lo)) / (np.log(gtop_hi) - np.log(gtop_lo))
    u1 = (np.log(p['gbot']) - np.log(gbot_lo)) / (np.log(gbot_hi) - np.log(gbot_lo))
    u2 = (p['ortho_top'] - ot_lo) / (ot_hi - ot_lo)
    u3 = (p['ortho_bottom'] - ob_lo) / (ob_hi - ob_lo)
    return np.clip(np.array([u0, u1, u2, u3], dtype=float), 0.0, 1.0)


def case_config(values):
    """Generate configuration for single-case Run 1 forward solve."""
    cfg = cc.config('run6')
    for k in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
        cfg['parameters'][k]['value'] = float(values[k])
        cfg['cases'][0]['operation'][k] = float(values[k])
    cfg['hardening']['beta'] = float(values['beta'])
    cfg['hardening']['floor'] = float(values['floor'])
    return cfg


def jread(p):
    return json.loads(Path(p).read_text())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def admissible(pt, others):
    return all(np.linalg.norm(pt - o) >= MIN_SEP for o in others)


def lhs_design(seed=SEED, n_points=Q * N_DESIGN_WAVES):
    """Deterministic Latin Hypercube sample of n_points satisfying min separation."""
    from scipy.stats import qmc
    rng_seed = seed
    for attempt in range(500):
        sampler = qmc.LatinHypercube(d=D, seed=rng_seed + attempt)
        pts = sampler.random(n_points)
        ok = True
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.linalg.norm(pts[i] - pts[j]) < MIN_SEP:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return pts
    raise RuntimeError(f'Could not generate {n_points} LHS points satisfying MIN_SEP={MIN_SEP}')


def classify(res):
    if res.get('valid'):
        return dict(res, failure_class=None)
    err = str(res.get('error', '')).lower()
    if 'timeout' in err or 'timed out' in err:
        cls = 'timeout'
    elif any(k in err for k in ('solver', 'indefinite', 'nan', 'convergence', 'newton')):
        cls = 'solver'
    elif 'registration' in err:
        cls = 'registration'
    else:
        cls = 'infrastructure'
    return dict(res, failure_class=cls)


def load_results():
    """Load all saved evaluation results in attempt order."""
    results_dir = BASE / 'results'
    if not results_dir.exists():
        return {}
    out = {}
    for p in sorted(results_dir.glob('a*.json')):
        a = int(p.stem[1:])
        res = jread(p)
        out[a] = classify(res)
    return out


def propose(res, wave):
    """Success-aware LogEI proposal for batch of Q points."""
    import warnings
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from calibration.ewcal.batch_bayes import fit_gp_model
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    recs = list(res.values())
    valid = [r for r in recs if r['valid']]
    if len(valid) < 2:
        raise RuntimeError('Need at least 2 valid observations to fit GP')

    torch.manual_seed(SEED + wave)
    warnings.filterwarnings('ignore')

    att = [r for r in recs if r['valid'] or r['failure_class'] in ('solver', 'registration', 'timeout')]
    X = np.array([r['x'] for r in att])
    y = np.array([int(r['valid']) for r in att])

    gp, best_f = fit_gp_model([dict(normalized_parameters=r['x'], objective_mm=r['objective_mm']) for r in valid])

    rng = np.random.default_rng(SEED + wave)
    pool = qmc.Sobol(D, seed=SEED + wave).random(2**14)
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
    taken = [x for x in X]
    batch = []
    cur = gp

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

    return np.array(batch)


def make_wave(res, wave):
    """Generate or retrieve Q proposals for the given wave."""
    ids = list(range(Q * (wave - 1) + 1, Q * wave + 1))
    proposals_dir = BASE / 'proposals'
    proposals_dir.mkdir(parents=True, exist_ok=True)
    paths = [proposals_dir / f'a{a:04d}.json' for a in ids]

    if all(p.exists() for p in paths):
        return [jread(p) for p in paths]

    if wave <= N_DESIGN_WAVES:
        design = lhs_design()
        start = Q * (wave - 1)
        pts = design[start:start + Q]
        source = 'initial_lhs'
    else:
        pts = propose(res, wave)
        source = 'log_ei'

    proposals = []
    for a, pt in zip(ids, pts):
        prop = {
            'attempt': a,
            'wave': wave,
            'source': source,
            'x': pt.tolist(),
            'parameters': decode(pt)
        }
        save_json(proposals_dir / f'a{a:04d}.json', prop)
        proposals.append(prop)
    return proposals


def evaluate(attempt):
    """Execute a single attempt evaluation."""
    prop_path = BASE / 'proposals' / f'a{attempt:04d}.json'
    prop = jread(prop_path)
    cand_dir = BASE / 'candidates' / f'a{attempt:04d}'
    cand_dir.mkdir(parents=True, exist_ok=True)

    cfg = case_config(prop['parameters'])
    cfg['output_dir'] = str(cand_dir)

    evaluator = Evaluator(cfg)
    raw = evaluator.evaluate(prop['x'])

    res = {
        'attempt': attempt,
        'wave': prop['wave'],
        'source': prop['source'],
        'x': prop['x'],
        'parameters': prop['parameters'],
        'case': 'Run 1',
        'historical_case': 'run6',
        'valid': bool(raw.get('valid', False)),
        'error': raw.get('error'),
        'objective_mm': float(raw['objective_mm']) if raw.get('valid') else None,
        'seconds': raw.get('seconds'),
        'failure_class': raw.get('failure_class'),
    }
    if res['valid'] and 'cases' in raw and raw['cases']:
        res['case_rms_mm'] = {'Run 1': raw['cases'][0]['case_rms']}
        res['metrics'] = raw['cases'][0]['metrics']

    res = classify(res)
    results_dir = BASE / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / f'a{attempt:04d}.json', res)
    return res


def dispatch_wave(proposals):
    """Run Q candidate evaluations in parallel subprocesses."""
    procs = []
    env = os.environ.copy()
    env['BLIS_NUM_THREADS'] = '1'
    for v in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT'):
        env.pop(v, None)

    for p in proposals:
        cmd = [sys.executable, str(Path(__file__).resolve()), 'evaluate', str(p['attempt'])]
        procs.append((p['attempt'], subprocess.Popen(cmd, env=env)))

    for a, proc in procs:
        ret = proc.wait()
        if ret != 0:
            res_path = BASE / 'results' / f'a{a:04d}.json'
            if not res_path.exists():
                prop = jread(BASE / 'proposals' / f'a{a:04d}.json')
                save_json(res_path, {
                    'attempt': a,
                    'wave': prop['wave'],
                    'source': prop['source'],
                    'x': prop['x'],
                    'parameters': prop['parameters'],
                    'case': 'Run 1',
                    'historical_case': 'run6',
                    'valid': False,
                    'error': f'Subprocess crashed with exit code {ret}',
                    'failure_class': 'infrastructure',
                    'objective_mm': None
                })


def check_early_stopping(res, wave):
    """Check if relative improvement over last EARLY_WINDOW waves is < EARLY_REL."""
    if wave < EARLY_FROM:
        return False, None
    valid = [r for r in res.values() if r['valid']]
    if not valid:
        return False, None

    # Find best up to wave - EARLY_WINDOW and best up to wave
    window_best = min(r['objective_mm'] for r in valid if r['wave'] <= wave - EARLY_WINDOW)
    current_best = min(r['objective_mm'] for r in valid if r['wave'] <= wave)
    rel_impr = (window_best - current_best) / window_best
    if rel_impr < EARLY_REL:
        return True, f'Improvement {rel_impr:.4%} over last {EARLY_WINDOW} waves < {EARLY_REL:.2%}'
    return False, None


def failure_streak(res):
    """Longest streak of consecutive failures."""
    streak = 0
    max_streak = 0
    for a in sorted(res.keys()):
        if not res[a]['valid']:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def write_summary(res, status='running', reason=None):
    """Write summary.json and evaluations.jsonl."""
    BASE.mkdir(parents=True, exist_ok=True)
    valid = [r for r in res.values() if r['valid']]
    best = min(valid, key=lambda r: r['objective_mm']) if valid else None

    summary = {
        'status': status,
        'stop_reason': reason,
        'total_evaluations': len(res),
        'valid_evaluations': len(valid),
        'best': best,
        'failure_counts': {
            cls: sum(1 for r in res.values() if r.get('failure_class') == cls)
            for cls in FAILURES
        },
        'max_consecutive_failures': failure_streak(res),
    }
    save_json(BASE / 'summary.json', summary)

    lines = [json.dumps(res[a]) for a in sorted(res.keys())]
    (BASE / 'evaluations.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''))
    return summary


def evaluate_transfer(params, case_name):
    """Evaluate frozen best parameters on an excluded case (Run 2 or Run 3)."""
    pub_name = PUBLIC_NAMES.get(case_name, case_name)
    transfer_dir = BASE / 'transfer' / case_name
    transfer_dir.mkdir(parents=True, exist_ok=True)

    cfg = cc.config(case_name)
    for k in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
        cfg['parameters'][k]['value'] = float(params[k])
        cfg['cases'][0]['operation'][k] = float(params[k])
    cfg['hardening']['beta'] = float(params['beta'])
    cfg['hardening']['floor'] = float(params['floor'])
    cfg['output_dir'] = str(transfer_dir)

    evaluator = Evaluator(cfg)
    x = evaluator.parameters.encode(params)
    raw = evaluator.evaluate(x)

    res = {
        'case': pub_name,
        'historical_case': case_name,
        'parameters': params,
        'valid': bool(raw.get('valid', False)),
        'error': raw.get('error'),
        'objective_mm': float(raw['objective_mm']) if raw.get('valid') else None,
        'seconds': raw.get('seconds'),
        'failure_class': raw.get('failure_class'),
    }
    if res['valid'] and 'cases' in raw and raw['cases']:
        res['case_rms_mm'] = {pub_name: raw['cases'][0]['case_rms']}
        res['metrics'] = raw['cases'][0]['metrics']

    res = classify(res)
    save_json(BASE / 'transfer' / f'{case_name}_result.json', res)
    return res


def run():
    """Main BO loop with automatic transfer evaluation on Run 2 and Run 3 at finish."""
    BASE.mkdir(parents=True, exist_ok=True)
    res = load_results()
    current_attempt = max(res.keys(), default=0)
    current_wave = (current_attempt // Q) + 1

    print(f'Starting Run 1 clean calibration at wave {current_wave} (existing attempts: {len(res)})')
    print(f'Target: Run 1 (historical run6), 24 LHS + 100 BO = 124 total evaluations (31 waves of 4)')

    status = 'completed'
    reason = 'Completed all 31 waves (124 evaluations)'

    for wave in range(current_wave, TOTAL_WAVES + 1):
        if failure_streak(res) >= MAX_CONSEC_FAIL:
            print(f'Stopping: {MAX_CONSEC_FAIL} consecutive failures reached')
            status = 'stopped_failure_limit'
            reason = f'{MAX_CONSEC_FAIL} consecutive failures'
            break

        proposals = make_wave(res, wave)
        print(f'--- Wave {wave}/{TOTAL_WAVES} (attempts {proposals[0]["attempt"]}-{proposals[-1]["attempt"]}) ---')
        dispatch_wave(proposals)
        res = load_results()

        valid = [r for r in res.values() if r['valid']]
        best_obj = min((r['objective_mm'] for r in valid), default=float('nan'))
        print(f'Wave {wave} complete. Valid: {len(valid)}/{len(res)}. Best Run 1 RMS: {best_obj:.4f} mm')

        stop, stop_reason = check_early_stopping(res, wave)
        if stop:
            print(f'Early stopping triggered at wave {wave}: {stop_reason}')
            status = 'early_stopped'
            reason = stop_reason
            break

        write_summary(res, status='running')

    summary = write_summary(res, status=status, reason=reason)
    print(f'Run 1 calibration finished: status={status}. Best attempt: {summary["best"]["attempt"]} ({summary["best"]["objective_mm"]:.4f} mm)')

    # Perform frozen transfer evaluations on Run 2 (run9) and Run 3 (run10)
    if summary['best']:
        print('Running exploratory transfer evaluations on Run 2 (run9) and Run 3 (run10)...')
        t2 = evaluate_transfer(summary['best']['parameters'], 'run9')
        print(f'Transfer to Run 2 (run9): valid={t2["valid"]}, RMS={t2["objective_mm"]}')
        t3 = evaluate_transfer(summary['best']['parameters'], 'run10')
        print(f'Transfer to Run 3 (run10): valid={t3["valid"]}, RMS={t3["objective_mm"]}')
        summary['transfers'] = {
            'Run 2': {'historical_case': 'run9', 'valid': t2['valid'], 'objective_mm': t2['objective_mm']},
            'Run 3': {'historical_case': 'run10', 'valid': t3['valid'], 'objective_mm': t3['objective_mm']}
        }
        save_json(BASE / 'summary.json', summary)

    print('All calibration and transfer tasks completed successfully!')


def main():
    if len(sys.argv) < 2:
        print('Usage: run1_clean_calibration.py [run | evaluate <attempt> | summary]')
        sys.exit(1)
    action = sys.argv[1]
    if action == 'run':
        run()
    elif action == 'evaluate':
        if len(sys.argv) < 3:
            print('Specify attempt number: evaluate <attempt>')
            sys.exit(1)
        evaluate(int(sys.argv[2]))
    elif action == 'summary':
        res = load_results()
        s = write_summary(res, status='inspected')
        print(json.dumps(s, indent=2))
    else:
        raise ValueError(f'Unknown action: {action}')


if __name__ == '__main__':
    main()
