"""Budgeted bounded fitting and local identifiability diagnostics."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from .common import save_json, NAMES

def remember_best(evaluator, record):
    if not record['valid']:
        return
    path = evaluator.base/'best.json'
    old = json.loads(path.read_text()) if path.exists() else None
    if old is None or old['identity'] != evaluator.identity or record['objective_mm'] < old['objective_mm']:
        save_json(path, record)

def residual(record):
    vectors = []
    total_weight = sum(c['weight'] for c in record['cases'])
    for case in record['cases']:
        with np.load(Path(case['directory'])/'comparison.npz') as data:
            vectors.append((data['closest_mm']-data['target_mm']).ravel() *
                           np.sqrt(case['weight']/total_weight/len(data['target_mm'])))
    return np.concatenate(vectors)

def sensitivity(evaluator):
    x = evaluator.parameters.initial()
    baseline = evaluator.evaluate(x); remember_best(evaluator, baseline)
    if not baseline['valid']:
        raise RuntimeError('Initial evaluation failed; resolve it before sensitivity analysis')
    columns, trials = [], []
    step = evaluator.config['optimization']['sensitivity_step']
    for j, name in enumerate(NAMES):
        low, high = x.copy(), x.copy()
        low[j] = max(0, x[j]-step); high[j] = min(1, x[j]+step)
        a, b = evaluator.evaluate(low), evaluator.evaluate(high)
        for r in (a, b): remember_best(evaluator, r)
        if not a['valid'] or not b['valid']:
            raise RuntimeError(f'Cannot form sensitivity for {name}: a perturbation failed')
        columns.append((residual(b)-residual(a))/(high[j]-low[j]))
        trials.append({'parameter': name, 'low': a['directory'], 'high': b['directory']})
    jac = np.column_stack(columns)
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    denom = np.outer(np.linalg.norm(jac, axis=0), np.linalg.norm(jac, axis=0))
    similarity = np.divide(jac.T @ jac, denom, out=np.zeros((4,4)), where=denom > 0)
    info = {'parameters': list(NAMES), 'baseline': baseline['directory'], 'trials': trials,
            'singular_values_mm_per_normalized_parameter': s.tolist(),
            'weakest_combination_normalized': dict(zip(NAMES, vt[-1].tolist())),
            'column_cosine_similarity': similarity.tolist(),
            'condition_number': float(s[0]/s[-1]) if s[-1] > 1e-15 else None,
            'note': 'Local, pose-profiled sensitivity only; not parameter confidence intervals.'}
    save_json(evaluator.base/'sensitivity.json', info)
    np.savez_compressed(evaluator.base/'sensitivity.npz', jacobian=jac, singular_values=s, right_vectors=vt)
    return info

def optimize(evaluator, resume=False):
    settings = evaluator.config['optimization']
    budget = settings['max_evaluations']
    if budget < 2:
        raise ValueError('max_evaluations must be at least 2')
    records = []
    def objective(x):
        record = evaluator.evaluate(np.clip(x, 0, 1))
        records.append(record); remember_best(evaluator, record)
        return record['objective_mm']
    start = evaluator.parameters.initial()
    bestpath = evaluator.base/'best.json'
    if resume and bestpath.exists():
        old = json.loads(bestpath.read_text())
        if old['identity'] != evaluator.identity:
            raise ValueError('Resume identity differs (config, executable, target, or Python source changed)')
        start = np.array(old['normalized_parameters'])
    objective(start)
    if not records[-1]['valid']:
        raise RuntimeError('Starting evaluation failed. Resolve before optimization.')
    coarse = min(settings['coarse_samples'], max(0, budget-2))
    if coarse:
        samples = qmc.LatinHypercube(d=4, seed=settings['seed']).random(coarse)
        for x in samples: objective(x)
    best = min((r for r in records if r['valid']), key=lambda r: r['objective_mm'])
    remaining = budget-len(records)
    result = minimize(objective, best['normalized_parameters'], method='Powell', bounds=[(0,1)]*4,
                      options={'maxfev': remaining, 'xtol': settings['xtol'], 'ftol': settings['ftol']})
    best = min((r for r in records if r['valid']), key=lambda r: r['objective_mm'])
    summary = {'best': best, 'optimizer_success': bool(result.success), 'message': str(result.message),
               'objective_calls': len(records), 'fresh_solver_evaluations': evaluator.attempts,
               'failed_calls': sum(not r['valid'] for r in records),
               'note': 'A budget-limited pilot fit; optimizer success does not establish uniqueness or validation.'}
    save_json(evaluator.base/'optimization.json', summary)
    return summary
