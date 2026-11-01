#!/usr/bin/env python
"""Joint Run 1/3 calibration with scalar or multiobjective acquisition.

JOINT_SPEC selects the fresh scalar search or an isolated Pareto continuation.
Run 2 is not read. Scientific execution requires the applicable review.
Usage: python calibration/bo/joint_calibration.py prepare | run | evaluate <attempt> | summary
"""
import copy
import hashlib
import itertools
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
from calibration.ewcal.common import save_json
from calibration.ewcal.runner import Evaluator

SPEC_PATH = Path(os.environ.get('JOINT_SPEC', ROOT / 'calibration/bo/fresh_lhs_joint4.json'))
SPEC = json.loads(SPEC_PATH.read_text())
MULTI = SPEC.get('method') == 'multiobjective'
if (SPEC.get('method', 'scalar') not in ('scalar', 'multiobjective') or SPEC.get('arm') != 'success'
        or 'warm_start' in SPEC or 'initial_from_warm' in SPEC
        or SPEC.get('active') != ['gtop', 'ortho_top', 'beta', 'floor']
        or SPEC.get('fixed') != {'gbot': 0.0, 'ortho_bottom': 0.0}
        or (not MULTI and not SPEC.get('fresh_lhs'))
        or (MULTI and (SPEC.get('fresh_lhs') or SPEC.get('seed_source') != 'run/fresh_lhs_joint4'))):
    raise ValueError('Expected the fresh four-parameter success-aware specification or its explicit multiobjective continuation')
BASE_REL = SPEC['base']
BASE = ROOT / BASE_REL
NAMES = ('gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor')
LOG = np.array([True, True, False, False, True, False])
LO = np.array([5e-5, 1e-5, -1.0, -1.0, 0.02, 0.0])
HI = np.array([1e-2, 1e-3, 1.0, 1.0, 1.0, 1.0])
LO[0], HI[0] = SPEC['gtop']
LO[1], HI[1] = SPEC['gbot']
LO[4], HI[4] = SPEC['beta']
LO[5], HI[5] = SPEC['floor']
SHIFT = np.zeros(6)
SHIFT[1] = SPEC['gbot_shift']
ACTIVE = [NAMES.index(n) for n in SPEC['active']]
FIXED = SPEC['fixed']
D = len(ACTIVE)
Q, WAVES, N_DESIGN_WAVES = 4, SPEC.get('waves'), SPEC.get('design_waves', 0)
MIN_SEP, SEED = SPEC.get('min_separation', 0.10), SPEC['seed']
EARLY_STOP_FROM_WAVE, EARLY_WINDOW, EARLY_REL = SPEC.get('early_from'), SPEC.get('early_window'), 0.005
MAX_CONSEC_FAIL = 8


def _t(v):
    return np.where(LOG, np.log(np.where(LOG, v + SHIFT, 1.0)), v)


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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_results():
    """Validate and load the completed fresh-search pairs without changing them."""
    if not MULTI:
        return {}
    from calibration.bo.pareto_bayes import CASES, FAILURES, case_errors
    source = ROOT / SPEC['seed_source']
    source_spec = jread(ROOT / 'calibration/bo/fresh_lhs_joint4.json')
    for key in ('active', 'fixed', 'gtop', 'gbot', 'gbot_shift', 'beta', 'floor'):
        if source_spec[key] != SPEC[key]:
            raise ValueError(f'Seed and continuation differ in {key}')
    summary = jread(source / 'summary.json')
    paths = sorted((source / 'results').glob('a*.json'))
    if summary['status'] != 'early_stop_no_improvement' or len(paths) != 88 or summary['valid'] != 86:
        raise ValueError('Expected the completed 88-attempt fresh search with 86 valid pairs')
    out = {}
    expected_sources = {name: sha256(ROOT / 'calibration/ewcal' / name)
                        for name in ('common.py', 'geometry.py', 'material_frame.py')}
    old_runner = sha256(ROOT / 'run/diagnostics/run2_blis_a0084_r1/reviewed_sources/runner.py')
    for path in paths:
        r = jread(path)
        a = r['attempt']
        if path.stem != f'a{a:04d}' or a in (-i for i in out):
            raise ValueError(f'Invalid or duplicate seed attempt: {path}')
        vals = decode(r['x'])
        if any(not math.isclose(vals[k], float(r['parameters'][k]), rel_tol=1e-10, abs_tol=1e-12)
               for k in NAMES):
            raise ValueError(f'Seed parameter codec mismatch: {path}')
        candidate = source / 'candidates' / f'a{a:04d}'
        raw_path = candidate / 'evaluations/attempt_0001/result.json'
        raw = jread(raw_path)
        identity_path = candidate / 'cache' / raw['identity'] / 'identity.json'
        identity = jread(identity_path)
        expected_cfg = joint_config(vals)
        expected_cfg['output_dir'] = f"{SPEC['seed_source']}/candidates/a{a:04d}"
        expected_cfg['report_dir'] = expected_cfg['output_dir'] + '/report'
        if (hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != raw['identity']
                or identity['config'] != expected_cfg or identity['executable'] != cc.BINARY_SHA
                or identity['targets'] != {c: cc.TARGET_SHA[c] for c in CASES}
                or any(identity['python_sources'].get(k) != v for k, v in expected_sources.items())
                or identity['python_sources'].get('runner.py') != old_runner
                or any(not math.isclose(vals[k], float(raw['parameters'][k]), rel_tol=1e-10, abs_tol=1e-12)
                       for k in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'))
                or bool(raw['valid']) != bool(r['valid'])):
            raise ValueError(f'Seed scientific identity mismatch: {path}')
        if r['valid']:
            errors = case_errors(r)
            expected = math.sqrt(float(np.mean(errors ** 2)))
            if (not math.isclose(float(r['objective_mm']), expected, rel_tol=1e-10)
                    or [c['id'] for c in raw['cases']] != list(CASES)
                    or any(not math.isclose(float(c['case_rms']), float(e), rel_tol=1e-10)
                           for c, e in zip(raw['cases'], errors))):
                raise ValueError(f'Seed case or objective mismatch: {path}')
        else:
            r = classify(r)
            if r.get('failure_class') not in FAILURES:
                raise ValueError(f'Unrecognized seed failure: {path}')
        out[-a] = dict(r, attempt=-a, original_attempt=a, wave=0, source='seed')
    if (set(out) != set(range(-88, 0)) or sum(r['valid'] for r in out.values()) != 86
            or not np.allclose(np.max([case_errors(r) for r in out.values() if r['valid']], axis=0),
                               SPEC['reference_mm'], rtol=1e-12, atol=0)):
        raise ValueError('Seed IDs, valid count or frozen reference do not match the protocol')
    return out


def plan_files(seed):
    """Inputs whose identity affects this continuation; the seed remains in place."""
    source = ROOT / SPEC['seed_source']
    paths = [Path(__file__), SPEC_PATH, ROOT / 'calibration/bo/pareto_bayes.py',
             ROOT / 'calibration/bo/run_multiobjective_joint4.sbatch',
             ROOT / 'calibration/tests/test_multiobjective_joint4.py',
             ROOT / 'calibration/bo/fresh_lhs_joint4.json', ROOT / 'calibration/bo/run_convention_corrected.py',
             ROOT / 'shellsim/configs/default_adaptive_cutback.json', ROOT / 'bin/shell',
             ROOT / 'run/diagnostics/run2_blis_a0084_r1/reviewed_sources/runner.py', source / 'summary.json']
    paths.extend((ROOT / 'calibration/ewcal').glob('*.py'))
    for case in ('run6', 'run10'):
        paths.append(ROOT / cc.REQUESTS[case])
        target = ROOT / f'run/putong_convention_check/preserved_samples/{case}.npz'
        paths.extend((target, target.with_suffix('.json')))
    for r in seed.values():
        a = r['original_attempt']
        candidate = source / 'candidates' / f'a{a:04d}'
        raw_path = candidate / 'evaluations/attempt_0001/result.json'
        raw = jread(raw_path)
        paths.extend((source / 'results' / f'a{a:04d}.json', raw_path,
                      candidate / 'cache' / raw['identity'] / 'identity.json'))
        if not r['valid']:
            paths.extend((candidate / 'evaluations/attempt_0001').glob('case_*/solver.log'))
    return sorted({str(Path(p).absolute()) for p in paths})


def software_versions():
    import botorch, scipy, sklearn, torch, vtk
    return {'numpy': np.__version__, 'scipy': scipy.__version__, 'vtk': vtk.vtkVersion.GetVTKVersion(),
            'torch': torch.__version__, 'botorch': botorch.__version__, 'sklearn': sklearn.__version__}


def prepare():
    """Pin the seed and scientific inputs without launching a shell solve."""
    if not MULTI:
        raise ValueError('Preparation is only defined for the multiobjective continuation')
    if BASE.exists() and any(BASE.iterdir()):
        raise FileExistsError(f'Output already exists: {BASE}')
    seed = seed_results()
    cfg = joint_config(decode([0.5] * D))
    identity, _ = Evaluator.identity_inputs(cfg)
    plan = {'seed_total': len(seed), 'seed_valid': sum(r['valid'] for r in seed.values()),
            'reference_mm': SPEC['reference_mm'], 'evaluator_identity': identity, 'versions': software_versions(),
            'files': {p: sha256(p) for p in plan_files(seed)}}
    BASE.mkdir(parents=True, exist_ok=True)
    save_json(BASE / 'plan.json', plan)
    print(f'Prepared {BASE / "plan.json"}: {plan["seed_valid"]} valid paired seed observations; no solver launched')


def verify_plan():
    plan = jread(BASE / 'plan.json')
    for path, expected in plan['files'].items():
        if sha256(path) != expected:
            raise RuntimeError(f'Reviewed input changed: {path}')
    if plan['reference_mm'] != SPEC['reference_mm'] or plan['versions'] != software_versions():
        raise RuntimeError('Reviewed reference or software versions changed')
    return plan


def enough_slurm_time():
    """Avoid launching two 1-hour case solves into an expiring Slurm allocation."""
    if 'SLURM_JOB_END_TIME' not in os.environ:
        raise RuntimeError('SLURM_JOB_END_TIME is required for the continuation')
    return int(os.environ['SLURM_JOB_END_TIME']) - time.time() >= 2 * 3600 + 300


def design():
    """Reproduce the configured fresh LHS, without historical points."""
    path = BASE / 'design.json'
    if path.exists():
        return np.array(jread(path)['x'])
    from scipy.stats import qmc
    for k in range(200):
        pts = qmc.LatinHypercube(d=D, seed=SEED + k, optimization='random-cd').random(Q * N_DESIGN_WAVES)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts)) * 9
        if d.min() >= MIN_SEP:
            save_json(path, {'seed': SEED + k, 'x': pts.tolist()})
            return pts
    raise RuntimeError('Could not build design satisfying separation')


def admissible(pt, others):
    return all(np.linalg.norm(pt - o) >= MIN_SEP for o in others)


def propose(res, wave):
    """Success-aware LogEI: valid-only objective GP, success classifier and greedy batch fantasies."""
    import warnings
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from calibration.ewcal.batch_bayes import fit_gp_model
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    recs = list(res.values())
    if sum(bool(r['valid']) for r in recs) < 2:
        raise RuntimeError('Need at least 2 valid observations')
    torch.manual_seed(SEED + wave)
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
        if MULTI:
            proposals = [jread(p) for p in paths]
            for a, p in zip(ids, proposals):
                vals = decode(p['x'])
                if (p['attempt'] != a or p['wave'] != wave or p.get('source') != 'greedy_qlogehvi'
                        or any(not math.isclose(vals[k], float(p['parameters'][k]), rel_tol=1e-10, abs_tol=1e-12)
                               for k in NAMES)):
                    raise ValueError(f'Invalid preserved proposal {a}')
            for a, p in zip(ids, proposals):
                others = [r['x'] for r in res.values() if r['attempt'] != a]
                others += [q['x'] for b, q in zip(ids, proposals) if b != a]
                if not admissible(np.asarray(p['x'], dtype=float), others):
                    raise ValueError(f'Preserved proposal {a} violates separation')
        return ids
    assert not any(p.exists() for p in paths), 'partial wave proposal'
    if MULTI:
        from calibration.bo.pareto_bayes import propose as propose_multiobjective
        xs, scores = propose_multiobjective(list(res.values()), SPEC['reference_mm'], seed=SEED + wave,
                                             batch_size=Q, min_sep=MIN_SEP)
        infos = [{'source': 'greedy_qlogehvi', 'log_ehvi_times_success': score} for score in scores]
    elif wave <= N_DESIGN_WAVES:
        xs = design()[Q * (wave - 1):Q * wave]
        infos = [{'source': 'design'}] * Q
    else:
        xs, rep = propose(res, wave)
        infos = [{'source': 'qlogei', 'repaired': rep}] * Q
    for a, p, x, info in zip(ids, paths, xs, infos):
        save_json(p, {'attempt': a, 'wave': wave, 'x': list(map(float, x)), 'parameters': decode(x), **info})
    return ids


def evaluate(att):
    if MULTI:
        if os.environ.get('BLIS_NUM_THREADS') != '1' or any(
                k in os.environ for k in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT')):
            raise RuntimeError('The solver process requires the approved single-thread BLIS environment')
    prop = jread(BASE / 'proposals' / f'a{att:04d}.json')
    vals = decode(prop['x'])
    cfg = joint_config(vals)
    cfg['output_dir'] = f'{BASE_REL}/candidates/a{att:04d}'
    cfg['report_dir'] = cfg['output_dir'] + '/report'
    ev = Evaluator(cfg)
    p4 = {k: vals[k] for k in cfg['parameters']}
    # The Evaluator's logarithmic codec cannot represent gbot = 0; pass exact values through.
    ev.parameters = SimpleNamespace(decode=lambda _x: dict(p4))
    x4 = [0.0] * 4
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
    summary_path = BASE / 'summary.json'
    if summary_path.is_file() and jread(summary_path).get('status') in (
            'budget_complete', 'early_stop_no_improvement', 'stopped_consecutive_failures',
            'pareto_plateau', 'allocation_limit'):
        raise RuntimeError('Completed campaign is preserved; use a new output directory for any authorized search')
    seed = seed_results() if MULTI else {}
    if MULTI:
        verify_plan()
    for d in ('proposals', 'results', 'logs'):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', BLIS_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    env.pop('PYTHONOPTIMIZE', None)
    for key in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT'):
        env.pop(key, None)
    waves = itertools.count(1) if MULTI else range(1, WAVES + 1)
    for wave in waves:
        if MULTI:
            verify_plan()
            if not enough_slurm_time():
                summary('allocation_limit', seed); return
        res = results()
        combined = {**seed, **res}
        if failure_streak(combined) >= MAX_CONSEC_FAIL:      # latched across restarts
            summary('stopped_consecutive_failures', seed); return
        ids = make_wave(combined, wave)
        todo = [a for a in ids if a not in res]
        for a in todo:
            d = BASE / f'candidates/a{a:04d}/evaluations'
            if d.exists() and any(d.iterdir()):
                raise RuntimeError(f'Attempt {a} was interrupted; decide resume policy manually')
        if MULTI and todo and not enough_slurm_time():
            summary('allocation_limit', seed); return
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
        if failure_streak({**seed, **res}) >= MAX_CONSEC_FAIL:
            summary('stopped_consecutive_failures', seed); return
        if MULTI:
            if pareto_stalled(seed, res, wave):
                summary('pareto_plateau', seed); return
            summary('running', seed)
        elif stalled(res, wave):
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
    v = [r['objective_mm'] for r in res.values() if r['valid'] and r['wave'] <= wave]
    return min(v) if v else math.inf


def stalled(res, wave):
    if wave < EARLY_STOP_FROM_WAVE:
        return False
    now, before = best_through(res, wave), best_through(res, wave - EARLY_WINDOW)
    return (before - now) < EARLY_REL * before


def pareto_stalled(seed, res, wave):
    """Four full batches with no new strictly nondominated observed point."""
    from calibration.bo.pareto_bayes import case_errors, observed_front
    window = SPEC['plateau_batches']
    if wave < window:
        return False
    for w in range(wave - window + 1, wave + 1):
        before = list(seed.values()) + [r for r in res.values() if r['wave'] < w]
        current = [r for r in res.values() if r['wave'] == w]
        if len(current) != Q:
            raise ValueError(f'Incomplete batch {w}')
        old = [case_errors(r) for r in before if r['valid']]
        front = observed_front(before + current)
        if any(r['wave'] == w and not any(np.array_equal(case_errors(r), p) for p in old) for r in front):
            return False
    return True


def summary(status=None, seed=None):
    res = results()
    if MULTI:
        from calibration.bo.pareto_bayes import case_errors, observed_front
        seed = seed if seed is not None else seed_results()
        valid = [r for r in list(seed.values()) + list(res.values()) if r['valid']]
        front = observed_front(valid)
        rows = [{'source': r.get('source', 'continuation'),
                 'attempt': r.get('original_attempt', r['attempt']), 'wave': r['wave'],
                 'parameters': r['parameters'], 'run1_mm': float(case_errors(r)[0]),
                 'run3_mm': float(case_errors(r)[1])} for r in front]
        save_json(BASE / 'summary.json', {
            'status': status, 'seed_evaluations': len(seed), 'seed_valid': sum(r['valid'] for r in seed.values()),
            'new_evaluations': len(res), 'new_valid': sum(r['valid'] for r in res.values()),
            'new_failures': {k: sum(r['failure_class'] == k for r in res.values())
                             for k in ('solver', 'registration', 'timeout', 'infrastructure')},
            'reference_mm': SPEC['reference_mm'], 'frontier': rows})
        print('PARETO', status, 'new', len(res), 'valid', len(valid), 'frontier', len(rows), flush=True)
        return
    res = results()
    valid = [r for r in res.values() if r['valid']]
    best = min(valid, key=lambda r: r['objective_mm']) if valid else None
    save_json(BASE / 'summary.json', {'status': status, 'evaluations': len(res), 'valid': len(valid),
              'failures': {k: sum(1 for r in res.values() if r['failure_class'] == k)
                           for k in ('solver', 'registration', 'timeout', 'infrastructure')},
              'best': best, 'first_design_attempt': res.get(1)})
    print('SUMMARY', status, len(res), best and best['objective_mm'], flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'evaluate':
        evaluate(int(sys.argv[2]))
    elif sys.argv[1] == 'summary':
        print(json.dumps(jread(BASE / 'summary.json'), indent=2))
    else:
        raise SystemExit(2)
