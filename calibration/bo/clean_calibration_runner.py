#!/usr/bin/env python3
"""Generalized Clean Bayesian Optimization calibration on the revised surface metric.

Supports single-case calibration for Run 1 ('run6'), Run 2 ('run9'), or Run 3 ('run10')
with automatic cross-case transfer evaluations on completion.

Usage:
    clean_calibration_runner.py <spec_json> run
    clean_calibration_runner.py <spec_json> evaluate <attempt>
    clean_calibration_runner.py <spec_json> summary
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

PUBLIC_NAMES = {'run6': 'Run 1', 'run9': 'Run 2', 'run10': 'Run 3'}
HISTORICAL_NAMES = {'Run 1': 'run6', 'Run 2': 'run9', 'Run 3': 'run10'}
FAILURES = ('solver', 'registration', 'timeout', 'infrastructure')


class CalibrationContext:
    def __init__(self, spec_path):
        self.spec_path = Path(spec_path).resolve()
        self.spec = json.loads(self.spec_path.read_text())
        self.base_rel = self.spec['base']
        self.base = ROOT / self.base_rel
        self.case = self.spec['case']
        self.case_label = self.spec.get('case_label', PUBLIC_NAMES.get(self.case, self.case))
        self.active_names = tuple(self.spec['active'])
        self.d = len(self.active_names)
        self.q = 4
        self.n_design_waves = self.spec.get('design_waves', 6)
        self.total_waves = self.spec.get('waves', 31)
        self.seed = self.spec['seed']
        self.min_sep = self.spec.get('min_separation', 0.05)
        self.early_from = self.spec.get('early_from', 16)
        self.early_window = self.spec.get('early_window', 6)
        self.early_rel = self.spec.get('early_rel', 0.005)
        self.max_consec_fail = 8

        self.bounds = {
            'gtop': tuple(self.spec['gtop']),
            'gbot': tuple(self.spec['gbot']),
            'ortho_top': tuple(self.spec['ortho_top']),
            'ortho_bottom': tuple(self.spec['ortho_bottom']),
        }
        self.fixed = self.spec['fixed']

    def decode(self, u):
        """Map normalized u in [0, 1]^4 to physical dictionary."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        gtop_lo, gtop_hi = self.bounds['gtop']
        gbot_lo, gbot_hi = self.bounds['gbot']
        ot_lo, ot_hi = self.bounds['ortho_top']
        ob_lo, ob_hi = self.bounds['ortho_bottom']

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
        out.update(self.fixed)
        return out

    def encode(self, p):
        """Map physical parameters to normalized u in [0, 1]^4."""
        gtop_lo, gtop_hi = self.bounds['gtop']
        gbot_lo, gbot_hi = self.bounds['gbot']
        ot_lo, ot_hi = self.bounds['ortho_top']
        ob_lo, ob_hi = self.bounds['ortho_bottom']

        u0 = (np.log(p['gtop']) - np.log(gtop_lo)) / (np.log(gtop_hi) - np.log(gtop_lo))
        u1 = (np.log(p['gbot']) - np.log(gbot_lo)) / (np.log(gbot_hi) - np.log(gbot_lo))
        u2 = (p['ortho_top'] - ot_lo) / (ot_hi - ot_lo)
        u3 = (p['ortho_bottom'] - ob_lo) / (ob_hi - ob_lo)
        return np.clip(np.array([u0, u1, u2, u3], dtype=float), 0.0, 1.0)

    def case_config(self, values):
        """Generate configuration for single-case forward solve."""
        cfg = cc.config(self.case)
        for k in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
            cfg['parameters'][k]['value'] = float(values[k])
            cfg['cases'][0]['operation'][k] = float(values[k])
        cfg['hardening']['beta'] = float(values['beta'])
        cfg['hardening']['floor'] = float(values['floor'])
        return cfg

    def admissible(self, pt, others):
        return all(np.linalg.norm(pt - o) >= self.min_sep for o in others)

    def lhs_design(self, seed=None, n_points=None):
        from scipy.stats import qmc
        seed = self.seed if seed is None else seed
        n_points = (self.q * self.n_design_waves) if n_points is None else n_points
        rng_seed = seed
        for attempt in range(500):
            sampler = qmc.LatinHypercube(d=self.d, seed=rng_seed + attempt)
            pts = sampler.random(n_points)
            ok = True
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if np.linalg.norm(pts[i] - pts[j]) < self.min_sep:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return pts
        raise RuntimeError(f'Could not generate {n_points} LHS points satisfying min_sep={self.min_sep}')

    @staticmethod
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

    def load_results(self):
        results_dir = self.base / 'results'
        if not results_dir.exists():
            return {}
        out = {}
        for p in sorted(results_dir.glob('a*.json')):
            a = int(p.stem[1:])
            res = json.loads(p.read_text())
            out[a] = self.classify(res)
        return out

    def propose(self, res, wave):
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

        torch.manual_seed(self.seed + wave)
        warnings.filterwarnings('ignore')

        att = [r for r in recs if r['valid'] or r['failure_class'] in ('solver', 'registration', 'timeout')]
        X = np.array([r['x'] for r in att])
        y = np.array([int(r['valid']) for r in att])

        gp, best_f = fit_gp_model([dict(normalized_parameters=r['x'], objective_mm=r['objective_mm']) for r in valid])

        rng = np.random.default_rng(self.seed + wave)
        pool = qmc.Sobol(self.d, seed=self.seed + wave).random(2**14)
        top = np.array([r['x'] for r in sorted(valid, key=lambda r: r['objective_mm'])[:3]])
        local = np.clip(top[rng.integers(len(top), size=4000)] + rng.normal(0, 0.05, (4000, self.d)), 0, 1)
        pool = np.vstack([pool, local])

        if y.min() == y.max():
            logp = np.zeros(len(pool))
        else:
            k = ConstantKernel(1.0, (0.1, 10)) * Matern(length_scale=np.ones(self.d), length_scale_bounds=(0.1, 20), nu=2.5)
            gpc = GaussianProcessClassifier(k, n_restarts_optimizer=2, random_state=self.seed + wave).fit(X, y)
            logp = np.log(np.clip(gpc.predict_proba(pool)[:, 1], 1e-3, 1))

        P = torch.tensor(pool, dtype=torch.double).unsqueeze(1)
        taken = [x for x in X]
        batch = []
        cur = gp

        for _ in range(self.q):
            with torch.no_grad():
                score = LogExpectedImprovement(cur, best_f=best_f)(P).numpy() + logp
            for i in np.argsort(-score):
                if self.admissible(pool[i], taken + batch):
                    break
            else:
                raise RuntimeError('No admissible candidate in pool')
            batch.append(pool[i])
            with torch.no_grad():
                mu = cur.posterior(torch.tensor(pool[i:i + 1], dtype=torch.double)).mean
            cur = cur.condition_on_observations(torch.tensor(pool[i:i + 1], dtype=torch.double), mu)

        return np.array(batch)

    def make_wave(self, res, wave):
        ids = list(range(self.q * (wave - 1) + 1, self.q * wave + 1))
        proposals_dir = self.base / 'proposals'
        proposals_dir.mkdir(parents=True, exist_ok=True)
        paths = [proposals_dir / f'a{a:04d}.json' for a in ids]

        if all(p.exists() for p in paths):
            return [json.loads(p.read_text()) for p in paths]

        if wave <= self.n_design_waves:
            design = self.lhs_design()
            start = self.q * (wave - 1)
            pts = design[start:start + self.q]
            source = 'initial_lhs'
        else:
            pts = self.propose(res, wave)
            source = 'log_ei'

        proposals = []
        for a, pt in zip(ids, pts):
            prop = {
                'attempt': a,
                'wave': wave,
                'source': source,
                'x': pt.tolist(),
                'parameters': self.decode(pt)
            }
            save_json(proposals_dir / f'a{a:04d}.json', prop)
            proposals.append(prop)
        return proposals

    def evaluate(self, attempt):
        prop_path = self.base / 'proposals' / f'a{attempt:04d}.json'
        prop = json.loads(prop_path.read_text())
        cand_dir = self.base / 'candidates' / f'a{attempt:04d}'
        cand_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.case_config(prop['parameters'])
        cfg['output_dir'] = str(cand_dir)

        evaluator = Evaluator(cfg)
        raw = evaluator.evaluate(prop['x'])

        res = {
            'attempt': attempt,
            'wave': prop['wave'],
            'source': prop['source'],
            'x': prop['x'],
            'parameters': prop['parameters'],
            'case': self.case_label,
            'historical_case': self.case,
            'valid': bool(raw.get('valid', False)),
            'error': raw.get('error'),
            'objective_mm': float(raw['objective_mm']) if raw.get('valid') else None,
            'seconds': raw.get('seconds'),
            'failure_class': raw.get('failure_class'),
        }
        if res['valid'] and 'cases' in raw and raw['cases']:
            res['case_rms_mm'] = {self.case_label: raw['cases'][0]['case_rms']}
            res['metrics'] = raw['cases'][0]['metrics']

        res = self.classify(res)
        results_dir = self.base / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        save_json(results_dir / f'a{attempt:04d}.json', res)
        return res

    def dispatch_wave(self, proposals):
        procs = []
        env = os.environ.copy()
        env['BLIS_NUM_THREADS'] = '1'
        for v in ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT'):
            env.pop(v, None)

        for p in proposals:
            cmd = [sys.executable, str(Path(__file__).resolve()), str(self.spec_path), 'evaluate', str(p['attempt'])]
            procs.append((p['attempt'], subprocess.Popen(cmd, env=env)))

        for a, proc in procs:
            ret = proc.wait()
            if ret != 0:
                res_path = self.base / 'results' / f'a{a:04d}.json'
                if not res_path.exists():
                    prop = json.loads((self.base / 'proposals' / f'a{a:04d}.json').read_text())
                    save_json(res_path, {
                        'attempt': a,
                        'wave': prop['wave'],
                        'source': prop['source'],
                        'x': prop['x'],
                        'parameters': prop['parameters'],
                        'case': self.case_label,
                        'historical_case': self.case,
                        'valid': False,
                        'error': f'Subprocess crashed with exit code {ret}',
                        'failure_class': 'infrastructure',
                        'objective_mm': None
                    })

    @staticmethod
    def failure_streak(res):
        streak = 0
        for a in sorted(res.keys()):
            if not res[a]['valid']:
                streak += 1
            else:
                streak = 0
        return streak

    def check_early_stop(self, res, wave):
        if wave < self.early_from:
            return False, f'Wave {wave} < early_from {self.early_from}'
        past_wave = wave - self.early_window
        attempts_past = self.q * past_wave
        attempts_curr = self.q * wave

        valid_past = [r['objective_mm'] for a, r in res.items() if a <= attempts_past and r['valid']]
        valid_curr = [r['objective_mm'] for a, r in res.items() if a <= attempts_curr and r['valid']]

        if not valid_past or not valid_curr:
            return False, 'Insufficient valid evaluations'

        best_past = min(valid_past)
        best_curr = min(valid_curr)

        rel_imp = (best_past - best_curr) / max(best_past, 1e-6)
        if rel_imp < self.early_rel:
            return True, f'Improvement {rel_imp * 100:.4f}% over last {self.early_window} waves < {self.early_rel * 100:.2f}%'
        return False, f'Improvement {rel_imp * 100:.4f}% >= {self.early_rel * 100:.2f}%'

    def write_summary(self, res, status='running', reason=None):
        valid = [r for r in res.values() if r['valid']]
        best = min(valid, key=lambda r: r['objective_mm']) if valid else None

        counts = {k: 0 for k in FAILURES}
        for r in res.values():
            if not r['valid']:
                c = r.get('failure_class', 'infrastructure')
                counts[c] = counts.get(c, 0) + 1

        summary = {
            'target_case': self.case_label,
            'historical_case': self.case,
            'status': status,
            'stop_reason': reason,
            'total_evaluations': len(res),
            'valid_evaluations': len(valid),
            'best': best,
            'failure_counts': counts,
            'max_consecutive_failures': self.failure_streak(res),
        }
        self.base.mkdir(parents=True, exist_ok=True)
        save_json(self.base / 'summary.json', summary)

        lines = [json.dumps(res[a]) for a in sorted(res.keys())]
        (self.base / 'evaluations.jsonl').write_text('\n'.join(lines) + ('\n' if lines else ''))
        return summary

    def evaluate_transfer(self, params, case_name):
        pub_name = PUBLIC_NAMES.get(case_name, case_name)
        transfer_dir = self.base / 'transfer' / case_name
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

        res = self.classify(res)
        save_json(self.base / 'transfer' / f'{case_name}_result.json', res)
        return res

    def run(self):
        res = self.load_results()
        current_attempt = max(res.keys(), default=0)
        current_wave = (current_attempt // self.q) + 1

        print(f'Starting {self.case_label} ({self.case}) clean calibration at wave {current_wave} (existing attempts: {len(res)})')
        print(f'Target: {self.case_label}, 24 LHS + 100 BO = 124 total evaluations (31 waves of 4)')

        status = 'completed'
        reason = f'Completed all {self.total_waves} waves ({self.total_waves * self.q} evaluations)'

        for wave in range(current_wave, self.total_waves + 1):
            print(f'--- Wave {wave}/{self.total_waves} (attempts {self.q * (wave - 1) + 1}-{self.q * wave}) ---')
            props = self.make_wave(res, wave)
            self.dispatch_wave(props)
            res = self.load_results()
            self.write_summary(res, status='running')

            streak = self.failure_streak(res)
            if streak >= self.max_consec_fail:
                status = 'aborted'
                reason = f'Hit {streak} consecutive failures'
                print(f'ABORT: {reason}')
                break

            valid = [r for r in res.values() if r['valid']]
            best_obj = min(r['objective_mm'] for r in valid) if valid else float('inf')
            print(f'Wave {wave} complete. Valid: {len(valid)}/{len(res)}. Best {self.case_label} RMS: {best_obj:.4f} mm')

            stop, stop_msg = self.check_early_stop(res, wave)
            if stop:
                status = 'early_stopped'
                reason = stop_msg
                print(f'EARLY STOP: {reason}')
                break

        summary = self.write_summary(res, status=status, reason=reason)
        print(f'{self.case_label} calibration finished: status={status}. Best attempt: {summary["best"]["attempt"]} ({summary["best"]["objective_mm"]:.4f} mm)')

        # Perform frozen transfer evaluations on the other two cases
        if summary['best']:
            other_cases = [c for c in ('run6', 'run9', 'run10') if c != self.case]
            print(f'Running exploratory transfer evaluations on {other_cases}...')
            transfers = {}
            for oc in other_cases:
                pub = PUBLIC_NAMES.get(oc, oc)
                t_res = self.evaluate_transfer(summary['best']['parameters'], oc)
                print(f'Transfer to {pub} ({oc}): valid={t_res["valid"]}, RMS={t_res["objective_mm"]}')
                transfers[pub] = {
                    'historical_case': oc,
                    'valid': t_res['valid'],
                    'objective_mm': t_res['objective_mm']
                }
            summary['transfers'] = transfers
            save_json(self.base / 'summary.json', summary)

        print(f'All calibration and transfer tasks for {self.case_label} completed successfully!')


def run_cli(spec_path, args):
    ctx = CalibrationContext(spec_path)
    if not args:
        print('Usage: clean_calibration_runner.py <spec_json> [run | evaluate <attempt> | summary]')
        sys.exit(1)
    action = args[0]
    if action == 'run':
        ctx.run()
    elif action == 'evaluate':
        if len(args) < 2:
            print('Specify attempt number: evaluate <attempt>')
            sys.exit(1)
        ctx.evaluate(int(args[1]))
    elif action == 'summary':
        res = ctx.load_results()
        s = ctx.write_summary(res, status='inspected')
        print(json.dumps(s, indent=2))
    else:
        raise ValueError(f'Unknown action: {action}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: clean_calibration_runner.py <spec_json> [run | evaluate <attempt> | summary]')
        sys.exit(1)
    run_cli(sys.argv[1], sys.argv[2:])
