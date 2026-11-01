#!/usr/bin/env python
"""Offline replay (read-only): would a 'does it solve' model beat failure imputation in the joint Run6+Run10 search?

Reads run/joint_run6_run10/results only; launches no solver; never touches Run 9.
Usage: replay_solve_model.py [out.json]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_joint_run6_run10 as J
from calibration.ewcal.batch_bayes import fit_gp_model

warnings.filterwarnings('ignore')
Q = J.Q


def load():
    recs = [r for r in J.results().values() if r['valid'] or r['failure_class'] in ('solver', 'timeout', 'registration')]
    recs.sort(key=lambda r: r['attempt'])
    X = np.array([r['x'] for r in recs]); ok = np.array([bool(r['valid']) for r in recs])
    Jv = np.array([r['objective_mm'] if r['valid'] else np.nan for r in recs])
    wave = np.array([(r['attempt'] - 1) // Q + 1 for r in recs])
    return recs, X, ok, Jv, wave


def prior_p(Xtr, ytr, Xte):
    return np.full(len(Xte), (ytr.sum() + 1) / (len(ytr) + 2))


def logit_p(Xtr, ytr, Xte):
    if ytr.min() == ytr.max():
        return prior_p(Xtr, ytr, Xte)
    return LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def gpc_p(Xtr, ytr, Xte):
    if ytr.min() == ytr.max():
        return prior_p(Xtr, ytr, Xte)
    k = ConstantKernel(1.0, (0.1, 10)) * Matern(length_scale=np.ones(6), length_scale_bounds=(0.1, 20), nu=2.5)
    return GaussianProcessClassifier(k, n_restarts_optimizer=2, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def imput_score(Xtr, oktr, Jtr, Xte):
    """Score from the current pipeline: GP on -log J with failures imputed at 1.5x worst valid J; high = likely valid."""
    worst = np.nanmax(Jtr)
    obj = np.where(oktr, Jtr, 1.5 * worst)
    gp, _ = fit_gp_model([dict(normalized_parameters=x.tolist(), objective_mm=o) for x, o in zip(Xtr, obj)])
    with torch.no_grad():
        return gp.posterior(torch.tensor(Xte, dtype=torch.double)).mean.squeeze(-1).numpy()


def metrics(y, p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return dict(brier=float(np.mean((p - y) ** 2)), logloss=float(log_loss(y, p, labels=[0, 1])),
                auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else None)


def sequential(X, ok, Jv, wave, first=4):
    y = ok.astype(int)
    rows = {m: [] for m in ('prior', 'logistic', 'gpc', 'imputed_gp_score')}
    ys = []
    for w in range(first, wave.max() + 1):
        te, tr = wave == w, wave < w
        if not te.any() or tr.sum() < 8:
            continue
        ys.append(y[te])
        rows['prior'].append(prior_p(X[tr], y[tr], X[te]))
        rows['logistic'].append(logit_p(X[tr], y[tr], X[te]))
        rows['gpc'].append(gpc_p(X[tr], y[tr], X[te]))
        rows['imputed_gp_score'].append(imput_score(X[tr], ok[tr], Jv[tr], X[te]))
    ys = np.concatenate(ys)
    out = {'n_test': int(len(ys)), 'success_rate': float(ys.mean())}
    for m, v in rows.items():
        v = np.concatenate(v)
        out[m] = dict(auc=float(roc_auc_score(ys, v))) if m == 'imputed_gp_score' else metrics(ys, v)
    return out


def loo(X, ok):
    y = ok.astype(int); n = len(y)
    res = {}
    for name, f in (('prior', prior_p), ('logistic', logit_p), ('gpc', gpc_p)):
        p = np.array([f(X[np.arange(n) != i], y[np.arange(n) != i], X[i:i + 1])[0] for i in range(n)])
        res[name] = metrics(y, p)
    return res


def alt_batch(Xtr, oktr, Jtr, taken, seed):
    """Alternative proposal: LogEI(valid-only GP) + log P(success) (GPC), greedy q=4 with the same separation rule."""
    from botorch.acquisition.analytic import LogExpectedImprovement
    v = oktr
    gp, best = fit_gp_model([dict(normalized_parameters=x.tolist(), objective_mm=o) for x, o in zip(Xtr[v], Jtr[v])])
    pool = qmc.Sobol(6, seed=seed).random(2 ** 13)
    with torch.no_grad():
        lei = LogExpectedImprovement(gp, best_f=best)(torch.tensor(pool, dtype=torch.double).unsqueeze(1)).numpy()
    score = lei + np.log(np.clip(gpc_p(Xtr, oktr.astype(int), pool), 1e-3, 1))
    batch = []
    for i in np.argsort(-score):
        if all(np.linalg.norm(pool[i] - o) >= J.MIN_SEP for o in list(taken) + batch):
            batch.append(pool[i])
        if len(batch) == Q:
            break
    return np.array(batch)


def replay_proposals(X, ok, Jv, wave, first=4):
    """Per BO wave: success-model probability for actual vs alternative batches, and nearest later-observed outcome."""
    rows, y = [], ok.astype(int)
    for w in range(first, wave.max() + 1):
        te, tr = wave == w, wave < w
        if not te.any():
            continue
        alt = alt_batch(X[tr], ok[tr], Jv[tr], X[tr], seed=2026 + w)
        pa, pb = gpc_p(X[tr], y[tr], X[te]), gpc_p(X[tr], y[tr], alt)
        later = ~(wave <= w)
        nn = []
        for a in alt:
            if later.any():
                d = np.linalg.norm(X[later] - a, axis=1); i = np.argmin(d)
                nn.append((float(d[i]), bool(ok[later][i])))
        rows.append(dict(wave=int(w), actual_mean_p=float(pa.mean()), actual_success=float(y[te].mean()),
                         alt_mean_p=float(pb.mean()), alt_nearest_later=nn))
    return rows


def box_stats(recs, X, ok):
    beta = np.array([r['parameters']['beta'] for r in recs]); floor = np.array([r['parameters']['floor'] for r in recs])
    inbox = (floor <= 0.10) & (beta >= 0.01) & (beta <= 0.30)
    mid = (floor <= 0.50)
    f = lambda m: dict(n=int(m.sum()), success=int(ok[m].sum()), rate=float(ok[m].mean()) if m.any() else None)
    return dict(all=f(np.ones(len(ok), bool)), lit_box_floor_le_0p10_beta_0p01_0p30=f(inbox), floor_le_0p5=f(mid),
                floor_gt_0p5=f(~mid), outside_lit_box=f(~inbox))


def main():
    torch.set_num_threads(4)
    recs, X, ok, Jv, wave = load()
    out = dict(n=int(len(ok)), n_valid=int(ok.sum()), n_fail=int((~ok).sum()), waves=int(wave.max()))
    for k, f in (('box', lambda: box_stats(recs, X, ok)), ('sequential', lambda: sequential(X, ok, Jv, wave)),
                 ('loo', lambda: loo(X, ok)), ('proposals', lambda: replay_proposals(X, ok, Jv, wave))):
        out[k] = f(); print('done', k, flush=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'proposals'}, indent=1))
    for r in out['proposals']:
        print(r['wave'], 'actual p=%.2f succ=%.2f | alt p=%.2f' % (r['actual_mean_p'], r['actual_success'], r['alt_mean_p']),
              [(round(d, 2), s) for d, s in r['alt_nearest_later']])


if __name__ == '__main__':
    main()
