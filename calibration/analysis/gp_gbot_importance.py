"""GP diagnostic: does gbot (and ortho_bottom) matter for the joint Run6+Run10 objective?

Post-hoc, solver-free. Pools every valid joint evaluation under the shared-footprint metric
(Stage 1, Stage 2 Arm A/B, Stage 3), encodes physical parameters in one common box, fits ARD Matern-5/2 GPs to
log J and compares: (1) K-fold predictive error with vs without gbot / ortho_bottom inputs, (2) ARD lengthscales,
(3) total-effect sensitivity of the GP mean in a local box around the best point, (4) GP profile along gbot at the
best point, (5) direct paired evaluations (Stage-3 wave 1 = Arm A points with gbot set to 0).
Points are adaptive BO samples (clustered, dependent): CV errors are optimistic and descriptive only.
"""
import glob
import json
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = {'stage1': 'run/joint_run6_run10', 'armA': 'run/joint_stage2/arm_A',
             'armB': 'run/joint_stage2/arm_B', 'stage3': 'run/joint_stage3_gbot0'}
NAMES = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor']
SHIFT = 1e-5


def enc(p):
    return np.array([
        np.log(p['gtop'] / 5e-5) / np.log(1e-2 / 5e-5),
        np.log((p['gbot'] + SHIFT) / SHIFT) / np.log((1e-3 + SHIFT) / SHIFT),
        (p['ortho_top'] + 1) / 2, (p['ortho_bottom'] + 1) / 2,
        np.log(p['beta'] / 0.01) / np.log(1.0 / 0.01), p['floor']])


def dec_gbot(u):
    return SHIFT * np.exp(u * np.log((1e-3 + SHIFT) / SHIFT)) - SHIFT


def load():
    rows = []
    for c, d in CAMPAIGNS.items():
        for f in sorted(glob.glob(str(ROOT / d / 'results/a*.json'))):
            r = json.loads(Path(f).read_text())
            if r.get('valid'):
                rows.append(dict(campaign=c, attempt=r['attempt'], p=r['parameters'], J=r['objective_mm'],
                                 r6=r['case_rms_mm']['run6'], r10=r['case_rms_mm']['run10']))
    return rows


def gp(d, seed=0):
    k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.ones(d), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-3, (1e-8, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=seed)


def cv(X, y, cols, Jmask, reps=5):
    errs, errs_basin = [], []
    for rep in range(reps):
        pred = np.empty_like(y)
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(X):
            pred[te] = gp(len(cols), rep).fit(X[tr][:, cols], y[tr]).predict(X[te][:, cols])
        errs.append(np.sqrt(np.mean((pred - y) ** 2)))
        errs_basin.append(np.sqrt(np.mean((np.exp(pred[Jmask]) - np.exp(y[Jmask])) ** 2)))
    return float(np.mean(errs)), float(np.std(errs)), float(np.mean(errs_basin)), float(np.std(errs_basin))


def total_effects(m, center, half=0.15, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)
    A = lo + (hi - lo) * rng.random((n, 6)); B = lo + (hi - lo) * rng.random((n, 6))
    fA, fB = m.predict(A), m.predict(B)
    var = np.var(np.r_[fA, fB])
    out = {}
    for i, nm in enumerate(NAMES):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        out[nm] = float(0.5 * np.mean((fA - m.predict(ABi)) ** 2) / var)   # Jansen total-effect estimator
    return out, (lo.tolist(), hi.tolist())


def main():
    rows = load()
    X = np.array([enc(r['p']) for r in rows]); J = np.array([r['J'] for r in rows]); y = np.log(J)
    basin = J <= 1.0
    best = rows[int(np.argmin(J))]
    res = {'n_valid': len(rows), 'n_by_campaign': {c: sum(r['campaign'] == c for r in rows) for c in CAMPAIGNS},
           'n_basin_J_le_1mm': int(basin.sum()), 'n_gbot_exact_zero': int(sum(r['p']['gbot'] == 0 for r in rows)),
           'best': {k: best[k] for k in ('campaign', 'attempt', 'J', 'r6', 'r10', 'p')}}

    sets = {'all6': [0, 1, 2, 3, 4, 5], 'no_gbot': [0, 2, 3, 4, 5], 'no_obot': [0, 1, 2, 4, 5],
            'no_gbot_no_obot': [0, 2, 4, 5], 'no_beta': [0, 1, 2, 3, 5], 'no_floor': [0, 1, 2, 3, 4]}
    res['cv'] = {}
    for name, cols in sets.items():
        for sub, mask in (('all', np.ones(len(y), bool)), ('basin', basin)):
            a, sa, b, sb = cv(X[mask], y[mask], cols, J[mask] <= 0.8)
            res['cv'][f'{name}/{sub}'] = dict(rmse_logJ=a, sd=sa, rmse_mm_J_le_0p8=b, sd_mm=sb)
            print(f'CV {name:16s} {sub:5s}: RMSE log J {a:.4f}±{sa:.4f} | J<=0.8 mm RMSE {b:.4f}±{sb:.4f} mm', flush=True)

    for sub, mask in (('all', np.ones(len(y), bool)), ('basin', basin)):
        m = gp(6).fit(X[mask], y[mask])
        ls = m.kernel_.k1.k2.length_scale
        res[f'lengthscales_{sub}'] = dict(zip(NAMES, map(float, ls)))
        res[f'kernel_{sub}'] = str(m.kernel_)
        print(f'lengthscales {sub}:', {k: round(float(v), 3) for k, v in zip(NAMES, ls)})
        te, box = total_effects(m, enc(best['p']))
        res[f'total_effect_local_{sub}'] = te; res['local_box'] = box
        print(f'total effects (local box ±0.15 around best) {sub}:', {k: round(v, 3) for k, v in te.items()})
        xb = enc(best['p']); prof = []
        for g in [0, 1e-6, 3e-6, 1e-5, 2e-5, 5e-5, 1e-4, 3e-4]:
            x = xb.copy(); x[1] = enc(dict(best['p'], gbot=g))[1]
            mu, sd = m.predict(x[None], return_std=True)
            prof.append(dict(gbot=g, J_mm=float(np.exp(mu[0])), lo95=float(np.exp(mu[0] - 2 * sd[0])),
                             hi95=float(np.exp(mu[0] + 2 * sd[0]))))
        res[f'profile_gbot_at_best_{sub}'] = prof
        print(f'gbot profile at best ({sub}):', [(p['gbot'], round(p['J_mm'], 4), round(p['lo95'], 4), round(p['hi95'], 4)) for p in prof])

    a = {r['attempt']: r for r in rows if r['campaign'] == 'armA'}
    s3 = {r['attempt']: r for r in rows if r['campaign'] == 'stage3'}
    pairs = []
    for s, src in zip([1, 2, 3, 4], [26, 43, 30, 25]):
        if s in s3 and src in a:
            pairs.append(dict(armA=src, gbot_before=a[src]['p']['gbot'], J_before=a[src]['J'], J_gbot0=s3[s]['J'],
                              dJ=s3[s]['J'] - a[src]['J'], dr6=s3[s]['r6'] - a[src]['r6'], dr10=s3[s]['r10'] - a[src]['r10']))
    res['paired_gbot0'] = pairs
    for p in pairs:
        print(f"paired a{p['armA']:04d}: gbot {p['gbot_before']:.2e}->0  J {p['J_before']:.4f}->{p['J_gbot0']:.4f} (dJ {p['dJ']:+.4f}; dRun6 {p['dr6']:+.4f}, dRun10 {p['dr10']:+.4f})")
    out = ROOT / 'run/joint_stage3_gbot0/gp_gbot_importance.json'
    out.write_text(json.dumps(res, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
