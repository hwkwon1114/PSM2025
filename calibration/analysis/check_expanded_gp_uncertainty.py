"""One offline production-family GP fit; never writes to the terminal campaign."""
import hashlib
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import botorch
import gpytorch
from calibration.ewcal.batch_bayes import fit_gp_model
from calibration.ewcal.common import Parameters
import os
assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
from calibration.analysis.success_aware_bo.model import training_records

OUT = ROOT / 'run/optimization_review/run6_expanded_bo/gp_uncertainty'
OUT.mkdir(parents=True, exist_ok=True)
STATE = ROOT / 'run/run6_expanded_bo/bayes_state.json'
CONFIG = ROOT / 'calibration/configs/run6.json'
NAMES = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def loo(K, y, mean):
    inv = torch.linalg.inv(K)
    alpha = inv @ (y - mean)
    diagonal = inv.diagonal()
    return y - alpha / diagonal, 1 / diagonal

# Check analytic LOO against directly conditioning a tiny, known Gaussian.
Ktest = torch.tensor([[2., .4, .2], [.4, 1.5, .3], [.2, .3, 1.]], dtype=torch.double)
yt = torch.tensor([.2, -.5, .9], dtype=torch.double)
mt = torch.tensor([.1, .1, .1], dtype=torch.double)
mu, var = loo(Ktest, yt, mt)
for i in range(3):
    keep = torch.arange(3) != i
    cross = Ktest[i, keep]
    k = Ktest[keep][:, keep]
    assert torch.allclose(mu[i], mt[i] + cross @ torch.linalg.solve(k, yt[keep]-mt[keep]))
    assert torch.allclose(var[i], Ktest[i, i] - cross @ torch.linalg.solve(k, cross))

if any((OUT / name).exists() for name in ('gp_uncertainty.json', 'gp_uncertainty.npz', 'gp_uncertainty.png')):
    raise FileExistsError('Preserve existing diagnostic outputs; do not silently rerun.')
pins = {str(p.relative_to(ROOT)): digest(p) for p in [STATE, CONFIG,
    ROOT / 'calibration/ewcal/batch_bayes.py',
    ROOT / 'calibration/analysis/success_aware_bo/model.py',
    ROOT / 'calibration/ewcal/common.py', Path(__file__),
    ROOT / 'calibration/analysis/render_expanded_gp_uncertainty.py']}
state = json.loads(STATE.read_text())
assert state['status'] == 'budget_complete'
assert len(state['observations']) == 160
records, completions, excluded = training_records(state['observations'])
assert len(records) == 147 and len(completions) == 160 and not excluded
assert sum(y == 0 for _, y in completions) == 13
best = min(records, key=lambda r: r['objective_mm'])
xbest = np.asarray(best['normalized_parameters'])
assert best['attempt_id'] == 153
cfg = json.loads(CONFIG.read_text())
codec = Parameters(cfg)
for record in state['observations']:
    np.testing.assert_allclose(codec.encode(record['parameters']), record['normalized_parameters'], atol=1e-12, rtol=0)
summary_path = STATE.with_name('bayes_summary.json')
assert json.loads(summary_path.read_text())['best'] == best
pins[str(summary_path.relative_to(ROOT))] = digest(summary_path)
pins['run/run6_expanded_bo/corrected_history.json'] = digest(ROOT / 'run/run6_expanded_bo/corrected_history.json')
torch.set_num_interop_threads(1)
torch.set_num_threads(4)
torch.manual_seed(2026)
np.random.seed(2026)
with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter('always')
    gp, _ = fit_gp_model(records)
gp.eval()
print('FIT_COMPLETE', flush=True)

@torch.no_grad()
def posterior(x):
    # q=1 batches avoid constructing a 4096-by-4096 joint covariance.
    p = gp.posterior(torch.as_tensor(x, dtype=torch.double).unsqueeze(-2), observation_noise=False)
    mean = p.mean.reshape(-1).numpy()
    sd = p.variance.reshape(-1).sqrt().numpy()
    values = np.stack([np.exp(-mean), np.exp(-mean-1.95996398454*sd),
                       np.exp(-mean+1.95996398454*sd), sd], axis=-1)
    assert np.isfinite(values).all() and (values[:, :3] > 0).all()
    assert (values[:, 1] <= values[:, 0]).all() and (values[:, 0] <= values[:, 2]).all()
    return values

sobol = torch.quasirandom.SobolEngine(4, scramble=True, seed=2026).draw(4096, dtype=torch.double).numpy()
local = np.maximum(0, xbest-.1) + sobol * (np.minimum(1, xbest+.1)-np.maximum(0, xbest-.1))
global_post, local_post, best_post = posterior(sobol), posterior(local), posterior(xbest[None, :])
profile_x = []
profile_post = []
for dim in range(4):
    u = np.unique(np.r_[np.linspace(0, 1, 151), xbest[dim]])
    x = np.tile(xbest, (len(u), 1))
    x[:, dim] = u
    profile_x.append(x)
    profile_post.append(posterior(x))

with torch.no_grad():
    X = gp.train_inputs[0]
    cov = gp.covar_module(X).to_dense()
    cov += gp.likelihood.noise.reshape(-1)[0] * torch.eye(len(X), dtype=torch.double)
    lm, lv = loo(cov, gp.train_targets, gp.mean_module(X))
    if hasattr(gp, 'outcome_transform'):
        lm, lv = gp.outcome_transform.untransform(lm[:, None], lv[:, None])
    lm, ls = lm.reshape(-1).numpy(), lv.reshape(-1).sqrt().numpy()
truth = np.asarray([-np.log(r['objective_mm']) for r in records])
resid = (truth-lm)/ls
assert np.isfinite(resid).all()
loo_rms = np.exp(-lm)
actual = np.exp(-truth)

def summaries(p):
    return {'latent_log_rms_sd_p05_p50_p95_max': np.quantile(p[:, 3], [.05, .5, .95, 1]).tolist(),
            'latent_95_width_mm_p05_p50_p95_max': np.quantile(p[:, 2]-p[:, 1], [.05, .5, .95, 1]).tolist(),
            'latent_95_upper_lower_ratio_p50_p95': np.quantile(p[:, 2]/p[:, 1], [.5, .95]).tolist()}

report = {'question': 'Remaining model-conditional loss-GP uncertainty; no proposals or new mechanics',
    'data': {'observations': 160, 'loss_points': 147, 'timeouts_not_loss_targets': 13,
             'loss_attempts': [r['attempt_id'] for r in records], 'input_sha256': pins, 'best_parameters': best['parameters']},
    'model': {'recipe': 'production fit_gp_model, Matern5/2 ARD, normalized inputs, -log RMS',
              'seed': 2026, 'lengthscales_normalized': dict(zip(NAMES, gp.covar_module.base_kernel.lengthscale.detach().flatten().tolist())),
              'outputscale': gp.covar_module.outputscale.item(),
              'noise_model_scale': gp.likelihood.noise.detach().flatten().tolist(),
              'outcome_transform': str(getattr(gp, 'outcome_transform', None)),
              'warnings': [str(w.message) for w in notices]},
    'global_uniform_normalized_domain': summaries(global_post),
    'local_clipped_0_1_box_each_coordinate': summaries(local_post),
    'best': {'attempt': 153, 'observed_rms_mm': best['objective_mm'],
             'predicted_median_mm': best_post[0, 0], 'latent_95_mm': best_post[0, 1:3].tolist(),
             'latent_log_rms_sd': best_post[0, 3]},
    'fixed_hyperparameter_loo': {'n': len(records),
        'log_rms_rmse': float(np.sqrt(np.mean((truth-lm)**2))),
        'rms_mm_rmse': float(np.sqrt(np.mean((actual-loo_rms)**2))),
        'predictive_95_coverage': float(np.mean(np.abs(resid) <= 1.95996398454)),
        'standardized_residual_rms': float(np.sqrt(np.mean(resid**2))),
        'best_loo_median_mm': float(loo_rms[[r['attempt_id'] for r in records].index(153)]),
        'limit': 'Hyperparameters estimated on all data; analytic deletion is not independently refitted CV or held-out calibration.'},
    'limits': 'Plug-in hyperparameters; latent bands exclude noise, hyperparameter uncertainty, alignment and physics discrepancy. Sobol domain weighting is uniform in normalized coordinates (log-growth axes). Slices are conditional, not marginalized. Same adaptive specimen; no parameter posterior or global-optimum certificate. Completion uncertainty not mapped.',
    'versions': {'torch': torch.__version__, 'botorch': botorch.__version__, 'gpytorch': gpytorch.__version__, 'numpy': np.__version__, 'python': sys.version, 'matplotlib': matplotlib.__version__},
    'regeneration': 'python calibration/analysis/check_expanded_gp_uncertainty.py && python calibration/analysis/render_expanded_gp_uncertainty.py',
    'job_id': os.environ['SLURM_JOB_ID'],
    'source_sha256': digest(Path(__file__))}
for p, expected in pins.items():
    assert digest(ROOT / p) == expected, f'Input drift: {p}'
(OUT / 'gp_uncertainty.json').write_text(json.dumps(report, indent=2, allow_nan=False))
np.savez_compressed(OUT / 'gp_uncertainty.npz', domain_x=sobol, domain_posterior=global_post,
    local_x=local, local_posterior=local_post, profile_x=np.asarray(profile_x),
    profile_posterior=np.asarray(profile_post), loo_log_mean=lm, loo_log_sd=ls,
    observed_rms=actual, loss_attempts=np.asarray([r['attempt_id'] for r in records]))
print(json.dumps(report, indent=2), flush=True)

print('DIAGNOSTIC_COMPLETE; posterior arrays saved; no campaign writes', flush=True)
