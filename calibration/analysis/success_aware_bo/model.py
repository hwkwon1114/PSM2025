"""Isolated, solver-free success-aware Run6 proposal implementation.

Losses stay real. Completion is a separate Bernoulli GP with a fixed prior;
Laplace/kernel/independence assumptions are not predictive calibration evidence.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.safe_math import log_fatplus, logmeanexp
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.distributions import MultivariateNormal
from scipy.stats import beta

from calibration.ewcal.batch_bayes import (fit_gp_model, is_admissible,
                                          validate_observation_record)


def training_records(records):
    """Exclude data/infrastructure failures, rather than treating them as failures of mechanics."""
    losses, completion, excluded, seen = [], [], [], set()
    for r in sorted(records, key=lambda r: r['attempt_id']):
        validate_observation_record(r)
        if r['attempt_id'] in seen:
            raise ValueError('Duplicate attempt identity')
        seen.add(r['attempt_id'])
        if r['valid']:
            losses.append(r)
            completion.append((r, 1))
        else:
            termination = r.get('solver_termination', {})
            timeout = (r['failure_class'] == 'timeout' and termination.get('timeout') is True
                       and type(termination.get('limit_seconds')) in (int, float)
                       and termination['limit_seconds'] == 3600
                       and r.get('failure_phase') == 'solver')
            numerical = (r['failure_class'] == 'solver' and termination.get('numerical') is True
                         and r.get('failure_phase') == 'solver')
            if timeout or numerical:
                completion.append((r, 0))
            else:
                excluded.append({'attempt_id': r['attempt_id'], 'failure_class': r['failure_class'],
                                 'reason': 'No verified within-cap mechanics-completion label'})
    return losses, completion, excluded


class CompletionGP(torch.nn.Module):
    """Fixed Matérn5/2 Bernoulli-logistic GP, Laplace approximation.

    Single-class/no-label data use an exchangeable Beta rate, not a local map.
    """
    def __init__(self, X, y, *, fit=True):
        super().__init__()
        X = torch.as_tensor(X, dtype=torch.double).reshape(-1, 4)
        y = torch.as_tensor(y, dtype=torch.double).reshape(-1)
        if X.shape[0] != y.numel() or not torch.isfinite(X).all() or ((X < 0) | (X > 1)).any():
            raise ValueError('Bad completion inputs')
        if not torch.isfinite(y).all() or ((y != 0) & (y != 1)).any():
            raise ValueError('Completion labels must be binary')
        self.register_buffer('X', X)
        self.register_buffer('y', y)
        self.register_buffer('a', torch.zeros_like(y))
        self.register_buffer('sqrt_w', torch.zeros_like(y))
        self.register_buffer('L', torch.eye(len(y), dtype=torch.double))
        self.register_buffer('beta_shapes', torch.tensor([1+y.sum(), 1+len(y)-y.sum()], dtype=torch.double))
        self.mode = 'laplace' if y.numel() and y.min() != y.max() else 'exchangeable_beta'
        self.iterations = 0
        if fit and self.mode == 'laplace':
            self._fit()

    @staticmethod
    def kernel(X, Z):
        distance = torch.cdist(X / .3, Z / .3)
        r = math.sqrt(5) * distance
        return 4 * (1 + r + r.square()/3) * torch.exp(-r)

    @torch.no_grad()
    def _fit(self):
        K = self.kernel(self.X, self.X) + 1e-6 * torch.eye(len(self.y), dtype=torch.double)
        chol_k = torch.linalg.cholesky(K)
        f = torch.zeros_like(self.y)
        def objective(z):
            return (F.logsigmoid((2*self.y-1)*z).sum()
                    - .5 * z @ torch.cholesky_solve(z[:, None], chol_k).squeeze(-1))
        for iteration in range(50):
            p = f.sigmoid()
            w = p * (1-p)
            sw = w.sqrt()
            L = torch.linalg.cholesky(torch.eye(len(f), dtype=f.dtype) + sw[:, None]*K*sw[None, :])
            b = w*f + self.y-p
            a = b - sw * torch.cholesky_solve((sw*(K @ b))[:, None], L).squeeze(-1)
            direction = K @ a - f
            step = 1.
            current = objective(f)
            while objective(f+step*direction) < current-1e-12:
                step *= .5
                if step < 2**-20:
                    raise RuntimeError('Completion GP Newton line search failed')
            f += step*direction
            if (step*direction).abs().max() < 1e-8:
                self.iterations = iteration+1
                break
        else:
            raise RuntimeError('Completion GP did not converge within 50 Newton steps')
        self.a.copy_(torch.cholesky_solve(f[:, None], chol_k).squeeze(-1))
        self.sqrt_w.copy_((f.sigmoid()*(1-f.sigmoid())).sqrt())
        self.L.copy_(torch.linalg.cholesky(torch.eye(len(f), dtype=f.dtype)
                                         + self.sqrt_w[:, None]*K*self.sqrt_w[None, :]))

    def posterior(self, X):
        if self.mode != 'laplace':
            raise ValueError('Single-class fallback has a Beta rate, not a Gaussian posterior')
        cross = self.kernel(X, self.X)
        mean = cross @ self.a
        weighted = cross * self.sqrt_w
        v = torch.linalg.solve_triangular(self.L, weighted.transpose(-1, -2), upper=False)
        cov = self.kernel(X, X) - v.transpose(-1, -2) @ v
        cov = .5 * (cov + cov.transpose(-1, -2)) + 1e-8 * torch.eye(X.shape[-2], dtype=X.dtype, device=X.device)
        if not torch.isfinite(mean).all() or not torch.isfinite(cov).all():
            raise ValueError('Nonfinite completion posterior')
        return GPyTorchPosterior(MultivariateNormal(mean, covariance_matrix=cov))

    def beta_logits_from_uniform(self, uniform):
        """One shared batch rate; uniforms come from the joint loss/rate rule."""
        u = uniform.detach().cpu().numpy().ravel()
        p = beta.ppf(np.clip(u, 1e-12, 1-1e-12), *self.beta_shapes.cpu().numpy())
        return torch.from_numpy(np.log(p)-np.log1p(-p)).to(uniform)

    @torch.no_grad()
    def probabilities(self, X):
        """Latent rate uncertainty, NOT a calibrated Bernoulli prediction interval."""
        if self.mode != 'laplace':
            a, b = self.beta_shapes.cpu().numpy()
            shape = X.shape[:-1]
            return {'mean': np.full(shape, a/(a+b)).tolist(),
                    'p05': np.full(shape, beta.ppf(.05, a, b)).tolist(),
                    'p95': np.full(shape, beta.ppf(.95, a, b)).tolist()}
        posterior = self.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).sqrt()
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        p = (mean[..., None] + math.sqrt(2)*std[..., None]*torch.from_numpy(nodes)).sigmoid()
        pmean = (p * torch.from_numpy(weights)).sum(-1)/math.sqrt(math.pi)
        return {'mean': pmean.tolist(), 'p05': (mean-1.64485362695*std).sigmoid().tolist(),
                'p95': (mean+1.64485362695*std).sigmoid().tolist()}


def log_successful_improvement(loss_draws, completion_logits, best_f):
    """Integrate all 2**q completion outcomes analytically, not all-success gating."""
    logs, order = log_fatplus(loss_draws-best_f, tau=1e-6).sort(dim=-1, descending=True)
    h = completion_logits.gather(-1, order)
    log_failure = F.logsigmoid(-h)
    preceding_failures = torch.cat([torch.zeros_like(log_failure[..., :1]),
                                   log_failure[..., :-1].cumsum(-1)], dim=-1)
    return torch.logsumexp(logs + F.logsigmoid(h) + preceding_failures, dim=-1)


def joint_base_samples(q, samples, seed, *, beta_rate=False):
    """Distinct coordinates of ONE scrambled Sobol rule, not paired restarted rules."""
    if q < 1 or samples < 1:
        raise ValueError('Positive q and sample count required')
    dimension = q + (1 if beta_rate else q)
    uniform = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed).draw(
        samples, dtype=torch.double)
    eps = torch.finfo(uniform.dtype).eps
    normal = torch.special.ndtri(uniform.clamp(eps, 1-eps))
    return normal[:, :q], uniform[:, q:] if beta_rate else normal[:, q:]


def posterior_draws(posterior, base_normal):
    """Transform fixed joint-rule coordinates through each posterior covariance.

    Reuse base coordinates across t-batches for common random numbers. Only a
    single-output Gaussian posterior is supported by this Run6 acquisition.
    """
    shape = posterior.base_sample_shape
    n, q = base_normal.shape
    if posterior.mean.shape[-2:] != torch.Size([q, 1]) or shape[-1] != q:
        raise ValueError('Expected a single-output q-point Gaussian posterior')
    # GPyTorch's single-output base shape omits the output singleton axis.
    base = base_normal.reshape(n, *((1,)*(len(shape)-1)), q).expand(n, *shape)
    return posterior.rsample_from_base_samples(torch.Size([n]), base).squeeze(-1)


class SuccessAwareLogEI(MCAcquisitionFunction):
    def __init__(self, loss_model, completion_model, best_f, *, samples=1024, seed=2026):
        super().__init__(model=loss_model)
        self.completion_model = completion_model
        self.samples, self.seed, self._q = samples, seed, None
        self.register_buffer('best_f', torch.as_tensor(best_f, dtype=torch.double).detach())
        self.register_buffer('loss_base', torch.empty(0, dtype=torch.double))
        self.register_buffer('completion_base', torch.empty(0, dtype=torch.double))

    @t_batch_mode_transform()
    def forward(self, X):
        q = X.shape[-2]
        if self._q != q:
            loss_base, completion_base = joint_base_samples(
                q, self.samples, self.seed, beta_rate=self.completion_model.mode != 'laplace')
            self.loss_base, self.completion_base = loss_base.to(X), completion_base.to(X)
            self._q = q
        draws = posterior_draws(self.model.posterior(X), self.loss_base)
        if self.completion_model.mode == 'laplace':
            logits = posterior_draws(self.completion_model.posterior(X), self.completion_base)
        else:
            rate = self.completion_model.beta_logits_from_uniform(self.completion_base)
            logits = rate.reshape((-1,) + (1,)*(draws.ndim-1)).expand_as(draws)
        return logmeanexp(log_successful_improvement(draws, logits, self.best_f), dim=0)


def fit_models(records):
    losses, outcomes, excluded = training_records(records)
    if len(losses) < 3:
        raise ValueError('Need at least three real losses; no fake targets or silent proposal fallback')
    loss, best_f = fit_gp_model(losses)
    success = CompletionGP([r['normalized_parameters'] for r, _ in outcomes], [y for _, y in outcomes])
    loss.eval()
    for parameter in loss.parameters():
        parameter.requires_grad_(False)
    return loss, success, best_f, {'loss_attempts': [r['attempt_id'] for r in losses],
                                 'completion_attempts': [r['attempt_id'] for r, _ in outcomes],
                                 'completion_labels': [y for _, y in outcomes], 'excluded': excluded,
                                 'completion_mode': success.mode, 'newton_steps': success.iterations}


def repair_batch(raw, records, seed):
    """Preserve the existing random separation policy and expose its interventions."""
    attempted = [r['normalized_parameters'] for r in records]
    rng = np.random.default_rng(seed)
    result, repaired = [], []
    for i, candidate in enumerate(np.asarray(raw)):
        if not is_admissible(candidate, [], attempted, result, failure_radius=0., min_separation=.08):
            repaired.append(i)
            for _ in range(10000):
                candidate = rng.random(4)
                if is_admissible(candidate, [], attempted, result, failure_radius=0., min_separation=.08):
                    break
            else:
                raise RuntimeError('Could not repair candidate separation')
        result.append(np.array(candidate, copy=True))
    return np.asarray(result), repaired


def propose(records, loss, success, best_f, *, seed, restarts=10, initializations=512, timeout=180,
            acquisition_seed=2026):
    """Pure proposal API; optimizer seed is separate from acquisition scramble."""
    from botorch.optim import optimize_acqf
    acquisition = SuccessAwareLogEI(loss, success, best_f, seed=acquisition_seed)
    torch.manual_seed(seed)
    raw, value = optimize_acqf(acquisition, bounds=torch.tensor([[0.]*4, [1.]*4], dtype=torch.double),
                              q=4, num_restarts=restarts, raw_samples=initializations,
                              options={'maxiter': 120, 'batch_limit': 4, 'init_batch_limit': 32},
                              timeout_sec=timeout, retry_on_optimization_warning=False)
    raw = raw.detach().cpu().numpy()
    final, interventions = repair_batch(raw, records, seed)
    if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError('Invalid raw proposal')
    return {'raw': raw.tolist(), 'final': final.tolist(), 'repaired_indices': interventions,
            'optimizer_log_acquisition': float(value), 'seed': seed,
            'restarts': restarts, 'initializations': initializations,
            'acquisition_seed': acquisition_seed, 'posterior_samples': 1024}
