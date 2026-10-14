"""Joint admissible proposals; isolated from the active mechanics controller."""
import io
import logging
import warnings

import numpy as np
import torch
from botorch.exceptions.errors import CandidateGenerationError
from botorch.generation.gen import gen_candidates_scipy
from botorch.logging import logger
from botorch.optim import optimize_acqf

from calibration.analysis.success_aware_bo.model import SuccessAwareLogEI
from calibration.ewcal.batch_bayes import is_admissible

SEPARATION = .08
SEARCH_DISTANCE = SEPARATION + 1e-6


def history_tensor(attempted):
    X = torch.as_tensor(attempted, dtype=torch.double)
    if X.numel() == 0 and X.shape == torch.Size([0]):
        X = X.reshape(0, 4)
    if X.ndim != 2 or X.shape[-1] != 4 or not torch.isfinite(X).all() or ((X < 0) | (X > 1)).any():
        raise ValueError('Historical points must be finite normalized four-vectors')
    return X


def constraints(history):
    """Nearest-distance scalar constraints preserve every exclusion, not just losses."""
    result = []
    if len(history):
        def historical(x):
            return (x-history).square().sum(-1).min() - SEARCH_DISTANCE**2
        result.append((historical, True))

    def pairwise(X):
        i, j = torch.triu_indices(4, 4, offset=1, device=X.device)
        return (X[i]-X[j]).square().sum(-1).min() - SEARCH_DISTANCE**2
    result.append((pairwise, False))
    return result


def admissible(batch, history):
    batch = np.asarray(batch)
    return batch.shape == (4, 4) and all(
        is_admissible(p, [], history, batch[:i], failure_radius=0., min_separation=SEPARATION)
        for i, p in enumerate(batch))


def feasible_initializations(history, count, seed):
    engine = torch.quasirandom.SobolEngine(16, scramble=True, seed=seed)
    accepted, drawn = [], 0
    # A finite generation ceiling is a failure path, not permission to relax spacing.
    while sum(len(x) for x in accepted) < count and drawn < 100*count:
        n = min(count, 100*count-drawn)
        X = engine.draw(n, dtype=torch.double).reshape(n, 4, 4)
        drawn += n
        pair = (X[:, :, None, :]-X[:, None, :, :]).square().sum(-1)
        i, j = torch.triu_indices(4, 4, offset=1)
        keep = (pair[:, i, j] >= SEARCH_DISTANCE**2).all(-1)
        if len(history):
            distances = (X[:, :, None, :]-history).square().sum(-1)
            keep &= (distances >= SEARCH_DISTANCE**2).all(dim=(-1, -2))
        accepted.append(X[keep])
    if sum(len(x) for x in accepted) < count:
        raise RuntimeError('Feasible initialization generation exhausted; spacing unchanged')
    return torch.cat(accepted)[:count], drawn


def scores(acquisition, batches):
    with torch.no_grad():
        values = torch.cat([acquisition(x).reshape(-1) for x in batches.split(32)])
    if len(values) != len(batches) or not torch.isfinite(values).all():
        raise ValueError('Nonfinite or incorrectly shaped acquisition')
    return values


def optimize_separated(acquisition, attempted, *, seed, restarts=10, initializations=512, timeout=180):
    """Select the best strictly admissible joint batch, never random member repair.

    shortcut: nearest-distance constraints have kinks; investigate smooth separate
    constraints only if convergence diagnostics show that this limits utility.
    """
    if (type(restarts) is not int or type(initializations) is not int
            or not 1 <= restarts <= initializations or not np.isfinite(timeout) or timeout <= 0):
        raise ValueError('Positive timeout and 1 <= restarts <= initializations required')
    history = history_tensor(attempted)
    pool, drawn = feasible_initializations(history, initializations, seed)
    initial_values = scores(acquisition, pool)
    top = torch.argsort(initial_values, descending=True, stable=True)[:restarts]
    diagnostics = []

    def generate(*args, **kwargs):
        # Capture below BoTorch's warning-containment boundary; never add retries.
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                candidate, value = gen_candidates_scipy(*args, **kwargs)
                if not torch.isfinite(candidate).all() or not torch.isfinite(value).all():
                    raise ValueError('Nonfinite optimizer candidates or acquisition')
                return candidate, value
        finally:
            logger.setLevel(level)
            logger.removeHandler(handler)
            handler.close()
            diagnostics.append({
                'warnings': [{'category': w.category.__name__, 'message': str(w.message),
                              'filename': w.filename, 'lineno': w.lineno} for w in captured],
                'info_messages': output.getvalue().splitlines()})

    feasibility_error = None
    try:
        optimized, _ = optimize_acqf(
            acquisition, bounds=torch.tensor([[0.]*4, [1.]*4], dtype=torch.double), q=4,
            num_restarts=restarts, raw_samples=None, batch_initial_conditions=pool[top],
            nonlinear_inequality_constraints=constraints(history), return_best_only=False,
            options={'maxiter': 120, 'batch_limit': 1}, timeout_sec=timeout,
            retry_on_optimization_warning=False, gen_candidates=generate)
    except CandidateGenerationError as error:
        # Exact installed-BoTorch aggregate failure only; generator errors propagate.
        expected = (f'The optimizer produced infeasible candidates. {restarts} out of '
                    f'{restarts} batches of candidates were infeasible. Please make sure '
                    'the constraints are satisfiable and relax them if needed. ')
        if str(error) != expected or len(diagnostics) != restarts:
            raise
        feasibility_error = str(error)
        optimized = pool.new_empty((0, 4, 4))
    optimized = optimized.detach()
    if optimized.shape != (0 if feasibility_error else restarts, 4, 4):
        raise ValueError('Unexpected constrained optimizer result shape')
    history_np = history.numpy()
    valid = [i for i, X in enumerate(optimized) if admissible(X.numpy(), history_np)]
    candidates = torch.cat([pool[top[:1]], optimized[valid]])
    values = scores(acquisition, candidates)
    index = int(values.argmax())
    final = candidates[index].numpy()
    if not admissible(final, history_np):
        raise ValueError('Final proposal violates strict separation')
    return {'final': final.tolist(), 'repaired_indices': [],
            'selection': 'scored_initialization' if index == 0 else 'constrained_optimizer',
            'selected_restart': None if index == 0 else valid[index-1],
            'optimizer_log_acquisition': float(values[index]),
            'best_initial_log_acquisition': float(initial_values[top[0]]),
            'best_initial_batch': pool[top[0]].tolist(),
            'generator_diagnostics': diagnostics,
            'warnings': [f"{w['category']}: {w['message']}"
                         for diagnostic in diagnostics for w in diagnostic['warnings']],
            'optimizer_feasibility_error': feasibility_error,
            'convergence_status': 'not_inferred_from_selection_or_absence_of_warnings',
            'inadmissible_optimizer_restarts': [i for i in range(restarts) if i not in valid],
            'seed': seed, 'restarts': restarts, 'initializations': initializations,
            'initial_batches_drawn': drawn, 'min_separation': SEPARATION,
            'search_distance': SEARCH_DISTANCE}


def propose(records, loss, success, best_f, *, seed, restarts=10, initializations=512, timeout=180,
            acquisition_seed=2026):
    """Success-aware separation-aware proposal API; no fit, solver or controller writes."""
    acquisition = SuccessAwareLogEI(loss, success, best_f, seed=acquisition_seed)
    result = optimize_separated(acquisition, [r['normalized_parameters'] for r in records],
                                seed=seed, restarts=restarts,
                                initializations=initializations, timeout=timeout)
    return {**result, 'acquisition_seed': acquisition_seed, 'posterior_samples': 1024}
