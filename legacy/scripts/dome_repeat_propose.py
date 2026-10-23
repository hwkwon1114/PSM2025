"""Finite-pool mixed-integer BO proposals; failed solves have no shape score."""
import numpy as np
import torch
from scipy.stats import qmc
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from dome_bo_repeat_domain import decode, encode, sequence


def propose(base, records, seed, count=2, initial=False):
    """Select unique feasible decoded geometries. records include all prior attempts."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    seen = {tuple(r['parameters']) for r in records}
    pool, physical = [], []
    for point in qmc.Sobol(8,scramble=True,seed=seed).random_base2(14):
        values = decode(point)
        key = tuple(values)
        if key in seen:
            continue
        try:
            sequence(base,values)
        except ValueError:
            continue
        seen.add(key)
        pool.append(encode(values)); physical.append(values)
    if len(pool) < count:
        raise RuntimeError('Insufficient feasible novel candidates')
    valid = [r for r in records if r['status'] == 'accepted']
    if initial or len(valid) < 2:
        indices = list(range(count))
        method = 'feasible_sobol'
    else:
        x = torch.tensor([encode(r['parameters']) for r in valid],dtype=torch.double)
        y = torch.tensor([[r['coverage']] for r in valid],dtype=torch.double)
        if not torch.isfinite(y).all() or not ((y >= 0) & (y <= 1)).all():
            raise ValueError('Invalid accepted coverage')
        model = SingleTaskGP(x,y,train_Yvar=torch.full_like(y,1e-6),outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model))
        acquisition = LogExpectedImprovement(model,best_f=y.max())
        candidates = torch.tensor(pool,dtype=torch.double)
        with torch.no_grad():
            scores = torch.cat([acquisition(chunk[:,None,:]) for chunk in candidates.split(256)]).numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite acquisition')
        # Second candidate is constrained to a distinct neighborhood to avoid
        # spending a parallel solve on a near-duplicate; not a qEI claim.
        indices = []
        for index in np.argsort(-scores,kind='stable'):
            if all(np.linalg.norm(np.asarray(pool[index])-pool[j]) >= .1 for j in indices):
                indices.append(int(index))
                if len(indices) == count: break
        if len(indices) != count:
            raise RuntimeError('Insufficient separated acquisition candidates')
        method = 'log_ei_discrete_pool_separated'
    return {'parameters':[physical[i] for i in indices], 'method':method,
            'seed':seed,'feasible_pool_size':len(pool),'valid_training_count':len(valid)}
