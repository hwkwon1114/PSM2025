"""Success-aware, two-objective hypervolume search for the joint calibration."""
import math

import numpy as np


CASES = ('run6', 'run10')  # historical identifiers: public Run 1, Run 3
FAILURES = ('solver', 'registration', 'timeout')


def case_errors(record):
    """Return finite positive paired discrepancies; failures have no objective values."""
    if not record['valid']:
        raise ValueError('A failed evaluation does not have paired objective values')
    errors = np.asarray([record['case_rms_mm'][case] for case in CASES], dtype=float)
    if not np.isfinite(errors).all() or (errors <= 0).any():
        raise ValueError('Both case discrepancies must be finite and positive')
    return errors


def nondominated(points):
    """Flags for minimization in both coordinates; equal points are retained."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite N-by-2 discrepancies')
    return np.array([not np.any(np.all(points <= p, axis=1) & np.any(points < p, axis=1))
                     for p in points], dtype=bool)


def observed_front(records):
    """Return observed valid nondominated records; preserve original IDs and order."""
    valid = [r for r in records if r['valid']]
    if not valid:
        return []
    return [r for r, keep in zip(valid, nondominated([case_errors(r) for r in valid])) if keep]


def propose(records, reference_mm, *, seed, batch_size=4, min_sep=0.01,
            pool_power=14, local_count=4000, mc_samples=128, chunk_size=256):
    """Greedy q=1 log-EHVI × estimated P(success), with posterior-mean batch fantasies.

    A random Sobol pool and local perturbations of the observed frontier are
    screened; this is an approximate acquisition optimizer, not a certificate.
    """
    import torch
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch.models import ModelListGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from calibration.ewcal.batch_bayes import fit_gp_model

    records = list(records)
    valid = [r for r in records if r['valid']]
    if len(valid) < 2:
        raise ValueError('At least two valid paired observations are required')
    if any(not r['valid'] and r.get('failure_class') not in FAILURES for r in records):
        raise ValueError('Infrastructure or unknown failures cannot be feasibility observations')
    X = np.asarray([r['x'] for r in records], dtype=float)
    if X.ndim != 2 or X.shape[1] != 4 or not np.isfinite(X).all() or (X < 0).any() or (X > 1).any():
        raise ValueError('Expected four finite normalized coordinates in [0, 1]')
    reference_mm = np.asarray(reference_mm, dtype=float)
    if reference_mm.shape != (2,) or not np.isfinite(reference_mm).all() or (reference_mm <= 0).any():
        raise ValueError('Expected two finite positive reference errors')

    torch.manual_seed(seed)
    models = []
    for i in range(2):
        observations = [{'normalized_parameters': r['x'], 'objective_mm': case_errors(r)[i]}
                        for r in valid]
        gp, _ = fit_gp_model(observations)
        models.append(gp)
    model = ModelListGP(*models)
    Y = torch.tensor(-np.log([case_errors(r) for r in valid]), dtype=torch.double)
    ref = torch.tensor(-np.log(reference_mm), dtype=torch.double)
    if not torch.any(torch.all(Y > ref, dim=1)):
        raise ValueError('No valid paired observation dominates the hypervolume reference')

    rng = np.random.default_rng(seed)
    pool = qmc.Sobol(4, seed=seed).random_base2(pool_power)
    fronts = np.asarray([r['x'] for r in observed_front(valid)], dtype=float)
    if local_count:
        centres = fronts[rng.integers(len(fronts), size=local_count)]
        local = np.clip(centres + rng.normal(0, 0.05, (local_count, 4)), 0, 1)
        pool = np.vstack([pool, local])
    labels = np.asarray([int(r['valid']) for r in records])
    if labels.min() == labels.max():
        logp = np.zeros(len(pool))
    else:
        kernel = ConstantKernel(1.0, (0.1, 10)) * Matern(
            length_scale=np.ones(4), length_scale_bounds=(0.1, 20), nu=2.5)
        classifier = GaussianProcessClassifier(kernel, n_restarts_optimizer=2, random_state=seed).fit(X, labels)
        logp = np.log(np.clip(classifier.predict_proba(pool)[:, 1], 1e-3, 1.0))

    batch, values = [], []
    for _ in range(batch_size):
        acquisition = qLogExpectedHypervolumeImprovement(
            model=model, ref_point=ref,
            partitioning=NondominatedPartitioning(ref_point=ref, Y=Y),
            sampler=SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(pool), chunk_size):
                P = torch.tensor(pool[start:start + chunk_size], dtype=torch.double).unsqueeze(1)
                chunks.append(acquisition(P).cpu().numpy())
        score = np.concatenate(chunks) + logp
        taken = np.vstack([X, *batch]) if batch else X
        for idx in np.argsort(-score):
            if math.isfinite(float(score[idx])) and np.all(np.linalg.norm(taken - pool[idx], axis=1) >= min_sep):
                break
        else:
            raise RuntimeError('No finite, sufficiently separated acquisition candidate')
        x = pool[idx].copy()
        batch.append(x)
        values.append(float(score[idx]))
        xt = torch.tensor(x[None, :], dtype=torch.double)
        if len(batch) < batch_size:
            with torch.no_grad():
                mean = model.posterior(xt).mean.detach().reshape(1, 2)
            model = model.condition_on_observations([xt, xt], mean)
            Y = torch.cat([Y, mean], dim=0)
    return np.asarray(batch), values
