import sys
import torch
import numpy as np

torch.set_default_dtype(torch.double)

print("Importing botorch modules...")
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf

print("Preparing tensors...")
X_valid = torch.tensor([
    [0.477, 0.677, 0.25, 0.25],
    [0.127, 0.216, 0.253, 0.273],
    [0.738, 0.744, 0.747, 0.261],
    [0.457, 0.89, 0.021, 0.712],
    [0.591, 0.132, 0.344, 0.108],
    [0.569, 0.332, 0.0, 0.484],
    [0.802, 0.904, 0.781, 0.875],
], dtype=torch.double)

y_valid = torch.tensor([
    [-np.log(17.266)],
    [-np.log(7.2575)],
    [-np.log(18.347)],
    [-np.log(19.309)],
    [-np.log(20.784)],
    [-np.log(24.473)],
    [-np.log(22.654)],
], dtype=torch.double)

X_all = torch.tensor([
    [0.477, 0.677, 0.25, 0.25],
    [0.527, 0.529, 0.735, 0.284],
    [0.127, 0.216, 0.253, 0.273],
    [0.249, 0.299, 0.954, 0.437],
    [0.895, 0.446, 0.102, 0.836],
    [0.738, 0.744, 0.747, 0.261],
    [0.457, 0.89, 0.021, 0.712],
    [0.997, 0.37, 0.337, 0.549],
    [0.803, 0.701, 0.423, 0.523],
    [0.945, 0.975, 0.672, 0.046],
    [0.591, 0.132, 0.344, 0.108],
    [0.033, 0.137, 0.722, 0.0],
    [0.569, 0.332, 0.0, 0.484],
    [0.802, 0.904, 0.781, 0.875],
    [0.0, 0.172, 1.0, 0.808],
    [0.0, 0.939, 0.018, 0.665],
    [0.597, 0.335, 0.453, 0.598],
    [0.832, 0.508, 0.126, 0.052],
    [0.0, 0.849, 0.0, 0.0],
    [0.209, 0.187, 0.413, 0.458],
], dtype=torch.double)

g_all = torch.tensor([
    [-0.5911],
    [0.0500],
    [-0.4479],
    [0.0500],
    [0.0500],
    [-0.9337],
    [-0.6050],
    [0.0800],
    [0.0500],
    [0.0500],
    [-0.6273],
    [0.0800],
    [-0.6329],
    [-0.1045],
    [0.0500],
    [0.0800],
    [0.5000],
    [0.0500],
    [0.0800],
    [0.0500],
], dtype=torch.double)

print("Fitting Model 0 (Objective GP)...")
m0 = SingleTaskGP(X_valid, y_valid, outcome_transform=Standardize(m=1))
mll0 = ExactMarginalLogLikelihood(m0.likelihood, m0)
fit_gpytorch_mll(mll0)

print("Fitting Model 1 (Constraint GP)...")
m1 = SingleTaskGP(X_all, g_all, outcome_transform=Standardize(m=1))
mll1 = ExactMarginalLogLikelihood(m1.likelihood, m1)
fit_gpytorch_mll(mll1)

model_list = ModelListGP(m0, m1)
print("ModelListGP successfully created!")

best_f = y_valid.max()
print(f"best_f = {best_f.item():.4f}")

try:
    print("Initializing qLogExpectedImprovement with constraints...")
    acq = qLogExpectedImprovement(
        model=model_list,
        best_f=best_f,
        objective=GenericMCObjective(objective=lambda Y, X=None: Y[..., 0]),
        constraints=[lambda Y: Y[..., 1]],  # evaluates g(x) <= 0
    )
    print("Acq initialized cleanly!")

    bounds = torch.stack([torch.zeros(4), torch.ones(4)])
    print("Optimizing acquisition...")
    candidates, acq_val = optimize_acqf(
        acq,
        bounds=bounds,
        q=4,
        num_restarts=4,
        raw_samples=64,
    )
    print(f"Acquisition value: {acq_val.item():.4f}")
    print("Candidates:")
    for i, c in enumerate(candidates):
        with torch.no_grad():
            post_g = m1.posterior(c.unsqueeze(0))
            mean_g = post_g.mean.item()
        print(f"  Candidate {i+1}: {c.tolist()}, predicted g(x)={mean_g:+.4f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
