#!/usr/bin/env python3
"""Warped-feature Gaussian Process surrogate with anisotropic Matérn-5/2 ARD kernel and Bernoulli feasibility classifier.

Implements the council-recommended surrogate architecture:
1. Physical coordinate warping of recipe-measured coordinates: log total dose, logit balance, grading power, log aspect ratio, log centre stroke pitches (stroke layout) or innermost terrace spacing (terrace layout).
2. Anisotropic Matérn-5/2 ARD GaussianProcessRegressor predicting continuous curvature objective J_combined.
3. GaussianProcessClassifier modeling the non-convex solver-convergence boundary (accepted equilibrium); dome success is NOT part of the label.
4. Constrained Expected Improvement (CEI = EI * P_feasible) acquisition function.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, RBF
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)


FEATURE_NAMES = [
    "log_total_dose",       # ln(effective Voce dose at the centre, face-shared)
    "logit_balance",        # logit(share of centre sweeps given to orientation A)
    "decay_power",          # alpha -> pitch-grading exponent of the terrace spacing
    "log_aspect_ratio",     # ln(along/cross capsule extent)
    "log_pitch_a",          # ln(centre stroke pitch of A [stroke] or innermost terrace spacing [terrace] / ref. half-width)
    "log_pitch_b"           # ln(same for orientation B)
]

# Raw coordinates are the quantities MEASURED on the emitted recipe
# (crosshatch_generator.raw_from_recipe): [total_dose, balance, decay_power, aspect, pitch_a, pitch_b]
DEFAULT_BOUNDS = [
    (0.5, 4.8),     # total_dose (combined Voce effective dose; asymptote 4.95 single-pass equivalents)
    (0.15, 0.85),   # dose_balance (A share of the centre sweeps)
    (1.0, 3.5),     # decay_power
    (1.0, 1.5),     # aspect_ratio (along/cross capsule extent; generator requires >= 1)
    (8.0, 30.0),    # center_pitch_a (mm, innermost terrace spacing; generator rescales to fit the sheet)
    (8.0, 30.0)     # center_pitch_b
]
PARAM_NAMES = ["total_dose", "dose_balance", "decay_power", "aspect_ratio", "center_pitch_a", "center_pitch_b"]
REFERENCE_LENGTH_MM = 95.0   # sheet envelope half-width (crosshatch_generator.A_E_MM)


def warp_features(raw_params: np.ndarray) -> np.ndarray:
    """Transforms recipe-measured raw coordinates into warped dimensionless coordinates."""
    raw = np.atleast_2d(np.asarray(raw_params, float))
    dose = np.clip(raw[:, 0], 1e-3, None)
    balance = np.clip(raw[:, 1], 1e-3, 1.0 - 1e-3)
    warped = np.column_stack([
        np.log(dose),
        np.log(balance / (1.0 - balance)),
        raw[:, 2],
        np.log(np.clip(raw[:, 3], 1e-3, None)),
        np.log(np.clip(raw[:, 4], 1e-3, None) / REFERENCE_LENGTH_MM),
        np.log(np.clip(raw[:, 5], 1e-3, None) / REFERENCE_LENGTH_MM),
    ])
    return warped.squeeze() if np.ndim(raw_params) == 1 else warped


class DomeGPSurrogate:
    """Surrogate modeling pipeline for double-curvature dome optimization."""

    def __init__(self, warp=None, n_features: int = 6, feature_names=None):
        """``warp`` maps recipe-measured raw coordinates to warped features (default: the 6-coordinate
        cross-hatch warp); ``n_features`` is the warped dimension."""
        self.warp = warp_features if warp is None else warp
        self.n_features = int(n_features)
        self.feature_names = FEATURE_NAMES if feature_names is None else list(feature_names)
        self.scaler = StandardScaler()
        self.regressor = None
        self.classifier = None
        self.X_raw = None
        self.y = None
        self.z = None
        self.is_fitted = False

    def fit(self, X_raw: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Fits the continuous GP regressor and the feasibility classifier.

        Parameters:
            X_raw: (N, 6) recipe-MEASURED coordinates (crosshatch_generator.raw_from_recipe).
            y: (N,) objective J_combined; ``NaN`` for solves that produced no scoreable state
               (solver failure). Failed solves still inform the feasibility model.
            z: (N,) feasibility labels (1 = native equilibrium accepted, 0 = numerical non-acceptance).
               A feasible observation must be scored.
        """
        self.X_raw = np.asarray(X_raw, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=int)
        N = len(self.X_raw)
        if N < 1 or self.y.shape != (N,) or self.z.shape != (N,):
            raise ValueError("X_raw, y and z must describe the same N >= 1 observations.")
        if np.any((self.z == 1) & ~np.isfinite(self.y)):
            raise ValueError("A feasible observation (z=1) must have a finite objective.")
        scored = np.isfinite(self.y)

        Phi = np.atleast_2d(self.warp(self.X_raw))
        Phi_scaled = self.scaler.fit_transform(Phi)
        self._Phi_scaled = Phi_scaled

        # 1. Anisotropic Matérn-5/2 ARD regressor on the scored observations only.
        init_lengths = np.ones(self.n_features)
        if scored.sum() >= 2:
            kernel_reg = (
                ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e3))
                * Matern(length_scale=init_lengths, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
            )
            self.regressor = GaussianProcessRegressor(kernel=kernel_reg, n_restarts_optimizer=5,
                                                      normalize_y=True, random_state=42)
            self.regressor.fit(Phi_scaled[scored], self.y[scored])
        else:
            self.regressor = None                       # not enough scored solves: acquisition = feasibility

        # 2. Feasibility model on ALL observations (failures included).
        if len(np.unique(self.z)) > 1:
            kernel_clf = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=init_lengths,
                                                                    length_scale_bounds=(1e-2, 1e2), nu=2.5)
            self.classifier = GaussianProcessClassifier(kernel=kernel_clf, n_restarts_optimizer=3,
                                                        random_state=42)
            self.classifier.fit(Phi_scaled, self.z)
        else:
            self.classifier = None                      # single class: smoothed Beta estimate (see _p_feasible)
        self.is_fitted = True

    def _p_feasible(self, Phi_scaled: np.ndarray) -> np.ndarray:
        """P(feasible). Two classes: GP classifier. One class: kernel-smoothed Beta(1,1) posterior mean
        (1 + sum_i k(x,x_i) z_i) / (2 + sum_i k(x,x_i)); it is never exactly 0 or 1, drops near observed
        failures and stays at the prior 1/2 far from all data, so exploration is not killed."""
        if self.classifier is not None:
            classes = list(self.classifier.classes_)
            return self.classifier.predict_proba(Phi_scaled)[:, classes.index(1)]
        d2 = ((Phi_scaled[:, None, :] - self._Phi_scaled[None, :, :]) ** 2).sum(axis=2)
        k = np.exp(-0.5 * d2 / 1.0 ** 2)
        return (1.0 + k @ (self.z == 1).astype(float)) / (2.0 + k.sum(axis=1))

    def predict(self, X_raw: np.ndarray) -> dict:
        """Posterior mean, std and feasibility probability at recipe-measured coordinates.
        ``mean``/``std`` are NaN when fewer than two scored solves exist."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate model must be fitted before predict.")
        X_raw = np.asarray(X_raw, float)
        Phi_scaled = self.scaler.transform(np.atleast_2d(self.warp(X_raw)))
        p_feasible = self._p_feasible(Phi_scaled)
        if self.regressor is not None:
            mu, sigma = self.regressor.predict(Phi_scaled, return_std=True)
        else:
            mu = sigma = np.full(len(Phi_scaled), np.nan)
        one = X_raw.ndim == 1
        return {"mean": mu.squeeze() if one else mu, "std": sigma.squeeze() if one else sigma,
                "p_feasible": p_feasible.squeeze() if one else p_feasible}

    def _y_best(self) -> float:
        scored = np.isfinite(self.y)
        feasible = scored & (self.z == 1)
        pool = feasible if feasible.any() else scored
        return float(np.max(self.y[pool])) if pool.any() else 0.0

    def constrained_expected_improvement(self, X_raw: np.ndarray, y_best: float = None, xi: float = 0.01) -> np.ndarray:
        """CEI = EI * P_feasible at recipe-measured coordinates. With fewer than two scored solves there
        is no objective model and the acquisition is the feasibility probability itself."""
        X_raw = np.asarray(X_raw, float)
        preds = self.predict(X_raw)
        p_feas = np.atleast_1d(preds["p_feasible"]).astype(float)
        if self.regressor is None:
            return p_feas.squeeze() if X_raw.ndim == 1 else p_feas
        y_best = self._y_best() if y_best is None else y_best
        mu, sigma = np.atleast_1d(preds["mean"]), np.atleast_1d(preds["std"])
        improvement = mu - y_best - xi
        cei = np.zeros_like(mu)
        ok = sigma > 1e-8
        if np.any(ok):
            z = improvement[ok] / sigma[ok]
            ei = improvement[ok] * norm.cdf(z) + sigma[ok] * norm.pdf(z)
            cei[ok] = np.maximum(0.0, ei) * p_feas[ok]
        return cei.squeeze() if X_raw.ndim == 1 else cei

    def propose(self, bounds: list = DEFAULT_BOUNDS, n_candidates: int = 2000, xi: float = 0.01,
                realize=None, rng=None, n_local: int = 200) -> dict:
        """Proposes the next design by maximising CEI evaluated at the REALIZED coordinates.

        ``realize`` maps a requested parameter vector to the coordinates measured on the recipe it
        produces (e.g. ``lambda r: generator_from_raw(r).realized_raw()``); acquisition is scored there,
        consistently with how the surrogate is trained. Without ``realize`` the request is assumed to be
        realized exactly. The realization is piecewise constant (integer sweeps/repeats), so refinement is
        a derivative-free local random search. Returns {"request", "realized", "cei"}."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate model must be fitted before proposing.")
        rng = np.random.default_rng() if rng is None else rng
        low = np.array([b[0] for b in bounds], float)
        high = np.array([b[1] for b in bounds], float)
        realize = (lambda r: np.asarray(r, float)) if realize is None else realize
        y_best = self._y_best() if self.regressor is not None else 0.0

        def score(requests):
            realized = np.array([realize(r) for r in requests], float)
            return realized, self.constrained_expected_improvement(realized, y_best, xi=xi)

        requests = rng.uniform(low, high, size=(n_candidates, len(bounds)))
        realized, cei = score(requests)
        best = int(np.argmax(cei))
        best_req, best_real, best_cei = requests[best], realized[best], float(cei[best])
        for scale in (0.15, 0.05):                       # shrinking local search around the incumbent
            local = np.clip(best_req + rng.normal(0.0, scale * (high - low), size=(n_local, len(bounds))), low, high)
            realized, cei = score(local)
            j = int(np.argmax(cei))
            if cei[j] > best_cei:
                best_req, best_real, best_cei = local[j], realized[j], float(cei[j])
        return {"request": best_req, "realized": best_real, "cei": best_cei}

    def propose_candidate(self, bounds: list = DEFAULT_BOUNDS, n_candidates: int = 2000, xi: float = 0.01,
                          realize=None, rng=None) -> np.ndarray:
        """Requested parameter vector of :meth:`propose`."""
        return self.propose(bounds, n_candidates, xi, realize=realize, rng=rng)["request"]

    def summary(self) -> dict:
        """Returns model hyperparameters and diagnostics."""
        if not self.is_fitted:
            return {"status": "unfitted"}
        if self.regressor is None:
            return {"n_samples": len(self.X_raw), "n_scored": int(np.isfinite(self.y).sum()),
                    "n_feasible": int(np.sum(self.z == 1)), "n_infeasible": int(np.sum(self.z == 0)),
                    "status": "feasibility-only (fewer than two scored solves)"}

        k_reg = self.regressor.kernel_
        length_scales = k_reg.k1.k2.length_scale.tolist() if hasattr(k_reg, 'k1') else []
        noise = k_reg.k2.noise_level if hasattr(k_reg, 'k2') else 0.0

        return {
            "n_samples": len(self.X_raw),
            "n_scored": int(np.isfinite(self.y).sum()),
            "n_feasible": int(np.sum(self.z == 1)),
            "n_infeasible": int(np.sum(self.z == 0)),
            "feature_length_scales": dict(zip(self.feature_names, length_scales)),
            "noise_level": float(noise),
            "log_marginal_likelihood": float(self.regressor.log_marginal_likelihood_value_)
        }


def main():
    parser = argparse.ArgumentParser(description="Dome Gaussian Process Surrogate Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run self-contained demonstration with synthetic data")
    args = parser.parse_args()

    if args.demo:
        print("Running Dome GP Surrogate demonstration...")
        np.random.seed(42)
        low = np.array([b[0] for b in DEFAULT_BOUNDS])
        high = np.array([b[1] for b in DEFAULT_BOUNDS])
        X = np.random.uniform(low, high, size=(25, len(DEFAULT_BOUNDS)))

        # Synthetic ground truth (demo only): prefers balanced dose, high grading power, aspect ~ 1.25
        dose, balance, alpha, gamma_e, pa, pb = X.T
        z = ((np.abs(balance - 0.5) < 0.3) & (pa <= 26.0) & (pb <= 26.0)).astype(int)
        y = np.sin(dose) - np.abs(balance - 0.5) + 0.5 * alpha - 0.2 * (gamma_e - 1.25) ** 2 + np.random.normal(0, 0.02, 25)

        surrogate = DomeGPSurrogate()
        surrogate.fit(X, y, z)

        summary = surrogate.summary()
        print("\n--- Fitted Surrogate Summary ---")
        print(json.dumps(summary, indent=2))

        candidate = surrogate.propose_candidate()
        preds = surrogate.predict(candidate)
        print("\n--- Proposed Next Candidate Design ---")
        for name, val in zip(PARAM_NAMES, candidate):
            print(f"  {name}: {val:.3f}")
        print(f"Predicted mean J_combined: {preds['mean']:.4f}")
        print(f"Predicted std: {preds['std']:.4f}")
        print(f"Predicted P(feasible): {preds['p_feasible']:.2%}")


if __name__ == "__main__":
    main()
