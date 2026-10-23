#!/usr/bin/env python3
"""Unit tests for warped-feature GP surrogate and feasibility classifier."""
import sys
import warnings
from pathlib import Path
import unittest
import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dome_bo'))

from crosshatch_generator import generator_from_raw
from dome_gp_surrogate import (
    warp_features,
    DomeGPSurrogate,
    DEFAULT_BOUNDS,
    FEATURE_NAMES,
    REFERENCE_LENGTH_MM,
)


class TestDomeGPSurrogate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore')

    def test_warp_features_transformation(self):
        """Recipe-measured raw coordinates -> log dose, logit balance, power, log aspect, log pitch/L."""
        raw = np.array([4.0, 0.25, 2.0, 1.25, 19.0, 47.5])
        warped = warp_features(raw)
        self.assertEqual(len(warped), 6)
        self.assertAlmostEqual(warped[0], np.log(4.0))
        self.assertAlmostEqual(warped[1], np.log(0.25 / 0.75))
        self.assertAlmostEqual(warped[2], 2.0)
        self.assertAlmostEqual(warped[3], np.log(1.25))
        self.assertAlmostEqual(warped[4], np.log(19.0 / REFERENCE_LENGTH_MM))
        self.assertAlmostEqual(warped[5], np.log(47.5 / REFERENCE_LENGTH_MM))
        self.assertTrue(np.allclose(warp_features(np.vstack([raw, raw]))[1], warped))

    def test_surrogate_fitting_and_prediction(self):
        """Verifies that GP regressor and classifier fit and predict correctly."""
        np.random.seed(42)
        low = np.array([b[0] for b in DEFAULT_BOUNDS])
        high = np.array([b[1] for b in DEFAULT_BOUNDS])
        X = np.random.uniform(low, high, size=(15, 6))

        # Synthetic outputs
        y = np.sum(X[:, :2], axis=1)
        z = (X[:, 0] > 2.5).astype(int)

        surrogate = DomeGPSurrogate()
        surrogate.fit(X, y, z)

        self.assertTrue(surrogate.is_fitted)

        # Predict on training points
        preds = surrogate.predict(X)
        self.assertEqual(len(preds["mean"]), 15)
        self.assertEqual(len(preds["std"]), 15)
        self.assertEqual(len(preds["p_feasible"]), 15)

        # Regressor should have low error on training points
        rmse = np.sqrt(np.mean((preds["mean"] - y) ** 2))
        self.assertLess(rmse, 0.1)

    def test_constrained_expected_improvement_penalty(self):
        """Verifies that CEI is heavily suppressed where feasibility probability is low."""
        np.random.seed(42)
        low = np.array([b[0] for b in DEFAULT_BOUNDS])
        high = np.array([b[1] for b in DEFAULT_BOUNDS])
        X = np.random.uniform(low, high, size=(20, 6))

        # Feasibility is only 1 when X[:, 0] > 3.5
        z = (X[:, 0] > 3.5).astype(int)
        y = X[:, 0] * 2.0

        surrogate = DomeGPSurrogate()
        surrogate.fit(X, y, z)

        # Test point in infeasible region
        x_infeas = np.array([1.0, 1.0, 2.0, 1.0, 6.0, 4.0])
        preds_infeas = surrogate.predict(x_infeas)
        self.assertLess(preds_infeas["p_feasible"], 0.25)

        cei = surrogate.constrained_expected_improvement(x_infeas, y_best=5.0)
        # CEI should be nearly zero due to low feasibility
        self.assertLess(cei, 0.05)

    def test_candidate_proposal_respects_bounds(self):
        """Verifies that proposed candidate lies strictly within physical bounds."""
        np.random.seed(42)
        low = np.array([b[0] for b in DEFAULT_BOUNDS])
        high = np.array([b[1] for b in DEFAULT_BOUNDS])
        X = np.random.uniform(low, high, size=(12, 6))
        y = np.random.normal(size=12)
        z = np.random.choice([0, 1], size=12)

        surrogate = DomeGPSurrogate()
        surrogate.fit(X, y, z)

        candidate = surrogate.propose_candidate(bounds=DEFAULT_BOUNDS, n_candidates=100)
        self.assertEqual(len(candidate), 6)
        for val, (b_low, b_high) in zip(candidate, DEFAULT_BOUNDS):
            self.assertGreaterEqual(val, b_low - 1e-6)
            self.assertLessEqual(val, b_high + 1e-6)


    # ---- failure learning -------------------------------------------------------------------
    @staticmethod
    def _points(n, seed):
        rng = np.random.default_rng(seed)
        low = np.array([b[0] for b in DEFAULT_BOUNDS]); high = np.array([b[1] for b in DEFAULT_BOUNDS])
        return rng.uniform(low, high, size=(n, 6))

    def test_unscored_failed_solves_update_the_feasibility_model(self):
        """A solver failure has no objective (NaN) but must still teach the classifier."""
        X = self._points(10, 1)
        z = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        y = np.where(z == 1, X[:, 0], np.nan)               # failed solves (z=0) are unscored
        s = DomeGPSurrogate(); s.fit(X, y, z)
        summary = s.summary()
        self.assertEqual((summary["n_samples"], summary["n_scored"], summary["n_infeasible"]), (10, 6, 4))
        self.assertEqual(len(s.regressor.X_train_), 6)       # regressor only sees scored solves
        self.assertEqual(len(s._Phi_scaled), 10)              # classifier sees every observation
        self.assertTrue(np.isfinite(s.constrained_expected_improvement(X[:3])).all())
        with self.assertRaises(ValueError):                    # a feasible observation must be scored
            DomeGPSurrogate().fit(X, np.where(z == 1, np.nan, 1.0), z)

    def test_all_infeasible_observations_keep_feasibility_and_acquisition_alive(self):
        X = self._points(6, 2)
        for y in (np.full(6, np.nan), np.linspace(-1.0, 0.0, 6)):     # all unscored / all scored-but-failed
            s = DomeGPSurrogate(); s.fit(X, y, np.zeros(6, int))
            far = self._points(40, 3)
            p = s.predict(far)["p_feasible"]
            self.assertTrue(((p > 0.0) & (p < 1.0)).all())
            self.assertGreater(float(np.max(s.constrained_expected_improvement(far))), 0.0)
            near = s.predict(X[0])["p_feasible"]
            self.assertLess(near, float(np.max(p)))            # lower where a failure was observed
            proposal = s.propose(n_candidates=64, n_local=16, rng=np.random.default_rng(4))
            self.assertGreater(proposal["cei"], 0.0)

    def test_single_scored_solve_falls_back_to_feasibility_acquisition(self):
        X = self._points(5, 5)
        y = np.array([np.nan, np.nan, 0.3, np.nan, np.nan]); z = np.array([0, 0, 1, 0, 0])
        s = DomeGPSurrogate(); s.fit(X, y, z)
        self.assertIsNone(s.regressor)
        far = self._points(8, 6)
        self.assertTrue(np.allclose(s.constrained_expected_improvement(far), s.predict(far)["p_feasible"]))

    # ---- acquisition on realized coordinates ------------------------------------------------
    def test_acquisition_is_scored_at_realized_not_requested_coordinates(self):
        realize = lambda r: generator_from_raw(r).realized_raw()
        X_req = self._points(8, 7)
        X = np.array([realize(r) for r in X_req])
        s = DomeGPSurrogate(); s.fit(X, np.sin(X[:, 0]) + 0.1 * X[:, 2], np.array([1, 0, 1, 1, 0, 1, 1, 0]))
        proposal = s.propose(n_candidates=40, n_local=10, realize=realize, rng=np.random.default_rng(8))
        req, real = proposal["request"], proposal["realized"]
        self.assertTrue(np.allclose(real, realize(req)))
        self.assertFalse(np.allclose(req, real), "the request is not the realized recipe (integer sweeps, sheet fit)")
        self.assertAlmostEqual(proposal["cei"], float(s.constrained_expected_improvement(real)), places=12)
        # a small-dose request realizes a much larger dose (>= 2 hits), so scoring the request would mislead
        small = np.array([0.5, 0.5, 2.0, 1.2, 14.0, 14.0])
        self.assertGreater(realize(small)[0], 1.5 * small[0])


if __name__ == "__main__":
    unittest.main()
