"""Regression tests for the metric-controlled initial-design benchmark."""

from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from python.active_sampling.initial_design_benchmark import (
    DistanceOracle,
    balanced_schedule,
    constrained_farthest_first,
    d2_orbit,
    ordered_trajectory,
)
from python.active_sampling.zigzag_action import generate_actions


class InitialDesignBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, _ = generate_actions(120, seed=7301)

    def test_d2_orbit_preserves_order_and_process_channels(self) -> None:
        curve = ordered_trajectory(self.samples[0], n_points=10)
        orbit = d2_orbit(curve)
        expected_process = np.broadcast_to(curve[None, :, 2:], orbit[:, :, 2:].shape)
        np.testing.assert_allclose(orbit[:, :, 2:], expected_process)
        np.testing.assert_allclose(orbit[1, :, 0], -curve[:, 0])
        np.testing.assert_allclose(orbit[1, :, 1], curve[:, 1])
        self.assertFalse(np.allclose(orbit[0], curve[::-1]))

    def test_d2_metrics_identify_same_symmetry_orbit(self) -> None:
        for metric in ("d2_ordered_l2", "d2_frechet", "d2_dtw"):
            oracle = DistanceOracle(self.samples, metric=metric, n_points=8)
            curve = oracle.curves[0]
            reflected = curve.copy()
            reflected[:, 0] *= -1.0
            original = oracle.curves[0]
            oracle.curves[0] = reflected
            self.assertAlmostEqual(oracle.from_index(0)[0], 0.0, places=12)
            oracle.curves[0] = original

    def test_balanced_schedule_has_expected_counts(self) -> None:
        schedule = balanced_schedule(500)
        classes = [category for category, _ in schedule]
        self.assertEqual(classes.count("local"), 150)
        self.assertEqual(classes.count("medium"), 250)
        self.assertEqual(classes.count("near_full"), 100)
        strip_counts = Counter(strip for _, strip in schedule)
        self.assertLessEqual(max(strip_counts.values()) - min(strip_counts.values()), 1)

    def test_small_maximin_design_is_unique_and_reproducible(self) -> None:
        schedule = balanced_schedule(48)
        oracle = DistanceOracle(self.samples, metric="d2_ordered_l2", n_points=8)
        first, insertion, _ = constrained_farthest_first(self.samples, schedule, oracle)
        second, _, _ = constrained_farthest_first(self.samples, schedule, oracle)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 48)
        self.assertTrue(np.all(insertion[1:] >= 0.0))


if __name__ == "__main__":
    unittest.main()
