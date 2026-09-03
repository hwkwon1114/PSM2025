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
from python.active_sampling.spatial_kernel_design import (
    constrained_information_greedy,
    information_gain,
    raster_convergence_diagnostic,
    rbf_kernel,
    spatial_field_features,
    spatial_treatment_fields,
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

    def test_spatial_fields_are_symmetric_target_metric_increments(self) -> None:
        fields = spatial_treatment_fields(self.samples[:8], grid_shape=(10, 8))
        self.assertEqual(fields.shape, (8, 10, 8, 6))
        self.assertTrue(np.all(np.isfinite(fields)))
        self.assertGreater(np.max(np.abs(fields)), 0.0)

        features, scales = spatial_field_features(fields)
        self.assertEqual(features.shape, (8, 10 * 8 * 6))
        self.assertEqual(scales.shape, (6,))
        self.assertTrue(np.all(scales > 0.0))
        reused, returned_scales = spatial_field_features(
            fields,
            channel_scale=scales,
        )
        np.testing.assert_allclose(reused, features)
        np.testing.assert_allclose(returned_scales, scales)

    def test_spatial_rbf_kernel_is_positive_semidefinite(self) -> None:
        fields = spatial_treatment_fields(self.samples[:24], grid_shape=(10, 8))
        features, _ = spatial_field_features(fields)
        kernel, bandwidth = rbf_kernel(features)
        self.assertGreater(bandwidth, 0.0)
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-10)

    def test_spatial_kernel_converges_with_subsampling(self) -> None:
        diagnostic = raster_convergence_diagnostic(
            self.samples[:24],
            grid_shape=(8, 6),
            tested_subsamples=(2, 4),
            reference_subsamples=8,
        )
        coarse = diagnostic["comparisons"]["2"]
        fine = diagnostic["comparisons"]["4"]
        self.assertLess(
            fine["relative_kernel_frobenius_error"],
            coarse["relative_kernel_frobenius_error"],
        )

    def test_spatial_information_design_is_unique_and_reproducible(self) -> None:
        schedule = balanced_schedule(48)
        fields = spatial_treatment_fields(self.samples, grid_shape=(10, 8))
        features, _ = spatial_field_features(fields)
        kernel, _ = rbf_kernel(features)
        first, conditional = constrained_information_greedy(
            kernel, self.samples, schedule
        )
        second, _ = constrained_information_greedy(
            kernel, self.samples, schedule
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 48)
        self.assertTrue(np.all(conditional > 0.0))
        self.assertGreater(information_gain(kernel, first, 1e-4), 0.0)


if __name__ == "__main__":
    unittest.main()
