#!/usr/bin/env python3
"""Tests of the smoothed dome objective and its post-solve connectivity gate (SI units)."""
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dome_bo'))

from dome_continuous_objective import (smooth_quantile, softplus, gauss_bonnet_integral,
                                       evaluate_dome_objective, material_areas, FLOOR)
from dome_curvature import coverage


def grid_mesh(nx=40, ny=60, half_u=0.1, half_v=0.15):
    """Structured triangulated sheet in metres; returns uv, faces, face centres."""
    u, v = np.linspace(-half_u, half_u, nx + 1), np.linspace(-half_v, half_v, ny + 1)
    U, V = np.meshgrid(u, v, indexing='ij')
    uv = np.column_stack((U.ravel(), V.ravel()))
    idx = lambda i, j: i * (ny + 1) + j
    faces = []
    for i in range(nx):
        for j in range(ny):
            faces += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    faces = np.array(faces)
    return uv, faces, uv[faces].mean(axis=1)


def evaluate(values, valid, uv, faces):
    return evaluate_dome_objective(values, valid, uv, faces)


class TestSmoothPrimitives(unittest.TestCase):

    def test_softplus_does_not_saturate(self):
        for v in (0.01, 0.5, 1.0, 2.0):
            self.assertAlmostEqual(float(softplus(np.array([v]), tau=1e-4)[0]), v, places=5)
        self.assertLess(float(softplus(np.array([-0.01]), tau=1e-4)[0]), 1e-10)

    def test_smooth_quantile_accuracy_shift_and_continuity(self):
        rng = np.random.default_rng(42)
        values, w = rng.normal(1.0, 0.2, 500), np.ones(500)
        q = smooth_quantile(values, w, q=0.20, epsilon=0.005)
        self.assertAlmostEqual(q, np.percentile(values, 20), delta=0.02)
        self.assertAlmostEqual(smooth_quantile(values + 0.5, w, q=0.20, epsilon=0.005), q + 0.5, delta=1e-4)
        vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        qs = [smooth_quantile(vals + d, np.ones(5), q=0.20, epsilon=0.01) for d in np.linspace(0, 0.1, 21)]
        self.assertTrue(np.all(np.diff(qs) > 0))
        self.assertLess(np.max(np.abs(np.diff(qs))), 0.02)

    def test_saddle_flux_fraction_is_scale_free(self):
        K = np.array([2.0, 1.0, -1.0, -0.5])
        w = np.ones(4)
        a = gauss_bonnet_integral(K, w)["saddle_flux_fraction"]
        b = gauss_bonnet_integral(0.25 * K, w)["saddle_flux_fraction"]
        self.assertAlmostEqual(a, 1.5 / 4.5, places=6)
        self.assertAlmostEqual(a, b, places=3)


class TestObjectiveAndGate(unittest.TestCase):

    def setUp(self):
        self.uv, self.faces, self.centers = grid_mesh()
        self.n = len(self.faces)
        self.valid = np.ones(self.n, bool)

    def values(self, k1, k2=None):
        k2 = k1 if k2 is None else k2
        return np.column_stack((np.full(self.n, k1, float), np.full(self.n, k2, float)))

    def patch_values(self, mask, c=3.0, other=(0.0, 0.0)):
        vals = np.tile(np.array(other, float), (self.n, 1))
        vals[mask] = c
        return vals

    def test_uniform_dome_is_qualified_for_either_orientation(self):
        pos, neg = evaluate(self.values(3.0), self.valid, self.uv, self.faces), \
                   evaluate(self.values(-3.0), self.valid, self.uv, self.faces)
        for r in (pos, neg):
            self.assertTrue(r["qualified"])
            self.assertAlmostEqual(r["largest_connected_fraction"], 1.0, places=9)
        self.assertEqual((pos["dominant_sign"], neg["dominant_sign"]), (1, -1))
        self.assertAlmostEqual(pos["j_combined"], neg["j_combined"], places=9)

    def test_weakening_the_dome_never_raises_the_objective(self):
        """Regression: halving curvature must not improve J, including for negative-oriented states."""
        mask = (np.abs(self.centers[:, 0]) < 0.07)
        for sign in (1, -1):
            base = sign * self.patch_values(mask, 3.0, other=(0.4, -0.3))
            j = [evaluate(base * s, self.valid, self.uv, self.faces)["j_combined"] for s in (0.25, 0.5, 1.0, 2.0)]
            self.assertTrue(all(b >= a - 1e-12 for a, b in zip(j, j[1:])), j)
            self.assertGreater(j[-1], j[0] + 1e-3)

    def test_weakening_never_raises_the_objective_at_physical_magnitudes_with_a_saddle_rim(self):
        """Regression (auditor): at realistic curvatures (~0.3 1/m, K ~ 0.03 1/m^2, near the saddle-share
        regulariser) a dome core with a saddle rim must score lower the more it is flattened."""
        core = (np.abs(self.centers[:, 0]) < 0.073) & (np.abs(self.centers[:, 1]) < 0.114)   # ~87 % of the ROI
        rim = ~core
        for sign in (1, -1):
            vals = np.zeros((self.n, 2))
            vals[core] = (0.3, 0.3)
            vals[rim] = (0.3, -0.12)
            j = [evaluate(sign * vals * s, self.valid, self.uv, self.faces)["j_combined"]
                 for s in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)]
            # Flattening never raises J (non-decreasing); J is exactly 0 where the smoothed 20th percentile
            # of the dome strength is <= 0 (below the 0.1 1/m gate floor) and strictly increasing once positive.
            self.assertTrue(all(b >= a for a, b in zip(j, j[1:])), (sign, j))
            positive = [x for x in j if x > 0.0]
            self.assertGreaterEqual(len(positive), 3, (sign, j))
            self.assertTrue(all(b > a for a, b in zip(positive, positive[1:])), (sign, j))
            self.assertGreater(j[-1], 0.0)

    def test_weakening_the_saved_solved_states_never_raises_the_objective(self):
        """Same property on real post-solve states (skipped if the run directory is absent)."""
        import json
        from dome_curvature import geometric_curvature
        found = 0
        for path in sorted((ROOT / 'run/dome').glob('*/*/solve/bilayer_zigzag_sequence_native_hybrid_state.json'))[:4] + \
                sorted((ROOT / 'run/dome').glob('proposed*/bilayer_zigzag_sequence_native_hybrid_state.json')):
            st = json.loads(path.read_text())
            uv, faces = np.asarray(st['material_uv'], float), np.asarray(st['faces'], int)
            vals, _, valid = geometric_curvature(np.asarray(st['positions'], float), uv, faces, radius=0.015)
            j = [evaluate(vals * s, valid, uv, faces)["j_combined"] for s in (0.1, 0.25, 0.5, 1.0)]
            self.assertTrue(all(b >= a for a, b in zip(j, j[1:])), (path.parent.name, j))      # never raised
            positive = [x for x in j if x > 0.0]                                                # strict once scored
            self.assertTrue(all(b > a for a, b in zip(positive, positive[1:])), (path.parent.name, j))
            found += 1
        if not found:
            self.skipTest('no saved solved states')

    def test_objective_is_continuous_across_the_gate_thresholds(self):
        """Regression (auditor): J must not jump when curvature crosses the hard-gate floor, and the gate
        decision must not leak into J."""
        lo = evaluate(self.values(FLOOR * (1 - 1e-5)), self.valid, self.uv, self.faces)
        hi = evaluate(self.values(FLOOR * (1 + 1e-5)), self.valid, self.uv, self.faces)
        self.assertNotEqual(lo["largest_connected_fraction"], hi["largest_connected_fraction"])  # gate flips
        self.assertAlmostEqual(lo["j_combined"], hi["j_combined"], delta=1e-4)
        # Anisotropy threshold crossing (|k|min = 0.1 |k|max) is also invisible to J.
        a = evaluate(self.values(3.0, 0.3 * (1 - 1e-5)), self.valid, self.uv, self.faces)
        b = evaluate(self.values(3.0, 0.3 * (1 + 1e-5)), self.valid, self.uv, self.faces)
        self.assertAlmostEqual(a["j_combined"], b["j_combined"], delta=1e-4)

    def test_objective_is_exactly_invariant_to_reversing_the_normal_even_below_the_floor(self):
        rng = np.random.default_rng(11)
        vals = rng.normal(0.0, 0.05, size=(self.n, 2))               # mixed-sign, mostly below the floor
        for scale in (1.0, 0.1):
            a = evaluate(vals * scale, self.valid, self.uv, self.faces)["j_combined"]
            b = evaluate(-vals * scale, self.valid, self.uv, self.faces)["j_combined"]
            self.assertAlmostEqual(a, b, places=12)
        for k in (0.099999, 0.100001):
            a = evaluate(self.values(k), self.valid, self.uv, self.faces)["j_combined"]
            b = evaluate(self.values(-k), self.valid, self.uv, self.faces)["j_combined"]
            self.assertAlmostEqual(a, b, places=12)

    def test_objective_is_lipschitz_in_a_smooth_curvature_sweep(self):
        ks = np.concatenate((np.linspace(0.0, 0.1, 101), np.linspace(0.1, 1.0, 91)[1:]))
        j = np.array([evaluate(self.values(k), self.valid, self.uv, self.faces)["j_combined"] for k in ks])
        slope = np.abs(np.diff(j)) / np.diff(ks)
        self.assertLess(slope.max(), 10.0)                          # Lipschitz: a jump would blow this up
        self.assertTrue(np.all(np.diff(j) >= -1e-9))                # stronger uniform dome never scores lower

    def test_cylinder_and_saddle_do_not_qualify(self):
        cyl = evaluate(self.values(3.0, 0.0), self.valid, self.uv, self.faces)
        sad = evaluate(self.values(3.0, -3.0), self.valid, self.uv, self.faces)
        self.assertFalse(cyl["qualified"])
        self.assertEqual(cyl["largest_connected_fraction"], 0.0)
        self.assertFalse(sad["qualified"])
        self.assertAlmostEqual(sad["saddle_flux_fraction"], 1.0, places=3)

    def test_below_floor_curvature_is_not_coherent(self):
        r = evaluate(self.values(0.5 * FLOOR), self.valid, self.uv, self.faces)
        self.assertEqual(r["largest_connected_fraction"], 0.0)

    def test_gate_threshold_is_80_percent_of_roi_area(self):
        roi_v = 0.12
        for frac, expect in ((0.78, False), (0.82, True)):
            cut = -roi_v + 2 * roi_v * frac
            mask = self.centers[:, 1] < cut
            r = evaluate(self.patch_values(mask), self.valid, self.uv, self.faces)
            self.assertEqual(r["qualified"], expect, (frac, r["largest_connected_fraction"]))
            self.assertAlmostEqual(r["largest_connected_fraction"], frac, delta=0.02)

    def test_disconnected_patches_count_by_largest_component(self):
        v = self.centers[:, 1]
        mask = ((v > -0.115) & (v < -0.05)) | ((v > 0.0) & (v < 0.115))   # two stripes with a gap
        r = evaluate(self.patch_values(mask), self.valid, self.uv, self.faces)
        self.assertGreater(r["total_coherent_fraction"], r["largest_connected_fraction"] + 0.1)
        self.assertFalse(r["qualified"])

    def test_invalid_faces_stay_in_the_denominator(self):
        valid = self.centers[:, 1] < 0.0           # half of the ROI invalid
        r = evaluate(self.values(3.0), valid, self.uv, self.faces)
        self.assertAlmostEqual(r["valid_area_fraction"], 0.5, delta=0.02)
        self.assertAlmostEqual(r["largest_connected_fraction"], 0.5, delta=0.02)
        self.assertFalse(r["qualified"])

    def test_saddle_region_lowers_objective_at_nearly_equal_dome_strength(self):
        """A saddle patch covering 10 % of the ROI (below the 20 % quantile mass, so the dome strength is
        almost unchanged) lowers J through the saddle-flux share: J ~ S (1 - f) with f = 0.1."""
        clean = self.values(3.0)
        saddled = clean.copy()
        saddled[self.centers[:, 0] > 0.064] = (3.0, -3.0)
        rc, rs = evaluate(clean, self.valid, self.uv, self.faces), evaluate(saddled, self.valid, self.uv, self.faces)
        self.assertAlmostEqual(rs["saddle_flux_fraction"], 0.1, delta=0.01)
        self.assertGreater(rc["j_combined"], rs["j_combined"] + 0.05 * rc["j_combined"])
        self.assertAlmostEqual(rs["j_combined"], rs["dome_strength"] * (1.0 - rs["saddle_flux_fraction"]), places=12)

    def test_objective_is_non_negative_and_zero_for_a_flat_state(self):
        flat = evaluate(self.values(0.0), self.valid, self.uv, self.faces)
        self.assertEqual((flat["j_combined"], flat["saddle_flux_fraction"]), (0.0, 0.0))
        half = np.zeros((self.n, 2))
        half[self.centers[:, 0] < 0.0] = 3.0
        self.assertGreaterEqual(evaluate(half, self.valid, self.uv, self.faces)["j_combined"], 0.0)

    def test_total_coherent_fraction_matches_repository_coverage(self):
        rng = np.random.default_rng(3)
        vals = rng.normal(0.0, 2.0, size=(self.n, 2))
        vals[rng.random(self.n) < 0.7] *= 0.0
        vals[:, 0] += 1.0
        r = evaluate(vals, self.valid, self.uv, self.faces)
        cov = coverage(vals, self.valid, self.uv, self.faces)["sign_fractions"]
        self.assertAlmostEqual(r["total_coherent_fraction"], cov[str(r["dominant_sign"])], places=9)

    def test_rejects_millimetre_uv(self):
        with self.assertRaises(ValueError):
            evaluate(self.values(3.0), self.valid, self.uv * 1000.0, self.faces)


if __name__ == "__main__":
    unittest.main()
