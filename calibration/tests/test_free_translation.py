"""Solver-free controls for the bounded saved-shape translation diagnostic."""
import copy
import unittest
import numpy as np
from calibration.analysis import diagnose_free_translation as diagnostic
from calibration.ewcal.geometry import align_target, triangle_mesh


class FreeTranslationTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [.2, .4, 4.]])
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        self.mesh = triangle_mesh(self.p, self.faces)
        self.settings = {'rotation_bound_deg': 5, 'translation_bounds_mm': [5, 5, 20],
                         'fit_points': 2000, 'icp_iterations': 240, 'metric': 'symmetric'}

    def test_free_bounds_only_change_translation(self):
        before = copy.deepcopy(self.settings)
        free = diagnostic.free_settings(self.settings)
        self.assertEqual(self.settings, before)
        self.assertTrue(np.isposinf(free['translation_bounds_mm']).all())
        self.assertEqual({k: v for k, v in free.items() if k != 'translation_bounds_mm'},
                         {k: v for k, v in before.items() if k != 'translation_bounds_mm'})

    def test_known_offset_outside_cap_is_rescued_without_shape_change(self):
        # Keep the known offset within ICP's local capture range, but outside the cap.
        p = self.p * 10
        mesh = triangle_mesh(p, self.faces)
        target = p - [0., 8., 0.]
        saved = target.copy()
        bounded, _, _, _ = align_target(mesh, target, self.settings, p, self.faces)
        free, _, _, _ = align_target(mesh, target, diagnostic.free_settings(self.settings), p, self.faces)
        self.assertFalse(bounded['registration_qualified'])
        self.assertTrue(bounded['pose_near_bound'])
        self.assertTrue(free['registration_qualified'])
        # Production ICP stops at 0.005mm motion or 1e-4mm cost stagnation,
        # so a known-pose control must use that accuracy rather than exact interpolation.
        self.assertLess(free['rms_symmetric_mm'], .005)
        np.testing.assert_allclose(free['pose'][3:], [0., 8., 0.], atol=.005)
        np.testing.assert_array_equal(target, saved)
        from calibration.ewcal.geometry import points
        np.testing.assert_array_equal(points(mesh), p)

    def test_free_translation_keeps_rotation_gate(self):
        # Proper near-limit rotation is still rejected even with infinite translations.
        from scipy.spatial.transform import Rotation
        target = self.p @ Rotation.from_rotvec([0., 0., np.deg2rad(4.9)]).as_matrix()
        metrics, _, _, _ = align_target(self.mesh, target, diagnostic.free_settings(self.settings), self.p, self.faces)
        self.assertTrue(metrics['registration_converged'])
        self.assertGreater(metrics['rotation_angle_deg'], 4.75)
        self.assertTrue(metrics['pose_near_bound'])
        self.assertFalse(metrics['registration_qualified'])

    def test_proximity_weights_both_full_directions(self):
        result = diagnostic.proximity([0., 3., 5., 9.], [0., 4., 10.], [.1, .2, .7])
        self.assertEqual(result['3']['scan_fraction'], .5)
        self.assertAlmostEqual(result['3']['simulation_area_fraction'], .1)
        self.assertAlmostEqual(result['5']['scan_fraction'], .75)
        self.assertAlmostEqual(result['5']['simulation_area_fraction'], .3)
        with self.assertRaises(AssertionError):
            diagnostic.proximity([np.nan], [0.], [1.])

    def test_timeout_exclusion_duplicate_and_missing_evidence_refused(self):
        good = {'attempt_id': 1, 'valid': True, 'metrics': {'pose': [0.] * 6}}
        timeout = {'attempt_id': 2, 'valid': False, 'failure_class': 'timeout'}
        selected, excluded = diagnostic.select_completed([timeout, good])
        self.assertEqual(selected, [good])
        self.assertEqual(excluded, [{'attempt_id': 2, 'failure_class': 'timeout'}])
        for bad in ([good, good], [{'attempt_id': 1, 'valid': False}], [dict(timeout, metrics=good['metrics'])]):
            with self.assertRaises(AssertionError):
                diagnostic.select_completed(bad)

    def test_fidelity_refuses_changed_score_and_nan(self):
        good = dict.fromkeys(diagnostic.RMS_KEYS, 2.)
        diagnostic.check_fidelity(good, good)
        for value in (2.001, float('nan')):
            with self.assertRaises(AssertionError):
                diagnostic.check_fidelity(dict(good, rms_symmetric_mm=value), good)

    def test_missing_input_and_existing_output_refused(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from pathlib import Path
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(diagnostic.prior, 'BASE', root):
                with self.assertRaises(FileNotFoundError):
                    diagnostic.prior.verify_inputs({'input_sha256': {'missing': '0' * 64}})
            path = root / 'existing.json'
            path.write_text('original')
            with self.assertRaises(FileExistsError):
                diagnostic.prior.write_new(path, {})
            self.assertEqual(path.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
