"""Small correctness fixtures only; no shell simulation or parameter fitting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_shape_refinement import interpolate, load_state, same_mesh, shape_metrics


class ShapeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])

    def test_proper_rigid_transform(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        metrics = shape_metrics(self.points, self.points@rotation+[2, -3, 4], np.arange(1, 6))
        self.assertLess(metrics['sampled_max_mm'], 1e-10)
        self.assertAlmostEqual(np.linalg.det(metrics['rotation']), 1)

    def test_reflection_is_not_removed(self):
        reflection = np.diag([-1., 1, 1])
        self.assertGreater(shape_metrics(self.points, self.points@reflection, np.ones(5))['rms_mm'], 1)

    def test_boundary_max_is_not_trimmed(self):
        other = self.points.copy()
        other[-1, 2] += .001
        metrics = shape_metrics(self.points, other, [1, 1, 1, 1, 0])
        self.assertLess(metrics['rms_mm'], 1e-10)
        self.assertLess(metrics['p95_mm'], 1e-10)
        self.assertAlmostEqual(metrics['sampled_max_mm'], 1)
        self.assertEqual(metrics['sampled_max_index'], 4)

    def test_weighted_tail_not_unweighted_percentile(self):
        other = self.points.copy()
        other[-1, 2] += .001
        metrics = shape_metrics(self.points, other, [1, 1, 1, 1, .001])
        self.assertLess(metrics['p95_mm'], metrics['sampled_max_mm']*.1)
        self.assertGreater(np.quantile(metrics['distances_mm'], .95), metrics['p95_mm']*10)

    def test_bad_inputs(self):
        for weights in ([0]*5, [-1]*5, [float('nan')]*5, [1, 2]):
            with self.assertRaises(ValueError):
                shape_metrics(self.points, self.points, weights)
        with self.assertRaises(ValueError):
            shape_metrics(self.points, self.points*float('nan'), [1]*5)

    def state(self):
        xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        return dict(schema_version=1, units='metres', positions=xyz.tolist(),
                    reference_positions=xyz.tolist(), material_uv=xyz[:, :2].tolist(),
                    faces=[[0, 1, 2]], edges=[[0, 1], [1, 2], [2, 0]],
                    selected_dofs=np.r_[xyz.ravel(order='F'), [0]*3][:, None].tolist(),
                    abar_top=[np.eye(2).tolist()], abar_bottom=[np.eye(2).tolist()],
                    bbar=[np.zeros((2, 2)).tolist()], material={'young': 1., 'nu': .33, 'thickness': .00077},
                    proof=dict(accepted=True, termination='gradient_target', curvature='positive_pivots', energy=0.,
                               gradient_norm=2e-15, gradient_tolerance=5e-14,
                               verification_evaluations=1, verification_hessians=1))

    def test_roundtrip_fixed_objective_and_failed_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.json'
            original = self.state()
            path.write_text(json.dumps(original))
            state = load_state(path)
            self.assertLess(same_mesh(state, state)['rms_mm'], 1e-10)
            changed = copy.deepcopy(state)
            changed['abar_top'][0, 0, 0] += .01
            with self.assertRaises(ValueError):
                same_mesh(state, changed)
            original['proof']['accepted'] = False
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                load_state(path)
            original['proof']['accepted'] = True
            original['selected_dofs'][0][0] += .001
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                load_state(path)

    def test_actual_triangle_affine_interpolation_and_no_extrapolation(self):
        state = self.state()
        state['material_uv'] = np.asarray(state['material_uv'])
        state['faces'] = np.asarray(state['faces'])
        uv = np.array([[.1, .1], [.4, .2]])
        state['positions'] = np.c_[state['material_uv'], 2*state['material_uv'][:, 0]+state['material_uv'][:, 1]]
        np.testing.assert_allclose(interpolate(state, uv), np.c_[uv, 2*uv[:, 0]+uv[:, 1]])
        with self.assertRaises(ValueError):
            interpolate(state, np.array([[1., 1.]]))


if __name__ == '__main__':
    unittest.main()
