import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dome_curvature', ROOT / 'dome_bo/dome_curvature.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def mesh():
    x, y = np.meshgrid(np.linspace(-.1, .1, 41), np.linspace(-.15, .15, 61))
    uv = np.column_stack((x.ravel(), y.ravel()))
    faces = []
    for j in range(60):
        for i in range(40):
            a = j*41+i
            faces.extend(((a, a+1, a+42), (a, a+42, a+41)))
    return uv, np.array(faces)


class CurvatureTest(unittest.TestCase):
    def test_analytic_surfaces_and_rigid_pose(self):
        uv, faces = mesh()
        x, y = uv.T
        radius = 2.0
        surfaces = {
            'plane': (np.zeros_like(x), [0, 0]),
            'sphere': (radius-np.sqrt(radius**2-x*x-y*y), [.5, .5]),
            'cylinder': (radius-np.sqrt(radius**2-x*x), [0, .5]),
            'saddle': ((x*x-y*y)/4, [-.5, .5]),
        }
        angle = .73
        rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        for name, (z, expected) in surfaces.items():
            positions = np.column_stack((uv, z))
            values, _, valid = m.geometric_curvature(positions, uv, faces)
            moved, _, moved_valid = m.geometric_curvature(positions @ rot.T + [2, -1, 3], uv, faces)
            np.testing.assert_array_equal(valid, moved_valid)
            np.testing.assert_allclose(values[valid], moved[valid], atol=1e-8)
            np.testing.assert_allclose(np.median(values[valid], axis=0), expected, atol=.005)
            result = m.coverage(values, valid, uv, faces)
            self.assertEqual(result['qualifies'], name == 'sphere')
            self.assertGreater(result['valid_area_fraction'], .99)

    def test_invalid_and_mixed_sign_area_cannot_pass(self):
        uv, faces = mesh()
        values = np.ones((len(faces), 2))
        values[uv[faces].mean(axis=1)[:, 0] < 0] = -1
        self.assertFalse(m.coverage(values, np.ones(len(faces), bool), uv, faces)['qualifies'])
        values[:] = 1
        valid = uv[faces].mean(axis=1)[:, 0] > 0
        self.assertFalse(m.coverage(values, valid, uv, faces)['qualifies'])
        values[:] = np.nan
        self.assertFalse(m.coverage(values, np.ones(len(faces), bool), uv, faces)['qualifies'])


if __name__ == '__main__':
    unittest.main()
