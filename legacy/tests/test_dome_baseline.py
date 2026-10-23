"""Input-only checks: no solver or scientific execution."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_dome', ROOT / 'legacy/scripts/prepare_dome_baseline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BaselineTest(unittest.TestCase):
    def test_only_approved_changes(self):
        original = json.loads(m.SOURCE.read_text())
        before = copy.deepcopy(original)
        calibration = json.loads(m.CALIBRATION.read_text())
        actual = m.prepare(original, calibration)
        self.assertEqual(original, before)
        expected = copy.deepcopy(original)
        expected['hardening'] = calibration['hardening']
        expected['boundary_conditions'] = {'enabled': False}
        for i, path in enumerate(expected['toolpaths']):
            op = path['operation']
            del op['ortho']
            for key in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
                op[key] = calibration['toolpaths'][0]['operation'][key]
            if i:
                op['lv_mm'] = 196.0
                op['shift_uv_mm'] = [0.0, 118.0 if i == 1 else -118.0]
        self.assertEqual(actual, expected)
        self.assertEqual(sum(p['repeat'] for p in actual['toolpaths']), 8)

    def test_original_sides_do_not_fit(self):
        source = json.loads(m.SOURCE.read_text())
        bounds = m.footprint(source['toolpaths'][1]['operation'])
        self.assertGreater(bounds[1], 100)
        self.assertGreater(bounds[3], 150)

    def test_invalid_dimensions(self):
        op = json.loads(m.SOURCE.read_text())['toolpaths'][0]['operation']
        for key, value in [('n_strips', 2.5), ('width_mm', -1), ('lv_mm', float('nan'))]:
            with self.assertRaises(ValueError):
                m.footprint(dict(op, **{key: value}))


if __name__ == '__main__':
    unittest.main()
