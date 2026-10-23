import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
import dome_bo_domain as domain


class DomainTest(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())

    def test_roundtrip_and_preservation(self):
        values = domain.coordinates(self.base)
        decoded = domain.decode(domain.encode(values))
        for a,b in zip(values,decoded):
            self.assertAlmostEqual(a,b)
        self.assertEqual(domain.sequence(self.base,values),self.base)
        self.assertEqual(domain.decode([0]*6),[100,1,6,120,.5,4])
        self.assertEqual(domain.decode([1]*6),[196,15,26,196,5,18])

    def test_invalid_footprint_and_parameters(self):
        with self.assertRaises(ValueError):
            domain.sequence(self.base,domain.decode([1]*6))
        for point in ([0]*5, [float('nan')]*6, [-.1]*6):
            with self.assertRaises(ValueError): domain.decode(point)
        with self.assertRaises(ValueError): domain.encode([140,5,10.5,196,2,8])

    def test_warm_start_recipe_fairness(self):
        root = ROOT/'legacy/configs/dome'
        for name in ('middle_n18_repeat2','middle_n18_length168','middle_n18_sides_n14'):
            self.assertTrue(domain.compatible(self.base,json.loads((root/(name+'.json')).read_text())))
        for name in ('middle_n18_uniform','middle_repeat4','sides_repeat6'):
            self.assertFalse(domain.compatible(self.base,json.loads((root/(name+'.json')).read_text())))


if __name__ == '__main__': unittest.main()
