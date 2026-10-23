"""Exercise cadence analysis on saved coarse fixtures; never runs a solver."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/analysis'))
from analyze_never_clamped import compare
from analyze_four_conditions import add_final_only


class CadenceAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / 'run/two_gripper_comparison/fixtures/final_only_v1'
        baseline = ROOT / 'run/two_gripper_comparison/fixtures/never_clamped_v2/paired_regression'
        cls.metrics, cls.arrays = compare(cls.root / 'per_cycle', baseline, 'fixture')

    def test_full_contrast(self):
        metrics, arrays = copy.deepcopy(self.metrics), copy.deepcopy(self.arrays)
        add_final_only(metrics, arrays, self.root / 'final', self.root / 'per_cycle')
        self.assertTrue(all(metrics['cadence']['reference_and_target_history_exact_matches'].values()))
        self.assertGreaterEqual(metrics['cadence']['area_weighted_rms_difference_mm'], 0)
        self.assertEqual(arrays['cadence_distance_mm'].size, metrics['vertices'])
        self.assertIn('paired_release', metrics)

    def test_rejects_wrong_history_and_cadence(self):
        original = json.loads
        for defect in ('history', 'cadence'):
            def corrupted(text):
                obj = original(text)
                payload = obj.get('payload', {})
                if defect == 'history' and payload.get('identity', {}).get('recipe', {}).get('equilibrium_cadence') == 'final_only':
                    payload['counts'] = []
                if defect == 'cadence' and payload.get('solve_mode') == 'final_only':
                    payload['equilibrium_solves'] = 2
                return obj
            with self.subTest(defect=defect), patch('analyze_four_conditions.json.loads', side_effect=corrupted):
                with self.assertRaises(ValueError):
                    add_final_only(copy.deepcopy(self.metrics), copy.deepcopy(self.arrays),
                                   self.root / 'final', self.root / 'per_cycle')


if __name__ == '__main__':
    unittest.main()
