"""Read-only checks of the bounded final-only integration fixtures."""
import csv
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'run/two_gripper_comparison/fixtures/final_only_v1'


def load(case, name):
    return json.loads((FIX / case / name).read_text())['payload']


class FinalOnlyTests(unittest.TestCase):
    def test_matched_targets_and_single_solve(self):
        final = load('final', 'never_clamped_final_checkpoint.json')
        per_cycle = load('per_cycle', 'never_clamped_final_checkpoint.json')
        for key in ('top', 'bottom', 'counts', 'cycle'):
            self.assertEqual(final[key], per_cycle[key], key)
        for key in ('rest', 'faces', 'edges', 'uv', 'bbar', 'vertex_mask', 'edge_mask'):
            self.assertEqual(final['identity'][key], per_cycle['identity'][key], key)
        self.assertEqual(final['cycle'], 2)
        self.assertFalse(any(any(row) for row in final['identity']['vertex_mask']))
        self.assertFalse(any(any(row) for row in final['identity']['edge_mask']))
        self.assertEqual(sorted(p.name for p in (FIX / 'final').glob('accepted_cycle_*.json')),
                         ['accepted_cycle_002.json'])
        result = load('final', 'never_clamped_result.json')
        self.assertEqual(result['equilibrium_solves'], 1)
        self.assertEqual(result['remaining_rigid_modes'], 6)
        self.assertEqual(result['fixed_vertex_dofs'], 0)
        self.assertEqual(result['proof'], final['proof'])
        self.assertTrue(final['proof']['accepted'])
        self.assertEqual(final['proof']['termination'], 'gradient_target')
        self.assertEqual(final['proof']['curvature'], 'positive_pivots')
        self.assertLessEqual(final['proof']['gradient_norm'], 5e-14)
        with (FIX / 'final/sequence_convergence.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['minimization_performed'] for r in rows], ['1'])
        self.assertEqual(float(rows[-1]['lambda_trial']), 1)
        with (FIX / 'final/sequence_history.csv').open() as f:
            self.assertEqual([r['solved'] for r in csv.DictReader(f)], ['0', '1'])
        a, b = [json.loads((FIX / c / 'execution.json').read_text())['identity']
                for c in ('final', 'per_cycle')]
        self.assertEqual(a['options'].pop('-sequence_solve_mode'), 'final_only')
        self.assertEqual(b['options'].pop('-sequence_solve_mode'), 'every_cycle')
        self.assertEqual(a['options'].pop('-sequence_adaptive'), 'false')
        self.assertEqual(b['options'].pop('-sequence_adaptive'), 'true')
        self.assertEqual(a, b)
        initial = 'bilayer_zigzag_sequence_cycle_000_initial.vtp'
        self.assertEqual((FIX / 'final' / initial).read_bytes(), (FIX / 'per_cycle' / initial).read_bytes())

    def test_terminal_recovery_and_cross_cadence_rejection(self):
        self.assertEqual(load('final', 'never_clamped_final_checkpoint.json'),
                         load('terminal', 'never_clamped_final_checkpoint.json'))
        self.assertEqual(load('terminal', 'never_clamped_result.json')['equilibrium_solves'], 0)
        self.assertFalse(list((FIX / 'terminal').glob('accepted_cycle_*.json')))
        self.assertNotIn('[native_hybrid] phase=', (FIX / 'terminal/solver.log').read_text())
        self.assertEqual((FIX / 'final/never_clamped_final.vtp').read_bytes(),
                         (FIX / 'terminal/never_clamped_final.vtp').read_bytes())
        self.assertIn('Resume build/recipe identity mismatch', (FIX / 'cross_cadence.log').read_text())
        self.assertFalse((FIX / 'invalid_resume').exists())


if __name__ == '__main__':
    unittest.main()
