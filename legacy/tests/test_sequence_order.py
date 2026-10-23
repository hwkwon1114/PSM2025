"""Counts, cadence acceptance and material-correspondence metrics; no solver launches."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'legacy/scripts'), str(ROOT / 'legacy/analysis')]
import run_sequence_order as runner
from analyze_sequence_order import compare, load_case, quantiles


class SequenceOrderTests(unittest.TestCase):
    def test_matrix_and_recipe_preservation(self):
        original = json.loads(runner.SOURCE.read_text())
        for name, letters in runner.SCHEDULES.items():
            config = runner.sequence(name)
            self.assertEqual(config['hardening'], original['hardening'])
            self.assertEqual(config['defaults'], original['defaults'])
            self.assertEqual(len(config['toolpaths']), len(letters))
            for recipe, letter in zip(config['toolpaths'], letters):
                self.assertEqual(recipe['operation'], original['toolpaths']['AB'.index(letter)]['operation'])
                self.assertEqual(recipe['repeat'], 1)
                self.assertTrue(recipe['id'].startswith(letter + '_'))
            if name.endswith('_only'):
                self.assertEqual(len(letters), 10)
            else:
                self.assertEqual(letters.count('A'), 10)
                self.assertEqual(letters.count('B'), 10)

    def test_options_change_only_intended_factor(self):
        base = runner.options('every_cycle')
        final = runner.options('final_only')
        self.assertEqual({k for k in base if base[k] != final[k]}, {'-sequence_solve_mode'})
        self.assertEqual(base, runner.options('every_cycle', 'repeat'))
        self.assertEqual({k for k in base if base[k] != runner.options('every_cycle', 'fine')[k]}, {'-res'})
        self.assertEqual({k for k in base if base[k] != runner.options('every_cycle', 'tight')[k]},
                         {'-native_hybrid_grad_tol', '-equilibrium_grad_tol'})
        self.assertEqual(base['-sequence_adaptive'], 'false')

    def test_unaccepted_and_nonfinite_proofs(self):
        proof = dict(accepted=True, termination='gradient_target', curvature='positive_pivots',
                     gradient_norm=1e-15, gradient_tolerance=5e-14)
        self.assertTrue(runner.accepted(proof, 5e-14))
        for key, value in [('accepted', False), ('termination', 'time_limit'),
                           ('curvature', 'negative_pivots'), ('gradient_norm', float('nan')),
                           ('gradient_norm', 1e-10), ('gradient_tolerance', 1e-12)]:
            self.assertFalse(runner.accepted(dict(proof, **{key: value}), 5e-14))

    def test_alignment_targets_and_reference_rejection(self):
        reference = np.array([[0, 0, 0], [.1, 0, 0], [0, .15, 0], [.1, .15, 0]])
        positions = reference.copy(); positions[3, 2] = .013
        state = dict(reference_positions=reference.tolist(), positions=positions.tolist(),
                     material_uv=reference[:, :2].tolist(), faces=[[0, 1, 2], [1, 3, 2]],
                     edges=[], bbar=[[[0, 0], [0, 0]]]*2, material={'young': 1},
                     abar_top=[[[1, 0], [0, 1]]]*2, abar_bottom=[[[1, 0], [0, 1]]]*2,
                     counts=[10, 10], proof={'energy': 1})
        transformed = copy.deepcopy(state)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        transformed['positions'] = (positions @ rotation + [2, 3, 4]).tolist()
        result, arrays = compare(transformed, state)
        self.assertLess(result['area_weighted_rms_mm'], 1e-9)
        self.assertAlmostEqual(np.linalg.det(result['rotation']), 1)
        transformed['abar_top'][0][0][0] += .01
        result, _ = compare(transformed, state)
        self.assertFalse(result['same_final_objective_and_history'])
        self.assertIsNone(result['normalized_energy_difference'])
        transformed['reference_positions'][0][0] = .01
        with self.assertRaises(ValueError):
            compare(transformed, state)

    def test_weighted_quantiles(self):
        q = quantiles(np.array([0., 10.]), np.array([.9, .1]))
        self.assertEqual(q['q50'], 0)
        self.assertEqual(q['q90'], 0)
        self.assertAlmostEqual(q['q99'], 9)

    @unittest.skipUnless(os.environ.get('SEQUENCE_ORDER_FIXTURES'), 'Needs retained compute fixtures')
    def test_real_fixture_acceptance_and_exact_targets(self):
        root = Path(os.environ['SEQUENCE_ORDER_FIXTURES'])
        states = []
        for mode in ('every_cycle', 'final_only'):
            folder = root / ('ab_alternate_' + mode + '_base')
            state, _, _ = load_case(folder)
            states.append(state)
            execution = json.loads((folder / 'execution.json').read_text())
            config = json.loads((folder / 'sequence.json').read_text())
            original = runner.accepted
            with patch.object(runner, 'accepted', return_value=False):
                with self.assertRaises(ValueError):
                    runner.verify(folder, config, execution['options'])
            self.assertIs(runner.accepted, original)
        result, _ = compare(*states)
        self.assertTrue(result['same_final_objective_and_history'])


if __name__ == '__main__':
    unittest.main()
