import unittest
import warnings
from unittest.mock import patch

from botorch.exceptions.errors import CandidateGenerationError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import _process_scipy_result
from botorch.logging import logger
from scipy.optimize import OptimizeResult

import numpy as np
import torch

from calibration.analysis.success_aware_bo.separation import (
    SEARCH_DISTANCE, admissible, constraints, feasible_initializations,
    history_tensor, optimize_separated)


class Quadratic(torch.nn.Module):
    def forward(self, X):
        return -(X-.5).square().sum(dim=(-1, -2))


class SeparationTests(unittest.TestCase):
    def test_gradients_and_boundaries(self):
        history = history_tensor([[.5]*4])
        historical, pairwise = [f for f, _ in constraints(history)]
        x = torch.tensor([.5+SEARCH_DISTANCE, .5, .5, .5], dtype=torch.double,
                         requires_grad=True)
        torch.testing.assert_close(historical(x), torch.zeros((), dtype=torch.double), atol=1e-16, rtol=0)
        historical(x).backward()
        torch.testing.assert_close(x.grad, torch.tensor([2*SEARCH_DISTANCE, 0., 0., 0.], dtype=torch.double))
        X = torch.tensor([[.1]*4, [.3]*4, [.6]*4, [.9]*4], dtype=torch.double, requires_grad=True)
        pairwise(X).backward()
        self.assertTrue(torch.isfinite(X.grad).all())
        X = np.array([[0.]*4, [.08, 0., 0., 0.], [.5]*4, [1.]*4])
        self.assertTrue(admissible(X, []))
        X[1, 0] = .079999
        self.assertFalse(admissible(X, []))
        self.assertFalse(admissible(np.zeros((3, 4)), []))

    def test_feasible_seeded_pool_and_history(self):
        history = history_tensor([[.5]*4, [.1]*4])
        a, drawn = feasible_initializations(history, 32, 3101)
        b, twin_drawn = feasible_initializations(history, 32, 3101)
        self.assertEqual(drawn, twin_drawn)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for X in a:
            self.assertTrue(admissible(X.numpy(), history.numpy()))
        self.assertFalse(admissible(a[0].numpy(), [a[0, 0].numpy()]))

    def test_toy_joint_constrained_optimization(self):
        result = optimize_separated(Quadratic(), [[.5]*4], seed=3101,
                                    restarts=2, initializations=32, timeout=20)
        self.assertTrue(admissible(result['final'], [[.5]*4]))
        self.assertEqual(result['repaired_indices'], [])
        self.assertGreaterEqual(result['optimizer_log_acquisition'], result['best_initial_log_acquisition'])
        self.assertEqual(result['selection'], 'constrained_optimizer')
        # Unconstrained optimum is forbidden; a near-shell joint batch is possible.
        self.assertGreater(result['optimizer_log_acquisition'], -.05)

    def test_invalid_optimizer_preserves_scored_initialization(self):
        with patch('calibration.analysis.success_aware_bo.separation.optimize_acqf',
                   return_value=(torch.zeros(2, 4, 4, dtype=torch.double), torch.zeros(2))):
            result = optimize_separated(Quadratic(), [], seed=1, restarts=2, initializations=16)
        self.assertTrue(admissible(result['final'], []))
        self.assertEqual(result['selection'], 'scored_initialization')
        self.assertEqual(result['inadmissible_optimizer_restarts'], [0, 1])
        self.assertEqual(result['optimizer_log_acquisition'], result['best_initial_log_acquisition'])

    def test_real_warning_containment_and_termination_reporting(self):
        level, handlers = logger.level, list(logger.handlers)

        def generate(initial, acquisition, **kwargs):
            warnings.warn('fixture optimization failure', OptimizationWarning)
            _process_scipy_result(OptimizeResult(success=False, status=9,
                                  message='Iteration limit reached'), kwargs['options'])
            _process_scipy_result(OptimizeResult(success=False, status=1,
                                  message='Optimization timed out after 0.01 seconds.'), kwargs['options'])
            return initial.clone(), acquisition(initial)

        with patch('calibration.analysis.success_aware_bo.separation.gen_candidates_scipy',
                   side_effect=generate) as generator, warnings.catch_warnings(record=True) as outer:
            warnings.simplefilter('always')
            result = optimize_separated(Quadratic(), [], seed=1, restarts=2, initializations=16)
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(len(result['generator_diagnostics']), 2)
        self.assertEqual(len(result['warnings']), 2)
        self.assertTrue(all('fixture optimization failure' in w for w in result['warnings']))
        self.assertFalse(outer)  # BoTorch/selector containment must not erase the saved evidence.
        for diagnostic in result['generator_diagnostics']:
            self.assertEqual(diagnostic['warnings'][0]['category'], 'OptimizationWarning')
            self.assertTrue(any('iteration limit' in m for m in diagnostic['info_messages']))
            self.assertTrue(any('timed out' in m for m in diagnostic['info_messages']))
        self.assertEqual(logger.level, level)
        self.assertEqual(logger.handlers, handlers)
        self.assertEqual(result['convergence_status'],
                         'not_inferred_from_selection_or_absence_of_warnings')

    def test_real_all_infeasible_aggregate_uses_scored_initialization(self):
        def generate(initial, acquisition, **kwargs):
            invalid = torch.zeros_like(initial)
            warnings.warn('fixture infeasible restart', OptimizationWarning)
            return invalid, acquisition(invalid)

        with patch('calibration.analysis.success_aware_bo.separation.gen_candidates_scipy',
                   side_effect=generate) as generator:
            result = optimize_separated(Quadratic(), [], seed=1, restarts=2, initializations=16)
        self.assertEqual(generator.call_count, 2)
        self.assertTrue(admissible(result['final'], []))
        self.assertEqual(result['selection'], 'scored_initialization')
        self.assertEqual(result['inadmissible_optimizer_restarts'], [0, 1])
        self.assertIn('2 out of 2', result['optimizer_feasibility_error'])
        self.assertEqual(result['optimizer_log_acquisition'], result['best_initial_log_acquisition'])
        self.assertEqual(len(result['warnings']), 2)

    def test_generator_errors_propagate_and_logging_is_restored(self):
        level, handlers = logger.level, list(logger.handlers)
        errors = [RuntimeError('programming fixture'), CandidateGenerationError('other failure'),
                  CandidateGenerationError('The optimizer produced infeasible candidates. 2 out of '
                      '2 batches of candidates were infeasible. Please make sure '
                      'the constraints are satisfiable and relax them if needed. ')]
        for error in errors:
            with self.subTest(error=str(error)), patch(
                    'calibration.analysis.success_aware_bo.separation.gen_candidates_scipy',
                    side_effect=error), self.assertRaises(type(error)):
                optimize_separated(Quadratic(), [], seed=1, restarts=2, initializations=16)
            self.assertEqual(logger.level, level)
            self.assertEqual(logger.handlers, handlers)

    def test_nonfinite_generator_results_are_not_feasibility_fallback(self):
        for field in ('candidate', 'value'):
            def generate(initial, acquisition, **kwargs):
                candidate, value = initial.clone(), acquisition(initial)
                (candidate if field == 'candidate' else value).fill_(float('nan'))
                return candidate, value
            with self.subTest(field=field), patch(
                    'calibration.analysis.success_aware_bo.separation.gen_candidates_scipy',
                    side_effect=generate), self.assertRaisesRegex(ValueError, 'Nonfinite optimizer'):
                optimize_separated(Quadratic(), [], seed=1, restarts=2, initializations=16)

    def test_worse_optimizer_cannot_displace_better_initialization(self):
        corners = torch.tensor([[[0.]*4, [1.]*4, [1., 0., 1., 0.], [0., 1., 0., 1.]]], dtype=torch.double)
        with patch('calibration.analysis.success_aware_bo.separation.optimize_acqf',
                   return_value=(corners, torch.zeros(1))):
            result = optimize_separated(Quadratic(), [], seed=1, restarts=1, initializations=16)
        self.assertEqual(result['selection'], 'scored_initialization')

    def test_bad_inputs_and_nonfinite_scores_fail(self):
        for X in ([[float('nan')]*4], [[1.1]*4], [[.1]*3], [[], []]):
            with self.assertRaises(ValueError):
                history_tensor(X)
        with self.assertRaises(ValueError):
            optimize_separated(Quadratic(), [], seed=1, restarts=17, initializations=16)
        class Bad(torch.nn.Module):
            def forward(self, X):
                return torch.full((len(X),), float('nan'), dtype=torch.double)
        with self.assertRaises(ValueError):
            optimize_separated(Bad(), [], seed=1, restarts=1, initializations=16)

    def test_initialization_exhaustion_is_not_random_fallback(self):
        dense = torch.zeros((1, 4), dtype=torch.double)
        with patch('calibration.analysis.success_aware_bo.separation.SEARCH_DISTANCE', 3.):
            with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                feasible_initializations(dense, 4, 1)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
