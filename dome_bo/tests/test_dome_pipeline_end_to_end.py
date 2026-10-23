#!/usr/bin/env python3
"""End-to-end dome pipeline on a coarse fixture (pipeline correctness only, not dome-quality evidence).

GP proposes a candidate -> generator emits a recipe -> the real simulator solves THAT recipe ->
curvature, smoothed objective and connectivity gate are evaluated on THAT solved state -> the
surrogate is updated with the observation. Provenance (recipe hash, hit accounting, parameters) is
checked to tie every stage to the same candidate.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ('.', 'dome_bo'):
    sys.path.insert(0, str(ROOT / sub))

from crosshatch_generator import PitchGradedCrossHatch, generator_from_raw, raw_from_recipe
from dome_continuous_objective import evaluate_state
from dome_gp_surrogate import DomeGPSurrogate, DEFAULT_BOUNDS

SHELL = ROOT / 'bin/shell'
FIXTURE_OPTIONS = {
    '-sim': 'bilayer_growth', '-case': 'custom', '-geometry': 'rectangle', '-lx': '.1', '-ly': '.15',
    '-res': '.04', '-h_total': '.003', '-growth_type': 'zigzag_sequence',
    '-sequence_solve_mode': 'final_only', '-metric_update': 'multiplicative', '-enable_passE': 'false',
    '-nsteps': '1', '-equilibrium_solver': 'native_hybrid', '-native_hybrid_grad_tol': '5e-14',
    '-equilibrium_grad_tol': '5e-14', '-native_hybrid_warmup_attempts': '200',
    '-native_hybrid_seconds': '120', '-native_hybrid_max_hessians': '40',
    '-native_hybrid_on_newton_stall': 'cutback', '-hessian_threads': '1', '-sequence_adaptive': 'true',
    '-sequence_initial_step': '1.0', '-sequence_max_step': '1.0', '-sequence_min_step': '.015625',
    '-sequence_max_retries': '8', '-native_hybrid_export_state': 'true', '-cycle_write_state': 'false',
    '-export_stl': 'false',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def solve_recipe(recipe: dict, out: Path, **overrides) -> dict:
    """Runs the real simulator on `recipe`; returns the solved state and the run's own result record."""
    cycle_file = out / 'sequence.json'
    cycle_file.write_text(json.dumps(recipe))
    options = dict(FIXTURE_OPTIONS, **overrides)
    cmd = [str(SHELL)] + [x for kv in options.items() for x in kv] + ['-cycle_file', str(cycle_file)]
    proc = subprocess.run(cmd, cwd=out, env=dict(os.environ, OMP_NUM_THREADS='1'),
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-600:] + proc.stderr[-600:])
    return {'state': json.loads((out / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text()),
            'result': json.loads((out / 'sequence_result.json').read_text()),
            'recipe_sha256': sha256(cycle_file.read_bytes())}


@unittest.skipUnless(SHELL.exists(), 'bin/shell not built')
class TestDomePipelineEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore')

    def test_candidate_to_solved_state_to_surrogate_update(self):
        # Seeds are real recipes: the surrogate only ever sees coordinates MEASURED on emitted recipes.
        requests = np.array([[1.5, 0.5, 1.5, 1.0, 8.0, 12.0], [3.0, 0.4, 2.0, 1.25, 8.0, 15.0],
                             [4.0, 0.6, 3.0, 1.5, 9.0, 20.0], [2.0, 0.3, 2.5, 1.1, 14.0, 10.0]])
        seeds = np.array([generator_from_raw(r).realized_raw() for r in requests])
        y_seed = np.array([-0.85, -0.62, -0.52, -0.70])      # synthetic seed objectives (fixture only)
        z_seed = np.array([0, 1, 1, 0])
        surrogate = DomeGPSurrogate()
        surrogate.fit(seeds, y_seed, z_seed)
        # Acquisition is scored at the coordinates the candidate's emitted recipe would realize.
        realize = lambda r: generator_from_raw(r).realized_raw()
        proposal = surrogate.propose(bounds=DEFAULT_BOUNDS, n_candidates=48, n_local=8,
                                     realize=realize, rng=np.random.default_rng(7))
        candidate = proposal['request']
        self.assertTrue(np.allclose(proposal['realized'], realize(candidate)))
        for x, (lo, hi) in zip(candidate, DEFAULT_BOUNDS):
            self.assertTrue(lo - 1e-9 <= x <= hi + 1e-9)

        generator = generator_from_raw(candidate)
        recipe = generator.generate_recipe()
        realized = generator.realized_raw(recipe)
        self.assertTrue(np.allclose(realized, raw_from_recipe(recipe, generator.decay_power)))
        expected_cycles = sum(tp['repeat'] for tp in recipe['toolpaths'])
        # Default (terrace) layout: one simulator cycle per sweep of each nested stroke.
        self.assertEqual(generator.layout, 'terrace')
        self.assertGreater(len(recipe['toolpaths']), 2)
        self.assertGreater(expected_cycles, 0)

        with tempfile.TemporaryDirectory() as tmp:
            run = solve_recipe(recipe, Path(tmp))

        # Provenance: the solver consumed exactly this candidate's recipe and applied every hit.
        self.assertEqual(run['recipe_sha256'], sha256(json.dumps(recipe).encode()))
        self.assertTrue(run['result']['completed'])
        self.assertEqual(run['result']['cycles'], expected_cycles)
        self.assertTrue(run['result']['solves'][0]['native_hybrid']['accepted'])
        self.assertEqual(run['state']['proof'], run['result']['solves'][0]['native_hybrid'])

        # Objective and gate on the solved state of THIS recipe.
        obj = evaluate_state(run['state'])
        for key in ('j_quantile', 'j_combined', 'saddle_flux_fraction', 'largest_connected_fraction'):
            self.assertTrue(np.isfinite(obj[key]), key)
        self.assertEqual(obj['qualified'], obj['largest_connected_fraction'] >= 0.8)
        self.assertAlmostEqual(obj['valid_area_fraction'], 1.0, places=6)

        self.assertTrue(np.allclose(realized, proposal['realized']))     # the scored point is the solved recipe's

        # Surrogate update with the observed (not prescreened) outcome.
        z_new = int(obj['qualified'])
        surrogate.fit(np.vstack([seeds, realized]), np.append(y_seed, obj['j_combined']), np.append(z_seed, z_new))
        self.assertTrue(np.allclose(surrogate.X_raw[-1], realized))      # realized, not requested, coordinates
        self.assertEqual(surrogate.summary()['n_samples'], 5)
        pred = surrogate.predict(realized.reshape(1, -1))
        self.assertTrue(np.isfinite(pred['mean'][0]) and pred['std'][0] > 0)
        self.assertTrue(0.0 <= pred['p_feasible'][0] <= 1.0)

    def test_failed_solve_updates_the_feasibility_model(self):
        """A genuine solver failure (the real simulator exits non-zero) is recorded as an unscored
        infeasible observation and the surrogate refits on it."""
        requests = np.array([[1.5, 0.5, 1.5, 1.0, 8.0, 12.0], [3.0, 0.4, 2.0, 1.25, 8.0, 15.0],
                             [4.0, 0.6, 3.0, 1.5, 9.0, 20.0]])
        seeds = np.array([generator_from_raw(r).realized_raw() for r in requests])
        failing = generator_from_raw([2.0, 0.5, 2.0, 1.2, 14.0, 14.0])
        recipe = failing.generate_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                solve_recipe(recipe, Path(tmp), **{'-native_hybrid_seconds': '0.001', '-sequence_max_retries': '1'})
        X = np.vstack([seeds, failing.realized_raw(recipe)])
        y = np.array([-0.85, -0.62, -0.52, np.nan])                     # the failed solve is unscored
        z = np.array([0, 1, 1, 0])
        surrogate = DomeGPSurrogate()
        surrogate.fit(X, y, z)
        self.assertEqual((surrogate.summary()['n_samples'], surrogate.summary()['n_scored']), (4, 3))
        pred = surrogate.predict(X[-1].reshape(1, -1))
        self.assertTrue(0.0 <= pred['p_feasible'][0] <= 1.0)
        realize = lambda r: generator_from_raw(r).realized_raw()
        nxt = surrogate.propose(n_candidates=24, n_local=6, realize=realize, rng=np.random.default_rng(9))
        self.assertTrue(np.isfinite(nxt['cei']))

    def test_two_different_candidates_give_different_solved_states(self):
        """The objective responds to the design: different recipes must not collapse to one state."""
        a = PitchGradedCrossHatch(total_dose=1.0)
        b = PitchGradedCrossHatch(total_dose=4.5)
        results = []
        for g in (a, b):
            with tempfile.TemporaryDirectory() as tmp:
                run = solve_recipe(g.generate_recipe(), Path(tmp))
            results.append(evaluate_state(run['state']))
        self.assertNotEqual(results[0]['j_combined'], results[1]['j_combined'])
        self.assertNotEqual(results[0]['mean_K_pos'], results[1]['mean_K_pos'])


if __name__ == '__main__':
    unittest.main()
