#!/usr/bin/env python3
"""Square-sheet dome search v2: J_amp objective, qLogEI batches, warm start, plateau stop, controller wiring.

Solver behaviour, no-retry rules and error-vs-failure classification are ``square_campaign.Campaign`` (see
test_square_pipeline.py); only what v2 changes is tested here. Real-data tests skip without the retained mesh-study cells.
"""
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ('.', 'dome_bo', 'dome_bo/tests'):
    sys.path.insert(0, str(ROOT / sub))

import sag_objective as so
import square_attempt as sa
import square_hatch as sh
import square_v2_attempt as v2a
import square_v2_campaign as v2c
from test_square_pipeline import sheet_state

warnings.filterwarnings('ignore', category=ResourceWarning)
STUDY = ROOT / 'run/dome_square/mesh_study'


class TestObjective(unittest.TestCase):

    def score(self, height):
        return so.score(sheet_state(height))

    def test_dome_is_rewarded_by_amplitude_up_to_the_cap(self):
        big = self.score(lambda x, y: .55 * (x * x + y * y))            # sag 2.9 mm: capped
        small = self.score(lambda x, y: .2 * (x * x + y * y))           # sag 1.06 mm: scaled by sag / 1.5
        self.assertAlmostEqual(big['j_amp'], big['j_fit'], places=9)
        self.assertAlmostEqual(small['j_amp'], small['j_fit'] * small['sag_mm'] / so.A_REF_MM, places=9)
        self.assertGreater(big['j_amp'], small['j_amp'])
        self.assertLess(small['fit_rms_mm'], 1e-6)

    def test_normal_reversal_leaves_the_objective_unchanged(self):
        up = self.score(lambda x, y: .55 * (x * x + y * y))
        down = self.score(lambda x, y: -.55 * (x * x + y * y))
        self.assertAlmostEqual(up['j_amp'], down['j_amp'], places=9)

    def test_amplitude_alone_does_not_pay_cylinder_saddle_or_flat(self):
        cylinder = self.score(lambda x, y: 1.1 * x * x)                 # sag 2.9 mm, one curvature only
        self.assertGreater(cylinder['sag_mm'], 2.0)
        for metrics in (cylinder, self.score(lambda x, y: .55 * (x * x - y * y)), self.score(lambda x, y: 0 * x)):
            self.assertEqual(metrics['j_amp'], 0.0)

    def test_runner_analysis_keeps_the_v1_finiteness_keys(self):
        metrics = v2a.analyze_state(sheet_state(lambda x, y: .55 * (x * x + y * y)))
        self.assertEqual(metrics['j_combined'], metrics['j_fit'])
        for key in ('j_amp', 'sag_mm', 'fit_rms_mm', 'largest_connected_fraction', 'qualified'):
            self.assertIn(key, metrics)


class TestProposal(unittest.TestCase):

    def data(self, n=12, pool=300, seed=2):
        rng = np.random.default_rng(seed)
        draw = lambda m: np.array([sh.realized_raw(sh.decode(u)) for u in (rng.random(sh.N_DIM) for _ in range(m))])
        X = draw(n)
        return X, rng.random(n), draw(pool)

    def test_batch_is_k_distinct_finite_and_reproducible(self):
        X, y, cand = self.data()
        picked, values = v2c.propose(X, y, cand, 4, seed=7)
        self.assertEqual(len(set(picked)), 4)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual((picked, values), v2c.propose(X, y, cand, 4, seed=7))


def fake_runner(path: Path, fail=()):
    path.write_text(f'''
import json, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'dome_bo')!r})
import square_attempt as sa
d = Path(sys.argv[sys.argv.index("--attempt-dir") + 1])
req = json.loads((d / "request.json").read_text())
n = req["attempt"]
(d / "solve").mkdir()
(d / "solve/sequence.json").write_text(json.dumps(req["recipe"]))
st = {{"recipe_sha256": req["recipe_sha256"], "binary_sha256": sa.sha(sa.BINARY), "elapsed_seconds": 1.0}}
if n in {list(fail)!r}:
    st.update(state="failed", failure_type="EquilibriumNotReached")
else:
    st.update(state="accepted", metrics={{"j_amp": 0.01 * n, "j_combined": 0.02 * n, "largest_connected_fraction": 0.1,
                                          "qualified": False}})
(d / "status.json").write_text(json.dumps(st))
''')
    return path


def synthetic_warm_start(binary_sha):
    rng = np.random.default_rng(11)
    out = []
    for a in range(6):
        u = rng.random(sh.N_DIM)
        out.append({'attempt': a, 'u': list(map(float, u)), 'realized': list(map(float, sh.realized_raw(sh.decode(u)))),
                    'j_amp': 0.05 * (a + 1)})
    return out


class TestController(unittest.TestCase):

    def setUp(self):
        self._warm = v2c.warm_start_records
        v2c.warm_start_records = synthetic_warm_start

    def tearDown(self):
        v2c.warm_start_records = self._warm

    def campaign(self, tmp, fail=(), **kw):
        return v2c.Campaign(Path(tmp) / 'run', launcher='', runner=fake_runner(Path(tmp) / 'fake.py', fail), **kw)

    def test_design_then_qlogei_waves_score_j_amp_and_skip_warm_start_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, fail=(5,), max_attempts=22)
            seen = []
            real = v2c.propose
            v2c.propose = lambda X, y, *a: (seen.append((len(X), len(y))), real(X, y, *a))[1]
            try:
                self.assertIn('attempts reached', camp.run())
            finally:
                v2c.propose = real
            ledger = camp.ledger()
            self.assertEqual(seen, [(6 + 17, 6 + 17)])                  # six warm start + 17 accepted; failed attempt 5 not fitted
            self.assertEqual([r['source'] for r in ledger], ['anchor'] * 2 + ['design'] * 16 + ['bo'] * 4)
            self.assertEqual([r['attempt'] for r in ledger], list(range(1, 23)))
            for r in ledger:
                if r['state'] == 'accepted':
                    self.assertEqual(r['y'], r['metrics']['j_amp'])
                else:
                    self.assertIsNone(r['y'])
            keys = {sh.decode_key(np.array(r['u'])) for r in ledger}
            self.assertEqual(len(keys), 22)
            self.assertFalse(keys & {sh.decode_key(np.array(w['u'])) for w in camp.warm_start()})

    def test_warm_start_is_part_of_the_frozen_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, max_attempts=4)
            camp.run()
            warm = camp.warm_start()
            warm[0]['j_amp'] += 1
            camp.warm_path.write_text(json.dumps(warm))
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                camp.freeze_identity()

    def test_nonfinite_j_amp_is_an_error_not_a_training_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, max_attempts=4)
            camp.frozen_binary = sa.sha(sa.BINARY)
            d = camp.reserve(1, 0, sc_anchor(), 'anchor', None)
            runner = fake_runner(Path(tmp) / 'f.py').read_text().replace('"j_amp": 0.01 * n', '"j_amp": float("nan")')
            (Path(tmp) / 'f.py').write_text(runner)
            import subprocess
            subprocess.run([sys.executable, str(Path(tmp) / 'f.py'), '--attempt-dir', str(d)], check=True)
            record = camp.collect(d, 0)
            self.assertEqual((record['state'], record['failure_type']), ('error', 'NonfiniteObjective'))

    def test_plateau_counts_only_bo_waves_and_includes_the_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp)
            camp.run = None                                              # controller state only
            camp.root.mkdir(parents=True)
            camp.warm_path.write_text(json.dumps([{'j_amp': 1.0}]))

            def rec(wave, y, source='bo'):
                return {'wave': wave, 'source': source, 'y': y}

            flat = [rec(0, 5.0, 'design')] + [rec(w, 0.5) for w in range(1, 5)]
            self.assertFalse(camp.plateaued(flat[:4]))                   # three BO waves: window not complete
            self.assertTrue(camp.plateaued([rec(0, 1.0, 'design')] + [rec(w, 1.04) for w in range(1, 5)]))
            self.assertFalse(camp.plateaued([rec(0, 1.0, 'design')] + [rec(w, 1.0) for w in range(1, 4)] + [rec(4, 1.06)]))
            self.assertTrue(camp.plateaued([rec(0, 1.0, 'design')] + [rec(w, None) for w in range(1, 5)]))


def sc_anchor():
    return v2c.sc.anchors()[0]


@unittest.skipUnless((STUDY / 'sag_objective.json').exists(), 'retained mesh-study cells not present')
class TestRetainedCells(unittest.TestCase):

    def test_warm_start_reproduces_the_recorded_2mm_scores_and_is_mesh_stable_against_1mm(self):
        records = v2c.warm_start_records(sa.sha(sa.BINARY))
        recorded = json.loads((STUDY / 'sag_objective.json').read_text())
        self.assertEqual([r['attempt'] for r in records], v2c.mc.CAND)
        for r in records:
            a = r['attempt']
            self.assertAlmostEqual(r['j_amp'], recorded[f'{a}_2']['jamp_1.5'], places=9)
            self.assertLess(abs(r['j_amp'] - recorded[f'{a}_1']['jamp_1.5']), 0.01)
            self.assertLess(abs(r['sag_mm'] - recorded[f'{a}_1']['sag_mm']), 0.1)
            self.assertEqual(r['realized'], list(sh.realized_raw(sh.decode(np.array(r['u'])))))


@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'), 'needs a compute allocation')
class TestRealSolverFixture(unittest.TestCase):

    def test_v2_runner_scores_j_amp_on_the_adaptive_cutback_route(self):
        recipe = sh.decode(sh.unit_from_physical(8, 8, 192, 192, 192, 192, 3, 3, 1.0))
        for tp in recipe['toolpaths']:
            tp['operation']['lv_mm'] *= .3
            tp['operation']['width_mm'] *= .3
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'attempt'
            d.mkdir()
            (d / 'request.json').write_text(json.dumps({'attempt': 1, 'recipe': recipe, 'recipe_sha256': sh.recipe_sha(recipe)}))
            status = v2a.run_attempt(d, mesh=('.1', '.1', '.05'), threads=1, fixture=True)
            self.assertEqual(status['state'], 'accepted', status.get('error'))
            self.assertTrue(np.isfinite(status['metrics']['j_amp']))
            self.assertEqual(status['tolerance'], v2a.TOLERANCE)
            self.assertIn('-native_hybrid_on_newton_stall', status['command'])


if __name__ == '__main__':
    unittest.main()
