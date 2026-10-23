#!/usr/bin/env python3
"""Square-sheet BO pipeline: generator, ROI gate/objective, failure learning, batch selection, controller.

Pipeline correctness only; none of these tests requires an untuned candidate to pass the dome gate.
The real-solver test needs a compute allocation (it runs bin of the reviewed repaired route on a
coarse fixture) and is skipped without SLURM_JOB_ID.
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
for sub in ('.', 'dome_bo'):
    sys.path.insert(0, str(ROOT / sub))

import square_hatch as sh
import square_attempt as sa
import square_campaign as sc
from dome_continuous_objective import evaluate_dome_objective

warnings.filterwarnings('ignore')
REF = sh.unit_from_physical(8, 8, 192, 192, 192, 192, 3, 3, 1.0)


def sheet_state(height, n=41, half=0.1):
    """Analytic square-sheet state (metres): z = height(x, y) on a structured mesh."""
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g, indexing='ij')
    uv = np.column_stack((X.ravel(), Y.ravel()))
    idx = lambda i, j: i * n + j
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    pos = np.column_stack((uv, height(uv[:, 0], uv[:, 1])))
    return {'positions': pos.tolist(), 'material_uv': uv.tolist(), 'faces': np.array(faces).tolist()}


class TestGenerator(unittest.TestCase):

    def test_random_unit_points_decode_inside_the_sheet_or_raise(self):
        rng = np.random.default_rng(1)
        for _ in range(400):
            recipe = sh.decode(rng.random(sh.N_DIM))
            for tp in recipe['toolpaths']:
                box = sh.operation_footprint(tp['operation'])
                self.assertLessEqual(max(map(abs, box)), sh.SHEET_HALF_MM + 1e-9)
                self.assertGreaterEqual(tp['operation']['n_strips'], 2)
            self.assertLessEqual(sum(tp['repeat'] for tp in recipe['toolpaths']), 12)
            self.assertEqual(recipe['boundary_conditions'], {'enabled': False})
            self.assertNotIn('metadata', recipe)

    def test_realized_coordinates_are_measured_on_the_emitted_recipe(self):
        recipe = sh.decode(REF)
        raw = sh.realized_raw(recipe)
        np.testing.assert_allclose(raw, [8, 8, 192, 192, 192, 192, 3, 3, 1.0], rtol=1e-7)
        u = REF.copy()
        u[2] = 0.5                                              # request span 146 mm, realised by whole strips
        r2 = sh.realized_raw(sh.decode(u))
        self.assertLessEqual(abs(r2[2] - 146.0), 4.0)           # within half a pitch of the request
        self.assertNotEqual(r2[2], 146.0)                       # realised != requested
        # distinct unit points may decode to the identical recipe: identical key, identical hash
        v = REF.copy()
        v[6] += 0.01
        self.assertEqual(sh.decode_key(REF), sh.decode_key(v))
        self.assertEqual(sh.recipe_sha(sh.decode(REF)), sh.recipe_sha(sh.decode(v)))

    def test_actual_treatment_changes_with_every_search_coordinate(self):
        base = sh.treatment(sh.decode(REF), 61)
        for i in range(sh.N_DIM):
            u = REF.copy()
            u[i] = 0.7 if i < 6 else 0.0            # differs from REF in every coordinate
            other = sh.treatment(sh.decode(u), 61)
            change = max(np.abs(other['top_trace'] - base['top_trace']).max(), np.abs(other['hits'] - base['hits']).max())
            self.assertGreater(change, 1e-9, sh.COORD_NAMES[i])

    def test_face_shared_hardening_is_order_dependent_and_repeat_saturating(self):
        t1 = sh.treatment(sh.decode(sh.unit_from_physical(8, 8, 192, 192, 192, 192, 1, 1, 1.0)), 21)
        t6 = sh.treatment(sh.decode(sh.unit_from_physical(8, 8, 192, 192, 192, 192, 6, 6, 1.0)), 21)
        c = 10
        self.assertGreater(t6['top_trace'][c, c], t1['top_trace'][c, c])
        self.assertLess(t6['top_trace'][c, c] / t1['top_trace'][c, c], 6.0)    # hardening saturates

    def test_anchors_and_design_are_unique_decodable_points(self):
        pts = sc.anchors() + sc.initial_design(n_design=6, starts=4, n_pool_log2=7)
        keys = [sh.decode_key(u) for u in pts]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertAlmostEqual(sh.realized_raw(sh.decode(sc.anchors()[1]))[8], 0.4)


class TestSquareRoiGate(unittest.TestCase):
    R = 1.5

    def dome(self, scale=1.0, sign=1.0):
        return sheet_state(lambda x, y: sign * scale * (self.R - np.sqrt(self.R ** 2 - x ** 2 - y ** 2)))

    def test_uniform_dome_passes_and_normal_reversal_is_invariant(self):
        a = sa.analyze_state(self.dome())
        b = sa.analyze_state(self.dome(sign=-1.0))
        self.assertTrue(a['qualified'])
        self.assertGreater(a['largest_connected_fraction'], 0.95)
        self.assertAlmostEqual(a['j_combined'], b['j_combined'], places=9)
        self.assertEqual(a['roi_half_m'], 0.08)

    def test_cylinder_saddle_and_central_bump_fail_the_gate(self):
        cyl = sa.analyze_state(sheet_state(lambda x, y: (self.R - np.sqrt(self.R ** 2 - x ** 2))))
        sad = sa.analyze_state(sheet_state(lambda x, y: 0.5 * (x ** 2 - y ** 2) / self.R))
        bump = sa.analyze_state(sheet_state(lambda x, y: 0.004 * np.exp(-(x ** 2 + y ** 2) / (2 * 0.02 ** 2))))
        for r in (cyl, sad, bump):
            self.assertFalse(r['qualified'])
        self.assertGreater(sad['whole_opposite_sign_fraction'], 0.8)
        self.assertLess(bump['largest_connected_fraction'], 0.5)

    def test_flattening_never_raises_the_search_objective(self):
        js = [sa.analyze_state(self.dome(scale=s))['j_combined'] for s in (0.2, 0.5, 1.0, 2.0)]
        self.assertTrue(all(b >= a for a, b in zip(js, js[1:])), js)
        self.assertGreater(js[-1], js[0])

    def test_roi_is_the_central_square_not_the_whole_sheet(self):
        # a spherical cap confined to r <= 50 mm on an otherwise flat sheet
        state = sheet_state(lambda x, y: self.R - np.sqrt(self.R ** 2 - np.minimum(np.hypot(x, y), 0.05) ** 2))
        self.assertFalse(sa.analyze_state(state, roi_half=0.08)['qualified'])
        self.assertTrue(sa.analyze_state(state, roi_half=0.03)['qualified'])


class TestFailureLearningAndBatches(unittest.TestCase):

    def fit(self, n_scored=6, n_failed=4, seed=3):
        rng = np.random.default_rng(seed)
        pts = [rng.random(sh.N_DIM) for _ in range(n_scored + n_failed)]
        raw = np.array([sh.realized_raw(sh.decode(u)) for u in pts])
        y = np.array([0.1 * rng.random() for _ in range(n_scored)] + [np.nan] * n_failed)
        z = np.array([1] * n_scored + [0] * n_failed)
        sur = sc.make_surrogate()
        sur.fit(raw, y, z)
        return sur, raw

    def test_failures_without_scores_update_feasibility_and_all_failed_is_defined(self):
        sur, raw = self.fit()
        self.assertEqual(sur.summary()['n_infeasible'], 4)
        p = sur.predict(raw)['p_feasible']
        self.assertTrue(np.all((p > 0) & (p < 1)))
        sur0, raw0 = self.fit(n_scored=1, n_failed=5)           # fewer than two scored solves
        cei = sur0.constrained_expected_improvement(raw0)
        self.assertTrue(np.all(np.isfinite(cei)))

    def test_batch_is_new_distinct_designs_scored_at_realized_coordinates(self):
        sur, raw = self.fit()
        keys = {tuple(np.round(r, 6)) for r in raw}
        batch = sc.select_batch(sur, keys, np.random.default_rng(5), k=4, n_pool=600)
        bkeys = [sh.decode_key(u) for u, _ in batch]
        self.assertEqual(len(set(bkeys)), 4)
        self.assertFalse(set(bkeys) & keys)
        for (u, cei) in batch:
            realized = sh.realized_raw(sh.decode(u))
            self.assertAlmostEqual(float(sur.constrained_expected_improvement(realized)), cei, places=9)


def fake_runner(path: Path, pass_attempts=(), fail_all=False, error_attempts=(), hang_attempts=(), crash_attempts=(),
                bad_digest=(), wrong_binary=(), late_exit=(), late_hang=()):
    path.write_text(f'''
import json, sys, time
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'dome_bo')!r})
import square_attempt as sa
d = Path(sys.argv[sys.argv.index("--attempt-dir") + 1])
req = json.loads((d / "request.json").read_text())
n = req["attempt"]
(d / "solve").mkdir()
rec = req["recipe"]
(d / "solve/sequence.json").write_text(json.dumps(rec))
digest = "0" * 64 if n in {list(bad_digest)!r} else req["recipe_sha256"]
binary = "1" * 64 if n in {list(wrong_binary)!r} else sa.sha(sa.BINARY)
if n in {list(hang_attempts)!r}:
    time.sleep(60)
if n in {list(crash_attempts)!r}:
    sys.exit(3)                                    # no terminal status at all
fail = {fail_all!r} or n % 3 == 0
if n in {list(error_attempts)!r}:
    st = {{"state": "error", "failure_type": "AnalysisTimeout", "recipe_sha256": digest, "binary_sha256": binary}}
elif n in {list(pass_attempts)!r}:
    m = {{"j_combined": 0.5, "qualified": True, "largest_connected_fraction": 0.9}}
    st = {{"state": "accepted", "recipe_sha256": digest, "binary_sha256": binary, "metrics": m, "elapsed_seconds": 1.0}}
elif fail:
    st = {{"state": "failed", "failure_type": "EquilibriumNotReached", "recipe_sha256": digest, "binary_sha256": binary, "elapsed_seconds": 1.0}}
else:
    m = {{"j_combined": 0.01 * n, "qualified": False, "largest_connected_fraction": 0.3}}
    st = {{"state": "accepted", "recipe_sha256": digest, "binary_sha256": binary, "metrics": m, "elapsed_seconds": 1.0}}
(d / "status.json").write_text(json.dumps(st))
if n in {list(late_exit)!r}:
    sys.exit(5)                                    # terminal status published, launcher step then fails
if n in {list(late_hang)!r}:
    time.sleep(60)                                 # terminal status published, step then hangs
''')
    return path


class TestController(unittest.TestCase):

    def campaign(self, tmp, **kw):
        keys = ('pass_attempts', 'fail_all', 'error_attempts', 'hang_attempts', 'crash_attempts', 'bad_digest',
                'wrong_binary', 'late_exit', 'late_hang')
        runner = fake_runner(Path(tmp) / 'fake.py', **{k: kw.pop(k) for k in keys if k in kw})
        return sc.Campaign(Path(tmp) / 'run', launcher='', runner=runner, **kw)

    def test_reservation_ledger_provenance_and_failure_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, max_attempts=8)
            self.assertIn('attempts reached', camp.run())
            ledger = camp.ledger()
            self.assertEqual([r['attempt'] for r in ledger], list(range(1, 9)))
            self.assertEqual([r['source'] for r in ledger][:2], ['anchor', 'anchor'])
            for r in ledger:
                req = json.loads((camp.attempts_dir / f"attempt_{r['attempt']:03d}/request.json").read_text())
                self.assertEqual(req['recipe_sha256'], r['recipe_sha256'])
                self.assertEqual(sh.recipe_sha(sh.decode(np.array(r['u']))), r['recipe_sha256'])
                if r['state'] == 'failed':
                    self.assertIsNone(r['y'])
                    self.assertEqual(r['z'], 0)
                else:
                    self.assertTrue(np.isfinite(r['y']))
            self.assertTrue(any(r['state'] == 'failed' for r in ledger))
            self.assertEqual(len({sh.decode_key(np.array(r['u'])) for r in ledger}), 8)

    def test_stops_after_a_wave_with_a_gate_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, pass_attempts=(2,))
            self.assertIn('passed the primary gate', camp.run())
            self.assertEqual(len(camp.ledger()), 4)                  # the whole wave is retained

    def test_stops_after_two_all_failed_waves(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, fail_all=True)
            self.assertIn('every solve failed', camp.run())
            self.assertEqual(len(camp.ledger()), 8)

    def test_identity_change_and_interrupted_attempts_stop_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, max_attempts=4)
            camp.run()
            ident = json.loads((camp.root / 'identity.json').read_text())
            ident['seed'] = 1
            ident['bo_seed'] = 1
            (camp.root / 'identity.json').write_text(json.dumps(ident))
            with self.assertRaises(RuntimeError):
                self.campaign(tmp, max_attempts=4).run()
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, max_attempts=4)
            d = camp.reserve(1, 0, REF, 'anchor', None)              # reserved, never collected
            with self.assertRaises(RuntimeError):
                self.campaign(tmp, max_attempts=4).run()
            rec = camp.ledger()
            self.assertEqual((rec[0]['state'], rec[0]['failure_type']), ('error', 'interrupted'))
            self.assertIsNone(rec[0]['z'])                           # never a numerical-feasibility label
            again = self.campaign(tmp, max_attempts=4)               # the stop is durable across re-entry
            self.assertIn('researcher decision required', again.run())
            self.assertEqual(len(again.ledger()), 1)                 # nothing launched, nothing replayed

    def run_and_ledger(self, tmp, **kw):
        camp = self.campaign(tmp, **kw)
        reason = camp.run()
        return camp, reason, camp.ledger()

    def test_infrastructure_and_analysis_errors_are_not_training_labels_and_stop_the_campaign(self):
        cases = {'error_attempts': (2,), 'crash_attempts': (2,), 'bad_digest': (2,), 'wrong_binary': (2,)}
        for key, attempts in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                camp, reason, ledger = self.run_and_ledger(tmp, **{key: attempts})
                self.assertIn('researcher decision required', reason, key)
                self.assertEqual(len(ledger), 4, key)                # the wave's sibling results are retained
                bad = [r for r in ledger if r['state'] == 'error']
                self.assertEqual([r['attempt'] for r in bad], [2], key)
                self.assertIsNone(bad[0]['z'])
                self.assertIsNone(bad[0]['y'])
                self.assertNotIn(2, [r['attempt'] for r in ledger if r['state'] in ('accepted', 'failed')])
                with self.assertRaises(RuntimeError):                # no wave may follow an error record
                    camp.next_wave()

    def test_genuine_numerical_failure_remains_an_unscored_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp, reason, ledger = self.run_and_ledger(tmp, max_attempts=8)
            failed = [r for r in ledger if r['state'] == 'failed']
            self.assertTrue(failed)
            self.assertTrue(all(r['z'] == 0 and r['y'] is None for r in failed))
            self.assertIn('attempts reached', reason)

    def test_terminal_status_followed_by_launcher_failure_is_an_infrastructure_stop(self):
        for key, kw, ftype in (('late_exit', {}, 'LauncherNonzeroExit'), ('late_hang', {'step_timeout': 3}, 'LauncherTimeout')):
            for n in (1, 3):                                         # attempt 1 -> accepted, attempt 3 -> failed
                with tempfile.TemporaryDirectory() as tmp:
                    camp = self.campaign(tmp, **{key: (n,)}, **kw)
                    reason = camp.run()
                    ledger = camp.ledger()
                    self.assertEqual(len(ledger), 4, (key, n))       # the wave is retained, no further launch
                    rec = [r for r in ledger if r['attempt'] == n][0]
                    self.assertEqual((rec['state'], rec['failure_type']), ('error', ftype), (key, n))
                    self.assertIsNone(rec['y'])
                    self.assertIsNone(rec['z'])
                    self.assertIn('researcher decision required', reason)
                    with self.assertRaises(RuntimeError):
                        camp.next_wave()

    def test_slow_surrogate_fit_is_bounded_by_the_proposal_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, proposal_timeout=1)
            camp.root.mkdir(parents=True, exist_ok=True)
            initial = sc.anchors() + sc.initial_design()
            for i, u in enumerate(initial):                          # a complete initial design on the ledger
                camp.append({'attempt': i + 1, 'wave': i // 4, 'source': 'design', 'u': list(map(float, u)),
                             'realized': list(map(float, sh.realized_raw(sh.decode(u)))),
                             'recipe_sha256': sh.recipe_sha(sh.decode(u)), 'state': 'accepted', 'z': 1,
                             'y': 0.01 * i, 'qualified': False})
            class Slow:
                def fit(self, *args, **kwargs):
                    import time
                    time.sleep(30)
            original = sc.make_surrogate
            sc.make_surrogate = Slow
            try:
                import time
                t0 = time.monotonic()
                with self.assertRaises(sc.ProposalTimeout):
                    camp.next_wave()
                self.assertLess(time.monotonic() - t0, 10)
            finally:
                sc.make_surrogate = original
            self.assertEqual(len(camp.ledger()), len(initial))       # nothing was reserved or launched

    def test_step_timeout_terminates_the_launcher_and_records_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, hang_attempts=(3,), step_timeout=3)
            reason = camp.run()
            ledger = camp.ledger()
            self.assertEqual(len(ledger), 4)
            hung = [r for r in ledger if r['attempt'] == 3][0]
            self.assertEqual((hung['state'], hung['failure_type']), ('error', 'LauncherTimeout'))
            self.assertIn('researcher decision required', reason)

    def test_identity_is_rechecked_between_waves(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, max_attempts=8)
            camp.freeze_identity()
            ident = json.loads((camp.root / 'identity.json').read_text())
            ident['files']['dome_bo/square_hatch.py'] = '0' * 64      # a code file changed since the freeze
            (camp.root / 'identity.json').write_text(json.dumps(ident))
            with self.assertRaises(RuntimeError):
                camp.run()
            self.assertEqual(camp.ledger(), [])
            self.assertIn('dome_bo/run_square_campaign.sbatch', camp.identity()['files'])

    def test_relative_roots_are_made_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                camp = sc.Campaign(Path('rel/campaign'), launcher='', runner=Path(tmp) / 'fake.py')
                self.assertTrue(camp.root.is_absolute())
                self.assertTrue(camp.attempts_dir.is_absolute())
            finally:
                os.chdir(cwd)

    def test_allocation_reserve_stops_before_a_wave(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            camp = self.campaign(tmp, end_epoch=time.time() + 100)
            self.assertIn('allocation reserve', camp.run())
            self.assertEqual(camp.ledger(), [])


class TestRunnerPaths(unittest.TestCase):

    def test_relative_attempt_dir_gives_an_absolute_cycle_file_the_solver_can_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                d = Path('attempts/attempt_001')
                d.mkdir(parents=True)
                recipe = sh.decode(REF)
                (d / 'request.json').write_text(json.dumps(
                    {'attempt': 1, 'recipe': recipe, 'recipe_sha256': sh.recipe_sha(recipe)}))
                stub = Path(tmp) / 'stub_solver.sh'
                stub.write_text('#!/bin/bash\n'                             # records the -cycle_file it can read
                                'while [ $# -gt 0 ]; do [ "$1" = "-cycle_file" ] && f="$2"; shift; done\n'
                                'cp "$f" cycle_file_seen.json && echo "equilibrium corrector failed" >&2; exit 1\n')
                stub.chmod(0o755)
                status = sa.run_attempt(d, binary=stub)
                seen = Path(tmp) / 'attempts/attempt_001/solve/cycle_file_seen.json'
                self.assertTrue(seen.exists())                               # the solver could open the file
                self.assertEqual(sh.recipe_sha(json.loads(seen.read_text())), sh.recipe_sha(recipe))
                cmd = status['command']
                self.assertTrue(Path(cmd[cmd.index('-cycle_file') + 1]).is_absolute())
                self.assertEqual((status['state'], status['failure_type']), ('failed', 'EquilibriumNotReached'))
            finally:
                os.chdir(cwd)

    def test_failure_classification(self):
        def attempt(script, tmp):
            d = Path(tmp) / 'a'
            d.mkdir()
            recipe = sh.decode(REF)
            (d / 'request.json').write_text(json.dumps(
                {'attempt': 1, 'recipe': recipe, 'recipe_sha256': sh.recipe_sha(recipe)}))
            stub = Path(tmp) / 's.sh'
            stub.write_text('#!/bin/bash\n' + script)
            stub.chmod(0o755)
            return sa.run_attempt(d, binary=stub, timeout=2)

        with tempfile.TemporaryDirectory() as t1:
            st = attempt('echo "what(): zigzag_sequence: equilibrium corrector failed at cycle 3" >&2; exit 134', t1)
            self.assertEqual(st['state'], 'failed')
        with tempfile.TemporaryDirectory() as t2:
            st = attempt('echo "bad cycle file" >&2; exit 1', t2)          # configuration/process problem
            self.assertEqual((st['state'], st['failure_type']), ('error', 'SolverProcessError'))
        with tempfile.TemporaryDirectory() as t3:
            st = attempt('sleep 30', t3)                                   # declared solver timeout
            self.assertEqual((st['state'], st['failure_type']), ('failed', 'SolverTimeout'))
        with tempfile.TemporaryDirectory() as t4:
            st = attempt('exit 0', t4)                                     # exit 0 but no artifacts to verify
            self.assertEqual(st['state'], 'error')


@unittest.skipUnless(os.environ.get('SLURM_JOB_ID'), 'real solver fixture needs a compute allocation')
class TestRealSolverFixture(unittest.TestCase):
    """The reviewed solver route on a coarse 80 mm square: provenance and failure recording, not quality."""

    def shrunk(self, scale=0.3):
        recipe = sh.decode(REF)
        for tp in recipe['toolpaths']:
            tp['operation']['lv_mm'] *= scale
            tp['operation']['width_mm'] *= scale
        return recipe

    def reserve(self, d, recipe):
        d.mkdir(parents=True)
        (d / 'request.json').write_text(json.dumps({'attempt': 1, 'recipe': recipe, 'recipe_sha256': sh.recipe_sha(recipe)}))

    def test_accepted_attempt_is_verified_scored_and_tied_to_the_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'attempt'
            recipe = self.shrunk()
            self.reserve(d, recipe)
            status = sa.run_attempt(d, mesh=('.04', '.04', '.15'), threads=1, fixture=True)
            self.assertEqual(status['state'], 'accepted', status.get('error'))
            self.assertEqual(status['recipe_sha256'], sh.recipe_sha(recipe))
            written = json.loads((d / 'solve/sequence.json').read_text())
            self.assertEqual(sh.recipe_sha(written), sh.recipe_sha(recipe))
            self.assertTrue(np.isfinite(status['metrics']['j_combined']))
            self.assertEqual(status['binary_sha256'], sa.sha(sa.BINARY))

    def test_unfit_recipe_is_a_recorded_error_never_a_training_label(self):
        """A recipe the simulator rejects (here 192 mm strokes on an 80 mm sheet) is a configuration problem, not
        a numerical non-acceptance: state 'error' (the campaign stops for the researcher), no metrics, no exception."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'attempt'
            self.reserve(d, sh.decode(REF))
            status = sa.run_attempt(d, mesh=('.04', '.04', '.15'), threads=1, fixture=True)
            self.assertEqual(status['state'], 'error')
            self.assertEqual(status['failure_type'], 'SolverProcessError')
            self.assertNotIn('metrics', status)
            self.assertTrue((d / 'status.json').exists())


if __name__ == '__main__':
    unittest.main()
