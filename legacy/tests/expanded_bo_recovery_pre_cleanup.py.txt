"""Solver/GP-free recovery fixtures; production data is read-only."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from calibration.bo import run_run6_expanded_bo_recovery as recovery
from calibration.bo import run_run6_expanded_bo as runner
from calibration.bo import run_run6_bo_refinement as controller
from calibration.ewcal.common import Parameters, save_json


class TestRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_history, cls.observations, cls.pins, cls.identity, cls.migration = recovery.validated_source()
        cls.best = runner.select_best(cls.observations)

    def setUp(self):
        self.runner_globals = {k: getattr(runner, k) for k in
                               ('BASE', 'HISTORY', 'REVIEW', 'CONFIG', 'META', 'publish_summary')}
        self.controller_globals = {k: getattr(controller, k) for k in
                                   ('BASE', 'HISTORY', 'CONFIG', 'META', 'CAMPAIGN_TYPE')}
        interop = patch('torch.set_num_interop_threads')
        interop.start()
        self.addCleanup(interop.stop)

    def tearDown(self):
        for k, v in self.runner_globals.items():
            setattr(runner, k, v)
        for k, v in self.controller_globals.items():
            setattr(controller, k, v)

    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.multiple(recovery, BASE=base, CONFIG=base/'frozen_config.json',
                                HISTORY=base/'corrected_history.json', REVIEW=base/'review'):
                history = dict(self.old_history, metadata=recovery.META,
                               observations=copy.deepcopy(self.observations),
                               evaluator_identity=self.identity, identity_migration=self.migration)
                save_json(recovery.CONFIG, self.old_history['config'])
                save_json(recovery.HISTORY, history)
                recovery.configure()
                yield base, history, controller.initial_state(history, controller.sha(recovery.HISTORY))

    @staticmethod
    def inventory(base):
        return {str(p.relative_to(base)): controller.sha(p) if p.is_file() else None
                for p in sorted(base.rglob('*'))}

    def test_exact_seeds_and_honest_identity_migration(self):
        self.assertEqual([o['attempt_id'] for o in self.observations], list(range(1, 205)))
        self.assertEqual(sum(o['valid'] for o in self.observations[160:]), 41)
        self.assertNotEqual(self.identity, recovery.OLD_IDENTITY)
        self.assertTrue(all(o['identity'] == recovery.OLD_IDENTITY for o in self.observations[160:]))
        self.assertEqual(self.best['attempt_id'], 187)
        with self.fixture() as (_, history, state):
            self.assertEqual(controller.validate_ledger(state, self.identity, 2026, history=history), 205)
            self.assertEqual(history['observations'], self.observations)

    def test_scientific_identity_drift_rejected(self):
        receipt = recovery.SOURCE/'cache'/recovery.OLD_IDENTITY/'identity.json'
        old = controller.read(receipt)
        old['versions']['numpy'] = 'not-the-frozen-version'
        with self.assertRaisesRegex(ValueError, 'beyond the unused CLI'):
            recovery.identity_migration(self.old_history['config'], old)
        changed = copy.deepcopy(self.old_history['config'])
        changed['solver']['timeout_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'beyond the unused CLI'):
            recovery.identity_migration(changed, controller.read(receipt))

    def test_corrupted_seed_and_excess_budget_rejected(self):
        with self.fixture() as (_, history, state):
            state['observations'][0]['parameters']['gtop'] += .0001
            with self.assertRaises(ValueError):
                controller.validate_ledger(state, self.identity, 2026, history=history)
            state = controller.initial_state(history, controller.sha(recovery.HISTORY))
            state['reserved_attempts'] = [{}] * 40
            with self.assertRaises(ValueError):
                controller.validate_ledger(state, self.identity, 2026, history=history)

    def execute(self, evaluator, reservations, state, state_file, eval_log, base, seed,
                lock, ledger_validator):
        for att, pt, reservation, idx in reservations:
            folder = Path(reservation['publication_dir'])
            folder.mkdir(parents=True)
            o = copy.deepcopy(self.best)
            o.update(attempt_id=att, normalized_parameters=pt.tolist(), identity=self.identity,
                     directory=str(folder), parameters=Parameters(self.old_history['config']).decode(pt),
                     proposal_idx=idx)
            reservation['status'] = 'completed'
            save_json(folder/'result.json', o)
            state['observations'].append(o)
            state['completed_attempts'].append(att)
        ledger_validator(state, self.identity, seed, runner.batch.validate_observation_record)
        save_json(state_file, state)
        runner.batch.sync_evaluations_log(eval_log, state['observations'])

    def test_nine_waves_remaining_seeds_and_no_replay(self):
        with self.fixture() as (base, history, _):
            pts = np.tile(self.best['normalized_parameters'], (4, 1))
            with patch.object(runner, 'check_review'), patch.object(controller, 'verify_inputs'), \
                    patch.object(runner, 'production_evaluator', return_value=None), \
                    patch.object(runner.batch, 'get_remaining_allocation_seconds', return_value=43200), \
                    patch.object(runner, 'bounded_proposal', return_value=(pts, {})) as propose, \
                    patch.object(runner.batch, '_execute_wave', side_effect=self.execute):
                runner.run('fixture')
            saved = controller.read(base/'bayes_state.json')
            summary = controller.read(base/'bayes_summary.json')
            self.assertEqual(saved['status'], 'budget_complete')
            self.assertEqual(saved['observations'][:204], self.observations)
            self.assertEqual([o['attempt_id'] for o in saved['observations']], list(range(1, 241)))
            self.assertEqual([c.args[1] for c in propose.call_args_list], list(range(2048, 2057)))
            self.assertEqual(summary['new_completed'], 36)
            self.assertEqual(summary['combined_continuation']['new_completed'], 80)
            self.assertEqual(summary['combined_continuation']['new_valid'], 77)
            with patch.object(runner, 'check_review'), patch.object(controller, 'verify_inputs'), \
                    patch.object(runner, 'bounded_proposal') as no_replay:
                runner.run('fixture')
                no_replay.assert_not_called()

    def test_real_constructor_worker_and_original_inventory(self):
        import vtk
        original = self.inventory(recovery.SOURCE)
        cleanup = controller.sha(runner.ROOT/'calibration/configs/run6.json')
        with self.fixture() as (base, history, state):
            evaluator = runner.production_evaluator(history['config'])
            self.assertEqual(evaluator.identity, self.identity)
            self.assertEqual(evaluator.base, base)
            self.assertEqual(evaluator.config, self.old_history['config'])
            self.assertEqual(evaluator.identity_record['config'], self.old_history['config'])
            self.assertTrue((evaluator.cache/'identity.json').is_file())
            points = vtk.vtkPoints()
            points.SetDataTypeToDouble()
            for point in ((0, 0, 0), (300, 0, 0), (0, 400, 0)):
                points.InsertNextPoint(*point)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(3)
            for i in range(3):
                cells.InsertCellPoint(i)
            mesh = vtk.vtkPolyData()
            mesh.SetPoints(points)
            mesh.SetPolys(cells)
            namespace = evaluator.evaluate.__func__.__globals__

            def mechanics(args, cwd, timeout, *extra, **kwargs):
                save_json(cwd/'sequence_result.json', {'fixture': True})
                save_json(cwd/'bilayer_zigzag_sequence_native_hybrid_state.json', {
                    'schema_version': 1, 'units': 'metres',
                    'reference_positions': [[0, 0, 0], [.3, 0, 0], [0, .4, 0]], 'faces': [[0, 1, 2]],
                })

            def output(*args):
                owned = vtk.vtkPolyData()
                owned.DeepCopy(mesh)
                return {'fixture': True}, owned

            def score(mesh, target, reference, faces):
                return copy.deepcopy(self.best['cases'][0]['metrics']), mesh, reference, np.zeros(3), reference

            validator = lambda s, i, seed, v: controller.validate_ledger(s, i, seed, v, history=history)

            def wave(first, wave_idx):
                reservations = []
                for idx in range(4):
                    pt = np.array(self.best['normalized_parameters'])
                    pt[0] += .001 * (first-204+idx)
                    att = first+idx
                    folder = base/'evaluations'/f'attempt_{att:04d}'
                    r = dict(attempt_id=att, x=pt.tolist(), wave_idx=wave_idx, proposal_idx=idx,
                             evaluator_identity=self.identity, publication_dir=str(folder),
                             result_file=str(folder/'result.json'), status='reserved', created_at=0)
                    state['reserved_attempts'].append(r)
                    reservations.append((att, pt, r, idx))
                state.update(charged_attempts=first+3, waves_reserved=wave_idx)
                validator(state, self.identity, 2026, runner.batch.validate_observation_record)
                runner.batch._execute_wave(evaluator, reservations, state, base/'bayes_state.json',
                    base/'evaluations.jsonl', base, 2026, threading.Lock(), ledger_validator=validator)

            with patch.dict(namespace, launch=mechanics, check_output=output, score_material_frame=score):
                wave(205, 1)
            self.assertEqual(len(list(evaluator.cache.glob('*.json'))), 5)
            self.assertTrue(all(o['valid'] for o in state['observations'][204:]))
            for o in state['observations'][204:]:
                self.assertEqual(controller.read(Path(o['directory'])/'request.json')['config'], history['config'])

            def timeout(*args, **kwargs):
                raise namespace['SolverProcessFailure']('fixture timeout', {
                    'timeout': True, 'numerical': False, 'reason': 'controller_elapsed_cap'})

            with patch.dict(namespace, launch=timeout):
                wave(209, 2)
            self.assertTrue(all(o['failure_class'] == 'timeout' for o in state['observations'][208:]))
            self.assertEqual(state['observations'][:204], self.observations)
            checkpoint = copy.deepcopy(state)
            self.assertEqual(runner.batch.reconcile_reservations(base, state, self.identity), (0, 0))
            self.assertEqual(state, checkpoint)
            validator(state, self.identity, 2026, runner.batch.validate_observation_record)
        self.assertEqual(self.inventory(recovery.SOURCE), original)
        self.assertEqual(controller.sha(runner.ROOT/'calibration/configs/run6.json'), cleanup)

    def test_deadline_real_constructor_no_proposals(self):
        with self.fixture() as (base, _, _):
            with patch.object(runner, 'check_review'), patch.object(controller, 'verify_inputs'), \
                    patch.object(runner.batch, 'get_remaining_allocation_seconds', return_value=0), \
                    patch.object(runner, 'bounded_proposal') as propose:
                runner.run('fixture')
                propose.assert_not_called()
            self.assertEqual(controller.read(base/'bayes_summary.json')['status'], 'deadline_stop')

    def test_preserved_failure_marker_prohibits_relaunch(self):
        with self.fixture() as (base, _, _):
            save_json(base/'controller_failure.json', {'fixture': 'preserved'})
            digest = controller.sha(base/'controller_failure.json')
            with self.assertRaisesRegex(RuntimeError, 'no relaunch'), patch.object(runner, 'bounded_proposal') as propose:
                runner.run('fixture')
            propose.assert_not_called()
            self.assertEqual(controller.sha(base/'controller_failure.json'), digest)


if __name__ == '__main__':
    unittest.main()
