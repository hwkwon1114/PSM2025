"""No-solver checks for the frozen Run 2 starting-state diagnostic."""
import copy
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from calibration.analysis import run2_initial_state_diagnostic as diag
from calibration.ewcal.runner import sequence_json


class InitialStateDiagnosticTests(unittest.TestCase):
    def test_only_seed_and_output_paths_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'diagnostic'
            cfg, params = diag.make_configs(out, diag.frozen_parameters())
            flat, seeded = copy.deepcopy(cfg['flat']), copy.deepcopy(cfg['seeded'])
            for c in (flat, seeded):
                c.pop('output_dir')
                c.pop('report_dir')
            args = seeded['solver']['args']
            self.assertEqual(args.pop('-prescribed_at_cycle'), 32)
            self.assertEqual(args.pop('-prescribed_disp_csv'), str(out / 'inputs/seed_displacement_m.csv'))
            self.assertEqual(args.pop('-prescribed_reference'), 'rest')
            self.assertFalse(args.pop('-prescribed_reequilibrate'))
            self.assertEqual(args.pop('-prescribed_tag'), 'a0016_seed')
            self.assertEqual(flat, seeded)
            self.assertEqual(flat['solver']['args']['-native_hybrid_seconds'], 3600)
            self.assertEqual(flat['solver']['timeout_seconds'], 3900)
            self.assertTrue(flat['solver']['args']['-native_hybrid_export_trace'])
            expected = json.loads(diag.SEQUENCE.read_text())
            self.assertEqual(sequence_json(flat, flat['cases'][0], params), expected)
            self.assertEqual(params['gbot'], 0.0)

    def test_seed_roundtrip_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'diagnostic'
            manifest = diag.prepare(out)
            diag.verify_inputs(out)
            data = json.loads(diag.STATE.read_text())
            with (out / 'inputs/seed_displacement_m.csv').open() as f:
                displacement = np.loadtxt(f, delimiter=',', skiprows=1)
            self.assertEqual(manifest['vertices'], len(data['positions']))
            np.testing.assert_array_equal(displacement[:, 0], np.arange(manifest['vertices']))
            np.testing.assert_allclose(np.asarray(data['reference_positions']) + displacement[:, 1:],
                                       data['positions'], atol=1e-16, rtol=0)
            with (out / 'inputs/seed_displacement_m.csv').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'displacement file changed'):
                diag.verify_inputs(out)
            with self.assertRaises(FileExistsError):
                diag.prepare(out)

    def test_driver_continues_only_for_evidenced_numerical_failures(self):
        # The native report/exception format is from a saved corrector failure;
        # cycle 33 is the diagnostic case's final physical cycle.
        report = ('[native_hybrid] phase=newton live_handoff=1 warmup_attempts=2000 '
                  'total_attempts=2009 total_evaluations=2073 total_hessians=10 '
                  'status=indefinite_curvature curvature=indefinite accepted=0 '
                  'energy=2.7797365802220555e-11 gradient=4.3178051144144816e-11\n')
        ending = ("terminate called after throwing an instance of 'std::runtime_error'\n"
                  '  what():  zigzag_sequence: equilibrium corrector failed at cycle 33, '
                  'lambda=0.015625, gradient_norm=0.000000, termination=indefinite_curvature.\n')
        aborted = {'valid': False, 'failure_class': 'infrastructure', 'failure_phase': 'solver',
                   'solver_termination': {'return_code': -signal.SIGABRT},
                   'seconds': 10, 'error': 'abort', 'objective_mm': 1000000.0}
        timed_out = {'valid': False, 'failure_class': 'timeout', 'failure_phase': 'solver',
                     'solver_termination': {'reason': 'controller_elapsed_cap'},
                     'seconds': 3900, 'error': 'timeout', 'objective_mm': 1000000.0}
        valid = {'valid': True, 'failure_class': None, 'seconds': 2, 'error': None,
                 'objective_mm': .5, 'cases': [{'manifest': {'total_energy': 1e-12}}]}

        for name, first, log, expected, continues in (
                ('corrector', aborted, report + ending, 'solver', True),
                ('timeout', timed_out, '', 'timeout', True),
                ('unknown_abort', aborted, 'unrelated abort\n', 'infrastructure', False),
                ('mismatched_status', aborted, report + ending.replace('termination=indefinite_curvature',
                                                                      'termination=newton_stagnation'),
                 'infrastructure', False)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'diagnostic'
                configs = {n: {'output_dir': str(out / n)} for n in ('flat', 'seeded')}
                calls = []

                class FakeEvaluator:
                    def __init__(self, cfg):
                        self.base = Path(cfg['output_dir'])

                    def evaluate(self, *_args, **_kwargs):
                        calls.append(self.base.name)
                        folder = self.base / 'evaluations/attempt_0001/case_00'
                        folder.mkdir(parents=True)
                        (folder / 'solver.log').write_text(log if len(calls) == 1 else '')
                        return copy.deepcopy(first if len(calls) == 1 else valid)

                compared = {'history_matches_original_for_both': True, 'target_form_comparison': None}
                with (mock.patch.object(diag, 'OUT', out),
                      mock.patch.object(diag, 'verify_inputs', return_value=(configs, {'gtop': 0.0})),
                      mock.patch.object(diag, 'Evaluator', FakeEvaluator),
                      mock.patch.object(diag, 'verify_saved_start', return_value={}),
                      mock.patch.object(diag, 'compare_completed', return_value=compared)):
                    if continues:
                        diag.run()
                        self.assertEqual(calls, ['flat', 'seeded'])
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'infrastructure failure'):
                            diag.run()
                        self.assertEqual(calls, ['flat'])
                summary = json.loads((out / 'partial_summary.json').read_text())
                self.assertEqual(summary['flat']['failure_class'], expected)
                self.assertEqual(summary['flat']['evaluator_failure_class'], first['failure_class'])
                self.assertIsNone(summary['flat']['objective_mm'])
                raw = json.loads((out / 'flat/forward_result.json').read_text())
                self.assertEqual(raw['failure_class'], first['failure_class'])
                self.assertEqual(raw['objective_mm'], 1000000.0)
                if name == 'corrector':
                    self.assertEqual(summary['flat']['classification_evidence']['status'],
                                     'indefinite_curvature')


if __name__ == '__main__':
    unittest.main()
