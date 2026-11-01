"""Bounded no-solver checks for the Run 2 BLIS comparison."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from calibration.analysis import run2_blis_diagnostic as diag
from calibration.ewcal.runner import SolverProcessFailure, command


class BlisDiagnosticTests(unittest.TestCase):
    def test_frozen_forward_plan_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pair'
            diag.prepare(out)
            plan = diag.verify_plan(out)
            cfg = plan['config']
            self.assertEqual(cfg['cases'][0]['repeat'], 33)
            self.assertEqual(cfg['solver']['solve_mode'], 'final_only')
            self.assertEqual(cfg['solver']['timeout_seconds'], 3900)
            self.assertEqual(cfg['solver']['args']['-native_hybrid_seconds'], 3600)
            self.assertTrue(cfg['solver']['args']['-native_hybrid_export_trace'])
            self.assertFalse(any(k.startswith('-prescribed') for k in cfg['solver']['args']))
            self.assertEqual(plan['frozen_parameters']['gbot'], 0.0)
            self.assertEqual(plan['sequence'], json.loads(diag.original.SEQUENCE.read_text()))
            first = command(cfg, cfg['cases'][0], out / 'unset/sequence.json')
            second = command(cfg, cfg['cases'][0], out / 'one/sequence.json')
            template = list(plan['command_template'])
            first[first.index(str(out / 'unset/sequence.json'))] = 'SEQUENCE'
            second[second.index(str(out / 'one/sequence.json'))] = 'SEQUENCE'
            template[template.index(str(out / 'ARM/sequence.json'))] = 'SEQUENCE'
            self.assertEqual(first, template)
            self.assertEqual(second, template)
            with self.assertRaises(FileExistsError):
                diag.prepare(out)
            plan['input_sha256'][str(diag.original.BEST)] = 'wrong'
            (out / 'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'changed'):
                diag.verify_plan(out)

    def test_blis_settings_and_common_environment(self):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for key in diag.THREAD_ENV[4:10]:
            env.pop(key, None)
        program = ('import json,sys; from calibration.analysis import run2_blis_diagnostic as d; '
                   'print(json.dumps(d.runtime_environment(sys.argv[1], d.libraries())))')
        outputs = []
        for arm in ('unset', 'one'):
            child = env.copy()
            if arm == 'one':
                child['BLIS_NUM_THREADS'] = '1'
            raw = subprocess.check_output([sys.executable, '-c', program, arm], env=child, text=True)
            outputs.append(json.loads(raw))
        self.assertEqual([v['blis_configured_threads_parent_process'] for v in outputs], [8, 1])
        self.assertEqual(outputs[0]['common_environment_sha256'], outputs[1]['common_environment_sha256'])
        self.assertEqual(outputs[0]['cpu_affinity'], outputs[1]['cpu_affinity'])

    def test_timeout_continues_but_unknown_failure_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pair'
            diag.prepare(out)
            common = {'common_environment_sha256': 'same', 'hostname': 'node', 'cpu_affinity': [1, 2],
                      'slurm_job_id': 'job', 'slurm_step_id': 'step'}
            def timeout(_args, folder, *_rest, **_kwargs):
                (folder / 'solver.log').write_text('')
                raise SolverProcessFailure('timeout', {'reason': 'controller_elapsed_cap', 'timeout': True})
            with (mock.patch.dict(os.environ),
                  mock.patch.object(diag, 'runtime_environment', return_value=common),
                  mock.patch.object(diag, 'launch', side_effect=timeout) as launched):
                os.environ.pop('BLIS_NUM_THREADS', None)
                diag.run_arm('unset', out)
                os.environ['BLIS_NUM_THREADS'] = '1'
                diag.run_arm('one', out)
            unset_env = launched.call_args_list[0].kwargs['solver_environment']
            one_env = launched.call_args_list[1].kwargs['solver_environment']
            self.assertNotIn('BLIS_NUM_THREADS', unset_env)
            self.assertEqual(one_env['BLIS_NUM_THREADS'], '1')
            for arm in ('unset', 'one'):
                result = json.loads((out / arm / 'result.json').read_text())
                self.assertFalse(result['valid'])
                self.assertEqual(result['failure_class'], 'timeout')
                self.assertIsNone(result['start_check'])
                self.assertNotIn('energy', result)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pair'
            diag.prepare(out)
            def unknown(_args, folder, *_rest, **_kwargs):
                (folder / 'solver.log').write_text('unknown crash\n')
                raise SolverProcessFailure('unknown', {'return_code': -6})
            with (mock.patch.object(diag, 'runtime_environment', return_value=common),
                  mock.patch.object(diag, 'launch', side_effect=unknown)):
                with self.assertRaisesRegex(RuntimeError, 'unknown'):
                    diag.run_arm('unset', out)
            result = json.loads((out / 'unset/result.json').read_text())
            self.assertEqual(result['failure_class'], 'infrastructure')
            self.assertFalse((out / 'one').exists())

    def test_mismatched_affinity_is_rejected_before_second_solve(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pair'
            (out / 'unset').mkdir(parents=True)
            env = {'common_environment_sha256': 'same', 'hostname': 'node', 'cpu_affinity': [1],
                   'slurm_job_id': 'job', 'slurm_step_id': 'step'}
            (out / 'unset/result.json').write_text(json.dumps({'environment': env}))
            changed = copy.deepcopy(env)
            changed['cpu_affinity'] = [2]
            with (mock.patch.object(diag, 'verify_plan', return_value={'libraries': {}}),
                  mock.patch.object(diag, 'runtime_environment', return_value=changed),
                  mock.patch.object(diag, 'launch') as launched):
                with self.assertRaisesRegex(ValueError, 'cpu_affinity'):
                    diag.run_arm('one', out)
                launched.assert_not_called()

    def test_final_geometry_requires_matched_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pair'
            state = {'reference_positions': [[0, 0, 0], [1, 0, 0]], 'material_uv': [[0, 0], [1, 0]],
                     'faces': [[0, 1, 0]], 'edges': [[0, 1]], 'abar_top': [1], 'abar_bottom': [1],
                     'bbar': [0], 'material': {'thickness': 1}, 'positions': [[0, 0, 0], [1, 0, 0]]}
            for arm in ('unset', 'one'):
                folder = out / arm
                folder.mkdir(parents=True)
                (folder / 'result.json').write_text(json.dumps({'valid': True}))
                s = copy.deepcopy(state)
                if arm == 'one':
                    s['positions'][1][2] = 0.002
                (folder / 'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps(s))
            with mock.patch.object(diag, 'verify_plan'):
                diag.summarize(out)
            report = json.loads((out / 'summary.json').read_text())
            self.assertAlmostEqual(report['final_shape_comparison']['vertex_rms_mm'], 2 / 2**0.5)
            changed = copy.deepcopy(state)
            changed['abar_top'] = [2]
            (out / 'one/bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps(changed))
            with mock.patch.object(diag, 'verify_plan'):
                with self.assertRaisesRegex(ValueError, 'load identity'):
                    diag.summarize(out)


if __name__ == '__main__':
    unittest.main()
