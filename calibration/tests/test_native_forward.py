"""Bounded plumbing fixtures using synthetic geometry; NOT physical evidence."""
import copy
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from calibration.ewcal.common import ROOT, Parameters, load_config, save_json
from calibration.ewcal.runner import forward, check_output, SolverFailure


class NativeForwardTests(unittest.TestCase):
    def config(self):
        c = load_config(ROOT/'calibration/configs/run6.json')
        c['solver']['executable'] = str(ROOT/'calibration/tests/mock_solver.py')
        c['solver']['command_prefix'] = [sys.executable]
        c['solver']['args']['-equilibrium_solver'] = 'native_hybrid'
        c['solver']['acceptance']['max_native_hybrid_gradient_norm'] = 5e-14
        c['cases'][0]['target'] = '/nonexistent/experimental-scan.npz'
        return c

    def test_forward_needs_no_scan_and_keeps_four_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            c = self.config(); x = Parameters(c).initial()
            out = Path(temp)/'forward'
            result = forward(c, x, out)
            self.assertTrue(result['valid']); self.assertEqual(result['mode'], 'forward_only')
            self.assertNotIn('objective_mm', result)
            # Existing normalized/logarithmic parameter roundtrip has ordinary
            # float64 rounding; the emitted operation must match decoded values exactly.
            for key in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
                self.assertAlmostEqual(result['parameters'][key], c['parameters'][key]['initial'], places=14)
            op = json.loads((out/'case_00/sequence.json').read_text())['toolpaths'][0]['operation']
            self.assertTrue(all(op[k] == v for k, v in result['parameters'].items()))
            with self.assertRaises(FileExistsError): forward(c, x, out)

    def test_native_acceptance_requires_numerical_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            c = self.config(); result = forward(c, Parameters(c).initial(), Path(temp)/'forward')
            self.assertTrue(result['valid'])
            folder = Path(result['cases'][0]['directory']); original = json.loads((folder/'sequence_result.json').read_text())
            corruptions = [('accepted', False), ('termination', 'time_cap'), ('curvature', 'negative_pivots'),
                           ('gradient_norm', None), ('energy', None), ('gradient_tolerance', 1e-7), ('verification_hessians', 0), ('verification_hessians', None), ('verification_evaluations', True)]
            for key, value in corruptions:
                with self.subTest(key=key):
                    m = copy.deepcopy(original); m['solves'][0]['native_hybrid'][key] = value
                    save_json(folder/'sequence_result.json', m)
                    with self.assertRaises(SolverFailure): check_output(c, c['cases'][0], folder)
            for key, value in [('return_code', 1), ('reported_eps', 1e-7)]:
                m = copy.deepcopy(original); m['solves'][0][key] = value; save_json(folder/'sequence_result.json', m)
                with self.assertRaises(SolverFailure): check_output(c, c['cases'][0], folder)
            m = copy.deepcopy(original); m['backend'] = 'hlbfgs'; save_json(folder/'sequence_result.json', m)
            with self.assertRaises(SolverFailure): check_output(c, c['cases'][0], folder)
            # Synthetic parser-only evidence; not an adaptive optimizer run.
            adaptive = copy.deepcopy(c); adaptive['solver']['args']['-native_hybrid_handoff'] = 'adaptive'
            m = copy.deepcopy(original); proof = m['solves'][0]['native_hybrid']
            proof.update(warmup_attempts=2, newton_attempts=1, probe_attempts=1, probes=1,
                         fallbacks=0, attempts=3, lbfgs_seconds=.2, newton_seconds=.1,
                         policy_events=[{'event': 'newton_probe'}], handoff_policy={
                'mode': 'adaptive', 'version': 'gradient-cost-v1', 'minimum_warmup': 50,
                'window': 20, 'plateau_ratio': .90, 'relative_gradient': .01,
                'trial_gradient_ratio': .95, 'rate_advantage': 1.1, 'maximum_probes': 8,
                'maximum_fallbacks': 2, 'newton_window': 4, 'newton_plateau_ratio': .90})
            save_json(folder/'sequence_result.json', m); check_output(adaptive, adaptive['cases'][0], folder)
            for key, value in [('attempts', 2), ('probe_attempts', 2), ('fallbacks', 3),
                               ('newton_attempts', True), ('lbfgs_seconds', None), ('policy_events', [])]:
                bad = copy.deepcopy(m); bad['solves'][0]['native_hybrid'][key] = value
                save_json(folder/'sequence_result.json', bad)
                with self.assertRaises(SolverFailure): check_output(adaptive, adaptive['cases'][0], folder)
            bad = copy.deepcopy(m); bad['solves'][0]['native_hybrid']['handoff_policy']['window'] = 40
            save_json(folder/'sequence_result.json', bad)
            with self.assertRaises(SolverFailure): check_output(adaptive, adaptive['cases'][0], folder)
            save_json(folder/'sequence_result.json', original)
            c['solver']['acceptance'].pop('max_native_hybrid_gradient_norm')
            with self.assertRaises(SolverFailure): check_output(c, c['cases'][0], folder)

    def test_existing_cli_forward_and_binary_override(self):
        with tempfile.TemporaryDirectory() as temp:
            c = self.config(); c['solver']['executable'] = '/nonexistent/unqualified-binary'
            root = Path(temp); save_json(root/'config.json', c)
            child = subprocess.run([sys.executable, '-B', '-m', 'calibration.ewcal', 'forward',
                '--config', str(root/'config.json'), '--executable', str(ROOT/'calibration/tests/mock_solver.py'),
                '--case', c['cases'][0]['id'], '--output', str(root/'forward')],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            self.assertEqual(child.returncode, 0, child.stdout.decode())
            self.assertTrue(json.loads((root/'forward/result.json').read_text())['valid'])

    def test_failure_is_reported_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            c = self.config(); c['solver']['args']['-mock_fail'] = True
            result = forward(c, Parameters(c).initial(), Path(temp)/'forward')
            self.assertFalse(result['valid']); self.assertEqual(result['cases'], [])
            self.assertIn('exit code 4', result['error'])


if __name__ == '__main__': unittest.main()
