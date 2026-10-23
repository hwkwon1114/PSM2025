"""Frozen-input checks for the approved Run 6/9 two-gripper comparison.

Run: python3 -m unittest discover -s calibration/tests -p test_two_gripper_setup.py
No solver is invoked. These checks do not validate checkpoint or release code.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'run6': ('run6_expanded_bo_continue20_recovery', 29,
             'ae885273b4da727a1815c4104d41328a09f96749e478cbf89dbe2ff58df92bb8'),
    'run9': ('run9_transfer_expanded', 33,
             'e3fe1bb10eaf4b45858b5902567dd7c6428da814642b1ea1951ed1d1ed4e86b5'),
}


class TwoGripperSetupTests(unittest.TestCase):
    def test_only_boundary_conditions_change(self):
        for name, (directory, repeats, digest) in CASES.items():
            with self.subTest(run=name):
                source = ROOT / 'run' / directory / 'evaluations/attempt_0209/case_00/sequence.json'
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
                original = json.loads(source.read_text())
                config = json.loads((ROOT / 'legacy/configs/boundary_conditions' /
                                     (name + '_two_gripper_cycle.json')).read_text())
                self.assertEqual(len(config['toolpaths']), 1)
                self.assertEqual(config['toolpaths'][0]['repeat'], repeats)
                original.pop('boundary_conditions')
                bc = config.pop('boundary_conditions')
                self.assertEqual(config, original)
                self.assertTrue(bc['enabled'])
                self.assertTrue(bc['release_after_final_cycle'])
                self.assertEqual(bc['coordinate_system'], 'material_uv')
                self.assertEqual(bc['application'], 'persistent')
                self.assertEqual(len(bc['regions']), 2)
                for region, x in zip(bc['regions'], (-50, 50)):
                    self.assertEqual(region['shape'], 'rectangle')
                    self.assertEqual(region['center_uv_mm'], [x, 140])
                    self.assertEqual(region['size_uv_mm'], [50, 20])
                    self.assertEqual(region['rotation_deg'], 0)
                    self.assertGreaterEqual(x - 25, -100)
                    self.assertLessEqual(x + 25, 100)
                    self.assertEqual(region['center_uv_mm'][1] + 10, 150)
                    self.assertEqual(region['center_uv_mm'][1] - 10 - 65, 65)

    def test_saved_coarse_restart_fixture(self):
        root = ROOT / 'run/two_gripper_comparison/fixtures'
        if not (root / 'final_resumed/paired_release_result.json').exists():
            self.skipTest('Run the three bounded coarse integration fixtures first')
        def load(run, filename):
            return json.loads((root / run / filename).read_text())['payload']
        a = load('final_uninterrupted', 'paired_clamped_checkpoint.json')
        b = load('final_resumed', 'paired_clamped_checkpoint.json')
        for key in ('identity', 'state', 'top', 'bottom', 'counts', 'cycle'):
            self.assertEqual(a[key], b[key], key)
        self.assertFalse((root / 'final_interrupted/accepted_cycle_002.json').exists())
        self.assertFalse((root / 'final_resumed/accepted_cycle_001.json').exists())
        starts = []
        for branch, rigid in (('putong', 0), ('free', 6)):
            a = load('final_uninterrupted', f'release_{branch}.json')
            b = load('final_resumed', f'release_{branch}.json')
            self.assertEqual(a['state'], b['state'])
            self.assertEqual(a['remaining_rigid_modes_start'], rigid)
            self.assertTrue(a['fixed_coordinates_unchanged'])
            self.assertTrue(a['target_history_unchanged'])
            self.assertTrue(a['proof']['accepted'])
            self.assertLessEqual(a['proof']['gradient_norm'], 5e-14)
            starts.append(a['start_state_checksum'])
        self.assertEqual(starts[0], starts[1])

    def test_never_clamped_coarse_controls(self):
        root = ROOT / 'run/two_gripper_comparison/fixtures/never_clamped_v2'
        if not (root / 'resumed/never_clamped_result.json').exists():
            self.skipTest('Run bounded never-clamped integration fixtures first')

        def load(run, filename):
            return json.loads((root / run / filename).read_text())['payload']

        full = load('uninterrupted', 'never_clamped_final_checkpoint.json')
        resumed = load('resumed', 'never_clamped_final_checkpoint.json')
        clamped = load('paired_regression', 'paired_clamped_checkpoint.json')
        for key in ('identity', 'state', 'top', 'bottom', 'counts', 'cycle'):
            self.assertEqual(full[key], resumed[key], key)
        for key in ('top', 'bottom', 'counts', 'cycle'):
            self.assertEqual(full[key], clamped[key], key)
        for key in ('rest', 'faces', 'edges', 'uv', 'bbar'):
            self.assertEqual(full['identity'][key], clamped['identity'][key], key)
        self.assertFalse((root / 'interrupted/accepted_cycle_002.json').exists())
        self.assertFalse((root / 'resumed/accepted_cycle_001.json').exists())
        for run, cycles in [('uninterrupted', (1, 2)), ('interrupted', (1,)), ('resumed', (2,))]:
            for cycle in cycles:
                checkpoint = load(run, 'accepted_cycle_%03d.json' % cycle)
                for mask in ('vertex_mask', 'edge_mask'):
                    self.assertFalse(any(any(row) for row in checkpoint['identity'][mask]))
                proof = checkpoint['proof']
                self.assertTrue(proof['accepted'])
                self.assertEqual(proof['termination'], 'gradient_target')
                self.assertEqual(proof['curvature'], 'positive_pivots')
                self.assertLessEqual(proof['gradient_norm'], proof['gradient_tolerance'])
                self.assertLessEqual(proof['gradient_tolerance'], 5e-14)
            if run != 'interrupted':
                result = load(run, 'never_clamped_result.json')
                self.assertEqual(result['remaining_rigid_modes'], 6)
                self.assertEqual(result['fixed_vertex_dofs'], 0)
                self.assertTrue(result['complete'])
                self.assertFalse((root / run / 'paired_release_result.json').exists())
                self.assertEqual(load(run, 'accepted_cycle_002.json'),
                                 load(run, 'never_clamped_final_checkpoint.json'))
        # Inputs may differ only in the physical-BC mode and checkpoint switches.
        runs = [json.loads((root / run / 'execution.json').read_text())['identity']
                for run in ('uninterrupted', 'paired_regression')]
        for identity in runs:
            identity['sequence'].pop('boundary_conditions')
            identity['options'].pop('-sequence_paired_release')
            identity['options'].pop('-sequence_never_clamped', None)
        self.assertEqual(runs[0], runs[1])
        terminal = load('terminal_resume', 'never_clamped_final_checkpoint.json')
        self.assertEqual(full, terminal)
        self.assertFalse(list((root / 'terminal_resume').glob('accepted_cycle_*.json')))
        self.assertNotIn('[native_hybrid] phase=', (root / 'terminal_resume/solver.log').read_text())
        for run in ('uninterrupted', 'resumed', 'terminal_resume'):
            result = load(run, 'never_clamped_result.json')
            vtp = root / run / result['final_vtp']
            self.assertTrue(vtp.is_file())
            self.assertEqual(vtp.read_bytes(), (root / 'uninterrupted/never_clamped_final.vtp').read_bytes())
        rejected = (root / 'clamped_resume_rejected.log').read_text()
        self.assertIn('Resume build/recipe identity mismatch', rejected)
        self.assertFalse((root / 'invalid_clamped_resume').exists())

    def test_run9_archived_command_input_alias(self):
        case = ROOT / 'run/run9_transfer_expanded/evaluations/attempt_0209/case_00'
        cmd = json.loads((case / 'command.json').read_text())
        recorded = Path(cmd[cmd.index('-cycle_file') + 1])
        # The archived command points to attempt_9901; its content must agree.
        self.assertEqual(recorded.read_bytes(), (case / 'sequence.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
