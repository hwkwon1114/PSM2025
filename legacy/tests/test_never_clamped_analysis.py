"""Read-only analysis checks against saved coarse fixtures; no simulation."""
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/analysis'))
from analyze_never_clamped import accepted, compare


class FormingAnalysisTests(unittest.TestCase):
    def test_saved_coarse_comparison(self):
        root = ROOT / 'run/two_gripper_comparison/fixtures/never_clamped_v2'
        metrics, arrays = compare(root / 'uninterrupted', root / 'paired_regression', 'fixture')
        self.assertEqual(metrics['cycles'], 2)
        self.assertTrue(all(metrics['reference_and_target_history_exact_matches'].values()))
        self.assertEqual(len(metrics['accepted_repeats']), 2)
        self.assertEqual(arrays['forming_delta_mm'].shape, arrays['free_mm'].shape)
        self.assertGreaterEqual(metrics['maximum_difference_mm'], metrics['area_weighted_rms_difference_mm'])
        with self.assertRaisesRegex(ValueError, 'predecessor-chain'):
            compare(root / 'resumed', root / 'paired_regression', 'fixture')

    def test_rejects_mismatched_history_and_rigid_modes(self):
        root = ROOT / 'run/two_gripper_comparison/fixtures/never_clamped_v2'
        original_loads = json.loads
        for defect, message in [('history', 'target/history'), ('rigid', 'proof/link/constraints')]:
            def altered(text):
                obj = original_loads(text)
                payload = obj.get('payload', {})
                if defect == 'history' and 'identity' in payload:
                    if not any(any(row) for row in payload['identity']['vertex_mask']):
                        payload['counts'] = []
                if defect == 'rigid' and 'remaining_rigid_modes' in payload:
                    payload['remaining_rigid_modes'] = 5
                return obj
            with self.subTest(defect=defect), patch('analyze_never_clamped.json.loads', side_effect=altered):
                with self.assertRaisesRegex(ValueError, message):
                    compare(root / 'uninterrupted', root / 'paired_regression', 'fixture')

    def test_plot_layout_guard(self):
        from plot_never_clamped import plt, save
        with tempfile.TemporaryDirectory() as folder:
            for defect in ('outside', 'overlap'):
                fig, ax = plt.subplots()
                ax.text(1.5 if defect == 'outside' else .5, .5, 'label', transform=ax.transAxes)
                if defect == 'overlap':
                    ax.text(.5, .5, 'collision', transform=ax.transAxes)
                with self.assertRaises(ValueError):
                    save(fig, Path(folder), defect)
                plt.close(fig)
            fig, ax = plt.subplots()
            ax.text(.2, .5, 'clean', transform=ax.transAxes)
            save(fig, Path(folder), 'clean')
            self.assertTrue((Path(folder) / 'clean.pdf').is_file())

    def test_acceptance_is_fail_closed(self):
        proof = dict(accepted=True, termination='gradient_target', curvature='positive_pivots',
                     energy=0., gradient_norm=1e-15, gradient_tolerance=5e-14)
        self.assertTrue(accepted(proof))
        for key, value in [('accepted', False), ('termination', 'timeout'),
                           ('curvature', 'unknown'), ('energy', float('nan')),
                           ('gradient_norm', float('nan')), ('gradient_norm', 1e-12),
                           ('gradient_tolerance', 1e-12)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(accepted(dict(proof, **{key: value})))


if __name__ == '__main__':
    unittest.main()
