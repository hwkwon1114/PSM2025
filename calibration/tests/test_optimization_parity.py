"""Unit tests for the four-tier optimization result equivalence verifier."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from calibration.analysis.optimization_parity import (
    CLASS_THRESHOLDS,
    compare_evaluation_records,
    compare_manifests_and_proofs,
    compare_vtp_meshes,
    kabsch_align_3d,
    load_proof_jsonl,
    summarize_proof_timings,
    validate_native_proof_certificate,
    verify_case_directories,
    verify_multi_point_suite,
)

ROOT = Path(__file__).resolve().parents[2]


class OptimizationParityTests(unittest.TestCase):
    def setUp(self):
        self.a0055_eval_path = ROOT / 'run/run1_clean_24lhs/candidates/a0055/evaluations.jsonl'
        self.a0055_case_dir = (
            ROOT / 'run/run1_clean_24lhs/candidates/a0055/evaluations/20261101-213248-b49cffef86/case_00'
        )
        self.a0082_eval_path = ROOT / 'run/run2_clean_24lhs/candidates/a0082/evaluations.jsonl'

    def test_class_a1_bit_exact_gate_on_production_a0055(self):
        """Baseline vs itself passes Class A1 bit-exact gate; any count or ULP drift fails A1."""
        rec = json.loads(self.a0055_eval_path.read_text().splitlines()[-1])
        rep = compare_evaluation_records(rec, copy.deepcopy(rec), opt_class='A1')
        self.assertTrue(rep['passed'], rep['errors'])
        self.assertEqual(rep['abs_objective_diff_mm'], 0.0)
        self.assertEqual(rep['manifest_report']['abs_energy_diff'], 0.0)
        self.assertEqual(rep['manifest_report']['abs_gradient_norm_diff'], 0.0)

        # Even a 1-step difference in warmup_attempts or evaluations must fail Class A1
        mutated = copy.deepcopy(rec)
        mutated['cases'][0]['manifest']['solves'][0]['native_hybrid']['evaluations'] -= 1
        rep_fail = compare_evaluation_records(rec, mutated, opt_class='A1')
        self.assertFalse(rep_fail['passed'])
        self.assertTrue(any('evaluations' in e for e in rep_fail['errors']))

    def test_proof_timing_summary_on_a0055_captures_cutback_waste(self):
        """Proof JSONL summary accurately quantifies L-BFGS, Newton, and cutback waste."""
        proof_path = self.a0055_case_dir / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl'
        rows = load_proof_jsonl(proof_path)
        summary = summarize_proof_timings(rows)
        self.assertEqual(summary['substeps'], 5)
        self.assertEqual(summary['accepted_substeps'], 3)
        self.assertEqual(summary['rejected_substeps'], 2)
        self.assertEqual(summary['total_attempts'], 10028)
        self.assertAlmostEqual(summary['lbfgs_seconds'], 806.3311018366367, places=6)
        self.assertAlmostEqual(summary['newton_seconds'], 99.27637815661728, places=6)
        self.assertAlmostEqual(summary['wasted_cutback_seconds'], 371.139486476779, places=6)

    def test_class_a2_accepts_machine_precision_reassociation_and_rejects_drift(self):
        """Class A2 allows ~1e-13 energy/mesh rounding from fused loops but rejects 1e-5 mm drift."""
        rec = json.loads(self.a0055_eval_path.read_text().splitlines()[-1])
        cand = copy.deepcopy(rec)
        nh = cand['cases'][0]['manifest']['solves'][0]['native_hybrid']
        nh['energy'] *= (1.0 + 1e-13)
        nh['gradient_norm'] = 9.1e-15
        nh['attempts'] = 2005  # within +/- 2% of 2006
        cand['objective_mm'] += 2e-10

        rep_ok = compare_evaluation_records(rec, cand, opt_class='A2')
        self.assertTrue(rep_ok['passed'], rep_ok['errors'])

        # Same candidate must fail Class A1 (which requires strict 0.0 diff)
        rep_a1 = compare_evaluation_records(rec, cand, opt_class='A1')
        self.assertFalse(rep_a1['passed'])

        # Excessive objective drift (1e-5 mm > 1e-9 mm) must fail Class A2
        bad_cand = copy.deepcopy(cand)
        bad_cand['objective_mm'] += 1e-5
        rep_bad = compare_evaluation_records(rec, bad_cand, opt_class='A2')
        self.assertFalse(rep_bad['passed'])

    def test_kabsch_rigid_body_invariance_and_nonrigid_detection(self):
        """6-DOF Kabsch alignment removes arbitrary free-shell rigid rotation/translation."""
        # Non-coplanar 3D curved shell sample points (in meters)
        ref_pts = [
            (-0.10, -0.08, 0.0012),
            (0.10, -0.08, 0.0015),
            (0.10, 0.12, -0.0021),
            (-0.10, 0.12, -0.0018),
            (0.00, 0.02, 0.0085),
            (0.04, -0.03, 0.0042),
        ]
        # Apply an exact 3D rotation around Z (0.05 rad) and X (-0.03 rad) + translation
        tz, tx = 0.05, -0.03
        cz, sz = math.cos(tz), math.sin(tz)
        cx, sx = math.cos(tx), math.sin(tx)

        def transform(pt):
            x, y, z = pt
            # Rotate around Z
            x1 = cz * x - sz * y
            y1 = sz * x + cz * y
            z1 = z
            # Rotate around X and translate
            x2 = x1 + 0.0005
            y2 = cx * y1 - sx * z1 - 0.0003
            z2 = sx * y1 + cx * z1 + 0.0011
            return (x2, y2, z2)

        rotated_pts = [transform(p) for p in ref_pts]
        metrics = kabsch_align_3d(ref_pts, rotated_pts, scale_to_mm=1000.0)
        self.assertGreater(metrics['raw_max_vertex_diff_mm'], 1.0)
        self.assertLess(metrics['aligned_max_vertex_diff_mm'], 1e-12)
        self.assertLess(metrics['aligned_rms_vertex_diff_mm'], 1e-12)

        # Now add a non-rigid 0.02 mm (2e-5 m) local deflection to vertex 4
        deformed_pts = list(rotated_pts)
        deformed_pts[4] = (deformed_pts[4][0], deformed_pts[4][1], deformed_pts[4][2] + 2e-5)
        def_metrics = kabsch_align_3d(ref_pts, deformed_pts, scale_to_mm=1000.0)
        self.assertGreater(def_metrics['aligned_max_vertex_diff_mm'], 0.01)

    def test_class_b_algorithmic_gate_and_certificate_enforcement(self):
        """Class B accepts faster adaptive/continuation trajectories to the same minimum."""
        rec = json.loads(self.a0055_eval_path.read_text().splitlines()[-1])
        fast_cand = copy.deepcopy(rec)
        nh = fast_cand['cases'][0]['manifest']['solves'][0]['native_hybrid']
        # Simulate adaptive handoff + strain-jump step sizing: 350 warmup steps instead of 2000
        nh['warmup_attempts'] = 350
        nh['attempts'] = 356
        nh['evaluations'] = 372
        nh['lbfgs_seconds'] = 28.0
        nh['newton_seconds'] = 12.0
        nh['energy'] *= (1.0 + 2e-12)
        fast_cand['objective_mm'] += 1e-7
        fast_cand['seconds'] = 240.0

        rep = compare_evaluation_records(rec, fast_cand, opt_class='B')
        self.assertTrue(rep['passed'], rep['errors'])
        self.assertGreater(rep['wall_speedup_x'], 4.0)

        # Corrupting the curvature certificate or verification Hessian must fail Class B
        bad_cert = copy.deepcopy(fast_cand)
        bad_cert['cases'][0]['manifest']['solves'][0]['native_hybrid']['curvature'] = 'indefinite'
        self.assertFalse(compare_evaluation_records(rec, bad_cert, opt_class='B')['passed'])

        no_verif = copy.deepcopy(fast_cand)
        no_verif['cases'][0]['manifest']['solves'][0]['native_hybrid']['verification_hessians'] = 0
        self.assertFalse(compare_evaluation_records(rec, no_verif, opt_class='B')['passed'])

    def test_tier4_multi_point_suite_and_rank_order_gate(self):
        """Tier 4 suite checks valid points, failure preservation, and rank order."""
        r1 = json.loads(self.a0055_eval_path.read_text().splitlines()[-1])
        r2 = json.loads(self.a0082_eval_path.read_text().splitlines()[-1])
        failed_solver = {
            'valid': False,
            'failure_class': 'solver',
            'objective_mm': None,
            'seconds': 400.0,
        }

        base_suite = [r1, r2, failed_solver]
        cand_suite = copy.deepcopy(base_suite)
        cand_suite[0]['objective_mm'] += 2e-7
        cand_suite[1]['objective_mm'] -= 1e-7

        suite_ok = verify_multi_point_suite(base_suite, cand_suite, opt_class='B')
        self.assertTrue(suite_ok['passed'], suite_ok['errors'])
        self.assertEqual(suite_ok['rank_swaps'], 0)

        # Falsely accepting a known solver failure must be rejected
        bad_failure_suite = copy.deepcopy(cand_suite)
        bad_failure_suite[2] = copy.deepcopy(r2)
        suite_fail = verify_multi_point_suite(base_suite, bad_failure_suite, opt_class='B')
        self.assertFalse(suite_fail['passed'])

    def test_case_directory_and_vtp_mesh_verification_on_a0055(self):
        """Full case_00 directory verification checks manifest, proof JSONL, and 3D VTP mesh."""
        if not self.a0055_case_dir.is_dir():
            self.skipTest("a0055 case_00 directory not present")
        rep = verify_case_directories(self.a0055_case_dir, self.a0055_case_dir, opt_class='A1')
        self.assertTrue(rep['passed'], rep['errors'])
        self.assertEqual(rep['mesh_parity']['vertex_count'], 15251)
        self.assertEqual(rep['mesh_parity']['face_count'], 30000)
        self.assertEqual(rep['mesh_parity']['raw_max_vertex_diff_mm'], 0.0)
        self.assertEqual(rep['mesh_parity']['aligned_max_vertex_diff_mm'], 0.0)
        self.assertEqual(rep['mesh_parity']['max_mean_diff'], 0.0)
        self.assertEqual(rep['mesh_parity']['max_gauss_diff'], 0.0)


if __name__ == '__main__':
    unittest.main()
