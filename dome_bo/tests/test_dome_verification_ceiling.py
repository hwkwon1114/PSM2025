"""Endpoint and accepted-substep tolerances must satisfy the requested verification ceiling."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_dome_continuation import verify


class VerificationCeilingTest(unittest.TestCase):
    def check_case(self, endpoint_tol, substep_tol, ceiling):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            def proof(tol):
                return dict(accepted=True, termination='gradient_target', curvature='positive_pivots',
                            gradient_norm=1e-15, gradient_tolerance=tol)
            end = proof(endpoint_tol)
            (out / 'mesh.vtp').touch()
            (out / 'sequence_result.json').write_text(json.dumps(dict(
                completed=True, cycles=1, equilibrium_solves=1, solve_mode='final_only',
                solves=[dict(cycle=1, native_hybrid=end)], final_mesh='mesh.vtp')))
            (out / 'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps(dict(proof=end)))
            (out / 'sequence_history.csv').write_text('cycle,solved,hit_events\n1,1,1\n')
            (out / 'sequence_convergence.csv').write_text(
                'executed_cycle_index,lambda_from,lambda_trial,equilibrium_accepted\n1,0,0.5,1\n1,0.5,1,1\n')
            (out / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').write_text(
                json.dumps(proof(substep_tol)) + '\n' + json.dumps(end) + '\n')
            return verify(out, 1, gradient_ceiling=ceiling)

    def test_screening_unchanged(self):
        self.check_case(5e-14, 5e-14, 5e-14)

    def test_strict_pass(self):
        self.check_case(1e-14, 1e-14, 1e-14)

    def test_loose_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unaccepted native endpoint'):
            self.check_case(5e-14, 1e-14, 1e-14)

    def test_loose_substep_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unqualified accepted substep'):
            self.check_case(1e-14, 5e-14, 1e-14)


if __name__ == '__main__':
    unittest.main()
