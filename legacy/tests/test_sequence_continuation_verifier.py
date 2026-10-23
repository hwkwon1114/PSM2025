"""Synthetic stage/rollback records, not numerical equilibrium evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'legacy/scripts'))
from verify_sequence_continuation import verify


class StageVerifierTests(unittest.TestCase):
    def fixture(self, out):
        proof = dict(accepted=True, termination='gradient_target', curvature='positive_pivots',
                     gradient_norm=1e-15, gradient_tolerance=5e-14)
        failed = dict(proof, accepted=False, termination='time_cap')
        result = dict(completed=True, cycles=2, equilibrium_solves=2, solve_mode='every_cycle',
                      solves=[dict(cycle=i, native_hybrid=proof) for i in (1, 2)], final_mesh='final.vtp')
        (out/'sequence_result.json').write_text(json.dumps(result))
        (out/'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps({'proof':proof}))
        (out/'final.vtp').touch()
        (out/'sequence_history.csv').write_text('cycle,solved,hit_events\n1,1,2\n2,1,2\n')
        (out/'bilayer_zigzag_sequence_summary.csv').write_text(
            'executed_cycle_index,toolpath_id,final_file\n1,A,final.vtp\n2,B,final.vtp\n')
        (out/'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').write_text(
            '\n'.join(json.dumps(p) for p in [failed,proof,proof,proof]))
        (out/'sequence_convergence.csv').write_text(
            'executed_cycle_index,lambda_from,lambda_trial,equilibrium_accepted\n'
            '1,0,1,0\n1,0,0.5,1\n1,0.5,1,1\n2,0,1,1\n')
        return {'toolpaths':[{'id':'A'},{'id':'B'}]}, {'-sequence_solve_mode':'every_cycle','-native_hybrid_grad_tol':'5e-14'}

    def test_stage_reset_and_rejection_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); config,opts=self.fixture(out)
            self.assertTrue(verify(out,config,opts)['completed'])

    def test_bad_stage_partial_and_rollback_are_rejected(self):
        for before,after in [('2,0,1,1','2,0,0.9,1'),('2,0,1,1','1,0,1,1'),
                             ('1,0,0.5,1','1,1,0.5,1')]:
            with self.subTest(after=after), tempfile.TemporaryDirectory() as folder:
                out=Path(folder);config,opts=self.fixture(out)
                p=out/'sequence_convergence.csv';p.write_text(p.read_text().replace(before,after))
                with self.assertRaises(ValueError):verify(out,config,opts)

    def test_retained_numerical_fixture_when_available(self):
        root = Path(__file__).resolve().parents[2] / 'run/sequence_order/continuation_fixtures_v2'
        if not root.exists():
            self.skipTest('Retained numerical fixture unavailable')
        states = []
        for mode in ('every_cycle', 'final_only'):
            out = root / f'ab_alternate_{mode}_base'
            execution = json.loads((out/'execution.json').read_text())
            self.assertEqual(execution['status'], 'accepted')
            verify(out, json.loads((out/'sequence.json').read_text()), execution['options'])
            states.append(json.loads((out/'bilayer_zigzag_sequence_native_hybrid_state.json').read_text()))
        for key in ('abar_top','abar_bottom','reference_positions','bbar','material'):
            self.assertEqual(states[0][key], states[1][key])

    def test_unqualified_intermediate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);config,opts=self.fixture(out)
            p=out/'bilayer_zigzag_sequence_native_hybrid_proof.jsonl'
            proofs=[json.loads(x) for x in p.read_text().splitlines()]
            proofs[1]['curvature']='indefinite'
            p.write_text('\n'.join(json.dumps(x) for x in proofs))
            with self.assertRaisesRegex(ValueError,'Unqualified'):verify(out,config,opts)
