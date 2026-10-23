"""Acceptance fixtures: no solver execution."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dome_bo'))
from run_dome_continuation import verify

class ContinuationTest(unittest.TestCase):
    def fixture(self,path,partial=False):
        proof={'accepted':True,'termination':'gradient_target','curvature':'positive_pivots',
               'gradient_norm':1e-15,'gradient_tolerance':5e-14}
        result={'completed':True,'cycles':1,'equilibrium_solves':1,'solve_mode':'final_only',
                'solves':[{'cycle':1,'native_hybrid':proof}],'final_mesh':'final.vtp'}
        (path/'sequence_result.json').write_text(json.dumps(result))
        (path/'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps({'proof':proof}))
        (path/'final.vtp').touch()
        (path/'sequence_history.csv').write_text('cycle,solved,hit_events\n1,1,1\n')
        failed=dict(proof,accepted=False,termination='time_cap')
        proofs=[failed,proof,proof]
        (path/'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in proofs))
        rows='executed_cycle_index,lambda_from,lambda_trial,equilibrium_accepted\n1,0,1,0\n1,0,0.5,1\n1,0.5,'+('0.9' if partial else '1')+',1\n'
        (path/'sequence_convergence.csv').write_text(rows)
    def test_rollback_and_exact_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.fixture(path)
            self.assertTrue(verify(path,1)['completed'])
            self.fixture(path,partial=True)
            with self.assertRaisesRegex(ValueError,'Partial-load'):verify(path,1)
    def test_proof_row_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.fixture(path)
            p=path/'sequence_convergence.csv'
            p.write_text(p.read_text().replace('1,0,1,0','1,0,1,1'))
            with self.assertRaisesRegex(ValueError,'load/proof'):verify(path,1)

if __name__=='__main__':unittest.main()
