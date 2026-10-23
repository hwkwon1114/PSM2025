"""Endpoint checker regressions using retained correctness fixture, no solver."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run_dome', ROOT/'legacy/scripts/run_dome.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
FIXTURE = ROOT/'run/dome/fixture_three_paths_v1'


class RunnerTest(unittest.TestCase):
    def test_retained_fixture(self):
        self.assertTrue(m.verify(FIXTURE, 3)['completed'])

    def test_reject_corrupt_evidence(self):
        source = json.loads((FIXTURE/'sequence_result.json').read_text())
        for fault in ('cycles', 'acceptance', 'history', 'state'):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                result = copy.deepcopy(source)
                state = json.loads((FIXTURE/'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
                history = (FIXTURE/'sequence_history.csv').read_text()
                if fault == 'cycles':
                    result['cycles'] = 2
                elif fault == 'acceptance':
                    result['solves'][0]['native_hybrid']['accepted'] = False
                elif fault == 'history':
                    history = history.replace('1,0,', '1,1,')
                else:
                    state['proof']['energy'] = -1
                (out/'sequence_result.json').write_text(json.dumps(result))
                (out/'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(json.dumps(state))
                (out/'sequence_history.csv').write_text(history)
                (out/result['final_mesh']).touch()
                with self.assertRaises(ValueError):
                    m.verify(out, 3)


if __name__ == '__main__':
    unittest.main()
