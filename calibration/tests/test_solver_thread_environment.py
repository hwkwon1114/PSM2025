"""No-solver checks for production and explicitly controlled BLIS environments."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from calibration.ewcal.runner import launch
from shellsim import simulate


LOOP_VARS = ('BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT')


class SolverThreadEnvironmentTests(unittest.TestCase):
    def test_calibration_solver_default_and_explicit_diagnostic(self):
        code = ('import json, os; print(json.dumps({k: os.environ.get(k) for k in '
                f'{repr(("BLIS_NUM_THREADS", "OMP_NUM_THREADS") + LOOP_VARS)}' + '}))')
        overrides = {'BLIS_NUM_THREADS': '12', 'OMP_NUM_THREADS': '8', **{key: '4' for key in LOOP_VARS}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, overrides):
            out = Path(tmp)
            default = out / 'default'
            default.mkdir()
            launch([sys.executable, '-c', code], default, 10)
            values = json.loads((default / 'solver.log').read_text())
            self.assertEqual(values['BLIS_NUM_THREADS'], '1')
            self.assertEqual(values['OMP_NUM_THREADS'], '8')
            self.assertTrue(all(values[key] is None for key in LOOP_VARS))
            self.assertEqual(os.environ['BLIS_NUM_THREADS'], '12')

            # The reviewed BLIS diagnostic deliberately compares an unset environment.
            explicit = os.environ.copy()
            explicit.pop('BLIS_NUM_THREADS')
            for key in LOOP_VARS:
                explicit.pop(key)
            diagnostic = out / 'diagnostic'
            diagnostic.mkdir()
            launch([sys.executable, '-c', code], diagnostic, 10, solver_environment=explicit)
            values = json.loads((diagnostic / 'solver.log').read_text())
            self.assertIsNone(values['BLIS_NUM_THREADS'])
            self.assertEqual(values['OMP_NUM_THREADS'], '8')
            self.assertTrue(all(values[key] is None for key in LOOP_VARS))

    def test_shellsim_solver_receives_single_blis_thread(self):
        overrides = {'BLIS_NUM_THREADS': '12', **{key: '4' for key in LOOP_VARS}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, overrides):
            root = Path(tmp)
            sequence = root / 'input.json'
            sequence.write_text('{}')
            captured = {}

            def fake_solver(_command, *, cwd, env, **_kwargs):
                captured.update(env)
                (cwd / 'bilayer_zigzag_sequence_native_hybrid_state.json').write_text(
                    json.dumps({'proof': {'accepted': True, 'termination': 'gradient_target'}}))

            with mock.patch.object(simulate.subprocess, 'run', side_effect=fake_solver):
                simulate.run(sequence, root / 'out', threads=8)
            self.assertEqual(captured['BLIS_NUM_THREADS'], '1')
            self.assertEqual(captured['OMP_NUM_THREADS'], '8')
            self.assertTrue(all(key not in captured for key in LOOP_VARS))
            self.assertEqual(os.environ['BLIS_NUM_THREADS'], '12')


if __name__ == '__main__':
    unittest.main()
