import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/scripts'))
from calibration.ewcal.common import load_config
from shellsim.solver_settings import solver_options, typed


class SharedSolverSettings(unittest.TestCase):
    def test_calibration_configs_carry_the_shared_options(self):
        shared = {k: typed(v) for k, v in solver_options().items()}
        for run in ('run6', 'run7', 'run9', 'run10'):
            args = load_config(ROOT / f'calibration/configs/{run}.json')['solver']['args']
            self.assertEqual({k: args[k] for k in shared}, shared, run)

    def test_solver_config_path_is_honored(self):
        import json, tempfile
        cfg = json.loads((ROOT / 'calibration/configs/run6.json').read_text())
        alt = Path(tempfile.mkdtemp()) / 'solver.json'
        alt.write_text(json.dumps({'options': {'-native_hybrid_seconds': '123'}}))
        cfg['solver']['solver_config'] = str(alt)
        run = Path(tempfile.mkdtemp()) / 'run6.json'
        run.write_text(json.dumps(cfg))
        self.assertEqual(load_config(run)['solver']['args']['-native_hybrid_seconds'], 123)

    def test_driver_options_carry_the_shared_options(self):
        import run_overlapping_sequence as overlap
        tight = overlap.options('final_only', 'tight')
        expected = solver_options(tolerance='1e-14', threads=8)
        self.assertEqual({k: tight[k] for k in expected}, expected)


if __name__ == '__main__':
    unittest.main()
