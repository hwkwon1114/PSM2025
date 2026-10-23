"""Real CLI solver integration tests verifying C++/Python interoperability, pending cycles, native hybrid mode, and legacy rejection."""
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from calibration.ewcal.common import ROOT, load_config
from calibration.ewcal.runner import check_output


class RealSolverCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = ROOT / "bin" / "shell"
        if not cls.solver.is_file():
            raise unittest.SkipTest("bin/shell not built")

    def test_cli_pending_cycles(self):
        """Verify that pending cycles (-sequence_minimize_every 2) do not abort with NaN energy."""
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            seq_path = tmp_path / "seq.json"
            seq_data = json.loads((ROOT / "shellsim/examples/trajectories/01_putong_1step.json").read_text())
            seq_data["toolpaths"][0]["repeat"] = 2
            seq_path.write_text(json.dumps(seq_data))

            cmd = [
                str(self.solver),
                "-sim", "bilayer_growth",
                "-case", "custom",
                "-geometry", "rectangle",
                "-lx", "0.1", "-ly", "0.15", "-res", "0.05",
                "-growth_type", "zigzag_sequence",
                "-cycle_file", str(seq_path),
                "-sequence_minimize_every", "2",
                "-sequence_adaptive", "false",
                "-sequence_solve_mode", "every_cycle",
            ]
            proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Solver stdout: {proc.stdout}\nstderr: {proc.stderr}")

            # Check history
            history_file = tmp_path / "sequence_history.csv"
            self.assertTrue(history_file.is_file())
            with history_file.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["solved"], "0")  # Cycle 1 was pending
            self.assertEqual(rows[1]["solved"], "1")  # Cycle 2 was solved

            # Check result manifest
            result_file = tmp_path / "sequence_result.json"
            self.assertTrue(result_file.is_file())
            result = json.loads(result_file.read_text())
            self.assertTrue(result["completed"])
            self.assertEqual(result["cycles"], 2)
            self.assertEqual(result["equilibrium_solves"], 1)

    def test_cli_final_only_mode(self):
        """Verify sequence_solve_mode final_only runs exactly 1 solve and passes check_output."""
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            seq_path = tmp_path / "seq.json"
            seq_data = json.loads((ROOT / "shellsim/examples/trajectories/01_putong_1step.json").read_text())
            seq_data["toolpaths"][0]["repeat"] = 2
            seq_path.write_text(json.dumps(seq_data))

            cmd = [
                str(self.solver),
                "-sim", "bilayer_growth",
                "-case", "custom",
                "-geometry", "rectangle",
                "-lx", "0.1", "-ly", "0.15", "-res", "0.05",
                "-growth_type", "zigzag_sequence",
                "-cycle_file", str(seq_path),
                "-equilibrium_solver", "native_hybrid",
                "-sequence_adaptive", "false",
                "-sequence_solve_mode", "final_only",
            ]
            proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Solver stdout: {proc.stdout}\nstderr: {proc.stderr}")

            config = load_config(ROOT / "calibration/configs/run6.json")
            config["solver"]["solve_mode"] = "final_only"
            config["solver"]["args"]["-equilibrium_solver"] = "native_hybrid"
            case = config["cases"][0]
            case["repeat"] = 2

            manifest, mesh = check_output(config, case, tmp_path)
            self.assertEqual(manifest["equilibrium_solves"], 1)
            self.assertEqual(manifest["backend"], "native_hybrid")
            self.assertGreater(mesh.GetNumberOfPoints(), 0)

    def test_cli_native_hybrid_mode(self):
        """Verify that production native_hybrid equilibrium solver initializes and runs cleanly."""
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            seq_path = tmp_path / "seq.json"
            seq_data = json.loads((ROOT / "shellsim/examples/trajectories/01_putong_1step.json").read_text())
            seq_data["toolpaths"][0]["repeat"] = 2
            seq_path.write_text(json.dumps(seq_data))

            cmd = [
                str(self.solver),
                "-sim", "bilayer_growth",
                "-case", "custom",
                "-geometry", "rectangle",
                "-lx", "0.1", "-ly", "0.15", "-res", "0.05",
                "-growth_type", "zigzag_sequence",
                "-cycle_file", str(seq_path),
                "-equilibrium_solver", "native_hybrid",
                "-sequence_adaptive", "false",
                "-sequence_solve_mode", "final_only",
            ]
            proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Solver stdout: {proc.stdout}\nstderr: {proc.stderr}")

            result_file = tmp_path / "sequence_result.json"
            self.assertTrue(result_file.is_file())
            result = json.loads(result_file.read_text())
            self.assertEqual(result["backend"], "native_hybrid")

    def test_cli_legacy_solver_rejected(self):
        """Verify that attempting to invoke removed legacy solvers is explicitly rejected."""
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            seq_path = tmp_path / "seq.json"
            seq_data = json.loads((ROOT / "shellsim/examples/trajectories/01_putong_1step.json").read_text())
            seq_data["toolpaths"][0]["repeat"] = 2
            seq_path.write_text(json.dumps(seq_data))

            for legacy in ["hlbfgs", "trust_region", "hybrid"]:
                cmd = [
                    str(self.solver),
                    "-sim", "bilayer_growth",
                    "-case", "custom",
                    "-geometry", "rectangle",
                    "-lx", "0.1", "-ly", "0.15", "-res", "0.05",
                    "-growth_type", "zigzag_sequence",
                    "-cycle_file", str(seq_path),
                    "-equilibrium_solver", legacy,
                ]
                proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("has been removed", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
