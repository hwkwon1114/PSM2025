"""Launch-wiring checks only; no solver execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'legacy/scripts'))
import run_sequence_order_continuation as runner


class ContinuationTests(unittest.TestCase):
    def test_declared_changes_only(self):
        original = json.loads((runner.SOURCE / 'execution.json').read_text())['options']
        changed = runner.continuation_options(original)
        keys = set(original) | set(changed)
        self.assertEqual({k for k in keys if original.get(k) != changed.get(k)}, {
            '-sequence_adaptive', '-sequence_initial_step', '-sequence_max_step',
            '-sequence_min_step', '-sequence_step_growth', '-sequence_max_retries', '-native_hybrid_seconds'})
        self.assertEqual(changed['-native_hybrid_handoff'], 'fixed')
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, SLURM_JOB_ID='mock', SLURM_CPUS_PER_TASK='8'), \
                patch.object(runner.subprocess, 'run') as launch, \
                patch.object(runner, 'verify', return_value={'solves': [{'native_hybrid': {'mock': True}}]}) as verify:
            out = Path(directory) / 'run'
            runner.run(out)
            self.assertEqual((out / 'sequence.json').read_bytes(), (runner.SOURCE / 'sequence.json').read_bytes())
            self.assertEqual(launch.call_args.kwargs['timeout'], 7200)
            self.assertEqual(verify.call_args.args[0], out)
            self.assertEqual(len(verify.call_args.args[1]['toolpaths']), 20)
            self.assertEqual(verify.call_args.args[2]['-sequence_solve_mode'], 'final_only')
            with self.assertRaises(FileExistsError):
                runner.run(out)
            self.assertEqual(launch.call_count, 1)

    def test_timeout_is_not_acceptance(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, SLURM_JOB_ID='mock', SLURM_CPUS_PER_TASK='8'), \
                patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('mock', 3600)), \
                patch.object(runner, 'verify') as verify:
            out = Path(directory) / 'failed'
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run(out)
            self.assertEqual(json.loads((out / 'execution.json').read_text())['status'], 'failed')
            verify.assert_not_called()
