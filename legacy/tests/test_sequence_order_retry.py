"""Mocked launch wiring only; no numerical simulation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'legacy/scripts'))
import retry_sequence_order_lumped as retry


class RetryTests(unittest.TestCase):
    def test_only_runtime_changes_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'retry'
            with patch.dict(os.environ, SLURM_JOB_ID='mock', SLURM_CPUS_PER_TASK='8'), \
                    patch.object(retry.subprocess, 'run') as launch, patch.object(retry, 'verify') as verify:
                retry.retry(out)
                old = json.loads((retry.ROOT / 'run/sequence_order/ab_block_final_only_base/execution.json').read_text())
                new = json.loads((out / 'execution.json').read_text())
                differences = {k for k in new['options'] if new['options'][k] != old['options'][k]}
                self.assertEqual(differences, {'-native_hybrid_seconds'})
                self.assertEqual(new['options']['-native_hybrid_seconds'], '1800')
                self.assertEqual(new['sequence_sha256'], retry.sha(out / 'sequence.json'))
                self.assertEqual(launch.call_args.kwargs['timeout'], 2070)
                self.assertEqual(new['status'], 'accepted')
                verify.assert_called_once()
                with self.assertRaises(FileExistsError):
                    retry.retry(out)
                self.assertEqual(launch.call_count, 1)

    def test_failure_remains_unaccepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'failed'
            with patch.dict(os.environ, SLURM_JOB_ID='mock', SLURM_CPUS_PER_TASK='8'), \
                    patch.object(retry.subprocess, 'run', side_effect=subprocess.TimeoutExpired('mock', 2070)), \
                    patch.object(retry, 'verify') as verify:
                with self.assertRaises(subprocess.TimeoutExpired):
                    retry.retry(out)
                self.assertEqual(json.loads((out / 'execution.json').read_text())['status'], 'failed')
                verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
