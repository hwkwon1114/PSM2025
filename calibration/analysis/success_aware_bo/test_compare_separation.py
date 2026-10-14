import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibration.analysis.success_aware_bo import compare_separation as driver


class DriverTests(unittest.TestCase):
    def test_exclusive_output_and_partial_refusal(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(driver, 'OUT', Path(temp)):
            driver.write_new('started.json', {'partial': True})
            with self.assertRaises(FileExistsError):
                driver.write_new('started.json', {'partial': False})
            with self.assertRaises(FileExistsError):
                driver.run()
            self.assertTrue(json.loads((Path(temp)/'started.json').read_text())['partial'])

    def test_review_packet_drift_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root/'source.py'
            artifact.write_text('original')
            packet = root/'review_packet.json'
            packet.write_text(json.dumps({'artifact_sha256': {'source.py': driver.sha(artifact)}}))
            with patch.object(driver, 'ROOT', root), patch.object(driver, 'PACKET', packet):
                self.assertEqual(len(driver.check_packet()['artifact_sha256']), 1)
                artifact.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    driver.check_packet()


if __name__ == '__main__':
    unittest.main()
