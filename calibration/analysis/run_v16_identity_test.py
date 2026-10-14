import sys
from pathlib import Path

base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import unittest
from calibration.tests.test_batch_v16 import TestV16Recovery

suite = unittest.TestSuite()
suite.addTest(TestV16Recovery('test_real_identity_import_and_bound_restart'))
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
