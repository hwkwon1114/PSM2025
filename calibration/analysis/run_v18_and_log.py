import sys
from pathlib import Path
base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import unittest

loader = unittest.TestLoader()
suite = loader.loadTestsFromName("calibration.tests.test_batch_v16")

log_path = base / "run/optimization_review/test_suite_v18_final.log"
with open(log_path, "w") as f:
    runner = unittest.TextTestRunner(stream=f, verbosity=2)
    result = runner.run(suite)

print(f"Ran {result.testsRun} tests. Errors: {len(result.errors)}, Failures: {len(result.failures)}")
sys.exit(0 if result.wasSuccessful() else 1)
