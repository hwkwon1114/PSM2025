import sys
from pathlib import Path
base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

print("1. importing modules...", flush=True)
import json
import tempfile
import copy
from unittest.mock import patch
import numpy as np

print("2. importing controller...", flush=True)
from calibration.ewcal import batch_bayes as controller
from calibration.ewcal.runner import Evaluator

print("3. loading config...", flush=True)
ROOT = base
FIXTURES = ROOT / "run/optimization_review/v15_fixtures"
MANIFEST = FIXTURES / "predecessor_manifest.json"

cfg = json.loads((ROOT / "calibration/configs/run6_stage2_bayesian_optimization.json").read_text())
tmp = tempfile.TemporaryDirectory()
base_dir = Path(tmp.name)
cfg["output_dir"] = str(base_dir)
state_file = base_dir / "bayes_state.json"
original = (FIXTURES / "predecessor_state.json").read_bytes()
state_file.write_bytes(original)

print("4. initializing Evaluator(cfg)...", flush=True)
evaluator = Evaluator(cfg)
print("5. Evaluator initialized successfully!", flush=True)

historical = json.loads(original)["observations"][11]["identity"]
print(f"Evaluator identity: {evaluator.identity}")
print(f"Historical identity: {historical}")

print("6. running batch_bayes restart...", flush=True)
with patch.object(evaluator, "evaluate") as evaluate, \
     patch.object(controller, "propose_q_batch") as propose, \
     patch.object(controller, "get_remaining_allocation_seconds", return_value=1000):
    try:
        records = controller.run_batch_bayes(evaluator, output_dir=base_dir,
                                             predecessor_manifest_path=MANIFEST)
        print(f"7. Result records count: {len(records)}", flush=True)
    except Exception as e:
        print(f"7. Exception raised: {type(e).__name__}: {e}", flush=True)

print("Done!", flush=True)
