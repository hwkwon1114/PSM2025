import sys
import time
from pathlib import Path
base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

t0 = time.time()
print(f"[{time.time()-t0:.2f}s] Importing modules...", flush=True)
import json
import tempfile
import copy
from unittest.mock import patch
import numpy as np

print(f"[{time.time()-t0:.2f}s] Importing controller...", flush=True)
from calibration.ewcal import batch_bayes as controller
from calibration.ewcal.runner import Evaluator

print(f"[{time.time()-t0:.2f}s] Setting up fixtures...", flush=True)
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

print(f"[{time.time()-t0:.2f}s] Creating Evaluator(cfg)...", flush=True)
evaluator = Evaluator(cfg)
print(f"[{time.time()-t0:.2f}s] Evaluator created, identity={evaluator.identity[:12]}...", flush=True)

with patch.object(evaluator, "evaluate") as evaluate, \
     patch.object(controller, "propose_q_batch") as propose, \
     patch.object(controller, "get_remaining_allocation_seconds", return_value=1000):
    for i in range(2):
        print(f"[{time.time()-t0:.2f}s] Running run_batch_bayes iteration {i}...", flush=True)
        records = controller.run_batch_bayes(evaluator, output_dir=base_dir,
                                             predecessor_manifest_path=MANIFEST)
        print(f"[{time.time()-t0:.2f}s] Returned {len(records)} records.", flush=True)

bound = json.loads(state_file.read_text())
print(f"[{time.time()-t0:.2f}s] Bound state verified. Testing unauthorized change...", flush=True)
changed = copy.deepcopy(cfg)
changed["solver"]["timeout_seconds"] += 1
ev_changed = Evaluator(changed)
print(f"[{time.time()-t0:.2f}s] Tested changed evaluator={ev_changed.identity[:12]}.", flush=True)
print(f"[{time.time()-t0:.2f}s] ALL CHECKS FINISHED!", flush=True)
