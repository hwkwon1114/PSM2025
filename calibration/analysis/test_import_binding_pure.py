import sys
from pathlib import Path
base = Path(__file__).resolve().parents[2]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import json
import hashlib
from calibration.ewcal.batch_integrity import validate_history_binding
from calibration.ewcal.runner import Evaluator

print("Loading config and predecessor state...", flush=True)
FIXTURES = base / "run/optimization_review/v15_fixtures"
MANIFEST = FIXTURES / "predecessor_manifest.json"
cfg = json.loads((base / "calibration/configs/run6_stage2_bayesian_optimization.json").read_text())
original = (FIXTURES / "predecessor_state.json").read_bytes()
state = json.loads(original)

print("Calling Evaluator.identity_inputs...", flush=True)
identity_inputs, _ = Evaluator.identity_inputs(cfg)
identity = hashlib.sha256(json.dumps(identity_inputs, sort_keys=True).encode()).hexdigest()

print("Calling validate_history_binding...", flush=True)
try:
    binding = validate_history_binding(state, identity, identity_inputs, original, MANIFEST)
    print("SUCCESS: binding =", binding, flush=True)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
