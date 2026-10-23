"""Frozen Run 6 optimum (attempt 209) -> one forward evaluation on a transfer sheet (run7 or run10). No re-fitting.
Usage: run_transfer.py run7|run10"""
import json, sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
RUN = sys.argv[1]
ATTEMPT = {'run7': 9901, 'run10': 9902}[RUN]
sys.path.insert(0, str(BASE))
from calibration.ewcal.common import Parameters, save_json
from calibration.ewcal.runner import Evaluator
best = json.loads((BASE/'run/run6_expanded_bo_continue20_recovery/bayes_summary.json').read_text())['best']
cfg = json.loads((BASE/f'calibration/configs/{RUN}.json').read_text())
out = BASE/f'run/{RUN}_transfer'
if (out/f'evaluations/attempt_{ATTEMPT}').exists(): raise FileExistsError('transfer already run')
x = Parameters(cfg).encode(best['parameters']).tolist()
rec = Evaluator(cfg).evaluate(x, use_cache=False, attempt_id=ATTEMPT)
if not rec.get('valid'): raise RuntimeError(f"{rec.get('failure_class')}: {rec.get('error')}")
m = rec['cases'][0]['metrics']
save_json(out/f'{RUN}_transfer_summary.json', {'run6_attempt': 209, 'run6_objective_mm': 1.182313603689339,
          'parameters': best['parameters'], 'metrics': m})
print({k: m[k] for k in ('rms_symmetric_mm','rms_target_to_sim_mm','rms_sim_to_target_weighted_mm','p95_target_to_sim_mm','max_target_to_sim_mm')})
