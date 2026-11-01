"""Researcher-requested exploratory Run 2 (historical Run 9) check of a frozen joint-fit candidate.
Report-only: must not feed back into candidate selection. Same Run 9 config as run/convention_corrected/run9
(Attempt 209 reference, 0.678 mm revised metric); only the six parameters differ.
Usage: run_run9_interim.py <campaign_base> <attempt>"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_convention_corrected as cc
from calibration.ewcal.common import save_json
from calibration.ewcal.runner import Evaluator


def main():
    base, att = sys.argv[1], int(sys.argv[2])
    src = json.loads((ROOT / base / f'results/a{att:04d}.json').read_text())
    assert src['valid']
    vals = src['parameters']
    cfg = cc.config('run9')
    cfg['hardening']['beta'], cfg['hardening']['floor'] = float(vals['beta']), float(vals['floor'])
    out = f'run/run9_interim/{Path(base).name}_a{att:04d}'
    if '--long' in sys.argv:   # diagnostic re-run: line-buffered solver log + 3 h cap; numerics unchanged
        cfg['solver']['command_prefix'] = ['stdbuf', '-oL', '-eL']
        cfg['solver']['timeout_seconds'] = 10800
        cfg['solver']['args']['-native_hybrid_seconds'] = 10200  # solver's own cap fires first, so its report + trace are written
        cfg['solver']['args']['-native_hybrid_export_trace'] = 'true'
        out += '_long'
    cfg['output_dir'], cfg['report_dir'] = out, out + '/report'
    if (ROOT / out / 'evaluations').exists():
        raise FileExistsError(out)
    p4 = {k: float(vals[k]) for k in cfg['parameters']}
    ev = Evaluator(cfg)
    ev.parameters = SimpleNamespace(decode=lambda _x: dict(p4))  # exact values; gbot = 0 is outside the log codec
    save_json(ROOT / out / 'source.json', {'campaign': base, 'attempt': att, 'parameters': vals,
                                           'joint_objective_mm': src['objective_mm'], 'case_rms_mm': src['case_rms_mm'],
                                           'note': 'researcher-requested exploratory frozen-parameter check; report-only, not used for selection'})
    rec = ev.evaluate([0.0] * 4, use_cache=False, attempt_id=1)
    save_json(ROOT / out / 'forward_result.json', rec)
    print('valid', rec['valid'], 'run9 rms_mm', rec['objective_mm'], rec.get('error'))


if __name__ == '__main__':
    main()
