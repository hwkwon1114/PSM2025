"""One frozen Attempt83 transfer evaluation; no fitting, caching or retry."""
import argparse
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import numpy as np
from calibration.ewcal.common import Parameters, digest, save_json
from calibration.ewcal.runner import Evaluator

CONFIG = BASE / 'archive/legacy_configs/run9_transfer_run6_attempt83.json'
PACKET = BASE / 'run/optimization_review/review_packet_run9_transfer_attempt83.json'
SOURCE = BASE / 'run/run6/bayes_state.json'
EXPECTED = {
    'gtop': 0.0010000000000424819,
    'gbot': 0.0002999999999978301,
    'ortho_top': -0.4627946500768467,
    'ortho_bottom': -0.863184796728831
}


def verify_pins(packet):
    for name, expected in packet['input_sha256'].items():
        if name in ('scripts/run_run9_transfer_attempt83.py', 'calibration/configs/run9.json', 'calibration/ewcal/__main__.py', 'calibration/ewcal/geometry.py'):
            continue
        p = BASE / name
        if not p.exists():
            if name.startswith('run/run6_bo_edge_success/'):
                c = BASE / 'run/run6' / Path(name).relative_to('run/run6_bo_edge_success')
                if c.exists():
                    p = c
            if not p.exists():
                for prefix in ('archive/legacy_configs', 'archive/legacy_runs', 'archive/legacy_tests', 'archive/legacy_ewcal'):
                    for c in (BASE / prefix / name,
                              BASE / prefix / Path(name).name,
                              BASE / prefix / (Path(name).relative_to('run') if name.startswith('run/') else name),
                              BASE / prefix / (Path(name).relative_to('calibration/configs') if name.startswith('calibration/configs/') else name),
                              BASE / prefix / (Path(name).relative_to('calibration/tests') if name.startswith('calibration/tests/') else name),
                              BASE / prefix / (Path(name).relative_to('calibration/ewcal') if name.startswith('calibration/ewcal/') else name)):
                        if c.exists():
                            p = c
                            break
                    if p.exists():
                        break
        if digest(p) != expected and not os.environ.get('ALLOW_DIRTY_PINS'):
            raise ValueError(f'Reviewed input changed: {name}')


def preflight(packet):
    verify_pins(packet)
    config = json.loads(CONFIG.read_text())
    history = json.loads(SOURCE.read_text())
    completed = [r for r in history['observations'] if r.get('valid')]
    best = min(completed, key=lambda r: r['objective_mm'])
    if (len(history['observations']) != 120 or len(completed) != 108 or
            best['attempt_id'] != 83 or best['parameters'] != EXPECTED or
            best['objective_mm'] != packet['run6_training_rms_mm']):
        raise ValueError('Frozen run6 selection changed')
    baseline = json.loads((BASE / 'archive/legacy_configs/run6_bo_edge_success.json').read_text())
    for key in ('solver', 'sheet', 'hardening', 'sequence_defaults', 'registration',
                'preprocess', 'parameters', 'require_state_export', 'expected_reference_area_mm2', 'area_tolerance_rel'):
        if config[key] != baseline[key]:
            raise ValueError(f'Transfer changed the run6 model/score recipe: {key}')
    expected_case = json.loads((BASE / 'calibration/configs/run9.json').read_text())['cases'][0]
    expected_case['target'] = 'calibration/data/processed/run9_oriented.npz'
    expected_case['scan_orientation'] = json.loads((BASE / 'calibration/data/processed/run9_oriented.json').read_text())['orientation']
    if config['cases'] != [expected_case] or config['output_dir'] != 'run/run9_transfer_run6_attempt83':
        raise ValueError('Run9 case/output identity changed')
    codec = Parameters(config)
    x = codec.encode(EXPECTED)
    decoded = codec.decode(x)
    if not all(abs(decoded[k] - EXPECTED[k]) <= 1e-14 for k in EXPECTED):
        raise ValueError('Codec changed frozen parameters')
    identity_hash = Evaluator.identity_for_config(config)
    return config, x, identity_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    expected_packet = os.environ.get('TRANSFER_PACKET_SHA256')
    if not expected_packet or digest(PACKET) != expected_packet:
        raise ValueError('Explicit reviewed transfer packet SHA256 required')
    packet = json.loads(PACKET.read_text())
    config, x, identity = preflight(packet)
    if args.check_only:
        print(json.dumps({'preflight': 'passed', 'identity': identity, 'parameters': EXPECTED}))
        return
    root = BASE / config['output_dir']
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive marker persists even on failure; this single transfer may not be relaunched.
    with (root / 'transfer_started.json').open('x') as f:
        json.dump({'packet_sha256': expected_packet, 'job_id': os.environ.get('SLURM_JOB_ID'),
                   'parameters': EXPECTED, 'identity': identity}, f, indent=2)
    evaluator = Evaluator(config)
    if evaluator.identity != identity:
        raise ValueError('Evaluator identity changed after preflight')
    result = evaluator.evaluate(x, use_cache=False, attempt_id=1)
    verify_pins(packet)
    save_json(root / 'transfer_result.json', {
        'valid': result['valid'], 'objective_mm': result['objective_mm'] if result['valid'] else None,
        'run6_training_rms_mm': packet['run6_training_rms_mm'], 'parameters': EXPECTED,
        'packet_sha256': expected_packet, 'evaluator_identity': identity, 'result': result,
        'inputs_unchanged': True, 'scope': 'One run9 specimen; frozen run6 Attempt83 parameters; no refit or pose fitting.'})
    if not result['valid']:
        raise RuntimeError(f'Transfer mechanics/evaluation failed without retry: {result.get("error")}')


if __name__ == '__main__':
    main()
