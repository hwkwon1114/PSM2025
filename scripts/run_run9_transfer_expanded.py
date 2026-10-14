#!/usr/bin/env python
"""Transfer evaluation on Run 9 for the optimum found by Run 6 expanded BO.

Enforces frozen-optimum completion gate, fail-closed data pin verification,
deep ledger validation, parameter round-trip consistency, and one-shot execution marker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import numpy as np
from calibration.ewcal.common import Parameters, digest, save_json
from calibration.ewcal.runner import Evaluator
from scripts import run_run6_bo_refinement as controller
from scripts import run_run6_expanded_bo as expanded_bo
from calibration.ewcal import batch_bayes as batch

CONFIG_RUN9_DEFAULT = BASE / 'calibration/configs/run9.json'
SUMMARY_EXPANDED_DEFAULT = BASE / 'run/run6_expanded_bo/bayes_summary.json'
STATE_EXPANDED_DEFAULT = BASE / 'run/run6_expanded_bo/bayes_state.json'
HISTORY_EXPANDED_DEFAULT = BASE / 'run/run6_expanded_bo/corrected_history.json'
FAILURE_EXPANDED_DEFAULT = BASE / 'run/run6_expanded_bo/controller_failure.json'
REVIEW_PACKET_DEFAULT = BASE / 'run/optimization_review/run6_expanded_bo/review_packet.json'
OUT_DIR_DEFAULT = BASE / 'run/run9_transfer_expanded'

BASELINE_ATTEMPT83_RUN9_MM = 2.1951052062325785
BASELINE_ATTEMPT25_RUN9_MM = 2.7562086968032733


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_transfer_pins(packet, base_dir=BASE):
    """Fail closed if any pinned artifact is missing or altered."""
    pins = packet.get('artifact_sha256', {})
    if not pins:
        raise ValueError('Review packet has empty artifact pins')
    for name, expected in pins.items():
        p = (base_dir / name) if not Path(name).is_absolute() else Path(name)
        if not p.exists():
            raise FileNotFoundError(f'Required pinned artifact missing: {name}')
        actual = sha(p)
        if actual != expected:
            raise ValueError(f'Reviewed artifact digest mismatch for {name}: expected {expected}, got {actual}')


def preflight(
    expected_packet_sha=None,
    summary_path=SUMMARY_EXPANDED_DEFAULT,
    state_path=STATE_EXPANDED_DEFAULT,
    history_path=HISTORY_EXPANDED_DEFAULT,
    failure_path=FAILURE_EXPANDED_DEFAULT,
    review_packet_path=REVIEW_PACKET_DEFAULT,
    config_run9_path=CONFIG_RUN9_DEFAULT,
    out_dir=OUT_DIR_DEFAULT,
    base_dir=BASE,
):
    """Validate all prerequisites strictly before creating the one-shot execution marker.
    
    Raises exceptions on any inconsistency or omission without creating artifacts.
    """
    if not expected_packet_sha:
        raise ValueError('Expected packet digest must be provided via --packet-sha256 or TRANSFER_PACKET_SHA256')

    summary_path = Path(summary_path)
    state_path = Path(state_path)
    history_path = Path(history_path)
    failure_path = Path(failure_path)
    review_packet_path = Path(review_packet_path)
    config_run9_path = Path(config_run9_path)
    out_dir = Path(out_dir)

    # Explicitly configure shared controller for expanded BO
    expanded_bo.configure(base=state_path.parent, history=history_path)

    # 1. Reject if controller failure marker exists
    if failure_path.exists():
        raise RuntimeError(f'Expanded BO campaign halted with controller failure at {failure_path}; cannot perform transfer')

    # 2. Check existence of critical campaign artifacts
    for p, name in [
        (review_packet_path, 'Review packet'),
        (summary_path, 'Expanded BO summary'),
        (state_path, 'Expanded BO state'),
        (history_path, 'Expanded BO history'),
        (config_run9_path, 'Run 9 configuration'),
    ]:
        if not p.exists():
            raise FileNotFoundError(f'{name} not found at {p}')

    # 3. Verify review packet digest and all artifact pins
    packet_sha = sha(review_packet_path)
    if packet_sha != expected_packet_sha:
        raise ValueError(f'Review packet SHA mismatch: expected {expected_packet_sha}, got {packet_sha}')

    packet = json.loads(review_packet_path.read_text())
    verify_transfer_pins(packet, base_dir=base_dir)

    # 4. Load history and state, then run complete ledger validation
    history = json.loads(history_path.read_text())
    history_sha = sha(history_path)
    controller.verify_inputs(history)

    state = json.loads(state_path.read_text())
    state_sha = sha(state_path)

    summary = json.loads(summary_path.read_text())
    summary_sha = sha(summary_path)

    # Validate ledger invariants
    next_id = controller.validate_ledger(state, history['evaluator_identity'], 2026, history=history)

    # Terminal completion and accounting checks
    if state.get('status') != 'budget_complete':
        raise ValueError(f'State status is {state.get("status")}; must be budget_complete before transfer')
    if summary.get('status') != 'budget_complete':
        raise ValueError(f'Summary status is {summary.get("status")}; must be budget_complete before transfer')
    if summary.get('new_completed') != 40:
        raise ValueError(f'Summary new_completed is {summary.get("new_completed")}; expected exactly 40')

    observations = state.get('observations', [])
    if len(observations) != 160:
        raise ValueError(f'Observation count is {len(observations)}; expected exactly 160 (120 seed + 40 new)')
    if state.get('charged_attempts') != 160:
        raise ValueError(f'Charged attempts is {state.get("charged_attempts")}; expected 160')
    if len(state.get('completed_attempts', [])) != 160:
        raise ValueError(f'Completed attempts count is {len(state.get("completed_attempts"))}; expected 160')

    # Publication consistency for all new attempts (121..160)
    for att in range(121, 161):
        res = [r for r in state.get('reserved_attempts', []) if r['attempt_id'] == att]
        if not res:
            raise ValueError(f'Reservation missing for attempt {att}')
        res_file = Path(res[0]['result_file'])
        if not res_file.exists():
            raise FileNotFoundError(f'Publication result missing for attempt {att} at {res_file}')

    best_in_state = expanded_bo.select_best(observations)
    if best_in_state is None:
        raise ValueError('No valid observations found in state')
    best_in_summary = summary.get('best', {})

    # Best record consistency between state and summary
    if best_in_state['attempt_id'] != best_in_summary.get('attempt_id'):
        raise ValueError(f'Summary best attempt ({best_in_summary.get("attempt_id")}) does not match ledger best ({best_in_state["attempt_id"]})')
    if not np.isclose(best_in_state['objective_mm'], best_in_summary.get('objective_mm', np.nan), rtol=1e-12, atol=1e-12):
        raise ValueError('Summary best objective does not match ledger best objective')
    if best_in_state.get('parameters') != best_in_summary.get('parameters'):
        raise ValueError('Summary best parameters do not match ledger best parameters')

    parameters = best_in_state['parameters']
    # Validate parameters are finite and contain no NaN/Inf
    for k, v in parameters.items():
        if not np.isfinite(v):
            raise ValueError(f'Parameter {k} is non-finite: {v}')

    # Validate parameter round-trip on Run 6 codec
    codec_run6 = Parameters(history['config'])
    norm_run6 = codec_run6.encode(parameters)
    roundtrip_run6 = codec_run6.decode(norm_run6)
    for k, v in parameters.items():
        if not np.isclose(roundtrip_run6[k], v, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Run 6 parameter round-trip mismatch for {k}: {v} vs {roundtrip_run6[k]}')
    if not np.allclose(best_in_state['normalized_parameters'], norm_run6, rtol=1e-12, atol=1e-12):
        raise ValueError('Run 6 normalized coordinates in state do not match encoded parameters')

    # 5. Run 9 Evaluator and Target Setup preflight
    config_run9 = json.loads(config_run9_path.read_text())
    identity_run9 = Evaluator.identity_for_config(config_run9)

    # Check target scan files exist
    for case in config_run9.get('cases', []):
        for path_key in ['scan', 'target']:
            tp = case.get(path_key)
            if tp:
                p_full = (BASE / tp) if not Path(tp).is_absolute() else Path(tp)
                if not p_full.exists():
                    raise FileNotFoundError(f'Run 9 target scan missing: {p_full}')

    # Validate parameter encoding with Run 9 codec
    codec_run9 = Parameters(config_run9)
    norm_run9 = codec_run9.encode(parameters).tolist()
    roundtrip_run9 = codec_run9.decode(np.array(norm_run9))
    for k, v in parameters.items():
        if not np.isclose(roundtrip_run9[k], v, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Run 9 parameter round-trip mismatch for {k}: {v} vs {roundtrip_run9[k]}')

    # Pre-execution checks for attempt 9901 and one-shot marker
    eval_attempt_9901 = out_dir / 'evaluations' / 'attempt_9901'
    if eval_attempt_9901.exists():
        raise FileExistsError(f'Transfer attempt 9901 publication already exists at {eval_attempt_9901}')

    start_marker = out_dir / 'transfer_started.json'
    if start_marker.exists():
        raise FileExistsError(f'Transfer one-shot marker already exists at {start_marker}')

    return {
        'packet_sha256': packet_sha,
        'summary_sha256': summary_sha,
        'state_sha256': state_sha,
        'history_sha256': history_sha,
        'run6_attempt_id': best_in_state['attempt_id'],
        'run6_objective_mm': best_in_state['objective_mm'],
        'parameters': parameters,
        'normalized_parameters': norm_run9,
        'config_run9': config_run9,
        'out_dir': out_dir,
    }


def evaluate_transfer(
    expected_packet_sha=None,
    summary_path=SUMMARY_EXPANDED_DEFAULT,
    state_path=STATE_EXPANDED_DEFAULT,
    history_path=HISTORY_EXPANDED_DEFAULT,
    failure_path=FAILURE_EXPANDED_DEFAULT,
    review_packet_path=REVIEW_PACKET_DEFAULT,
    config_run9_path=CONFIG_RUN9_DEFAULT,
    out_dir=OUT_DIR_DEFAULT,
    base_dir=BASE,
    evaluator_cls=Evaluator,
):
    # Preflight validation runs FIRST. If any check fails, no marker is created.
    info = preflight(
        expected_packet_sha=expected_packet_sha,
        summary_path=summary_path,
        state_path=state_path,
        history_path=history_path,
        failure_path=failure_path,
        review_packet_path=review_packet_path,
        config_run9_path=config_run9_path,
        out_dir=out_dir,
        base_dir=base_dir,
    )

    out_path = Path(info['out_dir'])
    out_path.mkdir(parents=True, exist_ok=True)

    # One-shot execution marker: created with exclusive 'x', persists on failure.
    start_marker = out_path / 'transfer_started.json'
    with start_marker.open('x') as f:
        json.dump({
            'packet_sha256': info['packet_sha256'],
            'summary_sha256': info['summary_sha256'],
            'state_sha256': info['state_sha256'],
            'run6_attempt_id': info['run6_attempt_id'],
            'run6_objective_mm': info['run6_objective_mm'],
            'parameters': info['parameters'],
            'normalized_parameters': info['normalized_parameters'],
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        }, f, indent=2)

    config = info['config_run9']
    config['output_dir'] = str(out_path.relative_to(base_dir) if out_path.is_relative_to(base_dir) else out_path)
    evaluator = evaluator_cls(config)

    print(f'Evaluating transfer on Run 9 with frozen parameters from Attempt {info["run6_attempt_id"]}: {info["parameters"]}')
    record = evaluator.evaluate(info['normalized_parameters'], use_cache=False, attempt_id=9901)

    if not record.get('valid'):
        raise RuntimeError(f'Transfer evaluation failed: {record.get("failure_class")} - {record.get("error")}')

    metrics = record['cases'][0]['metrics']
    rms_sym = metrics['rms_symmetric_mm']
    print(f'Run 9 Transfer Symmetric RMS: {rms_sym:.6f} mm')

    diff_83 = rms_sym - BASELINE_ATTEMPT83_RUN9_MM
    diff_25 = rms_sym - BASELINE_ATTEMPT25_RUN9_MM

    print(f'  vs Attempt 83 baseline ({BASELINE_ATTEMPT83_RUN9_MM:.4f} mm): {diff_83:+.6f} mm ({diff_83/BASELINE_ATTEMPT83_RUN9_MM*100:+.2f}%)')
    print(f'  vs Attempt 25 baseline ({BASELINE_ATTEMPT25_RUN9_MM:.4f} mm): {diff_25:+.6f} mm ({diff_25/BASELINE_ATTEMPT25_RUN9_MM*100:+.2f}%)')

    summary_transfer = {
        'packet_sha256': info['packet_sha256'],
        'history_sha256': info['history_sha256'],
        'state_sha256': info['state_sha256'],
        'summary_sha256': info['summary_sha256'],
        'run6_best_attempt_id': info['run6_attempt_id'],
        'run6_best_objective_mm': info['run6_objective_mm'],
        'parameters': info['parameters'],
        'normalized_parameters': info['normalized_parameters'],
        'rms_symmetric_mm': rms_sym,
        'metrics': metrics,
        'comparison': {
            'baseline_attempt83_mm': BASELINE_ATTEMPT83_RUN9_MM,
            'delta_vs_attempt83_mm': diff_83,
            'delta_vs_attempt83_rel_pct': diff_83 / BASELINE_ATTEMPT83_RUN9_MM * 100,
            'baseline_attempt25_mm': BASELINE_ATTEMPT25_RUN9_MM,
            'delta_vs_attempt25_mm': diff_25,
            'delta_vs_attempt25_rel_pct': diff_25 / BASELINE_ATTEMPT25_RUN9_MM * 100,
        },
    }
    save_json(out_path / 'run9_transfer_expanded_summary.json', summary_transfer)
    return summary_transfer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet-sha256', help='Expected review packet SHA256')
    args = parser.parse_args()
    evaluate_transfer(args.packet_sha256 or os.environ.get('TRANSFER_PACKET_SHA256'))
