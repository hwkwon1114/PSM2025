"""Compare flat and model-derived initial positions for frozen a0084, Run 2.

Prepare with ``prepare`` (no solver), then run once with ``run`` after review.
This is an exploratory numerical diagnostic, not a new calibration or validation.
"""
import csv
import json
import re
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_convention_corrected as cc
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.geometry import points, read_mesh
from calibration.ewcal.runner import Evaluator, sequence_json, validate_mesh_topology

OUT = ROOT / 'run/diagnostics/run2_initial_state_a0084'
BEST = ROOT / 'run/fresh_lhs_joint4/results/a0084.json'
SUMMARY = ROOT / 'run/fresh_lhs_joint4/summary.json'
FLAT_PREVIOUS = ROOT / 'run/run9_interim/fresh_lhs_joint4_a0084/evaluations/attempt_0001/case_00'
SEED_PREVIOUS = ROOT / 'run/run9_interim/joint_stage4_reduced4_a0016_long/evaluations/attempt_0001/case_00'
STATE = SEED_PREVIOUS / 'bilayer_zigzag_sequence_native_hybrid_state.json'
INITIAL_NAME = 'bilayer_zigzag_sequence_cycle_000_initial.vtp'
SEQUENCE = FLAT_PREVIOUS / 'sequence.json'
BATCH = ROOT / 'calibration/bo/run2_initial_state_diagnostic.sbatch'
EXPECTED = {
    BEST: '1fd2db5d0ac66883bb71a8375a5d702eb4d9d5d3cc1d186a55151bc4b02884db',
    STATE: 'ad6a635b628017395b8b6f5557546528ebd6116e5c9cb8b41cee6ed1c35b9601',
    FLAT_PREVIOUS / INITIAL_NAME: 'e470b59773859c42c26b8f6eb770d024016bf59a55a462872329f841a2749991',
    SEED_PREVIOUS / INITIAL_NAME: 'e470b59773859c42c26b8f6eb770d024016bf59a55a462872329f841a2749991',
    ROOT / 'bin/shell': cc.BINARY_SHA,
}


def frozen_parameters():
    best = json.loads(BEST.read_text())
    summary = json.loads(SUMMARY.read_text())
    if not best['valid'] or best['attempt'] != 84 or summary['best'] != best:
        raise ValueError('Fresh-search best is not the frozen valid a0084')
    return best['parameters']


def make_configs(out, vals):
    base = cc.config('run9')
    base['hardening']['beta'] = float(vals['beta'])
    base['hardening']['floor'] = float(vals['floor'])
    base['solver']['command_prefix'] = ['/usr/bin/stdbuf', '-oL', '-eL']
    base['solver']['timeout_seconds'] = 3900
    args = base['solver']['args']
    args['-native_hybrid_seconds'] = 3600
    args['-native_hybrid_export_trace'] = True
    if (base['solver']['solve_mode'] != 'final_only' or
            float(args['-sequence_slope_feedback']) != 0.0 or base['cases'][0]['repeat'] != 33):
        raise ValueError('Run 2 final-only, zero-slope-feedback contract changed')
    p4 = {key: float(vals[key]) for key in base['parameters']}
    if sequence_json(base, base['cases'][0], p4) != json.loads(SEQUENCE.read_text()):
        raise ValueError('Frozen Run 2 physical sequence differs from original attempt')

    # Separate configs differ only in output paths and this position-only seed.
    import copy
    configs = {}
    for start in ('flat', 'seeded'):
        cfg = copy.deepcopy(base)
        cfg['output_dir'] = str(out / start)
        cfg['report_dir'] = str(out / start / 'report')
        if start == 'seeded':
            cfg['solver']['args'].update({
                '-prescribed_at_cycle': 32,
                '-prescribed_disp_csv': str(out / 'inputs/seed_displacement_m.csv'),
                '-prescribed_reference': 'rest',
                '-prescribed_reequilibrate': False,
                '-prescribed_tag': 'a0016_seed',
            })
        configs[start] = cfg
    return configs, p4


def prepare(out=OUT):
    for path, expected in EXPECTED.items():
        if digest(path) != expected:
            raise ValueError(f'Frozen input identity mismatch: {path}')
    vals = frozen_parameters()
    configs, _ = make_configs(out, vals)
    data = json.loads(STATE.read_text())
    proof = data['proof']
    if (data['units'] != 'metres' or not proof['accepted'] or
            proof['termination'] != 'gradient_target' or
            proof['curvature'] != 'positive_pivots' or not np.isfinite(proof['gradient_norm']) or
            proof['gradient_norm'] > 5e-14 or proof['verification_hessians'] < 1):
        raise ValueError('Seed model is not an accepted native equilibrium')
    ref = np.asarray(data['reference_positions'], float)
    pos = np.asarray(data['positions'], float)
    faces = np.asarray(data['faces'])
    mesh = read_mesh(FLAT_PREVIOUS / INITIAL_NAME)
    validate_mesh_topology(mesh, faces)
    if (ref.shape != pos.shape or ref.ndim != 2 or ref.shape[1] != 3 or
            not np.isfinite(ref).all() or not np.isfinite(pos).all() or
            points(mesh).shape != ref.shape):
        raise ValueError('Invalid seed/reference vertex arrays')
    reference_delta_m = float(np.max(np.abs(points(mesh) - ref)))
    # The initial VTP stores float32 points, whereas the native reference is double.
    if reference_delta_m > 2e-8:
        raise ValueError('Seed reference positions do not match the original initial mesh')
    displacement = pos - ref
    if np.max(np.abs(displacement)) == 0:
        raise ValueError('Alternative seed is unexpectedly flat')

    inputs = out / 'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    seed = inputs / 'seed_displacement_m.csv'
    with seed.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vertex_index', 'dx', 'dy', 'dz'])
        for index, row in enumerate(displacement):
            writer.writerow([index, *(format(float(v), '.17g') for v in row)])
    files = set(EXPECTED) | {
        SUMMARY, SEQUENCE, Path(__file__).resolve(), BATCH,
        ROOT / 'calibration/tests/test_run2_initial_state_diagnostic.py',
        ROOT / 'calibration/bo/run_convention_corrected.py',
        ROOT / 'shellsim/configs/default_adaptive_cutback.json',
        ROOT / 'run/run9_transfer_expanded/evaluations/attempt_0209/request.json',
        ROOT / 'run/putong_convention_check/preserved_samples/run9.npz',
        ROOT / 'run/putong_convention_check/preserved_samples/run9.json',
    } | set((ROOT / 'calibration/ewcal').glob('*.py'))
    manifest = {
        'purpose': 'exploratory Run 2 initial-position diagnostic, not validation',
        'candidate': 'fresh_lhs_joint4/a0084', 'parameters': vals,
        'seed_source': str(STATE), 'seed_type': 'a0016 model positions only; edge directors not copied',
        'vertices': len(ref), 'faces': len(faces),
        'max_reference_mesh_difference_m': reference_delta_m,
        'max_absolute_displacement_mm_by_axis': (1000*np.max(np.abs(displacement), axis=0)).tolist(),
        'seed_csv_sha256': digest(seed),
        'input_sha256': {str(p): digest(p) for p in sorted(files)},
        'configs': configs,
    }
    save_json(inputs / 'manifest.json', manifest)
    print(f'Prepared {len(ref)} seed vertices; max reference difference {reference_delta_m:.3g} m; no solver invoked.')
    return manifest


def verify_inputs(out=OUT):
    manifest = json.loads((out / 'inputs/manifest.json').read_text())
    for name, expected in manifest['input_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Reviewed input identity changed: {name}')
    if digest(out / 'inputs/seed_displacement_m.csv') != manifest['seed_csv_sha256']:
        raise ValueError('Reviewed displacement file changed')
    configs, p4 = make_configs(out, frozen_parameters())
    if configs != manifest['configs']:
        raise ValueError('Effective diagnostic configuration changed')
    return configs, p4


def diagnostic_failure(rec, folder):
    """Refine only a fully evidenced native corrector abort; preserve raw records."""
    raw = rec.get('failure_class')
    term = rec.get('solver_termination', {})
    if (rec.get('valid') or raw != 'infrastructure' or rec.get('failure_phase') != 'solver' or
            term.get('return_code') != -signal.SIGABRT):
        return raw, None
    try:
        lines = (folder / 'solver.log').read_text(errors='replace').splitlines()
        reports = [line for line in lines if line.startswith('[native_hybrid] ')]
        terminal = [line for line in lines[-3:] if line.startswith('  what():  zigzag_sequence: equilibrium corrector failed at cycle ')]
        if not reports or len(terminal) != 1:
            return raw, None
        report = dict(re.findall(r'(\w+)=([^\s]+)', reports[-1]))
        match = re.search(r' at cycle (\d+), lambda=([^,]+), gradient_norm=([^,]+), termination=([a-z_]+)\.$', terminal[0])
        known = {'time_cap', 'iteration_cap', 'evaluation_cap', 'hessian_cap',
                 'newton_stagnation', 'indefinite_curvature'}
        status = report['status']
        if (match is None or int(match[1]) != 33 or status not in known or match[4] != status or
                report['accepted'] != '0' or not np.isfinite(float(report['energy'])) or
                not np.isfinite(float(report['gradient'])) or float(report['gradient']) < 0 or
                not np.isfinite(float(match[2])) or not np.isfinite(float(match[3])) or
                int(report['total_attempts']) < 1 or int(report['total_evaluations']) < 1 or
                int(report['total_hessians']) < 0):
            return raw, None
    except (OSError, KeyError, ValueError, TypeError, OverflowError):
        return raw, None
    return ('timeout' if status == 'time_cap' else 'solver'), {
        'reason': 'matched_native_corrector_failure', 'status': status,
        'native_report': reports[-1], 'terminal_exception': terminal[0]}


def verify_saved_start(name, folder):
    if digest(folder / INITIAL_NAME) != EXPECTED[FLAT_PREVIOUS / INITIAL_NAME]:
        raise ValueError(f'{name} exported initial mesh differs from the frozen original')
    if json.loads((folder / 'sequence.json').read_text()) != json.loads(SEQUENCE.read_text()):
        raise ValueError(f'{name} saved physical sequence differs from the original')
    if name == 'seeded':
        imposed = folder / 'bilayer_zigzag_sequence_a0016_seed_after_cycle_032_imposed_vertices.csv'
        data = np.genfromtxt(imposed, names=True, delimiter=',')
        expected = np.asarray(json.loads(STATE.read_text())['positions'])
        position = np.column_stack([data['x'], data['y'], data['z']])
        if (len(data) != len(expected) or not np.array_equal(data['vertex_index'], np.arange(len(data))) or
                any(np.any(data[k]) for k in ('bc_fixed_x', 'bc_fixed_y', 'bc_fixed_z')) or
                not np.isfinite(position).all() or np.max(np.abs(position - expected)) > 1e-12):
            raise ValueError('Seeded vertex positions were not imposed as intended')
        return {'seed_position_max_difference_m': float(np.max(np.abs(position - expected))),
                'fixed_vertex_dofs': 0}
    return {'initial_mesh_sha256': digest(folder / INITIAL_NAME)}


def compare_completed(results):
    case = {name: OUT / name / 'evaluations/attempt_0001/case_00' for name in ('flat', 'seeded')}
    original_history = (FLAT_PREVIOUS / 'sequence_history.csv').read_bytes()
    histories_match = all((case[n] / 'sequence_history.csv').is_file() and
                          (case[n] / 'sequence_history.csv').read_bytes() == original_history for n in case)
    report = {'results': results, 'history_matches_original_for_both': histories_match,
              'target_form_comparison': None, 'final_shape_comparison': None}
    if all(results[n]['valid'] for n in case):
        states = {n: json.loads((case[n] / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
                  for n in case}
        keys = ('reference_positions', 'material_uv', 'faces', 'edges', 'abar_top', 'abar_bottom', 'bbar')
        same = {key: bool(np.array_equal(states['flat'][key], states['seeded'][key])) for key in keys}
        same['material'] = states['flat']['material'] == states['seeded']['material']
        report['target_form_comparison'] = same
        if histories_match and all(same.values()):
            a = points(read_mesh(case['flat'] / 'prediction_mm.vtp'))
            b = points(read_mesh(case['seeded'] / 'prediction_mm.vtp'))
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Final material-frame geometry is not comparable')
            delta = b - a
            norm = np.linalg.norm(delta, axis=1)
            report['final_shape_comparison'] = {
                'frame': 'per-solution material-reference leveling; same corresponding mesh vertices',
                'vertex_displacement_rms_mm': float(np.sqrt(np.mean(norm**2))),
                'vertex_displacement_p95_mm': float(np.percentile(norm, 95)),
                'height_difference_rms_mm': float(np.sqrt(np.mean(delta[:, 2]**2))),
            }
    save_json(OUT / 'summary.json', report)
    return report


def run():
    configs, p4 = verify_inputs()
    if any((OUT / name).exists() for name in ('flat', 'seeded')):
        raise FileExistsError('A diagnostic run already exists; no automatic restart or overwrite')
    results = {}
    for name in ('flat', 'seeded'):
        print(f'\n=== {name} start ===', flush=True)
        ev = Evaluator(configs[name])
        # The legacy positive-gbot parameter codec cannot represent the frozen zero.
        ev.parameters = SimpleNamespace(decode=lambda _x: dict(p4))
        rec = ev.evaluate([0.0] * len(p4), use_cache=False, attempt_id=1)
        save_json(OUT / name / 'forward_result.json', rec)
        folder = OUT / name / 'evaluations/attempt_0001/case_00'
        failure, evidence = diagnostic_failure(rec, folder)
        results[name] = {'valid': rec['valid'], 'failure_class': failure,
                         'evaluator_failure_class': rec.get('failure_class'),
                         'classification_evidence': evidence,
                         'error': rec.get('error'), 'seconds': rec['seconds'],
                         'objective_mm': rec['objective_mm'] if rec['valid'] else None}
        save_json(OUT / 'partial_summary.json', results)
        if not rec['valid'] and failure not in ('timeout', 'solver', 'registration'):
            raise RuntimeError(f"{name} infrastructure failure; stop: {rec.get('error')}")
        results[name]['start_check'] = verify_saved_start(name, folder)
        if rec['valid']:
            results[name]['energy'] = rec['cases'][0]['manifest']['total_energy']
        save_json(OUT / 'partial_summary.json', results)
    report = compare_completed(results)
    if not report['history_matches_original_for_both']:
        raise RuntimeError('Growth histories differ; diagnostic comparison is invalid')
    if report['target_form_comparison'] is not None and not all(report['target_form_comparison'].values()):
        raise RuntimeError('Terminal physical load identities differ; diagnostic comparison is invalid')
    print('Both starts attempted; compare retained outputs without selecting parameters.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'run'):
        raise SystemExit('Usage: run2_initial_state_diagnostic.py prepare|run')
    prepare() if sys.argv[1] == 'prepare' else run()
