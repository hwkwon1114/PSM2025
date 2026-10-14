"""Researcher-authorized 240-step saved-registration reassessment; no solver/ledger writes."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PLAN = BASE / 'run/optimization_review/registration_240_reassessment_plan.json'
OUTPUT = BASE / 'run/optimization_review/registration_240_reassessment.json'
CONFIG = 'calibration/configs/run6_stage2_bayesian_optimization.json'
BEFORE = 'run/optimization_review/registration_240_config_before.json'
STATE = 'run/stage2_bayesian_optimization/bayes_state.json'
MANIFEST = 'calibration/data/processed/reconciled_training_manifest_v1.json'
REUSE = 'run/optimization_review/attempt25_registration_diagnostic.json'
EXPECTED_FAILED = [2, 4, 5, 8, 9, 10, 12, 15, 16, 18, 19, 20, 21, 22, 25, 26, 28, 29, 32]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((BASE / name).read_text())


def write_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def select_failed(observations, historical):
    ids = [o['attempt_id'] for o in observations]
    assert len(ids) == len(set(ids)), 'Duplicate campaign attempt'
    history = {o['attempt_id']: o for o in historical}
    assert len(history) == len(historical), 'Duplicate historical attempt'
    selected, excluded = [], []
    for o in sorted(observations, key=lambda o: o['attempt_id']):
        assert type(o['valid']) is bool
        if o['valid']:
            continue
        att = o['attempt_id']
        if o['failure_class'] != 'registration':
            excluded.append({'attempt_id': att, 'failure_class': o['failure_class'],
                             'reason': 'Did not reach registration; no mechanics rerun'})
            continue
        if att <= 12:
            h = history[att]
            assert not h['registration_qualified']
            mesh, native = BASE / h['mesh_file'], BASE / h['state_file']
            scale = 1.0 if mesh.name == 'prediction_mm.vtp' else 1000.0
            assert scale == 1.0 or mesh.name.startswith('bilayer_zigzag_sequence_cycle_029_')
            expected = {str(mesh.relative_to(BASE)): h['mesh_sha256'],
                        str(native.relative_to(BASE)): h['state_sha256']}
            original = {k: h.get(k) for k in ('rms_symmetric_mm', 'stop_reason', 'pose_near_bound')}
            params = h['parameters']
        else:
            case = o['rejected_case']
            folder = Path(case['directory'])
            mesh = folder / 'prediction_mm.vtp'
            native = folder / 'bilayer_zigzag_sequence_native_hybrid_state.json'
            scale, expected, params = 1.0, {}, o['parameters']
            original = {k: case['metrics'][k] for k in
                        ('rms_symmetric_mm', 'stop_reason', 'pose_near_bound', 'iterations_used')}
        selected.append({'attempt_id': att, 'parameters': params,
                         'mesh_file': str(mesh.relative_to(BASE)),
                         'state_file': str(native.relative_to(BASE)),
                         'sequence_file': str((native.parent / 'sequence_result.json').relative_to(BASE)),
                         'scale_to_mm': scale, 'historical_expected_sha256': expected,
                         'gauge': 'reviewed_legacy_raw_solver' if scale == 1000 else 'saved_prediction_mm',
                         'original_valid': False, 'original_metrics': original})
    return selected, excluded


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists(), 'Preserve existing reassessment'
    cfg, before, state = read(CONFIG), read(BEFORE), read(STATE)
    assert sha(BASE / BEFORE) == '7d2e38aaa06f66be8d570e28d26a2fa0c4a108352ef08a8e65d6c4d73fbdc549'
    assert sha(BASE / STATE) == '9a372c10b18f43f69786af844cc6e504887a76ffdd5d4a4c76316fba627912c6'
    expected = json.loads(json.dumps(before))
    expected['registration']['icp_iterations'] = 240
    assert cfg == expected and before['registration']['icp_iterations'] == 60
    assert len(state['observations']) == 32
    selected, excluded = select_failed(state['observations'], read(MANIFEST)['observations'])
    assert [o['attempt_id'] for o in selected] == EXPECTED_FAILED
    assert [o['attempt_id'] for o in excluded] == [17, 24, 27]
    names = [CONFIG, BEFORE, STATE, MANIFEST, REUSE,
             'run/stage2_bayesian_optimization/evaluations.jsonl',
             'run/stage2_bayesian_optimization/registration_metrics_sidecar.json',
             'calibration/data/processed/run6_oriented.npz',
             'calibration/data/processed/run6_oriented.json',
             'calibration/ewcal/geometry.py', str(Path(__file__).relative_to(BASE))]
    for o in selected:
        names.extend(o[k] for k in ('mesh_file', 'state_file', 'sequence_file'))
        if o['attempt_id'] > 12:
            names.append(str((BASE / o['state_file']).parent.parent.relative_to(BASE) / 'result.json'))
    hashes = {n: sha(BASE / n) for n in sorted(set(names))}
    assert hashes['calibration/ewcal/geometry.py'] == 'f7aebfee26552c2f661fa0acd9796a3540fb23ab3327bf85533e660da9196e43'
    assert hashes['calibration/data/processed/run6_oriented.npz'] == 'f2faaddbc5492b6482dd93a68f1cd9699a5ac17f740b577df5f1a8b7392f812e'
    for o in selected:
        assert all(hashes[n] == h for n, h in o['historical_expected_sha256'].items())
    write_new(PLAN, {'schema_version': 1, 'registration': cfg['registration'],
                     'config': cfg, 'input_sha256': hashes, 'endpoints': selected,
                     'excluded': excluded, 'target_rms_mm': 3.0,
                     'claim_limit': 'Saved-shape reassessment only; original campaign/sidecar remain frozen'})


def verify_inputs(plan):
    assert all(sha(BASE / n) == h for n, h in plan['input_sha256'].items()), 'Pinned input changed'


def load_geometry(endpoint, config):
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy
    from calibration.ewcal.geometry import points, read_mesh, reference_dual_areas
    mesh = read_mesh(BASE / endpoint['mesh_file'], scale=endpoint['scale_to_mm'])
    st = read(endpoint['state_file'])
    assert st['schema_version'] == 1 and st['units'] in ('metres', 'm')
    ref = np.asarray(st['reference_positions'], dtype=float) * 1000.0
    faces = np.asarray(st['faces'])
    assert ref.shape == points(mesh).shape and np.isfinite(ref).all()
    assert faces.ndim == 2 and faces.shape[1] == 3 and np.issubdtype(faces.dtype, np.integer)
    assert faces.size and np.all(faces >= 0) and np.all(faces < len(ref))
    cells = vtk_to_numpy(mesh.GetPolys().GetData())
    assert cells.size == 4 * len(faces)
    cells = cells.reshape(-1, 4)
    assert np.all(cells[:, 0] == 3) and np.array_equal(cells[:, 1:], faces)
    area = float(reference_dual_areas(ref, faces).sum())
    expected = config['expected_reference_area_mm2']
    assert abs(area - expected) / expected <= config['area_tolerance_rel']
    seq = read(endpoint['sequence_file'])
    assert seq['schema_version'] == 1 and seq['completed'] is True and seq['cycles'] == 29
    assert seq['backend'] == 'native_hybrid' and seq['equilibrium_solves'] == 1
    assert len(seq['solves']) == 1
    solve = seq['solves'][0]
    native = solve['native_hybrid']
    assert solve['cycle'] == 29 and native['accepted'] is True
    assert native['termination'] == 'gradient_target' and native['curvature'] == 'positive_pivots'
    assert np.isfinite(native['gradient_norm']) and 0 <= native['gradient_norm'] <= 5e-14
    return mesh, ref, faces


def run():
    import numpy as np
    import scipy
    import vtk
    from calibration.ewcal.geometry import align_target
    assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
    assert not OUTPUT.exists(), 'Preserve prior results; no automatic restart'
    plan = json.loads(PLAN.read_text())
    verify_inputs(plan)
    assert plan['registration']['icp_iterations'] == 240
    with np.load(BASE / 'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as data:
        target = data['points_mm'].copy()
    assert target.ndim == 2 and target.shape[1] == 3 and len(target) >= 1000 and np.isfinite(target).all()
    report = {'schema_version': 1, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'job_id': os.environ['SLURM_JOB_ID'], 'hostname': platform.node(),
              'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'vtk': vtk.vtkVersion.GetVTKVersion(), 'plan_sha256': sha(PLAN),
              'registration': plan['registration'], 'excluded': plan['excluded'], 'records': []}
    write_new(OUTPUT, report)
    start = time.monotonic()
    for endpoint in plan['endpoints']:
        tick = time.monotonic()
        mesh, ref, faces = load_geometry(endpoint, plan['config'])
        att = endpoint['attempt_id']
        if att == 25:
            prior = read(REUSE)
            assert prior['status'] == 'complete' and prior['control_matches'] and prior['inputs_unchanged']
            assert prior['attempt_id'] == att and prior['parameters'] == endpoint['parameters']
            assert dict(prior['registration_settings'], icp_iterations=240) == plan['registration']
            assert all(plan['input_sha256'][n] == h for n, h in prior['input_sha256'].items() if n != CONFIG)
            assert prior['metrics_240']['registration_qualified'] and prior['diagnostic_meets_target']
            metrics = prior['metrics_240']
            origin = 'reused_verified_attempt25_diagnostic'
        else:
            metrics, _, _, _ = align_target(mesh, target, plan['registration'], ref, faces)
            origin = 'new_240_reassessment'
        assert metrics['iterations_used'] <= 240 and np.isfinite(metrics['rms_symmetric_mm'])
        report['records'].append(dict(endpoint, metrics=metrics, source=origin,
                                      qualified=metrics['registration_qualified'],
                                      target_met=bool(metrics['registration_qualified'] and metrics['rms_symmetric_mm'] <= 3),
                                      elapsed_seconds=time.monotonic() - tick))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'attempt_id': att, 'qualified': metrics['registration_qualified'],
                          'rms_mm': metrics['rms_symmetric_mm'], 'stop': metrics['stop_reason'],
                          'iterations': metrics['iterations_used'], 'near_bound': metrics['pose_near_bound']}), flush=True)
    verify_inputs(plan)
    report.update(status='complete', inputs_unchanged=True, elapsed_seconds=time.monotonic() - start)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, str(BASE))
    assert sys.flags.optimize == 0, 'Assertions must be enabled'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        run()
