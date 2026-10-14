"""Fixed saved-shape translation-cap diagnostic; no production settings/record writes."""
import argparse
import copy
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from calibration.analysis import reassess_failed_registration as prior

BASE = prior.BASE
PLAN = BASE / 'run/optimization_review/free_translation_plan.json'
OUTPUT = BASE / 'run/optimization_review/free_translation_diagnostic.json'
CSV = OUTPUT.with_suffix('.csv')
STATE = 'run/run6_bo_refinement_240/bayes_state.json'
HISTORY = 'run/run6_bo_refinement_240/corrected_history.json'
LEGACY = 'run/optimization_review/legacy_frame_reassessment_retry.json'
TARGET = 'calibration/data/processed/run6_oriented.npz'
TIMEOUTS = [17, 24, 27, 35, 37, 39]
RMS_KEYS = ('rms_symmetric_mm', 'rms_target_to_sim_mm', 'rms_sim_to_target_weighted_mm')


def record_metrics(record):
    return record.get('metrics') or (record.get('cases') or [record.get('rejected_case') or {}])[0].get('metrics')


def select_completed(observations):
    ids = [o['attempt_id'] for o in observations]
    assert len(ids) == len(set(ids)), 'Duplicate attempt'
    selected, excluded = [], []
    for o in sorted(observations, key=lambda r: r['attempt_id']):
        assert type(o['valid']) is bool
        if o.get('failure_class') in ('timeout', 'solver', 'infrastructure'):
            assert not o['valid'] and not record_metrics(o), 'Mechanics failure has registration data'
            excluded.append({'attempt_id': o['attempt_id'], 'failure_class': o['failure_class']})
        else:
            assert record_metrics(o), 'Missing completed registration'
            selected.append(o)
    return selected, excluded


def free_settings(settings):
    result = copy.deepcopy(settings)
    result['translation_bounds_mm'] = [float('inf')] * 3
    return result


def check_fidelity(current, saved):
    assert all(abs(current[k] - saved[k]) <= 1e-6 for k in RMS_KEYS), 'Saved-pose RMS mismatch'


def proximity(forward, reverse, weights):
    import numpy as np
    forward, reverse, weights = map(np.asarray, (forward, reverse, weights))
    assert forward.ndim == reverse.ndim == weights.ndim == 1
    assert len(forward) and len(reverse) and reverse.shape == weights.shape
    assert all(np.isfinite(a).all() for a in (forward, reverse, weights))
    assert np.all(forward >= 0) and np.all(reverse >= 0) and np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.)
    return {str(threshold): {'scan_fraction': float(np.mean(forward <= threshold)),
                            'simulation_area_fraction': float(weights[reverse <= threshold].sum())}
            for threshold in (3, 5)}


def evaluate_saved_pose(mesh, target, ref, faces, pose):
    import numpy as np
    from scipy.spatial import cKDTree
    from calibration.ewcal.geometry import (SurfaceDistance, compute_bidirectional_metrics,
        points, pose_points, reference_dual_areas)
    pose = np.asarray(pose, float)
    assert pose.shape == (6,) and np.isfinite(pose).all()
    aligned = pose_points(target, pose)
    forward, _ = SurfaceDistance(mesh).closest(aligned)
    reverse, _ = cKDTree(aligned).query(points(mesh))
    areas = reference_dual_areas(ref, faces)
    return compute_bidirectional_metrics(mesh, aligned, ref, faces, forward), proximity(forward, reverse, areas / areas.sum())


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists() and not CSV.exists(), 'Preserve existing diagnostic'
    state, history, legacy = map(prior.read, (STATE, HISTORY, LEGACY))
    assert state['status'] == 'consecutive_failures'
    assert state['charged_attempts'] == len(state['completed_attempts']) == 40
    assert len(state['observations']) == 40 and state['waves_reserved'] == 2
    assert state['history_sha256'] == prior.sha(BASE / HISTORY)
    assert state['evaluator_identity'] == history['evaluator_identity']
    prior.verify_inputs(history)
    assert legacy['status'] == 'complete' and legacy['inputs_unchanged'] is True
    selected, excluded = select_completed(state['observations'])
    assert [e['attempt_id'] for e in excluded] == TIMEOUTS
    assert len(selected) == 34 and sum(o['valid'] for o in selected) == 14
    legacy_by_id = {r['attempt_id']: r for r in legacy['records']}
    assert set(legacy_by_id) == {2, 4, 5}
    endpoints = []
    names = set(history['input_sha256']) | {STATE, HISTORY, LEGACY, TARGET,
        'run/run6_bo_refinement_240/evaluations.jsonl',
        'calibration/configs/run6_bo_refinement_240.json',
        'calibration/analysis/reassess_failed_registration.py',
        'calibration/analysis/diagnose_free_translation.py', 'calibration/tests/test_free_translation.py'}
    for o in selected:
        att = o['attempt_id']
        if 'mesh_file' in o:
            mesh, native = BASE / o['mesh_file'], BASE / o['state_file']
            scale = 1000. if att in legacy_by_id else 1.
        else:
            case = (o.get('cases') or [o['rejected_case']])[0]
            directory = Path(case['directory'])
            mesh = directory / 'prediction_mm.vtp'
            native = directory / 'bilayer_zigzag_sequence_native_hybrid_state.json'
            scale = 1.
        endpoint = {'attempt_id': att, 'parameters': o['parameters'], 'original_valid': o['valid'],
            'saved_metrics': record_metrics(o), 'scale_to_mm': scale,
            'mesh_file': str(mesh.resolve().relative_to(BASE)),
            'state_file': str(native.resolve().relative_to(BASE)),
            'sequence_file': str((native.parent / 'sequence_result.json').resolve().relative_to(BASE))}
        if att in legacy_by_id:
            saved = legacy_by_id[att]
            assert all(endpoint[k] == saved[k] for k in ('mesh_file', 'state_file', 'scale_to_mm', 'parameters'))
            endpoint['normalization'] = saved['normalization']
        endpoints.append(endpoint)
        names.update(endpoint[k] for k in ('mesh_file', 'state_file', 'sequence_file'))
    for o in state['observations']:
        if o['attempt_id'] >= 33:
            names.add(str((Path(o['directory']) / 'result.json').resolve().relative_to(BASE)))
    cfg = history['config']
    assert cfg['registration']['icp_iterations'] == 240
    assert cfg['registration']['translation_bounds_mm'] == [5, 5, 20]
    assert cfg['registration']['rotation_bound_deg'] == 5
    assert cfg['registration']['metric'] == 'symmetric'
    prior.write_new(PLAN, {'schema_version': 1, 'config': cfg, 'endpoints': endpoints, 'excluded': excluded,
        'input_sha256': {n: prior.sha(BASE / n) for n in sorted(names)},
        'translation_policy': 'all translations unrestricted; infinities created only in memory',
        'claim_limit': 'Diagnostic pose comparison only; no production qualification/training migration or material-correspondence claim'})


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def run():
    import numpy as np
    import scipy
    import vtk
    from calibration.ewcal.geometry import align_target, points, triangle_mesh
    assert os.environ.get('SLURM_JOB_ID'), 'Owned compute allocation required'
    assert not OUTPUT.exists() and not CSV.exists(), 'No overwrite/automatic restart'
    plan = json.loads(PLAN.read_text())
    prior.verify_inputs(plan)
    settings = plan['config']['registration']
    with np.load(BASE / TARGET, allow_pickle=False) as data:
        target = data['points_mm'].copy()
    assert target.shape == (14503, 3) and np.isfinite(target).all()
    report = {'schema_version': 1, 'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
        'hostname': platform.node(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'vtk': vtk.vtkVersion.GetVTKVersion(), 'python': sys.version,
        'plan_sha256': prior.sha(PLAN), 'bounded_settings': settings,
        'diagnostic_translation_bounds_mm': None, 'rotation_bound_deg': settings['rotation_bound_deg'],
        'excluded': plan['excluded'], 'claim_limit': plan['claim_limit'], 'records': []}
    prior.write_new(OUTPUT, report)
    start = time.monotonic()
    for endpoint in plan['endpoints']:
        tick = time.monotonic()
        mesh, ref, faces = prior.load_geometry(endpoint, plan['config'])
        if 'normalization' in endpoint:
            transform = endpoint['normalization']
            r, t = np.asarray(transform['row_vector_rotation']), np.asarray(transform['translation_mm'])
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0)
            assert abs(np.linalg.det(r) - 1) <= 1e-12
            mesh = triangle_mesh(points(mesh) @ r + t, faces)
        baseline, coverage_before = evaluate_saved_pose(mesh, target, ref, faces, endpoint['saved_metrics']['pose'])
        check_fidelity(baseline, endpoint['saved_metrics'])
        metrics, _, _, _ = align_target(mesh, target, free_settings(settings), ref, faces)
        current, coverage_after = evaluate_saved_pose(mesh, target, ref, faces, metrics['pose'])
        check_fidelity(current, metrics)
        assert np.isfinite(metrics['rms_symmetric_mm']) and 1 <= metrics['iterations_used'] <= 240
        last = metrics['trace'][-1]
        stable = all(last[k] < .005 for k in ('delta_rotation_deg', 'delta_translation_mm', 'delta_point_motion_rms_mm'))
        row = {'attempt_id': endpoint['attempt_id'], 'parameters': endpoint['parameters'],
            'original_valid': endpoint['original_valid'], 'saved_pose': endpoint['saved_metrics']['pose'],
            'saved_metrics_reproduced': baseline, 'proximity_before': coverage_before,
            'free_translation_metrics': metrics, 'proximity_after': coverage_after,
            'diagnostic_converged_interior_rotation': metrics['registration_qualified'],
            'physical_motion_stable': stable,
            'translation_norm_mm': float(np.linalg.norm(metrics['pose'][3:])),
            'rms_change_mm': metrics['rms_symmetric_mm'] - baseline['rms_symmetric_mm'],
            'elapsed_seconds': time.monotonic() - tick}
        report['records'].append(row)
        save(report)
        print(json.dumps({'attempt': row['attempt_id'], 'old_rms': baseline['rms_symmetric_mm'],
            'free_rms': metrics['rms_symmetric_mm'], 'interior_converged': metrics['registration_qualified'],
            'pose': metrics['pose'], 'stop': metrics['stop_reason'], 'iterations': metrics['iterations_used']}), flush=True)
    prior.verify_inputs(plan)
    assert [r['attempt_id'] for r in report['records']] == [e['attempt_id'] for e in plan['endpoints']]
    report.update(status='complete', inputs_unchanged=True, elapsed_seconds=time.monotonic() - start)
    save(report)
    with CSV.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['attempt', 'original_valid', 'old_rms_mm', 'free_rms_mm', 'diagnostic_interior_converged',
            'physical_motion_stable', 'rotation_deg', 'tx_mm', 'ty_mm', 'tz_mm', 'iterations', 'stop',
            'scan_within3mm', 'sim_area_within3mm', 'scan_within5mm', 'sim_area_within5mm'])
        for row in report['records']:
            m = row['free_translation_metrics']; c = row['proximity_after']
            writer.writerow([row['attempt_id'], row['original_valid'], row['saved_metrics_reproduced']['rms_symmetric_mm'],
                m['rms_symmetric_mm'], row['diagnostic_converged_interior_rotation'], row['physical_motion_stable'],
                m['rotation_angle_deg'], *m['pose'][3:], m['iterations_used'], m['stop_reason'],
                c['3']['scan_fraction'], c['3']['simulation_area_fraction'], c['5']['scan_fraction'], c['5']['simulation_area_fraction']])


if __name__ == '__main__':
    assert sys.flags.optimize == 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    prepare() if parser.parse_args().prepare else run()
