"""Three authorized saved-frame diagnostics; no mechanics, training or original writes."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from calibration.analysis import reassess_failed_registration as prior

BASE = prior.BASE
PLAN = BASE / 'run/optimization_review/legacy_frame_reassessment_retry_plan.json'
OUTPUT = BASE / 'run/optimization_review/legacy_frame_reassessment_retry.json'
OLD_PLAN = 'run/optimization_review/alignment_investigation_plan.json'
AUDIT = 'run/optimization_review/alignment_investigation.json'
RAW_REPORT = 'run/optimization_review/registration_240_reassessment.json'
IDS = [2, 4, 5]


def select_endpoints(endpoints):
    ids = [o['attempt_id'] for o in endpoints]
    assert len(ids) == len(set(ids)), 'Duplicate endpoint'
    selected = sorted((o for o in endpoints if o['attempt_id'] in IDS), key=lambda o: o['attempt_id'])
    assert [o['attempt_id'] for o in selected] == IDS, 'Missing selected endpoint'
    assert all(o['gauge'] == 'legacy_raw_solver' and o['scale_to_mm'] == 1000.0 for o in selected)
    return selected


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists(), 'Preserve existing diagnostic'
    previous = prior.read(OLD_PLAN)
    prior.verify_inputs(previous)
    audit, raw = prior.read(AUDIT), prior.read(RAW_REPORT)
    assert all(audit[k] is True for k in ('baseline_matches', 'inputs_unchanged'))
    assert audit['status'] == raw['status'] == 'complete' and raw['inputs_unchanged'] is True
    cfg = prior.read(prior.CONFIG)
    assert cfg == previous['config'] and cfg['registration']['icp_iterations'] == 240
    endpoints = select_endpoints(previous['endpoints'])
    import xml.etree.ElementTree as ET
    for endpoint in endpoints:
        storage = ET.parse(BASE / endpoint['mesh_file']).find('./PolyData/Piece/Points/DataArray').get('type')
        assert storage == 'Float32', 'Unexpected raw VTP point storage'
        endpoint['point_storage_type'] = storage
    baselines = [next(o for o in raw['records'] if o['attempt_id'] == att) for att in IDS]
    audits = [next(o for o in audit['frame_audit'] if o['attempt_id'] == att) for att in IDS]
    for endpoint, baseline, row in zip(endpoints, baselines, audits):
        assert not baseline['qualified'] and baseline['metrics']['pose_near_bound']
        assert all(endpoint[k] == baseline[k] == row[k] for k in ('parameters', 'mesh_file', 'state_file', 'scale_to_mm'))
    assert prior.sha(BASE / 'run/optimization_review/legacy_frame_reassessment_initial_source.py') == '4d6b3735d3d38ee576e632cfbf31c946876835dc6a74588f6c7572f8c878a23c'
    assert prior.sha(BASE / 'run/optimization_review/legacy_frame_reassessment_initial_tests.py') == 'e5c5b664f409f652e6ce1fae5430a08953922f450275a39300605c6341c622e3'
    names = set(previous['input_sha256']) | {OLD_PLAN, AUDIT, RAW_REPORT,
        str(Path(__file__).relative_to(BASE)), 'calibration/tests/test_legacy_frames.py',
        'run/optimization_review/legacy_frame_reassessment_initial_source.py',
        'run/optimization_review/legacy_frame_reassessment_initial_tests.py',
        'run/optimization_review/legacy_frame_reassessment_plan.json',
        'run/optimization_review/legacy_frame_reassessment.json',
        'run/optimization_review/legacy_frame_reassessment.log'}
    prior.write_new(PLAN, {'schema_version': 1, 'input_sha256': {n: prior.sha(BASE / n) for n in sorted(names)},
        'config': cfg, 'endpoints': endpoints, 'raw_baselines': baselines, 'original_audits': audits,
        'claim_limit': 'Separate saved-frame diagnoses only; original labels/history remain unchanged'})


def serialized_native_mm(native_positions_m):
    """Represent native doubles in the original raw VTP's Float32 metre precision."""
    import numpy as np
    p = np.asarray(native_positions_m, dtype=float)
    assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
    serialized = p.astype(np.float32).astype(float) * 1000.0
    assert np.isfinite(serialized).all()
    return serialized


def canonicalize(mesh, ref, faces, native_positions_mm):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from calibration.ewcal.geometry import points, rigid_fit, remove_solver_rigid_motion, triangle_mesh
    p = points(mesh)
    assert p.shape == ref.shape == native_positions_mm.shape and np.isfinite(native_positions_mm).all()
    assert np.max(np.abs(p - native_positions_mm)) <= 1e-10, 'Raw mesh/serialized native final positions differ'
    r, t = rigid_fit(p, ref)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0) and abs(np.linalg.det(r) - 1) <= 1e-12
    canonical = remove_solver_rigid_motion(mesh, triangle_mesh(ref, faces))
    q = points(canonical)
    np.testing.assert_allclose(q, p @ r + t, atol=1e-10, rtol=0)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    old_lengths = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    new_lengths = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1)
    edge_error = float(np.max(np.abs(new_lengths - old_lengths)))
    assert edge_error <= 1e-9, 'Rigid transform changed shape'
    rr, tt = rigid_fit(q, ref)
    residual_angle = float(np.rad2deg(Rotation.from_matrix(rr).magnitude()))
    assert residual_angle <= 1e-6 and np.linalg.norm(tt) <= 1e-6
    np.testing.assert_array_equal(points(mesh), p)
    return canonical, {'row_vector_rotation': r.tolist(), 'translation_mm': t.tolist(),
        'rotation_deg': float(np.rad2deg(Rotation.from_matrix(r).magnitude())),
        'centroid_offset_mm': (p.mean(0) - ref.mean(0)).tolist(),
        'canonical_residual_rotation_deg': residual_angle,
        'canonical_residual_translation_mm': tt.tolist(), 'max_edge_length_change_mm': edge_error,
        'max_serialized_native_position_difference_mm': float(np.max(np.abs(p - native_positions_mm))),
        'original_mesh_unchanged': True}


def run():
    import numpy as np
    import scipy
    import vtk
    from calibration.ewcal.geometry import align_target
    assert os.environ.get('SLURM_JOB_ID'), 'Owned compute allocation required'
    assert not OUTPUT.exists(), 'No automatic overwrite/restart'
    plan = json.loads(PLAN.read_text())
    prior.verify_inputs(plan)
    assert select_endpoints(plan['endpoints']) == plan['endpoints']
    assert plan['config']['registration']['icp_iterations'] == 240
    with np.load(BASE / 'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as data:
        target = data['points_mm'].copy()
    assert target.shape == (14503, 3) and np.isfinite(target).all()
    report = {'schema_version': 1, 'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
        'hostname': platform.node(), 'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'vtk': vtk.vtkVersion.GetVTKVersion(), 'plan_sha256': prior.sha(PLAN),
        'registration': plan['config']['registration'], 'claim_limit': plan['claim_limit'], 'records': []}
    prior.write_new(OUTPUT, report)
    start = time.monotonic()
    for endpoint, baseline, audit in zip(plan['endpoints'], plan['raw_baselines'], plan['original_audits']):
        assert endpoint['attempt_id'] == baseline['attempt_id'] == audit['attempt_id']
        tick = time.monotonic()
        mesh, ref, faces = prior.load_geometry(endpoint, plan['config'])
        assert endpoint['point_storage_type'] == 'Float32'
        native_m = np.asarray(prior.read(endpoint['state_file'])['positions'], dtype=float)
        expected_raw = serialized_native_mm(native_m)
        canonical, transform = canonicalize(mesh, ref, faces, expected_raw)
        transform['point_storage_type'] = 'Float32'
        transform['max_native_float32_rounding_mm'] = float(np.max(np.abs(native_m * 1000.0 - expected_raw)))
        assert abs(transform['rotation_deg'] - audit['rotation_to_reference_deg']) <= 1e-6
        np.testing.assert_allclose(transform['centroid_offset_mm'], audit['centroid_offset_from_reference_mm'], atol=1e-6, rtol=0)
        print('START attempt', endpoint['attempt_id'], flush=True)
        metrics, _, _, _ = align_target(canonical, target, plan['config']['registration'], ref, faces)
        assert 1 <= metrics['iterations_used'] <= 240 and np.isfinite(metrics['rms_symmetric_mm'])
        row = dict(endpoint, raw_metrics=baseline['metrics'], normalization=transform, metrics=metrics,
            qualified=metrics['registration_qualified'], elapsed_seconds=time.monotonic() - tick)
        report['records'].append(row)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'attempt_id': row['attempt_id'], 'qualified': row['qualified'],
            **{k: metrics[k] for k in ('rms_symmetric_mm', 'iterations_used', 'stop_reason', 'pose_near_bound')}}), flush=True)
    prior.verify_inputs(plan)
    assert [r['attempt_id'] for r in report['records']] == IDS
    report.update(status='complete', inputs_unchanged=True, elapsed_seconds=time.monotonic() - start)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    assert sys.flags.optimize == 0, 'Assertions must be enabled'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    prepare() if parser.parse_args().prepare else run()
