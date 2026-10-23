"""Fixed saved-frame audit and two-start full-symmetric pose diagnostic; no recipe adoption."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibration.analysis import reassess_failed_registration as prior
from calibration.ewcal.geometry import (SurfaceDistance, compute_bidirectional_metrics, points,
    pose_points, reference_dual_areas, rigid_fit, rotvec_to_u, u_to_rotvec)

BASE = prior.BASE
PLAN = BASE / 'run/optimization_review/alignment_investigation_plan.json'
OUTPUT = BASE / 'run/optimization_review/alignment_investigation.json'
OLD_PLAN = 'run/optimization_review/registration_240_reassessment_plan.json'
OLD_REPORT = 'run/optimization_review/registration_240_reassessment.json'


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists(), 'Preserve prior investigation'
    previous = prior.read(OLD_PLAN)
    prior.verify_inputs(previous)
    history = {o['attempt_id']: o for o in prior.read(prior.MANIFEST)['observations']}
    observations = prior.read(prior.STATE)['observations']
    assert len(observations) == 32 and len({o['attempt_id'] for o in observations}) == 32
    endpoints = []
    for o in sorted(observations, key=lambda o: o['attempt_id']):
        att = o['attempt_id']
        if att in (17, 24, 27):
            assert o['failure_class'] == 'timeout' and not o['valid']
            continue
        if att <= 12:
            h = history[att]
            mesh, native = BASE / h['mesh_file'], BASE / h['state_file']
            scale = 1.0 if mesh.name == 'prediction_mm.vtp' else 1000.0
            assert scale == 1.0 or mesh.name.startswith('bilayer_zigzag_sequence_cycle_029_')
            assert prior.sha(mesh) == h['mesh_sha256'] and prior.sha(native) == h['state_sha256']
        else:
            case = o['cases'][0] if o['valid'] else o['rejected_case']
            directory = Path(case['directory'])
            mesh, native = directory / 'prediction_mm.vtp', directory / 'bilayer_zigzag_sequence_native_hybrid_state.json'
            scale = 1.0
        endpoints.append({'attempt_id': att, 'parameters': o['parameters'], 'original_valid': o['valid'],
                          'mesh_file': str(mesh.relative_to(BASE)), 'state_file': str(native.relative_to(BASE)),
                          'sequence_file': str((native.parent / 'sequence_result.json').relative_to(BASE)),
                          'scale_to_mm': scale, 'gauge': 'legacy_raw_solver' if scale == 1000 else 'prediction_mm'})
    assert len(endpoints) == 29
    names = set(previous['input_sha256']) | {OLD_PLAN, OLD_REPORT, str(Path(__file__).relative_to(BASE))}
    for o in endpoints:
        names.update(o[k] for k in ('mesh_file', 'state_file', 'sequence_file'))
    prior.write_new(PLAN, {'schema_version': 1, 'input_sha256': {n: prior.sha(BASE / n) for n in sorted(names)},
                          'config': prior.read(prior.CONFIG), 'endpoints': endpoints,
                          'starts': ['saved_123_pose', 'identity'], 'max_nfev_per_start': 80,
                          'claim_limit': 'Diagnostic optimizer convergence is NOT production ICP qualification'})


def symmetric_residual(mesh, target, ref, faces, settings, trace=None):
    """Residual norm squared equals the unchanged FULL reported symmetric RMS squared."""
    surface = SurfaceDistance(mesh)
    sim = points(mesh)
    areas = reference_dual_areas(ref, faces)
    weights = areas / areas.sum()
    angle = np.deg2rad(settings['rotation_bound_deg'])
    def residual(x):
        pose = np.r_[u_to_rotvec(x[:3], angle), x[3:]]
        aligned = pose_points(target, pose)
        forward, _ = surface.closest(aligned)
        reverse, _ = cKDTree(aligned).query(sim)
        fwd_mse, rev_mse = float(np.mean(forward**2)), float(np.sum(weights * reverse**2))
        if trace is not None:
            trace.append({'evaluation': len(trace) + 1, 'pose': pose.tolist(),
                          'rms_target_to_sim_mm': float(np.sqrt(fwd_mse)),
                          'rms_sim_to_target_weighted_mm': float(np.sqrt(rev_mse)),
                          'rms_symmetric_mm': float(np.sqrt(.5 * (fwd_mse + rev_mse)))})
        return np.r_[forward * np.sqrt(.5 / len(target)), reverse * np.sqrt(.5 * weights)]
    return residual


def evaluate_pose(mesh, target, ref, faces, settings, pose):
    aligned = pose_points(target, pose)
    distances, _ = SurfaceDistance(mesh).closest(aligned)
    metrics = compute_bidirectional_metrics(mesh, aligned, ref, faces, distances)
    angle = np.deg2rad(settings['rotation_bound_deg'])
    bounds = np.asarray(settings['translation_bounds_mm'])
    metrics.update(pose=np.asarray(pose).tolist(), pose_finite=bool(np.isfinite(pose).all()),
                   rotation_angle_deg=float(np.rad2deg(np.linalg.norm(pose[:3]))),
                   pose_near_bound=bool(np.linalg.norm(pose[:3]) > .95 * angle or np.any(np.abs(pose[3:]) > .95 * bounds)))
    return metrics


def fit_symmetric(mesh, target, ref, faces, settings, pose):
    angle = np.deg2rad(settings['rotation_bound_deg'])
    bounds = np.asarray(settings['translation_bounds_mm'], dtype=float)
    x = np.r_[rotvec_to_u(np.asarray(pose[:3]), angle), pose[3:]]
    trace = []
    residual = symmetric_residual(mesh, target, ref, faces, settings, trace)
    result = least_squares(residual, x, bounds=(np.r_[[-np.inf]*3, -bounds], np.r_[[np.inf]*3, bounds]),
                           max_nfev=80, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    pose = np.r_[u_to_rotvec(result.x[:3], angle), result.x[3:]]
    metrics = evaluate_pose(mesh, target, ref, faces, settings, pose)
    assert np.isclose(2 * result.cost, metrics['rms_symmetric_mm']**2, atol=1e-8, rtol=1e-10)
    return {'metrics': metrics, 'optimizer_success': bool(result.success), 'status': int(result.status),
            'message': str(result.message), 'nfev': int(result.nfev), 'njev': int(result.njev or 0),
            'actual_residual_calls': len(trace), 'optimality': float(result.optimality),
            'diagnostic_interior_success': bool(result.success and metrics['pose_finite'] and not metrics['pose_near_bound']),
            'production_qualification_not_evaluated': True, 'trace': trace}


def run():
    assert os.environ.get('SLURM_JOB_ID'), 'Owned compute allocation required'
    assert not OUTPUT.exists(), 'No automatic restart/overwrite'
    plan = json.loads(PLAN.read_text())
    prior.verify_inputs(plan)
    with np.load(BASE / 'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as data:
        target = data['points_mm'].copy()
    assert target.shape == (14503, 3) and np.isfinite(target).all()
    settings = plan['config']['registration']
    assert settings['icp_iterations'] == 240 and plan['starts'] == ['saved_123_pose', 'identity']
    metadata = prior.read('calibration/data/processed/run6_oriented.json')
    report = {'schema_version': 1, 'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
              'plan_sha256': prior.sha(PLAN), 'scan_metadata': metadata,
              'frame_audit': [], 'pose_trials': [], 'claim_limit': plan['claim_limit']}
    prior.write_new(OUTPUT, report)
    start = time.monotonic()
    for endpoint in plan['endpoints']:
        mesh, ref, faces = prior.load_geometry(endpoint, plan['config'])
        p = points(mesh)
        r, t = rigid_fit(p, ref)
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-10) and np.linalg.det(r) > .999999
        report['frame_audit'].append(dict(endpoint, points=len(p), faces=len(faces),
            centroid_offset_from_reference_mm=(p.mean(0) - ref.mean(0)).tolist(),
            rotation_to_reference_deg=float(np.rad2deg(Rotation.from_matrix(r).magnitude())),
            translation_to_reference_mm=t.tolist(),
            deformation_rms_to_flat_reference_mm=float(np.sqrt(np.mean(np.sum((p @ r + t - ref)**2, axis=1)))),
            reference_extents_mm=np.ptp(ref, axis=0).tolist()))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    endpoint = next(o for o in plan['endpoints'] if o['attempt_id'] == 25)
    mesh, ref, faces = prior.load_geometry(endpoint, plan['config'])
    baseline = prior.read(prior.REUSE)['metrics_240']
    metrics = evaluate_pose(mesh, target, ref, faces, settings, baseline['pose'])
    assert all(abs(metrics[k] - baseline[k]) <= 1e-6 for k in metrics if k.endswith('_mm') and isinstance(metrics[k], float))
    assert metrics['pose_near_bound'] == baseline['pose_near_bound']
    report.update(baseline_metrics=metrics, baseline_matches=True, parameters=endpoint['parameters'])
    for name, seed in zip(plan['starts'], (baseline['pose'], np.zeros(6))):
        print('START', name, flush=True)
        tick = time.monotonic()
        trial = fit_symmetric(mesh, target, ref, faces, settings, seed)
        trial.update(start=name, elapsed_seconds=time.monotonic() - tick)
        report['pose_trials'].append(trial)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k:v for k,v in trial.items() if k not in ('trace', 'metrics')}) + ' RMS=' +
              str(trial['metrics']['rms_symmetric_mm']), flush=True)
    prior.verify_inputs(plan)
    report.update(status='complete', inputs_unchanged=True, elapsed_seconds=time.monotonic() - start)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    assert sys.flags.optimize == 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        run()
