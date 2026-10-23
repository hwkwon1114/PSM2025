#!/usr/bin/env python3
"""Paired-release metrics in material correspondence; no solver invocation.

Scan scoring uses the frozen Attempt 209 comparison targets, not newly tuned poses.
Numerical vertex distributions are descriptive, not independent replicates or CIs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {'run6': 'run6_expanded_bo_continue20_recovery', 'run9': 'run9_transfer_expanded'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align(moving, target, weights):
    cm = weights @ moving
    ct = weights @ target
    u, singular, vt = np.linalg.svd((moving - cm).T @ (weights[:, None] * (target - ct)))
    if singular[1] <= singular[0] * 1e-12:
        raise ValueError('Degenerate rigid alignment')
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return (moving - cm) @ rotation + ct, rotation, ct - cm @ rotation


def area_weights(reference, faces):
    triangles = reference[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('Invalid reference triangles')
    weights = np.zeros(len(reference))
    for k in range(3):
        np.add.at(weights, faces[:, k], areas / 3)
    if np.any(weights <= 0):
        raise ValueError('Unreferenced vertices')
    return weights / weights.sum()


def analyze(folder, name, score_scans=False):
    inputs = {}
    def read(filename):
        path = folder / filename
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())
    execution = read('execution.json')
    if execution['status'] != 'completed':
        raise ValueError('Execution not complete')
    parent_envelope = read('paired_clamped_checkpoint.json')
    parent = parent_envelope['payload']
    pair = read('paired_release_result.json')['payload']
    if not pair['complete'] or pair['clamped_checkpoint_checksum'] != parent_envelope['checksum']:
        raise ValueError('Invalid parent link')
    uv = np.asarray(parent['identity']['uv'], float) * 1000
    nv = len(uv)
    faces = np.asarray(parent['identity']['faces'], int)
    def xyz(state):
        p = np.asarray(state, float).reshape(-1)
        if not np.isfinite(p).all() or len(p) < 3 * nv:
            raise ValueError('Invalid state')
        return p[:3 * nv].reshape(3, nv).T * 1000
    reference = xyz(parent['identity']['rest'])
    clamped = xyz(parent['state'])
    weights = area_weights(reference, faces)
    states, diagnostics = {}, {}
    starts = []
    for index, branch in enumerate(('putong', 'free')):
        end = read('release_' + branch + '.json')['payload']
        if end != pair['endpoints'][index] or end['parent_checksum'] != parent_envelope['checksum']:
            raise ValueError('Endpoint mismatch')
        proof = end['proof']
        if not (proof['accepted'] and proof['termination'] == 'gradient_target' and
                proof['curvature'] == 'positive_pivots' and
                0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= 5e-14 and
                end['fixed_coordinates_unchanged'] and end['target_history_unchanged']):
            raise ValueError('Unaccepted release')
        mask = np.asarray(end['vertex_mask'], bool)
        if mask.shape != (nv, 3):
            raise ValueError('Invalid mask')
        p = xyz(end['state'])
        if not np.array_equal(p[mask], clamped[mask]):
            raise ValueError('Fixed coordinates changed')
        starts.append(end['start_state_checksum'])
        states[branch] = p
        raw = np.asarray(end['unmasked_gradient'], float).reshape(-1)
        if not np.isfinite(raw).all():
            raise ValueError('Invalid unmasked gradient')
        fixed = raw[:3 * nv].reshape(3, nv).T[mask]
        diagnostics[branch] = {
            'energy_normalized': proof['energy'],
            'energy_change_from_clamped': proof['energy'] - parent['proof']['energy'],
            'solver_gradient_norm': proof['gradient_norm'],
            'unmasked_generalized_gradient_norm': end['unmasked_gradient_norm'],
            'fixed_coordinate_gradient_norm': float(np.linalg.norm(fixed)),
            'fixed_dofs': int(mask.sum()),
            'remaining_rigid_modes_start': end['remaining_rigid_modes_start'],
            'remaining_rigid_modes_end': end['remaining_rigid_modes_end'],
            'gauge_geometry_start': end['gauge_geometry_start'],
            'gauge_geometry_end': end['gauge_geometry_end'],
            'local_curvature_check': proof['curvature'],
        }
    if starts[0] != starts[1]:
        raise ValueError('Different release start states')
    putong, rotation, translation = align(states['putong'], states['free'], weights)
    delta = putong - states['free']
    distances = np.linalg.norm(delta, axis=1)
    result = {
        'case': name, 'vertices': nv, 'triangles': len(faces), 'cycles': pair['cycles'],
        'area_weighted_rms_difference_mm': float(np.sqrt(weights @ distances ** 2)),
        'vertex_rms_difference_mm': float(np.sqrt(np.mean(distances ** 2))),
        'maximum_difference_mm': float(distances.max()),
        'vertex_quantiles_mm': dict(zip(('q50', 'q90', 'q99'), np.quantile(distances, [.5, .9, .99]).tolist())),
        'energy_putong_minus_free_normalized': diagnostics['putong']['energy_normalized'] - diagnostics['free']['energy_normalized'],
        'putong_to_free_rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
        'diagnostics': diagnostics,
        'scope': 'Full-vertex deterministic paired comparison; spatial vertices are not independent replicates. E=1 energy normalization; no physical-force claim.',
    }
    if score_scans:
        if name not in BASELINES:
            raise ValueError('Only verified Run 6/9 frozen targets support scan scoring')
        sys.path.insert(0, str(ROOT))
        from calibration.ewcal.geometry import read_mesh, points
        from calibration.ewcal.material_frame import score_material_frame
        from vtk.util.numpy_support import vtk_to_numpy
        contract_path = ROOT / 'run/two_gripper_comparison/checks/frozen_scan_contract_vtp.json'
        contract = json.loads(contract_path.read_text())[name]
        if not contract['matches_1e_8_mm']:
            raise ValueError('Frozen scan contract not reproduced')
        for source, expected in contract['sha256'].items():
            if digest(Path(source)) != expected:
                raise ValueError('Frozen scoring source changed: ' + source)
        inputs[str(contract_path)] = digest(contract_path)
        baseline = ROOT / 'run' / BASELINES[name] / 'evaluations/attempt_0209'
        target_path = baseline / 'case_00/comparison.npz'
        inputs[str(target_path)] = digest(target_path)
        # Frozen comparison target is already oriented/centered; scoring only
        # removes its centroid again, not a new optimized scan registration.
        with np.load(target_path) as data:
            target = data['target_mm'].copy()
        baseline_record = json.loads((baseline / 'result.json').read_text())
        inputs[str(baseline / 'result.json')] = digest(baseline / 'result.json')
        if baseline_record['cases'][0]['metrics']['scoring_method'] != 'material_reference':
            raise ValueError('Frozen baseline scoring contract not material_reference')
        for branch in ('putong', 'free'):
            # Match the baseline's serialized VTP geometry for scan scoring.
            # Paired intrinsic differences above still use full JSON precision.
            mesh_path = folder / ('release_' + branch + '.vtp')
            mesh = read_mesh(mesh_path, scale=1000)
            mesh_points = points(mesh)
            vtk_faces = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)
            if not (np.all(vtk_faces[:, 0] == 3) and np.array_equal(vtk_faces[:, 1:], faces)):
                raise ValueError('Release VTP topology mismatch')
            if not np.allclose(mesh_points, states[branch], rtol=4*np.finfo(np.float32).eps, atol=1e-9):
                raise ValueError('Release VTP differs beyond serialization precision')
            inputs[str(mesh_path)] = digest(mesh_path)
            metrics, _, _, _, _ = score_material_frame(mesh, target, reference, faces)
            diagnostics[branch]['scan_metrics'] = metrics
            diagnostics[branch]['scan_mesh_serialization_max_difference_mm'] = float(
                np.max(np.linalg.norm(mesh_points - states[branch], axis=1)))
    result['input_sha256'] = inputs
    arrays = dict(reference_mm=reference, material_uv_mm=uv, faces=faces, weights=weights,
                  free_mm=states['free'], putong_aligned_mm=putong,
                  delta_mm=delta, distance_mm=distances, clamped_mm=clamped)
    return result, arrays


def self_test():
    p = np.array([[0., 0, 0], [1., 0, .2], [0, 2., .3], [1, 2, .5]])
    angle = .6
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    q = p @ r + [4, -2, 7]
    fitted, rotation, _ = align(p, q, np.full(4, .25))
    assert np.max(np.abs(fitted - q)) < 1e-12
    assert abs(np.linalg.det(rotation) - 1) < 1e-12
    print('Proper rigid-alignment self-test passed (synthetic math fixture only)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'run/two_gripper_comparison')
    parser.add_argument('--cases', nargs='+', default=['run6', 'run9'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--score-scans', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error('--output is required')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in args.cases:
        result, arrays = analyze(args.root / name, name, args.score_scans)
        np.savez_compressed(args.output / (name + '_full_field.npz'), **arrays)
        results.append(result)
    report = {'analysis_script_sha256': digest(Path(__file__)), 'numpy_version': np.__version__,
              'cases': results, 'alignment': 'Proper area-weighted Kabsch, matching material vertices; no reflection or scaling'}
    (args.output / 'paired_release_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for case in results:
        print(case['case'], 'RMS(mm)=', case['area_weighted_rms_difference_mm'], 'max(mm)=', case['maximum_difference_mm'])


if __name__ == '__main__':
    main()
