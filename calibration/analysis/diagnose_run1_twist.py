"""Read-only Run 1 residual and yaw diagnostics; not the calibration metric.

Input is a snapshot of finished fresh-search results and two archived controls.
No solver, Run 2 data, archived results, or running campaign state is changed.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.geometry import read_mesh, points, shared_footprint_faces

CAMPAIGN = ROOT / 'run/fresh_lhs_joint4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basis(xy):
    u, v = xy[:, 0] / 100.0, xy[:, 1] / 150.0
    return np.column_stack((np.ones(len(u)), u, v, u * u, u * v, v * v))


def quadratic(xy, height):
    co = np.linalg.lstsq(basis(xy), height, rcond=None)[0]
    # Principal height-curvature direction in the public x–z plane.
    hessian = np.array([[2 * co[3] / 100**2, co[4] / (100 * 150)],
                        [co[4] / (100 * 150), 2 * co[5] / 150**2]])
    eig, vec = np.linalg.eigh(hessian)
    major = vec[:, np.argmax(np.abs(eig))]
    angle = (np.degrees(np.arctan2(major[1], major[0])) + 90) % 180 - 90
    return {'coefficients_mm': co.tolist(), 'principal_axis_deg_from_x': float(angle),
            'quadratic_hessian_eigenvalues_per_mm': eig.tolist()}


def load_case(folder):
    path = folder / 'comparison.npz'
    with np.load(path, allow_pickle=False) as f:
        target, closest, distance = (f[k].copy() for k in ('target_mm', 'closest_mm', 'distance_mm'))
    assert target.shape == closest.shape and target.shape[1] == 3 and distance.shape == (len(target),)
    assert np.isfinite(target).all() and np.isfinite(closest).all() and np.isfinite(distance).all()
    assert np.allclose(np.linalg.norm(target - closest, axis=1), distance, atol=1e-8, rtol=1e-8)
    return target, closest, distance


def residual_info(target, closest, distance):
    a = basis(target[:, :2])
    residual = target[:, 2] - closest[:, 2]
    co = np.linalg.lstsq(a, residual, rcond=None)[0]
    uv = a[:, 4]
    uv_amp = float(uv @ residual / (uv @ uv))
    return {'forward_rms_mm': float(np.sqrt(np.mean(distance**2))),
            'height_rms_mm': float(np.sqrt(np.mean(residual**2))),
            'quadratic_xy_coefficient_mm': float(co[4]),
            'quadratic_remainder_rms_mm': float(np.sqrt(np.mean((residual - a @ co)**2))),
            'bilinear_only_coefficient_mm': uv_amp,
            'bilinear_removed_height_rms_mm': float(np.sqrt(np.mean((residual - uv_amp * uv)**2))),
            'scan_quadratic': quadratic(target[:, :2], target[:, 2]),
            'closest_quadratic': quadratic(target[:, :2], closest[:, 2])}, residual


def yaw_diagnostic(folder, target):
    # All angles use the same measured interior points and exact saved mesh faces.
    mesh = read_mesh(folder / 'prediction_mm.vtp')
    p = points(mesh)
    conn = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    tri = mtri.Triangulation(p[:, 0], p[:, 1], conn)
    interpolation = mtri.LinearTriInterpolator(tri, p[:, 2])
    xy = target[:, :2]
    roi = (np.abs(xy[:, 0]) <= 75) & (np.abs(xy[:, 1]) <= 115)
    angles = np.linspace(-8, 8, 65)
    predicted = []
    for theta in np.deg2rad(angles):
        c, s = np.cos(theta), np.sin(theta)
        # The mesh is rotated theta around its height axis; query its inverse.
        x = c * xy[:, 0] + s * xy[:, 1]
        y = -s * xy[:, 0] + c * xy[:, 1]
        predicted.append(interpolation(x, y).filled(np.nan))
    predicted = np.asarray(predicted)
    common = roi & np.all(np.isfinite(predicted), axis=0)
    if common.sum() < 1000:
        raise ValueError('insufficient common footprint for the yaw diagnostic')
    residual = target[common, 2][None, :] - predicted[:, common]
    raw = np.sqrt(np.mean(residual**2, axis=1))
    centered = np.sqrt(np.mean((residual - residual.mean(axis=1, keepdims=True))**2, axis=1))
    i = int(np.argmin(centered)); k = int(np.flatnonzero(angles == 0)[0])
    best_residual = residual[i] - residual[i].mean()
    return {'common_points': int(common.sum()), 'all_scan_points': len(target),
            'fraction_all_scan_points': float(common.mean()),
            'roi_internal_mm': {'absolute_u_max': 75, 'absolute_v_max': 115},
            'angle_grid_deg': angles.tolist(), 'height_rms_centered_by_angle_mm': centered.tolist(),
            'zero_yaw_raw_height_rms_mm': float(raw[k]),
            'zero_yaw_centered_height_rms_mm': float(centered[k]),
            'posthoc_min_yaw_deg': float(angles[i]),
            'posthoc_min_raw_height_rms_mm': float(raw[i]),
            'posthoc_min_centered_height_rms_mm': float(centered[i]),
            'posthoc_min_at_grid_boundary': bool(i in (0, len(angles) - 1)),
            'posthoc_offset_mm': float(residual[i].mean()),
            'scan_quadratic_fixed_roi': quadratic(xy[common], target[common, 2]),
            'simulation_quadratic_fixed_roi_zero_yaw': quadratic(xy[common], predicted[k, common]),
            'simulation_quadratic_fixed_roi_posthoc_yaw': quadratic(xy[common], predicted[i, common]),
            'posthoc_height_residual_quadratic_rms_mm': float(np.sqrt(np.mean(best_residual**2))),
            'mesh_sha256': digest(folder / 'prediction_mm.vtp')}


def scan_area_weighted_rms(target, distance):
    xy = target[:, :2]
    f = shared_footprint_faces(xy)
    a, b, c = xy[f[:, 0]], xy[f[:, 1]], xy[f[:, 2]]
    twice = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                   (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    weights = np.zeros(len(xy))
    for col in range(3):
        np.add.at(weights, f[:, col], twice / 6)
    weights /= weights.sum()
    return {'unweighted_forward_rms_mm': float(np.sqrt(np.mean(distance**2))),
            'area_weighted_forward_rms_mm': float(np.sqrt(np.sum(weights * distance**2))),
            'zero_area_points': int(np.sum(weights == 0)),
            'relative_weight_p05_p50_p95': np.quantile(weights * len(weights), [.05, .50, .95]).tolist()}


def main(output):
    # Small controls protect against an x-z normalization or yaw-sign mistake.
    x, y = np.meshgrid(np.linspace(-100, 100, 13), np.linspace(-150, 150, 15))
    xy = np.column_stack((x.ravel(), y.ravel()))
    co = np.array([2, 0.2, -0.1, -4, 1.7, -0.5])
    assert np.allclose(quadratic(xy, basis(xy) @ co)['coefficients_mm'], co, atol=1e-12)
    assert np.allclose(basis(np.array([[100, 150], [-100, 150]]))[:, 4], [1, -1])

    paths = sorted((CAMPAIGN / 'results').glob('a*.json'))
    if not paths:
        raise ValueError('no completed fresh-search result files')
    saved, invalid, raw_residuals = [], [], {}
    reference_target = None
    for path in paths:
        record = json.loads(path.read_text())
        if not record['valid']:
            invalid.append({'attempt': record['attempt'], 'failure_class': record.get('failure_class')})
            continue
        folder = CAMPAIGN / f"candidates/a{record['attempt']:04d}/evaluations/attempt_0001/case_00"
        target, closest, distance = load_case(folder)
        if reference_target is None:
            reference_target = target
        assert target.shape == reference_target.shape and np.allclose(target, reference_target, atol=1e-9, rtol=0)
        info, raw = residual_info(target, closest, distance)
        assert abs(record['case_rms_mm']['run6'] - record['case_metrics']['run6']['rms_symmetric_mm']) < 1e-10
        item = {'attempt': record['attempt'], 'wave': record['wave'],
                'joint_rms_mm': record['objective_mm'], 'run1_rms_mm': record['case_rms_mm']['run6'],
                'parameters': record['parameters'], 'residual': info,
                'result_sha256': digest(path), 'comparison_sha256': digest(folder / 'comparison.npz')}
        saved.append(item)
        raw_residuals[record['attempt']] = raw
    best_joint = min(saved, key=lambda r: r['joint_rms_mm'])
    best_run1 = min(saved, key=lambda r: r['run1_rms_mm'])
    for r in saved:
        r['height_residual_correlation_with_best_joint'] = float(
            np.corrcoef(raw_residuals[r['attempt']], raw_residuals[best_joint['attempt']])[0, 1])
    controls = {
        'fresh_best_joint': CAMPAIGN / f"candidates/a{best_joint['attempt']:04d}/evaluations/attempt_0001/case_00",
        'fresh_best_run1': CAMPAIGN / f"candidates/a{best_run1['attempt']:04d}/evaluations/attempt_0001/case_00",
        'previous_joint_best': ROOT / 'run/joint_stage4_reduced4/candidates/a0016/evaluations/attempt_0001/case_00',
        'previous_baseline': ROOT / 'run/joint_stage2/arm_A/candidates/a0001/evaluations/attempt_0001/case_00',
    }
    details = {}
    for label, folder in controls.items():
        t, c, d = load_case(folder)
        assert np.allclose(t, reference_target, atol=1e-9, rtol=0)
        info, _ = residual_info(t, c, d)
        details[label] = {'path': str(folder.relative_to(ROOT)), 'residual': info,
                          'weighting': scan_area_weighted_rms(t, d),
                          'yaw': yaw_diagnostic(folder, t), 'comparison_sha256': digest(folder / 'comparison.npz')}
    out = {'created_utc': datetime.now(timezone.utc).isoformat(),
           'scope': 'Post-hoc diagnostic on completed training Run 1 only; no solver, no Run 2, no campaign changes.',
           'coordinates': 'Internal u,v,h correspond to public x,z-65,y; all values mm.',
           'basis': '[1, x/100, (z-65)/150, (x/100)^2, (x/100)((z-65)/150), ((z-65)/150)^2]',
           'limits': 'Closest-point height residual and counterfactual fixed-ROI yaw-height RMS are not the optimized 3-D symmetric metric. Post-hoc yaw does not establish physically valid registration.',
           'source_script_sha256': digest(Path(__file__)), 'result_snapshot_files': [p.name for p in paths],
           'valid_count': len(saved), 'invalid': invalid, 'best_joint_attempt': best_joint['attempt'],
           'best_run1_attempt': best_run1['attempt'], 'records': saved, 'controls': details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(output), 'valid': len(saved), 'invalid': len(invalid),
                      'best_joint': best_joint['attempt'], 'best_run1': best_run1['attempt'],
                      'controls': {k: {'yaw_deg': v['yaw']['posthoc_min_yaw_deg'],
                                        'height_rms_zero_mm': v['yaw']['zero_yaw_centered_height_rms_mm'],
                                        'height_rms_best_mm': v['yaw']['posthoc_min_centered_height_rms_mm']}
                                   for k, v in details.items()}}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
