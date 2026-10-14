"""Align run9 by sheet edges, adopting run6's face-turnover convention.

Run with --self-test for bounded synthetic rotation checks. No shell solves.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage, stats

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.geometry import points, read_mesh
from calibration.ewcal.prepare import prepare_case


def edge_angles(xy, spacing=1.):
    """Fit x(y) for long-side boundaries of the largest dense component."""
    keys = np.floor(xy / spacing).astype(int)
    origin = keys.min(axis=0)
    keys -= origin
    shape = tuple(keys.max(axis=0) + 1)
    if max(shape) > 2000:
        raise ValueError('Unexpected scan extent for occupancy diagnostic')
    counts = np.zeros(shape, dtype=int)
    np.add.at(counts, (keys[:, 0], keys[:, 1]), 1)
    labels, n = ndimage.label(counts >= 5, structure=np.ones((3, 3)))
    if not n:
        raise ValueError('No dense scan component')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    component = int(np.argmax(sizes))
    ix, iy = np.where(labels == component)
    ylo, yhi = iy.min(), iy.max()
    rows = np.arange(int(np.ceil(ylo + .2*(yhi-ylo))), int(np.floor(yhi - .2*(yhi-ylo))) + 1)
    ys, left, right = [], [], []
    for j in rows:
        columns = ix[iy == j]
        if not len(columns):
            continue
        ys.append((j + origin[1] + .5)*spacing)
        left.append((columns.min() + origin[0] + .5)*spacing)
        right.append((columns.max() + origin[0] + .5)*spacing)
    if len(ys) < 20:
        raise ValueError('Too few edge bins')
    fits = [stats.theilslopes(values, ys) for values in (left, right)]
    angles = np.degrees(np.arctan([f[0] for f in fits]))
    return {'angles_deg': angles.tolist(), 'mean_angle_deg': float(angles.mean()),
            'rows': len(ys), 'component_cells': int(sizes[component]),
            'spacing_mm': spacing, 'min_vertices_per_cell': 5,
            'central_long_axis_fraction': .6,
            'angle_disagreement_deg': float(abs(angles[0]-angles[1]))}


def yaw_matrix(angle_deg):
    a = np.deg2rad(angle_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def self_test():
    # Use the specimen's scale: a 1mm boundary bin on a 60mm-long toy
    # rectangle can alone induce >1deg slope quantization.
    x, y = np.meshgrid(np.arange(-100, 100, .25), np.arange(-150, 150, .25))
    flat = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    for angle in (-12., 7., 0.):
        rotated = flat @ yaw_matrix(angle).T
        estimate = edge_angles(rotated[:, :2])
        # x(y) slope has the opposite sign of the applied yaw.
        assert abs(estimate['mean_angle_deg'] + angle) < .5, estimate
        aligned = rotated @ yaw_matrix(estimate['mean_angle_deg']).T
        residual = edge_angles(aligned[:, :2])
        assert max(abs(a) for a in residual['angles_deg']) < .5, residual
        assert np.allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(aligned, axis=1))
    print('PASS: three known yaw fixtures, residual edge angles and rigid length preservation')


def run():
    config_path = BASE / 'calibration/configs/run9.json'
    config = json.loads(config_path.read_text())
    original_meta_path = BASE / 'calibration/data/processed/run9.json'
    original_meta = json.loads(original_meta_path.read_text())
    raw_path = Path(original_meta['source'])
    original_npz = BASE / 'calibration/data/processed/run9.npz'
    dest = BASE / 'calibration/data/processed/run9_oriented.npz'
    out = BASE / 'run/run9_preprocessing'
    report_path = out / 'edge_alignment.json'
    image_path = out / 'run9_edge_alignment.png'
    for path in [dest, dest.with_suffix('.json'), dest.with_suffix('.png'), report_path, image_path]:
        if path.exists():
            raise FileExistsError(path)
    pins = {str(p): digest(p) for p in [raw_path, original_npz, original_meta_path, config_path]}
    assert pins[str(raw_path)] == original_meta['source_sha256']
    assert pins[str(original_npz)] == original_meta['target_sha256']
    raw = np.unique(points(read_mesh(raw_path)), axis=0)
    transform = original_meta['transform']
    basis = np.array(transform['basis_columns'])
    q = (raw - np.array(transform['origin_scanner_mm'])) @ basis - np.array(transform['shift_local_mm'])
    turnover = np.diag([1., -1., -1.])
    turned = q @ turnover
    estimates = [edge_angles(turned[:, :2], spacing) for spacing in (1., 2.)]
    for e in estimates:
        if e['angle_disagreement_deg'] > 1.:
            raise ValueError(f'Edge disagreement: {e}')
    angle = estimates[0]['mean_angle_deg']
    if abs(angle - estimates[1]['mean_angle_deg']) > .5:
        raise ValueError(f'Yaw sensitivity exceeds .5deg: {estimates}')
    rotation = yaw_matrix(angle)
    aligned = turned @ rotation.T
    long_residual = edge_angles(aligned[:, :2])
    short_residual = edge_angles(aligned[:, [1, 0]])
    if max(abs(a) for e in (long_residual, short_residual) for a in e['angles_deg']) > .5:
        save_json(report_path, {
            'status': 'alignment_gate_failed_no_target_written',
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'inputs_sha256': pins, 'script_sha256': digest(Path(__file__)),
            'estimates': estimates, 'yaw_deg_after_turnover': angle,
            'long_edges_residual': long_residual, 'short_edges_residual': short_residual,
            'limit_deg': .5,
            'limits': 'Proposed rotation is not an accepted target. No target or config written; no curvature fitting. Face sign adopts researcher-selected run6 convention.'})
        raise ValueError(f'Residual edge tilt exceeds .5deg: {long_residual}, {short_residual}')
    composed_basis = basis @ turnover @ rotation.T
    assert np.allclose(composed_basis.T @ composed_basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(composed_basis) > .999999
    assert np.allclose((q + np.array(transform['shift_local_mm'])) @ turnover @ rotation.T,
                       (raw - np.array(transform['origin_scanner_mm'])) @ composed_basis)
    case = copy.deepcopy(config['cases'][0])
    case['target'] = str(dest.relative_to(BASE))
    case['scan_orientation'] = {'flip_z': False, 'flip_xy': False, 'basis_columns': composed_basis.tolist()}
    meta = prepare_case(config, case)
    meta.update({'target_npz': case['target'],
                 'orientation_notes': f'Canonical x-turnover [1,-1,-1] followed by {angle:+.9f} deg z-yaw from run9 sheet edges. Full composed basis is recorded in transform and orientation.',
                 'face_convention': 'Researcher-adopted run6 x-turnover convention; physical face not independently labelled in STL.',
                 'edge_alignment': {'estimates': estimates, 'yaw_deg_after_turnover': angle,
                                    'long_edges_residual': long_residual, 'short_edges_residual': short_residual}})
    save_json(dest.with_suffix('.json'), meta)
    with np.load(dest, allow_pickle=False) as data:
        cloud = data['points_mm']
    path_file = BASE / 'calibration/data/raw/run9_zigzag_path.csv'
    with path_file.open() as f:
        path = np.array([[float(r[k]) for k in ('x_mm', 'y_mm')] for r in csv.DictReader(f)])
    path[:, 1] -= 93.75
    assert np.all(np.abs(path) <= [98, 148])
    # Plot before/after yaw with the same transformed-face height scale.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    idx = np.random.default_rng(2026).choice(len(turned), 40000, replace=False)
    lo, hi = cloud[:, 2].min(), cloud[:, 2].max()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, data, title in [(axes[0, 0], turned[idx], 'Before edge alignment'), (axes[0, 1], cloud, 'Edge-aligned scan and nominal path')]:
        artist = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=2, cmap='viridis', vmin=lo, vmax=hi)
        ax.set(xlabel='Nominal x (mm)', ylabel='Nominal y (mm)', title=title, aspect='equal')
        ax.plot([-98, 98, 98, -98, -98], [-148, -148, 148, 148, -148], color='#D55E00', linestyle='--', lw=1)
        ax.set_xlim(-140, 140); ax.set_ylim(-190, 190)
        fig.colorbar(artist, ax=ax, label='Height (mm)', extend='both')
    axes[0, 1].plot(path[:, 0], path[:, 1], color='#111111', lw=.8, label='Nominal CSV path')
    axes[0, 1].plot(path[0, 0], path[0, 1], marker='*', color='#D55E00', ms=10, label='Path start')
    axes[0, 1].legend(loc='lower left', fontsize=8)
    for ax, i, label in [(axes[1, 0], 0, 'Nominal x (mm)'), (axes[1, 1], 1, 'Nominal y (mm)')]:
        ax.scatter(cloud[:, i], cloud[:, 2], s=2, color='#0072B2')
        ax.set(xlabel=label, ylabel='Height (mm)', title='Edge-aligned scan projection')
        ax.set_ylim(lo-.5, hi+.5)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        b = text.get_window_extent(renderer)
        if b.width and b.height:
            assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height, text.get_text()
    fig.savefig(image_path, dpi=160); plt.close(fig)
    for p, h in pins.items():
        assert digest(Path(p)) == h, p
    report = {'status': 'edge_alignment_complete', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'inputs_sha256': pins, 'script_sha256': digest(Path(__file__)),
              'toolpath_sha256': digest(path_file), 'estimates': estimates,
              'yaw_deg_after_turnover': angle, 'long_edges_residual': long_residual,
              'short_edges_residual': short_residual, 'prepared_metadata': meta,
              'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
              'limits': 'No curve fitting or mechanics. Face sign follows researcher-adopted run6 convention, not physical STL landmarks. Geometric edge matching does not identify a labelled start corner. Occupancy threshold/connected component only selects points for edge estimation; existing prepare_case filters generate the target.'}
    save_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run()
