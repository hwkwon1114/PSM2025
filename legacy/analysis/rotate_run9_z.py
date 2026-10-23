"""Run9 z-only edge alignment and visual diagnostic; deliberately no face flip."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.orient_run9_edges import edge_angles, yaw_matrix
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.geometry import points, read_mesh
from calibration.ewcal.prepare import prepare_case

parser = argparse.ArgumentParser()
parser.add_argument('--render-only', action='store_true', help='Reuse the saved z-only target after a plotting failure')
args = parser.parse_args()
out = BASE / 'run/run9_preprocessing'
image_path = out / 'run9_z_rotation.png'
report_path = out / 'z_rotation.json'
dest = BASE / 'calibration/data/processed/run9_z_aligned.npz'
checked_outputs = [image_path, report_path] if args.render_only else [image_path, report_path, dest, dest.with_suffix('.json'), dest.with_suffix('.png')]
for p in checked_outputs:
    if p.exists():
        raise FileExistsError(p)
config_path = BASE / 'calibration/configs/run9.json'
meta_path = BASE / 'calibration/data/processed/run9.json'
config = json.loads(config_path.read_text())
meta = json.loads(meta_path.read_text())
source = Path(meta['source'])
original_target = BASE / 'calibration/data/processed/run9.npz'
path_file = BASE / 'calibration/data/raw/run9_zigzag_path.csv'
pins = {str(p): digest(p) for p in [config_path, meta_path, source, original_target, path_file]}
assert digest(source) == meta['source_sha256']
assert digest(original_target) == meta['target_sha256']
raw = np.unique(points(read_mesh(source, meta['scan_scale_to_mm'])), axis=0)
t = meta['transform']
basis = np.array(t['basis_columns'])
q = (raw - np.array(t['origin_scanner_mm'])) @ basis - np.array(t['shift_local_mm'])
# A long edge x(y) needs +atan(dx/dy); a short edge y(x) needs -atan(dy/dx).
long_edges = edge_angles(q[:, :2])
short_edges = edge_angles(q[:, [1, 0]])
angle = .5 * (long_edges['mean_angle_deg'] - short_edges['mean_angle_deg'])
rotation = yaw_matrix(angle)
rotated = q @ rotation.T
composed_basis = basis @ rotation.T
assert np.allclose(composed_basis.T @ composed_basis, np.eye(3), atol=1e-12)
assert np.linalg.det(composed_basis) > .999999
assert np.allclose(q[:, 2], rotated[:, 2])  # no normal/face flip
assert np.allclose(np.linalg.norm(q, axis=1), np.linalg.norm(rotated, axis=1))
residual_long = edge_angles(rotated[:, :2])
residual_short = edge_angles(rotated[:, [1, 0]])
case = copy.deepcopy(config['cases'][0])
case['target'] = str(dest.relative_to(BASE))
case['scan_orientation'] = {'flip_z': False, 'flip_xy': False, 'basis_columns': composed_basis.tolist()}
if args.render_only:
    prepared = json.loads(dest.with_suffix('.json').read_text())
    assert digest(dest) == prepared['target_sha256']
    assert np.allclose(prepared['transform']['basis_columns'], composed_basis)
    assert abs(prepared['z_alignment']['yaw_deg'] - angle) < 1e-9
else:
    prepared = prepare_case(config, case)
prepared.update({'target_npz': case['target'],
                 'orientation_notes': f'Z-only yaw {angle:+.9f} deg estimated from all four run9 sheet edges; no turnover or face flip. Recentered using existing preprocessing.',
                 'z_alignment': {'yaw_deg': angle, 'long_edges_before': long_edges,
                                 'short_edges_before': short_edges,
                                 'long_edges_after': residual_long, 'short_edges_after': residual_short},
                 'face_convention': 'Unchanged from preliminary run9 frame; pending researcher visual inspection.'})
if not args.render_only:
    save_json(dest.with_suffix('.json'), prepared)
with np.load(dest, allow_pickle=False) as f:
    cloud = f['points_mm']
with path_file.open() as f:
    path = np.array([[float(r[k]) for k in ('x_mm', 'y_mm')] for r in csv.DictReader(f)])
path[:, 1] -= 93.75
assert np.all(np.abs(path) <= [98, 148])
idx = np.random.default_rng(2026).choice(len(q), 40000, replace=False)
# Apply the newly saved complete transform to raw vertices, including centering.
nt = prepared['transform']
aligned = (raw - np.array(nt['origin_scanner_mm'])) @ np.array(nt['basis_columns']) - np.array(nt['shift_local_mm'])
# Centering may shift the height zero, but does not change physical height differences.
before_display = q[idx].copy()
before_display[:, 2] += np.median(aligned[:, 2]) - np.median(q[:, 2])
lo, hi = cloud[:, 2].min(), cloud[:, 2].max()
fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)
for ax, data, title in zip(axes, [before_display, aligned[idx], cloud],
                          ['Before z-rotation', 'After z-rotation (uncropped)', 'Prepared scan and nominal path']):
    artist = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=2, cmap='viridis', vmin=lo, vmax=hi)
    ax.set(xlabel='Nominal x (mm)', ylabel='Nominal y (mm)', title=title, aspect='equal')
    ax.set_xlim(-140, 140); ax.set_ylim(-190, 190)
    ax.set_xticks([-100, 0, 100]); ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    ax.plot([-98, 98, 98, -98, -98], [-148, -148, 148, 148, -148],
            color='#D55E00', linestyle='--', lw=1, label='Preprocessing crop')
axes[0].legend(loc='lower left', fontsize=8)
axes[2].plot(path[:, 0], path[:, 1], color='#111111', lw=.8, label='Nominal CSV path')
axes[2].plot(path[0, 0], path[0, 1], marker='*', color='#D55E00', ms=10, label='Path start')
axes[2].legend(loc='lower left', fontsize=8)
fig.colorbar(artist, ax=axes.tolist(), label='Height (mm)', extend='both', shrink=.8,
             ticks=np.linspace(lo, hi, 5), format='%.1f')
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
report = {'status': 'z_only_visual_diagnostic_complete', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
          'inputs_sha256': pins, 'script_sha256': digest(Path(__file__)),
          'yaw_deg': angle, 'long_edges_before': long_edges, 'short_edges_before': short_edges,
          'long_edges_after': residual_long, 'short_edges_after': residual_short,
          'prepared_metadata': prepared, 'render_only': args.render_only, 'raw_display_subset': 40000, 'seed': 2026,
          'raw_fraction_outside_xy_crop_after': float(np.mean(np.any(np.abs(aligned[:, :2]) > [98, 148], axis=1))),
          'raw_display_outside_plot_limits_after': int(np.sum(np.any(np.abs(aligned[idx, :2]) > [140, 190], axis=1))),
          'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
          'limits': 'Z-only proper rigid rotation from geometric boundaries, no curvature fitting, flip, stretching, calibration or config update. Target recentered with original preprocessing rules. Nominal path start is not a measured material landmark. Approximate bin coverage is not physical sheet-area coverage.'}
save_json(report_path, report)
print(json.dumps(report, indent=2))
