"""Solver-free run9 scan/nominal-path diagnostic; no orientation adjustment."""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.ewcal.geometry import read_mesh, points
from calibration.ewcal.common import digest

out = BASE / 'run/run9_preprocessing'
image = out / 'run9_scan_toolpath_orientation.png'
report = out / 'orientation_diagnostic.json'
if image.exists() or report.exists():
    raise FileExistsError('Preserve prior diagnostics; choose a new output destination')
meta_path = BASE / 'calibration/data/processed/run9.json'
meta = json.loads(meta_path.read_text())
source = Path(meta['source'])
target = BASE / 'calibration/data/processed/run9.npz'
assert digest(source) == meta['source_sha256']
assert digest(target) == meta['target_sha256']
raw = np.unique(points(read_mesh(source, meta['scan_scale_to_mm'])), axis=0)
t = meta['transform']
q = (raw - np.array(t['origin_scanner_mm'])) @ np.array(t['basis_columns']) - np.array(t['shift_local_mm'])
with np.load(target, allow_pickle=False) as data:
    cloud = data['points_mm']
assert np.isfinite(q).all() and np.isfinite(cloud).all()
path_file = BASE / 'calibration/data/raw/run9_zigzag_path.csv'
with path_file.open() as f:
    path = np.array([[float(r[k]) for k in ('x_mm', 'y_mm', 'z_mm')] for r in csv.DictReader(f)])
path[:, 1] -= 93.75
path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0, axis=1)]]
assert path.shape == (33, 3)
assert np.allclose(np.ptp(path, axis=0), [98.1848, 187.5, 0], atol=1e-4)
# Error-independent seeded display subset only; summaries use all unique vertices.
idx = np.random.default_rng(2026).choice(len(q), min(40000, len(q)), replace=False)
display = q[idx]
lo, hi = np.quantile(cloud[:, 2], [0, 1])
fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
for ax, data, title in [(axes[0, 0], display, 'Uncropped scan'), (axes[0, 1], cloud, 'Prepared scan and nominal path')]:
    artist = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=2, cmap='viridis', vmin=lo, vmax=hi)
    ax.set(xlabel='Nominal x (mm)', ylabel='Nominal y (mm)', title=title, aspect='equal')
    ax.plot([-98, 98, 98, -98, -98], [-148, -148, 148, 148, -148], color='#D55E00', linestyle='--', lw=1, label='Preprocessing crop')
    ax.set_xlim(-140, 140)
    ax.set_ylim(-190, 190)
    fig.colorbar(artist, ax=ax, label='Height (mm)', extend='both')
axes[0, 0].legend(loc='lower left', fontsize=8)
axes[0, 1].plot(path[:, 0], path[:, 1], color='#111111', lw=.8, label='Nominal CSV path')
axes[0, 1].plot(path[0, 0], path[0, 1], marker='*', color='#D55E00', ms=10, label='Path start')
axes[0, 1].legend(loc='lower left', fontsize=8)
for ax, axis, label in [(axes[1, 0], 0, 'Nominal x (mm)'), (axes[1, 1], 1, 'Nominal y (mm)')]:
    ax.scatter(cloud[:, axis], cloud[:, 2], s=2, color='#0072B2')
    ax.set(xlabel=label, ylabel='Height (mm)', title='Prepared-scan projection')
    ax.set_ylim(lo - .5, hi + .5)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in fig.findobj(matplotlib.text.Text):
    if not text.get_visible() or not text.get_text():
        continue
    b = text.get_window_extent(renderer)
    if b.width and b.height:
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height, text.get_text()
fig.savefig(image, dpi=160)
plt.close(fig)
summary = {
    'question': 'Does the fixed preprocessing frame plausibly align scan axes with the nominal run9 path?',
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    'source_sha256': digest(source), 'target_sha256': digest(target),
    'metadata_sha256': digest(meta_path), 'toolpath_sha256': digest(path_file),
    'script_sha256': digest(Path(__file__)),
    'raw_unique_vertices': len(q), 'display_raw_points': len(display), 'display_seed': 2026,
    'raw_xyz_min_mm': q.min(axis=0).tolist(), 'raw_xyz_max_mm': q.max(axis=0).tolist(),
    'raw_quantiles_mm': np.quantile(q, [.005, .5, .995], axis=0).tolist(),
    'raw_fraction_outside_xy_crop': float(np.mean(np.any(np.abs(q[:, :2]) > [98, 148], axis=1))),
    'accepted_points': len(cloud),
    'prepared_xyz_min_mm': cloud.min(axis=0).tolist(), 'prepared_xyz_max_mm': cloud.max(axis=0).tolist(),
    'path_points_after_consecutive_deduplication': len(path),
    'path_centering': 'y_csv - 93.75 mm; unchanged x; no rotation or data-fitted translation',
    'nominal_path_fits_crop': bool(np.all(np.abs(path[:, :2]) <= [98, 148])),
    'raw_display_outside_plot_limits': int(np.sum(np.any(np.abs(display[:, :2]) > [140, 190], axis=1))),
    'height_color_limits_mm': [float(lo), float(hi)],
    'limits': 'CSV overlay is a nominal coordinate reference, not observed wheel marks. No physical corner/face labels; no rotation/turnover verified. Coverage is projected-bin coverage, not physical sheet area. Raw scatter is a seeded display subset; all summaries use all unique vertices. Height color is clipped to prepared range, with extended colorbars.',
    'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
}
assert digest(target) == meta['target_sha256']
assert digest(source) == meta['source_sha256']
report.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
