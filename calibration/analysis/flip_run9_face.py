"""180deg x-turnover of the saved z-aligned run9 target; no reprocessing."""
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
from calibration.ewcal.common import digest, save_json

source = BASE / 'calibration/data/processed/run9_z_aligned.npz'
source_meta = source.with_suffix('.json')
dest = BASE / 'calibration/data/processed/run9_oriented.npz'
out = BASE / 'run/run9_preprocessing'
image_path = out / 'run9_face_flip.png'
report_path = out / 'face_flip.json'
for p in [dest, dest.with_suffix('.json'), image_path, report_path]:
    if p.exists():
        raise FileExistsError(p)
config_path = BASE / 'calibration/configs/run9.json'
meta = json.loads(source_meta.read_text())
pins = {str(p): digest(p) for p in [source, source_meta, config_path]}
assert digest(source) == meta['target_sha256']
with np.load(source, allow_pickle=False) as f:
    arrays = {k: f[k].copy() for k in f.files}
cloud = arrays['points_mm'].copy()
turnover = np.diag([1., -1., -1.])
flipped = cloud @ turnover
assert np.array_equal(flipped @ turnover, cloud)
assert np.array_equal(flipped[:, 0], cloud[:, 0])
assert np.array_equal(flipped[:, 1:], -cloud[:, 1:])
assert np.allclose(np.linalg.norm(cloud, axis=1), np.linalg.norm(flipped, axis=1))
t = meta['transform']
basis = np.array(t['basis_columns']) @ turnover
shift = np.array(t['shift_local_mm']) @ turnover
assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
assert np.linalg.det(basis) > .999999
# Check the complete composed transform against exact saved point reversal.
scanner_relative = (cloud + np.array(t['shift_local_mm'])) @ np.array(t['basis_columns']).T
assert np.allclose(scanner_relative @ basis - shift, flipped, atol=1e-10)
arrays['points_mm'] = flipped
np.savez_compressed(dest, **arrays)
meta['target_sha256'] = digest(dest)
meta['target_npz'] = str(dest.relative_to(BASE))
meta['transform']['basis_columns'] = basis.tolist()
meta['transform']['shift_local_mm'] = shift.tolist()
meta['orientation'] = {'flip_z': False, 'flip_xy': False, 'basis_columns': basis.tolist()}
meta['orientation_notes'] = 'First +8.897781769 deg z-yaw from run9 sheet edges, then 180deg x-turnover [1,-1,-1]. Complete composed basis/shift are recorded; no additional flip flags are needed.'
meta['face_convention'] = 'Researcher requested x-turnover after inspecting z-only alignment; physical face/start corner are not independently identified from STL.'
meta['face_flip'] = {'source_target': str(source.relative_to(BASE)), 'source_target_sha256': pins[str(source)],
                     'matrix': turnover.tolist(), 'formula': '(x,y,z) -> (x,-y,-z)',
                     'reprocessed': False, 'point_count': len(flipped)}
save_json(dest.with_suffix('.json'), meta)
with np.load(dest, allow_pickle=False) as f:
    assert np.array_equal(f['points_mm'], flipped)
    for k in arrays:
        if k != 'points_mm':
            assert np.array_equal(f[k], arrays[k]), k
path_file = BASE / 'calibration/data/raw/run9_zigzag_path.csv'
with path_file.open() as f:
    path = np.array([[float(r[k]) for k in ('x_mm', 'y_mm')] for r in csv.DictReader(f)])
path[:, 1] -= 93.75
# Fixed nominal path in the model frame, not transformed with the scan.
height_limit = float(np.max(np.abs(cloud[:, 2])))
fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
for column, (data, title) in enumerate([(cloud, 'Z-aligned, before flip'), (flipped, 'After x-turnover')]):
    ax = axes[0, column]
    artist = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=3, cmap='viridis', vmin=-height_limit, vmax=height_limit)
    ax.plot(path[:, 0], path[:, 1], color='#111111', lw=.8, label='Nominal CSV path')
    ax.plot(path[0, 0], path[0, 1], marker='*', color='#D55E00', ms=10, label='Path start')
    ax.set(xlabel='Nominal x (mm)', ylabel='Nominal y (mm)', title=title, aspect='equal', xlim=(-110, 110), ylim=(-160, 160))
    ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    ax.legend(loc='lower left', fontsize=8)
    fig.colorbar(artist, ax=ax, label='Height (mm)', ticks=np.linspace(-height_limit, height_limit, 5), format='%.1f')
    ax = axes[1, column]
    ax.scatter(data[:, 0], data[:, 2], s=2, color='#0072B2')
    ax.set(xlabel='Nominal x (mm)', ylabel='Height (mm)', title='X–height projection', xlim=(-110, 110), ylim=(-6, 6))
    ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
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
report = {'status': 'face_flip_complete', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
          'inputs_sha256': pins, 'script_sha256': digest(Path(__file__)), 'toolpath_sha256': digest(path_file),
          'target_sha256': digest(dest), 'formula': '(x,y,z) -> (x,-y,-z)', 'points': len(flipped),
          'flipped_height_range_mm': [float(flipped[:, 2].min()), float(flipped[:, 2].max())],
          'height_color_limits_mm': [-height_limit, height_limit],
          'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
          'limits': 'No new processing, smoothing, curvature fitting, solver or config change. Face convention requested by researcher; no independently labelled experimental face/corner.'}
save_json(report_path, report)
print(json.dumps(report, indent=2))
