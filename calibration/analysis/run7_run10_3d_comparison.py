"""Run 7 / Run 10 frozen-transfer 3D figure in the Run 6/Run 9 format (run6_run9_3d_comparison.py):
isometric residual, end-on residual, shape overlay. Row colour scales are independent because Run 7 residuals
(order 20 mm) would hide Run 10 residuals (order 2 mm)."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import numpy as np
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import load_evaluation_data
from calibration.ewcal.common import save_json

out = BASE / 'calibration/analysis/plots'; stem = 'run7_run10_3d_isometric_and_overlay'
targets = [out / f'{stem}.png', out / f'{stem}.pdf']
if any(t.exists() for t in targets): raise FileExistsError(targets)
pins = {}
cases = [('run7', 'run/run7_transfer/evaluations/attempt_9901/case_00', 'Run 7'),
         ('run10', 'run/run10_transfer/evaluations/attempt_9902/case_00', 'Run 10')]
rows = [(label, load_evaluation_data(name, case, pins, BASE)) for name, case, label in cases]
ASPECT = [1.0, 1.5, 0.35]
rc = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7,
      'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'axes.linewidth': .6, 'savefig.facecolor': 'white'}
panel = iter('abcdef')
with plt.rc_context(rc):
    fig = plt.figure(figsize=(15., 8.2))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1., 1., .04, 1.05], wspace=.25, hspace=.22)
    for i, (label, r) in enumerate(rows):
        scan, q, dz, faces = r['scan'], r['prediction'], r['dz'], r['faces']
        lim = float(np.nanmax(np.abs(dz))); norm = TwoSlopeNorm(vmin=-lim, vcenter=0., vmax=lim)
        sc = None
        for j, (view, kind) in enumerate([((25, -55), 'residual (isometric)'), ((10, -15), 'residual (end-on)')]):
            ax = fig.add_subplot(gs[i, j], projection='3d')
            sc = ax.scatter(scan[:, 0], scan[:, 1], scan[:, 2], c=dz, cmap='RdBu_r', norm=norm, s=2, rasterized=True)
            ax.set_box_aspect(ASPECT); ax.view_init(elev=view[0], azim=view[1])
            ax.set(title=f'({next(panel)}) {label}: 3D {kind}', xlabel='x (mm)', ylabel='y (mm)'); ax.set_zlabel('z (mm)', labelpad=1)
        cax = fig.add_subplot(gs[i, 2])
        fig.colorbar(sc, cax=cax, label='Δz: prediction − experiment (mm)', ticks=np.linspace(-lim, lim, 5), format='%.1f')
        ax = fig.add_subplot(gs[i, 3], projection='3d')
        he = ax.scatter(scan[:, 0], scan[:, 1], scan[:, 2], color='#D55E00', s=2, alpha=.5, label='Experiment', rasterized=True)
        ax.plot_trisurf(q[:, 0], q[:, 1], q[:, 2], triangles=faces, color='#0072B2', alpha=.35, edgecolor='none')
        ax.set_box_aspect(ASPECT); ax.view_init(elev=25, azim=-55)
        ax.set(title=f'({next(panel)}) {label}: 3D shape overlay', xlabel='x (mm)', ylabel='y (mm)'); ax.set_zlabel('z (mm)', labelpad=1)
        ax.legend(handles=[he, mpatches.Patch(color='#0072B2', alpha=.5, label='Prediction')], loc='upper right', framealpha=.9, fontsize=8)
    fig.savefig(targets[0], dpi=300, bbox_inches='tight'); fig.savefig(targets[1], bbox_inches='tight', dpi=150)
    plt.close(fig)
stats = [r['statistics'] for _, r in rows]
save_json(out / f'{stem}.provenance.json', {'input_sha256': pins, 'statistics': stats})
(out / f'{stem}.caption.md').write_text(
 "Frozen Run 6 attempt-209 constants applied without refit. Top: Run 7 (175 mm, 0.5 deg, 100 zigzags, 13 cycles). "
 "Bottom: Run 10 (175 mm, 1.5 deg, 24 zigzags, 50 cycles). Columns: isometric residual (elev 25, azim -55), end-on residual (elev 10, azim -15), "
 "shape overlay. dz = prediction z interpolated at scan xy minus scan z. **Colour scales differ by row** (symmetric about 0): "
 + ', '.join(f"{lab} +/-{np.nanmax(np.abs(r['dz'])):.1f} mm" for lab, r in rows)
 + ". z axis is exaggerated (box aspect 1:1.5:0.35), as in the Run 6/Run 9 figure. Single forward solve per run.\n")
for lab, r in rows:
    s = r['statistics']; print(lab, 'rms dz', s['rms_dz_mm'], 'supported', s['supported_points'], 'of', s['accepted_scan_points'])
