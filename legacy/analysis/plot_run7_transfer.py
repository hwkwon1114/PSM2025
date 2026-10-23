"""Run 7 frozen-transfer figure: scan vs prediction height maps, signed difference, profiles, error distribution.
All plotted scan points are the full prepared cloud; profile slices are fixed y bands (not error-selected)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.ewcal.geometry import points, read_mesh
STYLE = Path('/home/pxl1051/.pi/agent/skills/scientific-visualization/assets/publication.mplstyle')
case = BASE/'run/run7_transfer/evaluations/attempt_9901/case_00'
out = BASE/'calibration/analysis/plots'; stem = 'run7_transfer_comparison'
targets = [out/f'{stem}.pdf', out/f'{stem}.png']
if any(t.exists() for t in targets): raise FileExistsError(targets)
with np.load(case/'comparison.npz') as f: scan, dist = f['target_mm'], f['distance_mm']
sim = points(read_mesh(case/'prediction_mm.vtp'))
metrics = json.loads((BASE/'run/run7_transfer/run7_transfer_summary.json').read_text())['metrics']
_, k = cKDTree(sim[:, :2]).query(scan[:, :2]); dz = sim[k, 2] - scan[:, 2]   # prediction - scan, nearest in xy
vmin, vmax = np.percentile(np.r_[scan[:, 2], sim[:, 2]], [0, 100])
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '#333333'
with plt.style.context(str(STYLE)):
    fig = plt.figure(figsize=(7.2, 7.4), layout='constrained')
    gs = fig.add_gridspec(3, 3, height_ratios=[1.6, 1, 1])
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    s = ax[0].scatter(scan[:, 0], scan[:, 1], c=scan[:, 2], s=3, cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
    tri = mtri.Triangulation(sim[:, 0], sim[:, 1])
    ax[1].tripcolor(tri, sim[:, 2], cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud', rasterized=True)
    fig.colorbar(s, ax=ax[:2], shrink=.6, label='Height (mm)', location='bottom')
    lim = np.percentile(np.abs(dz), 99)
    d = ax[2].scatter(scan[:, 0], scan[:, 1], c=dz, s=3, cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
    fig.colorbar(d, ax=ax[2], shrink=.6, label='Prediction − scan (mm)', location='bottom')
    for a, t in zip(ax, ['(a) Scan', '(b) Prediction', '(c) Height difference']):
        a.set(aspect='equal', xlabel='x (mm)', xlim=(-105, 105), ylim=(-155, 155)); a.set_title(t, loc='left')
    ax[0].set_ylabel('y (mm)')
    prof = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for a, y0, lab in zip(prof, (-100, 0, 100), '(d) (e) (f)'.split()):
        ms, mm = np.abs(scan[:, 1]-y0) < 5, np.abs(sim[:, 1]-y0) < 5
        o = np.argsort(sim[mm, 0])
        a.scatter(scan[ms, 0], scan[ms, 2], s=2, color=GREY, label='Scan', rasterized=True)
        a.plot(sim[mm, 0][o], sim[mm, 2][o], color=ORANGE, ls='--', lw=1.3, label='Prediction')
        a.set(xlabel='x (mm)', ylim=(vmin-1, vmax+1)); a.set_title(f'{lab} y = {y0:+d} mm', loc='left')
    prof[0].set_ylabel('Height (mm)'); prof[0].legend(loc='lower left', markerscale=3)
    h = fig.add_subplot(gs[2, :2])
    h.hist(dist, bins=60, color=BLUE, alpha=.85)
    h.axvline(metrics['rms_target_to_sim_mm'], color=ORANGE, ls='--', label='Run 7 RMS 7.6 mm')
    h.axvline(1.18, color=GREY, ls=':', label='Run 6 training RMS 1.2 mm')
    h.set(xlabel='Scan-to-prediction distance (mm)', ylabel='Scan points'); h.set_title('(g) Distance distribution', loc='left'); h.legend()
    r = fig.add_subplot(gs[2, 2])
    r.hist([scan[:, 2], sim[:, 2]], bins=40, color=[GREY, ORANGE], label=['Scan', 'Prediction'], histtype='step', lw=1.3)
    r.set(xlabel='Height (mm)', ylabel='Count'); r.set_title('(h) Height range', loc='left'); r.legend()
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    undrawn = set()   # tick labels outside the view limits exist but are not drawn
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            undrawn |= {id(tk.label1) for tk in axis.get_major_ticks() if not lo <= tk.get_loc() <= hi}
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text() and id(t) not in undrawn:
            b = t.get_window_extent(rend)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, t.get_text()
    fig.savefig(targets[0]); fig.savefig(targets[1], dpi=200)
print('ok')
