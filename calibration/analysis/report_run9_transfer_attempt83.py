"""Report Attempt 83 frozen-parameter transfer; never solve or realign geometry."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.geometry import points, read_mesh, reference_dual_areas

root = BASE / 'run/run9'
record = json.loads((root / 'transfer_result.json').read_text())
if not record['valid']:
    raise RuntimeError('No accepted prediction: report numerical failure, not transfer error')
case = record['result']['cases'][0]
folder = Path(case['directory'])
output = root / 'report'
output.mkdir(exist_ok=True)
inputs = [root/'transfer_result.json', folder/'prediction_mm.vtp', folder/'comparison.npz',
          folder/'bilayer_zigzag_sequence_native_hybrid_state.json']
pins = {str(p.relative_to(BASE)): digest(p) for p in inputs}
with np.load(folder/'comparison.npz', allow_pickle=False) as f:
    scan, forward, closest = f['target_mm'].copy(), f['distance_mm'].copy(), f['closest_mm'].copy()
sim = points(read_mesh(folder/'prediction_mm.vtp'))
state = json.loads(inputs[-1].read_text())
reference = np.array(state['reference_positions']) * 1000
faces = np.array(state['faces'])
weights = reference_dual_areas(reference, faces)
weights /= weights.sum()
reverse, nearest = cKDTree(scan).query(sim)
metrics = case['metrics']
assert np.isfinite(scan).all() and np.isfinite(sim).all()
assert np.allclose(np.linalg.norm(scan - closest, axis=1), forward, atol=1e-8)
fwd_rms = float(np.sqrt(np.mean(forward**2)))
rev_rms = float(np.sqrt(np.sum(weights * reverse**2)))
sym_rms = float(np.sqrt(.5 * (fwd_rms**2 + rev_rms**2)))
for value, key in [(fwd_rms,'rms_target_to_sim_mm'),(rev_rms,'rms_sim_to_target_weighted_mm'),(sym_rms,'rms_symmetric_mm')]:
    assert abs(value - metrics[key]) < 1e-8
assert abs(np.quantile(reverse,.95) - metrics['p95_sim_to_target_mm']) < 1e-8
order = np.argsort(reverse)
weighted_p95 = float(reverse[order][np.searchsorted(np.cumsum(weights[order]), .95)])
np.savez_compressed(output/'full_distributions.npz', scan_mm=scan, prediction_mm=sim,
                    scan_to_surface_mm=forward, prediction_to_scan_mm=reverse,
                    prediction_reference_area_weights=weights, nearest_scan_index=nearest)

# Full held-out distributions lead; every point and vertex appears in spatial views.
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
ax = axes[0,0]
ordered = np.sort(forward)
ax.step(np.r_[0,ordered], np.r_[0,np.arange(1,len(ordered)+1)/len(ordered)], where='post',
        color='#D55E00', label='Scan → surface', linestyle='-')
ax.step(np.r_[0,reverse[order]], np.r_[0,np.cumsum(weights[order])], where='post',
        color='#0072B2', label='Model → scan (area)', linestyle='--')
ax.set(xlabel='Distance (mm)', ylabel='Cumulative fraction', title='(a) Full error distributions', ylim=(0,1), xlim=(0,max(forward.max(),reverse.max())*1.02))
ax.legend(loc='lower right', fontsize=8)
lo = float(min(scan[:,2].min(),sim[:,2].min()))
hi = float(max(scan[:,2].max(),sim[:,2].max()))
norm = Normalize(lo,hi)
xy = np.vstack([scan[:,:2],sim[:,:2]])
xlim = (float(xy[:,0].min()-5),float(xy[:,0].max()+5))
ylim = (float(xy[:,1].min()-5),float(xy[:,1].max()+5))
for ax, cloud, title in [(axes[0,1],scan,'(b) Run9 experiment'),(axes[0,2],sim,'(c) Frozen-parameter prediction')]:
    artist = ax.scatter(cloud[:,0],cloud[:,1],c=cloud[:,2],s=3,cmap='viridis',norm=norm,rasterized=True)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title=title,xlim=xlim,ylim=ylim,aspect='equal')
fig.colorbar(artist,ax=axes[0,1:].tolist(),label='Height (mm)',shrink=.8,ticks=np.linspace(lo,hi,5),format='%.1f')
for ax, dim, title in [(axes[1,0],0,'(d) X–height projection'),(axes[1,1],1,'(e) Y–height projection')]:
    ax.scatter(scan[:,dim],scan[:,2],s=3,color='#D55E00',marker='o',alpha=.45,label='Experiment',rasterized=True)
    ax.scatter(sim[:,dim],sim[:,2],s=3,color='#0072B2',marker='x',alpha=.45,label='Prediction',rasterized=True)
    ax.set(xlabel=('x' if dim==0 else 'y')+' (mm)',ylabel='Height (mm)',title=title,
           xlim=xlim if dim==0 else ylim,ylim=(lo-.5,hi+.5))
    ax.legend(loc='best',fontsize=8)
ax = axes[1,2]
artist = ax.scatter(scan[:,0],scan[:,1],c=forward,s=3,cmap='magma',vmin=0,vmax=float(forward.max()),rasterized=True)
ax.set(xlabel='x (mm)',ylabel='y (mm)',title='(f) Scan-to-surface distance',xlim=xlim,ylim=ylim,aspect='equal')
fig.colorbar(artist,ax=ax,label='Distance (mm)',ticks=np.linspace(0,forward.max(),5),format='%.1f')
fig.canvas.draw(); renderer = fig.canvas.get_renderer()
for ax in fig.axes:
    texts = [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]
    legend = ax.get_legend()
    if legend: texts.extend(legend.get_texts())
    for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
        for tick in axis.get_major_ticks():
            if min(limits) <= tick.get_loc() <= max(limits):
                texts.extend([tick.label1,tick.label2])
    for text in texts:
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1,text.get_text()
for extension in ('png','pdf'):
    fig.savefig(output/f'run9_transfer_comparison.{extension}',dpi=200)
plt.close(fig)
for path,h in pins.items(): assert digest(BASE/path)==h
summary = {'metrics':metrics,'reverse_reference_area_weighted_p95_mm':weighted_p95,
           'scan_points':len(scan),'model_vertices':len(sim),'model_faces':len(faces),
           'parameters':record['parameters'],'run6_training_rms_mm':record['run6_training_rms_mm'],
           'run9_minus_run6_rms_mm':sym_rms-record['run6_training_rms_mm'],
           'run9_to_run6_rms_ratio':sym_rms/record['run6_training_rms_mm'],
           'scan_height_range_mm':[float(scan[:,2].min()),float(scan[:,2].max())],
           'prediction_height_range_mm':[float(sim[:,2].min()),float(sim[:,2].max())],
           'input_sha256':pins,'script_sha256':digest(Path(__file__)),
           'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,
           'comparison':'Saved production material-reference coordinates; no new solve or alignment.',
           'selection':'All held-out points/vertices; no spatial downsampling or favorable region selection.',
           'caveats':'One specimen, no population-generalization CI; nominal visually adopted axes/face; cropped scan top approximates midsurface. Reverse production p95 is unweighted, RMS/CDF are reference-area weighted.'}
save_json(output/'summary.json',summary)
print(json.dumps(summary,indent=2))
