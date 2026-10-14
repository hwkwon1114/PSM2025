"""Report one saved frozen-parameter transfer; never solve or realign geometry."""
import argparse
import ast
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=BASE / 'run/run9_transfer_run6_attempt25')
parser.add_argument('--baseline-root', type=Path)
args = parser.parse_args()
root = args.root.resolve()
expanded = (root / 'run9_transfer_expanded_summary.json').exists()
if expanded:
    publication = root / 'evaluations/attempt_9901/result.json'
    transfer = json.loads((root / 'run9_transfer_expanded_summary.json').read_text())
    record = {'valid': True, 'result': json.loads(publication.read_text()),
              'parameters': transfer['parameters'], 'run6_training_rms_mm': transfer['run6_best_objective_mm']}
else:
    publication = root / 'transfer_result.json'
    record = json.loads(publication.read_text())
if not record['valid'] or not record['result']['valid']:
    raise RuntimeError('No accepted prediction: report numerical failure, not transfer error')
case = record['result']['cases'][0]
folder = Path(case['directory'])
output = root / 'report'
output.mkdir(exist_ok=False)
inputs = [publication, folder/'prediction_mm.vtp', folder/'comparison.npz',
          folder/'bilayer_zigzag_sequence_native_hybrid_state.json']
if expanded:
    inputs += [root/'run9_transfer_expanded_summary.json', root/'evaluations/attempt_9901/request.json']
pins = {str(p.relative_to(BASE)): digest(p) for p in inputs}
with np.load(folder/'comparison.npz', allow_pickle=False) as f:
    scan, forward, closest = f['target_mm'].copy(), f['distance_mm'].copy(), f['closest_mm'].copy()
sim = points(read_mesh(folder/'prediction_mm.vtp'))
state = json.loads((folder/'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
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

baseline = None
if args.baseline_root:
    baseline_root = args.baseline_root.resolve()
    baseline_record_path = baseline_root/'transfer_result.json'
    baseline_record = json.loads(baseline_record_path.read_text())
    assert baseline_record['valid'] and baseline_record['result']['valid']
    baseline_case = baseline_record['result']['cases'][0]
    baseline_folder = Path(baseline_case['directory'])
    baseline_request = baseline_folder.parent/'request.json'
    new_request = root/'evaluations/attempt_9901/request.json'
    old_config = json.loads(baseline_request.read_text())['config']
    new_config = json.loads(new_request.read_text())['config']
    # Search bounds and output locations do not affect a frozen forward prediction.
    for key in set(old_config) | set(new_config):
        if key not in {'parameters', 'output_dir', 'report_dir'}:
            assert old_config.get(key) == new_config.get(key), ('Baseline recipe mismatch', key)
    old_packet_path = BASE/'run/optimization_review/review_packet_run9_transfer_attempt83.json'
    old_packet = json.loads(old_packet_path.read_text())
    current_packet_path = BASE/'run/optimization_review/run6_expanded_bo/run9_transfer_packet.json'
    current_packet = json.loads(current_packet_path.read_text())
    assert digest(old_packet_path) == baseline_record['packet_sha256']
    old_geometry_path = BASE/'run/optimization_review/run6_edge_alignment/snapshot/calibration/ewcal/geometry.py'
    geometry_key = 'calibration/ewcal/geometry.py'
    assert digest(old_geometry_path) == old_packet['input_sha256'][geometry_key]
    old_geometry_ast = ast.parse(old_geometry_path.read_text())
    old_geometry_ast.body = [node for node in old_geometry_ast.body
                             if not isinstance(node, ast.FunctionDef) or node.name != 'register']
    assert ast.dump(old_geometry_ast) == ast.dump(ast.parse((BASE/geometry_key).read_text())), 'Geometry change beyond unused register alias removal'
    pins[str(old_geometry_path.relative_to(BASE))] = digest(old_geometry_path)
    for path in ['calibration/ewcal/material_frame.py',
                 'run/native_shape_refinement/build_9060950/bin/shell',
                 'calibration/data/processed/run9_oriented.npz', 'calibration/data/processed/run9_oriented.json']:
        assert old_packet['input_sha256'][path] == current_packet['artifact_sha256'][path] == digest(BASE/path), path
    with np.load(baseline_folder/'comparison.npz', allow_pickle=False) as f:
        assert np.array_equal(scan, f['target_mm']), 'Not the same saved Run9 target coordinates'
        baseline_forward = f['distance_mm'].copy()
        assert np.allclose(np.linalg.norm(scan-f['closest_mm'], axis=1), baseline_forward, atol=1e-8)
    baseline_sim = points(read_mesh(baseline_folder/'prediction_mm.vtp'))
    baseline_state_path = baseline_folder/'bilayer_zigzag_sequence_native_hybrid_state.json'
    baseline_state = json.loads(baseline_state_path.read_text())
    baseline_weights = reference_dual_areas(np.array(baseline_state['reference_positions'])*1000,
                                          np.array(baseline_state['faces']))
    baseline_weights /= baseline_weights.sum()
    baseline_reverse = cKDTree(scan).query(baseline_sim)[0]
    baseline_metrics = baseline_case['metrics']
    baseline_fwd_rms = float(np.sqrt(np.mean(baseline_forward**2)))
    baseline_rev_rms = float(np.sqrt(np.sum(baseline_weights*baseline_reverse**2)))
    baseline_sym_rms = float(np.sqrt(.5*(baseline_fwd_rms**2+baseline_rev_rms**2)))
    for value, key in [(baseline_fwd_rms,'rms_target_to_sim_mm'), (baseline_rev_rms,'rms_sim_to_target_weighted_mm'),
                       (baseline_sym_rms,'rms_symmetric_mm')]:
        assert abs(value-baseline_metrics[key]) < 1e-8
    for p in [baseline_record_path, baseline_request, old_packet_path, current_packet_path,
              baseline_folder/'comparison.npz', baseline_folder/'prediction_mm.vtp', baseline_state_path]:
        pins[str(p.relative_to(BASE))] = digest(p)
    delta = sym_rms-baseline_sym_rms
    baseline = {'attempt_id':83, 'metrics':baseline_metrics, 'delta_rms_mm':delta,
                'delta_rms_pct':100*delta/baseline_sym_rms,
                'verified_same_recipe_and_target':True,
                'geometry_source_change':'AST-identical except unused legacy register alias removed; archived old source digest verified.',
                'runner_baseline_rounding_difference_mm':transfer['comparison']['baseline_attempt83_mm']-baseline_sym_rms,
                'height_range_mm':[float(baseline_sim[:,2].min()),float(baseline_sim[:,2].max())]}
    np.savez_compressed(output/'baseline_distributions.npz', prediction_mm=baseline_sim,
                        scan_to_surface_mm=baseline_forward, prediction_to_scan_mm=baseline_reverse,
                        prediction_reference_area_weights=baseline_weights)

# Full distributions lead; every point and vertex appears in spatial views.
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
ax = axes[0,0]
ordered = np.sort(forward)
ax.step(np.r_[0,ordered], np.r_[0,np.arange(1,len(ordered)+1)/len(ordered)], where='post',
        color='#D55E00', label='Scan → surface', linestyle='-')
ax.step(np.r_[0,reverse[order]], np.r_[0,np.cumsum(weights[order])], where='post',
        color='#0072B2', label='Model → scan (area)', linestyle='--')
ax.set(xlabel='Distance (mm)', ylabel='Cumulative fraction', title='(a) Full error distributions', ylim=(0,1), xlim=(0,max(forward.max(),reverse.max())*1.02))
if baseline:
    ax.lines[0].set(color='#0072B2', label='153: scan → surface', linestyle='-')
    ax.lines[1].set(color='#0072B2', label='153: model → scan (area)', linestyle='-.')
    old_order = np.argsort(baseline_reverse)
    ax.step(np.r_[0,np.sort(baseline_forward)], np.r_[0,np.arange(1,len(baseline_forward)+1)/len(baseline_forward)],
            where='post',color='#009E73',linestyle='--',label='83: scan → surface')
    ax.step(np.r_[0,baseline_reverse[old_order]],np.r_[0,np.cumsum(baseline_weights[old_order])],
            where='post',color='#009E73',linestyle=':',label='83: model → scan (area)')
    ax.set_xlim(0,max(forward.max(),reverse.max(),baseline_forward.max(),baseline_reverse.max())*1.02)
ax.legend(loc='lower right', fontsize=8)
clouds = [scan, sim] + ([baseline_sim] if baseline else [])
lo = float(min(c[:,2].min() for c in clouds))
hi = float(max(c[:,2].max() for c in clouds))
norm = Normalize(lo,hi)
xy = np.vstack([c[:,:2] for c in clouds])
xlim = (float(xy[:,0].min()-5),float(xy[:,0].max()+5))
ylim = (float(xy[:,1].min()-5),float(xy[:,1].max()+5))
for ax, cloud, title in [(axes[0,1],scan,'(b) Run9 experiment'),(axes[0,2],sim,'(c) Frozen-parameter prediction')]:
    artist = ax.scatter(cloud[:,0],cloud[:,1],c=cloud[:,2],s=3,cmap='viridis',norm=norm,rasterized=True)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title=title,xlim=xlim,ylim=ylim,aspect='equal')
fig.colorbar(artist,ax=axes[0,1:].tolist(),label='Height (mm)',shrink=.8,ticks=np.linspace(lo,hi,5),format='%.1f')
for ax, dim, title in [(axes[1,0],0,'(d) X–height projection'),(axes[1,1],1,'(e) Y–height projection')]:
    ax.scatter(scan[:,dim],scan[:,2],s=3,color='#D55E00',marker='o',alpha=.45,label='Experiment',rasterized=True)
    ax.scatter(sim[:,dim],sim[:,2],s=3,color='#0072B2',marker='x',alpha=.45,label='Attempt 153' if baseline else 'Prediction',rasterized=True)
    if baseline:
        ax.scatter(baseline_sim[:,dim],baseline_sim[:,2],s=3,color='#009E73',marker='^',alpha=.35,label='Attempt 83',rasterized=True)
    ax.set(xlabel=('x' if dim==0 else 'y')+' (mm)',ylabel='Height (mm)',title=title,
           xlim=xlim if dim==0 else ylim,ylim=(lo-.5,hi+.5))
    ax.legend(loc='best',fontsize=8)
ax = axes[1,2]
artist = ax.scatter(scan[:,0],scan[:,1],c=forward,s=3,cmap='magma',vmin=0,vmax=float(forward.max()),rasterized=True)
ax.set(xlabel='x (mm)',ylabel='y (mm)',title='(f) Scan-to-surface distance',xlim=xlim,ylim=ylim,aspect='equal')
fig.colorbar(artist,ax=ax,label='Distance (mm)',ticks=np.linspace(0,forward.max(),5),format='%.1f')
fig.canvas.draw(); renderer = fig.canvas.get_renderer()
# Check only actually drawn tick labels; off-limits automatic ticks are not drawn.
for ax in fig.axes:
    texts = [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]
    legend = ax.get_legend()
    if legend: texts.extend(legend.get_texts())
    for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
        for tick in axis.get_major_ticks():
            if min(limits) <= tick.get_loc() <= max(limits):
                texts.extend([tick.label1,tick.label2])
    visible = [t for t in texts if t.get_visible() and t.get_text()]
    boxes = [t.get_window_extent(renderer) for t in visible]
    for text,b in zip(visible,boxes):
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1,text.get_text()
    for i,b in enumerate(boxes):
        for j in range(i+1,len(boxes)):
            other = boxes[j]
            assert min(b.x1,other.x1)-max(b.x0,other.x0) <= 1 or min(b.y1,other.y1)-max(b.y0,other.y0) <= 1, (visible[i].get_text(),visible[j].get_text())
for extension in ('png','pdf'):
    fig.savefig(output/f'run9_transfer_comparison.{extension}',dpi=200)
plt.close(fig)
for path,h in pins.items(): assert digest(BASE/path)==h
summary = {'metrics':metrics,'baseline_attempt83':baseline,
           'attempt25_comparison':'Withheld: unverified source/scoring identity; runner delta not adopted.',
           'reverse_reference_area_weighted_p95_mm':weighted_p95,
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
(output/'run9_transfer_comparison.caption.md').write_text(
    'Run 9 frozen-parameter prediction. (a) Full scan-to-surface and reference-area-weighted '
    'model-to-scan error distributions; (b,c) experimental and predicted heights with shared scales; '
    '(d,e) full X–height/Y–height projections; (f) scan-to-surface distance, not signed height residual. '
    'All scan cells and mesh vertices are included; no spatial sampling, pose refit or smoothing. '
    'Experiment is vermilion circles; Attempt153 is blue crosses; verified Attempt83 baseline is green triangles '
    'when supplied. Curves are solid/dash-dot for 153 and dashed/dotted for 83. Production reverse RMS/CDF '
    'are reference-area weighted; production reverse p95 is unweighted. One previously evaluated secondary '
    'specimen, not an independent population validation; no confidence intervals. Conditional prediction '
    'does not establish physical parameter identification. PNG/PDF are 14×9-inch standalone views.\n')
print(json.dumps(summary,indent=2))
