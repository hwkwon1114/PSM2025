"""Read-only frozen-checkpoint gallery; no mechanics, registration or rescoring."""
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from calibration.ewcal.geometry import points, pose_points, read_mesh

SNAPSHOT = BASE / 'run/optimization_review/deformation_constraint_snapshot'
OUT = SNAPSHOT / 'point_gallery'
BLUE, ORANGE = '#0072B2', '#D55E00'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9,
         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'pdf.fonttype': 42, 'figure.facecolor': 'white'}


def metrics(record):
    if 'metrics' in record:
        return record['metrics']
    case = record.get('rejected_case')
    if case is None and record.get('cases'):
        assert len(record['cases']) == 1
        case = record['cases'][0]
    return case.get('metrics', {}) if case else {}


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [fig._suptitle, *fig.legends]
    for ax in fig.axes:
        artists.extend([ax.title, ax._left_title, *ax.texts])
    boxes = []
    for artist in artists:
        if artist is None or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, 'Text outside canvas'
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            assert min(a.x1,b.x1)-max(a.x0,b.x0) <= 1 or min(a.y1,b.y1)-max(a.y0,b.y0) <= 1, 'Title/annotation overlap'


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use compute allocation'
    # Bounded self-check: malformed placement is detected, clean placement passes.
    with plt.style.context(STYLE):
        fixture, ax = plt.subplots()
        a = ax.text(.5,.5,'same',transform=ax.transAxes)
        b = ax.text(.5,.5,'same',transform=ax.transAxes)
        try:
            check_layout(fixture)
        except AssertionError:
            pass
        else:
            raise AssertionError('Layout check missed overlap')
        b.remove();check_layout(fixture)
        a.set_position((100,100))
        try:
            check_layout(fixture)
        except AssertionError:
            pass
        else:
            raise AssertionError('Layout check missed overflow')
        plt.close(fixture)
    assert metrics({'metrics': {'pose': [0]*6}})['pose'] == [0]*6
    assert metrics({'cases': [{'metrics': {'test': 1}}]}) == {'test': 1}
    assert metrics({'rejected_case': {'metrics': {'test': 2}}}) == {'test': 2}
    assert metrics({'failure_class': 'timeout'}) == {}
    OUT.mkdir(exist_ok=False)
    outputs = [OUT/'deformation_gallery.pdf', OUT/'pose_constraints.png', OUT/'pose_constraints.pdf',
               OUT/'pose_metrics.csv', OUT/'provenance.json', OUT/'deformation_gallery.caption.md']
    assert not any(p.exists() for p in outputs), 'Do not overwrite figures'
    hashes = {}
    def pin(path):
        path = Path(path)
        if not path.is_absolute():
            path = BASE/path
        hashes[str(path.relative_to(BASE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):
        return json.loads(pin(path).read_text())
    state = read(SNAPSHOT/'checkpoint_snapshot.json')
    receipt = read(SNAPSHOT/'snapshot_receipt.json')
    history = read('run/run6_bo_refinement_240/corrected_history.json')
    legacy = {r['attempt_id']: r for r in read('run/optimization_review/legacy_frame_reassessment_retry.json')['records']}
    pin(Path(__file__))
    pin('calibration/ewcal/geometry.py')
    with np.load(pin('calibration/data/processed/run6_oriented.npz'), allow_pickle=False) as f:
        scan0 = f['points_mm'].copy()
    assert scan0.ndim == 2 and scan0.shape[1] == 3 and np.isfinite(scan0).all()
    config = history['config']['registration']
    assert config['rotation_bound_deg'] == 5 and config['translation_bounds_mm'] == [5,5,20]
    records = {r['attempt_id']: r for r in state['observations']}
    assert len(records) == len(state['observations'])
    ids = sorted(set(records) | {r['attempt_id'] for r in state['reserved_attempts']})
    assert ids == list(range(1, max(ids)+1))
    rows, geometry = [], {}
    for att in ids:
        r = records.get(att)
        m = metrics(r) if r else {}
        row = {'attempt_id': att, 'status': 'qualified' if r and r.get('valid') else
               (r.get('failure_class') or 'registration') if r else 'pending',
               'scoring_source': r.get('scoring_source','production240') if r else 'not published',
               'rms_symmetric_mm': m.get('rms_symmetric_mm'),
               'registration_converged': m.get('registration_converged'),
               'stop_reason': m.get('stop_reason'), 'iterations_used': m.get('iterations_used'),
               'pose_near_bound': m.get('pose_near_bound')}
        if r:
            row.update(r['parameters'])
        if m:
            pose = np.asarray(m['pose'], float)
            assert pose.shape == (6,) and np.isfinite(pose).all()
            angle = np.rad2deg(np.linalg.norm(pose[:3]))
            assert abs(angle-m['rotation_angle_deg']) < 1e-6
            row.update(rotation_deg=float(angle), tx_mm=float(pose[3]), ty_mm=float(pose[4]), tz_mm=float(pose[5]))
            occupancy = np.r_[angle/5, abs(pose[3:])/np.array([5,5,20])]
            active = [name for name,value in zip(['rotation','tx','ty','tz'],occupancy) if value > .95]
            assert bool(active) == bool(m['pose_near_bound']), 'Bound flag mismatch'
            row['near_bound_components'] = ','.join(active)
            trace = m.get('trace') or []
            if trace:
                assert len(trace) == m['iterations_used']
                last = trace[-1]
                for key in ['delta_rotation_deg','delta_translation_mm','delta_point_motion_rms_mm','delta_rms_mm']:
                    row['last_'+key] = last[key]
                row['physical_motion_stable'] = all(last[k] < .005 for k in ['delta_rotation_deg','delta_translation_mm','delta_point_motion_rms_mm'])
            # A missing old convergence flag/trace stays missing, not fabricated from validity.
            if r.get('mesh_file'):
                mesh_path, state_path = r['mesh_file'], r['state_file']
            else:
                case = r.get('rejected_case') or r['cases'][0]
                folder = Path(case['directory'])
                mesh_path, state_path = folder/'prediction_mm.vtp', folder/'bilayer_zigzag_sequence_native_hybrid_state.json'
            native = read(state_path)
            faces = np.asarray(native['faces'], int)
            mesh = read_mesh(pin(mesh_path), scale=1000 if att in legacy else 1)
            sim = points(mesh)
            if att in legacy:
                norm = legacy[att]['normalization']
                sim = sim @ np.array(norm['row_vector_rotation']) + np.array(norm['translation_mm'])
                np.testing.assert_allclose(pose, legacy[att]['metrics']['pose'], rtol=0, atol=1e-12)
            assert faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < len(sim)
            assert np.isfinite(sim).all()
            aligned_scan = pose_points(scan0, pose)
            geometry[att] = (sim, faces, aligned_scan)
        else:
            assert not r or r.get('failure_class') in ['timeout','solver','infrastructure']
        rows.append(row)
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/'pose_metrics.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    bounds = np.vstack([np.array([g[0].min(0),g[0].max(0),g[2].min(0),g[2].max(0)]) for g in geometry.values()])
    low, high = bounds.min(0), bounds.max(0)
    margin = np.maximum((high-low)*.035,1)
    low, high = low-margin, high+margin
    indices = np.linspace(0,len(scan0)-1,min(2500,len(scan0)),dtype=int)
    legend = [Line2D([],[],color=BLUE,marker='o',ls='none',label='Simulation vertices'),
              Line2D([],[],color=ORANGE,marker='x',ls='none',label='Scan')]
    pages=[]
    # New campaign first; then complete history. Order within groups independent of error.
    ordered = [r for r in rows if r['attempt_id']>=33] + [r for r in rows if r['attempt_id']<33]
    with plt.style.context(STYLE), PdfPages(OUT/'deformation_gallery.pdf') as pdf:
        for page,start in enumerate(range(0,len(ordered),4),1):
            fig=plt.figure(figsize=(12,11),layout='constrained')
            fig.get_layout_engine().set(rect=(0,0,1,.94))
            fig.suptitle(f'Saved deformations — page {page} (frozen snapshot)',y=.993,fontsize=12)
            fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.97),ncol=2,frameon=False)
            grid=fig.add_gridspec(4,3)
            for i,row in enumerate(ordered[start:start+4]):
                att=row['attempt_id']; g=geometry.get(att)
                ax3=fig.add_subplot(grid[i,0],projection='3d' if g else None)
                axx=fig.add_subplot(grid[i,1]);axy=fig.add_subplot(grid[i,2])
                status=row['status'];rms=row['rms_symmetric_mm']
                ax3.set_title(f'{att:02d} | {status}'+(f' | RMS {rms:.2f} mm' if rms is not None else ''),loc='left')
                if g is None:
                    for ax in [ax3,axx,axy]:ax.set_axis_off()
                    axx.text(.5,.5,'No published final deformation\n'+status,ha='center',va='center',transform=axx.transAxes)
                    continue
                sim,faces,scan=g
                sim_index=np.linspace(0,len(sim)-1,min(2000,len(sim)),dtype=int)
                ax3.scatter(*sim[sim_index].T,c=BLUE,marker='o',s=2,alpha=.6,rasterized=True)
                ax3.scatter(*scan[indices].T,c=ORANGE,marker='x',s=2,alpha=.7,rasterized=True)
                ax3.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='x (mm)',ylabel='y (mm)')
                ax3.set_zticks([]);ax3.set_box_aspect(high-low);ax3.view_init(elev=24,azim=-62)
                for axis,dim,name in [(axx,0,'x–z projection'),(axy,1,'y–z projection')]:
                    axis.scatter(sim[:,dim],sim[:,2],c=BLUE,s=1,marker='o',alpha=.35,rasterized=True)
                    axis.scatter(scan[indices,dim],scan[indices,2],c=ORANGE,s=3,marker='x',alpha=.7,rasterized=True)
                    axis.set(xlim=(low[dim],high[dim]),ylim=(low[2],high[2]),xlabel=f'{"xy"[dim]} (mm)',ylabel='z (mm)',aspect='equal')
                    axis.spines[['top','right']].set_visible(False)
                axx.set_title(f'x–z | rot {row["rotation_deg"]:.2f}° | '+(row['near_bound_components'] or 'interior'),loc='left')
                axy.set_title(f'y–z | {row["stop_reason"]} ({row["iterations_used"]} steps)',loc='left')
            check_layout(fig)
            png=OUT/f'deformation_gallery_page_{page:02d}.png';assert not png.exists()
            fig.savefig(png,dpi=150);pdf.savefig(fig,dpi=180);plt.close(fig);pages.append(str(png.relative_to(BASE)))
            print('PAGE',page,flush=True)
        fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
        valid_rows=[r for r in rows if r['rms_symmetric_mm'] is not None]
        for group,color,marker in [('qualified',BLUE,'o'),('registration',ORANGE,'x')]:
            rr=[r for r in valid_rows if r['status']==group]
            axes[0].scatter([r['attempt_id'] for r in rr],[r['rms_symmetric_mm'] for r in rr],c=color,marker=marker,label=group)
        axes[0].set(ylabel='Symmetric RMS (mm)',yscale='log');axes[0].legend(ncol=2)
        for key,bound,color,marker in [('rotation_deg',5,'#000000','o'),('tx_mm',5,BLUE,'s'),('ty_mm',5,ORANGE,'^'),('tz_mm',20,'#009E73','D')]:
            axes[1].scatter([r['attempt_id'] for r in valid_rows],[abs(r[key])/bound for r in valid_rows],c=color,marker=marker,s=24,label=key.replace('_mm','').replace('_deg',''))
        axes[1].axhline(.95,c='black',ls='--',lw=1,label='Near-bound gate');axes[1].set(ylabel='Absolute pose / bound',ylim=(-.025,1.12));axes[1].legend(ncol=5,fontsize=8)
        for key,color,marker,name in [('last_delta_rotation_deg',BLUE,'o','Rotation step (deg)'),('last_delta_point_motion_rms_mm',ORANGE,'x','Point motion (mm)')]:
            rr=[r for r in rows if r.get(key) is not None]
            axes[2].scatter([r['attempt_id'] for r in rr],[max(r[key]/.005,1e-7) for r in rr],c=color,marker=marker,label=name)
        axes[2].axhline(1,c='black',ls='--',lw=1,label='Physical-motion threshold');axes[2].set(ylabel='Final step / threshold',yscale='log',xlabel='Attempt');axes[2].legend(ncol=3)
        for ax in axes:
            ax.axvline(32.5,c='gray',ls=':',lw=1);ax.set_xlim(.5,max(ids)+.5);ax.spines[['top','right']].set_visible(False)
        check_layout(fig);fig.savefig(OUT/'pose_constraints.png',dpi=170);fig.savefig(OUT/'pose_constraints.pdf',dpi=200);plt.close(fig)
    for name,digest in hashes.items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,'Frozen input changed'
    caption=f'''# Saved deformation and constraint audit

Frozen checkpoint at {receipt['snapshot_time']}, job9577873. All {len(ids)} history/admitted attempt IDs accounted: {len(geometry)} published final mechanical shapes and {len(ids)-len(geometry)} timeout/pending placeholders. New IDs33 onward appear first, then history1–32; none selected by error. Later results are not included. 3D view uses up to2000 uniformly indexed simulation vertices (no triangle surface); projection views retain ALL simulation vertices. Exactly{len(indices)} uniformly indexed scan points per panel from{len(scan0)} used in all display views (no metric recalculation). Blue circles: simulation; orange crosses: scan under the SAVED pose. Identical global axis limits, equal physical scale, camera and sampling across all panels; projections are full point projections, not cross-sections. Thin3D z ticks omitted; z is quantified in projections. Scan translations/rotations change per saved record, not new fits. Legacy2/4/5 use previously saved canonical row-vector rigid transform and1000m→mm conversion, no new rigid fitting. Historical early-converged metrics reused; saved240 reassessments override original60-step poses for failed history. CSV preserves parameters, pose components, qualification/stopping evidence and final trace changes when available.

Constraint overview includes every finite published symmetric RMS; rejected values are DIAGNOSTIC, never accepted objectives or penalty placeholders. Missing/timeout losses absent, not zero. Gray line separates old32 from new campaign. Hard bounds: rotation magnitude5degrees, tx/ty±5mm, tz±20mm. Gate rejects absolute component/magnitude occupancy>0.95, not merely hard-bound exceedance. Final steps are normalized by0.005deg or0.005mm physical-motion thresholds; the third physical test (translation) remains inCSV. Values below1 pass that individual test, not necessarily all. The alternative objective-stagnation stop requires two consecutive subset-RMS changes<0.0001mm after at least six iterations. Thus current 'registration_converged' does NOT ensure each physical-motion threshold passed; bound rejection is separate. Missing old trace/explicit convergence evidence stays missing. Zero final steps are displayed at1e-7 normalized only to permit log axis, not changed inCSV. No confidence intervals: adaptive deterministic simulations on one calibration scan, not held-out replicates.

Plots can reveal mismatched shapes, bound-active poses and numerical stopping ambiguity; they do not establish that wider constraints would improve an admissible physical calibration. No mechanics/registration/GP/bounds/labels changed; BO remains unchanged. Provenance inprovenance.json; source calibration/analysis/plot_deformation_constraints.py.
'''
    (OUT/'deformation_gallery.caption.md').write_text(caption)
    (OUT/'provenance.json').write_text(json.dumps({'input_sha256':hashes,'snapshot':receipt,'rows':rows,'shared_limits_mm':[low.tolist(),high.tolist()], 'scan_display_indices':indices.tolist(),'outputs':pages+[str(p.relative_to(BASE)) for p in outputs[:-1]],'job_id':os.environ['SLURM_JOB_ID'],'hostname':platform.node(),'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'simulation_style':{'color':BLUE,'marker':'o'},'scan_style':{'color':ORANGE,'marker':'x'},'no_new_alignment':True},indent=2)+'\n')
    print('COMPLETE',len(geometry),'shapes',len(ids),'IDs',flush=True)


if __name__ == '__main__':
    main()
