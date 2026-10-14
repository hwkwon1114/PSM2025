"""Fixed refined run6 scan frame and coherent saved-history replay; no solver/GP."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import numpy as np
from scipy.spatial import cKDTree
from calibration.analysis import migrate_run6_edge_frame as original
from calibration.analysis import rescore_run6_edge_history as replay
from calibration.analysis.inspect_subbin_run6 import raw_edges
from calibration.analysis.orient_run9_edges import yaw_matrix
from calibration.ewcal.geometry import points, read_mesh
from calibration.ewcal.material_frame import score_material_frame
from calibration.ewcal.prepare import prepare_case

ROOT = original.ROOT / 'refined'
OUT = BASE / 'run/run6_bo_edge_aligned/refined'
TARGET = 'calibration/data/processed/run6_edge_aligned_refined.npz'
CONFIG = 'calibration/configs/run6_bo_edge_aligned_refined.json'
POLICY = BASE / 'run/run6_bo_edge_aligned/subbin_edge_inspection.json'
PROTOCOL = 'run/stage2_bayesian_optimization/stage2_cluster_protocol.md'
read, save, sha = original.read, original.save, original.sha


def baseline():
    manifest = original.verify()
    rows = manifest['observations']
    if ([r['attempt_id'] for r in rows] != list(range(1, 81)) or
            len(manifest['sources']) != 70 or
            [r['attempt_id'] for r in rows if not r['valid']] != [17,24,27,35,37,39,45,47,59,69]):
        raise ValueError('Wrong frozen campaign accounting')
    policy = read(POLICY)
    if sha(BASE / 'calibration/analysis/inspect_subbin_run6.py') != policy['source_sha256']:
        raise ValueError('Saved refined estimator source changed')
    return manifest, policy


def fixed_rotation(policy, observed_angle):
    angle = policy['additional_yaw_deg']
    if not np.isfinite(angle) or abs(angle - 0.05586775269924428) > 1e-12:
        raise ValueError('Unselected edge policy')
    if not np.isfinite(observed_angle) or abs(observed_angle - angle) > 1e-12:
        raise ValueError('Saved geometric estimate does not reproduce')
    rotation = yaw_matrix(angle)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
    return angle, rotation


def figure(before, aligned, cloud):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with original.snap('calibration/data/raw/run6_zigzag_path.csv').open() as f:
        path = np.array([[float(row[k]) for k in ('x_mm','y_mm')] for row in csv.DictReader(f)])
    path[:, 1] -= 65.
    indices = np.random.default_rng(2026).choice(len(before), min(40000, len(before)), replace=False)
    panels = (before[indices], aligned[indices], cloud)
    lo = min(float(p[:, 2].min()) for p in panels)
    hi = max(float(p[:, 2].max()) for p in panels)
    with plt.rc_context({'pdf.fonttype':42, 'font.size':9}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), layout='constrained')
        for ax, p, title in zip(axes, panels, ('Original frame','Refined edge frame','Prepared target')):
            artist = ax.scatter(p[:,0], p[:,1], c=p[:,2], s=1, cmap='cividis', vmin=lo, vmax=hi, rasterized=True)
            ax.plot(path[:,0], path[:,1], color='#222222', lw=.6, label='Nominal CSV path')
            ax.plot(path[0,0], path[0,1], marker='*', color='#D55E00', ms=7, label='Start')
            ax.plot([-98,98,98,-98,-98], [-148,-148,148,148,-148], color='#D55E00', ls='--', lw=.7, label='Crop')
            ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)', aspect='equal', xlim=(-125,125), ylim=(-175,175))
            ax.set_xticks([-100,0,100]); ax.set_yticks([-150,-100,-50,0,50,100,150])
        axes[-1].legend(loc='lower left', fontsize=7)
        cbar = fig.colorbar(artist, ax=axes.tolist(), label='Height (mm)', shrink=.8)
        ticks = np.linspace(lo, hi, 5)
        cbar.set_ticks(ticks, labels=[f'{v:.1f}' for v in ticks])
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        for text in fig.findobj(matplotlib.text.Text):
            if text.get_visible() and text.get_text():
                b = text.get_window_extent(renderer)
                if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                    raise AssertionError('Figure text outside canvas: '+text.get_text())
        for ext in ('png','pdf'):
            dest = ROOT / f'run6_refined_frame_comparison.{ext}'
            if dest.exists():
                raise FileExistsError(dest)
            fig.savefig(dest, dpi=250)
        plt.close(fig)
    caption = ('Run6 fixed scan-frame comparison. One calibration specimen; unchanged x-turnover, '
               'geometric additional yaw +0.05586775269924428 degrees, no curvature/RMS fitting. '
               'Uncropped panels use the same fixed-seed 2026 subset (up to 40,000 unique raw vertices); '
               'target panel uses all prepared cells. Shared height scale and equal spatial aspect. '
               'The original panel is shifted only in z to the new common height origin. '
               'Nominal CSV path and crop are overlays, not identified physical landmarks. '
               'No statistical uncertainty or improvement in measured physical fit is established.\n')
    with (ROOT/'run6_refined_frame_comparison.caption.md').open('x') as f:
        f.write(caption)
    return {'display_seed':2026, 'display_count':len(indices),
            'outside_axes_before':int(np.any(np.abs(before[indices,:2]) > [125,175], axis=1).sum()),
            'outside_axes_after':int(np.any(np.abs(aligned[indices,:2]) > [125,175], axis=1).sum())}


def prepare():
    manifest, policy = baseline()
    destinations = [BASE/TARGET, (BASE/TARGET).with_suffix('.json'), (BASE/TARGET).with_suffix('.png'), BASE/CONFIG]
    if ROOT.exists() or OUT.exists() or any(p.exists() for p in destinations):
        raise FileExistsError('Preserve previous/partial refined migration')
    ROOT.mkdir()
    old_meta = read(original.snap('calibration/data/processed/run6.json'))
    transform = old_meta['transform']; basis = np.array(transform['basis_columns'])
    turn = np.diag([1.,-1.,-1.]); old_rotation = turn @ yaw_matrix(1.5).T
    with np.load(BASE/'calibration/data/processed/run6.npz', allow_pickle=False) as f:
        base_cloud = f['points_mm'].copy()
    with np.load(BASE/'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as f:
        old_cloud = f['points_mm'].copy()
    np.testing.assert_allclose(base_cloud @ old_rotation, old_cloud, atol=1e-9, rtol=0)
    raw = np.unique(points(read_mesh(BASE/'calibration/data/raw/run6.stl')), axis=0)
    before = ((raw-np.array(transform['origin_scanner_mm'])) @ basis-np.array(transform['shift_local_mm'])) @ old_rotation
    primary = raw_edges(before[:,:2])
    angle, rotation = fixed_rotation(policy, primary['mean_angle_deg'])
    after = before @ rotation.T
    np.testing.assert_array_equal(before[:,2], after[:,2])
    np.testing.assert_allclose(np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1), atol=1e-10, rtol=0)
    full_basis = basis @ old_rotation @ rotation.T
    np.testing.assert_allclose(full_basis.T @ full_basis, np.eye(3), atol=1e-12, rtol=0)
    config = read(original.snap('calibration/configs/run6_bo_material_frame.json'))
    config['cases'][0]['target'] = TARGET
    config['cases'][0]['scan_orientation'] = {'basis_columns':full_basis.tolist(), 'flip_z':False, 'flip_xy':False}
    config['output_dir'] = str(OUT.relative_to(BASE)); config['report_dir'] = str((OUT/'report').relative_to(BASE))
    metadata = prepare_case(config, config['cases'][0])
    metadata.update(edge_policy='fixed_refined_raw_boundary', additional_yaw_deg=angle,
                    total_yaw_after_turnover_deg=1.5+angle, geometric_policy_sha256=sha(POLICY))
    (BASE/TARGET).with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    save(BASE/CONFIG, config)
    finish()


def finish():
    """Finalize diagnostics from already-created immutable target; never rebuild it."""
    manifest, policy = baseline()
    if (ROOT/'prepared.json').exists() or OUT.exists():
        raise FileExistsError('Preparation already finalized or replay started')
    metadata = read((BASE/TARGET).with_suffix('.json')); config = read(BASE/CONFIG)
    target_hash, config_hash = sha(BASE/TARGET), sha(BASE/CONFIG)
    old_meta = read(original.snap('calibration/data/processed/run6.json'))
    transform = old_meta['transform']; basis = np.array(transform['basis_columns'])
    old_rotation = np.diag([1.,-1.,-1.]) @ yaw_matrix(1.5).T
    raw = np.unique(points(read_mesh(BASE/'calibration/data/raw/run6.stl')), axis=0)
    before = ((raw-np.array(transform['origin_scanner_mm'])) @ basis-np.array(transform['shift_local_mm'])) @ old_rotation
    angle, rotation = fixed_rotation(policy, raw_edges(before[:,:2])['mean_angle_deg'])
    expected_basis = basis @ old_rotation @ rotation.T
    np.testing.assert_allclose(config['cases'][0]['scan_orientation']['basis_columns'], expected_basis, atol=1e-12, rtol=0)
    np.testing.assert_allclose(metadata['transform']['basis_columns'], expected_basis, atol=1e-12, rtol=0)
    if metadata['additional_yaw_deg'] != angle or metadata['geometric_policy_sha256'] != sha(POLICY):
        raise ValueError('Prepared metadata differs from selected geometry policy')
    with np.load(BASE/'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as f:
        old_cloud = f['points_mm'].copy()
    with np.load(BASE/TARGET, allow_pickle=False) as f:
        cloud = f['points_mm'].copy()
    new_transform = metadata['transform']
    aligned = ((raw-np.array(new_transform['origin_scanner_mm'])) @ np.array(new_transform['basis_columns'])
               - np.array(new_transform['shift_local_mm']))
    height_shift = float(np.median(aligned[:,2]-before[:,2]))
    np.testing.assert_allclose(aligned[:,2]-before[:,2], height_shift, atol=1e-10, rtol=0)
    before_display = before.copy(); before_display[:,2] += height_shift
    display = figure(before_display, aligned, cloud)
    # Compare crop membership in the two complete fixed coordinate gauges.
    def retained(p):
        z = p[:,2]-np.median(p[:,2]); floor = config['preprocess']['z_outlier_floor_mm']
        return np.all(np.abs(p[:,:2]) <= [98,148], axis=1) & (np.abs(z) <= max(floor,12*np.median(np.abs(z))))
    old_keep, new_keep = retained(before), retained(aligned)
    baseline()
    if sha(BASE/TARGET) != target_hash or sha(BASE/CONFIG) != config_hash:
        raise ValueError('Target/config changed during diagnostic finalization')
    save(ROOT/'prepared.json', {'status':'prepared', 'additional_yaw_deg':angle, 'total_yaw_deg':1.5+angle,
        'target_sha256':sha(BASE/TARGET), 'config_sha256':sha(BASE/CONFIG), 'policy_sha256':sha(POLICY),
        'source_sha256':sha(Path(__file__)), 'old_point_count':len(old_cloud), 'new_point_count':len(cloud),
        'height_shift_mm':height_shift, 'crop_membership':{'old_retained':int(old_keep.sum()), 'new_retained':int(new_keep.sum()),
        'intersection':int((old_keep & new_keep).sum()), 'newly_included':int((new_keep & ~old_keep).sum()),
        'newly_excluded':int((old_keep & ~new_keep).sum())}, 'display':display,
        'tail_sensitivity_additional_yaw_deg':policy['tail_sensitivity']['mean_angle_deg'],
        'residual_edges':policy['residual_edges'], 'metadata':metadata, 'slurm_job':os.environ.get('SLURM_JOB_ID'),
        'python':sys.version, 'numpy':np.__version__, 'scope':'Fixed target preparation only; no mechanics/GP/BO.'})
    print('PREPARED', angle, len(cloud), flush=True)


def preflight():
    manifest, policy = baseline()
    prepared = read(ROOT/'prepared.json'); config = read(BASE/CONFIG)
    if sha(BASE/TARGET) != prepared['target_sha256'] or sha(BASE/CONFIG) != prepared['config_sha256']:
        raise ValueError('Prepared target/config changed')
    prior = read(original.snap('calibration/configs/run6_bo_material_frame.json'))
    normalized = copy.deepcopy(config)
    # Only target/orientation and isolated destinations may change.
    normalized['cases'][0]['target'] = prior['cases'][0]['target']
    normalized['cases'][0]['scan_orientation'] = prior['cases'][0]['scan_orientation']
    normalized['output_dir'] = prior['output_dir']; normalized['report_dir'] = prior['report_dir']
    if normalized != prior or config['output_dir'] != str(OUT.relative_to(BASE)):
        raise ValueError('Unapproved scientific recipe or output destination')
    identity, target, eid = replay.current_bindings(config)
    source_identity = read(original.snap('run/run6_bo_material_frame/corrected_history.json'))['identity_inputs']
    if identity['executable'] != source_identity['executable']:
        raise ValueError('Solver executable identity changed')
    with np.load(BASE/'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as f:
        old_target = f['points_mm'].copy()
    receipts = []
    for attempt in (2,25,41):
        entry = next(row for row in manifest['sources'] if row['attempt_id'] == attempt)
        mesh, ref, faces, weights = replay.load(entry)
        old, *_ = score_material_frame(mesh, old_target, ref, faces)
        error = abs(old['rms_symmetric_mm']-entry['old_rms_mm'])
        if error > 1e-6:
            raise ValueError(f'Old-policy fidelity failed: attempt{attempt}, {error}')
        new, *_ = score_material_frame(mesh, target, ref, faces)
        receipts.append({'attempt_id':attempt, 'old_rms_mm':entry['old_rms_mm'], 'replayed_old_rms_mm':old['rms_symmetric_mm'],
                         'absolute_error_mm':error, 'refined_rms_mm':new['rms_symmetric_mm']})
    pins = dict(manifest['original_input_sha256'])
    paths = [CONFIG, TARGET, str((BASE/TARGET).with_suffix('.json').relative_to(BASE)), PROTOCOL,
             str(POLICY.relative_to(BASE)), 'calibration/analysis/inspect_subbin_run6.py',
             'calibration/analysis/refined_edge_migration.py', 'calibration/tests/test_refined_edge_migration.py',
             'calibration/analysis/migrate_run6_edge_frame.py', 'calibration/analysis/rescore_run6_edge_history.py',
             'calibration/tests/test_edge_history_migration.py', config['solver']['executable'],
             str((original.ROOT/'manifest.json').relative_to(BASE)), str((ROOT/'prepared.json').relative_to(BASE)),
             'calibration/configs/run6_bo_edge_aligned.json', 'calibration/data/processed/run6_edge_aligned.npz',
             'calibration/data/processed/run6_edge_aligned.json', 'run/run6_bo_edge_aligned/concurrent_migration_notice.json']
    paths += [str(p.relative_to(BASE)) for p in (BASE/'calibration/ewcal').glob('*.py')]
    for path in paths:
        actual = sha(BASE/path)
        if path in pins and pins[path] != actual:
            raise ValueError('Preserved source changed: '+path)
        pins[path] = actual
    baseline()
    save(ROOT/'preflight.json', {'status':'passed', 'fixtures':receipts, 'slurm_job':os.environ.get('SLURM_JOB_ID')})
    save(ROOT/'rescore_bindings.json', {'input_sha256':pins, 'identity_inputs':identity, 'evaluator_identity':eid, 'config':config})
    print(json.dumps(receipts, indent=2), flush=True)


def checked_bindings():
    manifest, _ = baseline(); binding = read(ROOT/'rescore_bindings.json')
    for path, expected in binding['input_sha256'].items():
        if sha(BASE/path) != expected:
            raise ValueError('Reviewed migration input changed: '+path)
    return manifest, binding


def rescore():
    manifest, binding = checked_bindings()
    clearance = read(ROOT/'review_clearance.json')
    if clearance['verdict'] not in ('APPROVED','APPROVED_WITH_CAVEATS') or clearance['launch_blockers']:
        raise ValueError('Missing reviewed clearance')
    for path, expected in clearance['reviewed_artifact_sha256'].items():
        if sha(BASE/path) != expected:
            raise ValueError('Review/packet changed: '+path)
    config = binding['config']; identity, target, eid = replay.current_bindings(config)
    if identity != binding['identity_inputs'] or eid != binding['evaluator_identity']:
        raise ValueError('Runtime evaluator identity changed')
    OUT.mkdir(exist_ok=False)
    save(OUT/'migration_started.json', {'bindings_sha256':sha(ROOT/'rescore_bindings.json'),
         'clearance_sha256':sha(ROOT/'review_clearance.json'), 'slurm_job':os.environ.get('SLURM_JOB_ID')})
    originals = {row['attempt_id']:row for row in manifest['observations']}
    results, table = {}, []; start = time.monotonic()
    for entry in sorted(manifest['sources'], key=lambda row:row['attempt_id']):
        for key in ('mesh_file','state_file','sequence_file'):
            if sha(BASE/entry[key]) != binding['input_sha256'][entry[key]]:
                raise ValueError('Saved shape drift: '+entry[key])
        mesh, ref, faces, weights = replay.load(entry)
        metrics, framed, scan, forward, closest = score_material_frame(mesh, target, ref, faces)
        folder = OUT/'rescored_shapes'/f'attempt_{entry["attempt_id"]:04d}'
        folder.mkdir(parents=True, exist_ok=False)
        prediction = points(framed); reverse, _ = cKDTree(scan).query(prediction)
        row = replay.migrated_observation(originals[entry['attempt_id']], metrics, eid, str(folder.relative_to(BASE)))
        np.savez_compressed(folder/'comparison.npz', target_mm=scan, prediction_mm=prediction, faces=faces,
                            distance_mm=forward, closest_mm=closest, reverse_distance_mm=reverse, reference_area_weights=weights)
        save(folder/'result.json', {'attempt_id':entry['attempt_id'], 'parameters':entry['parameters'],
             'metrics':metrics, 'source':entry, 'evaluator_identity':eid})
        results[entry['attempt_id']] = row
        table.append({'attempt_id':entry['attempt_id'], 'old_rms_mm':entry['old_rms_mm'],
                      'refined_rms_mm':metrics['rms_symmetric_mm'], 'difference_mm':metrics['rms_symmetric_mm']-entry['old_rms_mm']})
        print('RESCORE', entry['attempt_id'], metrics['rms_symmetric_mm'], flush=True)
    observations = [results.get(row['attempt_id']) or replay.migrated_observation(row,None,eid,None)
                    for row in manifest['observations']]
    if len(results) != 70 or [r['attempt_id'] for r in observations] != list(range(1,81)):
        raise ValueError('Incomplete migration')
    for row in observations:
        old = originals[row['attempt_id']]
        if row['parameters'] != old['parameters'] or row['normalized_parameters'] != old['normalized_parameters']:
            raise ValueError('Parameter/attempt identity changed')
        if not row['valid'] and row['objective_mm'] is not None:
            raise ValueError('Timeout became a measured loss')
    checked_bindings()
    history = {'schema_version':1, 'config':config, 'evaluator_identity':eid, 'identity_inputs':identity,
        'observations':observations, 'input_sha256':binding['input_sha256'], 'target_policy':'fixed_refined_raw_boundary',
        'scoring_method':'material_reference', 'no_new_mechanics_or_gp':True, 'not_a_resumed_checkpoint':True}
    save(OUT/'corrected_history.json', history)
    best = min(results.values(), key=lambda row:row['objective_mm'])
    differences = np.array([row['difference_mm'] for row in table])
    save(OUT/'migration_summary.json', {'status':'complete', 'attempt_count':80, 'scored_count':70,
        'timeout_ids':manifest['timeout_ids'], 'best_attempt_id':best['attempt_id'], 'best_rms_mm':best['objective_mm'],
        'best_parameters':best['parameters'], 'evaluator_identity':eid, 'history_sha256':sha(OUT/'corrected_history.json'),
        'score_table':table, 'paired_difference_quantiles_mm':dict(zip(('min','p05','median','p95','max'),
             np.quantile(differences,[0,.05,.5,.95,1]).tolist())), 'seconds':time.monotonic()-start,
        'slurm_job':os.environ.get('SLURM_JOB_ID'),
        'interpretation':'Same saved shapes, changed target policy; not BO improvement or physical parameter identification.'})
    save(OUT/'best.json', {'attempt_id':best['attempt_id'], 'parameters':best['parameters'],
         'objective_mm':best['objective_mm'], 'metrics':best['metrics'], 'evaluator_identity':eid,
         'source':'rescored_existing_shapes'})
    print('COMPLETE', best['attempt_id'], best['objective_mm'], flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare','finish','preflight','rescore'))
    args = parser.parse_args(); globals()[args.stage]()
