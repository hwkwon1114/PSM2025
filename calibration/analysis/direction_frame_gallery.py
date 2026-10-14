"""Fixed material-reference directions, centroid translation, all saved deformations."""
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from calibration.analysis import reassess_failed_registration as prior
from calibration.analysis.diagnose_free_translation import evaluate_saved_pose
from calibration.ewcal.geometry import points, reference_dual_areas, rigid_fit, triangle_mesh

BASE = prior.BASE
PLAN = BASE / 'run/optimization_review/free_translation_plan.json'
OUT = BASE / 'run/optimization_review/direction_frame_gallery'
BLUE, ORANGE = '#0072B2', '#D55E00'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9,
         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'pdf.fonttype': 42, 'figure.facecolor': 'white'}


def direction_frame(p, reference, faces):
    """Proper material-correspondence rotation; no experimental points enter the fit."""
    p, reference = np.asarray(p, float), np.asarray(reference, float)
    if p.shape != reference.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Matched final/reference vertices required')
    if not np.isfinite(p).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite frame input')
    covariance = (p - p.mean(0)).T @ (reference - reference.mean(0))
    singular = np.linalg.svd(covariance, compute_uv=False)
    # The undeformed sheet is planar (rank two), not a 3D landmark set.
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-10:
        raise ValueError('Material direction frame is degenerate')
    rotation, _ = rigid_fit(p, reference)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
    assert abs(np.linalg.det(rotation) - 1) < 1e-12
    areas = reference_dual_areas(reference, faces)
    weights = areas / areas.sum()
    centroid = np.sum(weights[:, None] * p, axis=0)
    translation = -centroid @ rotation
    q = p @ rotation + translation
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    error = float(np.max(np.abs(np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1) -
                                    np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1))))
    assert error <= 1e-9, 'Rigid frame changed shape'
    assert np.linalg.norm(np.sum(weights[:, None] * q, axis=0)) < 1e-9
    return q, {'row_vector_rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
               'centroid_original_mm': centroid.tolist(), 'singular_values': singular.tolist(),
               'direction_condition_ratio': float(singular[1] / singular[0]),
               'max_edge_length_change_mm': error, 'determinant': float(np.linalg.det(rotation))}


def check_layout(fig):
    """Reusable title/legend/annotation checks, including tick and axis-label canvas extents."""
    from calibration.analysis.plot_deformation_constraints import check_layout as check
    check(fig)
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        # Locator-generated ticks outside view limits are not actually drawn.
        drawn = [ax.xaxis.label, ax.yaxis.label]
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in axis.get_major_ticks():
                if min(limits) <= tick.get_loc() <= max(limits):
                    drawn.extend([tick.label1, tick.label2])
        for text in drawn:
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, f'Axis text outside canvas: {text.get_text()!r} {box.bounds} canvas {fig.bbox.bounds}'


def save(report):
    (OUT / 'direction_frame_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def render_gallery(report, geometry, output=OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    scan = next(iter(geometry.values()))[1]
    bounds = np.vstack([np.vstack([sim.min(0), sim.max(0), target.min(0), target.max(0)])
                        for sim, target in geometry.values()])
    low, high = bounds.min(0), bounds.max(0)
    margin = np.maximum((high - low) * .035, 1.)
    low, high = low - margin, high + margin
    indices = np.linspace(0, len(scan)-1, min(2500, len(scan)), dtype=int)
    records = {r['attempt_id']: r for r in report['records']}
    missing = {r['attempt_id']: r for r in report['excluded']}
    ordered = list(range(33, 41)) + list(range(1, 33))
    legend = [Line2D([], [], color=BLUE, marker='o', ls='none', label='Simulation'),
              Line2D([], [], color=ORANGE, marker='x', ls='none', label='Scan')]
    pages = []
    with plt.style.context(STYLE), PdfPages(output / 'direction_frame_gallery.pdf') as pdf:
        for page, start in enumerate(range(0, len(ordered), 4), 1):
            fig = plt.figure(figsize=(12, 11), layout='constrained')
            # mplot3d labels protrude past axes; reserve an explicit outer margin.
            fig.get_layout_engine().set(rect=(.035, .055, .93, .875))
            fig.suptitle(f'Direction-first comparison — page {page}', y=.993, fontsize=12)
            fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .97), ncol=2, frameon=False)
            grid = fig.add_gridspec(4, 3)
            for row, att in enumerate(ordered[start:start+4]):
                data = geometry.get(att)
                ax3 = fig.add_subplot(grid[row, 0], projection='3d' if data else None)
                axx = fig.add_subplot(grid[row, 1]); axy = fig.add_subplot(grid[row, 2])
                if data is None:
                    assert att in missing
                    ax3.set_title(f'{att:02d} | mechanics timeout', loc='left')
                    for ax in [ax3, axx, axy]:
                        ax.set_axis_off()
                    axx.text(.5, .5, 'No completed mechanical shape', ha='center', va='center', transform=axx.transAxes)
                    continue
                sim, target = data; r = records[att]
                ax3.set_title(f'{att:02d} | RMS {r["metrics"]["rms_symmetric_mm"]:.2f} mm', loc='left')
                sample = np.linspace(0, len(sim)-1, min(2000, len(sim)), dtype=int)
                ax3.scatter(*sim[sample].T, c=BLUE, marker='o', s=1, alpha=.65, rasterized=True)
                ax3.scatter(*target[indices].T, c=ORANGE, marker='x', s=2, alpha=.7, rasterized=True)
                ax3.set(xlim=(low[0],high[0]), ylim=(low[1],high[1]), zlim=(low[2],high[2]), xlabel='x (mm)', ylabel='y (mm)')
                ax3.set_zticks([]); ax3.xaxis.labelpad=0; ax3.yaxis.labelpad=0
                ax3.set_xticks([-100,0,100]); ax3.set_yticks([-150,0,150])
                ax3.set_box_aspect(high-low); ax3.view_init(elev=24, azim=-62)
                for ax, dim, name in [(axx, 0, 'Short'), (axy, 1, 'Long')]:
                    ax.scatter(sim[:, dim], sim[:, 2], c=BLUE, s=1, marker='o', alpha=.35, rasterized=True)
                    ax.scatter(target[indices, dim], target[indices, 2], c=ORANGE, s=3, marker='x', alpha=.7, rasterized=True)
                    ax.set(xlim=(low[dim], high[dim]), ylim=(low[2],high[2]), xlabel=f'{name} {"xy"[dim]} (mm)', ylabel='z (mm)', aspect='equal')
                    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(4))
                    ax.spines[['top','right']].set_visible(False)
                axx.set_title('Short-direction projection', loc='left')
                axy.set_title('Long-direction projection', loc='left')
            check_layout(fig)
            png = output / f'direction_frame_gallery_page_{page:02d}.png'
            fig.savefig(png, dpi=150); pdf.savefig(fig, dpi=180)
            plt.close(fig); pages.append(png.name)
            print('PAGE', page, flush=True)
    report.update(gallery_pages=pages, gallery_pdf='direction_frame_gallery.pdf',
                  shared_limits_mm=[low.tolist(), high.tolist()], scan_display_indices=indices.tolist())


def main():
    import scipy
    import vtk
    assert os.environ.get('SLURM_JOB_ID'), 'Owned compute allocation required'
    assert not OUT.exists(), 'Preserve prior diagnostic output'
    plan = json.loads(PLAN.read_text())
    prior.verify_inputs(plan)
    assert len(plan['endpoints']) == 34
    assert [r['attempt_id'] for r in plan['excluded']] == [17,24,27,35,37,39]
    names = [str(PLAN.relative_to(BASE)), 'calibration/analysis/direction_frame_gallery.py',
             'calibration/tests/test_direction_frame.py', 'calibration/analysis/plot_deformation_constraints.py',
             'calibration/data/processed/run6_oriented.json']
    pins = {**plan['input_sha256'], **{n: prior.sha(BASE/n) for n in names}}
    with np.load(BASE / 'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as f:
        raw_scan = f['points_mm'].copy()
    assert raw_scan.shape == (14503,3) and np.isfinite(raw_scan).all()
    scan_centroid = raw_scan.mean(0)
    scan = raw_scan - scan_centroid
    OUT.mkdir()
    report = {'schema_version': 1, 'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
        'hostname': platform.node(), 'python': sys.version, 'numpy': np.__version__,
        'scipy': scipy.__version__, 'vtk': vtk.vtkVersion.GetVTKVersion(),
        'input_sha256': pins, 'scan_centroid_original_mm': scan_centroid.tolist(),
        'scan_metadata': prior.read('calibration/data/processed/run6_oriented.json'),
        'algorithm': 'proper matched material-reference Kabsch rotation; simulation reference-area centroid and scan point centroid at zero; no cross-shape fitting',
        'claim_limit': 'Nominal-frame diagnostic only; experimental material landmarks unverified, production labels/GP not changed',
        'excluded': plan['excluded'], 'records': []}
    save(report)
    start = time.monotonic(); geometry = {}
    for endpoint in plan['endpoints']:
        mesh, ref, faces = prior.load_geometry(endpoint, plan['config'])
        p = points(mesh)
        q, transform = direction_frame(p, ref, faces)
        metric, coverage = evaluate_saved_pose(triangle_mesh(q, faces), scan, ref, faces, np.zeros(6))
        assert all(np.isfinite(v) for v in metric.values())
        assert np.allclose(np.ptp(ref, axis=0), [200,300,0], atol=1e-6)
        row = {'attempt_id': endpoint['attempt_id'], 'parameters': endpoint['parameters'],
               'historical_valid': endpoint['original_valid'], 'shape_scored': True,
               'historical_rms_mm': endpoint['saved_metrics']['rms_symmetric_mm'],
               'metrics': metric, 'proximity': coverage, 'simulation_transform': transform}
        report['records'].append(row); geometry[row['attempt_id']] = (q, scan)
        save(report)
        print('SCORE', row['attempt_id'], metric['rms_symmetric_mm'], flush=True)
    assert len(report['records']) == 34 and all(r['shape_scored'] for r in report['records'])
    values = np.array([r['metrics']['rms_symmetric_mm'] for r in report['records']])
    report['aggregate'] = {'count': 34, 'rms_mm_quantiles_0_25_50_75_100': np.quantile(values,[0,.25,.5,.75,1]).tolist(),
        'historically_qualified_count': sum(r['historical_valid'] for r in report['records']),
        'historically_rejected_but_now_scored_count': sum(not r['historical_valid'] for r in report['records']),
        'best_attempt_id': min(report['records'], key=lambda r: r['metrics']['rms_symmetric_mm'])['attempt_id']}
    columns = ['attempt_id', 'historical_valid', 'historical_rms_mm', 'direction_rms_mm',
               'scan_to_sim_mm', 'sim_to_scan_mm', 'gtop', 'gbot', 'ortho_top', 'ortho_bottom',
               'direction_condition_ratio', 'scan_within5mm', 'sim_area_within5mm']
    with (OUT/'direction_frame_metrics.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for r in report['records']:
            m = r['metrics']; c = r['proximity']['5']
            writer.writerow({'attempt_id': r['attempt_id'], 'historical_valid': r['historical_valid'],
                'historical_rms_mm': r['historical_rms_mm'], 'direction_rms_mm': m['rms_symmetric_mm'],
                'scan_to_sim_mm': m['rms_target_to_sim_mm'], 'sim_to_scan_mm': m['rms_sim_to_target_weighted_mm'],
                **r['parameters'], 'direction_condition_ratio': r['simulation_transform']['direction_condition_ratio'],
                'scan_within5mm': c['scan_fraction'], 'sim_area_within5mm': c['simulation_area_fraction']})
    report['analysis_complete'] = True; save(report)
    render_gallery(report, geometry)
    import matplotlib
    report['matplotlib'] = matplotlib.__version__
    prior.verify_inputs({'input_sha256': pins})
    report.update(status='complete', inputs_unchanged=True, elapsed_seconds=time.monotonic()-start,
                  automatic_layout_checks='all ten pages; title/annotation/legend overlap and axis-text canvas bounds')
    save(report)
    caption = '''# Direction-first deformation gallery

All 40 terminal attempt IDs shown, new33–40 first then1–32, ordered independently of loss. All34 mechanically completed shapes get a finite full symmetric RMS, including20 previously registration-rejected shapes; six timeouts have placeholders, not zero scores. This is one adaptive calibration scan, not held-out validation or independent replicates.

Simulation orientation uses a PROPER matched-vertex rigid Kabsch rotation onto its undeformed material reference coordinates, not PCA of deformed bounding boxes or rotation fitted against experimental curvature. Reference x=200mm short, y=300mm long. No scaling, deformation, reflection, per-case face flip, curve-matching ICP or translation caps. Each simulation's reference-area-weighted centroid and the scan's point centroid are independently zeroed. Scan directions/signs remain the fixed preprocessed convention (documented x-turnover plus1.5degree nominal-axis yaw); actual scan material edge/corner labels are NOT available or independently verified. The closed-form material-reference frame is a nominal-direction diagnostic, not proof of exact physical correspondence; strongly folded geometry may make the global frame ambiguous, condition ratios recorded in JSON/CSV. Centroid choice can retain discrepancies from scan cropping/midsurface approximation.

Blue circles=simulation vertices, orange crosses=scan. Identical physical limits/camera/equal aspect for all shapes. Short/long projections are full point projections, not sections. All simulation vertices in projections; uniformly indexed≤2000vertices in3D, same2500of14503scanpoints for display. Metrics use ALL14503scanpoints and all simulation vertices/reference-area reverse weights, not displayed subsets. Dense scatter rasterized withinPDF, text/axes vector. No uncertainty intervals. 3D z ticks omitted for readability; projections quantify z. Historical RMS/qualification retained as annotations in CSV, NOT filtering/gates or comparable same-policy best claims. Large RMS is an intended shape disagreement, not alignment failure.

This gallery/scoring does not change production BO losses, GP training, labels, source/config or mechanical results. Reproduction: python -m calibration.analysis.direction_frame_gallery in owned compute allocation; provenance/transforms/parameters/aggregate in direction_frame_results.json, full table in direction_frame_metrics.csv. Visual inspection status recorded separately in existing verification_summary.md.
'''
    (OUT/'direction_frame_gallery.caption.md').write_text(caption)
    print('COMPLETE', report['aggregate'], flush=True)


def render_saved():
    """Layout-only recovery from all 34 completed, frozen direct scores."""
    assert os.environ.get('SLURM_JOB_ID')
    report_path = OUT / 'direction_frame_results.json'
    result_digest = prior.sha(report_path)
    report = json.loads(report_path.read_text())
    assert report['analysis_complete'] and len(report['records']) == 34
    plan = json.loads(PLAN.read_text())
    prior.verify_inputs(plan)
    frozen = {n:h for n,h in report['input_sha256'].items()
              if n != 'calibration/analysis/direction_frame_gallery.py'}
    prior.verify_inputs({'input_sha256': frozen})
    with np.load(BASE / 'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as f:
        scan = f['points_mm'] - np.array(report['scan_centroid_original_mm'])
    rows = {r['attempt_id']:r for r in report['records']}
    geometry = {}
    for endpoint in plan['endpoints']:
        mesh, _, _ = prior.load_geometry(endpoint, plan['config'])
        transform = rows[endpoint['attempt_id']]['simulation_transform']
        q = points(mesh) @ np.array(transform['row_vector_rotation']) + np.array(transform['translation_mm'])
        geometry[endpoint['attempt_id']] = (q, scan)
    output = OUT / 'rendered_gallery_v2'
    output.mkdir(exist_ok=False)
    render_gallery(report, geometry, output)
    prior.verify_inputs({'input_sha256': frozen})
    assert prior.sha(report_path) == result_digest, 'Saved scores changed during render'
    record = {'status':'render_complete', 'job_id':os.environ['SLURM_JOB_ID'],
        'input_result_sha256':prior.sha(report_path), 'input_plan_sha256':prior.sha(PLAN),
        'render_source_sha256':prior.sha(Path(__file__)), 'frozen_inputs_unchanged':True,
        'scoring_repeated':False, 'gallery_pages':report['gallery_pages'],
        'shared_limits_mm':report['shared_limits_mm'], 'scan_display_indices':report['scan_display_indices']}
    (output/'render_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('RENDER_COMPLETE', len(geometry), 'shapes, ten pages', flush=True)


if __name__ == '__main__':
    assert sys.flags.optimize == 0
    if sys.argv[1:] == ['--render-saved']:
        render_saved()
    else:
        assert not sys.argv[1:]
        main()
