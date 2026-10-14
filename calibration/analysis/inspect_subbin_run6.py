"""Resolve bin-center yaw quantization using the same uncropped sheet boundaries."""
import os
from pathlib import Path
import numpy as np
from scipy import ndimage, stats
from calibration.analysis.edge_align_run6 import ROOT, OUT, read, verify as verify_pins, write_new
from calibration.analysis.orient_run9_edges import yaw_matrix
from calibration.ewcal.common import digest
from calibration.ewcal.geometry import points, read_mesh


def verify(pins):
    amendment = read(OUT / 'authority_amendment.json')
    if (amendment['path'] != 'run/stage2_bayesian_optimization/stage2_cluster_protocol.md' or
            amendment['before_sha256'] != pins[amendment['path']] or
            amendment['reconstructed_before_matches_frozen'] is not True):
        raise ValueError('Unrecognized authority amendment')
    verify_pins(dict(pins, **{amendment['path']: amendment['after_sha256']}))


def raw_edges(xy, tail=.05, spacing=1.):
    keys = np.floor(xy / spacing).astype(int)
    origin = keys.min(0)
    shifted = keys - origin
    shape = tuple(shifted.max(0) + 1)
    if max(shape) > 2000:
        raise ValueError('Unexpected extent')
    counts = np.zeros(shape, dtype=int)
    np.add.at(counts, (shifted[:, 0], shifted[:, 1]), 1)
    labels, n = ndimage.label(counts >= 5, structure=np.ones((3, 3)))
    if not n:
        raise ValueError('No dense component')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    ix, iy = np.where(labels == np.argmax(sizes))
    rows = np.arange(int(np.ceil(iy.min() + .2*np.ptp(iy))), int(np.floor(iy.max() - .2*np.ptp(iy))) + 1)
    ys, left, right, vertices = [], [], [], []
    for row in rows:
        columns = ix[iy == row]
        if not len(columns):
            continue
        row_points = xy[shifted[:, 1] == row]
        # One neighboring cell recovers sparse partial boundary cells; fragments
        # farther from the dense boundary are not part of the edge estimate.
        xleft = row_points[(row_points[:, 0] >= (columns.min()+origin[0]-1)*spacing) &
                           (row_points[:, 0] < (columns.min()+origin[0]+1)*spacing), 0]
        xright = row_points[(row_points[:, 0] >= (columns.max()+origin[0])*spacing) &
                            (row_points[:, 0] < (columns.max()+origin[0]+2)*spacing), 0]
        if min(len(xleft), len(xright)) < 5:
            continue
        ys.append((row+origin[1]+.5)*spacing)
        left.append(float(np.quantile(xleft, tail)))
        right.append(float(np.quantile(xright, 1-tail)))
        vertices.append([len(xleft), len(xright)])
    if len(ys) < 20:
        raise ValueError('Too few raw boundary rows')
    fits = [stats.theilslopes(values, ys) for values in (left, right)]
    angles = [float(np.degrees(np.arctan(f.slope))) for f in fits]
    intervals = [[float(np.degrees(np.arctan(f.low_slope))), float(np.degrees(np.arctan(f.high_slope)))] for f in fits]
    return {'angles_deg': angles, 'mean_angle_deg': float(np.mean(angles)),
            'slope_intervals_deg': intervals, 'rows': len(ys), 'tail_fraction': tail,
            'spacing_mm': spacing, 'row_coordinate_mm': ys, 'left_coordinate_mm': left,
            'right_coordinate_mm': right, 'boundary_vertex_counts': vertices,
            'fit_intercepts_mm': [float(f.intercept) for f in fits],
            'interval_limit': 'Descriptive Theil-Sen slope intervals; scan rows/vertices are dependent, not calibrated specimen uncertainty.'}


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    report_path = OUT / 'subbin_edge_inspection.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    frozen = read(OUT / 'frozen_campaign.json')
    verify(frozen['input_sha256'])
    meta = read(ROOT / 'calibration/data/processed/run6.json')['transform']
    raw = np.unique(points(read_mesh(ROOT / 'calibration/data/raw/run6.stl')), axis=0)
    before = ((raw - np.asarray(meta['origin_scanner_mm'])) @ np.asarray(meta['basis_columns']) - np.asarray(meta['shift_local_mm'])) @ np.diag([1., -1., -1.]) @ yaw_matrix(1.5).T
    primary = raw_edges(before[:, :2])
    sensitivity = raw_edges(before[:, :2], tail=.1)
    angle = primary['mean_angle_deg']
    after = before @ yaw_matrix(angle).T
    residual = {'long': raw_edges(after[:, :2]), 'short': raw_edges(after[:, [1, 0]])}
    sampled = np.load(OUT / 'uncropped_display_sample.npz', allow_pickle=False)
    display_before = sampled['before'].copy()
    display_after = display_before @ yaw_matrix(angle).T
    np.savez_compressed(OUT / 'subbin_display_sample.npz', before=display_before, after=display_after)
    report = {'status': 'subbin_inspection_only_no_target', 'job_id': os.environ['SLURM_JOB_ID'],
              'source_sha256': digest(Path(__file__)), 'primary': primary, 'tail_sensitivity': sensitivity,
              'additional_yaw_deg': angle, 'total_yaw_after_turnover_deg': 1.5+angle,
              'residual_edges': residual, 'height_unchanged_exactly': bool(np.array_equal(before[:, 2], after[:, 2])),
              'composed_scanner_basis': (np.asarray(meta['basis_columns']) @ np.diag([1., -1., -1.]) @ yaw_matrix(1.5+angle).T).tolist()}
    verify(frozen['input_sha256'])
    write_new(report_path, report)
    print({k: {q:v for q,v in e.items() if q in ('angles_deg','mean_angle_deg','slope_intervals_deg','rows')} for k,e in [('primary',primary), ('sensitivity',sensitivity), *residual.items()]}, flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2), layout='constrained')
    lo, hi = display_before[:, 2].min(), display_before[:, 2].max()
    for ax, cloud, title in zip(axes, [display_before, display_after], ['Current scan frame', 'Proposed edge frame']):
        artist = ax.scatter(cloud[:,0],cloud[:,1],c=cloud[:,2],s=.5,cmap='cividis',vmin=lo,vmax=hi,rasterized=True)
        ax.plot([-100,100,100,-100,-100],[-150,-150,150,150,-150],color='#D55E00',linestyle='--',lw=.8)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title=title,aspect='equal',xlim=(-125,125),ylim=(-180,180))
    fig.colorbar(artist, ax=axes, label='Height (mm)', shrink=.8)
    fig.savefig(OUT/'subbin_edge_proposal.png',dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
