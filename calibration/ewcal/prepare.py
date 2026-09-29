"""Prepare a fixed, spatially balanced scan cloud without reconstructing a surface."""
from pathlib import Path
import numpy as np
from .common import resolve, digest, save_json
from .geometry import read_mesh, points

def prepare_case(config, case):
    source, dest = resolve(case['scan']), resolve(case['target'])
    settings = config['preprocess']
    p = np.unique(points(read_mesh(source, case.get('scan_scale_to_mm', 1.0))), axis=0)
    center = np.median(p, axis=0)
    # Iterated trimmed PCA reduces isolated scanner fragments; no shape smoothing.
    work = p
    for _ in range(3):
        center = np.median(work, axis=0)
        _, _, vt = np.linalg.svd(work-center, full_matrices=False)
        q = (p-center) @ vt.T
        lo, hi = np.quantile(q, [.005, .995], axis=0)
        work = p[np.all((q >= lo) & (q <= hi), axis=1)]
    y, z = vt[0].copy(), vt[2].copy()
    if y[1] < 0: y *= -1
    if z[2] < 0: z *= -1
    x = np.cross(y, z); x /= np.linalg.norm(x); z = np.cross(x, y)
    basis = np.column_stack([x, y, z])
    orient = case.get('scan_orientation', {})
    if 'basis_columns' in orient:
        basis = np.asarray(orient['basis_columns'], float)
    if basis.shape != (3, 3) or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or np.linalg.det(basis) < .999:
        raise ValueError('Scan basis must be an orthonormal right-handed 3x3 matrix')
    # flip_z also flips x to preserve handedness; flip_xy is a 180 degree turn.
    if orient.get('flip_z', False): basis = basis @ np.diag([-1, 1, -1])
    if orient.get('flip_xy', False): basis = basis @ np.diag([-1, -1, 1])
    q = (p-center) @ basis
    low, high = np.quantile(q, [.005, .995], axis=0)
    shift = (low+high)/2; shift[2] = np.median(q[:, 2])
    q -= shift
    dimensions = np.array(config['sheet']['size_mm'])
    observed = high[:2]-low[:2]
    if np.any(observed < .85*dimensions) or np.any(observed > 1.2*dimensions):
        raise ValueError(f'Scan size/orientation check failed: robust extents {observed}, expected {dimensions}')
    half = dimensions/2 - settings['edge_inset_mm']
    keep = np.all(np.abs(q[:, :2]) <= half, axis=1)
    # Gross z outlier gate only: mild deformations may have very small MAD.
    zmad = np.median(np.abs(q[:, 2]-np.median(q[:, 2])))
    keep &= np.abs(q[:, 2]-np.median(q[:, 2])) <= max(settings['z_outlier_floor_mm'], 12*zmad)
    q = q[keep]
    spacing = settings['spacing_mm']
    keys = np.floor((q[:, :2]+dimensions/2)/spacing).astype(int)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(labels); splits = np.flatnonzero(np.diff(labels[order]))+1
    cloud, spread, counts = [], [], []
    for group in np.split(order, splits):
        cell = q[group]
        median = np.median(cell, axis=0)
        mad = np.median(np.abs(cell[:, 2]-median[2]))
        if mad > settings['max_cell_z_mad_mm']:
            continue
        cloud.append(median); spread.append(mad); counts.append(len(group))
    cloud = np.asarray(cloud)
    if len(cloud) < settings['min_points']:
        raise ValueError(f'Only {len(cloud)} accepted scan cells')
    # One sample per projected material-plane bin: overlapping scan patches cannot
    # gain weight solely because they contain more triangles.
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, points_mm=cloud, z_mad_mm=spread, raw_count=counts)
    expected_cells = np.prod(2*half/spacing)
    meta = {'source': str(source), 'source_sha256': digest(source), 'target_sha256': digest(dest),
            'units': 'mm', 'unique_vertices': int(len(p)), 'accepted_points': len(cloud),
            'robust_scan_extents_mm': observed.tolist(),
            'coverage_fraction_approx': float(len(cloud)/expected_cells),
            'transform': {'origin_scanner_mm': center.tolist(), 'basis_columns': basis.tolist(),
                          'shift_local_mm': shift.tolist(),
                          'formula': '(scanner_mm-origin) @ basis_columns - shift_local_mm'},
            'orientation': orient, 'preprocess': settings, 'sheet_size_mm': dimensions.tolist(),
            'scan_scale_to_mm': case.get('scan_scale_to_mm', 1.0),
            'note': 'Top scan approximated as midsurface; orientation must be checked against photo/toolpath.'}
    save_json(dest.with_suffix('.json'), meta)
    preview(dest.with_suffix('.png'), cloud, np.asarray(spread), case['id'])
    return meta

def preview(path, cloud, spread, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    for ax, color, label in zip(axes, [cloud[:, 2], spread], ['Height (mm)', 'Within-cell z MAD (mm)']):
        artist = ax.scatter(cloud[:, 0], cloud[:, 1], c=color, s=3, cmap='viridis')
        ax.set(xlabel='Material x (mm)', ylabel='Material y (mm)', title=label, aspect='equal')
        fig.colorbar(artist, ax=ax)
    fig.suptitle(f'{title}: prepared scan — check orientation before fitting')
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160)
    Path(path).write_bytes(buffer.getvalue()); plt.close(fig)
