#!/usr/bin/env python3
"""Compare screening (4 mm, 5e-14) and verification (2 mm, 1e-14) states of the same recipe.

Reports the primary gate (central 160 mm ROI) at curvature radii 10/15/20 mm and floors .05/.1/.2 1/m, the
whole-sheet opposite-sign area, and the rigid-aligned (Kabsch) RMS shape difference of the two meshes sampled at
common material points. Diagnostics only: the frozen acceptance gate is the 15 mm / 0.1 1/m central-ROI result.
Usage: analyze_square_verification.py SCREEN_STATE.json VERIFY_STATE.json OUT.json [SCREEN_HISTORY.csv VERIFY_HISTORY.csv]
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import square_attempt as sa            # noqa: E402
import dome_continuous_objective as dco  # noqa: E402


_curv = sa.geometric_curvature
_cache = {}


def _cached_curvature(positions, uv, faces, radius=0.015, **kw):
    """Curvature depends on the radius, not on the floor: fit once per (mesh, radius) and reuse across floors."""
    key = (hash(np.asarray(positions).tobytes()), radius)
    if key not in _cache:
        _cache[key] = _curv(positions, uv, faces, radius=radius, **kw)
    return _cache[key]


sa.geometric_curvature = _cached_curvature


def sample(state, grid):
    uv = np.asarray(state['material_uv'], float)
    pos = np.asarray(state['positions'], float)
    return LinearNDInterpolator(uv, pos)(grid)


def kabsch_rms(a, b):
    """RMS of b after the best rigid (rotation+translation) alignment onto a; also height range of a."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b - cb).T @ (a - ca))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    d = np.sqrt(np.sum((((b - cb) @ r) + ca - a) ** 2, axis=1))
    return float(np.sqrt(np.mean(d ** 2))), {f'p{q}': float(np.percentile(d, q)) for q in (50, 90, 99, 100)}


def material_metric_deviation(state, key):
    """Per-face material-coordinate metric C - I, C = Dm^-T abar Dm^-1, with the simulator's edge basis
    Dm = [u2 - u1, u0 - u2] (GrowthHelper.hpp). Mesh independent for identical treatment. Returns centroids (n,2) and
    (n,3) components (C-I)_00, (C-I)_01, (C-I)_11."""
    uv = np.asarray(state['material_uv'], float)
    faces = np.asarray(state['faces'], int)
    abar = np.asarray(state[key], float)
    u0, u1, u2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    dm = np.stack([u2 - u1, u0 - u2], axis=2)                       # (n,2,2), columns are the edge vectors
    dinv = np.linalg.inv(dm)
    c = np.einsum('nji,njk,nkl->nil', dinv, abar, dinv)             # Dm^-T abar Dm^-1
    c = c - np.eye(2)
    return uv[faces].mean(1), np.stack([c[:, 0, 0], 0.5 * (c[:, 0, 1] + c[:, 1, 0]), c[:, 1, 1]], axis=1)


def total_hits(state_path, n_faces):
    """Per-face total hit count (before + this cycle) of the final mapping VTP next to the state file."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    files = sorted(Path(state_path).parent.glob('*_mapping.vtp'))
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(files[-1]))
    reader.Update()
    cells = reader.GetOutput().GetCellData()
    total = vtk_to_numpy(cells.GetArray('total_hits_before_cycle')) + vtk_to_numpy(cells.GetArray('hits_this_cycle'))
    if len(total) != n_faces:
        raise ValueError('mapping/state face count mismatch')
    return total.astype(float)


def treatment_comparability(states, paths):
    """Same-recipe treatment on the two meshes at common ROI material points (nearest face centroid): normalized
    metric deviation C-I per stroke family field and total hit count. Differences at stroke edges are expected from the
    4 mm vs 2 mm discretisation; report means, tails and relative RMS."""
    half = 0.08 * 0.999
    g = np.linspace(-half, half, 81)
    grid = np.array([[x, y] for x in g for y in g])
    out = {}
    for key in ('abar_top', 'abar_bottom'):
        fields = []
        for st in states:
            c, t = material_metric_deviation(st, key)
            fields.append(t[cKDTree(c).query(grid)[1]])
        a, b = fields
        out[key] = {'mean_screen': a.mean(0).tolist(), 'mean_verify': b.mean(0).tolist(),
                    'max_abs_screen': np.abs(a).max(0).tolist(), 'max_abs_verify': np.abs(b).max(0).tolist(),
                    'rms_diff_over_rms_screen': (np.sqrt(np.mean((a - b) ** 2, axis=0)) /
                                                 np.maximum(np.sqrt(np.mean(a ** 2, axis=0)), 1e-30)).tolist()}
    hits = []
    for st, pth in zip(states, paths):
        c = np.asarray(st['material_uv'], float)[np.asarray(st['faces'], int)].mean(1)
        hits.append(total_hits(pth, len(c))[cKDTree(c).query(grid)[1]])
    out['total_hits'] = {'screen_histogram': np.bincount(hits[0].astype(int)).tolist(),
                         'verify_histogram': np.bincount(hits[1].astype(int)).tolist(),
                         'fraction_points_equal': float(np.mean(hits[0] == hits[1])),
                         'max_abs_diff': float(np.abs(hits[0] - hits[1]).max())}
    return out


def main(screen_path, verify_path, out_path):
    states = {k: json.loads(Path(p).read_text()) for k, p in (('screen', screen_path), ('verify', verify_path))}
    result = {}
    for name, st in states.items():
        table = {}
        for radius in (0.010, 0.015, 0.020):
            for floor in (0.05, 0.1, 0.2):
                dco.FLOOR = floor
                m = sa.analyze_state(st, radius=radius)
                table[f'r{radius * 1e3:.0f}mm_floor{floor}'] = {k: m[k] for k in (
                    'largest_connected_fraction', 'qualified', 'dome_strength', 'whole_opposite_sign_fraction')}
        dco.FLOOR = 0.1
        result[name] = table
    uv = np.asarray(states['verify']['material_uv'], float)
    half = 0.08 * 0.999
    g = np.linspace(-half, half, 41)
    grid = np.array([[x, y] for x in g for y in g])
    a, b = sample(states['screen'], grid), sample(states['verify'], grid)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if not ok.all():
        raise ValueError('non-finite shape samples in the ROI grid')
    rms, tails = kabsch_rms(a[ok], b[ok])
    result['shape'] = {'n_points': int(ok.sum()), 'aligned_rms_m': rms, 'aligned_distance_percentiles_m': tails,
                       'screen_z_range_m': float(np.ptp(a[ok][:, 2])), 'verify_z_range_m': float(np.ptp(b[ok][:, 2])),
                       'uv_extent_m': [float(uv.min()), float(uv.max())]}
    if len(sys.argv) > 5:   # optional: screening/verification sequence_history.csv -> per-face hit-event comparison
        import csv
        hist = {}
        for k, pth in (('screen', sys.argv[4]), ('verify', sys.argv[5])):
            rows = list(csv.DictReader(open(pth)))
            nf = len(states[k]['faces'])
            hist[k] = {'hit_events_per_face': [float(r['hit_events']) / nf for r in rows],
                       'max_total_hits': [int(r['max_total_hits']) for r in rows]}
        result['hit_history'] = hist
    result['treatment'] = treatment_comparability((states['screen'], states['verify']), (screen_path, verify_path))
    Path(out_path).write_text(json.dumps(result, indent=2) + '\n')
    base = result['verify']['r15mm_floor0.1']
    print(json.dumps({'gate_15mm_0.1': base, 'shape_rms_m': rms, 'treatment': result['treatment']}))


if __name__ == '__main__':
    main(*sys.argv[1:4])
