"""Blind comparison of the Attempt 209 prediction with the Hyunwoo Run 1 scan (card REPLICATE-HW1, F4).

Usage: python shape_design/compare_hw1.py [--solve run/shape_design/hw_run1/solve_mesh2]
Writes <solve>/comparison/{comparison.json, profiles.npz}. Rigid alignment only; nothing is fitted to the scan.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.transform import Rotation
import vtk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.geometry import SurfaceDistance, rigid_fit, triangle_mesh

DEBRIS_MM = 5.0          # |h| beyond this from the levelled plane is debris
SAMPLE_MM = 2.0          # scan subsample spacing for alignment
PROFILE_BIN_MM = 5.0     # card F4
PROFILE_SUB_MM = 1.0
PROFILE_U_MAX = 105.0
PROFILE_HALF_V = 40.0
REGIONS_U = {'Region 1': (-100.0, -50.0), 'Region 3': (-40.0, 40.0), 'Region 2': (40.0, 100.0)}
ORIENTATIONS = {'identity': (1, 1), 'rot180_normal': (-1, -1), 'turn_over_u_axis': (1, -1),
                'turn_over_v_axis': (-1, 1)}


def read_ply(path):
    """Binary little-endian PLY with float xyz vertices and uchar/int triangle lists."""
    data = Path(path).read_bytes()
    end = data.index(b'end_header') + len(b'end_header')
    end += 2 if data[end:end + 2] == b'\r\n' else 1
    header = data[:end].decode('ascii').split()
    if 'binary_little_endian' not in header:
        raise ValueError('Expected binary_little_endian PLY')
    nv = int(header[header.index('vertex') + 1])
    nf = int(header[header.index('face') + 1])
    if header.count('property') != 4 or 'uchar' not in header:
        raise ValueError('Unexpected PLY properties')
    v = np.frombuffer(data, '<f4', 3 * nv, end).reshape(nv, 3).astype(float)
    rec = np.frombuffer(data, np.dtype([('n', 'u1'), ('i', '<i4', 3)]), nf, end + 12 * nv)
    if end + 12 * nv + 13 * nf != len(data) or (rec['n'] != 3).any():
        raise ValueError('PLY is not a pure triangle mesh of the declared size')
    return v, rec['i'].astype(int)


def level(points):
    """Right-handed PCA frame: u = longest in-plane axis, v = second, h = normal. Returns frame, debris mask."""
    def frame(p):
        c = p.mean(axis=0)
        w, vec = np.linalg.eigh(np.cov((p - c).T))
        basis = vec[:, ::-1].copy()
        if np.linalg.det(basis) < 0:
            basis[:, 2] *= -1
        return c, basis
    c, b = frame(points)
    keep = np.abs((points - c) @ b[:, 2]) <= DEBRIS_MM
    c, b = frame(points[keep])
    return c, b, keep


def subsample(p, spacing=SAMPLE_MM):
    """One actual point per spacing x spacing (u, v) cell: the one nearest the cell centre."""
    cell = np.floor(p[:, :2] / spacing).astype(int)
    off = np.linalg.norm(p[:, :2] - (cell + 0.5) * spacing, axis=1)
    order = np.lexsort((off, cell[:, 1], cell[:, 0]))
    _, first = np.unique(cell[order], axis=0, return_index=True)
    return p[order[first]]


class Surface(SurfaceDistance):
    """Closest points plus the unit normal of the closest triangle."""
    def __init__(self, vertices, faces):
        super().__init__(triangle_mesh(vertices, faces))
        e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        n = np.cross(e1, e2)
        self.normals = n / np.linalg.norm(n, axis=1, keepdims=True)

    def closest_with_normals(self, p):
        cp = np.empty_like(p, dtype=float); d = np.empty(len(p)); ids = np.empty(len(p), int)
        cell, sub, d2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
        q = [0.0, 0.0, 0.0]
        for i, x in enumerate(p):
            self.locator.FindClosestPoint(x, q, cell, sub, d2)
            cp[i] = q; d[i] = np.sqrt(max(0.0, float(d2))); ids[i] = int(cell)
        return d, cp, self.normals[ids]


def _pairs(sample, scan, pred_v, pred, r, t):
    """Forward: scan samples (current pose) -> prediction surface (points, closest points, normals).
    Backward: prediction vertices -> scan surface; returns the scan closest points in the current pose."""
    x = sample @ r + t
    d_f, cp, n_f = pred.closest_with_normals(x)
    y_local = (pred_v - t) @ r.T                          # prediction vertices in the scan's local frame
    d_b, cq, _ = scan.closest_with_normals(y_local)
    return (x, cp, n_f), (cq @ r + t, pred_v), d_f, d_b


def _system(fwd, back):
    """Linearized step [omega, dt]: forward point-to-plane rows; backward point-to-point rows, so the
    sheet edges constrain the in-plane placement."""
    x, cp, n = fwd
    p, q = back
    rows_f = np.column_stack((np.cross(x, n), n))
    rhs_f = np.einsum('ij,ij->i', cp - x, n)
    skew = np.zeros((len(p), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -p[:, 2], p[:, 1], -p[:, 0]
    skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = p[:, 2], -p[:, 1], p[:, 0]
    rows_b = np.concatenate((-skew, np.broadcast_to(np.eye(3), skew.shape)), axis=2).reshape(-1, 6)
    rhs_b = (q - p).ravel()
    # weight so each direction contributes like its share of the symmetric mean
    wf, wb = 1 / np.sqrt(len(x)), 1 / np.sqrt(len(p))
    return np.vstack((wf * rows_f, wb * rows_b)), np.concatenate((wf * rhs_f, wb * rhs_b))


def objective(d_f, d_b):
    return float(np.sqrt(0.5 * (np.mean(d_f ** 2) + np.mean(d_b ** 2))))


def icp(sample, scan, pred_v, pred, init, kabsch_iters=30, iters=100, tol=1e-9):
    """Symmetric rigid ICP of the scan onto the prediction; (R, t) row vectors: aligned = local @ R + t.
    Stage 1: forward point-to-point Kabsch (robust). Stage 2: symmetric point-to-plane with monotone
    step acceptance (in-plane placement is then constrained by the sheet edges through the backward term)."""
    r, t = init
    for _ in range(kabsch_iters):
        x = sample @ r + t
        _, cp = pred.closest(x)
        dr, dt = rigid_fit(x, cp)
        r, t = r @ dr, t @ dr + dt
    fwd, back, d_f, d_b = _pairs(sample, scan, pred_v, pred, r, t)
    best = objective(d_f, d_b)
    status = {}
    for _ in range(iters):
        sol = np.linalg.lstsq(*_system(fwd, back), rcond=None)[0]
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125):
            dr = Rotation.from_rotvec(step * sol[:3]).as_matrix().T   # row-vector convention: x @ dr
            r2, t2 = r @ dr, t @ dr + step * sol[3:]
            fwd2, back2, f2, b2 = _pairs(sample, scan, pred_v, pred, r2, t2)
            value = objective(f2, b2)
            if value <= best:
                r, t, fwd, back, d_f, d_b, accepted = r2, t2, fwd2, back2, f2, b2, True
                improvement, best = best - value, value
                break
        status = {'iterations': _ + 1, 'last_step_norm': float(np.linalg.norm(sol)), 'accepted_last': accepted}
        if not accepted or (improvement < tol and np.linalg.norm(sol) < 1e-6):
            break
    return r, t, d_f, d_b, status


def profile(p, half_v=PROFILE_HALF_V, bin_mm=PROFILE_BIN_MM, u_max=PROFILE_U_MAX, sub_mm=PROFILE_SUB_MM):
    """Card F4 width profile: mean height over |v| <= half_v in bin_mm bins along u (material frame), mean
    removed. Height is linearly interpolated on a fixed sub_mm grid so scan and prediction are sampled
    identically within each bin."""
    edges = np.arange(-u_max, u_max + 1e-9, bin_mm)
    offsets = np.arange(sub_mm / 2, bin_mm, sub_mm)
    u_sub = (edges[:-1, None] + offsets[None, :]).ravel()
    v = np.arange(-half_v, half_v + 1e-9, sub_mm)
    U, V = np.meshgrid(u_sub, v, indexing='ij')
    m = (np.abs(p[:, 0]) <= u_max + 5) & (np.abs(p[:, 1]) <= half_v + 5)
    h = LinearNDInterpolator(p[m, :2], p[m, 2])(U, V)
    if not np.isfinite(h).all():
        raise ValueError('Profile grid not covered by the surface')
    prof = h.mean(axis=1).reshape(len(edges) - 1, len(offsets)).mean(axis=1)
    return (edges[:-1] + edges[1:]) / 2, prof - prof.mean()


def region_curvature(u, prof):
    """Second-derivative coefficient 2a of a quadratic fit of the profile in each region (1/mm)."""
    out = {}
    for name, (lo, hi) in REGIONS_U.items():
        m = (u >= lo) & (u <= hi)
        out[name] = float(2 * np.polyfit(u[m], prof[m], 2)[0])
    return out


def compare(pred_v, pred_f, scan_v, scan_f):
    """Align the levelled scan to the prediction for each proper rectangle orientation; report metrics."""
    c, b, keep = level(scan_v)
    scan_local = (scan_v - c) @ b                    # right-handed (u, v, h)
    sample = subsample(scan_local[keep])
    pred = Surface(pred_v, pred_f)
    clean_f = scan_f[keep[scan_f].all(axis=1)]          # debris triangles removed from the backward surface
    scan = Surface(scan_local, clean_f)
    results = {}
    for name, (su, sv) in ORIENTATIONS.items():
        r0 = np.diag([su, sv, su * sv]).astype(float)
        r, t, d_fwd, d_back, status = icp(sample, scan, pred_v, pred, (r0, np.zeros(3)))
        results[name] = {'R': r.tolist(), 't': t.tolist(), 'forward_rms_mm': float(np.sqrt(np.mean(d_fwd ** 2))),
                         'backward_rms_mm': float(np.sqrt(np.mean(d_back ** 2))),
                         'symmetric_rms_mm': objective(d_fwd, d_back),
                         'forward_p95_mm': float(np.percentile(d_fwd, 95)),
                         'backward_p95_mm': float(np.percentile(d_back, 95)), 'icp': status}
    best = min(results, key=lambda k: results[k]['symmetric_rms_mm'])
    r, t = np.array(results[best]['R']), np.array(results[best]['t'])
    scan_in_pred = scan_local[keep] @ r + t
    u, p_pred = profile(pred_v)
    _, p_scan = profile(scan_in_pred)
    ptv = lambda x: float(np.max(x) - np.min(x))
    curv_pred, curv_scan = region_curvature(u, p_pred), region_curvature(u, p_scan)
    summary = {'best_orientation': best, 'orientations': results,
               'profile_ptv_pred_mm': ptv(p_pred), 'profile_ptv_scan_mm': ptv(p_scan),
               'profile_ptv_ratio_pred_over_scan': ptv(p_pred) / ptv(p_scan),
               'profile_rms_diff_mm': float(np.sqrt(np.mean((p_pred - p_scan) ** 2))),
               'region_curvature_pred_per_mm': curv_pred, 'region_curvature_scan_per_mm': curv_scan,
               'region_sign_match': {k: bool(np.sign(curv_pred[k]) == np.sign(curv_scan[k])) for k in REGIONS_U},
               'scan_points': int(len(scan_v)), 'scan_debris_removed': int((~keep).sum()),
               'scan_triangles_removed': int(len(scan_f) - len(clean_f)),
               'scan_alignment_samples': int(len(sample))}
    return summary, {'u_mm': u, 'profile_pred_mm': p_pred, 'profile_scan_mm': p_scan,
                     'scan_in_pred_frame_mm': scan_in_pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solve', type=Path, default=ROOT / 'run/shape_design/hw_run1/solve_mesh2')
    ap.add_argument('--scan', type=Path, default=ROOT / 'shape_design/data/Hyunwoo_Run_1_24_Passes.ply')
    a = ap.parse_args()
    rec = json.loads((a.solve / 'result.json').read_text())
    if rec['status'] != 'accepted':
        raise SystemExit('Solve not accepted')
    out = a.solve / 'comparison'
    out.mkdir(exist_ok=False)
    shape = np.load(a.solve / 'shape.npz')
    scan_v, scan_f = read_ply(a.scan)
    summary, arrays = compare(shape['aligned_mm'], shape['faces'], scan_v, scan_f)
    summary['inputs'] = {'shape_npz_sha256': digest(a.solve / 'shape.npz'), 'scan_sha256': digest(a.scan),
                         'code_sha256': digest(Path(__file__))}
    save_json(out / 'comparison.json', summary)
    np.savez_compressed(out / 'profiles.npz', **arrays)
    print(json.dumps({k: v for k, v in summary.items() if k != 'orientations'}, indent=1))
    for k, v in summary['orientations'].items():
        print(k, round(v['symmetric_rms_mm'], 4))


if __name__ == '__main__':
    main()
