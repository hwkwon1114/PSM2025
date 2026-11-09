#!/usr/bin/env python3
"""Pilot for card EDGE-ANCHOR-1: scan placement from physical sheet edges vs the current scan anchor.

Solver-free and read-only on all inputs. Writes only to run/diagnostics/scan_edge_anchor_pilot/.

Current production scoring (calibration/ewcal/material_frame.py) places the simulation by a Kabsch fit to
its own flat material reference and the scan by its point centroid; the scan's in-plane origin/yaw were
fixed during preprocessing by 0.5/99.5 % raw-extent quantile midpoints and a long-edge yaw.  This pilot
fits all four raw sheet edges, compares the edge-derived origin/yaw with the current one, and rescores
saved Run 1 simulations under
  A  scan edge-anchored, simulation unchanged (Kabsch material frame),
  B  scan and simulation both anchored by the same projected-edge functional.
The scored domain (all 14,504 scan points, fail-closed footprint) is identical in every variant.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.common import digest
from calibration.ewcal.geometry import (SurfaceDistance, compute_shared_footprint_metrics, points, read_mesh,
                                        rigid_fit, sample_mesh_on_shared_footprint, set_points)
from calibration.ewcal.material_frame import direction_frame, score_material_frame

OUT = ROOT / 'run/diagnostics/scan_edge_anchor_pilot'
RAW = ROOT / 'calibration/data/raw/run6.stl'
TARGET = ROOT / 'run/putong_convention_check/preserved_samples/run6.npz'
TARGET_SHA = '00d64e18d28eab6c7254d674af8424d169cac3672abaa2bcf072cab59673a348'
RAW_SHA = 'a67d0a8d13e89cdf9c7c47a8190d412e3524c72222f2aebcde95014bef332df9'
HALF = np.array([100.0, 150.0])     # sheet half-sizes, internal (u, v) mm
SIDE_FRACTION = 0.8                 # central fraction of each side used for line fits (corners excluded)


# ---------------------------------------------------------------- edge geometry
def yaw_matrix(angle_deg):
    a = np.deg2rad(angle_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def fit_frame(sides):
    """sides: dict left/right -> (y, x) samples, bottom/top -> (x, y) samples, in the current frame.
    Lines x = a + b y (long sides), y = c + d x (short sides), Theil-Sen.
    Returns centre = intersection of the two midlines, yaw (deg, CCW sheet rotation) and line coefficients."""
    fit = {k: stats.theilslopes(v[1], v[0])[:2] for k, v in sides.items()}   # (slope, intercept)
    bL, aL = fit['left']; bR, aR = fit['right']; dB, cB = fit['bottom']; dT, cT = fit['top']
    a, b = (aL + aR) / 2, (bL + bR) / 2      # vertical midline  x = a + b y
    c, d = (cB + cT) / 2, (dB + dT) / 2      # horizontal midline y = c + d x
    y0 = (c + d * a) / (1 - b * d); x0 = a + b * y0
    yaw_long = float(np.degrees(np.arctan(-b))); yaw_short = float(np.degrees(np.arctan(d)))
    angles = {'left': float(np.degrees(np.arctan(-bL))), 'right': float(np.degrees(np.arctan(-bR))),
              'bottom': float(np.degrees(np.arctan(dB))), 'top': float(np.degrees(np.arctan(dT)))}
    width = float((aR - aL) * np.cos(np.arctan(b))); length = float((cT - cB) * np.cos(np.arctan(d)))
    return {'centre_mm': [float(x0), float(y0)], 'yaw_deg': (yaw_long + yaw_short) / 2,
            'yaw_long_deg': yaw_long, 'yaw_short_deg': yaw_short, 'side_angles_deg': angles,
            'width_mm': width, 'length_mm': length,
            'lines': {k: {'slope': float(v[0]), 'intercept': float(v[1])} for k, v in fit.items()}}


def scan_sides(xy, cell=0.5, rank=5):
    """Boundary samples of the largest dense projected component; per 1 mm strip, the rank-th extreme point."""
    keys = np.floor(xy / cell).astype(int); origin = keys.min(0); keys -= origin
    counts = np.zeros(tuple(keys.max(0) + 1), int); np.add.at(counts, (keys[:, 0], keys[:, 1]), 1)
    labels, n = ndimage.label(counts >= 2, structure=np.ones((3, 3)))
    if not n: raise ValueError('No dense scan component')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    comp = ndimage.binary_fill_holes(labels == int(np.argmax(sizes)))
    p = xy[comp[keys[:, 0], keys[:, 1]]]
    out = {}
    for name, along, across, lim, sign in (('left', 1, 0, HALF[1], -1), ('right', 1, 0, HALF[1], 1),
                                           ('bottom', 0, 1, HALF[0], -1), ('top', 0, 1, HALF[0], 1)):
        s, v = [], []
        for t in np.arange(-SIDE_FRACTION * lim, SIDE_FRACTION * lim, 1.0):
            m = (p[:, along] >= t) & (p[:, along] < t + 1)
            if m.sum() < 50: continue
            vals = np.sort(sign * p[m, across])[::-1]
            s.append(t + .5); v.append(sign * vals[min(rank, len(vals) - 1)])
        out[name] = (np.array(s), np.array(v))
    return out, p


def sim_sides(q, ref):
    """Exact material-boundary vertices of the simulation, in its current projected frame."""
    tol = 1e-6; out = {}
    for name, ax, val in (('left', 0, -HALF[0]), ('right', 0, HALF[0]), ('bottom', 1, -HALF[1]), ('top', 1, HALF[1])):
        other = 1 - ax
        m = (np.abs(ref[:, ax] - val) < tol) & (np.abs(ref[:, other]) <= SIDE_FRACTION * HALF[other])
        if m.sum() < 10: raise ValueError(f'Too few boundary vertices on {name}')
        out[name] = (q[m, other], q[m, ax])
    return out


# ---------------------------------------------------------------- scoring variants
def score_variant(mesh, target, ref, faces, scan_anchor=None, sim_edge=False):
    """Copy of score_material_frame with an optional scan anchor (centre, yaw) and optional simulation
    edge anchoring.  With scan_anchor=None and sim_edge=False it must equal production exactly."""
    q, _ = direction_frame(points(mesh), ref, faces)
    sim_frame = None
    if sim_edge:
        sim_frame = fit_frame(sim_sides(q, ref))
        q = (q - np.array([*sim_frame['centre_mm'], 0.])) @ yaw_matrix(sim_frame['yaw_deg'])
    target = np.asarray(target, float)
    if scan_anchor is None:
        t0 = target - target.mean(0)
    else:
        c = np.array([*scan_anchor['centre_mm'], target[:, 2].mean()])
        t0 = (target - c) @ yaw_matrix(scan_anchor['yaw_deg'])
    ref_t = np.column_stack([t0[:, 0], t0[:, 1], np.zeros(len(t0))])
    r_scan, t_scan = rigid_fit(t0, ref_t)
    aligned = t0 @ r_scan + t_scan
    flat = np.column_stack([aligned[:, 0], aligned[:, 1], np.zeros(len(aligned))])
    for _ in range(6):
        s0 = sample_mesh_on_shared_footprint(q, ref, faces, aligned[:, :2])
        r, t = rigid_fit(s0, flat); q = q @ r + t
        if np.linalg.norm(r - np.eye(3)) < 1e-13 and np.linalg.norm(t) < 1e-13: break
    framed = set_points(mesh, q)
    d, _ = SurfaceDistance(framed).closest(aligned)
    m = compute_shared_footprint_metrics(framed, aligned, ref, faces, d)
    return m, aligned, d, sim_frame


# ---------------------------------------------------------------- known-answer tests
def rect_cloud(rng, shift, yaw, n=2_400_000, hole=True):
    xy = rng.uniform(-HALF, HALF, (n, 2))   # ~40 points/mm^2, similar to the raw scan
    if hole:   # missing interior patch: biases the centroid, not the edges
        xy = xy[~((xy[:, 0] > 20) & (xy[:, 0] < 80) & (xy[:, 1] > 30) & (xy[:, 1] < 140))]
    p = np.column_stack([xy, np.zeros(len(xy))]) @ yaw_matrix(yaw).T + np.array([*shift, 0])
    return p


def self_test():
    rng = np.random.default_rng(20261101)
    for shift, yaw in (((0., 0.), 0.), ((1.3, -0.7), 1.0), ((-2.0, 1.5), -0.6)):
        p = rect_cloud(rng, shift, yaw)
        f = fit_frame(scan_sides(p[:, :2])[0])
        assert np.allclose(f['centre_mm'], shift, atol=0.05), (f['centre_mm'], shift)
        assert abs(f['yaw_deg'] - yaw) < 0.01, (f['yaw_deg'], yaw)
        assert abs(f['width_mm'] - 200) < 0.5 and abs(f['length_mm'] - 300) < 0.5, f
    cen = p[:, :2].mean(0)
    assert np.linalg.norm(cen - np.array(shift)) > 1.0      # centroid is biased by the hole, edges are not
    print('PASS self-test: edge frame recovers shift/yaw within 0.05 mm/0.01 deg with a missing patch; '
          f'centroid bias {np.linalg.norm(cen - np.array(shift)):.2f} mm')


def reproduce_target(raw_q, meta):
    """Re-run the recorded crop/bin step in the preserved frame; must reproduce the frozen target."""
    s = meta['preprocess']; dims = np.array(meta['sheet_size_mm'], float); half = dims / 2 - s['edge_inset_mm']
    q = raw_q[np.all(np.abs(raw_q[:, :2]) <= half, axis=1)]
    zmad = np.median(np.abs(q[:, 2] - np.median(q[:, 2])))
    q = q[np.abs(q[:, 2] - np.median(q[:, 2])) <= max(s['z_outlier_floor_mm'], 12 * zmad)]
    keys = np.floor((q[:, :2] + dims / 2) / s['spacing_mm']).astype(int)
    _, lab = np.unique(keys, axis=0, return_inverse=True); lab = lab.ravel()
    order = np.argsort(lab, kind='stable'); splits = np.flatnonzero(np.diff(lab[order])) + 1
    cells = [np.median(q[g], axis=0) for g in np.split(order, splits)
             if np.median(np.abs(q[g, 2] - np.median(q[g, 2]))) <= s['max_cell_z_mad_mm']]
    return np.asarray(cells)


# ---------------------------------------------------------------- main
def case_dirs():
    rows = [('Attempt 209 (corrected)', ROOT / 'run/convention_corrected/run6/evaluations/attempt_0001/case_00')]
    for line in (ROOT / 'run/run1_clean_24lhs/evaluations.jsonl').read_text().splitlines():
        r = json.loads(line)
        if not r['valid']: continue
        d = sorted(glob.glob(str(ROOT / f"run/run1_clean_24lhs/candidates/a{r['attempt']:04d}/evaluations/*/case_00")))
        if len(d) == 1: rows.append((f"clean a{r['attempt']:04d}", Path(d[0]), r['objective_mm']))
    for a in (14, 16):   # valid wave-4 results already in candidates/, not yet in evaluations.jsonl
        for d in glob.glob(str(ROOT / f'run/run1_clean_24lhs/candidates/a{a:04d}/evaluations/*/case_00')):
            if (Path(d) / 'comparison.npz').exists() and not any(str(x[1]) == d for x in rows):
                rows.append((f'clean a{a:04d}', Path(d)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--self-test-only', action='store_true'); args = ap.parse_args()
    self_test()
    if args.self_test_only: return
    assert digest(TARGET) == TARGET_SHA and digest(RAW) == RAW_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(TARGET.with_suffix('.json').read_text()); tr = meta['transform']
    raw = np.unique(points(read_mesh(RAW, 1.0)), axis=0)
    q = (raw - np.array(tr['origin_scanner_mm'])) @ np.array(tr['basis_columns']) - np.array(tr['shift_local_mm'])
    target = np.load(TARGET)['points_mm']

    rep = reproduce_target(q, meta)
    key = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))]
    assert rep.shape == target.shape, (rep.shape, target.shape)
    rep_err = float(np.abs(key(rep) - key(target)).max())
    assert rep_err < 1e-6, rep_err
    print(f'PASS reproduce frozen target from raw STL: {len(rep)} cells, max |diff| {rep_err:.2e} mm')

    zmed = np.median(q[:, 2]); keep = np.abs(q[:, 2] - zmed) <= 20
    sides, comp = scan_sides(q[keep, :2])
    frame = fit_frame(sides)
    current = {'centre_mm': target[:, :2].mean(0).tolist(), 'yaw_deg': 0.0}
    print('scan edge frame:', json.dumps({k: frame[k] for k in ('centre_mm', 'yaw_deg', 'yaw_long_deg',
          'yaw_short_deg', 'side_angles_deg', 'width_mm', 'length_mm')}))

    results = []
    for row in case_dirs():
        label, d = row[0], row[1]
        st = json.loads((d / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
        ref = np.asarray(st['reference_positions'], float) * 1000.0; faces = np.asarray(st['faces'], int)
        mesh = read_mesh(d / 'prediction_mm.vtp', 1.0)
        prod, *_ = score_material_frame(mesh, target, ref, faces)
        m0, al0, d0, _ = score_variant(mesh, target, ref, faces)
        assert abs(m0['rms_symmetric_mm'] - prod['rms_symmetric_mm']) < 1e-12
        if len(row) > 2: assert abs(prod['rms_symmetric_mm'] - row[2]) < 1e-6, (label, prod['rms_symmetric_mm'], row[2])
        mA, alA, dA, _ = score_variant(mesh, target, ref, faces, scan_anchor=frame)
        mB, alB, dB, sf = score_variant(mesh, target, ref, faces, scan_anchor=frame, sim_edge=True)
        rec = {'label': label, 'dir': str(d.relative_to(ROOT)), 'current': m0, 'A': mA, 'B': mB,
               'sim_edge_frame': {k: sf[k] for k in ('centre_mm', 'yaw_deg', 'width_mm', 'length_mm')}}
        results.append(rec)
        np.savez_compressed(OUT / f"pointwise_{label.split()[-1].strip('()')}.npz",
                            xy_current=al0[:, :2], d_current=d0, xy_A=alA[:, :2], d_A=dA, xy_B=alB[:, :2], d_B=dB)
        print(f"{label:26s} current {m0['rms_symmetric_mm']:.4f}  A {mA['rms_symmetric_mm']:.4f}  "
              f"B {mB['rms_symmetric_mm']:.4f}  sim-edge centre {np.round(sf['centre_mm'], 3)} yaw {sf['yaw_deg']:.3f}")

    sub = comp[np.random.default_rng(0).choice(len(comp), 60000, replace=False)]
    np.savez_compressed(OUT / 'scan_boundary.npz', outline_sample=sub,
                        **{f'side_{k}_{i}': v[i] for k, v in sides.items() for i in (0, 1)})
    (OUT / 'summary.json').write_text(json.dumps({
        'inputs': {'raw_sha256': RAW_SHA, 'target_sha256': TARGET_SHA,
                   'script_sha256': digest(Path(__file__))},
        'reproduction_max_abs_mm': rep_err, 'scan_edge_frame': frame, 'current_scan_anchor': current,
        'side_fraction': SIDE_FRACTION, 'results': results}, indent=1, default=float) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
