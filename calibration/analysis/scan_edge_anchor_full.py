#!/usr/bin/env python3
"""Card EDGE-ANCHOR-1 full pass: edge-anchored scan placement for Runs 1-3 and all saved shared-footprint evaluations.

Solver-free; read-only on every input; writes only run/diagnostics/scan_edge_anchor/.
Formulas F1-F5 and gates G1-G3 are defined in cards/EDGE-ANCHOR-1.md.  The pilot module
(scan_edge_anchor_check.py) is imported for unchanged helpers and left untouched.
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.scan_edge_anchor_check import fit_frame, rect_cloud, reproduce_target, yaw_matrix
from calibration.ewcal.common import digest
from calibration.ewcal.geometry import (SurfaceDistance, compute_shared_footprint_metrics, points, read_mesh,
                                        rigid_fit, sample_mesh_on_shared_footprint, set_points)
from calibration.ewcal.material_frame import direction_frame, score_material_frame

OUT = ROOT / 'run/diagnostics/scan_edge_anchor'
HALF = np.array([100.0, 150.0])
SCANS = {  # historical id -> (public name, frozen target, sha256)
    'run6': ('Run 1', 'run/putong_convention_check/preserved_samples/run6.npz',
             '00d64e18d28eab6c7254d674af8424d169cac3672abaa2bcf072cab59673a348'),
    'run9': ('Run 2', 'run/putong_convention_check/preserved_samples/run9.npz',
             'aa1fc209fa1b1eee7929a050957c3551ec959d09ed0fec92b02769a01451890e'),
    'run10': ('Run 3', 'run/putong_convention_check/preserved_samples/run10.npz',
              '97a0e13ba494487578107c39fc0eff8e44706eb8f0ba783ab0b78583adcbd578')}
CAMPAIGNS = ['convention_corrected', 'run1_clean_24lhs', 'joint_run6_run10', 'joint_stage2/arm_A',
             'joint_stage2/arm_B', 'joint_stage3_gbot0', 'joint_stage4_reduced4', 'fresh_lhs_joint4',
             'multiobjective_joint4', 'pairwise_floor1']
PRIMARY = dict(rank=5, side_fraction=0.8)
SENSITIVITY = [dict(rank=r, side_fraction=f) for r in (1, 5, 20) for f in (0.6, 0.8, 0.9)]
GATE_CENTRE_MM, GATE_YAW_DEG, GATE_REL, GATE_J_MAX = 0.5, 0.2, 0.01, 2.0


# ---------------------------------------------------------------- F1/F2 with explicit settings
def component_points(xy, cell=0.5):
    keys = np.floor(xy / cell).astype(int); origin = keys.min(0); keys -= origin
    counts = np.zeros(tuple(keys.max(0) + 1), int); np.add.at(counts, (keys[:, 0], keys[:, 1]), 1)
    labels, n = ndimage.label(counts >= 2, structure=np.ones((3, 3)))
    if not n: raise ValueError('No dense scan component')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    comp = ndimage.binary_fill_holes(labels == int(np.argmax(sizes)))
    return xy[comp[keys[:, 0], keys[:, 1]]]


def scan_sides(p, rank=5, side_fraction=0.8):
    out = {}
    for name, along, across, lim, sign in (('left', 1, 0, HALF[1], -1), ('right', 1, 0, HALF[1], 1),
                                           ('bottom', 0, 1, HALF[0], -1), ('top', 0, 1, HALF[0], 1)):
        s, v = [], []
        for t in np.arange(-side_fraction * lim, side_fraction * lim, 1.0):
            m = (p[:, along] >= t) & (p[:, along] < t + 1)
            if m.sum() < 50: continue
            vals = np.sort(sign * p[m, across])[::-1]
            s.append(t + .5); v.append(sign * vals[min(rank, len(vals) - 1)])
        if len(s) < 20: raise ValueError(f'Too few edge strips on {name}')
        out[name] = (np.array(s), np.array(v))
    return out


def sim_sides(q, ref, side_fraction=0.8):
    out = {}
    for name, ax, val in (('left', 0, -HALF[0]), ('right', 0, HALF[0]), ('bottom', 1, -HALF[1]), ('top', 1, HALF[1])):
        other = 1 - ax
        m = (np.abs(ref[:, ax] - val) < 1e-6) & (np.abs(ref[:, other]) <= side_fraction * HALF[other])
        if m.sum() < 10: raise ValueError(f'Too few boundary vertices on {name}')
        out[name] = (q[m, other], q[m, ax])
    return out


def score_variant(mesh, target, ref, faces, scan_anchor=None, sim_edge=False):
    """F3/F4; identical to score_material_frame when scan_anchor is None and sim_edge is False."""
    q, _ = direction_frame(points(mesh), ref, faces)
    sim_frame = None
    if sim_edge:
        sim_frame = fit_frame(sim_sides(q, ref, PRIMARY['side_fraction']))
        q = (q - np.array([*sim_frame['centre_mm'], 0.])) @ yaw_matrix(sim_frame['yaw_deg'])
    target = np.asarray(target, float)
    if scan_anchor is None:
        t0 = target - target.mean(0)
    else:
        t0 = (target - np.array([*scan_anchor['centre_mm'], target[:, 2].mean()])) @ yaw_matrix(scan_anchor['yaw_deg'])
    r_scan, t_scan = rigid_fit(t0, np.column_stack([t0[:, :2], np.zeros(len(t0))]))
    aligned = t0 @ r_scan + t_scan
    flat = np.column_stack([aligned[:, :2], np.zeros(len(aligned))])
    for _ in range(6):
        s0 = sample_mesh_on_shared_footprint(q, ref, faces, aligned[:, :2])
        r, t = rigid_fit(s0, flat); q = q @ r + t
        if np.linalg.norm(r - np.eye(3)) < 1e-13 and np.linalg.norm(t) < 1e-13: break
    framed = set_points(mesh, q)
    d, _ = SurfaceDistance(framed).closest(aligned)
    return compute_shared_footprint_metrics(framed, aligned, ref, faces, d), aligned, d, sim_frame, q


# ---------------------------------------------------------------- known-answer checks
def k1():
    rng = np.random.default_rng(20261101)
    for shift, yaw in (((0., 0.), 0.), ((1.3, -0.7), 1.0), ((-2.0, 1.5), -0.6)):
        p = rect_cloud(rng, shift, yaw)
        f = fit_frame(scan_sides(component_points(p[:, :2]), **PRIMARY))
        assert np.allclose(f['centre_mm'], shift, atol=0.05) and abs(f['yaw_deg'] - yaw) < 0.01, (f, shift, yaw)
    return 'PASS'


def surface_samples(q, faces, density, rng):
    """Area-uniform random points on the triangulated surface q (density points per mm^2 of 3-D area)."""
    a, b, c = q[faces[:, 0]], q[faces[:, 1]], q[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = rng.poisson(area * density)
    idx = np.repeat(np.arange(len(faces)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])


def k4(case_dir, meta):
    """Synthetic scan = the simulated surface itself, displaced in-plane by a known anchor error and then
    cropped/binned by the production rule in a fixed window.  A and B must recover J ~ 0."""
    st = json.loads((case_dir / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    ref = np.asarray(st['reference_positions'], float) * 1000.0; faces = np.asarray(st['faces'], int)
    mesh = read_mesh(case_dir / 'prediction_mm.vtp', 1.0)
    rng = np.random.default_rng(7)
    _, _, _, _, q = score_variant(mesh, np.load(ROOT / SCANS['run6'][1])['points_mm'], ref, faces)
    out = {}
    for name, shift, yaw in (('aligned', (0., 0.), 0.), ('displaced', (1.0, -1.5), 0.4)):
        raw = surface_samples(q, faces, 40.0, rng) @ yaw_matrix(yaw).T + np.array([*shift, 0.])
        target = reproduce_target(raw, meta)
        frame = fit_frame(scan_sides(component_points(raw[:, :2]), **PRIMARY))
        rec = {'true_shift_mm': shift, 'true_yaw_deg': yaw, 'edge_frame': frame}
        for v, kw in (('current', {}), ('A', {'scan_anchor': frame}), ('B', {'scan_anchor': frame, 'sim_edge': True})):
            rec[v] = score_variant(mesh, target, ref, faces, **kw)[0]['rms_symmetric_mm']
        out[name] = rec
    ok = all(out[k][v] <= 0.01 for k in out for v in ('A', 'B')) and out['aligned']['current'] <= 0.01
    out['status'] = 'PASS' if ok else 'FAIL'
    return out


# ---------------------------------------------------------------- inventory and scoring
def inventory():
    rows, skipped = [], {}
    for camp in CAMPAIGNS:
        for res in sorted(glob.glob(str(ROOT / 'run' / camp / '**/evaluations/*/result.json'), recursive=True)):
            try:
                r = json.loads(Path(res).read_text()); rq = json.loads((Path(res).parent / 'request.json').read_text())
            except (OSError, json.JSONDecodeError):
                skipped['unreadable'] = skipped.get('unreadable', 0) + 1; continue
            if not r.get('valid'):
                skipped['invalid'] = skipped.get('invalid', 0) + 1; continue
            targets = {c['id']: c['target'] for c in rq['config']['cases']}
            reg = rq['config'].get('registration', {}).get('method')
            for c in r['cases']:
                sid = c['id']; m = c.get('metrics') or {}
                if sid not in SCANS or targets.get(sid) != SCANS[sid][1] or reg != 'material_reference' \
                        or 'rms_symmetric_mm' not in m:
                    skipped['other_target_or_metric'] = skipped.get('other_target_or_metric', 0) + 1; continue
                d = Path(c['directory'])
                if not (d / 'prediction_mm.vtp').exists():
                    skipped['missing_mesh'] = skipped.get('missing_mesh', 0) + 1; continue
                rows.append({'campaign': camp, 'result': str(Path(res).relative_to(ROOT)), 'scan': sid,
                             'dir': str(d), 'stored_mm': m['rms_symmetric_mm'],
                             'objective_mm': r.get('objective_mm'), 'n_cases': len(r['cases'])})
    return rows, skipped


_G = {}


def _init(frames):
    _G['frames'] = frames
    _G['targets'] = {k: np.load(ROOT / v[1])['points_mm'] for k, v in SCANS.items()}


def score_row(row):
    d = Path(row['dir']); t0 = time.time()
    try:
        st = json.loads((d / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
        ref = np.asarray(st['reference_positions'], float) * 1000.0; faces = np.asarray(st['faces'], int)
        mesh = read_mesh(d / 'prediction_mm.vtp', 1.0); tgt = _G['targets'][row['scan']]
        fr = _G['frames'][row['scan']]
        cur = score_variant(mesh, tgt, ref, faces)[0]['rms_symmetric_mm']
        prod = score_material_frame(mesh, tgt, ref, faces)[0]['rms_symmetric_mm']
        mA = score_variant(mesh, tgt, ref, faces, scan_anchor=fr)[0]['rms_symmetric_mm']
        mB, _, _, sf, _ = score_variant(mesh, tgt, ref, faces, scan_anchor=fr, sim_edge=True)
        return dict(row, current=cur, production=prod, A=mA, B=mB['rms_symmetric_mm'],
                    sim_edge_centre_mm=sf['centre_mm'], sim_edge_yaw_deg=sf['yaw_deg'], seconds=time.time() - t0)
    except Exception as e:   # recorded, not hidden
        return dict(row, error=f'{type(e).__name__}: {e}')


def discordant(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return int(sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i, j in itertools.combinations(range(len(a)), 2))), \
        len(a) * (len(a) - 1) // 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'card': 'cards/EDGE-ANCHOR-1.md', 'script_sha256': digest(Path(__file__)),
               'pilot_module_sha256': digest(ROOT / 'calibration/analysis/scan_edge_anchor_check.py'),
               'primary_settings': PRIMARY, 'K1': k1()}
    print('K1', summary['K1'], flush=True)

    frames, scans = {}, {}
    for sid, (name, tpath, sha) in SCANS.items():
        assert digest(ROOT / tpath) == sha, sid
        meta = json.loads((ROOT / tpath).with_suffix('.json').read_text()); tr = meta['transform']
        assert digest(Path(meta['source'])) == meta['source_sha256'], sid
        raw = np.unique(points(read_mesh(meta['source'], meta.get('scan_scale_to_mm', 1.0))), axis=0)
        q = (raw - np.array(tr['origin_scanner_mm'])) @ np.array(tr['basis_columns']) - np.array(tr['shift_local_mm'])
        target = np.load(ROOT / tpath)['points_mm']
        rep = reproduce_target(q, meta); key = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))]
        k2 = float(np.abs(key(rep) - key(target)).max()) if rep.shape == target.shape else float('inf')
        if not k2 < 1e-6:
            raise SystemExit(f'K2 FAIL for {sid}: shapes {rep.shape} vs {target.shape}, max diff {k2}')
        comp = component_points(q[np.abs(q[:, 2] - np.median(q[:, 2])) <= 20, :2])
        sens = []
        for s in SENSITIVITY:
            f = fit_frame(scan_sides(comp, **s))
            sens.append(dict(s, **{k: f[k] for k in ('centre_mm', 'yaw_deg', 'yaw_long_deg', 'yaw_short_deg',
                                                       'width_mm', 'length_mm')}, _frame=f))
        frames[sid] = next(x['_frame'] for x in sens if x['rank'] == PRIMARY['rank']
                           and x['side_fraction'] == PRIMARY['side_fraction'])
        sides = scan_sides(comp, **PRIMARY)
        np.savez_compressed(OUT / f'boundary_{sid}.npz', outline_sample=comp[np.random.default_rng(0).choice(len(comp), 60000, replace=False)],
                            **{f'side_{k}_{i}': v[i] for k, v in sides.items() for i in (0, 1)})
        scans[sid] = {'name': name, 'K2_max_abs_mm': k2, 'n_target': len(target),
                      'current_anchor_centre_mm': target[:, :2].mean(0).tolist(), 'edge_frame': frames[sid],
                      'sensitivity': [{k: v for k, v in x.items() if k != '_frame'} for x in sens]}
        print(sid, 'K2 PASS', json.dumps({k: frames[sid][k] for k in ('centre_mm', 'yaw_deg', 'width_mm', 'length_mm')}), flush=True)
    summary['scans'] = scans

    c209 = ROOT / 'run/convention_corrected/run6/evaluations/attempt_0001/case_00'
    summary['K4'] = k4(c209, json.loads((ROOT / SCANS['run6'][1]).with_suffix('.json').read_text()))
    print('K4', json.dumps({k: (v if k == 'status' else {x: v[x] for x in ('current', 'A', 'B')}) for k, v in summary['K4'].items()}), flush=True)
    if summary['K4']['status'] != 'PASS':
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
        raise SystemExit('K4 FAIL')

    rows, skipped = inventory()
    print(f'{len(rows)} case-evaluations to score; skipped {skipped}', flush=True)
    with Pool(args.workers, initializer=_init, initargs=(frames,)) as pool:
        scored = pool.map(score_row, rows, chunksize=4)
    errors = [r for r in scored if 'error' in r]; ok = [r for r in scored if 'error' not in r]
    k3 = [r for r in ok if abs(r['current'] - r['production']) > 1e-12 or abs(r['production'] - r['stored_mm']) > 1e-6]
    summary.update(n_scored=len(ok), n_errors=len(errors), errors=errors[:50], skipped=skipped,
                   K3={'status': 'PASS' if not k3 else 'FAIL', 'n_mismatch': len(k3),
                       'mismatch_examples': [{k: r[k] for k in ('result', 'scan', 'stored_mm', 'production')} for r in k3[:20]]})
    (OUT / 'case_scores.json').write_text(json.dumps(scored, indent=0, default=float) + '\n')

    # gates (K3 mismatches are excluded from gates and reported)
    good = [r for r in ok if r not in k3]
    gates = {}
    for sid in SCANS:
        rs = [r for r in good if r['scan'] == sid]
        fr = frames[sid]
        g1 = bool(np.hypot(*fr['centre_mm']) <= GATE_CENTRE_MM and abs(fr['yaw_deg']) <= GATE_YAW_DEG)
        rel = {v: [abs(r[v] - r['current']) / r['current'] for r in rs if r['current'] < GATE_J_MAX] for v in ('A', 'B')}
        g2 = bool(all(x <= GATE_REL for v in rel for x in rel[v]))
        disc = {}
        for camp in CAMPAIGNS:
            sub = [r for r in rs if r['campaign'] == camp]
            if len(sub) >= 2:
                disc[camp] = {v: discordant([r['current'] for r in sub], [r[v] for r in sub]) for v in ('A', 'B')}
        g3 = all(x[v][0] == 0 for x in disc.values() for v in ('A', 'B'))
        absd = {v: [1e3 * (r[v] - r['current']) for r in rs] for v in ('A', 'B')}
        gates[sid] = {'n': len(rs), 'G1': g1, 'G2': g2, 'G3': bool(g3),
                      'max_rel_change_J_lt_2mm': {v: (max(rel[v]) if rel[v] else None) for v in rel},
                      'delta_um_quantiles': {v: np.quantile(absd[v], [0, .05, .5, .95, 1]).tolist() if absd[v] else None for v in absd},
                      'discordant_pairs_by_campaign': disc}
        print(sid, json.dumps({k: gates[sid][k] for k in ('n', 'G1', 'G2', 'G3', 'max_rel_change_J_lt_2mm')}), flush=True)
    summary['gates'] = gates
    summary['decision'] = ('negligible: keep current anchor' if all(g['G1'] and g['G2'] and g['G3'] for g in gates.values())
                           else 'material: at least one gate failed')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print('DECISION:', summary['decision'], flush=True)


if __name__ == '__main__':
    main()
