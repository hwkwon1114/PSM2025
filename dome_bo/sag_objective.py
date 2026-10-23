#!/usr/bin/env python3
"""Capped-sag dome objective on the global surface fit (no solves): J_amp = J_fit * min(sag / A_ref, 1).
sag = |mean height of central ROI (r < 0.2 h) - mean height of ROI rim (0.85 h < r <= h)| of the fitted surface after
removing its best plane over the ROI; J_fit = existing dome_continuous_objective on the degree-d fit curvature.
Usage: sag_objective.py --mesh   -> mesh-study cells (6 candidates x 4/2/1 mm), then ranking table
       sag_objective.py --screen -> the 32 screening states (4 mm)"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import mesh_convergence as mc   # noqa: E402
import square_attempt as sa     # noqa: E402
import surface_fit_scan as sf   # noqa: E402

DEG = 8
A_REF_MM = 1.5          # researcher decision 2026-10-23 (docs/dome_bo/square_bo_v2_protocol.md)
CAPS = (1.0, 1.5, 2.0)


def fit_state(state_path, d=DEG):
    """Degree-d surface fit of a solver state (dict or path): (pos, uv, faces, vertex areas, fit residual, dome-objective metrics)."""
    st = state_path if isinstance(state_path, dict) else json.loads(Path(state_path).read_text())
    pos, uv, faces = (np.asarray(st[k], float if k != 'faces' else int) for k in ('positions', 'material_uv', 'faces'))
    af, _ = sa.dco.material_areas(uv, faces)
    av = np.zeros(len(uv))
    for k in range(3):
        np.add.at(av, faces[:, k], af / 3)
    vals, resid = sf.fit_curvature(pos, uv, faces, av, d)
    m = sa.dco.evaluate_dome_objective(vals, np.ones(len(faces), bool), uv, faces,
                                       roi_half=(sa.ROI_HALF_M, sa.ROI_HALF_M))
    return pos, uv, faces, av, resid, m


def evaluate_sag(pos, uv, resid):
    h = sa.ROI_HALF_M
    zf = pos[:, 2] - resid[:, 2]
    roi = (np.abs(uv[:, 0]) <= h) & (np.abs(uv[:, 1]) <= h)
    A = np.c_[uv[roi], np.ones(roi.sum())]
    z = zf[roi] - A @ np.linalg.lstsq(A, zf[roi], rcond=None)[0]
    r = np.hypot(uv[roi, 0], uv[roi, 1])
    return float(abs(z[r < 0.2 * h].mean() - z[(r > 0.85 * h) & (r <= h)].mean()) * 1e3)


def evaluate(state_path, d=DEG):
    pos, uv, faces, av, resid, m = fit_state(state_path, d)
    sag = evaluate_sag(pos, uv, resid)
    return dict(j=float(m['j_combined']), sag_mm=float(sag), **{f'jamp_{c}': float(m['j_combined'] * min(sag / c, 1)) for c in CAPS})


def score(state, a_ref=A_REF_MM, d=DEG):
    """Search objective J_amp = J_fit * min(sag / a_ref, 1) with the diagnostics recorded for every attempt."""
    pos, uv, faces, av, resid, m = fit_state(state, d)
    sag = evaluate_sag(pos, uv, resid)
    return {'j_amp': float(m['j_combined'] * min(sag / a_ref, 1)), 'j_fit': float(m['j_combined']), 'sag_mm': sag,
            'fit_rms_mm': float(np.sqrt((av[:, None] * resid ** 2).sum() / av.sum() / 3) * 1e3),
            **{k: m[k] for k in ('largest_connected_fraction', 'total_coherent_fraction', 'saddle_flux_fraction',
                                 'qualified')}}


def mesh():
    out = {}
    for a in mc.CAND:
        for r, h in mc.MESH:
            out[f'{a}_{h}'] = evaluate(mc.cell_dir(a, r) / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json')
    (ROOT / 'run/dome_square/mesh_study/sag_objective.json').write_text(json.dumps(out, indent=1) + '\n')
    print('att mesh(mm) J sag_mm', *[f'Jamp{c}' for c in CAPS])
    for a in mc.CAND:
        for _, h in mc.MESH:
            v = out[f'{a}_{h}']
            print(a, h, '%.3f %.2f' % (v['j'], v['sag_mm']), *['%.3f' % v[f'jamp_{c}'] for c in CAPS])
    for k in ('j', 'sag_mm') + tuple(f'jamp_{c}' for c in CAPS):
        t = [[out[f'{a}_{h}'][k] for a in mc.CAND] for _, h in mc.MESH]
        print(k, 'Spearman 4v2, 2v1: %.2f %.2f' % (spearmanr(t[0], t[1])[0], spearmanr(t[1], t[2])[0]),
              'max |2mm-1mm| %.3f' % np.max(np.abs(np.array(t[1]) - np.array(t[2]))))


def screen():
    camp = ROOT / 'run/dome_square/campaign'
    led = {r['attempt']: r for r in map(json.loads, (camp / 'ledger.jsonl').read_text().splitlines())}
    rows = []
    for a in sorted(led):
        if led[a]['state'] == 'accepted':
            sp = glob.glob(str(camp / f'attempts/attempt_{a:03d}/solve/*native_hybrid_state.json'))[0]
            rows.append((a, evaluate(sp)))
    for c in CAPS:
        print(f'--- cap {c} mm: top 8 by J_amp')
        for a, v in sorted(rows, key=lambda t: -t[1][f'jamp_{c}'])[:8]:
            print(a, 'J %.3f sag %.2f Jamp %.3f' % (v['j'], v['sag_mm'], v[f'jamp_{c}']))


if __name__ == '__main__':
    mesh() if sys.argv[1] == '--mesh' else screen()
