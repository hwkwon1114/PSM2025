#!/usr/bin/env python3
"""Global smooth-surface curvature on retained mesh-study states (no solves).
Fit X(u,v) (x,y,z) with a total-degree-d Legendre polynomial in material coordinates (area-weighted least squares),
take principal curvatures analytically from the fitted fundamental forms, then apply the existing objective and gate.
Usage: surface_fit_scan.py --cell A RES   -> run/dome_square/mesh_study/surface_fit/a{A}_{h}mm.json
       surface_fit_scan.py --summary"""
import json
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as L
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import mesh_convergence as mc  # noqa: E402
import square_attempt as sa    # noqa: E402

OUT = mc.STUDY / 'surface_fit'
DEGREES = (4, 6, 8, 10, 12)
HALF = 0.1   # sheet half-width (m): material coordinates are scaled to [-1, 1]


def basis(t_u, t_v, d, du=0, dv=0):
    cols = []
    for i in range(d + 1):
        for j in range(d + 1 - i):
            ci = np.zeros(i + 1); ci[i] = 1
            cj = np.zeros(j + 1); cj[j] = 1
            fu = L.legval(t_u, L.legder(ci, du)) if du else L.legval(t_u, ci)
            fv = L.legval(t_v, L.legder(cj, dv)) if dv else L.legval(t_v, cj)
            cols.append(fu * fv / HALF ** (du + dv))
    return np.column_stack(cols)


def fit_curvature(pos, uv, faces, area_v, d):
    tu, tv = uv[:, 0] / HALF, uv[:, 1] / HALF
    w = np.sqrt(area_v)[:, None]
    coef, *_ = np.linalg.lstsq(basis(tu, tv, d) * w, pos * w, rcond=None)
    resid = pos - basis(tu, tv, d) @ coef
    cen = uv[faces].mean(1)
    cu, cv = cen[:, 0] / HALF, cen[:, 1] / HALF
    D = {k: basis(cu, cv, d, *k) @ coef for k in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))}
    Xu, Xv, Xuu, Xuv, Xvv = D[(1, 0)], D[(0, 1)], D[(2, 0)], D[(1, 1)], D[(0, 2)]
    n = np.cross(Xu, Xv)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    E, F, G = (Xu * Xu).sum(1), (Xu * Xv).sum(1), (Xv * Xv).sum(1)
    Lc, M, N = (Xuu * n).sum(1), (Xuv * n).sum(1), (Xvv * n).sum(1)
    den = E * G - F * F
    K = (Lc * N - M * M) / den
    H = (E * N - 2 * F * M + G * Lc) / (2 * den)
    s = np.sqrt(np.maximum(H * H - K, 0))
    return np.column_stack((H - s, H + s)), resid


def cell(a, r):
    h = dict(mc.MESH)[r]
    st = json.loads((mc.cell_dir(a, r) / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    pos, uv, faces = (np.asarray(st[k], float if k != 'faces' else int) for k in ('positions', 'material_uv', 'faces'))
    area_f, _ = sa.dco.material_areas(uv, faces)
    area_v = np.zeros(len(uv))
    for k in range(3):
        np.add.at(area_v, faces[:, k], area_f / 3)
    res = {}
    for d in DEGREES:
        vals, resid = fit_curvature(pos, uv, faces, area_v, d)
        m = sa.dco.evaluate_dome_objective(vals, np.ones(len(faces), bool), uv, faces,
                                           roi_half=(sa.ROI_HALF_M, sa.ROI_HALF_M))
        rw = np.sqrt((area_v[:, None] * resid ** 2).sum() / area_v.sum() / 3)
        res[d] = {k: m[k] for k in ('largest_connected_fraction', 'total_coherent_fraction', 'j_combined',
                                   'saddle_flux_fraction')}
        res[d]['fit_rms_residual_mm'] = float(rw * 1e3)
    OUT.mkdir(exist_ok=True)
    (OUT / f'a{a}_{h}mm.json').write_text(json.dumps(res, indent=2) + '\n')


def summary():
    for d in DEGREES:
        print(f'--- degree {d}: gate / total fraction / J / fit rms (mm) at 4, 2, 1 mm')
        tab = {}
        for a in mc.CAND:
            row = []
            for _, h in mc.MESH:
                f = OUT / f'a{a}_{h}mm.json'
                row.append(json.loads(f.read_text())[str(d)] if f.exists() else None)
            tab[a] = row
            print(a, ' | '.join('missing' if x is None else '%.2f %.2f %.3f %.3f' % (
                x['largest_connected_fraction'], x['total_coherent_fraction'], x['j_combined'],
                x['fit_rms_residual_mm']) for x in row))
        if all(x is not None for r in tab.values() for x in r):
            for k in ('largest_connected_fraction', 'j_combined'):
                print(k, 'Spearman 4v2, 2v1:', ' '.join('%.2f' % spearmanr([tab[a][i][k] for a in mc.CAND],
                      [tab[a][i + 1][k] for a in mc.CAND])[0] for i in (0, 1)))


if __name__ == '__main__':
    if sys.argv[1] == '--cell':
        cell(int(sys.argv[2]), sys.argv[3])
    else:
        summary()
