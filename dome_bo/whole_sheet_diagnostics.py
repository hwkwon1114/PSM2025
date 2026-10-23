#!/usr/bin/env python3
"""Whole-sheet geometry/curvature diagnostics and reference-area shape comparison of screening vs verification states.

Diagnostics only (frozen primary gate is unchanged: 15 mm, floor 0.1 1/m, ratio 0.1, 80 % connected ROI area).
 * qualification mask = dome_continuous_objective.coherent_masks (floor AND weaker/stronger ratio >= 0.1, validity)
 * whole-sheet and ROI area-weighted curvature distributions and category fractions
 * fold/geometry checks on the actual faces: adjacent-face normal angle, normal-orientation consistency, triangle
   area and edge-length strain vs reference
 * shape comparison with reference-area (material) weights on the FINE mesh faces: coarse position by barycentric
   interpolation on the coarse mesh's actual faces (matplotlib trifinder), area-weighted Kabsch alignment
 * unexaggerated (equal-axis) geometry views and corrected qualification maps
Usage: whole_sheet_diagnostics.py SCREEN_STATE VERIFY_STATE OUTDIR
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import square_attempt as sa            # noqa: E402
import dome_continuous_objective as dco  # noqa: E402

ROI = 0.08


def wquant(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return {f'p{q}': float(x[o][min(np.searchsorted(c, q / 100), len(x) - 1)]) for q in qs}


def analyse(path):
    st = json.loads(Path(path).read_text())
    pos = np.asarray(st['positions'], float)
    ref = np.asarray(st['reference_positions'], float)
    uv = np.asarray(st['material_uv'], float)
    faces = np.asarray(st['faces'], int)
    cache = Path(path).with_name('curvature_r15mm_cache.npz')      # frozen estimator, cached to avoid recomputation
    if cache.exists():
        z_ = np.load(cache)
        vals, valid = z_['vals'], z_['valid']
    else:
        vals, _, valid = sa.geometric_curvature(pos, uv, faces, radius=0.015)
        np.savez(cache, vals=vals, valid=valid)
    area, cen = dco.material_areas(uv, faces)
    roi = (np.abs(cen[:, 0]) <= ROI) & (np.abs(cen[:, 1]) <= ROI)
    finite = np.isfinite(vals).all(1) & valid
    safe = np.where(finite[:, None], vals, 0.0)
    masks = dco.coherent_masks(vals, valid)
    qual = masks[-1] | masks[1]
    small = np.min(np.abs(safe), 1)
    large = np.max(np.abs(safe), 1)
    saddle = finite & (safe[:, 0] * safe[:, 1] < 0) & (small >= dco.FLOOR) & (small >= 0.1 * large)
    cat = np.zeros(len(faces), int)               # 0 weak/not qualifying, 1 qualifying dome sign -/+, 2 saddle, 3 invalid
    cat[qual] = 1
    cat[saddle] = 2
    cat[~finite] = 3
    out = {'n_faces': int(len(faces))}
    for name, sel in (('whole', np.ones(len(faces), bool)), ('roi', roi)):
        w = area[sel]
        d = {'area_fraction': {k: float(area[sel & (cat == v)].sum() / w.sum()) for k, v in
                               (('weak_or_nonqualifying', 0), ('qualifying_one_sign', 1), ('saddle_ratio_and_floor', 2),
                                ('invalid', 3))}}
        f = finite[sel]
        for lab, col in (('k_small_abs', small), ('k_large_abs', large)):
            d[lab] = wquant(col[sel][f], w[f], (1, 10, 50, 90, 99, 100))
        sgn = np.sign(safe[sel][:, 0] + safe[sel][:, 1])
        d['mean_curvature_sign_fraction_negative'] = float(w[f & (sgn < 0)].sum() / w[f].sum())
        out[name] = d
    # geometry / fold checks on actual faces
    tri = pos[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    edges = {}
    for i, f in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.setdefault(tuple(sorted((f[a], f[b]))), []).append(i)
    pairs = np.array([v for v in edges.values() if len(v) == 2])
    cosang = np.sum(n[pairs[:, 0]] * n[pairs[:, 1]], 1)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    # orientation consistency: all normals on the same side of the mean normal
    mean_n = n.mean(0)
    mean_n /= np.linalg.norm(mean_n)
    area3 = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rtri = ref[faces]
    area0 = 0.5 * np.linalg.norm(np.cross(rtri[:, 1] - rtri[:, 0], rtri[:, 2] - rtri[:, 0]), axis=1)
    elen = lambda p: np.linalg.norm(p[faces[pairs[:, 0], 0]] - p[faces[pairs[:, 0], 1]], axis=1)
    out['geometry'] = {
        'adjacent_face_normal_angle_deg': {'median': float(np.median(ang)), 'p99': float(np.percentile(ang, 99)),
                                           'max': float(ang.max())},
        'min_normal_dot_mean_normal': float((n @ mean_n).min()),
        'faces_flipped_relative_to_mean_normal': int(((n @ mean_n) < 0).sum()),
        'area_ratio_deformed_over_reference': {'min': float((area3 / area0).min()), 'median': float(np.median(area3 / area0)),
                                               'max': float((area3 / area0).max())},
        'height_range_m_true_scale_after_plane_removal': None}
    A = np.c_[uv, np.ones(len(uv))]
    z = pos[:, 2] - A @ np.linalg.lstsq(A, pos[:, 2], rcond=None)[0]
    out['geometry']['height_range_m_true_scale_after_plane_removal'] = float(np.ptp(z))
    return st, out, dict(vals=vals, cat=cat, area=area, cen=cen, roi=roi, z=z, uv=uv, pos=pos, faces=faces)


def weighted_kabsch(a, b, w):
    w = w / w.sum()
    ca, cb = (w[:, None] * a).sum(0), (w[:, None] * b).sum(0)
    u, _, vt = np.linalg.svd(((b - cb) * w[:, None]).T @ (a - ca))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return ((b - cb) @ r) + ca


def shape_difference(coarse, fine):
    """Fine-face centroids (material) -> coarse position by barycentric interpolation on the coarse actual faces."""
    cu, cp, cf = coarse['uv'], coarse['pos'], coarse['faces']
    fu, fp, ff = fine['uv'], fine['pos'], fine['faces']
    trif = Triangulation(cu[:, 0], cu[:, 1], cf).get_trifinder()
    cen = fu[ff].mean(1)
    pts_fine = fp[ff].mean(1)
    ti = trif(cen[:, 0], cen[:, 1])
    if (ti < 0).any():
        raise ValueError('fine-face centroid outside the coarse mesh')
    t = cf[ti]
    u0, u1, u2 = cu[t[:, 0]], cu[t[:, 1]], cu[t[:, 2]]
    m = np.stack([u1 - u0, u2 - u0], axis=2)
    lam = np.linalg.solve(m, (cen - u0)[:, :, None])[:, :, 0]
    pts_c = cp[t[:, 0]] + lam[:, :1] * (cp[t[:, 1]] - cp[t[:, 0]]) + lam[:, 1:] * (cp[t[:, 2]] - cp[t[:, 0]])
    res = {}
    for name, sel in (('roi', fine['roi']), ('whole', np.ones(len(ff), bool))):
        w = fine['area'][sel]
        aligned = weighted_kabsch(pts_fine[sel], pts_c[sel], w)        # coarse aligned onto fine
        d = np.linalg.norm(aligned - pts_fine[sel], axis=1)
        res[name] = {'n_fine_faces': int(sel.sum()), 'area_weighted_rms_m': float(np.sqrt((w * d ** 2).sum() / w.sum())),
                     'distance_percentiles_m': wquant(d, w, (50, 90, 99, 100)),
                     'fine_height_range_m': float(np.ptp(pts_fine[sel][:, 2]))}
    return res


def figures(coarse, fine, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    colors = matplotlib.colors.ListedColormap(['#dddddd', '#4477aa', '#222222', '#ee9988'])
    for row, (name, d) in enumerate((('screening 4 mm', coarse), ('verification 2 mm', fine))):
        uv = d['uv'] * 1e3
        for col, (vals, cmap, lab, clim) in enumerate((
                (d['z'][d['faces']].mean(1) * 1e3, 'viridis', 'height (mm), plane removed', None),
                (d['cat'], colors, 'grey weak | blue qualifying one sign | black saddle | salmon invalid', (-0.5, 3.5)),
                (np.log10(np.maximum(np.min(np.abs(np.nan_to_num(d['vals'])), 1), 1e-3)), 'magma',
                 'log10 weaker |curvature| (1/m)', (-1.5, 0.5)))):
            ax = axes[row, col]
            pc = PolyCollection(uv[d['faces']][:, :, :2], array=vals, cmap=cmap, edgecolors='none')
            if clim:
                pc.set_clim(*clim)
            ax.add_collection(pc)
            ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_aspect('equal')
            ax.add_patch(plt.Rectangle((-80, -80), 160, 160, fill=False, ec='k', lw=1, ls='--'))
            ax.set_title(f'{name}: {lab}', fontsize=8); ax.set_xlabel('u (mm)'); ax.set_ylabel('v (mm)')
            if col != 1:
                fig.colorbar(pc, ax=ax, shrink=0.7)
    fig.savefig(out_dir / 'whole_sheet_maps.png', dpi=140)
    plt.close(fig)
    # unexaggerated geometry: equal-axis side profiles through the centre (true scale) + whole-sheet 3D, equal aspect
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    for j, (name, d, c) in enumerate((('screening 4 mm', coarse, 'C0'), ('verification 2 mm', fine, 'C3'))):
        pos = d['pos'] * 1e3
        uv = d['uv'] * 1e3
        ax = fig.add_subplot(2, 3, 1 + 3 * j)
        for axis, ls in ((0, '-'), (1, '--')):
            other = 1 - axis
            sel = np.abs(uv[:, other]) < 1.5
            o = np.argsort(pos[sel][:, axis])
            ax.plot(pos[sel][o][:, axis], pos[sel][o][:, 2] - pos[:, 2].mean(), ls, c=c, lw=1,
                    label=f'section through centre, along {"u" if axis == 0 else "v"}')
        ax.set_aspect('equal'); ax.set_xlim(-110, 110); ax.set_ylim(-110, 110)
        ax.set_title(f'{name}: centre sections, EQUAL axes (no exaggeration), mm', fontsize=8); ax.legend(fontsize=6)
        ax = fig.add_subplot(2, 3, 2 + 3 * j, projection='3d')
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=d['faces'], color=c, alpha=0.7, linewidth=0.1,
                        edgecolor='k')
        ax.set_box_aspect((np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.ptp(pos[:, 2]))); ax.view_init(18, -60)
        ax.set_title(f'{name}: 3D true aspect (all axes same scale)', fontsize=8)
        ax = fig.add_subplot(2, 3, 3 + 3 * j)
        ax.hist(np.degrees(np.arccos(np.clip(0, -1, 1))) * 0 + 0, bins=1)   # placeholder removed below
        ax.remove()
    fig.savefig(out_dir / 'unexaggerated_geometry.png', dpi=140)
    plt.close(fig)


def main(screen, verify, out_dir):
    out_dir = Path(out_dir)
    t0 = time.time()
    sst, sres, sd = analyse(screen)
    print('screen analysed', time.time() - t0, flush=True)
    vst, vres, vd = analyse(verify)
    print('verify analysed', time.time() - t0, flush=True)
    result = {'screen': sres, 'verify': vres, 'shape_area_weighted_fine_faces': shape_difference(sd, vd)}
    print('shape done', time.time() - t0, flush=True)
    (out_dir / 'whole_sheet_diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    figures(sd, vd, out_dir)
    print(json.dumps(result, indent=1)[:6000])


if __name__ == '__main__':
    main(*sys.argv[1:4])
