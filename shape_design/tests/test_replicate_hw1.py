"""Solver-free known-answer tests for card REPLICATE-HW1 (K1, K3, K4)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shape_design import compare_hw1 as cmp, family, replicate_hw1 as rep

FRONT = ('gtop', 'gbot', 'ortho_top', 'ortho_bottom')


@pytest.fixture(scope='module')
def passes():
    return rep.read_passes()


def test_csv_structure(passes):
    assert len(passes) == 24
    order = [(p['region'], p['direction']) for p in passes[:6]]
    assert order == [('Region 1', 'F'), ('Region 2', 'F'), ('Region 3', 'F'),
                     ('Region 3', 'R'), ('Region 2', 'R'), ('Region 1', 'R')]
    for j, p in enumerate(passes):
        assert p['round'] == j // 6 + 1
        assert (p['face'] == 'back') == (p['region'] == 'Region 3')
        q = passes[j % 6]
        assert np.array_equal(p['legs'], q['legs'])                       # rounds repeat exactly
    for f, r in ((0, 5), (1, 4), (2, 3)):                                 # R retraces F backwards
        assert np.array_equal(passes[r]['legs'], passes[f]['legs'][::-1, ::-1])


def test_k1_every_pass_regenerated_exactly(passes):
    for p in passes:
        g = rep.fit_operation(p['legs'])
        assert g['max_abs_err_mm'] <= 1e-12
        assert g['lv_mm'] == 104.0 and math.isclose(abs(g['alpha_deg']), math.degrees(math.atan(5 / 104)))
        legs = rep.cpp_strips(g['lv_mm'], g['alpha_deg'], g['n_strips'], g['rotation_deg'],
                              g['center_uv_mm'], g['reverse_traversal']) + rep.ORIGIN
        assert np.allclose(legs, p['legs'], rtol=0, atol=1e-12)
        # inside the sheet including the band (C++ footprint check)
        lo = legs.reshape(-1, 2).min(axis=0) - family.BAND_MM / 2
        hi = legs.reshape(-1, 2).max(axis=0) + family.BAND_MM / 2
        assert (lo >= 0).all() and hi[0] <= rep.WIDTH_MM and hi[1] <= rep.LENGTH_MM


def test_cpp_reverse_copy():
    fwd = family.strip_segments(10.0, -45.0, 4)
    rev = rep.cpp_strips(10.0, -45.0, 4, 0.0, (0, 0), True)
    assert np.allclose(rev, [[(-10, -5), (-10, 5)], [(-10, 5), (0, -5)], [(0, -5), (10, 5)], [(10, 5), (10, -5)]])
    assert np.allclose(rev, fwd[::-1, ::-1])


def test_k2_hits_from_csv_equal_hits_from_reconstruction(passes):
    v, f = family.solver_grid(2.0)
    c = v[f][:, :, :2].mean(axis=1)
    direct = rep.predicted_hits(c, passes)
    n = np.zeros(len(c), int)
    for p in passes:
        g = rep.fit_operation(p['legs'])
        for a, b in rep.cpp_strips(g['lv_mm'], g['alpha_deg'], g['n_strips'], g['rotation_deg'],
                                   g['center_uv_mm'], g['reverse_traversal']):
            n += family._distance(c, a, b) <= family.BAND_MM / 2
    assert np.array_equal(direct, n)
    assert direct.max() > 0 and 0.2 < (direct > 0).mean() < 0.5


def test_k3_sequence(passes):
    params, config = family.load_frozen()
    seq = rep.sequence(params, config, passes)
    src = json.loads(family.SOURCE_SEQUENCE.read_text())
    assert seq['defaults'] == src['defaults'] and seq['hardening'] == src['hardening'] == family.HARDENING
    assert len(seq['toolpaths']) == 24 and len({t['id'] for t in seq['toolpaths']}) == 24
    for p, t in zip(passes, seq['toolpaths']):
        op = t['operation']
        got = tuple(op[k] for k in FRONT)
        want = tuple(params[k] for k in FRONT)
        if p['face'] == 'front':
            assert got == want
        else:
            assert got == (want[1], want[0], want[3], want[2])
        assert op['width_mm'] == 2.0 and t['repeat'] == 1 and op['shift_uv_mm'] == [0.0, 0.0]
    cfg, case = rep.case_config(config, 0.61, 2.0, 3600.0, 24)
    assert cfg['sheet'] == {'size_mm': [220.0, 124.0], 'thickness_mm': 0.61}
    assert case['solver_args'] == {'-lx': 0.11, '-ly': 0.062, '-res': 2.0 / 220.0}


def _surface(edge, scale=0.5, odd=1.0):
    u = np.arange(-110, 110 + 1e-9, edge)
    v = np.arange(-62, 62 + 1e-9, edge)
    U, V = np.meshgrid(u, v, indexing='ij')
    U, V = U.ravel(), V.ravel()
    h = scale * (2e-4 * U ** 2 - 3e-4 * V ** 2 + odd * (1.5e-6 * U ** 3 + 2e-5 * U * V) + 1e-8 * U ** 4)
    nu, nv = len(u), len(v)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1
            faces += [(a, b, c), (a, c, d)]
    return np.column_stack((U, V, h)), np.array(faces)


@pytest.fixture(scope='module')
def fixtures():
    pred_v, pred_f = _surface(2.0)
    rot = Rotation.from_euler('xyz', [170, 3, 20], degrees=True).as_matrix()   # turned over + yaw
    move = lambda p: p @ rot.T + np.array([300.0, -500.0, 190.0])
    scan_v, scan_f = _surface(1.0)
    # one debris point, attached to a triangle so the backward surface must drop it
    scan_v = np.vstack([move(scan_v), [[300.0, -500.0, 250.0]]])
    scan_f = np.vstack([scan_f, [[0, 1, len(scan_v) - 1]]])
    # even surface for the scale test: an odd (tilt-like) term would be partly absorbed by rigid alignment
    even_v, even_f = _surface(2.0, odd=0.0)
    half_v, _ = _surface(1.0, scale=0.25, odd=0.0)
    return pred_v, pred_f, scan_v, scan_f, (even_v, even_f, move(half_v))


def test_k4_self_comparison_recovers_pose(fixtures):
    pred_v, pred_f, scan_v, scan_f, _ = fixtures
    s, _ = cmp.compare(pred_v, pred_f, scan_v, scan_f)
    rms = sorted(o['symmetric_rms_mm'] for o in s['orientations'].values())
    assert rms[0] < 0.02 and rms[1] > 10 * rms[0]
    assert s['scan_debris_removed'] == 1 and s['scan_triangles_removed'] == 1
    best = s['orientations'][s['best_orientation']]
    assert best['backward_p95_mm'] < 0.01 and best['icp']['accepted_last'] is not None
    assert s['profile_rms_diff_mm'] < 0.005 and abs(s['profile_ptv_ratio_pred_over_scan'] - 1) < 0.005
    assert all(s['region_sign_match'].values())


def test_profile_bins_are_card_5mm():
    u = np.arange(-110, 110.01, 0.5)
    v = np.arange(-62, 62.01, 0.5)
    U, V = np.meshgrid(u, v, indexing='ij')
    pts = np.column_stack((U.ravel(), V.ravel(), U.ravel() ** 2 * 1e-3))
    c, prof = cmp.profile(pts)
    assert np.allclose(np.diff(c), 5.0) and c[0] == -102.5 and c[-1] == 102.5
    exact = np.array([np.mean(np.arange(x - 2, x + 2.01, 1.0) ** 2) * 1e-3 for x in c])
    assert np.allclose(prof, exact - exact.mean(), atol=1e-12)


def test_k4_scaled_copy_ratio(fixtures):
    _, _, _, _, (even_v, even_f, half) = fixtures
    scan_f = _surface(1.0)[1]
    s, _ = cmp.compare(even_v, even_f, half, scan_f)
    assert abs(s['profile_ptv_ratio_pred_over_scan'] - 2) < 0.03   # ~1% from rigid fit of unequal shapes
    assert all(s['region_sign_match'].values())


def test_read_ply_roundtrip(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    head = b'ply\r\nformat binary_little_endian 1.0\r\nelement vertex 3\r\nproperty float x\r\nproperty float y\r\n' \
           b'property float z\r\nelement face 1\r\nproperty list uchar int vertex_indices\r\nend_header\r\n'
    body = v.astype('<f4').tobytes() + np.array([3], 'u1').tobytes() + np.array([0, 1, 2], '<i4').tobytes()
    path = tmp_path / 't.ply'
    path.write_bytes(head + body)
    gv, gf = cmp.read_ply(path)
    assert np.array_equal(gv, v) and gf.tolist() == [[0, 1, 2]]
