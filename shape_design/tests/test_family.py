"""Solver-free known-answer tests for the double-curvature family (protocol checks 1-5)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.geometry import rigid_fit
from shape_design import analysis, family


def grid(edge=2.0):
    u = np.arange(-110, 110 + 1e-9, edge)
    v = np.arange(-62, 62 + 1e-9, edge)
    U, V = np.meshgrid(u, v, indexing='ij')
    ref = np.column_stack((U.ravel(), V.ravel(), np.zeros(U.size)))
    nu, nv = len(u), len(v)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1
            faces += [(a, b, c), (a, c, d)]
    return ref, np.array(faces)


def test_strip_builder_matches_cpp_formula():
    seg = family.strip_segments(10.0, -45.0, 4)
    want = [[(10, -5), (10, 5)], [(10, 5), (0, -5)], [(0, -5), (-10, 5)], [(-10, 5), (-10, -5)]]
    assert np.allclose(seg, want, atol=1e-12)


@pytest.mark.parametrize('name', ['F', 'C'])
def test_pitch_extent_and_footprint(name):
    lv, a, n, rot = family.pass_geometry(name, 1)
    seg = family.strip_segments(lv, a, n)
    pitch = abs(seg[1:-1, 1, 0] - seg[1:-1, 0, 0])
    assert np.allclose(pitch, 130 * math.tan(math.radians(1)), rtol=1e-12)
    assert family.footprint_ok(seg, rot, margin_mm=8.0)
    material = family.rotate(seg.reshape(-1, 2), rot)
    extent = np.abs(material).max(axis=0)
    assert np.allclose(extent, [100.0, 52.0] if name == 'F' else [100.0, 52.19], atol=0.2)


@pytest.mark.parametrize('variant', family.VARIANTS)
@pytest.mark.parametrize('k', family.K_VALUES)
def test_rounds_masks_and_pass_count(variant, k):
    n_center, n_edge = family.rounds(k)
    passes = family.build_passes(variant, k)
    assert len(passes) == 4 * max(n_center, n_edge)
    for name in 'FC':
        sub = [p for p in passes if p['name'] == name]
        center = [p['active'][p['center']].all() for p in sub]
        edge = [p['active'][~p['center']].all() for p in sub]
        assert sum(center) == 2 * n_center and sum(edge) == 2 * n_edge
        for p in sub:   # each region is either fully on or fully off within a pass
            assert p['active'][p['center']].all() or not p['active'][p['center']].any()
            assert p['active'][~p['center']].all() or not p['active'][~p['center']].any()
            assert (p['center'] == p['center'][::-1]).all()       # symmetric mask
            assert 0.45 < p['center'].mean() < 0.55
    if variant == 'oneface':
        assert all(p['face'] == 1 for p in passes)
    else:
        for name in 'FC':
            assert sum(p['face'] for p in passes if p['name'] == name) == 0


def canonical(segments):
    """Order-independent segment set: endpoints ordered lexicographically as (u, v) pairs."""
    s = np.round(np.asarray(segments, float), 9)
    swap = (s[:, 0, 0] > s[:, 1, 0]) | ((s[:, 0, 0] == s[:, 1, 0]) & (s[:, 0, 1] > s[:, 1, 1]))
    s[swap] = s[swap][:, ::-1]
    s = s.reshape(len(s), -1)
    return s[np.lexsort(s.T[::-1])]


@pytest.mark.parametrize('name', ['F', 'C'])
def test_back_pass_is_mirror_of_front(name):
    front = family.pass_geometry(name, 1)
    back = family.pass_geometry(name, -1)
    f = family.rotate(family.strip_segments(*front[:3]).reshape(-1, 2), front[3]).reshape(-1, 2, 2)
    b = family.rotate(family.strip_segments(*back[:3]).reshape(-1, 2), back[3]).reshape(-1, 2, 2)
    mirrored = f * np.array([-1.0, 1.0])
    assert np.allclose(canonical(mirrored), canonical(b), atol=1e-9)
    # Indexed: strip k of the back pass is the mirror of front strip k (same traversal order).
    assert np.allclose(mirrored, b, atol=1e-9)
    assert not np.allclose(f, b, atol=1e-3)          # an unmirrored back pass would fail


def test_sequence_json_uses_frozen_parameters():
    params, config = family.load_frozen()
    assert params == family.FROZEN
    for variant in family.VARIANTS:
        for k in (-4, 0, 4):
            seq = family.sequence(variant, k, params, config)
            json.dumps(seq)
            assert seq['hardening'] == family.HARDENING
            assert seq['defaults'] == {'start_mode': 'left_bottom_up', 'profile': {'mode': 'uniform'}}
            ids = [t['id'] for t in seq['toolpaths']]
            assert len(set(ids)) == len(ids)
            for t, p in zip(seq['toolpaths'], family.build_passes(variant, k)):
                op = t['operation']
                assert len(op['gtop']) == len(op['gbot']) == op['n_strips']
                want = family.layer_parameters(params, p['face'])
                assert set(op['gtop']) <= {0.0, want[0]} and set(op['gbot']) <= {0.0, want[1]}
                assert (op['ortho_top'], op['ortho_bottom']) == want[2:]
                assert [g != 0 for g in op['gtop']] == list(p['active'])
                assert family.footprint_ok(p['segments'], op['rotation_deg'], margin_mm=8.0)
            cfg, case = family.case_config(config, variant, k)
            assert case['repeat'] == len(seq['toolpaths'])
            assert cfg['solver']['args']['-equilibrium_grad_tol'] == config['solver']['args']['-equilibrium_grad_tol']
            assert np.isclose(2 * case['solver_args']['-lx'] * case['solver_args']['-res'], 0.002)
            assert cfg['sheet']['thickness_mm'] == 0.61


def test_predicted_hits_follow_regions():
    ref, faces = grid()
    centroids = ref[faces][:, :, :2].mean(axis=1)
    near = lambda u, v: np.argmin(np.hypot(centroids[:, 0] - u, centroids[:, 1] - v))
    win = lambda u, v: (np.abs(centroids[:, 0] - u) <= 6) & (np.abs(centroids[:, 1] - v) <= 6)
    dome = family.predicted_hits(centroids, 'flip', 4)
    saddle = family.predicted_hits(centroids, 'flip', -4)
    assert dome[win(0, 0)].mean() > 20 and saddle[win(0, 0)].max() == 0
    assert dome[win(80, 40)].max() == 0 and saddle[win(80, 40)].mean() > 20
    assert dome[near(105, 0)] == 0 and dome[near(0, 58)] == 0           # untreated margins
    # Same passes on one face: identical up to mirrored-strip aliasing on the asymmetric triangulation.
    assert abs(family.predicted_hits(centroids, 'oneface', 4).sum() / dome.sum() - 1) < 0.02


def test_solver_grid_matches_cpp_counts():
    for edge, shape in ((2.0, (111, 63)), (1.0, (221, 125))):
        v, f = family.solver_grid(edge)
        assert len(v) == shape[0] * shape[1] and len(f) == 2 * (shape[0] - 1) * (shape[1] - 1)
        assert np.allclose(np.ptp(v[:, :2], axis=0), [220, 124])
        assert f.min() == 0 and f.max() == len(v) - 1


def test_treatment_known_answer_single_and_repeated_hit():
    p0 = family.build_passes('oneface', 4)[0]                 # round 0, feed pass, front face
    k = int(np.flatnonzero(p0['active'])[len(np.flatnonzero(p0['active'])) // 2])
    seg = p0['segments'][k]
    point = family.rotate(seg.mean(axis=0)[None], p0['rotation_deg'])
    g = family.FROZEN
    d = family.rotate((seg[1] - seg[0])[None], p0['rotation_deg'])[0]
    d /= np.linalg.norm(d)
    e = np.array([-d[1], d[0]])
    one_top = g['gtop'] * (1 + g['ortho_top']) * np.outer(d, d) + g['gtop'] * (1 - g['ortho_top']) * np.outer(e, e)
    one_bot = g['gbot'] * (1 + g['ortho_bottom']) * np.outer(d, d) + g['gbot'] * (1 - g['ortho_bottom']) * np.outer(e, e)
    n, top, bot = family.treatment(point, 'oneface', 4, passes=[p0])
    assert n[0] == 1 and np.allclose(top[0], one_top, rtol=1e-12) and np.allclose(bot[0], one_bot, rtol=1e-12)
    n, top, _ = family.treatment(point, 'oneface', 4, passes=[p0, p0])
    assert n[0] == 2 and np.allclose(top[0], (1 + math.exp(-family.HARDENING['beta'])) * one_top, rtol=1e-12)
    back = dict(p0, face=-1)                                  # same footprint, layers swapped
    n, top, bot = family.treatment(point, 'oneface', 4, passes=[p0, back])
    q = math.exp(-family.HARDENING['beta'])
    assert np.allclose(top[0] - bot[0], (1 - q) * (one_top - one_bot), rtol=1e-12)


def test_curl_diagnostic_reference_and_finiteness():
    ref = family.curl_diagnostic('oneface', 0)
    assert np.allclose(ref['tiles'], 1.0) and np.isclose(ref['treated_rectangle_ratio'], 1.0)
    flip = family.curl_diagnostic('flip', 0)
    assert np.isfinite(flip['tiles']).all() and len(flip['tiles']) == 50
    assert 'face_one_sided_fraction' not in flip  # a different footprint cannot define this statistic


def test_mirrored_footprint_has_exclusive_back_hits():
    vertices, faces = family.solver_grid(2.0)
    centers = vertices[faces][:, :, :2].mean(axis=1)
    point = np.array([-98.66666666666667, -49.333333333333336])
    idx = np.argmin(np.linalg.norm(centers - point, axis=1))
    assert np.linalg.norm(centers[idx] - point) < 1e-8
    passes = family.build_passes('flip', 0)
    back = [p for p in passes if p['face'] == -1]
    front = [p for p in passes if p['face'] == 1]
    nback = family.treatment(centers[idx:idx+1], 'flip', 0, passes=back)[0][0]
    nfront = family.treatment(centers[idx:idx+1], 'flip', 0, passes=front)[0][0]
    assert (nback, nfront) == (8, 0)
    assert family.predicted_hits(centers[idx:idx+1], 'oneface', 0)[0] == 0


def test_quadratic_roi_includes_boundary_rows():
    v, _ = family.solver_grid(2.0)
    jitter = v + 1.5e-6 * np.sign(v)                           # Float32-scale perturbation
    assert analysis.treated(jitter).sum() == analysis.treated(v).sum() == 101 * 53


def surface(kind, R=1000.0):
    ref, faces = grid()
    u, v = ref[:, 0], ref[:, 1]
    h = {'dome': (u * u + v * v) / (2 * R), 'saddle': (u * u - v * v) / (2 * R), 'cylinder': u * u / (2 * R)}[kind]
    return ref, faces, np.column_stack((u, v, h))


@pytest.mark.parametrize('kind,label', [('dome', 'dome-like'), ('saddle', 'saddle-like'), ('cylinder', 'cylinder-like')])
def test_quadratic_fit_labels(kind, label):
    ref, _, shape = surface(kind)
    fit = analysis.quadratic_fit(ref, analysis.height(shape))
    assert fit['label'] == label
    expected = {'dome': 1e-6, 'saddle': -1e-6, 'cylinder': 0.0}[kind]
    assert np.isclose(fit['K_fit_per_mm2'], expected, atol=1e-12)


def test_height_is_invariant_to_rigid_motion():
    ref, _, shape = surface('dome')
    moved = shape @ Rotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix().T + [5.0, -3.0, 7.0]
    rot, t = rigid_fit(moved, ref)
    aligned = moved @ rot + t
    assert np.allclose(analysis.height(aligned), analysis.height(shape), atol=1e-8)


@pytest.mark.parametrize('kind,sign', [('dome', 'positive'), ('saddle', 'negative')])
def test_local_curvature_sign(kind, sign):
    ref, faces, shape = surface(kind)
    out = analysis.local_curvature(ref, shape, faces)
    assert out[f'{sign}_area_fraction'] > 0.95
    assert np.isclose(abs(out['median_gauss_per_m2']), 1.0, rtol=0.05)


def test_pca_recovers_one_parameter_family():
    ref, _, dome = surface('dome')
    _, _, saddle = surface('saddle')
    s = np.linspace(0, 1, 9)
    fields = [analysis.height(si * dome + (1 - si) * saddle) for si in s]
    out = analysis.pca(fields, s)
    assert out['explained_ratio'][0] > 0.999999
    assert np.all(np.diff(out['scores'][:, 0]) > 0)
