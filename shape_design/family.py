"""Toolpath family for the 124 x 220 x 0.61 mm Attempt 209 double-curvature study.

Protocol: docs/shape_design/attempt209_double_curvature.md. Lengths in mm unless noted.
Material coordinates: u across the 220 mm width, v along the 124 mm feed direction.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / 'run/convention_corrected/run6'
SOURCE_SEQUENCE = SOURCE_DIR / 'evaluations/attempt_0001/case_00/sequence.json'
SOURCE_CONFIG = SOURCE_DIR / 'proposed_config.json'

WIDTH_MM, LENGTH_MM, THICKNESS_MM = 220.0, 124.0, 0.61
PITCH_MM = 130.0 * math.tan(math.radians(1.0))      # Run 1 strip pitch
BAND_MM = 2.0
PASSES = {'F': {'lv_mm': 104.0, 'n_strips': 90, 'rotation_deg': 0.0},   # strokes along v (feed)
          'C': {'lv_mm': 200.0, 'n_strips': 48, 'rotation_deg': 90.0}}  # sheet rotated: strokes along u
CENTER_FRACTION = 0.5
BASE_ROUNDS = 4
K_VALUES = tuple(range(-4, 5))
VARIANTS = ('flip', 'oneface')
# (pass, face relative to the round's Thue-Morse face); chosen from the hardening law only.
ROUND_ORDER = {'flip': (('F', 1), ('F', -1), ('C', -1), ('C', 1)),
               'oneface': (('F', 1), ('F', 1), ('C', 1), ('C', 1))}
FROZEN = {'gtop': 0.00028835116227798787, 'gbot': 1.4587994001079146e-05,
          'ortho_top': -0.3989157893255133, 'ortho_bottom': 0.960306323937336}
HARDENING = {'model': 'voce_decay', 'history_variable': 'hit_count', 'beta': 0.2231435513142097,
             'floor': 0, 'scope': 'face_shared', 'count_rule': 'plastic_hits'}


def mesh_res(edge_mm=2.0):
    """`-res` giving `edge_mm` cells: the rectangle generator uses edge = 2 * lx * res."""
    return edge_mm / WIDTH_MM


def load_frozen():
    """Attempt 209 growth/hardening and solver config, checked against the documented values."""
    seq = json.loads(SOURCE_SEQUENCE.read_text())
    op = seq['toolpaths'][0]['operation']
    params = {k: op[k] for k in FROZEN}
    if params != FROZEN or seq['hardening'] != HARDENING:
        raise ValueError('Source sequence does not hold the frozen Attempt 209 parameters')
    config = json.loads(SOURCE_CONFIG.read_text())
    if config['hardening'] != HARDENING or config['sequence_defaults'] != seq['defaults']:
        raise ValueError('Source config and sequence disagree')
    return params, config


def alpha_deg(lv_mm):
    """Negative tilt like Run 1, with the Run 1 pitch."""
    return -math.degrees(math.atan(PITCH_MM / lv_mm))


def strip_segments(lv_mm, alpha, n_strips):
    """Python copy of C++ zigzag::buildZigZagStrips (local frame, mm): list of (a, b)."""
    if n_strips < 2:
        raise ValueError('n_strips must be >= 2')
    n_incl = n_strips - 2
    p = lv_mm * math.tan(math.radians(alpha))
    y_bot, y_top = -0.5 * lv_mm, 0.5 * lv_mm
    width = n_incl * p
    x_left, x_right = -0.5 * width, 0.5 * width
    pts = [(x_left, y_bot), (x_left, y_top)]
    for j in range(1, n_incl + 1):
        pts.append((x_left + j * p, y_bot if j % 2 == 1 else y_top))
    last = pts[-1]
    pts.append((x_right, y_bot if abs(last[1] - y_top) < 1e-15 else y_top))
    segments = [(pts[0], pts[1])] + [(pts[k], pts[k + 1]) for k in range(1, n_incl + 1)]
    segments.append((pts[-2], pts[-1]))
    return np.array(segments, float)            # (n_strips, 2 endpoints, 2 coords)


def rotate(xy, degrees):
    t = math.radians(degrees)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return xy @ r.T


def pass_geometry(name, face):
    """(lv, alpha, n_strips, rotation) of a pass; a back-face pass is mirrored u -> -u."""
    g = PASSES[name]
    a = alpha_deg(g['lv_mm'])
    rot = g['rotation_deg']
    if face < 0:
        a, rot = -a, -rot
    return g['lv_mm'], a, g['n_strips'], rot


def footprint_ok(segments, rotation_deg, margin_mm=0.0):
    """C++ checkTransformedFootprintInsidePanel on the 220 x 124 mm panel."""
    lo = segments.reshape(-1, 2).min(axis=0) - BAND_MM / 2
    hi = segments.reshape(-1, 2).max(axis=0) + BAND_MM / 2
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    c = rotate(corners, rotation_deg)
    half = np.array([WIDTH_MM, LENGTH_MM]) / 2 - margin_mm
    return bool((np.abs(c) <= half + 1e-9).all())


def central_mask(segments):
    """True where the strip midpoint lies in the middle half of the stepping-axis extent."""
    x = segments[:, :, 0].mean(axis=1)
    half = np.abs(segments[:, :, 0]).max()
    return np.abs(x) <= CENTER_FRACTION * half + 1e-9


def thue_morse(j):
    return 1 if bin(j).count('1') % 2 == 0 else -1


def rounds(k):
    n_center, n_edge = BASE_ROUNDS + k, BASE_ROUNDS - k
    if min(n_center, n_edge) < 0:
        raise ValueError('k outside the family')
    return n_center, n_edge


def build_passes(variant, k):
    """Ordered list of passes: dicts with name, face, round, geometry and active strip mask."""
    n_center, n_edge = rounds(k)
    out = []
    for j in range(max(n_center, n_edge)):
        for name, rel in ROUND_ORDER[variant]:
            face = thue_morse(j) * rel if variant == 'flip' else 1
            lv, a, n, rot = pass_geometry(name, face)
            seg = strip_segments(lv, a, n)
            center = central_mask(seg)
            active = np.where(center, j < n_center, j < n_edge)
            if not active.any():
                raise AssertionError('empty pass')
            out.append({'name': name, 'face': face, 'round': j, 'lv_mm': lv, 'alpha_deg': a,
                        'n_strips': n, 'rotation_deg': rot, 'segments': seg, 'active': active,
                        'center': center})
    return out


def layer_parameters(params, face):
    if face > 0:
        return params['gtop'], params['gbot'], params['ortho_top'], params['ortho_bottom']
    return params['gbot'], params['gtop'], params['ortho_bottom'], params['ortho_top']


def sequence(variant, k, params, config):
    toolpaths = []
    for i, p in enumerate(build_passes(variant, k)):
        gt, gb, ot, ob = layer_parameters(params, p['face'])
        op = {'type': 'zigzag', 'lv_mm': p['lv_mm'], 'alpha_deg': p['alpha_deg'],
              'n_strips': p['n_strips'], 'width_mm': BAND_MM, 'center_uv_mm': [0.0, 0.0],
              'shift_uv_mm': [0.0, 0.0], 'shift_frame': 'material', 'rotation_deg': p['rotation_deg'],
              'gtop': [gt if a else 0.0 for a in p['active']],
              'gbot': [gb if a else 0.0 for a in p['active']],
              'ortho_top': ot, 'ortho_bottom': ob}
        face = 'front' if p['face'] > 0 else 'back'
        toolpaths.append({'id': f"p{i:02d}_r{p['round']}_{p['name']}_{face}", 'enabled': True,
                          'repeat': 1, 'operation': op})
    return {'schema_version': 1, 'units': {'length': 'mm', 'angle': 'deg', 'growth': 'engineering_strain'},
            'defaults': copy.deepcopy(config['sequence_defaults']),
            'hardening': copy.deepcopy(config['hardening']),
            'boundary_conditions': {'enabled': False}, 'toolpaths': toolpaths}


def solver_grid(edge_mm=2.0):
    """Copy of C++ RectangularPlate_RightAngle (open): vertices (mm, z=0) and faces."""
    def axis(half):
        n = math.ceil(2 * half / edge_mm)
        n = n if n % 2 == 1 else n + 1
        return n, 2 * half / (n - 1)
    nx, dx = axis(WIDTH_MM / 2)
    ny, dy = axis(LENGTH_MM / 2)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    vertices = np.column_stack((-WIDTH_MM / 2 + i.ravel() * dx, -LENGTH_MM / 2 + j.ravel() * dy, np.zeros(nx * ny)))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    v0, v1 = (i * ny + j).ravel(), ((i + 1) * ny + j).ravel()
    v2, v3 = ((i + 1) * ny + j + 1).ravel(), (i * ny + j + 1).ravel()
    faces = np.empty((2 * len(v0), 3), int)
    faces[0::2] = np.column_stack((v0, v1, v2))
    faces[1::2] = np.column_stack((v0, v2, v3))
    return vertices, faces


def _distance(points, a, b):
    d = b - a
    t = np.clip(((points - a) @ d) / (d @ d), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


def treatment(face_centroids_mm, variant, k, params=FROZEN, passes=None):
    """Emulate hits in solver order (pass, then strip): returns plastic hit counts and the
    hardening-weighted additive top/bottom growth tensors (2x2, material u-v) per face.
    A small-strain linearization of the solver's multiplicative metric update."""
    c = np.asarray(face_centroids_mm, float)
    n = np.zeros(len(c), int)
    top, bot = np.zeros((len(c), 2, 2)), np.zeros((len(c), 2, 2))
    for p in build_passes(variant, k) if passes is None else passes:
        gt, gb, ot, ob = layer_parameters(params, p['face'])
        local = rotate(c, -p['rotation_deg'])
        for seg, act in zip(p['segments'], p['active']):
            if not act:
                continue
            hit = _distance(local, seg[0], seg[1]) <= BAND_MM / 2
            if not hit.any():
                continue
            q = HARDENING['floor'] + (1 - HARDENING['floor']) * np.exp(-HARDENING['beta'] * n[hit])
            d = rotate((seg[1] - seg[0])[None], p['rotation_deg'])[0]
            d /= np.linalg.norm(d)
            e = np.array([-d[1], d[0]])
            along, across = np.outer(d, d), np.outer(e, e)
            top[hit] += q[:, None, None] * (gt * (1 + ot) * along + gt * (1 - ot) * across)
            bot[hit] += q[:, None, None] * (gb * (1 + ob) * along + gb * (1 - ob) * across)
            n[hit] += 1
    return n, top, bot


def predicted_hits(face_centroids_mm, variant, k):
    """Plastic hit count per face, using the C++ centroid-distance rule; centroids centred on the panel."""
    return treatment(face_centroids_mm, variant, k)[0]


def curl_diagnostic(variant, k, edge_mm=2.0, tiles=(10, 5)):
    """Pre-solve residual-curl diagnostic from actual strip geometry and hardening order.

    Layer difference D = top - bottom (linearized). Ratio = |mean D(variant)| / |mean D(oneface)|
    (Frobenius norm) over equal-area tiles of the treated rectangle, same k. Not a shape bound."""
    verts, faces = solver_grid(edge_mm)
    cen = verts[faces][:, :, :2].mean(axis=1)
    _, t1, b1 = treatment(cen, variant, k)
    _, t0, b0 = treatment(cen, 'oneface', k)
    d1, d0 = t1 - b1, t0 - b0
    out = {'tiles': []}
    norm = lambda m: float(np.linalg.norm(m))
    ux = np.linspace(-100, 100, tiles[0] + 1)
    vz = np.linspace(-52, 52, tiles[1] + 1)
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            m = (cen[:, 0] >= ux[i]) & (cen[:, 0] < ux[i + 1]) & (cen[:, 1] >= vz[j]) & (cen[:, 1] < vz[j + 1])
            ref = norm(d0[m].mean(axis=0))
            if ref > 0:
                out['tiles'].append(norm(d1[m].mean(axis=0)) / ref)
    m = (np.abs(cen[:, 0]) <= 100) & (np.abs(cen[:, 1]) <= 52)
    out['treated_rectangle_ratio'] = norm(d1[m].mean(axis=0)) / norm(d0[m].mean(axis=0))
    out['tile_ratio_median'] = float(np.median(out['tiles']))
    out['tile_ratio_max'] = float(np.max(out['tiles']))
    out['tile_size_mm'] = [200 / tiles[0], 104 / tiles[1]]
    return out


def case_config(config, variant, k, edge_mm=2.0, timeout_s=3600):
    """Calibration-runner config/case for one family member; only geometry and timeout differ."""
    cfg = copy.deepcopy(config)
    cfg['sheet'] = {'size_mm': [WIDTH_MM, LENGTH_MM], 'thickness_mm': THICKNESS_MM}
    cfg['solver']['timeout_seconds'] = timeout_s
    cfg['solver']['args']['-native_hybrid_seconds'] = timeout_s
    case = {'id': f'{variant}_k{k:+d}', 'repeat': len(build_passes(variant, k)),
            'solver_args': {'-lx': WIDTH_MM / 2000, '-ly': LENGTH_MM / 2000, '-res': mesh_res(edge_mm)}}
    return cfg, case
