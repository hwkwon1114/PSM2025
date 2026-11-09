"""Replicate collaborator experiment "Hyunwoo Run 1" (24 wheeling passes) with the frozen Attempt 209 model.

Card: cards/REPLICATE-HW1.md. Lengths in mm.
CSV frame: x in [0, 220] across the width, y in [0, 124] along the feed, z = 0 nominal plane.
Assumption (card A1): back-face rows are given in the same front-view material frame as front rows.
Solver material frame: u = x - 110, v = y - 62 (sheet centred, as in shape_design.family).

Each CSV pass is a zig-zag polyline: vertical, inclined legs, vertical. It is mapped to one solver
`zigzag` operation whose regenerated directed, ordered strips must equal the CSV legs exactly.
Back-face passes swap the calibrated layer parameters (gtop<->gbot, ortho_top<->ortho_bottom).
"""
import copy
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from shape_design import family

CSV_PATH = ROOT / 'shape_design/data/Wheeling_Toolpath_24_Passes_Consolidated.csv'
SCAN_PATH = ROOT / 'shape_design/data/Hyunwoo_Run_1_24_Passes.ply'
WIDTH_MM, LENGTH_MM = 220.0, 124.0
ORIGIN = np.array([WIDTH_MM / 2, LENGTH_MM / 2])   # CSV (x, y) of the sheet centre
TOL_MM = 1e-9


def read_passes(path=CSV_PATH):
    """Ordered passes: dict(pass, round, region, face, direction, legs[(a, b)] in CSV mm)."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    passes = {}
    for r in rows:
        p = passes.setdefault(int(r['pass_number']), {
            'pass': int(r['pass_number']), 'round': int(r['round']), 'region': r['region'],
            'face': r['face'], 'direction': r['direction'], 'points': []})
        for key in ('round', 'region', 'face', 'direction'):
            if str(p[key]) != r[key]:
                raise ValueError(f'Inconsistent {key} within pass {p["pass"]}')
        if float(r['z_mm']) != 0.0:
            raise ValueError('Non-planar CSV point')
        p['points'].append((int(r['leg_index']), int(r['point_index']), r['point_role'],
                            float(r['x_mm']), float(r['y_mm'])))
    out = []
    for number in sorted(passes):
        p = passes[number]
        pts = sorted(p.pop('points'), key=lambda t: t[1])
        if [t[1] for t in pts] != list(range(len(pts))) or len(pts) % 2:
            raise ValueError(f'Pass {number}: point indices not contiguous pairs')
        legs = []
        for k in range(0, len(pts), 2):
            s, e = pts[k], pts[k + 1]
            if s[2] != 'start' or e[2] != 'end' or s[0] != e[0] or s[0] != k // 2 + 1:
                raise ValueError(f'Pass {number}: malformed leg at point {k}')
            legs.append(((s[3], s[4]), (e[3], e[4])))
        p['legs'] = np.array(legs, float)
        out.append(p)
    if [p['pass'] for p in out] != list(range(1, len(out) + 1)):
        raise ValueError('Pass numbers not contiguous')
    return out


def cpp_strips(lv, alpha, n_strips, rotation_deg, center_uv, reverse):
    """Python copy of C++ buildZigZagStrips (+ reverse_traversal), placed in material (u, v)."""
    seg = family.strip_segments(lv, alpha, n_strips)
    if reverse:
        seg = seg[::-1, ::-1]
    placed = family.rotate(seg.reshape(-1, 2), rotation_deg).reshape(seg.shape) + np.asarray(center_uv)
    return placed


def fit_operation(legs_csv):
    """Unique (alpha sign, rotation, reverse) reproducing the CSV legs exactly; returns geometry dict."""
    legs = legs_csv - ORIGIN                      # material (u, v)
    n = len(legs)
    d = legs[:, 1] - legs[:, 0]
    vertical = np.abs(d[:, 0]) < TOL_MM
    if not (vertical[0] and vertical[-1] and not vertical[1:-1].any()):
        raise ValueError('Expected vertical, inclined..., vertical legs')
    lv = float(np.ptp(legs[:, :, 1]))
    pitch = np.abs(d[1:-1, 0])
    if np.ptp(pitch) > TOL_MM:
        raise ValueError('Non-uniform pitch')
    alpha = math.degrees(math.atan(pitch[0] / lv))
    lo, hi = legs.reshape(-1, 2).min(axis=0), legs.reshape(-1, 2).max(axis=0)
    center = (lo + hi) / 2
    found = []
    for sign, rot, rev in itertools.product((1, -1), (0.0, 180.0), (False, True)):
        trial = cpp_strips(lv, sign * alpha, n, rot, center, rev)
        err = float(np.abs(trial - legs).max())
        if err <= 1e-9:
            found.append({'lv_mm': lv, 'alpha_deg': sign * alpha, 'n_strips': n, 'rotation_deg': rot,
                          'center_uv_mm': [float(center[0]), float(center[1])], 'reverse_traversal': rev,
                          'max_abs_err_mm': err})
    if not found:
        raise ValueError('No zigzag operation reproduces this pass')
    # alpha -> -alpha with 180 deg rotation can be equivalent; take the first and record all.
    best = dict(found[0])
    best['equivalent_solutions'] = len(found)
    return best


def layer_parameters(params, face):
    """Front face: calibrated (worked = top). Back face: swap the layers (researcher instruction)."""
    if face == 'front':
        return params['gtop'], params['gbot'], params['ortho_top'], params['ortho_bottom']
    if face == 'back':
        return params['gbot'], params['gtop'], params['ortho_bottom'], params['ortho_top']
    raise ValueError(face)


def sequence(params, config, passes=None):
    toolpaths = []
    for p in passes or read_passes():
        g = fit_operation(p['legs'])
        gt, gb, ot, ob = layer_parameters(params, p['face'])
        op = {'type': 'zigzag', 'lv_mm': g['lv_mm'], 'alpha_deg': g['alpha_deg'], 'n_strips': g['n_strips'],
              'width_mm': family.BAND_MM, 'center_uv_mm': g['center_uv_mm'], 'shift_uv_mm': [0.0, 0.0],
              'shift_frame': 'material', 'rotation_deg': g['rotation_deg'],
              'reverse_traversal': g['reverse_traversal'],
              'gtop': gt, 'gbot': gb, 'ortho_top': ot, 'ortho_bottom': ob}
        region = p['region'].replace(' ', '').lower()
        toolpaths.append({'id': f"p{p['pass']:02d}_r{p['round']}_{region}_{p['face']}_{p['direction']}",
                          'enabled': True, 'repeat': 1, 'operation': op})
    return {'schema_version': 1, 'units': {'length': 'mm', 'angle': 'deg', 'growth': 'engineering_strain'},
            'defaults': copy.deepcopy(config['sequence_defaults']),
            'hardening': copy.deepcopy(config['hardening']),
            'boundary_conditions': {'enabled': False}, 'toolpaths': toolpaths}


def predicted_hits(face_centroids_uv, passes=None):
    """Plastic hit count per face directly from the CSV legs (independent of the zigzag fit)."""
    c = np.asarray(face_centroids_uv, float)
    n = np.zeros(len(c), int)
    for p in passes or read_passes():
        for a, b in p['legs'] - ORIGIN:
            n += family._distance(c, a, b) <= family.BAND_MM / 2
    return n


def case_config(config, thickness_mm, edge_mm, timeout_s, n_passes):
    cfg = copy.deepcopy(config)
    cfg['sheet'] = {'size_mm': [WIDTH_MM, LENGTH_MM], 'thickness_mm': thickness_mm}
    cfg['solver']['timeout_seconds'] = timeout_s
    cfg['solver']['args']['-native_hybrid_seconds'] = timeout_s
    case = {'id': 'hw_run1', 'repeat': n_passes,
            'solver_args': {'-lx': WIDTH_MM / 2000, '-ly': LENGTH_MM / 2000, '-res': family.mesh_res(edge_mm)}}
    return cfg, case
