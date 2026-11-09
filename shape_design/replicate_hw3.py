"""Replicate collaborator experiment "Run 3" (30 wheeling passes) with the frozen Attempt 209 model.

Length, width, thickness: 220 x 124 x 0.61 mm (same aluminum sheet as Run 1).
CSV frame: x in [-27.7425, 30.0] mm across width (centered around 1.13 mm),
           y in [0, 55] mm along feed (half-width patch from centerline to 55 mm),
           z = 0 nominal plane.
Solver material frame: u = x, v = y (origin at sheet center for u, centerline for v).
Center of toolpath: [u0, v0] = [1.12875, 27.5] mm.

All 30 passes traverse the same 14-strip zigzag patch:
- 15 unique waypoints, 14 strips (1 vertical, 12 diagonal at alpha = 5.0 deg, 1 vertical).
- Odd passes (1, 3, ..., 29): forward traversal (reverse_traversal = True in C++ strip convention).
- Even passes (2, 4, ..., 30): reverse traversal (reverse_traversal = False in C++ strip convention).
- Constitutive parameters: frozen Attempt 209 (worked front face).
"""
import copy
import csv
import itertools
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from shape_design import family

CSV_PATH = ROOT / 'shape_design/data/Run3_toolpath.csv'
SCAN_PATH = ROOT / 'shape_design/data/Run_3.ply'
WIDTH_MM, LENGTH_MM, THICKNESS_MM = 220.0, 124.0, 0.61
TOL_MM = 1e-4


def read_passes(path=CSV_PATH):
    """Ordered passes: dict(pass, direction, legs[(a, b)] in CSV mm)."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    raw_passes = {}
    for r in rows:
        p_num = int(r['pass_number'])
        p = raw_passes.setdefault(p_num, {
            'pass': p_num,
            'direction': r['direction'],
            'points': []
        })
        if float(r['z_mm']) != 0.0:
            raise ValueError(f'Pass {p_num}: non-planar CSV point')
        p['points'].append((int(r['point_in_pass']), float(r['x_mm']), float(r['y_mm'])))

    out = []
    for number in sorted(raw_passes):
        p = raw_passes[number]
        # Sort by point_in_pass
        pts_in = sorted(p.pop('points'), key=lambda t: t[0])
        # Deduplicate consecutive identical waypoints (CNC dwell points at turn vertices)
        unique_pts = []
        for _, x, y in pts_in:
            pt = np.array([x, y])
            if not unique_pts or np.linalg.norm(pt - unique_pts[-1]) > 1e-6:
                unique_pts.append(pt)

        if len(unique_pts) != 15:
            raise ValueError(f'Pass {number}: expected 15 unique waypoints, got {len(unique_pts)}')

        legs = []
        for k in range(len(unique_pts) - 1):
            legs.append((unique_pts[k], unique_pts[k + 1]))
        p['legs'] = np.array(legs, float)
        out.append(p)

    if [p['pass'] for p in out] != list(range(1, len(out) + 1)):
        raise ValueError('Pass numbers not contiguous 1..N')
    return out


def fit_operation(legs_csv, pass_number):
    """Verify and return the C++ zigzag operation dict reproducing the CSV legs exactly."""
    lv = 55.0
    alpha = 5.0
    n = 14
    center = np.array([1.12875, 27.5])
    rev = (pass_number % 2 == 1)

    trial = family.strip_segments(lv, alpha, n)
    if rev:
        trial = trial[::-1, ::-1]
    trial = family.rotate(trial.reshape(-1, 2), 0.0).reshape(trial.shape) + center

    err = float(np.abs(trial - legs_csv).max())
    if err > TOL_MM:
        raise ValueError(f'Pass {pass_number}: max error {err:.2e} exceeds tolerance {TOL_MM}')

    return {
        'lv_mm': lv,
        'alpha_deg': alpha,
        'n_strips': n,
        'rotation_deg': 0.0,
        'center_uv_mm': [float(center[0]), float(center[1])],
        'reverse_traversal': rev,
        'max_abs_err_mm': err
    }


def sequence(params, config, passes=None):
    toolpaths = []
    all_passes = passes or read_passes()
    for p in all_passes:
        g = fit_operation(p['legs'], p['pass'])
        op = {
            'type': 'zigzag',
            'lv_mm': g['lv_mm'],
            'alpha_deg': g['alpha_deg'],
            'n_strips': g['n_strips'],
            'width_mm': family.BAND_MM,
            'center_uv_mm': g['center_uv_mm'],
            'shift_uv_mm': [0.0, 0.0],
            'shift_frame': 'material',
            'rotation_deg': g['rotation_deg'],
            'reverse_traversal': g['reverse_traversal'],
            'gtop': params['gtop'],
            'gbot': params['gbot'],
            'ortho_top': params['ortho_top'],
            'ortho_bottom': params['ortho_bottom']
        }
        toolpaths.append({
            'id': f"p{p['pass']:02d}_{p['direction']}",
            'enabled': True,
            'repeat': 1,
            'operation': op
        })

    return {
        'schema_version': 1,
        'units': {'length': 'mm', 'angle': 'deg', 'growth': 'engineering_strain'},
        'defaults': copy.deepcopy(config['sequence_defaults']),
        'hardening': copy.deepcopy(config['hardening']),
        'boundary_conditions': {'enabled': False},
        'toolpaths': toolpaths
    }


def predicted_hits(face_centroids_uv, passes=None):
    """Plastic hit count per face directly from the CSV legs (independent of the zigzag fit)."""
    c = np.asarray(face_centroids_uv, float)
    n = np.zeros(len(c), int)
    for p in passes or read_passes():
        for a, b in p['legs']:
            n += family._distance(c, a, b) <= family.BAND_MM / 2
    return n


def case_config(config, thickness_mm, edge_mm, timeout_s, n_passes):
    cfg = copy.deepcopy(config)
    cfg['sheet'] = {'size_mm': [WIDTH_MM, LENGTH_MM], 'thickness_mm': thickness_mm}
    cfg['solver']['timeout_seconds'] = timeout_s
    cfg['solver']['args']['-native_hybrid_seconds'] = timeout_s
    case = {
        'id': 'hw_run3',
        'repeat': n_passes,
        'solver_args': {
            '-lx': WIDTH_MM / 2000,
            '-ly': LENGTH_MM / 2000,
            '-res': family.mesh_res(edge_mm)
        }
    }
    return cfg, case

