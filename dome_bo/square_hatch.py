#!/usr/bin/env python3
"""Two-family hatch parameterization for the 200 x 200 x 0.77 mm square-sheet BO stage.

Frozen in docs/reports/dome_toolpath_investigation.md ("Square-sheet BO stage"). Nothing here
imposes monotone decay, pitch grading or cross-hatch layering; those are possible search outcomes.

Search vector: 9 unit-cube coordinates u in [0,1]^9
    0,1  stroke pitch p_A, p_B            6 .. 16 mm (linear)
    2,3  cross span  S_A, S_B             96 .. 196 mm (linear; realised span = (n-2) p + w)
    4,5  stroke length Lv_A, Lv_B         96 .. 192 mm (linear)
    6,7  repeats r_A, r_B                 {1..6}  (r = 1 + min(5, floor(6 u)))
    8    along-stroke top-growth profile  end ratio e in [0.4, 1] (center_peak, power 2; e = 1 is uniform)
Family A (rotation 0: strokes along v, stepping across u) is applied first, then family B
(rotation 90). Stroke width is fixed at 8 mm, shift 0, alpha = atan(p / Lv). All growth coefficients are
the calibrated Attempt209 constants (unchanged). ``realized_raw`` measures the surrogate coordinates on
the EMITTED recipe, so acquisition and training use the same quantities.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
from crosshatch_generator import (BETA_VOCE, G_TOP_DEFAULT, G_BOT_DEFAULT, ORTHO_TOP_DEFAULT,
                                  ORTHO_BOT_DEFAULT, decode_treatment_field, operation_footprint)

SHEET_HALF_MM = 100.0
WIDTH_MM = 8.0
PROFILE_POWER = 2.0
N_DIM = 9
PITCH = (6.0, 16.0)
SPAN = (96.0, 196.0)
LENGTH = (96.0, 192.0)
END_RATIO = (0.4, 1.0)
MAX_REPEAT = 6
COORD_NAMES = ['pitch_A', 'pitch_B', 'span_A', 'span_B', 'length_A', 'length_B', 'repeat_A', 'repeat_B', 'end_ratio']


def _lerp(u, bounds):
    return bounds[0] + float(np.clip(u, 0.0, 1.0)) * (bounds[1] - bounds[0])


def repeat_from_unit(u: float) -> int:
    return 1 + min(MAX_REPEAT - 1, int(np.floor(MAX_REPEAT * float(np.clip(u, 0.0, 1.0)))))


def unit_from_repeat(r: int) -> float:
    return (r - 1 + 0.5) / MAX_REPEAT


def unit_from_physical(pitch_a, pitch_b, span_a, span_b, length_a, length_b, rep_a, rep_b, end_ratio):
    """Unit-cube point that decodes to the given physical values (reference designs)."""
    f = lambda v, b: (v - b[0]) / (b[1] - b[0])
    return np.array([f(pitch_a, PITCH), f(pitch_b, PITCH), f(span_a, SPAN), f(span_b, SPAN),
                     f(length_a, LENGTH), f(length_b, LENGTH), unit_from_repeat(rep_a), unit_from_repeat(rep_b),
                     f(end_ratio, END_RATIO)])


def strips_for_span(span_mm: float, pitch_mm: float) -> int:
    """n_strips >= 2 with realised span (n-2) p + w nearest to the request and never beyond the sheet."""
    n = 2 + int(np.floor((span_mm - WIDTH_MM) / pitch_mm + 0.5))
    while n > 2 and (n - 2) * pitch_mm + WIDTH_MM > 2.0 * SHEET_HALF_MM:
        n -= 1
    return max(2, n)


def _operation(pitch, span, length, rotation, end_ratio):
    n = strips_for_span(span, pitch)
    alpha = float(np.degrees(np.arctan(pitch / length)))
    profile = ({'mode': 'uniform'} if end_ratio >= 1.0 - 1e-12 else
               {'mode': 'center_peak', 'top_end_ratio': float(end_ratio), 'top_profile_power': PROFILE_POWER})
    return {'type': 'zigzag', 'lv_mm': float(length), 'alpha_deg': round(alpha, 9), 'n_strips': int(n),
            'width_mm': WIDTH_MM, 'center_uv_mm': [0.0, 0.0], 'shift_uv_mm': [0.0, 0.0],
            'shift_frame': 'material', 'rotation_deg': float(rotation),
            'gtop': G_TOP_DEFAULT, 'gbot': G_BOT_DEFAULT,
            'ortho_top': ORTHO_TOP_DEFAULT, 'ortho_bottom': ORTHO_BOT_DEFAULT, 'profile': profile}


def decode(u) -> dict:
    """Unit-cube point -> simulator recipe. Raises if the footprint leaves the sheet."""
    u = np.asarray(u, float)
    if u.shape != (N_DIM,) or not np.all(np.isfinite(u)):
        raise ValueError(f"expected {N_DIM} finite unit coordinates")
    p_a, p_b = _lerp(u[0], PITCH), _lerp(u[1], PITCH)
    s_a, s_b = _lerp(u[2], SPAN), _lerp(u[3], SPAN)
    l_a, l_b = _lerp(u[4], LENGTH), _lerp(u[5], LENGTH)
    e = _lerp(u[8], END_RATIO)
    families = [('A', 0.0, _operation(p_a, s_a, l_a, 0.0, e), repeat_from_unit(u[6])),
                ('B', 90.0, _operation(p_b, s_b, l_b, 90.0, e), repeat_from_unit(u[7]))]
    toolpaths = []
    for name, _, op, repeat in families:
        box = operation_footprint(op)
        if max(abs(box[0]), abs(box[1]), abs(box[2]), abs(box[3])) > SHEET_HALF_MM + 1e-9:
            raise ValueError(f"family {name} footprint {box} leaves the sheet")
        toolpaths.append({'id': f'square_{name}', 'enabled': True, 'repeat': int(repeat), 'operation': op})
    return {"schema_version": 1,
            "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
            "hardening": {"model": "voce_decay", "history_variable": "hit_count", "beta": BETA_VOCE,
                          "floor": 0, "scope": "face_shared", "count_rule": "plastic_hits"},
            "defaults": {"start_mode": "left_bottom_up"},
            "boundary_conditions": {"enabled": False},
            "toolpaths": toolpaths}


def realized_raw(recipe: dict) -> np.ndarray:
    """[pitch_A, pitch_B, span_A, span_B, length_A, length_B, repeat_A, repeat_B, end_ratio] measured on
    the emitted recipe (pitch = Lv tan(alpha), span = (n-2) pitch + w)."""
    out = []
    ops = {tp['id'][-1]: tp for tp in recipe['toolpaths']}
    pitches, spans, lengths, repeats = [], [], [], []
    for fam in 'AB':
        op = ops[fam]['operation']
        pitch = op['lv_mm'] * np.tan(np.radians(op['alpha_deg']))
        pitches.append(pitch)
        spans.append((op['n_strips'] - 2) * pitch + op['width_mm'])
        lengths.append(op['lv_mm'])
        repeats.append(ops[fam]['repeat'])
    prof = ops['A']['operation']['profile']
    end = 1.0 if prof['mode'] == 'uniform' else prof['top_end_ratio']
    return np.array(pitches + spans + lengths + repeats + [end], float)


def warp(raw) -> np.ndarray:
    """Surrogate warp: log of pitch/span/length/repeats, linear end ratio."""
    raw = np.atleast_2d(np.asarray(raw, float))
    out = np.column_stack([np.log(np.clip(raw[:, :8], 1e-9, None)), raw[:, 8]])
    return out.squeeze() if np.ndim(raw) == 1 else out


def recipe_sha(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def decode_key(u) -> tuple:
    """Identity of a decoded design (distinct unit points can decode to the same recipe)."""
    r = realized_raw(decode(u))
    return tuple(np.round(r, 6))


def treatment(recipe: dict, n: int = 81) -> dict:
    """Simulator-faithful cumulative treatment on an n x n material grid (mm); for fixtures/diagnostics."""
    g = np.linspace(-SHEET_HALF_MM, SHEET_HALF_MM, n)
    U, V = np.meshgrid(g, g)
    return decode_treatment_field(recipe, U, V)
