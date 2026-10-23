#!/usr/bin/env python3
"""Prepare the approved adapted collaborator baseline; never runs the solver."""
import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'calibration/data/processed/zigzag_doublecurv.json'
CALIBRATION = ROOT / 'run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00/sequence.json'


def footprint(op):
    """Width-inflated transformed rectangle, matching the C++ placement check (mm)."""
    lv, width, count, angle = (op[k] for k in ('lv_mm', 'width_mm', 'n_strips', 'alpha_deg'))
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError('n_strips must be an integer >=2')
    if not all(math.isfinite(v) for v in (lv, width, angle)) or lv <= 0 or width <= 0 or not 0 <= angle < 90:
        raise ValueError('Invalid zigzag dimensions')
    if op['shift_frame'] != 'material':
        raise ValueError('Only material-frame placement is supported here')
    x = ((count - 2) * lv * math.tan(math.radians(angle)) + width) / 2
    y = (lv + width) / 2
    a = math.radians(op['rotation_deg'])
    c, s = math.cos(a), math.sin(a)
    center, shift = op['center_uv_mm'], op['shift_uv_mm']
    points = [(center[0] + shift[0] + c*u - s*v,
               center[1] + shift[1] + s*u + c*v)
              for u in (-x, x) for v in (-y, y)]
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('Nonfinite placement')
    return [min(p[0] for p in points), max(p[0] for p in points),
            min(p[1] for p in points), max(p[1] for p in points)]


def prepare(source, calibration):
    result = copy.deepcopy(source)
    calibrated = calibration['toolpaths'][0]['operation']
    result['hardening'] = copy.deepcopy(calibration['hardening'])
    result['boundary_conditions'] = {'enabled': False}
    if len(result['toolpaths']) != 3:
        raise ValueError('Expected collaborator A/B/C layout')
    for index, path in enumerate(result['toolpaths']):
        op = path['operation']
        op.pop('ortho', None)
        for key in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
            op[key] = calibrated[key]
        if index:
            op['lv_mm'] = 196.0
            op['shift_uv_mm'] = [0.0, 118.0 if index == 1 else -118.0]
        lo_u, hi_u, lo_v, hi_v = footprint(op)
        if lo_u < -100-1e-9 or hi_u > 100+1e-9 or lo_v < -150-1e-9 or hi_v > 150+1e-9:
            raise ValueError('Footprint outside approved sheet')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = prepare(json.loads(SOURCE.read_text()), json.loads(CALIBRATION.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
