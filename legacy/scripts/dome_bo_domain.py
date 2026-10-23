"""Approved six-dimensional dome domain, with exact integer/footprint checks."""
import copy
import math
from prepare_dome_baseline import footprint

NAMES = ('middle_length', 'middle_angle', 'middle_count', 'side_length', 'side_angle', 'side_count')
BOUNDS = ((100.,196.), (1.,15.), (6,26), (120.,196.), (.5,5.), (4,18))


def decode(point):
    if len(point) != 6 or not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
        raise ValueError('Expected six finite coordinates in [0,1]')
    result = [lo+v*(hi-lo) for v,(lo,hi) in zip(point,BOUNDS)]
    for i in (2,5):
        result[i] = int(math.floor(result[i]+.5))
    return result


def encode(values):
    if len(values) != 6 or not all(math.isfinite(v) and lo <= v <= hi for v,(lo,hi) in zip(values,BOUNDS)):
        raise ValueError('Physical parameters outside approved bounds')
    if any(isinstance(values[i],bool) or int(values[i]) != values[i] for i in (2,5)):
        raise ValueError('Counts must be integers')
    return [(v-lo)/(hi-lo) for v,(lo,hi) in zip(values,BOUNDS)]


def sequence(base, values):
    encode(values)
    config = copy.deepcopy(base)
    for i,path in enumerate(config['toolpaths']):
        length,angle,count = values[:3] if i == 0 else values[3:]
        op = path['operation']
        op.update(lv_mm=length,alpha_deg=angle,n_strips=int(count))
        lo,hi,bottom,top = footprint(op)
        if lo < -100-1e-9 or hi > 100+1e-9 or bottom < -150-1e-9 or top > 150+1e-9:
            raise ValueError('Footprint outside approved sheet')
    return config


def coordinates(config):
    a,b,c = [p['operation'] for p in config['toolpaths']]
    keys = ('lv_mm','alpha_deg','n_strips')
    if any(b[k] != c[k] for k in keys):
        raise ValueError('Side geometry is not symmetric')
    return [a[k] for k in keys]+[b[k] for k in keys]


def compatible(base, candidate):
    """Reject retained observations with different fixed recipes (not just geometry)."""
    values = coordinates(candidate)
    return sequence(base, values) == candidate
