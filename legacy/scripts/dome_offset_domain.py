"""Nine-variable dome domain; symmetric side offset allows inward overlap."""
import copy
import math
import dome_bo_repeat_domain as repeats

NAMES = repeats.NAMES + ('side_offset_mm',)
BOUNDS = repeats.BOUNDS + ((0.,118.),)


def decode(point):
    if len(point)!=9 or not math.isfinite(point[8]) or not 0<=point[8]<=1:
        raise ValueError('Expected nine finite unit coordinates')
    return repeats.decode(point[:8])+[118.*point[8]]


def encode(values):
    if len(values)!=9 or not math.isfinite(values[8]) or not 0<=values[8]<=118:
        raise ValueError('Side offset must be within0..118mm')
    return repeats.encode(values[:8])+[values[8]/118.]


def sequence(base, values):
    encode(values)
    shifted=copy.deepcopy(base)
    shifted['toolpaths'][1]['operation']['shift_uv_mm']=[0.,values[8]]
    shifted['toolpaths'][2]['operation']['shift_uv_mm']=[0.,-values[8]]
    # Validate after shifting: inward paths may have larger feasible spans.
    return repeats.sequence(shifted,values[:8])


def coordinates(config):
    b,c=[p['operation']['shift_uv_mm'] for p in config['toolpaths'][1:]]
    if b[0]!=0 or c[0]!=0 or b[1]!=-c[1]:
        raise ValueError('Expected symmetric side centers')
    return repeats.coordinates(config)+[b[1]]


def compatible(base,candidate):
    return sequence(base,coordinates(candidate))==candidate
