"""Eight-variable domain for stage2; stage1 code and results remain unchanged."""
import copy
import math
import dome_bo_domain as geometry

NAMES = geometry.NAMES + ('middle_repeats', 'side_repeats')
BOUNDS = geometry.BOUNDS + ((1,30),(1,30))


def decode(point):
    if len(point)!=8 or not all(math.isfinite(v) and 0<=v<=1 for v in point):
        raise ValueError('Expected eight finite unit coordinates')
    return geometry.decode(point[:6])+[int(math.floor(1+29*v+.5)) for v in point[6:]]


def encode(values):
    if len(values)!=8:
        raise ValueError('Expected eight physical parameters')
    repeats=values[6:]
    if any(not math.isfinite(v) or isinstance(v,bool) or int(v)!=v or not 1<=v<=30 for v in repeats):
        raise ValueError('Repeat counts must be integers in1..30')
    return geometry.encode(values[:6])+[(v-1)/29 for v in repeats]


def sequence(base,values):
    encode(values)
    result=geometry.sequence(base,values[:6])
    for i,path in enumerate(result['toolpaths']):
        path['repeat']=int(values[6 if i==0 else 7])
    return result


def coordinates(config):
    paths=config['toolpaths']
    if paths[1]['repeat']!=paths[2]['repeat']:
        raise ValueError('Side repeat counts must be symmetric')
    return geometry.coordinates(config)+[paths[0]['repeat'],paths[1]['repeat']]


def compatible(base,candidate):
    return sequence(base,coordinates(candidate))==candidate
