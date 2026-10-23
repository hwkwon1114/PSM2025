#!/usr/bin/env python3
"""Intermediate middle length on N18, preserving the transverse envelope."""
import copy
import json
import math
from prepare_dome_baseline import ROOT, footprint

source = ROOT/'legacy/configs/dome/middle_n18_repeat2.json'
base = json.loads(source.read_text())
config = copy.deepcopy(base)
op = config['toolpaths'][0]['operation']
span = (op['n_strips']-2)*op['lv_mm']*math.tan(math.radians(op['alpha_deg']))
op['lv_mm'] = 168.0
op['alpha_deg'] = math.degrees(math.atan(span/((op['n_strips']-2)*op['lv_mm'])))
for path in config['toolpaths']:
    lo, hi, bottom, top = footprint(path['operation'])
    assert lo >= -100-1e-9 and hi <= 100+1e-9 and bottom >= -150-1e-9 and top <= 150+1e-9
restored = copy.deepcopy(config)
for key in ('lv_mm','alpha_deg'):
    restored['toolpaths'][0]['operation'][key] = base['toolpaths'][0]['operation'][key]
assert restored == base
with (source.parent/'middle_n18_length168.json').open('x') as stream:
    json.dump(config,stream,indent=2); stream.write('\n')
print('middle_n18_length168', op['alpha_deg'], footprint(op))
