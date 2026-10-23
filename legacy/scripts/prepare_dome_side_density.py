#!/usr/bin/env python3
"""Side-density contrast against the best retained N18-middle configuration."""
import copy
import json
import math
from prepare_dome_baseline import ROOT, footprint
base = json.loads((ROOT/'legacy/configs/dome/middle_n18_repeat2.json').read_text())
config = copy.deepcopy(base)
for old, path in zip(base['toolpaths'][1:], config['toolpaths'][1:]):
    op = path['operation']
    span = (op['n_strips']-2)*op['lv_mm']*math.tan(math.radians(op['alpha_deg']))
    op['n_strips'] = 14
    op['alpha_deg'] = math.degrees(math.atan(span/(12*op['lv_mm'])))
    assert all(abs(a-b)<1e-9 for a,b in zip(footprint(op), footprint(old['operation'])))
restored = copy.deepcopy(config)
for old, path in zip(base['toolpaths'][1:],restored['toolpaths'][1:]):
    for key in ('n_strips','alpha_deg'):
        path['operation'][key] = old['operation'][key]
assert restored == base
with (ROOT/'legacy/configs/dome/middle_n18_sides_n14.json').open('x') as stream:
    json.dump(config,stream,indent=2); stream.write('\n')
print(config['toolpaths'][1]['operation']['alpha_deg'])
