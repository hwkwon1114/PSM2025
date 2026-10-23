#!/usr/bin/env python3
"""Predeclared middle-density contrasts at fixed continuous envelope."""
import copy
import json
import math
from prepare_dome_baseline import ROOT, footprint

base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
original = base['toolpaths'][0]['operation']
span = (original['n_strips']-2)*original['lv_mm']*math.tan(math.radians(original['alpha_deg']))
for repeats in (2, 4):
    config = copy.deepcopy(base)
    path = config['toolpaths'][0]
    path['repeat'] = repeats
    op = path['operation']
    op['n_strips'] = 18
    op['alpha_deg'] = math.degrees(math.atan(span/(16*op['lv_mm'])))
    assert all(abs(a-b)<1e-9 for a,b in zip(footprint(op), footprint(original)))
    restored = copy.deepcopy(config)
    restored['toolpaths'][0]['repeat'] = 2
    restored['toolpaths'][0]['operation']['n_strips'] = original['n_strips']
    restored['toolpaths'][0]['operation']['alpha_deg'] = original['alpha_deg']
    assert restored == base
    target = ROOT/f'legacy/configs/dome/middle_n18_repeat{repeats}.json'
    with target.open('x') as stream:
        json.dump(config,stream,indent=2); stream.write('\n')
    print(target.name, op['alpha_deg'])
