#!/usr/bin/env python3
"""Three predeclared baseline contrasts; input generation only, no simulation."""
import copy
import json
import math
from pathlib import Path
from prepare_dome_baseline import ROOT, footprint

base_path = ROOT/'legacy/configs/dome/baseline_attempt209.json'
base = json.loads(base_path.read_text())
contrasts = {}
middle = copy.deepcopy(base)
middle['toolpaths'][0]['repeat'] = 4
contrasts['middle_repeat4'] = middle
sides = copy.deepcopy(base)
for path in sides['toolpaths'][1:]:
    path['repeat'] = 6
contrasts['sides_repeat6'] = sides
longer = copy.deepcopy(base)
op = longer['toolpaths'][0]['operation']
span = (op['n_strips']-2)*op['lv_mm']*math.tan(math.radians(op['alpha_deg']))
op['lv_mm'] = 196.
op['alpha_deg'] = math.degrees(math.atan(span/((op['n_strips']-2)*op['lv_mm'])))
contrasts['middle_length196'] = longer
for name, config in contrasts.items():
    assert config['hardening'] == base['hardening']
    for original, changed in zip(base['toolpaths'],config['toolpaths']):
        for key in ('gtop','gbot','ortho_top','ortho_bottom','width_mm','rotation_deg'):
            assert original['operation'][key] == changed['operation'][key]
        lo, hi, bottom, top = footprint(changed['operation'])
        assert lo >= -100-1e-9 and hi <= 100+1e-9 and bottom >= -150-1e-9 and top <= 150+1e-9
    target = base_path.parent/(name+'.json')
    with target.open('x') as stream:
        json.dump(config,stream,indent=2)
        stream.write('\n')
    print(name, 'cycles', sum(p['repeat'] for p in config['toolpaths']))
