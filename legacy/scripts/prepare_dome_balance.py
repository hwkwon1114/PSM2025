#!/usr/bin/env python3
"""Two predeclared lower-repeat contrasts on the retained N18 layout."""
import copy
import json
from prepare_dome_baseline import ROOT, footprint

base = json.loads((ROOT/'legacy/configs/dome/middle_n18_repeat2.json').read_text())
for name, repeats in [('middle_n18_repeat1', (1,3,3)), ('middle_n18_sides1', (2,1,1))]:
    config = copy.deepcopy(base)
    for path, count in zip(config['toolpaths'], repeats):
        path['repeat'] = count
        lo, hi, bottom, top = footprint(path['operation'])
        assert lo >= -100-1e-9 and hi <= 100+1e-9 and bottom >= -150-1e-9 and top <= 150+1e-9
    restored = copy.deepcopy(config)
    for old, path in zip(base['toolpaths'], restored['toolpaths']):
        path['repeat'] = old['repeat']
    assert restored == base
    target = ROOT/f'legacy/configs/dome/{name}.json'
    with target.open('x') as stream:
        json.dump(config, stream, indent=2); stream.write('\n')
    print(name, sum(repeats))
