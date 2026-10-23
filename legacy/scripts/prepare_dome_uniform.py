#!/usr/bin/env python3
"""Researcher-authorized middle-profile-only diagnostic; no solver execution."""
import copy
import json
from prepare_dome_baseline import ROOT
source = ROOT/'legacy/configs/dome/middle_n18_repeat2.json'
base = json.loads(source.read_text())
config = copy.deepcopy(base)
assert 'profile' not in config['toolpaths'][0]['operation']
config['toolpaths'][0]['operation']['profile'] = {'mode':'uniform'}
restored = copy.deepcopy(config)
del restored['toolpaths'][0]['operation']['profile']
assert restored == base
with (source.parent/'middle_n18_uniform.json').open('x') as stream:
    json.dump(config,stream,indent=2); stream.write('\n')
