#!/usr/bin/env python3
"""Descriptive estimator-scale sensitivity on retained endpoints, not new scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dome_bo'))
from dome_curvature import geometric_curvature, coverage

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('runs', nargs='+', type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
report = {}
for run in args.runs:
    state = json.loads((run/'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    assert state['proof']['accepted'] is True
    positions, uv, faces = [np.asarray(state[key]) for key in ('positions','material_uv','faces')]
    scales = {}
    for radius in (.01,.015,.02):
        k, _, valid = geometric_curvature(positions, uv, faces, radius=radius)
        score = coverage(k, valid, uv, faces)
        center = uv[faces].mean(axis=1)
        roi = (abs(center[:,0]) <= .08) & (abs(center[:,1]) <= .12)
        edges = uv[faces][:,1:]-uv[faces][:,0,None]
        area = abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])/2
        score['saddle_area_fraction'] = float(area[roi & valid & (np.prod(k,axis=1)<0)].sum()/area[roi].sum())
        score['coverage_threshold_sensitivity'] = {str(t): max(score['sign_fractions'].values())>=t for t in (.7,.8,.9)}
        scales[str(radius)] = score
    report[run.name] = scales
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
    stream.write('\n')
