#!/usr/bin/env python3
"""Matched strip segmentation experiment; sequential solves, not a Markov proof."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import run_forward_model_diagnostics as runner
from run_solver_characteristics import zigzag
from analyze_sequence_ablation_v2 import load_vtp, metric_vector, proper_kabsch, cell_array


def cases():
    result = []
    for label, growth in [('low', .00012), ('high', .0012)]:
        for schedule, size in [('all',8), ('all_repeat',8), ('pairs',2), ('strips',1)]:
            paths=[]
            for start in range(0,8,size):
                path=zigzag(f'S{start:02d}',lv_mm=200,alpha_deg=9.462322208,
                            n_strips=8,width_mm=16,gtop=growth)
                path['operation']['active_strips']=list(range(start,start+size))
                paths.append(path)
            result.append(dict(name=f'segmentation_{label}_{schedule}',family='segmentation',
                res=.015,growth_per_hit=growth,sequence=runner.sequence(paths),
                note=f'gtop={growth}; fixed eight-strip footprint, ordered groups of {size}; '
                     'no hardening, top-only isotropic growth, every group equilibrated.'))
    return result


def analyze(specs):
    meshes={};results=[]
    for spec in specs:
        directory=runner.RUN_ROOT/'current'/spec['name']
        r=json.loads((directory/'result.json').read_text())
        if r['return_code']!=0: continue
        data,x,tri=load_vtp(directory/r['final_vtp'])
        meshes[spec['name']]=(data,x,tri)
    for label in ('low','high'):
        names=[s['name'] for s in specs if s['name'].startswith(f'segmentation_{label}_')]
        for a,b in itertools.combinations(names,2):
            if a not in meshes or b not in meshes:continue
            da,xa,ta=meshes[a];db,xb,tb=meshes[b]
            assert np.array_equal(ta,tb)
            ma,mb=metric_vector(da),metric_vector(db)
            _,rms,maximum=proper_kabsch(xa,xb)
            results.append(dict(left=a,right=b,rms_mm=rms*1000,max_mm=maximum*1000,
                metric_relative_difference=float(np.max(np.abs(ma-mb))/max(np.max(np.abs(ma)),np.max(np.abs(mb)))),
                hit_maps_identical=bool(np.array_equal(cell_array(da,'total_hit_count'),cell_array(db,'total_hit_count')))))
    (runner.RUN_ROOT/'segmentation_comparison.json').write_text(json.dumps(dict(comparisons=results,
        interpretation='Different shapes at equal target metrics demonstrate solve-schedule sensitivity, not proof of stable minima or non-Markovian full-state dynamics.'),indent=2)+'\n')
    print(json.dumps(results,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    specs=cases()
    if args.list:
        for s in specs:print(s['name'],len(s['sequence']['toolpaths']),'solves',s['sequence']['toolpaths'][0]['operation']['gtop'])
        return
    options=argparse.Namespace(force=False,timeout=2700,threads=8,sequence_adaptive=False,
                              grad_tol=1e-11,seed_amplitude=.004,material_baseline=runner.MATERIAL_BASELINE)
    failures=[]
    for spec in specs:
        if runner.run_case(spec,'current',options,None)!=0:failures.append(spec['name'])
    analyze(specs)
    if failures:raise SystemExit('Failed cases: '+', '.join(failures))


if __name__=='__main__':main()
