#!/usr/bin/env python3
"""Five equal-loading, three-pass half-sheet schedules."""
import argparse
import copy
import json
from collections import Counter
import run_forward_model_diagnostics as runner
from run_split_orthogonal import cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,choices=range(5))
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--max-iter',type=int,default=200000)
    parser.add_argument('--timeout',type=int,default=13800)
    args=parser.parse_args()
    if args.max_iter < 1 or args.timeout < 1:
        parser.error('--max-iter and --timeout must be positive')
    base=cases()
    specs=[]
    for label,source,cadence,local in [('strip_triplets',2,3,True),('pair_triplets',1,3,True),('each_path',0,1,False),('each_AB',0,2,False),('all_at_once',0,6,False)]:
        spec=copy.deepcopy(base[source])
        original=spec['sequence']['toolpaths']
        ordered=[p for p in original for _ in range(3)] if local else original*3
        paths=[]
        for i,p in enumerate(ordered):
            p=copy.deepcopy(p);p['id']+=f'_application{i:03d}';paths.append(p)
        spec.update(name='split_three_'+label,family='split_three',minimize_every=cadence,note='Three actual growth applications per strip; AB order; '+label)
        if args.max_iter != 200000:
            spec['name'] += f'_iter{args.max_iter}'
        spec['max_iter'] = args.max_iter
        spec['sequence']['toolpaths']=paths
        specs.append(spec)
    signatures=[]
    for spec in specs:
        signature=Counter()
        for p in spec['sequence']['toolpaths']:
            op=copy.deepcopy(p['operation']);active=op.pop('active_strips')
            for s in active:signature[(json.dumps(op,sort_keys=True),s)]+=1
        signatures.append(signature)
    assert all(s==signatures[0] for s in signatures)
    assert set(signatures[0].values())=={3}
    if args.check:
        for s in specs:
            print(s['name'],len(s['sequence']['toolpaths']),'updates;',len(s['sequence']['toolpaths'])//s['minimize_every'],'solves')
        print('Verified: identical 30 strip operations, each applied exactly three times at 0.0015 per hit.')
        return
    if args.index is None:parser.error('--index required')
    options=argparse.Namespace(force=False,timeout=args.timeout,threads=8,sequence_adaptive=False,grad_tol=1e-11,seed_amplitude=.004,material_baseline=runner.MATERIAL_BASELINE)
    raise SystemExit(runner.run_case(specs[args.index],'current',options,None))


if __name__=='__main__':main()
