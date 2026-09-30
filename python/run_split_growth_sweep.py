#!/usr/bin/env python3
"""One-pass split-half amplitude sweep; 0.05 is an extreme numerical stress test."""
import argparse
import copy
import json
from collections import Counter
import run_forward_model_diagnostics as runner
from run_split_orthogonal import cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,choices=range(18))
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--max-iter',type=int,default=200000)
    parser.add_argument('--timeout',type=int,default=13800)
    args=parser.parse_args()
    if args.max_iter < 1 or args.timeout < 1:
        parser.error('--max-iter and --timeout must be positive')
    specs=[]
    for growth in (.005,.01,.05):
        signatures=[]
        for original in cases():
            spec=copy.deepcopy(original)
            spec.update(name=original['name']+'_g'+str(growth).replace('.','p'),family='split_growth_sweep',growth_per_hit=growth,
                note=original['note']+f' Growth sweep g={growth}; one application per strip; physical validity and mesh resolution unverified at high growth.')
            if args.max_iter != 200000:
                spec['name'] += f'_iter{args.max_iter}'
            spec['max_iter'] = args.max_iter
            signature=Counter()
            for path in spec['sequence']['toolpaths']:
                path['operation']['gtop']=growth
                op=copy.deepcopy(path['operation']);active=op.pop('active_strips')
                for s in active:signature[(json.dumps(op,sort_keys=True),s)]+=1
            signatures.append(signature)
            specs.append(spec)
        assert all(s==signatures[0] for s in signatures)
        assert len(signatures[0])==30 and set(signatures[0].values())=={1}
    if args.check:
        for i,spec in enumerate(specs):
            command=runner.build_command(runner.IMPLEMENTATIONS['current']['binary'],spec,runner.RUN_ROOT/'current'/spec['name'],'current',argparse.Namespace(sequence_adaptive=False,grad_tol=1e-11))
            assert command[command.index('-max_iter')+1]==str(args.max_iter)
            assert command[command.index('-sequence_minimize_every')+1]=='1'
            print(i,spec['name'],len(spec['sequence']['toolpaths']),'solves')
        print('Verified identical per-strip loading within each six-case amplitude group; no repetitions, hardening, or adaptive subdivision added.')
        return
    if args.index is None:parser.error('--index required')
    options=argparse.Namespace(force=False,timeout=args.timeout,threads=8,sequence_adaptive=False,grad_tol=1e-11,seed_amplitude=.004,material_baseline=runner.MATERIAL_BASELINE)
    raise SystemExit(runner.run_case(specs[args.index],'current',options,None))


if __name__=='__main__':main()
