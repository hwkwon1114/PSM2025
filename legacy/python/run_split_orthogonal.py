#!/usr/bin/env python3
"""Run approved half-sheet paths in both orders and three solve schedules."""
import argparse
import copy
import json
from collections import Counter
import run_forward_model_diagnostics as runner


def cases():
    source = runner.RUN_ROOT / 'split_orthogonal_proposal/sequence.json'
    original = json.loads(source.read_text())['toolpaths']
    result = []
    for order in ('AB', 'BA'):
        for schedule, size in [('whole', None), ('pairs', 2), ('strips', 1)]:
            paths = []
            for index in ([0, 1] if order == 'AB' else [1, 0]):
                base = original[index]
                count = base['operation']['n_strips']
                group = size or count
                for start in range(0, count, group):
                    path = copy.deepcopy(base)
                    path['id'] += f'_s{start:02d}'
                    path['repeat'] = 1
                    path['operation']['gtop'] = .0015
                    path['operation']['active_strips'] = list(range(start, min(start+group,count)))
                    paths.append(path)
            result.append(dict(name=f'split_orthogonal_{order}_{schedule}',family='split_orthogonal',
                res=.015,growth_per_hit=.0015,max_iter=200000,sequence=runner.sequence(paths),
                note=f'Approved left/right footprints; {order}; {schedule}; top-only isotropic, no hardening; warm-start every group.'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(6))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    specs = cases()
    if args.check:
        signatures = []
        for spec in specs:
            hits = Counter()
            for path in spec['sequence']['toolpaths']:
                op = copy.deepcopy(path['operation'])
                active = op.pop('active_strips')
                for strip in active:
                    hits[(json.dumps(op,sort_keys=True),strip)] += 1
            signatures.append(hits)
            print(spec['name'], len(spec['sequence']['toolpaths']), 'solves', sum(hits.values()), 'strip applications')
        assert all(s == signatures[0] for s in signatures)
        assert set(signatures[0].values()) == {1}
        assert [len(s['sequence']['toolpaths']) for s in specs] == [2,15,30,2,15,30]
        print('All six schedules apply exactly the same 30 strip operations once each.')
        return
    if args.index is None:
        parser.error('--index is required unless --check is used')
    options = argparse.Namespace(force=False,timeout=13800,threads=8,sequence_adaptive=False,
        grad_tol=1e-11,seed_amplitude=.004,material_baseline=runner.MATERIAL_BASELINE)
    raise SystemExit(runner.run_case(specs[args.index], 'current', options, None))


if __name__ == '__main__':
    main()
