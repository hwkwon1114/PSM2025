#!/usr/bin/env python3
"""Bounded g=0.005 recheck: 2 meshes x 2 orders x 2 absolute residual gates.

Fresh loading trajectories, not VTP checkpoint restarts. Acceptance is unchanged;
HLBFGS relative stopping is disabled and the absolute stop is half the gate.
"""
import argparse
import copy
from pathlib import Path
import run_forward_model_diagnostics as runner
from run_nested_order_mesh_tol_check import build_specs


def specs():
    out=[]
    for res in (0.015, 0.01):
        for original in build_specs():
            if original['growth_per_hit'] != 0.005:
                continue
            spec=copy.deepcopy(original)
            spec['res']=res
            spec['name']=spec['name'].replace('nested_mesh_tol_', 'nested_absstop_').replace('res0p01_', 'res'+str(res).replace('.', 'p')+'_')
            spec['family']='nested_absolute_stop'
            spec['hlbfgs_absolute_gradient_tol']=0.5*spec['grad_tol']
            spec['note']='Fresh g=0.005 coarse/fine recheck; analytic HLBFGS gradient; absolute stopping at half the unchanged acceptance gate; no change to loading or continuation.'
            out.append(spec)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(8))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--binary', type=Path, required=True)
    args=parser.parse_args()
    runner.IMPLEMENTATIONS['current']['binary']=args.binary.resolve()
    for i,spec in enumerate(specs()):
        options=argparse.Namespace(force=False,timeout=85800,threads=8,sequence_adaptive=True,grad_tol=spec['grad_tol'],seed_amplitude=.004,material_baseline=runner.MATERIAL_BASELINE)
        if args.check:
            cmd=runner.build_command(args.binary,spec,runner.RUN_ROOT/'current'/spec['name'],'current',options)
            flags=dict(zip(cmd[1::2],cmd[2::2]))
            assert float(flags['-hlbfgs_absolute_gradient_tol'])==0.5*float(flags['-equilibrium_grad_tol'])
            assert float(flags['-res'])==spec['res']
            print(i,spec['name'],'absolute stop',spec['hlbfgs_absolute_gradient_tol'],'gate',spec['grad_tol'])
        elif args.index==i:
            if not args.binary.is_file():
                parser.error('verified binary missing')
            raise SystemExit(runner.run_case(spec,'current',options,None))
    if not args.check and args.index is None:
        parser.error('--index required')


if __name__=='__main__':main()
