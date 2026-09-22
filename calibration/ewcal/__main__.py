"""Run from repository root: python -m calibration.ewcal --help."""
import argparse
import importlib
import json
import sys
from pathlib import Path
import numpy as np
from .common import ROOT, resolve, load_config, Parameters, save_json

def main():
    p = argparse.ArgumentParser(description='English Wheel final-surface calibration')
    p.add_argument('action', choices=['doctor', 'prepare', 'dry-run', 'evaluate', 'sensitivity', 'optimize', 'report', 'compare-history', 'bayes'])
    p.add_argument('--config', default='calibration/configs/run6.json')
    p.add_argument('--budget', type=int, help='Bayes total evaluation budget, including completed trials on resume')
    p.add_argument('--parameters', help='JSON file mapping gtop, gbot, ortho_top, ortho_bottom')
    p.add_argument('--case', help='Prepare just this case ID')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--resume', action='store_true', help='Restart optimization from saved best of same identity')
    p.add_argument('--result', help='Result JSON for report; defaults to output_dir/best.json')
    p.add_argument('--output', help='Report directory or dry-run directory')
    p.add_argument('--a', help='First solver case directory for compare-history')
    p.add_argument('--b', help='Second solver case directory for compare-history')
    args = p.parse_args()
    if args.budget is not None and args.action != 'bayes':
        p.error('--budget is supported only for bayes')
    if args.action == 'compare-history':
        from .geometry import read_mesh, points, SurfaceDistance, remove_solver_rigid_motion
        from vtk.util.numpy_support import vtk_to_numpy
        if not args.a or not args.b: p.error('compare-history requires --a and --b')
        dirs = [Path(args.a), Path(args.b)]
        manifests = [json.loads((d/'sequence_result.json').read_text()) for d in dirs]
        meshes = [read_mesh(d/m['final_mesh']) for d,m in zip(dirs, manifests)]
        names = ['abar_top_11','abar_top_12','abar_top_22','abar_bot_11','abar_bot_12','abar_bot_22','total_pass_count']
        # Field names for hit history are detected explicitly below.
        available = {meshes[0].GetCellData().GetArrayName(i) for i in range(meshes[0].GetCellData().GetNumberOfArrays())}
        names = names[:6] + [n for n in sorted(available) if 'total' in n and ('hit' in n or 'pass' in n)]
        if len(names) == 6: raise ValueError('No cumulative hit-count field found')
        checks = {}
        for name in names:
            arrays = [mesh.GetCellData().GetArray(name) for mesh in meshes]
            if any(a is None for a in arrays): raise ValueError(f'Missing field {name}')
            a, b = map(vtk_to_numpy, arrays)
            if a.shape != b.shape: raise ValueError('Mesh topology differs')
            checks[name] = {'max_abs_difference': float(np.max(np.abs(a-b))),
                            'equal': bool(np.allclose(a,b,rtol=1e-12,atol=1e-15))}
        print(json.dumps(checks, indent=2))
        if not all(v['equal'] for v in checks.values()): return 1
        return 0
    c = load_config(resolve(args.config))
    if args.action == 'doctor':
        print('Python:', sys.executable); print('Repository:', ROOT)
        for name in ['numpy','scipy','matplotlib','vtk'] + (['sklearn'] if 'bayesian' in c else []):
            module = importlib.import_module(name)
            print(name, getattr(module,'__version__','installed'))
        exe = resolve(c['solver']['executable'])
        print('Solver:', exe, 'exists:', exe.is_file())
        print('Rectangle size must be checked by the initial evaluation:', c['sheet']['size_mm'], 'mm')
        return 0 if exe.is_file() else 1
    if args.action == 'prepare':
        from .prepare import prepare_case
        selected = [r for r in c['cases'] if not args.case or r['id']==args.case]
        if not selected: raise ValueError('Unknown case ID')
        for r in selected: print(json.dumps(prepare_case(c,r),indent=2))
        return 0
    if args.action == 'report':
        from .report import report
        print(report(args.result or resolve(c['output_dir'])/'best.json',
                     args.output or resolve(c['report_dir'])))
        return 0
    params = Parameters(c)
    x = params.encode(json.loads(Path(args.parameters).read_text())) if args.parameters else params.initial()
    if args.action == 'dry-run':
        from .runner import sequence_json, command
        folder = resolve(args.output or 'calibration/runs/dry-run'); folder.mkdir(parents=True, exist_ok=True)
        for i, case in enumerate(c['cases']):
            seq = folder/f'case_{i:02d}.json'
            save_json(seq, sequence_json(c,case,params.decode(x)))
            cmd = command(c,case,seq); save_json(folder/f'command_{i:02d}.json',cmd)
            import shlex
            print(shlex.join(cmd))
        return 0
    if args.action == 'bayes':
        if args.parameters:
            p.error('Set the initial values in the config for bayes')
        from .bayes import run_bayesian
        result = run_bayesian(c, resume=args.resume, budget=args.budget)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'budget_complete' else 1
    from .runner import Evaluator
    from .optimize import remember_best, sensitivity, optimize
    evaluator = Evaluator(c)
    if args.action == 'evaluate':
        record = evaluator.evaluate(x, use_cache=not args.no_cache); remember_best(evaluator,record)
        print(json.dumps(record,indent=2)); return 0 if record['valid'] else 1
    if args.parameters: raise ValueError('--parameters is supported for evaluate and dry-run only')
    if args.action == 'sensitivity': print(json.dumps(sensitivity(evaluator),indent=2))
    if args.action == 'optimize': print(json.dumps(optimize(evaluator,resume=args.resume),indent=2))
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. Bayesian observations are checkpointed; resume with --resume.', file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
