"""Run from repository root: python -m calibration.ewcal [action] ..."""
import argparse
import json
import sys
from pathlib import Path
from .common import ROOT, resolve, load_config, Parameters, save_json
from .runner import forward, Evaluator


def main():
    p = argparse.ArgumentParser(description='English Wheel calibration CLI')
    p.add_argument('action', choices=[
        'forward', 'doctor', 'evaluate'])
    p.add_argument('--config', default='calibration/configs/run6.json')
    p.add_argument('--executable', help='Solver binary override')
    p.add_argument('--budget', type=int, help='Bayes total evaluation budget')
    p.add_argument('--parameters', help='JSON file mapping parameters')
    p.add_argument('--case', help='Case ID')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--result', help='Result JSON')
    p.add_argument('--output', help='Output directory')
    p.add_argument('--a', help='First case directory')
    p.add_argument('--b', help='Second case directory')
    args = p.parse_args()

    c = load_config(resolve(args.config))
    if args.executable:
        c['solver']['executable'] = args.executable

    params = Parameters(c)
    x = params.encode(json.loads(Path(args.parameters).read_text())) if args.parameters else params.initial()

    if args.action == 'forward':
        out = Path(args.output) if args.output else Path(c.get('output_dir', 'run/forward'))
        if args.case:
            c['cases'] = [case for case in c['cases'] if case['id'] == args.case]
            if not c['cases']:
                raise ValueError(f"Case '{args.case}' not found in configuration")
        result = forward(c, x, out)
        save_json(out / 'result.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result.get('valid') else 1

    if args.action == 'doctor':
        print('Python:', sys.executable)
        exe = resolve(c['solver']['executable'])
        print('Solver:', exe, 'exists:', exe.is_file())
        return 0 if exe.is_file() else 1

    if args.action == 'evaluate':
        evaluator = Evaluator(c)
        rec = evaluator.evaluate(x, use_cache=not args.no_cache)
        print(json.dumps(rec, indent=2))
        return 0 if rec.get('valid') else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
