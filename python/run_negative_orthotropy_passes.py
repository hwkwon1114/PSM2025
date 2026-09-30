#!/usr/bin/env python3
"""Three successive rotated zigzags on one plate; fresh trajectory."""
import argparse
import copy
import json
import os
import run_forward_model_diagnostics as runner

NAME = 'zigzag_negative_ortho_m0p5_passes_000_045_090'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = runner.RUN_ROOT / 'current/simple_zigzag_negative_ortho_m0p5_g0p005/sequence.json'
    seq = json.loads(source.read_text())
    template = seq['toolpaths'][0]
    seq['toolpaths'] = []
    for angle in (0, 45, 90):
        path = copy.deepcopy(template)
        path['id'] = f'zigzag_rot{angle:03d}'
        path['operation']['rotation_deg'] = angle
        seq['toolpaths'].append(path)
    binary = runner.RUN_ROOT / 'absolute_stop_audit/source/bin/shell'
    runner.IMPLEMENTATIONS['current']['binary'] = binary
    case = dict(name=NAME, family='negative_orthotropy_passes', sequence=seq,
                res=.015, growth_per_hit=.005, hlbfgs_absolute_gradient_tol=5e-12,
                note='Fresh single-plate trajectory: centered six-strip zigzags at 0,45,90 degrees; '
                     'cumulative multiplicative growth, top .005 per hit, bottom 0, ortho -.5; '
                     'no hardening; absolute acceptance 1e-11; no stability certificate. '
                     '24-hour cap, no automatic restart. Template: '+str(source))
    options = argparse.Namespace(force=False, timeout=84600,
        threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
        sequence_adaptive=True, grad_tol=1e-11, seed_amplitude=.004,
        material_baseline=runner.MATERIAL_BASELINE)
    directory = runner.RUN_ROOT / 'current' / NAME
    if directory.exists():
        raise SystemExit(f'Existing trajectory preserved: {directory}')
    if args.check:
        assert binary.is_file()
        print(json.dumps(case, indent=2))
        print(' '.join(runner.build_command(binary, case, directory, 'current', options)))
        return
    raise SystemExit(runner.run_case(case, 'current', options, None))


if __name__ == '__main__':
    main()
