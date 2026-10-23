#!/usr/bin/env python3
"""One exploratory single-pass zigzag; no automatic retries or extensions."""
import argparse
import json
import os
import run_forward_model_diagnostics as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    binary = runner.RUN_ROOT / 'absolute_stop_audit/source/bin/shell'
    runner.IMPLEMENTATIONS['current']['binary'] = binary
    path = {
        'id': 'simple_negative_ortho', 'enabled': True, 'repeat': 1,
        'operation': {
            'type': 'zigzag', 'lv_mm': 140.0, 'alpha_deg': 9.13,
            'n_strips': 6, 'width_mm': 10.0,
            'center_uv_mm': [0.0, 0.0], 'rotation_deg': 0.0,
            'gtop': 0.005, 'gbot': 0.0, 'ortho': -0.5,
        },
    }
    case = {
        'name': 'simple_zigzag_negative_ortho_m0p5_g0p005',
        'family': 'single_negative_orthotropy',
        'sequence': runner.sequence([path]), 'res': 0.015,
        'growth_per_hit': 0.005, 'hlbfgs_absolute_gradient_tol': 5e-12,
        'note': 'Exploratory single centered six-strip zigzag, one pass; '
                'uniform top growth .005, bottom zero, ortho -.5; '
                'directional top growth .0025/.0075. No hardening or imposed seed. '
                'Absolute residual gate 1e-11; stability not certified. '
                'One 24-hour allocation; no automatic restart.',
    }
    options = argparse.Namespace(
        force=False, timeout=85800,
        threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
        sequence_adaptive=True, grad_tol=1e-11, seed_amplitude=.004,
        material_baseline=runner.MATERIAL_BASELINE,
    )
    case_dir = runner.RUN_ROOT / 'current' / case['name']
    if args.check:
        assert binary.is_file()
        assert not case_dir.exists(), 'Existing sample must not be overwritten'
        print(json.dumps(case, indent=2))
        print(' '.join(runner.build_command(binary, case, case_dir, 'current', options)))
        return
    if case_dir.exists():
        raise SystemExit(f'Existing sample preserved: {case_dir}')
    raise SystemExit(runner.run_case(case, 'current', options, None))


if __name__ == '__main__':
    main()
