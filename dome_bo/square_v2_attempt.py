#!/usr/bin/env python3
"""One isolated square-sheet attempt for the v2 search: solve, verify the endpoint, score J_amp on the global fit.

Reuses ``square_attempt.run_attempt`` (solve, endpoint verification, state 'accepted'/'failed'/'error') with the shared
adaptive-cutback solver settings; only the scoring differs. Defaults: 2 mm mesh, gradient gate 1e-14.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))

import sag_objective as so     # noqa: E402
import square_attempt as sa    # noqa: E402

MESH = sa.VERIFY_MESH
TOLERANCE = '1e-14'


def analyze_state(state: dict) -> dict:
    """J_amp plus fit diagnostics. ``j_combined`` is J_fit, so the v1 finiteness check in ``run_attempt`` still applies."""
    metrics = so.score(state)
    return dict(metrics, j_combined=metrics['j_fit'])


def run_attempt(attempt_dir, mesh=MESH, tolerance=TOLERANCE, threads=8, fixture=False, timeout=None):
    v1_analyze, sa.analyze_state = sa.analyze_state, analyze_state     # run_attempt looks it up at call time
    try:
        return sa.run_attempt(attempt_dir, mesh, tolerance, threads, fixture, timeout=timeout)
    finally:
        sa.analyze_state = v1_analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-dir', required=True, type=Path)
    parser.add_argument('--mesh', nargs=3, metavar=('LX', 'LY', 'RES'), default=MESH)
    parser.add_argument('--tolerance', default=TOLERANCE)
    parser.add_argument('--timeout', type=float, help='solver process ceiling (s); default 3600')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Needs a compute allocation')
    status = run_attempt(args.attempt_dir, tuple(args.mesh), args.tolerance, args.threads, args.fixture, args.timeout)
    print(json.dumps({k: status.get(k) for k in ('attempt', 'state', 'failure_type', 'elapsed_seconds')}))
    sys.exit(1 if status.get('state') == 'error' else 0)


if __name__ == '__main__':
    main()
