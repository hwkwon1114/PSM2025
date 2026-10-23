#!/usr/bin/env python3
"""Mesh/tolerance convergence study driver: one fresh solve of an existing square-campaign recipe at a given mesh.

Fixed: frozen binary, request.json of the campaign attempt, tolerance 1e-14. Reuses square_attempt.run_attempt unchanged.
Usage: mesh_study.py TASK_INDEX   (index into TASKS; Slurm array)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import square_attempt as sa  # noqa: E402

CANDIDATES = [30, 31, 21, 32, 27, 19]
RES = {'.005': 16200, '.01': 5400, '.02': 5400}           # relative resolution -> 1/2/4 mm edge; solver ceiling (s)
ANALYSIS_CEILING = 3600
WATCHDOG_1MM = '14400'      # -native_hybrid_seconds only for 1 mm (emergency watchdog; deterministic 40-Hessian/2000-warmup caps unchanged)
STUDY = ROOT / 'run/dome_square/mesh_study'
# 1 mm first (longest), attempt 30 at 2 mm is reused from the verification run
TASKS = [(a, r) for r in ('.005', '.01', '.02') for a in CANDIDATES if (a, r) != (30, '.01')]


def main(i):
    attempt, res = TASKS[i]
    out = STUDY / f'a{attempt:03d}_res{res[1:]}'
    out.mkdir(parents=True, exist_ok=False)
    src = ROOT / f'run/dome_square/campaign/attempts/attempt_{attempt:03d}/request.json'
    (out / 'request.json').write_text(src.read_text())
    sa.ANALYSIS_TIMEOUT = ANALYSIS_CEILING
    if res == '.005':
        base = sa.solver_options
        sa.solver_options = lambda *a: {**base(*a), '-native_hybrid_seconds': WATCHDOG_1MM}
    status = sa.run_attempt(out, ('.1', '.1', res), '1e-14', int(os.environ.get('SLURM_CPUS_PER_TASK', 8)),
                            timeout=RES[res])
    print(json.dumps({k: status.get(k) for k in ('attempt', 'state', 'failure_type', 'elapsed_seconds')}), res)
    sys.exit(1 if status.get('state') == 'error' else 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        print(len(TASKS), TASKS)
    else:
        if not os.environ.get('SLURM_JOB_ID'):
            sys.exit('Needs a compute allocation')
        main(int(sys.argv[1]))
