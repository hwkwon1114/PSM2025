"""Run the shell solver on a toolpath sequence with the shared adaptive-cutback settings, then read the result."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .solver_settings import solver_options
from .toolpath import save

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'bin/shell'


def run(sequence, out_dir, lx=0.1, ly=0.1, res=0.02, h_total=0.00077, tolerance='1e-14', threads=4,
        timeout=3600, binary=BINARY):
    """Solve `sequence` (dict from toolpath.sequence, or a JSON path) on a rectangular sheet spanning +-lx, +-ly [m]
    (0.1 gives the 200 mm sheet); `res` sets the mesh (0.02 is a 4 mm mesh on it, 0.01 a 2 mm mesh); one equilibrium after the whole sequence. Returns the accepted state dict
    (`positions`, `reference_positions`, `faces`, `material_uv`, `proof`). Raises if the solver does not
    reach the gradient tolerance."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cycle_file = save(sequence, out / 'sequence.json') if isinstance(sequence, dict) else Path(sequence).resolve()
    options = {'-sim': 'bilayer_growth', '-case': 'custom', '-geometry': 'rectangle', '-lx': lx, '-ly': ly,
               '-res': res, '-h_total': h_total, '-growth_type': 'zigzag_sequence',
               '-sequence_solve_mode': 'final_only', '-metric_update': 'multiplicative', '-enable_passE': 'false',
               '-nsteps': '1', '-stepwise': 'false', '-export_stl': 'false', '-cycle_write_state': 'false'}
    options.update(solver_options(tolerance=tolerance, threads=threads))
    command = [str(binary)] + [str(v) for kv in options.items() for v in kv] + ['-cycle_file', str(cycle_file)]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1')
    with (out / 'solver.log').open('w') as log:
        subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    state = json.loads((out / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    proof = state['proof']
    if not (proof['accepted'] and proof['termination'] == 'gradient_target'):
        raise RuntimeError(f"solver did not reach the gradient target: {proof['termination']}")
    return state


def height(state):
    """(x, y, z) in metres of the deformed sheet vertices, z relative to the mean height."""
    p = np.asarray(state['positions'], float)
    return p[:, 0], p[:, 1], p[:, 2] - p[:, 2].mean()
