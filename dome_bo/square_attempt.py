#!/usr/bin/env python3
"""One isolated square-sheet attempt: solve the reserved recipe, verify the endpoint, score the shape.

Uses the adaptive-cutback solver configuration (``shellsim/configs/default_adaptive_cutback.json``) and the
endpoint verification of ``dome_bo/run_dome_continuation.py`` (history-compatible with cutbacks) on a square sheet. Solver failure is a recorded outcome (``status.json``), not an
exception for the campaign; a configuration/artifact/analysis/infrastructure problem is state 'error' (exit 1).

The scientific acceptance measurement is the primary gate of ``dome_continuous_objective`` on the central
160 x 160 mm region; ``j_combined`` is only the search surrogate.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('.', 'dome_bo'):
    sys.path.insert(0, str(ROOT / sub))

from run_dome_continuation import verify                    # noqa: E402  (reviewed endpoint verification)
from dome_curvature import geometric_curvature               # noqa: E402
import square_hatch as sh                                    # noqa: E402
import dome_continuous_objective as dco                      # noqa: E402

# Researcher decision (this goal, m04331): the new adaptive-cutback solver, frozen as a private read-only copy
# so that a later rebuild of bin/shell cannot change the campaign.
BINARY = ROOT / 'run/dome_square/frozen/shell'
SOLVER_CONFIG = ROOT / 'shellsim/configs/default_adaptive_cutback.json'
ROI_HALF_M = 0.08
SCREEN_MESH = ('.1', '.1', '.02')        # half-extents lx, ly (m) and relative resolution -> 4 mm edge, 5000 faces
VERIFY_MESH = ('.1', '.1', '.01')        # 2 mm edge, 20000 faces
SOLVER_TIMEOUT = 3600                    # frozen: solver process ceiling (s)
ANALYSIS_TIMEOUT = 600                   # frozen: analysis ceiling (s)
# A solver outcome that is a genuine numerical non-acceptance (a training label for the feasibility model).
# Everything else that goes wrong (config/artifact/analysis/infrastructure) is state 'error' and stops the run.
NUMERICAL_MARKER = 'equilibrium corrector failed'
NUMERICAL_VERIFY_MESSAGES = ('Unaccepted native endpoint', 'Partial-load endpoint cannot be scored')


class AnalysisTimeout(RuntimeError):
    pass


def _alarm(signum, frame):
    raise AnalysisTimeout(f'analysis exceeded {ANALYSIS_TIMEOUT} s')


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_state(state: dict, radius: float = 0.015, roi_half: float = ROI_HALF_M) -> dict:
    """Primary gate + search objective on the central ROI, plus whole-sheet diagnostics (SI inputs)."""
    positions = np.asarray(state['positions'], float)
    uv = np.asarray(state['material_uv'], float)
    faces = np.asarray(state['faces'], int)
    values, _, valid = geometric_curvature(positions, uv, faces, radius=radius)
    obj = dco.evaluate_dome_objective(values, valid, uv, faces, roi_half=(roi_half, roi_half))
    area, _ = dco.material_areas(uv, faces)
    finite = np.isfinite(values).all(axis=1) & valid
    safe = np.where(finite[:, None], values, 0.0)
    opposite = finite & (safe[:, 0] * safe[:, 1] < 0) & (np.min(np.abs(safe), axis=1) >= dco.FLOOR)
    obj.update(whole_valid_fraction=float(area[finite].sum() / area.sum()),
               whole_opposite_sign_fraction=float(area[opposite].sum() / area.sum()),
               max_abs_curvature=float(np.max(np.abs(safe))) if finite.any() else float('nan'),
               n_faces=int(len(faces)), curvature_radius_m=radius, roi_half_m=roi_half)
    return obj


def solver_options(mesh, tolerance, threads, fixture):
    """Fixed physics/geometry options plus the adaptive-cutback solver configuration (default_adaptive_cutback.json:
    lambda=1 first, deterministic 40-Hessian cap, Newton stall/indefinite cutback, wall clock = emergency watchdog)."""
    lx, ly, res = mesh
    options = {'-sim': 'bilayer_growth', '-case': 'custom', '-geometry': 'rectangle', '-lx': lx, '-ly': ly,
               '-res': res, '-h_total': '.0007700000000000001', '-growth_type': 'zigzag_sequence',
               '-sequence_solve_mode': 'final_only', '-metric_update': 'multiplicative',
               '-sequence_never_clamped': 'false', '-sequence_paired_release': 'false', '-enable_passE': 'false',
               '-nsteps': '1', '-tol': '3e-12', '-stepwise': 'false', '-cycle_write_state': 'false',
               '-export_stl': 'false'}
    options.update(json.loads(SOLVER_CONFIG.read_text())['options'])
    options.update({'-native_hybrid_grad_tol': tolerance, '-equilibrium_grad_tol': tolerance,
                    '-hessian_threads': str(threads)})
    if fixture:
        options.update({'-native_hybrid_warmup_attempts': '5', '-native_hybrid_seconds': '20'})
    return options


def run_attempt(attempt_dir: Path, mesh=SCREEN_MESH, tolerance='5e-14', threads=8, fixture=False,
                binary: Path = BINARY, timeout=None) -> dict:
    """State is 'accepted' (verified, scored), 'failed' (genuine numerical non-acceptance or declared solver
    timeout: a feasibility label, no objective) or 'error' (configuration, artifact, analysis or
    infrastructure problem: never a training label; the campaign stops for the researcher)."""
    attempt_dir = Path(attempt_dir).resolve()                  # absolute: the solver runs in another cwd
    request = json.loads((attempt_dir / 'request.json').read_text())
    recipe = request['recipe']
    out = attempt_dir / 'solve'
    out.mkdir(parents=True, exist_ok=False)
    if sh.recipe_sha(recipe) != request['recipe_sha256']:
        status = {'attempt': request['attempt'], 'recipe_sha256': request['recipe_sha256'], 'state': 'error',
                  'failure_type': 'RecipeDigestMismatch', 'error': 'request recipe differs from its reserved digest'}
        (attempt_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        return status
    (out / 'sequence.json').write_text(json.dumps(recipe, indent=2) + '\n')
    cycles = sum(p['repeat'] for p in recipe['toolpaths'] if p.get('enabled', True))
    options = solver_options(mesh, tolerance, threads, fixture)
    cycle_file = (out / 'sequence.json').resolve()
    command = [str(binary)] + [v for kv in options.items() for v in kv] + ['-cycle_file', str(cycle_file)]
    status = {'attempt': request['attempt'], 'recipe_sha256': request['recipe_sha256'],
              'binary_sha256': sha(binary), 'sequence_sha256': sha(cycle_file),
              'mesh': list(mesh), 'tolerance': tolerance, 'command': command,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'state': 'started'}
    if status['sequence_sha256'] != hashlib.sha256(json.dumps(recipe, indent=2).encode() + b'\n').hexdigest():
        raise RuntimeError('written sequence differs from reserved recipe')
    (attempt_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', OMP_DYNAMIC='false')
    start = time.monotonic()
    try:
        try:
            with (out / 'solver.log').open('w') as log:
                subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=timeout or (120 if fixture else SOLVER_TIMEOUT), check=True)
        except subprocess.TimeoutExpired as exc:
            status.update(state='failed', failure_type='SolverTimeout', error=str(exc)[:300])
        except subprocess.CalledProcessError as exc:
            numerical = NUMERICAL_MARKER in (out / 'solver.log').read_text(errors='replace')
            status.update(state='failed' if numerical else 'error', failure_type=(
                'EquilibriumNotReached' if numerical else 'SolverProcessError'),
                error=f'exit code {exc.returncode}')
        else:
            if sha(binary) != status['binary_sha256']:
                raise RuntimeError('binary changed during execution')
            try:
                verify(out, cycles, gradient_ceiling=float(tolerance))
                status['state'] = 'accepted'
            except ValueError as exc:
                numerical = str(exc) in NUMERICAL_VERIFY_MESSAGES
                status.update(state='failed' if numerical else 'error', failure_type=(
                    'EquilibriumNotReached' if numerical else 'ArtifactIntegrity'), error=str(exc)[:300])
        status['elapsed_seconds'] = time.monotonic() - start
        if status['state'] == 'accepted':
            state_file = out / 'bilayer_zigzag_sequence_native_hybrid_state.json'
            state = json.loads(state_file.read_text())
            status['state_sha256'] = sha(state_file)
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(ANALYSIS_TIMEOUT)
            try:
                metrics = analyze_state(state)
            finally:
                signal.alarm(0)
            if not (np.isfinite(metrics['j_combined']) and np.isfinite(metrics['largest_connected_fraction'])):
                raise ValueError('nonfinite objective for an accepted equilibrium')
            status['metrics'] = metrics
    except Exception as exc:                                    # configuration/artifact/analysis/infrastructure
        status.pop('metrics', None)
        status.update(state='error', failure_type=type(exc).__name__, error=str(exc)[:500])
        status['elapsed_seconds'] = time.monotonic() - start
    (attempt_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-dir', required=True, type=Path)
    parser.add_argument('--verify', action='store_true', help='fine mesh and 1e-14 tolerance')
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--mesh', nargs=3, metavar=('LX', 'LY', 'RES'))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Needs a compute allocation')
    mesh = tuple(args.mesh) if args.mesh else (VERIFY_MESH if args.verify else SCREEN_MESH)
    status = run_attempt(args.attempt_dir, mesh, '1e-14' if args.verify else '5e-14', args.threads, args.fixture)
    print(json.dumps({k: status.get(k) for k in ('attempt', 'state', 'failure_type', 'elapsed_seconds')}))
    sys.exit(1 if status.get('state') == 'error' else 0)


if __name__ == '__main__':
    main()
