"""Compare BLIS settings for the same frozen Run 2 flat-start forward solve.

No scan scoring, calibration, automatic retry, or overwriting of earlier results.
Prepare once without a solver; run the two arms only after independent review.
"""
import ctypes
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis import run2_initial_state_diagnostic as original
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.runner import (SolverProcessFailure, check_output, command, launch,
                                      sequence_json, validate_mesh_topology)

OUT = ROOT / 'run/diagnostics/run2_blis_a0084_r1'
BATCH = ROOT / 'calibration/bo/run2_blis_diagnostic.sbatch'
TEST = ROOT / 'calibration/tests/test_run2_blis_diagnostic.py'
PREVIOUS_HISTORY = (original.OUT / 'flat/evaluations/attempt_0001/case_00/sequence_history.csv')
LIBRARIES = ('libblis.so.4', 'libgomp.so.1', 'libopenblas.so.0', 'libtbb.so.12')
THREAD_ENV = ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_MAX_ACTIVE_LEVELS', 'OMP_NESTED',
              'BLIS_NUM_THREADS', 'BLIS_JC_NT', 'BLIS_PC_NT', 'BLIS_IC_NT', 'BLIS_JR_NT', 'BLIS_IR_NT',
              'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_LIBRARY_PATH', 'LD_PRELOAD')


def libraries():
    """Record the actual loader resolution, not just nominal library names."""
    text = subprocess.check_output(['ldd', str(ROOT / 'bin/shell')], text=True)
    result = {}
    for name in LIBRARIES:
        match = re.search(r'^\s*' + re.escape(name) + r'\s+=>\s+(\S+)', text, re.M)
        if not match or not Path(match[1]).is_file():
            raise RuntimeError(f'Missing linked library: {name}')
        path = Path(match[1]).resolve()
        result[name] = {'path': str(path), 'sha256': digest(path)}
    return result


def make_plan(out=OUT):
    for path in (original.BEST, original.FLAT_PREVIOUS / original.INITIAL_NAME, ROOT / 'bin/shell'):
        if digest(path) != original.EXPECTED[path]:
            raise ValueError(f'Frozen input identity mismatch: {path}')
    vals = original.frozen_parameters()
    cfg = original.make_configs(out, vals)[0]['flat']
    cfg['output_dir'], cfg['report_dir'] = str(out / 'forward'), str(out / 'forward/report')
    p4 = {k: float(vals[k]) for k in cfg['parameters']}
    sequence = sequence_json(cfg, cfg['cases'][0], p4)
    if sequence != json.loads(original.SEQUENCE.read_text()):
        raise ValueError('Physical sequence is not the original a0084 input')
    files = {Path(__file__).resolve(), BATCH, TEST, Path(original.__file__).resolve(),
             original.BEST, original.SUMMARY, original.SEQUENCE,
             original.FLAT_PREVIOUS / original.INITIAL_NAME, PREVIOUS_HISTORY,
             ROOT / 'bin/shell', ROOT / 'calibration/bo/run_convention_corrected.py',
             ROOT / 'shellsim/configs/default_adaptive_cutback.json',
             ROOT / 'run/run9_transfer_expanded/evaluations/attempt_0209/request.json',
             ROOT / 'run/putong_convention_check/preserved_samples/run9.npz',
             ROOT / 'run/putong_convention_check/preserved_samples/run9.json'}
    files.update((ROOT / 'calibration/ewcal').glob('*.py'))
    return {'purpose': 'numerical BLIS-setting diagnostic; no scan scoring or parameter selection',
            'python': {'executable': sys.executable, 'version': sys.version,
                       'numpy': np.__version__, 'psutil': psutil.__version__},
            'frozen_parameters': vals, 'config': cfg, 'sequence': sequence,
            'command_template': command(cfg, cfg['cases'][0], out / 'ARM/sequence.json'),
            'input_sha256': {str(p): digest(p) for p in sorted(files)}, 'libraries': libraries()}


def prepare(out=OUT):
    plan = make_plan(out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'plan.json').open('x') as f:
        json.dump(plan, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(f'Prepared {out / "plan.json"}; no solver invoked.', flush=True)


def verify_plan(out=OUT):
    saved = json.loads((out / 'plan.json').read_text())
    reconstructed = make_plan(out)
    if saved != reconstructed:
        raise ValueError('Reviewed configuration, inputs, code or linked libraries changed')
    return reconstructed


def runtime_environment(arm, linked):
    if arm not in ('unset', 'one'):
        raise ValueError('Expected arm unset or one')
    expected = {'OMP_NUM_THREADS': '8', 'OMP_DYNAMIC': 'FALSE',
                'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected common thread environment')
    if os.environ.get('BLIS_NUM_THREADS') != (None if arm == 'unset' else '1'):
        raise ValueError('BLIS setting does not match requested arm')
    if any(os.environ.get(k) is not None for k in THREAD_ENV[5:10]):
        raise ValueError('Per-loop BLIS settings must be unset')
    blis = ctypes.CDLL(linked['libblis.so.4']['path'])
    blis.bli_init()
    threads = int(blis.bli_thread_get_num_threads())
    if threads != (8 if arm == 'unset' else 1):
        raise ValueError(f'BLIS reports unexpected parent-process thread setting: {threads}')
    common = {k: v for k, v in os.environ.items() if k != 'BLIS_NUM_THREADS'}
    common_sha = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
    return {'thread_env': {k: os.environ.get(k) for k in THREAD_ENV},
            'common_environment_sha256': common_sha, 'hostname': socket.gethostname(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'blis_configured_threads_parent_process': threads,
            'blis_threading_enabled': bool(blis.bli_info_get_enable_threading()),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_step_id': os.environ.get('SLURM_STEP_ID')}


def monitor_solver(stop, stats):
    """Sample OS thread count and RSS; thread count is not simultaneous active work."""
    parent = psutil.Process()
    while not stop.wait(2):
        try:
            for proc in parent.children(recursive=True):
                if proc.name() == 'shell':
                    stats['samples'] += 1
                    stats['max_os_threads'] = max(stats['max_os_threads'], proc.num_threads())
                    stats['max_rss_bytes'] = max(stats['max_rss_bytes'], proc.memory_info().rss)
                    cpu = proc.cpu_times()
                    stats['last_cpu_seconds'] = max(stats['last_cpu_seconds'], cpu.user + cpu.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def failure_class(exc, folder):
    evidence = exc.evidence
    if evidence.get('timeout') or evidence.get('reason') in ('controller_elapsed_cap', 'native_time_cap_abort'):
        return 'timeout', evidence
    record = {'valid': False, 'failure_class': 'infrastructure', 'failure_phase': 'solver',
              'solver_termination': evidence}
    return original.diagnostic_failure(record, folder)


def run_arm(arm, out=OUT):
    plan = verify_plan(out)
    environment = runtime_environment(arm, plan['libraries'])
    if arm == 'one':
        first = json.loads((out / 'unset/result.json').read_text())
        for key in ('common_environment_sha256', 'hostname', 'cpu_affinity', 'slurm_job_id', 'slurm_step_id'):
            if first['environment'][key] != environment[key]:
                raise ValueError(f'Pair is not matched: {key}')
    folder = out / arm
    folder.mkdir(exist_ok=False)
    cfg = plan['config']
    case = cfg['cases'][0]
    seq = folder / 'sequence.json'
    save_json(seq, plan['sequence'])
    args = command(cfg, case, seq)
    expected = list(plan['command_template'])
    expected[expected.index(str(out / 'ARM/sequence.json'))] = str(seq)
    if args != expected:
        raise ValueError('Effective command differs from reviewed command')
    save_json(folder / 'command.json', args)
    result = {'arm': arm, 'environment': environment, 'valid': False, 'failure_class': None}
    stats = {'samples': 0, 'max_os_threads': 0, 'max_rss_bytes': 0, 'last_cpu_seconds': 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=monitor_solver, args=(stop, stats), daemon=True)
    started = time.monotonic()
    sampler.start()
    try:
        launch(args, folder, cfg['solver']['timeout_seconds'],
               cfg.get('monitoring', {}).get('heartbeat_seconds', 30),
               solver_environment=os.environ.copy())
    except SolverProcessFailure as exc:
        result['failure_class'], result['failure_evidence'] = failure_class(exc, folder)
        result['error'] = str(exc)
    except Exception as exc:
        result['failure_class'] = 'infrastructure'
        result['error'] = str(exc)
    finally:
        stop.set()
        sampler.join()
        result['solver_seconds'] = time.monotonic() - started
        result['process_samples'] = stats
    try:
        if result['failure_class'] and not (folder / original.INITIAL_NAME).is_file():
            result['start_check'] = None
            result['start_check_unavailable'] = 'Solver did not export an initial mesh'
        else:
            result['start_check'] = original.verify_saved_start('flat', folder)
        history = folder / 'sequence_history.csv'
        result['history_matches_previous'] = history.is_file() and history.read_bytes() == PREVIOUS_HISTORY.read_bytes()
        if result['failure_class'] is None:
            manifest, mesh = check_output(cfg, case, folder)
            state = json.loads((folder / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
            reference = np.asarray(state['reference_positions'], dtype=float)
            positions = np.asarray(state['positions'], dtype=float)
            if (state.get('units') not in ('metres', 'm') or reference.shape != positions.shape or
                    reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(positions).all() or
                    not np.isfinite(reference).all()):
                raise ValueError('Invalid accepted state arrays')
            validate_mesh_topology(mesh, np.asarray(state['faces']))
            if not result['history_matches_previous']:
                raise ValueError('Accepted solution has a different 33-cycle history')
            result.update(valid=True, energy=manifest['total_energy'],
                          proof=manifest['solves'][0]['native_hybrid'])
    except Exception as exc:
        result['failure_class'] = 'infrastructure'
        result['output_check_error'] = str(exc)
        if not result.get('error'):
            result['error'] = f'Output/identity check failed: {exc}'
    save_json(folder / 'result.json', result)
    print(f"{arm}: valid={result['valid']} failure={result['failure_class']} solver={result['solver_seconds']:.1f}s", flush=True)
    if result['failure_class'] not in (None, 'timeout', 'solver'):
        raise RuntimeError(result.get('error', 'Infrastructure failure'))


def summarize(out=OUT):
    verify_plan(out)
    results = {arm: json.loads((out / arm / 'result.json').read_text()) for arm in ('unset', 'one')}
    report = {'arms': results, 'terminal_load_identity': None, 'final_shape_comparison': None}
    if all(results[arm]['valid'] for arm in results):
        states = {arm: json.loads((out / arm / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
                  for arm in results}
        keys = ('reference_positions', 'material_uv', 'faces', 'edges', 'abar_top', 'abar_bottom', 'bbar')
        same = {key: bool(np.array_equal(states['unset'][key], states['one'][key])) for key in keys}
        same['material'] = states['unset']['material'] == states['one']['material']
        report['terminal_load_identity'] = same
        if not all(same.values()):
            raise ValueError('Terminal load identity differs; cannot compare equilibria')
        difference = 1000 * (np.asarray(states['unset']['positions']) - np.asarray(states['one']['positions']))
        norm = np.linalg.norm(difference, axis=1)
        report['final_shape_comparison'] = {'frame': 'native frame, corresponding vertices',
                                            'vertex_rms_mm': float(np.sqrt(np.mean(norm**2))),
                                            'vertex_p95_mm': float(np.percentile(norm, 95)),
                                            'vertex_max_mm': float(np.max(norm))}
    save_json(out / 'summary.json', report)
    print(f'Comparison saved: {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif len(sys.argv) == 3 and sys.argv[1] == 'run' and sys.argv[2] in ('unset', 'one'):
        run_arm(sys.argv[2])
    elif sys.argv[1:] == ['summarize']:
        summarize()
    else:
        raise SystemExit('Usage: run2_blis_diagnostic.py prepare | run unset|one | summarize')
