#!/usr/bin/env python3
"""Fixed two-toolpath cadence investigation; no optimization/search or automatic retry."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_dome_baseline import footprint
from run_dome import ROOT, sha

SOURCE = ROOT / 'legacy/configs/dome/baseline_attempt209.json'
BINARY = ROOT / 'run/native_shape_refinement/build_4375711/bin/shell'
SCHEDULES = {
    'a_only': 'A' * 10, 'b_only': 'B' * 10,
    'ab_block': 'A' * 10 + 'B' * 10, 'ba_block': 'B' * 10 + 'A' * 10,
    'ab_alternate': 'AB' * 10, 'ba_alternate': 'BA' * 10,
}


def sequence(name, fixture=False):
    source = json.loads(SOURCE.read_text())
    recipes = copy.deepcopy(source['toolpaths'][:2])
    letters = SCHEDULES[name]
    if fixture:
        letters = {'a_only': 'AA', 'b_only': 'BB', 'ab_block': 'AABB',
                   'ba_block': 'BBAA', 'ab_alternate': 'ABAB', 'ba_alternate': 'BABA'}[name]
        for recipe in recipes:
            recipe['operation'].update(lv_mm=20, n_strips=4, width_mm=4,
                                       alpha_deg=10, shift_uv_mm=[0, 0],
                                       gtop=2e-5, gbot=1e-5)
    paths = []
    for i, letter in enumerate(letters, 1):
        path = copy.deepcopy(recipes['AB'.index(letter)])
        path.update(id=f'{letter}_{i:02d}', repeat=1, enabled=True)
        bounds = footprint(path['operation'])
        hx, hy = (40, 50) if fixture else (100, 150)
        if bounds[0] < -hx-1e-9 or bounds[1] > hx+1e-9 or bounds[2] < -hy-1e-9 or bounds[3] > hy+1e-9:
            raise ValueError('Recipe footprint exceeds sheet')
        paths.append(path)
    source['toolpaths'] = paths
    if source['boundary_conditions'] != {'enabled': False}:
        raise ValueError('Expected free sheet')
    return source


def options(mode, variant='base', fixture=False):
    threads = 1 if fixture else 8
    tolerance = '1e-14' if variant == 'tight' else '5e-14'
    return {
        '-sim': 'bilayer_growth', '-case': 'custom', '-geometry': 'rectangle',
        '-lx': '.04' if fixture else '.1', '-ly': '.05' if fixture else '.15',
        '-res': '.15' if fixture else ('.0075' if variant == 'fine' else '.01'),
        '-h_total': '.0007700000000000001', '-growth_type': 'zigzag_sequence',
        '-sequence_solve_mode': mode, '-sequence_adaptive': 'false',
        '-sequence_warm_start': 'true', '-sequence_minimize_every': '1',
        '-sequence_slope_feedback': '0', '-metric_update': 'multiplicative',
        '-sequence_never_clamped': 'false', '-sequence_paired_release': 'false',
        '-enable_passE': 'false', '-nsteps': '1', '-tol': '3e-12',
        '-stepwise': 'false', '-equilibrium_solver': 'native_hybrid',
        '-native_hybrid_grad_tol': tolerance, '-equilibrium_grad_tol': tolerance,
        '-native_hybrid_refine_linear_solve': 'true',
        '-native_hybrid_warmup_attempts': '5' if fixture else '2000',
        '-native_hybrid_seconds': '20' if fixture else '600',
        '-native_hybrid_handoff': 'fixed', '-native_hybrid_export_state': 'true',
        '-seed_escape': 'false', '-hessian_threads': str(threads),
        '-cycle_write_state': 'false', '-export_stl': 'false',
    }


def accepted(proof, tolerance):
    return (proof.get('accepted') is True and proof.get('termination') == 'gradient_target'
            and proof.get('curvature') == 'positive_pivots'
            and 0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= tolerance)


def verify(out, config, opts):
    cycles = len(config['toolpaths'])
    mode = opts['-sequence_solve_mode']
    expected = list(range(1, cycles+1)) if mode == 'every_cycle' else [cycles]
    tolerance = float(opts['-native_hybrid_grad_tol'])
    result = json.loads((out / 'sequence_result.json').read_text())
    solves = result['solves']
    if not (result['completed'] is True and result['cycles'] == cycles
            and result['equilibrium_solves'] == len(expected) and result['solve_mode'] == mode
            and [row['cycle'] for row in solves] == expected
            and all(accepted(row['native_hybrid'], tolerance) for row in solves)):
        raise ValueError('Unaccepted equilibrium or wrong solve cadence')
    proofs = [json.loads(line) for line in
              (out / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').read_text().splitlines() if line.strip()]
    if proofs != [row['native_hybrid'] for row in solves]:
        raise ValueError('Extra/missing native solves or inconsistent proofs')
    with (out / 'sequence_convergence.csv').open() as stream:
        convergence = list(csv.DictReader(stream))
    if ([int(row['executed_cycle_index']) for row in convergence] != expected or
            any(not (row['equilibrium_accepted'] == '1' and
                     float(row['lambda_from']) == 0 and float(row['lambda_trial']) == 1)
                for row in convergence)):
        raise ValueError('Expected exactly one accepted full-load solve per stage')
    with (out / 'sequence_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    if ([int(row['cycle']) for row in history] != list(range(1, cycles+1)) or
            [int(row['solved']) for row in history] != [int(i in expected) for i in range(1, cycles+1)] or
            any(int(row['hit_events']) <= 0 for row in history)):
        raise ValueError('Application history/cadence mismatch')
    state = json.loads((out / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    if state['proof'] != proofs[-1] or not (out / result['final_mesh']).is_file():
        raise ValueError('Endpoint artifacts disagree')
    with (out / 'bilayer_zigzag_sequence_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    if ([int(row['executed_cycle_index']) for row in summary] != expected or
            [row['toolpath_id'] for row in summary] != [config['toolpaths'][i-1]['id'] for i in expected] or
            any(not (out / row['final_file']).is_file() for row in summary)):
        raise ValueError('Missing intermediate equilibrium or wrong application order')
    return result


def run(name, mode, variant, fixture, out):
    if variant != 'base' and (name != 'ab_block' or fixture):
        raise ValueError('Sensitivity checks are preselected ab_block only')
    config, opts = sequence(name, fixture), options(mode, variant, fixture)
    threads = int(opts['-hessian_threads'])
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) < threads:
        raise ValueError('Requires compute allocation with sufficient CPUs')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'sequence.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(BINARY)] + [v for pair in opts.items() for v in pair] + ['-cycle_file', str(path)]
    stages = len(config['toolpaths']) if mode == 'every_cycle' else 1
    timeout = 120 if fixture else 750 * stages + 120
    record = {'case': name, 'mode': mode, 'variant': variant, 'fixture': fixture,
              'source': str(SOURCE), 'source_sha256': sha(SOURCE), 'sequence_sha256': sha(path),
              'binary_sha256': sha(BINARY), 'runner_sha256': sha(Path(__file__)),
              'options': opts, 'command': command, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'timeout_seconds': timeout, 'status': 'started'}
    manifest = out / 'execution.json'
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    try:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='false')
        with (out / 'solver.log').open('w') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
        if sha(BINARY) != record['binary_sha256'] or sha(path) != record['sequence_sha256']:
            raise ValueError('Execution identity changed')
        verify(out, config, opts)
        record['status'] = 'accepted'
    except Exception as exc:
        record.update(status='failed', error=str(exc), failure_type=type(exc).__name__)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        manifest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=SCHEDULES, required=True)
    parser.add_argument('--mode', choices=['every_cycle', 'final_only'], required=True)
    parser.add_argument('--variant', choices=['base', 'repeat', 'fine', 'tight'], default='base')
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.mode, args.variant, args.fixture, args.output)
