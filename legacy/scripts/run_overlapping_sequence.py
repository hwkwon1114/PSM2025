#!/usr/bin/env python3
"""Overlapping perpendicular toolpath investigation runner (sequential and combined continuation)."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_dome_baseline import footprint
from verify_sequence_continuation import verify

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shellsim.solver_settings import solver_options  # noqa: E402
SOURCE = ROOT / 'legacy/configs/dome/overlapping_perpendicular_n18.json'
BINARY = ROOT / 'bin/shell'

SCHEDULES = {
    'a_only': 'A' * 10,
    'b_only': 'B' * 10,
    'ab_block': 'A' * 10 + 'B' * 10,
    'ba_block': 'B' * 10 + 'A' * 10,
    'ab_alternate': 'AB' * 10,
    'ba_alternate': 'BA' * 10,
}

FIXTURE_SCHEDULES = {
    'a_only': 'AA',
    'b_only': 'BB',
    'ab_block': 'AABB',
    'ba_block': 'BBAA',
    'ab_alternate': 'ABAB',
    'ba_alternate': 'BABA',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sequence(name, fixture=False):
    source = json.loads(SOURCE.read_text())
    recipes = copy.deepcopy(source['toolpaths'][:2])
    letters = FIXTURE_SCHEDULES[name] if fixture else SCHEDULES[name]
    if fixture:
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
            raise ValueError(f'Recipe footprint exceeds sheet: {bounds}')
        paths.append(path)
    source['toolpaths'] = paths
    if source.get('boundary_conditions') != {'enabled': False}:
        raise ValueError('Expected free sheet boundary conditions')
    return source


def options(mode, variant='base', fixture=False):
    threads = 1 if fixture else 8
    tolerance = '1e-14' if variant == 'tight' else '5e-14'
    return {
        '-sim': 'bilayer_growth',
        '-case': 'custom',
        '-geometry': 'rectangle',
        '-lx': '.04' if fixture else '.1',
        '-ly': '.05' if fixture else '.15',
        '-res': '.15' if fixture else ('.0075' if variant == 'fine' else '.01'),
        '-h_total': '.0007700000000000001',
        '-growth_type': 'zigzag_sequence',
        '-sequence_solve_mode': mode,
        '-metric_update': 'multiplicative',
        '-sequence_never_clamped': 'false',
        '-sequence_paired_release': 'false',
        '-enable_passE': 'false',
        '-nsteps': '1',
        '-stepwise': 'false',
        '-cycle_write_state': 'false',
        '-export_stl': 'false',
        **solver_options(tolerance=tolerance, threads=threads, fixture=fixture),
    }


def run(out, name='ab_block', mode='final_only', variant='base', fixture=False):
    out = Path(out).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f'Output directory already exists: {out}')
    
    threads = 1 if fixture else 8
    if not fixture:
        if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < threads:
            raise ValueError(f'Requires compute allocation with at least {threads} CPUs')
    
    config = sequence(name, fixture=fixture)
    opts = options(mode, variant=variant, fixture=fixture)
    
    cycles = len(config['toolpaths'])
    expected_cycles = (2 if name in ('a_only', 'b_only') else 4) if fixture else (10 if name in ('a_only', 'b_only') else 20)
    if cycles != expected_cycles:
        raise ValueError(f'Expected {expected_cycles} cycles, got {cycles}')
    
    timeout = (120 if fixture else
               (108000 if variant == 'fine' else 72000) if mode == 'every_cycle' else
               (10800 if variant == 'fine' else 7200))
    
    seq_path = out / 'sequence.json'
    seq_path.write_text(json.dumps(config, indent=2) + '\n')
    
    command = [str(BINARY)] + [v for pair in opts.items() for v in pair] + ['-cycle_file', str(seq_path)]
    
    manifest_path = out / 'execution.json'
    record = {
        'case': name,
        'mode': mode,
        'variant': variant,
        'fixture': fixture,
        'solving_recipe': 'continuation',
        'source': str(SOURCE),
        'source_sha256': sha(SOURCE),
        'binary_sha256': sha(BINARY),
        'sequence_sha256': sha(seq_path),
        'runner_sha256': sha(Path(__file__)),
        'verifier_sha256': sha(ROOT / 'legacy/scripts/verify_sequence_continuation.py'),
        'options': opts,
        'command': command,
        'timeout_seconds': timeout,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local_fixture'),
        'status': 'started'
    }
    manifest_path.write_text(json.dumps(record, indent=2) + '\n')
    
    start_time = time.monotonic()
    try:
        env = dict(os.environ,
                   OMP_NUM_THREADS=str(threads),
                   OMP_THREAD_LIMIT=str(threads),
                   OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1',
                   OMP_DYNAMIC='false')
        with (out / 'solver.log').open('w') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
            
        if sha(BINARY) != record['binary_sha256'] or sha(seq_path) != record['sequence_sha256']:
            raise ValueError('Execution identity changed during run')
            
        result = verify(out, config, opts)
        record['accepted_terminal_proof'] = result['solves'][-1]['native_hybrid']
        record['status'] = 'accepted'
    except Exception as exc:
        record.update(status='failed', error=str(exc), failure_type=type(exc).__name__)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start_time
        manifest_path.write_text(json.dumps(record, indent=2) + '\n')
        
    elapsed = record['elapsed_seconds']
    print(f'Case {name} ({mode}, {variant}) completed successfully in {elapsed:.1f} s.')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(SCHEDULES.keys()), default='ab_block')
    parser.add_argument('--mode', choices=['every_cycle', 'final_only'], default='final_only')
    parser.add_argument('--variant', choices=['base', 'repeat', 'tight', 'fine'], default='base')
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.case, args.mode, args.variant, args.fixture)
