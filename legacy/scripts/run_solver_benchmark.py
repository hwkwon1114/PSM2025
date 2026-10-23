#!/usr/bin/env python3
"""Benchmark runner comparing adaptive cutback vs conservative continuation solvers."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'legacy/scripts'))
from prepare_dome_baseline import footprint
from verify_sequence_continuation import verify

BINARY = ROOT / 'bin/shell'
CONFIG_DIR = ROOT / 'legacy/solver_benchmark/configs'
BENCHMARK_CASES_FILE = CONFIG_DIR / 'benchmark_cases.json'
CUTBACK_CONFIG_FILE = ROOT / 'shellsim/configs/default_adaptive_cutback.json'
CONTINUATION_CONFIG_FILE = CONFIG_DIR / 'conservative_continuation.json'
OUTPUT_ROOT = ROOT / 'run/solver_benchmark'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_sequence(case_name, fixture=False):
    with open(BENCHMARK_CASES_FILE) as f:
        cases_meta = {c['name']: c for c in json.load(f)['cases']}
    case_meta = cases_meta[case_name]
    recipe_path = ROOT / case_meta['recipe']
    source = json.loads(recipe_path.read_text())

    recipes = copy.deepcopy(source['toolpaths'][:2])
    letters_needed = case_meta['toolpath_filter']
    cycles = 2 if fixture else case_meta['cycles']

    if len(letters_needed) == 1:
        letters = letters_needed[0] * cycles
    elif len(letters_needed) == 2:
        # 'AB' block: A*(cycles//2) + B*(cycles//2)
        half = cycles // 2
        letters = 'A' * half + 'B' * half
    else:
        raise ValueError(f"Unsupported toolpath filter: {letters_needed}")

    if fixture:
        for recipe in recipes:
            recipe['operation'].update(lv_mm=20, n_strips=4, width_mm=4,
                                       alpha_deg=10, shift_uv_mm=[0, 0],
                                       gtop=2e-5, gbot=1e-5)

    paths = []
    for i, letter in enumerate(letters, 1):
        idx = 'AB'.index(letter)
        path = copy.deepcopy(recipes[idx])
        path.update(id=f'{letter}_{i:02d}', repeat=1, enabled=True)
        bounds = footprint(path['operation'])
        hx, hy = (40, 50) if fixture else (100, 150)
        if bounds[0] < -hx - 1e-9 or bounds[1] > hx + 1e-9 or bounds[2] < -hy - 1e-9 or bounds[3] > hy + 1e-9:
            raise ValueError(f"Recipe footprint exceeds sheet: {bounds}")
        paths.append(path)

    source['toolpaths'] = paths
    if source.get('boundary_conditions') != {'enabled': False}:
        raise ValueError("Expected free sheet boundary conditions")
    return source, case_meta


def get_options(policy_name, fixture=False):
    if policy_name in ('cutback', 'default'):
        config_file = CUTBACK_CONFIG_FILE
    elif policy_name == 'continuation':
        config_file = CONTINUATION_CONFIG_FILE
    elif policy_name == 'baseline':
        raise ValueError("The obsolete wall-clock baseline policy has been removed from the repository. "
                         "Use 'cutback' (default adaptive cutback solver).")
    else:
        raise ValueError(f"Unknown policy name: {policy_name}")
    with open(config_file) as f:
        cfg = json.load(f)['options']

    opts = {
        '-sim': 'bilayer_growth',
        '-case': 'custom',
        '-geometry': 'rectangle',
        '-lx': '.04' if fixture else '.1',
        '-ly': '.05' if fixture else '.15',
        '-res': '.15' if fixture else '.01',
        '-h_total': '.0007700000000000001',
        '-growth_type': 'zigzag_sequence',
        '-sequence_solve_mode': 'final_only',
        '-metric_update': 'multiplicative',
        '-sequence_never_clamped': 'false',
        '-sequence_paired_release': 'false',
        '-enable_passE': 'false',
        '-nsteps': '1',
        '-tol': '3e-12',
        '-stepwise': 'false',
        '-sequence_stability': 'true',
        '-cycle_write_state': 'false',
        '-export_stl': 'false',
    }
    opts.update(cfg)
    if fixture:
        opts['-native_hybrid_warmup_attempts'] = '10'
        opts['-native_hybrid_seconds'] = '30'
        opts['-hessian_threads'] = '1'
    return opts


def run_benchmark_case(case_name, policy_name, fixture=False, force=False):
    target_dir = OUTPUT_ROOT / 'matrix' / f'{case_name}_{policy_name}'
    if fixture:
        target_dir = OUTPUT_ROOT / 'fixtures' / f'{case_name}_{policy_name}'

    target_dir.mkdir(parents=True, exist_ok=True)
    state_file = target_dir / 'bilayer_zigzag_sequence_native_hybrid_state.json'
    proof_file = target_dir / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl'
    exec_file = target_dir / 'execution.json'

    seq_data, case_meta = build_sequence(case_name, fixture=fixture)
    seq_bytes = json.dumps(seq_data, indent=2).encode('utf-8')
    current_seq_sha = hashlib.sha256(seq_bytes).hexdigest()
    current_bin_sha = sha256_file(BINARY)
    opts = get_options(policy_name, fixture=fixture)
    seq_path = target_dir / 'sequence.json'
    run_opts = dict(opts)
    run_opts['-cycle_file'] = str(seq_path)

    # 1. Scientific integrity: verify existing runs completely before reuse
    if not force and exec_file.exists() and state_file.exists() and proof_file.exists() and seq_path.exists():
        try:
            record = json.loads(exec_file.read_text())
            disk_seq_sha = sha256_file(seq_path)
            if (record.get('status') == 'accepted' and
                record.get('exit_code') == 0 and
                record.get('binary_sha256') == current_bin_sha and
                record.get('sequence_sha256') == current_seq_sha and
                disk_seq_sha == current_seq_sha and
                record.get('options') == opts):
                # Verify mathematically
                v_res = verify(target_dir, seq_data, run_opts)
                if v_res.get('status') == 'accepted' or v_res.get('accepted') is not False:
                    print(f"Verified existing completed case: {target_dir.name} (reusing)")
                    return record
                else:
                    print(f"Existing run in {target_dir.name} failed verification. Rerun required.")
        except Exception as e:
            print(f"Existing run in {target_dir.name} corrupted or unparseable: {e}. Rerun required.")

    # 2. Operational safety: handle existing occupied or partial directories
    existing_files = [f for f in target_dir.glob('*') if f.is_file()]
    if existing_files:
        if not force:
            raise RuntimeError(
                f"Target directory {target_dir} contains unverified or partial files ({len(existing_files)} files). "
                f"Pass --force to safely archive existing artifacts to backups/ and rerun."
            )
        # Archive existing failure/partial evidence before cleaning
        backup_dir = OUTPUT_ROOT / 'backups' / f"{target_dir.name}_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Archiving existing artifacts from {target_dir.name} -> {backup_dir.name}")
        for f in existing_files:
            f.rename(backup_dir / f.name)

    seq_path.write_bytes(seq_bytes)

    cmd = [str(BINARY)]
    for k, v in run_opts.items():
        cmd.extend([k, str(v)])

    log_path = target_dir / 'solver.log'
    print(f"Executing: {case_name} | Policy: {policy_name} -> {target_dir.name}")
    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS': '8',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'BLIS_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
    })
    t0 = time.time()
    with open(log_path, 'w') as log_f:
        proc = subprocess.run(cmd, cwd=target_dir, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    t1 = time.time()
    wall_time = t1 - t0

    exec_record = {
        'case_name': case_name,
        'policy_name': policy_name,
        'fixture': fixture,
        'target_dir': str(target_dir),
        'exit_code': proc.returncode,
        'wall_time_seconds': wall_time,
        'binary': str(BINARY),
        'binary_sha256': current_bin_sha,
        'sequence_sha256': current_seq_sha,
        'options': opts,
        'case_meta': case_meta,
        'status': 'failed' if proc.returncode != 0 else 'pending_verification',
    }

    if proc.returncode != 0:
        print(f"FAILED: {target_dir.name} (exit code {proc.returncode})")
        exec_record['status'] = 'failed'
    else:
        try:
            verification = verify(target_dir, seq_data, run_opts)
            exec_record['verification'] = verification
            exec_record['status'] = 'accepted' if (verification.get('status') == 'accepted' or verification.get('accepted') is not False) else 'verification_failed'
        except Exception as exc:
            exec_record['status'] = 'verification_failed'
            exec_record['verification_error'] = str(exc)

    # Write execution record ONLY after verification is attempted
    exec_file.write_text(json.dumps(exec_record, indent=2))

    # 3. Parallel summary isolation: always write isolated per-case summary for both successes and failures
    case_summary_file = target_dir / 'case_summary.json'
    case_summary = {
        'case_name': case_name,
        'policy_name': policy_name,
        'status': exec_record['status'],
        'exit_code': proc.returncode,
        'wall_time_seconds': wall_time,
        'timestamp': time.time(),
    }
    case_summary_file.write_text(json.dumps(case_summary, indent=2))

    status_label = "SUCCESS" if exec_record['status'] == 'accepted' else "TERMINATED"
    print(f"{status_label}: {target_dir.name} (wall time: {wall_time:.1f}s, status: {exec_record['status']})")
    return exec_record


def aggregate_benchmark_records(fixture=False):
    """Aggregate individual per-case execution records without race conditions."""
    base_dir = OUTPUT_ROOT / ('fixtures' if fixture else 'matrix')
    records = []
    if not base_dir.exists():
        return records

    for case_dir in sorted(base_dir.iterdir()):
        exec_file = case_dir / 'execution.json'
        if case_dir.is_dir() and exec_file.exists():
            try:
                rec = json.loads(exec_file.read_text())
                records.append(rec)
            except Exception as e:
                print(f"Notice: skipped incomplete record in {case_dir.name}: {e}")

    summary_file = (OUTPUT_ROOT / 'fixtures' if fixture else OUTPUT_ROOT) / 'benchmark_summary.json'
    summary_file.write_text(json.dumps(records, indent=2))
    print(f"Aggregated {len(records)} benchmark records to {summary_file}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Solver benchmark runner")
    parser.add_argument('--case', choices=['simple_a_1cycle', 'moderate_nonoverlapping_20cycle', 'complex_overlapping_20cycle', 'all'], default='all')
    parser.add_argument('--policy', choices=['cutback', 'continuation', 'both', 'all'], default='cutback',
                        help='Solver policy to run (default: cutback; baseline has been removed)')
    parser.add_argument('--fixture', action='store_true', help="Run small fixture check")
    parser.add_argument('--force', action='store_true', help="Force re-run existing cases")
    parser.add_argument('--aggregate', action='store_true', help="Only aggregate existing case records to summary JSON")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_benchmark_records(fixture=args.fixture)
        return

    cases = ['simple_a_1cycle', 'moderate_nonoverlapping_20cycle', 'complex_overlapping_20cycle'] if args.case == 'all' else [args.case]
    if args.policy in ('all', 'both'):
        policies = ['cutback', 'continuation']
    else:
        policies = [args.policy]

    for c in cases:
        for p in policies:
            run_benchmark_case(c, p, fixture=args.fixture, force=args.force)

    # Only aggregate to shared benchmark_summary.json if running the full suite monolithicly
    # Parallel individual SLURM jobs only write isolated case_summary.json
    if args.case == 'all' and args.policy == 'both':
        aggregate_benchmark_records(fixture=args.fixture)


if __name__ == '__main__':
    main()
