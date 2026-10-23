#!/usr/bin/env python3
"""One two-gripper or matched never-clamped case; no automatic retries.

--case fixture is a two-repeat coarse correctness test (120 s process ceiling).
Production needs a Slurm allocation and a recorded independent review before use.
This runner is not itself an authorization/review gate.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shellsim.solver_settings import solver_options  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, choices=('run6', 'run9', 'fixture'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--never-clamped', action='store_true',
                   help='No physical clamps during forming; no release branches')
    p.add_argument('--final-only', action='store_true',
                   help='With --never-clamped: accumulate all growth before one equilibrium')
    p.add_argument('--stop-after', type=int, default=0, help='Fixture interruption only')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Run on a compute node in a Slurm allocation')
    fixture = args.case == 'fixture'
    if args.final_only and (not args.never_clamped or args.stop_after):
        p.error('--final-only requires --never-clamped and forbids --stop-after')
    if args.stop_after and (not fixture or args.stop_after != 1):
        p.error('Only fixture stop-after=1 is supported')
    threads = 1 if fixture else 8
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) < threads:
        p.error(f'Need at least {threads} allocated CPUs')
    source = ROOT / 'legacy/configs/boundary_conditions' / (
        ('run6' if fixture else args.case) + '_two_gripper_cycle.json')
    config = json.loads(source.read_text())
    if fixture:
        tp = config['toolpaths'][0]
        tp['repeat'] = 2
        tp['operation'].update(lv_mm=20, n_strips=4, width_mm=2,
                               alpha_deg=10, shift_uv_mm=[0, 0],
                               gtop=2e-5, gbot=1e-5)
        for region, x in zip(config['boundary_conditions']['regions'], (-10, 10)):
            region['center_uv_mm'] = [x, 28]
            region['size_uv_mm'] = [10, 4]
    if args.never_clamped:
        config['boundary_conditions'] = {'enabled': False}
    binary = ROOT / 'bin/shell'
    options = {
        '-sim': 'bilayer_growth', '-case': 'custom', '-geometry': 'rectangle',
        '-lx': '.02' if fixture else '.1', '-ly': '.03' if fixture else '.15',
        '-res': '.2' if fixture else '.01', '-h_total': '.0007700000000000001',
        '-growth_type': 'zigzag_sequence', '-sequence_solve_mode': 'every_cycle',
        '-sequence_paired_release': 'true', '-enable_passE': 'false', '-nsteps': '1',
        '-stepwise': 'false', '-cycle_write_state': 'false', '-export_stl': 'false',
        **solver_options(threads=threads, fixture=fixture),
    }
    if args.never_clamped:
        options['-sequence_paired_release'] = 'false'
        options['-sequence_never_clamped'] = 'true'
    if args.final_only:
        options['-sequence_solve_mode'] = 'final_only'
        options['-sequence_adaptive'] = 'false'
    identity = {'binary_sha256': sha(binary), 'runner_sha256': sha(Path(__file__)),
                'sequence': config, 'options': options}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume:
        resume = args.resume.resolve(strict=True)
        saved = json.loads(resume.read_text())['payload']
        if saved['identity']['runner_identity'] != digest:
            p.error('Resume build/recipe identity mismatch')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cycle = out / 'sequence.json'
    cycle.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(binary)]
    for key, value in options.items():
        command += [key, value]
    command += ['-cycle_file', str(cycle), '-sequence_checkpoint_identity', digest]
    if args.resume:
        command += ['-sequence_resume', str(resume)]
    if args.stop_after:
        command += ['-sequence_stop_after', str(args.stop_after)]
    manifest = {'identity': identity, 'identity_sha256': digest, 'command': command,
                'source_sequence': str(source), 'source_sha256': sha(source),
                'slurm_job_id': os.environ['SLURM_JOB_ID'],
                'resume': str(resume) if args.resume else None,
                'process_timeout_seconds': 120 if fixture else (1800 if args.final_only else 6 * 3600),
                'status': 'started'}
    (out / 'execution.json').write_text(json.dumps(manifest, indent=2) + '\n')
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
               OMP_DYNAMIC='false', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        with (out / 'solver.log').open('w') as log:
            result = subprocess.run(command, cwd=out, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=manifest['process_timeout_seconds'])
        if result.returncode:
            raise RuntimeError(f'Solver failed ({result.returncode}); see {out / "solver.log"}')
        if sha(binary) != identity['binary_sha256']:
            raise RuntimeError('Binary changed during execution')
        if args.stop_after:
            if not (out / 'accepted_cycle_001.json').is_file():
                raise RuntimeError('Fixture did not publish the expected checkpoint')
            manifest['status'] = 'fixture_stopped_after_accepted_repeat'
        elif args.never_clamped:
            result = json.loads((out / 'never_clamped_result.json').read_text())['payload']
            endpoint = json.loads((out / 'never_clamped_final_checkpoint.json').read_text())['payload']
            proof = endpoint['proof']
            expected_cycles = config['toolpaths'][0]['repeat']
            if not (result['complete'] is True and result['cycles'] == endpoint['cycle'] == expected_cycles and
                    result['remaining_rigid_modes'] == 6 and result['fixed_vertex_dofs'] == 0 and
                    not any(any(row) for row in endpoint['identity']['vertex_mask']) and
                    not any(any(row) for row in endpoint['identity']['edge_mask']) and
                    result['proof'] == proof and proof['accepted'] and
                    proof['termination'] == 'gradient_target' and proof['curvature'] == 'positive_pivots' and
                    0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= 5e-14):
                raise RuntimeError('Incomplete, constrained or unaccepted never-clamped endpoint')
            if args.final_only:
                expected_solves = 0 if args.resume else 1
                checkpoints = sorted(out.glob('accepted_cycle_*.json'))
                expected_checkpoints = [] if args.resume else [out / f'accepted_cycle_{expected_cycles:03d}.json']
                with (out / 'sequence_convergence.csv').open() as stream:
                    solves = [row for row in csv.DictReader(stream) if row['minimization_performed'] == '1']
                if not (result['solve_mode'] == 'final_only' and
                        result['equilibrium_solves'] == len(solves) == expected_solves and
                        checkpoints == expected_checkpoints and
                        all(int(row['executed_cycle_index']) == expected_cycles and
                            float(row['lambda_trial']) == 1 and row['equilibrium_accepted'] == '1'
                            for row in solves)):
                    raise RuntimeError('Final-only cadence or full-load acceptance mismatch')
            if result['final_vtp'] != 'never_clamped_final.vtp' or not (out / result['final_vtp']).is_file():
                raise RuntimeError('Never-clamped final VTP missing')
            if (out / 'paired_release_result.json').exists():
                raise RuntimeError('Never-clamped control unexpectedly performed releases')
            manifest['status'] = 'completed'
        else:
            pair = json.loads((out / 'paired_release_result.json').read_text())['payload']
            if pair['complete'] is not True or len(pair['endpoints']) != 2:
                raise RuntimeError('Incomplete paired release')
            if [e['branch'] for e in pair['endpoints']] != ['putong', 'free']:
                raise RuntimeError('Missing release branch')
            if len({e['start_state_checksum'] for e in pair['endpoints']}) != 1:
                raise RuntimeError('Branch start states differ')
            for e in pair['endpoints']:
                proof = e['proof']
                if not (proof['accepted'] and proof['termination'] == 'gradient_target' and
                        proof['curvature'] == 'positive_pivots' and
                        0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= 5e-14 and
                        e['fixed_coordinates_unchanged'] and e['target_history_unchanged']):
                    raise RuntimeError('Failed endpoint acceptance')
            manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - start
        (out / 'execution.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out, manifest['status'], flush=True)


if __name__ == '__main__':
    main()
