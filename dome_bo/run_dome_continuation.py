#!/usr/bin/env python3
"""One isolated three-path final-only run. Production requires independent review."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shellsim.solver_settings import solver_options  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(out, cycles, gradient_ceiling=5e-14):
    if not 0 < gradient_ceiling <= 5e-14:
        raise ValueError('Invalid verification gradient ceiling')
    result = json.loads((out/'sequence_result.json').read_text())
    if not (result['completed'] is True and result['cycles'] == cycles and
            result['equilibrium_solves'] == 1 and result['solve_mode'] == 'final_only' and
            len(result['solves']) == 1 and result['solves'][0]['cycle'] == cycles):
        raise ValueError('Incomplete final-only endpoint')
    proof = result['solves'][0]['native_hybrid']
    if not (proof['accepted'] is True and proof['termination'] == 'gradient_target' and
            proof['curvature'] == 'positive_pivots' and
            0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= gradient_ceiling):
        raise ValueError('Unaccepted native endpoint')
    state = json.loads((out/'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    if state['proof'] != proof or not (out/result['final_mesh']).is_file():
        raise ValueError('Endpoint artifacts disagree')
    with (out/'sequence_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    if ([int(row['cycle']) for row in history] != list(range(1, cycles+1)) or
            [int(row['solved']) for row in history] != [0]*(cycles-1)+[1] or
            any(int(row['hit_events']) <= 0 for row in history)):
        raise ValueError('Incomplete accumulation history or wrong solve cadence')
    with (out/'sequence_convergence.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    proofs = [json.loads(line) for line in (out/'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(proofs) or proofs[-1] != proof:
        raise ValueError('Missing or inconsistent continuation proof history')
    accepted_load = 0.0
    for row, subproof in zip(rows, proofs):
        trial = float(row['lambda_trial'])
        if (int(row['executed_cycle_index']) != cycles or
                float(row['lambda_from']) != accepted_load or
                not accepted_load < trial <= 1.0 or
                bool(int(row['equilibrium_accepted'])) != subproof['accepted']):
            raise ValueError('Invalid continuation load/proof history')
        if subproof['accepted']:
            if not (subproof['termination']=='gradient_target' and subproof['curvature']=='positive_pivots' and
                    0 <= subproof['gradient_norm'] <= subproof['gradient_tolerance'] <= gradient_ceiling):
                raise ValueError('Unqualified accepted substep')
            accepted_load = trial
    if accepted_load != 1.0 or float(rows[-1]['lambda_trial']) != 1.0:
        raise ValueError('Partial-load endpoint cannot be scored')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    threads = 1 if args.fixture else 8
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) < threads:
        parser.error('Needs a compute allocation with sufficient CPUs')
    source = args.sequence.resolve(strict=True)
    config = json.loads(source.read_text())
    if len(config['toolpaths']) != 3 or config.get('boundary_conditions') != {'enabled': False}:
        parser.error('Requires three recipes with disabled boundary conditions')
    if args.fixture:
        # Coarse correctness-only case, not a dome candidate or calibration evidence.
        for i, path in enumerate(config['toolpaths']):
            path['repeat'] = 1
            path['operation'].update(lv_mm=20, n_strips=4, width_mm=4,
                                     alpha_deg=10, shift_uv_mm=[0, (i-1)*15],
                                     gtop=2e-5, gbot=1e-5)
    cycles = sum(p['repeat'] for p in config['toolpaths'] if p.get('enabled', True))
    options = {'-sim':'bilayer_growth', '-case':'custom', '-geometry':'rectangle',
               '-lx':'.04' if args.fixture else '.1', '-ly':'.05' if args.fixture else '.15',
               '-res':'.15' if args.fixture else '.01', '-h_total':'.0007700000000000001',
               '-growth_type':'zigzag_sequence', '-sequence_solve_mode':'final_only',
               '-metric_update':'multiplicative', '-sequence_never_clamped':'false',
               '-sequence_paired_release':'false', '-enable_passE':'false', '-nsteps':'1',
               '-stepwise':'false', '-cycle_write_state':'false', '-export_stl':'false',
               **solver_options(threads=threads, fixture=args.fixture)}
    binary = ROOT/'bin/shell'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cycle_file = out/'sequence.json'
    cycle_file.write_text(json.dumps(config, indent=2)+'\n')
    command = [str(binary)] + [v for pair in options.items() for v in pair] + ['-cycle_file', str(cycle_file)]
    manifest = {'command':command, 'source':str(source), 'source_sha256':sha(source),
                'sequence_sha256':sha(cycle_file), 'binary_sha256':sha(binary),
                'runner_sha256':sha(Path(__file__)), 'fixture':args.fixture,
                'slurm_job_id':os.environ['SLURM_JOB_ID'], 'status':'started'}
    record = out/'execution.json'
    record.write_text(json.dumps(manifest, indent=2)+'\n')
    start = time.monotonic()
    try:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='false')
        with (out/'solver.log').open('w') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=120 if args.fixture else 3600, check=True)
        if sha(binary) != manifest['binary_sha256']:
            raise ValueError('Binary changed during execution')
        verify(out, cycles)
        manifest['status'] = 'accepted'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc), failure_type=type(exc).__name__)
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-start
        record.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
