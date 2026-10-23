#!/usr/bin/env python3
"""One authorized continuation-assisted ab_block case, not a one-solve baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run_sequence_order import ROOT, BINARY, sha, sequence, options
from verify_sequence_continuation import verify

SOURCE = ROOT / 'run/sequence_order/ab_block_final_only_base'


def continuation_options(original):
    options = dict(original)
    options.update({'-sequence_adaptive': 'true', '-sequence_initial_step': '.1',
                    '-sequence_max_step': '.1', '-sequence_min_step': '.015625',
                    '-sequence_step_growth': '2', '-sequence_max_retries': '8',
                    '-native_hybrid_seconds': '1800'})
    return options


def run(out, name='ab_block', mode='final_only', variant='base', fixture=False):
    prior = json.loads((SOURCE / 'execution.json').read_text())
    if (prior['case'], prior['mode'], prior['variant']) != ('ab_block', 'final_only', 'base'):
        raise ValueError('Wrong source case')
    if sha(BINARY) != prior['binary_sha256'] or sha(SOURCE / 'sequence.json') != prior['sequence_sha256']:
        raise ValueError('Changed source identity')
    threads = 1 if fixture else 8
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < threads:
        raise ValueError('Requires sufficient compute CPUs')
    if name not in ('ab_block', 'ba_block', 'ab_alternate', 'ba_alternate'):
        raise ValueError('Mixed cases only')
    if variant != 'base' and name != 'ab_block':
        raise ValueError('Sensitivity is preselected ab_block only')
    opts = continuation_options(options(mode, variant, fixture))
    if fixture:
        opts['-native_hybrid_seconds'] = '20'
    timeout = (120 if fixture else
               (108000 if variant == 'fine' else 72000) if mode == 'every_cycle' else
               (10800 if variant == 'fine' else 7200))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'sequence.json'
    path.write_text(json.dumps(sequence(name, fixture), indent=2) + '\n')
    config = json.loads(path.read_text())
    cycles = sum(p['repeat'] for p in config['toolpaths'] if p.get('enabled', True))
    if cycles != (4 if fixture else 20):
        raise ValueError('Wrong physical application count')
    command = [str(BINARY)] + [v for pair in opts.items() for v in pair] + ['-cycle_file', str(path)]
    record = dict(case=name, mode=mode, variant=variant, fixture=fixture,
                  solving_recipe='continuation',
                  source=str(SOURCE), source_execution_sha256=sha(SOURCE / 'execution.json'),
                  binary_sha256=sha(BINARY), sequence_sha256=sha(path),
                  runner_sha256=sha(Path(__file__)), verifier_sha256=sha(ROOT / 'legacy/scripts/verify_sequence_continuation.py'),
                  options=opts, command=command, timeout_seconds=timeout,
                  slurm_job_id=os.environ['SLURM_JOB_ID'], status='started')
    manifest = out / 'execution.json'
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    start = time.monotonic()
    try:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='false')
        with (out / 'solver.log').open('w') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
        if sha(BINARY) != record['binary_sha256'] or sha(path) != record['sequence_sha256']:
            raise ValueError('Execution identity changed')
        result = verify(out, config, opts)
        record['accepted_terminal_proof'] = result['solves'][-1]['native_hybrid']
        record['status'] = 'accepted'
    except Exception as exc:
        record.update(status='failed', error=str(exc), failure_type=type(exc).__name__)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-start
        manifest.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['ab_block','ba_block','ab_alternate','ba_alternate'], default='ab_block')
    parser.add_argument('--mode', choices=['every_cycle','final_only'], default='final_only')
    parser.add_argument('--variant', choices=['base','repeat','tight','fine'], default='base')
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    run(args.output, args.case, args.mode, args.variant, args.fixture)
