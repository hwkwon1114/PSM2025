#!/usr/bin/env python3
"""Researcher-authorized ab_block lumped restart; preserve the frozen runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run_sequence_order import ROOT, BINARY, sha, verify


def retry(out):
    old = ROOT / 'run/sequence_order/ab_block_final_only_base'
    prior = json.loads((old / 'execution.json').read_text())
    proof = json.loads((old / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').read_text())
    assert (prior['case'], prior['mode'], prior['variant'], prior['status']) == ('ab_block', 'final_only', 'base', 'failed')
    assert proof['termination'] == 'time_cap' and proof['accepted'] is False
    assert sha(BINARY) == prior['binary_sha256']
    assert sha(old / 'sequence.json') == prior['sequence_sha256']
    assert os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) >= 8
    opts = dict(prior['options'])
    assert opts['-native_hybrid_seconds'] == '600'
    opts['-native_hybrid_seconds'] = '1800'
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'sequence.json'
    path.write_bytes((old / 'sequence.json').read_bytes())
    command = [str(BINARY)] + [v for pair in opts.items() for v in pair] + ['-cycle_file', str(path)]
    record = dict(prior, options=opts, command=command, runner_sha256=sha(Path(__file__)),
                  parent_runner_sha256=sha(ROOT / 'legacy/scripts/run_sequence_order.py'),
                  retry_of=str(old), retry_of_execution_sha256=sha(old / 'execution.json'),
                  slurm_job_id=os.environ['SLURM_JOB_ID'], timeout_seconds=2070, status='started')
    for key in ('error', 'failure_type', 'elapsed_seconds'):
        record.pop(key, None)
    manifest = out / 'execution.json'
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    start = time.monotonic()
    try:
        env = dict(os.environ, OMP_NUM_THREADS='8', OMP_THREAD_LIMIT='8',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_DYNAMIC='false')
        with (out / 'solver.log').open('w') as log:
            subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=2070, check=True)
        if sha(BINARY) != record['binary_sha256'] or sha(path) != record['sequence_sha256']:
            raise ValueError('Execution identity changed')
        verify(out, json.loads(path.read_text()), opts)
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
    retry(parser.parse_args().output)
