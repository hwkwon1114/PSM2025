#!/usr/bin/env python3
"""Run a single test cell with a specified switch_attempts count to measure Newton behavior."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import qualify_cholmod_integration as dep
import run_convergence_benchmark as core
import run_nonconvex_screen as screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gate', type=Path, default=None)
    ap.add_argument('--switch-attempts', type=int, default=1000)
    ap.add_argument('--switch-mode', choices=['fixed', 'adaptive_curvature'], default='fixed')
    ap.add_argument('--gradient-gate', type=float, default=1e-5)
    ap.add_argument('--min-warmup', type=int, default=200)
    ap.add_argument('--check-interval', type=int, default=25)
    ap.add_argument('--start', type=str, default='cylinder_y_plus')
    a = ap.parse_args()
    out = a.out.resolve()
    root = a.root.resolve()
    sw = a.switch_attempts
    start_case = a.start

    total_cap = 1800
    report_reserve = 60
    entry = float(os.environ.get('PROFILED_PHASE_ENTRY', time.time()))
    deadline = entry + total_cap - 30
    fit_deadline = deadline - report_reserve

    signal.signal(signal.SIGTERM, core.stop)
    signal.signal(signal.SIGALRM, core.stop)
    signal.alarm(max(1, int(fit_deadline - time.time())))

    gate_dir = a.gate.resolve() if a.gate else (root / 'run/nonconvex_solver_benchmark/parallel_solver_gate_7723520')
    pg = screen.read(gate_dir / 'gate.json')
    binary = gate_dir / 'convergence_worker'
    runtime = dep.runtime(binary)
    identity_dep = dep.digest(runtime)

    target = screen.read(root / 'run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/gate.json')
    bundle = Path(target['bundle'])
    identity = screen.verify_bundle(bundle)

    base = dict(
        implementation=screen.sha(binary),
        dependency_identity=identity_dep,
        bundle=str(bundle),
        identity=identity,
        hessian_threads=4,
        guarded_hessian=False,
        reuse_symbolic=False
    )

    cell_spec = dict(start=start_case, recipe='hybrid', switch_attempts=sw, switch_mode=a.switch_mode)
    if a.switch_mode == 'adaptive_curvature':
        cell_spec.update(dict(
            gradient_tolerance_gate=a.gradient_gate,
            min_warmup_attempts=a.min_warmup,
            check_interval=a.check_interval
        ))

    protocol = dict(
        cells=[cell_spec],
        request_base=base,
        total_cap_seconds=total_cap,
        phase_entry_epoch=entry,
        report_reserve_seconds=report_reserve,
        finalization_reserve_seconds=30,
        cell_seconds=1200,
        optimizer_seconds=1140,
        worker_timeout=1190,
        max_attempts=100000,
        max_evaluations=200000,
        max_hessians=10000,
        acceptance='gradient_target + independent native <=5e-14 + positive restricted pivots',
        scientific_fits_authorized=True,
        allocated_cpus=4,
        study=f'Test single switch_attempts={sw} for {start_case}'
    )
    screen.atomic(out / 'protocol.json', protocol)
    screen.atomic(out / 'status.json', dict(stage='launch', completed=False))

    def checked(req, record, **kw):
        dep.verify(binary, runtime)
        return core.launch(binary, req, record, screen, **kw)

    ledger = core.suite(out, protocol, screen, checked, allowance=fit_deadline - time.time())
    screen.atomic(out / 'screen/ledger.json', ledger)

    cell = ledger['cells'][0]
    res = cell.get('result', {})
    cp_str = res.get('checkpoint', '')
    shifted = 0
    newton_steps = 0
    if cp_str and Path(cp_str).is_file():
        import json
        cp_data = json.loads(Path(cp_str).read_text())
        trace = cp_data.get('payload', {}).get('state', {}).get('trace', [])
        newton_records = [v for v in trace if 'hessian_oracle_seconds' in v]
        newton_steps = len(newton_records)
        shifted = sum(1 for v in newton_records if v.get('shift', 0.0) > 0.0)

    summary = {
        'start': start_case,
        'switch_attempts': sw,
        'accepted': res.get('accepted', False),
        'charged_seconds': cell.get('charged_seconds', 0),
        'final_energy': res.get('energy'),
        'final_gradient_norm': res.get('gradient_norm'),
        'total_evaluations': res.get('evaluations'),
        'newton_steps': newton_steps,
        'shifted_newton_steps': shifted,
        'completed': ledger['completed']
    }
    screen.atomic(out / 'summary.json', summary)
    print(f'SUMMARY: {summary}')


if __name__ == '__main__':
    main()
