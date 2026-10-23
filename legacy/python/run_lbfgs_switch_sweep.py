#!/usr/bin/env python3
"""L-BFGS switch sweep: evaluate Newton convergence across switch_attempts in [200, 500, 1000, 2000]."""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import qualify_cholmod_integration as dep
import run_convergence_benchmark as core
import run_nonconvex_screen as screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gate', type=Path, default=None)
    a = ap.parse_args()
    out = a.out.resolve()
    root = a.root.resolve()

    total_cap = 2400  # 40 minutes total budget
    report_reserve = 180
    entry = float(os.environ.get('PROFILED_PHASE_ENTRY', time.time()))
    deadline = entry + total_cap - 30
    fit_deadline = deadline - report_reserve

    signal.signal(signal.SIGTERM, core.stop)
    signal.signal(signal.SIGALRM, core.stop)
    signal.alarm(max(1, int(fit_deadline - time.time())))

    gate_dir = a.gate.resolve() if a.gate else (root / 'run/nonconvex_solver_benchmark/parallel_solver_gate_7723520')
    pg = screen.read(gate_dir / 'gate.json')
    assert pg['passed'] and pg['scientific_fits'] == 0
    assert pg['full_size_serial_parallel_state_exact'] and pg['full_size_disk_resume_state_exact']
    assert pg['thread_policy_mismatch_rejected'] and pg['unavailable_team_rejected']
    assert pg['threads'] == 4 and not pg['reuse_symbolic']
    assert (gate_dir / 'complete.txt').exists() and (gate_dir / 'exit_code.txt').read_text().strip() == '0'

    binary = gate_dir / 'convergence_worker'
    assert screen.sha(binary) == pg['worker_sha256']
    runtime = dep.runtime(binary)
    identity_dep = dep.digest(runtime)
    assert identity_dep == pg['dependency_identity']
    screen.atomic(out / 'runtime_manifest.json', dict(identity=identity_dep, **runtime))

    target = screen.read(root / 'run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/gate.json')
    bundle = Path(target['bundle'])
    identity = screen.verify_bundle(bundle)
    assert target['passed'] and identity == target['identity']

    base = dict(
        implementation=screen.sha(binary),
        dependency_identity=identity_dep,
        bundle=str(bundle),
        identity=identity,
        hessian_threads=4,
        guarded_hessian=False,
        reuse_symbolic=False
    )

    # 4 switch settings x 2 starts = 8 cells
    switch_values = [200, 500, 1000, 2000]
    starts = ['cylinder_y_plus', 'cylinder_y_minus']
    cells = []
    for sw in switch_values:
        for s in starts:
            cells.append(dict(start=s, recipe='hybrid', switch_attempts=sw))

    protocol = dict(
        cells=cells,
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
        study='L-BFGS switch attempts sweep: [200, 500, 1000, 2000] with 4 OpenMP threads on Quest'
    )
    screen.atomic(out / 'protocol.json', protocol)
    screen.atomic(out / 'status.json', dict(stage='launch_checks', completed=False, scientific_fits_started=0))

    def checked(req, record, **kw):
        dep.verify(binary, runtime)
        assert screen.verify_bundle(bundle) == identity
        return core.launch(binary, req, record, screen, **kw)

    # Dry run checks
    testroot = out / 'driver_checks'
    testroot.mkdir()
    def fake(req, *_, **kw):
        assert req['recipe'] == 'hybrid' and not req['toy']
        return dict(returncode=0, killed=False, simulated=True)
    ledger_dry = core.suite(testroot, protocol, screen, fake, dry=True)
    assert ledger_dry['completed'] and len(ledger_dry['cells']) == len(cells) and ledger_dry['scientific_fits_started'] == 0

    # Synthetic shell handoff check
    shell = out / 'shell_fixture'
    shell.mkdir()
    assert checked(dict(base, out=str(shell), shell_fixture=True), out / 'shell_process', timeout=35, kind='HybridQualification')['returncode'] == 0
    parity = screen.read(shell / 'shell_parity.json')
    assert parity['relative_hessian'] <= 1e-12 and parity['shell_optimizer_steps'] == 0

    # Toy restart check
    toy = out / 'toy_check'
    toy.mkdir()
    assert checked(dict(base, out=str(toy), recipe='hybrid', toy=True), out / 'toy_process', timeout=15)['returncode'] == 0
    res = screen.read(toy / 'result.json')
    assert res['status'] == 'fixture_stop' and res['phase'] == 'newton'

    # Execute scientific fits
    screen.atomic(out / 'status.json', dict(stage='optimization', completed=False, scientific_fits_started=0))
    ledger = core.suite(out, protocol, screen, checked, allowance=fit_deadline - time.time())
    screen.atomic(out / 'screen/ledger.json', ledger)

    # Generate summary report
    print('ALL SCIENTIFIC CELLS EXECUTED. Generating analysis...')
    summary = []
    for c in ledger['cells']:
        res = c.get('result', {})
        summary.append({
            'index': c['index'],
            'start': c['start'],
            'switch_attempts': c.get('switch_attempts'),
            'status': c.get('status'),
            'accepted': res.get('accepted', False),
            'charged_seconds': c.get('charged_seconds', 0),
            'energy': res.get('energy'),
            'gradient_norm': res.get('gradient_norm'),
            'iterations': res.get('iterations'),
            'evaluations': res.get('evaluations'),
            'hessians': res.get('hessians'),
            'curvature': res.get('curvature')
        })

    screen.atomic(out / 'sweep_summary.json', dict(
        cells=summary,
        total_charged_seconds=sum(r['charged_seconds'] for r in summary),
        completed=ledger['completed']
    ))
    screen.atomic(out / 'status.json', dict(stage='finished', completed=ledger['completed'], accepted=sum(1 for r in summary if r['accepted'])))
    print('L-BFGS SWITCH SWEEP COMPLETE.')


if __name__ == '__main__':
    main()
