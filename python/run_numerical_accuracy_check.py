#!/usr/bin/env python3
"""Build and execute the numerical gauge accuracy suite on Quest."""
import argparse, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

# Ensure python/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nonconvex_screen as screen
import qualify_cholmod_integration as dep

def build_worker(out: Path, root: Path) -> Path:
    frozen = root / 'run/nonconvex_solver_benchmark/development_7546843/source'
    row = next(r for r in screen.read(frozen / 'build/compile_commands.json') if r['file'].endswith('Test_NonconvexScreen.cpp'))
    args = shlex.split(row['command'])
    flags = []
    i = 0
    while i < len(args):
        if args[i] in ('-c', '-o'):
            i += 2
            continue
        flags.append(args[i])
        i += 1

    # Add test include dirs
    test_diag = root / 'test/diagnostics'
    flags.extend(['-I' + str(test_diag)])

    obj = out / 'numerical_accuracy.o'
    binary = out / 'numerical_accuracy_worker'
    source = test_diag / 'NumericalGaugeAccuracyTest.cpp'

    cmd = flags + ['-c', str(source), '-o', str(obj)]
    screen.atomic(out / 'compile_command.json', cmd)
    print(f"[Build] Compiling {source} -> {obj}...")
    t0 = time.monotonic()
    subprocess.run(cmd, check=True, timeout=380)
    print(f"[Build] Compiled in {time.monotonic() - t0:.2f}s")

    link = shlex.split(subprocess.check_output(['ninja', '-C', str(frozen / 'build'), '-t', 'commands', 'testshell'], text=True).splitlines()[-1])[2:-2]
    args = []
    i = 0
    while i < len(link):
        t = link[i]
        if t == '-o':
            args += ['-o', str(binary)]
            i += 2
            continue
        if t.endswith('.cpp.o') or t.startswith('-Wl,--dependency-file='):
            i += 1
            continue
        args.append(str(frozen / 'build' / t) if t.startswith('lib/') else t)
        i += 1
    args.insert(1, str(obj))
    screen.atomic(out / 'link_command.json', args)
    print(f"[Build] Linking {binary}...")
    t0 = time.monotonic()
    subprocess.run(args, check=True, timeout=60)
    print(f"[Build] Linked in {time.monotonic() - t0:.2f}s")
    return binary

def print_report(report):
    print("\n" + "="*80)
    print("NUMERICAL ACCURACY TEST RESULTS")
    print("="*80)
    print(f"Handoff State Energy:        {report['handoff_energy']:.12e}")
    print(f"Handoff Gradient Norm:       {report['handoff_gradient_norm']:.12e}")

    single = report['single_step_linear_solve']
    comp = single['comparison']
    qr = single['gauge_qr']
    hint = single['gauge_hint']

    print("\n--- 1. SINGLE-STEP RESTRICTED SOLVE & GAUGE INVARIANCE ---")
    print(f"Gauge QR Condition Number:   {qr['condition_number']:.4f}")
    print(f"Gauge Hint Condition Number: {hint['condition_number']:.4f}")
    print(f"CHOLMOD Linear Residual (QR):   {qr['linear_residual']:.4e}")
    print(f"CHOLMOD Linear Residual (Hint): {hint['linear_residual']:.4e}")
    print(f"Positive Definite (QR):     {qr['positive_definite']}")
    print(f"Positive Definite (Hint):   {hint['positive_definite']}")

    print(f"\nRaw Step Norm Diff ||dx_qr - dx_hint||: {comp['raw_diff_norm']:.6e}")
    print(f"  Rigid Component of Diff:              {comp['rigid_component_norm']:.6e} (pure gauge shift)")
    print(f"  Non-Rigid Component of Diff:          {comp['non_rigid_component_norm']:.6e} (strain-producing)")
    print(f"Relative Physical Displacement Diff:    {comp['relative_physical_step_diff']:.6e}")

    print(f"\nPredicted Energy Drop (QR):   {qr['predicted_energy_drop']:.12e}")
    print(f"Predicted Energy Drop (Hint): {hint['predicted_energy_drop']:.12e}")
    print(f"Rel. Diff in Predicted Drop:  {comp['relative_predicted_energy_diff']:.6e}")

    print(f"\nActual Step Energy (QR):      {qr['actual_step_energy']:.12e}")
    print(f"Actual Step Energy (Hint):    {hint['actual_step_energy']:.12e}")
    print(f"Rel. Diff in Step Energy:     {comp['relative_actual_step_energy_diff']:.6e}")
    print(f"Actual Step ||g|| Diff:       {comp['step_gradient_norm_diff']:.6e}")

    print("\n--- 2. MULTI-STEP TRAJECTORY CONVERGENCE & SHAPE PARITY ---")
    traj = report['trajectories']
    for name in ['qr', 'hint', 'guarded']:
        t = traj[name]
        print(f"Trajectory '{name}':")
        print(f"  Steps:                 {t['steps_taken']}")
        print(f"  Final Energy:          {t['final_energy']:.12e}")
        print(f"  Final Gradient Norm:   {t['final_gradient_norm']:.12e}")
        print(f"  Status:                {t['final_status']}")
        if 'rms_vertex_diff_vs_qr' in t:
            print(f"  Procrustes RMS vs QR:  {t['rms_vertex_diff_vs_qr']:.6e}")
            print(f"  Procrustes Max vs QR:  {t['max_vertex_diff_vs_qr']:.6e}")
            print(f"  Rel. Energy Diff vs QR:{t['relative_energy_diff_vs_qr']:.6e}")
        print()

    if "intrinsic_invariance" in report:
        inv = report["intrinsic_invariance"]
        print("--- 3. GAUGE-INVARIANT INTRINSIC STRAIN & METRIC PARITY (Hint vs QR) ---")
        print(f"Edge Length RMS Diff: {inv['edge_length_rms_diff']:.6e} m")
        print(f"Edge Length Max Diff: {inv['edge_length_max_diff']:.6e} m")
        print(f"Face Area RMS Diff:   {inv['face_area_rms_diff']:.6e} m^2")
        print(f"Face Area Max Diff:   {inv['face_area_max_diff']:.6e} m^2")
        print()

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Run numerical gauge accuracy diagnostics.")
    parser.add_argument('--out', type=Path, default=Path('run/nonconvex_solver_benchmark/numerical_gauge_accuracy'))
    parser.add_argument('--max-steps', type=int, default=15)
    parser.add_argument('--report-only', action='store_true', help='Only print report from existing results')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = args.out.resolve()

    if args.report_only:
        report_file = out / 'numerical_accuracy_report.json'
        if not report_file.exists():
            raise FileNotFoundError(f"Report file not found: {report_file}")
        with open(report_file) as f:
            data = json.load(f)
        print_report(data)
        return

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Build binary
    binary = build_worker(out, root)

    # 2. Prepare bundle and input state
    bundle = root / 'run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/bundle'
    identity = screen.verify_bundle(bundle)
    assert identity == '9f79ed808b6dd6ccca171f5c72538837fe6400653f0f6b4632dcc407e5e259f0'

    pilot = root / 'run/nonconvex_solver_benchmark/profiled_zigzag_pilot_7629263'
    cell = screen.read(pilot / 'screen/ledger.json')['cells'][0]
    handoff_cp = Path(cell['out']) / 'switch_target.json'

    # Extract handoff vector
    raw = screen.read(handoff_cp)
    x = np.array(raw['payload']['state']['x'], dtype='<f8')
    assert len(x) == 32267 and np.isfinite(x).all()
    handoff_file = out / 'handoff_x.f64'
    x.tofile(handoff_file)

    report_file = out / 'numerical_accuracy_report.json'
    request = {
        "bundle": str(bundle),
        "identity": identity,
        "handoff": str(handoff_file),
        "out": str(report_file),
        "max_steps": args.max_steps
    }
    request_file = out / 'request.json'
    screen.atomic(request_file, request)

    # 3. Launch worker
    print(f"[Run] Executing numerical accuracy tests via {binary}...")
    env = dict(os.environ, NUMERICAL_ACCURACY_REQUEST=str(request_file), OMP_NUM_THREADS="4")
    log_file = out / 'run.log'
    t0 = time.monotonic()
    with log_file.open('w') as log:
        proc = subprocess.run([str(binary), '--gtest_filter=NumericalGaugeAccuracy.Dispatch'],
                              env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    elapsed = time.monotonic() - t0
    print(f"[Run] Completed with returncode {proc.returncode} in {elapsed:.2f}s")

    if proc.returncode != 0:
        print("[Error] Worker failed! Check log:")
        with log_file.open('r') as log:
            print(log.read()[-2000:])
        sys.exit(proc.returncode)

    # 4. Parse and display findings
    report = screen.read(report_file)
    print_report(report)

if __name__ == '__main__':
    main()
