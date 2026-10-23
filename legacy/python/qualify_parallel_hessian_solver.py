#!/usr/bin/env python3
"""Bounded parallel-Newton integration/recovery gate; no convergence fits/retries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import run_convergence_benchmark as core
import run_nonconvex_screen as screen
import qualify_cholmod_integration as dep


def normalize(value):
    # Only operational telemetry/policy encoding differs between compared states.
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()
                if k not in ('elapsed', 'recipeId', 'hessian_threads', 'symbolic_cache_hit', 'symbolic_analyses') and not k.endswith('_seconds')}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--reuse-symbolic', action='store_true', default=False)
    a = ap.parse_args()
    out, root = a.out.resolve(), a.root.resolve()
    entry = time.monotonic()
    signal.signal(signal.SIGTERM, core.stop)
    signal.signal(signal.SIGALRM, core.stop)
    signal.alarm(320)  # Within 360s allocation; reserve finalization, never extend.
    gate = dict(passed=False, qualification_only=True, scientific_fits=0)
    screen.atomic(out/'status.json', dict(stage='build', **gate))
    try:
        frozen = root/'run/nonconvex_solver_benchmark/development_7546843/source'
        binary = core.build(out, frozen, screen, headers=out/'source',
                            translation_unit='QualifyParallelHessian.cpp')
        runtime = dep.runtime(binary)
        dependency = dep.digest(runtime)
        screen.atomic(out/'runtime_manifest.json', dict(identity=dependency, **runtime))
        target = screen.read(root/'run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/gate.json')
        bundle = Path(target['bundle'])
        identity = screen.verify_bundle(bundle)
        assert target['passed'] and identity == target['identity']
        base = dict(implementation=screen.sha(binary), dependency_identity=dependency,
                    bundle=str(bundle), identity=identity, hessian_threads=4,
                    guarded_hessian=False, reuse_symbolic=a.reuse_symbolic)
        screen.atomic(out/'protocol.json', dict(request_base=base, allocation_seconds=360,
            cpus=4, driver_seconds=320, finalization_reserve_seconds=40,
            shell_steps_per_path=2, scientific_fits=0, automatic_retries=False,
            scope='Synthetic handoff + bounded Newton steps; not a convergence trajectory'))
        serial = 0

        def launch(work, *, kind='ConvergenceBenchmark', resume=False, pause='', **extra):
            nonlocal serial
            if time.monotonic()-entry > 255:
                raise TimeoutError('insufficient reserve for next qualification check; no retry')
            dep.verify(binary, runtime)
            assert screen.verify_bundle(bundle) == identity
            work.mkdir(exist_ok=resume)
            req = dict(base, out=str(work), recipe='hybrid', toy=True,
                       resume=resume, pause=pause)
            req.update(extra)
            serial += 1
            return core.launch(binary, req, out/('process_%02d' % serial), screen,
                               timeout=60 if kind=='HybridQualification' else 15,
                               pause=bool(pause), kind=kind)

        controls = out/'controls'
        assert launch(controls, kind='ParallelIntegration')['returncode'] == 0
        assert screen.read(controls/'controls.json')['passed']
        serial_toy, parallel_toy = out/'toy_serial', out/'toy_parallel'
        for work, threads in ((serial_toy, 1), (parallel_toy, 4)):
            assert launch(work, hessian_threads=threads)['returncode'] == 0
        expected = screen.read(parallel_toy/'result.json')
        assert expected['status']=='fixture_stop' and expected['phase']=='newton'
        for file in ('final_x.f64', 'final_g.f64'):
            assert (serial_toy/file).read_bytes() == (parallel_toy/file).read_bytes()
        policy = screen.read(parallel_toy/'journal.json')['policy']
        assert 'hessianThreads' not in policy['lbfgs'] and policy['newton']['hessianThreads']==4
        assert policy['newton'].get('reuseNewtonSymbolic', False) == a.reuse_symbolic
        rows = []
        for point in ('before_target', 'after_target', 'after_commit', 'progress'):
            work = out/('toy_'+point)
            stopped = launch(work, pause=point)
            assert stopped['reason']=='safe_pause'
            journal = screen.read(work/'journal.json')
            source = work/journal['active']
            original = screen.sha(source)
            floor = max(stopped['elapsed'], journal['charged_elapsed'],
                        screen.read(work/'paused.json')['elapsed']) + .001
            wrong = out/('wrong_threads_'+point)
            shutil.copytree(work, wrong)
            assert launch(wrong, resume=True, prior_elapsed=floor, hessian_threads=2)['returncode'] != 0
            assert not (wrong/'completed.json').exists()
            if a.reuse_symbolic:
                wrong_sym = out/('wrong_symbolic_'+point)
                shutil.copytree(work, wrong_sym)
                assert launch(wrong_sym, resume=True, prior_elapsed=floor, reuse_symbolic=False)['returncode'] != 0
                assert not (wrong_sym/'completed.json').exists()
            if point=='progress':
                for label, changes in (('runtime', dict(dependency_identity='wrong')),
                                       ('counter_reset', dict(prior_elapsed=0)),
                                       ('exhausted', dict(prior_elapsed=60))):
                    dest = out/('reject_'+label)
                    shutil.copytree(work, dest)
                    args = dict(prior_elapsed=floor); args.update(changes)
                    assert launch(dest, resume=True, **args)['returncode'] != 0
                    assert not (dest/'completed.json').exists()
            restored = launch(work, resume=True, prior_elapsed=floor)
            assert restored['returncode']==0 and screen.sha(source)==original
            actual = screen.read(work/'result.json')
            for key in ('attempts','iterations','evaluations','hessians','hvps','status','phase'):
                assert expected[key]==actual[key], (point,key)
            for file in ('final_x.f64','final_g.f64'):
                assert (work/file).read_bytes()==(parallel_toy/file).read_bytes()
            assert actual['elapsed']>=floor
            assert launch(work, resume=True, prior_elapsed=floor+restored['elapsed'])['returncode'] != 0
            rows.append(dict(boundary=point, passed=True, source_unchanged=True))
        # Same built executable, real ShellProblem + CHOLMOD steps. These start
        # at a synthetic handoff, not at any completed scientific checkpoint.
        serial_shell, parallel_shell = out/'shell_serial', out/'shell_parallel'
        for work, threads in ((serial_shell,1),(parallel_shell,4)):
            assert launch(work, kind='HybridQualification', shell_fixture=True,
                          shell_steps=2, hessian_threads=threads)['returncode']==0
            parity = screen.read(work/'shell_parity.json')
            assert parity['shell_optimizer_steps']==2 and parity['hessian_threads_actual']==threads
            assert parity['relative_hessian']<=1e-12
        serial_state = screen.read(serial_shell/'final.json')['payload']['state']
        parallel_state = screen.read(parallel_shell/'final.json')['payload']['state']
        assert normalize(serial_state)==normalize(parallel_state), 'serial/parallel numerical state differs'
        resumed = out/'shell_resume'
        stopped = launch(resumed, kind='HybridQualification', shell_fixture=True,
                         shell_steps=2, pause='after_shell_step')
        assert stopped['reason']=='safe_pause'
        before = screen.sha(resumed/'progress.json')
        saved = screen.read(resumed/'progress.json')['payload']['state']
        assert saved['attempts']==2001 and saved['hessians']==1
        floor = max(stopped['elapsed'], saved['elapsed'])+.001
        wrong = out/'shell_wrong_threads';shutil.copytree(resumed,wrong)
        assert launch(wrong, kind='HybridQualification', shell_fixture=True, shell_steps=2,
                      resume=True, charged_elapsed=floor, hessian_threads=2)['returncode'] != 0
        if a.reuse_symbolic:
            wrong_sym_shell = out/'shell_wrong_symbolic';shutil.copytree(resumed,wrong_sym_shell)
            assert launch(wrong_sym_shell, kind='HybridQualification', shell_fixture=True, shell_steps=2,
                          resume=True, charged_elapsed=floor, reuse_symbolic=False)['returncode'] != 0
        assert launch(resumed, kind='HybridQualification', shell_fixture=True, shell_steps=2,
                      resume=True, charged_elapsed=floor)['returncode']==0
        after = screen.read(resumed/'final.json')['payload']['state']
        assert screen.sha(resumed/'progress.json')==before and after['elapsed']>=floor
        assert normalize(after)==normalize(parallel_state), 'full-size disk continuation differs'
        assert launch(resumed, kind='HybridQualification', shell_fixture=True, shell_steps=2,
                      resume=True, charged_elapsed=floor)['returncode'] != 0
        # A runtime that cannot deliver the requested team must fail, not silently
        # run serial while recording a parallel policy.
        previous = os.environ['OMP_THREAD_LIMIT']
        try:
            os.environ['OMP_THREAD_LIMIT']='1'
            denied = out/'team_unavailable'
            assert launch(denied, kind='HybridQualification', shell_fixture=True)['returncode'] != 0
            assert not (denied/'completed.json').exists()
        finally:
            os.environ['OMP_THREAD_LIMIT']=previous
        dep.verify(binary,runtime)
        gate.update(passed=True, worker_sha256=screen.sha(binary), dependency_identity=dependency,
            bundle_identity=identity, threads=4, reuse_symbolic=a.reuse_symbolic, toy_restart_checks=rows,
            full_size_serial_parallel_state_exact=True, full_size_disk_resume_state_exact=True,
            full_size_newton_steps_executed=6, thread_policy_mismatch_rejected=True,
            unavailable_team_rejected=True, cumulative_resource_floor_verified=True,
            completed_cells_refused=True,
            limits=['No full convergence fit or speedup claim.',
                    'Shell source handoff is synthetic, not 2000 live L-BFGS steps.',
                    'SIGKILL tested only after durable boundaries, not arbitrary mid-step.',
                    'Guarded numerical Hessian reuse off; composed with 4-thread parallel and symbolic reuse.',
                    'One full-size positive-cylinder state; no sanitizer/race-detector run.'])
    except BaseException as exc:
        gate['failure']=repr(exc)
        raise
    finally:
        signal.alarm(0)
        gate['elapsed_seconds']=time.monotonic()-entry
        screen.atomic(out/'gate.json',gate)
        screen.atomic(out/'status.json',dict(stage='complete' if gate['passed'] else 'failed_partial',**gate))
    print('PARALLEL_SOLVER_INTEGRATION_AND_RECOVERY_PASSED; NO_CONVERGENCE_FITS')


if __name__=='__main__':
    main()
