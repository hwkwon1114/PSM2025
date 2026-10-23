#!/usr/bin/env python3
"""Two authorized hybrid cells; bounded checks + fits + reporting; no retries."""
import argparse, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import run_nonconvex_screen as screen
import qualify_cholmod_integration as dep
import run_convergence_benchmark as core


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--guarded',action='store_true');modes.add_argument('--parallel',action='store_true');modes.add_argument('--composed',action='store_true')
    a=ap.parse_args();out=a.out.resolve();root=a.root.resolve()
    total_cap=3000 if a.guarded else 2700;report_reserve=210 if a.guarded else (180 if (a.parallel or a.composed) else 150)
    entry=float(os.environ['PROFILED_PHASE_ENTRY']);deadline=entry+total_cap-30;fit_deadline=deadline-report_reserve
    signal.signal(signal.SIGTERM,core.stop);signal.signal(signal.SIGALRM,core.stop)
    signal.alarm(max(1,int(fit_deadline-time.time())))
    prep=root/'run/nonconvex_solver_benchmark/convergence_prepare_7592200'
    target=root/'run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619'
    gate=screen.read(prep/'gate.json');tg=screen.read(target/'gate.json')
    screen.atomic(out/'status.json',dict(stage='launch_checks',completed=False,scientific_fits_started=0))
    ledger=None;failure=None;report_status=None
    try:
        assert gate['passed'] and gate['scientific_fits']==0 and tg['passed'] and tg['qualification_only']
        binary=prep/'convergence_worker';assert screen.sha(binary)==gate['worker_sha256']
        assert gate['worker_sha256']=='abb52ac5be1f502bc245862b2d7f5f096ad9ff9f6ae631931aaa80634d41f04b'
        for path,h in gate['source_hashes'].items():assert screen.sha(path)==h,path
        old_runtime=dep.runtime(binary);assert dep.digest(old_runtime)==gate['dependency_identity']
        if a.guarded:
            qualified=root/'run/nonconvex_solver_benchmark/guarded_hessian_gate_7637624'
            assert screen.read(qualified/'guarded_gate.json')['passed'] and screen.read(qualified/'completion.json')['passed']
            for line in (qualified/'source_sha256.txt').read_text().splitlines():
                h,path=line.split('  ',1);assert screen.sha(path)==h,path
            # The qualified numerical engine/adapter are copied verbatim. Only
            # handoff policy + request plumbing are new integration code.
            for name in ('NonconvexBenchmark.hpp','BenchmarkNewtonFactorization.hpp'):
                assert screen.sha(out/'source/test/diagnostics'/name)==screen.sha(qualified/'source/test/diagnostics'/name)
            binary=core.build(out,root/'run/nonconvex_solver_benchmark/development_7546843/source',screen,headers=out/'source')
        if a.parallel:
            qualified=root/'run/nonconvex_solver_benchmark/parallel_solver_gate_7651813'
            pg=screen.read(qualified/'gate.json');assert pg['passed'] and pg['scientific_fits']==0
            assert pg['full_size_serial_parallel_state_exact'] and pg['full_size_disk_resume_state_exact']
            assert pg['thread_policy_mismatch_rejected'] and pg['unavailable_team_rejected']
            assert len(pg['toy_restart_checks'])==4 and all(x['passed'] for x in pg['toy_restart_checks'])
            assert (qualified/'complete.txt').exists() and (qualified/'exit_code.txt').read_text().strip()=='0'
            for line in (qualified/'source.sha256').read_text().splitlines():
                h,path=line.split('  ',1);assert screen.sha(path)==h,path
            binary=qualified/'convergence_worker';assert screen.sha(binary)==pg['worker_sha256']
            assert pg['worker_sha256']=='b668179d815515ed68ff30bfc5c8fe526f32ea7de8a2c5bcac06841dbc82a393'
            assert int(os.environ['SLURM_CPUS_PER_TASK'])==4 and len(os.sched_getaffinity(0))>=4
            assert os.environ['OMP_THREAD_LIMIT']=='4' and os.environ['OMP_NUM_THREADS']=='1'
            assert screen.sha(out/'run_convergence_benchmark.py')==screen.sha(qualified/'run_convergence_benchmark.py')
            assert dep.digest(dep.runtime(binary))==pg['dependency_identity']
        if a.composed:
            gate_path = os.environ.get('COMPOSED_QUALIFIED_GATE')
            qualified = Path(gate_path) if gate_path else (root/'run/nonconvex_solver_benchmark/composed_solver_gate')
            pg=screen.read(qualified/'gate.json');assert pg['passed'] and pg['scientific_fits']==0
            assert pg['full_size_serial_parallel_state_exact'] and pg['full_size_disk_resume_state_exact']
            assert pg['thread_policy_mismatch_rejected'] and pg['unavailable_team_rejected']
            assert pg['threads']==4 and pg['reuse_symbolic']
            assert len(pg['toy_restart_checks'])==4 and all(x['passed'] for x in pg['toy_restart_checks'])
            assert (qualified/'complete.txt').exists() and (qualified/'exit_code.txt').read_text().strip()=='0'
            for line in (qualified/'source.sha256').read_text().splitlines():
                h,path=line.split('  ',1);assert screen.sha(path)==h,path
            binary=qualified/'convergence_worker';assert screen.sha(binary)==pg['worker_sha256']
            assert int(os.environ['SLURM_CPUS_PER_TASK'])==4 and len(os.sched_getaffinity(0))>=4
            assert os.environ['OMP_THREAD_LIMIT']=='4' and os.environ['OMP_NUM_THREADS']=='1'
            assert screen.sha(out/'run_convergence_benchmark.py')==screen.sha(qualified/'run_convergence_benchmark.py')
            assert dep.digest(dep.runtime(binary))==pg['dependency_identity']
        runtime=dep.runtime(binary);identity_dep=dep.digest(runtime)
        assert runtime['libraries']==old_runtime['libraries'] and runtime['loader_environment']==old_runtime['loader_environment']
        screen.atomic(out/'runtime_manifest.json',dict(identity=identity_dep,**runtime))
        bundle=Path(tg['bundle']);identity=screen.verify_bundle(bundle);assert identity==tg['identity']
        assert identity=='9f79ed808b6dd6ccca171f5c72538837fe6400653f0f6b4632dcc407e5e259f0'
        base=dict(implementation=screen.sha(binary),dependency_identity=identity_dep,bundle=str(bundle),identity=identity)
        if a.guarded:base.update(guarded_hessian=True,reuse_symbolic=False)
        if a.parallel:base.update(hessian_threads=4,guarded_hessian=False,reuse_symbolic=False)
        if a.composed:base.update(hessian_threads=4,guarded_hessian=False,reuse_symbolic=True)
        protocol=dict(cells=[dict(start=s,recipe='hybrid') for s in ('cylinder_y_plus','cylinder_y_minus')],request_base=base,
            total_cap_seconds=total_cap,phase_entry_epoch=entry,report_reserve_seconds=report_reserve,finalization_reserve_seconds=30,
            cell_seconds=1200,optimizer_seconds=1140,worker_timeout=1190,max_attempts=100000,max_evaluations=200000,max_hessians=10000,
            switch_attempts=2000,acceptance='gradient_target + independent native <=5e-14 + positive restricted pivots',
            target='all eight passes accumulated before optimization',no_automatic_retry=True,
            qualification=str(target),worker_preparation=str(prep),scientific_fits_authorized=True,
            interpretation='two dependent starts of one fixed objective; no general method ranking')
        if a.guarded:protocol.update(baseline=str(root/'run/nonconvex_solver_benchmark/profiled_zigzag_pilot_7629263'),guarded_qualification=str(qualified),symbolic_reuse=False)
        if a.parallel:protocol.update(baseline=str(root/'run/nonconvex_solver_benchmark/profiled_zigzag_pilot_7629263'),
            parallel_qualification=str(qualified),symbolic_reuse=False,guarded_hessian_reuse=False,
            allocated_cpus=4,max_allocated_core_seconds=4*total_cap,
            comparison='Historical serial baseline preserved; descriptive timing, not same-job randomized comparison',
            resource_accounting='Four CPUs reserved throughout, including serial L-BFGS/checkpoints/reporting')
        if a.composed:protocol.update(baseline=str(root/'run/nonconvex_solver_benchmark/profiled_zigzag_pilot_7629263'),
            composed_qualification=str(qualified),symbolic_reuse=True,guarded_hessian_reuse=False,
            allocated_cpus=4,max_allocated_core_seconds=4*total_cap,
            comparison='Historical serial baseline preserved; descriptive timing, not same-job randomized comparison',
            resource_accounting='Four CPUs reserved throughout, including serial L-BFGS/checkpoints/reporting')
        screen.atomic(out/'protocol.json',protocol)
        def checked(req,record,**kw):
            dep.verify(binary,runtime);assert screen.verify_bundle(bundle)==identity
            return core.launch(binary,req,record,screen,**kw)
        def fake(req,*_,**kw):
            assert req['recipe']=='hybrid' and not req['toy']
            return dict(returncode=0,killed=False,simulated=True)
        testroot=out/'driver_checks';testroot.mkdir()
        ledger_dry=core.suite(testroot,protocol,screen,fake,dry=True)
        assert ledger_dry['completed'] and len(ledger_dry['cells'])==2 and ledger_dry['scientific_fits_started']==0
        try:core.suite(testroot,protocol,screen,fake,dry=True)
        except FileExistsError:pass
        else:raise AssertionError('completed suite reopened')
        budgetroot=testroot/'budget';budgetroot.mkdir()
        def forbidden(*args,**kw):raise AssertionError('budget-exhausted dispatch')
        skipped=core.suite(budgetroot,protocol,screen,forbidden,allowance=1199)
        assert skipped['scientific_fits_started']==0 and all(c['status']=='global_budget_unstarted' for c in skipped['cells'])
        badroot=testroot/'failed_worker';badroot.mkdir()
        failed=core.suite(badroot,protocol,screen,lambda *a,**k:dict(returncode=1,killed=False))
        assert not failed['completed'] and len(failed['cells'])==1 and failed['cells'][0]['status']=='worker_failure'
        # Exercise exact frozen worker's final-target synthetic handoff, no shell steps.
        shell=out/'shell_fixture';shell.mkdir()
        assert checked(dict(base,out=str(shell),shell_fixture=True),out/'shell_process',timeout=35,kind='HybridQualification')['returncode']==0
        parity=screen.read(shell/'shell_parity.json');assert parity['relative_hessian']<=1e-12 and parity['shell_optimizer_steps']==0
        # Exact-worker handoff/restart checks, including changed-policy and terminal refusals.
        full=out/'toy_full';full.mkdir()
        assert checked(dict(base,out=str(full),recipe='hybrid',toy=True),out/'toy_full_process',timeout=12)['returncode']==0
        expected=screen.read(full/'result.json');assert expected['status']=='fixture_stop' and expected['phase']=='newton'
        policy=screen.read(full/'journal.json')['policy']
        assert policy['newton'].get('guardedHessianReuse',False)==a.guarded
        assert policy['newton'].get('reuseNewtonSymbolic',False)==a.composed
        if a.parallel or a.composed:assert policy['newton']['hessianThreads']==4 and 'hessianThreads' not in policy['lbfgs']
        restart_rows=[]
        for point in (['before_target','after_target','after_commit','progress'] if a.guarded else ['after_commit']):
            toy=out/('toy_'+point);toy.mkdir()
            p=checked(dict(base,out=str(toy),recipe='hybrid',toy=True,pause=point),out/('toy_pause_'+point),timeout=12,pause=True)
            assert p['reason']=='safe_pause'
            journal=screen.read(toy/'journal.json');saved=screen.read(toy/journal['active'])['payload']['state']
            floor=max(p['elapsed'],journal['charged_elapsed'],screen.read(toy/'paused.json')['elapsed'])+.001
            if a.guarded:
                wrong=out/('wrong_'+point);shutil.copytree(toy,wrong)
                req=dict(base,out=str(wrong),recipe='hybrid',toy=True,resume=True,prior_elapsed=floor,guarded_hessian=False)
                assert checked(req,out/('wrong_process_'+point),timeout=12)['returncode']!=0
            if a.parallel or a.composed:
                wrong=out/('wrong_threads_'+point);shutil.copytree(toy,wrong)
                req=dict(base,out=str(wrong),recipe='hybrid',toy=True,resume=True,prior_elapsed=floor,hessian_threads=1)
                assert checked(req,out/('wrong_process_'+point),timeout=12)['returncode']!=0
            if a.composed:
                wrong_sym=out/('wrong_symbolic_'+point);shutil.copytree(toy,wrong_sym)
                req=dict(base,out=str(wrong_sym),recipe='hybrid',toy=True,resume=True,prior_elapsed=floor,reuse_symbolic=False)
                assert checked(req,out/('wrong_sym_process_'+point),timeout=12)['returncode']!=0
            q=checked(dict(base,out=str(toy),recipe='hybrid',toy=True,resume=True,prior_elapsed=floor),out/('toy_resume_'+point),timeout=12)
            assert q['returncode']==0
            actual=screen.read(toy/'result.json')
            for key in ('attempts','iterations','evaluations','hessians','status','phase'):assert actual[key]==expected[key],(point,key)
            for name in ('final_x.f64','final_g.f64'):
                np.testing.assert_allclose(np.fromfile(toy/name,dtype='<f8'),np.fromfile(full/name,dtype='<f8'),rtol=1e-12,atol=1e-13)
            assert checked(dict(base,out=str(toy),recipe='hybrid',toy=True,resume=True,prior_elapsed=floor+q['elapsed']),out/('terminal_refusal_'+point),timeout=12)['returncode']!=0
            restart_rows.append(dict(boundary=point,passed=True,saved_cache_valid=saved['cacheValid'],saved_reuse_age=saved.get('hessianReuseAge',0)))
        screen.atomic(out/'launch_gate.json',dict(passed=True,scientific_fits=0,shell_optimizer_steps=0,
            dry_cells=2,reopening_refused=True,budget_skips_verified=True,worker_failure_stops_verified=True,
            synthetic_shell_parity=parity,toy_durable_resume=True,restart_checks=restart_rows,elapsed_since_phase_entry=time.time()-entry))
        screen.atomic(out/'status.json',dict(stage='running',completed=False,planned_cells=2))
        allowance=max(0,fit_deadline-time.time())
        ledger=core.suite(out,protocol,screen,checked,allowance=allowance)
    except BaseException as e:
        failure=repr(e)
        screen.atomic(out/'failure.json',dict(error=failure,no_retry=True,elapsed=time.time()-entry))
    finally:
        signal.alarm(0)
        if (out/'screen/ledger.json').exists():ledger=screen.read(out/'screen/ledger.json')
        remaining=int(deadline-time.time())
        if ledger is not None and remaining>5:
            with (out/'report.log').open('x') as log:
                try:
                    reporter='report_guarded_hessian_pilot.py' if (a.guarded or a.parallel or a.composed) else 'report_profiled_zigzag_pilot.py'
                    p=subprocess.run([sys.executable,str(out/reporter),'--root',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                    report_status='complete_unreviewed' if p.returncode==0 else 'failed_partial'
                except subprocess.TimeoutExpired:report_status='report_budget_exhausted_partial'
        else:report_status='unavailable_or_insufficient_budget'
        cells=[] if ledger is None else ledger['cells']
        screen.atomic(out/'status.json',dict(stage='finished' if failure is None else 'failed_or_interrupted',
            completed=bool(ledger and ledger['completed']),scientific_fits_started=0 if ledger is None else ledger['scientific_fits_started'],
            accepted=sum(c.get('result',{}).get('accepted',False) for c in cells),statuses=[c['status'] for c in cells],
            report_status=report_status,failure=failure,elapsed_since_phase_entry=time.time()-entry,no_retry=True))
    if failure is not None:raise RuntimeError(failure)

if __name__=='__main__':main()
