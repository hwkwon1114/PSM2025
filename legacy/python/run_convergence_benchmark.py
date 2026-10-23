#!/usr/bin/env python3
"""Gated convergence comparison; prep has only toy/synthetic workers, never fits."""
import argparse,csv,json,os,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np

class Stop(BaseException):pass
def stop(*_):raise Stop('allocation interruption; no automatic retry')

RECIPES=('lbfgs','newton','hybrid')
def cells():
 return [dict(start='cylinder_y_plus',recipe=r) for r in RECIPES]+[dict(start='cylinder_y_minus',recipe=r) for r in reversed(RECIPES)]

def build(out,frozen,runner,*,headers=None,translation_unit='ConvergenceBenchmark.cpp'):
 src=out/'source';build=frozen/'build';headers=frozen if headers is None else headers
 row=next(x for x in runner.read(build/'compile_commands.json') if x['file'].endswith('Test_NonconvexScreen.cpp'))
 args=shlex.split(row['command']);flags=[];i=0
 while i<len(args):
  if args[i] in ('-c','-o'):i+=2;continue
  flags.append(args[i]);i+=1
 obj=out/'convergence.o';binary=out/'convergence_worker'
 command=flags+['-I'+str(src),'-I'+str(headers/'test/testshell'),'-I'+str(headers/'test/diagnostics'),'-c',str(src/translation_unit),'-o',str(obj)]
 runner.atomic(out/'compile_command.json',command);subprocess.run(command,check=True,timeout=380)
 link=shlex.split(subprocess.check_output(['ninja','-C',str(build),'-t','commands','testshell'],text=True).splitlines()[-1])[2:-2]
 args=[];i=0
 while i<len(link):
  t=link[i]
  if t=='-o':args+=['-o',str(binary)];i+=2;continue
  if t.endswith('.cpp.o') or t.startswith('-Wl,--dependency-file='):i+=1;continue
  args.append(str(build/t) if t.startswith('lib/') else t);i+=1
 args.insert(1,str(obj));runner.atomic(out/'link_command.json',args);subprocess.run(args,check=True,timeout=50)
 return binary

def launch(binary,request,record,runner,timeout,*,pause=False,kind='ConvergenceBenchmark'):
 record.mkdir(parents=True,exist_ok=False);runner.atomic(record/'request.json',request)
 env=dict(os.environ,**{('CONVERGENCE_REQUEST' if kind=='ConvergenceBenchmark' else 'HYBRID_QUALIFICATION_REQUEST'):str(record/'request.json')})
 start=time.monotonic();killed=False
 with (record/'worker.log').open('x') as log:
  proc=subprocess.Popen([str(binary),'--gtest_filter='+kind+'.Dispatch'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while proc.poll() is None:
    if pause and (Path(request['out'])/'paused.json').exists():
     os.killpg(proc.pid,signal.SIGKILL);killed=True;break
    if time.monotonic()-start>=timeout:
     os.killpg(proc.pid,signal.SIGKILL);killed=True;break
    time.sleep(.03 if pause else .2)
   rc=proc.wait()
  finally:
   if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 result=dict(returncode=rc,killed=killed,elapsed=time.monotonic()-start,
   reason='safe_pause' if killed and pause and (Path(request['out'])/'paused.json').exists() else ('watchdog_cap' if killed else 'exit'))
 runner.atomic(record/'process.json',result);return result

def suite(out,protocol,runner,dispatch,*,dry=False,allowance=7150):
 screen=out/('dry_screen' if dry else 'screen');screen.mkdir(exist_ok=False)
 runner.atomic(screen/'protocol.json',protocol);ledger=dict(completed=False,cells=[],scientific_fits_started=0,dry_run=dry)
 runner.atomic(screen/'ledger.json',ledger);start=time.monotonic()
 for i,c in enumerate(protocol['cells']):
  if not dry and allowance-(time.monotonic()-start)<1200:
   ledger['cells'].append(dict(c,index=i,status='global_budget_unstarted',completed=True,charged_seconds=0));continue
  work=screen/('%02d_%s_%s'%(i,c['start'],c['recipe']));work.mkdir()
  request=dict(protocol['request_base'],**c,out=str(work),toy=False)
  row=dict(c,index=i,status='running',out=str(work),completed=False);ledger['cells'].append(row)
  if not dry:ledger['scientific_fits_started']+=1
  runner.atomic(screen/'ledger.json',ledger);t=time.monotonic()
  try:
   process=dispatch(request,work/'process',timeout=1190);row['process']=process
   if dry:row['status']='simulated_cap'
   elif process['returncode']==0 and (work/'completed.json').exists():
    result=runner.read(work/'result.json');row['result']=result;row['status']=result['status']
    if result['accepted']:
     assert result['status']=='gradient_target' and result['selected_gradient_norm']<=5e-14 and result['curvature']=='positive_pivots'
    row['checkpoint_sha256']=runner.sha(result['checkpoint'])
   else:row['status']='watchdog_cap' if process['killed'] else 'worker_failure'
  except BaseException as e:
   row.update(status='infrastructure_failure',error=repr(e));raise
  finally:
   row.update(completed=True,charged_seconds=time.monotonic()-t)
   ledger['elapsed_seconds']=time.monotonic()-start;runner.atomic(screen/'ledger.json',ledger)
  if row['status'] not in ('simulated_cap','time_cap','iteration_cap','evaluation_cap','hessian_cap','gradient_target'):
   break  # preserve evidence, stop on numerical/infrastructure failure
 ledger['completed']=len(ledger['cells'])==len(protocol['cells']) and all(c['completed'] for c in ledger['cells'])
 ledger['elapsed_seconds']=time.monotonic()-start;runner.atomic(screen/'ledger.json',ledger)
 if not dry:
  with (screen/'results.csv').open('x',newline='') as f:
   columns=['index','start','recipe','status','charged_seconds','accepted','energy','gradient_norm','selected_energy','selected_gradient_norm','curvature','attempts','iterations','evaluations','hessians','phase']
   w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
   for c in ledger['cells']:
    merged=dict(c.get('result',{}),**{k:v for k,v in c.items() if k!='result'});w.writerow({k:merged.get(k) for k in columns})
 return ledger

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--preparation',type=Path);a=ap.parse_args()
 out=a.out.resolve();frozen=a.frozen.resolve();sys.path.insert(0,str(frozen/'python'))
 import run_nonconvex_screen as runner
 import qualify_cholmod_integration as helper
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(570 if a.mode=='prepare' else 7150)
 base_root=frozen.parent.parent
 prior=base_root/'local_dihedral_solver_gate_7547943'
 g=runner.read(prior/'preflight/gate.json');assert g['passed'] and g['qualification_only']
 bundle=Path(g['bundle']);identity=runner.verify_bundle(bundle);assert identity==g['identity']
 assert runner.sha(frozen/'bin/testshell')==g['implementation'];assert runner.sha(Path(runner.__file__))==g['runner_sha256']
 if a.mode=='prepare':binary=build(out,frozen,runner)
 else:
  assert a.preparation is not None
  prep=a.preparation.resolve();gate=runner.read(prep/'gate.json');assert gate['passed'] and gate['scientific_fits']==0 and gate['dry_cells']==6
  assert gate['driver_sha256']==runner.sha(Path(__file__))
  binary=prep/'convergence_worker';assert runner.sha(binary)==gate['worker_sha256']
 manifest=helper.runtime(binary);dep=helper.digest(manifest)
 if a.mode=='pilot':
  assert dep==gate['dependency_identity'] and identity==gate['bundle_identity']
  for file,h in gate['source_hashes'].items():assert runner.sha(file)==h,'prepared source/library changed'
 helper.verify(binary,manifest);runner.atomic(out/'runtime_manifest.json',dict(identity=dep,**manifest))
 base=dict(implementation=runner.sha(binary),dependency_identity=dep,bundle=str(bundle),identity=identity)
 protocol=dict(cells=cells(),request_base=base,cell_seconds=1200,optimizer_seconds=1140,worker_timeout=1190,
  preparation_cap=600,fit_job_cap=7200,total_cap=7800,max_attempts=100000,max_evaluations=200000,max_hessians=10000,
  switch_attempts=2000,acceptance='gradient_target + independent native <=5e-14 + positive restricted pivots',
  no_automatic_retry=True,scientific_fits_authorized=a.mode=='pilot')
 runner.atomic(out/'protocol.json',protocol)
 serial=0
 def checked(request,record,**kw):
  assert runner.sha(binary)==base['implementation'];helper.verify(binary,manifest)
  assert runner.verify_bundle(bundle)==identity
  return launch(binary,request,record,runner,**kw)
 if a.mode=='prepare':
  def toy(work,recipe,*,resume=False,pause='',prior_elapsed=0,**extra):
   nonlocal serial
   serial+=1;work.mkdir(exist_ok=resume)
   request=dict(base,out=str(work),recipe=recipe,toy=True,resume=resume,pause=pause,prior_elapsed=prior_elapsed);request.update(extra)
   return checked(request,out/('toy_process_%02d'%serial),timeout=12,pause=bool(pause))
  rows=[]
  for recipe in RECIPES:
   work=out/('toy_full_'+recipe);assert toy(work,recipe)['returncode']==0
   expected=runner.read(work/'result.json')
   points=['progress'] if recipe!='hybrid' else ['before_target','after_target','after_commit','progress']
   for point in points:
    root=out/('toy_'+recipe+'_'+point);p=toy(root,recipe,pause=point);assert p['reason']=='safe_pause'
    # Inject recipe mismatch into a clone; keep the good pause untouched.
    wrong=out/('wrong_'+recipe+'_'+point);shutil.copytree(root,wrong)
    elapsed=max(p['elapsed'],runner.read(root/'journal.json')['charged_elapsed'],runner.read(root/'paused.json')['elapsed'])+.001
    assert toy(wrong,recipe,resume=True,prior_elapsed=elapsed,switch_attempts=7)['returncode']!=0
    q=toy(root,recipe,resume=True,prior_elapsed=elapsed);assert q['returncode']==0
    actual=runner.read(root/'result.json')
    for key in ('attempts','iterations','evaluations','hessians','status','phase'):assert expected[key]==actual[key],(recipe,point,key)
    for file in ('final_x.f64','final_g.f64'):
     np.testing.assert_allclose(np.fromfile(work/file,dtype='<f8'),np.fromfile(root/file,dtype='<f8'),rtol=1e-12,atol=1e-13)
    assert toy(root,recipe,resume=True,prior_elapsed=elapsed+q['elapsed'])['returncode']!=0
    rows.append(dict(recipe=recipe,boundary=point,passed=True))
  # Shared qualification helper + full-size synthetic handoff in this exact build.
  shell=out/'shell_fixture';shell.mkdir()
  req=dict(base,out=str(shell),shell_fixture=True)
  assert checked(req,out/'shell_process',timeout=35,kind='HybridQualification')['returncode']==0
  assert runner.read(shell/'shell_parity.json')['relative_hessian']<=1e-12
  def fake(req,*_,**kw):
   assert req['toy'] is False and req['recipe'] in RECIPES
   return dict(returncode=0,killed=False,simulated=True)
  ledger=suite(out,protocol,runner,fake,dry=True)
  assert ledger['completed'] and ledger['scientific_fits_started']==0
  try:suite(out,protocol,runner,fake,dry=True)
  except FileExistsError:pass
  else:raise AssertionError('completed suite reopened')
  sources={}
  for line in (out/'source_sha256.txt').read_text().splitlines():
   h,path=line.split('  ',1);assert runner.sha(path)==h;sources[path]=h
  helper.verify(binary,manifest)
  runner.atomic(out/'gate.json',dict(passed=True,scientific_fits=0,dry_cells=6,toy_restart_checks=rows,
   shell_optimizer_steps=0,worker_sha256=runner.sha(binary),dependency_identity=dep,bundle_identity=identity,
   driver_sha256=runner.sha(Path(__file__)),source_hashes=sources,
   limits=['Controlled toy pause/restart, synthetic shell handoff; arbitrary crash recovery is not authorized.']))
  print('CONVERGENCE_PREPARATION_PASSED; ZERO_SCIENTIFIC_FITS');return
 runner.atomic(out/'status.json',dict(stage='running',completed=False,planned_cells=6))
 try:
  ledger=suite(out,protocol,runner,checked)
  runner.atomic(out/'status.json',dict(stage='finished',completed=ledger['completed'],scientific_fits_started=ledger['scientific_fits_started'],
    accepted=sum(c.get('result',{}).get('accepted',False) for c in ledger['cells']),statuses=[c['status'] for c in ledger['cells']]))
 except BaseException as e:
  runner.atomic(out/'status.json',dict(stage='failed_or_interrupted',completed=False,error=repr(e),no_retry=True));raise
 print('CONVERGENCE_PILOT_FINISHED')
if __name__=='__main__':main()
