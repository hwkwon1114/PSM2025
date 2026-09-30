#!/usr/bin/env python3
"""Isolated build + toy transactional recovery + synthetic shell handoff; no fits."""
import argparse,hashlib,json,os,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np

def interrupted(*_):raise TimeoutError('qualification budget/interruption; no retries')
def main():
 signal.signal(signal.SIGTERM,interrupted)
 signal.signal(signal.SIGALRM,interrupted);signal.alarm(850)
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frozen',type=Path,required=True);a=ap.parse_args()
 out=a.out.resolve();frozen=a.frozen.resolve();sys.path.insert(0,str(frozen/'python'))
 import run_nonconvex_screen as runner
 import qualify_cholmod_integration as helper
 src=out/'source';build=frozen/'build'
 # Frozen compiler/link flags, including matching ABI; never rebuild frozen libs.
 cmd=shlex.split(next(x['command'] for x in runner.read(build/'compile_commands.json') if x['file'].endswith('Test_NonconvexScreen.cpp')))
 flags=[];i=0
 while i<len(cmd):
  if cmd[i] in ('-o','-c'):i+=2;continue
  flags.append(cmd[i]);i+=1
 obj=out/'hybrid.o';binary=out/'hybrid_worker'
 compile_cmd=flags+['-I'+str(src),'-I'+str(frozen/'test/testshell'),'-I'+str(frozen/'test/diagnostics'),'-o',str(obj),'-c',str(src/'QualifyHybridShell.cpp')]
 runner.atomic(out/'compile_command.json',compile_cmd)
 subprocess.run(compile_cmd,check=True,timeout=500)
 commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','testshell'],text=True).splitlines()
 link=shlex.split(commands[-1]);link=link[2:-2];new=[];i=0
 while i<len(link):
  token=link[i]
  if token=='-o':new+=['-o',str(binary)];i+=2;continue
  if token.endswith('.cpp.o') or token.startswith('-Wl,--dependency-file='):i+=1;continue
  new.append(str(build/token) if token.startswith('lib/') else token);i+=1
 new.insert(1,str(obj));runner.atomic(out/'link_command.json',new)
 subprocess.run(new,check=True,timeout=60)
 manifest=helper.runtime(binary);identity=helper.digest(manifest)
 runner.atomic(out/'runtime_manifest.json',dict(identity=identity,**manifest))
 bundle=Path(runner.read(frozen.parent.parent/'local_dihedral_solver_gate_7547943/preflight/gate.json')['bundle'])
 bundle_id=runner.verify_bundle(bundle)
 base=dict(implementation=runner.sha(binary),dependency_identity=identity,bundle=str(bundle),identity=bundle_id)
 serial=0
 def launch(work,*,resume=False,pause='',shell_fixture=False,charged_elapsed=0,**extra):
  nonlocal serial
  helper.verify(binary,manifest);assert runner.verify_bundle(bundle)==bundle_id
  work.mkdir(exist_ok=resume);req=dict(base,out=str(work),resume=resume,pause=pause,shell_fixture=shell_fixture,charged_elapsed=charged_elapsed);req.update(extra)
  serial+=1;request=out/('request_%02d.json'%serial);runner.atomic(request,req)
  env=dict(os.environ,HYBRID_QUALIFICATION_REQUEST=str(request));t=time.monotonic()
  with (out/('worker_%02d.log'%serial)).open('x') as f:
   p=subprocess.Popen([str(binary),'--gtest_filter=HybridQualification.Dispatch'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   killed=False
   try:
    while p.poll() is None:
     if pause and (work/'paused.json').exists():os.killpg(p.pid,signal.SIGKILL);killed=True;break
     if time.monotonic()-t>90:raise TimeoutError('worker cap; stop phase, no retries')
     time.sleep(.03)
    rc=p.wait()
   finally:
    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
  result=dict(returncode=rc,killed_at_pause=killed,elapsed=time.monotonic()-t,request=str(request))
  runner.atomic(out/('process_%02d.json'%serial),result);return result
 # Baseline toy plus controlled SIGKILL/restoration at every transaction boundary.
 baseline=out/'toy_full';assert launch(baseline)['returncode']==0
 ref=runner.read(baseline/'completed.json');rows=[]
 for point in ('before_target','after_target','after_commit','after_progress'):
  work=out/('toy_'+point);killed=launch(work,pause=point);assert killed['killed_at_pause']
  source_hash=runner.sha(work/'source.json')
  # Refuse recipe changes and cap reset before resuming the valid source.
  wrong=out/('wrong_'+point);shutil.copytree(work,wrong)
  assert launch(wrong,resume=True,switch_attempts=7)['returncode']!=0
  assert not (wrong/'completed.json').exists()
  restored=launch(work,resume=True,charged_elapsed=killed['elapsed'])
  assert restored['returncode']==0 and runner.sha(work/'source.json')==source_hash
  result=runner.read(work/'completed.json')
  for key in ('attempts','evaluations'):assert result[key]==ref[key]
  for key in ('x','g'):np.testing.assert_allclose(result[key],ref[key],rtol=1e-12,atol=1e-13)
  assert result['elapsed']>=killed['elapsed']
  assert launch(work,resume=True)['returncode']!=0, 'completed run reopened'
  rows.append(dict(boundary=point,passed=True,source_unchanged=True,charged_seconds=killed['elapsed']+restored['elapsed']))
 # Negative fixtures retain a genuine durable pre-switch pause, never rewrite
 # numerical checkpoint identities to force a successful restoration.
 origin=out/'wrong_before_target'
 for label,extra in [('exhausted',dict(charged_elapsed=120)),('dependency',dict(dependency_identity='wrong-runtime'))]:
  dest=out/('reject_'+label);shutil.copytree(origin,dest)
  assert launch(dest,resume=True,**extra)['returncode']!=0 and not (dest/'completed.json').exists()
 # Synthetic shell checkpoint at switch threshold; never execute shell Engine.step.
 shell=out/'shell_handoff';assert launch(shell,shell_fixture=True)['returncode']==0
 parity=runner.read(shell/'shell_parity.json');assert parity['shell_optimizer_steps']==0 and parity['dofs']==32267
 helper.verify(binary,manifest)
 runner.atomic(out/'qualification_gate.json',dict(passed=True,qualification_only=True,scientific_fits=0,
  shell_optimizer_steps=0,boundary_tests=rows,shell_parity=parity,dependency_identity=identity,
  binary_sha256=runner.sha(binary),limits=['Toy SIGKILL at durable boundaries only; arbitrary mid-step crash not resumable.',
  'Shell handoff is synthetic, not 2000 real L-BFGS steps or a convergence test.',
  'No production change or comparative fit authorization.']))
 print('HYBRID_TRANSACTION_AND_SHELL_FIXTURE_PASSED; ZERO_SCIENTIFIC_FITS')
if __name__=='__main__':main()
