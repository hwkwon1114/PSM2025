#!/usr/bin/env python3
"""Compile isolated opt-in candidate and run no-fit qualification once."""
import argparse,json,os,shlex,subprocess,time
from pathlib import Path
import numpy as np
import run_nonconvex_screen as screen
import qualify_cholmod_integration as dep


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--guarded',action='store_true');a=ap.parse_args()
 out=a.out;root=a.root;start=time.monotonic();frozen=root/'run/nonconvex_solver_benchmark/development_7546843/source'
 row=next(r for r in screen.read(frozen/'build/compile_commands.json') if r['file'].endswith('Test_NonconvexScreen.cpp'))
 args=shlex.split(row['command']);flags=[];i=0
 while i<len(args):
  if args[i] in ('-c','-o'):i+=2;continue
  flags.append(args[i]);i+=1
 obj=out/'qualification.o';binary=out/'qualification_worker'
 source='QualifyGuardedHessian.cpp' if a.guarded else 'QualifySymbolicReuse.cpp'
 cmd=flags+['-c',str(out/'source/test/diagnostics'/source),'-o',str(obj)]
 screen.atomic(out/'compile_command.json',cmd);subprocess.run(cmd,check=True,timeout=380)
 link=shlex.split(subprocess.check_output(['ninja','-C',str(frozen/'build'),'-t','commands','testshell'],text=True).splitlines()[-1])[2:-2]
 args=[];i=0
 while i<len(link):
  t=link[i]
  if t=='-o':args+=['-o',str(binary)];i+=2;continue
  if t.endswith('.cpp.o') or t.startswith('-Wl,--dependency-file='):i+=1;continue
  args.append(str(frozen/'build'/t) if t.startswith('lib/') else t);i+=1
 args.insert(1,str(obj));screen.atomic(out/'link_command.json',args);subprocess.run(args,check=True,timeout=45)
 runtime=dep.runtime(binary);screen.atomic(out/'runtime_manifest.json',runtime)
 bundle=root/'run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/bundle';identity=screen.verify_bundle(bundle)
 assert identity=='9f79ed808b6dd6ccca171f5c72538837fe6400653f0f6b4632dcc407e5e259f0'
 pilot=root/'run/nonconvex_solver_benchmark/profiled_zigzag_pilot_7629263';cell=screen.read(pilot/'screen/ledger.json')['cells'][0]
 final=Path(cell['result']['checkpoint']);assert screen.sha(final)==cell['checkpoint_sha256'] and cell['result']['accepted']
 handoff=Path(cell['out'])/'switch_target.json'
 states=[str(bundle/'cylinder_y_plus.f64')];checks=[];hashes={}
 for name,cp in [('handoff',handoff),('accepted',final)]:
  raw=screen.read(cp);x=np.array(raw['payload']['state']['x'],dtype='<f8');assert len(x)==32267 and np.isfinite(x).all()
  file=out/(name+'_x.f64');x.tofile(file);states.append(str(file));checks.append(dict(checkpoint=str(cp),state_file=str(file)));hashes[str(cp)]=screen.sha(cp);hashes[str(file)]=screen.sha(file)
 screen.atomic(out/'input_hashes.json',hashes)
 request=dict(bundle=str(bundle),identity=identity,out=str(out),states=states,checkpoints=checks)
 screen.atomic(out/'request.json',request)
 remaining=560-(time.monotonic()-start)
 if remaining<50:raise RuntimeError('insufficient original budget; no qualification launch')
 with (out/'tests.log').open('x') as log:
  tests='SymbolicReuseQualification.Dispatch:BenchmarkScreenPolicy.*'+(':GuardedHessianQualification.Dispatch' if a.guarded else '')
  subprocess.run([str(binary),'--gtest_filter='+tests],env=dict(os.environ,SYMBOLIC_REUSE_REQUEST=str(out/'request.json')),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
 dep.verify(binary,runtime);assert screen.verify_bundle(bundle)==identity
 for p,h in hashes.items():assert screen.sha(p)==h
 gate=screen.read(out/'gate.json');assert gate['passed'] and gate['scientific_fits']==0
 if a.guarded:
  guarded=screen.read(out/'guarded_gate.json');assert guarded['passed'] and guarded['scientific_fits']==0 and guarded['shell_optimizer_steps']==0
 screen.atomic(out/'completion.json',dict(passed=True,guarded_hessian=a.guarded,scientific_fits=0,elapsed=time.monotonic()-start,binary_sha256=screen.sha(binary),runtime_identity=dep.digest(runtime),production_unchanged=True))
 print('GUARDED_HESSIAN_QUALIFIED_NO_SCIENTIFIC_FITS' if a.guarded else 'SYMBOLIC_REUSE_QUALIFIED_NO_SCIENTIFIC_FITS')

if __name__=='__main__':main()
