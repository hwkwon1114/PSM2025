#!/usr/bin/env python3
"""Qualification only: dependency checks, operational fixtures, frozen near solve."""
import argparse,copy,hashlib,json,os,signal,subprocess
from pathlib import Path
from types import SimpleNamespace
import run_nonconvex_screen as screen

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def runtime(binary):
    text=subprocess.check_output(['ldd',str(binary)],text=True)
    if 'not found' in text:raise RuntimeError('unresolved runtime library')
    files={}
    for line in text.splitlines():
        parts=line.split();path=None
        if '=>' in parts and len(parts)>2 and parts[2].startswith('/'):path=parts[2]
        elif parts and parts[0].startswith('/'):path=parts[0]
        if path:files[path]=screen.sha(path)
    if not files:raise RuntimeError('empty runtime fingerprint')
    return dict(binary_sha256=screen.sha(binary),libraries=files,
                loader_environment={k:os.environ.get(k) for k in ['LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT']},
                ld_cache_sha256=screen.sha('/etc/ld.so.cache') if Path('/etc/ld.so.cache').exists() else None)
def verify(binary,manifest):
    if screen.sha(binary)!=manifest['binary_sha256']:raise RuntimeError('binary changed')
    if {k:os.environ.get(k) for k in manifest['loader_environment']}!=manifest['loader_environment']:raise RuntimeError('loader environment changed')
    cache=screen.sha('/etc/ld.so.cache') if Path('/etc/ld.so.cache').exists() else None
    if cache!=manifest['ld_cache_sha256']:raise RuntimeError('loader cache changed')
    for path,expected in manifest['libraries'].items():
        if screen.sha(path)!=expected:raise RuntimeError('runtime library identity changed: '+path)
def stop(*args):raise TimeoutError('integration qualification interrupted; no retries')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--binary',type=Path,required=True);args=ap.parse_args()
    out=args.root;binary=args.binary.resolve();signal.signal(signal.SIGTERM,stop)
    manifest=runtime(binary);identity=digest(manifest);screen.atomic(out/'runtime_manifest.json',dict(identity=identity,**manifest))
    bad=copy.deepcopy(manifest);bad['libraries'][next(iter(bad['libraries']))]='0'*64
    try:verify(binary,bad)
    except RuntimeError:pass
    else:raise AssertionError('altered dependency digest accepted')
    original=screen.launch
    def qualified(binary,request,**kwargs):
        if request['mode']=='fit':raise RuntimeError('no scientific fits authorized in integration qualification')
        verify(binary,manifest);r=dict(request)
        if r.get('dependency_identity',identity)!=identity:raise RuntimeError('requested dependency identity mismatch')
        r.update(newton_backend='cholmod',dependency_identity=identity)
        return original(binary,r,**kwargs)
    original_atomic=screen.atomic
    def qualification_atomic(path,value):
        if Path(path)==out/'preflight/gate.json':
            value=dict(value,qualification_only=True,newton_backend='cholmod',dependency_identity=identity)
        return original_atomic(path,value)
    screen.atomic=qualification_atomic
    screen.launch=qualified
    screen.preflight(SimpleNamespace(root=out/'preflight',binary=binary))
    assert runtime(binary)==manifest,'resolved runtime changed during preflight'
    prior=screen.PROJECT/'run/nonconvex_solver_benchmark/newton_pilot_7386943'
    fixture=prior/'near_equilibrium_matrix.json';expected=screen.read(prior/'near_gate.json')['fixture_sha256']
    assert screen.sha(fixture)==expected,'near fixture changed'
    verify(binary,manifest)
    env=dict(os.environ,NONCONVEX_NEAR_FIXTURE=str(fixture),NONCONVEX_NEAR_OUTPUT=str(out/'near_gate.json'))
    with (out/'near_gate.log').open('x') as log:
        subprocess.run([str(binary),'--gtest_filter=BenchmarkCholmod.FrozenNearGate','--gtest_output=xml:'+str(out/'near_tests.xml')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    near=screen.read(out/'near_gate.json');assert near['passed']
    gate_path=out/'preflight/gate.json';gate=screen.read(gate_path)
    gate.update(qualification_only=True,newton_backend='cholmod',dependency_identity=identity,
                dependency_manifest=str(out/'runtime_manifest.json'),dependency_mismatch_rejected=True,
                qualification_driver_sha256=screen.sha(Path(__file__)),near_gate=str(out/'near_gate.json'))
    screen.atomic(gate_path,gate)
    try:screen.screen(SimpleNamespace(root=out/'unauthorized_fit',preflight=out/'preflight',resume=False))
    except RuntimeError as e:assert 'qualification' in str(e)
    else:raise AssertionError('qualification gate authorized a fit')
    screen.atomic(out/'integration_gate.json',dict(passed=True,qualification_only=True,backend='cholmod',dependency_identity=identity,
        binary_sha256=screen.sha(binary),near_fixture_sha256=expected,process_preflight=str(gate_path),near_gate=str(out/'near_gate.json'),
        dependency_checks='ldd closure before/after preflight; binary/library/cache hashes and loader environment before each real process',
        scientific_fits=0,limits=['No whole-trajectory speedup established.','Unknown mid-step crashes remain terminal.','Physical stopping and production solvers unchanged.']))
if __name__=='__main__':main()
