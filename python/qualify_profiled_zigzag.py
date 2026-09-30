#!/usr/bin/env python3
"""Bounded no-fit qualification of profiled, hardened negative-orthotropy targets."""
import argparse, csv, json, math, os, shlex, shutil, subprocess, time
from pathlib import Path
import numpy as np
import run_nonconvex_screen as screen
import qualify_cholmod_integration as dep
from visualize_solver_characteristics import place_strips, point_segment_distance


def build(out, frozen):
    row=next(r for r in screen.read(frozen/'build/compile_commands.json') if r['file'].endswith('Test_NonconvexScreen.cpp'))
    args=shlex.split(row['command']);flags=[];i=0
    while i<len(args):
        if args[i] in ('-c','-o'):i+=2;continue
        flags.append(args[i]);i+=1
    obj=out/'prepare.o';binary=out/'prepare_target'
    command=flags+['-c',str(out/'source/PrepareProfiledZigzag.cpp'),'-o',str(obj)]
    screen.atomic(out/'compile_command.json',command)
    subprocess.run(command,check=True,timeout=300)
    link=shlex.split(subprocess.check_output(['ninja','-C',str(frozen/'build'),'-t','commands','testshell'],text=True).splitlines()[-1])[2:-2]
    args=[];i=0
    while i<len(link):
        token=link[i]
        if token=='-o':args+=['-o',str(binary)];i+=2;continue
        if token.endswith('.cpp.o') or token.startswith('-Wl,--dependency-file='):i+=1;continue
        args.append(str(frozen/'build'/token) if token.startswith('lib/') else token);i+=1
    args.insert(1,str(obj));screen.atomic(out/'link_command.json',args)
    subprocess.run(args,check=True,timeout=45)
    return binary


def independently_verify(out, bundle, sequence):
    X=np.fromfile(bundle/'reference.f64',dtype='<f8').reshape((5427,3),order='F')
    F=np.fromfile(bundle/'faces.i32',dtype='<i4').reshape((10560,3),order='F')
    uv=X[:,:2];centers=uv[F].mean(axis=1);nf=len(F)
    D=np.stack((uv[F[:,2]]-uv[F[:,1]],uv[F[:,0]]-uv[F[:,2]]),axis=2)
    S=np.tile(np.eye(2),(nf,1,1));history=np.zeros(nf,dtype=np.int32)
    rows=list(csv.DictReader((bundle/'hits.csv').open()));pos=0;cycle=0;counts=[];errors=[]
    assert sequence['hardening']==dict(model='voce_decay',history_variable='hit_count',beta=.223143551,floor=0.,scope='face_shared',count_rule='plastic_hits')
    def value(op,key,k):
        val=op[key];return val[k] if isinstance(val,list) else val
    for pi,path in enumerate(sequence['toolpaths']):
        if not path.get('enabled',True):continue
        op=path['operation'];placed=place_strips(op,(-.127,.127,-.1524,.1524))
        profile=op.get('profile',sequence['defaults']['profile'])
        for repeat in range(1,path['repeat']+1):
            cycle+=1;n=0
            for strip in placed['strips']:
                k=strip['strip_index'];a=strip['a_uv_m'];b=strip['b_uv_m'];d=b-a
                ids=np.flatnonzero(point_segment_distance(centers,a,b)<=placed['half_width_m'])
                for f in ids:
                    s=float(np.clip((centers[f]-a)@d/(d@d),0,1))
                    p=1. if profile['mode']=='uniform' else profile['top_end_ratio']+(1-profile['top_end_ratio'])*math.sin(math.pi*s)**profile['top_profile_power']
                    gt=value(op,'gtop',k)*p;gb=value(op,'gbot',k);ortho=value(op,'ortho',k)
                    assert ortho==-.5 and gb==0 and gt>0
                    q=math.exp(-sequence['hardening']['beta']*int(history[f]))
                    g1=q*gt*(1+ortho);g2=q*gt*(1-ortho);angle=strip['angle_material_rad']
                    r=rows[pos];pos+=1;n+=1
                    for field,val in [('cycle',cycle),('path',pi),('repeat',repeat),('strip',k),('face',f),('previous_hits',history[f])]:assert int(r[field])==val,(field,r,val)
                    for field,val in [('gtop',gt),('gbot',gb),('ortho',ortho),('profile',p),('s01',s),('q',q),('g1',g1),('g2',g2)]:
                        assert abs(float(r[field])-val)<=1e-13,(field,r,val)
                    assert abs(math.sin(float(r['angle'])-angle))<1e-13
                    c,ss=math.cos(angle),math.sin(angle);R=np.array([[c,-ss],[ss,c]])
                    G=(R*np.array([1+g1,1+g2]))@R.T
                    S[f]=G.T@S[f]@G;history[f]+=1
            expected=np.swapaxes(D,1,2)@S@D
            native=np.fromfile(bundle/f'cycle_{cycle}_top.f64',dtype='<f8').reshape((nf,2,2))
            error=float(np.linalg.norm(native-expected)/np.linalg.norm(expected));assert error<1e-12,error
            counts.append(n);errors.append(error)
    assert pos==len(rows) and cycle==8
    assert np.array_equal(history,np.fromfile(bundle/'history.i32',dtype='<i4'))
    bottom=np.fromfile(bundle/'bottom.f64',dtype='<f8');initial=np.fromfile(bundle/'initial_bottom.f64',dtype='<f8')
    bottomerror=float(np.linalg.norm(bottom-initial)/np.linalg.norm(initial));assert bottomerror<1e-12
    assert screen.sha(bundle/'top.f64')==screen.sha(bundle/'cycle_8_top.f64')
    eigen=np.linalg.eigvalsh(S);assert np.isfinite(eigen).all() and eigen.min()>0
    growth=np.sqrt(eigen)-1
    result=dict(passed=True,scope='all faces, strips, repeats and every native hit; no optimization',cycles=cycle,
        hit_counts=counts,total_hits=pos,faces_hit=int(np.count_nonzero(history)),max_face_hits=int(history.max()),
        cycle_relative_metric_errors=errors,bottom_relative_change=bottomerror,
        principal_natural_length_growth_min=float(growth.min()),principal_natural_length_growth_max=float(growth.max()),
        geometric_centroid_sampling_only=True,mesh_resolution_adequacy_not_established=True)
    screen.atomic(out/'target_verification.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    args=ap.parse_args();out=args.out;start=time.monotonic()
    screen.verify_bundle(args.baseline)
    binary=build(out,args.frozen);native_runtime=dep.runtime(binary);screen.atomic(out/'builder_runtime.json',native_runtime)
    bundle=out/'bundle'
    subprocess.run([str(binary),str(out/'sequence.json'),str(bundle)],check=True,timeout=60)
    for name in ('reference.f64','faces.i32','edges.i32','initial_top.f64','initial_bottom.f64','bbar.f64'):
        assert screen.sha(bundle/name)==screen.sha(args.baseline/name),name
    base=screen.read(args.baseline/'metadata.json')
    for name in ['constraints.u8']+[s+'.f64' for s in base['starts']]:shutil.copy2(args.baseline/name,bundle/name)
    result=independently_verify(out,bundle,screen.read(out/'sequence.json'))
    base.update(cycles=8,hit_counts=result['hit_counts'],target_recipe='profiled_voce_ortho_minus_0p5',sequence_sha256=screen.sha(out/'sequence.json'))
    screen.atomic(bundle/'metadata.json',base)
    screen.atomic(bundle/'manifest.json',dict(schema=1,sha256={p.name:screen.sha(p) for p in sorted(bundle.iterdir()) if p.is_file()}))
    identity=screen.verify_bundle(bundle)
    worker=args.frozen/'bin/testshell'
    assert screen.sha(worker)=='d994c9f63f784a595bfa113ac98fe147786bd6e18535a80020a5caa73102a0d1'
    runtime=dep.runtime(worker);dep_id=dep.digest(runtime);screen.atomic(out/'runtime_manifest.json',dict(identity=dep_id,**runtime))
    checks=out/'derivatives';checks.mkdir()
    request=dict(mode='preflight',bundle=str(bundle),identity=identity,out=str(checks),
        implementation='nonconvex-core-v8-local-dihedral-optin',newton_backend='cholmod',dependency_identity=dep_id,
        hessian_assembly='local_dihedral_csc',local_dihedral_parity=True)
    screen.atomic(out/'preflight_request.json',request)
    remaining=560-(time.monotonic()-start)
    if remaining<90:raise RuntimeError('insufficient original phase budget; no derivative launch')
    with (out/'preflight.log').open('x') as log:
        subprocess.run([str(worker),'--gtest_filter=NonconvexScreen.Dispatch'],env=dict(os.environ,NONCONVEX_SCREEN_REQUEST=str(out/'preflight_request.json')),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    checks_data=screen.read(checks/'preflight.json');assert len(checks_data)==6 and all(r['passed'] for r in checks_data)
    dep.verify(binary,native_runtime);dep.verify(worker,runtime);assert screen.verify_bundle(bundle)==identity
    screen.atomic(out/'gate.json',dict(passed=True,qualification_only=True,scientific_fits=0,optimizer_steps=0,
        bundle=str(bundle),identity=identity,dependency_identity=dep_id,binary_sha256=screen.sha(worker),
        elapsed=time.monotonic()-start,checkpoint_scope='initial state roundtrips; hybrid durable-boundary qualification inherited, not new arbitrary-crash testing',
        limitations=['mesh adequacy for 4 mm strips not established','physical growth calibration and self-contact not validated']))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
