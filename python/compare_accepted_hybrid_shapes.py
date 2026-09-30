#!/usr/bin/env python3
"""Read-only comparison of accepted +/- hybrid endpoints; no fitting/optimization.
Proper rigid Kabsch alignment with fixed material-vertex correspondence.
"""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stats(v):
 v=np.asarray(v);assert np.isfinite(v).all()
 return dict(count=len(v),rms=float(np.sqrt(np.mean(v*v))),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),max=float(v.max()))
def align(a,b,w):
 w=w/w.sum();ca=w@a;cb=w@b;x=a-ca;y=b-cb
 u,s,v=np.linalg.svd((x*w[:,None]).T@y)
 d=np.eye(3);d[2,2]=np.linalg.det(u@v);r=u@d@v;t=cb-ca@r
 assert abs(np.linalg.det(r)-1)<1e-12
 assert np.linalg.norm(r.T@r-np.eye(3))<1e-12
 return a@r+t,r,t

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(exist_ok=False);ledger=read(a.root/'screen/ledger.json');protocol=read(a.root/'protocol.json');bundle=Path(protocol['request_base']['bundle'])
 manifest=read(bundle/'manifest.json');hashes={}
 for name,h in manifest['sha256'].items():assert sha(bundle/name)==h,name
 assert sha(bundle/'manifest.json')==protocol['request_base']['identity']
 meta=read(bundle/'metadata.json');nv,nf,ne,nd=[meta[k] for k in ('nv','nf','ne','nd')];assert nd==3*nv+ne
 f=np.fromfile(bundle/'faces.i32',dtype='<i4').reshape(nf,3,order='F');e=np.fromfile(bundle/'edges.i32',dtype='<i4').reshape(ne,2,order='F');ref=np.fromfile(bundle/'reference.f64',dtype='<f8').reshape(nv,3,order='F')
 assert f.min()>=0 and f.max()<nv and e.min()>=0 and e.max()<nv
 pair=[]
 for start in ('cylinder_y_plus','cylinder_y_minus'):
  c=next(c for c in ledger['cells'] if c['recipe']=='hybrid' and c['start']==start)
  work=Path(c['out']);r=read(work/'result.json')
  assert r['accepted'] and r['status']=='gradient_target' and r['selected_gradient_norm']<=5e-14 and r['curvature']=='positive_pivots'
  assert sha(r['checkpoint'])==c['checkpoint_sha256']
  q=np.fromfile(work/'selected_x.f64',dtype='<f8');final=np.fromfile(work/'final_x.f64',dtype='<f8')
  assert q.size==nd and np.isfinite(q).all() and np.array_equal(q,final)
  for p in (work/'result.json',work/'selected_x.f64',work/'final_x.f64',Path(r['checkpoint'])):hashes[str(p)]=sha(p)
  pair.append((q[:3*nv].reshape(nv,3,order='F'),q[3*nv:],r))
 p,phip,rp=pair[0];m,phim,rm=pair[1]
 # Self/known transform tests protect row-vector rotation convention.
 testR=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
 checked,_,_=align(p,p@testR+np.array([.1,-.2,.03]),np.ones(nv));assert np.max(np.abs(checked-(p@testR+np.array([.1,-.2,.03]))))<1e-12
 aligned,R,t=align(m,p,np.ones(nv));dist=np.linalg.norm(aligned-p,axis=1)
 area=np.linalg.norm(np.cross(ref[f[:,1]]-ref[f[:,0]],ref[f[:,2]]-ref[f[:,0]]),axis=1)/2
 weights=np.zeros(nv)
 for k in range(3):np.add.at(weights,f[:,k],area/3)
 aw,Rw,tw=align(m,p,weights);dw=np.linalg.norm(aw-p,axis=1)
 lp=np.linalg.norm(p[e[:,1]]-p[e[:,0]],axis=1);lm=np.linalg.norm(m[e[:,1]]-m[e[:,0]],axis=1)
 def normals(v):
  n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);mag=np.linalg.norm(n,axis=1);assert (mag>0).all();return n/mag[:,None]
 np_,nm=normals(p),normals(aligned)
 angle=np.arctan2(np.linalg.norm(np.cross(np_,nm),axis=1),np.sum(np_*nm,axis=1))
 rawphi=phim-phip;wrapped=np.arctan2(np.sin(rawphi),np.cos(rawphi))
 span=float(np.linalg.norm(np.ptp(ref,axis=0)));assert span>0
 result=dict(job=os.getenv('SLURM_JOB_ID'),question='Do accepted hybrid endpoints coincide up to a proper rigid transform?',
  nv=nv,nf=nf,ne=ne,units='vertex positions in metres; director/normal angles in radians unless suffix states otherwise',
  selection='both accepted hybrid endpoints of the predeclared +/- starts, full matching vertex and edge IDs; no remeshing or vertex permutation',
  convention='minus @ R + t aligned to plus; SO(3), no reflections, scaling, or deformation',
  determinant=float(np.linalg.det(R)),R=R.tolist(),t_m=t.tolist(),
  raw_vertex_distance_m=stats(np.linalg.norm(m-p,axis=1)),
  aligned_vertex_distance_m=stats(dist),aligned_vertex_distance_mm=stats(dist*1000),
  reference_diagonal_m=span,relative_aligned_rms_to_reference_diagonal=float(np.sqrt(np.mean(dist**2))/span),
  reference_area_weighted_alignment=dict(vertex_errors_m=stats(dw),area_weighted_rms_m=float(np.sqrt(np.sum(weights*dw**2)/weights.sum()))),
  edge_length_absolute_difference_m=stats(np.abs(lm-lp)),edge_length_relative_difference=stats(np.abs(lm-lp)/lp),
  aligned_face_normal_difference_deg=stats(np.degrees(angle)),
  director_absolute_raw_difference_rad=stats(np.abs(rawphi)),director_absolute_wrapped_difference_rad=stats(np.abs(wrapped)),
  plus_energy=rp['energy'],minus_energy=rm['energy'],energy_absolute_difference=abs(rp['energy']-rm['energy']),
  native_gradients=[rp['selected_gradient_norm'],rm['selected_gradient_norm']],source_hashes=hashes,
  script_sha256=sha(__file__),python=sys.version,numpy=np.__version__,
  limits=['Numerical agreement is not exact equality, uniqueness or a global minimum certificate.',
  'Rigid-invariant stored director angles are compared at identical edges; no raw angle reflection equivalence assumed.',
  'No new objective/gradient or curvature evaluation; acceptance checks inherited from saved independently verified results.',
  'All vertex/edge/face records used; spatial samples are dependent, no inferential intervals.'])
 (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.out/'vertex_residuals.csv').open('x',newline='') as out:
  w=csv.writer(out);w.writerow(['vertex_id','raw_distance_m','aligned_distance_m','area_weighted_alignment_distance_m']);w.writerows(zip(range(nv),np.linalg.norm(m-p,axis=1),dist,dw))
 with (a.out/'edge_residuals.csv').open('x',newline='') as out:
  w=csv.writer(out);w.writerow(['edge_id','length_difference_m','director_difference_rad','wrapped_director_difference_rad']);w.writerows(zip(range(ne),lm-lp,rawphi,wrapped))
 print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','R','t_m')},indent=2))
if __name__=='__main__':main()
