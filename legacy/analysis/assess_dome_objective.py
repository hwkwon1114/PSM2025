"""Offline planning diagnostic only; never imported by production BO."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def scores(k, valid, areas, penalty=.5):
    valid=valid & np.isfinite(k).all(axis=1)
    total=areas.sum()
    if total<=0: raise ValueError('Empty area')
    output=[]
    for sign in (-1,1):
        m=np.min(sign*k,axis=1)
        target=np.maximum(.1,.1*np.max(abs(k),axis=1))
        u=np.full(len(k),-penalty)
        positive=valid & (m>=0);negative=valid & (m<0)
        u[positive]=np.minimum(m[positive]/target[positive],1)
        u[negative]=-penalty*abs(m[negative])/(.1+abs(m[negative]))
        output.append(float(areas@u/total))
    return dict(score=max(output),negative_sign=output[0],positive_sign=output[1])


def largest_component(faces, eligible, areas):
    ids=np.flatnonzero(eligible)
    if not len(ids):return 0.
    edges={};row=[];col=[]
    for local,face in enumerate(faces[ids]):
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            key=tuple(sorted((int(a),int(b))))
            if key in edges:
                other=edges[key];row.extend([local,other]);col.extend([other,local])
            else:edges[key]=local
    graph=coo_matrix((np.ones(len(row)),(row,col)),shape=(len(ids),len(ids))).tocsr()
    _,labels=connected_components(graph,directed=False)
    return float(np.bincount(labels,weights=areas[ids]).max()/areas.sum())


def assess(path):
    with np.load(path) as a:
        k,valid,uv,faces=[a[n] for n in ('geometric','valid','uv','faces')]
    triangles=uv[faces];c=triangles.mean(axis=1)
    roi=(abs(c[:,0])<=.08)&(abs(c[:,1])<=.12)
    e=triangles[:,1]-triangles[:,0];f=triangles[:,2]-triangles[:,0]
    areas=abs(e[:,0]*f[:,1]-e[:,1]*f[:,0])/2
    k,valid,faces,areas=k[roi],valid[roi],faces[roi],areas[roi]
    valid=valid & np.isfinite(k).all(axis=1)
    threshold=np.maximum(.1,.1*np.max(abs(k),axis=1))
    cover=[];connected=[]
    for sign in (-1,1):
        eligible=valid & (np.min(sign*k,axis=1)>=threshold)
        cover.append(float(areas[eligible].sum()/areas.sum()))
        connected.append(largest_component(faces,eligible,areas))
    return dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        coverage=max(cover),connected_coverage=max(connected),
        saddle_fraction=float(areas[valid & (k[:,0]*k[:,1]<0)].sum()/areas.sum()),
        invalid_fraction=float(areas[~valid].sum()/areas.sum()),
        continuous={str(p):scores(k,valid,areas,p) for p in (.5,1.)})


def self_test():
    a=np.ones(4);valid=np.ones(4,dtype=bool)
    def score(pair):return scores(np.tile(pair,(4,1)),valid,a)['score']
    assert score([0,0])==0 and score([0,.5])==0
    assert score([.2,.2])==1 and score([-.2,-.2])==1
    assert score([-.2,.2])<0
    k=np.array([[.2,.2],[.2,.2],[-.2,-.2],[-.2,-.2]])
    assert scores(k,valid,a)['score']<1
    assert scores(k,valid,a)=={'score':scores(-k,valid,a)['score'],
        'negative_sign':scores(-k,valid,a)['positive_sign'],'positive_sign':scores(-k,valid,a)['negative_sign']}
    for pair in ([.2,.2],[-.2,.2],[0,0]):
        k=np.tile(pair,(4,1));bad=valid.copy();bad[0]=False
        assert scores(k,bad,a)['score']<=scores(k,valid,a)['score']
    assert score([20,20])==score([.2,.2])
    assert score([.11,1.15])<score([.11,.95]) # ratio trade-off preserved, not erased
    faces=np.array([[0,1,2],[2,1,3],[4,5,6]])
    assert np.isclose(largest_component(faces,np.ones(3,dtype=bool),np.ones(3)),2/3)
    # Scalar score depends only on principal eigenvalues, invariant to tangent basis rotation.
    t=.71;Q=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    assert np.allclose(np.linalg.eigvalsh(Q@np.diag([-.2,.4])@Q.T),[-.2,.4])
    return 'passed: plane/cylinder/dome/saddle/mixed-sign/invalid/positive-saturation/ratio/connectivity/basis-invariance'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();tests=self_test()
    manifest=json.loads(args.manifest.read_text())
    result={'tests':tests,'scope':'exploratory offline comparison; no validation of future BO benefit',
            'groups':{name:[assess(Path(path)) for path in paths] for name,paths in manifest.items()}}
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
