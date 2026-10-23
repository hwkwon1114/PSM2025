"""LHS-derived constrained maximin design; final subset is not an exact LHS.

Retains a strict-LHS diagnostic for provenance. No global optimality claim.
"""
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist
from dome_bo_repeat_domain import BOUNDS, decode, encode, sequence


def physical(unit):
    bounds=np.asarray(BOUNDS,float)
    result=bounds[:,0]+unit*(bounds[:,1]-bounds[:,0])
    result[:,[2,5,6,7]]=np.floor(result[:,[2,5,6,7]]+.5)
    return result


def violations(values):
    # Fixed geometry: middle width8, side width4, rotations0/90, shifts+/-118.
    middle_span=(values[:,2]-2)*values[:,0]*np.tan(np.deg2rad(values[:,1]))+8
    side_span=(values[:,5]-2)*values[:,3]*np.tan(np.deg2rad(values[:,4]))+4
    return np.maximum(middle_span/200-1,0)**2+np.maximum(side_span/64-1,0)**2


def optimized_lhs(base, seed=20261019, n=24):
    unit=qmc.LatinHypercube(8,seed=seed).random(8192)
    values=physical(unit)
    indices=np.flatnonzero(violations(values)<=0)
    canonical=np.asarray([encode(decode(unit[i])) for i in indices])
    if len(indices)<n:
        raise RuntimeError('Insufficient feasible LHS-derived candidates')
    # Multistart farthest-point selection in decoded normalized coordinates.
    rng=np.random.default_rng(seed)
    best=None; best_distance=-1.
    for start in rng.choice(len(indices),size=min(32,len(indices)),replace=False):
        chosen=[int(start)]
        distances=np.linalg.norm(canonical-canonical[start],axis=1)
        for _ in range(n-1):
            index=int(np.argmax(distances)); chosen.append(index)
            distances=np.minimum(distances,np.linalg.norm(canonical-canonical[index],axis=1))
        separation=float(pdist(canonical[chosen]).min())
        if separation>best_distance:
            best=chosen; best_distance=separation
    selected=indices[best]
    parameters=[decode(unit[i]) for i in selected]
    for value in parameters: sequence(base,value)
    assert len({tuple(v) for v in parameters})==n
    return {'seed':seed,'method':'LHS_derived_feasible_multistart_maximin_subset',
            'pool_size':len(unit),'feasible_pool_size':len(indices),'starts':32,
            'normalized_min_distance':best_distance,
            'latent_unit_points':unit[selected].tolist(),'parameters':parameters,
            'integer_unique_counts':{str(j):len({v[j] for v in parameters}) for j in (2,5,6,7)},
            'limits':'Filtered and selected subset is NOT an exact Latin hypercube; best-found maximin heuristic, not global optimum.'}


def strict_lhs_diagnostic(base, seed=20261019, n=24, iterations=40000):
    rng=np.random.default_rng(seed)
    unit=qmc.LatinHypercube(8,seed=seed).random(n)
    def objective(points):
        values=physical(points)
        canonical=(values-np.asarray(BOUNDS)[:,0])/np.ptp(np.asarray(BOUNDS),axis=1)
        minimum=float(pdist(canonical).min())
        penalty=float(violations(values).sum())
        return 1000*penalty-minimum, penalty, minimum
    score,penalty,distance=objective(unit)
    best=None; best_distance=-1.
    for step in range(iterations):
        col=int(rng.integers(8)); a,b=rng.choice(n,2,replace=False)
        unit[a,col],unit[b,col]=unit[b,col],unit[a,col]
        trial,new_penalty,new_distance=objective(unit)
        temperature=.5*(1-step/iterations)**3+.0001
        if trial<score or rng.random()<np.exp(min(0,(score-trial)/temperature)):
            score,penalty,distance=trial,new_penalty,new_distance
        else:
            unit[a,col],unit[b,col]=unit[b,col],unit[a,col]
        if penalty==0 and distance>best_distance:
            best=unit.copy(); best_distance=distance
    if best is None:
        raise RuntimeError('No feasible stratified LHS found; do not silently replace sampling method')
    values=[decode(row) for row in best]
    for value in values: sequence(base,value)
    assert len({tuple(v) for v in values})==n
    strata=np.floor(best*n).astype(int)
    assert all(np.array_equal(np.sort(strata[:,j]),np.arange(n)) for j in range(8))
    return {'seed':seed,'method':'column_swap_annealed_feasible_maximin_LHS',
            'iterations':iterations,'normalized_min_distance':best_distance,
            'latent_unit_points':best.tolist(),'parameters':values,
            'integer_unique_counts':{str(j):len({v[j] for v in values}) for j in (2,5,6,7)},
            'limits':'Latent Latin strata preserved; rounded integer marginals may tie. Best-found design, not certified global optimum.'}
