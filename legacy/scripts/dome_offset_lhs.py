"""LHS-derived constrained maximin design; final subset is not an exact LHS.

Nine-variable replacement design. No global optimality claim.
"""
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist
from dome_offset_domain import BOUNDS, decode, encode, sequence


def physical(unit):
    bounds=np.asarray(BOUNDS,float)
    result=bounds[:,0]+unit*(bounds[:,1]-bounds[:,0])
    result[:,[2,5,6,7]]=np.floor(result[:,[2,5,6,7]]+.5)
    return result


def violations(values):
    # Fixed widths8/4 and rotations0/90; ninth coordinate moves both sides.
    middle_span=(values[:,2]-2)*values[:,0]*np.tan(np.deg2rad(values[:,1]))+8
    side_span=(values[:,5]-2)*values[:,3]*np.tan(np.deg2rad(values[:,4]))+4
    return np.maximum(middle_span/200-1,0)**2+np.maximum((side_span+2*values[:,8])/300-1,0)**2


def optimized_lhs(base, seed=20261020, n=24):
    unit=qmc.LatinHypercube(9,seed=seed).random(8192)
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



