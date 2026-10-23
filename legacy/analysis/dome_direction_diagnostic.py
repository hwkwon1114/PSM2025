#!/usr/bin/env python3
"""Descriptive material-direction diagnostics, not a dome acceptance test."""
import argparse
import json
import numpy as np


def summarize(path):
    with np.load(path) as data:
        uv, faces, positions, k, directions, valid = [data[key] for key in
            ('uv', 'faces', 'positions', 'geometric', 'directions', 'valid')]
    ref = uv[faces]
    current = positions[faces]
    dm = np.stack((ref[:,1]-ref[:,0], ref[:,2]-ref[:,0]), axis=-1)
    dx = np.stack((current[:,1]-current[:,0], current[:,2]-current[:,0]), axis=-1)
    jacobian = dx @ np.linalg.inv(dm)
    weak_index = np.argmin(abs(k), axis=1)
    weak = directions[np.arange(len(k)), weak_index]
    # Least-squares pullback of the physical tangent direction, then normalize
    # in orthonormal material coordinates, not in triangle edge coordinates.
    material = np.einsum('nij,nj->ni', np.linalg.pinv(jacobian), weak)
    norm = np.linalg.norm(material, axis=1)
    good = valid & np.isfinite(material).all(axis=1) & (norm > 0)
    material[good] /= norm[good,None]
    area = abs(np.linalg.det(dm))/2
    centers = ref.mean(axis=1)
    roi = (abs(centers[:,0]) <= .08) & (abs(centers[:,1]) <= .12)
    def stats(mask):
        use = mask & good
        weights = area[use]
        if not len(weights):
            return {'valid_area_fraction':0}
        weak_k = np.min(abs(k[use]), axis=1)
        strong_k = np.max(abs(k[use]), axis=1)
        avg = lambda x: float(np.average(x, weights=weights))
        return {'valid_area_fraction':float(weights.sum()/area[mask].sum()),
                'mean_abs_weak_m_inv':avg(weak_k), 'mean_abs_strong_m_inv':avg(strong_k),
                'weak_direction_mean_u_squared':avg(material[use,0]**2),
                'opposite_sign_fraction':avg(np.prod(k[use], axis=1)<0),
                'weak_below_point1_fraction':avg(weak_k<.1)}
    return {name:stats(mask) for name,mask in
            [('whole_sheet', np.ones(len(faces),bool)), ('central_roi',roi),
             ('middle',roi & (abs(centers[:,1])<=.074)),
             ('gaps',roi & (abs(centers[:,1])>.074) & (abs(centers[:,1])<.0866)),
             ('side_regions',roi & (abs(centers[:,1])>=.0866))]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fields')
    args = parser.parse_args()
    print(json.dumps(summarize(args.fields),indent=2,allow_nan=False))
