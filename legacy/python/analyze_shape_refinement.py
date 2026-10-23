#!/usr/bin/env python3
"""Read saved float64 shape exports. No solver, fitting or numerical gate changes.

Distances use a single proper area-weighted rigid alignment, fixed material
correspondence, reference-area RMS and inverse weighted-CDF P95. Maximum is
sampled, not a continuum supremum. Rejected endpoints are never comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def shape_metrics(a, b, weights):
    a, b, w = (np.asarray(v, dtype=np.float64) for v in (a, b, weights))
    if (a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape or
            w.shape != (len(a),) or len(a) < 3 or
            not all(np.isfinite(v).all() for v in (a, b, w)) or
            (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError('Invalid matched positions/area weights')
    w = w / w.sum()
    ca, cb = w @ a, w @ b
    u, _, vh = np.linalg.svd(((a-ca)*w[:, None]).T @ (b-cb))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    translation = cb-ca @ rotation
    distance = np.linalg.norm(a @ rotation+translation-b, axis=1)*1000
    positive = w > 0
    order = np.argsort(distance[positive], kind='stable')
    sorted_distance, sorted_weight = distance[positive][order], w[positive][order]
    cdf = np.cumsum(sorted_weight)
    cdf[-1] = 1.0
    tail = sorted_distance[np.searchsorted(cdf, .95, side='left')]
    return dict(rms_mm=float(np.sqrt(w @ (distance**2))),
                p95_mm=float(tail), sampled_max_mm=float(distance.max()),
                sampled_max_index=int(distance.argmax()), sample_count=len(a),
                rotation=rotation.tolist(), translation_m=translation.tolist(),
                distances_mm=distance.tolist())


def load_state(path):
    data = json.loads(Path(path).read_text())
    if data['schema_version'] != 1 or data['units'] != 'metres':
        raise ValueError('Unsupported export schema/units')
    proof = data['proof']
    if not (proof['accepted'] and proof['termination'] == 'gradient_target' and
            proof['curvature'] == 'positive_pivots' and
            np.isfinite(proof['energy']) and np.isfinite(proof['gradient_norm']) and
            0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= 1e-13 and
            proof['gradient_tolerance'] > 0 and
            proof['verification_evaluations'] >= 1 and proof['verification_hessians'] >= 1):
        raise ValueError('Unaccepted endpoint is not a shape-convergence observation')
    for key in ('positions', 'reference_positions', 'material_uv', 'selected_dofs',
                'abar_top', 'abar_bottom', 'bbar'):
        data[key] = np.asarray(data[key], dtype=np.float64)
        if not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite endpoint: '+key)
    f = np.asarray(data['faces'])
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('Invalid face connectivity')
    data['faces'] = f.astype(np.int64)
    nv = len(data['positions'])
    if (data['positions'].shape != (nv, 3) or data['reference_positions'].shape != (nv, 3)
            or data['material_uv'].shape != (nv, 2) or f.min() < 0 or f.max() >= nv
            or not np.array_equal(data['selected_dofs'].ravel()[:3*nv],
                                  data['positions'].ravel(order='F'))):
        raise ValueError('Malformed or unmatched selected state')
    return data


def same_mesh(a, b):
    for key in ('reference_positions', 'material_uv', 'faces', 'edges',
                'abar_top', 'abar_bottom', 'bbar', 'material'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError('Changed fixed objective/correspondence: '+key)
    ref, faces = a['reference_positions'], a['faces']
    area = np.linalg.norm(np.cross(ref[faces[:, 1]]-ref[faces[:, 0]],
                                   ref[faces[:, 2]]-ref[faces[:, 0]]), axis=1)/2
    if (area <= 0).any():
        raise ValueError('Degenerate reference face')
    weights = np.zeros(len(ref))
    for column in range(3):
        np.add.at(weights, faces[:, column], area/3)
    return shape_metrics(a['positions'], b['positions'], weights)


def interpolate(state, uv):
    # Use the actual connectivity, never a new Delaunay mesh of deformed points.
    from matplotlib.tri import Triangulation
    material, faces = state['material_uv'], state['faces']
    finder = Triangulation(material[:, 0], material[:, 1], triangles=faces).get_trifinder()
    face_ids = finder(uv[:, 0], uv[:, 1])
    if (face_ids < 0).any():
        raise ValueError('Uncovered material samples; extrapolation forbidden')
    vertices = faces[face_ids]
    base = material[vertices[:, 0]]
    basis = np.stack((material[vertices[:, 1]]-base,
                      material[vertices[:, 2]]-base), axis=2)
    bary = np.linalg.solve(basis, (uv-base)[..., None])[..., 0]
    if (bary < -1e-10).any() or (bary.sum(axis=1) > 1+1e-10).any():
        raise ValueError('Invalid material interpolation weights')
    xyz = state['positions'][vertices]
    return xyz[:, 0]+bary[:, :1]*(xyz[:, 1]-xyz[:, 0])+bary[:, 1:]*(xyz[:, 2]-xyz[:, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('states', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if len(args.states) < 3:
        parser.error('At least three ordered refinements required')
    if args.output.exists():
        raise FileExistsError('Preserve previous analysis')
    records = [load_state(p) for p in args.states]
    pairs = sorted(set([(i, i+1) for i in range(len(records)-1)]+
                       [(i, len(records)-1) for i in range(len(records)-1)]))
    comparisons = [dict(a=str(args.states[i]), b=str(args.states[j]),
                        **same_mesh(records[i], records[j])) for i, j in pairs]
    result = dict(schema_version=1, scope='same-mesh tolerance refinement, accepted endpoints only',
                  comparisons=comparisons,
                  inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.states},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numpy=np.__version__,
                  limits=['No absolute shape-error bound or convergence/optimality certificate invented.',
                          'Same-mesh reference-area vertex quadrature; spatial samples are dependent.',
                          'No cross-mesh claim or gate relaxation; failed cells must remain in the run ledger.'])
    with args.output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
