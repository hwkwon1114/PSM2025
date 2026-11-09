#!/usr/bin/env python3
"""Exploratory fixed-correspondence 2-D adaptation of Demisse et al.'s code.

Uses the authors' SE(2) segment representation and product distance, but keeps
physical x correspondence and performs no rotation/reflection optimization.
See the accompanying report for limitations and source provenance.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'calibration/paper_figures/convention_corrected/calibration_transfer_sections.npz'
OUTPUT = ROOT / 'scratch/curve_metric_pilot/results.json'


def normalize(p):
    centered = p - np.mean(p, axis=0)
    return centered / np.linalg.norm(centered)


def representations(points):
    p = normalize(np.asarray(points, float))
    p = p - p[0] + np.array([1., 0.])  # authors' default start point
    out = []
    for a, b in zip(p[:-1], p[1:]):
        _, _, vt = np.linalg.svd(np.vstack([a, b]) - (a+b)/2, full_matrices=True)
        v = vt.T
        p1, p2 = v.T @ a, v.T @ b
        u, _, w = np.linalg.svd(np.outer(p1, p2))
        vv = w.T
        e = np.diag([1., np.linalg.det(vv @ u.T)])
        rot = v @ (vv @ e @ u.T) @ v.T
        trans = b - rot @ a
        out.append((rot, trans))
    return out


def distance(a, b):
    aa, bb = representations(a), representations(b)
    if len(aa) != len(bb):
        raise ValueError('equal sampling required')
    total = 0.
    for (r1, t1), (r2, t2) in zip(aa, bb):
        q = r1.T @ r2
        angle = np.arctan2(q[1, 0] - q[0, 1], q[0, 0] + q[1, 1])
        total += angle**2 + np.sum((t1-t2)**2)
    return float(np.sqrt(total))


def main():
    # Source data sampled at 0.5 mm. Find the longest jointly supported interval;
    # do not interpolate over gaps. Resample each supported interval uniformly in x.
    data = np.load(INPUT, allow_pickle=False)
    x = data['x_mm']
    out = []
    for run in ('run6', 'run9', 'run10'):
        for z in (-35, 65, 165):
            a = data[f'{run}_experiment_y_z{z}']
            b = data[f'{run}_analytical_y_z{z}']
            good = np.isfinite(a) & np.isfinite(b)
            groups = np.split(np.flatnonzero(good), np.where(np.diff(np.flatnonzero(good)) != 1)[0]+1)
            group = max(groups, key=len)
            if len(group) < 100:
                raise ValueError(f'insufficient contiguous support: {run} {z}')
            xs = np.linspace(x[group[0]], x[group[-1]], 101)
            av, bv = np.interp(xs, x[group], a[group]), np.interp(xs, x[group], b[group])
            p, q = np.c_[xs, av], np.c_[xs, bv]
            out.append(dict(run=run, z_mm=z, supported_interval_mm=[float(xs[0]),float(xs[-1])],
                            joint_samples=int(good.sum()), interval_samples=len(group),
                            curve_distance=distance(p,q), correlation=float(np.corrcoef(av,bv)[0,1]),
                            experiment_ptv_mm=float(np.ptp(av)), model_ptv_mm=float(np.ptp(bv)),
                            mean_height_difference_mm=float(np.mean(bv-av)),
                            distance_51=distance(p[::2],q[::2])))
    t=np.linspace(-1,1,101); p=np.c_[t,0.2*t*t]
    angle=.4; rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    checks=dict(identity=distance(p,p), translated=distance(p,p+[3.,7.]),
                isotropic_scaled=distance(p,2*p), rotated_without_alignment=distance(p,p@rotation.T),
                inverted_bending=distance(p,np.c_[t,-.2*t*t]),
                anisotropic_height_scale=distance(p,np.c_[t,.4*t*t]),
                symmetry_difference=abs(distance(p,np.c_[t,.4*t*t])-distance(np.c_[t,.4*t*t],p)))
    assert checks['identity'] < 1e-10 and checks['translated'] < 1e-10
    assert checks['isotropic_scaled'] < 1e-10 and checks['symmetry_difference'] < 1e-10
    assert checks['inverted_bending'] > checks['identity'] and checks['anisotropic_height_scale'] > 1e-8
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    result=dict(input=str(INPUT.relative_to(ROOT)), input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                method='SE(2) authors-code segment distance, fixed uniform x correspondence, no pose fitting, no DP; 101 points',
                synthetic_checks=checks, sections=out)
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
