"""Geometry-only local curvature. SI units; native full-precision vertices required.

Neighborhoods are fixed in material coordinates (15 mm radius), not selected by
candidate error. Invalid faces stay in the coverage denominator.
"""
import numpy as np
from scipy.spatial import cKDTree

RADIUS = 0.015


def geometric_curvature(positions, uv, faces, radius=RADIUS):
    positions, uv = np.asarray(positions, float), np.asarray(uv, float)
    faces = np.asarray(faces, int)
    if not np.isfinite(positions).all() or not np.isfinite(uv).all():
        raise ValueError('Nonfinite coordinates')
    triangles = positions[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    centers = uv[faces].mean(axis=1)
    tree = cKDTree(uv)
    values = np.full((len(faces), 2), np.nan)
    directions = np.full((len(faces), 2, 3), np.nan)
    valid = np.zeros(len(faces), bool)
    for i, ids in enumerate(tree.query_ball_point(centers, radius)):
        if len(ids) < 12 or lengths[i] <= 1e-16:
            continue
        points = positions[ids] - triangles[i].mean(axis=0)
        _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        normal = axes[2]
        if normal @ normals[i] < 0:
            normal = -normal
        # Right-handed chart with a consistently mesh-oriented normal.
        tangent = axes[0]
        second = np.cross(normal, tangent)
        x, y, z = points @ tangent / radius, points @ second / radius, points @ normal / radius
        design = np.column_stack((np.ones(len(ids)), x, y, x*x/2, x*y, y*y/2))
        coeff, _, rank, singular = np.linalg.lstsq(design, z, rcond=None)
        if rank != 6 or singular[0]/singular[-1] > 1e6:
            continue
        _, p, q, a, b, c = coeff
        metric = np.array([[1+p*p, p*q], [p*q, 1+q*q]])
        curvature = np.array([[a, b], [b, c]]) / (radius*np.sqrt(1+p*p+q*q))
        # Symmetric generalized eigenproblem via Cholesky, not Hessian alone.
        lower = np.linalg.cholesky(metric)
        inv = np.linalg.inv(lower)
        eigenvalues, eigenvectors = np.linalg.eigh(inv @ curvature @ inv.T)
        coords = np.linalg.solve(lower.T, eigenvectors)
        basis = np.column_stack((tangent+p*normal, second+q*normal))
        physical = basis @ coords
        physical /= np.linalg.norm(physical, axis=0)
        values[i], directions[i], valid[i] = eigenvalues, physical.T, True
    return values, directions, valid


def coverage(values, valid, uv, faces, floor=0.1):
    """Separate coherent signs; invalid area is never removed."""
    values, valid, uv, faces = (np.asarray(x) for x in (values, valid, uv, faces))
    triangles = uv[faces]
    d1, d2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    area = abs(d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0])/2
    centers = triangles.mean(axis=1)
    roi = (abs(centers[:, 0]) <= .08) & (abs(centers[:, 1]) <= .12)
    denominator = area[roi].sum()
    if denominator <= 0:
        raise ValueError('Empty central region')
    small, large = np.min(abs(values), axis=1), np.max(abs(values), axis=1)
    eligible = valid & np.isfinite(values).all(axis=1) & (small >= floor) & (small >= .1*large)
    fractions = {str(sign): float(area[roi & eligible & (sign*values > 0).all(axis=1)].sum()/denominator)
                 for sign in (-1, 1)}
    return {'sign_fractions': fractions, 'qualifies': max(fractions.values()) >= .8,
            'valid_area_fraction': float(area[roi & valid].sum()/denominator)}
