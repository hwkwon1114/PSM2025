"""Shape descriptors for the double-curvature family (pre-specified in the protocol).

Inputs are saved by run_case.py: flat reference vertices and deformed vertices already
rigidly fitted to that reference (calibration `remove_solver_rigid_motion`), in mm.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dome_bo.dome_curvature import geometric_curvature

TREATED = (100.0, 52.0)       # |u|, |v| half-extents of the treated rectangle (mm)
TOL = 1e-3                    # mm; inclusive boundary tolerance, far below the 1-2 mm vertex spacing
RATIO = 0.2                   # min |kappa_min|/|kappa_max| for a dome/saddle label


def height(aligned_mm):
    """Height (public y) on the material vertex grid, mean removed."""
    h = np.asarray(aligned_mm, float)[:, 2]
    return h - h.mean()


def treated(reference_mm):
    r = np.asarray(reference_mm, float)
    return (np.abs(r[:, 0]) <= TREATED[0] + TOL) & (np.abs(r[:, 1]) <= TREATED[1] + TOL)


def quadratic_fit(reference_mm, h):
    """Least-squares quadric over the treated rectangle; curvatures in 1/mm (small-slope)."""
    r = np.asarray(reference_mm, float)
    m = treated(r)
    u, v, y = r[m, 0], r[m, 1], np.asarray(h, float)[m]
    design = np.column_stack((np.ones_like(u), u, v, u * u, u * v, v * v))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b, c = coef[3], coef[4], coef[5]
    hess = np.array([[2 * a, b], [b, 2 * c]])
    kappa = np.linalg.eigvalsh(hess)
    gauss = float(4 * a * c - b * b)
    big, small = np.max(np.abs(kappa)), np.min(np.abs(kappa))
    ratio = float(small / big) if big > 0 else 0.0
    if ratio >= RATIO and gauss > 0:
        label = 'dome-like'
    elif ratio >= RATIO and gauss < 0:
        label = 'saddle-like'
    else:
        label = 'cylinder-like'
    resid = y - design @ coef
    return {'coefficients': coef.tolist(), 'kappa_per_mm': kappa.tolist(), 'K_fit_per_mm2': gauss,
            'ratio': ratio, 'label': label, 'rms_residual_mm': float(np.sqrt(np.mean(resid ** 2)))}


def local_curvature(reference_mm, aligned_mm, faces):
    """15 mm local fits (reviewed dome_bo helper, SI units); area fractions of K sign in the treated rectangle."""
    ref = np.asarray(reference_mm, float)
    values, _, valid = geometric_curvature(np.asarray(aligned_mm, float) / 1000, ref[:, :2] / 1000, faces)
    tri = ref[faces][:, :, :2]
    d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / 2
    centers = tri.mean(axis=1)
    roi = (np.abs(centers[:, 0]) <= TREATED[0] + TOL) & (np.abs(centers[:, 1]) <= TREATED[1] + TOL)
    gauss = values[:, 0] * values[:, 1]                     # 1/m^2
    total = area[roi].sum()
    return {'gauss_per_m2': gauss, 'valid': valid,
            'positive_area_fraction': float(area[roi & valid & (gauss > 0)].sum() / total),
            'negative_area_fraction': float(area[roi & valid & (gauss < 0)].sum() / total),
            'invalid_area_fraction': float(area[roi & ~valid].sum() / total),
            'median_gauss_per_m2': float(np.median(gauss[roi & valid])) if (roi & valid).any() else float('nan')}


def pca(fields, order):
    """PCA of height fields (rows); PC signs fixed so scores correlate positively with `order`."""
    X = np.asarray(fields, float)
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    scores = U * S
    order = np.asarray(order, float)
    for i in range(len(S)):
        if np.std(scores[:, i]) > 0 and np.corrcoef(scores[:, i], order)[0, 1] < 0:
            scores[:, i] *= -1
            Vt[i] *= -1
    var = S ** 2
    return {'mean': mean, 'components': Vt, 'scores': scores,
            'explained_ratio': var / var.sum() if var.sum() > 0 else var}
