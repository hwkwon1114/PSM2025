"""Direction-first rigid gauge and symmetric shape score; no scan-fitted pose gates."""
import numpy as np
from .geometry import (SurfaceDistance, compute_bidirectional_metrics, points,
                       reference_dual_areas, rigid_fit, set_points)

METHOD = 'material_reference'


def direction_frame(p, reference, faces):
    """Same proper matched-vertex frame as the frozen direction-first gallery."""
    p, reference, faces = np.asarray(p, float), np.asarray(reference, float), np.asarray(faces)
    if p.shape != reference.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Matched final/reference vertices required')
    if not np.isfinite(p).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite frame input')
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1 or
            not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0) or np.any(faces >= len(p))):
        raise ValueError('Invalid material-reference topology')
    covariance = (p - p.mean(0)).T @ (reference - reference.mean(0))
    singular = np.linalg.svd(covariance, compute_uv=False)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-10:
        raise ValueError('Material direction frame is degenerate')
    rotation, _ = rigid_fit(p, reference)
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0) or
            abs(np.linalg.det(rotation) - 1) >= 1e-12):
        raise ValueError('Material direction frame is not a proper rigid rotation')
    areas = reference_dual_areas(reference, faces)
    weights = areas / areas.sum()
    centroid = np.sum(weights[:, None] * p, axis=0)
    translation = -centroid @ rotation
    q = p @ rotation + translation
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    error = float(np.max(np.abs(np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1) -
                               np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1))))
    if error > 1e-9 or np.linalg.norm(np.sum(weights[:, None] * q, axis=0)) >= 1e-9:
        raise ValueError('Material frame changed shape or failed centroid removal')
    return q, {'row_vector_rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
               'centroid_original_mm': centroid.tolist(), 'singular_values': singular.tolist(),
               'direction_condition_ratio': float(singular[1] / singular[0]),
               'determinant': float(np.linalg.det(rotation)), 'max_edge_length_change_mm': error}


def score_material_frame(mesh, target, reference, faces):
    """Return metrics, transformed mesh, centered scan, distances and closest points."""
    target = np.asarray(target, float)
    if target.ndim != 2 or target.shape[1] != 3 or len(target) < 3 or not np.isfinite(target).all():
        raise ValueError('Invalid scan for material-reference scoring')
    q, transform = direction_frame(points(mesh), reference, faces)
    framed_mesh = set_points(mesh, q)
    scan_centroid = target.mean(0)
    aligned = target - scan_centroid
    distances, closest = SurfaceDistance(framed_mesh).closest(aligned)
    metrics = compute_bidirectional_metrics(framed_mesh, aligned, reference, faces, distances)
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite material-reference shape score')
    metrics.update(rms_mm=metrics['rms_symmetric_mm'], scoring_method=METHOD, shape_scored=True,
                   simulation_transform=transform, scan_centroid_original_mm=scan_centroid.tolist())
    return metrics, framed_mesh, aligned, distances, closest
