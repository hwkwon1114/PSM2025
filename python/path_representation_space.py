"""Prototype a broad, physically band-limited trajectory representation space.

The module supports the companion notebook
``notebooks/path_representation_space.ipynb``.  It deliberately has no dependency on
the forming solver: the goal is to inspect path geometry, process channels, feasibility,
descriptor coverage, and initial-design selection before choosing a neural operator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wasserstein_distance


@dataclass(frozen=True)
class PathSample:
    """A resolution-independent path sampled densely for analysis and plotting."""

    xy: np.ndarray
    force: np.ndarray
    speed: np.ndarray
    source: str
    complexity: int
    label: str

    def reversed(self, label: str | None = None) -> "PathSample":
        """Return the same commanded path traversed in the opposite order."""

        return PathSample(
            xy=self.xy[::-1].copy(),
            force=self.force[::-1].copy(),
            speed=self.speed[::-1].copy(),
            source="reverse",
            complexity=self.complexity,
            label=label or f"reverse_{self.label}",
        )


DESCRIPTOR_NAMES = (
    "length",
    "duration",
    "curvature_mean",
    "curvature_p95",
    "signed_turn",
    "absolute_turn",
    "bbox_area",
    "grid_coverage",
    "revisit_fraction",
    "centroid_x",
    "centroid_y",
    "boundary_clearance",
    "closure_gap",
    "straightness",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "force_mean",
    "force_std",
    "force_start",
    "force_end",
    "speed_mean",
    "speed_std",
    "dose",
    "complexity",
)

RECTANGLE_SYMMETRIES = ("identity", "flip_x", "flip_y", "rotate_180")


def transform_path(path: PathSample, symmetry: str) -> PathSample:
    """Apply a centered-rectangle D2 symmetry without reversing time."""

    xy = path.xy.copy()
    if symmetry == "identity":
        pass
    elif symmetry == "flip_x":
        xy[:, 0] = 1.0 - xy[:, 0]
    elif symmetry == "flip_y":
        xy[:, 1] = 1.0 - xy[:, 1]
    elif symmetry == "rotate_180":
        xy = 1.0 - xy
    else:
        raise ValueError(f"Unknown rectangle symmetry: {symmetry}")
    return PathSample(
        xy=xy,
        force=path.force.copy(),
        speed=path.speed.copy(),
        source=path.source,
        complexity=path.complexity,
        label=f"{symmetry}_{path.label}",
    )


def symmetry_orbit(path: PathSample) -> list[PathSample]:
    return [transform_path(path, symmetry) for symmetry in RECTANGLE_SYMMETRIES]


def _clamped_bspline(control: np.ndarray, n_samples: int) -> np.ndarray:
    control = np.asarray(control, dtype=float)
    degree = min(3, len(control) - 1)
    interior_count = len(control) - degree - 1
    interior = (
        np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        if interior_count > 0
        else np.empty(0)
    )
    knots = np.concatenate(
        [np.zeros(degree + 1), interior, np.ones(degree + 1)]
    )
    parameter = np.linspace(0.0, 1.0, n_samples)
    return BSpline(knots, control, degree)(parameter)


def _process_profiles(
    rng: np.random.Generator, n_control: int, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    profile_control = max(4, min(7, n_control))
    force_control = rng.uniform(0.15, 1.0, profile_control)
    speed_control = rng.uniform(0.25, 1.0, profile_control)
    force = np.clip(
        _clamped_bspline(force_control[:, None], n_samples).ravel(), 0.05, 1.05
    )
    speed = np.clip(
        _clamped_bspline(speed_control[:, None], n_samples).ravel(), 0.15, 1.05
    )
    return force, speed


def _curvature(xy: np.ndarray) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, len(xy))
    d1 = np.gradient(xy, parameter, axis=0, edge_order=2)
    d2 = np.gradient(d1, parameter, axis=0, edge_order=2)
    numerator = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    denominator = np.linalg.norm(d1, axis=1) ** 3 + 1e-10
    return numerator / denominator


def _path_length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def is_feasible(path: PathSample) -> bool:
    """Apply deliberately broad geometric and process constraints."""

    if not all(
        np.all(np.isfinite(a)) for a in (path.xy, path.force, path.speed)
    ):
        return False
    if path.xy.min() < 0.0 or path.xy.max() > 1.0:
        return False
    length = _path_length(path.xy)
    if length < 0.05 or length > 7.0:
        return False
    if path.force.min() < 0.0 or path.force.max() > 1.1:
        return False
    if path.speed.min() < 0.1 or path.speed.max() > 1.1:
        return False
    # The percentile ignores isolated endpoint differentiation artifacts.
    if np.percentile(_curvature(path.xy), 99) > 180.0:
        return False
    return True


def make_freeform_path(
    rng: np.random.Generator,
    n_control: int | None = None,
    n_samples: int = 160,
    label: str = "freeform",
) -> PathSample:
    """Draw a smooth path without prescribing a human path topology."""

    n_control = int(n_control or rng.integers(4, 13))
    for _ in range(80):
        control = np.empty((n_control, 2), dtype=float)
        control[0] = rng.uniform(0.15, 0.85, 2)
        heading = rng.uniform(-np.pi, np.pi)
        for i in range(1, n_control):
            heading += rng.normal(0.0, 0.75)
            step = rng.uniform(0.06, 0.22)
            proposal = control[i - 1] + step * np.array(
                [np.cos(heading), np.sin(heading)]
            )
            if np.any(proposal < 0.06) or np.any(proposal > 0.94):
                heading += np.pi + rng.normal(0.0, 0.25)
                proposal = control[i - 1] + step * np.array(
                    [np.cos(heading), np.sin(heading)]
                )
            control[i] = np.clip(proposal, 0.07, 0.93)

        xy = _clamped_bspline(control, n_samples)
        force, speed = _process_profiles(rng, n_control, n_samples)
        path = PathSample(xy, force, speed, "freeform", n_control, label)
        if is_feasible(path):
            return path
    raise RuntimeError("Unable to generate a feasible free-form path")


def _rotate_translate_to_unit_square(
    local_xy: np.ndarray, rng: np.random.Generator, margin: float = 0.06
) -> np.ndarray:
    angle = rng.uniform(-np.pi, np.pi)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = local_xy @ rotation.T
    span = np.ptp(rotated, axis=0)
    allowed = max(1e-6, 1.0 - 2.0 * margin)
    if span.max() > allowed:
        rotated *= allowed / span.max()
        span = np.ptp(rotated, axis=0)
    lower = margin - rotated.min(axis=0)
    upper = 1.0 - margin - rotated.max(axis=0)
    shift = rng.uniform(lower, upper)
    return rotated + shift


def make_primitive_path(
    kind: str,
    rng: np.random.Generator,
    n_samples: int = 160,
    label: str | None = None,
) -> PathSample:
    """Generate a human-readable seed family without restricting later search."""

    parameter = np.linspace(0.0, 1.0, n_samples)
    if kind == "line":
        length = rng.uniform(0.15, 0.85)
        local = np.column_stack((length * (parameter - 0.5), np.zeros(n_samples)))
        complexity = 2
    elif kind == "arc":
        radius = rng.uniform(0.10, 0.36)
        span = rng.uniform(0.35 * np.pi, 1.8 * np.pi)
        angle = np.linspace(-0.5 * span, 0.5 * span, n_samples)
        local = radius * np.column_stack((np.cos(angle), np.sin(angle)))
        complexity = 4
    elif kind == "spiral":
        turns = rng.uniform(0.55, 2.6)
        angle = 2.0 * np.pi * turns * parameter
        radius = np.linspace(rng.uniform(0.02, 0.08), rng.uniform(0.18, 0.38), n_samples)
        local = radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
        complexity = int(np.ceil(4 * turns))
    elif kind == "zigzag":
        n_turns = int(rng.integers(2, 9))
        length = rng.uniform(0.35, 0.90)
        amplitude = rng.uniform(0.04, 0.23)
        x = length * (parameter - 0.5)
        y = amplitude * np.sin(np.pi * n_turns * parameter)
        local = np.column_stack((x, y))
        complexity = n_turns + 2
    else:
        raise ValueError(f"Unknown primitive kind: {kind}")

    xy = _rotate_translate_to_unit_square(local, rng)
    force, speed = _process_profiles(rng, max(4, complexity), n_samples)
    return PathSample(
        xy=xy,
        force=force,
        speed=speed,
        source=kind,
        complexity=complexity,
        label=label or kind,
    )


def describe_path(path: PathSample, grid_size: int = 20) -> np.ndarray:
    xy = path.xy
    segments = np.diff(xy, axis=0)
    ds = np.linalg.norm(segments, axis=1)
    length = float(ds.sum())
    midpoint_speed = 0.5 * (path.speed[:-1] + path.speed[1:])
    duration = float(np.sum(ds / np.maximum(midpoint_speed, 1e-6)))
    curvature = _curvature(xy)
    angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    turns = np.diff(angles)
    span = np.ptp(xy, axis=0)
    cells = np.floor(np.clip(xy, 0.0, 1.0 - 1e-12) * grid_size).astype(int)
    unique_cells = len(np.unique(cells, axis=0))
    boundary_clearance = float(
        np.min(np.column_stack((xy, 1.0 - xy)))
    )
    closure = float(np.linalg.norm(xy[-1] - xy[0]))
    dose = float(np.sum(0.5 * (path.force[:-1] + path.force[1:]) * ds))

    return np.array(
        [
            length,
            duration,
            float(np.mean(curvature)),
            float(np.percentile(curvature, 95)),
            float(turns.sum()) if len(turns) else 0.0,
            float(np.abs(turns).sum()) if len(turns) else 0.0,
            float(span.prod()),
            unique_cells / float(grid_size**2),
            1.0 - unique_cells / float(len(xy)),
            float(xy[:, 0].mean()),
            float(xy[:, 1].mean()),
            boundary_clearance,
            closure,
            closure / max(length, 1e-8),
            float(xy[0, 0]),
            float(xy[0, 1]),
            float(xy[-1, 0]),
            float(xy[-1, 1]),
            float(path.force.mean()),
            float(path.force.std()),
            float(path.force[0]),
            float(path.force[-1]),
            float(path.speed.mean()),
            float(path.speed.std()),
            dose,
            float(path.complexity),
        ],
        dtype=float,
    )


def descriptor_matrix(paths: Sequence[PathSample]) -> np.ndarray:
    return np.vstack([describe_path(path) for path in paths])


def standardize_descriptors(
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(descriptors, axis=0)
    q25, q75 = np.percentile(descriptors, [25, 75], axis=0)
    scale = q75 - q25
    fallback = descriptors.std(axis=0)
    scale = np.where(scale > 1e-10, scale, np.where(fallback > 1e-10, fallback, 1.0))
    return (descriptors - center) / scale, center, scale


def pca_embedding(standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = standardized - standardized.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ vt[:2].T
    explained = singular**2 / max(float(np.sum(singular**2)), 1e-12)
    return embedding, explained[:2]


def canonicalize_path(
    path: PathSample, n_samples: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample geometry and process channels on normalized cumulative arc length.

    This removes dependence on the original numerical discretization without erasing
    traversal direction.  Force and speed remain ordered channels along the path.
    """

    segment_length = np.linalg.norm(np.diff(path.xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    total = cumulative[-1]
    if total <= 1e-12:
        raise ValueError("Cannot canonicalize a zero-length path")
    cumulative /= total
    target = np.linspace(0.0, 1.0, n_samples)
    xy = np.column_stack(
        [np.interp(target, cumulative, path.xy[:, dim]) for dim in range(2)]
    )
    force = np.interp(target, cumulative, path.force)
    speed = np.interp(target, cumulative, path.speed)
    return xy, force, speed


def canonical_input_features(
    path: PathSample,
    n_samples: int = 64,
    tangent_weight: float = 0.15,
    force_weight: float = 0.50,
    speed_weight: float = 0.30,
    duration_weight: float = 0.35,
    dose_weight: float = 0.50,
) -> np.ndarray:
    """Encode the exact ordered function presented to a future surrogate.

    Sample-wise terms are scaled by ``1/sqrt(n_samples)`` so Euclidean distance
    approximates an integral path metric rather than growing with discretization.
    Global length, duration, and dose keep paths with the same normalized trace but
    different physical scale or process history distinct.
    """

    xy, force, speed = canonicalize_path(path, n_samples=n_samples)
    parameter = np.linspace(0.0, 1.0, n_samples)
    tangent = np.gradient(xy, parameter, axis=0, edge_order=2)
    scale = 1.0 / np.sqrt(n_samples)
    ordered = np.column_stack(
        (
            xy,
            np.sqrt(tangent_weight) * tangent,
            np.sqrt(force_weight) * force,
            np.sqrt(speed_weight) * speed,
        )
    ).ravel()
    ordered *= scale

    segments = np.diff(xy, axis=0)
    ds = np.linalg.norm(segments, axis=1)
    length = float(ds.sum())
    midpoint_speed = 0.5 * (speed[:-1] + speed[1:])
    duration = float(np.sum(ds / np.maximum(midpoint_speed, 1e-8)))
    dose = float(np.sum(0.5 * (force[:-1] + force[1:]) * ds))
    globals_ = np.array(
        [length, duration_weight * duration, dose_weight * dose], dtype=float
    )
    return np.concatenate((ordered, globals_))


def canonical_input_matrix(
    paths: Sequence[PathSample], n_samples: int = 64
) -> np.ndarray:
    return np.vstack(
        [canonical_input_features(path, n_samples=n_samples) for path in paths]
    )


def pairwise_squared_distances(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    norms = np.sum(features * features, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * features @ features.T
    return np.maximum(distances, 0.0)


def rbf_kernel(
    features: np.ndarray, bandwidth: float | None = None
) -> tuple[np.ndarray, float]:
    """Return a Gaussian kernel using a median-distance bandwidth by default."""

    distances2 = pairwise_squared_distances(features)
    if bandwidth is None:
        positive = distances2[np.triu_indices_from(distances2, k=1)]
        positive = positive[positive > 1e-14]
        if len(positive) == 0:
            bandwidth = 1.0
        else:
            bandwidth = float(np.sqrt(np.median(positive)))
    kernel = np.exp(-0.5 * distances2 / max(bandwidth**2, 1e-14))
    return kernel, float(bandwidth)


def kernel_d_optimal(
    kernel: np.ndarray, batch_size: int, jitter: float = 1e-10
) -> np.ndarray:
    """Greedy kernel D-optimal design via pivoted Cholesky residual variance."""

    kernel = np.asarray(kernel, dtype=float)
    n_paths = len(kernel)
    if kernel.shape != (n_paths, n_paths):
        raise ValueError("kernel must be square")
    if not 0 < batch_size <= n_paths:
        raise ValueError("batch_size must be between one and the pool size")

    factors = np.zeros((n_paths, batch_size), dtype=float)
    residual = np.maximum(np.diag(kernel).copy(), 0.0)
    selected: list[int] = []
    for column in range(batch_size):
        pivot = int(np.argmax(residual))
        selected.append(pivot)
        pivot_variance = max(residual[pivot], jitter)
        if column == 0:
            correction = kernel[:, pivot]
        else:
            correction = kernel[:, pivot] - factors[:, :column] @ factors[
                pivot, :column
            ]
        factors[:, column] = correction / np.sqrt(pivot_variance)
        residual = np.maximum(residual - factors[:, column] ** 2, 0.0)
        residual[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def mmd_to_pool(kernel: np.ndarray, selected: Sequence[int]) -> float:
    """Biased empirical MMD between a selected design and the candidate pool."""

    selected = np.asarray(selected, dtype=int)
    all_indices = np.arange(len(kernel))
    value = (
        kernel[np.ix_(selected, selected)].mean()
        - 2.0 * kernel[np.ix_(selected, all_indices)].mean()
        + kernel.mean()
    )
    return float(np.sqrt(max(value, 0.0)))


def kernel_effective_rank(kernel: np.ndarray, tolerance: float = 1e-12) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(kernel), 0.0)
    eigenvalues = eigenvalues[eigenvalues > tolerance]
    if not len(eigenvalues):
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def representation_collision_pairs(
    descriptor_features: np.ndarray,
    input_features: np.ndarray,
    n_pairs: int = 4,
) -> dict[str, list[tuple[int, int, float, float]]]:
    """Find descriptor/input neighborhoods that disagree most strongly."""

    descriptor_distance = np.sqrt(pairwise_squared_distances(descriptor_features))
    input_distance = np.sqrt(pairwise_squared_distances(input_features))
    upper = np.triu_indices(len(descriptor_features), k=1)
    descriptor_scale = max(float(np.median(descriptor_distance[upper])), 1e-12)
    input_scale = max(float(np.median(input_distance[upper])), 1e-12)
    descriptor_normalized = descriptor_distance / descriptor_scale
    input_normalized = input_distance / input_scale

    def ranked_pairs(
        neighborhood: np.ndarray, disagreement: np.ndarray
    ) -> list[tuple[int, int, float, float]]:
        threshold = np.quantile(neighborhood[upper], 0.10)
        mask = neighborhood[upper] <= threshold
        candidate_positions = np.flatnonzero(mask)
        order = candidate_positions[
            np.argsort(disagreement[upper][mask])[::-1]
        ]
        result = []
        for position in order[:n_pairs]:
            i, j = upper[0][position], upper[1][position]
            result.append(
                (
                    int(i),
                    int(j),
                    float(descriptor_normalized[i, j]),
                    float(input_normalized[i, j]),
                )
            )
        return result

    return {
        "descriptor_near_input_far": ranked_pairs(
            descriptor_normalized, input_normalized
        ),
        "input_near_descriptor_far": ranked_pairs(
            input_normalized, descriptor_normalized
        ),
    }


def metric_benchmark_subset(
    paths: Sequence[PathSample], seed: int = 31
) -> np.ndarray:
    """Return a reproducible 96-path subset for quadratic-time metrics."""

    rng = np.random.default_rng(seed)
    quotas = {
        "freeform": 36,
        "line": 12,
        "arc": 12,
        "spiral": 12,
        "zigzag": 12,
        "reverse": 12,
    }
    selected: list[int] = []
    for source, quota in quotas.items():
        members = np.asarray(
            [i for i, path in enumerate(paths) if path.source == source], dtype=int
        )
        selected.extend(
            rng.choice(members, size=min(quota, len(members)), replace=False).tolist()
        )
    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)


def augmented_trajectory(path: PathSample, n_samples: int = 32) -> np.ndarray:
    """Ordered curve for DTW/Fréchet with spatial, temporal, and process channels."""

    xy, force, speed = canonicalize_path(path, n_samples=n_samples)
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    midpoint_speed = 0.5 * (speed[:-1] + speed[1:])
    time = np.concatenate(
        ([0.0], np.cumsum(ds / np.maximum(midpoint_speed, 1e-8)))
    )
    return np.column_stack(
        (xy, 0.25 * time, 0.50 * force, 0.30 * speed)
    )


def discrete_frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    costs = cdist(curve_a, curve_b)
    dynamic = np.empty_like(costs)
    dynamic[0, 0] = costs[0, 0]
    for i in range(1, len(curve_a)):
        dynamic[i, 0] = max(dynamic[i - 1, 0], costs[i, 0])
    for j in range(1, len(curve_b)):
        dynamic[0, j] = max(dynamic[0, j - 1], costs[0, j])
    for i in range(1, len(curve_a)):
        for j in range(1, len(curve_b)):
            dynamic[i, j] = max(
                costs[i, j],
                min(
                    dynamic[i - 1, j],
                    dynamic[i - 1, j - 1],
                    dynamic[i, j - 1],
                ),
            )
    return float(dynamic[-1, -1])


def constrained_dtw_distance(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    window_fraction: float = 0.15,
) -> float:
    costs = cdist(sequence_a, sequence_b)
    n_a, n_b = costs.shape
    window = max(abs(n_a - n_b), int(np.ceil(window_fraction * max(n_a, n_b))))
    dynamic = np.full((n_a + 1, n_b + 1), np.inf, dtype=float)
    dynamic[0, 0] = 0.0
    for i in range(1, n_a + 1):
        lower = max(1, i - window)
        upper = min(n_b, i + window)
        for j in range(lower, upper + 1):
            dynamic[i, j] = costs[i - 1, j - 1] + min(
                dynamic[i - 1, j],
                dynamic[i, j - 1],
                dynamic[i - 1, j - 1],
            )
    return float(dynamic[n_a, n_b] / max(n_a + n_b, 1))


def path_signature_features(
    path: PathSample, n_samples: int = 32
) -> np.ndarray:
    """Truncated level-two signature with absolute-position/process anchors."""

    curve = augmented_trajectory(path, n_samples=n_samples)
    increments = np.diff(curve, axis=0)
    level_one = np.zeros(curve.shape[1], dtype=float)
    level_two = np.zeros((curve.shape[1], curve.shape[1]), dtype=float)
    for increment in increments:
        level_two += np.outer(level_one, increment)
        level_two += 0.5 * np.outer(increment, increment)
        level_one += increment
    length = _path_length(path.xy)
    dose = float(
        np.sum(
            0.5 * (path.force[:-1] + path.force[1:])
            * np.linalg.norm(np.diff(path.xy, axis=0), axis=1)
        )
    )
    return np.concatenate(
        (curve[0], curve[-1], level_one, level_two.ravel(), [length, dose])
    )


def sliced_occupation_distance(
    path_a: PathSample,
    path_b: PathSample,
    n_samples: int = 32,
    n_directions: int = 8,
) -> float:
    """Approximate Wasserstein distance between force-weighted path occupations."""

    xy_a, force_a, _ = canonicalize_path(path_a, n_samples=n_samples)
    xy_b, force_b, _ = canonicalize_path(path_b, n_samples=n_samples)
    weight_a = np.maximum(force_a, 1e-6)
    weight_b = np.maximum(force_b, 1e-6)
    distances = []
    for angle in np.linspace(0.0, np.pi, n_directions, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection_a = xy_a @ direction
        projection_b = xy_b @ direction
        distances.append(
            wasserstein_distance(
                projection_a,
                projection_b,
                u_weights=weight_a,
                v_weights=weight_b,
            )
        )
    return float(np.sqrt(np.mean(np.square(distances))))


def pairwise_custom_distance(
    paths: Sequence[PathSample],
    distance_function,
) -> np.ndarray:
    n_paths = len(paths)
    distances = np.zeros((n_paths, n_paths), dtype=float)
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            value = distance_function(paths[i], paths[j])
            distances[i, j] = distances[j, i] = value
    return distances


def synthetic_history_response(
    path: PathSample,
    grid_size: int = 14,
    n_samples: int = 64,
) -> np.ndarray:
    """A cheap spatial oracle with hardening, overlap, and order-dependent feedback.

    This is a controlled metric benchmark, not a calibrated forming model.  Its role is
    to make response informativeness testable before connecting the real shell solver.
    """

    xy, force, speed = canonicalize_path(path, n_samples=n_samples)
    axis = np.linspace(0.0, 1.0, grid_size)
    gx, gy = np.meshgrid(axis, axis, indexing="xy")
    hardening = np.zeros_like(gx)
    strain_x = np.zeros_like(gx)
    strain_y = np.zeros_like(gx)
    strain_xy = np.zeros_like(gx)
    global_state = np.zeros(2, dtype=float)
    footprint_width = 0.075

    for step in range(n_samples - 1):
        delta = xy[step + 1] - xy[step]
        ds = float(np.linalg.norm(delta))
        if ds <= 1e-12:
            continue
        tangent = delta / ds
        footprint = np.exp(
            -(
                (gx - xy[step, 0]) ** 2
                + (gy - xy[step, 1]) ** 2
            )
            / (2.0 * footprint_width**2)
        )
        local_hardening = float(
            np.sum(footprint * hardening) / max(np.sum(footprint), 1e-12)
        )
        base_dose = force[step] * ds / max(0.20 + speed[step], 1e-8)
        directional_memory = np.tanh(
            1.8 * (global_state[0] * tangent[1] - global_state[1] * tangent[0])
        )
        efficiency = np.exp(-0.65 * local_hardening) * (
            1.0 + 0.30 * directional_memory
        )
        increment = base_dose * efficiency * footprint
        strain_x += increment * tangent[0] ** 2
        strain_y += increment * tangent[1] ** 2
        strain_xy += increment * tangent[0] * tangent[1]
        hardening += base_dose * footprint
        global_state = 0.985 * global_state + base_dose * np.array(
            [tangent[0], tangent[1]]
        )

    moment_x = float(np.sum((gy - 0.5) * strain_x))
    moment_y = float(np.sum((gx - 0.5) * strain_y))
    cross_moment = float(np.sum((gx - 0.5) * (gy - 0.5) * strain_xy))
    globals_ = np.array(
        [
            moment_x,
            moment_y,
            cross_moment,
            float(hardening.mean()),
            float(hardening.max()),
            *global_state,
        ]
    )
    return np.concatenate(
        (strain_x.ravel(), strain_y.ravel(), strain_xy.ravel(), hardening.ravel(), globals_)
    )


def synthetic_history_response_matrix(paths: Sequence[PathSample]) -> np.ndarray:
    return np.vstack([synthetic_history_response(path) for path in paths])


def distance_matrix_from_features(features: np.ndarray) -> np.ndarray:
    return np.sqrt(pairwise_squared_distances(features))


def normalize_distance_matrix(distances: np.ndarray) -> np.ndarray:
    upper = distances[np.triu_indices_from(distances, k=1)]
    scale = max(float(np.median(upper[upper > 1e-14])), 1e-12)
    return distances / scale


def greedy_maximin_distance(distances: np.ndarray, batch_size: int) -> np.ndarray:
    """Farthest-first traversal using a precomputed distance matrix."""

    first = int(np.argmax(distances.mean(axis=1)))
    selected = [first]
    minimum = distances[:, first].copy()
    minimum[first] = -np.inf
    while len(selected) < batch_size:
        nxt = int(np.argmax(minimum))
        selected.append(nxt)
        minimum = np.minimum(minimum, distances[:, nxt])
        minimum[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def neighborhood_overlap(
    metric_distance: np.ndarray,
    response_distance: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    overlaps = []
    for index in range(len(metric_distance)):
        metric_neighbors = set(
            np.argsort(metric_distance[index])[1 : n_neighbors + 1]
        )
        response_neighbors = set(
            np.argsort(response_distance[index])[1 : n_neighbors + 1]
        )
        overlaps.append(len(metric_neighbors & response_neighbors) / n_neighbors)
    return float(np.mean(overlaps))


def triplet_agreement(
    metric_distance: np.ndarray,
    response_distance: np.ndarray,
    seed: int = 41,
    n_triplets: int = 5000,
) -> float:
    rng = np.random.default_rng(seed)
    agreement = 0
    valid = 0
    for _ in range(n_triplets):
        anchor, first, second = rng.choice(
            len(metric_distance), size=3, replace=False
        )
        metric_sign = np.sign(
            metric_distance[anchor, first] - metric_distance[anchor, second]
        )
        response_sign = np.sign(
            response_distance[anchor, first]
            - response_distance[anchor, second]
        )
        if metric_sign == 0 or response_sign == 0:
            continue
        agreement += metric_sign == response_sign
        valid += 1
    return float(agreement / max(valid, 1))


def knn_response_error(
    metric_distance: np.ndarray,
    responses_z: np.ndarray,
    selected: Sequence[int],
    n_neighbors: int = 4,
) -> float:
    selected = np.asarray(selected, dtype=int)
    test = np.setdiff1d(np.arange(len(metric_distance)), selected)
    predictions = []
    for index in test:
        nearest = selected[
            np.argsort(metric_distance[index, selected])[:n_neighbors]
        ]
        local_distance = metric_distance[index, nearest]
        weights = 1.0 / np.maximum(local_distance, 1e-6)
        weights /= weights.sum()
        predictions.append(weights @ responses_z[nearest])
    predictions = np.asarray(predictions)
    return float(np.sqrt(np.mean((predictions - responses_z[test]) ** 2)))


def evaluate_trajectory_metrics(
    paths: Sequence[PathSample],
    batch_size: int = 24,
    seed: int = 31,
) -> dict[str, object]:
    """Benchmark trajectory metrics against a controlled history-dependent response."""

    subset_indices = metric_benchmark_subset(paths, seed=seed)
    subset = [paths[i] for i in subset_indices]
    runtimes: dict[str, float] = {}
    distances: dict[str, np.ndarray] = {}

    start = perf_counter()
    canonical = canonical_input_matrix(subset, n_samples=32)
    distances["canonical H1"] = distance_matrix_from_features(canonical)
    runtimes["canonical H1"] = perf_counter() - start

    start = perf_counter()
    descriptor = descriptor_matrix(subset)
    descriptor_z, _, _ = standardize_descriptors(descriptor)
    distances["descriptors"] = distance_matrix_from_features(descriptor_z)
    runtimes["descriptors"] = perf_counter() - start

    augmented = [augmented_trajectory(path, n_samples=32) for path in subset]
    start = perf_counter()
    distances["weighted Frechet"] = pairwise_custom_distance(
        augmented, discrete_frechet_distance
    )
    runtimes["weighted Frechet"] = perf_counter() - start

    start = perf_counter()
    distances["constrained DTW"] = pairwise_custom_distance(
        augmented, constrained_dtw_distance
    )
    runtimes["constrained DTW"] = perf_counter() - start

    start = perf_counter()
    distances["occupation SW"] = pairwise_custom_distance(
        subset, sliced_occupation_distance
    )
    runtimes["occupation SW"] = perf_counter() - start

    start = perf_counter()
    signatures = np.vstack([path_signature_features(path) for path in subset])
    signature_z, _, _ = standardize_descriptors(signatures)
    distances["level-2 signature"] = distance_matrix_from_features(signature_z)
    runtimes["level-2 signature"] = perf_counter() - start

    normalized_components = {
        name: normalize_distance_matrix(distance)
        for name, distance in distances.items()
    }
    start = perf_counter()
    distances["multi-view"] = np.sqrt(
        np.mean(
            np.stack(
                [
                    normalized_components[name] ** 2
                    for name in (
                        "canonical H1",
                        "weighted Frechet",
                        "constrained DTW",
                        "occupation SW",
                        "level-2 signature",
                    )
                ]
            ),
            axis=0,
        )
    )
    runtimes["multi-view"] = perf_counter() - start

    distances = {
        name: normalize_distance_matrix(distance)
        for name, distance in distances.items()
    }

    responses = synthetic_history_response_matrix(subset)
    response_mean = responses.mean(axis=0)
    response_scale = responses.std(axis=0)
    active_response = response_scale > 0.01 * max(float(response_scale.max()), 1e-12)
    responses_z = (
        responses[:, active_response] - response_mean[active_response]
    ) / response_scale[active_response]
    response_distance = normalize_distance_matrix(
        distance_matrix_from_features(responses_z)
    )
    upper = np.triu_indices(len(subset), k=1)

    selections: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, distance in distances.items():
        selected = greedy_maximin_distance(distance, batch_size=batch_size)
        selections[name] = selected
        kernel = np.exp(-0.5 * distance**2)
        minimum_eigenvalue = float(np.linalg.eigvalsh(kernel).min())
        correlation = float(
            spearmanr(distance[upper], response_distance[upper]).statistic
        )
        response_radius = float(
            np.max(np.min(response_distance[:, selected], axis=1))
        )
        metrics[name] = {
            "response_spearman": correlation,
            "neighbor_overlap": neighborhood_overlap(
                distance, response_distance
            ),
            "triplet_agreement": triplet_agreement(
                distance, response_distance, seed=seed + 10
            ),
            "response_radius": response_radius,
            "knn_error": knn_response_error(
                distance, responses_z, selected
            ),
            "kernel_min_eigenvalue": minimum_eigenvalue,
            "runtime_seconds": runtimes[name],
        }

    return {
        "subset_indices": subset_indices,
        "paths": subset,
        "distances": distances,
        "responses": responses,
        "responses_z": responses_z,
        "active_response_features": active_response,
        "response_distance": response_distance,
        "selections": selections,
        "metrics": metrics,
    }


def quotient_distance(
    path_a: PathSample,
    path_b: PathSample,
    distance_function,
) -> float:
    """Distance between D2 symmetry orbits for an isometric base distance."""

    return float(
        min(
            distance_function(path_a, transform_path(path_b, symmetry))
            for symmetry in RECTANGLE_SYMMETRIES
        )
    )


def evaluate_flip_invariance(
    paths: Sequence[PathSample],
    seed: int = 53,
    n_paths: int = 14,
    n_pairs: int = 28,
) -> dict[str, object]:
    """Audit D2 isometry and quotient invariance for the original metrics."""

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=n_paths, replace=False)
    audit_paths = [paths[i] for i in indices]
    augmented_pool = [
        transform_path(path, symmetry)
        for path in audit_paths
        for symmetry in RECTANGLE_SYMMETRIES
    ]

    descriptor_all = descriptor_matrix(augmented_pool)
    _, descriptor_center, descriptor_scale = standardize_descriptors(descriptor_all)
    signature_all = np.vstack(
        [path_signature_features(path) for path in augmented_pool]
    )
    _, signature_center, signature_scale = standardize_descriptors(signature_all)

    def canonical_distance(a: PathSample, b: PathSample) -> float:
        return float(
            np.linalg.norm(canonical_input_features(a) - canonical_input_features(b))
        )

    def descriptor_distance(a: PathSample, b: PathSample) -> float:
        a_feature = (describe_path(a) - descriptor_center) / descriptor_scale
        b_feature = (describe_path(b) - descriptor_center) / descriptor_scale
        return float(np.linalg.norm(a_feature - b_feature))

    def frechet_distance(a: PathSample, b: PathSample) -> float:
        return discrete_frechet_distance(
            augmented_trajectory(a), augmented_trajectory(b)
        )

    def dtw_distance(a: PathSample, b: PathSample) -> float:
        return constrained_dtw_distance(
            augmented_trajectory(a), augmented_trajectory(b)
        )

    def signature_distance(a: PathSample, b: PathSample) -> float:
        a_feature = (path_signature_features(a) - signature_center) / signature_scale
        b_feature = (path_signature_features(b) - signature_center) / signature_scale
        return float(np.linalg.norm(a_feature - b_feature))

    base_functions = {
        "canonical H1": canonical_distance,
        "descriptors": descriptor_distance,
        "weighted Frechet": frechet_distance,
        "constrained DTW": dtw_distance,
        "occupation SW": sliced_occupation_distance,
        "level-2 signature": signature_distance,
    }

    pair_indices: list[tuple[int, int]] = []
    while len(pair_indices) < n_pairs:
        i, j = rng.choice(n_paths, size=2, replace=False)
        pair = (int(i), int(j))
        if pair not in pair_indices and pair[::-1] not in pair_indices:
            pair_indices.append(pair)

    scales: dict[str, float] = {}
    for name, function in base_functions.items():
        values = [function(audit_paths[i], audit_paths[j]) for i, j in pair_indices]
        scales[name] = max(float(np.median(values)), 1e-12)

    def multi_view_distance(a: PathSample, b: PathSample) -> float:
        values = [
            base_functions[name](a, b) / scales[name]
            for name in (
                "canonical H1",
                "weighted Frechet",
                "constrained DTW",
                "occupation SW",
                "level-2 signature",
            )
        ]
        return float(np.sqrt(np.mean(np.square(values))))

    functions = {**base_functions, "multi-view": multi_view_distance}
    scales["multi-view"] = max(
        float(
            np.median(
                [multi_view_distance(audit_paths[i], audit_paths[j]) for i, j in pair_indices]
            )
        ),
        1e-12,
    )

    metrics: dict[str, dict[str, float]] = {}
    non_identity = RECTANGLE_SYMMETRIES[1:]
    for name, function in functions.items():
        relative_errors = []
        for i, j in pair_indices:
            baseline = function(audit_paths[i], audit_paths[j])
            for symmetry in non_identity:
                transformed = function(
                    transform_path(audit_paths[i], symmetry),
                    transform_path(audit_paths[j], symmetry),
                )
                relative_errors.append(
                    abs(transformed - baseline) / max(abs(baseline), 1e-12)
                )

        raw_orbit = []
        quotient_orbit = []
        for path in audit_paths:
            for symmetry in non_identity:
                transformed_path = transform_path(path, symmetry)
                raw_orbit.append(function(path, transformed_path) / scales[name])
                quotient_orbit.append(
                    quotient_distance(path, transformed_path, function) / scales[name]
                )

        metrics[name] = {
            "simultaneous_isometry_mean_error": float(np.mean(relative_errors)),
            "simultaneous_isometry_max_error": float(np.max(relative_errors)),
            "raw_orbit_distance_median": float(np.median(raw_orbit)),
            "quotient_orbit_distance_max": float(np.max(quotient_orbit)),
        }

    return {
        "indices": indices,
        "paths": audit_paths,
        "metrics": metrics,
        "scales": scales,
        "example_orbit": symmetry_orbit(audit_paths[0]),
    }


def greedy_maximin(standardized: np.ndarray, batch_size: int) -> np.ndarray:
    """Greedy farthest-point design in standardized descriptor space."""

    n_paths = len(standardized)
    if not 0 < batch_size <= n_paths:
        raise ValueError("batch_size must be between one and the pool size")
    mean = standardized.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(standardized - mean, axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(standardized - standardized[first], axis=1)
    min_distance[first] = -np.inf
    while len(selected) < batch_size:
        nxt = int(np.argmax(min_distance))
        selected.append(nxt)
        distance = np.linalg.norm(standardized - standardized[nxt], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def covering_radius(standardized: np.ndarray, selected: Sequence[int]) -> float:
    selected = np.asarray(selected, dtype=int)
    distances = np.linalg.norm(
        standardized[:, None, :] - standardized[selected][None, :, :], axis=2
    )
    return float(np.max(np.min(distances, axis=1)))


def build_candidate_pool(
    seed: int = 7,
    n_freeform: int = 220,
    n_per_primitive: int = 45,
    n_reverse: int = 40,
) -> list[PathSample]:
    rng = np.random.default_rng(seed)
    paths: list[PathSample] = []

    for kind in ("line", "arc", "spiral", "zigzag"):
        count = 0
        attempts = 0
        while count < n_per_primitive and attempts < 20 * n_per_primitive:
            attempts += 1
            candidate = make_primitive_path(
                kind, rng, label=f"{kind}_{count:03d}"
            )
            if is_feasible(candidate):
                paths.append(candidate)
                count += 1
        if count != n_per_primitive:
            raise RuntimeError(f"Generated only {count}/{n_per_primitive} {kind} paths")

    for index in range(n_freeform):
        paths.append(make_freeform_path(rng, label=f"freeform_{index:03d}"))

    base_indices = rng.choice(len(paths), size=min(n_reverse, len(paths)), replace=False)
    paths.extend(paths[i].reversed() for i in base_indices)
    return paths


def source_counts(paths: Iterable[PathSample]) -> Counter:
    return Counter(path.source for path in paths)


def compare_initial_designs(
    paths: Sequence[PathSample],
    standardized: np.ndarray,
    batch_size: int = 24,
    n_random_trials: int = 100,
    seed: int = 13,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mixed = greedy_maximin(standardized, batch_size)
    zigzag_pool = np.array(
        [i for i, path in enumerate(paths) if path.source == "zigzag"], dtype=int
    )
    zigzag_local = greedy_maximin(standardized[zigzag_pool], batch_size)
    zigzag = zigzag_pool[zigzag_local]
    random_radii = np.array(
        [
            covering_radius(
                standardized,
                rng.choice(len(paths), size=batch_size, replace=False),
            )
            for _ in range(n_random_trials)
        ]
    )
    return {
        "mixed_indices": mixed,
        "zigzag_indices": zigzag,
        "mixed_radius": covering_radius(standardized, mixed),
        "zigzag_radius": covering_radius(standardized, zigzag),
        "random_radii": random_radii,
    }


def compare_representation_designs(
    paths: Sequence[PathSample],
    batch_size: int = 24,
    seed: int = 21,
    n_canonical_samples: int = 64,
) -> dict[str, object]:
    """Compare selection in descriptor, ordered-input, and RKHS geometries."""

    descriptors = descriptor_matrix(paths)
    descriptor_z, _, _ = standardize_descriptors(descriptors)
    ordered_input = canonical_input_matrix(paths, n_samples=n_canonical_samples)
    input_distances2 = pairwise_squared_distances(ordered_input)
    upper = np.triu_indices(len(paths), k=1)
    input_distance_scale = max(
        float(np.sqrt(np.median(input_distances2[upper]))), 1e-12
    )
    input_normalized = ordered_input / input_distance_scale

    input_kernel, input_bandwidth = rbf_kernel(ordered_input)
    descriptor_kernel, descriptor_bandwidth = rbf_kernel(descriptor_z)
    hybrid_kernel = input_kernel * descriptor_kernel

    rng = np.random.default_rng(seed)
    zigzag_pool = np.asarray(
        [i for i, path in enumerate(paths) if path.source == "zigzag"], dtype=int
    )
    selections = {
        "descriptor maximin": greedy_maximin(descriptor_z, batch_size),
        "input maximin": greedy_maximin(input_normalized, batch_size),
        "input kernel D-opt": kernel_d_optimal(input_kernel, batch_size),
        "hybrid kernel D-opt": kernel_d_optimal(hybrid_kernel, batch_size),
        "random": rng.choice(len(paths), size=batch_size, replace=False),
        "zigzag-only": zigzag_pool[
            greedy_maximin(input_normalized[zigzag_pool], batch_size)
        ],
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, selected in selections.items():
        selected_kernel = input_kernel[np.ix_(selected, selected)]
        metrics[name] = {
            "descriptor_radius": covering_radius(descriptor_z, selected),
            "input_radius": covering_radius(input_normalized, selected),
            "input_mmd": mmd_to_pool(input_kernel, selected),
            "kernel_effective_rank": kernel_effective_rank(selected_kernel),
        }

    return {
        "descriptors": descriptors,
        "descriptor_z": descriptor_z,
        "ordered_input": ordered_input,
        "input_normalized": input_normalized,
        "input_kernel": input_kernel,
        "descriptor_kernel": descriptor_kernel,
        "hybrid_kernel": hybrid_kernel,
        "input_bandwidth": input_bandwidth,
        "descriptor_bandwidth": descriptor_bandwidth,
        "selections": selections,
        "metrics": metrics,
        "collisions": representation_collision_pairs(
            descriptor_z, input_normalized
        ),
    }


def _colored_path(ax: plt.Axes, path: PathSample) -> None:
    points = path.xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = 0.5 * (path.force[:-1] + path.force[1:])
    collection = LineCollection(segments, cmap="viridis", linewidth=2.2)
    collection.set_array(colors)
    collection.set_clim(0.0, 1.05)
    ax.add_collection(collection)
    ax.scatter(*path.xy[0], s=24, marker="o", color="tab:green", zorder=3)
    ax.scatter(*path.xy[-1], s=28, marker="x", color="tab:red", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{path.source} · k={path.complexity}", fontsize=8)


def plot_gallery(
    paths: Sequence[PathSample],
    indices: Sequence[int],
    title: str,
    output: str | Path | None = None,
    n_columns: int = 6,
) -> plt.Figure:
    indices = list(indices)
    n_rows = int(np.ceil(len(indices) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(2.15 * n_columns, 2.15 * n_rows), squeeze=False
    )
    for ax, index in zip(axes.ravel(), indices):
        _colored_path(ax, paths[index])
    for ax in axes.ravel()[len(indices) :]:
        ax.axis("off")
    fig.suptitle(title + "\ncolor = normalized force; green = start; red = end")
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=170, bbox_inches="tight")
    return fig


def plot_embedding(
    paths: Sequence[PathSample],
    embedding: np.ndarray,
    selected: Sequence[int],
    explained: Sequence[float],
    output: str | Path | None = None,
    space_label: str = "descriptor",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sources = sorted(set(path.source for path in paths))
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, len(sources)))
    for color, source in zip(palette, sources):
        mask = np.array([path.source == source for path in paths])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=20,
            alpha=0.58,
            color=color,
            label=source,
        )
    selected = np.asarray(selected, dtype=int)
    ax.scatter(
        embedding[selected, 0],
        embedding[selected, 1],
        s=95,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="maximin selected",
    )
    ax.set_xlabel(f"{space_label} PC1 ({100 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"{space_label} PC2 ({100 * explained[1]:.1f}% variance)")
    ax.set_title(f"Representation space: {space_label}")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def plot_descriptor_coverage(
    descriptors: np.ndarray,
    selected: Sequence[int],
    output: str | Path | None = None,
) -> plt.Figure:
    chosen = (0, 2, 5, 6, 8, 12, 19, 24)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    selected = np.asarray(selected, dtype=int)
    for ax, descriptor_index in zip(axes.ravel(), chosen):
        values = descriptors[:, descriptor_index]
        ax.hist(values, bins=24, alpha=0.45, color="0.55", label="candidate pool")
        ax.hist(
            descriptors[selected, descriptor_index],
            bins=12,
            alpha=0.75,
            color="tab:orange",
            label="selected",
        )
        ax.set_title(DESCRIPTOR_NAMES[descriptor_index], fontsize=9)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Descriptor coverage of the maximin initial design")
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=170, bbox_inches="tight")
    return fig


def plot_design_metrics(
    metrics: dict[str, dict[str, float]],
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot coverage and diversity metrics without mixing their directions."""

    method_names = list(metrics)
    compact_names = [
        name.replace(" kernel ", "\nkernel ").replace(" maximin", "\nmaximin")
        for name in method_names
    ]
    specifications = (
        ("descriptor_radius", "Descriptor covering radius ↓"),
        ("input_radius", "Ordered-input covering radius ↓"),
        ("input_mmd", "Input-kernel MMD to pool ↓"),
        ("kernel_effective_rank", "Selected-kernel effective rank ↑"),
    )
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, len(method_names)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric_name, title) in zip(axes.ravel(), specifications):
        values = [metrics[name][metric_name] for name in method_names]
        ax.bar(np.arange(len(values)), values, color=colors, edgecolor="0.25")
        ax.set_xticks(np.arange(len(values)), compact_names, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for index, value in enumerate(values):
            ax.text(
                index,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle(
        "Initial-design diversity depends on which representation defines distance",
        fontsize=14,
    )
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def plot_collision_pairs(
    paths: Sequence[PathSample],
    collisions: dict[str, list[tuple[int, int, float, float]]],
    output: str | Path | None = None,
    pairs_per_kind: int = 3,
) -> plt.Figure:
    """Visualize cases where descriptor and ordered-input neighborhoods disagree."""

    rows: list[tuple[str, tuple[int, int, float, float]]] = []
    for kind in ("descriptor_near_input_far", "input_near_descriptor_far"):
        rows.extend((kind, pair) for pair in collisions[kind][:pairs_per_kind])
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(7.5, 3.3 * len(rows)), squeeze=False
    )
    for row_index, (kind, pair) in enumerate(rows):
        i, j, descriptor_distance, input_distance = pair
        _colored_path(axes[row_index, 0], paths[i])
        _colored_path(axes[row_index, 1], paths[j])
        axes[row_index, 0].set_title(paths[i].label, fontsize=8)
        axes[row_index, 1].set_title(paths[j].label, fontsize=8)
        label = (
            "descriptor-near / input-far"
            if kind == "descriptor_near_input_far"
            else "input-near / descriptor-far"
        )
        axes[row_index, 0].set_ylabel(
            f"{label}\n"
            f"d_desc={descriptor_distance:.2f}, d_input={input_distance:.2f}",
            fontsize=7.5,
        )
    fig.suptitle(
        "Representation-collision audit: the two metrics do not preserve all neighborhoods",
        fontsize=12,
        y=0.997,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    if output is not None:
        fig.savefig(output, dpi=170, bbox_inches="tight")
    return fig


def plot_metric_learning_results(
    metrics: dict[str, dict[str, float]],
    output: str | Path | None = None,
) -> plt.Figure:
    names = list(metrics)
    compact = [name.replace(" ", "\n") for name in names]
    specifications = (
        ("response_spearman", "Response-distance Spearman ↑"),
        ("neighbor_overlap", "5-neighbor response overlap ↑"),
        ("triplet_agreement", "Response triplet agreement ↑"),
        ("response_radius", "Selected response radius ↓"),
        ("knn_error", "Held-out response kNN error ↓"),
        ("runtime_seconds", "Pairwise metric runtime (s) ↓"),
    )
    colors = plt.cm.tab20(np.linspace(0.0, 0.75, len(names)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (metric_name, title) in zip(axes.ravel(), specifications):
        values = [metrics[name][metric_name] for name in names]
        ax.bar(np.arange(len(values)), values, color=colors, edgecolor="0.25")
        ax.set_xticks(np.arange(len(values)), compact, fontsize=7.2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        if metric_name == "runtime_seconds":
            ax.set_yscale("log")
        for index, value in enumerate(values):
            ax.text(
                index,
                value,
                f"{value:.2g}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )
    fig.suptitle(
        "Trajectory metrics judged by a controlled history-dependent learning task",
        fontsize=14,
    )
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def plot_metric_response_alignment(
    benchmark: dict[str, object],
    output: str | Path | None = None,
) -> plt.Figure:
    names = list(benchmark["distances"])
    response_distance = benchmark["response_distance"]
    upper = np.triu_indices(len(response_distance), k=1)
    rng = np.random.default_rng(101)
    sample_size = min(2200, len(upper[0]))
    positions = rng.choice(len(upper[0]), size=sample_size, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        distance = benchmark["distances"][name]
        ax.scatter(
            distance[upper][positions],
            response_distance[upper][positions],
            s=8,
            alpha=0.28,
        )
        rho = benchmark["metrics"][name]["response_spearman"]
        ax.set_title(f"{name}\nSpearman={rho:.2f}", fontsize=9)
        ax.set_xlabel("input distance", fontsize=8)
        ax.set_ylabel("response distance", fontsize=8)
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle(
        "Input similarity is only useful when it aligns with dynamical response",
        fontsize=13,
    )
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=175, bbox_inches="tight")
    return fig


def plot_flip_invariance_results(
    metrics: dict[str, dict[str, float]],
    output: str | Path | None = None,
) -> plt.Figure:
    names = list(metrics)
    compact = [name.replace(" ", "\n") for name in names]
    specifications = (
        ("simultaneous_isometry_max_error", "Max simultaneous-isometry error ↓", True),
        ("raw_orbit_distance_median", "Raw distance to flipped copy", False),
        ("quotient_orbit_distance_max", "Max quotient distance to flipped copy ↓", True),
    )
    colors = plt.cm.Pastel1(np.linspace(0.0, 1.0, len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (metric_name, title, log_scale) in zip(axes, specifications):
        values = [max(metrics[name][metric_name], 1e-18) for name in names]
        ax.bar(np.arange(len(values)), values, color=colors, edgecolor="0.3")
        ax.set_xticks(np.arange(len(values)), compact, fontsize=7)
        ax.set_title(title, fontsize=10)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Rectangle D2 symmetry: base metrics are isometric, quotienting creates invariance",
        fontsize=13,
    )
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def run_demo(output_directory: str | Path, seed: int = 7) -> dict[str, object]:
    """Generate the complete notebook experiment and save preview figures."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = build_candidate_pool(seed=seed)
    descriptors = descriptor_matrix(paths)
    standardized, center, scale = standardize_descriptors(descriptors)
    embedding, explained = pca_embedding(standardized)
    comparison = compare_initial_designs(paths, standardized)
    selected = comparison["mixed_indices"]
    representation_comparison = compare_representation_designs(paths, seed=seed + 14)
    input_embedding, input_explained = pca_embedding(
        representation_comparison["input_normalized"]
    )
    hybrid_selected = representation_comparison["selections"][
        "hybrid kernel D-opt"
    ]

    rng = np.random.default_rng(seed + 1)
    gallery_indices: list[int] = []
    for source in sorted(set(path.source for path in paths)):
        members = [i for i, path in enumerate(paths) if path.source == source]
        gallery_indices.extend(
            rng.choice(members, size=min(3, len(members)), replace=False).tolist()
        )

    figures = [
        plot_gallery(
            paths,
            gallery_indices,
            "Representative candidates from a topology-broad path space",
            output_directory / "candidate_gallery.png",
        ),
        plot_embedding(
            paths,
            embedding,
            selected,
            explained,
            output_directory / "descriptor_embedding.png",
        ),
        plot_gallery(
            paths,
            selected,
            "Descriptor-maximin initial design",
            output_directory / "selected_initial_design.png",
        ),
        plot_descriptor_coverage(
            descriptors,
            selected,
            output_directory / "descriptor_coverage.png",
        ),
        plot_embedding(
            paths,
            input_embedding,
            hybrid_selected,
            input_explained,
            output_directory / "ordered_input_embedding.png",
            space_label="canonical ordered input",
        ),
        plot_design_metrics(
            representation_comparison["metrics"],
            output_directory / "representation_design_comparison.png",
        ),
        plot_collision_pairs(
            paths,
            representation_comparison["collisions"],
            output_directory / "representation_collisions.png",
        ),
        plot_gallery(
            paths,
            hybrid_selected,
            "Hybrid input/descriptor kernel D-optimal design",
            output_directory / "hybrid_initial_design.png",
        ),
    ]
    for figure in figures:
        plt.close(figure)

    return {
        "paths": paths,
        "descriptors": descriptors,
        "standardized": standardized,
        "descriptor_center": center,
        "descriptor_scale": scale,
        "embedding": embedding,
        "explained": explained,
        "selected": selected,
        "comparison": comparison,
        "representation_comparison": representation_comparison,
        "input_embedding": input_embedding,
        "input_explained": input_explained,
        "source_counts": source_counts(paths),
        "output_directory": output_directory,
    }


if __name__ == "__main__":
    result = run_demo(
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "path_representation_outputs"
    )
    comparison = result["comparison"]
    random_radii = comparison["random_radii"]
    print(f"Generated {len(result['paths'])} feasible paths")
    print(f"Sources: {dict(result['source_counts'])}")
    print(f"Mixed maximin covering radius: {comparison['mixed_radius']:.3f}")
    print(f"Zigzag-only covering radius: {comparison['zigzag_radius']:.3f}")
    print(
        "Random covering radius: "
        f"{np.mean(random_radii):.3f} +/- {np.std(random_radii):.3f}"
    )
    print("\nRepresentation-aware design comparison:")
    for name, metrics in result["representation_comparison"]["metrics"].items():
        print(
            f"  {name:22s} "
            f"d_desc={metrics['descriptor_radius']:.3f}  "
            f"d_input={metrics['input_radius']:.3f}  "
            f"MMD={metrics['input_mmd']:.3f}  "
            f"rank={metrics['kernel_effective_rank']:.2f}"
        )
    print(f"Figures written to {result['output_directory']}")
