"""Spatial treatment-field kernels and information-gain initial designs."""

from __future__ import annotations

from collections import defaultdict
from math import cos, sin, sqrt
from typing import Sequence

import numpy as np

from .zigzag_action import DecodedZigzag, Panel, zigzag_vertices


FIELD_CHANNELS = (
    "top_11_minus_identity",
    "sqrt2_top_12",
    "top_22_minus_identity",
    "bottom_11_minus_identity",
    "sqrt2_bottom_12",
    "bottom_22_minus_identity",
)


def _point_segment_distance(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    direction = end - start
    length_squared = float(direction @ direction)
    if length_squared <= 1e-24:
        return np.linalg.norm(points - start, axis=1)
    coordinate = np.clip((points - start) @ direction / length_squared, 0.0, 1.0)
    closest = start + coordinate[:, None] * direction
    return np.linalg.norm(points - closest, axis=1)


def _growth_tensor(angle: float, growth_1: float, growth_2: float) -> np.ndarray:
    rotation = np.array(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]], dtype=float
    )
    stretch = np.diag([1.0 + growth_1, 1.0 + growth_2])
    return rotation @ stretch @ rotation.T


def spatial_treatment_fields(
    samples: Sequence[DecodedZigzag],
    panel: Panel = Panel(),
    grid_shape: tuple[int, int] = (20, 16),
    subsamples_per_axis: int = 4,
) -> np.ndarray:
    """Rasterize the final top/bottom target metrics in material coordinates.

    The channels contain the tensor difference from the identity metric. The
    off-diagonal channel is multiplied by sqrt(2), so Euclidean distance in the
    flattened representation equals the Frobenius field distance. Band overlap
    uses the simulator's multiplicative target-metric algebra in polyline order,
    while geometric band coverage is approximated by regular subcell quadrature.
    Intermediate equilibria are discarded.
    """

    n_v, n_u = grid_shape
    if n_v < 2 or n_u < 2:
        raise ValueError("grid dimensions must both be at least two")
    if subsamples_per_axis < 1:
        raise ValueError("subsamples_per_axis must be positive")
    fine_v = n_v * subsamples_per_axis
    fine_u = n_u * subsamples_per_axis
    u = -panel.half_x_m + (
        np.arange(fine_u, dtype=float) + 0.5
    ) * panel.width_m / fine_u
    v = -panel.half_y_m + (
        np.arange(fine_v, dtype=float) + 0.5
    ) * panel.height_m / fine_v
    grid_u, grid_v = np.meshgrid(u, v)
    points = np.column_stack((grid_u.ravel(), grid_v.ravel()))
    identity = np.eye(2, dtype=float)
    fields = np.empty((len(samples), n_v, n_u, len(FIELD_CHANNELS)), dtype=float)

    for sample_index, decoded in enumerate(samples):
        top = np.broadcast_to(identity, (len(points), 2, 2)).copy()
        bottom = top.copy()
        vertices = zigzag_vertices(decoded)
        action = decoded.action
        top_growth = (
            decoded.growth_top * (1.0 + action.orthotropy),
            decoded.growth_top * (1.0 - action.orthotropy),
        )
        bottom_growth = (
            decoded.growth_bottom * (1.0 + action.orthotropy),
            decoded.growth_bottom * (1.0 - action.orthotropy),
        )
        for start, end in zip(vertices[:-1], vertices[1:]):
            hit = _point_segment_distance(points, start, end) <= 0.5 * decoded.band_width_m
            if not np.any(hit):
                continue
            direction = end - start
            angle = float(np.arctan2(direction[1], direction[0]) % np.pi)
            growth_top = _growth_tensor(angle, *top_growth)
            growth_bottom = _growth_tensor(angle, *bottom_growth)
            top[hit] = np.einsum(
                "ab,nbc,cd->nad", growth_top, top[hit], growth_top
            )
            bottom[hit] = np.einsum(
                "ab,nbc,cd->nad", growth_bottom, bottom[hit], growth_bottom
            )

        encoded = np.column_stack(
            (
                top[:, 0, 0] - 1.0,
                sqrt(2.0) * top[:, 0, 1],
                top[:, 1, 1] - 1.0,
                bottom[:, 0, 0] - 1.0,
                sqrt(2.0) * bottom[:, 0, 1],
                bottom[:, 1, 1] - 1.0,
            )
        )
        fine_field = encoded.reshape(fine_v, fine_u, -1)
        fields[sample_index] = fine_field.reshape(
            n_v,
            subsamples_per_axis,
            n_u,
            subsamples_per_axis,
            len(FIELD_CHANNELS),
        ).mean(axis=(1, 3))
    return fields


def spatial_field_features(
    fields: np.ndarray,
    channel_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return channel-balanced integral features and their fitted scales."""

    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 4 or fields.shape[-1] != len(FIELD_CHANNELS):
        raise ValueError(
            f"fields must have shape (samples, v, u, {len(FIELD_CHANNELS)})"
        )
    if channel_scale is None:
        channel_scale = np.sqrt(np.mean(fields * fields, axis=(0, 1, 2)))
        channel_scale = np.where(channel_scale > 1e-14, channel_scale, 1.0)
    else:
        channel_scale = np.asarray(channel_scale, dtype=float)
        if channel_scale.shape != (len(FIELD_CHANNELS),):
            raise ValueError(
                f"channel_scale must have shape ({len(FIELD_CHANNELS)},)"
            )
        if np.any(channel_scale <= 0.0):
            raise ValueError("channel_scale entries must be positive")
    normalized = fields / channel_scale
    features = normalized.reshape(len(fields), -1) / sqrt(fields.shape[1] * fields.shape[2])
    features -= features.mean(axis=0, keepdims=True)
    return features, channel_scale


def rbf_kernel(
    features: np.ndarray,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, float]:
    """Build a positive-definite RBF kernel with a median-distance scale."""

    features = np.asarray(features, dtype=float)
    squared_norm = np.sum(features * features, axis=1)
    distances_squared = squared_norm[:, None] + squared_norm[None, :]
    distances_squared -= 2.0 * features @ features.T
    distances_squared = np.maximum(distances_squared, 0.0)
    if bandwidth is None:
        upper = distances_squared[np.triu_indices_from(distances_squared, k=1)]
        positive = upper[upper > 1e-14]
        bandwidth = (
            float(np.sqrt(np.median(positive))) if len(positive) else 1.0
        )
    kernel = np.exp(
        -0.5 * distances_squared / max(float(bandwidth) ** 2, 1e-14)
    )
    return kernel, float(bandwidth)


def raster_convergence_diagnostic(
    samples: Sequence[DecodedZigzag],
    panel: Panel = Panel(),
    grid_shape: tuple[int, int] = (20, 16),
    tested_subsamples: Sequence[int] = (8, 16),
    reference_subsamples: int = 32,
) -> dict[str, object]:
    """Measure raster feature/kernel convergence against a finer quadrature."""

    reference_fields = spatial_treatment_fields(
        samples,
        panel=panel,
        grid_shape=grid_shape,
        subsamples_per_axis=reference_subsamples,
    )
    reference_features, channel_scale = spatial_field_features(reference_fields)
    reference_kernel, bandwidth = rbf_kernel(reference_features)
    feature_norm = max(float(np.linalg.norm(reference_features)), 1e-15)
    kernel_norm = max(float(np.linalg.norm(reference_kernel)), 1e-15)
    comparisons = {}
    for subsamples in tested_subsamples:
        fields = spatial_treatment_fields(
            samples,
            panel=panel,
            grid_shape=grid_shape,
            subsamples_per_axis=int(subsamples),
        )
        features, _ = spatial_field_features(fields, channel_scale=channel_scale)
        kernel, _ = rbf_kernel(features, bandwidth=bandwidth)
        comparisons[str(subsamples)] = {
            "relative_feature_frobenius_error": float(
                np.linalg.norm(features - reference_features) / feature_norm
            ),
            "relative_kernel_frobenius_error": float(
                np.linalg.norm(kernel - reference_kernel) / kernel_norm
            ),
            "maximum_kernel_absolute_error": float(
                np.max(np.abs(kernel - reference_kernel))
            ),
        }
    return {
        "sample_count": len(samples),
        "grid_shape_vu": list(grid_shape),
        "reference_subsamples_per_axis": reference_subsamples,
        "reference_bandwidth": bandwidth,
        "comparisons": comparisons,
    }


def _members_by_cell(
    samples: Sequence[DecodedZigzag],
) -> dict[tuple[str, int], list[int]]:
    members: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        members[(sample.actual_scale_class, sample.action.n_strips)].append(index)
    return members


def constrained_information_greedy(
    kernel: np.ndarray,
    samples: Sequence[DecodedZigzag],
    schedule: Sequence[tuple[str, int]],
    noise_variance: float = 1e-4,
    jitter: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy log-determinant design under the shared balance schedule.

    Pivoted Cholesky residual variance is the marginal gain criterion. The
    independent observation-noise term regularizes nearly equivalent treatment
    fields and makes the selected Gram matrix strictly positive definite.
    """

    kernel = np.asarray(kernel, dtype=float)
    n_candidates = len(samples)
    if kernel.shape != (n_candidates, n_candidates):
        raise ValueError("kernel shape must match the candidate count")
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive")
    if len(schedule) > n_candidates:
        raise ValueError("schedule cannot exceed the candidate count")

    covariance = kernel.copy()
    covariance.flat[:: n_candidates + 1] += noise_variance
    factors = np.zeros((n_candidates, len(schedule)), dtype=float)
    residual = np.diag(covariance).copy()
    available = np.ones(n_candidates, dtype=bool)
    cells = _members_by_cell(samples)
    selected: list[int] = []
    conditional_variance: list[float] = []

    for column, cell in enumerate(schedule):
        eligible = np.asarray(
            [index for index in cells[cell] if available[index]], dtype=int
        )
        if not len(eligible):
            raise ValueError(f"Reference ensemble has insufficient members for {cell}")
        if column == 0:
            chosen = int(eligible[np.argmin(kernel[eligible].mean(axis=1))])
        else:
            chosen = int(eligible[np.argmax(residual[eligible])])
        pivot = max(float(residual[chosen]), jitter)
        correction = covariance[:, chosen]
        if column:
            correction = correction - factors[:, :column] @ factors[chosen, :column]
        factors[:, column] = correction / sqrt(pivot)
        residual = np.maximum(residual - factors[:, column] ** 2, 0.0)
        available[chosen] = False
        residual[~available] = -np.inf
        selected.append(chosen)
        conditional_variance.append(pivot)

    return np.asarray(selected, dtype=int), np.asarray(conditional_variance)


def information_gain(
    kernel: np.ndarray,
    selected: Sequence[int],
    noise_variance: float,
) -> float:
    """Return 0.5 log det(I + K_SS / noise_variance)."""

    indices = np.asarray(selected, dtype=int)
    submatrix = kernel[np.ix_(indices, indices)]
    sign, logdet = np.linalg.slogdet(
        np.eye(len(indices)) + submatrix / noise_variance
    )
    if sign <= 0.0:
        raise ValueError("selected information matrix is not positive definite")
    return 0.5 * float(logdet)
