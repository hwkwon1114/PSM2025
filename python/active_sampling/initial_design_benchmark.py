"""Compare initial zigzag designs before expensive solver labels are available.

The module deliberately separates three ideas that are often conflated:

* random feasible sampling;
* Latin-hypercube coordinate stratification; and
* constrained maximin selection under alternative trajectory metrics.

The maximin methods use one common feasible reference ensemble so differences among
them can be attributed to the metric.  This is a benchmark control, not a
mathematical restriction: :func:`constrained_farthest_first` only needs a distance
oracle and can later be connected to an on-demand proposal optimizer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import csv
import json
from math import ceil, cos, pi, sin
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .zigzag_action import (
    DecodedZigzag,
    Panel,
    ZigzagAction,
    _balanced_strip_targets,
    decode_action,
    generate_actions,
    zigzag_vertices,
)


SCALE_CLASSES = ("local", "medium", "near_full")
SCALE_PROPORTIONS = {"local": 0.30, "medium": 0.50, "near_full": 0.20}
METRIC_NAMES = ("action_euclidean", "d2_ordered_l2", "d2_frechet", "d2_dtw")


def _resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    if cumulative[-1] <= 1e-14:
        raise ValueError("Cannot resample a zero-length path")
    target = np.linspace(0.0, cumulative[-1], n_points)
    return np.column_stack(
        [np.interp(target, cumulative, points[:, axis]) for axis in range(2)]
    )


def ordered_trajectory(
    decoded: DecodedZigzag,
    panel: Panel = Panel(),
    n_points: int = 16,
) -> np.ndarray:
    """Return an ordered, dimensionless input curve for trajectory distances.

    Position is normalized by the panel half extents.  The progress coordinate
    preserves traversal order, and constant process channels make geometrically
    identical paths with different actuation distinguishable.  No response fields
    are used, so the metric cannot leak test labels into the initial design.
    """

    xy = _resample_polyline(zigzag_vertices(decoded), n_points=n_points)
    xy[:, 0] /= panel.half_x_m
    xy[:, 1] /= panel.half_y_m
    progress = np.linspace(0.0, 1.0, n_points)
    action = decoded.action
    process = np.array(
        [
            action.growth_membrane / 0.0045,
            action.growth_curvature / 0.0060,
            action.orthotropy / 1.6,
            (action.log_band_fraction - np.log(0.004))
            / (np.log(0.075) - np.log(0.004)),
            (action.n_strips - 3.0) / 11.0,
        ],
        dtype=float,
    )
    weights = np.array([0.45, 0.45, 0.25, 0.20, 0.20])
    repeated_process = np.broadcast_to(weights * process, (n_points, 5))
    return np.column_stack((xy, 0.30 * progress, repeated_process))


def d2_orbit(curve: np.ndarray) -> np.ndarray:
    """Apply rectangle D2 symmetries to space without reversing time."""

    signs = np.array(((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)))
    orbit = np.repeat(curve[None, :, :], 4, axis=0)
    orbit[:, :, :2] *= signs[:, None, :]
    return orbit


def action_feature_matrix(samples: Sequence[DecodedZigzag]) -> np.ndarray:
    """Coordinates for the non-invariant parameter-space baseline."""

    rows = []
    for decoded in samples:
        action = decoded.action
        rows.append(
            [
                action.log_area_scale,
                action.log_aspect,
                action.xi_u,
                action.xi_v,
                cos(2.0 * action.rotation_rad),
                sin(2.0 * action.rotation_rad),
                (action.n_strips - 3.0) / 11.0,
                action.log_band_fraction,
                action.growth_membrane,
                action.growth_curvature,
                action.orthotropy,
            ]
        )
    return np.asarray(rows, dtype=float)


def robust_standardize(features: np.ndarray) -> np.ndarray:
    center = np.median(features, axis=0)
    q25, q75 = np.percentile(features, [25.0, 75.0], axis=0)
    scale = q75 - q25
    fallback = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1.0))
    return (features - center) / scale


def _point_costs(pivot: np.ndarray, orbit_curves: np.ndarray) -> np.ndarray:
    delta = orbit_curves[:, :, None, :, :] - pivot[None, None, :, None, :]
    return np.linalg.norm(delta, axis=-1)


def _batched_frechet(costs: np.ndarray) -> np.ndarray:
    dynamic = np.empty_like(costs)
    dynamic[:, :, 0, 0] = costs[:, :, 0, 0]
    for i in range(1, costs.shape[2]):
        dynamic[:, :, i, 0] = np.maximum(
            dynamic[:, :, i - 1, 0], costs[:, :, i, 0]
        )
    for j in range(1, costs.shape[3]):
        dynamic[:, :, 0, j] = np.maximum(
            dynamic[:, :, 0, j - 1], costs[:, :, 0, j]
        )
    for i in range(1, costs.shape[2]):
        for j in range(1, costs.shape[3]):
            predecessor = np.minimum(
                np.minimum(dynamic[:, :, i - 1, j], dynamic[:, :, i - 1, j - 1]),
                dynamic[:, :, i, j - 1],
            )
            dynamic[:, :, i, j] = np.maximum(costs[:, :, i, j], predecessor)
    return dynamic[:, :, -1, -1].min(axis=1)


def _batched_dtw(costs: np.ndarray, window_fraction: float = 0.20) -> np.ndarray:
    n_left, n_right = costs.shape[2:]
    window = max(abs(n_left - n_right), int(ceil(window_fraction * max(n_left, n_right))))
    dynamic = np.full(
        (costs.shape[0], costs.shape[1], n_left + 1, n_right + 1),
        np.inf,
        dtype=float,
    )
    dynamic[:, :, 0, 0] = 0.0
    for i in range(1, n_left + 1):
        for j in range(max(1, i - window), min(n_right, i + window) + 1):
            predecessor = np.minimum(
                np.minimum(dynamic[:, :, i - 1, j], dynamic[:, :, i, j - 1]),
                dynamic[:, :, i - 1, j - 1],
            )
            dynamic[:, :, i, j] = costs[:, :, i - 1, j - 1] + predecessor
    return dynamic[:, :, n_left, n_right].min(axis=1) / max(n_left + n_right, 1)


class DistanceOracle:
    """Compute one-to-all distances without materializing a quadratic matrix."""

    def __init__(
        self,
        samples: Sequence[DecodedZigzag],
        metric: str,
        n_points: int = 16,
    ) -> None:
        if metric not in METRIC_NAMES:
            raise ValueError(f"Unknown metric {metric!r}; choose from {METRIC_NAMES}")
        self.metric = metric
        self.features = robust_standardize(action_feature_matrix(samples))
        self.curves = np.stack(
            [ordered_trajectory(sample, n_points=n_points) for sample in samples]
        )
        self.orbits = np.stack([d2_orbit(curve) for curve in self.curves])

    def from_index(self, index: int) -> np.ndarray:
        if self.metric == "action_euclidean":
            return np.linalg.norm(self.features - self.features[index], axis=1)
        if self.metric == "d2_ordered_l2":
            delta = self.orbits - self.curves[index][None, None, :, :]
            return np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1)).min(axis=1)
        costs = _point_costs(self.curves[index], self.orbits)
        if self.metric == "d2_frechet":
            return _batched_frechet(costs)
        return _batched_dtw(costs)

    def pair(self, left: int, right: int) -> float:
        return float(self.from_index(left)[right])


def balanced_schedule(n_samples: int) -> list[tuple[str, int]]:
    """Create a nested scale/strip schedule shared by every selection method."""

    class_targets = {
        "local": int(round(0.30 * n_samples)),
        "medium": int(round(0.50 * n_samples)),
    }
    class_targets["near_full"] = n_samples - sum(class_targets.values())
    cell_targets = {
        category: _balanced_strip_targets(class_targets[category], offset=3 * index)
        for index, category in enumerate(SCALE_CLASSES)
    }
    class_used = Counter()
    cell_used: dict[str, Counter] = {category: Counter() for category in SCALE_CLASSES}
    schedule: list[tuple[str, int]] = []
    for step in range(n_samples):
        categories = [c for c in SCALE_CLASSES if class_used[c] < class_targets[c]]
        category = max(
            categories,
            key=lambda c: (
                SCALE_PROPORTIONS[c] * (step + 1) - class_used[c],
                -SCALE_CLASSES.index(c),
            ),
        )
        used_in_class = class_used[category]
        strips = [n for n in range(3, 15) if cell_used[category][n] < cell_targets[category][n]]
        strip = max(
            strips,
            key=lambda n: (
                cell_targets[category][n] * (used_in_class + 1) / class_targets[category]
                - cell_used[category][n],
                -n,
            ),
        )
        schedule.append((category, strip))
        class_used[category] += 1
        cell_used[category][strip] += 1
    return schedule


def _members_by_cell(samples: Sequence[DecodedZigzag]) -> dict[tuple[str, int], list[int]]:
    members: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        members[(sample.actual_scale_class, sample.action.n_strips)].append(index)
    return members


def stratified_random_indices(
    samples: Sequence[DecodedZigzag],
    schedule: Sequence[tuple[str, int]],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cells = _members_by_cell(samples)
    for members in cells.values():
        rng.shuffle(members)
    used = Counter()
    selected = []
    for cell in schedule:
        if used[cell] >= len(cells[cell]):
            raise ValueError(f"Reference ensemble has insufficient members for {cell}")
        selected.append(cells[cell][used[cell]])
        used[cell] += 1
    return np.asarray(selected, dtype=int)


def constrained_farthest_first(
    samples: Sequence[DecodedZigzag],
    schedule: Sequence[tuple[str, int]],
    oracle: DistanceOracle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy maximin under a fixed, nested scale/strip balance schedule."""

    cells = _members_by_cell(samples)
    available = np.ones(len(samples), dtype=bool)
    minimum_distance = np.full(len(samples), np.inf)
    selected: list[int] = []
    insertion_distance: list[float] = []
    for step, cell in enumerate(schedule):
        eligible = np.asarray([index for index in cells[cell] if available[index]], dtype=int)
        if not len(eligible):
            raise ValueError(f"Reference ensemble has insufficient members for {cell}")
        if step == 0:
            norms = np.linalg.norm(oracle.features[eligible], axis=1)
            chosen = int(eligible[np.argmax(norms)])
            insertion_distance.append(float("nan"))
        else:
            chosen = int(eligible[np.argmax(minimum_distance[eligible])])
            insertion_distance.append(float(minimum_distance[chosen]))
        selected.append(chosen)
        available[chosen] = False
        minimum_distance = np.minimum(minimum_distance, oracle.from_index(chosen))
        minimum_distance[~available] = -np.inf
    return (
        np.asarray(selected, dtype=int),
        np.asarray(insertion_distance, dtype=float),
        minimum_distance,
    )


def _identified(decoded: DecodedZigzag, strategy: str, index: int) -> DecodedZigzag:
    action = ZigzagAction(
        **{
            **asdict(decoded.action),
            "sample_id": f"{strategy}_{index:04d}",
        }
    )
    return decode_action(action)


def _write_design(
    output_directory: Path,
    strategy: str,
    selected_samples: Sequence[DecodedZigzag],
    source_indices: Sequence[int] | None,
) -> None:
    path = output_directory / f"design_{strategy}_500.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for design_index, decoded in enumerate(selected_samples):
            identified = _identified(decoded, strategy, design_index)
            payload = {
                "strategy": strategy,
                "design_index": design_index,
                "source_candidate_index": (
                    None if source_indices is None else int(source_indices[design_index])
                ),
                "scale_class": identified.actual_scale_class,
                "action": asdict(identified.action),
                "sequence_v1": identified.sequence_v1(),
            }
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _counts(samples: Sequence[DecodedZigzag]) -> dict[str, dict[str, int]]:
    return {
        "scale_class": dict(Counter(sample.actual_scale_class for sample in samples)),
        "n_strips": {
            str(key): value
            for key, value in sorted(Counter(sample.action.n_strips for sample in samples).items())
        },
    }


def run_benchmark(
    output_directory: Path,
    n_samples: int = 500,
    reference_size: int = 1500,
    seed: int = 20260820,
    n_points: int = 16,
    metrics: Sequence[str] = METRIC_NAMES,
) -> dict:
    if n_samples != 500:
        raise ValueError("Output naming currently requires n_samples=500")
    if reference_size < 2 * n_samples:
        raise ValueError("reference_size must be at least twice n_samples")
    output_directory.mkdir(parents=True, exist_ok=True)
    schedule = balanced_schedule(n_samples)
    reference, reference_summary = generate_actions(reference_size, seed=seed + 10)
    lhs, lhs_summary = generate_actions(n_samples, seed=seed)

    # Reorder the direct LHS so every learning-curve prefix is approximately balanced.
    lhs_order = stratified_random_indices(lhs, schedule, seed=seed + 20)
    lhs = [lhs[index] for index in lhs_order]
    _write_design(output_directory, "lhs", lhs, source_indices=None)

    random_indices = stratified_random_indices(reference, schedule, seed=seed + 30)
    random_design = [reference[index] for index in random_indices]
    _write_design(output_directory, "random", random_design, random_indices)

    result = {
        "seed": seed,
        "n_samples": n_samples,
        "reference_size": reference_size,
        "trajectory_points": n_points,
        "reference_generation": reference_summary,
        "lhs_generation": lhs_summary,
        "designs": {
            "lhs": {"counts": _counts(lhs), "selection_seconds": 0.0},
            "random": {"counts": _counts(random_design), "selection_seconds": 0.0},
        },
    }
    diagnostic_rows = []
    for metric in metrics:
        start = perf_counter()
        oracle = DistanceOracle(reference, metric=metric, n_points=n_points)
        indices, insertion, remaining = constrained_farthest_first(reference, schedule, oracle)
        elapsed = perf_counter() - start
        design = [reference[index] for index in indices]
        strategy = f"maximin_{metric}"
        _write_design(output_directory, strategy, design, indices)
        finite_insertion = insertion[np.isfinite(insertion)]
        result["designs"][strategy] = {
            "counts": _counts(design),
            "selection_seconds": elapsed,
            "minimum_pair_distance": float(finite_insertion.min()),
            "insertion_distance_q05": float(np.quantile(finite_insertion, 0.05)),
            "insertion_distance_median": float(np.median(finite_insertion)),
            "reference_covering_radius": float(remaining[remaining >= 0.0].max()),
        }
        for design_index, distance in enumerate(insertion):
            diagnostic_rows.append(
                {
                    "strategy": strategy,
                    "design_size": design_index + 1,
                    "insertion_distance": distance,
                }
            )

    with (output_directory / "maximin_insertion_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    (output_directory / "initial_design_benchmark_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_diagnostics(diagnostic_rows, output_directory)
    return result


def _plot_diagnostics(rows: Sequence[dict], output_directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    strategies = sorted({row["strategy"] for row in rows})
    for strategy in strategies:
        members = [row for row in rows if row["strategy"] == strategy]
        x = np.asarray([row["design_size"] for row in members])
        y = np.asarray([row["insertion_distance"] for row in members], dtype=float)
        keep = np.isfinite(y)
        ax.plot(x[keep], y[keep], label=strategy.replace("maximin_", ""), alpha=0.85)
    ax.set_xlabel("initial-design size")
    ax.set_ylabel("distance to nearest earlier trajectory")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_title("Constrained farthest-first insertion distances")
    fig.tight_layout()
    fig.savefig(output_directory / "maximin_insertion_distances.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/initial_design_benchmark_outputs"),
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--reference-size", type=int, default=1500)
    parser.add_argument("--trajectory-points", type=int, default=16)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        default=list(METRIC_NAMES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        output_directory=args.output_dir,
        reference_size=args.reference_size,
        seed=args.seed,
        n_points=args.trajectory_points,
        metrics=args.metrics,
    )
    print(f"Wrote {len(result['designs'])} initial designs to {args.output_dir}")
    for name, diagnostic in result["designs"].items():
        minimum = diagnostic.get("minimum_pair_distance", "n/a")
        print(f"  {name}: min distance={minimum}, seconds={diagnostic['selection_seconds']:.2f}")


if __name__ == "__main__":
    main()
