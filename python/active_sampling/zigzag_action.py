"""Generate scale-aware zigzag actions compatible with zigzag_sequence schema v1."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from math import atan, cos, exp, log, pi, sin, sqrt, tan
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Panel:
    half_x_m: float = 0.127
    half_y_m: float = 0.1524

    @property
    def width_m(self) -> float:
        return 2.0 * self.half_x_m

    @property
    def height_m(self) -> float:
        return 2.0 * self.half_y_m


@dataclass(frozen=True)
class ZigzagAction:
    sample_id: str
    requested_scale_class: str
    log_area_scale: float
    log_aspect: float
    xi_u: float
    xi_v: float
    rotation_rad: float
    n_strips: int
    log_band_fraction: float
    growth_membrane: float
    growth_curvature: float
    orthotropy: float


@dataclass(frozen=True)
class DecodedZigzag:
    action: ZigzagAction
    width_m: float
    height_m: float
    pitch_m: float
    band_width_m: float
    alpha_rad: float
    center_u_m: float
    center_v_m: float
    rotated_half_u_m: float
    rotated_half_v_m: float
    coverage_fraction: float
    centerline_area_fraction: float
    pitch_band_ratio: float
    path_length_m: float
    growth_top: float
    growth_bottom: float
    actual_scale_class: str

    def operation_v1(self) -> dict:
        return {
            "type": "zigzag",
            "lv_mm": 1000.0 * self.height_m,
            "alpha_deg": 180.0 * self.alpha_rad / pi,
            "n_strips": self.action.n_strips,
            "width_mm": 1000.0 * self.band_width_m,
            "center_uv_mm": [
                1000.0 * self.center_u_m,
                1000.0 * self.center_v_m,
            ],
            "rotation_deg": 180.0 * self.action.rotation_rad / pi,
            "gtop": self.growth_top,
            "gbot": self.growth_bottom,
            "ortho": self.action.orthotropy,
            "profile": {"mode": "uniform"},
        }

    def sequence_v1(self) -> dict:
        return {
            "schema_version": 1,
            "units": {
                "length": "mm",
                "angle": "deg",
                "growth": "engineering_strain",
            },
            "hardening": {"model": "none"},
            "defaults": {
                "start_mode": "left_bottom_up",
                "profile": {"mode": "uniform"},
            },
            "toolpaths": [
                {
                    "id": self.action.sample_id,
                    "enabled": True,
                    "repeat": 1,
                    "operation": self.operation_v1(),
                }
            ],
        }


def scale_class(coverage_fraction: float) -> str:
    if coverage_fraction < 0.05:
        return "local"
    if coverage_fraction < 0.80:
        return "medium"
    return "near_full"


def decode_action(action: ZigzagAction, panel: Panel = Panel()) -> DecodedZigzag:
    area_scale = exp(action.log_area_scale)
    aspect = action.log_aspect
    normalized_width = sqrt(area_scale * exp(aspect))
    normalized_height = sqrt(area_scale * exp(-aspect))
    width = panel.width_m * normalized_width
    height = panel.height_m * normalized_height
    band = exp(action.log_band_fraction) * min(panel.width_m, panel.height_m)

    if action.n_strips < 3:
        raise ValueError("n_strips must be at least three for a zigzag")
    pitch = width / (action.n_strips - 2)
    alpha = atan(pitch / height)
    if not (np.deg2rad(1.0) <= alpha <= np.deg2rad(72.0)):
        raise ValueError("decoded alpha is outside [1,72] degrees")

    local_half_x = 0.5 * (width + band)
    local_half_y = 0.5 * (height + band)
    c = abs(cos(action.rotation_rad))
    s = abs(sin(action.rotation_rad))
    half_u = c * local_half_x + s * local_half_y
    half_v = s * local_half_x + c * local_half_y
    if half_u > panel.half_x_m or half_v > panel.half_y_m:
        raise ValueError("rotated footprint does not fit inside the panel")

    center_u = action.xi_u * (panel.half_x_m - half_u)
    center_v = action.xi_v * (panel.half_y_m - half_v)
    coverage = 4.0 * half_u * half_v / (panel.width_m * panel.height_m)
    centerline_area = width * height / (panel.width_m * panel.height_m)
    ratio = pitch / band
    if ratio < 0.15 or ratio > 30.0:
        raise ValueError("pitch-to-band ratio is outside [0.15,30]")

    growth_top = action.growth_membrane + action.growth_curvature
    growth_bottom = action.growth_membrane - action.growth_curvature
    eta = action.orthotropy
    stretches = (
        1.0 + growth_top * (1.0 + eta),
        1.0 + growth_top * (1.0 - eta),
        1.0 + growth_bottom * (1.0 + eta),
        1.0 + growth_bottom * (1.0 - eta),
    )
    if min(stretches) <= 0.0:
        raise ValueError("action produces a nonpositive incremental stretch")

    path_length = 2.0 * height + (action.n_strips - 2) * sqrt(
        height**2 + pitch**2
    )
    return DecodedZigzag(
        action=action,
        width_m=width,
        height_m=height,
        pitch_m=pitch,
        band_width_m=band,
        alpha_rad=alpha,
        center_u_m=center_u,
        center_v_m=center_v,
        rotated_half_u_m=half_u,
        rotated_half_v_m=half_v,
        coverage_fraction=coverage,
        centerline_area_fraction=centerline_area,
        pitch_band_ratio=ratio,
        path_length_m=path_length,
        growth_top=growth_top,
        growth_bottom=growth_bottom,
        actual_scale_class=scale_class(coverage),
    )


def zigzag_vertices(decoded: DecodedZigzag) -> np.ndarray:
    n_inclined = decoded.action.n_strips - 2
    x_left = -0.5 * decoded.width_m
    x_right = 0.5 * decoded.width_m
    y_bottom = -0.5 * decoded.height_m
    y_top = 0.5 * decoded.height_m
    points = [[x_left, y_bottom], [x_left, y_top]]
    for index in range(1, n_inclined + 1):
        points.append(
            [
                x_left + index * decoded.pitch_m,
                y_bottom if index % 2 else y_top,
            ]
        )
    last_y = points[-1][1]
    points.append([x_right, y_bottom if abs(last_y - y_top) < 1e-14 else y_top])
    local = np.asarray(points, dtype=float)
    angle = decoded.action.rotation_rad
    rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    return local @ rotation.T + np.array([decoded.center_u_m, decoded.center_v_m])


def _category_area_range(category: str) -> tuple[float, float]:
    if category == "local":
        return 0.0008, 0.035
    if category == "medium":
        return 0.02, 0.62
    if category == "near_full":
        return 0.58, 0.96
    raise ValueError(category)


def _rotation_from_unit(value: float, category: str) -> float:
    if category != "near_full":
        return pi * value
    # Near-full rectangles fit primarily close to their material axes.
    if value < 0.5:
        local = 2.0 * value
        return (local - 0.5) * np.deg2rad(24.0)
    local = 2.0 * (value - 0.5)
    return pi / 2.0 + (local - 0.5) * np.deg2rad(24.0)


def _balanced_strip_targets(total: int, offset: int = 0) -> dict[int, int]:
    values = list(range(3, 15))
    base, remainder = divmod(total, len(values))
    targets = {value: base for value in values}
    for index in range(remainder):
        targets[values[(offset + index) % len(values)]] += 1
    return targets


def generate_actions(
    n_samples: int = 500,
    seed: int = 20260820,
    panel: Panel = Panel(),
) -> tuple[list[DecodedZigzag], dict]:
    targets = {
        "local": int(round(0.30 * n_samples)),
        "medium": int(round(0.50 * n_samples)),
    }
    targets["near_full"] = n_samples - targets["local"] - targets["medium"]
    accepted: list[DecodedZigzag] = []
    rejected = Counter()
    proposed = Counter()
    strip_targets_by_class = {
        category: _balanced_strip_targets(targets[category], offset=3 * index)
        for index, category in enumerate(("local", "medium", "near_full"))
    }

    for category_index, category in enumerate(("local", "medium", "near_full")):
        target = targets[category]
        strip_targets = strip_targets_by_class[category]
        accepted_by_strip = Counter()
        category_samples: list[DecodedZigzag] = []
        batch_index = 0
        while len(category_samples) < target:
            sampler = qmc.LatinHypercube(
                d=10, seed=seed + 1000 * category_index + batch_index
            )
            unit = sampler.random(2048)
            area_min, area_max = _category_area_range(category)
            for row in unit:
                if len(category_samples) >= target:
                    break
                proposed[category] += 1
                log_area = log(area_min) + row[0] * (log(area_max) - log(area_min))
                area = exp(log_area)
                aspect_bound = min(log(4.0), max(0.0, -log(area)) * 0.96)
                log_aspect = (2.0 * row[1] - 1.0) * aspect_bound
                n_strips = 3 + min(int(row[9] * 12), 11)
                if accepted_by_strip[n_strips] >= strip_targets[n_strips]:
                    rejected[f"{category}:strip_quota_filled"] += 1
                    continue
                action = ZigzagAction(
                    sample_id="pending",
                    requested_scale_class=category,
                    log_area_scale=log_area,
                    log_aspect=log_aspect,
                    # Work in a first-quadrant representative of the D2 orbit.
                    xi_u=row[2],
                    xi_v=row[3],
                    rotation_rad=_rotation_from_unit(row[4], category) % pi,
                    n_strips=n_strips,
                    log_band_fraction=log(0.004)
                    + row[5] * (log(0.075) - log(0.004)),
                    growth_membrane=-0.0015 + 0.0045 * row[6],
                    growth_curvature=-0.0030 + 0.0060 * row[7],
                    orthotropy=-0.8 + 1.6 * row[8],
                )
                try:
                    decoded = decode_action(action, panel=panel)
                    if decoded.actual_scale_class != category:
                        rejected[f"{category}:class_mismatch"] += 1
                        continue
                except ValueError as error:
                    rejected[f"{category}:{str(error)}"] += 1
                    continue

                sample_number = len(accepted) + len(category_samples)
                identified_action = ZigzagAction(
                    **{
                        **asdict(action),
                        "sample_id": f"zigzag_{sample_number:04d}",
                    }
                )
                category_samples.append(decode_action(identified_action, panel=panel))
                accepted_by_strip[n_strips] += 1
            batch_index += 1
            if batch_index > 80:
                raise RuntimeError(f"Could not generate enough {category} samples")
        accepted.extend(category_samples)

    # Interleave scale classes rather than storing three contiguous blocks.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(accepted))
    shuffled = [accepted[index] for index in order]
    renumbered = []
    for index, decoded in enumerate(shuffled):
        action = ZigzagAction(
            **{**asdict(decoded.action), "sample_id": f"zigzag_{index:04d}"}
        )
        renumbered.append(decode_action(action, panel=panel))

    summary = {
        "seed": seed,
        "requested_samples": n_samples,
        "accepted_samples": len(renumbered),
        "panel_half_extents_m": [panel.half_x_m, panel.half_y_m],
        "panel_full_dimensions_m": [panel.width_m, panel.height_m],
        "canonical_symmetry_domain": "rectangle_D2_first_quadrant_center",
        "target_counts": targets,
        "strip_targets_by_class": strip_targets_by_class,
        "accepted_counts": dict(Counter(d.actual_scale_class for d in renumbered)),
        "proposed_counts": dict(proposed),
        "rejected_counts": dict(rejected),
    }
    return renumbered, summary


def _flat_record(decoded: DecodedZigzag) -> dict:
    action = decoded.action
    return {
        "sample_id": action.sample_id,
        "scale_class": decoded.actual_scale_class,
        "log_area_scale": action.log_area_scale,
        "log_aspect": action.log_aspect,
        "xi_u": action.xi_u,
        "xi_v": action.xi_v,
        "rotation_deg": 180.0 * action.rotation_rad / pi,
        "n_strips": action.n_strips,
        "log_band_fraction": action.log_band_fraction,
        "growth_membrane": action.growth_membrane,
        "growth_curvature": action.growth_curvature,
        "orthotropy": action.orthotropy,
        "width_mm": 1000.0 * decoded.width_m,
        "height_mm": 1000.0 * decoded.height_m,
        "pitch_mm": 1000.0 * decoded.pitch_m,
        "band_width_mm": 1000.0 * decoded.band_width_m,
        "alpha_deg": 180.0 * decoded.alpha_rad / pi,
        "center_u_mm": 1000.0 * decoded.center_u_m,
        "center_v_mm": 1000.0 * decoded.center_v_m,
        "coverage_fraction": decoded.coverage_fraction,
        "centerline_area_fraction": decoded.centerline_area_fraction,
        "pitch_band_ratio": decoded.pitch_band_ratio,
        "path_length_mm": 1000.0 * decoded.path_length_m,
        "growth_top": decoded.growth_top,
        "growth_bottom": decoded.growth_bottom,
    }


def write_samples(
    decoded_samples: Sequence[DecodedZigzag],
    summary: dict,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    records = [_flat_record(decoded) for decoded in decoded_samples]
    with (output_directory / "zigzag_samples_500.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with (output_directory / "zigzag_samples_500.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for decoded in decoded_samples:
            payload = {
                "sample_id": decoded.action.sample_id,
                "scale_class": decoded.actual_scale_class,
                "action": asdict(decoded.action),
                "derived": {
                    key: value
                    for key, value in _flat_record(decoded).items()
                    if key not in {"sample_id", "scale_class"}
                },
                "sequence_v1": decoded.sequence_v1(),
            }
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    numeric = {
        key: np.asarray([record[key] for record in records], dtype=float)
        for key in records[0]
        if key not in {"sample_id", "scale_class"}
    }
    summary = {
        **summary,
        "quantiles": {
            key: {
                "min": float(values.min()),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
            for key, values in numeric.items()
        },
    }
    (output_directory / "zigzag_samples_500_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for category in ("local", "medium", "near_full"):
        members = sorted(
            (
                decoded
                for decoded in decoded_samples
                if decoded.actual_scale_class == category
            ),
            key=lambda decoded: decoded.coverage_fraction,
        )
        representative = members[len(members) // 2]
        (output_directory / f"representative_{category}.json").write_text(
            json.dumps(representative.sequence_v1(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def plot_sample_gallery(
    samples: Sequence[DecodedZigzag], output_directory: Path, seed: int = 9
) -> None:
    rng = np.random.default_rng(seed)
    chosen = []
    for category in ("local", "medium", "near_full"):
        members = [sample for sample in samples if sample.actual_scale_class == category]
        chosen.extend(rng.choice(members, size=6, replace=False).tolist())
    panel = Panel()
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.5), squeeze=False)
    for ax, decoded in zip(axes.ravel(), chosen):
        points = zigzag_vertices(decoded)
        ax.plot(1000.0 * points[:, 0], 1000.0 * points[:, 1], "-o", ms=2, lw=1.5)
        ax.scatter(1000.0 * points[0, 0], 1000.0 * points[0, 1], c="green", s=18)
        ax.scatter(1000.0 * points[-1, 0], 1000.0 * points[-1, 1], c="red", marker="x", s=22)
        ax.set_xlim(-1000 * panel.half_x_m, 1000 * panel.half_x_m)
        ax.set_ylim(-1000 * panel.half_y_m, 1000 * panel.half_y_m)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{decoded.actual_scale_class}\n"
            f"A={decoded.coverage_fraction:.2f}, N={decoded.action.n_strips}",
            fontsize=8,
        )
    fig.suptitle("Scale-aware zigzag samples on the 254 × 304.8 mm material domain")
    fig.tight_layout()
    fig.savefig(output_directory / "zigzag_samples_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_coverage(
    samples: Sequence[DecodedZigzag], output_directory: Path
) -> None:
    records = [_flat_record(decoded) for decoded in samples]
    fields = (
        "coverage_fraction",
        "width_mm",
        "height_mm",
        "rotation_deg",
        "n_strips",
        "band_width_mm",
        "pitch_band_ratio",
        "growth_membrane",
        "growth_curvature",
    )
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, field in zip(axes.ravel(), fields):
        values = [record[field] for record in records]
        ax.hist(values, bins=24, color="tab:blue", alpha=0.72)
        ax.set_title(field)
        ax.grid(alpha=0.18)
    fig.suptitle("Coverage of the 500-sample zigzag action design")
    fig.tight_layout()
    fig.savefig(output_directory / "zigzag_parameter_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    colors = {"local": "tab:green", "medium": "tab:blue", "near_full": "tab:red"}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for category in colors:
        members = [decoded for decoded in samples if decoded.actual_scale_class == category]
        ax.scatter(
            [decoded.width_m / Panel().width_m for decoded in members],
            [decoded.height_m / Panel().height_m for decoded in members],
            c=[decoded.action.n_strips for decoded in members],
            cmap="viridis",
            s=34,
            alpha=0.75,
            marker={"local": "o", "medium": "s", "near_full": "^"}[category],
            label=category,
        )
    ax.set_xlabel("normalized centerline width")
    ax.set_ylabel("normalized centerline height")
    ax.set_title("Scale, aspect ratio, and strip-count coverage")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_directory / "zigzag_scale_aspect_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_and_write(
    output_directory: Path,
    n_samples: int = 500,
    seed: int = 20260820,
) -> tuple[list[DecodedZigzag], dict]:
    samples, summary = generate_actions(n_samples=n_samples, seed=seed)
    write_samples(samples, summary, output_directory)
    plot_sample_gallery(samples, output_directory)
    plot_parameter_coverage(samples, output_directory)
    return samples, summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output = root / "notebooks" / "zigzag_action_outputs"
    samples, summary = generate_and_write(output)
    print(f"Generated {len(samples)} zigzag actions")
    print(f"Scale counts: {summary['accepted_counts']}")
    print(f"Proposals: {summary['proposed_counts']}")
    print(f"Rejections: {sum(summary['rejected_counts'].values())}")
    print(f"Wrote outputs to {output}")
