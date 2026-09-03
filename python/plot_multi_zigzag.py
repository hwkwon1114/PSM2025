"""Render the patch-clipped centerlines defined by a multi_zigzag JSON file."""

from __future__ import annotations

import argparse
import json
from math import ceil, radians, tan
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


def next_even_at_least(value: int, minimum: int = 4) -> int:
    value = max(value, minimum)
    return value if value % 2 == 0 else value + 1


def decode_patch(patch: dict) -> tuple[np.ndarray, np.ndarray, int]:
    corners = np.asarray(patch["corners_mm"], dtype=float)
    center = corners.mean(axis=0)
    width_vector = corners[1] - corners[0]
    height_vector = corners[2] - corners[1]
    width = float(np.linalg.norm(width_vector))
    height = float(np.linalg.norm(height_vector))
    axis_x = width_vector / width
    axis_y = height_vector / height

    alpha = radians(float(patch.get("zigzag_alpha_deg", 15.0)))
    band = float(patch.get("zigzag_w_mm", 2.0))
    required = (width - band) / (height * abs(tan(alpha))) + 2.0
    n_strips = next_even_at_least(
        int(ceil(required)), int(patch.get("zigzag_N_min", 4))
    )
    n_strips = min(n_strips, int(patch.get("zigzag_N_max", 200)))

    pitch = height * tan(alpha)
    span = (n_strips - 2) * pitch
    x_left = -0.5 * span
    x_right = 0.5 * span
    y_bottom = -0.5 * height
    y_top = 0.5 * height
    points = [[x_left, y_bottom], [x_left, y_top]]
    for index in range(1, n_strips - 1):
        points.append(
            [x_left + index * pitch, y_bottom if index % 2 else y_top]
        )
    last_y = points[-1][1]
    points.append([x_right, y_bottom if np.isclose(last_y, y_top) else y_top])
    local = np.asarray(points)
    global_points = center + local[:, :1] * axis_x + local[:, 1:] * axis_y
    return corners, global_points, n_strips


def plot_config(config_path: Path, output_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(8.2, 8.8))
    axis.add_patch(
        Rectangle(
            (-127.0, -152.4),
            254.0,
            304.8,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            label="sheet boundary",
        )
    )
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for index, patch in enumerate(config["patches"]):
        if not patch.get("enabled", True):
            continue
        color = colors[index % len(colors)]
        corners, path, n_strips = decode_patch(patch)
        polygon = Polygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=2.0,
        )
        axis.add_patch(polygon)
        line, = axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=2.0,
            label=f"{patch['name']} (N={n_strips})",
        )
        line.set_clip_path(polygon)
        axis.text(
            *corners.mean(axis=0),
            patch["name"].replace("_", " "),
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
    axis.set_xlim(-135.0, 135.0)
    axis.set_ylim(-160.0, 160.0)
    axis.set_aspect("equal")
    axis.set_xlabel("material x (mm)")
    axis.set_ylabel("material y (mm)")
    axis.set_title("Multi-zigzag: orthogonal rectangular patches")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_config(args.config, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
