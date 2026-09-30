#!/usr/bin/env python3
"""Render a minimal, illustrative three-cycle zigzag sequence (not a run recipe).

Usage: python python/plot_toolpath_sequence_overview.py [--out OUTPUT_DIRECTORY]
Exports transparent PNG, PDF, and SVG images.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from visualize_solver_characteristics import place_strips


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run/project_overview")
    args = parser.parse_args()
    # Deliberately illustrative coverage domains: left, upper, and lower-right.
    operations = [
        dict(type="zigzag", lv_mm=142, alpha_deg=6.8, n_strips=6,
             width_mm=8, center_uv_mm=[-48, 0], rotation_deg=0),
        dict(type="zigzag", lv_mm=150, alpha_deg=7.5, n_strips=6,
             width_mm=8, center_uv_mm=[0, 46], rotation_deg=90),
        dict(type="zigzag", lv_mm=105, alpha_deg=9, n_strips=6,
             width_mm=8, center_uv_mm=[30, -29], rotation_deg=-38),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    fig.subplots_adjust(left=.025, right=.975, bottom=.06, top=.94, wspace=.38)
    for axis, operation in zip(axes, operations):
        axis.add_patch(FancyBboxPatch((-100, -100), 200, 200,
                                     boxstyle="round,pad=0,rounding_size=9",
                                     facecolor="#F0F4F7", edgecolor="#D6DFE6", lw=1))
        strips = place_strips(operation, (-.1, .1, -.1, .1))["strips"]
        points = np.vstack([strips[0]["a_uv_m"]] + [strip["b_uv_m"] for strip in strips]) * 1000
        axis.plot(points[:, 0], points[:, 1], color="#246A91", linewidth=3.5,
                  solid_capstyle="round", solid_joinstyle="round")
        axis.set(xlim=(-106, 106), ylim=(-106, 106), aspect="equal")
        axis.axis("off")
    fig.canvas.draw()
    for left, right in zip(axes[:-1], axes[1:]):
        a, b = left.get_position(), right.get_position()
        y = (a.y0 + a.y1) / 2
        fig.add_artist(FancyArrowPatch((a.x1+.012, y), (b.x0-.012, y),
                                      transform=fig.transFigure, arrowstyle="-|>",
                                      mutation_scale=15, lw=1.8, color="#A9B5BF"))
    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        target = args.out / f"zigzag_toolpath_sequence.{extension}"
        fig.savefig(target, dpi=300, transparent=True)
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
