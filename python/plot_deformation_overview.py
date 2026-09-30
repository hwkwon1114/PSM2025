#!/usr/bin/env python3
"""Render a minimal 3D surface colored by rigid-aligned displacement magnitude.

Uses an existing simulation, with true physical proportions and no exaggeration.
Exports transparent PNG/PDF/SVG and JSON containing source and color-scale units.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from analyze_sequence_ablation_v2 import load_vtp, proper_kabsch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "run/forward_model_diagnostics/current/nested_recheck_g0p0015_broad_to_central_standard"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--out", type=Path, default=ROOT / "run/project_overview")
    args = parser.parse_args()
    result = json.loads((args.case / "result.json").read_text())
    if result["return_code"] != 0 or result.get("timed_out", False):
        raise ValueError("Choose a completed simulation")
    final_path = args.case / result["final_vtp"]
    initial_path = args.case / "bilayer_zigzag_sequence_cycle_000_initial.vtp"
    _, reference, initial_triangles = load_vtp(initial_path)
    _, moving, triangles = load_vtp(final_path)
    if reference.shape != moving.shape or not np.array_equal(initial_triangles, triangles):
        raise ValueError("Initial and final meshes must have matching topology")
    aligned, rms, maximum = proper_kabsch(reference, moving)
    displacement_mm = np.linalg.norm(aligned - reference, axis=1) * 1000
    points = (aligned - aligned.mean(axis=0)) * 1000
    if not np.isfinite(points).all() or not np.isfinite(displacement_mm).all():
        raise ValueError("Nonfinite displacement data")
    norm = Normalize(0, max(float(displacement_mm.max()), 1e-12))
    fig = plt.figure(figsize=(6, 5))
    axis = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho")
    surface = axis.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                               triangles=triangles, linewidth=0, edgecolor="none",
                               antialiased=False, cmap="viridis", norm=norm, shade=False)
    surface.set_array(displacement_mm[triangles].mean(axis=1))
    extent = np.ptp(points, axis=0)
    extent = np.maximum(extent, extent.max() * 1e-6)
    midpoint = (points.min(axis=0) + points.max(axis=0)) / 2
    for setter, center, span in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), midpoint, extent):
        setter(center - span*.53, center + span*.53)
    axis.set_box_aspect(extent.copy())
    axis.view_init(elev=27, azim=-57)
    axis.set_axis_off()
    axis.patch.set_alpha(0)
    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        target = args.out / f"deformation_field_3d.{extension}"
        fig.savefig(target, dpi=300, transparent=True, pad_inches=0)
        print(target)
    plt.close(fig)
    evidence = dict(source=str(final_path), reference=str(initial_path),
                    field="displacement magnitude after proper rigid alignment to initial mesh",
                    units="mm", colormap="viridis", color_limits_mm=[0, float(displacement_mm.max())],
                    displacement_rms_mm=rms*1000, displacement_max_mm=maximum*1000,
                    surface_extents_mm=extent.tolist(), deformation_scale=1,
                    vertices=len(points), triangles=len(triangles))
    (args.out / "deformation_field_3d.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
