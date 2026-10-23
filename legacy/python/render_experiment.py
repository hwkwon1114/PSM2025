#!/usr/bin/env python3
"""Render arbitrary VTP experiment outputs in one consistent comparison layout.

Example:
  python render_experiment.py --case c3=/tmp/c3/final.vtp --case c4=/tmp/c4/final.vtp --out comparison.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def load_vtp(path):
    reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
    data = reader.GetOutput()
    if data.GetPoints() is None:
        raise RuntimeError("failed to read VTP: %s" % path)
    pd = data.GetPointData()
    current_array = pd.GetArray("X_current")
    if current_array is not None:
        points = vtk_to_numpy(current_array).reshape(-1, 3).astype(float)
    else:
        points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    u_array = pd.GetArray("material_u")
    v_array = pd.GetArray("material_v")
    if u_array is None or v_array is None:
        u = points[:, 0].copy()
        v = points[:, 1].copy()
    else:
        u = vtk_to_numpy(u_array).astype(float)
        v = vtk_to_numpy(v_array).astype(float)
    return points, triangles, u, v


def parse_case(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be LABEL=VTP_PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True,
                        help="comparison case as LABEL=VTP_PATH; repeat for multiple cases")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Experiment comparison")
    parser.add_argument("--z-scale", type=float, default=1.0,
                        help="visual z exaggeration; default 1 preserves physical aspect")
    args = parser.parse_args()
    loaded = [(label, load_vtp(path)) for label, path in args.case]
    n = len(loaded)
    fig = plt.figure(figsize=(6.2 * n, 8.0), constrained_layout=True)
    for i, (label, (points, triangles, u, v)) in enumerate(loaded):
        ax = fig.add_subplot(2, n, i + 1, projection="3d")
        p = points.copy(); p[:, 2] *= args.z_scale
        z = p[triangles][:, :, 2].mean(axis=1)
        norm = colors.Normalize(vmin=float(z.min()), vmax=float(z.max()) if z.max() > z.min() else float(z.min()) + 1e-12)
        ax.plot_trisurf(1000*p[:, 0], 1000*p[:, 1], 1000*p[:, 2],
                        triangles=triangles, cmap="viridis", norm=norm,
                        linewidth=0.0, edgecolor="none")
        ax.set_title(label); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.view_init(elev=25, azim=-60)
        aspect = np.ptp(points[:,2]) / max(np.ptp(points[:,:2]), 1e-12)
        ax.set_box_aspect((1.0, 1.0, max(0.08, args.z_scale * aspect)))
        ax2 = fig.add_subplot(2, n, n + i + 1)
        contour = ax2.tripcolor(1000*u, 1000*v, triangles, 1000*points[:,2], cmap="viridis", shading="gouraud")
        ax2.set_aspect("equal"); ax2.set_title("material-plane elevation")
        ax2.set_xlabel("u (mm)"); ax2.set_ylabel("v (mm)"); fig.colorbar(contour, ax=ax2, label="z (mm)")
    fig.suptitle(args.title, fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
