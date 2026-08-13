"""
Render the tolerance x mesh grid for the results artifact.

Produces:
  shapes_grid.png  -- 2x5 small multiples of the final deformed shape, coloured by
                      mean curvature on a shared diverging scale and a shared view,
                      so the panels are directly comparable
  results.json     -- the metrics table the artifact's charts read

Usage:
    python python/render_results.py <workdir> <outdir>
"""

import base64
import glob
import json
import os
import re
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection             # noqa: E402

MESHES = ["0.005", "0.0025"]
TOLS = ["1e-4", "1e-5", "1e-6", "1e-7", "2.2e-16"]

# diverging blue <-> red with a neutral grey midpoint, per the reference palette
DIVERGING = LinearSegmentedColormap.from_list(
    "bwr_ref", ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f0a09f", "#e34948", "#8f2b2a"])


def read_stage(rundir, stage=None):
    cands = [f for f in glob.glob(os.path.join(rundir, "*growth_[0-9]*.vtp"))
             if "energies" not in os.path.basename(f)]
    if not cands:
        return None

    def idx(f):
        m = re.search(r"growth_(\d+)\.vtp$", f)
        return int(m.group(1)) if m else -1

    fn = max(cands, key=idx) if stage is None else next(
        (f for f in cands if idx(f) == stage), None)
    if fn is None or idx(fn) == 0:
        return None

    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(fn)
    r.Update()
    pd = r.GetOutput()
    pts = v2n(pd.GetPoints().GetData()).astype(float)
    tris = v2n(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    cd = pd.GetCellData()
    meanc = v2n(cd.GetArray("mean")).astype(float)
    gauss = v2n(cd.GetArray("gauss")).astype(float)
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return dict(pts=pts, tris=tris, mean=meanc, gauss=gauss, areas=areas, stage=idx(fn))


def area_norm(field, areas):
    return np.sqrt(np.sum(areas * field**2) / np.sum(areas))


def cost_of(workdir, res, tol):
    fn = os.path.join(workdir, "grid_%s_%s.txt" % (res, tol))
    if not os.path.exists(fn):
        return {}
    p = open(fn).read().split()
    try:
        return dict(iters=int(p[4]), gnorm=float(p[5]), seconds=int(p[6]))
    except (IndexError, ValueError):
        return {}


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/projects/p33082/psm_scaling"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/psm_artifact"
    os.makedirs(outdir, exist_ok=True)

    cells = {}
    for res in MESHES:
        for tol in TOLS:
            d = os.path.join(workdir, "grid_res%s_tol%s" % (res, tol))
            data = read_stage(d)
            if data is not None:
                cells[(res, tol)] = data

    # shared symmetric colour scale, robust to a few extreme faces
    allm = np.concatenate([c["mean"] for c in cells.values()])
    vmax = float(np.percentile(np.abs(allm), 98))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # shared axis limits so panel-to-panel size differences are real, not framing
    allp = np.concatenate([c["pts"] for c in cells.values()])
    ctr = allp.mean(axis=0)
    rad = float(np.abs(allp - ctr).max()) * 0.62

    fig = plt.figure(figsize=(15.0, 7.4))
    fig.patch.set_alpha(0.0)

    metrics = []
    for i, res in enumerate(MESHES):
        ref = cells.get((res, "2.2e-16"))
        for j, tol in enumerate(TOLS):
            ax = fig.add_subplot(len(MESHES), len(TOLS), i * len(TOLS) + j + 1,
                                 projection="3d")
            ax.set_axis_off()
            ax.patch.set_alpha(0.0)

            c = cells.get((res, tol))
            if c is None:
                ax.text2D(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
                continue

            verts = c["pts"][c["tris"]]
            colors = DIVERGING(norm(c["mean"]))

            # Hand-rolled diffuse shading: matplotlib's shade=True conflicts with
            # edgecolors="none" here, and doing it manually keeps the curvature hue
            # readable while still giving the surface enough relief to read as 3D.
            n = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.where(ln > 0, ln, 1.0)
            light = np.array([-0.35, -0.45, 0.82])
            light = light / np.linalg.norm(light)
            lam = np.abs(n @ light)
            shade = (0.62 + 0.38 * lam)[:, None]
            colors[:, :3] = np.clip(colors[:, :3] * shade, 0, 1)

            coll = Poly3DCollection(verts, facecolors=colors, edgecolors="none",
                                    linewidths=0, shade=False)
            ax.add_collection3d(coll)

            ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
            ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
            ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-58)

            rel = None
            if ref is not None and len(c["mean"]) == len(ref["mean"]):
                hn = area_norm(ref["mean"], ref["areas"])
                rel = float(area_norm(c["mean"] - ref["mean"], ref["areas"]) / hn) if hn else None

            entry = dict(res=res, tol=tol, faces=int(len(c["mean"])),
                         relL2H=rel, stage=int(c["stage"]))
            entry.update(cost_of(workdir, res, tol))
            entry["intH"] = float(np.sum(c["areas"] * np.abs(c["mean"])))
            entry["intK"] = float(np.sum(c["areas"] * np.abs(c["gauss"])))
            entry["area"] = float(c["areas"].sum())
            metrics.append(entry)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=-0.16, hspace=-0.22)
    png = os.path.join(outdir, "shapes_grid.png")
    fig.savefig(png, dpi=132, transparent=True)
    plt.close(fig)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(dict(meshes=MESHES, tols=TOLS, vmax=vmax, metrics=metrics), f, indent=1)

    with open(os.path.join(outdir, "shapes_grid.b64"), "w") as f:
        f.write(base64.b64encode(open(png, "rb").read()).decode())

    print("wrote %s (%.1f KB)" % (png, os.path.getsize(png) / 1024))
    print("cells rendered: %d" % len(metrics))
    for m in metrics:
        print("  res=%-7s tol=%-8s faces=%-5d relL2H=%s iters=%s s=%s"
              % (m["res"], m["tol"], m["faces"],
                 ("%.3e" % m["relL2H"]) if m.get("relL2H") is not None else "ref",
                 m.get("iters", "?"), m.get("seconds", "?")))


if __name__ == "__main__":
    main()
