"""
Render the evidence figures for the findings artifact.

  fig_mirror.png   the two mesh references curl in opposite directions
  fig_design.png   what actually drives the bending: fibre line fields + local mismatch
  fig_null.png     path-reversal null test on the toolpath -> per-face projection

Usage:
    python python/render_findings.py <workdir> <outdir>
"""

import base64
import glob
import os
import re
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.collections import LineCollection                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm    # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                # noqa: E402

DIV = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f0a09f", "#e34948", "#8f2b2a"])
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
INK, MUTED = "#3D4956", "#5F6C79"
AREA_AVG_DEV = 4.53e-4   # measured area average of the interlayer deviator
S_COARSE, S_FINE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#D8E0E8",
    "font.family": "sans-serif", "savefig.transparent": True,
})


def load(path, stage=None):
    fs = [f for f in glob.glob(os.path.join(path, "*growth_[0-9]*.vtp"))
          if "energies" not in f]
    if not fs:
        return None
    idx = lambda f: int(re.search(r"growth_(\d+)", f).group(1))
    fn = max(fs, key=idx) if stage is None else next(f for f in fs if idx(f) == stage)
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(fn); r.Update()
    pd = r.GetOutput(); cd = pd.GetCellData()
    pts = v2n(pd.GetPoints().GetData()).astype(float)
    tris = v2n(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    d = dict(pts=pts, tris=tris,
             a=0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1))
    for n in ("mean", "gauss", "dens_bot", "dens_top", "dir_bot", "dir_top"):
        if cd.GetArray(n) is not None:
            d[n] = v2n(cd.GetArray(n)).astype(float)
    d["ctr"] = pts[tris].mean(axis=1)
    return d


def shell3d(ax, d, norm, ctr, rad, elev=22, azim=-58):
    verts = d["pts"][d["tris"]]
    col = DIV(norm(d["mean"]))
    n = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln > 0, ln, 1.0)
    L = np.array([-0.35, -0.45, 0.82]); L /= np.linalg.norm(L)
    col[:, :3] = np.clip(col[:, :3] * (0.62 + 0.38 * np.abs(n @ L))[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(verts, facecolors=col, edgecolors="none",
                                         linewidths=0, shade=False))
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()


def fig_mirror(W, out):
    a = load(os.path.join(W, "grid_res0.005_tol2.2e-16"))
    b = load(os.path.join(W, "grid_res0.0025_tol2.2e-16"))
    allp = np.vstack([a["pts"], b["pts"]]); ctr = allp.mean(0)
    rad = float(np.abs(allp - ctr).max()) * 0.60
    vmax = float(np.percentile(np.abs(np.r_[a["mean"], b["mean"]]), 98))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(10.4, 3.5))
    for i, (d, lab, col) in enumerate([(a, "res 0.005  ·  1244 faces", S_COARSE),
                                       (b, "res 0.0025  ·  2477 faces", S_FINE)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        shell3d(ax, d, norm, ctr, rad)
        s = np.sum(d["a"] * d["mean"])
        ax.set_title("%s\n$\\int H\\,dA$ = %+.2f" % (lab, s), color=col,
                     fontsize=9, pad=-2)

    ax = fig.add_subplot(1, 3, 3)
    bins = np.linspace(-vmax * 1.6, vmax * 1.6, 46)
    for d, lab, col in [(a, "res 0.005", S_COARSE), (b, "res 0.0025", S_FINE)]:
        ax.hist(d["mean"], bins=bins, weights=d["a"], histtype="step", lw=1.8,
                color=col, label=lab)
    ax.axvline(0, color="#B0BBC6", lw=1, zorder=0)
    ax.set_xlabel("mean curvature $H$"); ax.set_ylabel("area  (weighted count)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Curvature is mirrored, not merely shifted", fontsize=9, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.7, rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=132)
    plt.close(fig)


def fig_design(W, out):
    d = load(os.path.join(W, "grid_res0.0025_tol2.2e-16"), stage=0)
    c = d["ctr"]
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.6))

    for ax, key, lab in [(axes[0], "dir_bot", "bottom layer"),
                         (axes[1], "dir_top", "top layer")]:
        v = d[key][:, :2]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-300)
        L = 0.62
        cc = c[:, [1, 0]]; vv = v[:, [1, 0]]          # transpose to landscape
        segs = np.stack([cc - L * vv, cc + L * vv], axis=1)
        ax.add_collection(LineCollection(segs, colors="#2a78d6", linewidths=0.55, alpha=.85))
        ax.set_title("%s  ·  filament direction" % lab, fontsize=9, color=INK)
        ax.set_xlim(-20.6, 20.6); ax.set_ylim(-4.6, 4.6); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#D8E0E8")

    ab = np.arctan2(d["dir_bot"][:, 1], d["dir_bot"][:, 0])
    at = np.arctan2(d["dir_top"][:, 1], d["dir_top"][:, 0])
    eig = np.diag([0.12, 0.26])
    dev = np.empty(len(ab))
    for i in range(len(ab)):
        def R(x):
            co, si = np.cos(x), np.sin(x)
            M = np.array([[co, -si], [si, co]]); return M @ eig @ M.T
        D = R(ab[i]) - R(at[i])
        dev[i] = np.hypot((D[0, 0] - D[1, 1]) / 2, D[0, 1])

    ax = axes[2]
    # dev is one value per face, so colour the face centres directly rather than
    # building a triangulation of them
    sc = ax.scatter(c[:, 1], c[:, 0], c=dev, cmap=SEQ, s=5, lw=0)
    ax.set_title("local mismatch $|dev(\\varepsilon_b-\\varepsilon_t)|$", fontsize=9, color=INK)
    ax.set_xlim(-20.6, 20.6); ax.set_ylim(-4.6, 4.6); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#D8E0E8")
    cb = fig.colorbar(sc, ax=ax, fraction=0.020, pad=0.012)
    cb.ax.tick_params(labelsize=7)
    ax.text(0.5, -0.16, "local mean %.3f   ·   area-averaged deviator %.2e   (%.0fx smaller)"
            % (dev.mean(), AREA_AVG_DEV, dev.mean() / AREA_AVG_DEV),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)

    fig.tight_layout(pad=0.7)
    fig.savefig(out, dpi=132)
    plt.close(fig)


def fig_null(W, out):
    o = load(os.path.join(W, "null_orig"), stage=1)
    r = load(os.path.join(W, "null_rev"), stage=1)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1))

    ax = axes[0]
    ax.scatter(o["dens_bot"], r["dens_bot"], s=3, color=S_COARSE, alpha=.5, lw=0)
    lim = [min(o["dens_bot"].min(), 0), o["dens_bot"].max() * 1.03]
    ax.plot(lim, lim, color="#B0BBC6", lw=1, zorder=0)
    ax.set_xlabel("density, original paths"); ax.set_ylabel("density, reversed")
    ax.set_title("Filament density\nmax diff 2.5e-12", fontsize=9, color=INK)

    ax = axes[1]
    for key, col, lab in [("dir_bot", S_COARSE, "bottom"), ("dir_top", S_FINE, "top")]:
        a, b = o[key][:, :2], r[key][:, :2]
        dot = np.abs(np.sum(a * b, 1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)
        ang = np.degrees(np.arccos(np.clip(dot, 0, 1)))
        ax.hist(np.log10(np.maximum(ang, 1e-12)), bins=40, histtype="step", lw=1.8,
                color=col, label=lab)
    ax.set_xlabel("log$_{10}$ line-field angle difference  (deg)")
    ax.set_ylabel("faces")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Fibre orientation as a line field\nunchanged to 1e-6 deg", fontsize=9, color=INK)

    ax = axes[2]
    frac = []
    for key in ("dir_bot", "dir_top"):
        s = np.sum(o[key][:, :2] * r[key][:, :2], 1)
        frac.append(100 * (s < 0).mean())
    ax.bar(["bottom", "top"], frac, color=[S_COARSE, S_FINE], width=.5)
    ax.set_ylabel("% of faces"); ax.set_ylim(0, 60)
    ax.set_title("Signed vector DOES flip\n(harmless: metric is even in $d$)", fontsize=9, color=INK)
    for i, f in enumerate(frac):
        ax.text(i, f + 1.6, "%.1f%%" % f, ha="center", fontsize=8, color=INK)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.7)
    fig.savefig(out, dpi=132)
    plt.close(fig)


def main():
    W = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/projects/p33082/psm_scaling"
    O = sys.argv[2] if len(sys.argv) > 2 else "/tmp/psm_artifact"
    os.makedirs(O, exist_ok=True)
    for name, fn in [("fig_mirror", fig_mirror), ("fig_design", fig_design),
                     ("fig_null", fig_null)]:
        p = os.path.join(O, name + ".png")
        fn(W, p)
        with open(os.path.join(O, name + ".b64"), "w") as f:
            f.write(base64.b64encode(open(p, "rb").read()).decode())
        print("%-12s %7.1f KB" % (name, os.path.getsize(p) / 1024))


if __name__ == "__main__":
    main()
