#!/usr/bin/env python3
"""Final-state maps for the square-sheet mesh convergence study (retained states; no solves).
Usage: plot_mesh_study.py OUT_DIR  ->  mesh_study_attempt30_maps.png, mesh_study_candidates_height.png"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation, LinearTriInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import mesh_convergence as mc            # noqa: E402
import whole_sheet_diagnostics as wsd    # noqa: E402

MESH = [('.02', 4), ('.01', 2), ('.005', 1)]
COL = {4: '#0072B2', 2: '#E69F00', 1: '#009E73'}
CATS = ListedColormap(['#dddddd', '#4477aa', '#222222', '#ee9988'])


def load(a, r):
    sp = mc.cell_dir(a, r) / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json'
    return wsd.analyse(sp)[2]


def height_map(ax, d, vmin, vmax, cmap='viridis'):
    uv = d['uv'] * 1e3
    pc = PolyCollection(uv[d['faces']][:, :, :2], array=d['z'][d['faces']].mean(1) * 1e3, cmap=cmap,
                        edgecolors='none')
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_aspect('equal')
    ax.add_patch(plt.Rectangle((-80, -80), 160, 160, fill=False, ec='k', lw=0.8, ls='--'))
    return pc


def centre_profile(d, axis=0):
    uv = d['uv'] * 1e3
    tri = Triangulation(uv[:, 0], uv[:, 1], d['faces'])
    f = LinearTriInterpolator(tri, d['z'] * 1e3)
    s = np.linspace(-99.5, 99.5, 600)
    return s, (f(s, np.zeros_like(s)) if axis == 0 else f(np.zeros_like(s), s))


def fig_attempt(a, data, out):
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    allz = np.concatenate([data[(a, h)]['z'] for _, h in MESH]) * 1e3
    vmin, vmax = np.percentile(allz, [0.5, 99.5])
    for i, (r, h) in enumerate(MESH):
        d = data[(a, h)]
        w = d['area']
        gate = mc.json.loads((mc.STUDY / 'cell_metrics' / f'a{a}_{h}mm.json').read_text())['gate_fraction']
        pc = height_map(axes[i, 0], d, vmin, vmax)
        axes[i, 0].set_title(f'{h} mm: height, plane removed (mm)', fontsize=9)
        uv = d['uv'] * 1e3
        weaker = np.log10(np.maximum(np.min(np.abs(np.nan_to_num(d['vals'])), 1), 1e-3))
        pc2 = PolyCollection(uv[d['faces']][:, :, :2], array=weaker, cmap='magma', edgecolors='none')
        pc2.set_clim(-1.5, 0.5)
        axes[i, 1].add_collection(pc2)
        axes[i, 1].set_title(f'{h} mm: log10 weaker |curvature| (1/m)', fontsize=9)
        pc3 = PolyCollection(uv[d['faces']][:, :, :2], array=d['cat'], cmap=CATS, edgecolors='none')
        pc3.set_clim(-0.5, 3.5)
        axes[i, 2].add_collection(pc3)
        axes[i, 2].set_title(f'{h} mm: qualification (gate fraction {gate:.2f})', fontsize=9)
        for j in (1, 2):
            axes[i, j].set_xlim(-100, 100); axes[i, j].set_ylim(-100, 100); axes[i, j].set_aspect('equal')
            axes[i, j].add_patch(plt.Rectangle((-80, -80), 160, 160, fill=False, ec='k', lw=0.8, ls='--'))
        if i == 0:
            fig.colorbar(pc, ax=axes[:, 0], shrink=0.6, label='Height (mm)')
            fig.colorbar(pc2, ax=axes[:, 1], shrink=0.6, label='log10 weaker |curvature|')
        for j in range(3):
            axes[i, j].set_xlabel('u (mm)'); axes[i, j].set_ylabel('v (mm)')
        axes[i, 3].remove()
    ax = fig.add_subplot(3, 4, 4)
    ax2 = fig.add_subplot(3, 4, 8)
    ax3 = fig.add_subplot(3, 4, 12)
    ref = {}
    for r, h in MESH:
        d = data[(a, h)]
        for k, axx in ((0, ax), (1, ax2)):
            s, z = centre_profile(d, k)
            axx.plot(s, z, c=COL[h], lw=1.4, label=f'{h} mm')
            ref[(h, k)] = (s, z)
    for k, axx, t in ((0, ax, 'section v = 0'), (1, ax2, 'section u = 0')):
        axx.set_xlabel('Position (mm)'); axx.set_ylabel('Height (mm)'); axx.set_title(t, fontsize=9)
        axx.legend(fontsize=8)
    for k, ls in ((0, '-'), (1, '--')):
        for h in (4, 2):
            s, z = ref[(h, k)]
            z1 = ref[(1, k)][1]
            ax3.plot(s, z - z1, c=COL[h], ls=ls, lw=1.2, label=f'{h} mm - 1 mm, {"v" if k == 0 else "u"} section' )
    ax3.set_xlabel('Position (mm)'); ax3.set_ylabel('Height difference (mm)'); ax3.legend(fontsize=7)
    ax3.set_title('Difference to 1 mm', fontsize=9)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_candidates(cands, data, out):
    fig, axes = plt.subplots(len(cands), 4, figsize=(15, 3.6 * len(cands)), constrained_layout=True)
    for i, a in enumerate(cands):
        allz = np.concatenate([data[(a, h)]['z'] for _, h in MESH]) * 1e3
        vmin, vmax = np.percentile(allz, [0.5, 99.5])
        for j, (r, h) in enumerate(MESH):
            pc = height_map(axes[i, j], data[(a, h)], vmin, vmax)
            axes[i, j].set_title(f'Attempt {a}, {h} mm', fontsize=9)
            if j == 2:
                fig.colorbar(pc, ax=axes[i, :3], shrink=0.8, label='Height (mm), per-row scale')
        for h in (4, 2, 1):
            s, z = centre_profile(data[(a, h)], 0)
            axes[i, 3].plot(s, z, c=COL[h], lw=1.2, label=f'{h} mm')
        axes[i, 3].set_xlabel('u (mm), v = 0'); axes[i, 3].set_ylabel('Height (mm)')
        axes[i, 3].legend(fontsize=7)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cands = mc.CAND
    data = {(a, h): load(a, r) for a in cands for r, h in MESH}
    for a in (30, 31):
        fig_attempt(a, data, out / f'mesh_study_attempt{a}_maps.png')
    fig_candidates(cands, data, out / 'mesh_study_candidates_height.png')
    for a in cands:
        print(a, {h: round(float(np.ptp(data[(a, h)]['z'])) * 1e3, 3) for _, h in MESH})


if __name__ == '__main__':
    main(sys.argv[1])
