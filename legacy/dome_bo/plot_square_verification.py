#!/usr/bin/env python3
"""SUPERSEDED: the qualification mask here omits the weaker/stronger ratio >= 0.1 and fit-validity conditions; use
whole_sheet_diagnostics.py. Kept unchanged in logic for provenance.
Whole-sheet height (true scale, best-fit plane removed) and qualifying-curvature maps, screening vs verification.
Usage: plot_square_verification.py SCREEN_STATE VERIFY_STATE OUT.png  (frozen 15 mm radius, 0.1 1/m floor)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import square_attempt as sa            # noqa: E402
import dome_continuous_objective as dco  # noqa: E402

fig, axes = plt.subplots(2, 2, figsize=(9, 8.4), constrained_layout=True)
for row, (name, path) in enumerate((('screening 4 mm', sys.argv[1]), ('verification 2 mm', sys.argv[2]))):
    st = json.loads(Path(path).read_text())
    uv = np.asarray(st['material_uv'], float) * 1e3
    pos = np.asarray(st['positions'], float) * 1e3
    faces = np.asarray(st['faces'], int)
    A = np.c_[uv, np.ones(len(uv))]
    z = pos[:, 2] - A @ np.linalg.lstsq(A, pos[:, 2], rcond=None)[0]
    vals, _, valid = sa.geometric_curvature(np.asarray(st['positions'], float), uv * 1e-3, faces, radius=0.015)
    f = vals                           # geometric_curvature returns per-face principal curvatures
    k = np.sign(f.mean(1))
    ok = (np.abs(f).min(1) >= 0.1) & (f[:, 0] * f[:, 1] > 0)
    cat = np.where(ok, k, 0.0)         # -1/+1 qualifying sign, 0 not qualifying
    cat = np.where((np.abs(f).min(1) >= 0.1) & (f[:, 0] * f[:, 1] < 0), 2.0, cat)
    for col, (vals_, cmap, lab) in enumerate(((z[faces].mean(1), 'viridis', 'height (mm), plane removed'),
                                              (cat, matplotlib.colors.ListedColormap(['#4477aa', '#dddddd', '#cc6677', '#222222']),
                                               'qualifying face: blue/red = one sign, grey = none, black = saddle'))):
        ax = axes[row, col]
        pc = PolyCollection(uv[faces][:, :, :2], array=vals_ if col == 0 else cat + 1, cmap=cmap, edgecolors='none',
                            clim=None if col == 0 else (0, 3))
        ax.add_collection(pc); ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_aspect('equal')
        ax.add_patch(plt.Rectangle((-80, -80), 160, 160, fill=False, ec='k', lw=1, ls='--'))
        ax.set_title(f'{name}: {lab}', fontsize=8); ax.set_xlabel('u (mm)'); ax.set_ylabel('v (mm)')
        if col == 0:
            fig.colorbar(pc, ax=ax, shrink=0.8)
fig.savefig(sys.argv[3], dpi=160)
