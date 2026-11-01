"""Visualize the three accepted turned-over pilot shapes; no solver or recalibration."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_design.analysis import height, quadratic_fit

OUT = ROOT / 'run/shape_design/pilot_visualization'
CASES = [(-4, 'Edge-focused'), (0, 'Uniform'), (4, 'Center-focused')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    out = parser.parse_args().out_dir
    if any(out.glob('turned_over_pilot_shapes.*')):
        raise FileExistsError(f'Refusing to overwrite shapes in {out}')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, label in CASES:
        folder = ROOT / 'run/shape_design/pilot' / f'flip_k{k:+d}'
        result = json.loads((folder / 'result.json').read_text())
        if result['status'] != 'accepted' or result['hit_check']['mismatched_faces'] != 0:
            raise ValueError(f'Case is not accepted: {folder}')
        path = folder / 'shape.npz'
        with np.load(path) as d:
            ref, pos, faces = (d[k].copy() for k in ('reference_mm', 'aligned_mm', 'faces'))
        h = height(pos)
        rows.append(dict(k=k, label=label, ref=ref, pos=pos, faces=faces, h=h,
                         fit=quadratic_fit(ref, h), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    lim = max(np.max(np.abs(r['h'])) for r in rows)
    lim = np.ceil(lim * 10) / 10
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, left=.06, right=.88, bottom=.10, top=.94, hspace=.23, wspace=.22)
    last = None
    for col, r in enumerate(rows):
        p, h = r['pos'], r['h']
        ax = fig.add_subplot(gs[0, col], projection='3d')
        ax.plot_trisurf(p[:, 0], p[:, 1] + 65, h, triangles=r['faces'], cmap='RdBu_r', norm=norm,
                        linewidth=0, antialiased=False, rasterized=True, shade=False)
        ax.set(xlim=(-110, 110), ylim=(3, 127), zlim=(-lim, lim),
               xlabel='Width x (mm)', ylabel='Length z (mm)', zlabel='Height y (mm)')
        ax.set_yticks([3, 65, 127])
        ax.set_box_aspect((220, 124, 2*lim*10))
        ax.view_init(elev=25, azim=-65)
        ax.set_title(f"{r['label']} (k={r['k']:+d})", pad=14)
        ax.tick_params(labelsize=8, pad=1)
        ax = fig.add_subplot(gs[1, col])
        tri = mtri.Triangulation(r['ref'][:, 0], r['ref'][:, 1] + 65, r['faces'])
        last = ax.tripcolor(tri, h, shading='gouraud', cmap='RdBu_r', norm=norm, rasterized=True)
        ax.set_aspect('equal')
        ax.set(xlabel='Width x (mm)', ylabel='Length z (mm)', xlim=(-110,110), ylim=(3,127))
        ax.set_yticks([3, 65, 127])
        ax.set_title(f"{r['fit']['label']} · quadratic fit", fontsize=10)
    cax = fig.add_axes([.91, .18, .015, .29])
    fig.colorbar(last, cax=cax, label='Height y (mm)')
    fig.canvas.draw()
    for ax in fig.axes:
        for t in [ax.title, ax.xaxis.label, ax.yaxis.label, *([ax.zaxis.label] if hasattr(ax,'zaxis') else [])]:
            if t.get_visible() and t.get_text():
                b = t.get_window_extent(fig.canvas.get_renderer())
                if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1:
                    raise ValueError(f'Text outside page: {t.get_text()}')
    stem = out / 'turned_over_pilot_shapes'
    fig.savefig(str(stem)+'.png', dpi=200)
    fig.savefig(str(stem)+'.pdf', dpi=200)
    plt.close(fig)
    caption = ('# Turned-over pilot shapes\n\nThree accepted 2 mm mesh pilot simulations, selected as the predeclared edge, uniform and center settings. '
               'Top: aligned 3-D shapes; height axis visually exaggerated 10×. Bottom: heights on the material-reference x–z plane. '
               'All panels use one shared zero-centered color scale; each height field is mean-removed after rigid alignment. '
               'Labels come from the predeclared quadratic fit in the treated region, not a local curvature classification. '
               'The edge-focused saddle label is near the 0.2 curvature-ratio threshold. These are exploratory predictions, not experimental scans, '
               'The fine-mesh endpoints have accepted solves, but their comparison remains pending. Public length z = v + 65 mm. '
               'The one-face edge case failed and is not shown.\n')
    (out / 'turned_over_pilot_shapes.caption.md').write_text(caption)
    (out / 'turned_over_pilot_shapes.provenance.json').write_text(json.dumps([
        dict(k=r['k'], path=f"run/shape_design/pilot/flip_k{r['k']:+d}/shape.npz", sha256=r['sha256'], fit=r['fit']) for r in rows], indent=2)+'\n')


if __name__ == '__main__':
    main()
