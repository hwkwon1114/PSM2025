#!/usr/bin/env python3
"""Read-only toolpath geometry preview; no mesh, growth assembly or solver calls."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

from visualize_solver_characteristics import place_strips, capsule_polygon

STYLES = {'toolpath_A': ('A', '#0072B2', '-'),
          'toolpath_B_scalar_trial': ('B', '#D55E00', '--'),
          'toolpath_C_scalar_trial': ('C', '#009E73', '-.')}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    s = json.loads(args.sequence.read_text())
    assert s['units'] == dict(length='mm', angle='deg', growth='engineering_strain')
    bbox = (-.127, .127, -.1524, .1524)
    records, segments = [], []
    with plt.rc_context({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(7.4, 9), layout='constrained')
        ax.add_patch(Rectangle((-127, -152.4), 254, 304.8,
                               facecolor='#fafafa', edgecolor='#333333',
                               linewidth=1.2, linestyle='--'))
        for t in s['toolpaths']:
            if not t.get('enabled', True):
                continue
            op = t['operation']
            assert op['type'] == 'zigzag' and op['shift_frame'] == 'material'
            label, color, ls = STYLES[t['id']]
            placed = place_strips(op, bbox)
            pts = []
            for i, strip in enumerate(placed['strips']):
                a, b = strip['a_uv_m']*1000, strip['b_uv_m']*1000
                w = placed['half_width_m']*1000
                poly = capsule_polygon(a, b, w, arc=64)
                ax.add_patch(Polygon(poly, facecolor=color, alpha=.17, edgecolor='none'))
                if i == 0:
                    pts.append(a)
                pts.append(b)
                segments.append(dict(toolpath=t['id'], strip=i, u0_mm=a[0], v0_mm=a[1],
                                     u1_mm=b[0], v1_mm=b[1], width_mm=2*w))
            pts = np.asarray(pts)
            lo, hi = pts.min(axis=0)-w, pts.max(axis=0)+w
            clearance = min(lo[0]+127, 127-hi[0], lo[1]+152.4, 152.4-hi[1])
            ax.plot(pts[:,0], pts[:,1], color=color, linestyle=ls, linewidth=1.3,
                    label=f'{label} × {t.get("repeat",1)}')
            ax.plot(*pts[0], 'o', color=color, markeredgecolor='black', markersize=6)
            ax.plot(*pts[-1], '^', color=color, markeredgecolor='black', markersize=6)
            d = pts[1]-pts[0]
            ax.annotate('', xy=pts[0]+.6*d, xytext=pts[0]+.35*d,
                        arrowprops=dict(arrowstyle='->', color='#222222', lw=1.1))
            ax.text(0, {'A':79, 'B':163, 'C':-170}[label],
                    f'{label} · {op["width_mm"]:g} mm strips', ha='center', fontsize=12)
            records.append(dict(id=t['id'], repeats=t.get('repeat',1),
                                profile=op.get('profile',s['defaults']['profile']),
                                footprint_bbox_mm=[lo.tolist(),hi.tolist()],
                                minimum_sheet_clearance_mm=clearance,
                                start_mm=pts[0].tolist(), end_mm=pts[-1].tolist()))
        ax.set(xlabel='Material u (mm)', ylabel='Material v (mm)',
               xlim=(-143,143), ylim=(-183,183), aspect='equal')
        ax.set_xticks([-100,-50,0,50,100])
        ax.set_yticks([-150,-100,-50,0,50,100,150])
        ax.legend(loc='upper center', bbox_to_anchor=(.5,1.065), ncol=3, frameon=False)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        # Check annotation overlap/containment plus labels and legend canvas bounds.
        boxes=[text.get_window_extent(renderer) for text in ax.texts if text.get_text()]
        for bb in boxes:
            assert ax.bbox.contains(bb.x0,bb.y0) and ax.bbox.contains(bb.x1,bb.y1)
        for i, bb in enumerate(boxes):
            assert all(not bb.overlaps(other) for other in boxes[i+1:])
        for artist in [ax.xaxis.label,ax.yaxis.label,ax.get_legend(),
                       *ax.get_xticklabels(),*ax.get_yticklabels()]:
            bb=artist.get_window_extent(renderer)
            assert fig.bbox.contains(bb.x0,bb.y0) and fig.bbox.contains(bb.x1,bb.y1)
        fig.set_layout_engine('none')
        for ext in ('png','pdf'):
            fig.savefig(args.out/f'zigzag_abc_footprints.{ext}',dpi=200)
        plt.close(fig)
    with Image.open(args.out/'zigzag_abc_footprints.png') as im:
        im.convert('L').save(args.out/'zigzag_abc_footprints_grayscale.png')
    with (args.out/'segments.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=segments[0].keys());writer.writeheader();writer.writerows(segments)
    source=Path(__file__).resolve()
    provenance=dict(scope='geometry only; no fits, mesh or final growth tensor computation',
        sequence_path=str(args.sequence.resolve()), sequence_sha256=sha(args.sequence),
        source_sha256=sha(source), helper_sha256=sha(source.with_name('visualize_solver_characteristics.py')),
        job_id=os.environ.get('SLURM_JOB_ID'),sheet_mm=[254,304.8],
        sheet_is_assumed=True, coordinates='unscaled material frame, equal aspect',
        styles=STYLES, paths=records, matplotlib_version=matplotlib.__version__,
        numpy_version=np.__version__,annotation_and_canvas_extent_checks='passed',
        review='pending; PDF rendering and CVD simulation not performed')
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.out/'zigzag_abc_footprints.caption.md').write_text(
        '# Proposed zigzag geometry\n\n'
        'Material-frame centerlines and full-width capsule footprints from the supplied JSON. '
        'The dashed boundary assumes the previous 254 × 304.8 mm sheet; dimensions were not specified in the JSON. '
        'Circles mark starts, triangles mark ends, and arrows indicate initial traversal. '
        'Repeated paths occupy identical geometry and are drawn once, not offset. Order: A, A, B, B, B, C, C, C. '
        'B and C shifts are in the material frame, so their 90° rotation does not rotate the ±120 mm translation. '
        'A inherits the center-peak profile (top growth 0.0006 at segment ends, 0.0015 at centers before hardening); '
        'B and C override it with uniform 0.0015. Shading encodes geometric width only, NOT growth magnitude or hit counts. '
        'The Voce history, orthotropy and repeat-dependent target tensors are not computed by this preview. '
        'No deformation prediction, mesh or equilibrium solve is included.\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    main()
