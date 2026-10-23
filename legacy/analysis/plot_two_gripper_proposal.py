#!/usr/bin/env python3
"""Geometry-only preview of the proposed (not historical) two-gripper setup.

No solver input is changed. Coordinates are material coordinates in millimetres.
Regenerate with the project DHD Python environment. Output is a PNG preview.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'legacy/analysis/plots/two_gripper_top_edge_layout.png'


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserving existing preview: {OUT}')
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'}):
        fig, ax = plt.subplots(figsize=(8, 10), layout='constrained')
        ax.add_patch(Rectangle((-100, -150), 200, 300,
                               facecolor='white', edgecolor='#333333', lw=1.7))
        ax.add_patch(Rectangle((-100, 65), 200, 85,
                               facecolor='#F3EDF5', edgecolor='none'))
        for x in (-50, 50):
            assert x - 25 >= -100 and x + 25 <= 100
            ax.plot([x, x], [-150, 150], color='#AAAAAA', ls=':', lw=0.8)
            ax.add_patch(Rectangle((x - 25, 130), 50, 20,
                                   facecolor='#CC79A7', edgecolor='#663355',
                                   hatch='///', lw=1.5))
            ax.plot(x, 140, marker='+', color='black', ms=10, mew=1.4)
            ax.text(x, 114, f'({x:+.0f}, 140)', ha='center', va='center')
            ax.plot([x, x], [119, 130], color='#663355', lw=1)
        ax.plot([-100, 100], [65, 65], color='#0072B2', ls='--', lw=1.5)
        ax.text(0, 53, 'Toolpath upper turnaround: y = 65',
                ha='center', va='center', color='#005A8C')
        ax.annotate('', xy=(86, 130), xytext=(86, 65),
                    arrowprops={'arrowstyle': '<->', 'color': '#333333'})
        ax.plot([75, 100], [130, 130], color='#555555', lw=0.7)
        ax.text(104, 97.5, '65 mm\ngap', ha='left', va='center')
        ax.text(0, 20, 'Two proposed gripper pads', ha='center', va='center')
        ax.text(0, 3, 'Each: 50 mm (x) × 20 mm (y)', ha='center', va='center')
        ax.text(0, -29, 'Quarter-width centers: x = ±50 mm', ha='center', va='center')
        ax.text(0, -62, 'Sheet: 200 × 300 mm', ha='center', va='center')
        ax.text(0, -100, 'Layout only — no simulation\nToolpath traces not shown',
                ha='center', va='center', color='#555555')
        ax.set(xlim=(-112, 156), ylim=(-163, 163), aspect='equal',
               xlabel='Material x (mm)', ylabel='Material y (mm)',
               title='Two-gripper layout — aligned with the top edge')
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_yticks([-150, -100, -50, 0, 65, 130, 150])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [t for t in ax.texts if t.get_text()]
        for i, t in enumerate(labels):
            box = t.get_window_extent(renderer)
            assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1), t.get_text()
            for other in labels[i + 1:]:
                overlap = Bbox.intersection(box, other.get_window_extent(renderer))
                assert overlap is None or overlap.width <= 1 or overlap.height <= 1, (t.get_text(), other.get_text())
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, dpi=180)
        plt.close(fig)
        print(OUT)


if __name__ == '__main__':
    main()
