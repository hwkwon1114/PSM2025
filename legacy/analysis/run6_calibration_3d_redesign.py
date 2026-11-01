"""Run 6 3D visual draft from saved calibration results; no new fit/alignment."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import load_evaluation_data, heights_at_xy
from calibration.ewcal.common import digest


def main():
    out = BASE / 'calibration/paper_figures/redesign_run6_3d'
    out.mkdir(exist_ok=True)
    stem = out / 'run6_shape_agreement'
    pins = {}
    row = load_evaluation_data('run6', BASE / 'run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00', pins)
    scan, q, faces = row['scan'], row['prediction'], row['faces']
    tri = mtri.Triangulation(scan[:, 0], scan[:, 1])
    # Do not bridge gaps beyond neighboring accepted scan cells.
    xyz = scan[tri.triangles, :2]
    edges = np.linalg.norm(xyz - np.roll(xyz, 1, axis=1), axis=2)
    tri.set_mask(edges.max(axis=1) > 5)
    assert np.isfinite(scan).all() and np.isfinite(q).all()
    orange, blue = '#D55E00', '#0072B2'
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(9, 3.65))
        axes = [fig.add_subplot(1, 3, i+1, projection='3d', computed_zorder=False) for i in range(3)]
        axes[0].plot_trisurf(tri, scan[:, 2], color=orange, linewidth=0, antialiased=False, shade=True, rasterized=True)
        axes[1].plot_trisurf(q[:, 0], q[:, 1], q[:, 2], triangles=faces, color=blue, linewidth=0, antialiased=False, shade=True, rasterized=True)
        # Scan points remain discrete in the overlay: no interpolated measured surface.
        axes[2].scatter(*scan.T, s=.45, color=orange, alpha=.24, linewidths=0, depthshade=False, rasterized=True, zorder=1)
        lines = []
        for x in np.linspace(-98, 98, 11):
            lines.append(np.column_stack([np.full(151, x), np.linspace(-148, 148, 151)]))
        for y in np.linspace(-148, 148, 13):
            lines.append(np.column_stack([np.linspace(-98, 98, 101), np.full(101, y)]))
        for xy in lines:
            z, support, _ = heights_at_xy(q, faces, xy)
            axes[2].plot(xy[:, 0], xy[:, 1], z, color=blue, lw=.65, zorder=2)
        for ax, title in zip(axes, ['(a)  Measurement', '(b)  Prediction', '(c)  Direct overlay']):
            ax.set_title(title, loc='left', pad=4)
            ax.view_init(elev=24, azim=-55)
            ax.set_proj_type('ortho')
            ax.set(xlim=(-105,105), ylim=(-155,155), zlim=(-5,3),
                   xticks=[-100,0,100], yticks=[-100,0,100], zticks=[-4,0,2])
            ax.set_box_aspect((200,300,96), zoom=.86)  # 12x vertical exaggeration.
            ax.set_xlabel('$x$ (mm)', labelpad=-3)
            ax.set_ylabel('$y$ (mm)', labelpad=-3)
            ax.set_zlabel('$z$ (mm)', labelpad=-5)
            ax.tick_params(pad=-2)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill=False
                axis.pane.set_edgecolor('white')
                axis.line.set_color('#888888')
            ax.grid(False)
        fig.legend(handles=[Line2D([],[],marker='o',ls='',ms=4,color=orange,label='Measurement'),
                            Line2D([],[],lw=1,color=blue,label='Prediction')],
                   loc='lower center', bbox_to_anchor=(.5,.06), ncol=2, frameon=False)
        fig.text(.5,.035,'Run 6 · calibration     |     Vertical exaggeration ×12 in all panels',ha='center',fontsize=9,color='#444444')
        fig.subplots_adjust(left=.02,right=.97,bottom=.19,top=.90,wspace=.10)
        for ext in ('pdf','png'):
            fig.savefig(stem.with_suffix('.'+ext),dpi=350)
        plt.close(fig)
    assert all(digest(BASE / p)==h for p,h in pins.items())
    stem.with_suffix('.provenance.json').write_text(json.dumps({
        'input_sha256':pins,'script_sha256':digest(Path(__file__)),
        'statistics':row['statistics'],'numpy':np.__version__,'matplotlib':matplotlib.__version__,
        'camera':{'elevation':24,'azimuth':-55,'projection':'orthographic','vertical_exaggeration':12},
        'scan_triangles_masked':int(tri.mask.sum()),'scan_max_triangle_edge_mm':5,
        'overlay':'All accepted scan points; model sampled on 11 constant-x and 13 constant-y lines independent of residuals. Lines drawn over translucent scan points for legibility, not physical occlusion.'},indent=2)+'\n')
    stem.with_suffix('.caption.md').write_text(
        '# Run 6: measured and predicted deformation\n\n'
        '(a) Measured shape, (b) Attempt-209 calibrated prediction, and (c) direct 3D overlay '
        'of all 14,504 accepted scan points (orange) and a prediction wire grid (blue). '
        'Identical orthographic cameras and coordinate limits; vertical geometry is exaggerated '
        '12× in every panel, while tick labels retain physical mm. This exaggeration also magnifies '
        'vertical discrepancies. No refitting, new solve, or alignment changes. One calibration specimen, '
        'not held-out validation. Color denotes identity, not height or error.\n\n'
        'The measured surface in (a) is a piecewise-planar XY triangulation of accepted scan points; '
        'triangles with any edge longer than 5 mm are masked to avoid bridging missing regions. '
        'The prediction in (b) uses native mesh faces. Lighting emphasizes shape without changing geometry. '
        'Overlay grid locations are uniformly spaced and independent of error; model heights are evaluated '
        'on native facets without extrapolation. Scan points are translucent and model lines are drawn '
        'on top for visibility; the overlay is not an opaque-surface occlusion rendering. '
        'All input points contribute; no smoothing or error-based selection.\n\n'
        'Recorded symmetric geometric RMS: 1.182 mm; signed-height RMS: 0.651 mm. '
        'The metrics have different definitions. This shape-view draft complements, rather than replaces, '
        'full-field error maps and error distributions.\n\n'
        'Regenerate: `python calibration/analysis/run6_calibration_3d_redesign.py`. '
        'Hashes and rendering parameters are in the adjacent provenance JSON.\n')
    print(stem)


if __name__ == '__main__':
    main()
