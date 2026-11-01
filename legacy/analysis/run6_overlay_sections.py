"""Shaded, vertically exaggerated Run 6 overlay and fixed-y sections."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource, to_rgb
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.analysis.run6_run9_six_panel import load_evaluation_data, heights_at_xy
from calibration.ewcal.common import digest


def main():
    out = BASE / 'calibration/paper_figures/redesign_run6_sections'
    out.mkdir(exist_ok=True)
    stem = out / 'run6_overlay_sections'
    pins = {}
    row = load_evaluation_data('run6', BASE / 'run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00', pins)
    scan, q, faces = row['scan'], row['prediction'], row['faces']
    tri = mtri.Triangulation(scan[:, 0], scan[:, 1])
    xyz = scan[tri.triangles, :2]
    tri.set_mask(np.linalg.norm(xyz-np.roll(xyz,1,axis=1),axis=2).max(axis=1)>5)
    measured = mtri.LinearTriInterpolator(tri, scan[:, 2])
    orange, blue = '#D55E00', '#0072B2'
    ys = [-100, 0, 100]
    x = np.linspace(-100, 100, 401)
    sections = []
    for y in ys:
        xy = np.column_stack([x, np.full(len(x),y)])
        pred, support, _ = heights_at_xy(q,faces,xy)
        obs = measured(x,xy[:,1]).filled(np.nan)
        assert np.isfinite(obs).any() and support.any()
        sections.append((obs,pred))
    assert np.isfinite(scan).all() and np.isfinite(q).all()
    with plt.rc_context({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig = plt.figure(figsize=(7.2,6.3))
        gs = fig.add_gridspec(3,3,left=.10,right=.97,bottom=.09,top=.94,
                              height_ratios=[2.8,1.35,.7],hspace=.56,wspace=.32)
        ax = fig.add_subplot(gs[0,:],projection='3d',computed_zorder=False)
        # Lighting normals match the displayed 12x geometry, not a fitted surface.
        displayed = scan[tri.get_masked_triangles()].copy()
        displayed[:, :, 2] *= 12
        normals = np.cross(displayed[:, 1]-displayed[:, 0], displayed[:, 2]-displayed[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        intensity = LightSource(azdeg=315, altdeg=35).shade_normals(normals)
        colors = np.clip(np.array(to_rgb('#E6AA70'))[None, :] * (.45 + .55*intensity[:, None]), 0, 1)
        surface = ax.plot_trisurf(tri, scan[:, 2], shade=False, linewidth=0,
                                 antialiased=False, rasterized=True, zorder=1)
        surface.set_facecolors(colors)
        lines = [np.column_stack([np.full(151,u),np.linspace(-148,148,151)]) for u in np.linspace(-98,98,11)]
        lines += [np.column_stack([np.linspace(-98,98,101),np.full(101,v)]) for v in np.linspace(-148,148,13)]
        for xy in lines:
            z,_,_=heights_at_xy(q,faces,xy)
            ax.plot(xy[:,0],xy[:,1],z,color=blue,lw=.55,zorder=2)
        for y,(obs,pred) in zip(ys,sections):
            ax.plot(x,np.full(len(x),y),pred,color='#222222',lw=.9,ls='--',zorder=3)
        ax.set(xlim=(-105,105),ylim=(-155,155),zlim=(-5,3),xticks=[-100,0,100],yticks=[-100,0,100],zticks=[-4,2])
        # XY remains physical; only vertical display scale is enlarged.
        ax.set_box_aspect((210,310,8*12),zoom=1.0)
        ax.set_proj_type('ortho')
        ax.view_init(elev=0,azim=-90)
        ax.set_xlabel('$x$ (mm)',labelpad=10)
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_zlabel('$z$ (mm)',labelpad=1)
        ax.tick_params(pad=-1)
        ax.grid(False)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False
            axis.pane.set_edgecolor('white')
        fig.text(.10, .955, '(a)  x–z view · vertical scale ×12', fontsize=10, va='top')
        fig.legend(handles=[Line2D([],[],color=orange,ls='-',marker='o',ms=3,lw=1,label='Experiment'),
                            Line2D([],[],color=blue,ls='--',lw=1.3,label='Analytical')],
                   loc='upper right',bbox_to_anchor=(.97,.98),ncol=2,frameon=False)
        for i,(y,(obs,pred)) in enumerate(zip(ys,sections)):
            a = fig.add_subplot(gs[1,i])
            a.plot(x,obs,color=orange,lw=1.2,marker='o',markevery=32,ms=2)
            a.plot(x,pred,color=blue,lw=1.3,ls='--')
            a.set(xlim=(-100,100),ylim=(-5,3),xticks=[-100,0,100],yticks=[-4,-2,0,2])
            a.set_title(f'({chr(98+i)})  $y = {y}$ mm',loc='left')
            a.grid(axis='y',color='#e8e8e8',lw=.5)
            if i==0:
                a.set_ylabel('$z$ (mm)')
            else:
                a.tick_params(labelleft=False)
            a.tick_params(labelbottom=False)
            e = fig.add_subplot(gs[2,i])
            e.axhline(0,color='#888888',lw=.65)
            e.plot(x,pred-obs,color='#333333',lw=1)
            e.set(xlim=(-100,100),ylim=(-2.5,2.5),xticks=[-100,0,100],yticks=[-2,0,2],xlabel='$x$ (mm)')
            if i==0:
                e.set_ylabel('$\\Delta z$ (mm)')
            else:
                e.tick_params(labelleft=False)
        for ext in ('pdf','png'):
            fig.savefig(stem.with_suffix('.'+ext),dpi=350)
        plt.close(fig)
    assert all(digest(BASE/p)==h for p,h in pins.items())
    arrays={'x_mm':x,'y_mm':np.array(ys)}
    for i,(obs,pred) in enumerate(sections):
        arrays[f'measured_z_{i}']=obs
        arrays[f'predicted_z_{i}']=pred
    np.savez(stem.with_suffix('.npz'),**arrays)
    stem.with_suffix('.provenance.json').write_text(json.dumps({
        'input_sha256':pins,'script_sha256':digest(Path(__file__)),
        'statistics':row['statistics'],'numpy':np.__version__,'matplotlib':matplotlib.__version__,
        'camera':{'elevation':0,'azimuth':-90,'projection':'orthographic','vertical_exaggeration':12},
        'section_y_mm':ys,'section_x_spacing_mm':.5,'measured_max_triangle_edge_mm':5,
        'section_measured_finite_counts':[int(np.isfinite(s[0]).sum()) for s in sections],
        'section_joint_finite_counts':[int((np.isfinite(s[0]) & np.isfinite(s[1])).sum()) for s in sections]},indent=2)+'\n')
    stem.with_suffix('.caption.md').write_text(
        '# Run 6 calibration: shaded overlay and cross-sections\n\n'
        '(a) Experiment: a shaded piecewise-planar triangulation of all 14,504 accepted scan points (ochre); Analytical: the saved numerical shell-model wire grid (blue). '
        'Analytical is the requested display label, not a claim of a closed-form solution. '
        'Vertical geometry is exaggerated 12× for both datasets; XY scale and all tick values remain physical mm. '
        'This also magnifies vertical errors. Orthographic view from −y toward the origin (x–z projection); '
        'all y positions overlap in this view, which is not a y=constant section. Directional illumination uses normals of '
        'the displayed exaggerated triangles; brightness is not an error metric. No cast shadows. '
        'Black dashed lines locate the three fixed-y sections. '
        '(b–d) Sections at y = −100, 0, +100 mm selected independently of error. Orange solid lines '
        'with markers denote Experiment (interpolated scan heights); blue dashed lines denote Analytical (interpolated model heights). '
        'The aligned lower plots show Δz = prediction minus measurement. Section plots use distinct '
        'horizontal and vertical scales with shared limits across columns; they are not true-aspect geometry.\n\n'
        'Sections are sampled every 0.5 mm. Measured heights are linearly interpolated on XY scan '
        'triangles, excluding triangles with any edge >5 mm. Markers mark interpolated values, '
        'not raw observations. Model heights use native facets. Unsupported points remain gaps; '
        'there is no extrapolation, smoothing fit, new solve, refit or registration. Overlay model '
        'lines are drawn over the shaded experimental surface for legibility, not physical occlusion: '
        'a visible wire does not imply the model is above the experiment. '
        'Existing preprocessing and Attempt-209 material-frame alignment are unchanged. '
        'One calibration specimen; not held-out validation. Recorded symmetric geometric RMS '
        'is 1.182 mm; full-field signed-height RMS is 0.651 mm, a distinct metric.\n\n'
        'Regenerate: `python calibration/analysis/run6_overlay_sections.py`. '
        'Source hashes, exact section arrays and provenance accompany this draft.\n')
    print(stem)


if __name__ == '__main__':
    main()
