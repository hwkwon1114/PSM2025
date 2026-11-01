#!/usr/bin/env python3
"""Publication figures from frozen Attempt 209 and corrected forward results; no solver."""
import csv
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis import convention_corrected_figures as shapes
from calibration.analysis.plot_pairwise_shape_overlays import grid_lines
from calibration.ewcal.common import digest

OUT = ROOT / 'calibration/paper_figures/attempt209_publication'
HISTORY = ROOT / 'run/run6_expanded_bo_continue20_recovery/evaluations.jsonl'
SUMMARY = ROOT / 'run/run6_expanded_bo_continue20_recovery/bayes_summary.json'
CONFIG = ROOT / 'run/run6_expanded_bo_continue20_recovery/frozen_config.json'
RUNS = ('run6', 'run9', 'run10')
NAMES = ('Run 1', 'Run 2', 'Run 3')
PARAMS = ('gtop', 'gbot', 'ortho_top', 'ortho_bottom')
PLABELS = (r'$g_{\mathrm{top}}$', r'$g_{\mathrm{bottom}}$', r'$o_{\mathrm{top}}$', r'$o_{\mathrm{bottom}}$')
BLUE, ORANGE = '#0072B2', '#D55E00'
STYLE = {'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
         'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16,
         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
         'savefig.facecolor': 'white'}


def save(fig, name, caption):
    if isinstance(fig.get_layout_engine(), matplotlib.layout_engine.ConstrainedLayoutEngine):
        fig.get_layout_engine().set(w_pad=.16, h_pad=.12, wspace=.12, hspace=.12)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    for ax in fig.axes:
        texts.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                      ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()])
        axes = [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]
        if hasattr(ax, 'zaxis'):
            axes.append((ax.zaxis, ax.get_zlim()))
            texts.append(ax.zaxis.label)
        # Read rendered ticks directly: get_*ticklabels() resets 3-D projected positions.
        for axis, limits in axes:
            lo, hi = sorted(limits)
            for tick in axis.majorTicks:
                if lo <= tick.get_loc() <= hi:
                    texts.extend([tick.label1, tick.label2])
        if ax.get_legend():
            texts.extend(ax.get_legend().get_texts())
    for text in texts:
        if text.get_visible() and text.get_text():
            if plt.rcParams['font.size'] == 16 and not np.isclose(text.get_fontsize(), 16):
                raise ValueError(f'{name}: non-uniform font: {text.get_text()} = {text.get_fontsize()}')
            bounds = text.get_window_extent(renderer).extents
            if np.isfinite(bounds).all() and (bounds[0] < -1 or bounds[1] < -1 or
                                               bounds[2] > fig.bbox.width + 1 or bounds[3] > fig.bbox.height + 1):
                raise ValueError(f'{name}: text outside page: {text.get_text()}')
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300)
    plt.close(fig)
    (OUT / f'{name}.caption.md').write_text(caption + '\n')


def history_figure(records):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 2]})
    for ax, subset, heading in zip(axes, (records[:80], records[80:]),
                                   ('(a) Rescored prior', '(b) Search and continuation')):
        valid = [r for r in subset if r['valid']]
        bad = [r for r in subset if not r['valid']]
        ax.scatter([r['attempt_id'] for r in valid], [r['objective_mm'] for r in valid],
                   s=12, color='#777777', alpha=.75, label='Valid')
        ax.scatter([r['attempt_id'] for r in bad], [.03]*len(bad),
                   transform=ax.get_xaxis_transform(), marker='x', color=ORANGE,
                   s=24, clip_on=False, label='Invalid')
        ax.set(yscale='log', xlabel='Archived attempt ID', ylabel='Historical RMS (mm)', title=heading)
        ax.grid(axis='y', color='#e5e5e5', lw=.5)
    best = np.inf
    incumbent = []
    for r in records[80:]:
        if r['valid']:
            best = min(best, r['objective_mm'])
        incumbent.append(best)
    axes[1].step([r['attempt_id'] for r in records[80:]], incumbent, color=BLUE,
                 lw=1.5, where='post', label='Incumbent')
    sel = records[208]
    axes[1].scatter([209], [sel['objective_mm']], s=75, marker='*', color=BLUE, zorder=5,
                    label='Selected')
    axes[1].legend(frameon=False, loc='upper right')
    save(fig, 'optimization_history',
         '# Historical Run 1 optimization history\n\nThe first 80 records were retrospectively rescored prior observations, not a contemporaneous trajectory. Attempts 81–120 are refined-edge BO; attempts 121–240 are expanded-bounds BO and its continuation; the incumbent is the running minimum over valid records in that stage. Crosses denote invalid attempts, not objective values. Attempt IDs are archival order, not completion time. Attempt 209 is selected at 1.182314 mm under the historical material-reference metric. This search used earlier solver/convention settings and an older metric; its objective is not numerically comparable with the corrected-convention shared-footprint shape scores. The axes are logarithmic; this single adaptive search does not establish global convergence.')


def encode(parameters, bounds):
    out = []
    for i, name in enumerate(PARAMS):
        low, high = bounds[name]
        value = parameters[name]
        if i < 2:
            low, high, value = np.log(low), np.log(high), np.log(value)
        out.append((value-low)/(high-low))
    return np.array(out)


def sensitivity_figure(records, chosen, bounds):
    valid = [r for r in records if r['valid']]
    X = np.array([encode(r['parameters'], bounds) for r in valid])
    y = np.log([r['objective_mm'] for r in valid])
    if not (np.isfinite(X).all() and np.isfinite(y).all() and len(valid) == 221):
        raise ValueError('Unexpected valid historical observations')
    kernel = ConstantKernel(1, (0.1, 10)) * Matern(np.ones(4), length_scale_bounds=(.05, 20), nu=2.5)
    def fit(x, z):
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True,
                                         n_restarts_optimizer=1, random_state=2026)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, z)
        return model, [str(w.message) for w in caught]
    gp, warn = fit(X, y)
    earlier = np.array([r['attempt_id'] <= 200 for r in valid])
    check, check_warnings = fit(X[earlier], y[earlier])
    mu, sd = check.predict(X[~earlier], return_std=True)
    rmse = float(np.sqrt(np.mean((mu-y[~earlier])**2)))
    coverage = float(np.mean(np.abs(mu-y[~earlier]) <= 1.96*sd))
    x0 = encode(chosen, bounds)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout='constrained')
    curves = {}
    for i, (ax, name, label) in enumerate(zip(axes.flat, PARAMS, PLABELS)):
        u = np.linspace(0, 1, 200)
        xx = np.tile(x0, (len(u), 1)); xx[:, i] = u
        mu, sd = gp.predict(xx, return_std=True)
        low, high = bounds[name]
        physical = np.exp(np.log(low)+(np.log(high)-np.log(low))*u) if i < 2 else low+(high-low)*u
        ax.fill_between(physical, np.exp(mu-1.96*sd), np.exp(mu+1.96*sd),
                        color=BLUE, alpha=.17, linewidth=0)
        ax.plot(physical, np.exp(mu), color=BLUE, lw=1.5)
        ax.axvline(chosen[name], color=ORANGE, ls='--', lw=1.2)
        ax.plot([r['parameters'][name] for r in valid], np.full(len(valid), .025), '|',
                color='#777777', alpha=.35, markersize=4, transform=ax.get_xaxis_transform())
        if i < 2: ax.set_xscale('log')
        ax.set_yscale('log'); ax.set_xlabel(label); ax.set_ylabel('Historical RMS (mm)')
        ax.set_title(f'({chr(97+i)}) {label}', loc='left')
        ax.grid(axis='y', color='#e5e5e5', lw=.5)
        curves[name] = {'physical': physical.tolist(), 'median': np.exp(mu).tolist(),
                        'lower': np.exp(mu-1.96*sd).tolist(), 'upper': np.exp(mu+1.96*sd).tolist()}
    axes[0,0].legend(handles=[Line2D([], [], color=BLUE, label='GP median'),
                              Line2D([], [], color=ORANGE, ls='--', label='Selected')],
                      frameon=False)
    save(fig, 'historical_gp_sensitivity',
         '# Conditional sensitivity of the historical Run 1 objective\n\nOne-dimensional slices of a post-hoc Matérn-5/2 Gaussian process fitted to 221 valid archived observations; other parameters are fixed at Attempt 209. Curves are the back-transformed log-objective predictive median; shading is the pointwise 95% latent-GP interval conditional on fitted kernel hyperparameters, not a parameter confidence interval. Rugs show locations of all valid observations projected onto each parameter axis, not observations made on the displayed slice. The two growth coordinates are logarithmic; orthotropy coordinates are linear. The fit combines adaptively sampled records, including 80 rescored prior observations and later records from changing search stages. It models the historical material-reference objective and earlier numerical convention, not the corrected-convention shared-footprint scores used in the shape figures. A chronological check trained on valid IDs 1–200 and predicted valid IDs 201–240; these are adaptive, not independent validation. Smoothness and bands do not establish causal sensitivity, uniqueness, or physical identifiability.')
    return {'valid': len(valid), 'kernel': str(gp.kernel_), 'warnings': warn,
            'check_training': int(earlier.sum()), 'check_later': int((~earlier).sum()),
            'check_log_rmse': rmse, 'check_pointwise_coverage': coverage,
            'check_warnings': check_warnings, 'curves': curves}


def toolpaths(rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.5), sharex=True, sharey=True, layout='constrained')
    for i, (ax, row, label) in enumerate(zip(axes, rows, NAMES)):
        # CSV x is mirrored relative to the saved sequence's negative-angle
        # builder; verify the plotted start against buildZigZagStrips.
        path = row['path'].copy(); path[:, 0] *= -1
        seq = json.loads((shapes.case_dir(row['name']) / 'sequence.json').read_text())
        op = seq['toolpaths'][0]['operation']
        assert op.get('rotation_deg', 0) == 0 and not op.get('reverse_traversal', False)
        expected_start = -.5 * (op['n_strips'] - 2) * op['lv_mm'] * np.tan(np.deg2rad(op['alpha_deg']))
        assert np.isclose(path[0, 0], expected_start + op['shift_uv_mm'][0], atol=1e-3)
        assert np.isclose(path[0, 1], -op['lv_mm']/2 + op['shift_uv_mm'][1], atol=1e-3)
        st = row['statistics']
        ax.fill([-100, 100, 100, -100], [-85, -85, 215, 215], color='#f2f2f2', zorder=0)
        ax.plot(path[:,0], path[:,1]+65, color=BLUE, lw=.7)
        ax.scatter([path[0,0]], [path[0,1]+65], s=24, color=ORANGE, zorder=3)
        ax.set(xlim=(-105,105), ylim=(-90,220), xticks=[-100,0,100], yticks=[-85,65,215],
               xlabel='Width $x$ (mm)', title=f'({chr(97+i)}) {label}')
        ax.set_aspect('equal')
        if i == 0: ax.set_ylabel('Length $z$ (mm)')
        ax.text(.5, -.2, f"{st['repeat']} cycles", transform=ax.transAxes, ha='center', va='top')
    save(fig, 'toolpaths',
         '# Corrected toolpaths\n\nNominal centerlines from the corrected-convention sequences for Runs 1, 2 and 3. Orange circles mark the start. The grey rectangle is the undeformed sheet, not a measured gripper. Each centerline is shown once; cycle counts are repetitions. Line width is graphical, not the effective physical track width. Public coordinates are width x, length z and height y; z = internal v + 65 mm. The source CSV x coordinates are mirrored to match the saved sequence’s negative-angle model builder, and the first point is checked against that builder. This is the nominal model toolpath, not independent confirmation of the physical travel direction. No solve or registration was performed for this figure.')


def residuals(rows):
    lim = np.ceil(max(np.max(np.abs(r['dz'][r['support']])) for r in rows)*2)/2
    fig = plt.figure(figsize=(14, 12), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[2.3,1.2])
    maps = []
    for i, (r, label) in enumerate(zip(rows, NAMES)):
        ax = fig.add_subplot(grid[0,i]); maps.append(ax)
        p, m = r['scan'], r['support']
        im = ax.scatter(p[m,0], p[m,1]+65, c=r['dz'][m], s=2, marker='s', linewidths=0,
                        cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
        if (~m).any(): ax.scatter(p[~m,0], p[~m,1]+65, c='black', marker='x', s=3, linewidths=.4)
        ax.set(xlim=(-103,103), ylim=(-88,218), xticks=[-100,0,100], yticks=[-85,65,215],
               xlabel='Width $x$ (mm)', title=f'({chr(97+i)}) {label}')
        ax.set_aspect('equal'); ax.set_facecolor('#eeeeee')
        if i == 0: ax.set_ylabel('Length $z$ (mm)')
        else: ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=maps, label=r'Height difference $\Delta y$ (mm)', shrink=.8)
    ax=fig.add_subplot(grid[1,:])
    for row,label,c,ls in zip(rows,NAMES,[BLUE,ORANGE,'#333333'],['-','--',':']):
        d=np.sort(np.abs(row['dz'][row['support']]))
        ax.step(d,np.arange(1,len(d)+1)/len(d),where='post',label=label,color=c,ls=ls,lw=1.4)
    ax.set(xlabel='Absolute height difference (mm)',ylabel='Fraction of scan points',ylim=(0,1.01),title='(d) Supported-point distributions')
    ax.legend(frameon=False,ncol=3,loc='lower right');ax.grid(color='#e5e5e5',lw=.5)
    save(fig, 'height_residuals',
         '# Corrected-convention height differences and distributions\n\nRuns 1–3 use the same frozen Attempt 209 parameters. The signed height difference is simulation minus experiment at scan x–z coordinates using native-facet interpolation, not the symmetric geometric optimization objective. All accepted scan points supported by the model are shown; unsupported points, if present, are black crosses. Grey denotes background. Shared color limits include all supported values without clipping. The distributions weight each supported scan point equally; spatially dependent points from one specimen per run are not independent replicates. Runs 2 and 3 were previously inspected and are exploratory transfers, not blind validation.')


def sections_figure(rows):
    x=np.linspace(-100,100,401)
    results=[shapes.sections(r,x) for r in rows]
    values=np.concatenate([np.concatenate((e[np.isfinite(e)],m[np.isfinite(m)])) for row in results for e,m in row])
    ylim=(np.floor(values.min()),np.ceil(values.max()))
    fig,axs=plt.subplots(3,3,figsize=(14,13),sharex=True,sharey=True,layout='constrained')
    for i,(row,label) in enumerate(zip(results,NAMES)):
        for j,(e,m) in enumerate(row):
            ax=axs[i,j];ax.plot(x,e,color=ORANGE,lw=1.3,marker='o',markevery=40,ms=2.5)
            ax.plot(x,m,color=BLUE,lw=1.3,ls='--');ax.set(xlim=(-100,100),ylim=ylim,xticks=[-100,0,100])
            ax.grid(axis='y',color='#e5e5e5',lw=.5)
            if i==0: ax.set_title(f'$z={shapes.SECTIONS_Z[j]}$ mm')
            if j==0: ax.set_ylabel(f'{label}\nHeight $y$ (mm)')
            if i==2: ax.set_xlabel('Width $x$ (mm)')
    fig.legend(handles=[Line2D([],[],color=ORANGE,marker='o',ms=3,label='Experiment'),
                        Line2D([],[],color=BLUE,ls='--',label='Simulation')],
               loc='outside upper center',ncol=2,frameon=False)
    save(fig,'fixed_sections',
         '# Fixed-location sections\n\nRows are Runs 1–3; columns are z = −35, 65 and 165 mm, fixed without selecting by error. Experiment and simulation are interpolated separately in the fixed saved frame; scan triangles with edges longer than 5 mm are excluded, and unsupported portions remain gaps. Sampling is every 0.5 mm; plotted markers are interpolated values, not raw observations. All panels use the same limits. Runs 2 and 3 are previously seen exploratory transfers. No new solve, registration or smoothing was performed.')


def overlay_figures(rows):
    rng=np.random.default_rng(20261029)
    indices=[np.sort(rng.choice(len(r['scan']),size=min(2500,len(r['scan'])),replace=False)) for r in rows]
    wires=[grid_lines(r['prediction'],r['faces']) for r in rows]
    heights=np.concatenate([a[:,2] for r in rows for a in (r['scan'],r['prediction'])])
    ylim=(float(np.floor(heights.min())),float(np.ceil(heights.max())))
    fig,axes=plt.subplots(3,1,figsize=(7.2,7.2),sharex=True,sharey=True,layout='constrained')
    for i,(ax,r,ind,lines,label) in enumerate(zip(axes,rows,indices,wires,NAMES)):
        p=r['scan'][ind];ax.scatter(p[:,0],p[:,2],s=2,color=ORANGE,alpha=.35,rasterized=True)
        for line in lines: ax.plot(line[:,0],line[:,2],color=BLUE,lw=.65,alpha=.8)
        ax.set(xlim=(-105,105),ylim=ylim,xticks=[-100,0,100],ylabel='Height $y$ (mm)',title=f'({chr(97+i)}) {label}')
        ax.grid(color='#e7e7e7',lw=.5)
    axes[-1].set_xlabel('Width $x$ (mm)')
    fig.legend(handles=[Line2D([],[],color=ORANGE,marker='o',ls='',ms=4,label='Experiment (sampled)'),
                        Line2D([],[],color=BLUE,lw=1,label='Simulation grid')],loc='outside upper center',ncol=2,frameon=False)
    save(fig,'shape_overlays_xy',
         '# Width–height projections\n\nRun 1 calibration and Runs 2–3 exploratory transfers using frozen Attempt 209 parameters in the corrected convention. Points are 2,500 scan points per case sampled uniformly without replacement using seed 20261029. Lines are model grids interpolated on saved native facets and masked outside the mesh. The x–y view projects all lengths z together; it is not a fixed-z section, and overlapping points need not be corresponding material positions. Axes have unequal display scales. No new solve or registration.')
    fig=plt.figure(figsize=(7.2,9.5),layout='constrained')
    axs=fig.subplots(3,1,subplot_kw={'projection':'3d'})
    for i,(ax,r,ind,lines,label) in enumerate(zip(axs,rows,indices,wires,NAMES)):
        p=r['scan'][ind];ax.scatter(p[:,0],p[:,1]+65,p[:,2],s=1.5,color=ORANGE,alpha=.55,depthshade=False,rasterized=True)
        for line in lines: ax.plot(line[:,0],line[:,1]+65,line[:,2],color=BLUE,lw=.7)
        ax.set(xlim=(-105,105),ylim=(-90,220),zlim=ylim,xticks=[-100,0,100],yticks=[-85,65,215],
               xlabel='Width $x$ (mm)',ylabel='Length $z$ (mm)',zlabel='Height $y$ (mm)',title=f'({chr(97+i)}) {label}')
        ax.view_init(elev=23,azim=-65);ax.set_box_aspect((210,310,(ylim[1]-ylim[0])*8),zoom=1.05)
        ax.grid(False)
    fig.legend(handles=[Line2D([],[],color=ORANGE,marker='o',ls='',ms=4,label='Experiment (sampled)'),
                        Line2D([],[],color=BLUE,lw=1,label='Simulation grid')],loc='outside upper center',ncol=2,frameon=False)
    save(fig,'shape_overlays_3d',
         '# Three-dimensional shape overlays\n\nCorrected-convention saved surfaces using frozen Attempt 209 parameters. Run 1 was used for calibration; Runs 2 and 3 are previously seen exploratory transfers. Orange points are the same fixed-seed sample as in the x–y view; blue lines are piecewise-linear grid lines on the saved simulation mesh. The height axis is displayed 8× relative to the width and length axes; numerical ticks remain in mm. All panels use the same camera and axis limits. No new solve or alignment.')
    return {'sample_indices': {k:v.tolist() for k,v in zip(RUNS,indices)}, 'height_limits_mm':ylim}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    OUT = parser.parse_args().out_dir
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing outputs: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    pins={}; rows=[shapes.load(run,pins) for run in RUNS]
    for p in (HISTORY,SUMMARY,CONFIG): pins[str(p.relative_to(ROOT))]=digest(p)
    summary=json.loads(SUMMARY.read_text());config=json.loads(CONFIG.read_text());chosen=summary['best']['parameters']
    bounds={k:config['parameters'][k]['bounds'] for k in PARAMS}
    records=sorted((json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()),key=lambda r:r['attempt_id'])
    assert [r['attempt_id'] for r in records] == list(range(1,241))
    assert all(r['frozen']==chosen and r['forward_metrics']['scoring_method']=='material_reference' for r in rows)
    for run in RUNS:
        seq = json.loads((shapes.case_dir(run) / 'sequence.json').read_text())
        assert seq['hardening']['beta'] == 0.2231435513142097 and seq['hardening']['floor'] == 0
    assert records[208]['valid'] and records[208]['parameters']==chosen
    assert records[208]['objective_mm']==min(r['objective_mm'] for r in records if r['valid'])
    assert all((r.get('metrics') or r['cases'][0]['metrics'])['scoring_method']=='material_reference' for r in records if r['valid'])
    with plt.rc_context(STYLE):
        history_figure(records);gp=sensitivity_figure(records,chosen,bounds)
        toolpaths(rows);residuals(rows);sections_figure(rows);overlay=overlay_figures(rows)
    assert all(digest(ROOT/p)==h for p,h in pins.items())
    output={'input_sha256':pins,'script_sha256':digest(Path(__file__)),'frozen_parameters':chosen,
            'hardening':{'beta':0.2231435513142097,'floor':0.0},
            'historical_objective_mm':records[208]['objective_mm'],
            'corrected_symmetric_rms_mm':{name:row['forward_metrics']['rms_symmetric_mm'] for name,row in zip(NAMES,rows)},
            'gp':gp,'overlay':overlay,'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    (OUT/'provenance.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    (OUT/'README.md').write_text('# Attempt 209 publication figures\n\nSeven reproducible PDF and PNG figures with accompanying captions. Shape figures use corrected-convention forward evaluations at the frozen historical Attempt 209 parameters. Optimization history and post-hoc sensitivity use the older historical objective and are not on the same scale as the corrected shape scores. Runs 2 and 3 are exploratory transfers, not unseen validation. The GP is a conditional retrospective diagnostic, not a physical sensitivity or parameter confidence interval. Figures do not establish global optimality. No solver or search was run. See `provenance.json` and `calibration/analysis/attempt209_publication.py` for data identities and settings.\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('gp','overlay')},indent=2))
    print(json.dumps({k:v for k,v in gp.items() if k!='curves'},indent=2))

if __name__ == '__main__': main()
