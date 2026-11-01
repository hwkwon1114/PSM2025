"""Run 6/9/10 residual maps and fixed-y sections from the convention-corrected forward evaluations.

Reads saved outputs only (no solve, refit or registration). Public convention: x width, z length, y height (internal v maps to z = v + 65 mm; internal h is height y).
Experiment = centred scan in comparison.npz;
Analytical = framed prediction mesh prediction_mm.vtp, i.e. the objects scored by the material-frame metric.
"""
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
from calibration.analysis.run6_run9_six_panel import load_evaluation_data, heights_at_xy, pin
from calibration.ewcal.common import digest

RUNS = [('run6', 'Run 6'), ('run9', 'Run 9'), ('run10', 'Run 10')]
ORANGE, BLUE = '#D55E00', '#0072B2'
SECTIONS_Z = [-35, 65, 165]  # public z (mm); internal v = -100, 0, 100
Z_OFFSET = 65.0
MAX_EDGE_MM = 5.0
STYLE = {'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'legend.fontsize': 7, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
OUT = BASE / 'calibration/paper_figures/convention_corrected'


def case_dir(run):
    return BASE / f'run/convention_corrected/{run}/evaluations/attempt_0001/case_00'


def load(run, pins):
    row = load_evaluation_data(run, case_dir(run), pins)
    forward = pin(BASE / f'run/convention_corrected/{run}/forward_result.json', pins)
    result = json.loads(forward.read_text())
    assert result['valid'] and len(result['cases']) == 1
    with np.load(case_dir(run) / 'comparison.npz', allow_pickle=False) as f:
        row['distance'] = f['distance_mm'].copy()
    row['forward_metrics'] = result['cases'][0]['metrics']
    row['frozen'] = result['parameters']
    scan = row['scan']
    tri = mtri.Triangulation(scan[:, 0], scan[:, 1] + Z_OFFSET)
    corners = scan[tri.triangles, :2]
    long_edge = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1) > MAX_EDGE_MM
    tri.set_mask(long_edge | ~row['support'][tri.triangles].all(axis=1))
    row['tri'] = tri
    return row


def sections(row, x):
    # Same long-edge exclusion for the experiment as the maps (unsupported-model triangles only affect the maps).
    corners = row['scan'][row['tri'].triangles, :2]
    keep = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1) <= MAX_EDGE_MM
    exp_tri = mtri.Triangulation(row['scan'][:, 0], row['scan'][:, 1], row['tri'].triangles[keep])
    interp = mtri.LinearTriInterpolator(exp_tri, row['scan'][:, 2])
    out = []
    for zpub in SECTIONS_Z:
        xy = np.column_stack([x, np.full(len(x), zpub - Z_OFFSET)])  # internal v
        model, support, _ = heights_at_xy(row['prediction'], row['faces'], xy)
        exp = interp(x, xy[:, 1]).filled(np.nan)
        assert np.isfinite(exp).sum() > 100 and support.sum() > 100
        out.append((exp, model))
    return out


def map_axes(ax, first_col, last_row):
    ax.set_aspect('equal')
    ax.set(xlim=(-102, 102), ylim=(-87, 217), xticks=[-100, 0, 100], yticks=[-85, 65, 215])
    if not first_col:
        ax.tick_params(labelleft=False)
    if not last_row:
        ax.tick_params(labelbottom=False)


def residual_maps(rows, stem):
    zlo = min(min(r['scan'][:, 2].min(), np.nanmin(r['scan'][:, 2] + r['dz'])) for r in rows)
    zhi = max(max(r['scan'][:, 2].max(), np.nanmax(r['scan'][:, 2] + r['dz'])) for r in rows)
    zlim = (np.floor(zlo * 2) / 2, np.ceil(zhi * 2) / 2)
    dlim = np.ceil(max(np.nanpercentile(np.abs(r['dz']), 99.5) for r in rows) * 2) / 2
    dmax = np.ceil(max(r['distance'].max() for r in rows) * 4) / 4
    bins = np.linspace(0, dmax, 26)
    # Use one shared histogram limit without clipping a highly concentrated run.
    max_fraction = max(np.histogram(r['distance'], bins=bins)[0].max() / len(r['distance']) for r in rows)
    fraction_limit = np.ceil(max_fraction * 1.1 * 20) / 20
    clipped = {}
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 8.1))
        gs = fig.add_gridspec(5, 5, left=.115, right=.985, bottom=.055, top=.955, hspace=.12, wspace=.24,
                              height_ratios=[1, 1, 1, .2, .05], width_ratios=[1, 1, 1, .12, 1.15])
        for i, (row, (run, label)) in enumerate(zip(rows, RUNS)):
            last = i == 2
            scan, dz, tri = row['scan'], row['dz'], row['tri']
            analytical = scan[:, 2] + dz
            axes = [fig.add_subplot(gs[i, j]) for j in (0, 1, 2, 4)]
            for ax, field, kw in ((axes[0], scan[:, 2], {'cmap': 'viridis', 'vmin': zlim[0], 'vmax': zlim[1]}),
                                  (axes[1], analytical, {'cmap': 'viridis', 'vmin': zlim[0], 'vmax': zlim[1]}),
                                  (axes[2], dz, {'cmap': 'RdBu_r', 'vmin': -dlim, 'vmax': dlim})):
                im = ax.tripcolor(tri, np.nan_to_num(field), shading='gouraud', rasterized=True, **kw)
                if ax is axes[0] and i == 0:
                    im_z = im
                if ax is axes[2] and i == 0:
                    im_d = im
            for j, ax in enumerate(axes[:3]):
                map_axes(ax, j == 0, last)
            axes[0].set_ylabel(f'{label}\n$z$ (mm)')
            if last:
                for ax in axes[:3]:
                    ax.set_xlabel('$x$ (mm)')
            clipped[run] = int((np.abs(dz[row['support']]) > dlim).sum())
            d = row['distance']
            h = axes[3]
            h.hist(d, bins=bins, weights=np.full(len(d), 1 / len(d)), color='#9aa5ad', edgecolor='white', lw=.3)
            rms = float(np.sqrt(np.mean(d ** 2)))
            p95 = float(np.quantile(d, .95))
            h.axvline(rms, ymax=.7, color=BLUE, lw=1.1)
            h.axvline(p95, ymax=.7, color=ORANGE, lw=1.1, ls='--')
            h.legend(handles=[Line2D([], [], color=BLUE, lw=1.1, label=f'RMS {rms:.2f} mm'),
                              Line2D([], [], color=ORANGE, lw=1.1, ls='--', label=f'95th pct. {p95:.2f} mm')],
                     frameon=False, loc='upper right', handlelength=1.6)
            h.set(xlim=(0, dmax), ylim=(0, fraction_limit))
            h.set_ylabel('Fraction of scan points')
            if last:
                h.set_xlabel('Distance (mm)')
            else:
                h.tick_params(labelbottom=False)
            if i == 0:
                for ax, t in zip(axes, ['(a) Experiment', '(b) Analytical', '(c) Height difference', '(d) Distance']):
                    ax.set_title(t, loc='left')
        cz = fig.colorbar(im_z, cax=fig.add_subplot(gs[4, 0:2]), orientation='horizontal')
        cz.set_label('Height $y$ (mm)')
        cd = fig.colorbar(im_d, cax=fig.add_subplot(gs[4, 2]), orientation='horizontal', ticks=[-dlim, 0, dlim])
        cd.set_label('$\\Delta y$ (mm)')
        for ext in ('pdf', 'png'):
            fig.savefig(stem.with_suffix('.' + ext), dpi=350)
        plt.close(fig)
    return {'height_limits_mm': list(map(float, zlim)), 'delta_z_limit_mm': float(dlim),
            'points_beyond_delta_z_limit': clipped, 'distance_axis_max_mm': float(dmax),
            'histogram_fraction_axis_max': float(fraction_limit),
            'max_distance_mm': {r['name']: float(r['distance'].max()) for r in rows}}


def section_figure(rows, stem, x):
    data = [sections(r, x) for r in rows]
    allz = np.concatenate([np.concatenate([e[np.isfinite(e)], m[np.isfinite(m)]]) for d in data for e, m in d])
    zlim = (np.floor(allz.min()), np.ceil(allz.max()))
    alld = np.concatenate([(m - e)[np.isfinite(m - e)] for d in data for e, m in d])
    dlim = np.ceil(np.abs(alld).max() * 2) / 2
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 7.4))
        gs = fig.add_gridspec(6, 3, left=.105, right=.985, bottom=.07, top=.925, hspace=.2, wspace=.22,
                              height_ratios=[2.3, 1, 2.3, 1, 2.3, 1])
        for i, (row, (run, label)) in enumerate(zip(rows, RUNS)):
            for j, zpub in enumerate(SECTIONS_Z):
                exp, model = data[i][j]
                a = fig.add_subplot(gs[2 * i, j])
                a.plot(x, exp, color=ORANGE, lw=1.1, marker='o', markevery=32, ms=2)
                a.plot(x, model, color=BLUE, lw=1.2, ls='--')
                a.set(xlim=(-100, 100), ylim=zlim, xticks=[-100, 0, 100])
                a.grid(axis='y', color='#e8e8e8', lw=.5)
                a.tick_params(labelbottom=False)
                e = fig.add_subplot(gs[2 * i + 1, j], sharex=a)
                e.axhline(0, color='#888888', lw=.6)
                e.plot(x, model - exp, color='#333333', lw=.9)
                e.set(xlim=(-100, 100), ylim=(-dlim, dlim), yticks=[-dlim, 0, dlim], xticks=[-100, 0, 100])
                if j == 0:
                    a.set_ylabel(f'{label}\n$y$ (mm)')
                    e.set_ylabel('$\\Delta y$ (mm)')
                else:
                    a.tick_params(labelleft=False)
                    e.tick_params(labelleft=False)
                if i == 0:
                    a.set_title(f'({chr(97 + j)}) $z = {zpub}$ mm', loc='left')
                if i == 2:
                    e.set_xlabel('$x$ (mm)')
                else:
                    e.tick_params(labelbottom=False)
        fig.legend(handles=[Line2D([], [], color=ORANGE, marker='o', ms=3, lw=1.1, label='Experiment'),
                            Line2D([], [], color=BLUE, ls='--', lw=1.2, label='Analytical')],
                   loc='upper right', bbox_to_anchor=(.985, .995), ncol=2, frameon=False)
        for ext in ('pdf', 'png'):
            fig.savefig(stem.with_suffix('.' + ext), dpi=350)
        plt.close(fig)
    return data, {'z_limits_mm': list(map(float, zlim)), 'delta_z_limit_mm': float(dlim)}


def main():
    OUT.mkdir(exist_ok=True)
    pins = {}
    rows = [load(run, pins) for run, _ in RUNS]
    for r in rows:
        assert np.isfinite(r['scan']).all() and np.isfinite(r['prediction']).all()
        assert r['frozen'] == rows[0]['frozen']
    maps = residual_maps(rows, OUT / 'calibration_transfer_residual_maps')
    x = np.linspace(-100, 100, 401)
    data, sect = section_figure(rows, OUT / 'calibration_transfer_sections', x)
    assert all(digest(BASE / p) == h for p, h in pins.items())
    arrays = {'x_mm': x, 'z_mm': np.array(SECTIONS_Z)}
    counts = {}
    for r, d in zip(rows, data):
        for j, (e, m) in enumerate(d):
            arrays[f"{r['name']}_experiment_y_z{SECTIONS_Z[j]}"] = e
            arrays[f"{r['name']}_analytical_y_z{SECTIONS_Z[j]}"] = m
            counts[f"{r['name']}_z{SECTIONS_Z[j]}"] = int((np.isfinite(e) & np.isfinite(m)).sum())
    np.savez(OUT / 'calibration_transfer_sections.npz', **arrays)
    keys = ['rms_symmetric_mm', 'rms_target_to_sim_mm', 'rms_sim_to_target_weighted_mm', 'p95_target_to_sim_mm',
            'p95_sim_to_target_mm', 'max_target_to_sim_mm', 'max_sim_to_target_mm']
    table = {r['name']: {k: r['forward_metrics'][k] for k in keys} for r in rows}
    stats = {r['name']: {k: v for k, v in r['statistics'].items() if k not in ('metrics', 'parameters')} for r in rows}
    (OUT / 'calibration_transfer_metrics.json').write_text(json.dumps(table, indent=2) + '\n')
    (OUT / 'calibration_transfer.provenance.json').write_text(json.dumps({
        'input_sha256': pins, 'script_sha256': digest(Path(__file__)), 'frozen_parameters': rows[0]['frozen'],
        'maps': maps, 'sections': {**sect, 'x_spacing_mm': .5, 'joint_finite_counts': counts,
                                  'experiment_max_triangle_edge_mm': MAX_EDGE_MM},
        'height_difference_statistics': stats, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__}, indent=2) + '\n')
    print(json.dumps(table, indent=1))
    print(json.dumps({'maps': maps, 'sections': sect, 'height_difference_statistics': stats}, indent=1))


if __name__ == '__main__':
    main()
