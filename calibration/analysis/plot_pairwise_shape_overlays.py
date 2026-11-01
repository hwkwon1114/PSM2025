#!/usr/bin/env python3
"""Plot the nine saved selected-fit/case overlaps; no optimization or solver runs."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.report_pairwise_floor1 import BASE, OUT, PAIRS, digest, load_pair
from calibration.ewcal.geometry import points, read_mesh

STEM = 'pairwise_shape_overlays_3x3_v2'
CASES = ('run6', 'run9', 'run10')
PAIR_ORDER = ('12', '23', '13')
FIT_LABELS = {'12': 'Fit: Runs 1 + 2', '23': 'Fit: Runs 2 + 3', '13': 'Fit: Runs 1 + 3'}
SCAN_COLOR, MODEL_COLOR = '#D55E00', '#0072B2'
N_SCATTER = 2500
SEED = 20261029
HEIGHT_STRETCH = 8


def grid_lines(q, faces):
    """Linear interpolation on the saved model facets, masked outside the mesh."""
    tri = mtri.Triangulation(q[:, 0], q[:, 1], faces)
    interpolator = mtri.LinearTriInterpolator(tri, q[:, 2])
    lines = []
    for u in np.linspace(-95, 95, 9):
        v = np.linspace(-145, 145, 151)
        x = np.full_like(v, u)
        lines.append(np.column_stack((x, v, interpolator(x, v).filled(np.nan))))
    for v in np.linspace(-145, 145, 13):
        x = np.linspace(-95, 95, 101)
        y = np.full_like(x, v)
        lines.append(np.column_stack((x, y, interpolator(x, y).filled(np.nan))))
    return lines


def load_panel(group, case, scan_reference):
    pair = group['pair']
    transfer = case == group['transfer']['case']
    if transfer:
        if group['transfer']['status'] != 'valid':
            raise ValueError(f'{pair}: excluded-case transfer is not valid')
        transfer_record = json.loads((BASE / f'pair_{pair}' / 'transfer_result.json').read_text())
        folder = Path(transfer_record['cases'][0]['directory'])
        rms = float(transfer_record['objective_mm'])
    else:
        index = PAIRS[pair].index(case)
        folder = Path(group['best']['directory']) / f'case_{index:02d}'
        rms = float(group['best']['case_rms_mm'][case])
    files = [folder / 'comparison.npz', folder / 'prediction_mm.vtp',
             folder / 'bilayer_zigzag_sequence_native_hybrid_state.json', folder / 'sequence.json']
    with np.load(files[0], allow_pickle=False) as f:
        target = f['target_mm'].copy()
    q = points(read_mesh(files[1]))
    state = json.loads(files[2].read_text())
    seq = json.loads(files[3].read_text())
    faces = np.asarray(state['faces'], dtype=int)
    native = np.asarray(state['positions'], dtype=float) * 1000
    params = group['best']['parameters']
    operation = seq['toolpaths'][0]['operation']
    if (seq['toolpaths'][0]['id'] != case or
        any(operation[k] != params[k] for k in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom')) or
        any(seq['hardening'][k] != params[k] for k in ('beta', 'floor'))):
        raise ValueError(f'{pair}/{case}: saved sequence differs from frozen parameters')
    if (target.ndim != 2 or target.shape[1] != 3 or not np.isfinite(target).all() or
        q.shape != native.shape or not np.isfinite(q).all() or
        faces.ndim != 2 or faces.shape[1] != 3):
        raise ValueError(f'{pair}/{case}: invalid saved geometry')
    for a, b in ((0, 1), (1, 2), (2, 0)):
        original = np.linalg.norm(native[faces[:, a]] - native[faces[:, b]], axis=1)
        transformed = np.linalg.norm(q[faces[:, a]] - q[faces[:, b]], axis=1)
        if np.max(np.abs(original - transformed)) >= 1e-3:
            raise ValueError(f'{pair}/{case}: saved mesh is not a rigid transform of native mesh')
    if case in scan_reference:
        if target.shape != scan_reference[case].shape or not np.allclose(target, scan_reference[case], atol=1e-8, rtol=0):
            raise ValueError(f'{pair}/{case}: scan differs across compared fits')
    else:
        scan_reference[case] = target.copy()
    return {'pair': pair, 'case': case, 'transfer': transfer, 'rms_mm': rms,
            'scan': target, 'model': q, 'faces': faces, 'wires': grid_lines(q, faces),
            'inputs': {str(p.relative_to(ROOT)): digest(p) for p in files},
            'attempt': group['best']['attempt'], 'parameters': params}


def main():
    output = [OUT / f'{STEM}.{ext}' for ext in ('png', 'pdf', 'caption.md', 'json')]
    if any(p.exists() or p.is_symlink() for p in output):
        raise FileExistsError('Refusing to overwrite existing shape-overlap outputs')
    groups = [load_pair(BASE, pair) for pair in PAIR_ORDER]
    reference = {}
    panels = [[load_panel(g, case, reference) for case in CASES] for g in groups]
    sampled = {}
    for i, case in enumerate(CASES):
        n = len(reference[case])
        indices = np.random.default_rng(SEED + i).choice(n, size=min(N_SCATTER, n), replace=False)
        sampled[case] = np.sort(indices)
    arrays = [p[k] for row in panels for p in row for k in ('scan', 'model')]
    xyz_min = np.min([a.min(axis=0) for a in arrays], axis=0)
    xyz_max = np.max([a.max(axis=0) for a in arrays], axis=0)
    # Matplotlib coordinates (width x, length z, height y).  The +65 mm
    # translation converts internal v to public length z without refitting.
    xlim = (float(min(-100, xyz_min[0]) - 4), float(max(100, xyz_max[0]) + 4))
    zlim = (float(min(-150, xyz_min[1]) + 65 - 4), float(max(150, xyz_max[1]) + 65 + 4))
    ylim = (float(np.floor(xyz_min[2]) - .5), float(np.ceil(xyz_max[2]) + .5))
    aspect = (xlim[1]-xlim[0], zlim[1]-zlim[0], (ylim[1]-ylim[0])*HEIGHT_STRETCH)
    style = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 8,
             'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 10,
             'pdf.fonttype': 42, 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13, 13.5))
        fig.subplots_adjust(left=.075, right=.99, top=.91, bottom=.035, wspace=.02, hspace=.20)
        fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=SCAN_COLOR, markersize=5, label='Experiment (sampled points)'),
                            Line2D([], [], color=MODEL_COLOR, linewidth=1.4, label='Simulation (mesh grid)')],
                   loc='upper center', bbox_to_anchor=(.5, .975), ncol=2, frameon=False)
        fig.text(.5, .945, 'Height axis exaggerated 8×; tick values in mm', ha='center', fontsize=9)
        for i, row in enumerate(panels):
            for j, panel in enumerate(row):
                ax = fig.add_subplot(3, 3, i*3+j+1, projection='3d')
                scan = panel['scan'][sampled[panel['case']]]
                ax.scatter(scan[:, 0], scan[:, 1]+65, scan[:, 2], s=1.5, color=SCAN_COLOR,
                           alpha=.55, depthshade=False, rasterized=True)
                for line in panel['wires']:
                    ax.plot(line[:, 0], line[:, 1]+65, line[:, 2], color=MODEL_COLOR, linewidth=.65, alpha=.85)
                ax.set(xlim=xlim, ylim=zlim, zlim=ylim, xlabel='Width x (mm)',
                       ylabel='Length z (mm)', zlabel='Height y (mm)' if j == 0 else '')
                ax.set_xticks([-100, 0, 100]); ax.set_yticks([-85, 65, 215])
                ax.set_zticks([-6, -3, 0, 3]); ax.tick_params(pad=1)
                ax.set_box_aspect(aspect, zoom=1.07)
                ax.view_init(elev=23, azim=-65)
                ax.grid(False)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.pane.set_alpha(0.0)
                label = 'excluded check' if panel['transfer'] else 'calibration'
                ax.set_title(f'Run {j+1} · {label}\nSurface RMS = {panel["rms_mm"]:.3f} mm', pad=3)
            fig.text(.018, .79-i*.293, FIT_LABELS[PAIR_ORDER[i]], rotation=90,
                     va='center', ha='center', fontsize=11, fontweight='bold')
        fig.savefig(output[0], dpi=200)
        fig.savefig(output[1], dpi=200)
        plt.close(fig)
    output[2].write_text(
        '# Final shape overlaps for the three pairwise calibrations\n\n'
        'Rows show the lowest-valid-objective frozen fits calibrated on Runs 1+2, 2+3 and 1+3; '
        'columns show Runs 1, 2 and 3. Each panel overlays the saved experiment (orange points) with '
        'the saved simulation (blue grid). The excluded run in each row is labeled as an exploratory '
        'check, not blind validation: all runs had previously been seen. Titles report the full '
        'shared-footprint symmetric surface RMS in mm, not an RMS of the displayed point sample. '
        'For visual clarity, 2,500 scan points per case were uniformly sampled without replacement '
        f'using deterministic seeds {SEED}, {SEED+1} and {SEED+2}; the same points are used '
        'in all three rows for that case. Simulation grid lines are piecewise-linear interpolations '
        'on saved model facets, masked beyond the mesh; neither surface was refit or smoothed. '
        'Both inputs are the saved material-frame scored surfaces, without a new registration. '
        'Public coordinates are x = width, z = length and y = height; internal v is shifted by '
        '+65 mm. All panels use identical limits, camera and 8× height-axis exaggeration; numeric '
        'tick values remain in mm. Spatially dependent points are not independent replicates. '
        'Full error aggregates and directional p95 are in pairwise_case_error_comparison and '
        'pairwise_floor1_results.md.\n')
    output[3].write_text(json.dumps({
        'script_sha256': digest(Path(__file__)), 'seed': SEED, 'scatter_count': N_SCATTER,
        'camera': {'elevation_deg': 23, 'azimuth_deg': -65, 'height_axis_exaggeration': HEIGHT_STRETCH},
        'limits': {'x_width_mm': xlim, 'z_length_mm': zlim, 'y_height_mm': ylim},
        'sample_indices': {k: v.tolist() for k, v in sampled.items()},
        'panels': [[{k: v for k, v in p.items() if k not in ('scan','model','faces','wires')} for p in row] for row in panels],
        'selection_and_results_sha256': {g['pair']: g['hashes'] for g in groups},
    }, indent=2, allow_nan=False) + '\n')
    print('\n'.join(str(p) for p in output))


if __name__ == '__main__':
    main()
