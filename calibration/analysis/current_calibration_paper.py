"""Figures for the completed joint calibration, using saved evaluations only.

Public Run 1/2/3 are historical run6/run9/run10. No solver, optimization,
registration, or mutation of archived records is performed. The small post-hoc
GP is descriptive and is not the optimizer's saved GP or a parameter posterior.
"""
import json
import os
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['JOINT_SPEC'] = str(ROOT / 'calibration/bo/stage4_reduced4.json')
from calibration.bo import run_joint_run6_run10 as joint
from calibration.case_labels import PUBLIC_TO_ARCHIVE, archive_case_id, public_case_label
from calibration.analysis import convention_corrected_figures as shapes
from calibration.analysis.run6_run9_six_panel import load_evaluation_data, pin
from calibration.ewcal.common import digest

OUT = ROOT / 'calibration/paper_figures/current_joint_calibration'
CAMPAIGN = ROOT / 'run/joint_stage4_reduced4'
JOINT_CASES = CAMPAIGN / 'candidates/a0016/evaluations/attempt_0001'
CHECK_CASE = ROOT / 'run/run9_interim/joint_stage4_reduced4_a0016_long/evaluations/attempt_0001'
BASELINE = ROOT / 'run/joint_stage2/arm_A/results/a0001.json'
NAMES = ['gtop', 'ortho_top', 'beta', 'floor']
PLOT_NAMES = [r'$g_{\rm top}$', r'$o_{\rm top}$', r'$\beta$', r'$f$']
BLUE, ORANGE, RED = '#0072B2', '#D55E00', '#C44E52'
STYLE = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}


def read_json(path):
    return json.loads(path.read_text())


def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
    plt.close(fig)


def deduplicate(records):
    # Warm-start and rerun points are not independent observations. Later records
    # replace earlier ones at identical four-dimensional parameter coordinates.
    result = {}
    for r in records:
        if r['valid']:
            x = joint.encode(r['parameters'])
            result[tuple(np.round(x, 11))] = (x, float(r['objective_mm']))
    return np.array([v[0] for v in result.values()]), np.array([v[1] for v in result.values()])


def fit_gp(records):
    X, J = deduplicate(records)
    if len(X) < 8 or np.any(~np.isfinite(J)) or np.any(J <= 0):
        raise ValueError('Insufficient finite positive GP observations')
    # Fixed nugget is numerical regularization on standardized log J, not a
    # measured noise model; the intervals are conditional on fitted hyperparameters.
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
        np.ones(4), length_scale_bounds=(0.05, 20.0), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True,
                                  n_restarts_optimizer=1, random_state=2026)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        gp.fit(X, np.log(J))
    return gp, X, J, [str(w.message) for w in caught]


def history(warm, records, pins):
    from matplotlib.lines import Line2D

    def load_campaign(relative):
        rows = []
        for path in sorted((ROOT / relative / 'results').glob('a*.json')):
            row = read_json(pin(path, pins))
            backup = Path(str(path) + '.pre_metric_revision.bak')
            row['rescored'] = backup.is_file()
            if row['rescored']:
                pin(backup, pins)
            rows.append(row)
        return rows

    prior = load_campaign('run/joint_run6_run10')
    arm_a = load_campaign('run/joint_stage2/arm_A')
    arm_b = load_campaign('run/joint_stage2/arm_B')
    extension = load_campaign('run/joint_stage3_gbot0')
    for relative in ('calibration/bo/stage2_arm_A.json', 'calibration/bo/stage2_arm_B.json',
                     'calibration/bo/stage3_gbot0.json'):
        pin(ROOT / relative, pins)
    if [len(rows) for rows in (prior, arm_a, arm_b, extension, records)] != [48, 44, 60, 24, 48]:
        raise ValueError('Unexpected saved campaign sizes')
    assert all(a['parameters'] == b['parameters'] and a['valid'] == b['valid']
               for a, b in zip(arm_a[:12], arm_b[:12]))
    eligible = [r for r in extension if r['parameters']['gbot'] == 0.0]
    assert len(eligible) == len(warm) == 10
    by_id = {r['attempt']: r for r in eligible}
    assert all(-r['attempt'] in by_id and
               r['objective_mm'] == by_id[-r['attempt']]['objective_mm'] for r in warm)
    assert all(np.isfinite(r['objective_mm']) and r['objective_mm'] > 0
               for rows in (prior, arm_a, arm_b, extension, records)
               for r in rows if r['valid'])
    a_best = min(r['objective_mm'] for r in arm_a if r['valid'])
    warm_best = min(r['objective_mm'] for r in warm if r['valid'])
    prior_extension_best = min(r['objective_mm'] for r in extension if r['valid'])
    assert np.isclose(min(warm_best, *(r['objective_mm'] for r in records if r['valid'])),
                      read_json(CAMPAIGN / 'summary.json')['best']['objective_mm'])

    rng = np.random.default_rng(2026)

    def draw(ax, rows, *, warm_value=None, alternate=False, offset=0.0, failures=True, visible_max=None):
        waves = sorted({r['wave'] for r in rows})
        x, incumbent = ([0], [warm_value]) if warm_value is not None else ([], [])
        best = warm_value if warm_value is not None else float('inf')
        color, marker = ('#666666', '^') if alternate else (BLUE, 'o')
        for wave in waves:
            subset = [r for r in rows if r['wave'] == wave]
            valid = [r for r in subset if r['valid']]
            shown = [r for r in valid if visible_max is None or r['objective_mm'] <= visible_max]
            jitter = rng.uniform(-.07 if offset else -.14, .07 if offset else .14, len(shown))
            for open_marker in (False, True):
                idx = [i for i, r in enumerate(shown) if r.get('rescored', False) == open_marker]
                if idx:
                    ax.scatter(wave + offset + jitter[idx], [shown[i]['objective_mm'] for i in idx],
                               s=19, marker=marker, edgecolors=color, linewidths=.75,
                               facecolors='none' if open_marker else color, alpha=.8, zorder=2)
            failed = [r for r in subset if not r['valid']]
            if failures and failed:
                spread = .16 if offset else .32
                ax.scatter(wave + offset + np.linspace(-spread, spread, len(failed)),
                           np.full(len(failed), .025), transform=ax.get_xaxis_transform(),
                           marker='+' if alternate else 'x', color=RED, s=26, linewidths=1,
                           zorder=4)
            best = min([best] + [r['objective_mm'] for r in valid])
            if np.isfinite(best):
                x.append(wave); incumbent.append(float(best))
        ax.step(x, incumbent, where='post', color='#555555' if alternate else ORANGE,
                ls='--' if alternate else '-', lw=1.5, zorder=3)
        if warm_value is not None:
            ax.scatter([0], [warm_value], marker='D', color=ORANGE, s=25, zorder=5)
        return {'waves': waves, 'incumbent_mm': incumbent, 'warm_best_mm': warm_value,
                'attempts': len(rows), 'valid': sum(r['valid'] for r in rows),
                'rescored_valid': sum(r.get('rescored', False) and r['valid'] for r in rows),
                # Older per-result failure classes can be stale; do not present
                # them as diagnosed causes in this cross-stage display.
                'failures': [dict(attempt=r['attempt'], wave=r['wave'])
                             for r in rows if not r['valid']]}

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.0, 8.8))
        grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .8])
        a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        c, d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
        detail = fig.add_subplot(grid[2, :])
        fig.subplots_adjust(left=.11, right=.98, bottom=.07, top=.90, hspace=.58, wspace=.32)
        h_prior = draw(a, prior)
        h_a = draw(b, arm_a, offset=-.23)
        h_b = draw(b, arm_b, alternate=True, offset=.23)
        h_extension = draw(c, extension, warm_value=a_best)
        h_final = draw(d, records, warm_value=warm_best)
        draw(detail, records, warm_value=warm_best, failures=False, visible_max=.70)
        for ax, title, last in ((a, '(a) Earlier joint search', 12),
                                (b, '(b) Comparison arms', 15),
                                (c, '(c) Bottom-growth extension', 6),
                                (d, '(d) Final reduced search', 12)):
            ax.set_title(title, loc='left')
            ax.set_yscale('log'); ax.set_ylim(.42, 35)
            ax.set_xlim(-.45, last + .55)
            ax.set_xlabel('Batch'); ax.set_ylabel('Joint RMS $J$ (mm)')
            ax.set_xticks([0, 3, 6, 9, 12, 15] if last == 15 else
                          [0, 3, 6] if last == 6 else [0, 3, 6, 9, 12])
            ax.grid(axis='y', color='#e4e4e4', lw=.6)
        detail.set_title('(e) Final search detail', loc='left')
        detail.set_xlim(-.45, 12.55); detail.set_ylim(.47, .72)
        detail.set_xticks(range(0, 13, 2))
        detail.set_xlabel('Batch (0 = eligible warm start)')
        detail.set_ylabel('Joint RMS $J$ (mm)')
        detail.grid(axis='y', color='#e4e4e4', lw=.6)
        legend = [Line2D([], [], marker='o', color=BLUE, ls='', markersize=5, label='Valid: A / others'),
                  Line2D([], [], marker='^', color='#666666', ls='', markersize=5, label='Valid: B'),
                  Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor=BLUE,
                         color=BLUE, ls='', markersize=5, label='Retrospectively rescored'),
                  Line2D([], [], color=ORANGE, lw=1.5, label='Incumbent: A / others'),
                  Line2D([], [], color='#555555', ls='--', lw=1.5, label='Incumbent: B'),
                  Line2D([], [], marker='x', color=RED, ls='', markersize=5, label='Failed: × A, + B'),
                  Line2D([], [], marker='D', color=ORANGE, ls='', markersize=5, label='Warm-start best')]
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .985),
                   ncol=3, frameon=False, fontsize=8)
        save(fig, 'optimization_history')
    return {'earlier_joint_search': h_prior, 'stage2_success_aware': h_a,
            'stage2_failure_imputation': h_b, 'bottom_growth_extension': h_extension,
            'final_reduced_search': h_final, 'stage2_matched_design_attempts_per_arm': 12,
            'final_warm_attempts_from_extension': sorted(-r['attempt'] for r in warm),
            'extension_best_overall_mm': prior_extension_best,
            'failure_note': 'Failure markers exclude objective scores; causes are omitted because some older per-result classes are stale.',
            'note': 'Earlier joint search and comparison arms are separate; only eligible extension records warm-start the final search.'}


def sensitivity(warm, records, selected):
    valid = [r for r in records if r['valid']]
    gp, X, J, warnings_fit = fit_gp(warm + records)
    train = warm + [r for r in records if r['wave'] <= 8]
    hold = [r for r in valid if r['wave'] > 8]
    gp_early, X_early, _, warnings_early = fit_gp(train)
    Xhold = np.array([joint.encode(r['parameters']) for r in hold])
    yhold = np.log([r['objective_mm'] for r in hold])
    pred, sd = gp_early.predict(Xhold, return_std=True)
    err = pred - yhold
    check = {'training_unique': len(X_early), 'holdout_valid': len(hold),
             'holdout_waves': sorted({r['wave'] for r in hold}),
             'rmse_logJ': float(np.sqrt(np.mean(err ** 2))),
             'mae_logJ': float(np.mean(np.abs(err))),
             'interval_coverage': float(np.mean(np.abs(err) <= 1.96 * sd)),
             'note': 'Later adaptive valid-only points, not an independent experimental test.'}
    x0 = joint.encode(selected['parameters'])
    slices = {}
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.0))
        fig.subplots_adjust(left=.12, right=.98, bottom=.10, top=.86, hspace=.49, wspace=.30)
        for i, ax in enumerate(axs.flat):
            # Plot within the range seen among valid points, not the whole search box.
            lo, hi = float(X[:, i].min()), float(X[:, i].max())
            grid = np.linspace(lo, hi, 160)
            U = np.tile(x0, (len(grid), 1)); U[:, i] = grid
            mu, std = gp.predict(U, return_std=True)
            physical = np.array([joint.decode(u)[NAMES[i]] for u in U])
            data_physical = np.array([joint.decode(u)[NAMES[i]] for u in X])
            center = selected['parameters'][NAMES[i]]
            median = np.exp(mu); lower = np.exp(mu - 1.96 * std); upper = np.exp(mu + 1.96 * std)
            ax.fill_between(physical, lower, upper, color=BLUE, alpha=.17, lw=0,
                            label='Conditional GP 95% interval')
            ax.plot(physical, median, color=BLUE, lw=1.7, label='GP median')
            ax.scatter([center], [selected['objective_mm']], s=32, color=ORANGE, marker='o',
                       edgecolor='white', linewidth=.5, zorder=4, label='Selected evaluation')
            ax.plot(data_physical, np.full(len(X), .02), '|', ms=5, color='#777777', alpha=.55,
                    transform=ax.get_xaxis_transform())
            if i in (0, 2):
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(axis='y', color='#e4e4e4', lw=.6)
            ax.set_title(f'({chr(97+i)}) {PLOT_NAMES[i]}', loc='left')
            ax.set_xlabel(PLOT_NAMES[i]); ax.set_ylabel('Joint RMS $J$ (mm)')
            slices[NAMES[i]] = {'normalized_range': [lo, hi], 'physical_range': [float(physical[0]), float(physical[-1])],
                                'x': physical.tolist(), 'median_mm': median.tolist(),
                                'lower_95_mm': lower.tolist(), 'upper_95_mm': upper.tolist()}
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .985),
                   ncol=3, frameon=False, fontsize=8)
        save(fig, 'gp_sensitivity')
    return {'n_unique_valid': len(X), 'kernel': str(gp.kernel_), 'warnings': warnings_fit,
            'early_kernel': str(gp_early.kernel_), 'early_warnings': warnings_early,
            'forward_wave_check': check, 'slices': slices}


def shape_figures(selected, pins):
    joint_res = read_json(pin(JOINT_CASES / 'result.json', pins))
    check_res = read_json(pin(CHECK_CASE / 'result.json', pins))
    assert joint_res['valid'] and check_res['valid']
    cases = [(archive_case_id('run1'), 'Run 1 (fit)', JOINT_CASES / 'case_00', joint_res['cases'][0]),
             (archive_case_id('run2'), 'Run 2 (check)', CHECK_CASE / 'case_00', check_res['cases'][0]),
             (archive_case_id('run3'), 'Run 3 (fit)', JOINT_CASES / 'case_01', joint_res['cases'][1])]
    shapes.RUNS = [(old, label) for old, label, _, _ in cases]
    rows = []
    summary = {}
    arrays = {}
    for old, label, case_dir, case in cases:
        assert case['id'] == old
        seq = read_json(pin(case_dir / 'sequence.json', pins))
        op = seq['toolpaths'][0]['operation']; hard = seq['hardening']
        for key in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom'):
            assert np.isclose(op[key], selected['parameters'][key], rtol=0, atol=1e-14), (old, key)
        for key in ('beta', 'floor'):
            assert np.isclose(hard[key], selected['parameters'][key], rtol=0, atol=1e-14), (old, key)
        row = load_evaluation_data(old, case_dir, pins)
        with np.load(case_dir / 'comparison.npz', allow_pickle=False) as f:
            row['distance'] = f['distance_mm'].copy()
        metrics = case['metrics']
        assert np.isclose(np.sqrt(np.mean(row['distance'] ** 2)), metrics['rms_target_to_sim_mm'], atol=1e-9)
        assert np.isclose(np.quantile(row['distance'], .95), metrics['p95_target_to_sim_mm'], atol=1e-9)
        row['forward_metrics'] = metrics
        scan = row['scan']
        tri = mtri.Triangulation(scan[:, 0], scan[:, 1] + shapes.Z_OFFSET)
        corners = scan[tri.triangles, :2]
        long_edge = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1) > shapes.MAX_EDGE_MM
        tri.set_mask(long_edge | ~row['support'][tri.triangles].all(axis=1))
        row['tri'] = tri
        rows.append(row)
        summary[label] = {'rms_symmetric_mm': metrics['rms_symmetric_mm'],
                          'rms_scan_to_model_mm': metrics['rms_target_to_sim_mm'],
                          'p95_scan_to_model_mm': metrics['p95_target_to_sim_mm'],
                          'scan_points': len(scan), 'height_supported_points': int(row['support'].sum()),
                          'height_unsupported_points': int((~row['support']).sum())}
    maps = shapes.residual_maps(rows, OUT / 'shape_fit_maps')
    x = np.linspace(-100, 100, 401)
    sections, settings = shapes.section_figure(rows, OUT / 'shape_fit_sections', x)
    # The legacy helper uses internal (u,v,h) names; expose public y-height names.
    maps['delta_y_limit_mm'] = maps.pop('delta_z_limit_mm')
    maps['points_beyond_delta_y_limit'] = maps.pop('points_beyond_delta_z_limit')
    settings['height_y_limits_mm'] = settings.pop('z_limits_mm')
    settings['delta_y_limit_mm'] = settings.pop('delta_z_limit_mm')
    arrays['x_mm'] = x; arrays['z_mm'] = np.array(shapes.SECTIONS_Z)
    for i, (row, sections_row) in enumerate(zip(rows, sections), start=1):
        for z, (exp, model) in zip(shapes.SECTIONS_Z, sections_row):
            arrays[f'run{i}_experiment_y_z{z}'] = exp
            arrays[f'run{i}_analytical_y_z{z}'] = model
            assert np.count_nonzero(np.isfinite(exp) & np.isfinite(model)) > 100
    np.savez_compressed(OUT / 'shape_fit_sections_data.npz', **arrays)
    return summary, {'maps': maps, 'sections': settings}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pins = {}
    for source in ('calibration/analysis/convention_corrected_figures.py',
                   'calibration/analysis/run6_run9_six_panel.py', 'calibration/case_labels.py',
                   'calibration/bo/run_joint_run6_run10.py', 'calibration/bo/stage4_reduced4.json'):
        pin(ROOT / source, pins)
    summary = read_json(pin(CAMPAIGN / 'summary.json', pins))
    selected = summary['best']
    assert selected['attempt'] == 16
    warm = joint.warm()
    records = [joint.classify(read_json(pin(p, pins))) for p in sorted((CAMPAIGN / 'results').glob('a*.json'))]
    for p in sorted((ROOT / 'run/joint_stage3_gbot0/results').glob('a*.json')):
        pin(p, pins)
    base = read_json(pin(BASELINE, pins))
    print('Drawing history and fitting small post-hoc GP...', flush=True)
    history_info = history(warm, records, pins)
    sensitivity_info = sensitivity(warm, records, selected)
    print('Drawing saved-surface comparisons...', flush=True)
    case_info, shape_info = shape_figures(selected, pins)
    provenance = {'selected': selected, 'baseline': base, 'public_case_metrics': case_info,
                  'history': history_info, 'gp': sensitivity_info, 'shape_figure_settings': shape_info,
                  'public_run_mapping': {public_case_label(v): v for v in PUBLIC_TO_ARCHIVE.values()},
                  'input_sha256': pins, 'script_sha256': digest(Path(__file__)),
                  'matplotlib': matplotlib.__version__, 'numpy': np.__version__}
    assert all(digest(ROOT / p) == h for p, h in pins.items())
    (OUT / 'figure_data_and_provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'history': history_info, 'gp_kernel': sensitivity_info['kernel'],
                      'gp_warnings': sensitivity_info['warnings'], 'gp_check': sensitivity_info['forward_wave_check'],
                      'cases': case_info, 'shapes': shape_info}, indent=2), flush=True)


if __name__ == '__main__':
    main()
