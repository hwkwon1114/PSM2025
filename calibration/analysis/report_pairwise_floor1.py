#!/usr/bin/env python3
"""Compare completed, frozen three-pair calibrations without rerunning a solver."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'run/pairwise_floor1'
OUT = ROOT / 'calibration/paper_figures/pairwise_floor1'
PAIRS = {'12': ('run6', 'run9'), '23': ('run9', 'run10'), '13': ('run6', 'run10')}
EXCLUDED = {'12': 'run10', '23': 'run6', '13': 'run9'}
PUBLIC = {'run6': 'Run 1', 'run9': 'Run 2', 'run10': 'Run 3'}
PARAMS = ('gtop', 'ortho_top', 'beta', 'floor')
LABELS = (r'$g_{\rm top}$', r'$o_{\rm top}$', r'$\beta$', 'Floor')
COLORS = {'12': '#0072B2', '23': '#D55E00', '13': '#009E73'}
TERMINAL = {'budget_complete', 'early_stop_no_improvement', 'exact_zero_optimum',
            'stopped_consecutive_failures', 'allocation_limit'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def failure_class(record):
    """Match the reviewed controller's on-read native numerical-failure classification."""
    kind = record.get('failure_class')
    if not record['valid'] and kind == 'infrastructure' and str(record.get('error', '')).startswith('Solver exit code'):
        for path in sorted(Path(record['directory']).glob('case_*/solver.log')):
            if 'zigzag_sequence: equilibrium corrector failed' in path.read_text(errors='replace')[-4000:]:
                return 'solver'
    return kind


def load_pair(base, pair):
    folder = base / f'pair_{pair}'
    summary = read(folder / 'summary.json')
    if summary.get('status') not in TERMINAL:
        raise ValueError(f'{pair}: search is not in a reportable terminal state')
    paths = sorted((folder / 'results').glob('a*.json'))
    records = []
    for path in paths:
        r = read(path)
        if path.name != f"a{r['attempt']:04d}.json" or r['wave'] != (r['attempt'] - 1) // 4 + 1:
            raise ValueError(f'{path}: inconsistent attempt or wave')
        proposal = read(folder / 'proposals' / path.name)
        if any(r[k] != proposal[k] for k in ('attempt', 'wave', 'x', 'parameters')):
            raise ValueError(f'{path}: proposal mismatch')
        if r['valid']:
            if set(r['case_rms_mm']) != set(PAIRS[pair]):
                raise ValueError(f'{path}: wrong pair')
            e = [float(r['case_rms_mm'][name]) for name in PAIRS[pair]]
            if not all(math.isfinite(x) and x >= 0 for x in e):
                raise ValueError(f'{path}: invalid discrepancy')
            if not math.isclose(float(r['objective_mm']), math.hypot(*e) / math.sqrt(2), rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError(f'{path}: objective mismatch')
        elif failure_class(r) not in {'solver', 'timeout', 'registration', 'infrastructure'}:
            raise ValueError(f'{path}: unknown failure')
        records.append(r)
    if len(records) != summary['evaluations'] or len({r['attempt'] for r in records}) != len(records):
        raise ValueError(f'{pair}: summary count or attempt IDs do not match')
    valid = [r for r in records if r['valid']]
    if len(valid) != summary['valid'] or not valid:
        raise ValueError(f'{pair}: no valid selection or summary mismatch')
    best = min(valid, key=lambda r: (r['objective_mm'], r['attempt']))
    selection = read(folder / 'selection.json')
    if (selection['pair'] != pair or selection['attempt'] != best['attempt']
            or selection['parameters'] != best['parameters']
            or selection['case_rms_mm'] != best['case_rms_mm']
            or selection['excluded_case'] != EXCLUDED[pair]
            or selection['result_sha256'] != digest(folder / 'results' / f"a{best['attempt']:04d}.json")
            or selection['plan_sha256'] != digest(folder / 'plan.json')):
        raise ValueError(f'{pair}: frozen selection mismatch')
    transfer_path = folder / 'transfer_result.json'
    unavailable_path = folder / 'transfer_unavailable.json'
    if transfer_path.is_file() == unavailable_path.is_file():
        raise ValueError(f'{pair}: missing or conflicting transfer outcome')
    if transfer_path.is_file():
        transfer = read(transfer_path)
        if transfer['valid']:
            cases = transfer['cases']
            if len(cases) != 1 or cases[0]['id'] != EXCLUDED[pair]:
                raise ValueError(f'{pair}: wrong transfer case')
            score = float(cases[0]['case_rms'])
            if not math.isfinite(score) or score < 0 or not math.isclose(score, transfer['objective_mm'], rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError(f'{pair}: invalid transfer score')
            outcome = {'status': 'valid', 'case': EXCLUDED[pair], 'rms_mm': score}
        else:
            outcome = {'status': 'failed', 'case': EXCLUDED[pair], 'failure_class': transfer.get('failure_class'),
                       'error': transfer.get('error')}
    else:
        outcome = {'status': 'unavailable', 'case': EXCLUDED[pair], **read(unavailable_path)}
    inputs = [folder / 'summary.json', folder / 'selection.json', folder / 'plan.json',
              transfer_path if transfer_path.is_file() else unavailable_path, *paths]
    inputs += [folder / 'proposals' / p.name for p in paths]
    return {'pair': pair, 'summary': summary, 'records': records, 'best': best,
            'transfer': outcome, 'hashes': {str(p): digest(p) for p in inputs}}


def validate_matched_design(groups):
    designs = [{r['attempt']: r['x'] for r in g['records'] if r['attempt'] <= 16} for g in groups]
    common = set.intersection(*(set(x) for x in designs))
    if any(any(design[a] != designs[0][a] for design in designs[1:]) for a in common):
        raise ValueError('Initial LHS designs differ across paired searches')
    return len(common)


def make_history(groups):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, sharey=True, layout='constrained')
    for ax, group in zip(axes, groups):
        records = group['records']
        pair = group['pair']
        incumbent = math.inf
        waves, values = [], []
        for wave in sorted({r['wave'] for r in records}):
            batch = [r for r in records if r['wave'] == wave]
            valid = [r for r in batch if r['valid']]
            if valid:
                incumbent = min(incumbent, *(r['objective_mm'] for r in valid))
            if math.isfinite(incumbent):
                waves.append(max(r['attempt'] for r in batch))
                values.append(incumbent)
        valid = [r for r in records if r['valid']]
        failed = [r for r in records if not r['valid']]
        ax.scatter([r['attempt'] for r in valid], [r['objective_mm'] for r in valid],
                   s=15, alpha=.55, color=COLORS[pair], label='Valid candidate')
        ax.step(waves, values, where='post', color=COLORS[pair], lw=1.8, label='Batch incumbent')
        ax.scatter([r['attempt'] for r in failed], [0.025] * len(failed), transform=ax.get_xaxis_transform(),
                   marker='x', s=25, color='#333333', label='Failed (no score)', clip_on=False)
        ax.axvline(16.5, color='#888888', lw=.8, ls='--')
        ax.set_ylabel('Joint RMS (mm)')
        ax.set_title(f'Run {pair[0]} + Run {pair[1]}', loc='left')
        ax.grid(alpha=.2)
    if any(r['objective_mm'] == 0 for g in groups for r in g['records'] if r['valid']):
        axes[0].set_yscale('symlog', linthresh=1e-6)
    else:
        axes[0].set_yscale('log')
    axes[-1].set_xlabel('Candidate evaluation number (within 4-candidate batches)')
    axes[0].legend(loc='upper right', fontsize=8, frameon=False)
    return fig


def make_parameters(groups):
    fig, axes = plt.subplots(3, 4, figsize=(10, 6.4), sharey=True, layout='constrained')
    for row, group in enumerate(groups):
        pair = group['pair']
        valid = [r for r in group['records'] if r['valid']]
        best = group['best']
        for col, (key, label) in enumerate(zip(PARAMS, LABELS)):
            ax = axes[row, col]
            ax.scatter([r['parameters'][key] for r in valid], [r['objective_mm'] for r in valid],
                       s=13, alpha=.45, color=COLORS[pair])
            ax.scatter(best['parameters'][key], best['objective_mm'], marker='*', s=90,
                       color=COLORS[pair], edgecolor='black', linewidth=.5, zorder=4)
            if key in ('gtop', 'beta'):
                ax.set_xscale('log')
            if row == 2:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(f'Run {pair[0]} + {pair[1]}\nJoint RMS (mm)')
            ax.grid(alpha=.18)
    if any(r['objective_mm'] == 0 for g in groups for r in g['records'] if r['valid']):
        axes[0, 0].set_yscale('symlog', linthresh=1e-6)
    else:
        axes[0, 0].set_yscale('log')
    return fig


def write_report(groups, out):
    targets = [out / f'{stem}.{ext}' for stem in ('pairwise_optimization_history', 'pairwise_parameter_observations')
               for ext in ('pdf', 'png')]
    targets += [out / 'pairwise_results.csv', out / 'pairwise_provenance.json', out / 'pairwise_figures.caption.md']
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Refusing to overwrite existing report artifacts')
    matched = validate_matched_design(groups)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        for figure, stem in ((make_history(groups), 'pairwise_optimization_history'),
                             (make_parameters(groups), 'pairwise_parameter_observations')):
            figure.savefig(out / f'{stem}.pdf')
            figure.savefig(out / f'{stem}.png', dpi=300)
            plt.close(figure)
    with (out / 'pairwise_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'attempt', 'wave', 'valid', 'failure_class', 'joint_rms_mm',
                         'run1_rms_mm', 'run2_rms_mm', 'run3_rms_mm', 'selected', *PARAMS])
        for group in groups:
            for r in group['records']:
                errors = r.get('case_rms_mm', {}) if r['valid'] else {}
                writer.writerow([group['pair'], r['attempt'], r['wave'], r['valid'], failure_class(r) or '',
                                 r['objective_mm'] if r['valid'] else '', *(errors.get(name, '') for name in ('run6','run9','run10')),
                                 r['attempt'] == group['best']['attempt'], *(r['parameters'][p] for p in PARAMS)])
    payload = {'matched_initial_candidates': matched,
               'pairs': {g['pair']: {'status': g['summary']['status'], 'selection': g['best'],
                                    'transfer': g['transfer'], 'source_sha256': g['hashes']} for g in groups},
               'script_sha256': digest(Path(__file__))}
    (out / 'pairwise_provenance.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    counts = '; '.join(f"{g['pair']}: {g['summary']['evaluations']} attempted, {g['summary']['valid']} valid, "
                       f"status {g['summary']['status']}" for g in groups)
    (out / 'pairwise_figures.caption.md').write_text(
        '# Pairwise calibration figures\n\n'
        'Each pair minimizes the equal-weight joint RMS of its two training cases. The three searches use the same '
        'initial 16-point LHS and then adaptive BO, with the same four-parameter bounds including floor [0, 1]. '
        f'{matched} saved initial candidates could be matched across all three; {counts}. '
        'Candidates are dependent adaptive evaluations, not independent experimental replicates.\n\n'
        '**Optimization history:** dots show valid candidates; the step is the best valid objective through each completed batch. '
        'Crosses at the axis bottom denote failed candidates and have no objective value. The dashed line separates LHS from BO. '
        'The axis is logarithmic unless an exact zero is observed, when it is symmetric-log with a 1e-6 mm linear region.\n\n'
        '**Parameter observations:** all valid evaluated candidates are shown against each physical parameter; stars mark the '
        'minimum joint objective in each search. Growth and beta use logarithmic axes. These are projections of the sampled '
        'four-dimensional response, not profiles, confidence intervals or proofs of identifiability. At floor = 1 beta has '
        'no mechanical effect. Search stopping and bounds limit all comparisons.\n\n'
        'Each chosen fit is frozen before the excluded-case check; outcomes and failures are in pairwise_provenance.json. '
        'All three cases were previously seen, so the checks are exploratory, not blind validation. See the CSV for every '
        'attempt, including failures, and provenance JSON for input identities and frozen transfer outcomes.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    groups = [load_pair(args.base, p) for p in ('12', '23', '13')]
    write_report(groups, args.out)
    print(f'Wrote three-pair comparison to {args.out}')


if __name__ == '__main__':
    main()
