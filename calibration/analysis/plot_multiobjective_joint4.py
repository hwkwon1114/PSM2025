"""Compare the seed and completed Run 1/Run 3 continuation frontiers; no solver calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.plot_fresh_pareto_tradeoff import (
    CONFIG, check_annotation_layout, load_results, nondominated, sha256,
)
from calibration.case_labels import archive_case_id

CAMPAIGN = ROOT / 'run/multiobjective_joint4'
OUTPUT = ROOT / 'calibration/paper_figures/multiobjective_joint4'
STEM = 'run1_run3_pareto_continuation'
PARAMETERS = ('gtop', 'ortho_top', 'beta', 'floor')
TERMINAL = {'pareto_plateau', 'allocation_limit', 'stopped_consecutive_failures'}


def observations(records, source):
    """Make source-qualified records, leaving failed pairs missing rather than penalized."""
    result = []
    ids = set()
    for record in records:
        if type(record['valid']) is not bool or type(record['attempt']) is not int:
            raise ValueError(f'Invalid {source} record types')
        attempt = record['attempt']
        if attempt <= 0 or attempt in ids:
            raise ValueError(f'Duplicate or invalid {source} attempt: {attempt}')
        ids.add(attempt)
        row = {'source': source, 'attempt': attempt, 'wave': int(record['wave']),
               'valid': bool(record['valid']), 'parameters': record['parameters']}
        if row['valid']:
            case = record['case_rms_mm']
            row['run1'] = float(case[archive_case_id('run1')])
            row['run3'] = float(case[archive_case_id('run3')])
            if not all(math.isfinite(x) and x > 0 for x in (row['run1'], row['run3'])):
                raise ValueError(f'Invalid discrepancy in {source} {attempt}')
            row['joint'] = math.sqrt((row['run1']**2 + row['run3']**2) / 2)
            if not math.isclose(float(record['objective_mm']), row['joint'], rel_tol=1e-10):
                raise ValueError(f'Inconsistent joint objective in {source} {attempt}')
        result.append(row)
    return result


def mark_frontiers(seed, continuation):
    """Mark both initial and pooled observed frontiers; equal coordinates survive."""
    initial = [r for r in seed if r['valid']]
    pooled = initial + [r for r in continuation if r['valid']]
    if not initial:
        raise ValueError('Cannot construct a frontier without a valid seed')
    for r in seed + continuation:
        r['initial_front'] = r['pooled_front'] = False
    for r, flag in zip(initial, nondominated([(r['run1'], r['run3']) for r in initial])):
        r['initial_front'] = flag
    for r, flag in zip(pooled, nondominated([(r['run1'], r['run3']) for r in pooled])):
        r['pooled_front'] = flag
    return pooled


def load_campaign():
    summary_path = CAMPAIGN / 'summary.json'
    plan_path = CAMPAIGN / 'plan.json'
    summary = json.loads(summary_path.read_text())
    if summary['status'] not in TERMINAL:
        raise ValueError('Continuation has no terminal controller summary; inspect job and partial results first')
    plan = json.loads(plan_path.read_text())
    seed_raw, seed_valid, _, seed_paths = load_results()
    for path in [*seed_paths, ROOT / 'run/fresh_lhs_joint4/summary.json', CONFIG]:
        if plan['files'].get(str(path.absolute())) != sha256(path):
            raise ValueError(f'Seed source differs from the frozen plan: {path}')
    paths = sorted((CAMPAIGN / 'results').glob('a*.json'))
    new_raw = []
    for path in paths:
        record = json.loads(path.read_text())
        if path.stem != f"a{record['attempt']:04d}":
            raise ValueError(f'Attempt ID does not match filename: {path}')
        new_raw.append(record)
    seed, new = observations(seed_raw, 'seed'), observations(new_raw, 'continuation')
    pooled = mark_frontiers(seed, new)
    failures = summary['new_failures']
    if (any(type(n) is not int or n < 0 for n in failures.values())
            or len(seed) != summary['seed_evaluations'] or len(seed_valid) != summary['seed_valid']
            or len(new) != summary['new_evaluations']
            or sum(r['valid'] for r in new) != summary['new_valid']
            or sum(failures.values()) != sum(not r['valid'] for r in new)):
        raise ValueError('Result counts do not match the terminal summary')
    expected = {(r['source'], r['attempt']): (r['run1'], r['run3'])
                for r in pooled if r['pooled_front']}
    declared = {(r['source'], r['attempt']): (r['run1_mm'], r['run3_mm'])
                for r in summary['frontier']}
    if (len(declared) != len(summary['frontier']) or expected.keys() != declared.keys()
            or any(not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                           for a, b in zip(expected[k], declared[k])) for k in expected)):
        raise ValueError('Recomputed pooled frontier differs from the terminal summary')
    return seed, new, summary, [*seed_paths, *paths, summary_path, plan_path, CONFIG]


def figure(seed, new):
    valid = mark_frontiers(seed, new)
    initial = [r for r in seed if r['initial_front']]
    final = [r for r in valid if r['pooled_front']]
    seed_valid = [r for r in seed if r['valid']]
    new_valid = [r for r in new if r['valid']]
    style = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
             'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
             'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55))
        fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.27, wspace=.36)
        for ax in axes:
            for rows, marker, color, size in [(seed_valid, '.', '#777777', 16),
                                               (new_valid, 'x', '#D55E00', 24)]:
                ax.scatter([r['run1'] for r in rows], [r['run3'] for r in rows],
                           marker=marker, s=size, color=color, linewidths=.8, alpha=.7, zorder=2)
            ax.scatter([r['run1'] for r in initial], [r['run3'] for r in initial],
                       marker='o', s=58, facecolors='none', edgecolors='#0072B2',
                       linewidths=1.1, zorder=3)
            ax.scatter([r['run1'] for r in final], [r['run3'] for r in final],
                       marker='s', s=38, facecolors='none', edgecolors='black',
                       linewidths=1.0, zorder=4)
            ax.set(xlabel='Run 1 discrepancy (mm)', ylabel='Run 3 discrepancy (mm)')
            ax.grid(color='#dddddd', linewidth=.5, zorder=0)
        axes[0].set_title('(a)  All valid evaluations', loc='left')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[1].set_title('(b)  Frontier detail', loc='left')
        for key, setter in [('run1', axes[1].set_xlim), ('run3', axes[1].set_ylim)]:
            values = [r[key] for r in initial + final]
            lo, hi = min(values), max(values)
            padding = max((hi - lo) * .12, hi * .02, .001)
            setter(max(0, lo - padding), hi + padding)
        fig.legend(handles=[
            Line2D([], [], marker='.', color='#777777', linestyle='', label=f'Seed ({len(seed_valid)})'),
            Line2D([], [], marker='x', color='#D55E00', linestyle='', label=f'New ({len(new_valid)})'),
            Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='#0072B2',
                   color='#0072B2', linestyle='', label=f'Initial frontier ({len(initial)})'),
            Line2D([], [], marker='s', markerfacecolor='none', markeredgecolor='black',
                   color='black', linestyle='', label=f'Pooled frontier ({len(final)})'),
        ], loc='lower center', bbox_to_anchor=(.5, .04), ncol=4, frameon=False)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUTPUT)
    out = parser.parse_args().out_dir
    targets = [out / f'{STEM}.{ext}' for ext in ('pdf', 'png', 'csv', 'caption.md')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Refusing to overwrite an existing figure or data file')
    seed, new, summary, paths = load_campaign()
    valid = [r for r in seed + new if r['valid']]
    initial = [r for r in seed if r['initial_front']]
    final = [r for r in valid if r['pooled_front']]
    fig = figure(seed, new)
    check_annotation_layout(fig)
    xlim, ylim = fig.axes[1].get_xlim(), fig.axes[1].get_ylim()
    zoom_n = sum(xlim[0] <= r['run1'] <= xlim[1] and ylim[0] <= r['run3'] <= ylim[1] for r in valid)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'pdf.fonttype': 42, 'savefig.facecolor': 'white'}):
        fig.savefig(targets[0])
        fig.savefig(targets[1], dpi=300)
    plt.close(fig)
    with targets[2].open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source', 'attempt', 'wave', 'valid', 'run1_rms_mm', 'run3_rms_mm',
                         'joint_rms_mm', 'initial_frontier', 'pooled_frontier', *PARAMETERS])
        for r in seed + new:
            writer.writerow([r['source'], f"a{r['attempt']:04d}", r['wave'], r['valid'],
                             r.get('run1', ''), r.get('run3', ''), r.get('joint', ''),
                             r['initial_front'], r['pooled_front'],
                             *(r['parameters'][name] for name in PARAMETERS)])
    aggregate = hashlib.sha256()
    for path in sorted(paths):
        aggregate.update(f'{path.relative_to(ROOT)} {sha256(path)}\n'.encode())
    with targets[3].open('w') as stream:
        stream.write(f'''# Run 1–Run 3 Pareto continuation

Observed shared-footprint symmetric surface RMS discrepancies, in millimetres; lower is better on both axes. (a) All {len(valid)} valid paired evaluations on logarithmic axes. (b) Linear-scale frontier detail, containing {zoom_n} valid pairs; {len(valid)-zoom_n} lie outside the detail limits. Grey points are the {sum(r['valid'] for r in seed)} valid seed pairs; orange crosses are the {sum(r['valid'] for r in new)} valid new pairs. Blue circles mark the {len(initial)} seed-only nondominated points; black squares mark the {len(final)} pooled nondominated points. A point can carry both markers. Each point is one shared parameter set, not an independent experimental replicate; no interpolated frontier or uncertainty interval is shown.

The seed contains {len(seed)} attempts and the continuation contains {len(new)} attempts, including {sum(not r['valid'] for r in seed)} and {sum(not r['valid'] for r in new)} invalid attempts, respectively. Invalid pairs are retained without invented coordinates in the CSV. The controller stopped with `{summary['status']}`. Both cases were used for calibration, and new proposals were adaptive and conditioned on all seed data. This is a pooled observed frontier within the chosen parameter bounds, not a global-optimum certificate, an independent optimizer comparison, parameter identification or held-out validation. The original seed and new observations used different BLIS settings; exact numerical trajectory equivalence is not established. Run 2 was not used.

Source: `run/fresh_lhs_joint4/results/a*.json`, `run/multiobjective_joint4/results/a*.json`, their summaries, `run/multiobjective_joint4/plan.json` and `calibration/bo/fresh_lhs_joint4.json`. Source-file aggregate SHA-256 (sorted `project-relative-path SHA-256\\n` entries): `{aggregate.hexdigest()}`. Generating script SHA-256: `{sha256(Path(__file__))}`; imported seed-figure helper SHA-256: `{sha256(ROOT / 'calibration/analysis/plot_fresh_pareto_tradeoff.py')}`. Matplotlib {matplotlib.__version__}; Python {sys.version.split()[0]}; figure 7.4 × 3.55 inches, PNG 300 dpi. The scheduler exit and individual artifacts must also be checked separately; a terminal controller summary alone is not a successful-job certificate.

Regenerate to a **new** directory: `python calibration/analysis/plot_multiobjective_joint4.py --out-dir /path/to/new/output`.
''')
    print(f"Controller {summary['status']}: {len(seed)} seed + {len(new)} new attempts, "
          f'{len(initial)} initial and {len(final)} pooled nondominated points; {zoom_n} points in detail')
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
