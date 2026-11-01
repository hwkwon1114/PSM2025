"""Plot the observed Run 1/3 trade-off from the completed fresh search only."""
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
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.case_labels import archive_case_id

CAMPAIGN = ROOT / 'run/fresh_lhs_joint4'
CONFIG = ROOT / 'calibration/bo/fresh_lhs_joint4.json'
DEFAULT_OUTPUT = ROOT / 'calibration/paper_figures/fresh_lhs_joint4'
STEM = 'run1_run3_pareto_tradeoff'
PARAMETERS = ('gtop', 'ortho_top', 'beta', 'floor')


def nondominated(points):
    """Return flags for observed points minimizing both coordinates, including ties."""
    return [not any(x <= a and y <= b and (x < a or y < b)
                    for x, y in points) for a, b in points]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results():
    summary = json.loads((CAMPAIGN / 'summary.json').read_text())
    config = json.loads(CONFIG.read_text())
    paths = sorted((CAMPAIGN / 'results').glob('a*.json'))
    records = []
    for path in paths:
        r = json.loads(path.read_text())
        if path.stem != f"a{r['attempt']:04d}":
            raise ValueError(f'Attempt ID does not match filename: {path}')
        r['phase'] = 'LHS' if r['wave'] <= config['design_waves'] else 'BO'
        if r['valid']:
            e = r['case_rms_mm']
            r['run1'] = float(e[archive_case_id('run1')])
            r['run3'] = float(e[archive_case_id('run3')])
            if not all(math.isfinite(x) and x > 0 for x in (r['run1'], r['run3'])):
                raise ValueError(f'Invalid discrepancy in {path}')
            expected = math.sqrt((r['run1']**2 + r['run3']**2) / 2)
            if not math.isclose(float(r['objective_mm']), expected, rel_tol=1e-10):
                raise ValueError(f'Unexpected joint objective in {path}')
        records.append(r)
    valid = [r for r in records if r['valid']]
    if (len(records) != summary['evaluations'] or len(valid) != summary['valid']
            or len({r['attempt'] for r in records}) != len(records)):
        raise ValueError('Saved results do not match summary counts or have duplicate IDs')
    best = min(valid, key=lambda r: r['objective_mm'])
    if (best['attempt'] != summary['best']['attempt']
            or not math.isclose(best['objective_mm'], summary['best']['objective_mm'], rel_tol=1e-12)):
        raise ValueError('Selected candidate does not match the saved summary')
    flags = nondominated([(r['run1'], r['run3']) for r in valid])
    for r, flag in zip(valid, flags):
        r['pareto'] = flag
    if not best['pareto']:
        raise ValueError('Joint optimum is unexpectedly dominated')
    return records, valid, best, paths


def check_annotation_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        boxes = []
        for text in ax.texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            box = Text.get_window_extent(text, renderer)
            bounds = ax.get_window_extent(renderer)
            if (box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1
                    or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1):
                raise ValueError(f'Annotation outside axes: {text.get_text()}')
            for other in boxes:
                if box.overlaps(other):
                    raise ValueError('Overlapping plot annotations')
            for collection in ax.collections:
                if not isinstance(collection, PathCollection) or not collection.get_visible():
                    continue
                size = max(collection.get_sizes(), default=0)
                radius = math.sqrt(size) * fig.dpi / 144 + 1
                for x, y in collection.get_offset_transform().transform(collection.get_offsets()):
                    if (box.x0 - radius <= x <= box.x1 + radius
                            and box.y0 - radius <= y <= box.y1 + radius):
                        raise ValueError(f'Annotation overlaps a plotted point: {text.get_text()}')
            boxes.append(box)


def make_figure(valid, best):
    style = {'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
             'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
             'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'savefig.facecolor': 'white'}
    blue, orange, gray = '#0072B2', '#D55E00', '#777777'
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55))
        fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.26, wspace=.36)
        for ax in axes:
            for phase, marker in [('LHS', 'x'), ('BO', '.')]:
                subset = [r for r in valid if r['phase'] == phase]
                ax.scatter([r['run1'] for r in subset], [r['run3'] for r in subset],
                           marker=marker, s=24 if phase == 'LHS' else 18,
                           color=gray, linewidths=.8, alpha=.65, zorder=2)
            front = [r for r in valid if r['pareto']]
            ax.scatter([r['run1'] for r in front], [r['run3'] for r in front],
                       marker='o', s=46, facecolors='white', edgecolors=blue,
                       linewidths=1.2, zorder=3)
            ax.scatter(best['run1'], best['run3'], marker='*', s=120,
                       color=orange, edgecolors='black', linewidths=.5, zorder=4)
            ax.set(xlabel='Run 1 discrepancy (mm)', ylabel='Run 3 discrepancy (mm)')
            ax.grid(color='#dddddd', linewidth=.5, zorder=0)
        axes[0].set_title('(a)  All valid candidates', loc='left')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlim(.55, 40)
        axes[0].set_ylim(.14, 26)
        for axis, ticks in ((axes[0].xaxis, [.6, 1, 3, 10, 30]),
                            (axes[0].yaxis, [.2, .5, 1, 3, 10, 20])):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            axis.set_minor_locator(NullLocator())
        axes[1].set_title('(b)  Frontier detail', loc='left')
        axes[1].set(xlim=(.60, .70), ylim=(.15, .32),
                    xticks=[.60, .62, .64, .66, .68, .70],
                    yticks=[.15, .20, .25, .30])
        for label, row, pos in [
                ('Run 1 min', min(valid, key=lambda r: r['run1']), (.05, .83)),
                ('Joint min', best, (.04, .27)),
                ('Run 3 min', min(valid, key=lambda r: r['run3']), (.77, .04))]:
            axes[1].annotate(label, (row['run1'], row['run3']), xytext=pos,
                             textcoords='axes fraction', fontsize=8,
                             arrowprops={'arrowstyle': '-', 'color': '#333333', 'lw': .7})
        n_lhs = sum(r['phase'] == 'LHS' for r in valid)
        n_bo = len(valid) - n_lhs
        n_front = sum(r['pareto'] for r in valid)
        fig.legend(handles=[
            Line2D([], [], marker='x', color=gray, linestyle='', label=f'LHS ({n_lhs})'),
            Line2D([], [], marker='.', color=gray, linestyle='', label=f'BO ({n_bo})'),
            Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor=blue,
                   color=blue, linestyle='', label=f'Observed frontier ({n_front})'),
            Line2D([], [], marker='*', markerfacecolor=orange, markeredgecolor='black',
                   markeredgewidth=.5, color=orange, markersize=9, linestyle='', label='Minimum joint $J$')
        ], loc='lower center', bbox_to_anchor=(.5, .04), ncol=4, frameon=False)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    out = parser.parse_args().out_dir
    targets = [out / f'{STEM}.{ext}' for ext in ('pdf', 'png', 'csv', 'caption.md')]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('Refusing to overwrite a previous figure or data file')
    records, valid, best, paths = load_results()
    n_front = sum(r['pareto'] for r in valid)
    zoom_n = sum(.60 <= r['run1'] <= .70 and .15 <= r['run3'] <= .32 for r in valid)
    out.mkdir(parents=True, exist_ok=True)
    fig = make_figure(valid, best)
    check_annotation_layout(fig)
    with plt.rc_context({'pdf.fonttype': 42, 'savefig.facecolor': 'white'}):
        fig.savefig(targets[0])
        fig.savefig(targets[1], dpi=300)
    plt.close(fig)
    with targets[2].open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['attempt', 'wave', 'phase', 'valid', 'run1_rms_mm', 'run3_rms_mm',
                         'joint_rms_mm', 'observed_nondominated', 'selected', *PARAMETERS])
        for r in records:
            writer.writerow([f"a{r['attempt']:04d}", r['wave'], r['phase'], r['valid'],
                             r.get('run1', ''), r.get('run3', ''),
                             r['objective_mm'] if r['valid'] else '', r.get('pareto', ''),
                             r['attempt'] == best['attempt'],
                             *(r['parameters'][name] for name in PARAMETERS)])
    aggregate = hashlib.sha256()
    for path in paths:
        aggregate.update(f'{path.name} {sha256(path)}\n'.encode())
    with targets[3].open('w') as stream:
        stream.write(f'''# Run 1–Run 3 calibration trade-off

Observed symmetric shared-footprint surface RMS discrepancies for the fresh joint LHS-to-BO search. Both axes are training/calibration cases in millimetres; lower is better. (a) All {len(valid)} valid candidates, with logarithmic axes. (b) Linear-scale detail showing {zoom_n} of those candidates; the other {len(valid)-zoom_n} lie outside the detail limits. Crosses are valid LHS points; dots are valid BO points. Blue rings identify the {n_front} points not dominated by any other valid saved candidate; the star is the selected minimum of J = sqrt((E1² + E3²)/2). The three text labels mark the smallest observed Run 1 discrepancy, the selected minimum J, and the smallest observed Run 3 discrepancy. Each point is a shared four-parameter set, not an independent experimental replicate.

The full search contains {len(records)} attempts; {len(records)-len(valid)} invalid attempts have no plottable pair and are excluded, not assigned penalty coordinates. The search optimized the single joint objective, not a multiobjective acquisition, so the nondominated subset is an observed frontier only—not a proven global frontier or evidence of an irreducible trade-off. Candidate results are adaptive, both cases were used for calibration, and no uncertainty interval or held-out validation is implied. The BO bounds and model were informed by earlier work. No solver, new optimization, scan scoring or Run 2 data was used for this figure.

Source: `run/fresh_lhs_joint4/results/a*.json` and `run/fresh_lhs_joint4/summary.json`; settings: `calibration/bo/fresh_lhs_joint4.json`. Public Run 1/3 map to historical result keys through `calibration/case_labels.py`. Invalid points are retained without invented scores in `{STEM}.csv`. Result-file aggregate SHA-256 (sorted `filename SHA-256\\n` entries): `{aggregate.hexdigest()}`. Summary SHA-256: `{sha256(CAMPAIGN / 'summary.json')}`; config SHA-256: `{sha256(CONFIG)}`; source SHA-256: `{sha256(Path(__file__))}`. Matplotlib {matplotlib.__version__}; Python {sys.version.split()[0]}; figure 7.4 × 3.55 inches, PNG 300 dpi.

Regenerate to a **new** directory: `python calibration/analysis/plot_fresh_pareto_tradeoff.py --out-dir /path/to/new/output`.
''')
    print(f'{len(records)} attempts, {len(valid)} valid, {n_front} nondominated; detail contains {zoom_n} valid points')
    print(f"Selected: a{best['attempt']:04d}; outputs: {out}")


if __name__ == '__main__':
    main()
