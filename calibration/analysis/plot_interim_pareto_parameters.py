"""One-time, solver-free snapshot of the running Run 1/3 Pareto continuation."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.plot_fresh_pareto_tradeoff import (
    CONFIG, check_annotation_layout, load_results, make_figure, sha256,
)
from calibration.analysis.plot_multiobjective_joint4 import observations, mark_frontiers

CAMPAIGN = ROOT / 'run/multiobjective_joint4'
OUT = ROOT / 'calibration/paper_figures/multiobjective_joint4_interim'
STEM = 'run1_run3_pareto_tradeoff_with_parameters'
PARAMS = ('gtop', 'ortho_top', 'beta', 'floor')


def main():
    targets = [OUT / f'{STEM}.{ext}' for ext in ('png', 'csv', 'caption.md')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite the previous interim snapshot')
    summary_bytes = (CAMPAIGN / 'summary.json').read_bytes()
    summary = json.loads(summary_bytes)
    plan = json.loads((CAMPAIGN / 'plan.json').read_text())
    seed_raw, seed_valid, _, seed_paths = load_results()
    for path in [*seed_paths, ROOT / 'run/fresh_lhs_joint4/summary.json', CONFIG]:
        if plan['files'].get(str(path.absolute())) != sha256(path):
            raise ValueError(f'Seed differs from campaign plan: {path}')
    count = summary['new_evaluations']
    paths = [CAMPAIGN / 'results' / f'a{i:04d}.json' for i in range(1, count + 1)]
    raw = [json.loads(p.read_text()) for p in paths]
    if any(r['attempt'] != i for i, r in enumerate(raw, 1)):
        raise ValueError('Continuation result IDs do not match the snapshot')
    seed = observations(seed_raw, 'seed')
    new = observations(raw, 'continuation')
    valid = mark_frontiers(seed, new)
    failures = summary['new_failures']
    if (len(seed) != summary['seed_evaluations'] or len(seed_valid) != summary['seed_valid']
            or sum(r['valid'] for r in new) != summary['new_valid']
            or sum(failures.values()) != sum(not r['valid'] for r in new)):
        raise ValueError('Saved results do not match the captured summary counts')
    actual = {(r['source'], r['attempt']): (r['run1'], r['run3']) for r in valid if r['pooled_front']}
    declared = {(r['source'], r['attempt']): (r['run1_mm'], r['run3_mm']) for r in summary['frontier']}
    if (len(declared) != len(summary['frontier']) or actual.keys() != declared.keys()
            or any(not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                           for a, b in zip(actual[k], declared[k])) for k in actual)):
        raise ValueError('Saved frontier differs from recomputed snapshot')
    for r in valid:
        r['pareto'] = r['pooled_front']
        r['phase'] = 'LHS' if r['source'] == 'seed' and r['wave'] <= 4 else 'BO'
    best = min(valid, key=lambda r: r['joint'])
    chosen = [min(valid, key=lambda r: r['run1']), best, min(valid, key=lambda r: r['run3'])]
    if not all(r['pareto'] for r in chosen):
        raise ValueError('Selected illustrative points must be nondominated')
    fig = make_figure(valid, best)
    fig.suptitle(f'Interim snapshot: {len(seed_valid)} seed + {summary["new_valid"]} new valid evaluations', fontsize=10)
    annotations = fig.axes[1].texts
    preferred = [text.get_position() for text in annotations]
    for text in annotations:
        text.set_visible(False)
    for text, origin in zip(annotations, preferred):
        text.set_visible(True)
        candidates = [origin] + sorted(((x, y) for x in (.03, .17, .31, .45, .60, .75)
                                        for y in (.04, .18, .32, .46, .60, .74, .88)),
                                       key=lambda p: (p[0]-origin[0])**2 + (p[1]-origin[1])**2)
        for position in candidates:
            text.set_position(position)
            try:
                check_annotation_layout(fig)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f'Cannot place annotation without overlap: {text.get_text()}')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, facecolor='white')
    plt.close(fig)
    image = Image.open(buffer).convert('RGB')
    output = Image.new('RGB', (image.width, image.height + 320), 'white')
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    regular = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 32)
    bold = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 32)
    x = [100, 710, 1035, 1400, 1710]
    y = image.height + 22
    for xpos, text in zip(x, ['Point', 'g_top', 'ortho_top', 'beta', 'floor']):
        draw.text((xpos, y), text, font=bold, fill='#222222')
    for i, (name, row) in enumerate(zip(('Run 1 min', 'Joint min', 'Run 3 min'), chosen)):
        source = 'new' if row['source'] == 'continuation' else 'seed'
        fields = [f'{name} ({source} a{row["attempt"]:04d})'] + [f'{row["parameters"][p]:.4g}' for p in PARAMS]
        for xpos, text in zip(x, fields):
            draw.text((xpos, y + 60 * (i + 1)), text, font=regular, fill='#222222')
    OUT.mkdir(parents=True, exist_ok=True)
    output.save(targets[0])
    with targets[1].open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source', 'attempt', 'wave', 'valid', 'run1_rms_mm', 'run3_rms_mm',
                         'joint_rms_mm', 'observed_frontier', *PARAMS])
        for r in seed + new:
            writer.writerow([r['source'], f'a{r["attempt"]:04d}', r['wave'], r['valid'],
                             r.get('run1', ''), r.get('run3', ''), r.get('joint', ''),
                             r['pooled_front'], *(r['parameters'][p] for p in PARAMS)])
    digest = hashlib.sha256()
    for path in sorted([*seed_paths, *paths]):
        digest.update(f'{path.relative_to(ROOT)} {sha256(path)}\n'.encode())
    targets[2].write_text(
        f'# Interim Run 1–Run 3 Pareto snapshot\n\n'
        f'Controller status at snapshot: {summary["status"]}. The plot includes {len(seed_valid)} valid seed '
        f'evaluations and {summary["new_valid"]} valid continuation evaluations, with '
        f'{len(actual)} pooled nondominated points. Seed failures: {len(seed)-len(seed_valid)}; '
        f'continuation failures: {sum(failures.values())}; failures have no plot coordinates. '
        f'The seed was optimized for a scalar joint objective; continuation used multiobjective acquisition. '
        f'LHS/BO markers follow initial seed design vs all subsequent adaptive evaluations. '
        f'The detail panel has fixed limits Run 1 [0.60,0.70] and Run 3 [0.15,0.32] mm. '
        f'The table marks the minimum Run 1, minimum equal-weight joint RMS and minimum Run 3 observed points; '
        f'values are rounded to four significant digits; CSV retains exact values. '
        f'Source-qualified IDs distinguish seed from continuation attempts. '
        f'All points are adaptive calibration results, not independent experiments or a global frontier. '
        f'The continuation is still running; floor range is [0,0.5]. No solver or Run 2 score was used.\n\n'
        f'Captured summary SHA-256 `{hashlib.sha256(summary_bytes).hexdigest()}`; '
        f'aggregate of sorted result path and SHA-256 pairs `{digest.hexdigest()}`; '
        f'generating script SHA-256 `{sha256(Path(__file__))}`.\n')
    print(f'{len(seed_valid)} seed + {summary["new_valid"]} new valid; {len(actual)} on frontier; {sum(failures.values())} new failures')
    for name, r in zip(('Run 1 min', 'Joint min', 'Run 3 min'), chosen):
        print(name, r['source'], f'a{r["attempt"]:04d}', r['run1'], r['run3'], {p: r['parameters'][p] for p in PARAMS})
    print(targets[0])


if __name__ == '__main__':
    main()
