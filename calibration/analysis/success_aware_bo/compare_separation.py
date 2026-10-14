"""§17: exactly three no-refit proposal cells, immutable baseline and paired MC."""
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from calibration.analysis.success_aware_bo.diagnose import (
    ROOT, OUT as PRIOR, evaluate, restore_models, sha, verify)
from calibration.analysis.success_aware_bo.separation import admissible, propose

OUT = PRIOR / 'separation'
PACKET = OUT / 'review_packet.json'


def write_new(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def check_packet():
    packet = json.loads(PACKET.read_text())
    for name, expected in packet['artifact_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Reviewed or preserved artifact changed: {name}')
    return packet


def run():
    # These markers protect even failed/partial runs from accidental replay.
    if (OUT/'started.json').exists() or list(OUT.glob('cell_*.json')):
        raise FileExistsError('Completed/partial run exists; preserve and consult before replay')
    packet = check_packet()
    snapshot, protected = verify()
    clearance = OUT/'preexecution_review_astra_focused_v3.md'
    binding = json.loads((OUT/'launch_binding.json').read_text())
    if (binding['packet_sha256'] != sha(PACKET) or binding['clearance_sha256'] != sha(clearance)
            or binding['verdict'] not in ('APPROVED', 'APPROVED_WITH_CAVEATS')):
        raise ValueError('Independent clearance/launch binding missing or mismatched')
    started = time.monotonic()
    fixed_hash = sha(PRIOR/'fixed_models_joint.pt')
    write_new('started.json', {'packet_sha256': sha(PACKET), 'fixed_model_sha256': fixed_hash,
                              'job_id': os.environ.get('SLURM_JOB_ID'), 'platform': platform.platform(),
                              'torch_version': torch.__version__, 'threads': torch.get_num_threads(),
                              'protected_sources_checked': protected})
    loss, success, best, fit_info = restore_models(snapshot)
    old = [json.loads((PRIOR/f'cell_10_{seed}.json').read_text()) for seed in (3101, 3102, 3103)]
    cells = []
    history = np.asarray([r['normalized_parameters'] for r in snapshot['observations']])
    for seed in (3101, 3102, 3103):
        tick = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            cell = propose(snapshot['observations'], loss, success, best, seed=seed)
        batch = np.asarray(cell['final'])
        if not admissible(batch, history) or cell['repaired_indices']:
            raise ValueError('Final batch failed strict unchanged policy')
        historical = np.linalg.norm(batch[:, None]-history, axis=-1)
        pairwise = np.linalg.norm(batch[:, None]-batch[None], axis=-1)
        cell.update(seconds=time.monotonic()-tick,
                    warnings=cell['warnings'] + [str(w.message) for w in captured],
                    min_historical_distance=float(historical.min()),
                    min_pairwise_distance=float(pairwise[np.triu_indices(4, 1)].min()),
                    fixed_model_sha256=fixed_hash)
        write_new(f'cell_{seed}.json', cell)
        cells.append(cell)
        print('CELL', seed, cell['seconds'], cell['selection'], cell['optimizer_log_acquisition'],
              len(cell['warnings']), flush=True)
    batches = [batch for base, cell in zip(old, cells)
               for batch in (base['raw'], base['final'], cell['final'])]
    labels = [f'seed{seed}_{stage}' for seed in (3101, 3102, 3103)
              for stage in ('baseline_raw_inadmissible', 'baseline_random_repaired', 'separation_aware')]
    reference = evaluate(loss, success, best, batches, 8192, range(9201, 9205))
    write_new('common_evaluation.json', reference)
    values = np.asarray(reference['utility_by_seed'])
    contrasts = []
    for i, seed in enumerate((3101, 3102, 3103)):
        baseline, separated = values[:, 3*i+1], values[:, 3*i+2]
        difference = separated-baseline
        mean, sd = float(difference.mean()), float(difference.std(ddof=1))
        contrasts.append({'seed': seed, 'paired_mean_difference': mean, 'paired_scramble_sd': sd,
                          'positive_above_twice_spread': mean > 0 and mean > 2*sd,
                          'relative_uplift': float(separated.mean()/baseline.mean()-1)
                          if baseline.mean() > 1e-12 else None})
    check_packet()
    verify()
    if sha(PRIOR/'fixed_models_joint.pt') != fixed_hash:
        raise ValueError('Frozen checkpoint changed')
    write_new('diagnostic.json', {
        'packet_sha256': sha(PACKET), 'snapshot_sha256': sha(PRIOR/'snapshot.json'),
        'fixed_model_sha256': fixed_hash, 'fit_info': fit_info, 'cells': cells,
        'baseline_cells': old, 'labels': labels, 'common_evaluation': reference,
        'common_mean': values.mean(axis=0).tolist(),
        'common_scramble_sd': values.std(axis=0, ddof=1).tolist(), 'contrasts': contrasts,
        'favorable_numerical_signal': sum(c['positive_above_twice_spread'] for c in contrasts) >= 2,
        'seconds': time.monotonic()-started, 'protected_sources_checked': protected,
        'preserved_artifacts_checked': len(packet['artifact_sha256']),
        'limits': 'Approximate model-conditional numerical signal, not physical fit or production adoption. '
                  'Joint method comparison, not equal-work/pure optimizer ablation. All cells reported.'})
    print('RESULT', contrasts, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run()
