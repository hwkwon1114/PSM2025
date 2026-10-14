"""Stdlib-only descriptive analysis of the completed, predeclared diagnostic.

All batches/contrasts are retained. Scramble SD is conditional numerical spread,
not an interval over independent datasets or a predictive calibration claim.
"""
import hashlib
import json
import math
import statistics as s
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'run/optimization_review/success_aware_bo'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d = json.loads((OUT/'diagnostic.json').read_text())
snapshot = json.loads((OUT/'snapshot.json').read_text())
assert d['snapshot_sha256'] == sha(OUT/'snapshot.json')
assert d['fixed_model_sha256'] == sha(OUT/'fixed_models_joint.pt')
assert len(d['cells']) == 6 and len(d['labels']) == 12
assert {(c['restarts'], c['seed']) for c in d['cells']} == {(r, seed) for r in (10, 30) for seed in (3101, 3102, 3103)}
assert all(c['acquisition_seed'] == 2026 and c['posterior_samples'] == 1024 for c in d['cells'])
means, spread = d['common_mean'], d['common_scramble_sd']
prior = [r['normalized_parameters'] for r in snapshot['observations']]
distance = lambda a, b: math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
rows = []
for i, cell in enumerate(d['cells']):
    raw, final = cell['raw'], cell['final']
    categories = []
    for j, candidate in enumerate(raw):
        old_distance = min(distance(candidate, previous) for previous in prior)
        previous_selected_distance = min((distance(candidate, raw[k]) for k in range(j)), default=None)
        categories.append({'candidate': j, 'minimum_historical_distance': old_distance,
                           'minimum_previous_raw_distance': previous_selected_distance,
                           'near_history': old_distance < .08,
                           'near_previous_raw': previous_selected_distance is not None and previous_selected_distance < .08})
    minimum_final_history = min(distance(x, y) for x in final for y in prior)
    minimum_final_pair = min(distance(final[j], final[k]) for j in range(4) for k in range(j))
    assert minimum_final_history >= .08 and minimum_final_pair >= .08
    probability = {}
    for offset, stage in ((0, 'raw'), (1, 'final')):
        index = 2*i+offset
        probability[stage] = {key: d['completion_probability'][key][index]
                              for key in ('mean', 'p05', 'p95')}
    rows.append({'restarts': cell['restarts'], 'initializations': cell['initializations'], 'seed': cell['seed'],
                 'seconds': cell['seconds'], 'warnings': cell['warnings'], 'repair_indices': cell['repaired_indices'],
                 'raw_mean': means[2*i], 'raw_scramble_sd': spread[2*i],
                 'final_mean': means[2*i+1], 'final_scramble_sd': spread[2*i+1],
                 'relative_repair_change': means[2*i+1]/means[2*i]-1,
                 'raw_separation_diagnostics': categories,
                 'minimum_final_history_distance': minimum_final_history,
                 'minimum_final_pair_distance': minimum_final_pair,
                 'completion_probability': probability})
mc = []
for run in d['mc_evaluations']+[d['common_evaluation']]:
    columns = list(zip(*run['utility_by_seed']))
    average = [s.mean(column) for column in columns]
    sd = [s.stdev(column) for column in columns]
    raw = list(range(0, 12, 2)); final = list(range(1, 12, 2))
    mc.append({'samples': run['samples'], 'seeds': run['seeds'], 'mean': average, 'scramble_sd': sd,
               'relative_error_vs_reference': [a/b-1 for a, b in zip(average, means)],
               'raw_max_absolute_relative_mean_difference': max(abs(average[j]/means[j]-1) for j in raw),
               'raw_max_relative_scramble_sd': max(sd[j]/average[j] for j in raw),
               'rankings_raw_by_scramble': [[d['labels'][j] for j in sorted(raw, key=lambda k: utility[k], reverse=True)]
                                            for utility in run['utility_by_seed']],
               'rankings_final_by_scramble': [[d['labels'][j] for j in sorted(final, key=lambda k: utility[k], reverse=True)]
                                              for utility in run['utility_by_seed']]})
contrasts = []
for contrast in d['stronger_minus_current']:
    contrasts.append({**contrast,
                      'relative_rule_and_spread_pass': contrast['relative_uplift'] > .05 and
                      contrast['paired_mean'] > contrast['paired_scramble_sd']})
counts = {stage: sum(c['relative_rule_and_spread_pass'] for c in contrasts if c['stage'] == stage)
          for stage in ('raw', 'final')}
result = {'goal_id': 'mv5rga6i-tfgstn', 'diagnostic_sha256': sha(OUT/'diagnostic.json'),
          'snapshot_sha256': d['snapshot_sha256'], 'fixed_model_sha256': d['fixed_model_sha256'],
          'estimand': 'Conditional expected smoothed best-successful improvement in -log(real RMS) units; not physical RMS.',
          'independence': 'One fixed model/dataset; three optimizer seeds and four independent scramble randomizations. Nested sample-count sequences reuse scrambles.',
          'uncertainty': 'Paired common-scramble contrast SD; descriptive integration spread, not a confidence interval or dataset-level replication.',
          'rows': rows, 'mc_summary': mc, 'contrasts': contrasts,
          'predeclared_relative_rule_pass_counts': counts,
          'mean_current_seconds': s.mean(c['seconds'] for c in d['cells'] if c['restarts'] == 10),
          'mean_stronger_seconds': s.mean(c['seconds'] for c in d['cells'] if c['restarts'] == 30),
          'best_raw_relative_uplift': max(means[6::2])/max(means[:6:2])-1,
          'interpretation': 'Raw numerical criterion fails (one qualifying seed). Final relative-only rule formally passes two seeds, but one contrast is ~5e-20 at the smoothing floor and final contrasts also include random repair; neither justifies a general optimizer improvement claim. Do not amend the criterion post hoc.',
          'recommendation': 'No automatic setting change. Main observed bottleneck is separation-blind optimization followed by random replacement. Propose separately authorized/reviewed admissible acquisition optimization before spending on more restarts or mechanics.',
          'limits': 'No success-aware proposals were physically evaluated; classifier probabilities are uncalibrated model-conditional latent-rate summaries. Six adaptive failures, fixed prior/Laplace and loss-completion independence remain assumptions.'}
with (OUT/'analysis.json').open('x') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
print('ANALYSIS', counts, 'best-raw uplift', result['best_raw_relative_uplift'], flush=True)
for item in mc:
    print('MC', item['samples'], 'max raw relative mean difference', item['raw_max_absolute_relative_mean_difference'],
          'max raw relative scramble SD', item['raw_max_relative_scramble_sd'], flush=True)
