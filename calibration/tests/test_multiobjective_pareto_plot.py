"""No-solver tests of source-qualified, pooled Pareto reporting."""
import json
import math

import matplotlib.pyplot as plt
import pytest

from calibration.analysis import plot_multiobjective_joint4 as plot


def record(attempt, run1=None, run3=None):
    valid = run1 is not None and run3 is not None
    row = {'attempt': attempt, 'wave': 1, 'valid': valid,
           'parameters': dict(gtop=.001, ortho_top=0, beta=.1, floor=.2)}
    if valid:
        row['case_rms_mm'] = {'run6': run1, 'run10': run3}
        row['objective_mm'] = math.sqrt((run1**2 + run3**2) / 2)
    return row


def test_initial_and_pooled_frontier_with_failure_and_overlapping_ids():
    seed = plot.observations([record(1, 1, 3), record(2, 2, 2),
                              record(3, 3, 1), record(4, 4, 4)], 'seed')
    new = plot.observations([record(1, 1.5, 1.5), record(2)], 'continuation')
    pooled = plot.mark_frontiers(seed, new)
    assert len(pooled) == 5
    assert {(r['source'], r['attempt']) for r in seed if r['initial_front']} == {
        ('seed', 1), ('seed', 2), ('seed', 3)}
    assert {(r['source'], r['attempt']) for r in pooled if r['pooled_front']} == {
        ('seed', 1), ('seed', 3), ('continuation', 1)}
    assert not new[1]['valid'] and 'run1' not in new[1] and not new[1]['pooled_front']
    fig = plot.figure(seed, new)
    plot.check_annotation_layout(fig)
    assert len(fig.axes) == 2
    x, y = fig.axes[1].get_xlim(), fig.axes[1].get_ylim()
    assert all(x[0] <= r['run1'] <= x[1] and y[0] <= r['run3'] <= y[1]
               for r in pooled if r['pooled_front'])
    plt.close(fig)


def test_invalid_pairs_and_duplicate_ids_fail_closed():
    with pytest.raises(ValueError, match='Duplicate'):
        plot.observations([record(1, 1, 1), record(1, 2, 2)], 'seed')
    bad = record(1, 1, 1)
    bad['case_rms_mm']['run6'] = 0
    with pytest.raises(ValueError, match='Invalid discrepancy'):
        plot.observations([bad], 'seed')
    bad = record(1, 1, 1)
    bad['objective_mm'] = 2
    with pytest.raises(ValueError, match='Inconsistent joint'):
        plot.observations([bad], 'seed')


def test_load_checks_frozen_inputs_and_recomputes_the_summary_frontier(tmp_path, monkeypatch):
    campaign = tmp_path / 'run/multiobjective_joint4'
    result_dir = campaign / 'results'
    result_dir.mkdir(parents=True)
    fresh_summary = tmp_path / 'run/fresh_lhs_joint4/summary.json'
    fresh_summary.parent.mkdir(parents=True)
    fresh_summary.write_text('seed summary')
    config = tmp_path / 'fresh.json'
    config.write_text('seed config')
    seed_path = tmp_path / 'seed.json'
    seed_path.write_text('seed')
    seed_raw = [record(1, 1, 3), record(2, 2, 2), record(3, 3, 1)]
    (result_dir / 'a0001.json').write_text(json.dumps(record(1, 1.5, 1.5)))
    plan = {'files': {str(p.absolute()): plot.sha256(p) for p in (seed_path, fresh_summary, config)}}
    (campaign / 'plan.json').write_text(json.dumps(plan))
    summary = {'status': 'pareto_plateau', 'seed_evaluations': 3, 'seed_valid': 3,
               'new_evaluations': 1, 'new_valid': 1,
               'new_failures': {'solver': 0, 'registration': 0, 'timeout': 0, 'infrastructure': 0},
               'frontier': [{'source': 'seed', 'attempt': 1, 'run1_mm': 1, 'run3_mm': 3},
                            {'source': 'seed', 'attempt': 3, 'run1_mm': 3, 'run3_mm': 1},
                            {'source': 'continuation', 'attempt': 1, 'run1_mm': 1.5, 'run3_mm': 1.5}]}
    (campaign / 'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(plot, 'ROOT', tmp_path)
    monkeypatch.setattr(plot, 'CAMPAIGN', campaign)
    monkeypatch.setattr(plot, 'CONFIG', config)
    monkeypatch.setattr(plot, 'load_results', lambda: (seed_raw, seed_raw, seed_raw[0], [seed_path]))
    seed, new, _, _ = plot.load_campaign()
    assert sum(r['initial_front'] for r in seed) == 3
    assert sum(r['pooled_front'] for r in seed + new) == 3
    seed_path.write_text('changed')
    with pytest.raises(ValueError, match='differs from the frozen plan'):
        plot.load_campaign()
    seed_path.write_text('seed')
    summary['frontier'][0]['run1_mm'] = 9
    (campaign / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='differs from the terminal summary'):
        plot.load_campaign()


def test_live_or_stale_running_summary_is_not_presented_as_completed(tmp_path, monkeypatch):
    (tmp_path / 'summary.json').write_text(json.dumps({'status': 'running'}))
    monkeypatch.setattr(plot, 'CAMPAIGN', tmp_path)
    with pytest.raises(ValueError, match='no terminal controller summary'):
        plot.load_campaign()
