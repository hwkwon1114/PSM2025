"""Stage-wise acceptance for this investigation's continuation comparisons."""
import csv
import json
from run_sequence_order import accepted


def verify(out, config, opts):
    cycles = len(config['toolpaths'])
    mode = opts['-sequence_solve_mode']
    expected = list(range(1, cycles + 1)) if mode == 'every_cycle' else [cycles]
    tolerance = float(opts['-native_hybrid_grad_tol'])
    result = json.loads((out / 'sequence_result.json').read_text())
    solves = result['solves']
    if not (result['completed'] is True and result['cycles'] == cycles and
            result['solve_mode'] == mode and result['equilibrium_solves'] == len(expected) and
            [s['cycle'] for s in solves] == expected):
        raise ValueError('Incomplete physical-stage sequence')
    proofs = [json.loads(line) for line in
              (out / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl').read_text().splitlines() if line.strip()]
    with (out / 'sequence_convergence.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != len(proofs):
        raise ValueError('Missing continuation proof rows')
    stage, load = 0, 0.0
    for row, proof in zip(rows, proofs):
        if stage >= len(expected) or int(row['executed_cycle_index']) != expected[stage]:
            raise ValueError('Wrong continuation stage order')
        trial = float(row['lambda_trial'])
        if (float(row['lambda_from']) != load or not load < trial <= 1 or
                row['equilibrium_accepted'] not in ('0', '1') or
                (row['equilibrium_accepted'] == '1') != proof['accepted']):
            raise ValueError('Invalid rollback/load proof')
        if proof['accepted']:
            if not accepted(proof, tolerance):
                raise ValueError('Unqualified accepted substep')
            load = trial
            if load == 1:
                if solves[stage]['native_hybrid'] != proof:
                    raise ValueError('Stage endpoint proof mismatch')
                stage, load = stage + 1, 0.0
    if stage != len(expected) or load != 0:
        raise ValueError('Partial-load endpoint')
    with (out / 'sequence_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    if ([int(r['cycle']) for r in history] != list(range(1, cycles + 1)) or
            [int(r['solved']) for r in history] != [int(i in expected) for i in range(1, cycles + 1)] or
            any(int(r['hit_events']) <= 0 for r in history)):
        raise ValueError('Wrong application history')
    with (out / 'bilayer_zigzag_sequence_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    if ([int(r['executed_cycle_index']) for r in summary] != expected or
            [r['toolpath_id'] for r in summary] != [config['toolpaths'][i-1]['id'] for i in expected] or
            any(not (out / r['final_file']).is_file() for r in summary)):
        raise ValueError('Missing or mismatched intermediate shapes')
    state = json.loads((out / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    if state['proof'] != solves[-1]['native_hybrid'] or not (out / result['final_mesh']).is_file():
        raise ValueError('Final artifacts disagree')
    return result
