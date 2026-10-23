#!/usr/bin/env python3
"""Add matched final-only cadence contrast; retain both reviewed contrasts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_never_clamped import accepted, compare
from analyze_two_gripper_release import ROOT, align, digest


def add_final_only(metrics, arrays, control, per_cycle):
    inputs = metrics['input_sha256']

    def load(folder, name):
        path = folder / name
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    execution = load(control, 'execution.json')
    old_execution = load(per_cycle, 'execution.json')
    envelope = load(control, 'never_clamped_final_checkpoint.json')
    state = envelope['payload']
    previous = load(per_cycle, 'never_clamped_final_checkpoint.json')['payload']
    result = load(control, 'never_clamped_result.json')['payload']
    cycles = previous['cycle']
    if not (execution['status'] == 'completed' and execution['resume'] is None and
            result['complete'] and result['cycles'] == state['cycle'] == cycles and
            result['solve_mode'] == 'final_only' and result['equilibrium_solves'] == 1 and
            result['resumed_cycle'] == 0 and result['fixed_vertex_dofs'] == 0 and
            result['remaining_rigid_modes'] == 6 and result['proof'] == state['proof'] and
            result['checkpoint_checksum'] == envelope['checksum'] and accepted(state['proof'])):
        raise ValueError('Invalid final-only completion, cadence, constraints or proof')
    if any(np.asarray(state['identity'][key], bool).any() for key in ('vertex_mask', 'edge_mask')):
        raise ValueError('Physical constraints in final-only state')
    matches = {key: state['identity'][key] == previous['identity'][key]
               for key in ('rest', 'faces', 'edges', 'uv', 'bbar')}
    matches.update({key: state[key] == previous[key] for key in ('top', 'bottom', 'counts')})
    if not all(matches.values()):
        raise ValueError('Final-only reference/target/history mismatch: ' + str(matches))
    a, b = execution['identity'], old_execution['identity']
    if a['sequence'] != b['sequence']:
        raise ValueError('Final-only recipe differs from per-cycle recipe')
    opts_a, opts_b = dict(a['options']), dict(b['options'])
    if not (opts_a.pop('-sequence_solve_mode') == 'final_only' and
            opts_b.pop('-sequence_solve_mode') == 'every_cycle' and
            opts_a.pop('-sequence_adaptive') == 'false' and
            opts_b.pop('-sequence_adaptive') == 'true' and opts_a == opts_b):
        raise ValueError('Unexpected non-cadence solver option difference')
    initial = 'bilayer_zigzag_sequence_cycle_000_initial.vtp'
    for folder in (control, per_cycle):
        inputs[str(folder / initial)] = digest(folder / initial)
    if (control / initial).read_bytes() != (per_cycle / initial).read_bytes():
        raise ValueError('Initial exported reference states differ')
    files = sorted(control.glob('accepted_cycle_*.json'))
    if files != [control / ('accepted_cycle_%03d.json' % cycles)]:
        raise ValueError('Unexpected intermediate accepted checkpoint')
    if load(control, files[0].name) != envelope:
        raise ValueError('Final accepted checkpoint differs from endpoint')
    for filename, column, expected in [('sequence_history.csv', 'solved', ['0'] * (cycles - 1) + ['1']),
                                       ('sequence_convergence.csv', 'minimization_performed', ['1'])]:
        path = control / filename
        inputs[str(path)] = digest(path)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if [row[column] for row in rows] != expected:
            raise ValueError('Unexpected equilibrium history: ' + filename)
    if not (int(rows[0]['executed_cycle_index']) == cycles and float(rows[0]['lambda_trial']) == 1
            and rows[0]['equilibrium_accepted'] == '1'):
        raise ValueError('Final-only solve is not accepted at full target')
    dofs = np.asarray(state['state'], float).reshape(-1)
    if dofs.size != np.asarray(previous['state']).size or not np.isfinite(dofs).all():
        raise ValueError('Invalid final-only state')
    nv = metrics['vertices']
    moving = dofs[:3 * nv].reshape(3, nv).T * 1000
    aligned, rotation, translation = align(moving, arrays['never_clamped_mm'], arrays['weights'])
    delta = aligned - arrays['never_clamped_mm']
    distances = np.linalg.norm(delta, axis=1)
    metrics['cadence'] = {
        'contrast': 'final-only minus never-clamped per-cycle; proper material-correspondence alignment',
        'reference_and_target_history_exact_matches': matches,
        'area_weighted_rms_difference_mm': float(np.sqrt(arrays['weights'] @ distances ** 2)),
        'vertex_rms_difference_mm': float(np.sqrt(np.mean(distances ** 2))),
        'maximum_difference_mm': float(distances.max()),
        'vertex_quantiles_mm': dict(zip(('q50', 'q90', 'q99'), np.quantile(distances, [.5, .9, .99]).tolist())),
        'rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
        'energy_final_only_minus_per_cycle_normalized': state['proof']['energy'] - previous['proof']['energy'],
        'final_only_proof': state['proof'],
        'final_only_remaining_rigid_modes': result['remaining_rigid_modes'],
        'final_only_fixed_dofs': result['fixed_vertex_dofs'],
        'scope': 'Descriptive deterministic-case contrast. No shape-error bound, global uniqueness or population replication. E=1.',
    }
    arrays.update(final_only_mm=moving, final_only_aligned_mm=aligned,
                  cadence_delta_mm=delta, cadence_distance_mm=distances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'run/two_gripper_comparison')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in ('run6', 'run9'):
        per_cycle = args.root / (name + '_never_clamped')
        metrics, arrays = compare(per_cycle, args.root / name, name)
        add_final_only(metrics, arrays, args.root / (name + '_final_only'), per_cycle)
        cases.append(metrics)
        np.savez_compressed(args.output / (name + '_full_field.npz'), **arrays)
    scripts = {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('analyze_never_clamped.py'),
                                            Path(__file__).with_name('analyze_two_gripper_release.py')]}
    (args.output / 'four_condition_metrics.json').write_text(json.dumps(
        dict(cases=cases, analysis_sha256=scripts, numpy_version=np.__version__,
             scan_scope='Existing frozen-contract scores retained in prior report; no new scan scores here.'),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
