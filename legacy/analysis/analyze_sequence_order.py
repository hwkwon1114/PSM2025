#!/usr/bin/env python3
"""All-case descriptive sequence/cadence comparisons; no simulation or scan fitting."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dome_bo'))
from run_sequence_order import SCHEDULES, verify
from verify_sequence_continuation import verify as verify_continuation
from analyze_two_gripper_release import align, area_weights, digest
from dome_curvature import geometric_curvature


def quantiles(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return dict(zip(('q50', 'q90', 'q99'),
                    np.interp([.5, .9, .99], cumulative, values[order]).tolist()))


def compare(a, b):
    """a minus b, identical material correspondence; outputs millimetres."""
    for key in ('reference_positions', 'material_uv', 'faces', 'edges', 'bbar', 'material'):
        if a[key] != b[key]:
            raise ValueError('Reference/objective mismatch: ' + key)
    weights = area_weights(np.asarray(b['reference_positions']), np.asarray(b['faces']))
    target = np.asarray(b['positions']) * 1000
    moving, rotation, translation = align(np.asarray(a['positions']) * 1000, target, weights)
    delta = moving - target
    distance = np.linalg.norm(delta, axis=1)
    fields = {}
    for key in ('abar_top', 'abar_bottom', 'counts'):
        left, right = np.asarray(a[key]), np.asarray(b[key])
        if left.shape != right.shape or not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise ValueError('Invalid target/history arrays: ' + key)
        change = left - right
        scale = np.linalg.norm(right)
        fields[key] = {'exact': np.array_equal(left, right),
                       'maximum_absolute': float(abs(change).max()),
                       'rms_absolute': float(np.sqrt(np.mean(change**2))),
                       'relative_l2': float(np.linalg.norm(change) / scale) if scale else None}
    same = all(fields[key]['exact'] for key in fields)
    diagonal = np.linalg.norm(np.ptp(np.asarray(b['reference_positions']), axis=0)) * 1000
    rms = float(np.sqrt(weights @ distance**2))
    result = {'area_weighted_rms_mm': rms, 'relative_rms': rms / diagonal,
              'maximum_mm': float(distance.max()), 'area_weighted_quantiles_mm': quantiles(distance, weights),
              'targets_and_history': fields, 'same_final_objective_and_history': same,
              'normalized_energy_difference': a['proof']['energy'] - b['proof']['energy'] if same else None,
              'rotation': rotation.tolist(), 'translation_mm': translation.tolist(),
              'scope': 'Deterministic material-correspondence contrast; not independent vertex replicates or proof of global minima.'}
    return result, dict(delta_mm=delta, distance_mm=distance, weights=weights, aligned_mm=moving)


def load_case(folder):
    execution = json.loads((folder / 'execution.json').read_text())
    if execution['status'] != 'accepted':
        raise ValueError('Not an accepted execution: ' + str(folder))
    config = json.loads((folder / 'sequence.json').read_text())
    if digest(folder / 'sequence.json') != execution['sequence_sha256']:
        raise ValueError('Changed sequence identity')
    verifier = verify_continuation if execution['options']['-sequence_adaptive'] == 'true' else verify
    result = verifier(folder, config, execution['options'])
    state = json.loads((folder / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(folder / result['final_mesh']))
    reader.Update()
    data = reader.GetOutput()
    cells = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)
    if not (np.all(cells[:, 0] == 3) and np.array_equal(cells[:, 1:], state['faces'])):
        raise ValueError('VTP/native topology mismatch')
    arrays = {}
    for key in ('total_hit_count', 'mean', 'gauss'):
        array = data.GetCellData().GetArray(key)
        if array is None:
            raise ValueError('Missing VTP field ' + key)
        arrays[key] = vtk_to_numpy(array).astype(float)
        if not np.isfinite(arrays[key]).all():
            raise ValueError('Nonfinite VTP field ' + key)
    counts = arrays['total_hit_count']
    if np.any(counts != np.rint(counts)) or np.any(counts < 0):
        raise ValueError('Invalid hit counts')
    state['counts'] = counts.tolist()
    for key in ('positions', 'reference_positions', 'selected_dofs', 'abar_top', 'abar_bottom', 'bbar'):
        if not np.isfinite(np.asarray(state[key])).all():
            raise ValueError('Nonfinite native state: ' + key)
    for key in ('abar_top', 'abar_bottom'):
        forms = np.asarray(state[key])
        if not np.allclose(forms, forms.transpose(0, 2, 1), rtol=1e-12, atol=1e-18) or np.any(np.linalg.eigvalsh(forms) <= 0):
            raise ValueError('Invalid target metric')
    geometric, directions, valid = geometric_curvature(state['positions'], state['material_uv'], state['faces'])
    arrays.update(geometric_principal_m_inverse=geometric, geometric_valid=valid,
                  positions=np.asarray(state['positions']), uv=np.asarray(state['material_uv']),
                  faces=np.asarray(state['faces']), counts=counts)
    vertices = np.asarray(state['reference_positions'])
    triangles = vertices[np.asarray(state['faces'])]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    arrays['face_weights'] = areas / areas.sum()
    report = {'execution': execution, 'proof': state['proof'],
              'input_sha256': {str(p): digest(p) for p in
                               (folder / 'execution.json', folder / 'sequence.json',
                                folder / 'bilayer_zigzag_sequence_native_hybrid_state.json',
                                folder / result['final_mesh'])},
              'geometric_valid_area_fraction': float(arrays['face_weights'][valid].sum()),
              'curvature_limits': 'Geometry fit: frozen 15mm material radius, invalid fits retained. VTP mean/gauss use discrete directors and serialized precision; not interchangeable with geometric fits.'}
    return state, arrays, report


def analyze(root, output, continuation=False):
    output.mkdir(parents=True, exist_ok=False)
    report = {'cases': {}, 'cadence': {}, 'order': {}, 'controls': {}, 'sensitivity': {},
              'analysis_sha256': {str(p): digest(p) for p in
                                 (Path(__file__), Path(__file__).with_name('dome_curvature.py'),
                                  Path(__file__).with_name('analyze_two_gripper_release.py'))}}
    loaded = {}
    for variant in ('base', 'repeat', 'tight', 'fine'):
        for name in (SCHEDULES if variant == 'base' else ('ab_block',)):
            for mode in ('every_cycle', 'final_only'):
                key = f'{name}_{mode}_{variant}'
                folder = root / key
                if continuation and name == 'ab_block' and mode == 'final_only' and variant == 'base':
                    folder = ROOT / 'run/sequence_order/ab_block_final_only_continuation_v1'
                if not (folder / 'execution.json').exists():
                    report['cases'][key] = {'status': 'not_run'}
                    continue
                record = json.loads((folder / 'execution.json').read_text())
                if record['status'] != 'accepted':
                    report['cases'][key] = record
                    continue
                state, arrays, metrics = load_case(folder)
                metrics['folder'] = str(folder.resolve())
                if continuation and metrics['execution']['options']['-sequence_adaptive'] != 'true':
                    raise ValueError('Direct solve in continuation comparison: ' + key)
                loaded[key] = state
                report['cases'][key] = metrics
                np.savez_compressed(output / (key + '.npz'), **arrays)

    def contrast(group, label, left, right, cadence=False):
        if left not in loaded or right not in loaded:
            return
        result, arrays = compare(loaded[left], loaded[right])
        if cadence:
            a = report['cases'][left]['execution']
            b = report['cases'][right]['execution']
            oa, ob = dict(a['options']), dict(b['options'])
            oa.pop('-sequence_solve_mode'); ob.pop('-sequence_solve_mode')
            initial = 'bilayer_zigzag_sequence_cycle_000_initial.vtp'
            if (a['sequence_sha256'] != b['sequence_sha256'] or a['binary_sha256'] != b['binary_sha256'] or
                    oa != ob or digest(Path(report['cases'][left]['folder']) / initial) != digest(Path(report['cases'][right]['folder']) / initial)):
                raise ValueError('Unmatched cadence settings/initial states: ' + label)
            if not result['same_final_objective_and_history']:
                raise ValueError('Cadence pair changed final targets/history: ' + label)
        report[group][label] = dict(left=left, right=right, **result)
        np.savez_compressed(output / (group + '_' + label + '.npz'), **arrays)

    for name in SCHEDULES:
        contrast('cadence', name, name+'_final_only_base', name+'_every_cycle_base', cadence=True)
    mixed = [name for name in SCHEDULES if not name.endswith('_only')]
    for mode in ('every_cycle', 'final_only'):
        for a, b in itertools.combinations(mixed, 2):
            contrast('order', f'{a}_vs_{b}_{mode}', f'{a}_{mode}_base', f'{b}_{mode}_base')
        contrast('controls', 'a_vs_b_' + mode, f'a_only_{mode}_base', f'b_only_{mode}_base')
        for variant in ('repeat', 'tight'):
            contrast('sensitivity', mode+'_'+variant, f'ab_block_{mode}_{variant}', f'ab_block_{mode}_base')
    for variant in ('repeat', 'tight', 'fine'):
        contrast('sensitivity', 'cadence_'+variant,
                 f'ab_block_final_only_{variant}', f'ab_block_every_cycle_{variant}', cadence=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--continuation', action='store_true', help='Explicitly reuse accepted ab_block combined continuation baseline')
    args = parser.parse_args()
    analyze(args.root, args.output, args.continuation)
