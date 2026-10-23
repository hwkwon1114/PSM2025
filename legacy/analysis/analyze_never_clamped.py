#!/usr/bin/env python3
"""Matched forming-restraint contrast; descriptive spatial distributions only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from analyze_two_gripper_release import BASELINES, ROOT, align, analyze, digest


def accepted(proof):
    return (proof['accepted'] and proof['termination'] == 'gradient_target'
            and proof['curvature'] == 'positive_pivots'
            and np.isfinite(proof['energy'])
            and 0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= 5e-14)


def compare(control, baseline, name, score_scans=False):
    release, arrays = analyze(baseline, name, score_scans)
    inputs = dict(release['input_sha256'])

    def load(folder, filename):
        path = folder / filename
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    execution = load(control, 'execution.json')
    if execution['status'] != 'completed':
        raise ValueError('Control execution incomplete')
    envelope = load(control, 'never_clamped_final_checkpoint.json')
    final = envelope['payload']
    result = load(control, 'never_clamped_result.json')['payload']
    parent = load(baseline, 'paired_clamped_checkpoint.json')['payload']
    if not (result['complete'] and result['checkpoint_checksum'] == envelope['checksum']
            and result['cycles'] == final['cycle'] == parent['cycle'] == release['cycles']
            and result['fixed_vertex_dofs'] == 0 and result['remaining_rigid_modes'] == 6
            and result['proof'] == final['proof'] and accepted(final['proof'])):
        raise ValueError('Invalid final control proof/link/constraints')
    matches = {}
    for key in ('rest', 'faces', 'edges', 'uv', 'bbar'):
        matches[key] = final['identity'][key] == parent['identity'][key]
    for key in ('top', 'bottom', 'counts'):
        matches[key] = final[key] == parent[key]
    if not all(matches.values()):
        raise ValueError('Unmatched reference or accumulated target/history: ' + str(matches))
    # This analysis handles fresh runs only; any resumption needs the complete
    # predecessor chain, not a silently truncated per-repeat audit.
    if execution['resume']:
        raise ValueError('Resumed production requires explicit predecessor-chain audit')
    cycles = []
    for cycle in range(1, final['cycle'] + 1):
        checkpoint = load(control, 'accepted_cycle_%03d.json' % cycle)
        state = checkpoint['payload']
        if state['cycle'] != cycle or not accepted(state['proof']):
            raise ValueError('Unaccepted or misordered control repeat')
        if state['identity'] != final['identity']:
            raise ValueError('Control identity changed between repeats')
        if any(np.asarray(state['identity'][key], bool).any()
               for key in ('vertex_mask', 'edge_mask')):
            raise ValueError('Physical fixed coordinates in control')
        if cycle == final['cycle'] and checkpoint != envelope:
            raise ValueError('Final accepted checkpoint differs from endpoint')
        cycles.append({'cycle': cycle, 'proof': state['proof']})
    nv = release['vertices']
    state = np.asarray(final['state'], float).reshape(-1)
    if state.size != np.asarray(parent['state']).size or not np.isfinite(state).all():
        raise ValueError('Invalid full endpoint DOFs')
    moving = state[:3 * nv].reshape(3, nv).T * 1000
    aligned, rotation, translation = align(moving, arrays['free_mm'], arrays['weights'])
    delta = aligned - arrays['free_mm']
    distances = np.linalg.norm(delta, axis=1)
    raw_gradient = np.asarray(result['unmasked_gradient'], float)
    if not np.isfinite(raw_gradient).all() or not np.isfinite(result['unmasked_energy']):
        raise ValueError('Nonfinite control diagnostics')
    metrics = {
        'case': name, 'cycles': final['cycle'], 'vertices': nv,
        'reference_and_target_history_exact_matches': matches,
        'area_weighted_rms_difference_mm': float(np.sqrt(arrays['weights'] @ distances ** 2)),
        'vertex_rms_difference_mm': float(np.sqrt(np.mean(distances ** 2))),
        'maximum_difference_mm': float(distances.max()),
        'vertex_quantiles_mm': dict(zip(('q50', 'q90', 'q99'), np.quantile(distances, [.5, .9, .99]).tolist())),
        'never_clamped_to_released_rotation': rotation.tolist(),
        'translation_mm': translation.tolist(),
        'energy_never_clamped_minus_released_normalized': final['proof']['energy'] - release['diagnostics']['free']['energy_normalized'],
        'never_clamped_proof': final['proof'],
        'never_clamped_unmasked_gradient_norm': float(np.linalg.norm(raw_gradient)),
        'never_clamped_fixed_dofs': result['fixed_vertex_dofs'],
        'never_clamped_remaining_rigid_modes': result['remaining_rigid_modes'],
        'accepted_repeats': cycles,
        'paired_release': release,
        'input_sha256': inputs,
        'scope': 'Two deterministic cases, no vertex-independent inference. E=1 normalized energies; positive pivots indicate local stability only. No scan comparison in this artifact.',
    }
    if score_scans:
        sys.path.insert(0, str(ROOT))
        from calibration.ewcal.geometry import read_mesh, points
        from calibration.ewcal.material_frame import score_material_frame
        from vtk.util.numpy_support import vtk_to_numpy
        # analyze() above has already checked the frozen scoring source hashes.
        target_path = ROOT / 'run' / BASELINES[name] / 'evaluations/attempt_0209/case_00/comparison.npz'
        with np.load(target_path) as data:
            target = data['target_mm'].copy()
        mesh_path = control / result['final_vtp']
        mesh = read_mesh(mesh_path, scale=1000)
        mesh_points = points(mesh)
        vtk_faces = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)
        if not (np.all(vtk_faces[:, 0] == 3) and np.array_equal(vtk_faces[:, 1:], arrays['faces'])):
            raise ValueError('Control VTP topology mismatch')
        if not np.allclose(mesh_points, moving, rtol=4*np.finfo(np.float32).eps, atol=1e-9):
            raise ValueError('Control VTP differs beyond serialization precision')
        scan, _, _, _, _ = score_material_frame(mesh, target, arrays['reference_mm'], arrays['faces'])
        metrics['never_clamped_scan_metrics'] = scan
        metrics['never_clamped_scan_serialization_max_difference_mm'] = float(
            np.max(np.linalg.norm(mesh_points - moving, axis=1)))
        inputs[str(mesh_path)] = digest(mesh_path)
        metrics['scope'] = metrics['scope'].replace('No scan comparison in this artifact.',
            'Scan scoring uses the verified frozen material-reference contract; intrinsic contrasts use full checkpoint precision.')
    arrays.update(never_clamped_mm=moving, never_clamped_aligned_mm=aligned,
                  forming_delta_mm=delta, forming_distance_mm=distances)
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'run/two_gripper_comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-scans', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in ('run6', 'run9'):
        metrics, arrays = compare(args.root / (name + '_never_clamped'), args.root / name, name, args.score_scans)
        cases.append(metrics)
        np.savez_compressed(args.output / (name + '_full_field.npz'), **arrays)
    report = {'cases': cases, 'analysis_script_sha256': digest(Path(__file__)),
              'release_analysis_sha256': digest(Path(__file__).with_name('analyze_two_gripper_release.py')),
              'numpy_version': np.__version__}
    (args.output / 'forming_restraint_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
