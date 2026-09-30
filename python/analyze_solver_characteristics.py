#!/usr/bin/env python3
"""Separate numerical characteristics; never infer physical accuracy from solver success."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from analyze_sequence_ablation_v2 import load_vtp, final_vtp, metric_vector, proper_kabsch, quadratic_fit, point_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('run/solver_characteristics'))
    args = parser.parse_args()
    records, loaded = [], {}
    for marker in sorted(args.root.glob('*/*/result.json')):
        directory = marker.parent
        result = json.loads(marker.read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        row = {'family': directory.parent.name, 'case': directory.name,
               'return_code': result.get('return_code'),
               'wall_seconds': result.get('wall_seconds'),
               'physical_accuracy_validated': False}
        row.update({key: manifest.get(key) for key in
                    ('endpoint', 'compare_with', 'group', 'role', 'omp_num_threads')})
        row['status'] = result.get('status')
        if result.get('return_code') == 0:
            data, points, triangles = load_vtp(final_vtp(directory, result))
            material = np.column_stack((point_array(data, 'material_u'),
                                        point_array(data, 'material_v')))
            flat = np.column_stack((material, np.zeros(len(material))))
            aligned, _, _ = proper_kabsch(flat, points)
            row.update(quadratic_fit(points))
            row['aligned_height_span_m'] = float(np.ptp(aligned[:, 2]))
            with (directory / 'sequence_convergence.csv').open() as stream:
                attempts = list(csv.DictReader(stream))
            physical = [r for r in attempts if r.get('state_kind', 'physical') == 'physical']
            endpoints = [r for r in physical if r['equilibrium_accepted'] == '1'
                         and float(r['lambda_trial']) == 1.0]
            row['accepted_endpoint_count'] = len(endpoints)
            row['rejected_attempt_count'] = sum(r['equilibrium_accepted'] != '1' for r in physical)
            row['max_endpoint_gradient_norm'] = max(
                (float(r['final_gradient_norm']) for r in endpoints), default=None)
            loaded[(row['family'], row['case'])] = (points, triangles, material, metric_vector(data))
        records.append(row)
    comparisons = []
    for left, right in itertools.combinations(records, 2):
        if (right['case'] not in (left['compare_with'] or [])
                and left['case'] not in (right['compare_with'] or [])):
            continue
        key_l, key_r = (left['family'], left['case']), (right['family'], right['case'])
        if key_l not in loaded or key_r not in loaded:
            continue
        x, tri, uv, metric = loaded[key_l]
        y, tri_r, uv_r, metric_r = loaded[key_r]
        if x.shape != y.shape or not np.array_equal(tri, tri_r) or not np.array_equal(uv, uv_r):
            comparisons.append({'family': left['family'], 'left': left['case'],
                                'right': right['case'], 'comparable': False,
                                'reason': 'Different material discretization; no nodewise comparison.'})
            continue
        _, rms, maximum = proper_kabsch(x, y)
        metric_error = float(np.max(np.abs(metric - metric_r)))
        metric_scale = max(float(np.max(np.abs(metric))), float(np.max(np.abs(metric_r))), np.finfo(float).tiny)
        expected_identical = any(
            (owner['endpoint'] or {}).get('reference_case') == other['case']
            and (owner['endpoint'] or {}).get('expected_identical_abar') is True
            for owner, other in ((left, right), (right, left)))
        comparisons.append({'family': left['family'], 'left': left['case'],
                            'right': right['case'], 'comparable': True,
                            'aligned_vector_rms_m': rms,
                            'aligned_vector_max_m': maximum,
                            'raw_vector_rms_m': float(np.sqrt(np.mean(np.sum((x-y)**2, axis=1)))),
                            'expected_identical_metric': expected_identical,
                            'metric_matches': metric_error <= 1e-12 * metric_scale,
                            'metric_max_absolute': metric_error,
                            'metric_max_relative': metric_error / metric_scale,
                            'metric_bitwise_equal': bool(np.array_equal(metric, metric_r)),
                            'interpretation': 'Different metrics are different loads; compare shape differences with repeated-run floor before branch claims.'})
    output = {'cases': records, 'within_family_comparisons': comparisons,
              'physical_accuracy_validated': False,
              'limitations': ['Residual acceptance is not physical validation.',
                              'Initial-state perturbations are not optimizer stochasticity.',
                              'Geometry alone does not identify residual stress or plastic internal state.']}
    destination = args.root / 'characteristics_analysis.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
