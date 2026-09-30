#!/usr/bin/env python3
"""Diagnose forward geometry diversity; no target fitting or physical validation."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_sequence_ablation_v2 import (load_vtp, point_array, cell_array,
                                         metric_vector, proper_kabsch, areas)
from visualize_solver_characteristics import (
    CYCLE_RE, expand_schedule, load_mesh, place_strips, validate_reconstruction,
    toolpath_figure, sequence_overview_figure)


def render_toolpaths(directory):
    """Reconstruct each executed path and verify it against exported face hits."""
    schedule = expand_schedule(json.loads((directory / 'sequence.json').read_text()))
    mappings = {}
    for path in directory.glob('*_mapping.vtp'):
        match = CYCLE_RE.search(path.name)
        if match:
            mappings[int(match.group(1))] = path
    if set(mappings) != set(range(1, len(schedule) + 1)):
        raise RuntimeError(f'{directory}: mapping exports do not cover the full sequence')
    case = {'tier': directory.parent.name, 'name': directory.name}
    overview, validations, figures = [], [], []
    for index, entry in enumerate(schedule, 1):
        mesh = load_mesh(mappings[index])
        uv = mesh['material']
        bbox = (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())
        hits = cell_array(mesh['data'], 'hits_this_cycle')
        placement = place_strips(entry['operation'], bbox)
        validation = validate_reconstruction(placement, mesh, hits)
        if (not validation['contact_face_set_identical']
                or not validation['hit_count_identical']
                or validation['strip_index_mismatches']
                or validation['max_growth_angle_error_rad'] > 1e-10):
            raise RuntimeError(f'{directory}, cycle {index}: path mismatch: {validation}')
        zero_growth = all(np.all(np.asarray(entry['operation'].get(key, 0)) == 0)
                          for key in ('gtop', 'gbot'))
        validation.update(cycle_index=index, toolpath_id=entry['toolpath_id'],
                          mapping_vtp=mappings[index].name, zero_growth=bool(zero_growth))
        validations.append(validation)
        filename = f'toolpath_cycle_{index:03d}.png'
        toolpath_figure(directory / filename, case, index, entry, placement,
                        mesh, hits, validation, bbox, 140)
        figures.append(filename)
        label = entry['toolpath_id'] + ('\nzero growth; equilibrium probe' if zero_growth else '')
        overview.append(dict(cycle_index=index, toolpath_id=label, placement=placement,
                             mesh=mesh, hits=hits, bbox=bbox))
    filename = 'toolpath_sequence_overview.png'
    sequence_overview_figure(directory / filename, case, overview, 140)
    figures.append(filename)
    return {'figures': figures, 'cycles': validations,
            'coordinates': 'reference material coordinates; not a tracked physical tool',
            'shading': 'geometric face hits, including zero-growth probe hits'}


def analyze(root):
    records, meshes = [], {}
    for marker in sorted(root.glob('*/*/result.json')):
        result = json.loads(marker.read_text())
        directory = marker.parent
        record = {'implementation': directory.parent.name, 'case': directory.name,
                  'return_code': result.get('return_code'), 'status': result.get('status')}
        records.append(record)
        summaries = sorted(directory.glob('*_summary.csv'))
        if summaries:
            with summaries[0].open() as stream:
                summary_rows = list(csv.DictReader(stream))
            if summary_rows and 'total_energy' in summary_rows[-1]:
                record['final_energy_model_units'] = float(summary_rows[-1]['total_energy'])
        filename = result.get('final_vtp')
        if result.get('return_code') != 0 or not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = directory / path
        data, x, tri = load_vtp(path)
        uv = np.column_stack([point_array(data, name) for name in ('material_u', 'material_v')])
        reference = np.column_stack([uv, np.zeros(len(uv))])
        aligned, _, _ = proper_kabsch(reference, x)
        weights = areas(reference, tri)
        weights /= weights.sum()
        gauss, mean = cell_array(data, 'gauss'), cell_array(data, 'mean')
        discriminant = mean**2 - gauss
        # Negative roundoff is clipped only for reconstructing principal values;
        # report its magnitude so invalid curvature estimates cannot hide.
        root_disc = np.sqrt(np.maximum(discriminant, 0))
        k1, k2 = mean + root_disc, mean - root_disc
        record.update(height_span_mm=float(np.ptp(aligned[:, 2])*1000),
                      mean_abs_gaussian_curvature_m2=float(weights @ np.abs(gauss)),
                      minimum_curvature_discriminant=float(discriminant.min()),
                      rms_principal_curvatures_m_inv=[float(np.sqrt(weights @ (k*k))) for k in (k1,k2)])
        record['curvature_area_fractions'] = {}
        for threshold in (1e-4, 1e-3, 1e-2):
            record['curvature_area_fractions'][str(threshold)] = {
                'K_positive': float(weights @ (gauss > threshold)),
                'K_negative': float(weights @ (gauss < -threshold)),
                'K_near_zero': float(weights @ (np.abs(gauss) <= threshold))}
        try:
            metric = metric_vector(data)
        except RuntimeError:
            metric = None
        meshes[(directory.parent.name, directory.name)] = (x, tri, uv, metric)
        record['toolpaths'] = render_toolpaths(directory)
        fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
        axis = fig.add_subplot(131, projection='3d')
        axis.plot_trisurf(aligned[:,0]*1000, aligned[:,1]*1000, aligned[:,2]*1000,
                         triangles=tri, cmap='coolwarm', linewidth=0)
        extent = np.ptp(aligned, axis=0).max()*1000
        center = aligned.mean(axis=0)*1000
        for setter, value in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), center):
            setter(value-extent/2, value+extent/2)
        axis.set_box_aspect((1,1,1))
        axis.set_title('Aligned shape, true equal scale (mm)')
        for pos, values, title in ((132, mean, 'Mean curvature (1/m)\nraw per-face values'),
                                   (133, gauss, 'Gaussian curvature (1/m²)\nraw per-face values')):
            ax = fig.add_subplot(pos)
            limit = max(float(np.max(np.abs(values))), 1e-12)
            image = ax.tripcolor(uv[:,0]*1000, uv[:,1]*1000, tri, facecolors=values,
                                cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set_aspect('equal'); ax.set_title(title)
            ax.set_xlabel('material u (mm)'); ax.set_ylabel('material v (mm)')
            fig.colorbar(image, ax=ax)
        fig.suptitle(f'{directory.parent.name}: {directory.name}')
        fig.savefig(directory / 'diagnosis.png', dpi=140)
        plt.close(fig)
    comparisons = []
    for (left, a), (right, b) in itertools.combinations(meshes.items(), 2):
        if left[1] != right[1] and not (left[0] == right[0] and 'frozen' in left[1] and 'frozen' in right[1]):
            continue
        x, tri, uv, metric = a
        y, triy, uvy, metricy = b
        row = {'left': list(left), 'right': list(right)}
        comparisons.append(row)
        if x.shape != y.shape or not np.array_equal(tri, triy) or not np.array_equal(uv, uvy):
            row['comparable'] = False
            continue
        _, rms, maximum = proper_kabsch(x, y)
        row.update(comparable=True, aligned_rms_mm=rms*1000, aligned_max_mm=maximum*1000)
        if metric is not None and metricy is not None:
            row['metric_relative_difference'] = float(np.max(np.abs(metric-metricy))/max(np.max(np.abs(metric)), np.max(np.abs(metricy)), np.finfo(float).tiny))
    output = {'cases': records, 'comparisons': comparisons,
              'physical_validation': False,
              'note': 'Curvature thresholds are diagnostic sensitivities, not proof of stable minima. Shapes alone do not identify plastic physics.'}
    (root / 'diagnosis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('run/forward_model_diagnostics'))
    analyze(parser.parse_args().root)
