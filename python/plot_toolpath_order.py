#!/usr/bin/env python3
"""Render the completed nested-path order comparison; no simulations launched."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_sequence_ablation_v2 import cell_array, metric_vector, proper_kabsch
from visualize_solver_characteristics import (discover_case, load_mesh, place_strips,
    validate_reconstruction, draw_toolpath, midline_profile)


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'run/solver_characteristics/shrinking_crown'
    names = ['crown_nested_broad_to_central', 'crown_nested_central_to_broad']
    labels = ['A: Broad → central', 'B: Central → broad']
    cases = [discover_case(base / name) for name in names]
    assert [s['operation'] for s in cases[0]['schedule']] == [s['operation'] for s in reversed(cases[1]['schedule'])]
    for case in cases:
        assert case['result']['return_code'] == 0
        assert case['result']['convergence']['all_expected_cycles_accepted']
    meshes = [load_mesh(c['final_vtp']) for c in cases]
    assert np.array_equal(meshes[0]['material'], meshes[1]['material'])
    assert np.array_equal(meshes[0]['triangles'], meshes[1]['triangles'])
    uv = meshes[0]['material']
    reference = np.column_stack([uv, np.zeros(len(uv))])
    aligned = [proper_kabsch(reference, m['points'])[0] for m in meshes]
    matched, rms, maximum = proper_kabsch(meshes[0]['points'], meshes[1]['points'])
    difference = np.linalg.norm(matched - meshes[0]['points'], axis=1)*1000
    metrics = [metric_vector(m['data']) for m in meshes]
    metric_error = float(np.max(np.abs(metrics[0]-metrics[1])) / max(np.max(np.abs(m)) for m in metrics))
    height = [float(np.ptp(x[:, 2])*1000) for x in aligned]
    deformation_rms = proper_kabsch(reference, meshes[0]['points'])[1]
    bbox = (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, .65])
    paths = grid[0].subgridspec(2, 6)
    validations = []
    for row, case in enumerate(cases):
        for col, entry in enumerate(case['schedule']):
            mesh = load_mesh(case['mapping_vtps'][col+1])
            hits = cell_array(mesh['data'], 'hits_this_cycle')
            placement = place_strips(entry['operation'], bbox)
            v = validate_reconstruction(placement, mesh, hits)
            assert v['hit_count_identical'] and v['contact_face_set_identical']
            assert v['strip_index_mismatches'] == 0 and v['max_growth_angle_error_rad'] < 1e-10
            validations.append(v)
            ax = fig.add_subplot(paths[row, col])
            draw_toolpath(ax, placement, mesh, hits, bbox, show_arrows=False)
            ax.set_title(f'Pass {col+1}', fontsize=10)
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
    maps = grid[1].subgridspec(1, 3)
    limit = max(np.max(np.abs(x[:, 2]))*1000 for x in aligned)
    for i in range(3):
        ax = fig.add_subplot(maps[0, i])
        if i < 2:
            values = aligned[i][:, 2]*1000
            image = ax.tripcolor(uv[:, 0]*1000, uv[:, 1]*1000, meshes[i]['triangles'], values,
                                shading='gouraud', cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set_title(f'{labels[i]}\nHeight span = {height[i]:.3f} mm')
            barlabel = 'Aligned out-of-plane displacement (mm)'
        else:
            image = ax.tripcolor(uv[:, 0]*1000, uv[:, 1]*1000, meshes[0]['triangles'], difference,
                                shading='gouraud', cmap='magma', vmin=0, vmax=maximum*1000)
            ax.set_title(f'Final shape difference |B − A|\nRigid-aligned RMS = {rms*1000:.3f} mm')
            barlabel = '3D displacement difference (mm)'
        ax.set_aspect('equal'); ax.set_xlabel('Material u (mm)'); ax.set_ylabel('Material v (mm)')
        fig.colorbar(image, ax=ax, label=barlabel, shrink=.85)
    bottom = grid[2].subgridspec(1, 3)
    for direction in range(2):
        ax = fig.add_subplot(bottom[0, direction])
        for mesh, x, label in zip(meshes, aligned, labels):
            coordinate, w = midline_profile(mesh, x[:, 2]*1000, direction)
            ax.plot(coordinate, w, label=label, linewidth=2)
        ax.set_title('Midline v = 0' if direction == 0 else 'Midline u = 0')
        ax.set_xlabel('Material ' + ('u' if direction == 0 else 'v') + ' (mm)')
        ax.set_ylabel('Aligned height (mm)'); ax.set_ylim(-limit*1.1, limit*1.1)
        ax.grid(alpha=.25); ax.legend(fontsize=9)
    ax = fig.add_subplot(bottom[0, 2]); ax.axis('off')
    ax.text(0, 1, '\n'.join([
        'Same paths; reversed execution order',
        'Equilibrium accepted after all 6 paths',
        f'Final metric relative difference: {metric_error:.2e}',
        f'Shape RMS difference: {rms*1000:.3f} mm',
        f'Maximum difference: {maximum*1000:.3f} mm',
        f'RMS / deformation RMS of A: {100*rms/deformation_rms:.1f}%',
        'Plate: 254 × 304.8 × 0.6 mm; res = 0.015',
        'Top-only growth: 120 microstrain per hit',
        '',
        'Evidence of numerical order sensitivity.',
        'Not proof of stable minima or physical accuracy.',
    ]), va='top', fontsize=10, linespacing=1.6)
    fig.suptitle('Toolpath order changes the computed final shape\n'
                 'Top: actual path footprints and exported face hits. Middle: shared height scale; rigid motion removed.',
                 fontsize=16, fontweight='bold')
    out = root / 'run/forward_model_diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'toolpath_order_effect.{suffix}', dpi=180)
    plt.close(fig)
    evidence = dict(cases=names, metric_relative_difference=metric_error,
                    shape_rms_mm=rms*1000, shape_max_mm=maximum*1000,
                    height_spans_mm=height, rms_percent_of_reference_deformation=100*rms/deformation_rms,
                    toolpath_validations=validations)
    (out/'toolpath_order_effect.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k != 'toolpath_validations'}, indent=2))
    print(out / 'toolpath_order_effect.png')
    progression(cases, labels, out)


def progression(cases, labels, out, title='Deformation progression: same toolpaths, opposite order'):
    states = [[load_mesh(case['final_vtps'][i]) for i in range(1, 7)] for case in cases]
    heights = []
    for row in states:
        row_heights = []
        for mesh in row:
            uv = mesh['material']
            ref = np.column_stack([uv, np.zeros(len(uv))])
            row_heights.append(proper_kabsch(ref, mesh['points'])[0][:, 2]*1000)
        heights.append(row_heights)
    limit = max(float(np.max(np.abs(w))) for row in heights for w in row)
    fig, axes = plt.subplots(2*len(cases), 6, figsize=(16, 6*len(cases)), constrained_layout=True)
    for order, case in enumerate(cases):
        uv = states[order][0]['material']
        bbox = (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())
        for col, entry in enumerate(case['schedule']):
            ax = axes[2*order, col]
            mapping = load_mesh(case['mapping_vtps'][col+1])
            hits = cell_array(mapping['data'], 'hits_this_cycle')
            draw_toolpath(ax, place_strips(entry['operation'], bbox), mapping, hits, bbox,
                          show_arrows=False)
            ax.set_title(f'Path {col+1}', fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            mesh = states[order][col]
            ax = axes[2*order+1, col]
            image = ax.tripcolor(uv[:, 0]*1000, uv[:, 1]*1000, mesh['triangles'],
                                heights[order][col], shading='gouraud',
                                cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set_xlim(bbox[0]*1000, bbox[1]*1000)
            ax.set_ylim(bbox[2]*1000, bbox[3]*1000)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'After path {col+1}', fontsize=11)
        axes[2*order, 0].set_ylabel(labels[order] + '\nToolpath', fontsize=12, fontweight='bold')
        axes[2*order+1, 0].set_ylabel('Deformation', fontsize=12, fontweight='bold')
    fig.colorbar(image, ax=axes, shrink=.65, pad=.015,
                 label='Out-of-plane displacement after rigid alignment (mm)')
    fig.suptitle(title, fontsize=18, fontweight='bold')
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'toolpath_order_progression.{suffix}', dpi=180)
    plt.close(fig)
    print(out / 'toolpath_order_progression.png')


if __name__ == '__main__':
    main()
