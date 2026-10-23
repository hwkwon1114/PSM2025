#!/usr/bin/env python3
"""Postprocess the three centered cross-wheeling schedules without rerunning physics."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from run_forward_model_diagnostics import RUN_ROOT, crosswheel_cases
from visualize_solver_characteristics import CYCLE_RE, expand_schedule, load_mesh, midline_profile
from analyze_sequence_ablation_v2 import proper_kabsch, quadratic_fit, metric_vector, cell_array
from plot_toolpath_order import progression


def main():
    out = RUN_ROOT / 'crosswheel_comparison'
    out.mkdir(exist_ok=True)
    cases, records, endpoints = [], [], []
    for spec in crosswheel_cases():
        directory = RUN_ROOT / 'current' / spec['name']
        result = json.loads((directory/'result.json').read_text())
        if result['return_code'] != 0:
            raise RuntimeError(f'{directory}: simulation failed')
        case = dict(schedule=expand_schedule(spec['sequence']), final_vtps={}, mapping_vtps={})
        for p in directory.glob('*.vtp'):
            m = CYCLE_RE.search(p.name)
            if m:
                case['final_vtps' if m[4]=='final' else 'mapping_vtps'][int(m[1])] = p
        cases.append(case)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for cycle in range(1, 7):
            mesh = load_mesh(case['final_vtps'][cycle])
            ref = np.column_stack([mesh['material'], np.zeros(len(mesh['points']))])
            aligned, rms, _ = proper_kabsch(ref, mesh['points'])
            row = dict(case=spec['name'], cycle=cycle, height_mm=float(np.ptp(aligned[:,2])*1000),
                       deformation_rms_mm=rms*1000, height_over_thickness=float(np.ptp(aligned[:,2])/.0006),
                       broad_shape_fit=quadratic_fit(aligned))
            records.append(row)
            for direction, ax in enumerate(axes):
                s, w = midline_profile(mesh, aligned[:,2]*1000, direction)
                ax.plot(s, w, label=f'Path {cycle}')
                ax.set_xlabel(('u' if direction==0 else 'v')+' (mm)')
                ax.set_ylabel('Aligned height (mm)'); ax.legend(); ax.grid(alpha=.2)
        endpoints.append(mesh)
        fig.suptitle(spec['name'])
        fig.savefig(out/(spec['name']+'_profiles.png'), dpi=180)
        plt.close(fig)
        with (directory/'sequence_convergence.csv').open() as stream:
            convergence = list(csv.DictReader(stream))
        with next(directory.glob('*_summary.csv')).open() as stream:
            energy = list(csv.DictReader(stream))
        (out/(spec['name']+'_solver.json')).write_text(json.dumps(dict(convergence=convergence,energy=energy),indent=2)+'\n')
    comparisons = []
    for a, b in itertools.combinations(range(3), 2):
        x, y = endpoints[a], endpoints[b]
        assert np.array_equal(x['material'], y['material']) and np.array_equal(x['triangles'], y['triangles'])
        _, rms, maximum = proper_kabsch(x['points'], y['points'])
        ma, mb = metric_vector(x['data']), metric_vector(y['data'])
        ha, hb = [cell_array(m['data'], 'total_hit_count') for m in (x,y)]
        comparisons.append(dict(left=crosswheel_cases()[a]['name'], right=crosswheel_cases()[b]['name'],
            rms_mm=rms*1000, max_mm=maximum*1000,
            metric_relative_difference=float(np.max(np.abs(ma-mb))/max(np.max(np.abs(ma)),np.max(np.abs(mb)))),
            hit_maps_identical=bool(np.array_equal(ha,hb)), max_hit_difference=float(np.max(np.abs(ha-hb)))))
    progression(cases, ['Single direction', 'Alternating 0°/90°', 'Blocked 0° then 90°'], out,
                title='Centered cross-wheeling: toolpaths and deformation after every path')
    (out/'comparison.json').write_text(json.dumps(dict(cycles=records,comparisons=comparisons),indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__ == '__main__':
    main()
