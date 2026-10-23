#!/usr/bin/env python3
"""Analyse the mesh/tolerance convergence study from retained states (no solves). Criteria C1-C3 are the ones frozen in
docs/reports/dome_toolpath_investigation.md ("Square-sheet mesh/tolerance convergence study"). Usage: mesh_convergence.py OUT_JSON"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import square_attempt as sa            # noqa: E402
import whole_sheet_diagnostics as wsd  # noqa: E402

CAND = [30, 31, 21, 32, 27, 19]
MESH = [('.02', 4), ('.01', 2), ('.005', 1)]
RMS_MAX_M, GATE_DELTA_MAX, GATE_PASS, RHO_MIN = 1.0e-4, 0.05, 0.80, 0.9
STUDY = ROOT / 'run/dome_square/mesh_study'


def cell_dir(a, r):
    if (a, r) == (30, '.01'):
        return ROOT / 'run/dome_square/verification/attempt_030'
    return STUDY / f'a{a:03d}_res{r[1:]}'


def cell_metrics(a, h, r):
    """Heavy per-cell stage (cacheable): geometry/curvature diagnostics + gate at 10/15/20 mm. 15 mm reuses the cached
    frozen-estimator curvature (identical to analyze_state at 15 mm); 10/20 mm are computed fresh."""
    d = cell_dir(a, r)
    status = json.loads((d / 'status.json').read_text())
    rec = {'state': status['state'], 'failure_type': status.get('failure_type'),
           'elapsed_seconds': status.get('elapsed_seconds')}
    sp = d / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json'
    if status['state'] == 'accepted' and sp.exists():
        st, geo, dd = wsd.analyse(sp)
        rec.update(n_faces=geo['n_faces'], geometry=geo['geometry'], roi_area_fraction=geo['roi']['area_fraction'],
                   whole_area_fraction=geo['whole']['area_fraction'],
                   whole_k_small_abs=geo['whole']['k_small_abs'], whole_k_large_abs=geo['whole']['k_large_abs'])
        obj = sa.dco.evaluate_dome_objective(dd['vals'], np.load(sp.with_name('curvature_r15mm_cache.npz'))['valid'],
                                             dd['uv'], dd['faces'], roi_half=(sa.ROI_HALF_M, sa.ROI_HALF_M))
        rec['r15mm'] = {k: obj[k] for k in ('largest_connected_fraction', 'j_combined', 'total_coherent_fraction')}
        for radius in (0.010, 0.020):
            m = sa.analyze_state(st, radius=radius)
            rec[f'r{int(radius * 1000)}mm'] = {k: m[k] for k in ('largest_connected_fraction', 'j_combined',
                                                                  'total_coherent_fraction')}
        rec['gate_fraction'] = rec['r15mm']['largest_connected_fraction']
    return rec


def main(out_json):
    cells, data = {}, {}
    for a in CAND:
        for r, h in MESH:
            key = f'a{a}_{h}mm'
            f = STUDY / 'cell_metrics' / f'{key}.json'
            if not (cell_dir(a, r) / 'status.json').exists():
                cells[key] = {'state': 'missing'}
                continue
            if not f.exists():
                raise SystemExit(f'run the per-cell stage first: mesh_convergence.py --cell {a} {r} ({f} missing)')
            rec = json.loads(f.read_text())
            cells[key] = rec
            if 'gate_fraction' in rec:
                data[(a, h)] = wsd.analyse(cell_dir(a, r) / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json')[2]
    pairs = {}
    for (r0, h0), (r1, h1) in zip(MESH[:-1], MESH[1:]):
        name = f'{h0}mm_to_{h1}mm'
        per = {}
        for a in CAND:
            if (a, h0) in data and (a, h1) in data:
                sd = wsd.shape_difference(data[(a, h0)], data[(a, h1)])
                g0, g1 = cells[f'a{a}_{h0}mm']['gate_fraction'], cells[f'a{a}_{h1}mm']['gate_fraction']
                per[a] = {'roi_rms_mm': sd['roi']['area_weighted_rms_m'] * 1e3,
                          'roi_p99_mm': sd['roi']['distance_percentiles_m']['p99'] * 1e3,
                          'whole_rms_mm': sd['whole']['area_weighted_rms_m'] * 1e3,
                          'gate_coarse': g0, 'gate_fine': g1, 'gate_delta_pp': 100 * (g1 - g0),
                          'C1_shape': sd['roi']['area_weighted_rms_m'] <= RMS_MAX_M,
                          'C2_gate': abs(g1 - g0) <= GATE_DELTA_MAX and ((g0 >= GATE_PASS) == (g1 >= GATE_PASS))}
        both = [a for a in CAND if a in per]
        c3 = None
        if len(both) == len(CAND):
            x = [per[a]['gate_coarse'] for a in CAND]
            y = [per[a]['gate_fine'] for a in CAND]
            rho = float(spearmanr(x, y)[0])
            c3 = {'spearman_rho': rho, 'top_coarse': CAND[int(np.argmax(x))], 'top_fine': CAND[int(np.argmax(y))]}
            c3['C3_ranking'] = bool(rho >= RHO_MIN and c3['top_coarse'] == c3['top_fine'])
        met = (len(both) == len(CAND) and all(per[a]['C1_shape'] and per[a]['C2_gate'] for a in CAND)
               and bool(c3 and c3['C3_ranking']))
        pairs[name] = {'per_candidate': per, 'C3': c3, 'all_criteria_met': met}
    result = {'criteria': {'C1_roi_rms_max_mm': RMS_MAX_M * 1e3, 'C2_gate_delta_max_pp': 100 * GATE_DELTA_MAX,
                           'C3_spearman_min': RHO_MIN}, 'cells': cells, 'pairs': pairs}
    Path(out_json).write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps({k: v['all_criteria_met'] for k, v in pairs.items()}))
    for key, c in cells.items():
        print(key, c.get('state'), c.get('gate_fraction'), c.get('elapsed_seconds'))


if __name__ == '__main__':
    if sys.argv[1] == '--cell':
        a, r = int(sys.argv[2]), sys.argv[3]
        h = dict(MESH)[r]
        out = STUDY / 'cell_metrics'
        out.mkdir(exist_ok=True)
        (out / f'a{a}_{h}mm.json').write_text(json.dumps(cell_metrics(a, h, r), indent=2, default=float) + '\n')
    else:
        main(sys.argv[1])
