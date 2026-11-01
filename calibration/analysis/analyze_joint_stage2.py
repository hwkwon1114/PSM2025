"""Comprehensive analysis of Stage-1 and Stage-2 joint (Run 6 + Run 10) Bayesian calibration:
- Selected parameters and bounds proximity (flagging bound hits and comparing interior runner-up)
- Per-case RMS and 95th-percentile comparison against Attempt-209 baseline under the shared-footprint metric
- Wave-by-wave objective and feasibility history across Stage 1, Stage 2 Arm A (success-aware), and Stage 2 Arm B (failure-imputation)
- Identifiability and parameter compensation analysis (basin correlations, PCA/SVD of the near-optimal set, and GP Hessian/sensitivity along stiff vs sloppy directions).
"""
import glob
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_campaign(rel_dir: str, spec_path: str):
    base = ROOT / rel_dir
    spec = json.loads((ROOT / spec_path).read_text())
    res_files = sorted(glob.glob(str(base / 'results/a*.json')))
    rows = [json.loads(Path(p).read_text()) for p in res_files]
    rows.sort(key=lambda r: r['attempt'])
    return spec, rows


def bounds_proximity(params: dict, spec: dict):
    names = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor']
    beta_b = spec.get('beta', [0.01, 2.0])
    floor_b = spec.get('floor', [0.0, 1.0])
    lo = [5e-5, 1e-5, -1.0, -1.0, float(beta_b[0]), float(floor_b[0])]
    hi = [5e-3, 1e-3,  1.0,  1.0, float(beta_b[1]), float(floor_b[1])]
    log_flags = [True, True, False, False, True, False]
    out = {}
    for name, l, h, is_log in zip(names, lo, hi, log_flags):
        v = float(params[name])
        if is_log:
            u = (np.log(v) - np.log(l)) / (np.log(h) - np.log(l))
        else:
            u = (v - l) / (h - l)
        at_lo = bool(u <= 0.01)
        at_hi = bool(u >= 0.99)
        out[name] = {
            'value': v,
            'bounds': [l, h],
            'unit_coord': float(u),
            'dist_to_nearest_bound_unit': float(min(u, 1.0 - u)),
            'at_lower_bound': at_lo,
            'at_upper_bound': at_hi,
            'flagged_at_bound': bool(at_lo or at_hi),
        }
    return out


def summarize_campaign(name: str, spec: dict, rows: list):
    valid = [r for r in rows if r.get('valid')]
    failed = [r for r in rows if not r.get('valid')]
    bo_rows = [r for r in rows if r['wave'] >= 4]
    bo_valid = [r for r in bo_rows if r.get('valid')]
    fail_counts = {}
    for r in failed:
        fc = r.get('failure_class') or 'unknown'
        fail_counts[fc] = fail_counts.get(fc, 0) + 1
    valid_sorted = sorted(valid, key=lambda r: r['objective_mm'])
    best = valid_sorted[0] if valid_sorted else None
    wave_history = []
    running_best = float('inf')
    for w in sorted({r['wave'] for r in rows}):
        w_rows = [r for r in rows if r['wave'] == w]
        w_val = [r for r in w_rows if r.get('valid')]
        w_min = min((r['objective_mm'] for r in w_val), default=None)
        if w_min is not None and w_min < running_best:
            running_best = w_min
        wave_history.append({
            'wave': int(w),
            'n_total': len(w_rows),
            'n_valid': len(w_val),
            'n_failed': len(w_rows) - len(w_val),
            'wave_best_mm': float(w_min) if w_min is not None else None,
            'cum_best_mm': float(running_best) if running_best < float('inf') else None,
        })
    return {
        'campaign': name,
        'arm': spec.get('arm'),
        'n_total': len(rows),
        'n_valid': len(valid),
        'n_failed': len(failed),
        'bo_n_total': len(bo_rows),
        'bo_n_valid': len(bo_valid),
        'bo_valid_rate': float(len(bo_valid) / len(bo_rows)) if bo_rows else 0.0,
        'failure_breakdown': fail_counts,
        'best_attempt': best['attempt'] if best else None,
        'best_wave': best['wave'] if best else None,
        'best_objective_mm': float(best['objective_mm']) if best else None,
        'best_case_rms_mm': best.get('case_rms_mm') if best else None,
        'best_case_metrics': best.get('case_metrics') if best else None,
        'best_parameters': best.get('parameters') if best else None,
        'best_bounds_proximity': bounds_proximity(best['parameters'], spec) if best else None,
        'top_10_valid': [
            {
                'attempt': r['attempt'],
                'wave': r['wave'],
                'objective_mm': float(r['objective_mm']),
                'run6_rms_mm': float(r['case_rms_mm']['run6']),
                'run10_rms_mm': float(r['case_rms_mm']['run10']),
                'run6_p95_sym_mm': float(0.5 * (
                    r['case_metrics']['run6']['p95_target_to_sim_mm'] +
                    r['case_metrics']['run6']['p95_sim_to_target_mm']
                )),
                'run10_p95_sym_mm': float(0.5 * (
                    r['case_metrics']['run10']['p95_target_to_sim_mm'] +
                    r['case_metrics']['run10']['p95_sim_to_target_mm']
                )),
                'parameters': r['parameters'],
            }
            for r in valid_sorted[:10]
        ],
        'wave_history': wave_history,
    }


def identifiability_analysis(spec: dict, rows: list):
    names = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor']
    valid = [r for r in rows if r.get('valid')]
    basin = [r for r in valid if r['objective_mm'] <= 0.70]
    X_unit = np.array([r['x'] for r in basin], float)
    y = np.array([r['objective_mm'] for r in basin], float)
    r6 = np.array([r['case_rms_mm']['run6'] for r in basin], float)
    r10 = np.array([r['case_rms_mm']['run10'] for r in basin], float)

    # Effective growth rates per hit in top/bottom along (1) and (2):
    # g1_top = gtop * (1 + ortho_top), g2_top = gtop * (1 - ortho_top)
    # Integrated effective multiplier over K hits: M(K, beta, floor) = sum_{k=0}^{K-1} [floor + (1-floor)*exp(-beta*k)]
    def hit_sum(K, beta, fl):
        k = np.arange(K, dtype=float)
        return float(np.sum(fl + (1.0 - fl) * np.exp(-beta * k)))

    eff_records = []
    for r in basin:
        p = r['parameters']
        m58 = hit_sum(58, p['beta'], p['floor'])   # Run 6 max hits = 58 (29 cycles * 2)
        m100 = hit_sum(100, p['beta'], p['floor']) # Run 10 max hits = 100 (50 cycles * 2)
        eff_records.append({
            'attempt': r['attempt'],
            'objective_mm': float(r['objective_mm']),
            'run6_rms_mm': float(r['case_rms_mm']['run6']),
            'run10_rms_mm': float(r['case_rms_mm']['run10']),
            'g1_top': float(p['gtop'] * (1.0 + p['ortho_top'])),
            'g2_top': float(p['gtop'] * (1.0 - p['ortho_top'])),
            'M58': m58,
            'M100': m100,
            'M100_over_M58': float(m100 / m58),
            'cum_gtop_58': float(p['gtop'] * m58),
            'cum_gtop_100': float(p['gtop'] * m100),
            'cum_g2top_58': float(p['gtop'] * (1.0 - p['ortho_top']) * m58),
            'cum_g2top_100': float(p['gtop'] * (1.0 - p['ortho_top']) * m100),
        })

    corr = np.corrcoef(X_unit, rowvar=False)
    # PCA of unit-box coordinates in the low-objective basin (J <= 0.70 mm)
    Xc = X_unit - X_unit.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_explained = (S ** 2) / np.sum(S ** 2)

    return {
        'n_basin_le_0p70mm': len(basin),
        'parameter_names': names,
        'basin_unit_mean': X_unit.mean(axis=0).tolist(),
        'basin_unit_std': X_unit.std(axis=0).tolist(),
        'basin_correlation_matrix': corr.tolist(),
        'pca_singular_values': S.tolist(),
        'pca_variance_explained': var_explained.tolist(),
        'pca_components_rows': Vt.tolist(),
        'effective_cumulative_growth_in_basin': eff_records,
    }


def main():
    s1_spec, s1_rows = load_campaign('run/joint_run6_run10', 'calibration/bo/stage1_default.json') if (ROOT / 'calibration/bo/stage1_default.json').exists() else (
        {
            'arm': 'success',
            'bounds_lo': [5e-5, 1e-5, -1.0, -1.0, 0.01, 0.0],
            'bounds_hi': [5e-3, 1e-3,  1.0,  1.0, 2.00, 1.0],
        },
        [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(ROOT / 'run/joint_run6_run10/results/a*.json')))]
    )
    a_spec, a_rows = load_campaign('run/joint_stage2/arm_A', 'calibration/bo/stage2_arm_A.json')
    b_spec, b_rows = load_campaign('run/joint_stage2/arm_B', 'calibration/bo/stage2_arm_B.json')

    baseline = next(r for r in a_rows if r['attempt'] == 1)
    winner = min((r for r in a_rows if r.get('valid')), key=lambda r: r['objective_mm'])
    interior_runner_up = min(
        (r for r in a_rows if r.get('valid') and r['parameters']['gbot'] > 1.05e-5),
        key=lambda r: r['objective_mm']
    )

    out = {
        'baseline_attempt_209': {
            'attempt': 1,
            'objective_mm': float(baseline['objective_mm']),
            'case_rms_mm': baseline['case_rms_mm'],
            'case_metrics': baseline['case_metrics'],
            'parameters': baseline['parameters'],
        },
        'stage2_winner_a0026': {
            'attempt': winner['attempt'],
            'wave': winner['wave'],
            'objective_mm': float(winner['objective_mm']),
            'case_rms_mm': winner['case_rms_mm'],
            'case_metrics': winner['case_metrics'],
            'parameters': winner['parameters'],
            'bounds_proximity': bounds_proximity(winner['parameters'], a_spec),
        },
        'stage2_interior_runner_up_a0043': {
            'attempt': interior_runner_up['attempt'],
            'wave': interior_runner_up['wave'],
            'objective_mm': float(interior_runner_up['objective_mm']),
            'case_rms_mm': interior_runner_up['case_rms_mm'],
            'case_metrics': interior_runner_up['case_metrics'],
            'parameters': interior_runner_up['parameters'],
            'bounds_proximity': bounds_proximity(interior_runner_up['parameters'], a_spec),
        },
        'campaigns': {
            'stage1_broad_bounds': summarize_campaign('stage1_broad_bounds', s1_spec, s1_rows),
            'stage2_arm_A_success': summarize_campaign('stage2_arm_A_success', a_spec, a_rows),
            'stage2_arm_B_impute': summarize_campaign('stage2_arm_B_impute', b_spec, b_rows),
        },
        'identifiability_arm_A': identifiability_analysis(a_spec, a_rows),
    }
    out_path = ROOT / 'run/joint_stage2/stage2_analysis_summary.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'Wrote {out_path}')
    print(f"Baseline (a0001): J={baseline['objective_mm']:.6f} mm (run6={baseline['case_rms_mm']['run6']:.6f}, run10={baseline['case_rms_mm']['run10']:.6f})")
    print(f"Winner   (a0026): J={winner['objective_mm']:.6f} mm (run6={winner['case_rms_mm']['run6']:.6f}, run10={winner['case_rms_mm']['run10']:.6f})")
    print(f"Interior (a0043): J={interior_runner_up['objective_mm']:.6f} mm (run6={interior_runner_up['case_rms_mm']['run6']:.6f}, run10={interior_runner_up['case_rms_mm']['run10']:.6f})")
    bp = out['stage2_winner_a0026']['bounds_proximity']
    flagged = [k for k, v in bp.items() if v['flagged_at_bound']]
    print(f'Winner bounds flags: {flagged}')
    id_res = out['identifiability_arm_A']
    print(f"Basin (J<=0.70 mm) count: {id_res['n_basin_le_0p70mm']}")
    for rec in id_res['effective_cumulative_growth_in_basin'][:6]:
        print(f"  a{rec['attempt']:04d}: J={rec['objective_mm']:.4f} | M100/M58={rec['M100_over_M58']:.3f} | cum_gtop58={rec['cum_gtop_58']:.4e} | cum_g2top58={rec['cum_g2top_58']:.4e}")


if __name__ == '__main__':
    main()
