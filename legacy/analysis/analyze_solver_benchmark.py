#!/usr/bin/env python3
"""Rigorous post-execution analysis and qualification of solver benchmark matrix."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/scripts'))
sys.path.insert(0, str(ROOT / 'legacy/analysis'))

from analyze_two_gripper_release import align, area_weights

def fit_midline_curvatures(uv, z):
    u = uv[:, 0] / 1000.0
    v = uv[:, 1] / 1000.0
    zm = z / 1000.0
    M = np.column_stack((0.5 * u**2, u * v, 0.5 * v**2, u, v, np.ones_like(u)))
    coeffs, _, _, _ = np.linalg.lstsq(M, zm, rcond=None)
    H = np.array([[coeffs[0], coeffs[1]], [coeffs[1], coeffs[2]]])
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0]), float(eigvals[1])

MATRIX_DIR = ROOT / 'run/solver_benchmark/matrix'
OUTPUT_FILE = ROOT / 'run/solver_benchmark/benchmark_analysis.json'


def load_state(case_dir):
    state_file = case_dir / 'bilayer_zigzag_sequence_native_hybrid_state.json'
    proof_file = case_dir / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl'   # canonical full history; per-substep files are partial
    csv_file = case_dir / 'sequence_convergence.csv'
    stability_file = case_dir / 'sequence_stability.csv'
    exec_file = case_dir / 'execution.json'

    state = None
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except Exception:
            state = None

    proofs = []
    if proof_file and proof_file.exists():
        with open(proof_file) as f:
            for line in f:
                if line.strip():
                    proofs.append(json.loads(line))

    csv_rows = []
    if csv_file and csv_file.exists():
        with open(csv_file) as f:
            header = f.readline().strip().split(',')
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    csv_rows.append(dict(zip(header, parts)))

    stability = {}
    if stability_file and stability_file.exists():
        try:
            lines = [l.strip() for l in stability_file.read_text().splitlines() if l.strip()]
            if len(lines) >= 2:
                header = lines[0].split(',')
                parts = lines[-1].split(',')
                stability = dict(zip(header, parts))
                for k in ['smallest_ritz_value', 'residual_absolute', 'residual_relative']:
                    if k in stability:
                        try: stability[k] = float(stability[k])
                        except (ValueError, TypeError): pass
                for k in ['residual_converged', 'operator_evaluations', 'rigid_modes']:
                    if k in stability:
                        try: stability[k] = int(stability[k])
                        except (ValueError, TypeError): pass
        except Exception:
            pass

    exec_data = {}
    if exec_file and exec_file.exists():
        try:
            exec_data = json.loads(exec_file.read_text())
        except Exception:
            pass

    return state, proofs, csv_rows, stability, exec_data


def analyze():
    cases = ['simple_a_1cycle', 'moderate_nonoverlapping_20cycle', 'complex_overlapping_20cycle']
    policies = ['cutback', 'baseline', 'continuation']

    results = {}

    for c in cases:
        results[c] = {}
        for p in policies:
            case_name = f"{c}_{p}"
            case_dir = MATRIX_DIR / case_name
            state, proofs, csv_rows, stability, exec_data = load_state(case_dir)

            total_lbfgs_sec = sum(pr.get('lbfgs_seconds', 0.0) for pr in proofs)
            total_newton_sec = sum(pr.get('newton_seconds', 0.0) for pr in proofs)
            total_solver_sec = total_lbfgs_sec + total_newton_sec
            total_hessians = sum(pr.get('hessians', 0) for pr in proofs)
            total_evals = sum(pr.get('evaluations', 0) for pr in proofs)
            total_attempts = sum(pr.get('attempts', 0) for pr in proofs)

            attempt1_lbfgs = proofs[0].get('lbfgs_seconds', 0.0) if proofs else 0.0
            attempt1_newton = proofs[0].get('newton_seconds', 0.0) if proofs else 0.0
            attempt1_solver_sec = attempt1_lbfgs + attempt1_newton
            attempt1_status = proofs[0].get('termination', 'unknown') if proofs else 'unknown'

            if state is None:
                results[c][p] = {
                    "accepted": False,
                    "status": proofs[-1].get('terminal', {}).get('status', 'not_accepted') if proofs else 'not_accepted',
                    "attempts_count": len(proofs),
                    "total_attempts": total_attempts,
                    "total_evaluations": total_evals,
                    "total_hessians": total_hessians,
                    "total_lbfgs_seconds": total_lbfgs_sec,
                    "total_newton_seconds": total_newton_sec,
                    "total_solver_seconds": total_solver_sec,
                    "attempt1_lbfgs_seconds": attempt1_lbfgs,
                    "attempt1_newton_seconds": attempt1_newton,
                    "attempt1_solver_seconds": attempt1_solver_sec,
                    "attempt1_status": attempt1_status,
                    "final_energy": None,
                    "final_gradient_norm": None,
                    "curvature_k1": None,
                    "curvature_k2": None,
                    "state_positions": None,
                    "weights": None,
                    "ref_flat": None,
                    "stability": stability,
                    "binary_sha256": exec_data.get('binary_sha256', ''),
                    "wall_time_seconds": exec_data.get('wall_time_seconds', 0.0),
                    "proof_policy": proofs[-1].get('handoff_policy', {}) if proofs else {}
                }
                continue

            # Curvatures
            uv = np.asarray(state['material_uv']) * 1000.0
            faces = np.asarray(state['faces'])
            ref_flat = np.column_stack((uv, np.zeros(len(uv))))
            weights = area_weights(ref_flat, faces)

            pos = np.asarray(state['positions']) * 1000.0
            pos_aligned, _, _ = align(pos, ref_flat, weights)
            k1, k2 = fit_midline_curvatures(uv, pos_aligned[:, 2])

            final_energy = float(csv_rows[-1]['recomputed_energy']) if csv_rows else float(state.get('energy', 0.0))
            final_grad = float(csv_rows[-1]['final_gradient_norm']) if csv_rows else float(state.get('gradient_norm', 0.0))

            results[c][p] = {
                "attempts_count": len(proofs),
                "total_attempts": total_attempts,
                "total_evaluations": total_evals,
                "total_hessians": total_hessians,
                "total_lbfgs_seconds": total_lbfgs_sec,
                "total_newton_seconds": total_newton_sec,
                "total_solver_seconds": total_solver_sec,
                "attempt1_lbfgs_seconds": attempt1_lbfgs,
                "attempt1_newton_seconds": attempt1_newton,
                "attempt1_solver_seconds": attempt1_solver_sec,
                "attempt1_status": attempt1_status,
                "final_energy": final_energy,
                "final_gradient_norm": final_grad,
                "curvature_k1": k1,
                "curvature_k2": k2,
                "state_positions": pos_aligned,
                "weights": weights,
                "ref_flat": ref_flat,
                "stability": stability,
                "binary_sha256": exec_data.get('binary_sha256', ''),
                "wall_time_seconds": exec_data.get('wall_time_seconds', 0.0),
                "proof_policy": proofs[-1].get('handoff_policy', {}) if proofs else {}
            }

    # Pairwise shape comparisons
    comparisons = {}
    for c in cases:
        comparisons[c] = {}
        avail = [p for p in ['cutback', 'continuation', 'baseline'] if p in results[c] and results[c][p].get('state_positions') is not None]
        pairs = []
        if 'cutback' in avail and 'continuation' in avail:
            pairs.append(('cutback', 'continuation'))
        if 'cutback' in avail and 'baseline' in avail:
            pairs.append(('cutback', 'baseline'))
        if 'baseline' in avail and 'continuation' in avail:
            pairs.append(('baseline', 'continuation'))

        for p1, p2 in pairs:
            d1 = results[c][p1]
            d2 = results[c][p2]

            pos1 = d1['state_positions']
            pos2 = d2['state_positions']
            weights = d1['weights']
            is_bitwise = bool(np.array_equal(pos1, pos2))

            aligned1, _, _ = align(pos1, pos2, weights)
            diff = aligned1 - pos2
            rms_mm = float(np.sqrt(np.sum(weights * np.sum(diff**2, axis=1)) / np.sum(weights)))
            max_mm = float(np.max(np.linalg.norm(diff, axis=1)))
            energy_diff = abs(d1['final_energy'] - d2['final_energy'])

            comparisons[c][f"{p1}_vs_{p2}"] = {
                "rms_displacement_mm": rms_mm,
                "rms_displacement_nm": rms_mm * 1e6,
                "max_displacement_mm": max_mm,
                "max_displacement_nm": max_mm * 1e6,
                "energy_difference_J": energy_diff,
                "energy_diff_relative": energy_diff / max(abs(d1['final_energy']), 1e-300),
                "k1_shift_pct": abs(d1['curvature_k1'] - d2['curvature_k1']) / max(abs(d1['curvature_k1']), 1e-12) * 100.0,
                "k2_shift_pct": abs(d1['curvature_k2'] - d2['curvature_k2']) / max(abs(d1['curvature_k2']), 1e-12) * 100.0,
                "bitwise_identical": is_bitwise
            }

    # Clean out large arrays before writing JSON
    summary_data = {
        "cases": {},
        "comparisons": comparisons,
        "qualification_analysis": {}
    }

    physical_gate_nm = 36100.0  # 36.1 um physical resolution gate (1e-4 sheet diagonal)
    exploratory_gate_nm = 100.0 # 100 nm tight exploratory gate

    for c in cases:
        summary_data["qualification_analysis"][c] = {}
        if 'cutback_vs_baseline' in comparisons[c]:
            cmp_cb = comparisons[c]['cutback_vs_baseline']
            passed_tight = cmp_cb['rms_displacement_nm'] <= exploratory_gate_nm
            bitwise = cmp_cb.get('bitwise_identical', False)
            verdict_text = "PASS (bitwise identical positions)" if bitwise else (
                f"PASS ({cmp_cb['rms_displacement_nm']:.2f} nm RMS <= {exploratory_gate_nm:.1f} nm gate)" if passed_tight
                else f"FAIL ({cmp_cb['rms_displacement_nm']:.2f} nm RMS > {exploratory_gate_nm:.1f} nm gate)"
            )
            summary_data["qualification_analysis"][c]["cutback_vs_baseline"] = {
                "rms_nm": cmp_cb['rms_displacement_nm'],
                "max_nm": cmp_cb['max_displacement_nm'],
                "bitwise_identical": bitwise,
                "verdict": verdict_text
            }

        if 'cutback_vs_continuation' in comparisons[c]:
            cmp_cc = comparisons[c]['cutback_vs_continuation']
            passed_phys = cmp_cc['rms_displacement_nm'] <= physical_gate_nm
            passed_tight = cmp_cc['rms_displacement_nm'] <= exploratory_gate_nm
            verdict_text = (
                f"PASS ({cmp_cc['rms_displacement_nm']:.2f} nm RMS <= {exploratory_gate_nm:.1f} nm exploratory gate)" if passed_tight
                else (f"PASS ({cmp_cc['rms_displacement_nm']:.2f} nm RMS, well below {physical_gate_nm/1e3:.1f} um physical resolution gate)" if passed_phys
                      else f"FAIL ({cmp_cc['rms_displacement_nm']:.2f} nm RMS > {physical_gate_nm/1e3:.1f} um gate)")
            )
            summary_data["qualification_analysis"][c]["cutback_vs_continuation"] = {
                "rms_nm": cmp_cc['rms_displacement_nm'],
                "max_nm": cmp_cc['max_displacement_nm'],
                "gate_physical_resolution_nm": physical_gate_nm,
                "verdict": verdict_text,
                "relative_to_sheet_diagonal": cmp_cc['rms_displacement_nm'] / 360555127.5,
                "stability": results[c]['cutback'].get('stability', {})
            }

        if c == 'complex_overlapping_20cycle' and 'baseline' in results[c]:
            summary_data["qualification_analysis"][c]["telemetry_verification"] = {
                "cutback_attempt1_solver_seconds": results[c]['cutback']['attempt1_solver_seconds'],
                "baseline_attempt1_solver_seconds": results[c]['baseline']['attempt1_solver_seconds'],
                "cutback_total_hessians": results[c]['cutback']['total_hessians'],
                "baseline_total_hessians": results[c]['baseline']['total_hessians'],
                "time_saved_attempt1_seconds": results[c]['baseline']['attempt1_solver_seconds'] - results[c]['cutback']['attempt1_solver_seconds']
            }

    for c in cases:
        summary_data["cases"][c] = {}
        for p in results[c]:
            if 'state_positions' in results[c][p]:
                d = dict(results[c][p])
                del d['state_positions']
                del d['weights']
                del d['ref_flat']
                summary_data["cases"][c][p] = d

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Successfully generated {OUTPUT_FILE}")
    print("\nSummary Results:")
    for c in cases:
        print(f"\n--- {c} ---")
        for p in policies:
            print(f"  {p:12s}: attempts={summary_data['cases'][c][p]['attempts_count']} "
                  f"solver_sec={summary_data['cases'][c][p]['total_solver_seconds']:.2f}s "
                  f"hessians={summary_data['cases'][c][p]['total_hessians']}")
        for pair, comp in comparisons[c].items():
            print(f"  {pair:24s}: RMS={comp['rms_displacement_nm']:.4f} nm, Max={comp['max_displacement_nm']:.4f} nm")


if __name__ == '__main__':
    analyze()
