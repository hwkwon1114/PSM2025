#!/usr/bin/env python3
"""Compare the proposed cross-hatch against retained Attempts 020 / 045 on post-solve states only.

All three states must come from the same solver settings (thickness, resolution, tolerances); the
script refuses to report a comparison otherwise. The proposed design is reported ONLY if its own solve
was accepted; a failed/unaccepted solve is reported as such, never replaced by a prescreen.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'legacy/analysis'))
sys.path.insert(0, str(ROOT / 'dome_bo'))

from dome_curvature import geometric_curvature
from dome_continuous_objective import evaluate_dome_objective

BASELINES = {
    'Attempt 020 (center cross)': ROOT / 'run/dome/bo_continuation_stage1/attempt_020/solve',
    'Attempt 045 (best 3-toolpath BO)': ROOT / 'run/dome/bo_continuation_stage1/attempt_045/solve',
}
PROPOSED_NAME = 'Proposed pitch-graded cross-hatch'
PROPOSED_DIR = ROOT / 'run/dome/proposed_matched_solve'
OUT_DIR = ROOT / 'legacy/analysis/plots'
SETTING_KEYS = ('-h_total', '-res', '-lx', '-ly', '-native_hybrid_grad_tol', '-equilibrium_grad_tol')


def run_settings(solve_dir: Path) -> dict:
    """Solver settings actually used, parsed from the run's argumentparser.log (last value wins)."""
    tokens = (solve_dir / 'argumentparser.log').read_text().split()
    found = {}
    for i, tok in enumerate(tokens[:-1]):
        if tok in SETTING_KEYS:
            found[tok] = float(tokens[i + 1])
    return found


def load(solve_dir: Path) -> dict:
    result = json.loads((solve_dir / 'sequence_result.json').read_text())
    proof = result['solves'][-1]['native_hybrid']
    state = json.loads((solve_dir / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    accepted = bool(result.get('completed')) and bool(proof['accepted'])
    out = {'dir': str(solve_dir), 'accepted': accepted, 'gradient_norm': proof['gradient_norm'],
           'settings': run_settings(solve_dir), 'cycles': result['cycles']}
    if accepted:
        positions = np.asarray(state['positions'], float)
        uv = np.asarray(state['material_uv'], float)
        faces = np.asarray(state['faces'], int)
        values, _, valid = geometric_curvature(positions, uv, faces)
        out.update(objective=evaluate_dome_objective(values, valid, uv, faces), uv=uv, faces=faces, values=values)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {name: load(path) for name, path in BASELINES.items()}
    if (PROPOSED_DIR / 'sequence_result.json').exists():
        runs[PROPOSED_NAME] = load(PROPOSED_DIR)
    else:
        runs[PROPOSED_NAME] = {'accepted': False, 'dir': str(PROPOSED_DIR), 'settings': {}, 'cycles': None,
                               'gradient_norm': None, 'reason': 'no completed solve in output directory'}

    reference = runs[next(iter(BASELINES))]['settings']
    mismatched = {n: r['settings'] for n, r in runs.items() if r['settings'] != reference}
    summary = {}
    for name, r in runs.items():
        entry = {'solve_dir': r['dir'], 'accepted_solve': r['accepted'], 'settings': r['settings'],
                 'cycles': r['cycles'], 'gradient_norm': r['gradient_norm']}
        if r['accepted']:
            o = r['objective']
            entry.update({k: o[k] for k in ('j_combined', 'j_quantile', 'saddle_flux_fraction',
                                            'total_coherent_fraction', 'largest_connected_fraction',
                                            'valid_area_fraction', 'qualified')})
        summary[name] = entry
    summary['_settings_matched'] = not mismatched
    summary['_settings_mismatch_detail'] = mismatched
    (OUT_DIR / 'dome_parameterization_comparison_metrics.json').write_text(json.dumps(summary, indent=2))

    print('=' * 100)
    print(f"{'Configuration':<36} | {'solve':<10} | {'coherent area':>13} | {'largest connected':>17} | gate >=80%")
    print('-' * 100)
    for name, r in runs.items():
        if r['accepted']:
            o = r['objective']
            print(f"{name:<36} | {'accepted':<10} | {o['total_coherent_fraction']*100:>12.1f}% | "
                  f"{o['largest_connected_fraction']*100:>16.1f}% | {'PASS' if o['qualified'] else 'FAIL'}")
        else:
            print(f"{name:<36} | {'NOT ACCEPTED':<10} | {'n/a':>13} | {'n/a':>17} | n/a")
    print('=' * 100)
    print('Settings matched across all runs:' , not mismatched)
    for n, s in mismatched.items():
        print('  mismatch', n, s, 'vs reference', reference)

    # Plot only accepted, setting-matched post-solve states (principal-curvature sign map).
    shown = [(n, r) for n, r in runs.items() if r['accepted']]
    fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 6), layout='constrained', squeeze=False)
    norm = Normalize(vmin=-3.0, vmax=3.0)
    for ax, (name, r) in zip(axes[0], shown):
        K = np.nan_to_num(r['values'][:, 0] * r['values'][:, 1])
        tri = Triangulation(r['uv'][:, 0] * 1000.0, r['uv'][:, 1] * 1000.0, r['faces'])
        tc = ax.tripcolor(tri, facecolors=K, cmap='RdBu_r', norm=norm, rasterized=True)
        ax.add_patch(Rectangle((-80, -120), 160, 240, fill=False, ls='--', ec='black', lw=1.2))
        o = r['objective']
        ax.set_title(f"{name}\nconnected {o['largest_connected_fraction']*100:.1f}% "
                     f"({'PASS' if o['qualified'] else 'FAIL'})", fontsize=9)
        ax.set_xlabel('Material u (mm)')
        ax.set_ylabel('Material v (mm)')
        ax.set_aspect('equal')
    fig.colorbar(tc, ax=axes[0].tolist(), shrink=0.7, label='Face Gaussian curvature K (m$^{-2}$)')
    stem = 'dome_parameterization_comparison_baselines'
    fig.savefig(OUT_DIR / f'{stem}.png', dpi=200)
    fig.savefig(OUT_DIR / f'{stem}.pdf')
    plt.close(fig)
    note = ('Post-solve Gaussian curvature of accepted equilibria computed with the repository geometric '
            'curvature (15 mm material neighbourhood). Gate = largest edge-connected coherent same-sign '
            'region (|k|min >= 0.1 1/m, |k|min >= 0.1|k|max) >= 80% of the 160x240 mm ROI. '
            f"Settings matched across runs: {not mismatched}.")
    (OUT_DIR / f'{stem}.caption.md').write_text(note + '\n')


if __name__ == '__main__':
    main()
