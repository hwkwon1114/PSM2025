"""Complete arm_B attempt 13 whose case_00 (run6) finished solving at 14:21:58
and hit the pre-14:27 interpolate_mesh_z ValueError before case_01 (run10) ran.
Scores the existing solved case_00 and runs case_01 using the exact Evaluator contract.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['JOINT_SPEC'] = str(ROOT / 'calibration/bo/stage2_arm_B.json')
from calibration.bo import run_joint_run6_run10 as rj
from calibration.ewcal.common import load_json, save_json
from calibration.ewcal.geometry import reference_dual_areas, write_mesh
from calibration.ewcal.material_frame import score_material_frame
from calibration.ewcal.runner import (Evaluator, RegistrationFailure, SolverFailure,
                                      SolverProcessFailure, check_output, command, launch,
                                      sanitize_for_json, sequence_json, validate_mesh_topology)


def backup_once(p: Path):
    bak = p.with_suffix(p.suffix + '.pre_curled_footprint_fix.bak')
    if p.exists() and not bak.exists():
        shutil.copy2(p, bak)


def score_solved_case(ev: Evaluator, case: dict, casefolder: Path):
    manifest, mesh = check_output(ev.config, case, casefolder)
    state_path = casefolder / 'bilayer_zigzag_sequence_native_hybrid_state.json'
    st_data = json.loads(state_path.read_text())
    st_ref = np.asarray(st_data['reference_positions'], float)
    st_faces = np.asarray(st_data['faces'])
    validate_mesh_topology(mesh, st_faces)
    st_ref_mm = st_ref * 1000.0
    va = reference_dual_areas(st_ref_mm, st_faces)
    total_area = float(np.sum(va))
    expected_area = float(ev.config.get('expected_reference_area_mm2', 60000.0))
    area_tol = float(ev.config.get('area_tolerance_rel', 0.01))
    if abs(total_area - expected_area) / expected_area > area_tol:
        raise SolverFailure(f'Reference area {total_area:.2f} deviates from {expected_area:.2f}')
    metrics, mesh, aligned, distance, closest = score_material_frame(
        mesh, ev.target[case['id']], st_ref_mm, st_faces)
    write_mesh(mesh, casefolder / 'prediction_mm.vtp')
    np.savez_compressed(casefolder / 'comparison.npz', target_mm=aligned,
                        closest_mm=closest, distance_mm=distance)
    case_rms = float(metrics['rms_symmetric_mm'])
    return {'id': case['id'], 'directory': str(casefolder),
            'metrics': sanitize_for_json(metrics), 'manifest': manifest,
            'case_rms': case_rms, 'weight': case.get('weight', 1.0)}


def main():
    base = ROOT / 'run/joint_stage2/arm_B'
    top_res_path = base / 'results/a0013.json'
    if top_res_path.exists():
        cur = load_json(top_res_path)
        if cur.get('valid') or cur.get('failure_class') in ('solver', 'timeout', 'registration'):
            print('arm_B a0013 already completed:', cur.get('valid'), cur.get('failure_class'), cur.get('objective_mm'))
            return
    prop = load_json(base / 'proposals/a0013.json')
    vals = rj.decode(prop['x'])
    out = base / 'candidates/a0013'
    folder = out / 'evaluations/attempt_0001'
    for p in (top_res_path, folder / 'result.json', out / 'evaluations.jsonl'):
        backup_once(p)
    cfg = rj.joint_config(vals)
    cfg['output_dir'] = f'{rj.BASE_REL}/candidates/a0013'
    cfg['report_dir'] = cfg['output_dir'] + '/report'
    ev = Evaluator(cfg)
    params_4 = {k: vals[k] for k in cfg['parameters']}
    rec = load_json(folder / 'result.json')
    t0 = time.monotonic()
    phase = 'infrastructure'
    try:
        c0_case = cfg['cases'][0]
        c0_folder = folder / 'case_00'
        c0_rec = score_solved_case(ev, c0_case, c0_folder)
        print(f"Scored existing case_00 ({c0_case['id']}): rms={c0_rec['case_rms']:.6f} mm", flush=True)
        rec['cases'] = [c0_rec]
        c1_case = cfg['cases'][1]
        c1_folder = folder / 'case_01'
        if not (c1_folder / 'sequence_result.json').is_file():
            if c1_folder.exists():
                shutil.rmtree(c1_folder)
            c1_folder.mkdir(exist_ok=True)
            seq = c1_folder / 'sequence.json'
            save_json(seq, sequence_json(cfg, c1_case, params_4))
            args = command(cfg, c1_case, seq)
            save_json(c1_folder / 'command.json', args)
            phase = 'solver'
            launch(args, c1_folder, cfg['solver']['timeout_seconds'],
                   cfg.get('monitoring', {}).get('heartbeat_seconds', 30))
        phase = 'registration_validation'
        c1_rec = score_solved_case(ev, c1_case, c1_folder)
        print(f"Solved & scored case_01 ({c1_case['id']}): rms={c1_rec['case_rms']:.6f} mm", flush=True)
        cases = [c0_rec, c1_rec]
        weights = np.array([r['weight'] for r in cases])
        errors = np.array([r['case_rms'] for r in cases])
        obj = float(np.sqrt(np.average(errors**2, weights=weights)))
        rec['cases'] = cases
        rec['objective_mm'] = obj
        rec['valid'] = True
        rec['failure_class'] = None
        rec.pop('error', None)
        rec.pop('failure_phase', None)
    except (SolverFailure, OSError, ValueError, KeyError, TypeError) as e:
        rec['valid'] = False
        rec['error'] = str(e)
        if isinstance(e, RegistrationFailure):
            rec['failure_class'] = 'registration'
        elif isinstance(e, SolverProcessFailure):
            if e.evidence.get('timeout', False) or e.evidence.get('reason') in ('controller_elapsed_cap', 'native_time_cap_abort'):
                rec['failure_class'] = 'timeout'
            elif e.evidence.get('numerical') is True and phase == 'solver':
                rec['failure_class'] = 'solver'
            else:
                rec['failure_class'] = 'infrastructure'
            rec['solver_termination'] = e.evidence
        else:
            rec['failure_class'] = 'infrastructure'
        rec['failure_phase'] = phase
        rec['objective_mm'] = cfg['optimization']['failure_penalty_mm']
    rec['seconds'] = float(rec.get('seconds', 1730.6)) + (time.monotonic() - t0)
    rec = sanitize_for_json(rec)
    save_json(folder / 'result.json', rec)
    (out / 'evaluations.jsonl').write_text(json.dumps(rec, allow_nan=False) + '\n')
    m = {c['id']: c.get('metrics', {}) for c in rec.get('cases', [])}
    keep = ('rms_mm', 'rms_symmetric_mm', 'p95_target_to_sim_mm', 'p95_sim_to_target_mm',
            'max_target_to_sim_mm', 'max_sim_to_target_mm')
    save_json(top_res_path, {
        'attempt': 13, 'wave': prop['wave'], 'x': prop['x'], 'parameters': vals,
        'valid': bool(rec['valid']), 'failure_class': rec.get('failure_class'), 'error': rec.get('error'),
        'objective_mm': rec['objective_mm'],
        'case_rms_mm': {c['id']: c['case_rms'] for c in rec.get('cases', [])} if rec['valid'] else {},
        'case_metrics': {k: {n: v.get(n) for n in keep} for k, v in m.items()} if rec['valid'] else {},
        'seconds': rec['seconds'], 'directory': rec['directory']})
    print(f"Finished arm_B a0013: valid={rec['valid']} failure_class={rec.get('failure_class')} objective_mm={rec['objective_mm']}", flush=True)


if __name__ == '__main__':
    main()
