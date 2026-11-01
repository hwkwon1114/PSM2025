"""Rescore saved evaluations with the revised shared-footprint double-sided surface metric (J(S,S)=0).
Preserves backups (*.pre_metric_revision.bak) before updating result.json, comparison.npz, prediction_mm.vtp,
campaign results/aNNNN.json, and evaluations.jsonl.
"""
import json
import shutil
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from calibration.ewcal.common import load_json, save_json
from calibration.ewcal.geometry import read_mesh, write_mesh
from calibration.ewcal.material_frame import score_material_frame


def backup_once(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + '.pre_metric_revision.bak')
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
    return bak if bak.exists() else path


def load_ref_and_faces(case_dir: Path):
    st = load_json(case_dir / 'bilayer_zigzag_sequence_native_hybrid_state.json')
    return np.asarray(st['reference_positions'], float) * 1000.0, np.asarray(st['faces'], int)


def rescore_case_dir(case_dir: Path, ref_pts=None, faces=None):
    vtp = case_dir / 'prediction_mm.vtp'
    if not vtp.exists():
        return None
    src_vtp = backup_once(vtp)
    src_cmp = backup_once(case_dir / 'comparison.npz')
    target_pts = np.load(src_cmp)['target_mm']
    if ref_pts is None or faces is None:
        ref_pts, faces = load_ref_and_faces(case_dir)
    mesh = read_mesh(src_vtp, 1.0)
    metrics, framed, aligned, distances, closest = score_material_frame(mesh, target_pts, ref_pts, faces)
    write_mesh(framed, vtp)
    np.savez_compressed(case_dir / 'comparison.npz', target_mm=aligned, closest_mm=closest, distance_mm=distances)
    return metrics


def rescore_convention_corrected():
    out = {}
    for run in ('run6', 'run9', 'run10'):
        root = BASE / 'run/convention_corrected' / run
        case_dir = root / 'evaluations/attempt_0001/case_00'
        m = rescore_case_dir(case_dir)
        for p in (root / 'evaluations/attempt_0001/result.json', root / 'forward_result.json'):
            backup_once(p)
            rec = load_json(p)
            rec['objective_mm'] = m['rms_mm']
            rec['cases'][0]['case_rms'] = m['rms_mm']
            rec['cases'][0]['metrics'] = m
            save_json(p, rec)
        jsonl = root / 'evaluations.jsonl'
        backup_once(jsonl)
        rec = load_json(root / 'forward_result.json')
        jsonl.write_text(json.dumps(rec, sort_keys=True) + '\n')
        out[run] = m['rms_mm']
    return out


def rescore_joint_campaign(campaign_dir: Path):
    ref_pts, faces = load_ref_and_faces(BASE / 'run/convention_corrected/run6/evaluations/attempt_0001/case_00')
    cand_root = campaign_dir / 'candidates'
    results_root = campaign_dir / 'results'
    rescored = []
    if not cand_root.exists():
        return rescored
    for a_dir in sorted(cand_root.glob('a*')):
        res_path = a_dir / 'evaluations/attempt_0001/result.json'
        if not res_path.exists():
            continue
        rec = load_json(res_path)
        if not rec.get('valid'):
            rescored.append((a_dir.name, False, None, None, None))
            continue
        backup_once(res_path)
        vals = []
        by_case_metrics = {}
        by_case_rms = {}
        for idx, case in enumerate(rec['cases']):
            c_dir = a_dir / f'evaluations/attempt_0001/case_{idx:02d}'
            cname = case.get('name') or case.get('id')
            m = rescore_case_dir(c_dir, ref_pts, faces)
            case['case_rms'] = m['rms_mm']
            case['metrics'] = m
            vals.append(m['rms_mm'])
            if cname:
                by_case_metrics[cname] = m
                by_case_rms[cname] = m['rms_mm']
        obj = float(np.sqrt(np.mean(np.square(vals))))
        rec['objective_mm'] = obj
        save_json(res_path, rec)
        for p in (a_dir / 'result.json', a_dir / 'evaluations.jsonl'):
            if p.exists():
                backup_once(p)
                if p.suffix == '.json':
                    r2 = load_json(p)
                    r2['objective_mm'] = obj
                    r2['cases'] = rec['cases']
                    save_json(p, r2)
                else:
                    p.write_text(json.dumps(rec, sort_keys=True) + '\n')
        top_res = results_root / f'{a_dir.name}.json'
        if top_res.exists():
            backup_once(top_res)
            tr = load_json(top_res)
            tr['objective_mm'] = obj
            for rk in ('case_rms_mm', 'case_rms'):
                if rk in tr:
                    for cname, v in by_case_rms.items():
                        tr[rk][cname] = v
            if 'case_metrics' in tr:
                for cname, m in by_case_metrics.items():
                    tr['case_metrics'][cname] = {k: m[k] for k in tr['case_metrics'][cname] if k in m}
            save_json(top_res, tr)
        rescored.append((a_dir.name, True, obj, vals[0], vals[1], by_case_metrics))
    top_jsonl = campaign_dir / 'evaluations.jsonl'
    if top_jsonl.exists():
        backup_once(top_jsonl)
        by_attempt = {int(x[0][1:]): x for x in rescored if x[1]}
        lines = []
        for line in top_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            att = r.get('attempt')
            if att in by_attempt:
                _, _, obj, v6, v10, by_m = by_attempt[att]
                r['objective_mm'] = obj
                for rk in ('case_rms_mm', 'case_rms'):
                    if rk in r:
                        r[rk]['run6'] = v6
                        r[rk]['run10'] = v10
                if 'case_metrics' in r:
                    for cname in ('run6', 'run10'):
                        if cname in by_m and cname in r['case_metrics']:
                            for k in list(r['case_metrics'][cname].keys()):
                                if k in by_m[cname]:
                                    r['case_metrics'][cname][k] = by_m[cname][k]
            lines.append(json.dumps(r, sort_keys=True))
        top_jsonl.write_text('\n'.join(lines) + '\n')
    summary_path = campaign_dir / 'summary.json'
    if summary_path.exists() and results_root.exists():
        backup_once(summary_path)
        srec = load_json(summary_path)
        valid_res = []
        for rp in sorted(results_root.glob('a*.json')):
            rr = load_json(rp)
            if rr.get('valid') and rr.get('objective_mm') is not None:
                valid_res.append(rr)
            if rr.get('attempt') == 1 and 'baseline_attempt1' in srec:
                srec['baseline_attempt1'] = rr
        if valid_res and 'best' in srec:
            valid_res.sort(key=lambda x: x['objective_mm'])
            srec['best'] = valid_res[0]
        save_json(summary_path, srec)
    return [(x[0], x[1], x[2], x[3], x[4]) for x in rescored]


if __name__ == '__main__':
    cc = rescore_convention_corrected()
    print('convention_corrected:', {k: round(v, 6) for k, v in cc.items()})
    for rel in ('run/joint_run6_run10', 'run/joint_stage2/arm_A', 'run/joint_stage2/arm_B'):
        cdir = BASE / rel
        rows = rescore_joint_campaign(cdir)
        valid = [x for x in rows if x[1]]
        valid.sort(key=lambda x: x[2])
        print(f'{rel}: rescored {len(valid)} valid of {len(rows)} total; top 5:')
        for row in valid[:5]:
            print(f'  {row[0]}: J={row[2]:.6f} mm (run6={row[3]:.6f}, run10={row[4]:.6f})')
