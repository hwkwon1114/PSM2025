#!/usr/bin/env python
"""Separate, fixed 40-evaluation corrected240 campaign; historical events stay frozen."""
import argparse
import copy
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from calibration.ewcal import batch_bayes as batch
from calibration.ewcal.common import Parameters, save_json
from calibration.ewcal.runner import Evaluator
from calibration.ewcal.progress import message

CONFIG = ROOT / 'calibration/configs/run6_bo_refinement_240.json'
BASE = ROOT / 'run/run6_bo_refinement_240'
HISTORY = BASE / 'corrected_history.json'
META = {'base_seed': 2026, 'history_count': 32, 'new_budget': 40, 'q': 4, 'waves': 10,
        'registration_cap': 240, 'epoch_start_attempt': 33}
CAMPAIGN_TYPE = 'refinement240_qlogei'
TERMINAL = {'budget_complete', 'consecutive_failures', 'stopped_infrastructure_failure', 'deadline_stop'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def select_material_recipe():
    """Select the new isolated scoring lineage; this is not launch authorization."""
    global CONFIG, BASE, HISTORY, META, CAMPAIGN_TYPE
    CONFIG = ROOT / 'calibration/configs/run6_bo_material_frame.json'
    BASE = ROOT / 'run/run6_bo_material_frame'
    HISTORY = BASE / 'corrected_history.json'
    META = {'base_seed': 2026, 'history_count': 40, 'new_budget': 40, 'q': 4, 'waves': 10,
            'scoring_method': 'material_reference', 'epoch_start_attempt': 41}
    CAMPAIGN_TYPE = 'material_frame_qlogei'


def validate_material_science(record):
    """No pose/cost convergence or low-RMS requirement for completed shapes."""
    if record.get('failure_class') in ('solver', 'timeout', 'infrastructure'):
        if record['valid']:
            raise ValueError('Mechanics failure cannot be a scored shape')
        return
    m = metrics(record)
    if not record['valid'] or not isinstance(m, dict):
        raise ValueError('Completed material-frame shapes must have a measured loss')
    if m.get('scoring_method') != 'material_reference' or m.get('shape_scored') is not True:
        raise ValueError('Wrong shape scoring lineage')
    if any(key in m or key in record for key in ('gp_eligible', 'registration_qualified', 'pose_near_bound')):
        raise ValueError('Obsolete pose qualification override in material-frame record')
    for key in ('rms_symmetric_mm', 'rms_target_to_sim_mm', 'rms_sim_to_target_weighted_mm'):
        value = m.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError('Malformed material-frame score')
    symmetric = np.sqrt(.5 * (m['rms_target_to_sim_mm']**2 + m['rms_sim_to_target_weighted_mm']**2))
    if not np.isclose(symmetric, m['rms_symmetric_mm'], rtol=1e-12, atol=1e-12) or not np.isclose(
            record['objective_mm'], symmetric, rtol=1e-12, atol=1e-12):
        raise ValueError('Material-frame score conflicts with directional errors')
    transform = m.get('simulation_transform', {})
    r = np.asarray(transform.get('row_vector_rotation'), float)
    t = np.asarray(transform.get('translation_mm'), float)
    centroid = np.asarray(m.get('scan_centroid_original_mm'), float)
    if (r.shape != (3, 3) or t.shape != (3,) or centroid.shape != (3,) or
            not np.isfinite(r).all() or not np.isfinite(t).all() or not np.isfinite(centroid).all() or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0) or
            abs(np.linalg.det(r) - 1) >= 1e-12):
        raise ValueError('Malformed/nonproper material-frame transform')
    condition = transform.get('direction_condition_ratio')
    edge_error = transform.get('max_edge_length_change_mm')
    if (isinstance(condition, bool) or not isinstance(condition, (int, float)) or
            not np.isfinite(condition) or not 1e-10 < condition <= 1 or
            isinstance(edge_error, bool) or not isinstance(edge_error, (int, float)) or
            not np.isfinite(edge_error) or not 0 <= edge_error <= 1e-9):
        raise ValueError('Malformed/degenerate material-frame diagnostics')


def material_records(original, report, codec):
    """Training copies only: all completed shapes, no inherited pose gates or penalties."""
    if sorted(r['attempt_id'] for r in original) != list(range(1, 41)):
        raise ValueError('Need all40 unique historical attempts')
    rows = {r['attempt_id']: r for r in report['records']}
    if len(rows) != 34 or len(report['records']) != 34:
        raise ValueError('Need all34 unique completed shapes')
    out = []
    for old in sorted(original, key=lambda r: r['attempt_id']):
        r = copy.deepcopy(old)
        att = r['attempt_id']
        if not np.allclose(codec.encode(r['parameters']), r['normalized_parameters'], rtol=0, atol=1e-10):
            raise ValueError('Historical parameter codec mismatch')
        r.update(training_only=True, original_valid=old['valid'], scoring_source='direction_first_gallery')
        for key in ('gp_eligible', 'registration_qualified', 'metrics', 'rejected_case', 'pose_diagnostics'):
            r.pop(key, None)
        if att in rows:
            row = rows[att]
            if row['parameters'] != r['parameters'] or row.get('shape_scored') is not True or old.get('failure_class') in ('solver', 'timeout', 'infrastructure'):
                raise ValueError('Direction-first score/physics binding mismatch')
            m = copy.deepcopy(row['metrics'])
            m.update(scoring_method='material_reference', shape_scored=True,
                     simulation_transform=copy.deepcopy(row['simulation_transform']),
                     scan_centroid_original_mm=copy.deepcopy(report['scan_centroid_original_mm']))
            r.update(valid=True, objective_mm=m['rms_symmetric_mm'], metrics=m)
            for key in ('failure_class', 'error', 'failure_phase'):
                r.pop(key, None)
            for case in r.get('cases', []):
                case['metrics'] = copy.deepcopy(m)
                case['case_rms'] = r['objective_mm']
        else:
            if r['valid'] or r.get('failure_class') != 'timeout' or att not in (17,24,27,35,37,39):
                raise ValueError('Missing completed-shape score')
            r.pop('objective_mm', None)
            r['scoring_source'] = 'mechanics_timeout_no_loss'
        batch.validate_observation_record(r)
        validate_material_science(r)
        out.append(r)
    if set(rows) != {r['attempt_id'] for r in out if r['valid']}:
        raise ValueError('Unknown/missing historical shape')
    return out


def prepare_material_history():
    if HISTORY.exists() or (BASE / 'bayes_state.json').exists():
        raise FileExistsError('Preserve existing material-frame campaign')
    cfg = read(CONFIG)
    expected = read(ROOT / 'calibration/configs/run6_bo_refinement_240.json')
    expected['registration'] = {'method': 'material_reference', 'metric': 'symmetric', 'require_state_export': True}
    expected['optimization']['max_evaluations'] = 80
    expected['bayesian']['failure_radius'] = 0
    expected['output_dir'], expected['report_dir'] = 'run/run6_bo_material_frame', 'run/run6_bo_material_frame/report'
    if cfg != expected:
        raise ValueError('Unapproved material-frame configuration change')
    oldpath = ROOT / 'run/run6_bo_refinement_240/bayes_state.json'
    reportpath = ROOT / 'run/optimization_review/direction_frame_gallery/direction_frame_results.json'
    renderpath = reportpath.parent / 'rendered_gallery_v2/render_provenance.json'
    old, report, render = read(oldpath), read(reportpath), read(renderpath)
    if old['status'] != 'consecutive_failures' or old['charged_attempts'] != 40 or not report.get('analysis_complete'):
        raise ValueError('Historical campaign/gallery incomplete')
    if render.get('status') != 'render_complete' or render.get('input_result_sha256') != sha(reportpath) or not render.get('frozen_inputs_unchanged'):
        raise ValueError('Unverified gallery score publication')
    # Production sources intentionally change; all original scientific DATA remains bound.
    pins = {n: d for n, d in report['input_sha256'].items() if not n.endswith('.py')}
    for n, digest in pins.items():
        if sha(ROOT / n) != digest:
            raise ValueError(f'Frozen gallery data changed: {n}')
    names = [CONFIG, Path(__file__), oldpath, reportpath, renderpath,
             ROOT / 'run/optimization_review/run_run6_bo_refinement_before_material_frame.py']
    names += list((ROOT / 'calibration/ewcal').glob('*.py'))
    pins.update({str(p.relative_to(ROOT)): sha(p) for p in names})
    observations = material_records(old['observations'], report, Parameters(cfg))
    payload = {'schema_version': 1, 'metadata': META, 'config': cfg,
               'evaluator_identity': Evaluator.identity_for_config(cfg),
               'identity_inputs': Evaluator.identity_inputs(cfg)[0],
               'observations': observations, 'input_sha256': pins,
               'limits': 'Recipe migration only; future40-new ceiling is a proposal, not launch authorization; nominal material axes'}
    verify_inputs(payload)
    BASE.mkdir(parents=True, exist_ok=True)
    write_new(HISTORY, payload)
    message('MATERIAL', f'All34 losses + six mechanics timeouts migrated: {sha(HISTORY)}')


def metrics(record):
    return record.get('metrics') or (record.get('cases') or [record.get('rejected_case') or {}])[0].get('metrics')


def corrected_records(original, reassessment, legacy, lookup, codec):
    """Explicit seed-only scoring migration, never overwrite original publications."""
    original = sorted(original, key=lambda r: r['attempt_id'])
    if [r['attempt_id'] for r in original] != list(range(1, 33)):
        raise ValueError('Need exactly 32 unique historical attempts')
    rescored = {r['attempt_id']: r for r in reassessment}
    canonical = {r['attempt_id']: r for r in legacy}
    if len(rescored) != 19 or set(canonical) != {2, 4, 5}:
        raise ValueError('Incomplete or duplicate reassessment coverage')
    out, margins = [], {}
    for old in original:
        r = copy.deepcopy(old)
        att = r['attempt_id']
        if not np.allclose(codec.encode(r['parameters']), r['normalized_parameters'], rtol=0, atol=1e-10):
            raise ValueError('Historical parameter codec mismatch')
        r['scoring_source'] = 'mechanics_failed_no_registration' if r.get('failure_class') in ('solver', 'timeout', 'infrastructure') else 'original_early_converged'
        r['original_valid'] = old['valid']
        r['training_only'] = True
        m = metrics(r)
        if att in rescored:
            row = canonical.get(att, rescored[att])
            if row['parameters'] != r['parameters']:
                raise ValueError('Reassessment parameter mismatch')
            m = copy.deepcopy(row['metrics'])
            r['valid'] = bool(row['qualified'])
            r['scoring_source'] = 'canonical240_legacy_replay' if att in canonical else 'saved240_reassessment'
            r['metrics'] = m
            r.pop('objective_mm', None)
            if r['valid']:
                r.pop('failure_class', None)
                r.pop('error', None)
                r['objective_mm'] = float(m['rms_symmetric_mm'])
            else:
                r['failure_class'], r['error'] = 'registration', 'Rejected under adopted240 pose/qualification gates'
        elif r['valid']:
            if not m or not m.get('registration_qualified'):
                raise ValueError('Missing qualified historical metrics')
            # Predecessor ledger metrics omit pose; the pinned 60-step replay supplies it.
            if m.get('pose') is None:
                s = lookup[att]
                m = {**m, **{k: s[k] for k in ('pose', 'rotation_angle_deg', 'pose_near_bound', 'stop_reason', 'iterations_used')}}
                r['metrics'] = m
            if m['iterations_used'] > 60 or not m.get('registration_converged', True):
                raise ValueError('Original score cannot be reused under unchanged early stop')
        if r['valid'] and (m['pose_near_bound'] or not m['registration_qualified']):
            raise ValueError('Qualified seed contradicts registration evidence')
        eligible = r.get('failure_class') not in ('timeout', 'solver', 'infrastructure')
        margin = batch.compute_registration_margin(r) if eligible else None
        margins[str(att)] = {'valid': r['valid'], 'failure_class': r.get('failure_class'),
                             'training_eligible': eligible, 'computed_margin': margin}
        if eligible and ((margin < 0) != r['valid']):
            raise ValueError('Registration margin/label mismatch')
        batch.validate_observation_record(r)
        out.append(r)
    if sum(r['valid'] for r in out) != 13 or sum(r['training_eligible'] for r in margins.values()) != 29:
        raise ValueError('Corrected history must have 13 losses and 29 registration observations')
    return out, margins


def verify_inputs(history):
    for name, expected in history['input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Pinned refinement input changed: {name}')


def prepare():
    if META.get('scoring_method') == 'material_reference':
        return prepare_material_history()
    if HISTORY.exists() or (BASE / 'bayes_state.json').exists():
        raise FileExistsError('Preserve existing corrected campaign')
    cfg = read(CONFIG)
    oldcfg = read(ROOT / 'calibration/configs/run6_stage2_bayesian_optimization.json')
    expected = copy.deepcopy(oldcfg)
    expected['optimization']['max_evaluations'] = 72
    expected['output_dir'], expected['report_dir'] = 'run/run6_bo_refinement_240', 'run/run6_bo_refinement_240/report'
    if cfg != expected or cfg['registration']['icp_iterations'] != 240:
        raise ValueError('Unapproved scientific config change')
    original_path = ROOT / 'run/stage2_bayesian_optimization/bayes_state.json'
    if sha(original_path) != '9a372c10b18f43f69786af844cc6e504887a76ffdd5d4a4c76316fba627912c6':
        raise ValueError('Frozen original campaign changed')
    rp = ROOT / 'run/optimization_review/registration_240_reassessment.json'
    lp = ROOT / 'run/optimization_review/legacy_frame_reassessment_retry.json'
    if sha(rp) != '7056b398e7cbbda4c2daf4cc4d15d3dfb0e988bbf4988d0b15e926df661945c2' or sha(lp) != '91f10f4ce0e456a7dd812cb272520df93c0955f34d3626cefc7a1524ab655ccb':
        raise ValueError('Verified reassessment evidence changed')
    reassess, legacy = read(rp), read(lp)
    if any(d['status'] != 'complete' or not d['inputs_unchanged'] for d in (reassess, legacy)):
        raise ValueError('Incomplete reassessment')
    observations, lookup = corrected_records(read(original_path)['observations'], reassess['records'],
                                             legacy['records'], batch.load_registration_sidecar(), Parameters(cfg))
    # Rebind data, not obsolete historical Python source versions. Preserve their reports verbatim.
    names = {str(CONFIG.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)),
             'calibration/bo/run_run6_bo_refinement.sbatch', str(original_path.relative_to(ROOT)),
             str(rp.relative_to(ROOT)), str(lp.relative_to(ROOT)), cfg['solver']['executable'],
             'run/stage2_bayesian_optimization/evaluations.jsonl',
             'run/stage2_bayesian_optimization/registration_metrics_sidecar.json',
             'calibration/configs/run6_stage2_bayesian_optimization.json'}
    for plan_name in ('registration_240_reassessment_plan.json', 'legacy_frame_reassessment_retry_plan.json'):
        plan = read(ROOT / 'run/optimization_review' / plan_name)
        for n, digest in plan['input_sha256'].items():
            if n.endswith('.py'):
                continue
            if sha(ROOT / n) != digest:
                raise ValueError(f'Previously verified data changed: {n}')
            names.add(n)
    for r in read(original_path)['observations']:
        for key in ('state_file', 'mesh_file'):
            if r.get(key):
                names.add(str(Path(r[key]).resolve().relative_to(ROOT.resolve())))
        for case in (r.get('cases') or []) + ([r['rejected_case']] if r.get('rejected_case') else []):
            folder = Path(case['directory'])
            for name in ('prediction_mm.vtp', 'bilayer_zigzag_sequence_native_hybrid_state.json', 'sequence_result.json'):
                p = folder / name
                if r['valid'] or r.get('failure_class') == 'registration':
                    if not p.is_file():
                        raise ValueError(f'Missing completed mechanics evidence: {p}')
                    names.add(str(p.resolve().relative_to(ROOT.resolve())))
        pub = Path(r.get('publication_dir') or r.get('directory') or BASE)
        for p in (pub / 'result.json', pub / 'request.json'):
            if p.exists():
                names.add(str(p.resolve().relative_to(ROOT.resolve())))
    sources = {str(p.relative_to(ROOT)) for p in (ROOT / 'calibration/ewcal').glob('*.py')}
    names.update(sources)
    payload = {'schema_version': 1, 'metadata': META, 'config': cfg,
               'evaluator_identity': Evaluator.identity_for_config(cfg),
               'identity_inputs': Evaluator.identity_inputs(cfg)[0],
               'observations': observations, 'registration_lookup': lookup,
               'input_sha256': {n: sha(ROOT / n) for n in sorted(names)},
               'limits': 'Corrected training copies only; no original events relabeled; no physical identification'}
    verify_inputs(payload)
    BASE.mkdir(parents=True, exist_ok=True)
    write_new(HISTORY, payload)
    message('REFINEMENT', f'Corrected history prepared: {sha(HISTORY)}')


def scientific_payload(record):
    """Bind the complete published payload; only the worker's proposal index is annotation."""
    return json.dumps({k: v for k, v in record.items() if k != 'proposal_idx'}, sort_keys=True, allow_nan=False)


def validate_new_science(record):
    if META.get('scoring_method') == 'material_reference':
        return validate_material_science(record)
    for key in ('gp_eligible', 'registration_qualified'):
        if key in record and (type(record[key]) is not bool or record[key] != record['valid']):
            raise ValueError('Contradictory top-level qualification override')
    if record.get('failure_class') in ('solver', 'timeout', 'infrastructure'):
        return
    m = metrics(record)
    if not isinstance(m, dict):
        raise ValueError('New registration result lacks measurements')
    for key in ('pose_near_bound', 'registration_qualified', 'registration_converged', 'outer_converged'):
        if type(m.get(key)) is not bool:
            raise ValueError(f'Malformed registration flag: {key}')
    if m['registration_qualified'] != record['valid']:
        raise ValueError('Registration qualification/validity conflict')
    if type(m.get('iterations_used')) is not int or not 1 <= m['iterations_used'] <= 240 or not isinstance(m.get('stop_reason'), str) or not m['stop_reason']:
        raise ValueError('Malformed registration stopping evidence')
    if record['valid'] and (m['pose_near_bound'] or not m['registration_converged'] or not m['outer_converged']):
        raise ValueError('Valid result contradicts convergence/bound evidence')
    for key in ('rms_symmetric_mm', 'rms_target_to_sim_mm', 'rms_sim_to_target_weighted_mm', 'rotation_angle_deg'):
        v = m.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f'Malformed registration scalar: {key}')
    pose = m.get('pose')
    if not isinstance(pose, list) or len(pose) != 6 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in pose):
        raise ValueError('Malformed registration pose')
    symmetric = np.sqrt(.5 * (m['rms_target_to_sim_mm']**2 + m['rms_sim_to_target_weighted_mm']**2))
    if not np.isclose(symmetric, m['rms_symmetric_mm'], rtol=1e-12, atol=1e-12):
        raise ValueError('Symmetric score contradicts directional metrics')
    if record['valid'] and not np.isclose(record['objective_mm'], symmetric, rtol=1e-12, atol=1e-12):
        raise ValueError('Objective contradicts published symmetric score')
    batch.compute_registration_margin(record)


def verify_publication_science(state):
    for record in state['observations']:
        if record['attempt_id'] < META['epoch_start_attempt']:
            continue  # Seed copies are bound to the immutable history, not rewritten publications.
        for name in ('result.json', 'trial_result.json'):
            path = BASE / 'evaluations' / f"attempt_{record['attempt_id']:04d}" / name
            if path.exists():
                publication = read(path)
                validate_new_science(publication)
                if scientific_payload(publication) != scientific_payload(record):
                    raise ValueError('Publication/checkpoint scientific-payload conflict')


def validate_ledger(state, identity, base_seed, validate=batch.validate_observation_record, *, history):
    if state.get('schema_version') != 3 or state.get('campaign_type') != CAMPAIGN_TYPE:
        raise ValueError('Wrong campaign discriminator')
    if state.get('metadata') != META or base_seed != 2026 or state.get('evaluator_identity') != identity:
        raise ValueError('Campaign metadata/identity changed')
    if state.get('status') not in TERMINAL | {'in_progress'}:
        raise ValueError('Unknown campaign status')
    streak = state.get('proposal_order_failure_streak')
    if type(streak) is not int or streak < 0:
        raise ValueError('Malformed stopping streak')
    if state.get('stopping_epoch') != {'epoch_start_attempt': META['epoch_start_attempt']}:
        raise ValueError('Stopping epoch changed')
    indexed = {}
    for r in state['observations']:
        validate(r)
        if r['attempt_id'] >= META['epoch_start_attempt']:
            validate_new_science(r)
        if r['attempt_id'] in indexed:
            raise ValueError('Duplicate observation')
        indexed[r['attempt_id']] = r
    if [indexed.get(a) for a in range(1, META['epoch_start_attempt'])] != history['observations']:
        raise ValueError('Corrected history changed in checkpoint')
    complete = state['completed_attempts']
    if any(type(a) is not int for a in complete) or len(set(complete)) != len(complete) or set(complete) != set(indexed):
        raise ValueError('Completed observation ledger mismatch')
    reserved = state['reserved_attempts']
    if len(reserved) > META['new_budget'] or len(reserved) % 4 or [r['attempt_id'] for r in reserved] != list(range(META['epoch_start_attempt'], META['epoch_start_attempt'] + len(reserved))):
        raise ValueError('Charged reservations must be contiguous complete q4 waves within40')
    if type(state.get('charged_attempts')) is not int or state['charged_attempts'] != META['history_count'] + len(reserved):
        raise ValueError('Charged count mismatch')
    if type(state.get('waves_reserved')) is not int or state['waves_reserved'] != len(reserved) // 4:
        raise ValueError('Wave count mismatch')
    if set(indexed) - set(range(1, META['epoch_start_attempt'])) - {r['attempt_id'] for r in reserved}:
        raise ValueError('Uncharged observation')
    for r in reserved:
        a = r['attempt_id']
        validate({'attempt_id': a, 'normalized_parameters': r['x'], 'valid': False,
                  'failure_class': 'infrastructure', 'error': 'reservation'})
        folder = BASE / 'evaluations' / f'attempt_{a:04d}'
        if r['evaluator_identity'] != identity or Path(r['publication_dir']).resolve() != folder.resolve() or Path(r['result_file']).resolve() != (folder / 'result.json').resolve():
            raise ValueError('Reservation identity/path mismatch')
        if type(r['wave_idx']) is not int or type(r['proposal_idx']) is not int or (r['wave_idx'], r['proposal_idx']) != ((a-META['epoch_start_attempt'])//4+1, (a-META['epoch_start_attempt'])%4):
            raise ValueError('Reservation wave/proposal mismatch')
        if r.get('status') not in ('reserved', 'completed', 'completed_interrupted'):
            raise ValueError('Unknown reservation status')
        if a in indexed:
            if r['status'] == 'reserved' or indexed[a].get('identity') != identity or not np.allclose(indexed[a]['normalized_parameters'], r['x'], rtol=0, atol=1e-10):
                raise ValueError('Publication/reservation conflict')
        elif a < META['epoch_start_attempt'] + len(reserved) - 4:
            raise ValueError('Unresolved earlier wave')
    verify_publication_science(state)
    return META['epoch_start_attempt'] + len(reserved)


def initial_state(history, digest):
    return {'schema_version': 3, 'campaign_type': CAMPAIGN_TYPE, 'metadata': META,
            'history_sha256': digest, 'evaluator_identity': history['evaluator_identity'],
            'observations': copy.deepcopy(history['observations']), 'reserved_attempts': [],
            'completed_attempts': list(range(1, META['epoch_start_attempt'])), 'charged_attempts': META['history_count'], 'waves_reserved': 0,
            'status': 'in_progress', 'proposal_order_failure_streak': 0,
            'stopping_epoch': {'epoch_start_attempt': META['epoch_start_attempt']}, 'wave_diagnostics': []}


def recover(state, history, identity):
    validate_ledger(state, identity, 2026, history=history)
    candidate = copy.deepcopy(state)
    batch.reconcile_reservations(BASE, candidate, identity)
    validate_ledger(candidate, identity, 2026, history=history)
    indexed = {r['attempt_id']: r for r in candidate['observations']}
    def same_science(left, right, validate):
        validate(left)
        validate(right)
        return scientific_payload(left) == scientific_payload(right)
    # Compare every line, including duplicates, before the generic reader collapses them.
    batch._read_evaluations_log(BASE / 'evaluations.jsonl', indexed, require_in_ledger=True,
                                record_comparator=same_science)
    return candidate


def update_stop(state):
    streak, crossed = batch.reconstruct_proposal_order_streak(state['observations'], epoch_start_attempt=META['epoch_start_attempt'])
    state['proposal_order_failure_streak'] = streak
    if any(r['attempt_id'] >= META['epoch_start_attempt'] and r.get('failure_class') == 'infrastructure' for r in state['observations']):
        state['status'] = 'stopped_infrastructure_failure'
    elif crossed:
        state['status'] = 'consecutive_failures'
    elif state['charged_attempts'] == META['history_count'] + META['new_budget'] and len(state['observations']) == state['charged_attempts']:
        state['status'] = 'budget_complete'


def run(expected_history_sha):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.time()
    if sha(HISTORY) != expected_history_sha:
        raise ValueError('Reviewed corrected-history digest mismatch')
    history, cfg = read(HISTORY), read(CONFIG)
    verify_inputs(history)
    identity = Evaluator.identity_for_config(cfg)
    if identity != history['evaluator_identity'] or cfg != history['config']:
        raise ValueError('Reviewed evaluator identity changed')
    with batch.ExclusiveLock(BASE / '.campaign.lock'):
        path = BASE / 'bayes_state.json'
        state = read(path) if path.exists() else initial_state(history, expected_history_sha)
        if state.get('history_sha256') != expected_history_sha:
            raise ValueError('Checkpoint history digest mismatch')
        state = recover(state, history, identity)
        update_stop(state)
        save_json(path, state)
        batch.sync_evaluations_log(BASE / 'evaluations.jsonl', state['observations'])
        if state['status'] in TERMINAL:
            message('REFINEMENT', f'Terminal state {state["status"]}; zero launches')
            return
        evaluator = Evaluator(cfg)
        lookup = {int(k): v for k, v in history.get('registration_lookup', {}).items()}
        validator = lambda s, i, seed, val: validate_ledger(s, i, seed, val, history=history)
        for wave in range(state['waves_reserved'] + 1, META['waves'] + 1):
            verify_inputs(history)
            if batch.get_remaining_allocation_seconds(start, 46800) < 3900:
                state['status'] = 'deadline_stop'
                break
            next_id = validate_ledger(state, identity, 2026, history=history)
            if len(state['observations']) != next_id - 1:
                raise ValueError('Unresolved reservations prohibit proposals')
            pts, diag = batch.propose_q_batch(state['observations'], q=4, seed=2026+wave,
                                             use_constrained_gp=META.get('scoring_method') != 'material_reference',
                                             failure_radius=0.0, sidecar_lookup=lookup)
            verify_inputs(history)
            if batch.get_remaining_allocation_seconds(start, 46800) < 3900:
                state['status'] = 'deadline_stop'
                break
            state['wave_diagnostics'].append({'wave_idx': wave, 'seed': 2026+wave, 'diagnostics': diag})
            reservations = []
            for idx, pt in enumerate(pts):
                att = next_id + idx
                folder = BASE / 'evaluations' / f'attempt_{att:04d}'
                r = {'attempt_id': att, 'x': pt.tolist(), 'wave_idx': wave, 'proposal_idx': idx,
                     'evaluator_identity': identity, 'publication_dir': str(folder),
                     'result_file': str(folder / 'result.json'), 'status': 'reserved', 'created_at': time.time()}
                state['reserved_attempts'].append(r)
                reservations.append((att, pt, r, idx))
            state['waves_reserved'], state['charged_attempts'] = wave, next_id + 3
            validator(state, identity, 2026, batch.validate_observation_record)
            save_json(path, state)
            message('REFINEMENT', f'Wave{wave}/10, Attempts{next_id}–{next_id+3},40-new ceiling')
            batch._execute_wave(evaluator, reservations, state, path, BASE / 'evaluations.jsonl',
                                BASE, 2026, threading.Lock(), ledger_validator=validator)
            update_stop(state)
            save_json(path, state)
            best = min((r for r in state['observations'] if r['valid']), key=lambda r: r['objective_mm'])
            save_json(BASE / 'bayes_summary.json', {'status': state['status'], 'new_charged': state['charged_attempts']-META['history_count'],
                      'new_completed': len(state['observations'])-META['history_count'], 'best': best,
                      'history_sha256': expected_history_sha})
            message('REFINEMENT', f'Wave{wave} finished; best={best["objective_mm"]:.9g}mm, Attempt{best["attempt_id"]}, status={state["status"]}')
            if state['status'] in TERMINAL:
                break
        save_json(path, state)
        verify_inputs(history)


def pilot():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    h = read(HISTORY)
    verify_inputs(h)
    pts, diag = batch.propose_q_batch(h['observations'], q=4, seed=2027,
                                     use_constrained_gp=META.get('scoring_method') != 'material_reference',
                                     failure_radius=0.0,
                                     sidecar_lookup={int(k): v for k, v in h.get('registration_lookup', {}).items()})
    expected_count = 34 if META.get('scoring_method') == 'material_reference' else 13
    if pts.shape != (4, 4) or diag['valid_count'] != expected_count or (
            META.get('scoring_method') != 'material_reference' and diag['registration_training_count'] != 29):
        raise ValueError('Pilot training/proposal count mismatch')
    codec = Parameters(h['config'])
    write_new(BASE / 'proposal_fixture.json', {'history_sha256': sha(HISTORY), 'diagnostics': diag,
              'parameters': [codec.decode(p) for p in pts], 'limits': 'Correctness fixture, not mechanical evaluation'})
    verify_inputs(h)
    message('REFINEMENT', f'Solver-free q4 proposal fixture passed: {diag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'pilot', 'run'])
    parser.add_argument('--history-sha256')
    parser.add_argument('--material-frame', action='store_true')
    args = parser.parse_args()
    if args.material_frame:
        select_material_recipe()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'pilot':
        pilot()
    elif not args.history_sha256 or len(args.history_sha256) != 64:
        parser.error('run requires reviewed --history-sha256')
    else:
        run(args.history_sha256)
