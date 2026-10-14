"""Fixed run6 scan-boundary yaw, isolated from live BO; no mechanics or GP fits."""
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np
from calibration.analysis.orient_run9_edges import edge_angles, yaw_matrix, self_test
from calibration.ewcal.common import digest, Parameters
from calibration.ewcal.geometry import points, read_mesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'run/run6_bo_edge_aligned'
OLD = ROOT / 'run/run6_bo_material_frame'
CONFIG = ROOT / 'calibration/configs/run6_bo_material_frame.json'
TARGET = ROOT / 'calibration/data/processed/run6_edge_aligned.npz'
NEW_CONFIG = ROOT / 'calibration/configs/run6_bo_edge_aligned.json'


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify(pins):
    for name, expected in pins.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def metrics(record):
    return record.get('metrics') or record['cases'][0]['metrics']


def freeze():
    if OUT.exists() or TARGET.exists() or NEW_CONFIG.exists():
        raise FileExistsError('Preserve existing edge-alignment outputs')
    cfg, state = read(CONFIG), read(OLD / 'bayes_state.json')
    if state['status'] != 'budget_complete' or state['charged_attempts'] != 80:
        raise ValueError('Need the completed 80-attempt material-frame campaign')
    observations = sorted(state['observations'], key=lambda r: r['attempt_id'])
    if [r['attempt_id'] for r in observations] != list(range(1, 81)):
        raise ValueError('Missing or duplicate historical attempt')
    prior = ROOT / 'run/optimization_review/free_translation_plan.json'
    plan = read(prior)
    original_endpoints = {e['attempt_id']: e for e in plan['endpoints']}
    names = {relative(CONFIG), relative(prior),
             'calibration/data/raw/run6.stl', 'calibration/data/raw/run6_zigzag_path.csv'}
    names.update(relative(p) for p in (ROOT / 'calibration/data/processed').glob('run6*') if p.is_file())
    names.update(relative(p) for p in (ROOT / 'calibration/ewcal').glob('*.py'))
    names.update(['calibration/analysis/orient_run9_edges.py', relative(Path(__file__)),
                  'calibration/analysis/reassess_failed_registration.py',
                  'scripts/run_run6_bo_refinement.py', 'scripts/run_run6_bo_material_frame.sbatch'])
    names.update(relative(p) for p in OLD.glob('*') if p.is_file())
    for root in ('run/stage2_bayesian_optimization', 'run/run6_bo_refinement_240'):
        names.update(relative(p) for p in (ROOT / root).glob('*') if p.is_file())
    expected = {n: h for n, h in plan['input_sha256'].items() if not n.endswith('.py')}
    names.update(expected)
    endpoints, failures = [], []
    codec = Parameters(cfg)
    for record in observations:
        att = record['attempt_id']
        if not np.allclose(codec.encode(record['parameters']), record['normalized_parameters'], atol=1e-10, rtol=0):
            raise ValueError(f'Parameter codec mismatch: {att}')
        if type(record['valid']) is not bool:
            raise ValueError('Invalid mechanics status')
        if not record['valid']:
            if record.get('failure_class') != 'timeout':
                raise ValueError('Unexpected mechanics failure class')
            failure = copy.deepcopy(record)
            if 'objective_mm' in failure:
                if failure['objective_mm'] != cfg['optimization']['failure_penalty_mm']:
                    raise ValueError('Unexpected historical timeout objective')
                # Historical evaluator publishes a penalty; it is NOT a measured loss.
                failure['historical_penalty_not_used'] = failure.pop('objective_mm')
            failures.append(failure)
        else:
            m = metrics(record)
            if m.get('scoring_method') != 'material_reference' or m.get('shape_scored') is not True:
                raise ValueError('Unexpected source scoring policy')
            if att <= 40:
                endpoint = copy.deepcopy(original_endpoints[att])
                if endpoint['parameters'] != record['parameters']:
                    raise ValueError('Historical geometry/parameter mismatch')
            else:
                if len(record['cases']) != 1 or record['cases'][0]['id'] != 'run6':
                    raise ValueError('Wrong publication case')
                folder = Path(record['cases'][0]['directory'])
                endpoint = {'attempt_id': att, 'parameters': record['parameters'], 'scale_to_mm': 1.,
                            'mesh_file': relative(folder / 'prediction_mm.vtp'),
                            'state_file': relative(folder / 'bilayer_zigzag_sequence_native_hybrid_state.json'),
                            'sequence_file': relative(folder / 'sequence_result.json')}
            endpoints.append(endpoint)
            names.update(endpoint[k] for k in ('mesh_file', 'state_file', 'sequence_file'))
        folder = record.get('directory') or record.get('publication_dir')
        if folder:
            names.update(relative(p) for p in Path(folder).glob('*.json'))
    pins = {n: digest(ROOT / n) for n in sorted(names)}
    for n, h in expected.items():
        if pins[n] != h:
            raise ValueError(f'Historical source drift: {n}')
    verify(pins)
    OUT.mkdir()
    for name in names:
        # Keep campaign/source snapshots; large geometry remains pinned in place.
        if name.endswith('.py') or name in [relative(CONFIG), relative(OLD / 'bayes_state.json')]:
            dest = OUT / 'snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
            if digest(dest) != pins[name]:
                raise ValueError('Snapshot did not match pinned source')
    payload = {'schema_version': 1, 'status': 'frozen', 'job_id': os.environ['SLURM_JOB_ID'],
               'config': cfg, 'source_campaign_sha256': pins[relative(OLD / 'bayes_state.json')],
               'source_evaluator_identity': state['evaluator_identity'],
               'observations': observations, 'endpoints': endpoints, 'failures': failures,
               'input_sha256': pins,
               'limits': 'Offline history migration only; no executable BO checkpoint or new mechanics.'}
    verify(pins)
    write_new(OUT / 'frozen_campaign.json', payload)
    print('FROZEN', len(observations), len(endpoints), len(failures), flush=True)
    return payload


def inspect():
    frozen = freeze()
    self_test()
    meta = read(ROOT / 'calibration/data/processed/run6.json')
    turned_rotation = np.diag([1., -1., -1.])
    nominal_yaw = yaw_matrix(1.5)
    old_rotation = turned_rotation @ nominal_yaw.T
    with np.load(ROOT / 'calibration/data/processed/run6.npz', allow_pickle=False) as d:
        base_cloud = d['points_mm'].copy()
    with np.load(ROOT / 'calibration/data/processed/run6_oriented.npz', allow_pickle=False) as d:
        old_cloud = d['points_mm'].copy()
    error = float(np.max(np.abs(base_cloud @ old_rotation - old_cloud)))
    if base_cloud.shape != old_cloud.shape or error > 1e-9:
        raise ValueError(f'Actual old target orientation does not match metadata recipe: {error}')
    raw = np.unique(points(read_mesh(ROOT / 'calibration/data/raw/run6.stl')), axis=0)
    tr = meta['transform']
    q = (raw - np.asarray(tr['origin_scanner_mm'])) @ np.asarray(tr['basis_columns'])
    before = (q - np.asarray(tr['shift_local_mm'])) @ old_rotation
    estimates = [edge_angles(before[:, :2], spacing) for spacing in (1., 2.)]
    angle = estimates[0]['mean_angle_deg']
    rotation = yaw_matrix(angle)
    after = before @ rotation.T
    residuals = {'long': edge_angles(after[:, :2]), 'short': edge_angles(after[:, [1, 0]])}
    sample = np.random.default_rng(2026).choice(len(raw), min(40000, len(raw)), replace=False)
    if not np.array_equal(before[:, 2], after[:, 2]):
        raise ValueError('Z-yaw changed raw height differences')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-12):
        raise ValueError('Not a proper rigid rotation')
    length_error = float(np.max(np.abs(np.linalg.norm(before[sample], axis=1) - np.linalg.norm(after[sample], axis=1))))
    verify(frozen['input_sha256'])
    np.savez_compressed(OUT / 'uncropped_display_sample.npz', before=before[sample], after=after[sample], indices=sample)
    report = {'status': 'inspection_only_no_new_target', 'job_id': os.environ['SLURM_JOB_ID'],
              'source_sha256': digest(Path(__file__)), 'source_target_recipe_error_mm': error,
              'source_metadata_limit': 'run6_oriented.json transform was inherited before turnover/yaw; verified actual NPZ mapping explicitly',
              'edge_estimates': estimates, 'additional_yaw_deg': angle, 'total_yaw_after_turnover_deg': 1.5 + angle,
              'residual_edges': residuals, 'row_vector_rotation': rotation.T.tolist(),
              'composed_scanner_basis': (np.asarray(tr['basis_columns']) @ old_rotation @ rotation.T).tolist(),
              'raw_height_unchanged_exactly': True, 'sample_norm_error_mm': length_error,
              'display_sample_count': len(sample), 'raw_vertex_count': len(raw), 'seed': 2026,
              'limits': 'Edge component selection affects fitting only, not preprocessing. No curvature or simulation fitting. Face convention retained; no physical labelled corner verified. Residuals/sensitivity are diagnostics, not an invented acceptance gate.'}
    write_new(OUT / 'edge_inspection.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    inspect()
