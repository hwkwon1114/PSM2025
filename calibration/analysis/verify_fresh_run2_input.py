"""No-solver identity and sequence checks for the frozen fresh-search Run 2 check."""
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.bo import run_convention_corrected as cc
from calibration.ewcal.runner import Evaluator, command, sequence_json

EXPECTED = {
    'calibration/bo/run_run9_interim.py': '504832afe0526e477000b13c75ddc51bd3dc33820e365878cfea6031f1315736',
    'calibration/bo/run_run9_interim.sbatch': 'dd9d85156367eabe26b079154b81ba66311821332e0afa9402e82e42e003dfa4',
    'calibration/bo/run_convention_corrected.py': '8821a2c6d9e07b07d25e334d7edd0d3e22651f47eab786401f22508506fd6490',
    'calibration/ewcal/runner.py': '4b93b4513a5d4dbc86c4c273fefb92d328e47d339bd3e30c0391ceebf73da763',
    'calibration/ewcal/material_frame.py': 'f8bbb76709e8cae9772b715af2dbd189070f92eb4f626197f7d6be1c8d99ce17',
    'calibration/ewcal/geometry.py': 'eebb68941f953806f64a086f3b75696756be435237e5c67027b67e3e236006ea',
    'calibration/ewcal/common.py': 'abd5b2e4206b2695e692c5dbfaa0ca8aed0a42f2a1037fcc426464b5243b81c7',
    'shellsim/configs/default_adaptive_cutback.json': '94252b22fd97d165de5fa0c1b203dbf0ef5c9d77639ef5474f9fa925fc187ce3',
    'bin/shell': '522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841',
    'run/fresh_lhs_joint4/summary.json': '78deeae0d0d31fe41e77206cbae60031dad394d4e10c2987b7ea28f201a55072',
    'run/fresh_lhs_joint4/results/a0084.json': '1fd2db5d0ac66883bb71a8375a5d702eb4d9d5d3cc1d186a55151bc4b02884db',
    'run/run9_transfer_expanded/evaluations/attempt_0209/request.json': '913b83ee8238cb60edcab7d409596058848c65b2df59f44358174f75424b164d',
    'run/putong_convention_check/preserved_samples/run9.npz': 'aa1fc209fa1b1eee7929a050957c3551ec959d09ed0fec92b02769a01451890e',
    'run/putong_convention_check/preserved_samples/run9.json': '297b0f5b29e4401633a7a67f553f1a8cdc9b1885100a54a47ad942358e501bca',
    'run/run9_interim/joint_stage4_reduced4_a0016_long/forward_result.json': 'f4003e03cd4465e667ea5d375ae02d30596ab49af6e83914a6ef31884ae45eee',
}


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    actual = {path: sha(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        assert actual[path] == expected, f'identity mismatch: {path}'
    src = json.loads((ROOT / 'run/fresh_lhs_joint4/results/a0084.json').read_text())
    summary = json.loads((ROOT / 'run/fresh_lhs_joint4/summary.json').read_text())
    assert src['valid'] and summary['best'] == src and src['attempt'] == 84
    vals = src['parameters']
    cfg = cc.config('run9')
    cfg['hardening']['beta'], cfg['hardening']['floor'] = vals['beta'], vals['floor']
    out = 'run/run9_interim/fresh_lhs_joint4_a0084'
    cfg['output_dir'], cfg['report_dir'] = out, out + '/report'
    assert cfg['cases'][0]['id'] == 'run9' and cfg['cases'][0]['repeat'] == 33
    assert cfg['solver']['timeout_seconds'] == 3600
    assert cfg['solver']['args']['-native_hybrid_seconds'] == 3600
    assert cfg['registration'] == {'method': 'material_reference', 'metric': 'symmetric', 'require_state_export': True}
    _, targets = Evaluator.identity_inputs(cfg)
    assert set(targets) == {'run9'} and len(targets['run9']) == 14468
    p4 = {k: float(vals[k]) for k in cfg['parameters']}
    # Match the actual launcher, which bypasses the old positive-gbot codec.
    params = SimpleNamespace(decode=lambda _x: dict(p4)).decode([0.0] * 4)
    seq = sequence_json(cfg, cfg['cases'][0], params)
    op = seq['toolpaths'][0]['operation']
    for key in cfg['parameters']:
        assert op[key] == vals[key]
    for key in ('beta', 'floor'):
        assert seq['hardening'][key] == vals[key]
    prior_sequences = {}
    for case in ('case_00', 'case_01'):
        path = f'run/fresh_lhs_joint4/candidates/a0084/evaluations/attempt_0001/{case}/sequence.json'
        prior = json.loads((ROOT / path).read_text())
        prior_sequences[path] = sha(path)
        for key in cfg['parameters']:
            assert prior['toolpaths'][0]['operation'][key] == op[key]
        for key in ('beta', 'floor'):
            assert prior['hardening'][key] == seq['hardening'][key]
    sequence_path = ROOT / out / 'evaluations/attempt_0001/case_00/sequence.json'
    cmd = command(cfg, cfg['cases'][0], sequence_path)
    assert cmd[0] == str(ROOT / 'bin/shell')
    assert cmd[cmd.index('-native_hybrid_seconds') + 1] == '3600'
    assert not (ROOT / out / 'evaluations').exists()
    manifest = {
        'status': 'PASS', 'solver_invoked': False, 'source_attempt': 84,
        'source_result': 'run/fresh_lhs_joint4/results/a0084.json',
        'joint_objective_mm': src['objective_mm'], 'physical_parameters': vals,
        'run2_target_points': len(targets['run9']), 'hardening_sequence': seq['hardening'],
        'run2_sequence': seq, 'runner_timeout_seconds': cfg['solver']['timeout_seconds'],
        'generated_solver_command': cmd, 'input_sha256_expected_and_verified': actual,
        'prior_sequence_sha256': prior_sequences,
        'fixture_sha256': sha('calibration/analysis/verify_fresh_run2_input.py'),
        'output_directory_absent': out + '/evaluations',
    }
    path = ROOT / 'run/run9_interim/verification/fresh_a0084_fixture.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write('\n')
    print('PASS: selected candidate, exact physical inputs, Run 1/3 sequences, target/sidecar, metric, caps, no existing evaluation, and all 15 expected hashes')
    print('Run 2 target points:', len(targets['run9']))
    print('Four physical inputs:', p4)
    print('Hardening:', seq['hardening'])
    print('Generated solver command:', shlex.join(cmd))
    for name, digest in actual.items():
        print(digest, name)
    print('Manifest:', path)
    print('No solver was invoked.')


if __name__ == '__main__':
    main()
