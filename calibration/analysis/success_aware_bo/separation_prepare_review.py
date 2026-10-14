"""Stdlib-only identity capture for §17; no fit/proposal/mechanics execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT/'run/optimization_review/success_aware_bo'
OUT = PRIOR/'separation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'review_packet.json').exists():
        raise FileExistsError('Preserve existing review packet')
    snapshot = json.loads((PRIOR/'snapshot.json').read_text())
    capture = json.loads((PRIOR/'capture_verification.json').read_text())
    if sha(PRIOR/'snapshot.json') != capture['snapshot_sha256']:
        raise ValueError('Frozen snapshot changed')
    if sha(PRIOR/'source_checkpoint.json') != snapshot['source_sha256_at_capture']:
        raise ValueError('Frozen source changed')
    joint = json.loads((PRIOR/'joint_rebinding.json').read_text())
    if sha(PRIOR/'fixed_models_joint.pt') != joint['successor_sha256']:
        raise ValueError('No-refit successor changed')
    protected = {**snapshot['protected_sha256'], **capture['publication_sha256']}
    for name, expected in protected.items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Protected scientific artifact changed: {name}')
    artifacts = dict(protected)
    for path in PRIOR.iterdir():
        if path.is_file():
            artifacts[str(path.relative_to(ROOT))] = sha(path)
    for path in (PRIOR/'b1_before_fix').iterdir():
        if path.is_file():
            artifacts[str(path.relative_to(ROOT))] = sha(path)
    for name in ('model.py', 'diagnose.py', 'analyze.py', 'test_model.py', 'separation.py',
                 'test_separation.py', 'compare_separation.py', 'test_compare_separation.py',
                 'separation_prepare_review.py'):
        path = ROOT/'calibration/analysis/success_aware_bo'/name
        artifacts[str(path.relative_to(ROOT))] = sha(path)
    for path in OUT.glob('tests_*.log'):
        artifacts[str(path.relative_to(ROOT))] = sha(path)
    for name in ('calibration/tests/test_batch_bayes.py',
                 'run/stage2_bayesian_optimization/stage2_cluster_protocol.md'):
        artifacts[name] = sha(ROOT/name)
    packet = {'schema_version': 1, 'scope': '§17 isolated separation-aware proposal and three-cell no-refit diagnostic',
              'artifact_sha256': artifacts, 'protected_count': len(protected),
              'snapshot_sha256': sha(PRIOR/'snapshot.json'),
              'fixed_model_sha256': sha(PRIOR/'fixed_models_joint.pt'),
              'limits': 'No mechanics, production changes, refits or baseline regeneration. '
                        'One4CPU8GB30min/process1500s diagnostic; three optimize180s cells. '
                        'Preserve partial output, no replay; independent Astra clearance required.'}
    with (OUT/'review_packet.json').open('x') as f:
        json.dump(packet, f, indent=2, allow_nan=False)
        f.write('\n')
    print('PROTECTED', len(protected), 'drift0; PRESERVED/REVIEWED', len(artifacts),
          'packet SHA256', sha(OUT/'review_packet.json'))


if __name__ == '__main__':
    main()
