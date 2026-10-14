"""Freeze completed Run6 records without locking or writing a live campaign."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'run/optimization_review/success_aware_bo'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / 'snapshot.json'
    if path.exists():
        raise FileExistsError('Do not replace the frozen investigation snapshot')
    history_path = ROOT / 'run/run6_bo_material_frame/corrected_history.json'
    history = json.loads(history_path.read_bytes())
    packet = json.loads((ROOT / 'run/optimization_review/review_packet_material_frame_bo.json').read_bytes())
    protected = dict(history['input_sha256'])
    protected.update({k: v for k, v in packet['artifact_sha256'].items()
                      if k not in ('run/stage2_bayesian_optimization/stage2_cluster_protocol.md',
                                   'run/sparse_direct_build/verification_summary.md')})
    protected[str(history_path.relative_to(ROOT))] = sha(history_path)
    for name, digest in protected.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Protected input drift: {name}')
    live_path = ROOT / 'run/run6_bo_material_frame/bayes_state.json'
    raw = live_path.read_bytes()  # The controller publishes this via atomic replacement.
    live = json.loads(raw)
    if live['history_sha256'] != sha(history_path) or live['evaluator_identity'] != history['evaluator_identity']:
        raise ValueError('Live lineage differs from the reviewed history')
    observations = sorted(live['observations'], key=lambda r: r['attempt_id'])
    seeds = [r for r in observations if r['attempt_id'] <= 40]
    if seeds != sorted(history['observations'], key=lambda r: r['attempt_id']):
        raise ValueError('Historical scientific payload differs')
    pending = [r['attempt_id'] for r in live['reserved_attempts'] if r['status'] != 'completed']
    payload = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
               'source': str(live_path.relative_to(ROOT)),
               'source_sha256_at_capture': hashlib.sha256(raw).hexdigest(),
               'source_evaluator_identity': live['evaluator_identity'],
               'config': history['config'], 'observations': observations,
               'pending_excluded': pending, 'protected_sha256': protected,
               'limits': 'Solver-free frozen diagnostic only; no live sources or campaign writes.'}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'snapshot': str(path), 'sha256': sha(path), 'completed': len(observations),
                      'valid': sum(r['valid'] for r in observations), 'pending_excluded': pending,
                      'protected_files': len(protected)}))


if __name__ == '__main__':
    main()
