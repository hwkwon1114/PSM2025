"""Bounded solver-free fixtures and fixed-model acquisition numerical diagnostics."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel

from calibration.analysis.success_aware_bo.model import (
    CompletionGP, SuccessAwareLogEI, fit_models, propose, training_records)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'run/optimization_review/success_aware_bo'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify():
    snapshot = json.loads((OUT/'snapshot.json').read_text())
    capture = json.loads((OUT/'capture_verification.json').read_text())
    if sha(OUT/'snapshot.json') != capture['snapshot_sha256']:
        raise ValueError('Frozen snapshot changed')
    if sha(OUT/'source_checkpoint.json') != snapshot['source_sha256_at_capture']:
        raise ValueError('Captured source bytes changed')
    protected = dict(snapshot['protected_sha256'])
    protected.update(capture['publication_sha256'])
    for path, expected in protected.items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Protected scientific source/publication changed: {path}')
    return snapshot, len(protected)


def code_identity():
    return {name: sha(Path(__file__).with_name(name)) for name in ('model.py', 'diagnose.py')}


def save_models(loss, success, best, info, snapshot):
    bundle = {'schema_version': 1, 'loss_state': loss.state_dict(), 'success_state': success.state_dict(),
              'best_f': best.detach(), 'fit_info': info, 'snapshot_sha256': sha(OUT/'snapshot.json'),
              'code_identity': code_identity()}
    with (OUT/'fixed_models.pt').open('xb') as f:
        torch.save(bundle, f)
    return bundle


def state_signatures(bundle):
    """Bit-level fingerprints of all learned buffers/parameters and fit metadata."""
    def tensor_signature(tensor):
        tensor = tensor.detach().cpu().contiguous()
        prefix = f'{tensor.dtype}|{tuple(tensor.shape)}|'.encode()
        return hashlib.sha256(prefix+tensor.numpy().tobytes()).hexdigest()
    return {**{key: {name: tensor_signature(tensor) for name, tensor in bundle[key].items()}
               for key in ('loss_state', 'success_state')},
            'best_f': tensor_signature(bundle['best_f']),
            'fit_info': hashlib.sha256(json.dumps(bundle['fit_info'], sort_keys=True,
                                                  allow_nan=False).encode()).hexdigest()}


def rebind_models(snapshot):
    """Explicit no-refit successor for the reviewed joint-coordinate correction."""
    source, successor = OUT/'fixed_models.pt', OUT/'fixed_models_joint.pt'
    if successor.exists() or (OUT/'joint_rebinding.json').exists():
        raise FileExistsError('Preserve completed/partial rebinding; do not silently repeat')
    packet_path = OUT/'review_packet_auto_route.json'
    packet = json.loads(packet_path.read_text())
    pinned = packet['artifact_sha256']
    relative_source = str(source.relative_to(ROOT))
    if sha(source) != pinned[relative_source]:
        raise ValueError('Original fitted checkpoint differs from initial review packet')
    prior_identity = {}
    for name in ('model.py', 'diagnose.py', 'test_model.py'):
        expected = pinned[f'calibration/analysis/success_aware_bo/{name}']
        if sha(OUT/'b1_before_fix'/name) != expected:
            raise ValueError('Archived pre-correction code differs from original packet')
        if name != 'test_model.py':
            prior_identity[name] = expected
    original = torch.load(source, map_location='cpu', weights_only=True)
    if (original['schema_version'] != 1 or original['code_identity'] != prior_identity
            or original['snapshot_sha256'] != sha(OUT/'snapshot.json')):
        raise ValueError('Original model/data/code lineage differs from initial review')
    before = state_signatures(original)
    source_hash = sha(source)
    lineage = {'no_refit': True, 'reason': 'Astra B1 joint Sobol coordinate correction',
               'source_path': relative_source, 'source_sha256': source_hash,
               'initial_packet_sha256': sha(packet_path), 'prior_code_identity': prior_identity,
               'new_code_identity': code_identity(), 'snapshot_sha256': sha(OUT/'snapshot.json')}
    updated = {**original, 'code_identity': code_identity(), 'lineage': lineage}
    with successor.open('xb') as f:
        torch.save(updated, f)
    reread = torch.load(successor, map_location='cpu', weights_only=True)
    if state_signatures(reread) != before or sha(source) != source_hash:
        raise ValueError('Rebinding changed scientific state or original evidence')
    loss, success, best, fit_info = restore_models(snapshot, checkpoint=successor)
    restored = {'loss_state': loss.state_dict(), 'success_state': success.state_dict(),
                'best_f': best, 'fit_info': fit_info}
    if state_signatures(restored) != before:
        raise ValueError('Restored successor changed learned state')
    # A fresh acquisition and a restored twin must agree, without any fitting.
    probe = torch.tensor(np.random.default_rng(3101).random((2, 4, 4)), dtype=torch.double,
                         requires_grad=True)
    value = SuccessAwareLogEI(loss, success, best, samples=128, seed=31)(probe)
    value.sum().backward()
    if not torch.isfinite(value).all() or not torch.isfinite(probe.grad).all():
        raise ValueError('Nonfinite corrected acquisition or gradient')
    twin_loss, twin_success, twin_best, _ = restore_models(snapshot, checkpoint=successor)
    twin = SuccessAwareLogEI(twin_loss, twin_success, twin_best, samples=128, seed=31)(probe)
    torch.testing.assert_close(value, twin, atol=1e-12, rtol=1e-12)
    verify()
    write_new('joint_rebinding.json', {**lineage, 'successor_sha256': sha(successor),
                                     'preserved_state_signatures': before,
                                     'tensor_count': len(before['loss_state'])+len(before['success_state'])+1,
                                     'gradient_finite': True, 'restore_error': float((value-twin).abs().max()),
                                     'no_new_fit_or_proposal_optimization': True})
    print('REBIND', sha(successor), 'all learned state preserved; restore error0', flush=True)


def restore_models(snapshot, *, checkpoint=None):
    checkpoint = checkpoint or (OUT/'fixed_models_joint.pt' if (OUT/'fixed_models_joint.pt').exists()
                                else OUT/'fixed_models.pt')
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if bundle['snapshot_sha256'] != sha(OUT/'snapshot.json') or bundle['code_identity'] != code_identity():
        raise ValueError('Model/data/code identity drift')
    losses, outcomes, _ = training_records(snapshot['observations'])
    X = torch.tensor([r['normalized_parameters'] for r in losses], dtype=torch.double)
    Y = torch.tensor([-np.log(r['objective_mm']) for r in losses], dtype=torch.double).unsqueeze(-1)
    loss = SingleTaskGP(X, Y, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4)))
    loss.load_state_dict(bundle['loss_state'])
    loss.eval()
    for parameter in loss.parameters():
        parameter.requires_grad_(False)
    success = CompletionGP([r['normalized_parameters'] for r, _ in outcomes], [y for _, y in outcomes], fit=False)
    success.load_state_dict(bundle['success_state'])
    return loss, success, bundle['best_f'], bundle['fit_info']


def fixture(snapshot, protected_count):
    if (OUT/'fixed_models.pt').exists():
        raise FileExistsError('Do not silently repeat the fixed-model fit')
    torch.manual_seed(2026)
    started = time.monotonic()
    loss, success, best, info = fit_models(snapshot['observations'])
    save_models(loss, success, best, info, snapshot)
    probe = torch.tensor(np.random.default_rng(3101).random((4, 4)), dtype=torch.double, requires_grad=True)
    acq = SuccessAwareLogEI(loss, success, best, samples=64, seed=31)
    value = acq(probe)
    value.sum().backward()
    if not torch.isfinite(value).all() or not torch.isfinite(probe.grad).all():
        raise ValueError('Nonfinite acquisition/gradient')
    restored_loss, restored_success, restored_best, _ = restore_models(snapshot)
    restored = SuccessAwareLogEI(restored_loss, restored_success, restored_best, samples=64, seed=31)(probe)
    torch.testing.assert_close(value, restored, atol=1e-12, rtol=1e-12)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        proposal = propose(snapshot['observations'], loss, success, best, seed=3101, timeout=120)
    verify()
    write_new('fixture.json', {'fit_info': info, 'fixed_model_sha256': sha(OUT/'fixed_models.pt'),
                              'proposal': proposal, 'warnings': [str(w.message) for w in captured],
                              'gradient_finite': True, 'restore_error': float((value-restored).abs().max()),
                              'protected_sources_checked': protected_count, 'seconds': time.monotonic()-started})
    print('FIXTURE', info, proposal, flush=True)


def evaluate(loss, success, best, batches, samples, seeds):
    X = torch.tensor(batches, dtype=torch.double)
    evaluations = []
    with torch.no_grad():
        for seed in seeds:
            log_values = SuccessAwareLogEI(loss, success, best, samples=samples, seed=seed)(X)
            if not torch.isfinite(log_values).all():
                raise ValueError('Nonfinite evaluation')
            evaluations.append(log_values.exp().tolist())
    return {'samples': samples, 'seeds': list(seeds), 'utility_by_seed': evaluations}


def diagnostic(snapshot, protected_count):
    if (OUT/'diagnostic.json').exists() or list(OUT.glob('cell_*.json')):
        raise FileExistsError('Completed/partial diagnostic exists: preserve and consult before replay')
    loss, success, best, info = restore_models(snapshot)
    fixed_hash = sha(OUT/'fixed_models_joint.pt')
    rebinding = json.loads((OUT/'joint_rebinding.json').read_text())
    if fixed_hash != rebinding['successor_sha256']:
        raise ValueError('Corrected checkpoint differs from no-refit rebinding proof')
    started = time.monotonic()
    cells = []
    for restarts, initializations in ((10, 512), (30, 2048)):
        for seed in (3101, 3102, 3103):
            t = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                result = propose(snapshot['observations'], loss, success, best, seed=seed,
                                 restarts=restarts, initializations=initializations, timeout=180)
            result.update(seconds=time.monotonic()-t, warnings=[str(w.message) for w in captured],
                          fixed_model_sha256=fixed_hash)
            write_new(f'cell_{restarts}_{seed}.json', result)
            cells.append(result)
            print('CELL', restarts, seed, result['seconds'], result['repaired_indices'], flush=True)
    batches = [c[key] for c in cells for key in ('raw', 'final')]
    labels = [f"r{c['restarts']}_seed{c['seed']}_{key}" for c in cells for key in ('raw', 'final')]
    evaluations = []
    for samples in (256, 1024, 4096):
        result = evaluate(loss, success, best, batches, samples, range(9101, 9105))
        write_new(f'mc_{samples}.json', result)
        evaluations.append(result)
    reference = evaluate(loss, success, best, batches, 8192, range(9201, 9205))
    write_new('common_evaluation.json', reference)
    expected = np.asarray(reference['utility_by_seed'])
    mean = expected.mean(axis=0)
    std = expected.std(axis=0, ddof=1)
    repair = [{'cell': labels[2*i], 'relative_utility_change': float(mean[2*i+1]/mean[2*i]-1)}
              for i in range(len(cells))]
    contrasts = []
    for i in range(3):
        for offset, stage in ((0, 'raw'), (1, 'final')):
            low, high = 2*i+offset, 2*(i+3)+offset
            paired = expected[:, high]-expected[:, low]
            contrasts.append({'seed': 3101+i, 'stage': stage,
                              'relative_uplift': float(mean[high]/mean[low]-1),
                              'paired_mean': float(paired.mean()),
                              'paired_scramble_sd': float(paired.std(ddof=1))})
    verify()
    if sha(OUT/'fixed_models_joint.pt') != fixed_hash:
        raise ValueError('Model checkpoint changed')
    write_new('diagnostic.json', {'snapshot_sha256': sha(OUT/'snapshot.json'),
                                 'fixed_model_sha256': fixed_hash, 'fit_info': info, 'labels': labels,
                                 'cells': cells, 'mc_evaluations': evaluations, 'common_evaluation': reference,
                                 'common_mean': mean.tolist(), 'common_scramble_sd': std.tolist(),
                                 'repair_effects': repair, 'stronger_minus_current': contrasts,
                                 'completion_probability': success.probabilities(torch.tensor(batches, dtype=torch.double)),
                                 'protected_sources_checked': protected_count,
                                 'seconds': time.monotonic()-started,
                                 'limits': 'Conditional numerical diagnostics, not predictive calibration or new mechanics evidence.'})
    print('DIAGNOSTIC', contrasts, repair, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fixture', 'rebind', 'diagnostic', 'verify'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    snapshot, checked = verify()
    if args.mode == 'fixture':
        fixture(snapshot, checked)
    elif args.mode == 'rebind':
        rebind_models(snapshot)
    elif args.mode == 'diagnostic':
        diagnostic(snapshot, checked)
    else:
        print('Protected science/publications unchanged:', checked)
