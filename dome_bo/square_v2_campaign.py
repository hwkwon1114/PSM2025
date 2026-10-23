#!/usr/bin/env python3
"""Four-wide controller for the square-sheet dome search v2 (protocol: docs/dome_bo/square_bo_v2_protocol.md).

Maximize J_amp = J_fit * min(sag / 1.5 mm, 1) on the 2 mm mesh with the shared adaptive-cutback solver settings.
Attempts 1-2: fixed anchors; 3-18: maximin Sobol design; then qLogEI batches of four (greedy, sequential) on a
GP over the warped realized coordinates. Failed solves have no objective value and are not fitted (no classifier).
The six retained 2 mm mesh-study cells are re-scored once into ``warm_start.json`` and enter the GP as observations
(they are not attempts and do not count toward the 96-solve budget). Everything else (reservation on disk before
launch, single-writer ledger, no retry, identity checks, error vs failure) is ``square_campaign.Campaign``.
"""
import argparse
import json
import signal
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings('ignore', message='Data \\(input features\\) is not contained to the unit cube')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))

import mesh_convergence as mc          # noqa: E402
import sag_objective as so             # noqa: E402
import square_attempt as sa            # noqa: E402
import square_campaign as sc           # noqa: E402
import square_hatch as sh              # noqa: E402
import square_v2_attempt as v2         # noqa: E402

MAX_ATTEMPTS = 96
N_DESIGN = 16
DESIGN_SEED = 20261024
BO_SEED = 20261025
PLATEAU_WAVES = 4                       # researcher decision 2026-10-23
PLATEAU_GAIN = 0.05
ROOT_DEFAULT = ROOT / 'run/dome_square_v2'
CAMPAIGN_V1 = ROOT / 'run/dome_square/campaign'
RUNNER = ROOT / 'dome_bo/square_v2_attempt.py'
IDENTITY_FILES = sc.IDENTITY_FILES + [
    'dome_bo/square_v2_attempt.py', 'dome_bo/square_v2_campaign.py', 'dome_bo/sag_objective.py',
    'dome_bo/surface_fit_scan.py', 'dome_bo/mesh_convergence.py', 'dome_bo/whole_sheet_diagnostics.py',
    'dome_bo/run_square_v2_campaign.sbatch']


def warm_start_records(binary_sha: str) -> list:
    """Re-score the retained 2 mm mesh-study cells; require the same physics, mesh, tolerance and binary."""
    records = []
    for a in mc.CAND:
        cell = mc.cell_dir(a, '.01')
        status = json.loads((cell / 'status.json').read_text())
        request = json.loads((CAMPAIGN_V1 / f'attempts/attempt_{a:03d}/request.json').read_text())
        state_file = cell / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json'
        expected = {'state': 'accepted', 'mesh': list(sa.VERIFY_MESH), 'tolerance': v2.TOLERANCE,
                    'binary_sha256': binary_sha, 'recipe_sha256': request['recipe_sha256'],
                    'state_sha256': sa.sha(state_file)}
        if {k: status.get(k) for k in expected} != expected:
            raise RuntimeError(f'attempt {a}: mesh-study cell differs from the v2 solve settings')
        records.append({'attempt': a, 'u': request['u'], 'realized': request['realized'],
                        'recipe_sha256': request['recipe_sha256'], 'state_sha256': expected['state_sha256'],
                        **so.score(json.loads(state_file.read_text()))})
    return records


def propose(X_raw, y, candidates, k, seed):
    """Greedy sequential qLogEI over a discrete pool: (indices into ``candidates``, joint qLogEI of the batch so far).

    ``X_raw``/``y``: realized coordinates and J_amp of every fitted observation; ``candidates``: realized coordinates."""
    phi = sh.warp(np.asarray(X_raw))
    mean, std = phi.mean(axis=0), phi.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    X = torch.tensor((phi - mean) / std, dtype=torch.double)
    Y = torch.tensor(np.asarray(y, float), dtype=torch.double).unsqueeze(-1)
    C = torch.tensor((sh.warp(np.asarray(candidates)) - mean) / std, dtype=torch.double)
    torch.manual_seed(seed)
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    acq = qLogExpectedImprovement(model, best_f=Y.max(), sampler=SobolQMCNormalSampler(torch.Size([128]), seed=seed))
    chosen, values = [], []
    for _ in range(k):
        acq.set_X_pending(C[chosen] if chosen else None)
        with torch.no_grad():
            score = acq(C.unsqueeze(1))
        score[chosen] = -float('inf')
        if not torch.isfinite(score.max()):
            raise RuntimeError('nonfinite qLogEI; refusing to propose')
        chosen.append(int(torch.argmax(score)))
        values.append(float(score[chosen[-1]]))
    return chosen, values


class Campaign(sc.Campaign):
    def __init__(self, root, launcher='srun -N1 -n1 -c8 --mem=32G --exact', **kw):
        kw.setdefault('mesh', sa.VERIFY_MESH)
        kw.setdefault('max_attempts', MAX_ATTEMPTS)
        kw.setdefault('runner', RUNNER)
        super().__init__(root, launcher, **kw)
        self.warm_path = self.root / 'warm_start.json'

    def identity(self) -> dict:
        return {'binary_sha256': sa.sha(sa.BINARY), 'mesh': self.mesh, 'tolerance': v2.TOLERANCE,
                'design_seed': DESIGN_SEED, 'bo_seed': BO_SEED, 'max_attempts': self.max_attempts,
                'warm_start_sha256': sa.sha(self.warm_path),
                'files': {f: sa.sha(ROOT / f) for f in IDENTITY_FILES}}

    def freeze_identity(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.warm_path.exists():
            self.warm_path.write_text(json.dumps(warm_start_records(sa.sha(sa.BINARY)), indent=2) + '\n')
        super().freeze_identity()

    def warm_start(self) -> list:
        return json.loads(self.warm_path.read_text())

    def stop_reason(self):
        records = self.ledger()
        if any(r['state'] == 'error' for r in records):
            bad = [(r['attempt'], r.get('failure_type')) for r in records if r['state'] == 'error']
            return f'infrastructure/configuration/analysis error or interruption {bad}; researcher decision required'
        if len(records) >= self.max_attempts:
            return f'{self.max_attempts} attempts reached'
        waves = sorted({r['wave'] for r in records})
        if len(waves) >= 2 and all(not any(r['z'] for r in records if r['wave'] == w) for w in waves[-2:]):
            return 'two consecutive waves in which every solve failed'
        if self.plateaued(records):
            return f'best J_amp improved by <= {PLATEAU_GAIN:.0%} over {PLATEAU_WAVES} consecutive BO waves'
        return None

    def plateaued(self, records) -> bool:
        """Only BO waves count; baseline = best J_amp (warm start included) before the last PLATEAU_WAVES of them."""
        bo_waves = sorted({r['wave'] for r in records if r['source'] == 'bo'})
        if len(bo_waves) < PLATEAU_WAVES:
            return False
        warm = [w['j_amp'] for w in self.warm_start()]

        def best_through(wave):
            return max(warm + [r['y'] for r in records if r['y'] is not None and r['wave'] <= wave])

        return best_through(bo_waves[-1]) <= best_through(bo_waves[-PLATEAU_WAVES] - 1) * (1 + PLATEAU_GAIN)

    def next_wave(self):
        records = self.ledger()
        n = len(records)
        if n >= self.max_attempts:
            return []
        if any(r['state'] == 'error' for r in records):
            raise RuntimeError('an error record is present; researcher decision required')
        want = min(sc.WAVE, self.max_attempts - n)
        initial = sc.anchors() + sc.initial_design(N_DESIGN, DESIGN_SEED)
        if n < len(initial):
            sources = ['anchor'] * sc.N_ANCHOR + ['design'] * N_DESIGN
            return [(initial[i], sources[i], None) for i in range(n, min(n + want, len(initial)))]
        warm = self.warm_start()
        wave = (n - len(initial)) // sc.WAVE
        rng = np.random.default_rng(BO_SEED + wave)
        seen = self.attempted_keys() | {sh.decode_key(np.array(w['u'])) for w in warm}
        pool = []
        for u in (rng.random(sh.N_DIM) for _ in range(4000)):
            try:
                key = sh.decode_key(u)
            except ValueError:
                continue
            if key not in seen:
                seen.add(key)
                pool.append(u)
        if len(pool) < want:
            raise RuntimeError('candidate pool exhausted')
        scored = [r for r in records if r['y'] is not None]
        X_raw = [w['realized'] for w in warm] + [r['realized'] for r in scored]
        y = [w['j_amp'] for w in warm] + [r['y'] for r in scored]
        signal.signal(signal.SIGALRM, sc._proposal_alarm)
        signal.alarm(self.proposal_timeout)                                     # whole proposal: fit + selection
        try:
            picked, values = propose(X_raw, y, [sh.realized_raw(sh.decode(u)) for u in pool], want, BO_SEED + wave)
        finally:
            signal.alarm(0)
        return [(pool[i], 'bo', v) for i, v in zip(picked, values)]            # ``cei`` field = qLogEI value

    def collect(self, d: Path, code=None) -> dict:
        record = super().collect(d, code)
        if record['state'] == 'accepted':
            if not np.isfinite(record['metrics']['j_amp']):
                return dict(record, state='error', failure_type='NonfiniteObjective', z=None, y=None)
            record['y'] = record['metrics']['j_amp']
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--end-epoch', type=float, default=None)
    parser.add_argument('--launcher', default='srun -N1 -n1 -c8 --mem=32G --exact')
    parser.add_argument('--max-attempts', type=int, default=MAX_ATTEMPTS)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    camp = Campaign(args.root, args.launcher, fixture=args.fixture, end_epoch=args.end_epoch,
                    max_attempts=args.max_attempts)
    print('STOP:', camp.run())


if __name__ == '__main__':
    main()
