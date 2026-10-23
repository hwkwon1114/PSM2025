#!/usr/bin/env python3
"""Four-wide staged BO controller for the square-sheet dome stage (frozen policy in
docs/reports/dome_toolpath_investigation.md, "Square-sheet BO stage").

Attempts 1-2: fixed reference anchors; 3-24: frozen maximin subset of unique decoded Sobol points;
25-96: constrained-EI batches of four from ``DomeGPSurrogate`` scored at REALIZED recipe coordinates.
Every attempt is reserved on disk before launch (``attempts/attempt_NNN/request.json``); every outcome is
appended to ``ledger.jsonl`` by this single writer. Failed solves are attempts with no objective value.
No automatic retry, extension or restart: a changed identity, an infrastructure/configuration/analysis error
(state 'error', never a training label) or an interrupted attempt stops the campaign for the researcher.
"""
import argparse
import hashlib
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
for sub in ('.', 'dome_bo'):
    sys.path.insert(0, str(ROOT / sub))

import square_hatch as sh                                         # noqa: E402
import square_attempt                                              # noqa: E402
from dome_gp_surrogate import DomeGPSurrogate                      # noqa: E402

MAX_ATTEMPTS = 96
WAVE = 4
N_ANCHOR = 2
N_DESIGN = 22
DESIGN_SEED = 20261022
BO_SEED = 20261023
RESERVE_SECONDS = 4800
STEP_TIMEOUT = 3900                  # frozen: per solve step (s); solver process itself is capped at 3600 s
ANALYSIS_TIMEOUT = 600               # frozen: per analysis (s), enforced in the runner
PROPOSAL_TIMEOUT = 300               # frozen: per wave proposal (s)
MIN_SEPARATION = 0.5                 # scaled warped-space Euclidean separation inside a BO batch
RUNNER = ROOT / 'dome_bo/square_attempt.py'
IDENTITY_FILES = ['dome_bo/square_hatch.py', 'dome_bo/square_attempt.py',
                  'dome_bo/square_campaign.py', 'dome_bo/crosshatch_generator.py',
                  'dome_bo/dome_continuous_objective.py', 'dome_bo/dome_curvature.py',
                  'dome_bo/dome_gp_surrogate.py', 'dome_bo/run_dome_continuation.py',
                  'dome_bo/run_square_campaign.sbatch', 'shellsim/configs/default_adaptive_cutback.json']


class ProposalTimeout(RuntimeError):
    pass


def _proposal_alarm(signum, frame):
    raise ProposalTimeout(f'proposal exceeded {PROPOSAL_TIMEOUT} s')


def make_surrogate() -> DomeGPSurrogate:
    return DomeGPSurrogate(warp=sh.warp, n_features=sh.N_DIM, feature_names=sh.COORD_NAMES)


def anchors():
    return [sh.unit_from_physical(8, 8, 192, 192, 192, 192, 3, 3, 1.0),
            sh.unit_from_physical(8, 8, 192, 192, 192, 192, 3, 3, 0.4)]


def _scaled(points):
    raw = np.array([sh.realized_raw(sh.decode(u)) for u in points])
    phi = sh.warp(raw)
    return phi / np.maximum(phi.std(axis=0), 1e-9)


def initial_design(n_design: int = N_DESIGN, seed: int = DESIGN_SEED, n_pool_log2: int = 9, starts: int = 32):
    """Frozen maximin subset of unique decoded feasible Sobol points (no exact-LHS claim). Anchors excluded."""
    pool = qmc.Sobol(d=sh.N_DIM, scramble=True, seed=seed).random_base2(n_pool_log2)
    seen = {sh.decode_key(u) for u in anchors()}
    unique = []
    for u in pool:
        try:
            key = sh.decode_key(u)
        except ValueError:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(u)
    unique = np.array(unique)
    feats = _scaled(unique)
    rng = np.random.default_rng(seed)
    best_set, best_score = None, -1.0
    for start in rng.choice(len(unique), size=starts, replace=False):
        chosen = [int(start)]
        dmin = np.linalg.norm(feats - feats[start], axis=1)
        while len(chosen) < n_design:
            nxt = int(np.argmax(dmin))
            chosen.append(nxt)
            dmin = np.minimum(dmin, np.linalg.norm(feats - feats[nxt], axis=1))
        sub = feats[chosen]
        score = min(np.linalg.norm(sub[i] - sub[j]) for i in range(len(sub)) for j in range(i))
        if score > best_score:
            best_set, best_score = chosen, score
    return [unique[i] for i in best_set]


def select_batch(surrogate: DomeGPSurrogate, attempted_keys: set, rng, k: int = WAVE, n_pool: int = 4000,
                 min_sep: float = MIN_SEPARATION):
    """Greedy constrained-EI batch of ``k`` NEW decoded designs, CEI scored at realized coordinates, with a
    minimum separation in scaled warped space (a batching heuristic, not a qEI claim)."""
    scored = np.isfinite(surrogate.y)
    top = surrogate.X_raw[scored][np.argsort(-surrogate.y[scored])[:5]] if scored.any() else np.empty((0, sh.N_DIM))
    pool = [rng.random(sh.N_DIM) for _ in range(n_pool)]
    seen, cand = set(attempted_keys), []
    for u in pool:
        try:
            key = sh.decode_key(u)
        except ValueError:
            continue
        if key not in seen:
            seen.add(key)
            cand.append(u)
    if len(cand) < k:
        raise RuntimeError('candidate pool exhausted')
    realized = np.array([sh.realized_raw(sh.decode(u)) for u in cand])
    cei = np.atleast_1d(surrogate.constrained_expected_improvement(realized))
    if not np.isfinite(cei).all():
        raise RuntimeError('nonfinite constrained expected improvement; refusing to propose')
    scaled = surrogate.scaler.transform(sh.warp(realized))
    chosen, spare = [], np.ones(len(cand), bool)
    for _ in range(k):
        dist = np.full(len(cand), np.inf)
        for c in chosen:
            dist = np.minimum(dist, np.linalg.norm(scaled - scaled[c], axis=1))
        ok = spare & (dist >= min_sep)
        if not ok.any():                                   # separation unattainable: relax, record in return
            ok = spare
        score = np.where(ok, cei + 1e-12 * np.minimum(dist, 1e3), -np.inf)
        pick = int(np.argmax(score))
        chosen.append(pick)
        spare[pick] = False
    return [(cand[i], float(cei[i])) for i in chosen]


class Campaign:
    def __init__(self, root: Path, launcher: str = 'srun -N1 -n1 -c8 --mem=32G --exact', mesh=None,
                 fixture: bool = False, end_epoch: float = None, max_attempts: int = MAX_ATTEMPTS, runner=None,
                 step_timeout: float = STEP_TIMEOUT + ANALYSIS_TIMEOUT, proposal_timeout: int = PROPOSAL_TIMEOUT):
        self.root = Path(root).resolve()                      # absolute: workers run in other directories
        self.step_timeout = step_timeout
        self.proposal_timeout = proposal_timeout
        self.launcher = shlex.split(launcher)
        self.mesh = list(mesh or square_attempt.SCREEN_MESH)
        self.fixture = fixture
        self.runner = Path(runner) if runner else RUNNER
        self.end_epoch = end_epoch
        self.max_attempts = max_attempts
        self.ledger_path = self.root / 'ledger.jsonl'
        self.attempts_dir = self.root / 'attempts'
        self.frozen_binary = None

    # ---- identity ---------------------------------------------------------------------------
    def identity(self) -> dict:
        ident = {'binary_sha256': square_attempt.sha(square_attempt.BINARY), 'mesh': self.mesh,
                 'design_seed': DESIGN_SEED, 'bo_seed': BO_SEED, 'max_attempts': self.max_attempts,
                 'files': {f: square_attempt.sha(ROOT / f) for f in IDENTITY_FILES}}
        return ident

    def freeze_identity(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'identity.json'
        ident = self.identity()
        if path.exists():
            if json.loads(path.read_text()) != ident:
                raise RuntimeError('identity changed since the campaign was frozen; researcher decision required')
        else:
            path.write_text(json.dumps(ident, indent=2) + '\n')

    # ---- ledger -----------------------------------------------------------------------------
    def ledger(self) -> list:
        if not self.ledger_path.exists():
            return []
        return [json.loads(line) for line in self.ledger_path.read_text().splitlines() if line.strip()]

    def append(self, record: dict):
        with self.ledger_path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def reserved_dirs(self):
        return sorted(self.attempts_dir.glob('attempt_*')) if self.attempts_dir.exists() else []

    def reconcile(self):
        """Reserved attempts without an outcome are recorded as interrupted (no retry) and stop the run."""
        done = {r['attempt'] for r in self.ledger()}
        orphans = [d for d in self.reserved_dirs() if int(d.name.split('_')[1]) not in done]
        for d in orphans:
            req = json.loads((d / 'request.json').read_text())
            self.append({'attempt': req['attempt'], 'wave': req['wave'], 'source': req['source'],
                         'u': req['u'], 'realized': req['realized'], 'recipe_sha256': req['recipe_sha256'],
                         'state': 'error', 'failure_type': 'interrupted', 'z': None, 'y': None})
        if orphans:
            raise RuntimeError(f'{len(orphans)} interrupted attempt(s) recorded; no automatic retry')

    # ---- planning ---------------------------------------------------------------------------
    def attempted_keys(self) -> set:
        return {sh.decode_key(np.array(r['u'])) for r in self.ledger()}

    def next_wave(self):
        records = self.ledger()
        n = len(records)
        if n >= self.max_attempts:
            return []
        if any(r['state'] == 'error' for r in records):
            raise RuntimeError('an error record is present; researcher decision required')
        want = min(WAVE, self.max_attempts - n)
        initial = anchors() + initial_design()
        if n < len(initial):
            sources = ['anchor'] * N_ANCHOR + ['design'] * N_DESIGN
            return [(initial[i], sources[i], None) for i in range(n, min(n + want, len(initial)))]
        rng = np.random.default_rng(BO_SEED + (n - len(initial)) // WAVE)       # BO waves are counted from zero
        signal.signal(signal.SIGALRM, _proposal_alarm)
        signal.alarm(self.proposal_timeout)                                     # whole proposal: fit + selection
        try:
            sur = make_surrogate()
            sur.fit(np.array([r['realized'] for r in records]),
                    np.array([np.nan if r['y'] is None else r['y'] for r in records]),
                    np.array([r['z'] for r in records]))
            return [(u, 'bo', cei) for u, cei in select_batch(sur, self.attempted_keys(), rng, k=want)]
        finally:
            signal.alarm(0)

    # ---- execution --------------------------------------------------------------------------
    def reserve(self, number: int, wave: int, u, source: str, cei) -> Path:
        recipe = sh.decode(u)
        d = self.attempts_dir / f'attempt_{number:03d}'
        d.mkdir(parents=True, exist_ok=False)
        request = {'attempt': number, 'wave': wave, 'source': source, 'cei': cei, 'u': list(map(float, u)),
                   'realized': list(map(float, sh.realized_raw(recipe))), 'recipe': recipe,
                   'recipe_sha256': sh.recipe_sha(recipe)}
        (d / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
        return d

    def launch(self, d: Path) -> subprocess.Popen:
        cmd = self.launcher + [sys.executable, str(self.runner), '--attempt-dir', str(d), '--mesh', *self.mesh]
        if self.fixture:
            cmd.append('--fixture')
        return subprocess.Popen(cmd, stdout=(d / 'runner.log').open('w'), stderr=subprocess.STDOUT)

    def wait_wave(self, procs):
        """Wait for the wave's launcher processes under the step ceiling; a timed-out step is terminated."""
        deadline = time.monotonic() + self.step_timeout
        codes = {}
        for d, p in procs:
            try:
                codes[d] = p.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                codes[d] = 'timeout'
        return codes

    def collect(self, d: Path, code=None) -> dict:
        req = json.loads((d / 'request.json').read_text())
        base = {'attempt': req['attempt'], 'wave': req['wave'], 'source': req['source'], 'u': req['u'],
                'realized': req['realized'], 'recipe_sha256': req['recipe_sha256'], 'launcher_exit': code}
        if sh.recipe_sha(req['recipe']) != req['recipe_sha256']:
            return dict(base, state='error', failure_type='RecipeDigestMismatch', z=None, y=None)
        status = json.loads((d / 'status.json').read_text()) if (d / 'status.json').exists() else {}
        state = status.get('state')
        if state not in ('accepted', 'failed'):
            # missing/started/error status: infrastructure, configuration, artifact or analysis problem
            return dict(base, state='error', z=None, y=None, elapsed_seconds=status.get('elapsed_seconds'),
                        failure_type=status.get('failure_type') or ('LauncherTimeout' if code == 'timeout'
                                                                    else 'NoTerminalStatus'))
        if code not in (None, 0):
            # the worker published a terminal status but its launcher step ended abnormally (teardown failure,
            # hang, node problem): infrastructure, never a clean numerical outcome; stop for the researcher
            return dict(base, state='error', z=None, y=None, elapsed_seconds=status.get('elapsed_seconds'),
                        failure_type='LauncherTimeout' if code == 'timeout' else 'LauncherNonzeroExit')
        sequence = d / 'solve/sequence.json'
        written = json.loads(sequence.read_text()) if sequence.exists() else None
        if (status.get('recipe_sha256') != req['recipe_sha256'] or written is None or
                sh.recipe_sha(written) != req['recipe_sha256']):
            return dict(base, state='error', failure_type='RecipeDigestMismatch', z=None, y=None)
        if status.get('binary_sha256') != self.frozen_binary:
            return dict(base, state='error', failure_type='BinaryIdentityMismatch', z=None, y=None)
        ok = state == 'accepted'
        metrics = status.get('metrics') if ok else None
        if ok and not (metrics and np.isfinite(metrics.get('j_combined', np.nan))):
            return dict(base, state='error', failure_type='NonfiniteObjective', z=None, y=None)
        record = dict(base, state=state, z=int(ok), y=metrics['j_combined'] if ok else None,
                      qualified=bool(metrics['qualified']) if ok else False,
                      largest_connected_fraction=metrics['largest_connected_fraction'] if ok else None,
                      elapsed_seconds=status.get('elapsed_seconds'))
        if ok:
            record['metrics'] = metrics
        else:
            record['failure_type'] = status.get('failure_type', 'unknown')
        return record

    def stop_reason(self):
        records = self.ledger()
        if any(r['state'] == 'error' for r in records):
            bad = [(r['attempt'], r.get('failure_type')) for r in records if r['state'] == 'error']
            return f'infrastructure/configuration/analysis error or interruption {bad}; researcher decision required'
        if len(records) >= self.max_attempts:
            return f'{self.max_attempts} attempts reached'
        waves = sorted({r['wave'] for r in records})
        if any(r.get('qualified') for r in records):
            return 'a screening candidate passed the primary gate (verify it)'
        if len(waves) >= 2 and all(not any(r['z'] for r in records if r['wave'] == w) for w in waves[-2:]):
            return 'two consecutive waves in which every solve failed'
        return None

    def run(self):
        self.freeze_identity()
        self.frozen_binary = json.loads((self.root / 'identity.json').read_text())['binary_sha256']
        self.reconcile()
        while True:
            reason = self.stop_reason()
            if reason:
                return reason
            if self.end_epoch is not None and time.time() + RESERVE_SECONDS > self.end_epoch:
                return 'insufficient allocation reserve for another wave'
            self.freeze_identity()                               # code/binary/launcher unchanged between waves
            plan = self.next_wave()
            if not plan:
                return 'no further attempts'
            records = self.ledger()
            wave = (max([r['wave'] for r in records]) + 1) if records else 0
            number = len(records) + 1
            dirs = [self.reserve(number + i, wave, u, source, cei) for i, (u, source, cei) in enumerate(plan)]
            procs = [(d, self.launch(d)) for d in dirs]
            codes = self.wait_wave(procs)
            for d, _ in procs:
                self.append(self.collect(d, codes[d]))
            self.freeze_identity()                               # and after collection: stop on a change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
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
