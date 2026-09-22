"""Configuration, parameter coordinates, and reproducible file identities."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

NAMES = ('gtop', 'gbot', 'ortho_top', 'ortho_bottom')
ROOT = Path(__file__).resolve().parents[2]

def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def load_config(path):
    c = json.loads(Path(path).read_text())
    if c.get('schema_version') != 1:
        raise ValueError('Expected calibration schema_version=1')
    for name in NAMES:
        p = c['parameters'][name]
        lo, hi = p['bounds']
        if not np.isfinite([lo, hi, p['initial']]).all() or not lo < hi:
            raise ValueError(f'Invalid bounds for {name}')
        if not lo <= p['initial'] <= hi:
            raise ValueError(f'Initial {name} lies outside bounds')
        if name.startswith('g') and lo <= 0:
            raise ValueError('This pilot uses positive log-scaled growth parameters')
        if name.startswith('ortho') and (lo < -1 or hi > 1):
            raise ValueError('Orthotropy must lie in [-1,1]')
    if c['solver']['solve_mode'] not in ('final_only', 'every_cycle'):
        raise ValueError('Invalid solve_mode')
    if c['solver']['args'].get('-enable_passE') is not False:
        raise ValueError('Hit-count hardening requires enable_passE=false')
    if c['solver']['args'].get('-nsteps') != 1:
        raise ValueError('Use nsteps=1 for sequences')
    ids = [r['id'] for r in c['cases']]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Cases need unique IDs')
    for r in c['cases']:
        if r['repeat'] < 1 or int(r['repeat']) != r['repeat']:
            raise ValueError('repeat must be a positive integer')
        if r.get('weight', 1) <= 0:
            raise ValueError('Case weights must be positive')
    return c

class Parameters:
    """Optimization coordinates in [0,1]; logarithmic growth, linear ortho."""
    def __init__(self, config):
        self.spec = config['parameters']
        self.lo = np.array([self.spec[n]['bounds'][0] for n in NAMES], float)
        self.hi = np.array([self.spec[n]['bounds'][1] for n in NAMES], float)
        self.lo[:2] = np.log(self.lo[:2]); self.hi[:2] = np.log(self.hi[:2])
    def encode(self, values):
        v = np.array([values[n] for n in NAMES], float)
        if np.any(v[:2] <= 0):
            raise ValueError('Growth must be positive')
        v[:2] = np.log(v[:2])
        x = (v - self.lo) / (self.hi - self.lo)
        if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
            raise ValueError('Parameters outside configured bounds')
        return np.clip(x, 0, 1)
    def decode(self, x):
        x = np.asarray(x, float)
        if x.shape != (4,) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError('Expected four finite normalized parameters in [0,1]')
        v = self.lo + x * (self.hi - self.lo)
        v[:2] = np.exp(v[:2])
        return dict(zip(NAMES, map(float, v)))
    def initial(self):
        return self.encode({n: self.spec[n]['initial'] for n in NAMES})
