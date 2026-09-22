"""Isolated solver runs, completion checks, objective evaluation and caching."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from .common import ROOT, resolve, digest, save_json, Parameters
from .geometry import read_mesh, points, remove_solver_rigid_motion, align_target, write_mesh
from .progress import message, latest_diagnostic

class SolverFailure(RuntimeError):
    pass

def sequence_json(config, case, parameters):
    op = copy.deepcopy(case['operation'])
    op.pop('ortho', None)
    op.update(parameters)
    return {'schema_version': 1,
            'units': {'length': 'mm', 'angle': 'deg', 'growth': 'engineering_strain'},
            'defaults': config['sequence_defaults'], 'hardening': config['hardening'],
            'boundary_conditions': {'enabled': False},
            'toolpaths': [{'id': case['id'], 'repeat': case['repeat'], 'operation': op}]}

def command(config, case, sequence_path):
    solver = config['solver']
    args = copy.deepcopy(solver['args'])
    args.update(case.get('solver_args', {}))
    args.update({'-cycle_file': str(sequence_path), '-sequence_solve_mode': solver['solve_mode'],
                 '-h_total': config['sheet']['thickness_mm']/1000,
                 '-growth_type': 'zigzag_sequence', '-enable_passE': False, '-nsteps': 1,
                 '-cycle_write_state': False, '-export_stl': False})
    result = list(solver.get('command_prefix', [])) + [str(resolve(solver['executable']))]
    for key, value in args.items():
        result += [key, str(value).lower() if isinstance(value, bool) else str(value)]
    return result

def launch(args, cwd, timeout, heartbeat_seconds=30):
    heartbeat_seconds = float(heartbeat_seconds)
    if heartbeat_seconds <= 0 or not np.isfinite(heartbeat_seconds):
        raise ValueError('heartbeat_seconds must be positive and finite')
    started = time.monotonic()
    with open(cwd/'solver.log', 'wb') as log:
        proc = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        message('SOLVER', f'PID={proc.pid}; log={cwd / "solver.log"}')
        try:
            while True:
                remaining = timeout-(time.monotonic()-started)
                if remaining <= 0:
                    raise SolverFailure(f'Solver exceeded {timeout}s: {cwd}/solver.log')
                try:
                    code = proc.wait(timeout=min(heartbeat_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    message('SOLVER', f'PID={proc.pid}; elapsed={time.monotonic()-started:.0f}s; '
                            + latest_diagnostic(cwd))
        except BaseException:
            # Ctrl+C must not leave a costly solver process running in the background.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # The wrapper may exit before descendants; clean up the whole group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    message('SOLVER', f'exit={code}; elapsed={time.monotonic()-started:.1f}s')
    if code != 0:
        raise SolverFailure(f'Solver exit code {code}: {cwd}/solver.log')


def check_output(config, case, folder):
    path = folder/'sequence_result.json'
    if not path.is_file():
        raise SolverFailure('Missing sequence_result.json: rebuild the patched solver')
    m = json.loads(path.read_text())
    count = case['repeat']
    expected_solves = 1 if config['solver']['solve_mode'] == 'final_only' else count
    if m.get('schema_version') != 1 or m.get('completed') is not True or m.get('cycles') != count:
        raise SolverFailure('Incomplete or incompatible sequence manifest')
    if m['equilibrium_solves'] != expected_solves or m['solve_mode'] != config['solver']['solve_mode']:
        raise SolverFailure('Wrong equilibrium schedule')
    if len(m['solves']) != expected_solves or not np.isfinite(m['total_energy']):
        raise SolverFailure('Invalid solve records or energy')
    checks = config['solver']['acceptance']
    for solve in m['solves']:
        code = solve['return_code']
        if not solve.get('return_code_available', False) or code < 0:
            raise SolverFailure(f'Unacceptable minimizer return code: {code}')
        if m['backend'] == 'hlbfgs' and not m['stepwise']:
            norm = solve['reported_eps']
            if not np.isfinite(norm) or norm > checks['max_hlbfgs_gradient_norm'] or norm < 0:
                raise SolverFailure(f'HLBFGS last gradient norm {norm} exceeds configured acceptance')
        else:
            codes = checks.get('other_backend_success_codes', {}).get(m['backend'])
            if codes is None or code not in codes:
                raise SolverFailure('Backend convergence is not configured; inspect its return-code contract')
    with open(folder/'sequence_history.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    if [int(r['cycle']) for r in rows] != list(range(1, count+1)):
        raise SolverFailure('Growth-history cycle coverage is incomplete')
    if sum(int(r['solved']) for r in rows) != expected_solves:
        raise SolverFailure('Growth history and manifest disagree')
    def safe_mesh(key):
        p = (folder/m[key]).resolve()
        if p.parent != folder.resolve() or not p.is_file():
            raise SolverFailure(f'Invalid {key} path')
        return read_mesh(p, scale=1000)
    initial, final = safe_mesh('initial_mesh'), safe_mesh('final_mesh')
    extent = np.ptp(points(initial), axis=0)
    wanted = np.array(config['sheet']['size_mm'])
    if not np.allclose(extent[:2], wanted, rtol=0, atol=.1) or extent[2] > .01:
        raise SolverFailure(f'Initial mesh extent {extent} mm; expected {wanted} mm and flat z. Check lx/ly conventions.')
    for i in range(final.GetCellData().GetNumberOfArrays()):
        data = vtk_to_numpy(final.GetCellData().GetArray(i))
        if not np.isfinite(data).all():
            raise SolverFailure('Nonfinite final cell field')
    return m, remove_solver_rigid_motion(final, initial)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.parameters = Parameters(config)
        self.base = resolve(config['output_dir']); self.base.mkdir(parents=True, exist_ok=True)
        self.target = {}
        identity = {'config': config, 'executable': digest(resolve(config['solver']['executable'])),
                    'targets': {}, 'python_sources': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                    'environment': {k: os.environ.get(k, '') for k in ['CONDA_PREFIX','LD_LIBRARY_PATH']}}
        import scipy, vtk
        identity['versions'] = {'numpy': np.__version__, 'scipy': scipy.__version__, 'vtk': vtk.vtkVersion.GetVTKVersion()}
        for case in config['cases']:
            p = resolve(case['target'])
            with np.load(p, allow_pickle=False) as data:
                cloud = data['points_mm'].copy()
            if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 10 or not np.isfinite(cloud).all():
                raise ValueError(f'Invalid target cloud {p}')
            sidecar = json.loads(p.with_suffix('.json').read_text())
            if sidecar['target_sha256'] != digest(p) or sidecar['preprocess'] != config['preprocess'] or sidecar['orientation'] != case.get('scan_orientation', {}):
                raise ValueError(f'Target preparation settings changed: run prepare again for {case["id"]}')
            self.target[case['id']] = cloud
            identity['targets'][case['id']] = digest(p)
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache = self.base/'cache'/self.identity
        self.cache.mkdir(parents=True, exist_ok=True)
        save_json(self.cache/'identity.json', identity)
        self.attempts = 0

    def evaluate(self, x, use_cache=True):
        parameters = self.parameters.decode(x)
        key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        cachefile = self.cache/(key+'.json')
        if use_cache and cachefile.exists():
            record = json.loads(cachefile.read_text())
            if Path(record['directory']).is_dir() and all(
                (Path(r['directory'])/name).is_file() for r in record['cases']
                for name in ['comparison.npz', 'prediction_mm.vtp', 'sequence_result.json']):
                message('CACHE', f"RMS={record['objective_mm']:.6g} mm; {record['directory']}")
                return record
        self.attempts += 1
        folder = self.base/'evaluations'/f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:10]}'
        folder.mkdir(parents=True)
        message('EVALUATE', f'{folder}')
        message('PARAMETERS', ', '.join(f'{k}={v:.8g}' for k,v in parameters.items()))
        record = {'parameters': parameters, 'normalized_parameters': list(map(float, x)),
                  'identity': self.identity, 'directory': str(folder), 'valid': False, 'cases': []}
        save_json(folder/'request.json', {'parameters': parameters, 'config': self.config})
        start = time.monotonic()
        try:
            for case in self.config['cases']:
                # Case IDs are labels; filesystem components are deliberately generated.
                casefolder = folder/f'case_{len(record["cases"]):02d}'
                casefolder.mkdir()
                message('CASE', f"{case['id']}; {casefolder}")
                seq = casefolder/'sequence.json'
                save_json(seq, sequence_json(self.config, case, parameters))
                args = command(self.config, case, seq)
                save_json(casefolder/'command.json', args)
                launch(args, casefolder, self.config['solver']['timeout_seconds'],
                       self.config.get('monitoring', {}).get('heartbeat_seconds', 30))
                message('CHECK', 'Reading meshes and checking solver completion, dimensions and convergence')
                manifest, mesh = check_output(self.config, case, casefolder)
                message('COMPARE', 'Rigid registration and point-to-triangle distances')
                metrics, aligned, distance, closest = align_target(mesh, self.target[case['id']], self.config['registration'])
                if not np.isfinite(metrics['rms_mm']):
                    raise SolverFailure('Nonfinite objective')
                if metrics['pose_near_bound'] and self.config['registration']['reject_at_bound']:
                    raise SolverFailure('Rigid registration reached its bounds: review scan orientation/registration range')
                message('COMPARE', f"RMS={metrics['rms_mm']:.6g} mm; writing comparison")
                write_mesh(mesh, casefolder/'prediction_mm.vtp')
                np.savez_compressed(casefolder/'comparison.npz', target_mm=aligned,
                                    closest_mm=closest, distance_mm=distance)
                record['cases'].append({'id': case['id'], 'directory': str(casefolder),
                                        'metrics': metrics, 'manifest': manifest,
                                        'weight': case.get('weight', 1.0)})
            weights = np.array([r['weight'] for r in record['cases']])
            errors = np.array([r['metrics']['rms_mm'] for r in record['cases']])
            record['objective_mm'] = float(np.sqrt(np.average(errors**2, weights=weights)))
            record['valid'] = True
        except (SolverFailure, OSError, ValueError, KeyError) as e:
            record['error'] = str(e)
            record['objective_mm'] = self.config['optimization']['failure_penalty_mm']
        record['seconds'] = time.monotonic()-start
        save_json(folder/'result.json', record)
        if record['valid']:
            save_json(cachefile, record)
        with open(self.base/'evaluations.jsonl', 'a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        print(f"{'OK' if record['valid'] else 'FAILED'} objective={record['objective_mm']:.6g} mm  {folder}", flush=True)
        if not record['valid']:
            print(record['error'], flush=True)
        return record
