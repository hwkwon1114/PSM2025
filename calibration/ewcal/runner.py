"""Isolated solver runs, completion checks, objective evaluation and caching."""
from pathlib import Path
import copy
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import uuid
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from .common import ROOT, resolve, digest, save_json, Parameters
from .geometry import read_mesh, points, remove_solver_rigid_motion, align_target, write_mesh, reference_dual_areas
from .material_frame import METHOD as MATERIAL_METHOD, score_material_frame
from .progress import message, latest_diagnostic


def sanitize_for_json(obj):
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

class SolverFailure(RuntimeError):
    pass

class SolverProcessFailure(SolverFailure):
    """Explicit process evidence; unknown exits are infrastructure, not numerics."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def termination_evidence(code, cwd):
    evidence = {'return_code': code, 'numerical': False, 'reason': 'unexplained_process_exit'}
    # Only the historically observed native time-cap abort is recognized here.
    # Signals alone, and a stale/partial progress marker alone, prove nothing.
    if code != -signal.SIGABRT:
        return evidence
    lines = (cwd/'solver.log').read_text(errors='replace').splitlines()
    reports = [line for line in lines if line.startswith('[native_hybrid] ')]
    if not reports:
        return evidence
    row = dict(re.findall(r'(\w+)=([^\s]+)', reports[-1]))
    try:
        if (row['status'] != 'time_cap' or row['accepted'] != '0' or
                not all(np.isfinite(float(row[k])) for k in ('energy', 'gradient')) or
                float(row['gradient']) < 0 or
                any(int(row[k]) < 1 for k in ('total_attempts', 'total_evaluations', 'total_hessians')) or
                not any(line.startswith('  what():  zigzag_sequence: equilibrium corrector failed at cycle ')
                        for line in lines[-3:])):
            return evidence
    except (KeyError, ValueError, OverflowError):
        return evidence
    evidence.update(numerical=True, timeout=True, reason='native_time_cap_abort', native_report=row)
    return evidence


class RegistrationFailure(SolverFailure):
    """Mechanics accepted, but the unchanged bounded pose contract rejected it."""
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
    if 'timeout_seconds' in solver and '-native_hybrid_seconds' not in args:
        args['-native_hybrid_seconds'] = float(solver['timeout_seconds'])
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
                    raise SolverProcessFailure(f'Solver exceeded {timeout}s: {cwd}/solver.log',
                        {'numerical': False, 'reason': 'controller_elapsed_cap', 'timeout': True,
                         'limit_seconds': timeout, 'elapsed_seconds': time.monotonic()-started})
                try:
                    code = proc.wait(timeout=min(heartbeat_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    message('SOLVER', f'PID={proc.pid}; elapsed={time.monotonic()-started:.0f}s; '
                            + latest_diagnostic(cwd))
        except BaseException as error:
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
            if isinstance(error, SolverProcessFailure):
                error.evidence['return_code'] = proc.returncode
                save_json(cwd/'termination.json', error.evidence)
            raise
    message('SOLVER', f'exit={code}; elapsed={time.monotonic()-started:.1f}s')
    if code != 0:
        evidence = termination_evidence(code, cwd)
        save_json(cwd/'termination.json', evidence)
        raise SolverProcessFailure(f'Solver exit code {code}: {cwd}/solver.log', evidence)


def validate_mesh_topology(mesh, st_faces):
    """Reject missing/mixed cells and require exact native triangle connectivity."""
    if mesh.GetNumberOfCells() != len(st_faces):
        raise SolverFailure(f'Cell count mismatch: mesh {mesh.GetNumberOfCells()} vs state {len(st_faces)}')
    if mesh.GetNumberOfPolys() != len(st_faces):
        raise SolverFailure('VTP mesh must contain only triangle polygons matching the native faces')
    polys = mesh.GetPolys()
    if polys is None or polys.GetData() is None:
        raise SolverFailure('Missing VTP polygon connectivity')
    from vtk.util.numpy_support import vtk_to_numpy
    polys_data = vtk_to_numpy(polys.GetData())
    if polys_data.ndim != 1 or polys_data.size != 4 * len(st_faces):
        raise SolverFailure('Malformed VTP triangle connectivity length')
    cells = polys_data.reshape(-1, 4)
    if not np.all(cells[:, 0] == 3):
        raise SolverFailure('VTP polygons must all be triangles')
    if not np.array_equal(cells[:, 1:], st_faces):
        raise SolverFailure('State faces do not match VTP mesh faces in ordering and connectivity')


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
    requested_backend = config['solver']['args'].get('-equilibrium_solver')
    if requested_backend is not None and m['backend'] != requested_backend:
        raise SolverFailure('Requested equilibrium backend does not match solver output')
    for solve in m['solves']:
        code = solve['return_code']
        if not solve.get('return_code_available', False) or code < 0:
            raise SolverFailure(f'Unacceptable minimizer return code: {code}')
        if m['backend'] == 'native_hybrid':
            proof = solve.get('native_hybrid', {})
            if not isinstance(proof, dict): raise SolverFailure('Missing native_hybrid numerical evidence')
            for name in ('verification_evaluations', 'verification_hessians'):
                verification_count = proof.get(name)
                if not isinstance(verification_count, int) or isinstance(verification_count, bool) or verification_count < 1:
                    raise SolverFailure('Missing native_hybrid independent verification counts')
            gate = checks.get('max_native_hybrid_gradient_norm')
            if not isinstance(gate, (float, int)) or isinstance(gate, bool) or not np.isfinite(gate) or not 0 < gate <= 1e-13:
                raise SolverFailure('Configure a finite native_hybrid gradient gate in (0, 1e-13]')
            norm = solve['reported_eps']; raw = proof.get('gradient_norm'); target = proof.get('gradient_tolerance')
            energy = proof.get('energy')
            for value in (norm, raw, target, energy):
                if not isinstance(value, (float, int)) or isinstance(value, bool) or not np.isfinite(value):
                    raise SolverFailure('Missing or nonfinite native_hybrid energy/gradient evidence')
            if (code != 0 or proof.get('accepted') is not True or
                    proof.get('termination') != 'gradient_target' or proof.get('curvature') != 'positive_pivots' or
                    norm < 0 or norm > gate or raw != norm or not 0 < target <= gate or norm > target or
                    proof.get('verification_evaluations', 0) < 1 or proof.get('verification_hessians', 0) < 1):
                raise SolverFailure('Native hybrid numerical acceptance failed')
            mode = config['solver']['args'].get('-native_hybrid_handoff', 'fixed')
            policy = proof.get('handoff_policy')
            if policy is not None and (not isinstance(policy, dict) or policy.get('mode') != mode):
                raise SolverFailure('Native hybrid handoff mode mismatch')
            if mode == 'adaptive':
                expected_policy = {
                    'mode': 'adaptive', 'version': 'gradient-cost-v1',
                    'minimum_warmup': config['solver']['args'].get('-native_hybrid_min_warmup', 50),
                    'window': config['solver']['args'].get('-native_hybrid_window', 20),
                    'plateau_ratio': config['solver']['args'].get('-native_hybrid_plateau_ratio', .90),
                    'relative_gradient': config['solver']['args'].get('-native_hybrid_relative_gradient', .01),
                    'trial_gradient_ratio': .95, 'rate_advantage': 1.1,
                    'maximum_probes': 8, 'maximum_fallbacks': 2,
                    'newton_window': 4, 'newton_plateau_ratio': .90,
                }
                if policy != expected_policy:
                    raise SolverFailure('Adaptive native policy controls mismatch')
                for name in ('warmup_attempts', 'newton_attempts', 'probe_attempts', 'probes', 'fallbacks', 'attempts'):
                    work_value = proof.get(name)
                    if not isinstance(work_value, int) or isinstance(work_value, bool) or work_value < 0:
                        raise SolverFailure('Missing adaptive method-work counters')
                if (proof['warmup_attempts'] + proof['newton_attempts'] != proof['attempts'] or
                        proof['probe_attempts'] > proof['newton_attempts'] or
                        proof['probes'] > 8 or proof['fallbacks'] > 2):
                    raise SolverFailure('Adaptive cumulative method-work counters disagree')
                for name in ('lbfgs_seconds', 'newton_seconds'):
                    seconds_value = proof.get(name)
                    if (not isinstance(seconds_value, (float, int)) or isinstance(seconds_value, bool) or
                            not np.isfinite(seconds_value) or seconds_value < 0):
                        raise SolverFailure('Missing adaptive method elapsed accounting')
                events = proof.get('policy_events')
                if (not isinstance(events, list) or any(not isinstance(event, dict) for event in events) or
                        sum(event.get('event') == 'newton_probe' for event in events) != proof['probes'] or
                        sum(event.get('event') == 'newton_to_lbfgs' for event in events) != proof['fallbacks']):
                    raise SolverFailure('Adaptive policy-event accounting disagrees')
        elif m['backend'] in ('hlbfgs', 'hybrid') and not m['stepwise']:
            norm = solve['reported_eps']
            if not np.isfinite(norm) or norm > checks['max_hlbfgs_gradient_norm'] or norm < 0:
                raise SolverFailure(f"{m['backend']} last gradient norm {norm} exceeds configured acceptance")
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

def forward(config, x, folder):
    """Real forward mechanics only: no scan access, fitting, cache or retries."""
    parameters = Parameters(config).decode(x)
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=False)
    record = {'mode': 'forward_only', 'valid': False, 'parameters': parameters,
              'directory': str(folder), 'cases': [],
              'executable_sha256': digest(resolve(config['solver']['executable']))}
    save_json(folder/'request.json', {'parameters': parameters, 'config': config})
    try:
        for case in config['cases']:
            casefolder = folder/f'case_{len(record["cases"]):02d}'; casefolder.mkdir()
            seq = casefolder/'sequence.json'; save_json(seq, sequence_json(config, case, parameters))
            args = command(config, case, seq); save_json(casefolder/'command.json', args)
            launch(args, casefolder, config['solver']['timeout_seconds'],
                   config.get('monitoring', {}).get('heartbeat_seconds', 30))
            manifest, mesh = check_output(config, case, casefolder)
            write_mesh(mesh, casefolder/'prediction_mm.vtp')
            record['cases'].append({'id': case['id'], 'directory': str(casefolder), 'manifest': manifest})
        if not record['cases']: raise SolverFailure('No forward cases configured')
        record['valid'] = True
    except (SolverFailure, OSError, ValueError, KeyError) as e:
        record['error'] = str(e)
    save_json(folder/'result.json', record)
    return record

class Evaluator:
    @staticmethod
    def identity_inputs(config):
        """Read and validate identity inputs without creating worker artifacts."""
        settings = config.get('registration', {})
        method = settings.get('method', 'bounded_icp')
        if method not in ('bounded_icp', MATERIAL_METHOD):
            raise ValueError(f'Unknown shape scoring method: {method}')
        if method == MATERIAL_METHOD and (settings != {'method': MATERIAL_METHOD, 'metric': 'symmetric',
                                                       'require_state_export': True}):
            raise ValueError('Material-reference scoring requires only method, symmetric metric and native state')
        targets = {}
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
            targets[case['id']] = cloud
            identity['targets'][case['id']] = digest(p)
        return identity, targets

    @staticmethod
    def identity_for_config(config):
        identity, _ = Evaluator.identity_inputs(config)
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def __init__(self, config):
        self.config = config
        self.parameters = Parameters(config)
        self.base = resolve(config['output_dir']); self.base.mkdir(parents=True, exist_ok=True)
        identity, self.target = self.identity_inputs(config)
        self.identity_record = identity
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache = self.base/'cache'/self.identity
        self.cache.mkdir(parents=True, exist_ok=True)
        save_json(self.cache/'identity.json', identity)
        self.attempts = 0

    def evaluate(self, x, use_cache=True, attempt_id=None, write_to_evaluations_log=True):
        parameters = self.parameters.decode(x)
        key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        cachefile = self.cache/(key+'.json')
        if use_cache and cachefile.exists():
            record = json.loads(cachefile.read_text())
            if Path(record['directory']).is_dir() and all(
                (Path(r['directory'])/name).is_file() for r in record['cases']
                for name in ['comparison.npz', 'prediction_mm.vtp', 'sequence_result.json']):
                message('CACHE', f"RMS={record['objective_mm']:.6g} mm; {record['directory']}")
                if attempt_id is not None:
                    folder = self.base/'evaluations'/f'attempt_{attempt_id:04d}'
                    if folder.exists():
                        raise FileExistsError(f"Attempt directory already exists: {folder}")
                    folder.mkdir(parents=True, exist_ok=False)
                    attempt_record = dict(record)
                    attempt_record['attempt_id'] = attempt_id
                    attempt_record['directory'] = str(folder)
                    attempt_record['cached_from'] = record['directory']
                    save_json(folder/'result.json', attempt_record)
                    save_json(folder/'request.json', {'parameters': parameters, 'config': self.config,
                                                    'identity': self.identity, 'cached_from': record['directory']})
                    return attempt_record
                return record
        self.attempts += 1
        if attempt_id is not None:
            folder = self.base/'evaluations'/f'attempt_{attempt_id:04d}'
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"Attempt directory already exists and is occupied: {folder}")
        else:
            folder = self.base/'evaluations'/f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:10]}'
        folder.mkdir(parents=True, exist_ok=False if attempt_id is not None else True)
        message('EVALUATE', f'{folder}')
        message('PARAMETERS', ', '.join(f'{k}={v:.8g}' for k,v in parameters.items()))
        record = {'parameters': parameters, 'normalized_parameters': list(map(float, x)),
                  'identity': self.identity, 'directory': str(folder), 'valid': False, 'cases': [],
                  'attempt_id': attempt_id}
        save_json(folder/'request.json', {'parameters': parameters, 'config': self.config,
                                        'identity': self.identity})
        start = time.monotonic()
        phase = 'infrastructure'
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
                phase = 'solver'
                launch(args, casefolder, self.config['solver']['timeout_seconds'],
                       self.config.get('monitoring', {}).get('heartbeat_seconds', 30))
                phase = 'output_validation'
                message('CHECK', 'Reading meshes and checking solver completion, dimensions and convergence')
                manifest, mesh = check_output(self.config, case, casefolder)
                phase = 'state_validation'
                state_path = casefolder/'bilayer_zigzag_sequence_native_hybrid_state.json'
                metric_choice = self.config.get('registration', {}).get('metric', 'directional')
                require_export = self.config.get('require_state_export', self.config.get('registration', {}).get('require_state_export', metric_choice in ('symmetric', 'sim_to_target')))
                if require_export and not state_path.is_file():
                    raise SolverFailure('Missing required state export: bilayer_zigzag_sequence_native_hybrid_state.json')

                st_ref_mm = None
                st_faces = None
                if state_path.is_file():
                    st_data = json.loads(state_path.read_text())
                    if st_data.get('schema_version') != 1:
                        raise SolverFailure('Invalid state export schema version')
                    if st_data.get('units') not in ('metres', 'm'):
                        raise SolverFailure(f"Invalid state export units '{st_data.get('units')}'; expected 'metres'")
                    st_ref = np.asarray(st_data.get('reference_positions', []), float)
                    if len(st_ref) == 0 or st_ref.ndim != 2 or st_ref.shape[1] != 3:
                        raise SolverFailure('Invalid or empty reference_positions in state export')
                    if not np.all(np.isfinite(st_ref)):
                        raise SolverFailure('Non-finite reference_positions in state export')
                    else:
                        mesh_pts = points(mesh)
                        if len(st_ref) != len(mesh_pts):
                            raise SolverFailure(f'Vertex correspondence mismatch: state {len(st_ref)} vs mesh {len(mesh_pts)}')
                        raw_faces = st_data.get('faces', [])
                        st_faces = np.asarray(raw_faces)
                        if len(st_faces) == 0 or st_faces.ndim != 2 or st_faces.shape[1] != 3:
                            raise SolverFailure('Invalid or empty faces in state export')
                        if not np.issubdtype(st_faces.dtype, np.integer):
                            raise SolverFailure(f'Faces in state export must contain integer indices, got {st_faces.dtype}')
                        if np.any(st_faces < 0) or np.any(st_faces >= len(st_ref)):
                            raise SolverFailure('Face vertex indices out of range in state export')

                        validate_mesh_topology(mesh, st_faces)

                        st_ref_mm = st_ref * 1000.0
                        try:
                            va = reference_dual_areas(st_ref_mm, st_faces)
                        except ValueError as exc:
                            raise SolverFailure(str(exc))
                        total_area = float(np.sum(va))
                        expected_area = float(self.config.get('expected_reference_area_mm2', 60000.0))
                        area_tol = float(self.config.get('area_tolerance_rel', 0.01))
                        if require_export and (abs(total_area - expected_area) / expected_area > area_tol):
                            raise SolverFailure(f'Reference area {total_area:.2f} mm^2 deviates from expected {expected_area:.2f} mm^2 by > {area_tol*100:.1f}%')

                phase = 'registration_validation'
                material_scoring = self.config['registration'].get('method') == MATERIAL_METHOD
                message('COMPARE', 'Material-reference directions and symmetric shape metrics' if material_scoring
                        else 'Rigid registration and bidirectional shape metrics')
                if material_scoring:
                    if st_ref_mm is None or st_faces is None:
                        raise SolverFailure('Material-reference scoring requires native reference geometry')
                    metrics, mesh, aligned, distance, closest = score_material_frame(
                        mesh, self.target[case['id']], st_ref_mm, st_faces)
                else:
                    metrics, aligned, distance, closest = align_target(
                        mesh, self.target[case['id']], self.config['registration'],
                        ref_positions_mm=st_ref_mm, faces=st_faces
                    )
                if not np.isfinite(metrics['rms_mm']):
                    raise SolverFailure('Nonfinite objective')
                # Preserve accepted mechanics and measured diagnostic geometry even
                # when this case is rejected by the unchanged registration gate.
                write_mesh(mesh, casefolder/'prediction_mm.vtp')
                np.savez_compressed(casefolder/'comparison.npz', target_mm=aligned,
                                    closest_mm=closest, distance_mm=distance)

                def reject(exc_cls, msg):
                    record['rejected_case'] = {'id': case['id'], 'directory': str(casefolder),
                                               'metrics': sanitize_for_json(metrics), 'manifest': manifest}
                    raise exc_cls(msg)

                # ICP convergence/pose gates are not part of the direction-first contract.
                if metric_choice == 'symmetric' and not material_scoring:
                    pose = metrics.get('pose')
                    pose_valid = False
                    if pose is not None and isinstance(pose, (list, np.ndarray)):
                        try:
                            pose_arr = np.asarray(pose, dtype=float)
                            if pose_arr.ndim == 1 and pose_arr.shape == (6,) and np.all(np.isfinite(pose_arr)):
                                pose_valid = True
                        except (TypeError, ValueError):
                            pose_valid = False
                    if not pose_valid:
                        reject(SolverFailure, "Missing, non-finite, or malformed registration pose in registration output (must be shape (6,) finite float)")
                    metrics['pose'] = [float(v) for v in pose_arr.tolist()]

                    trace = metrics.get('trace', [])
                    if not trace or not isinstance(trace, list) or not all(isinstance(t, dict) and t.get('success', False) for t in trace):
                        reject(RegistrationFailure, "Registration unqualified: inner least_squares failed")

                    if not metrics.get('outer_converged', False):
                        reject(RegistrationFailure, f"Registration failed to converge: stop_reason={metrics.get('stop_reason', 'unknown')}")

                    rms_sym = metrics.get('rms_symmetric_mm')
                    if rms_sym is None or not isinstance(rms_sym, (int, float)) or not np.isfinite(rms_sym) or rms_sym <= 0:
                        reject(SolverFailure, "Missing, non-finite, or non-positive symmetric RMS metric in registration output (must be finite positive)")

                    # Optional pose boundary rejection gate (strictly governed by reject_at_bound)
                    if metrics.get('pose_near_bound', False) and self.config.get('registration', {}).get('reject_at_bound', True):
                        reject(RegistrationFailure, "Registration pose near bound")
                    metrics['registration_qualified'] = True
                elif 'registration_qualified' in metrics:
                    if not metrics.get('outer_converged', False):
                        reject(RegistrationFailure, f"Registration failed to converge: stop_reason={metrics.get('stop_reason', 'unknown')}")
                    if not all(t.get('success', False) for t in metrics.get('trace', [])):
                        reject(RegistrationFailure, "Registration unqualified: inner least_squares failed")
                    if not np.isfinite(metrics.get('rms_mm', np.nan)):
                        reject(RegistrationFailure, "Registration unqualified: non-finite results")
                    if self.config.get('registration', {}).get('reject_at_bound', True) and metrics.get('pose_near_bound', False):
                        reject(RegistrationFailure, 'Rigid registration reached its bounds: review scan orientation/registration range')

                if metric_choice == 'symmetric':
                    if 'rms_symmetric_mm' not in metrics or metrics['rms_symmetric_mm'] is None or not np.isfinite(metrics['rms_symmetric_mm']):
                        reject(SolverFailure, "Missing, non-finite, or non-positive symmetric RMS metric in registration output (must be finite positive)")
                    case_rms = float(metrics['rms_symmetric_mm'])
                elif metric_choice == 'sim_to_target':
                    if 'rms_sim_to_target_weighted_mm' not in metrics or not np.isfinite(metrics['rms_sim_to_target_weighted_mm']):
                        reject(RegistrationFailure, "Missing or non-finite simulation-to-target RMS metric")
                    case_rms = float(metrics['rms_sim_to_target_weighted_mm'])
                else:
                    case_rms = float(metrics['rms_mm'])

                message('COMPARE', f"RMS={case_rms:.6g} mm (symmetric={metrics.get('rms_symmetric_mm', float('nan')):.6g} mm); writing comparison")
                record['cases'].append({'id': case['id'], 'directory': str(casefolder),
                                        'metrics': sanitize_for_json(metrics), 'manifest': manifest,
                                        'case_rms': case_rms,
                                        'weight': case.get('weight', 1.0)})
            weights = np.array([r['weight'] for r in record['cases']])
            errors = np.array([r['case_rms'] for r in record['cases']])
            record['objective_mm'] = float(np.sqrt(np.average(errors**2, weights=weights)))
            if self.config.get('lhs') and (not np.isfinite(record['objective_mm']) or record['objective_mm'] <= 0):
                raise SolverFailure('LHS requires a finite positive shape objective')
            record['valid'] = True
            record['failure_class'] = None
        except (SolverFailure, OSError, ValueError, KeyError, TypeError) as e:
            record['error'] = str(e)
            if isinstance(e, RegistrationFailure):
                record['failure_class'] = 'registration'
            elif isinstance(e, SolverProcessFailure):
                if e.evidence.get('timeout', False) or e.evidence.get('reason') in ('controller_elapsed_cap', 'native_time_cap_abort'):
                    record['failure_class'] = 'timeout'
                elif e.evidence.get('numerical') is True and phase == 'solver':
                    record['failure_class'] = 'solver'
                else:
                    record['failure_class'] = 'infrastructure'
            else:
                record['failure_class'] = 'infrastructure'
            record['failure_phase'] = phase
            if record['failure_class'] != 'registration':
                pose_diag = record.pop('rejected_case', None)
                if pose_diag:
                    record['pose_diagnostics'] = pose_diag.get('metrics', {})
            if isinstance(e, SolverProcessFailure):
                record['solver_termination'] = e.evidence
            record['objective_mm'] = self.config['optimization']['failure_penalty_mm']
        record['seconds'] = time.monotonic()-start
        record = sanitize_for_json(record)
        save_json(folder/'result.json', record)
        if record['valid']:
            save_json(cachefile, record)
        if write_to_evaluations_log:
            lock_path = self.base/'.evaluations_log.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    with open(self.base/'evaluations.jsonl', 'a') as f:
                        f.write(json.dumps(record, allow_nan=False)+'\n')
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        print(f"{'OK' if record['valid'] else 'FAILED'} objective={record['objective_mm']:.6g} mm  {folder}", flush=True)
        if not record['valid']:
            print(record['error'], flush=True)
        return record
