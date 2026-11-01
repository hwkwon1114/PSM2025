"""Solve one family member with the frozen Attempt 209 model; no fitting, no retries.

Usage: python shape_design/run_case.py STAGE VARIANT K [--mesh-mm 2.0] [--timeout 3600]
Writes run/shape_design/STAGE/VARIANT_kK/{sequence.json, command.json, shape.npz, result.json}.
Refuses to run unless code, binary, sources and the generated sequence match run/shape_design/plan/plan.json.
"""
import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.ewcal import runner
from calibration.ewcal.common import digest, save_json
from calibration.ewcal.geometry import read_mesh, points, rigid_fit
from shape_design import family

PLAN = ROOT / 'run/shape_design/plan/plan.json'
STATE = 'bilayer_zigzag_sequence_native_hybrid_state.json'


class Terminated(BaseException):
    """Scheduler SIGTERM: unwind so the solver group is killed and result.json is written."""


def on_term(signum, frame):
    raise Terminated(f'signal {signum}')


def faces_of(mesh):
    return vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(int)


def identities():
    files = [Path(family.__file__), Path(__file__), ROOT / 'shape_design/analysis.py',
             ROOT / 'shape_design/dry_run.py', ROOT / 'shape_design/run_family.sbatch',
             ROOT / 'shape_design/tests/test_family.py', ROOT / 'dome_bo/dome_curvature.py',
             ROOT / 'calibration/ewcal/runner.py', ROOT / 'calibration/ewcal/geometry.py',
             ROOT / 'calibration/ewcal/common.py', ROOT / 'bin/shell',
             family.SOURCE_SEQUENCE, family.SOURCE_CONFIG]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def case_name(variant, k, mesh_mm):
    return f'{variant}_k{k:+d}_mesh{mesh_mm:g}'


def main():
    signal.signal(signal.SIGTERM, on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('variant', choices=family.VARIANTS)
    ap.add_argument('k', type=int, choices=family.K_VALUES)
    ap.add_argument('--mesh-mm', type=float, default=2.0)
    ap.add_argument('--timeout', type=float, default=3600)
    a = ap.parse_args()
    folder = ROOT / 'run/shape_design' / a.stage / f'{a.variant}_k{a.k:+d}'
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f'Refusing to overwrite {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    record = {'stage': a.stage, 'variant': a.variant, 'k': a.k, 'mesh_mm': a.mesh_mm, 'timeout_s': a.timeout,
              'rounds_center_edge': family.rounds(a.k), 'status': 'started', 'identities': identities(),
              'plan_sha256': digest(PLAN)}
    started = time.time()
    try:
        plan = json.loads(PLAN.read_text())
        name = case_name(a.variant, a.k, a.mesh_mm)
        if plan['identities'] != record['identities'] or name not in plan['cases']:
            raise RuntimeError('Code/binary/source identities or case differ from the reviewed plan')
        if plan['cases'][name]['timeout_s'] != a.timeout:
            raise RuntimeError('Timeout differs from the reviewed plan')
        params, config = family.load_frozen()
        cfg, case = family.case_config(config, a.variant, a.k, a.mesh_mm, a.timeout)
        seq_path = folder / 'sequence.json'
        seq_path.write_text(json.dumps(family.sequence(a.variant, a.k, params, config), indent=1) + '\n')
        record['sequence_sha256'] = digest(seq_path)
        if record['sequence_sha256'] != plan['cases'][name]['sequence_sha256']:
            raise RuntimeError('Generated sequence differs from the reviewed plan')
        args = runner.command(cfg, case, seq_path)
        save_json(folder / 'command.json', args)
        runner.launch(args, folder, cfg['solver']['timeout_seconds'])
        manifest, _ = runner.check_output(cfg, case, folder)
        # Full-precision endpoint (VTP points are Float32); cross-checked against the VTP meshes.
        state = json.loads((folder / STATE).read_text())
        ref = np.asarray(state['reference_positions'], float) * 1000
        pos = np.asarray(state['positions'], float) * 1000
        faces = np.asarray(state['faces'], int)
        initial = read_mesh(folder / manifest['initial_mesh'], scale=1000)
        final = read_mesh(folder / manifest['final_mesh'], scale=1000)
        grid_v, grid_f = family.solver_grid(a.mesh_mm)
        checks = {'state_vs_initial_vtp_mm': float(np.abs(ref - points(initial)).max()),
                  'state_vs_final_vtp_mm': float(np.abs(pos - points(final)).max()),
                  'state_vs_python_grid_mm': float(np.abs(ref - grid_v).max()) if ref.shape == grid_v.shape else None}
        if (not np.array_equal(faces, faces_of(initial)) or not np.array_equal(faces, faces_of(final))
                or not np.array_equal(faces, grid_f) or checks['state_vs_python_grid_mm'] is None
                or checks['state_vs_python_grid_mm'] > 1e-9 or checks['state_vs_initial_vtp_mm'] > 1e-4
                or checks['state_vs_final_vtp_mm'] > 1e-4):
            raise runner.SolverFailure(f'State/VTP/grid correspondence failed: {checks}')
        centroids = ref[faces][:, :, :2].mean(axis=1)
        centroids -= (ref[:, :2].min(axis=0) + ref[:, :2].max(axis=0)) / 2
        expected = family.predicted_hits(centroids, a.variant, a.k)
        solver_hits = vtk_to_numpy(final.GetCellData().GetArray('total_hit_count')).astype(int)
        mismatch = int((expected != solver_hits).sum())
        record['hit_check'] = {'faces': len(faces), 'mismatched_faces': mismatch,
                               'max_hits': int(solver_hits.max()), 'mean_hits': float(solver_hits.mean())}
        record['correspondence'] = checks
        if mismatch:
            raise runner.SolverFailure(f'Solver hit counts differ from the toolpath prediction on {mismatch} faces')
        rot, t = rigid_fit(pos, ref)
        np.savez_compressed(folder / 'shape.npz', reference_mm=ref, aligned_mm=pos @ rot + t, faces=faces,
                            hits=solver_hits)
        proof = manifest['solves'][-1]['native_hybrid']
        record.update(status='accepted', cycles=manifest['cycles'],
                      proof={k: proof[k] for k in ('accepted', 'termination', 'curvature', 'gradient_norm',
                                                   'gradient_tolerance', 'energy', 'attempts', 'hessians')})
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_s'] = time.time() - started
        save_json(folder / 'result.json', record)


if __name__ == '__main__':
    main()
