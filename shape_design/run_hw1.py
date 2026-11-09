"""One frozen Attempt 209 solve of the Hyunwoo Run 1 toolpath (card REPLICATE-HW1); no fitting, no retries.

Usage:
  python shape_design/run_hw1.py plan              # write run/shape_design/hw_run1/plan.json (no solver)
  python shape_design/run_hw1.py solve             # refuses unless identities match plan.json
Outputs in run/shape_design/hw_run1/solve_mesh2/: sequence.json, command.json, shape.npz, result.json.
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
from shape_design import family, replicate_hw1 as rep

OUT = ROOT / 'run/shape_design/hw_run1'
PLAN = OUT / 'plan.json'
STATE = 'bilayer_zigzag_sequence_native_hybrid_state.json'
THICKNESS_MM, MESH_MM, TIMEOUT_S = 0.61, 2.0, 3600.0


class Terminated(BaseException):
    """Scheduler SIGTERM: unwind so the solver group is killed and result.json is written."""


def on_term(signum, frame):
    raise Terminated(f'signal {signum}')


def faces_of(mesh):
    return vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(int)


def identities():
    files = [Path(rep.__file__), Path(family.__file__), Path(__file__), ROOT / 'shape_design/compare_hw1.py',
             ROOT / 'shape_design/tests/test_replicate_hw1.py', ROOT / 'shape_design/run_hw1.sbatch', ROOT / 'calibration/ewcal/runner.py',
             ROOT / 'calibration/ewcal/geometry.py', ROOT / 'calibration/ewcal/common.py', ROOT / 'bin/shell',
             rep.CSV_PATH, rep.SCAN_PATH, family.SOURCE_SEQUENCE, family.SOURCE_CONFIG]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def build():
    params, config = family.load_frozen()
    passes = rep.read_passes()
    seq = rep.sequence(params, config, passes)
    cfg, case = rep.case_config(config, THICKNESS_MM, MESH_MM, TIMEOUT_S, len(seq['toolpaths']))
    return passes, seq, cfg, case


def write_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        raise SystemExit(f'Refusing to overwrite {PLAN}')
    passes, seq, cfg, case = build()
    seq_path = OUT / 'plan_sequence.json'
    seq_path.write_text(json.dumps(seq, indent=1) + '\n')
    args = runner.command(cfg, case, Path('sequence.json'))
    save_json(PLAN, {'card': 'cards/REPLICATE-HW1.md', 'identities': identities(),
                     'sequence_sha256': digest(seq_path), 'command_template': args,
                     'thickness_mm': THICKNESS_MM, 'mesh_mm': MESH_MM, 'timeout_s': TIMEOUT_S})
    print(PLAN)


def solve():
    signal.signal(signal.SIGTERM, on_term)
    folder = OUT / f'solve_mesh{MESH_MM:g}'
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f'Refusing to overwrite {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    record = {'card': 'cards/REPLICATE-HW1.md', 'thickness_mm': THICKNESS_MM, 'mesh_mm': MESH_MM,
              'timeout_s': TIMEOUT_S, 'status': 'started', 'identities': identities(), 'plan_sha256': digest(PLAN)}
    started = time.time()
    try:
        plan = json.loads(PLAN.read_text())
        if plan['identities'] != record['identities']:
            raise RuntimeError('Code/binary/source identities differ from the reviewed plan')
        passes, seq, cfg, case = build()
        seq_path = folder / 'sequence.json'
        seq_path.write_text(json.dumps(seq, indent=1) + '\n')
        record['sequence_sha256'] = digest(seq_path)
        if record['sequence_sha256'] != plan['sequence_sha256']:
            raise RuntimeError('Generated sequence differs from the reviewed plan')
        args = runner.command(cfg, case, seq_path)
        save_json(folder / 'command.json', args)
        runner.launch(args, folder, cfg['solver']['timeout_seconds'])
        manifest, _ = runner.check_output(cfg, case, folder)
        state = json.loads((folder / STATE).read_text())
        ref = np.asarray(state['reference_positions'], float) * 1000
        pos = np.asarray(state['positions'], float) * 1000
        faces = np.asarray(state['faces'], int)
        initial = read_mesh(folder / manifest['initial_mesh'], scale=1000)
        final = read_mesh(folder / manifest['final_mesh'], scale=1000)
        grid_v, grid_f = family.solver_grid(MESH_MM)
        checks = {'state_vs_initial_vtp_mm': float(np.abs(ref - points(initial)).max()),
                  'state_vs_final_vtp_mm': float(np.abs(pos - points(final)).max()),
                  'state_vs_python_grid_mm': float(np.abs(ref - grid_v).max()) if ref.shape == grid_v.shape else None}
        record['correspondence'] = checks
        if (not np.array_equal(faces, faces_of(initial)) or not np.array_equal(faces, faces_of(final))
                or not np.array_equal(faces, grid_f) or checks['state_vs_python_grid_mm'] is None
                or checks['state_vs_python_grid_mm'] > 1e-9 or checks['state_vs_initial_vtp_mm'] > 1e-4
                or checks['state_vs_final_vtp_mm'] > 1e-4):
            raise runner.SolverFailure(f'State/VTP/grid correspondence failed: {checks}')
        centroids = ref[faces][:, :, :2].mean(axis=1)
        centroids -= (ref[:, :2].min(axis=0) + ref[:, :2].max(axis=0)) / 2
        expected = rep.predicted_hits(centroids, passes)          # K2: from CSV legs directly
        solver_hits = vtk_to_numpy(final.GetCellData().GetArray('total_hit_count')).astype(int)
        mismatch = int((expected != solver_hits).sum())
        record['hit_check'] = {'faces': len(faces), 'mismatched_faces': mismatch,
                               'max_hits': int(solver_hits.max()), 'mean_hits': float(solver_hits.mean()),
                               'hit_fraction': float((solver_hits > 0).mean())}
        if mismatch:
            raise runner.SolverFailure(f'Solver hit counts differ from the CSV prediction on {mismatch} faces')
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
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('plan', 'solve'))
    a = ap.parse_args()
    write_plan() if a.mode == 'plan' else solve()
