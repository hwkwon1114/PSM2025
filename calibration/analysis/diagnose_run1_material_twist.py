"""Measure Run 1 model height cross-terms in the fixed material reference.

Uses the already-snapshotted valid candidates from diagnose_run1_twist.py.
No solver, optimization, Run 2 access, or source-result modifications.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.diagnose_run1_twist import basis
from calibration.ewcal.geometry import points, read_mesh, reference_dual_areas


def sha(data):
    return hashlib.sha256(data).hexdigest()


def coefficients(ref, faces, height):
    design = basis(ref[:, :2])
    root_weight = np.sqrt(reference_dual_areas(ref, faces))
    return np.linalg.lstsq(design * root_weight[:, None], height * root_weight, rcond=None)[0]


def main(input_path, output_path):
    source = json.loads(input_path.read_text())
    rows = []
    for record in source['records']:
        attempt = record['attempt']
        folder = ROOT / f'run/fresh_lhs_joint4/candidates/a{attempt:04d}/evaluations/attempt_0001/case_00'
        state_bytes = (folder / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_bytes()
        state = json.loads(state_bytes)
        ref = np.asarray(state['reference_positions'], float) * 1000
        native = np.asarray(state['positions'], float) * 1000
        faces = np.asarray(state['faces'], int)
        mesh_path = folder / 'prediction_mm.vtp'
        p = points(read_mesh(mesh_path))
        assert p.shape == ref.shape == native.shape and np.isfinite(p).all()
        max_edge_error = 0.0
        for i, j in ((0, 1), (1, 2), (2, 0)):
            framed = np.linalg.norm(p[faces[:, i]] - p[faces[:, j]], axis=1)
            original = np.linalg.norm(native[faces[:, i]] - native[faces[:, j]], axis=1)
            max_edge_error = max(max_edge_error, float(np.max(np.abs(framed - original))))
        if max_edge_error >= 1e-3:
            raise ValueError(f'a{attempt:04d}: framed/native edge discrepancy {max_edge_error} mm')
        if not rows:
            known = np.array([2.0, 0.2, -0.1, -4.0, 1.7, -0.5])
            assert np.allclose(coefficients(ref, faces, basis(ref[:, :2]) @ known), known, atol=1e-10)
        co = coefficients(ref, faces, p[:, 2])
        rows.append({'attempt': attempt, 'run1_rms_mm': record['run1_rms_mm'],
                     'joint_rms_mm': record['joint_rms_mm'], 'coefficients_mm': co.tolist(),
                     'max_framed_native_edge_difference_mm': max_edge_error,
                     'native_state_sha256': sha(state_bytes), 'prediction_sha256': sha(mesh_path.read_bytes())})
    out = {'created_utc': datetime.now(timezone.utc).isoformat(),
           'source_snapshot_sha256': sha(input_path.read_bytes()),
           'script_sha256': sha(Path(__file__).read_bytes()),
           'coordinates': 'height y vs reference u=x and v=z-65; area-weighted quadratic; mm',
           'basis': source['basis'],
           'caution': 'Material-reference model coefficients are descriptive, not fitted scan coefficients or a model impossibility proof.',
           'records': rows}
    with output_path.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    for label, group in [('all', rows), ('Run1 <= 0.8 mm', [r for r in rows if r['run1_rms_mm'] <= 0.8])]:
        uv = [r['coefficients_mm'][4] for r in group]
        print(label, 'n=', len(group), 'material cross-term min/median/max mm=',
              float(np.min(uv)), float(np.median(uv)), float(np.max(uv)), flush=True)
    print('saved', output_path, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
