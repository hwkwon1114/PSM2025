#!/usr/bin/env python3
"""Summarize an accepted dome endpoint; geometric/director disagreement is unresolved.

This numerical report alone does not satisfy spatial visualization or refinement
verification and therefore never claims a verified dome.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dome_bo'))
from dome_curvature import geometric_curvature, coverage


def analyze(directory):
    result = json.loads((directory/'sequence_result.json').read_text())
    state_path = directory/'bilayer_zigzag_sequence_native_hybrid_state.json'
    state = json.loads(state_path.read_text())
    proof = state['proof']
    if not (result['completed'] is True and proof['accepted'] is True and
            result['solves'][-1]['native_hybrid'] == proof and
            proof['curvature'] == 'positive_pivots' and proof['termination'] == 'gradient_target' and
            0 <= proof['gradient_norm'] <= proof['gradient_tolerance'] <= 5e-14):
        raise ValueError('Unaccepted or inconsistent endpoint')
    positions, uv, faces = (np.asarray(state[key]) for key in ('positions', 'material_uv', 'faces'))
    for key in ('abar_top', 'abar_bottom'):
        metrics = np.asarray(state[key])
        if not np.isfinite(metrics).all() or not np.allclose(metrics, metrics.transpose(0,2,1), atol=1e-18, rtol=1e-12):
            raise ValueError('Nonfinite/asymmetric targets')
        if np.any(np.linalg.eigvalsh(metrics) <= 0):
            raise ValueError('Non-SPD targets')
    geometric, directions, valid = geometric_curvature(positions, uv, faces)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(directory/result['final_mesh']))
    reader.Update()
    data = reader.GetOutput()
    connectivity = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1,4)
    if not np.all(connectivity[:,0] == 3) or not np.array_equal(connectivity[:,1:], faces):
        raise ValueError('VTP/native topology mismatch')
    fields = data.GetCellData()
    h = vtk_to_numpy(fields.GetArray('mean')).astype(float)
    k = vtk_to_numpy(fields.GetArray('gauss')).astype(float)
    disc = h*h-k
    allowance = 64*np.finfo(float).eps*np.maximum(np.maximum(h*h, abs(k)), 1e-30)
    director_valid = np.isfinite(h) & np.isfinite(k) & (disc >= -allowance)
    root = np.sqrt(np.maximum(disc, 0))
    major = h + np.where(h >= 0, root, -root)
    minor = np.divide(k, major, out=np.zeros_like(k), where=major != 0)
    director = np.sort(np.column_stack((minor, major)), axis=1)
    director[~director_valid] = np.nan
    geom_score = coverage(geometric, valid, uv, faces)
    director_score = coverage(director, director_valid, uv, faces)
    disagreement = geom_score['qualifies'] != director_score['qualifies']
    report = {'geometry':geom_score, 'director':director_score,
              'status':'unresolved' if disagreement else ('candidate_requires_verification' if geom_score['qualifies'] else 'not_qualifying'),
              'sensitivity':{str(floor):coverage(geometric, valid, uv, faces, floor) for floor in (.05,.2)},
              'native_state_sha256':hashlib.sha256(state_path.read_bytes()).hexdigest(),
              'limits':'Local geometric fit at frozen 15mm scale. No spatial veto, tighter-tolerance or finer-mesh verification in this report.'}
    return report, dict(geometric=geometric, director=director, valid=valid,
                        director_valid=director_valid, directions=directions, uv=uv, faces=faces, positions=positions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing existing analysis directory')
    report, arrays = analyze(args.directory.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'curvature_fields.npz', **arrays)
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
