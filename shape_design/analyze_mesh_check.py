"""Compare the saved 1 mm endpoints with the nine accepted 2 mm turned-over shapes.

No solver, registration or parameter fitting. PCA is descriptive across nine
model-generated shapes; the fine endpoints are projected into the coarse basis.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_design import analysis

OUT = ROOT / 'run/shape_design/mesh_comparison'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_vertices(coarse, fine, tolerance=1e-8):
    """One-to-one reference-coordinate mapping; reject grids that are not nested."""
    distance, indices = cKDTree(fine).query(coarse)
    if np.max(distance) > tolerance or len(np.unique(indices)) != len(indices):
        raise ValueError('Reference grids are not nested within tolerance')
    return indices, float(np.max(distance))


def load(stage, k, mesh):
    folder = ROOT / 'run/shape_design' / stage / f'flip_k{k:+d}'
    result_path, shape_path, sequence_path = (folder / 'result.json', folder / 'shape.npz', folder / 'sequence.json')
    record = json.loads(result_path.read_text())
    if (record['status'] != 'accepted' or record['mesh_mm'] != mesh or
            record['variant'] != 'flip' or record['k'] != k or
            record['hit_check']['mismatched_faces'] != 0 or
            digest(sequence_path) != record['sequence_sha256']):
        raise ValueError(f'Unaccepted, inconsistent or changed saved case: {folder}')
    with np.load(shape_path, allow_pickle=False) as data:
        ref, pos, faces = (data[name].copy() for name in ('reference_mm', 'aligned_mm', 'faces'))
    if ref.shape != pos.shape or not np.isfinite(pos).all():
        raise ValueError(f'Invalid mesh coordinates: {folder}')
    return {'ref': ref, 'pos': pos, 'faces': faces, 'height': analysis.height(pos),
            'inputs': {str(p.relative_to(ROOT)): digest(p) for p in (result_path, shape_path, sequence_path)}}


def curvature_summary(case):
    local = analysis.local_curvature(case['ref'], case['pos'], case['faces'])
    return {key: value for key, value in local.items() if key not in ('gauss_per_m2', 'valid')}


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    ks = np.arange(-4, 5)
    coarse = [load('pilot' if k in (-4, 0, 4) else 'sweep_flip', int(k), 2.0) for k in ks]
    reference = coarse[0]['ref']
    if any(not np.allclose(c['ref'], reference, atol=1e-9, rtol=0) or
           not np.array_equal(c['faces'], coarse[0]['faces']) for c in coarse):
        raise ValueError('Coarse cases do not share the same reference grid')
    fields = np.stack([c['height'] for c in coarse])
    fit = [analysis.quadratic_fit(c['ref'], c['height']) for c in coarse]
    pca = analysis.pca(fields, ks)
    scores = pca['scores'][:, :2]
    correlation = float(np.corrcoef(scores[:, 0], [f['K_fit_per_mm2'] for f in fit])[0, 1])
    results = {'coarse': [], 'endpoints': {}, 'pca_explained_ratio': pca['explained_ratio'].tolist(),
               'pc1_K_correlation': correlation, 'analysis_script_sha256': digest(Path(__file__)),
               'analysis_helper_sha256': digest(ROOT / 'shape_design/analysis.py'),
               'curvature_helper_sha256': digest(ROOT / 'dome_bo/dome_curvature.py'),
               'notes': 'Mean-removed normal height after each full-grid rigid alignment; fine nodes restricted to nested coarse reference vertices and mean removed again. Local K sign uses 15 mm neighborhoods, reference-face area weighting and includes invalid area in denominator. Two meshes only; no physical validation or global convergence claim.'}
    for i, k in enumerate(ks):
        results['coarse'].append({'k': int(k), 'fit': fit[i], 'pc1': float(scores[i, 0]),
                                  'pc2': float(scores[i, 1]), 'inputs': coarse[i]['inputs']})
    for k in (-4, 4):
        c = coarse[k+4]
        fine = load('mesh_flip', k, 1.0)
        indices, grid_error = match_vertices(reference, fine['ref'])
        fine_h = fine['height'][indices]
        fine_h -= fine_h.mean()
        diff = fine_h - c['height']
        projection = (fine_h - pca['mean']) @ pca['components'][:2].T
        results['endpoints'][str(k)] = {
            'coarse_fit': fit[k+4], 'fine_fit': analysis.quadratic_fit(fine['ref'], fine['height']),
            'coarse_local': curvature_summary(c), 'fine_local': curvature_summary(fine),
            'coarse_pc1': float(scores[k+4, 0]), 'fine_pc1': float(projection[0]),
            'coarse_pc2': float(scores[k+4, 1]), 'fine_pc2': float(projection[1]),
            'height_difference_rms_mm': float(np.sqrt(np.mean(diff**2))),
            'height_difference_p95_abs_mm': float(np.quantile(np.abs(diff), .95)),
            'height_difference_max_abs_mm': float(np.max(np.abs(diff))),
            'nested_reference_max_error_mm': grid_error, 'fine_inputs': fine['inputs']}
    results['coarse_pc1_order_minus4_less_plus4'] = bool(scores[0, 0] < scores[-1, 0])
    results['fine_pc1_order_minus4_less_plus4'] = bool(results['endpoints']['-4']['fine_pc1'] < results['endpoints']['4']['fine_pc1'])
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'pca.npz', reference_mm=reference, mean=pca['mean'],
                        components=pca['components'][:2], scores=scores, k=ks)
    print(json.dumps({'explained': results['pca_explained_ratio'][:2], 'correlation': correlation,
                      'endpoints': results['endpoints']}, indent=2))


if __name__ == '__main__':
    main()
