#!/usr/bin/env python3
"""Curvature-scale scan on retained mesh-study states (no solves): objective and gate at curvature radius R.
Usage: radius_scan.py --cell A RES R_MM  -> run/dome_square/mesh_study/radius_scan/a{A}_{h}mm_r{R}.json
       radius_scan.py --summary         -> prints table and Spearman rank agreement between meshes"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dome_bo'))
import mesh_convergence as mc  # noqa: E402
import square_attempt as sa    # noqa: E402

OUT = mc.STUDY / 'radius_scan'
KEYS = ('largest_connected_fraction', 'total_coherent_fraction', 'j_combined', 'saddle_flux_fraction')


def cell(a, r, radius_mm):
    h = dict(mc.MESH)[r]
    st = json.loads((mc.cell_dir(a, r) / 'solve/bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
    m = sa.analyze_state(st, radius=radius_mm / 1000)
    OUT.mkdir(exist_ok=True)
    (OUT / f'a{a}_{h}mm_r{radius_mm}.json').write_text(json.dumps({k: m[k] for k in KEYS}, indent=2) + '\n')


def summary(radii=(15, 20, 25, 30)):
    def get(a, h, R):
        if R in (10, 15, 20):
            return json.loads((mc.STUDY / 'cell_metrics' / f'a{a}_{h}mm.json').read_text())[f'r{R}mm']
        f = OUT / f'a{a}_{h}mm_r{R}.json'
        return json.loads(f.read_text()) if f.exists() else None
    for R in radii:
        print(f'--- curvature radius {R} mm: gate / total fraction / J  at 4, 2, 1 mm')
        tab = {}
        for a in mc.CAND:
            row = [get(a, h, R) for _, h in mc.MESH]
            tab[a] = row
            print(a, ' | '.join('missing' if x is None else '%.2f %.2f %.3f' % (
                x['largest_connected_fraction'], x['total_coherent_fraction'], x['j_combined']) for x in row))
        if all(x is not None for r in tab.values() for x in r):
            for k in ('largest_connected_fraction', 'j_combined'):
                print(k, 'Spearman 4v2, 2v1:', ' '.join('%.2f' % spearmanr([tab[a][i][k] for a in mc.CAND],
                      [tab[a][i + 1][k] for a in mc.CAND])[0] for i in (0, 1)))


if __name__ == '__main__':
    if sys.argv[1] == '--cell':
        cell(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        summary()
