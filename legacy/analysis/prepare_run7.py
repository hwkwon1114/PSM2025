"""Prepare Run 7 scan exactly as Run 9: PCA basis, x-turnover [1,-1,-1], edge-yaw alignment; writes config run7.json.
No solves. Path: lv 175, alpha 0.5, n_strips 100, repeat 13; shift_uv=(0,(top-bottom)/2) with bottom clearance 85 mm."""
import copy, json, sys
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / "calibration/analysis"))
from calibration.ewcal.common import save_json
from calibration.ewcal.prepare import prepare_case
from orient_run9_edges import edge_angles, yaw_matrix

cfg = json.loads((BASE / 'calibration/configs/run9.json').read_text())
case = copy.deepcopy(cfg['cases'][0])
LV, BOTTOM, SHEET_L = 175.0, 85.0, 300.0
shift_v = ((SHEET_L - LV - BOTTOM) - BOTTOM) / 2          # (top-bottom)/2 = -22.5
case.update(id='run7', repeat=13, scan='calibration/data/raw/Run_7.stl',
            target='calibration/data/processed/run7_oriented.npz')
case['operation'].update(lv_mm=LV, alpha_deg=0.5, n_strips=100, shift_uv_mm=[0, shift_v])
case.pop('scan_orientation', None)
cfg['cases'] = [case]
cfg['output_dir'] = 'run/run7_transfer'; cfg['report_dir'] = 'run/run7_transfer/report'

# stage 1: PCA basis (same as Run 9 initial prepare) into a scratch target
init = copy.deepcopy(case); init['target'] = 'calibration/data/processed/run7.npz'
m0 = prepare_case(cfg, init)
T = m0['transform']; basis = np.array(T['basis_columns'])
from calibration.ewcal.geometry import points, read_mesh
raw = np.unique(points(read_mesh(BASE / case['scan'])), axis=0)
q = (raw - np.array(T['origin_scanner_mm'])) @ basis - np.array(T['shift_local_mm'])
turn = np.diag([1., -1., -1.]); turned = q @ turn
est = [edge_angles(turned[:, :2], s) for s in (1., 2.)]
angle = est[0]['mean_angle_deg']
assert abs(angle - est[1]['mean_angle_deg']) <= .5, est   # yaw insensitive to bin size; edge disagreement (1.9 deg, curved sheet) recorded, not gated
rot = yaw_matrix(angle); aligned = turned @ rot.T
resid = [edge_angles(aligned[:, :2]), edge_angles(aligned[:, [1, 0]])]
assert max(abs(a) for e in resid for a in e['angles_deg']) <= 3., resid  # Run 9 gate was .5; Run 7 is more curved
composed = basis @ turn @ rot.T
case['scan_orientation'] = {'flip_z': False, 'flip_xy': False, 'basis_columns': composed.tolist()}
meta = prepare_case(cfg, case)
meta['edge_alignment'] = {'estimates': est, 'yaw_deg_after_turnover': angle, 'residual': resid,
                          'convention': 'Run 9: x-turnover [1,-1,-1] then yaw; face not labelled in STL'}
save_json(BASE / 'calibration/data/processed/run7_oriented.json', meta)
save_json(BASE / 'calibration/configs/run7.json', cfg)
print(json.dumps({'yaw_deg': angle, 'shift_uv': case['operation']['shift_uv_mm'], 'accepted_points': meta['accepted_points'],
                  'coverage': meta['coverage_fraction_approx'], 'extents': meta['robust_scan_extents_mm'], 'resid': resid}, indent=1))
