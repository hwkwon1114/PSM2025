"""Read-only-to-inputs scan convention diagnostic; no solver or optimization."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.ewcal import prepare
from calibration.ewcal.common import digest

OUT=ROOT/'run/putong_convention_check'

def main():
    OUT.mkdir(exist_ok=True)
    # Diagnostic is numerical only; suppress the helper's optional preview figure.
    prepare.preview=lambda *args, **kwargs: None
    report=[]
    for run in ('run6','run9','run10'):
        cfgpath=ROOT/f'calibration/configs/{run}.json'
        cfg=json.loads(cfgpath.read_text())
        case=cfg['cases'][0]
        oldpath=ROOT/case['target']
        oldmeta=json.loads(oldpath.with_suffix('.json').read_text())
        sources=[cfgpath,oldpath,oldpath.with_suffix('.json'),ROOT/case['scan']]
        pins={str(p.relative_to(ROOT)):digest(p) for p in sources}
        cfg=copy.deepcopy(cfg);case=cfg['cases'][0]
        case['scan_orientation']={'flip_z':True,'flip_xy':False}
        case['target']=str((OUT/f'{run}_putong.npz').relative_to(ROOT))
        if (ROOT/case['target']).exists():raise FileExistsError(case['target'])
        meta=prepare.prepare_case(cfg,case)
        old=oldmeta['transform'];new=meta['transform']
        bo=np.array(old['basis_columns']);bn=np.array(new['basis_columns'])
        # Exact transform of existing prepared points, independently of rebinning.
        r=bo.T@bn
        t=(np.array(old['origin_scanner_mm'])-np.array(new['origin_scanner_mm']))@bn+np.array(old['shift_local_mm'])@r-np.array(new['shift_local_mm'])
        with np.load(oldpath) as f: p=f['points_mm'].copy()
        mapped=p@r+t
        np.savez(OUT/f'{run}_old_samples_in_putong_frame.npz',points_mm=mapped)
        assert np.allclose(r.T@r,np.eye(3),atol=1e-8) and np.linalg.det(r)>.999999
        assert np.max(np.abs(np.linalg.norm(p[1:]-p[:-1],axis=1)-np.linalg.norm(mapped[1:]-mapped[:-1],axis=1)))<1e-8
        assert all(digest(ROOT/p)==h for p,h in pins.items())
        (OUT/f'{run}_preparation_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        report.append({'run':run,'input_sha256':pins,'old_to_putong_row_rotation':r.tolist(),'old_to_putong_translation_mm':t.tolist(),'rotation_angle_deg':float(Rotation.from_matrix(r).magnitude()*180/np.pi),'old_points':len(p),'new_points':meta['accepted_points'],'note':'Fresh PCA+flip_z keeps current preprocessing settings but removes custom edge basis. Rebinning/masks can change retained samples; exact mapped old samples saved separately. No model score or solver run.'})
    (OUT/'scan_convention_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
