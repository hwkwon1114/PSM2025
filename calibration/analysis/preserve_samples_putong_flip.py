"""Convention-only 180-degree in-plane rotation; preserve accepted scan samples."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.ewcal.common import digest
OUT=ROOT/'run/putong_convention_check/preserved_samples'
CASES={'run6':'run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/case_00','run9':'run/run9_transfer_expanded/evaluations/attempt_0209/case_00','run10':'run/run10_transfer/evaluations/attempt_9902/case_00'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    R=np.diag([-1.,-1.,1.])
    report=[]
    for run,folder in CASES.items():
        cfgpath=ROOT/f'calibration/configs/{run}.json';cfg=json.loads(cfgpath.read_text())
        path=ROOT/cfg['cases'][0]['target'];mp=path.with_suffix('.json')
        meta=json.loads(mp.read_text());comp=ROOT/folder/'comparison.npz'
        pins={str(p.relative_to(ROOT)):digest(p) for p in (cfgpath,path,mp,comp)}
        dest=OUT/f'{run}.npz'
        if dest.exists():raise FileExistsError(dest)
        with np.load(path) as f: data={k:f[k].copy() for k in f.files}
        old=data['points_mm'].copy();data['points_mm']=old@R
        assert np.array_equal(data['points_mm']@R,old)
        np.savez_compressed(dest,**data)
        # Preserve all archived comparison quantities, transform vector coordinates only.
        with np.load(comp) as f: comparison={k:f[k].copy() for k in f.files}
        keys=[]
        for k,v in comparison.items():
            if v.ndim==2 and v.shape[1]==3 and np.issubdtype(v.dtype,np.floating):
                comparison[k]=v@R;keys.append(k)
        np.savez_compressed(OUT/f'{run}_comparison_rotated.npz',**comparison)
        t=meta['transform'];t['basis_columns']=(np.array(t['basis_columns'])@R).tolist()
        t['shift_local_mm']=(np.array(t['shift_local_mm'])@R).tolist()
        meta['target_sha256']=digest(dest)
        meta['convention_change']={'source_target':str(path.relative_to(ROOT)),'row_rotation':R.tolist(),'preserved_samples':True,'rebinning':False,'edge_alignment_retained':True,'note':'180-degree in-plane rotation relates the former x-turnover to Putong y-turnover. This is NOT a new solve or placement-only correction.'}
        meta['orientation']={'basis_columns':t['basis_columns'],'flip_z':False,'flip_xy':False}
        dest.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        assert all(digest(ROOT/p)==h for p,h in pins.items())
        report.append({'run':run,'samples':len(old),'input_sha256':pins,'rotated_comparison_coordinate_keys':keys,'exact_roundtrip':True,'status':'coordinate-only scan/comparison copies; model not re-solved'})
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
