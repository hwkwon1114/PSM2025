"""One frozen-parameter forward solve in the researcher-corrected convention."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.ewcal.common import Parameters,save_json,digest
from calibration.ewcal.runner import Evaluator

REQUESTS={
 'run6':'run/run6_expanded_bo_continue20_recovery/evaluations/attempt_0209/request.json',
 'run9':'run/run9_transfer_expanded/evaluations/attempt_0209/request.json',
 'run10':'run/run10_transfer/evaluations/attempt_9902/request.json'}
BINARY_SHA='522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841'
TARGET_SHA={'run6':'00d64e18d28eab6c7254d674af8424d169cac3672abaa2bcf072cab59673a348','run9':'aa1fc209fa1b1eee7929a050957c3551ec959d09ed0fec92b02769a01451890e','run10':'97a0e13ba494487578107c39fc0eff8e44706eb8f0ba783ab0b78583adcbd578'}

def config(run):
    cfg=copy.deepcopy(json.loads((ROOT/REQUESTS[run]).read_text())['config'])
    # Researcher-requested solver upgrade; physical recipe remains archived.
    cfg['solver']['executable']='bin/shell'
    options=json.loads((ROOT/'shellsim/configs/default_adaptive_cutback.json').read_text())['options']
    cfg['solver']['args'].update(options)
    # Preserve approved wall limits and final-only history schedule, not shared defaults.
    cfg['solver']['args']['-native_hybrid_seconds']=3600
    cfg['solver']['args']['-sequence_minimize_every']=1  # ignored by final_only solve schedule
    assert cfg['solver']['solve_mode']=='final_only'
    assert digest(ROOT/cfg['solver']['executable'])==BINARY_SHA
    case=cfg['cases'][0]
    # Internal (u,v,h) -> public (x,y,z)=(u,h,v+65) mm.
    # Negative angle mirrors the generator's x progression: right-bottom -> up.
    case['operation']['alpha_deg']=-abs(case['operation']['alpha_deg'])
    case['operation']['shift_uv_mm']=[0.,case['operation']['lv_mm']/2-65.]
    target=ROOT/f'run/putong_convention_check/preserved_samples/{run}.npz'
    case['target']=str(target.relative_to(ROOT))
    assert digest(target)==TARGET_SHA[run]
    meta=json.loads(target.with_suffix('.json').read_text())
    case['scan_orientation']={'basis_columns':meta['transform']['basis_columns'],'flip_z':False,'flip_xy':False}
    cfg['output_dir']=f'run/convention_corrected/{run}'
    cfg['report_dir']=f'run/convention_corrected/{run}/report'
    return cfg

def main():
    run=sys.argv[1]
    assert run in ('run6','run9','run10')
    cfg=config(run)
    best={'parameters':json.loads((ROOT/REQUESTS['run6']).read_text())['parameters']}
    assert best['parameters']==json.loads((ROOT/REQUESTS[run]).read_text())['parameters']
    out=ROOT/cfg['output_dir']
    if '--prepare-only' in sys.argv:
        out.mkdir(parents=True,exist_ok=True)
        save_json(out/'proposed_config.json',cfg)
        save_json(out/'frozen_parameters.json',best['parameters'])
        return
    if (out/'evaluations').exists():raise FileExistsError('Existing corrected evaluation; no automatic restart')
    rec=Evaluator(cfg).evaluate(Parameters(cfg).encode(best['parameters']).tolist(),use_cache=False,attempt_id=1)
    save_json(out/'forward_result.json',rec)
    if not rec.get('valid'):raise RuntimeError(f"{rec.get('failure_class')}: {rec.get('error')}")

if __name__=='__main__':main()
