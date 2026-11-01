import json,sys
from pathlib import Path
import numpy as np
R=Path('/gpfs/projects/p33082/PSM2025-zigzag');sys.path.insert(0,str(R))
from calibration.bo.run_convention_corrected import config,REQUESTS
from calibration.ewcal.common import Parameters,digest
from calibration.ewcal.runner import command,sequence_json

def diff(a,b,p=''):
 if isinstance(a,dict) and isinstance(b,dict):return sum([diff(a.get(k),b.get(k),p+'/'+k) for k in sorted(set(a)|set(b))],[])
 if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):return sum([diff(x,y,p+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
 return [] if a==b else [{'path':p,'old':a,'new':b}]
out=[]
for run in REQUESTS:
 cfg=config(run);old=json.loads((R/REQUESTS[run]).read_text());folder=R/f'run/convention_corrected/{run}'
 assert cfg==json.loads((folder/'proposed_config.json').read_text())
 params=old['parameters'];codec=Parameters(cfg);decoded=codec.decode(codec.encode(params))
 assert all(np.isclose(decoded[k],v,rtol=1e-14,atol=0) for k,v in params.items())
 files=[R/'calibration/bo/run_convention_corrected.py',R/'shellsim/configs/default_adaptive_cutback.json',folder/'proposed_config.json',folder/'frozen_parameters.json',R/cfg['cases'][0]['target']]
 out.append({'run':run,'sha256':{str(p.relative_to(R)):digest(p) for p in files},'differences':diff(old['config'],cfg),'command':command(cfg,cfg['cases'][0],folder/'sequence_preview.json'),'sequence':sequence_json(cfg,cfg['cases'][0],decoded),'decoded':decoded,'roundtrip_rtol':1e-14,'roundtrip_pass':True})
(R/'docs/calibration/bayesian_calibration/convention_current_commands.json').write_text(json.dumps(out,indent=2)+'\n')
