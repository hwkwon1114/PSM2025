"""One-pass production-versus-frozen-gallery fidelity, no ICP, plotting or mechanics."""
import json
import os
from pathlib import Path
import numpy as np
from calibration.analysis import reassess_failed_registration as prior
from calibration.ewcal.material_frame import score_material_frame

ROOT = Path(__file__).resolve().parents[2]


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Owned compute allocation required')
    planpath = ROOT/'run/optimization_review/free_translation_plan.json'
    reportpath = ROOT/'run/optimization_review/direction_frame_gallery/direction_frame_results.json'
    output = ROOT/'run/optimization_review/material_bo_score_fidelity.json'
    if output.exists():
        raise FileExistsError('Preserve previous fidelity evidence')
    plan, report = json.loads(planpath.read_text()), json.loads(reportpath.read_text())
    pins = {n:d for n,d in report['input_sha256'].items() if not n.endswith('.py')}
    pins.update({str(p.relative_to(ROOT)):prior.sha(p) for p in [planpath,reportpath]})
    def verify():
        for name,digest in pins.items():
            if prior.sha(ROOT/name) != digest:
                raise ValueError(f'Frozen scientific input changed: {name}')
    verify()
    rows = {r['attempt_id']:r for r in report['records']}
    if len(rows)!=34 or len(plan['endpoints'])!=34 or not report.get('analysis_complete'):
        raise ValueError('Incomplete gallery evidence')
    with np.load(ROOT/'calibration/data/processed/run6_oriented.npz',allow_pickle=False) as d:
        scan=d['points_mm'].copy()
    result={'status':'running','job_id':os.environ['SLURM_JOB_ID'], 'input_sha256':pins,
            'source_sha256':prior.sha(Path(__file__)),
            'scorer_sha256':prior.sha(ROOT/'calibration/ewcal/material_frame.py'), 'records':[]}
    def save():
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for endpoint in plan['endpoints']:
        mesh,ref,faces=prior.load_geometry(endpoint,plan['config'])
        metrics,*_=score_material_frame(mesh,scan,ref,faces)
        frozen=rows[endpoint['attempt_id']]
        errors={k:abs(metrics[k]-v) for k,v in frozen['metrics'].items()}
        if any(v>1e-6 for v in errors.values()):
            raise ValueError(f'Production/gallery metric mismatch: {endpoint["attempt_id"]}: {errors}')
        transform=metrics['simulation_transform']
        for k in ('row_vector_rotation','translation_mm'):
            if not np.allclose(transform[k],frozen['simulation_transform'][k],rtol=0,atol=1e-9):
                raise ValueError('Production/gallery rigid-frame mismatch')
        result['records'].append({'attempt_id':endpoint['attempt_id'],
                                 'rms_mm':metrics['rms_symmetric_mm'],'max_metric_difference_mm':max(errors.values())})
        save()
        print('FIDELITY',endpoint['attempt_id'],max(errors.values()),flush=True)
    verify()
    result.update(status='complete',inputs_unchanged=True,completed_count=len(result['records']))
    save()


if __name__=='__main__': main()
