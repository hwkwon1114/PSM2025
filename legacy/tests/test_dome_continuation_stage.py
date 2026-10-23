"""Stage2 domain/design/controller checks without solver execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
import dome_offset_domain as domain
import dome_continuation_campaign as campaign

class StageTest(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
    def test_repeat_extremes(self):
        original=domain.coordinates(self.base)
        for middle,side in ((1,30),(30,1),(30,30)):
            params=original[:6]+[middle,side,118]
            config=domain.sequence(self.base,params)
            self.assertEqual([p['repeat'] for p in config['toolpaths']],[middle,side,side])
            self.assertEqual(domain.coordinates(config),params)
        with self.assertRaises(ValueError): domain.sequence(self.base,original[:6]+[1.5,30,118])
    def test_design_and_warmstarts(self):
        records=campaign.warm_start(self.base)
        self.assertEqual(len(records),1)
        self.assertEqual(sum(r['status']=='interrupted' for r in records),0)
        self.assertTrue(all('coverage' not in r for r in records if r['status']=='interrupted'))
        self.assertEqual(sum(r['status']=='accepted' for r in records),1)
        points=json.loads(campaign.INITIAL_DESIGN.read_text())['parameters']
        self.assertEqual(len({tuple(p) for p in points}),24)
        for point in points:
            domain.sequence(self.base,point)
            self.assertNotIn(point,[r['parameters'] for r in records])
    def test_64_attempts_and_initial_design(self):
        initial=json.loads(campaign.INITIAL_DESIGN.read_text())['parameters']
        observed=[]
        def evaluate(attempt):
            record=json.loads((attempt/'reservation.json').read_text())
            observed.append(record['parameters'])
            return {'status':'accepted','coverage':.2,'screening_pass':False}
        def propose(command,**kwargs):
            req=json.loads(Path(command[-2]).read_text()); wave=req['seed']-20261021
            params=domain.coordinates(self.base)
            batch=[]
            for j in range(2):
                p=params.copy(); p[0]=100+wave+j*.1; p[6:]=[7,9,50]; batch.append(p)
            campaign.save_new(Path(command[-1]),{'parameters':batch,'seed':req['seed']})
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'SLURM_JOB_ID':'test','SLURM_CPUS_PER_TASK':'16','DOME_BO_DEADLINE':'inf'}), patch.object(campaign,'evaluate',side_effect=evaluate), patch.object(campaign.subprocess,'run',side_effect=propose):
            out=Path(tmp)/'stage'
            campaign.campaign(out)
            self.assertEqual(len(observed),64)
            for i,point in enumerate(initial,1):
                self.assertEqual(json.loads((out/f'attempt_{i:03d}/reservation.json').read_text())['parameters'],point)
            self.assertEqual({tuple(p) for p in observed[:24]},{tuple(p) for p in initial})
            self.assertEqual(json.loads((out/'stop.json').read_text())['reason'],'64_continuation_evaluations_exhausted')

if __name__=='__main__': unittest.main()
