import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
import dome_bo_campaign as campaign


class CampaignTest(unittest.TestCase):
    def test_warm_start(self):
        base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
        records = campaign.warm_start(base)
        self.assertEqual(len(records),5)
        self.assertEqual(sum(r['status']=='accepted' for r in records),4)
        self.assertTrue(all('coverage' not in r for r in records if r['status']=='failed'))

    def run_fixture(self, status, expected_attempts, reason):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'campaign'
            def fake_process(command,**kwargs):
                request = json.loads(Path(command[-2]).read_text())
                # Vary length within valid bounds to avoid fake duplicate attempts.
                length = 120 + 2*(request['seed']%20)
                campaign.save_new(Path(command[-1]), {'seed':request['seed'],
                    'parameters':[[length,4,10,196,2,8],[length+1,4,10,196,2,8]]})
            def fake_evaluate(attempt):
                self.assertTrue((attempt/'reservation.json').exists())
                if status=='error': raise RuntimeError('fixture infrastructure error')
                return {'status':status,**({'coverage':.81,'screening_pass':True} if status=='accepted' else {})}
            with patch.dict('os.environ',{'SLURM_JOB_ID':'fixture','SLURM_CPUS_PER_TASK':'16'}), \
                 patch.object(campaign.subprocess,'run',side_effect=fake_process), \
                 patch.object(campaign,'evaluate',side_effect=fake_evaluate):
                campaign.campaign(out)
            self.assertEqual(len(list(out.glob('attempt_*'))),expected_attempts)
            self.assertEqual(json.loads((out/'stop.json').read_text())['reason'],reason)
            self.assertEqual(len(list(out.glob('attempt_*/observation.json'))),expected_attempts)

    def test_screening_stop(self):
        self.run_fixture('accepted',2,'screening_pass_requires_verification')
    def test_failure_stop(self):
        self.run_fixture('failed',4,'two_full_waves_failed_equilibrium')
    def test_infrastructure_stop(self):
        self.run_fixture('error',2,'infrastructure_or_analysis_error')


if __name__=='__main__': unittest.main()
