import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
from dome_bo_campaign import warm_start
from dome_bo_domain import sequence
from dome_bo_propose import propose


class ProposalTest(unittest.TestCase):
    def test_small_retained_fixture(self):
        base = json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
        records = warm_start(base)
        for initial in (True,False):
            result = propose(base,records,20261018,count=2,initial=initial)
            self.assertEqual(result['valid_training_count'],4)
            self.assertEqual(len(result['parameters']),2)
            self.assertNotEqual(result['parameters'][0],result['parameters'][1])
            for point in result['parameters']:
                sequence(base,point)
                self.assertNotIn(point,[r['parameters'] for r in records])


if __name__=='__main__': unittest.main()
