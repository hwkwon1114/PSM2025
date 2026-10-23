import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'legacy/scripts'))
import dome_offset_domain as domain

class OffsetTest(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'legacy/configs/dome/baseline_attempt209.json').read_text())
    def test_baseline_roundtrip(self):
        p=domain.coordinates(self.base)
        self.assertEqual(domain.sequence(self.base,p),self.base)
        for a,b in zip(domain.decode(domain.encode(p)),p): self.assertAlmostEqual(a,b)
    def test_inward_expansion_and_overlap(self):
        p=domain.coordinates(self.base)
        p[3:6]=[196,5,18]
        with self.assertRaises(ValueError): domain.sequence(self.base,p)
        p[8]=0
        c=domain.sequence(self.base,p)
        self.assertEqual(c['toolpaths'][1]['operation']['shift_uv_mm'],[0,0])
        self.assertEqual(c['toolpaths'][2]['operation']['shift_uv_mm'],[0,0])
        self.assertTrue(domain.compatible(self.base,c))
        self.assertEqual(self.base['toolpaths'][1]['operation']['shift_uv_mm'],[0,118])
    def test_invalid_offset(self):
        p=domain.coordinates(self.base)
        for v in (-1,119,float('nan')):
            with self.assertRaises(ValueError): domain.sequence(self.base,p[:8]+[v])

if __name__=='__main__': unittest.main()
