"""Small solver-free controls for the frozen edge-target history replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from calibration.analysis import rescore_run6_edge_history as migration
from calibration.ewcal.geometry import triangle_mesh
from calibration.ewcal.runner import SolverFailure

class EdgeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[-100.,-150.,0],[100.,-150.,0],[100.,150.,2],[-100.,150.,2]])
        self.ref=self.p.copy();self.ref[:,2]=0
        self.faces=np.array([[0,1,2],[0,2,3]])
        self.mesh=triangle_mesh(self.p,self.faces)
        self.state={'schema_version':1,'units':'m','reference_positions':(self.ref/1000).tolist(),
                    'positions':(self.p/1000).tolist(),'faces':self.faces.tolist()}
        self.sequence={'schema_version':1,'completed':True,'cycles':29,'backend':'native_hybrid','equilibrium_solves':1,
                       'solves':[{'cycle':29,'return_code':0,'native_hybrid':{'accepted':True,'termination':'gradient_target',
                                 'curvature':'positive_pivots','gradient_norm':1e-15}}]}
    def test_valid_saved_native(self):
        ref,faces,weights=migration.validate_saved(self.mesh,self.state,self.sequence)
        np.testing.assert_allclose(ref,self.ref);np.testing.assert_array_equal(faces,self.faces)
        self.assertAlmostEqual(weights.sum(),1)
    def test_invalid_schema_units_geometry(self):
        variants=[('units','mm'),('schema_version',2),('faces',[[0,1,8],[0,2,3]]),
                  ('faces',[[0,2,1],[0,2,3]]),('positions',(self.p/100).tolist())]
        for field,value in variants:
            with self.subTest(field=field,value=value):
                state=copy.deepcopy(self.state);state[field]=value
                with self.assertRaises((ValueError,SolverFailure)):migration.validate_saved(self.mesh,state,self.sequence)
    def test_incomplete_or_nonaccepted(self):
        for path,value in [('completed',False),('cycles',28),('accepted',False),('gradient_norm',1e-8),('curvature','negative_pivots')]:
            s=copy.deepcopy(self.sequence)
            if path in ('completed','cycles'):s[path]=value
            else:s['solves'][0]['native_hybrid'][path]=value
            with self.subTest(path=path),self.assertRaises(ValueError):migration.validate_saved(self.mesh,self.state,s)
    def test_timeout_is_not_penalty_loss(self):
        old={'attempt_id':47,'parameters':{'gtop':.001},'normalized_parameters':[0,0,0,0],
             'valid':False,'failure_class':'timeout','objective_mm':1000000,'directory':'original'}
        untouched=copy.deepcopy(old);new=migration.migrated_observation(old,None,'new',None)
        self.assertIsNone(new['objective_mm']);self.assertEqual(new['prior_objective_mm'],1000000)
        self.assertFalse(new['valid']);self.assertEqual(old,untouched)
        self.assertEqual(new['parameters'],old['parameters']);self.assertEqual(new['normalized_parameters'],old['normalized_parameters'])
    def test_completed_requires_real_score(self):
        old={'valid':True,'parameters':{},'objective_mm':5,'cases':[{'metrics':{'old':True}}]}
        untouched=copy.deepcopy(old)
        for metrics in [None,{'rms_symmetric_mm':float('nan')}]:
            with self.assertRaises(ValueError):migration.migrated_observation(old,metrics,'new','newdir')
        new=migration.migrated_observation(old,{'rms_symmetric_mm':2},'new','newdir')
        self.assertEqual(new['objective_mm'],2);self.assertEqual(old,untouched)
        self.assertEqual(new['cases'][0]['prior_metrics'],{'old':True})
    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';migration.save(p,{'first':True})
            with self.assertRaises(FileExistsError):migration.save(p,{'first':False})
            self.assertEqual(json.loads(p.read_text()),{'first':True})
    def test_drift_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'fixture.txt';source.write_text('old')
            migration.save(root/'rescore_bindings.json',{'input_sha256':{'fixture.txt':migration.sha(source)}})
            with patch.object(migration,'ROOT',root),patch.object(migration,'BASE',root),patch.object(migration,'verify',return_value={}):
                migration.check_bindings();source.write_text('new')
                with self.assertRaisesRegex(RuntimeError,'Reviewed migration drift'):migration.check_bindings()

if __name__=='__main__':unittest.main()
