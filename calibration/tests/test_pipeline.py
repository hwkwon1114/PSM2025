import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from calibration.ewcal.common import ROOT, Parameters, load_config, digest, save_json
from calibration.ewcal.geometry import triangle_mesh, SurfaceDistance, rigid_fit, points
from calibration.ewcal.runner import Evaluator, sequence_json, check_output, command, launch, SolverFailure
from calibration.ewcal.optimize import optimize
from calibration.ewcal.report import report

class GeometryTests(unittest.TestCase):
    def test_triangle_interior_not_nearest_vertex(self):
        mesh=triangle_mesh([[0,0,0],[10,0,0],[0,10,0]],[[0,1,2]])
        d,cp=SurfaceDistance(mesh).closest(np.array([[2.,2.,3.],[2.,2.,0.]]))
        np.testing.assert_allclose(d,[3,0],atol=1e-12)
        np.testing.assert_allclose(cp,[[2,2,0],[2,2,0]],atol=1e-12)
    def test_rigid_fit_does_not_scale_or_reflect(self):
        p=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
        angle=.3;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        q=p@r+np.array([2,3,4]);a,b=rigid_fit(p,q)
        np.testing.assert_allclose(p@a+b,q,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(a),1)
    def test_parameter_roundtrip_and_layer_json(self):
        c=load_config(ROOT/'calibration/configs/run6.json');p=Parameters(c);values=p.decode(p.initial())
        np.testing.assert_allclose(list(values.values()),[.003,.0001,-.5,-.5])
        values['ortho_bottom']=.2
        op=sequence_json(c,c['cases'][0],values)['toolpaths'][0]['operation']
        self.assertEqual(op['ortho_bottom'],.2);self.assertNotIn('ortho',op)

class PipelineTests(unittest.TestCase):
    def test_subprocess_objective_cache_failure_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=load_config(ROOT/'calibration/configs/run6.json')
            c['solver']['executable']=str(ROOT/'calibration/tests/mock_solver.py')
            c['solver']['command_prefix']=[sys.executable]
            c['output_dir']=str(root/'runs');c['report_dir']=str(root/'report')
            c['registration']['fit_points']=100;c['registration']['icp_iterations']=3
            target=root/'target.npz';case=c['cases'][0];case['target']=str(target)
            folder=root/'seed';folder.mkdir();seq=folder/'sequence.json'
            save_json(seq,sequence_json(c,case,Parameters(c).decode(Parameters(c).initial())))
            launch(command(c,case,seq),folder,60)
            manifest,mesh=check_output(c,case,folder);cloud=points(mesh)
            np.savez_compressed(target,points_mm=cloud)
            save_json(target.with_suffix('.json'),{'target_sha256':digest(target),'preprocess':c['preprocess'],'orientation':case['scan_orientation']})
            ev=Evaluator(c);result=ev.evaluate(ev.parameters.initial())
            self.assertTrue(result['valid']);self.assertLess(result['objective_mm'],1e-9)
            again=ev.evaluate(ev.parameters.initial());self.assertEqual(again['directory'],result['directory']);self.assertEqual(ev.attempts,1)
            path=report(Path(result['directory'])/'result.json',root/'report');self.assertTrue(Path(path).exists())
            bad=copy.deepcopy(c);bad['solver']['args']['-mock_fail']=True
            bad_ev=Evaluator(bad);failure=bad_ev.evaluate(bad_ev.parameters.initial())
            self.assertFalse(failure['valid']);self.assertIn('exit code 4',failure['error'])
            wrong=copy.deepcopy(c);wrong['sheet']['size_mm']=[204,204]
            with self.assertRaises(SolverFailure):check_output(wrong,case,folder)
            manifest['solves'][0]['reported_eps']=1.0;save_json(folder/'sequence_result.json',manifest)
            with self.assertRaises(SolverFailure):check_output(c,case,folder)
            c['optimization']['max_evaluations']=3;c['optimization']['coarse_samples']=0
            ev2=Evaluator(c);out=optimize(ev2)
            self.assertTrue(out['best']['valid']);self.assertLessEqual(out['objective_calls'],3)
if __name__=='__main__':unittest.main()
