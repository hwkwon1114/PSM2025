"""Algorithm tests use cheap synthetic functions, never the physical shell solver."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import ndtr
from calibration.ewcal.common import ROOT,Parameters,load_config,save_json
from calibration.ewcal.bayes import expected_improvement,bayesian_search,propose,settings_for
from calibration.ewcal.runner import launch,SolverFailure

class FakeEvaluator:
    def __init__(self,folder):
        self.config=load_config(ROOT/'calibration/configs/run6_bayes.json')
        self.config['bayesian'].update(initial_points=4,candidate_pool=128,local_candidates=32,
                                       acquisition_restarts=2,gp_restarts=0)
        self.config['output_dir']=str(folder)
        self.parameters=Parameters(self.config);self.base=Path(folder);self.identity='test-identity';self.attempts=0
        self.stop_on=None
    def evaluate(self,x):
        self.attempts+=1
        if self.stop_on==self.attempts:raise KeyboardInterrupt()
        # Coupled, unequal-sensitivity, positive loss in normalized parameter space.
        u=np.asarray(x)-np.array([.68,.22,.63,.34])
        loss=.15+(u[0]+.4*u[1])**2+.8*u[1]**2+.6*u[2]**2+.3*u[3]**2
        return {'valid':True,'objective_mm':float(loss),'normalized_parameters':list(map(float,x)),
                'parameters':self.parameters.decode(x),'identity':self.identity,
                'directory':str(self.base/f'trial_{self.attempts}'),'cases':[],'seconds':.01}

class BayesTests(unittest.TestCase):
    def test_expected_improvement_formula(self):
        value=expected_improvement(np.array([1.,2.]),np.array([1.,0.]),1.,0.)
        np.testing.assert_allclose(value,[1/np.sqrt(2*np.pi),0],atol=1e-14)
        self.assertEqual(float(expected_improvement(0.,0.,1.,.1)),.9)
    def test_resume_pending_and_monotone_best(self):
        with tempfile.TemporaryDirectory() as d:
            ev=FakeEvaluator(d);ev.stop_on=3
            with self.assertRaises(KeyboardInterrupt):bayesian_search(ev,budget=8)
            state=json.loads((Path(d)/'bayes_state.json').read_text())
            self.assertEqual(len(state['observations']),2);self.assertIsNotNone(state['pending'])
            expected=state['pending']['x']
            resumed=FakeEvaluator(d)
            result=bayesian_search(resumed,resume=True,budget=8)
            self.assertEqual(result['completed_attempts'],8)
            state=json.loads((Path(d)/'bayes_state.json').read_text())
            np.testing.assert_allclose(state['observations'][2]['normalized_parameters'],expected)
            self.assertEqual(resumed.attempts,6)
            self.assertTrue(any(p['acquisition']=='expected_improvement_log_rms' for p in state['proposals']))
            ys=[r['objective_mm'] for r in state['observations']]
            self.assertAlmostEqual(result['best']['objective_mm'],min(ys))
            self.assertLess(min(ys),ys[0])
            for r in state['observations']:
                self.assertTrue(np.all(np.array(r['normalized_parameters'])>=0))
                self.assertTrue(np.all(np.array(r['normalized_parameters'])<=1))
            self.assertTrue((Path(d)/'bayes_progress.png').is_file())
            with self.assertRaises(ValueError):bayesian_search(resumed,budget=9)
            changed=FakeEvaluator(d);changed.identity='changed'
            with self.assertRaises(ValueError):bayesian_search(changed,resume=True,budget=9)
    def test_failed_losses_excluded_and_failure_radius(self):
        with tempfile.TemporaryDirectory() as d:
            ev=FakeEvaluator(d);settings=settings_for(ev.config)
            records=[ev.evaluate(np.full(4,t)) for t in [.1,.3,.7,.9]]
            bad=ev.evaluate(np.full(4,.5));bad.update(valid=False,objective_mm=1e6,error='failed')
            with patch('calibration.ewcal.bayes.GaussianProcessRegressor.fit',autospec=True) as fit:
                fit.side_effect=ValueError('inspect then fallback')
                x,info=propose(records+[bad],settings,22)
                train_y=fit.call_args.args[2]
                self.assertEqual(len(train_y),4);self.assertLess(train_y.max(),np.log(1e6))
            self.assertGreater(np.linalg.norm(x-.5),settings['failure_radius'])
    def test_cli_bayesian_run_and_resume_with_synthetic_solver(self):
        from calibration.ewcal.runner import sequence_json, command, check_output
        from calibration.ewcal.geometry import points
        from calibration.ewcal.common import digest
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);c=load_config(ROOT/'calibration/configs/run6_bayes.json')
            c['solver']['executable']=str(ROOT/'calibration/tests/mock_solver.py')
            c['solver']['command_prefix']=[sys.executable]
            c['output_dir']=str(folder/'runs');c['report_dir']=str(folder/'reports')
            c['bayesian'].update(initial_points=3,candidate_pool=64,local_candidates=16,
                                  acquisition_restarts=1,gp_restarts=0)
            c['registration'].update(fit_points=100,icp_iterations=3)
            target=folder/'target.npz';case=c['cases'][0];case['target']=str(target)
            seed=folder/'seed';seed.mkdir();seq=seed/'sequence.json'
            p=Parameters(c);save_json(seq,sequence_json(c,case,p.decode(p.initial())))
            launch(command(c,case,seq),seed,60)
            _,mesh=check_output(c,case,seed)
            np.savez_compressed(target,points_mm=points(mesh))
            save_json(target.with_suffix('.json'),{'target_sha256':digest(target),
                       'preprocess':c['preprocess'],'orientation':case['scan_orientation']})
            config=folder/'config.json';save_json(config,c)
            args=[sys.executable,'-m','calibration.ewcal','bayes','--config',str(config)]
            first=subprocess.run(args+['--budget','2'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            second=subprocess.run(args+['--resume','--budget','4'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stdout+second.stderr)
            summary=json.loads((folder/'runs/bayes_summary.json').read_text())
            self.assertEqual(summary['completed_attempts'],4)
            self.assertEqual(summary['fresh_evaluations_this_invocation'],2)
            self.assertLess(summary['best']['objective_mm'],1e-9)
            self.assertTrue((folder/'runs/best_parameters.json').exists())

    def test_solver_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SolverFailure):
                launch([sys.executable,'-c','import time; time.sleep(20)'],Path(d),.15,.05)

if __name__=='__main__':unittest.main()
