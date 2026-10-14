"""Bounded solver-free integration checks; never fit the real history."""
import copy
import json
import subprocess
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
from scripts import run_run6_bo_edge as edge
from functools import lru_cache
from calibration.ewcal.geometry import triangle_mesh
from calibration.ewcal.material_frame import score_material_frame
from calibration.tests.test_direction_frame import fixture
from calibration.analysis.success_aware_bo.model import training_records


@lru_cache(maxsize=1)
def mock_measurement():
    p, ref, faces = fixture()
    target = p + 1
    target[-1, 2] += 0.5
    measurement, *_ = score_material_frame(triangle_mesh(p, faces), target, ref, faces)
    return measurement


def record(att, valid=True):
    measurement = copy.deepcopy(mock_measurement())
    return {
        'attempt_id': att,
        'normalized_parameters': [att / 100, 0.2, 0.3, 0.4],
        'identity': 'fixture_identity',
        'valid': valid,
        **(
            {'objective_mm': measurement['rms_symmetric_mm'], 'cases': [{'id': 'run6', 'metrics': measurement}]}
            if valid else {'failure_class': 'timeout', 'error': 'fixture timeout'}
        ),
    }


class EdgeBOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        base = Path(self.temp.name)
        patch.object(edge, 'BASE', base).start()
        patch.object(edge, 'HISTORY', base/'history.json').start()
        patch.object(edge, 'CONFIG', base/'config.json').start()
        edge.configure()
        self.history = {'observations': [record(a) for a in range(1,81)],
                        'evaluator_identity': 'fixture_identity', 'config': {},
                        'metadata': edge.META, 'input_sha256': {}}
        edge.write_new(edge.HISTORY, self.history)
        edge.write_new(edge.CONFIG, {})
        self.state = edge.controller.initial_state(self.history, edge.sha(edge.HISTORY))

    def reserve(self):
        start = 81+len(self.state['reserved_attempts'])
        for att in range(start,start+4):
            folder = edge.BASE/'evaluations'/f'attempt_{att:04d}'
            self.state['reserved_attempts'].append({
                'attempt_id':att, 'x':[.5,.2,.3,.4], 'wave_idx':(att-81)//4+1,
                'proposal_idx':(att-81)%4, 'evaluator_identity':'fixture_identity',
                'publication_dir':str(folder), 'result_file':str(folder/'result.json'), 'status':'reserved'})
        self.state['charged_attempts'] = start+3
        self.state['waves_reserved'] = len(self.state['reserved_attempts'])//4

    def test_actual_history_training_labels(self):
        h = edge.read(edge.SOURCE)
        losses, outcomes, excluded = training_records(h['observations'])
        self.assertEqual(len(losses),70)
        self.assertEqual(len(outcomes),80)
        self.assertEqual(sum(y==0 for _,y in outcomes),10)
        self.assertEqual(excluded,[])
        for r in h['observations']:
            edge.controller.validate_material_science(r)

    def test_seed_preservation_recovery_and_no_replay(self):
        self.reserve()
        recovered = edge.controller.recover(self.state,self.history,'fixture_identity')
        edge.controller.update_stop(recovered)
        self.assertEqual(recovered['status'],'stopped_infrastructure_failure')
        self.assertEqual(recovered['charged_attempts'],84)
        self.assertEqual(recovered['observations'][:80],self.history['observations'])
        self.assertEqual(len(recovered['observations']),84)
        bad=copy.deepcopy(recovered); bad['observations'][0]['objective_mm']+=1
        with self.assertRaises(ValueError): edge.controller.validate_ledger(bad,'fixture_identity',2026,history=self.history)

    def test_forty_attempt_ceiling(self):
        for _ in range(10): self.reserve()
        for reserved in self.state['reserved_attempts']:
            r=record(reserved['attempt_id']); r['normalized_parameters']=reserved['x']
            self.state['observations'].append(r)
            self.state['completed_attempts'].append(r['attempt_id'])
            reserved['status']='completed'
        self.assertEqual(edge.controller.validate_ledger(self.state,'fixture_identity',2026,history=self.history),121)
        edge.controller.update_stop(self.state)
        self.assertEqual(self.state['status'],'budget_complete')
        self.reserve()
        with self.assertRaises(ValueError): edge.controller.validate_ledger(self.state,'fixture_identity',2026,history=self.history)

    def test_subprocess_cap_and_failure_propagation(self):
        with patch.object(edge.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',1500)) as launch:
            with self.assertRaises(subprocess.TimeoutExpired): edge.bounded_proposal([],2027)
        self.assertEqual(launch.call_args.kwargs['timeout'],1500)
        self.assertEqual(launch.call_args.kwargs['input'],'[]')

    def test_full_mock_controller_stops_at_ten_waves_and_terminal_resume(self):
        def execute(evaluator,reservations,state,*args,**kwargs):
            for att,pt,res,idx in reservations:
                r=record(att); r['normalized_parameters']=pt.tolist(); r['proposal_idx']=idx
                folder=Path(res['publication_dir']); folder.mkdir(parents=True)
                edge.write_new(folder/'result.json',r)
                state['observations'].append(r); state['completed_attempts'].append(att)
                res['status']='completed'
            kwargs['ledger_validator'](state,'fixture_identity',2026,edge.batch.validate_observation_record)
        pts=np.array([[.2,.2,.3,.4],[.4,.2,.3,.4],[.6,.2,.3,.4],[.8,.2,.3,.4]])
        with patch.object(edge,'check_review'), patch.object(edge,'production_evaluator'), patch.object(edge,'Evaluator') as evaluator, \
             patch.object(edge,'bounded_proposal',return_value=(pts,{'fixture':True})) as propose, \
             patch.object(edge.batch,'get_remaining_allocation_seconds',return_value=46800), \
             patch.object(edge.batch,'_execute_wave',side_effect=execute) as worker, \
             patch('torch.set_num_interop_threads'):
            evaluator.identity_for_config.return_value='fixture_identity'
            edge.run('fixture')
            self.assertEqual(propose.call_count,10); self.assertEqual(worker.call_count,10)
            self.assertEqual(edge.read(edge.BASE/'bayes_state.json')['charged_attempts'],120)
            edge.run('fixture')
            self.assertEqual(worker.call_count,10)
        self.assertEqual(edge.read(edge.BASE/'bayes_summary.json')['status'],'budget_complete')

    def test_controller_failures_are_durable_and_do_not_relaunch(self):
        failures = [('proposal',subprocess.TimeoutExpired('fixture',1500)),
                    ('proposal',subprocess.CalledProcessError(1,'fixture')),
                    ('proposal',RuntimeError('model failure')),
                    ('proposal',json.JSONDecodeError('bad','',0)),
                    ('deadline',RuntimeError('deadline query failure')),
                    ('input',ValueError('input drift'))]
        for phase,error in failures:
            with self.subTest(phase=phase,error=type(error).__name__), tempfile.TemporaryDirectory(dir='/tmp') as temp:
                with patch.object(edge,'BASE',Path(temp)):
                    with patch.object(edge,'check_review',side_effect=[None,error] if phase=='input' else None), \
                         patch.object(edge,'production_evaluator'), patch.object(edge,'Evaluator') as evaluator, \
                         patch.object(edge,'bounded_proposal',side_effect=error if phase=='proposal' else AssertionError('not reached')) as propose, \
                         patch.object(edge.batch,'get_remaining_allocation_seconds',side_effect=error if phase=='deadline' else None,return_value=46800), \
                         patch.object(edge.batch,'_execute_wave') as worker, patch('torch.set_num_interop_threads'):
                        evaluator.identity_for_config.return_value='fixture_identity'
                        with self.assertRaises(type(error)): edge.run('fixture')
                        saved=edge.read(edge.BASE/'bayes_state.json')
                        self.assertEqual(saved['observations'],self.history['observations'])
                        self.assertEqual(saved['charged_attempts'],80)
                        marker=edge.read(edge.BASE/'controller_failure.json')
                        self.assertEqual(marker['accounting'],{'new_charged':0,'new_completed':0,'unresolved_attempts':[]})
                        calls=propose.call_count
                        with self.assertRaisesRegex(RuntimeError,'no relaunch'): edge.run('fixture')
                        self.assertEqual(propose.call_count,calls); worker.assert_not_called()

    def test_native_abort_published_cap_enters_completion_model(self):
        from calibration.ewcal.common import Parameters
        for known in (True,False):
            with self.subTest(known=known), tempfile.TemporaryDirectory(dir='/tmp') as temp:
                cfg=edge.read(edge.ROOT/'calibration/configs/run6.json')
                cfg['output_dir']=temp
                ev=edge.production_evaluator(cfg)
                def process(*args,**kwargs):
                    lines=('[native_hybrid] status=time_cap accepted=0 energy=1 gradient=1 total_attempts=1 total_evaluations=1 total_hessians=1\n'
                           '  what():  zigzag_sequence: equilibrium corrector failed at cycle 1\n') if known else 'unknown abort\n'
                    kwargs['stdout'].write(lines.encode()); kwargs['stdout'].flush()
                    p=Mock(); p.pid=12345; p.wait.return_value=-signal.SIGABRT
                    return p
                with patch.object(subprocess,'Popen',side_effect=process):
                    r=ev.evaluate(Parameters(cfg).initial(),attempt_id=81)
                published=edge.read(Path(r['directory'])/'result.json')
                self.assertEqual(r,published)
                losses,outcomes,excluded=training_records([published])
                self.assertEqual(losses,[])
                if known:
                    self.assertEqual(published['solver_termination']['limit_seconds'],3600)
                    self.assertEqual(published['failure_class'],'timeout')
                    self.assertEqual([label for _,label in outcomes],[0]); self.assertEqual(excluded,[])
                    termination=edge.read(Path(r['directory'])/'case_00/termination.json')
                    self.assertEqual(termination,published['solver_termination'])
                else:
                    self.assertEqual(published['failure_class'],'infrastructure')
                    self.assertEqual(outcomes,[]); self.assertEqual(len(excluded),1)


if __name__ == '__main__': unittest.main()
