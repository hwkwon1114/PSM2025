"""Sequential GP / expected-improvement optimization of the measured shape loss.

The GP models log(RMS), not a posterior distribution over physical parameters.
Only real, successful forward evaluations can become the reported best fit.
"""
from pathlib import Path
import csv
import hashlib
import json
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn import __version__ as sklearn_version
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from .common import NAMES, resolve, save_json
from .optimize import remember_best
from .progress import message


def expected_improvement(mean, std, best, xi=0.01):
    """Expected positive reduction below best-xi, for a Gaussian prediction."""
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    improvement = best-mean-xi
    safe = np.maximum(std, 1e-15)
    z = improvement/safe
    ei = improvement*ndtr(z) + std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return np.where(std > 1e-15, np.maximum(ei, 0), np.maximum(improvement, 0))


def settings_for(config):
    default = {'initial_points': 12, 'candidate_pool': 2048, 'local_candidates': 512,
               'acquisition_restarts': 5, 'gp_restarts': 2, 'gp_alpha': 1e-6,
               'xi': .01, 'min_separation': 1e-5, 'failure_radius': .03,
               'max_consecutive_failures': 3, 'objective_floor_mm': 1e-6}
    supplied = config.get('bayesian', {})
    unknown = set(supplied)-set(default)
    if unknown:
        raise ValueError(f'Unknown Bayesian settings: {sorted(unknown)}')
    default.update(supplied)
    for key in ['initial_points','candidate_pool','local_candidates','acquisition_restarts','gp_restarts','max_consecutive_failures']:
        if not isinstance(default[key], int) or default[key] < (0 if key == 'gp_restarts' else 1):
            raise ValueError(f'{key} must be a valid integer')
    if default['initial_points'] < 3 or default['candidate_pool'] < 16:
        raise ValueError('Use at least 3 initial points and 16 candidate proposals')
    for key in ['gp_alpha','min_separation','failure_radius','objective_floor_mm']:
        if not np.isfinite(default[key]) or default[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not np.isfinite(default['xi']) or default['xi'] < 0:
        raise ValueError('xi must be finite and nonnegative')
    return default


def minimum_distance(x, old):
    x = np.atleast_2d(x)
    if len(old) == 0:
        return np.full(len(x), np.inf)
    return np.sqrt(np.min(np.sum((x[:,None,:]-np.asarray(old)[None,:,:])**2,axis=2),axis=1))


def propose(records, settings, seed):
    """Optimize cheap acquisition only; this function never invokes the shell solver."""
    valid = [r for r in records if r['valid']]
    old = np.array([r['normalized_parameters'] for r in records])
    failed = [r['normalized_parameters'] for r in records if not r['valid']]
    rng = np.random.default_rng(seed)
    power = int(np.ceil(np.log2(settings['candidate_pool'])))
    candidates = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(power)[:settings['candidate_pool']]
    if valid:
        best = min(valid,key=lambda r:r['objective_mm'])
        local = np.clip(np.array(best['normalized_parameters']) +
                        rng.normal(0,.12,(settings['local_candidates'],4)),0,1)
        candidates = np.vstack([candidates,local])
    def allowed(x):
        return ((minimum_distance(x,old)>settings['min_separation']) &
                (minimum_distance(x,failed)>settings['failure_radius']))
    candidates = candidates[allowed(candidates)]
    if not len(candidates):
        raise RuntimeError('No admissible proposals; review failure exclusion radius and bounds')
    info = {'acquisition':'space_filling', 'warnings':[]}
    if len(valid) < 3:
        return candidates[np.argmax(minimum_distance(candidates,old))],info
    train_x = np.array([r['normalized_parameters'] for r in valid])
    train_y = np.log(np.maximum([r['objective_mm'] for r in valid],settings['objective_floor_mm']))
    kernel = ConstantKernel(1.0,(.01,100)) * Matern(np.full(4,.3),(.03,3.0),nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel,alpha=settings['gp_alpha'],normalize_y=True,
                                  n_restarts_optimizer=settings['gp_restarts'],random_state=seed)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always',ConvergenceWarning)
            gp.fit(train_x,train_y)
        info['warnings'] = [str(w.message) for w in notices]
    except (ValueError,np.linalg.LinAlgError) as exc:
        info['warnings'] = [f'GP fit failed; using space filling: {exc}']
        return candidates[np.argmax(minimum_distance(candidates,old))],info
    def acquisition(x):
        x = np.atleast_2d(x)
        mean,std = gp.predict(x,return_std=True)
        value = expected_improvement(mean,std,float(train_y.min()),settings['xi'])
        return np.where(allowed(x),value,0.0)
    score = acquisition(candidates)
    # Refine several promising candidates using only the surrogate.
    refined = []
    for i in np.argsort(score)[-settings['acquisition_restarts']:]:
        result = minimize(lambda x:-float(acquisition(x)[0]),candidates[i],
                          method='L-BFGS-B',bounds=[(0,1)]*4,options={'maxiter':60})
        if np.isfinite(result.x).all() and allowed(result.x)[0]:
            refined.append(np.clip(result.x,0,1))
    if refined:
        candidates = np.vstack([candidates,refined]);score = acquisition(candidates)
    if not np.isfinite(score).all() or score.max() <= 1e-14:
        info['warnings'].append('Expected improvement collapsed; using space filling')
        return candidates[np.argmax(minimum_distance(candidates,old))],info
    index = int(np.argmax(score));x = candidates[index]
    mean,std = gp.predict(x[None,:],return_std=True)
    info.update({'acquisition':'expected_improvement_log_rms', 'expected_improvement':float(score[index]),
                 'predicted_log_rms':float(mean[0]), 'predicted_log_std':float(std[0]),
                 'kernel':str(gp.kernel_)})
    return x,info


def write_trace(base, records):
    path = base/'bayes_trace.csv'
    with path.with_suffix('.tmp').open('w',newline='') as f:
        fields = ['evaluation','valid','objective_mm','best_so_far_mm',*NAMES,'seconds','directory','error']
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        best = np.inf
        for i,r in enumerate(records,1):
            if r['valid']:best=min(best,r['objective_mm'])
            writer.writerow({'evaluation':i,'valid':r['valid'],
                'objective_mm':r['objective_mm'] if r['valid'] else '',
                'best_so_far_mm':best if np.isfinite(best) else '',**r['parameters'],
                'seconds':r.get('seconds',0),'directory':r['directory'],'error':r.get('error','')})
    path.with_suffix('.tmp').replace(path)
    if not any(r['valid'] for r in records):return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    y = np.array([r['objective_mm'] if r['valid'] else np.nan for r in records])
    cumulative=np.minimum.accumulate(np.where(np.isfinite(y),y,np.inf))
    cumulative[~np.isfinite(cumulative)]=np.nan
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    n=np.arange(1,len(records)+1)
    ax.plot(n,y,'o',ms=4,label='Successful forward evaluations',color='#e58b27')
    ax.plot(n,cumulative,'-',lw=2,label='Best measured RMS',color='#286aab')
    if np.all(y[np.isfinite(y)]>0):ax.set_yscale('log')
    ax.set(xlabel='Evaluation attempt',ylabel='Surface RMS (mm)',title='Bayesian calibration progress')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=160);plt.close(fig)
    tmp=base/'bayes_progress.tmp';tmp.write_bytes(buffer.getvalue());tmp.replace(base/'bayes_progress.png')


def bayesian_search(evaluator, resume=False, budget=None):
    """Checkpoint each completed attempt and any pending proposal; total-budget resume."""
    settings=settings_for(evaluator.config)
    budget=evaluator.config['optimization']['max_evaluations'] if budget is None else budget
    if not isinstance(budget,int) or budget<1:raise ValueError('Budget must be a positive integer')
    seed=int(evaluator.config['optimization']['seed'])
    signature=hashlib.sha256(json.dumps({'evaluator':evaluator.identity,'settings':settings,
                                        'seed':seed,'sklearn':sklearn_version},sort_keys=True).encode()).hexdigest()
    path=evaluator.base/'bayes_state.json'
    if path.exists():
        if not resume:raise ValueError('Bayesian state already exists; use --resume or a new output_dir')
        state=json.loads(path.read_text())
        if state.get('signature')!=signature:
            raise ValueError('Resume identity changed (inputs, code, executable, bounds or GP settings); use a new output_dir')
    else:
        if resume:raise ValueError('No Bayesian checkpoint exists to resume')
        design=qmc.LatinHypercube(d=4,seed=seed).random(settings['initial_points']-1)
        state={'schema_version':1,'signature':signature,'observations':[], 'pending':None,
               'design':np.vstack([evaluator.parameters.initial(),design]).tolist(), 'proposals':[],
               'status':'running'}
        save_json(path,state)
    records=state['observations']
    valid=[r for r in records if r['valid']]
    if valid:remember_best(evaluator,min(valid,key=lambda r:r['objective_mm']))
    trailing_failures=0
    for r in reversed(records):
        if r['valid']:break
        trailing_failures+=1
    state['status']='running'
    while len(records)<budget:
        number=len(records)
        if state['pending'] is not None:
            pending=state['pending'];x=np.array(pending['x']);info=pending['info']
        elif number<len(state['design']):
            x=np.array(state['design'][number]);info={'acquisition':'initial_design','warnings':[]}
        else:
            message('BAYES',f'Fitting GP using {sum(r["valid"] for r in records)} successful observations')
            x,info=propose(records,settings,seed+number)
        state['pending']={'x':x.tolist(),'info':info}
        save_json(path,state)
        current_best=min((r['objective_mm'] for r in records if r['valid']),default=None)
        message('BAYES',f'Trial {number+1}/{budget}; stage={info["acquisition"]}; best RMS={current_best}')
        record=evaluator.evaluate(x)
        # A failure is never passed to the GP as an enormous artificial loss.
        if record['valid'] and (not np.isfinite(record['objective_mm']) or record['objective_mm']<0):
            raise RuntimeError('Evaluator marked an invalid numerical loss successful')
        records.append(record);state['proposals'].append(info);state['pending']=None
        save_json(path,state)  # Durable before plotting or reporting.
        remember_best(evaluator,record)
        write_trace(evaluator.base,records)
        valid=[r for r in records if r['valid']]
        if valid:
            best=min(valid,key=lambda r:r['objective_mm'])
            save_json(evaluator.base/'best_parameters.json',best['parameters'])
            message('BEST',f'{best["objective_mm"]:.6g} mm after {len(records)} attempts')
        trailing_failures=0 if record['valid'] else trailing_failures+1
        if number==0 and not record['valid']:
            state['status']='initial_evaluation_failed';break
        if trailing_failures>=settings['max_consecutive_failures']:
            state['status']='consecutive_failures';break
    if state['status']=='running':state['status']='budget_complete'
    save_json(path,state)
    valid=[r for r in records if r['valid']]
    best=min(valid,key=lambda r:r['objective_mm']) if valid else None
    summary={'method':'Gaussian process / expected improvement on log RMS', 'status':state['status'],
             'completed_attempts':len(records),'successful_evaluations':len(valid),
             'failed_evaluations':len(records)-len(valid),'requested_total_budget':budget,
             'fresh_evaluations_this_invocation':evaluator.attempts, 'best':best,
             'near_bound_parameters':[] if best is None else
                 [name for name,x in zip(NAMES,best['normalized_parameters']) if x<.02 or x>.98],
             'note':'Budget completion is not proof of a global optimum or parameter identifiability.'}
    save_json(evaluator.base/'bayes_summary.json',summary)
    return summary


def run_bayesian(config, resume=False, budget=None):
    # Linux server workflow: reject concurrent BO writers to the same checkpoint.
    import fcntl
    from .runner import Evaluator
    base=resolve(config['output_dir']);base.mkdir(parents=True,exist_ok=True)
    with (base/'bayes.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Bayesian run holds this output directory')
        evaluator=Evaluator(config)
        return bayesian_search(evaluator,resume=resume,budget=budget)
