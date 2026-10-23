#!/usr/bin/env python3
"""Read-only audit of all six long-run endpoints/curves; no numerical fits.
Outputs exclusive timing attribution, complete plotted trace tables, and
proper-rigid-aligned endpoint geometry using fixed material correspondences.
"""
import argparse,csv,hashlib,json,os,sys,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,cm,ticker
from matplotlib.text import Text
from PIL import Image

STYLES={'lbfgs':('L-BFGS','#0072B2','-','o'), 'newton':('Newton','#D55E00','--','s'), 'hybrid':('Hybrid','#009E73','-.','^')}
CATS=['Hessian','Other model','Factorization','Symbolic + solve','Energy / gradient','L-BFGS direction','Unattributed']
COLORS=['#0072B2','#56B4E9','#D55E00','#CC79A7','#009E73','#E69F00','#BBBBBB']
HATCH=['','..','//','xx','\\\\','++','']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def align(a,b):
 ca=a.mean(0);cb=b.mean(0);u,_,v=np.linalg.svd((a-ca).T@(b-cb));d=np.eye(3);d[2,2]=np.linalg.det(u@v);r=u@d@v
 assert abs(np.linalg.det(r)-1)<1e-12
 return (a-ca)@r+cb

def exclusive(trace,charged):
 def total(k):return sum(float(v.get(k,0)) for v in trace)
 values=[total('hessian_oracle_seconds'),sum(max(0,v.get('model_seconds',0)-v.get('hessian_oracle_seconds',0)) for v in trace),total('factorization_seconds'),total('symbolic_seconds')+total('solve_seconds'),total('energy_gradient_seconds'),total('lbfgs_direction_seconds')]
 residual=charged-sum(values);assert residual>-1e-5,(charged,values)
 return dict(zip(CATS,values+[max(0,residual)]))

def export(fig,out,stem):
 # Basic margin audit: labels must not leave the canvas. This does not certify
 # legend/data separation or projected 3D-label layout; visual review follows.
 fig.canvas.draw()
 # Freeze positions before PDF/PNG backends change font metrics or DPI. The
 # initial constrained-layout solve must not silently change between exports.
 fig.set_layout_engine(None)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
 issues=[]
 for ax in fig.axes:
  texts=[ax.title,ax.xaxis.label,ax.yaxis.label]+ax.texts
  legend=ax.get_legend()
  if legend:texts+=legend.get_texts()
  for text in texts:
   if not text.get_visible() or not text.get_text():continue
   b=text.get_window_extent(renderer)
   if b.x0<bounds.x0-2 or b.y0<bounds.y0-2 or b.x1>bounds.x1+2 or b.y1>bounds.y1+2:issues.append(text.get_text())
 assert not issues,('text outside canvas',stem,issues)
 fig.savefig(out/(stem+'.pdf'),dpi=240)
 fig.savefig(out/(stem+'.png'),dpi=160)
 plt.close(fig)
 with Image.open(out/(stem+'.png')) as im:im.convert('L').save(out/(stem+'_grayscale.png'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 ledger=read(a.root/'screen/ledger.json');protocol=read(a.root/'protocol.json');assert ledger['completed'] and len(ledger['cells'])==6
 assert {(c['start'],c['recipe']) for c in ledger['cells']}=={(s,r) for s in ['cylinder_y_plus','cylinder_y_minus'] for r in STYLES}
 bundle=Path(protocol['request_base']['bundle']);manifest=read(bundle/'manifest.json')
 for file,h in manifest['sha256'].items():assert sha(bundle/file)==h,file
 assert sha(bundle/'manifest.json')==protocol['request_base']['identity']
 meta=read(bundle/'metadata.json');nv,nf,nd=meta['nv'],meta['nf'],meta['nd']
 ref=np.fromfile(bundle/'reference.f64',dtype='<f8').reshape(nv,3,order='F');faces=np.fromfile(bundle/'faces.i32',dtype='<i4').reshape(nf,3,order='F')
 records=[];curves={};shapes={};provenance={str(a.root/'screen/ledger.json'):sha(a.root/'screen/ledger.json'),str(bundle/'manifest.json'):sha(bundle/'manifest.json')}
 for cell in ledger['cells']:
  result=cell['result'];cp=Path(result['checkpoint']);h=sha(cp);assert h==cell['checkpoint_sha256'];provenance[str(cp)]=h
  raw=read(cp);state=raw['payload']['state'];recipe=raw['payload']['config'];trace=state['trace']
  assert recipe['tolerance']==5e-14 and recipe['maxAttempts']==100000 and recipe['maxEvaluations']==200000
  if result['accepted']:assert result['status']=='gradient_target' and result['selected_gradient_norm']<=5e-14 and result['curvature']=='positive_pivots'
  valid=[v for v in trace if 'energy' in v and 'gradient_norm' in v]
  data=np.array([[v['elapsed'],v['energy'],v['gradient_norm'],v.get('iterations',np.nan)] for v in valid]);assert np.isfinite(data[:,:3]).all() and (data[:,1:3]>0).all() and (np.diff(data[:,0])>=0).all()
  curves[cell['index']]=data
  with (a.out/('trace_%02d.csv'%cell['index'])).open('x',newline='') as f:
   w=csv.writer(f);w.writerow(['optimizer_elapsed_s','energy','native_gradient_norm','iterations']);w.writerows(data)
  wdir=Path(cell['out']);q=np.fromfile(wdir/'final_x.f64',dtype='<f8');assert q.size==nd and np.isfinite(q).all()
  assert np.allclose(q,np.array(state['x']),rtol=0,atol=0)
  if result['accepted']:assert np.array_equal(q,np.fromfile(wdir/'selected_x.f64',dtype='<f8'))
  shapes[cell['index']]=q[:3*nv].reshape(nv,3,order='F');provenance[str(wdir/'final_x.f64')]=sha(wdir/'final_x.f64')
  phases={}
  for k in sorted({k for v in trace for k in v if k.endswith('_seconds')}):
   vals=[v[k] for v in trace if k in v];phases[k]=dict(seconds=sum(vals),median=statistics.median(vals),count=len(vals))
  nr=[v for v in trace if 'hessian_oracle_seconds' in v];steps=[v for v in trace if 'shift' in v];positive=[v for v in steps if v['shift']>0]
  hybrid=cell['recipe']=='hybrid';switch=next((v['elapsed'] for v in trace if v.get('iterations')==2000),None) if hybrid else None
  thresholds={}
  for target in [1e-8,1e-10,1e-12,5e-14]:
   indices=np.flatnonzero(data[:,2]<=target);thresholds[str(target)]=float(data[indices[0],0]) if len(indices) else None
  tail=data[data[:,0]<=.5*data[-1,0]][-1]
  r=dict(index=cell['index'],start=cell['start'],recipe=cell['recipe'],accepted=result['accepted'],status=result['status'],charged_seconds=cell['charged_seconds'],optimizer_elapsed=result['optimizer_elapsed'],energy=result['energy'],gradient=result['gradient_norm'],iterations=result['iterations'],evaluations=result['evaluations'],hessians=result['hessians'],phases=phases,exclusive_seconds=exclusive(trace,cell['charged_seconds']),switch_elapsed=switch,
   newton_updates=len(steps),shifted_updates=len(positive),unshifted_updates=len(steps)-len(positive),numerical_factorizations=sum(v.get('numerical_factorizations',0) for v in trace),
   backtracked_newton_updates=sum(v.get('alpha',1)<1 for v in steps),last_shifted_elapsed=positive[-1]['elapsed'] if positive else None,
   first_gradient_crossings_seconds=thresholds,second_half_energy_reduction=(tail[1]-result['energy'])/tail[1],trace_rows=len(trace),plotted_rows=len(valid),last_trace=trace[-1])
  records.append(r)
  del raw,state,trace
 # All endpoints aligned to the accepted + hybrid oriented in the reference frame.
 anchor=align(shapes[2],ref);aligned={i:align(v,anchor) for i,v in shapes.items()}
 differences={i:np.linalg.norm(v-aligned[2 if i<3 else 3],axis=1)*1000 for i,v in aligned.items()}
 for r in records:
  d=differences[r['index']];r['shape_difference_to_same_start_hybrid_mm']=dict(rms=float(np.sqrt(np.mean(d*d))),p95=float(np.quantile(d,.95)),maximum=float(d.max()))
  with (a.out/('shape_residual_%02d.csv'%r['index'])).open('x',newline='') as f:
   w=csv.writer(f);w.writerow(['vertex_id','distance_to_same_start_hybrid_mm']);w.writerows(enumerate(d))
 agg={}
 for name in STYLES:
  group=[r for r in records if r['recipe']==name];charged=sum(r['charged_seconds'] for r in group)
  exclusive_sum={key:sum(r['exclusive_seconds'][key] for r in group) for key in CATS}
  agg[name]=dict(charged_seconds=charged,exclusive_seconds=exclusive_sum,fractions={k:v/charged for k,v in exclusive_sum.items()})
 audit=dict(cells=records,aggregate=agg,source_hashes=provenance,job=os.getenv('SLURM_JOB_ID'),script_sha256=sha(__file__),numpy=np.__version__,matplotlib=matplotlib.__version__,styles=STYLES,
   caveats=['All six cells and both predeclared related starts shown; no held-out dataset or independent replications, no uncertainty intervals.',
   'Time curves use recorded cumulative optimizer elapsed time including in-loop checkpoints, not extra parent/final verification time.',
   'Model and Hessian timers are nested. Exclusive categories subtract Hessian only from present parent-model timers; orphan terminal Hessian calls counted once.',
   'Unattributed includes checkpointing, verification, uninstrumented work and incomplete timers; not a measured I/O total.',
   'All final shapes are fixed-ID proper-rigid aligned, no reflection/scaling. Capped endpoints are unfinished, not different certified minima.',
   'No fitting, gradients, Hessians, or curvature recomputed in this audit.',
   'Surface rendering does not test self-contact/interpenetration or physical model adequacy.'],render_review='pending')
 (a.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey='row',layout='constrained')
 for r in records:
  col=0 if r['index']<3 else 1;d=curves[r['index']];label,color,ls,marker=STYLES[r['recipe']]
  for i,scale in [(0,1e9),(1,1)]:
   axes[i,col].plot(d[:,0]/60,d[:,i+1]*scale,color=color,ls=ls,lw=1.2,label=label)
   axes[i,col].plot(d[-1,0]/60,d[-1,i+1]*scale,color=color,marker=marker,ms=5,mfc=color if r['accepted'] else 'white')
   if r['switch_elapsed'] is not None:axes[i,col].axvline(r['switch_elapsed']/60,color=color,ls=':',lw=.8)
 for j in range(2):
  axes[0,j].set_title(['Positive start','Negative start'][j]);axes[1,j].set_xlabel('Recorded optimizer time (min)');axes[1,j].set_yscale('log');axes[1,j].axhline(5e-14,color='black',ls=':',lw=1,label='Acceptance target')
  for i in range(2):axes[i,j].set_xlim(0,20);axes[i,j].grid(alpha=.2)
 axes[0,0].set_ylabel('Energy (×10⁻⁹)');axes[1,0].set_ylabel('Native gradient norm');axes[0,0].legend(loc='upper right');axes[1,0].text(.03,.08,'Target: 5×10⁻¹⁴',transform=axes[1,0].transAxes,fontsize=10)
 export(fig,a.out,'convergence_progress')
 # Non-overlapping time categories across ALL six fits, sorted by start/recipe.
 order=[0,1,2,5,4,3];fig,ax=plt.subplots(figsize=(10,5),layout='constrained');left=np.zeros(6)
 for k,color,hatch in zip(CATS,COLORS,HATCH):
  v=np.array([records[i]['exclusive_seconds'][k]/60 for i in order]);ax.barh(np.arange(6),v,left=left,label=k,color=color,hatch=hatch,edgecolor='white',linewidth=.4);left+=v
 ax.set_yticks(np.arange(6));ax.set_yticklabels([STYLES[records[i]['recipe']][0]+(' (+)' if i<3 else ' (−)') for i in order]);ax.invert_yaxis();ax.set_xlabel('Charged cell time (min)');ax.set_xlim(0,24);ax.legend(loc='upper right',fontsize=9);ax.grid(axis='x',alpha=.15)
 export(fig,a.out,'convergence_timing')
 # Final deformed midsurfaces: all six endpoints, true physical aspect.
 verts={i:v*1000 for i,v in aligned.items()};stack=np.concatenate(list(verts.values()));lo=stack.min(0);hi=stack.max(0);extent=hi-lo
 norm=colors.Normalize(lo[2],hi[2]);fig=plt.figure(figsize=(12,8),layout='constrained');axes=[]
 for pos,i in enumerate(order):
  r=records[i];ax=fig.add_subplot(2,3,pos+1,projection='3d');axes.append(ax);v=verts[i]
  surf=ax.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=faces,cmap='viridis',norm=norm,linewidth=0,antialiased=False,shade=False);surf.set_rasterized(True)
  ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(extent);ax.view_init(elev=27,azim=-60)
  ax.set_title(STYLES[r['recipe']][0]+(' (+)' if i<3 else ' (−)')+'\n'+('Accepted' if r['accepted'] else 'Time capped'),fontsize=10)
  ax.set_xlabel('x (mm)',labelpad=1);ax.set_ylabel('y (mm)',labelpad=1);ax.set_zlabel('z (mm)' if pos%3==2 else '',labelpad=1)
  for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(ticker.MaxNLocator(3));axis.set_tick_params(labelsize=8)
 fig.colorbar(cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axes,shrink=.6,pad=.03,label='Aligned height (mm)')
 export(fig,a.out,'convergence_final_shapes')
 # Fixed reference coordinates show every material-vertex displacement error.
 fig,axes=plt.subplots(2,3,figsize=(10,7),sharex=True,sharey=True);fig.subplots_adjust(left=.10,right=.86,bottom=.09,top=.94,hspace=.22,wspace=.18);vmax=max(d.max() for d in differences.values())
 for pos,i in enumerate(order):
  ax=axes.flat[pos];r=records[i]
  artist=ax.tripcolor(ref[:,0]*1000,ref[:,1]*1000,faces,differences[i],shading='gouraud',cmap='cividis',vmin=0,vmax=vmax,rasterized=True)
  ax.set_aspect('equal');ax.set_title(STYLES[r['recipe']][0]+(' (+)' if i<3 else ' (−)'),fontsize=10)
  if pos>=3:ax.set_xlabel('Reference x (mm)')
  if pos%3==0:ax.set_ylabel('Reference y (mm)')
 colorax=fig.add_axes([.89,.13,.022,.75]);fig.colorbar(artist,cax=colorax,label='Distance to same-start hybrid (mm)')
 export(fig,a.out,'convergence_shape_differences')
 captions={
 'convergence_progress':'All six saved trajectories. Energy scaled by 10^9; gradient logarithmic. All finite energy/gradient observations retained without smoothing/downsampling. Failure-only rows lacking observations omitted and counted in audit.json. Green vertical dotted lines mark the fixed hybrid switch after 2000 L-BFGS attempts. Filled endpoint symbols are accepted minima, open symbols are time caps. The horizontal target alone is not sufficient for acceptance: independent gradient and positive restricted pivots are also required. Time is recorded optimizer elapsed, not parent/final-verification time. Two related starts, no confidence intervals.',
 'convergence_timing':'All six charged cell times, split into disjoint measured categories and a residual. Hessian cost is removed from inclusive model time; symbolic and solve combined. Unattributed time includes checkpoint/verification/uninstrumented work and incomplete timers, and must not be called measured I/O. Bars start at zero; no censored cell is omitted. Within-loop timers are descriptive and compare trajectories visiting different states.',
 'convergence_final_shapes':'All six final midsurfaces at their actual stopping points, not all local minima. Proper rigid alignment to the accepted positive-start hybrid oriented to the reference frame; same vertex IDs, no reflection, deformation or magnification. Common axes, camera, height scale and true physical aspect. Full mesh (5427 vertices, 10560 faces) retained; meshes rasterized in PDF, text remains vector. Rows positive/negative starts, columns L-BFGS/Newton/hybrid. Capped shapes are unfinished trajectories.',
 'convergence_shape_differences':'Full material-vertex displacement fields after proper rigid alignment, relative to the accepted hybrid for the same start. Displayed on fixed reference coordinates with one shared sequential scale; no vertices or cases omitted, hybrid self-difference is exactly zero. These are differences between computed endpoints, not errors relative to physical truth. Numerical per-vertex data and RMS/p95/max provided. The two starts are related, not independent samples.'}
 for stem,text in captions.items():(a.out/(stem+'.caption.md')).write_text(text+'\n')
 print(json.dumps({'aggregate':agg,'cells':[{k:r[k] for k in ['index','recipe','charged_seconds','switch_elapsed','newton_updates','shifted_updates','unshifted_updates','last_shifted_elapsed','numerical_factorizations','backtracked_newton_updates','first_gradient_crossings_seconds','shape_difference_to_same_start_hybrid_mm']} for r in records]},indent=2))
if __name__=='__main__':main()
