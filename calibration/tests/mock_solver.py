"""Synthetic executable used only by tests; NOT the shell mechanics model."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from calibration.ewcal.geometry import triangle_mesh, write_mesh

def main():
    args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
    if args.get('-mock_fail')=='true': return 4
    seq=json.loads(Path(args['-cycle_file']).read_text())
    op=seq['toolpaths'][0]['operation']; cycles=seq['toolpaths'][0]['repeat']
    nx,ny=11,16
    x,y=np.meshgrid(np.linspace(-.1,.1,nx),np.linspace(-.15,.15,ny))
    p=np.column_stack([x.ravel(),y.ravel(),np.zeros(nx*ny)]); triangles=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;triangles += [[a,a+1,a+nx],[a+1,a+nx+1,a+nx]]
    write_mesh(triangle_mesh(p,triangles),'initial.vtp')
    q=p.copy();u=q[:,0]/.1;v=q[:,1]/.15
    q[:,2]=op['gtop']*(u*u+.5*v*v+.1*op['ortho_top']*u*v)
    q[:,2]+=10*op['gbot']*(.2*u*u+.7*v*v+.05*op['ortho_bottom']*u*v)
    mesh=triangle_mesh(q,triangles)
    for i,name in enumerate(['abar_top_11','abar_top_12','abar_top_22','abar_bot_11','abar_bot_12','abar_bot_22','total_hit_count']):
        arr=numpy_to_vtk(np.full(len(triangles),float(cycles+i)),deep=True);arr.SetName(name);mesh.GetCellData().AddArray(arr)
    write_mesh(mesh,'final.vtp')
    mode=args['-sequence_solve_mode'];solves=[cycles] if mode=='final_only' else list(range(1,cycles+1))
    Path('sequence_history.csv').write_text('cycle,solved,hit_events,max_total_hits\n'+''.join(f'{c},{int(c in solves)},100,{c}\n' for c in range(1,cycles+1)))
    m={'schema_version':1,'completed':True,'cycles':cycles,'solve_mode':mode,'equilibrium_solves':len(solves),
       'initial_mesh':'initial.vtp','final_mesh':'final.vtp','total_energy':.01,'backend':'hlbfgs','stepwise':False,
       'solves':[{'cycle':c,'return_code':0,'reported_eps':1e-12,'return_code_available':True} for c in solves]}
    Path('sequence_result.json').write_text(json.dumps(m));return 0
if __name__=='__main__':sys.exit(main())
