#!/bin/bash
#SBATCH --job-name=psm_exacth
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/pxl1051/build_verify_psm/exacth-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/exacth-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /gpfs/projects/p33082/PSM2025

echo "==== Stage 4: exact Hessian payoff (monolayer, res 24) ===="
python - <<'PY'
import time, numpy as np, sys
sys.path.insert(0,'python')
from verify_bindings import build_validation_disk, relax_in_plane
from branch_continuation import minimize_energy_newton
from stage_check import assess_stage, hessian_lowest_eig
E,nu,h,res=1.0,0.5,0.01,24
print("exact methods present:", hasattr(build_validation_disk(res=8), "hessian_vector_product_tinyad"))

# --- (1) SPEED: FD-Newton vs exact-Newton from the same curved warm start ---
s=build_validation_disk(res=res,E=E,nu=nu,h=h,swelling=1.0)
xb,_=relax_in_plane(s)
# seed onto the buckled branch so there's real Newton work to do
_,vecs,idx=hessian_lowest_eig(s,xb,k=1,mode="out_of_plane",want_vectors=True)
mode=np.zeros(s.n_dofs); mode[idx]=vecs[:,0]; nV=s.n_vertices
zmax=np.abs(mode[2*nV:3*nV]).max(); mode/= (zmax if zmax>0 else 1)
x0=xb+0.1*mode
for label,ex in [("FD-HvP",False),("exact-HvP",True)]:
    t=time.time()
    xr,r=minimize_energy_newton(s,x0,gtol=1e-8,maxiter=2000,exact_hvp=ex)
    dt=time.time()-t
    s.set_dofs(xr)
    print(f"  {label:9s}: {dt:6.2f}s  nit={r.nit:3d}  nhev={getattr(r,'nhev','?')}  njev={getattr(r,'njev','?')}  E={s.energy():.6e}  maxz={np.abs(s.vertices()[:,2]).max():.3e}")

# --- (2) CERTIFICATION: exact vs FD smallest eigenvalue at the flat saddle ---
print("\n  certification at flat swollen state (lowest 3 out-of-plane eigenvalues):")
for label,ex in [("FD  ",False),("exact",True)]:
    t=time.time(); vals=hessian_lowest_eig(s,xb,k=3,mode="out_of_plane",exact=ex); dt=time.time()-t
    print(f"    {label}: {dt:5.2f}s  lam = " + "  ".join(f"{v:+.6e}" for v in np.atleast_1d(vals)))
PY
echo "==== EXACTHESS_DONE ===="
