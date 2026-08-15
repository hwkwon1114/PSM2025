#!/bin/bash
#SBATCH --job-name=psm_ewheel
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/pxl1051/build_verify_psm/ewheel-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/ewheel-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /gpfs/projects/p33082/PSM2025

echo "==== English-wheel bilayer: exact-Hessian Newton vs FD-Hessian Newton ===="
python - <<'PY'
import time, numpy as np, sys
sys.path.insert(0,'python')
from branch_continuation import build_bilayer_onesided, branch_continuation, minimize_energy_newton
E,nu,h,res=1.0,0.5,0.01,12
build=lambda t: build_bilayer_onesided(t,res=res,E=E,nu=nu,h=h)
print("bilayer exact-Hessian available:", hasattr(build(0.1),"hessian_tinyad"))
sched=[0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.75,0.9,1.0]
for label,ex in [("FD-Newton",False),("exact-Newton",True)]:
    mzr=lambda sh,x0,ex=ex: minimize_energy_newton(sh,x0,gtol=1e-8,maxiter=2000,exact_hvp=ex)
    print(f"\n--- {label} (certification uses the exact Hessian either way) ---")
    t=time.time()
    r=branch_continuation(build,sched,E=E,h=h,minimizer=mzr,verbose=True)
    dt=time.time()-t
    cert=[x for x in r.records if x.accepted]
    print(f"  => {label}: {dt:.1f}s  certified {len(cert)}/{len(r.records)}  stopped_at={r.stopped_at}"
          + (f"  last t={cert[-1].t:.3f} max|z|={cert[-1].max_z:.3e} lam_min={cert[-1].lam_min:+.3e}" if cert else ""))
PY
echo "==== EWHEEL_DONE ===="
