#!/bin/bash
#SBATCH --job-name=psm_farrell
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/pxl1051/build_verify_psm/farrell-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/farrell-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /gpfs/projects/p33082/PSM2025

echo "==== Farrell residual-deflation on the english-wheel bilayer (t=0.4, res 12) ===="
python - <<'PY'
import numpy as np, sys
sys.path.insert(0,'python')
from branch_continuation import build_bilayer_onesided
from global_search import farrell_deflation
E,nu,h,res=1.0,0.5,0.01,12
build=lambda t: build_bilayer_onesided(t,res=res,E=E,nu=nu,h=h)
print("exact Hessian for certification:", hasattr(build(0.1),"hessian_tinyad"))
r=farrell_deflation(build,0.4,E=E,h=h,max_solutions=10)
print(f"\ntally: {r.tally}")
print(f"=> {len(r.minima)} distinct certified minima by Farrell deflation (ranked):")
for i,m in enumerate(r.minima):
    tag="  <== LOWEST FOUND" if i==0 else ""
    print(f"   [{i}] E={m.energy:.6e}  max|z|={m.max_z:.3e}  lam_min={m.lam_min:+.3e}  via {m.seed}{tag}")
PY
echo "==== FARRELL_DONE ===="
