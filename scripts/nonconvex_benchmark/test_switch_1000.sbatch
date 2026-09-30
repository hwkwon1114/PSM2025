#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=sw_1000
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/switch_1000_%j.out
set -eo pipefail
export PROFILED_PHASE_ENTRY=$(date +%s)
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
qualified="$root/run/nonconvex_solver_benchmark/parallel_solver_gate_7723520"
out="$root/run/nonconvex_solver_benchmark/switch_1000_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT

export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg

cpus=$(python -c 'import os; a=sorted(os.sched_getaffinity(0)); assert len(a)>=4; print(",".join(map(str,a[:4])))')
echo "$cpus" > "$out/pinned_cpus.txt"

taskset -c "$cpus" python "$root/python/test_switch_threshold.py" --out "$out" --root "$root" --gate "$qualified" --switch-attempts 1000 --start cylinder_y_plus
