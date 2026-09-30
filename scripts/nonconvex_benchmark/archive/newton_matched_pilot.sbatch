#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=newton_matched_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --nodelist=qnode0107
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/newton_pilot_%j.out
set -eo pipefail
begin=$SECONDS
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
build="$root/run/nonconvex_solver_benchmark/development_7380235"
binary="$build/source/bin/testshell"
out="$root/run/nonconvex_solver_benchmark/newton_pilot_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
mkdir "$out/source"
sha256sum -c "$build/test_binary.sha256"
cmp "$root/test/diagnostics/ShiftedNewtonFactorization.hpp" "$build/source/test/diagnostics/ShiftedNewtonFactorization.hpp"
cp "$root/python/run_nonconvex_screen.py" "$root/python/run_newton_pilot.py" "$root/python/prepare_near_equilibrium_matrix.py" "$out/source/"
cp "$root/test/diagnostics/ProfileFrozenFactorization.cpp" "$build/source/test/diagnostics/ShiftedNewtonFactorization.hpp" "$out/source/"
cp "$root/scripts/newton_matched_pilot.sbatch" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
find "$out/source" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/source_sha256.txt"
find "$CONDA_PREFIX/include/eigen3/Eigen" "$CONDA_PREFIX/include/nlohmann" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/near_dependency_sha256.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=false
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; conda list; } > "$out/environment.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
export PILOT_SETUP_SECONDS=$((SECONDS-begin))
printf 'Pilot root=%s CPU=%s; no scientific fits before all gates pass\n' "$out" "$cpu"
taskset -c "$cpu" python "$out/source/run_newton_pilot.py" --root "$out" --binary "$binary"
