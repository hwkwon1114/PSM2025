#!/bin/bash
#SBATCH --job-name=zz_precond
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/zzprecond-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zzprecond-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
RUN=$WT/run/zz_seq_demo

echo "==== BUILD ===="
cd "$WT/build_cmake"; ninja 2>&1 | grep -iE "error|FAILED|ninja: build stopped|Linking" | tail -20 || true
echo "==== BUILD_DONE ===="

run_case () {  # name res minimizer
  local name=$1 res=$2 minz=$3
  local d="$RUN/$name"; mkdir -p "$d"; cd "$d"
  echo "==== CASE $name (res=$res minimizer=$minz) ===="
  /usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
    -sim bilayer_growth -case custom -geometry rectangle \
    -lx 0.13 -ly 0.16 -res "$res" -h_total 0.0005 \
    -growth_type zigzag_sequence -cycle_file "$RUN/zigzag_sequence.json" \
    -enable_passE false -nsteps 1 -tol 1e-8 -minimizer "$minz" \
    -basename "$name" -export_stl false 2>&1 | \
    grep -iE "HLBFGS-precond\]|cycle=[0-9].*energy=|completed .* cycles|WALL |Energy went" | tail -30
  echo "==== ${name}_DONE ===="
}

# correctness + scaling: does precond reach the same energy, and does it win at finer mesh?
run_case base_r010    0.010 hlbfgs
run_case precond_r010 0.010 hlbfgs_precond
run_case base_r006    0.006 hlbfgs
run_case precond_r006 0.006 hlbfgs_precond
run_case base_r004    0.004 hlbfgs
run_case precond_r004 0.004 hlbfgs_precond
echo "==== ALLDONE ===="
