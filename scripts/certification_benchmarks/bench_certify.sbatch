#!/bin/bash
#SBATCH --job-name=zz_certify
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/zzcert-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zzcert-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
RUN=$WT/run/zz_seq_demo

echo "==== BUILD ===="
cd "$WT/build_cmake"; ninja 2>&1 | grep -iE "error|FAILED|ninja: build stopped|Linking" | tail -10 || true
echo "==== BUILD_DONE ===="

run_case () {  # name json tol
  local name=$1 json=$2 tol=$3
  local d="$RUN/$name"; mkdir -p "$d"; cd "$d"
  echo "==== CASE $name ($json tol=$tol) ===="
  /usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
    -sim bilayer_growth -case custom -geometry rectangle \
    -lx 0.13 -ly 0.16 -res 0.008 -h_total 0.0005 \
    -growth_type zigzag_sequence -cycle_file "$RUN/$json" \
    -enable_passE false -nsteps 1 -tol "$tol" -minimizer hlbfgs \
    -max_iter 50000 -certify_final true -basename "$name" -export_stl false 2>&1 | \
    grep -iE "certify_final|WALL " | tail -6
  awk -F, 'END{printf "   final max_U3=%.4e  E=%.4e\n", $18, $19}' \
      "$d/bilayer_zigzag_sequence_summary.csv" 2>/dev/null
}

# 1. the shallow continuation branch (fast)
run_case cert_shallow zigzag_cont30.json 1e-8
# 2. the deep tight-tol state (~30 min rerun of the 1e-12 brute force)
run_case cert_deep zigzag_large.json 1e-12
echo "==== ALLDONE ===="
