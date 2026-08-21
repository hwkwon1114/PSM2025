#!/bin/bash
#SBATCH --job-name=zz_large
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/zzlarge-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zzlarge-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
RUN=$WT/run/zz_seq_demo

echo "==== BUILD ===="
cd "$WT/build_cmake"; ninja 2>&1 | grep -iE "error|FAILED|ninja: build stopped|Linking" | tail -10 || true
echo "==== BUILD_DONE ===="

run_case () {  # name
  local name=$1
  local d="$RUN/$name"; mkdir -p "$d"; cd "$d"
  echo "==== CASE $name (HLBFGS) LARGE deformation ===="
  /usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
    -sim bilayer_growth -case custom -geometry rectangle \
    -lx 0.13 -ly 0.16 -res 0.008 -h_total 0.0005 \
    -growth_type zigzag_sequence -cycle_file "$RUN/zigzag_large.json" \
    -enable_passE false -nsteps 1 -tol 1e-8 -minimizer hlbfgs \
    -basename "$name" -export_stl false 2>&1 | \
    grep -iE "cycle=[0-9].*energy=|max_displacement|completed .* cycles|WALL |Energy went" | tail -30
  # deflection from the summary CSV (max_U3 column) -- confirms it's genuinely large
  awk -F, 'NR>1{print "   cycle "$1"  max_U3="$18"  total_energy="$19}' \
    "$d/bilayer_zigzag_sequence_summary.csv" 2>/dev/null | tail -6
  echo "==== ${name}_DONE ===="
}

run_case large_hlbfgs
echo "==== ALLDONE ===="
