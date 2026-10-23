#!/bin/bash
#SBATCH --job-name=zz_escape
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/zzescape-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zzescape-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
TRAJ=$WT/trajectories
RUN=$WT/run/zz_seq_demo

echo "==== BUILD ===="
cd "$WT/build_cmake"; ninja 2>&1 | grep -iE "error|FAILED|ninja: build stopped|Linking" | tail -10 || true
echo "==== BUILD_DONE ===="

d="$RUN/escape_demo"; mkdir -p "$d"; cd "$d"
echo "==== CASE escape_demo : continuation (30x0.0098 @1e-8) + certified saddle-escape ===="
/usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
  -sim bilayer_growth -case custom -geometry rectangle \
  -lx 0.13 -ly 0.16 -res 0.008 -h_total 0.0005 \
  -growth_type zigzag_sequence -cycle_file "$TRAJ/zigzag_cont30.json" \
  -enable_passE false -nsteps 1 -tol 1e-8 -minimizer hlbfgs \
  -max_iter 50000 -seed_escape true -seed_max 12 \
  -basename escape_demo -export_stl false 2>&1 | \
  grep -iE "certify_final|seed_escape|completed .* cycles|WALL " | tail -40
echo "==== ALLDONE ===="
