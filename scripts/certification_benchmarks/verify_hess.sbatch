#!/bin/bash
#SBATCH --job-name=zz_verify
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/pxl1051/build_verify_psm/zzverify-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zzverify-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
RUN=$WT/run/zz_seq_demo

echo "==== BUILD (TinyAD first compile) ===="
cd "$WT/build_cmake"
ninja 2>&1 | grep -iE "error|FAILED|ninja: build stopped|Linking|warning: .*TinyAD" | tail -30 || true
echo "==== BUILD_DONE ===="

cd "$RUN"
echo "==== VERIFY TinyAD Hessian vs CombinedOperator_Parametric ===="
"$WT/bin/shell" \
  -sim bilayer_growth -case custom -geometry rectangle \
  -lx 0.13 -ly 0.16 -res 0.01 -h_total 0.0005 \
  -growth_type zigzag_sequence -cycle_file "$RUN/zigzag_sequence.json" \
  -enable_passE false -nsteps 1 -verify_hessian true 2>&1 | grep -iE "verify_hessian"
echo "==== VERIFY_DONE ===="
