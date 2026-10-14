#!/usr/bin/env bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --nodelist=qnode0102
#SBATCH --job-name=shape-refine-build
#SBATCH --chdir=/gpfs/projects/p33082/PSM2025-zigzag
#SBATCH --output=run/native_shape_refinement/build-%j.out
#SBATCH --error=run/native_shape_refinement/build-%j.err
# Build only. No physical solves. Activate compiler environment without nounset,
# matching the previously successful native full build rather than runtime hooks.
set -eo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
source /gpfs/home/pxl1051/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export NATIVE_HANDOFF_FIXTURE_NATIVE_ONLY=1
bash scripts/build_native_shape_refinement.sh
