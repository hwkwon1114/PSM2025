#!/bin/bash
#SBATCH --job-name=build_psm
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/build_clean.log
#SBATCH --error=/gpfs/projects/p33082/PSM2025-zigzag/build_clean.log

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38

cd /gpfs/projects/p33082/PSM2025-zigzag
cmake -B build_cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
  -DVTK_DIR="$CONDA_PREFIX/lib/cmake/vtk-9.3" \
  -DTBB_DIR="$CONDA_PREFIX/lib/cmake/TBB"

ninja -C build_cmake -j 2 shell testshell
bin/testshell
