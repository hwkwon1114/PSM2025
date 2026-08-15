#!/bin/bash
#SBATCH --job-name=psm_pyshell
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/pyshell-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/pyshell-%j.out

# STAGE 2a: build the pyshell python module (with the TinyAD stretching binding)
# and run the verification. Fresh out-of-source build; the committed build_cmake/
# is bound to a different collaborator's checkout.

# NB: no `-u` (nounset) -- conda's (de)activate scripts reference unset vars.
set -eo pipefail

SRC=/gpfs/projects/p33082/PSM2025
BLD=/home/pxl1051/build_pyshell_psm
mkdir -p "$BLD"

echo "==== job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cores ===="

source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

VTKDIR="$CONDA_PREFIX/lib/cmake/$(ls "$CONDA_PREFIX/lib/cmake" | grep -E '^vtk-9\.' | head -n1)"
echo "VTK_DIR=$VTKDIR"
echo "PYTHON=$(which python)  $(python --version)"

rm -rf "$BLD"/CMakeCache.txt "$BLD"/CMakeFiles "$BLD"/build.ninja
cd "$BLD"

echo "==== configure (BUILD_PYTHON_BINDINGS=ON) ===="
cmake "$SRC" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/lib/cmake" \
  -DVTK_DIR="$VTKDIR" \
  -DTBB_DIR="$CONDA_PREFIX/lib/cmake/TBB" \
  -DTBB_ROOT="$CONDA_PREFIX" -DVTK_ROOT="$CONDA_PREFIX" -DGSL_ROOT_DIR="$CONDA_PREFIX"

echo "==== build pyshell (-j $SLURM_CPUS_PER_TASK) ===="
ninja -j "$SLURM_CPUS_PER_TASK" pyshell

echo "==== pyshell .so (emitted to project python/) ===="
ls -la "$SRC"/python/pyshell*.so

echo "==== VERIFY ===="
cd "$SRC/python"
python verify_tinyad_stretching.py
echo "==== PYSHELL_BUILD_VERIFY_DONE ===="
