#!/bin/bash
#SBATCH --job-name=psm_tinyad_fixes
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/tinyad_fixes-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/tinyad_fixes-%j.out

# Build pyshell with the two TinyAD correctness fixes (FIX #1 Dirichlet masking, FIX #2
# smooth flat-state dihedral) and run all acceptance gates. Fresh out-of-source build.

set -eo pipefail

SRC=/gpfs/projects/p33082/PSM2025
BLD=/home/pxl1051/build_pyshell_psm
mkdir -p "$BLD" /home/pxl1051/build_verify_psm

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

cd "$SRC/python"

echo "==== REGRESSION 1: monolayer 2b full energy+gradient (verify_tinyad_bending.py) ===="
python verify_tinyad_bending.py || echo "WARN: mono bending verify returned nonzero"

echo "==== REGRESSION 2: monolayer exact Hessian (verify_tinyad_hessian.py) ===="
python verify_tinyad_hessian.py || echo "WARN: mono hessian verify returned nonzero"

echo "==== REGRESSION 3: bilayer exact Hessian (verify_tinyad_bilayer_hessian.py) ===="
python verify_tinyad_bilayer_hessian.py || echo "WARN: bilayer hessian verify returned nonzero"

echo "==== NEW GATES: FIX #1 masking + FIX #2 flat finiteness (verify_tinyad_fixes.py) ===="
python verify_tinyad_fixes.py

echo "==== TINYAD_FIXES_BUILD_VERIFY_DONE ===="
