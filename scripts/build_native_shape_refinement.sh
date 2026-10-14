#!/usr/bin/env bash
# Full production target, isolated outputs, existing OFFLINE dependencies only.
set -euo pipefail
: "${SLURM_JOB_ID:?Compute-node build only}"
: "${CONDA_PREFIX:?Existing smcpp_vtk38 required}"
root=$(cd "$(dirname "$0")/.." && pwd);cd "$root"
stage=1
out="$root/run/native_shape_refinement/build_${SLURM_JOB_ID}"
[[ ! -e "$out" ]] || { echo 'Preserve existing build outputs' >&2; exit 1; }
mkdir -p "$out";exec > >(tee "$out/build.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
ss=/gpfs/software/2025/spack/opt/spack/linux-rhel8-x86_64/gcc-12.4.0/suite-sparse-7.7.0-m42btauyiweee53twzlcgjrlkdfsoffh
nv=/gpfs/software/2025/spack/opt/spack/linux-rhel8-x86_64/gcc-12.4.0/nvhpc-24.9-w6zqlov7fylj7q462chvqmpy6qwuqryt/Linux_x86_64/24.9/compilers/lib
libs="$ss/lib64:$CONDA_PREFIX/lib:$nv"
export LD_LIBRARY_PATH="$libs" OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
{ date -Is;hostname;printf 'Job=%s\n' "$SLURM_JOB_ID";"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" --version; } > "$out/environment.txt"
sha256sum CMakeLists.txt src/libshell/*.hpp src/libshell/*.cpp src/libshell/ReferenceHybrid/*.hpp \
    src/simulations/*.hpp src/simulations/*.cpp calibration/ewcal/*.py calibration/tests/test_native_forward.py \
    calibration/tests/mock_solver.py calibration/configs/run6.json test/testshell/Test_NativeHybridProduction.cpp "$0" > "$out/source.sha256"
set -x
timeout --signal=TERM --kill-after=3 90 "$CONDA_PREFIX/bin/cmake" -S "$root" -B "$out/cmake" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_FLAGS="-DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_MAX_STATIC_ALIGN_BYTES=64" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$libs -Wl,-rpath-link,$libs" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DTBB_ROOT="$CONDA_PREFIX" \
    -DSHELL_OUTPUT_ROOT="$out" -DENABLE_NATIVE_HYBRID=ON -DENABLE_CHOLMOD=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_SuiteSparse=ON -DCHOLMOD_INCLUDE_DIR="$ss/include" -DCHOLMOD_LIBRARY="$ss/lib64/libcholmod.so" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_TRIANGLE="$root/build_cmake/_deps/triangle-src" \
    -DFETCHCONTENT_SOURCE_DIR_LIBIGL="$root/build_cmake/_deps/libigl-src" \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$root/build_cmake/_deps/googletest-src" \
    -DLIBIGL_BUILD_TESTS=OFF -DLIBIGL_BUILD_TUTORIALS=OFF
targets=(shell)
if [[ "$stage" == 1 ]];then targets+=(native_hybrid_fixtures);fi
timeout --signal=TERM --kill-after=3 1000 "$CONDA_PREFIX/bin/cmake" --build "$out/cmake" --target "${targets[@]}" --parallel 1
sha256sum --check "$out/source.sha256" > "$out/source_check.txt"
sha256sum "$out/bin/shell" > "$out/binary.sha256"
# Tiny correctness fixtures only, never a four-parameter physical forward.
if [[ "$stage" == 1 ]];then
    start=$SECONDS
    timeout --signal=TERM --kill-after=2 30 "$out/bin/native_hybrid_fixtures" \
        --gtest_filter=NativeHybridProduction.* --gtest_output="xml:$out/native_fixtures.xml" \
        > "$out/native_fixtures.log" 2>&1
    printf '%s\n' "$((SECONDS-start))" > "$out/native_fixture_seconds.txt"
fi
# Native-only reassessment uses the final declared fixture command; Python
# reader source/configs are unchanged and its prior passing evidence is retained.
if [[ ${NATIVE_HANDOFF_FIXTURE_NATIVE_ONLY:-0} != 1 ]];then
    start=$SECONDS
    python_limit=30
    if [[ "$stage" == 1 ]];then python_limit=40;fi
    timeout --signal=TERM --kill-after=2 "$python_limit" "$CONDA_PREFIX/bin/python" -B -m unittest calibration.tests.test_native_forward -v \
        > "$out/python_fixtures.log" 2>&1
    printf '%s\n' "$((SECONDS-start))" > "$out/python_fixture_seconds.txt"
fi
printf 'FULL_NATIVE_EXECUTABLE_BUILT; TINY_FIXTURES_PASSED; NO_PHYSICAL_FORWARD_RUN\n' > "$out/complete.txt"
sha256sum "$root/python/analyze_shape_refinement.py" "$root/python/test_analyze_shape_refinement.py" > "$out/measurement_source.sha256"
