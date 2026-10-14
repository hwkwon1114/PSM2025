#!/bin/bash
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=2
export PYTHONPATH=.
/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/analysis/run_v16_identity_test.py
