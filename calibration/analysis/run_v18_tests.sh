#!/bin/bash
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=2
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -m unittest calibration.tests.test_batch_v16 -v
