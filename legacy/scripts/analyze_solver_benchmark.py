#!/usr/bin/env python3
"""Canonical entry point delegating to calibration/analysis/analyze_solver_benchmark.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'legacy/analysis'))

from analyze_solver_benchmark import analyze

if __name__ == '__main__':
    analyze()
