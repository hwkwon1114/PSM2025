#!/usr/bin/env bash
# calibration/paper_figures/reproduce_paper_figures.sh
# One-click reproduction of paper figures
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON:-}" ]; then
    if [ -d "$HOME/miniforge/envs/DHD" ]; then
        PYTHON_BIN="$HOME/miniforge/envs/DHD/bin/python"
    else
        PYTHON_BIN="$(which python3)"
    fi
else
    PYTHON_BIN="$PYTHON"
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" calibration/paper_figures/reproduce_paper_figures.py "$@"
