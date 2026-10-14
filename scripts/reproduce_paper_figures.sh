#!/usr/bin/env bash
# scripts/reproduce_paper_figures.sh
# One-click reproduction of paper figures
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-$(which python3)}"
if command -v conda &>/dev/null && [ -d "$HOME/miniforge/envs/DHD" ]; then
    PYTHON_BIN="$HOME/miniforge/envs/DHD/bin/python"
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" scripts/reproduce_paper_figures.py "$@"
