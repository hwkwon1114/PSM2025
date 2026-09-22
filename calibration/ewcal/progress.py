"""Low-cost, flushed progress messages and child-process diagnostics."""
from pathlib import Path
import time

def message(stage, text):
    print(f'[{time.strftime("%H:%M:%S")}] {stage}: {text}', flush=True)

def latest_diagnostic(folder):
    paths = list(Path(folder).glob('*_diagnostics.dat'))
    if not paths:
        return 'No diagnostics file yet; solver output may be buffered.'
    path = max(paths, key=lambda p: p.stat().st_mtime)
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-4096))
        lines = f.read().decode(errors='replace').splitlines()
    rows = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')]
    age = max(0, time.time()-path.stat().st_mtime)
    if not rows:
        return f'{path.name}: no numeric rows yet.'
    return f'{path.name}: {rows[-1][:220]} (file age {age:.0f}s; diagnostic columns depend on backend)'
