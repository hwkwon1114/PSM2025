#!/usr/bin/env python3
"""One-shot, non-destructive document relocation with byte-preservation audit.
Legacy aliases intentionally preserve scripts, archived provenance and bookmarks.
Does not execute notebooks, change scientific records, or delete build artifacts.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    audit = ROOT/'docs/reorganization_manifest.json'
    if audit.exists():
        raise SystemExit('Already reorganized; refusing to repeat migration')
    moves = []
    for directory in ('knowledge', 'reports', 'docs/reports'):
        for src in sorted((ROOT/directory).iterdir()):
            moves.append((src, ROOT/'docs'/src.name))
    moves += [(ROOT/'notebooks', ROOT/'docs/notebooks'),
              (ROOT/'paper_list', ROOT/'docs/papers')]
    destinations = [str(dst) for _, dst in moves]
    assert len(destinations) == len(set(destinations)), 'destination collision'
    assert all(not dst.exists() for _, dst in moves), 'existing destination'
    records = []
    for src, dst in moves:
        files = sorted(src.rglob('*')) if src.is_dir() else [src]
        for f in files:
            if f.is_file():
                target = dst/f.relative_to(src) if src.is_dir() else dst
                records.append(dict(old=str(f.relative_to(ROOT)), new=str(target.relative_to(ROOT)), sha256=digest(f)))
    for src, dst in moves:
        src.rename(dst)
    # Historical references retain their original spelling and contents.
    for old in ('knowledge', 'reports'):
        (ROOT/old).rmdir()
        (ROOT/old).symlink_to('docs', target_is_directory=True)
    (ROOT/'notebooks').symlink_to('docs/notebooks', target_is_directory=True)
    (ROOT/'paper_list').symlink_to('docs/papers', target_is_directory=True)
    # File aliases, not a self-referencing directory symlink.
    for src, dst in moves:
        if src.parent == ROOT/'docs/reports':
            src.symlink_to(Path('..')/dst.name, target_is_directory=dst.is_dir())
    for r in records:
        assert digest(ROOT/r['new']) == r['sha256'], r
        assert digest(ROOT/r['old']) == r['sha256'], r
    audit.write_text(json.dumps(dict(scope='byte-preserving relocation, not literature review or new experiment',
        historical_paths='compatibility aliases; do not treat as duplicate evidence',
        files=records, verified_files=len(records)), indent=2)+'\n')
    print(f'Relocated and verified {len(records)} files; historical aliases verified.')

if __name__ == '__main__':
    main()
