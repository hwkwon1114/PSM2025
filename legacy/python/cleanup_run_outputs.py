"""Safely compact completed run cases and prune superseded search candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

KEEP_COMMON = {
    "argumentparser.log",
    "bilayer_zigzag_sequence_final_deformed.stl",
    "bilayer_zigzag_sequence_summary.csv",
    "evaluation.json",
    "result.json",
    "result.tsv",
    "sequence.json",
    "solver_certification.txt",
}


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def ensure_run_path(path: Path, repo_root: Path) -> Path:
    run_root = (repo_root / "run").resolve()
    resolved = path.resolve()
    if resolved == run_root or run_root not in resolved.parents:
        raise ValueError(f"Refusing path outside a run subdirectory: {path}")
    return path


def completed_result(case_dir: Path) -> tuple[str, str] | None:
    json_result = case_dir / "result.json"
    tsv_result = case_dir / "result.tsv"
    if json_result.is_file() and not json_result.is_symlink():
        result = json.loads(json_result.read_text(encoding="utf-8"))
        status = int(result.get("status", -1))
        verdict = str(result.get("verdict", ""))
        final_file = str(result.get("final_file", ""))
    elif tsv_result.is_file() and not tsv_result.is_symlink():
        with tsv_result.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if not rows:
            return None
        result = rows[-1]
        status = int(result["status"])
        verdict = result["verdict"]
        final_file = result["final_file"]
    else:
        return None
    if status != 0 or not verdict.startswith("MINIMUM") or not final_file:
        return None
    final_relative = Path(final_file)
    if final_relative.is_absolute() or final_relative.name != final_file:
        return None
    final_path = case_dir / final_relative
    if final_path.is_symlink() or not final_path.is_file():
        return None
    if final_path.resolve().parent != case_dir.resolve():
        return None
    return final_relative.name, verdict


def compact_case(case_dir: Path, apply: bool) -> tuple[int, int]:
    if case_dir.is_symlink():
        print(f"SKIP symlink {case_dir}")
        return 0, 0
    completion = completed_result(case_dir)
    if completion is None:
        print(f"SKIP incomplete or uncertified {case_dir}")
        return 0, 0
    final_file, _ = completion
    keep = KEEP_COMMON | {final_file}
    removed_files = 0
    removed_bytes = 0
    for entry in sorted(case_dir.iterdir()):
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.name in keep
            or entry.suffix.lower() in {".csv", ".json", ".png", ".tsv", ".txt"}
        ):
            continue
        size = entry.stat().st_size
        print(f"{'DELETE' if apply else 'WOULD DELETE'} {entry} ({format_bytes(size)})")
        if apply:
            entry.unlink()
        removed_files += 1
        removed_bytes += size
    return removed_files, removed_bytes


def compact_roots(roots: list[Path], repo_root: Path, apply: bool) -> None:
    total_files = 0
    total_bytes = 0
    for root in roots:
        root = ensure_run_path(root, repo_root)
        if (root / "result.json").is_file() or (root / "result.tsv").is_file():
            cases = [root]
        else:
            cases = [entry for entry in sorted(root.iterdir()) if entry.is_dir()]
        for case_dir in cases:
            removed_files, removed_bytes = compact_case(case_dir, apply)
            total_files += removed_files
            total_bytes += removed_bytes
    action = "Removed" if apply else "Would remove"
    print(f"{action} {total_files} files totaling {format_bytes(total_bytes)}")


def directory_size(path: Path) -> int:
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if not entry.is_symlink() and entry.is_file()
    )


def prune_search(
    root: Path,
    keep_names: set[str],
    drop_names: set[str],
    repo_root: Path,
    apply: bool,
) -> None:
    root = ensure_run_path(root, repo_root)
    overlap = keep_names & drop_names
    if overlap:
        raise ValueError(f"Names cannot be both kept and dropped: {sorted(overlap)}")
    invalid_drop_names = [
        name
        for name in drop_names
        if Path(name).is_absolute() or Path(name).name != name or name in {".", ".."}
    ]
    if invalid_drop_names:
        raise ValueError(f"Drop names must be immediate children: {sorted(invalid_drop_names)}")
    candidates = {
        entry.name: entry
        for entry in root.iterdir()
        if (
            entry.is_dir()
            and not entry.is_symlink()
            and (entry / "result.json").is_file()
            and completed_result(entry) is not None
        )
    }
    missing = keep_names - candidates.keys()
    if missing:
        raise ValueError(f"Requested keep cases do not exist or are incomplete: {sorted(missing)}")
    explicit_drops = [root / name for name in sorted(drop_names) if (root / name).exists()]
    invalid_drops = [
        entry for entry in explicit_drops if entry.is_symlink() or not entry.is_dir()
    ]
    if invalid_drops:
        raise ValueError(f"Explicit drops must be real directories: {invalid_drops}")
    targets = [entry for name, entry in candidates.items() if name not in keep_names]
    for entry in explicit_drops:
        if entry not in targets:
            targets.append(entry)
    removed_bytes = 0
    for target in sorted(targets):
        size = directory_size(target)
        print(f"{'DELETE TREE' if apply else 'WOULD DELETE TREE'} {target} ({format_bytes(size)})")
        if apply:
            shutil.rmtree(target)
        removed_bytes += size
    action = "Removed" if apply else "Would remove"
    print(f"{action} {len(targets)} directories totaling {format_bytes(removed_bytes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    compact_parser = subparsers.add_parser("compact")
    compact_parser.add_argument("roots", nargs="+", type=Path)
    compact_parser.add_argument("--apply", action="store_true")

    prune_parser = subparsers.add_parser("prune-search")
    prune_parser.add_argument("root", type=Path)
    prune_parser.add_argument("--keep", nargs="+", required=True)
    prune_parser.add_argument("--drop", nargs="*", default=[])
    prune_parser.add_argument("--apply", action="store_true")

    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "compact":
        compact_roots(args.roots, repo_root, args.apply)
    else:
        prune_search(
            args.root,
            set(args.keep),
            set(args.drop),
            repo_root,
            args.apply,
        )


if __name__ == "__main__":
    main()
