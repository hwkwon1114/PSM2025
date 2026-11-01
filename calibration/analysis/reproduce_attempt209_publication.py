"""Recreate the seven delivered 16 pt figures in a fresh directory, without solver runs."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis import attempt209_horizontal_overlays as overlays
from calibration.analysis import attempt209_restyle_other_figures as other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f'Choose a fresh output directory: {args.out_dir}')
    other.main(['--out-dir', str(args.out_dir)])
    overlays.main(['--source-dir', str(args.out_dir), '--out-dir', str(args.out_dir)])
    (args.out_dir / 'README.md').write_text(
        '# Attempt 209 publication figures\n\nSeven PDF/PNG figures, reproduced from saved inputs. '
        'The historical optimization and conditional GP use the older objective; corrected shape scores are not its convergence history. '
        'Run 1 was calibrated; Runs 2–3 are previously seen exploratory transfers, not blind validation. '
        'The GP is descriptive, not a physical-parameter posterior. See individual captions and `provenance.json`.\n\n'
        'All visible text is 16 pt at the exported dimensions: x–y 14×5 inches; 3-D 18×7; '
        'other figures 14 inches wide. Text shrinks when resized; journal layout is not verified. '
        'Regenerate with `python calibration/analysis/reproduce_attempt209_publication.py --out-dir NEW_DIRECTORY`. '
        'Existing output directories are refused. No simulation or registration is run.\n')
    print(args.out_dir)


if __name__ == '__main__':
    main()
