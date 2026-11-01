#!/usr/bin/env python3
"""Stage the five non-overlay Attempt 209 figures with uniform 16 pt text."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis import attempt209_publication as pub
from calibration.ewcal.common import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=pub.ROOT / 'calibration/paper_figures/attempt209_publication')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args(argv)
    previous = json.loads((args.source_dir / 'provenance.json').read_text())
    stage = args.out_dir or Path(tempfile.mkdtemp(prefix='attempt209_other_16pt_', dir=ROOT / 'run'))
    if stage.exists() and any(stage.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {stage}')
    stage.mkdir(parents=True, exist_ok=True)
    pub.OUT = stage
    pins = {}
    rows = [pub.shapes.load(run, pins) for run in pub.RUNS]
    for path in (pub.HISTORY, pub.SUMMARY, pub.CONFIG):
        pins[str(path.relative_to(ROOT))] = digest(path)
    if any(previous['input_sha256'].get(path) != value for path, value in pins.items()):
        raise ValueError('Saved input identities differ from figure provenance')
    summary = json.loads(pub.SUMMARY.read_text())
    config = json.loads(pub.CONFIG.read_text())
    chosen = summary['best']['parameters']
    if chosen != previous['frozen_parameters'] or any(row['frozen'] != chosen for row in rows):
        raise ValueError('Frozen parameters changed')
    bounds = {key: config['parameters'][key]['bounds'] for key in pub.PARAMS}
    records = sorted((json.loads(line) for line in pub.HISTORY.read_text().splitlines() if line.strip()),
                     key=lambda record: record['attempt_id'])
    if [record['attempt_id'] for record in records] != list(range(1, 241)):
        raise ValueError('Unexpected historical record set')
    with plt.rc_context(pub.STYLE):
        pub.history_figure(records)
        gp = pub.sensitivity_figure(records, chosen, bounds)
        pub.toolpaths(rows)
        pub.residuals(rows)
        pub.sections_figure(rows)
    for name in pub.PARAMS:
        for quantity in ('physical', 'median', 'lower', 'upper'):
            if not np.allclose(gp['curves'][name][quantity], previous['gp']['curves'][name][quantity], rtol=1e-10, atol=1e-12):
                raise ValueError(f'GP curve changed: {name} {quantity}')
    if any(digest(ROOT/path) != value for path, value in pins.items()):
        raise ValueError('Input changed during rendering')
    previous['other_figures_16pt_revision'] = {
        'generator': str(Path(__file__).relative_to(ROOT)), 'generator_sha256': digest(Path(__file__)),
        'plot_source': str(Path(pub.__file__).relative_to(ROOT)), 'plot_source_sha256': digest(Path(pub.__file__)),
        'font_pt': 16, 'historical_gp_curves_match_prior': True,
        'figures': ['optimization_history', 'historical_gp_sensitivity', 'toolpaths', 'height_residuals', 'fixed_sections'],
        'size_inches': {'optimization_history': [14, 6], 'historical_gp_sensitivity': [14, 11],
                        'toolpaths': [14, 8.5], 'height_residuals': [14, 12], 'fixed_sections': [14, 13]},
    }
    (stage / 'provenance.json').write_text(json.dumps(previous, indent=2, allow_nan=False) + '\n')
    print('Staged figures:', stage)


if __name__ == '__main__':
    main()
