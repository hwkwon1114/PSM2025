"""Create a non-destructive colour-harmonized Attempt 209 publication set; no fitting or solver.

The six existing blue/orange/grey figures are copied byte-for-byte. Only the historical signed
residual map and the three Hyunwoo figures are redrawn. Historical input hashes are verified.
Usage: python calibration/analysis/build_attempt209_palette_set.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
from calibration.analysis import attempt209_publication as pub, publication_palette as palette
from calibration.analysis.restyle_publication_figures import STYLE_20PT
from calibration.ewcal.common import digest, save_json

SOURCE = ROOT / 'calibration/paper_figures/attempt209_publication'
DEST = SOURCE / 'harmonized_blue_orange'
UNCHANGED = ('fixed_sections', 'historical_gp_sensitivity', 'optimization_history',
             'shape_overlays_3d', 'shape_overlays_xy', 'toolpaths')


def main():
    DEST.mkdir(exist_ok=False)
    previous = json.loads((SOURCE / 'provenance.json').read_text())
    for stem in UNCHANGED:
        for ext in ('pdf', 'png', 'caption.md'):
            shutil.copy2(SOURCE / f'{stem}.{ext}', DEST / f'{stem}.{ext}')
    pins = {}
    rows = [pub.shapes.load(run, pins) for run in pub.RUNS]
    historical = previous['input_sha256']
    for path, recorded in pins.items():
        key = str(Path(path).resolve().relative_to(ROOT))
        if key not in historical or digest(ROOT / key) != historical[key]:
            raise ValueError(f'Historical source changed or not pinned: {key}')
    palette.register()
    pub.OUT = DEST
    with plt.rc_context(STYLE_20PT):
        pub.residuals(rows, cmap=palette.DIVERGING)
    subprocess.run([sys.executable, str(ROOT / 'calibration/analysis/hw_run1_publication.py'),
                    '--out', str(DEST)], cwd=ROOT, check=True)
    (DEST / 'README.md').write_text('''# Attempt 209: harmonized colour set

Blue `#0072B2` and orange `#D55E00`, plus black and grey. Line comparisons use orange for
experiment/scan and blue for simulation/prediction; the toolpath uses blue for front face and
orange for back face. Labels and line styles distinguish meanings. Signed height and residual maps
use the same blue–white–orange diverging scale, centred at zero; colour limits remain figure-specific
and residual signs are stated in captions. Six pre-existing figures were copied byte-for-byte;
only `height_residuals` and the three `hw_run1_*` figures were redrawn. No fitting or solver was run.
See original `../provenance.json`, `hw_run1_provenance.json`, and `palette_provenance.json`.
''')
    save_json(DEST / 'palette_provenance.json', {
        'script': str(Path(__file__).resolve().relative_to(ROOT)), 'script_sha256': digest(Path(__file__)),
        'palette_sha256': digest(Path(palette.__file__)), 'historical_generator_sha256': digest(Path(pub.__file__)),
        'source_provenance_sha256': digest(SOURCE / 'provenance.json'), 'verified_historical_inputs': pins,
        'unchanged_figures': list(UNCHANGED), 'redrawn_historical_figure': 'height_residuals',
        'redrawn_hyunwoo_figures': ['hw_run1_toolpath', 'hw_run1_shape_comparison', 'hw_run1_shape_3d'],
        'colors': {'blue': palette.BLUE, 'orange': palette.ORANGE, 'diverging': 'blue-white-orange'},
    })
    print('Created:', DEST)


if __name__ == '__main__':
    main()
