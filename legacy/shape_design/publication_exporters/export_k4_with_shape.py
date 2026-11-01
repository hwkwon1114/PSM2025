"""Illustrated k=+4 pilot treatment segments. Not a machine program."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'run/shape_design/pilot_toolpath_csv'
OUT = SRC / 'k_+4_illustrated_with_shape'
MANIFEST = SRC / 'manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [e for e in json.loads(MANIFEST.read_text()) if e['k'] == 4]
    if len(rows) != 32 or [e['pass'] for e in rows] != list(range(1, 33)):
        raise ValueError('Expected the 32 ordered k=+4 passes')
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    images = OUT / 'pass_images'
    images.mkdir()
    wb = xlsxwriter.Workbook(str(OUT / 'k_plus4_illustrated_toolpaths.xlsx'))
    info = wb.add_worksheet('Read me')
    info.set_column('A:A', 110)
    title = wb.add_format({'bold': True, 'font_size': 16})
    warning = wb.add_format({'bold': True, 'font_color': '#9C0006', 'text_wrap': True})
    info.write('A1', 'k = +4 — exploratory treatment strips', title)
    info.write('A3', 'NOT A MACHINE PROGRAM: each pair of points is a separate treated strip; do not connect pairs under load.', warning)
    info.write('A5', 'Pass order: see Pass order. Each pass tab contains coordinates and a picture of its active strips.')
    info.write('A6', 'Sheet: 220 mm width × 124 mm feed × 0.61 mm thickness. Four pass operations per round, eight rounds.')
    info.write('A7', 'x: width centered on sheet; y: feed from sheet edge (0–124 mm); z: nominal flat material plane.')
    info.write('A8', 'Back-face strokes are shown in the original material frame. Machine coordinates after flips/rotations must be established separately.')
    info.write('A9', 'No wheel-lift/transit path, pressure, speed, fixture, or machine-height set point is specified.')
    info.write('A10', 'Model-based extrapolation from Attempt 209: 0.61 mm thickness and back-face treatment were not calibrated.')
    shape = wb.add_worksheet('Simulated shape')
    shape.set_column('A:A', 120)
    shape.write('A1', 'k = +4: saved 2 mm pilot simulation (not a measurement)', title)
    shape.write('A2', '3-D height is exaggerated 10×; colors show height in mm after rigid alignment and mean removal.')
    shape.write('A3', 'Shape image uses public z = v + 65 mm; pass CSVs use y = v + 62 mm. Both describe the same sheet.')
    shape.insert_image('A5', str(SRC / 'selected_shape_images/flip_k+4_shape.png'), {'x_scale': .65, 'y_scale': .65})
    order = wb.add_worksheet('Pass order')
    order.set_column('A:G', 19)
    fields = ['pass', 'round', 'direction', 'face', 'segments', 'wheel_band_width_mm']
    order.write_row(0, 0, fields)
    source_hashes = {}
    for index, entry in enumerate(rows, 1):
        f = SRC / entry['file']
        if sha(f) != entry['csv_sha256']:
            raise ValueError(f'Changed source CSV: {f}')
        with f.open(newline='') as stream:
            data = list(csv.DictReader(stream))
        if len(data) != 2*entry['segments'] or [int(r['point_index']) for r in data] != list(range(len(data))):
            raise ValueError(f'Unexpected point pairs: {f}')
        order.write_row(index, 0, [entry[x] for x in fields])
        name = f'Pass {index:02d}'
        ws = wb.add_worksheet(name)
        ws.set_column('A:D', 15); ws.set_column('F:N', 13)
        ws.freeze_panes(3, 0)
        direction = 'Feed direction' if entry['direction'] == 'F' else 'Crosswise'
        ws.write('A1', f"Round {entry['round']} · {direction} · {entry['face']} face", title)
        ws.write('A2', 'Each row pair is a separate active strip; transfer and lifting are unspecified.', warning)
        ws.write_row(2, 0, ['point_index', 'x_mm', 'y_mm', 'z_mm'])
        xy = np.array([[float(r['x_mm']), float(r['y_mm'])] for r in data])
        for i, r in enumerate(data, 3):
            ws.write_row(i, 0, [int(r['point_index']),float(r['x_mm']),float(r['y_mm']),float(r['z_mm'])])
        fig, ax = plt.subplots(figsize=(7.4, 4.9), layout='constrained')
        color = '#0072B2' if entry['face'] == 'front' else '#D55E00'
        ax.add_collection(LineCollection(xy.reshape(-1, 2, 2), colors=color, linewidths=.9))
        ax.add_patch(Rectangle((-110, 0), 220, 124, fill=False, edgecolor='black', linewidth=1))
        ax.set(xlim=(-115,115), ylim=(-3,127), xlabel='Width x (mm)', ylabel='Feed y (mm)', aspect='equal',
               title=f'Pass {index:02d}: {direction}, {entry["face"]} face')
        image = images / f'pass_{index:02d}.png'
        fig.savefig(image, dpi=160)
        plt.close(fig)
        ws.insert_image('F3', str(image), {'x_scale': .7, 'y_scale': .7,
                                           'description': f'{direction} strips, {entry["face"]} face'})
        source_hashes[entry['file']] = entry['csv_sha256']
    wb.close()
    (OUT / 'README.md').write_text('''# k=+4 illustrated pilot toolpaths

Open `k_plus4_illustrated_toolpaths.xlsx`; `Pass order` lists the 32 operations and each pass sheet has its actual CSV coordinates and a picture. Individual pictures are also in `pass_images/`. Every line is one separate segment: lifting and travel between segments are unspecified. The coordinate frame is x centered across width, y from the sheet feed edge, z the nominal flat material plane, in mm. Back-side paths have already been mapped into the original material frame; they still need a machine-specific coordinate transform after physical flips or turns. This is a model prediction, not a machine-ready path, force specification, or fabrication guarantee.\n''')
    (OUT / 'provenance.json').write_text(json.dumps({'manifest_sha256': sha(MANIFEST), 'csv_sha256': source_hashes,
        'source_script_sha256': sha(Path(__file__)), 'rows': len(rows), 'no_solver': True}, indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
