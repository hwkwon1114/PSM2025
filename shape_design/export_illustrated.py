"""Illustrated CSV treatment segments for collaborator review, not machine execution."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.ewcal.common import digest

SOURCE = ROOT / 'run/shape_design/pilot_toolpath_csv'
FIELDS = ['pass', 'round', 'direction', 'face', 'segments', 'wheel_band_width_mm']
POINT_FIELDS = ['point_index', 'x_mm', 'y_mm', 'z_mm']


def load_passes(source, k):
    entries = [e for e in json.loads((source / 'manifest.json').read_text()) if e['k'] == k]
    count = 4 * (4 + abs(k))
    if [e['pass'] for e in entries] != list(range(1, count + 1)):
        raise ValueError(f'Expected {count} ordered k={k:+d} passes')
    data = []
    for entry in entries:
        path = source / entry['file']
        if digest(path) != entry['csv_sha256']:
            raise ValueError(f'Changed source CSV: {path}')
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != POINT_FIELDS:
                raise ValueError(f'Unexpected CSV columns: {path}')
            rows = list(reader)
        values = np.array([[float(row[key]) for key in POINT_FIELDS] for row in rows])
        if (values.shape != (2 * entry['segments'], 4) or not np.isfinite(values).all()
                or not np.array_equal(values[:, 0], np.arange(len(rows)))):
            raise ValueError(f'Invalid ordered point pairs: {path}')
        data.append(values)
    return entries, data


def pass_sheet(wb, entry, values, image_path, title, warning):
    number = entry['pass']
    direction = 'Feed direction' if entry['direction'] == 'F' else 'Crosswise'
    ws = wb.add_worksheet(f'Pass {number:02d}')
    ws.set_column('A:D', 15)
    ws.set_column('F:N', 13)
    ws.freeze_panes(3, 0)
    ws.merge_range('A1:N1', f"Round {entry['round']} · {direction} · {entry['face']} face", title)
    ws.merge_range('A2:N2', 'Each row pair is a separate treated strip; lifting and transfer are unspecified.', warning)
    ws.write_row(2, 0, POINT_FIELDS)
    for row, point in enumerate(values, 3):
        ws.write_row(row, 0, [int(point[0]), *point[1:]])
    fig, ax = plt.subplots(figsize=(7.4, 4.9), layout='constrained')
    color = '#0072B2' if entry['face'] == 'front' else '#D55E00'
    ax.add_collection(LineCollection(values[:, 1:3].reshape(-1, 2, 2), colors=color, linewidths=.9))
    ax.add_patch(Rectangle((-110, 0), 220, 124, fill=False, edgecolor='black', linewidth=1))
    ax.set(xlim=(-115, 115), ylim=(-3, 127), xlabel='Width x (mm)', ylabel='Feed y (mm)',
           aspect='equal', title=f'Pass {number:02d}: {direction}, {entry["face"]} face')
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    ws.insert_image('F3', str(image_path), {'x_scale': .7, 'y_scale': .7,
                                          'description': f'{direction} strips, {entry["face"]} face'})


def export(source, out, k, shape_image=None):
    entries, data = load_passes(source, k)
    if shape_image is not None and not shape_image.is_file():
        raise FileNotFoundError(shape_image)
    out.mkdir(parents=True, exist_ok=False)
    images = out / 'pass_images'
    images.mkdir()
    stem = 'minus4' if k == -4 else 'plus4' if k == 4 else '0'
    workbook = out / f'k_{stem}_illustrated_toolpaths.xlsx'
    with xlsxwriter.Workbook(str(workbook)) as wb:
        title = wb.add_format({'bold': True, 'font_size': 16})
        warning = wb.add_format({'bold': True, 'font_color': '#9C0006', 'text_wrap': True})
        info = wb.add_worksheet('Read me')
        info.set_column('A:A', 110)
        info.write('A1', f'k = {k:+d} — exploratory treatment strips', title)
        lines = [
            'NOT A MACHINE PROGRAM: each point pair is a separate strip; do not connect pairs under load.',
            'Use Pass order for the sequence and each numbered sheet for its coordinates and picture.',
            f'Sheet: 220 × 124 × 0.61 mm. {4 + abs(k)} rounds, four pattern operations per round.',
            'CSV axes: x width centered; y feed from sheet edge (0–124 mm); z nominal flat plane.',
            'Back-face paths use the original material frame; establish machine transforms after flips/turns.',
            'Wheel lift, travel, pressure, speed, fixtures and machine-height set points are unspecified.',
            'Attempt 209 extrapolation: thinner sheet and back-face treatment have not been calibrated.'
        ]
        if k == -4:
            lines.append('The coarse-mesh saddle-like fit is borderline and does not imply uniform saddle curvature.')
        for row, line in enumerate(lines, 2):
            info.write(row, 0, line, warning if row == 2 else None)
        if shape_image is not None:
            shutil.copyfile(shape_image, out / 'simulated_shape.png')
            shape = wb.add_worksheet('Simulated shape')
            shape.set_column('A:A', 120)
            shape.write('A1', f'k = {k:+d}: saved 2 mm pilot simulation, not a measurement', title)
            shape.write('A2', 'Height exaggerated 10×; colors in mm after rigid alignment and mean removal.')
            shape.write('A3', 'Shape: public z = v + 65 mm. CSV: feed y = v + 62 mm; same sheet, different origin.')
            shape.insert_image('A5', str(out / 'simulated_shape.png'), {'x_scale': .65, 'y_scale': .65})
        order = wb.add_worksheet('Pass order')
        order.set_column('A:F', 20)
        order.write_row(0, 0, FIELDS)
        for entry, values in zip(entries, data):
            order.write_row(entry['pass'], 0, [entry[key] for key in FIELDS])
            pass_sheet(wb, entry, values, images / f'pass_{entry["pass"]:02d}.png', title, warning)
    provenance = {'manifest_sha256': digest(source / 'manifest.json'),
                  'csv_sha256': {e['file']: e['csv_sha256'] for e in entries},
                  'source_script_sha256': digest(Path(__file__)), 'k': k, 'passes': len(entries),
                  'workbook_sha256': digest(workbook), 'no_solver': True}
    if shape_image is not None:
        provenance['shape_image'] = {'source': str(shape_image), 'sha256': digest(shape_image)}
        shape_provenance = shape_image.parent / 'provenance.json'
        if shape_provenance.is_file():
            provenance['shape_inputs'] = json.loads(shape_provenance.read_text())[str(k)]
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'README.md').write_text(
        f'# k={k:+d} illustrated toolpaths\n\nOpen `{workbook.name}`. Each numbered sheet contains one pass and its separate treatment segments. '
        'Follow `Pass order` for the face and direction; repeated passes are not separate experiments. '
        'CSV coordinates follow the historical Run 6 schema, not public figure axis names. '
        'Establish machine registration, lift/travel and force/speed with the experimentalist before use. '
        'This is not a machine program or a fabrication guarantee.\n' +
        ('\n`Simulated shape` shows the saved 2 mm pilot, not a refined-mesh result or an experiment.\n' if shape_image else ''))
    return workbook


def main(default_k=None, default_with_shape=False, default_out=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, choices=(-4, 0, 4), default=default_k, required=default_k is None)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out-dir', type=Path, default=default_out)
    parser.add_argument('--shape-image', type=Path)
    args = parser.parse_args()
    shape_image = args.shape_image
    if shape_image is None and default_with_shape:
        shape_image = SOURCE / f'selected_shape_images/flip_k{args.k:+d}_shape.png'
    out = args.out_dir or SOURCE / f'k_{args.k:+d}_illustrated'
    print(export(args.source, out, args.k, shape_image))


if __name__ == '__main__':
    main()
