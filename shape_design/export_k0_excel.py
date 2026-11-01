"""Export accepted k=0 CSV treatment segments to Excel; not a machine program."""
import argparse
import csv
import json
from pathlib import Path
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'run/shape_design/pilot_toolpath_csv'
OUTPUT = SOURCE / 'k_0_toolpaths.xlsx'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    entries = [r for r in json.loads((SOURCE / 'manifest.json').read_text()) if r['k'] == 0]
    if len(entries) != 16 or [r['pass'] for r in entries] != list(range(1, 17)):
        raise ValueError('Expected exactly 16 ordered k=0 passes')

    with xlsxwriter.Workbook(str(args.out)) as wb:
        intro = wb.add_worksheet('Overview')
        intro.set_column('A:A', 120)
        intro.set_row(0, 28)
        bold = wb.add_format({'bold': True, 'font_size': 14})
        intro.write('A1', 'k = 0: first-round treatment strips', bold)
        intro.write('A2', 'Four pattern passes per round; four rounds total. Passes 01–04 are shown; all 16 are in separate worksheets.')
        intro.write('A3', 'Coordinates: x width centred on the sheet; y feed measured from the sheet edge; z nominal surface height (mm).')
        intro.write('A4', 'Each line is one independent treatment segment. Wheel lift and travel between lines are not specified.')
        intro.write('A5', 'Front/back and feed/crosswise passes need machine-specific registration after flipping or rotating the sheet.')
        intro.write('A6', 'Exploratory model only: the 0.61 mm sheet and back-face working are calibration extrapolations. NOT machine-ready.')

        for entry in entries:
            number = entry['pass']
            src = SOURCE / entry['file']
            with src.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 2 * entry['segments']:
                raise ValueError(f'Incomplete point pairs: {src}')
            name = f'Pass {number:02d}'
            ws = wb.add_worksheet(name)
            ws.freeze_panes(1, 0)
            ws.set_column('A:A', 15); ws.set_column('B:D', 18)
            ws.write_row(0, 0, ['point_index', 'x_mm', 'y_mm', 'z_mm'])
            for i, row in enumerate(rows, 1):
                ws.write_row(i, 0, [int(row['point_index']), float(row['x_mm']), float(row['y_mm']), float(row['z_mm'])])
            if number <= 4:
                chart = wb.add_chart({'type': 'scatter', 'subtype': 'straight'})
                color = '#0072B2' if entry['face'] == 'front' else '#D55E00'
                for j in range(entry['segments']):
                    first = 1 + 2*j
                    chart.add_series({'categories': [name, first, 1, first + 1, 1],
                                      'values': [name, first, 2, first + 1, 2],
                                      'line': {'color': color, 'width': 0.75},
                                      'marker': {'type': 'none'}})
                direction = 'Feed direction' if entry['direction'] == 'F' else 'Crosswise'
                chart.set_title({'name': f'Pass {number:02d}: {direction}, {entry["face"]} face'})
                chart.set_x_axis({'name': 'Width x (mm)', 'min': -110, 'max': 110, 'major_unit': 50})
                chart.set_y_axis({'name': 'Feed from sheet edge (mm)', 'min': 0, 'max': 124, 'major_unit': 25})
                chart.set_legend({'none': True})
                chart.set_size({'width': 720, 'height': 390})
                intro.insert_chart(7 + (number-1)//2 * 21, (number-1)%2 * 8, chart)

        order = wb.add_worksheet('Pass order')
        headers = ['pass', 'round', 'direction', 'face', 'segments', 'wheel_band_width_mm']
        order.write_row(0, 0, headers)
        order.set_column('A:F', 20)
        for i, e in enumerate(entries, 1):
            order.write_row(i, 0, [e[h] for h in headers])
    print(args.out)


if __name__ == '__main__':
    main()
