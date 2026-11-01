"""Export accepted pilot treatment segments in historical Run 6 CSV schema.

These are material-frame coordinates, not executable machine programs. Each CSV
contains ordered *pairs* of endpoints; travel between pairs is unspecified.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_design.family import rotate, strip_segments

OUT = ROOT / 'run/shape_design/pilot_toolpath_csv'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for k in (-4, 0, 4):
        source = ROOT / 'run/shape_design/pilot' / f'flip_k{k:+d}'
        result = json.loads((source / 'result.json').read_text())
        seqpath = source / 'sequence.json'
        if result['status'] != 'accepted' or result['sequence_sha256'] != digest(seqpath):
            raise ValueError(f'Unaccepted or modified sequence: {source}')
        seq = json.loads(seqpath.read_text())
        target = out / f'k_{k:+d}'
        target.mkdir(exist_ok=True)
        for n, item in enumerate(seq['toolpaths']):
            op = item['operation']
            if not item['enabled'] or item['repeat'] != 1 or op['type'] != 'zigzag':
                raise ValueError(f'Unsupported sequence item {item["id"]}')
            if op.get('center_uv_mm', [0, 0]) != [0, 0] or op.get('shift_uv_mm', [0, 0]) != [0, 0]:
                raise ValueError('Unexpected translation')
            gtop, gbot = np.asarray(op['gtop']), np.asarray(op['gbot'])
            if len(gtop) != op['n_strips'] or len(gbot) != op['n_strips']:
                raise ValueError('Strip array mismatch')
            active = (gtop != 0) | (gbot != 0)
            segments = rotate(strip_segments(op['lv_mm'], op['alpha_deg'], op['n_strips']), op['rotation_deg'])
            indices = np.flatnonzero(active)
            name = f'pass_{n+1:02d}.csv'
            with (target / name).open('w', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(['point_index', 'x_mm', 'y_mm', 'z_mm'])
                for j, pt in enumerate(segments[active].reshape(-1, 2)):
                    # Historical CSV axes: x width, y feed, z height. Origin y=0 at sheet edge.
                    writer.writerow([j, f'{pt[0]:.8f}', f'{pt[1]+62:.8f}', '0.00000000'])
            tokens = item['id'].split('_')
            manifest.append({'k': k, 'pass': n+1, 'id': item['id'], 'round': int(tokens[1][1:])+1,
                             'direction': tokens[2], 'face': tokens[3], 'file': str((target/name).relative_to(out)),
                             'csv_sha256': digest(target/name), 'source_sequence_sha256': digest(seqpath),
                             'active_strips': indices.tolist(), 'segments': len(indices),
                             'wheel_band_width_mm': op['width_mm']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with (out / 'pass_manifest.csv').open('w', newline='') as stream:
        fields = ['k','pass','round','direction','face','file','segments','wheel_band_width_mm']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(manifest)
    (out / 'README.md').write_text('''# Exploratory Attempt 209 toolpaths — NOT machine-ready

Each `k_.../pass_XX.csv` follows the historical Run 6 header `point_index,x_mm,y_mm,z_mm`.
Use `pass_manifest.csv` for the exact pass order, rounds, F/C direction and front/back face.
Each pass CSV contains *separate treated segments*: rows 0–1 define strip 1, rows 2–3 strip 2, etc. Do NOT connect consecutive pairs with a loaded wheel; those transfers, wheel lift, approach, clearance and traverse are not specified by this export. The active strip indices in `manifest.json` refer to the original sequence. Inactive strips are omitted. Individual segments appear in solver order, including repeated geometry when the solver counts it separately.

Coordinates are flat initial sheet **material frame** in mm: x = width, centered (−110 to 110); y = feed length from sheet edge (0 to 124); z = 0 nominal material plane. This follows the historical CSV axis names, not the project public 3-D axes. Back-face strokes have already been reflected into the common material frame; `face` and `direction` indicate when to flip or rotate the sheet. A machine-specific transformation and verified flip/rotation protocol are still required. No pressure, wheel position offsets, process force or speed is included. The z=0 values are not an instruction to drive the wheel at zero machine height.

Parameter k determines active rounds per pass's stepping-axis region: center = 4+k, outer = 4−k. Each round has four pass operations. The three k values are the already accepted 2 mm model pilots. They use frozen Attempt 209 parameters and assumed back-face processing on a 0.61 mm sheet, beyond calibration; neither a real-world toolpath nor physical performance has been validated. The k=−4 saddle label is borderline. Review with the experimentalist before fabricating.
''')
    print(f'Exported {len(manifest)} pass files across 3 cases to {out}')


if __name__ == '__main__':
    main()
