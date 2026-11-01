"""Add a one-time parameter table to the existing saved Pareto PNG."""
import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'calibration/paper_figures/fresh_lhs_joint4'
STEM = 'run1_run3_pareto_tradeoff'
OUT = BASE / 'run1_run3_pareto_tradeoff_with_parameters.png'


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    source = BASE / f'{STEM}.png'
    data = BASE / f'{STEM}.csv'
    with data.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['valid'] == 'True']
    if len(rows) != 86:
        raise ValueError('Saved plot does not have the expected 86 valid candidates')
    selected = [min(rows, key=lambda r: float(r['run1_rms_mm'])),
                min(rows, key=lambda r: float(r['joint_rms_mm'])),
                min(rows, key=lambda r: float(r['run3_rms_mm']))]
    if not all(r['observed_nondominated'] == 'True' for r in selected):
        raise ValueError('A selected candidate is not on the observed frontier')
    image = Image.open(source).convert('RGB')
    output = Image.new('RGB', (image.width, image.height + 320), 'white')
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    regular = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'
    font = ImageFont.truetype(regular, 34)
    head = ImageFont.truetype(bold, 34)
    y = image.height + 22
    x = [100, 710, 1035, 1400, 1710]
    for xpos, text in zip(x, ['Point', 'g_top', 'ortho_top', 'beta', 'floor']):
        draw.text((xpos, y), text, font=head, fill='#222222')
    for i, (name, row) in enumerate(zip(['Run 1 min', 'Joint min', 'Run 3 min'], selected)):
        values = [f'{name} ({row["attempt"]})'] + [f'{float(row[k]):.4g}' for k in ('gtop', 'ortho_top', 'beta', 'floor')]
        for xpos, text in zip(x, values):
            draw.text((xpos, y + 60 * (i + 1)), text, font=font, fill='#222222')
    output.save(OUT)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (BASE / 'run1_run3_pareto_tradeoff_with_parameters.caption.md').write_text(
        'The existing fresh-search Pareto plot is unchanged; the table adds the four '
        'parameter values for its three labeled points. Values are rounded to four '
        'significant digits; exact values are in the original CSV. The 86 valid points '
        'are adaptive calibration evaluations, not independent replicates or a global '
        'frontier; floor was bounded by 0.5. No new solver or search was used.\n\n'
        f'Original PNG SHA-256: `{digest(source)}`; original CSV SHA-256: `{digest(data)}`.\n')
    print(f'Wrote {OUT}; ' + ', '.join(f'{n}: {r["attempt"]}' for n, r in zip(['Run 1', 'joint', 'Run 3'], selected)))


if __name__ == '__main__':
    main()
