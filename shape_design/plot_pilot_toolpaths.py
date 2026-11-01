"""Visualize active treatment strips in the accepted turned-over pilot sequences.

Plots unique active strip centerlines, not wheel travel, execution order, or contact pressure.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shape_design import family

OUT = ROOT / 'run/shape_design/pilot_visualization'
ROWS = [('F', 1, 'Feed · front'), ('F', -1, 'Feed · back'),
        ('C', 1, 'Crosswise · front'), ('C', -1, 'Crosswise · back')]
KS = (-4, 0, 4)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    out = parser.parse_args().out_dir
    if any(out.glob('turned_over_pilot_toolpaths.*')):
        raise FileExistsError(f'Refusing to overwrite toolpaths in {out}')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(4, 3, figsize=(14, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.98, top=.93, bottom=.08, wspace=.16, hspace=.30)
    record = []
    for col, k in enumerate(KS):
        folder = ROOT / 'run/shape_design/pilot' / f'flip_k{k:+d}'
        result = json.loads((folder / 'result.json').read_text())
        seqpath = folder / 'sequence.json'
        if result['status'] != 'accepted' or sha(seqpath) != result['sequence_sha256']:
            raise ValueError(f'Unverified pilot input {folder}')
        seq = json.loads(seqpath.read_text())
        passes = family.build_passes('flip', k)
        if len(passes) != len(seq['toolpaths']):
            raise ValueError('Pass count differs from sequence')
        for p, item in zip(passes, seq['toolpaths']):
            op = item['operation']
            if not (op['rotation_deg'] == p['rotation_deg'] and op['n_strips'] == p['n_strips']
                    and np.array_equal(np.asarray(op['gtop']) != 0, p['active'])):
                raise ValueError(f'Pass geometry/mask mismatch in {folder}')
        for row, (name, face, title) in enumerate(ROWS):
            ax = axs[row, col]
            subset = [p for p in passes if p['name'] == name and p['face'] == face]
            segs = np.concatenate([family.rotate(p['segments'][p['active']], p['rotation_deg']) for p in subset])
            # Repeated rounds retrace strips; plot each ordered centerline once.
            unique = np.unique(np.round(segs.reshape(-1, 4), 9), axis=0).reshape(-1, 2, 2)
            color = '#0072B2' if face == 1 else '#D55E00'
            unique[:, :, 1] += 65
            ax.add_collection(LineCollection(unique, colors=color, linewidths=.75, alpha=.85))
            ax.add_patch(Rectangle((-110,3),220,124, fill=False, edgecolor='black', linewidth=1))
            ax.set(xlim=(-115,115), ylim=(-2,132), aspect='equal')
            ax.set_xticks([-100,0,100]); ax.set_yticks([3,65,127])
            if row == 0:
                center, edge = family.rounds(k)
                ax.set_title(f'k = {k:+d}  |  center {center}, edge {edge} rounds', pad=8)
            if col == 0:
                ax.set_ylabel(f'{title}\nLength z (mm)')
            if row == 3:
                ax.set_xlabel('Width x (mm)')
            record.append(dict(k=k, direction=name, face=face, passes=len(subset),
                               unique_active_strips=len(unique), sequence=str(seqpath.relative_to(ROOT)),
                               sequence_sha256=sha(seqpath)))
    fig.canvas.draw()
    for ax in axs.flat:
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if text.get_visible() and text.get_text():
                b = text.get_window_extent(fig.canvas.get_renderer())
                if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                    raise ValueError(f'Text outside page: {text.get_text()}')
    out.mkdir(parents=True, exist_ok=True)
    stem = out / 'turned_over_pilot_toolpaths'
    for ext in ('png','pdf'):
        fig.savefig(str(stem)+'.'+ext, dpi=200)
    plt.close(fig)
    (out / 'turned_over_pilot_toolpaths.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    (out / 'turned_over_pilot_toolpaths.caption.md').write_text(
        '# Active treatment strips in the three pilot toolpaths\n\n'
        'Columns show the edge-focused (k = −4), uniform (k = 0) and center-focused (k = +4) turned-over pilots. '
        'Rows show feed-direction and crosswise strokes on front and back faces, all in the same fixed material frame; '
        'back-face placement incorporates the prescribed mirror and layer swap. Each line is a unique active strip centerline; '
        'the 2 mm width is not drawn and repeated rounds are collapsed. The black rectangle is the 220 × 124 mm sheet. '
        'Public length z = v + 65 mm; this differs from the historical CSV feed coordinate y = v + 62 mm. '
        'Inactive zero-growth strips, travel connections, order and actual wheel contact are not depicted. '
        'The middle half of each pass’s own stepping-axis span is its central region: width x for feed passes, length z '
        'for crosswise passes. Center and edge rounds are 4+k and 4−k respectively, with four pass operations per round. '
        'This is the prescribed simulation treatment pattern, not proof of the real machine path or pressure.\n')


if __name__ == '__main__':
    main()
