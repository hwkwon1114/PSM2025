"""Portable figures and a Markdown fit report from stored evaluation outputs."""
import json
from pathlib import Path
import numpy as np
from .common import NAMES
from .geometry import read_mesh, points

def report(result_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    record = json.loads(Path(result_path).read_text())
    if not record['valid']:
        raise ValueError('Cannot report a failed evaluation')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    text = ['# English Wheel final-surface fit', '',
            f"RMS objective: **{record['objective_mm']:.6g} mm**", '',
            'This is a calibration result, not independent validation or evidence of unique parameters.', '',
            '| Parameter | Value |', '|---|---:|']
    text += [f"| {n} | {record['parameters'][n]:.9g} |" for n in NAMES]
    text += ['', 'The top scan is treated as the midsurface. No geometric scale fitting is used.', '']
    for i, case in enumerate(record['cases']):
        folder = Path(case['directory'])
        with np.load(folder/'comparison.npz') as data:
            target, closest, d = data['target_mm'], data['closest_mm'], data['distance_mm']
        p = points(read_mesh(folder/'prediction_mm.vtp'))
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        ax = fig.add_subplot(221, projection='3d')
        skip = max(1, len(p)//2500)
        ax.scatter(*p[::skip].T, s=1, alpha=.4, label='Prediction')
        ax.scatter(*target[::3].T, s=1, alpha=.4, label='Scan')
        ax.set(xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)', title=case['id']); ax.legend()
        ax = fig.add_subplot(222)
        sc = ax.scatter(target[:,0], target[:,1], c=d, s=4, cmap='magma')
        ax.set(aspect='equal', xlabel='x (mm)', ylabel='y (mm)', title='Distance to predicted triangles')
        fig.colorbar(sc, ax=ax, label='mm')
        ax = fig.add_subplot(223); ax.hist(d, bins=40)
        ax.set(xlabel='Surface distance (mm)', ylabel='Scan cells')
        ax = fig.add_subplot(224)
        centerline = np.abs(target[:,0]) < 3
        order = np.argsort(target[centerline,1])
        ax.plot(target[centerline,1][order], target[centerline,2][order], '.', label='Scan near x=0')
        ax.plot(closest[centerline,1][order], closest[centerline,2][order], '.', label='Closest prediction')
        ax.set(xlabel='y (mm)', ylabel='z (mm)'); ax.legend()
        import io
        name = f'case_{i:02d}.png'; buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=170)
        (output/name).write_bytes(buffer.getvalue()); plt.close(fig)
        text += [f"## {case['id']}", '', f"RMS {case['metrics']['rms_mm']:.6g} mm; "
                 f"95th percentile {case['metrics']['p95_mm']:.6g} mm.", '', f'![Fit diagnostics]({name})', '',
                 f"Registration near bound: {case['metrics']['pose_near_bound']}", '',
                 f"Evaluation folder: `{folder}`", '']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    return str(output/'report.md')
