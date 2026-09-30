#!/usr/bin/env python3
"""Snapshot the eight-case ledger; analyze only certified-by-ledger endpoints.

No simulation, independent residual recomputation, or stability certification.
Snapshot mode uses stdlib only. Analysis/plotting belongs on a compute node.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'run/forward_model_diagnostics'
AUDIT = RUN/'absolute_stop_audit'
CYCLE = re.compile(r'_cycle_(\d+)_.*_final\.vtp$')


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(out):
    out.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    scheduler = subprocess.run(['sacct', '-j', '6958070,6958071', '-X', '-n', '-P',
        '--format=JobID,State,Elapsed,ExitCode'], capture_output=True, text=True, check=True).stdout
    (out/'sacct.txt').write_text(scheduler)
    states = {row.split('|')[0]: row.split('|')[1:] for row in scheduler.splitlines() if row.strip()}
    records = []
    for index, (res, order, variant) in enumerate(itertools.product(
            ('0p015', '0p01'), ('broad_to_central', 'central_to_broad'), ('standard', 'tight'))):
        name = f'nested_absstop_g0p005_{order}_res{res}_{variant}'
        directory = RUN/'current'/name
        dest = out/name
        dest.mkdir()
        sources = []
        for filename in ('manifest.json', 'sequence.json', 'sequence_convergence.csv', 'result.json'):
            path = directory/filename
            if not path.exists():
                continue
            data = path.read_bytes()
            (dest/filename).write_bytes(data)
            sources.append(dict(path=str(path), sha256=digest(data), bytes=len(data),
                                captured_utc=datetime.now(timezone.utc).isoformat()))
        with (directory/'run.log').open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size-20000))
            (dest/'run_tail.log').write_bytes(stream.read())
        manifest = json.loads((dest/'manifest.json').read_text())
        seqbytes = (dest/'sequence.json').read_bytes()
        seq = json.loads(seqbytes)
        flags = dict(zip(manifest['command'][1::2], manifest['command'][2::2]))
        result = json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else {}
        gate = float(flags['-equilibrium_grad_tol'])
        text = (dest/'sequence_convergence.csv').read_text()
        # Live writers may leave an unterminated row; never silently parse it as evidence.
        dropped_partial = not text.endswith('\n')
        if dropped_partial:
            text = text[:text.rfind('\n')+1]
        rows = list(csv.DictReader(io.StringIO(text)))
        accepted, violations, full, progress = [], [], set(), {}
        keys = []
        for number, row in enumerate(rows, 2):
            cycle, attempt = int(row['executed_cycle_index']), int(row['substep_attempt'])
            keys.append((cycle, attempt))
            norm = float(row['final_gradient_norm'])
            if row['equilibrium_accepted'] == '1':
                accepted.append(row)
                energy = float(row['recomputed_energy'])
                if not (math.isfinite(norm) and 0 <= norm <= gate and
                        int(row['solver_code']) in (1, 2, 3, 4) and math.isfinite(energy)):
                    violations.append(number)
                lam = float(row['lambda_trial'])
                progress[cycle] = max(progress.get(cycle, 0), lam)
                if abs(lam-1) < 1e-12:
                    full.add(cycle)
        expected = sum(int(p.get('repeat', 1)) for p in seq['toolpaths'] if p.get('enabled', True))
        exports = {int(CYCLE.search(p.name).group(1)): p for p in directory.glob('*_final.vtp')
                   if CYCLE.search(p.name)}
        full_endpoint = (result.get('return_code') == 0 and not result.get('timed_out') and
                         full == set(range(1, expected+1)) and not violations and
                         set(range(1, expected+1)) <= set(exports))
        if full_endpoint:
            path = exports[expected]
            assert result['final_vtp'] == path.name
            data = path.read_bytes()
            (dest/path.name).write_bytes(data)
            sources.append(dict(path=str(path), sha256=digest(data), bytes=len(data)))
        rejected = [r for r in rows if r['equilibrium_accepted'] != '1']
        last = rows[-1] if rows else None
        checks = dict(sequence_hash_matches=manifest['sequence_sha256'] == digest(seqbytes),
            stop_half_gate=float(flags['-hlbfgs_absolute_gradient_tol']) == gate*.5,
            unique_attempt_keys=len(set(keys)) == len(keys),
            all_accepted_residuals_and_energies_valid=not violations)
        assert all(checks.values()), (name, checks)
        record = dict(index=index, job=f'6958071_{index}', case=name, order=order,
            res=float(res.replace('p', '.')), variant=variant, gate=gate,
            scheduler=states.get(f'6958071_{index}'), return_code=result.get('return_code'),
            timed_out=result.get('timed_out'), wall_seconds=result.get('wall_seconds'),
            full_endpoint=full_endpoint, expected_cycles=expected, full_cycles=sorted(full),
            accepted_load_by_cycle=progress, attempts=len(rows), accepted_attempts=len(accepted),
            rejected_attempts=len(rejected), rejected_codes=dict(Counter(r['solver_code'] for r in rejected)),
            accepted_gate_violations=violations,
            max_accepted_gradient=max((float(r['final_gradient_norm']) for r in accepted), default=None),
            last_record=last, trailing_partial_row_excluded=dropped_partial,
            binary_sha256=manifest['binary_sha256'], flags=flags, checks=checks, sources=sources,
            endpoint_filename=exports[expected].name if full_endpoint else None)
        records.append(record)
        print(index, name, record['scheduler'], 'full cycles', sorted(full),
              'accepted loading', progress, 'rejected', len(rejected), flush=True)
    hashes = {r['binary_sha256'] for r in records}
    assert len(hashes) == 1
    # Scientific inputs must differ only in prescribed order, mesh and tolerance.
    ignore = {'-cycle_file', '-basename', '-res', '-tol', '-equilibrium_grad_tol', '-hlbfgs_absolute_gradient_tol'}
    fixed = [{k:v for k,v in r['flags'].items() if k not in ignore} for r in records]
    assert all(v == fixed[0] for v in fixed)
    sequences = [json.loads((out/r['case']/'sequence.json').read_text()) for r in records]
    multisets = [Counter(json.dumps({k:v for k,v in p.items() if k != 'id'}, sort_keys=True)
                        for p in seq['toolpaths']) for seq in sequences]
    assert all(m == multisets[0] for m in multisets)
    nonpaths = [{k:v for k,v in s.items() if k != 'toolpaths'} for s in sequences]
    assert all(s == nonpaths[0] for s in nonpaths)
    save(out/'ledger.json', dict(snapshot_start_utc=start,
         snapshot_end_utc=datetime.now(timezone.utc).isoformat(), records=records,
         scope='Saved ledger audit; running records are a non-atomic bounded snapshot. No independent residual recomputation.',
         all_same_binary=True, all_fixed_flags_equal=True, all_path_multisets_equal=True))


def analyze(out):
    import platform
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from analyze_sequence_ablation_v2 import load_vtp, point_array, cell_array, metric_vector, areas
    target = out/'analysis'
    target.mkdir(exist_ok=False)
    ledger = json.loads((out/'ledger.json').read_text())

    def align(ref, mov, w):
        w = w/w.sum()
        cr, cm = w@ref, w@mov
        u, _, vt = np.linalg.svd((mov-cm).T@(w[:, None]*(ref-cr)))
        if np.linalg.det(u@vt) < 0:
            u[:, -1] *= -1
        return (mov-cm)@(u@vt)+cr

    def rms(d, w):
        return float(np.sqrt(w@np.sum(d*d, axis=1)/w.sum()))

    # Exact rigid motion and affine interpolation sanity checks.
    rng = np.random.default_rng(42)
    test = rng.normal(size=(100, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    assert rms(align(test, test@q+3, np.ones(100))-test, np.ones(100)) < 1e-12
    tiny = mtri.Triangulation([0, 1, 0], [0, 0, 1], [[0, 1, 2]])
    assert np.allclose(mtri.LinearTriInterpolator(tiny, [2, 5, 6])([.2], [.3]), [3.8])
    loaded, summaries = {}, []
    for r in ledger['records']:
        if not r['full_endpoint']:
            continue
        path = out/r['case']/r['endpoint_filename']
        data, xyz, tri = load_vtp(path)
        uv = np.column_stack([point_array(data, key) for key in ('material_u', 'material_v')])
        ref = np.column_stack([uv, np.zeros(len(uv))])
        weights = np.zeros(len(uv))
        face_area = areas(ref, tri)
        assert np.all(face_area > 0) and np.isfinite(xyz).all()
        for j in range(3):
            np.add.at(weights, tri[:, j], face_area/3)
        weights /= weights.sum()
        aligned = align(ref, xyz, weights)
        metrics = metric_vector(data)
        hits = cell_array(data, 'total_hit_count')
        assert np.isfinite(metrics).all() and np.isfinite(hits).all()
        bbar = np.column_stack([cell_array(data, f'bbar_ref_{key}') for key in ('11', '12', '22')])
        loaded[r['index']] = dict(xyz=xyz, uv=uv, tri=tri, aligned=aligned, w=weights,
                                  metrics=metrics, hits=hits, bbar=bbar, record=r)
        summaries.append(dict(index=r['index'], case=r['case'], vertices=len(uv), faces=len(tri),
            height_span_mm=float(np.ptp(aligned[:, 2])*1000),
            deformation_rms_mm=rms(aligned-ref, weights)*1000,
            geometry_storage=data.GetPoints().GetData().GetDataTypeAsString(),
            final_energy=float(r['last_record']['recomputed_energy'])))
    comparisons = []
    distributions = {}
    for (ia, a), (ib, b) in itertools.combinations(loaded.items(), 2):
        ra, rb = a['record'], b['record']
        varying = [key for key in ('res', 'order', 'variant') if ra[key] != rb[key]]
        if len(varying) != 1:
            continue
        row = dict(left=ia, right=ib, varying=varying[0])
        if varying[0] != 'res':
            assert np.array_equal(a['tri'], b['tri']) and np.array_equal(a['uv'], b['uv'])
            samples = [(None, a['xyz'], b['xyz'], a['uv'], a['w'])]
            row['metric_max_absolute_difference'] = float(np.abs(a['metrics']-b['metrics']).max())
            row['metric_max_relative_difference'] = row['metric_max_absolute_difference']/float(max(np.abs(a['metrics']).max(), np.abs(b['metrics']).max()))
            row['hits_identical'] = bool(np.array_equal(a['hits'], b['hits']))
            row['bbar_max_absolute_difference'] = float(np.abs(a['bbar']-b['bbar']).max())
        else:
            samples = []
            for nu, nv in ((100, 120), (200, 240)):
                lo = np.maximum(a['uv'].min(axis=0), b['uv'].min(axis=0))
                hi = np.minimum(a['uv'].max(axis=0), b['uv'].max(axis=0))
                assert np.allclose(a['uv'].min(axis=0), b['uv'].min(axis=0), atol=1e-12)
                assert np.allclose(a['uv'].max(axis=0), b['uv'].max(axis=0), atol=1e-12)
                uu, vv = np.meshgrid(lo[0]+(np.arange(nu)+.5)*(hi[0]-lo[0])/nu,
                                     lo[1]+(np.arange(nv)+.5)*(hi[1]-lo[1])/nv)
                uv = np.column_stack([uu.ravel(), vv.ravel()])
                interpolated, hh = [], []
                for obj in (a, b):
                    mesh = mtri.Triangulation(obj['uv'][:, 0], obj['uv'][:, 1], obj['tri'])
                    values = [mtri.LinearTriInterpolator(mesh, obj['xyz'][:, k])(uv[:, 0], uv[:, 1]) for k in range(3)]
                    assert not any(np.ma.getmaskarray(v).any() for v in values), 'Uncovered common sample'
                    interpolated.append(np.column_stack([np.asarray(v) for v in values]))
                    face = mesh.get_trifinder()(uv[:, 0], uv[:, 1])
                    assert np.all(face >= 0)
                    hh.append(obj['hits'][face])
                samples.append((f'{nu}x{nv}', *interpolated, uv, np.ones(len(uv))/len(uv)))
                row.setdefault('hit_footprint_sensitivity', []).append(dict(grid=f'{nu}x{nv}',
                    mismatch_area_fraction=float(np.mean(hh[0] != hh[1])),
                    mean_absolute_hit_difference=float(np.mean(np.abs(hh[0]-hh[1])))))
            row['metric_note'] = 'Raw edge-basis metric components are not compared across meshes; hit-field differences quantify footprint discretization only.'
        row['evaluations'] = []
        for grid, x, y, uv, w in samples:
            w = w/w.sum()
            y_to_x = align(x, y, w)
            delta = np.linalg.norm(y_to_x-x, axis=1)*1000
            ref = np.column_stack([uv, np.zeros(len(uv))])
            deforms = [rms(align(ref, obj, w)-ref, w)*1000 for obj in (x, y)]
            srms = rms(y_to_x-x, w)*1000
            order = np.argsort(delta)
            quantiles = np.interp([.5, .9, .95], np.cumsum(w[order]), delta[order]).tolist()
            row['evaluations'].append(dict(grid=grid, samples=len(uv), aligned_rms_mm=srms,
                max_mm=float(delta.max()), median_p90_p95_mm=quantiles,
                deformation_rms_mm=deforms, rms_over_mean_deformation=srms/np.mean(deforms)))
            distributions[(ia, ib)] = (delta, w)
        comparisons.append(row)
    save(target/'comparison.json', dict(endpoints=summaries, comparisons=comparisons,
         snapshot=str(out), complete_endpoint_indices=sorted(loaded),
         methods='Area-lumped weighted rigid alignment on identical meshes. Cross-mesh: piecewise-linear positions at uniform common material cell centers, equal-area rigid alignment; 100x120 and 200x240 grids. No reflections, scaling, or smoothing.',
         limitations=['Ledger residuals only, not independent reevaluation.', 'Three-dimensional positions stored at exported precision; no precision recovery.',
                      'No stability certificate or independent replicates.', 'No cross-mesh tensor-metric equality test; spatial hit discretization may change loading.'],
         versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
         self_tests=['weighted rigid alignment', 'affine material interpolation']))

    def layout_check(fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            texts = list(ax.texts)+[ax.title, ax.xaxis.label, ax.yaxis.label]+ax.get_xticklabels()+ax.get_yticklabels()
            for text in texts:
                if not text.get_visible() or not text.get_text():
                    continue
                bb = text.get_window_extent(renderer)
                fb = fig.bbox
                if bb.x0 < fb.x0-1 or bb.y0 < fb.y0-1 or bb.x1 > fb.x1+1 or bb.y1 > fb.y1+1:
                    raise RuntimeError('Text outside figure: '+text.get_text())
            for text in ax.texts:
                bb = text.get_window_extent(renderer)
                ab = ax.get_window_extent(renderer)
                if bb.x0 < ab.x0-1 or bb.y0 < ab.y0-1 or bb.x1 > ab.x1+1 or bb.y1 > ab.y1+1:
                    raise RuntimeError('Annotation outside panel: '+text.get_text())

    def export(fig, stem):
        layout_check(fig)
        for ext in ('png', 'pdf'):
            fig.savefig(target/(stem+'.'+ext), dpi=180)
        plt.close(fig)

    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42}):
        # All eight planned conditions are visible, including failed/pending endpoints.
        fig, axes = plt.subplots(2, 4, figsize=(13, 7.8), layout='constrained')
        limit = max(float(np.abs(a['aligned'][:, 2]).max())*1000 for a in loaded.values())
        for r in ledger['records']:
            ax = axes[r['index']//4, r['index']%4]
            order = 'Broad → central' if r['order'] == 'broad_to_central' else 'Central → broad'
            ax.set_title(f'{r["index"]}: {order}\nres={r["res"]:g}; {r["variant"]}')
            ax.set(xlim=(-127, 127), ylim=(-152.4, 152.4), xticks=[-100, 0, 100], yticks=[-100, 0, 100],
                   xlabel='Material u (mm)', ylabel='Material v (mm)')
            ax.set_aspect('equal')
            if r['index'] in loaded:
                a = loaded[r['index']]
                image = ax.tripcolor(a['uv'][:, 0]*1000, a['uv'][:, 1]*1000, a['tri'],
                    a['aligned'][:, 2]*1000, shading='gouraud', cmap='RdBu_r', vmin=-limit, vmax=limit,
                    rasterized=True)
            else:
                status = 'Timed out' if r['timed_out'] else ('Failed' if r['return_code'] is not None else 'Running at snapshot')
                ax.text(.5, .5, f'{status}\n{len(r["full_cycles"])}/6 full cycles\nNo final-endpoint map',
                        ha='center', va='center', transform=ax.transAxes)
        fig.colorbar(image, ax=list(axes.ravel()), location='bottom', shrink=.55, pad=.06,
                     label='Rigidly aligned height (mm); common scale')
        export(fig, 'absolute_stop_endpoint_height')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        styles = {'order': ('#0072B2', '-'), 'res': ('#D55E00', '--'), 'variant': ('#009E73', '-.')}
        for row in comparisons:
            delta, w = distributions[(row['left'], row['right'])]
            ii = np.argsort(delta)
            label = f'{row["left"]} vs {row["right"]}: '+{'order':'order', 'res':'mesh', 'variant':'tolerance'}[row['varying']]
            color, ls = styles[row['varying']]
            axes[0].plot(delta[ii], np.cumsum(w[ii]), color=color, ls=ls, label=label)
        dmax = max(float(d.max()) for d, _ in distributions.values())
        axes[0].set(xlabel='Aligned position difference (mm)', ylabel='Material-area fraction',
                    xlim=(0, dmax*1.05), xticks=np.linspace(0, dmax, 5),
                    ylim=(0, 1), yticks=np.linspace(0, 1, 6))
        axes[0].legend(fontsize=8)
        # Failure-aware progress for every attempted case, not only successful runs.
        for r in ledger['records']:
            progress = sum(r['accepted_load_by_cycle'].values())
            axes[1].barh(r['index'], progress, color='.65', edgecolor='black', height=.65)
        axes[1].set(xlim=(0, 6), xticks=range(7), yticks=range(8), yticklabels=[f'Case {i}' for i in range(8)],
                    xlabel='Sum of accepted cycle loading (0–6)', ylabel='Array task')
        axes[1].invert_yaxis()
        export(fig, 'absolute_stop_shape_difference_distribution')
    (target/'figures.caption.md').write_text(
        'Eight-case absolute-stop recheck, partial snapshot. The height map shows all eight planned conditions; '
        'only fully completed six-cycle, residual-accepted trajectories have final-endpoint maps. '
        'Partial geometries are not substituted. Heights are after material-area-weighted proper rigid alignment '
        'to the flat plate; common spatial and color scales, no deformation magnification. '
        'The distribution figure uses every vertex with lumped reference-area weights for same-mesh comparisons '
        'and all 200×240 common material cell centers for cross-mesh comparisons. Curves show distributions '
        'of spatial differences, not uncertainty across independent runs. Cross-mesh positions are linearly '
        'interpolated on each actual material triangulation; hit-field discrepancies remain a loading confound. '
        'The progress bars sum accepted loading fractions across six cycles, not runtime completion fractions. '
        'Case IDs are the original array task indices. No stability or mesh-convergence claim follows.\n')
    print(json.dumps(dict(endpoints=summaries, comparisons=comparisons), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshot', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (snapshot if args.mode == 'snapshot' else analyze)(args.output.resolve())


if __name__ == '__main__':
    main()
