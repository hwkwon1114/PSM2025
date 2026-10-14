"""One offline production-family GP fit; saved data only, no campaign mutations."""
import itertools
import json
import hashlib
import os
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'run/optimization_review'
INPUTS = ['run/stage2_bayesian_optimization/bayes_state.json',
          'run/optimization_review/registration_240_reassessment.json',
          'run/optimization_review/legacy_frame_reassessment_retry.json',
          'calibration/configs/run6_stage2_bayesian_optimization.json',
          'calibration/ewcal/common.py', 'calibration/ewcal/batch_bayes.py',
          'calibration/analysis/plot_gp_landscape_240.py']
KEYS = ['gtop', 'gbot', 'ortho_top', 'ortho_bottom']
LABELS = [r'$g_{\mathrm{top}}$', r'$g_{\mathrm{bottom}}$', r'$o_{\mathrm{top}}$', r'$o_{\mathrm{bottom}}$']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualified_metrics(observation):
    if 'metrics' in observation:
        metrics = observation['metrics']
    else:
        assert len(observation['cases']) == 1 and observation['cases'][0]['id'] == 'run6'
        metrics = observation['cases'][0]['metrics']
    assert metrics['registration_qualified']
    return metrics


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
    targets = [OUT / ('gp_landscape_240_' + stem + '.' + ext)
               for stem in ('pairs', 'profiles') for ext in ('png', 'pdf')]
    report_path = OUT / 'gp_landscape_240.json'
    caption_path = OUT / 'gp_landscape_240.caption.md'
    assert not any(p.exists() for p in targets + [report_path, caption_path]), 'Preserve existing outputs'
    hashes = {p: sha(ROOT / p) for p in INPUTS}
    import numpy as np
    import torch
    import botorch
    import gpytorch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from calibration.ewcal.common import Parameters
    from calibration.ewcal.batch_bayes import fit_gp_model
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(2026); np.random.seed(2026)
    read = lambda p: json.loads((ROOT / p).read_text())
    state, rescoring, legacy, cfg = [read(p) for p in INPUTS[:4]]
    assert rescoring['status'] == legacy['status'] == 'complete'
    assert rescoring['inputs_unchanged'] and legacy['inputs_unchanged']
    assert not any(o['qualified'] for o in legacy['records'])
    assert cfg['registration']['icp_iterations'] == 240
    sample = {'registration_qualified': True, 'rms_symmetric_mm': 1.0}
    assert qualified_metrics({'metrics': sample}) == qualified_metrics({'cases': [{'id': 'run6', 'metrics': sample}]})
    rows = state['observations']
    assert sorted(o['attempt_id'] for o in rows) == list(range(1, 33))
    rescued = {o['attempt_id']: o for o in rescoring['records'] if o['qualified']}
    assert sorted(rescued) == [16, 19, 25]
    codec = Parameters(cfg)
    training, accounting = [], []
    for o in rows:
        att = o['attempt_id']
        source = 'original_qualified' if o['valid'] else 'excluded'
        loss = None
        if o['valid']:
            metrics = qualified_metrics(o)
            loss = float(metrics['rms_symmetric_mm'])
            assert abs(loss - o['objective_mm']) < 1e-6
        if att in rescued:
            assert not o['valid'] and o['parameters'] == rescued[att]['parameters']
            assert rescued[att]['metrics']['registration_qualified']
            loss = float(rescued[att]['metrics']['rms_symmetric_mm'])
            source = 'qualified_240_reassessment'
        accounting.append({'attempt_id': att, 'original_valid': o['valid'], 'loss_source': source})
        if loss is not None:
            assert np.isfinite(loss) and loss > 0
            u = codec.encode(o['parameters'])
            np.testing.assert_allclose(u, o['normalized_parameters'], atol=1e-12, rtol=0)
            training.append({'attempt_id': att, 'parameters': o['parameters'],
                             'normalized_parameters': u.tolist(), 'objective_mm': loss, 'source': source})
    assert len(training) == 13 and len({tuple(o['normalized_parameters']) for o in training}) == 13
    best = min(training, key=lambda o: o['objective_mm'])
    assert best['attempt_id'] == 25 and abs(best['objective_mm'] - 2.898934234236606) < 1e-12
    anchor = np.array(best['normalized_parameters'])
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        gp, _ = fit_gp_model(training)
    gp.eval()

    def posterior(x):
        with torch.no_grad():
            p = gp.posterior(torch.tensor(x, dtype=torch.double), observation_noise=False)
            mu, sd = p.mean[:, 0].numpy(), p.variance[:, 0].clamp_min(0).sqrt().numpy()
        median, lo, hi = np.exp(-mu), np.exp(-mu - 1.96 * sd), np.exp(-mu + 1.96 * sd)
        assert np.isfinite(np.array([median, lo, hi, sd])).all()
        assert np.all((lo <= median) & (median <= hi))
        return median, lo, hi, sd

    pairs, profiles = [], []
    for i, j in itertools.combinations(range(4), 2):
        a = np.unique(np.r_[np.linspace(0, 1, 31), anchor[i]])
        b = np.unique(np.r_[np.linspace(0, 1, 31), anchor[j]])
        aa, bb = np.meshgrid(a, b)
        grid = np.tile(anchor, (aa.size, 1)); grid[:, i] = aa.ravel(); grid[:, j] = bb.ravel()
        median, lo, hi, sd = posterior(grid)
        x = [codec.decode(np.array([a[k] if d == i else anchor[d] for d in range(4)]))[KEYS[i]] for k in range(len(a))]
        y = [codec.decode(np.array([b[k] if d == j else anchor[d] for d in range(4)]))[KEYS[j]] for k in range(len(b))]
        idx = int(np.argmin(median))
        pairs.append({'dims': [i, j], 'x': x, 'y': y, 'median_mm': median.reshape(aa.shape).tolist(),
                      'sd_log_rms': sd.reshape(aa.shape).tolist(), 'lower95_mm': lo.reshape(aa.shape).tolist(),
                      'upper95_mm': hi.reshape(aa.shape).tolist(), 'grid_minimum_median_mm': float(median[idx]),
                      'grid_minimum_parameters': codec.decode(grid[idx])})
    for d in range(4):
        u = np.unique(np.r_[np.linspace(0, 1, 101), anchor[d]])
        grid = np.tile(anchor, (len(u), 1)); grid[:, d] = u
        median, lo, hi, sd = posterior(grid)
        profiles.append({'dim': d, 'x': [codec.decode(v)[KEYS[d]] for v in grid],
                         'median_mm': median.tolist(), 'lower95_mm': lo.tolist(), 'upper95_mm': hi.tolist()})
    report = {'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'], 'input_sha256': hashes,
              'training': training, 'accounting': accounting, 'best': best, 'seed': 2026,
              'fit_warnings': [str(w.message) for w in notices], 'model': 'production-family Matern5/2 ARD, -log RMS, standardized outcome, inferred noise',
              'lengthscales_normalized': gp.covar_module.base_kernel.lengthscale.detach().flatten().tolist(),
              'noise_standardized': gp.likelihood.noise.detach().flatten().tolist(),
              'versions': {'numpy': np.__version__, 'torch': torch.__version__, 'botorch': botorch.__version__,
                           'gpytorch': gpytorch.__version__, 'matplotlib': matplotlib.__version__},
              'pairs': pairs, 'profiles': profiles,
              'limits': 'Conditional slices, not marginalization; fitted-hyperparameter latent bands, not validated coverage. No feasibility GP or proposals; adaptive single-scan calibration, not held-out.'}
    for p, h in hashes.items(): assert sha(ROOT / p) == h, 'Input drift'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('lengthscales_normalized', 'noise_standardized', 'fit_warnings')}), flush=True)
    print('PAIR_MINIMA', [(p['dims'], p['grid_minimum_median_mm']) for p in pairs], flush=True)

    def save(fig, stem):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = fig.bbox
        for ax in fig.axes:
            artists = [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.get_xticklabels() + ax.get_yticklabels()
            for artist in artists:
                if not artist.get_visible() or not artist.get_text(): continue
                b = artist.get_window_extent(renderer)
                assert b.x0 >= box.x0 - 2 and b.y0 >= box.y0 - 2 and b.x1 <= box.x1 + 2 and b.y1 <= box.y1 + 2, 'Clipped text'
        for ext in ('png', 'pdf'):
            fig.savefig(OUT / f'gp_landscape_240_{stem}.{ext}', dpi=180, bbox_inches=None)
        plt.close(fig)

    with plt.rc_context({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout='constrained')
        vmax = max(np.max(p['median_mm']) for p in pairs)
        vmin = min(np.min(p['median_mm']) for p in pairs)
        smax = max(np.max(p['sd_log_rms']) for p in pairs)
        for n, p in enumerate(pairs):
            i, j = p['dims']; row, col = divmod(n, 2)
            for k, (field, title, cmap, low, high) in enumerate([('median_mm', 'Median RMS', 'viridis_r', vmin, vmax),
                                                                 ('sd_log_rms', 'Latent SD', 'cividis', 0, smax)]):
                ax = axes[row, col * 2 + k]
                z = ax.pcolormesh(p['x'], p['y'], p[field], cmap=cmap, vmin=low, vmax=high, shading='auto', rasterized=True)
                ax.plot(best['parameters'][KEYS[i]], best['parameters'][KEYS[j]], '*', color='#D55E00', ms=9, mec='black', mew=.5)
                ax.set(xlabel=LABELS[i], ylabel=LABELS[j], title=title)
                if i < 2: ax.set_xscale('log')
                if j < 2: ax.set_yscale('log')
                cb = fig.colorbar(z, ax=ax, fraction=.046, pad=.03)
                cb.set_label('RMS (mm)' if k == 0 else 'SD of log RMS')
        save(fig, 'pairs')
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
        for ax, p in zip(axes.flat, profiles):
            d = p['dim']
            ax.fill_between(p['x'], p['lower95_mm'], p['upper95_mm'], color='#0072B2', alpha=.18, label='95% latent band')
            ax.plot(p['x'], p['median_mm'], color='#0072B2', label='GP median')
            ax.plot(best['parameters'][KEYS[d]], best['objective_mm'], '*', color='#D55E00', ms=10, label='Attempt 25')
            ax.plot([r['parameters'][KEYS[d]] for r in training], [.025] * len(training), '|', color='.35',
                    transform=ax.get_xaxis_transform(), label='Training projections')
            ax.set(xlabel=LABELS[d], ylabel='Symmetric RMS (mm)', yscale='log')
            if d < 2: ax.set_xscale('log')
            ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        save(fig, 'profiles')
    caption_path.write_text('''# Conditional GP landscape around Attempt 25

Six parameter-pair slices: left of each pair is predicted RMS median (mm; lower is better), right is latent SD of log RMS. Shared scales across comparable panels. Red stars mark Attempt25. The other two parameters are fixed at its values. Four1D profiles hold the other three parameters fixed; blue shading is pointwise95% latent-function Gaussian credible bands conditional on fitted kernel/noise, transformed via exp(-f). This is not observation-noise or hyperparameter uncertainty, calibrated coverage, parameter posterior, marginalization or nuisance profiling. Growth axes and profile RMS axes are logarithmic; orthotropy dimensionless/linear. Rugs are projections of all13 qualified training points, NOT observations on the conditional slice.

One double-precision production-family Matérn5/2 ARD loss GP with -log symmetric RMS and standardized outputs, seed2026. Ten original qualified points reuse their unchanged early-terminating registration scores; only16/19/25 are replaced by separately qualified240-cap evidence. Nineteen excluded attempts (16 registration rejects,3 timeouts) are not assigned penalty losses. All32 are accounted in JSON. No constraint GP is shown: apparently promising locations can be registration-infeasible or time out. Same adaptive Run6 calibration scan, no independent validation or physical-identification claim. Sparse data and plug-in hyperparameters can give misleadingly narrow uncertainty. Slice-grid minima are model estimates, not evaluated improvements or global optima. No mechanics, realignment, acquisition proposal, training migration or campaign write.

Source: calibration/analysis/plot_gp_landscape_240.py. Data/grids/training provenance/source hashes, fitted hyperparameters, warnings and versions: gp_landscape_240.json. PNG/PDF exports intended as standalone exploratory desktop figures (14×10in pairs,9×6in profiles), not scaled manuscript panels; numeric grids provide an accessible data alternative.
''')
    print('GP_LANDSCAPE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
