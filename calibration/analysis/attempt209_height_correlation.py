#!/usr/bin/env python3
"""Descriptive full-footprint height statistics at supported scan points."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.analysis.convention_corrected_figures import load


def summarize(scan, model):
    scan, model = np.asarray(scan), np.asarray(model)
    if scan.size < 2 or not (np.isfinite(scan).all() and np.isfinite(model).all()):
        raise ValueError('need finite matched heights')
    delta = model - scan
    return dict(n=int(scan.size), pearson_r=float(np.corrcoef(scan, model)[0, 1]),
                scan_height_sd_mm=float(np.std(scan)), model_height_sd_mm=float(np.std(model)),
                model_to_scan_sd=float(np.std(model)/np.std(scan)),
                mean_difference_mm=float(delta.mean()),
                vertical_rmse_mm=float(np.sqrt(np.mean(delta**2))),
                absolute_error_p50_mm=float(np.quantile(abs(delta), .5)),
                absolute_error_p95_mm=float(np.quantile(abs(delta), .95)))


def main():
    # A correlation is unaffected by offset/positive scale; RMSE is not.
    fixture = summarize(np.arange(5.), 2*np.arange(5.)+10)
    assert np.isclose(fixture['pearson_r'], 1) and np.isclose(fixture['model_to_scan_sd'], 2)
    assert fixture['vertical_rmse_mm'] > 0
    pins = {}
    results = {}
    for run in ('run6', 'run9', 'run10'):
        row = load(run, pins)
        mask = row['support'] & np.isfinite(row['dz'])
        scan = row['scan'][mask, 2]
        results[run] = summarize(scan, scan + row['dz'][mask])
        results[run]['total_scan_points'] = int(len(mask))
        results[run]['unsupported'] = int((~mask).sum())
    out = dict(description='Pearson height correlation and vertical residuals, equally weighted supported scan points; no registration',
               independent_unit='specimen', pins=pins, results=results)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
