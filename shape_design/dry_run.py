"""No-solver preparation: generate every sequence and command, record identities in plan.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.ewcal import runner
from calibration.ewcal.common import digest, save_json
from shape_design import family
from shape_design.run_case import identities

out = ROOT / 'run/shape_design/plan'
out.mkdir(parents=True, exist_ok=True)
params, config = family.load_frozen()
cases = {}
for mesh in (2.0, 1.0):
    for variant in family.VARIANTS:
        for k in family.K_VALUES:
            if mesh == 1.0 and not (variant == 'flip' and abs(k) == 4):
                continue
            name = f'{variant}_k{k:+d}_mesh{mesh:g}'
            cfg, case = family.case_config(config, variant, k, mesh, 7200 if mesh == 1.0 else 3600)
            seq = out / f'{name}.json'
            seq.write_text(json.dumps(family.sequence(variant, k, params, config), indent=1) + '\n')
            cmd = runner.command(cfg, case, seq)
            opts = dict(zip(cmd[1::2], cmd[2::2]))
            ref = dict(zip(*[iter(runner.command(config, config['cases'][0], seq)[1:])] * 2))
            changed = {key: (ref.get(key), value) for key, value in opts.items() if ref.get(key) != value}
            assert set(changed) <= {'-lx', '-ly', '-res', '-h_total', '-native_hybrid_seconds'}, changed
            cases[name] = {'sequence_sha256': digest(seq), 'passes': case['repeat'],
                           'timeout_s': cfg['solver']['timeout_seconds'],
                           'rounds_center_edge': family.rounds(k), 'changed_vs_attempt209_run1': changed}
curl = {f'flip_k{k:+d}_mesh2': family.curl_diagnostic('flip', k, 2.0) for k in family.K_VALUES}
curl.update({f'flip_k{k:+d}_mesh1': family.curl_diagnostic('flip', k, 1.0) for k in (-4, 4)})
save_json(out / 'curl_diagnostic.json', curl)
print({n: (round(c['treated_rectangle_ratio'], 4), round(c['tile_ratio_median'], 4),
           round(c['tile_ratio_max'], 4)) for n, c in curl.items()})
plan = {'identities': identities(), 'cases': cases,
        'curl_diagnostic_sha256': digest(out / 'curl_diagnostic.json'),
        'tests': {'shape_design/tests/test_family.py': digest(ROOT / 'shape_design/tests/test_family.py')},
        'launcher': {'shape_design/run_family.sbatch': digest(ROOT / 'shape_design/run_family.sbatch')}}
save_json(out / 'plan.json', plan)
print(json.dumps({k: (v['passes'], v['timeout_s'], sorted(v['changed_vs_attempt209_run1'])) for k, v in cases.items()}))
