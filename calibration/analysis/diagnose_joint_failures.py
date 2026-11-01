"""Read-only failure diagnosis across joint Run6+Run10 campaigns: failure mode from solver logs + parameter context.
Writes run/joint_stage4_reduced4/failure_diagnosis.json. No solver runs."""
import glob
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = {'stage1': 'run/joint_run6_run10', 'armA': 'run/joint_stage2/arm_A', 'armB': 'run/joint_stage2/arm_B',
             'stage3': 'run/joint_stage3_gbot0', 'stage4': 'run/joint_stage4_reduced4'}
N6, N10 = 58, 100   # max hits per element, Run 6 / Run 10


def growth_sum(beta, floor, n):
    return sum(floor + (1 - floor) * math.exp(-beta * k) for k in range(n))


def log_info(casedir):
    log = casedir / 'solver.log'
    if not log.exists():
        return None
    txt = log.read_text(errors='replace')
    cyc = re.findall(r"\[zigzag_sequence\] cycle=(\d+), toolpath='(\w+)', repeat=\d+/(\d+)", txt)
    what = re.findall(r'what\(\):\s*(.*)', txt)
    tail = txt[-20000:]
    return {'case': casedir.name, 'last_cycle_done': int(cyc[-1][0]) if cyc else 0,
            'n_cycles': int(cyc[-1][2]) if cyc else None, 'toolpath': cyc[-1][1] if cyc else None,
            'what': what[-1][:200] if what else None,
            'indefinite_tail': tail.count('curvature=indefinite'), 'positive_tail': tail.count('curvature=positive'),
            'completed': 'completed' in txt[-3000:] and 'physical cycles' in txt[-3000:]}


def main():
    rows = []
    for camp, d in CAMPAIGNS.items():
        for f in sorted(glob.glob(str(ROOT / d / 'results/a*.json'))):
            r = json.loads(Path(f).read_text())
            p = r['parameters']
            row = dict(campaign=camp, attempt=r['attempt'], valid=r['valid'], failure_class=r.get('failure_class'),
                       error=(r.get('error') or '')[:160], seconds=r.get('seconds'), J=r['objective_mm'] if r['valid'] else None,
                       **{k: p[k] for k in ('gtop', 'gbot', 'ortho_top', 'ortho_bottom', 'beta', 'floor')})
            row['G6'] = p['gtop'] * growth_sum(p['beta'], p['floor'], N6)      # cumulative top growth after 58 hits
            row['G10'] = p['gtop'] * growth_sum(p['beta'], p['floor'], N10)
            if not r['valid']:
                cases = sorted(Path(r['directory']).glob('case_*')) if r.get('directory') else []
                row['cases'] = [x for x in (log_info(c) for c in cases) if x]
                last = row['cases'][-1] if row['cases'] else {}
                w = (last.get('what') or row['error'] or '')
                if 'equilibrium corrector failed' in w:
                    mode = 'equilibrium_corrector'
                elif row['failure_class'] == 'timeout' or 'time' in w.lower():
                    mode = 'timeout'
                else:
                    mode = 'other'
                row['mode'] = mode
                row['failed_in'] = last.get('toolpath')
                row['failed_at_cycle'] = (last.get('last_cycle_done', 0) + 1) if last else None
                row['n_cycles'] = last.get('n_cycles')
            rows.append(row)
    out = ROOT / 'run/joint_stage4_reduced4/failure_diagnosis.json'
    out.write_text(json.dumps(rows, indent=1))
    F = [r for r in rows if not r['valid']]
    V = [r for r in rows if r['valid']]
    print(f'{len(rows)} evaluations, {len(F)} failed')
    for r in F:
        print(f"{r['campaign']:6s} a{r['attempt']:04d} {r['mode']:21s} {str(r['failed_in']):5s} cyc {r['failed_at_cycle']}/{r['n_cycles']} "
              f"t={r['seconds'] or 0:6.0f}s gtop={r['gtop']:.2e} gbot={r['gbot']:.1e} ot={r['ortho_top']:+.2f} "
              f"b={r['beta']:.3f} fl={r['floor']:.3f} G6={r['G6']:.2e} G10={r['G10']:.2e}")

    def q(v, x):
        v = sorted(v); return v[int(x * (len(v) - 1))]
    print('\nvalid vs failed (median [10%,90%]):')
    for k in ('gtop', 'ortho_top', 'beta', 'floor', 'G6', 'G10'):
        a = [r[k] for r in V]; b = [r[k] for r in F]
        print(f'  {k:9s} valid {q(a,.5):.3g} [{q(a,.1):.3g},{q(a,.9):.3g}]   failed {q(b,.5):.3g} [{q(b,.1):.3g},{q(b,.9):.3g}]')
    print('\nfailure rate by cumulative Run-6 growth G6 bin:')
    edges = [0, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e9]
    for lo, hi in zip(edges, edges[1:]):
        s = [r for r in rows if lo <= r['G6'] < hi]
        if s:
            print(f'  G6 [{lo:.1e},{hi:.1e}): {sum(not r["valid"] for r in s)}/{len(s)} failed')
    print('wrote', out)


if __name__ == '__main__':
    main()
