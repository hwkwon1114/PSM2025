"""Read-only census of saved shell solves: runtime, solver path and timeouts.

Question: why do some solves finish in minutes while identical-input solves
hit the 3600 s cap?  Uses only saved case folders; no solver runs.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ELAPSED = re.compile(r'elapsed=([0-9.]+)s')


def case_record(case: Path):
    rec = {'case': str(case.relative_to(ROOT))}
    term = case / 'termination.json'
    if term.is_file():
        t = json.loads(term.read_text())
        rec['termination_reason'] = t.get('reason')
        rec['timeout'] = bool(t.get('timeout'))
        rec['elapsed_s'] = t.get('elapsed_seconds')
    else:
        rec['timeout'] = False
    conv = case / 'sequence_convergence.csv'
    rows = []
    if conv.is_file():
        with conv.open() as fh:
            rows = [r for r in csv.DictReader(fh) if r.get('state_kind') == 'physical']
    rec['attempts'] = len(rows)
    rec['statuses'] = '|'.join(r.get('termination_status', '') for r in rows)
    rec['cutback'] = any(r.get('retry_count', '0') not in ('0', '') for r in rows)
    rec['indefinite'] = any('indefinite' in r.get('termination_status', '') for r in rows)
    rec['iterations'] = sum(int(r['iterations']) for r in rows if r.get('iterations', '').lstrip('-').isdigit())
    rec['accepted'] = any(r.get('equilibrium_accepted') == '1' for r in rows)
    log = case / 'solver.log'
    rec['log_bytes'] = log.stat().st_size if log.is_file() else None
    cmd = case / 'command.json'
    # command.json is written immediately before launch; solver.log last write ~ solver end.
    if log.is_file() and cmd.is_file():
        rec['wall_s_approx'] = log.stat().st_mtime - cmd.stat().st_mtime
    if cmd.is_file():
        c = json.loads(cmd.read_text())
        rec['toolpath'] = next((p.split('/')[-2] for p in c if str(p).endswith('sequence.json')), None)
        rec['stdbuf'] = c[0] == 'stdbuf'
    seqres = case / 'sequence_result.json'
    rec['solved'] = seqres.is_file()
    return rec


def main():
    roots = sys.argv[1:] or ['run']
    out = []
    for r in roots:
        for case in sorted((ROOT / r).rglob('case_0[0-9]')):
            if case.is_dir() and (case / 'command.json').is_file():
                out.append(case_record(case))
    dest = ROOT / 'run/diagnostics/timeout_census.json'
    dest.write_text(json.dumps(out, indent=1))
    print(len(out), 'cases ->', dest)


if __name__ == '__main__':
    main()
