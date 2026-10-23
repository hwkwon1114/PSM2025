#!/usr/bin/env python3
"""Account for a completed saved-matrix experiment, without new solver calls."""
import hashlib,json,re,statistics,sys,xml.etree.ElementTree as ET
from pathlib import Path
out=Path(sys.argv[1]).resolve();job=out.name.split('ordering_',1)[1]
assert (out/'complete.txt').exists()
tests=ET.parse(out.parent/f'development_{job}/tests/tests.xml').getroot()
assert int(tests.attrib['tests'])==33
assert all(int(tests.attrib.get(k,0))==0 for k in ['failures','errors','disabled'])
verified=[]
for manifest in ['source_sha256.txt','binary.sha256','input_sha256.txt']:
    for line in (out/manifest).read_text().splitlines():
        expected,path=line.split(maxsplit=1)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    verified.append(manifest)
profiles=[];integrity={};mask=(1<<64)-1
for file in sorted(out.glob('profile_*.jsonl')):
    rows=[json.loads(line) for line in file.read_text().splitlines()]
    assert [r['mode'] for r in rows]==['amd_internal','amd_external','geometric_nd','geometric_nd','amd_external','amd_internal']
    assert all(r['valid'] and r['same_shift_and_direction'] for r in rows)
    assert len({r['input'] for r in rows})==1
    path=rows[0]['input'];raw=Path(path).read_bytes()
    prefix=re.match(rb'\{"checksum":"([0-9]+)","payload":',raw)
    assert prefix and raw.endswith(b'}\n')
    # Native envelope writes compact payload.dump() directly, then outer } + LF.
    # Hash those exact bytes; Python JSON float formatting need not match C++.
    value=14695981039346656037
    for byte in memoryview(raw)[prefix.end():-2]:value=((value^byte)*1099511628211)&mask
    assert str(value)==prefix.group(1).decode(),path
    integrity[path]=dict(fnv1a_matches=True,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));del raw
    modes={}
    for name in ['amd_internal','amd_external','geometric_nd']:
        r=[x for x in rows if x['mode']==name]
        means={k:statistics.mean(x[k] for x in r) for k in ['seconds','ordering_seconds','permutation_seconds','symbolic_seconds','factor_seconds','solve_seconds','column_square_work_proxy']}
        means['failed_factor_seconds']=statistics.mean(sum(t['seconds'] for t in x['trials'] if not t['pivot_gate_passed']) for x in r)
        means['successful_factor_seconds']=statistics.mean(sum(t['seconds'] for t in x['trials'] if t['pivot_gate_passed']) for x in r)
        modes[name]=dict(means=means,seconds=[x['seconds'] for x in r],factor_nnz=r[0]['factor_nnz'],matrix_nnz=r[0]['matrix_nnz'],shift=r[0]['shift'],attempts=[len(x['trials']) for x in r])
    old=modes['amd_internal'];new=modes['geometric_nd']
    profiles.append(dict(input=path,base_iteration=rows[0]['base_iteration'],modes=modes,
        candidate_time_ratio=new['means']['seconds']/old['means']['seconds'],
        candidate_factor_nnz_ratio=new['factor_nnz']/old['factor_nnz'],
        candidate_work_proxy_ratio=new['means']['column_square_work_proxy']/old['means']['column_square_work_proxy'],
        max_linear_residual=max(r['linear_residual'] for r in rows),max_direction_difference=max(r['direction_difference'] for r in rows)))
assert [p['base_iteration'] for p in profiles]==[0,37]
summary=dict(job_id=int(job),tests_passed=33,profiles=profiles,checkpoint_integrity=integrity,verified_manifests=verified,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    decision='Do not adopt this geometric nested-dissection candidate; retain AMD. No optimizer or production ordering changed.',
    limits=['Two indexed states of one objective/mesh, two repetitions per mode; not independent problems or held-out validation.',
            'Frozen total includes ordering/permutation, symbolic analysis, shifts, solve and verification, but not Hessian construction or checkpoint reading.',
            'Column-square proxy is structural, not hardware-counter FLOPs; factor nonzeros are not peak resident memory.',
            'New full-size trajectory model-substage and L-BFGS timers have not been collected; instrumentation is regression-tested only.',
            'No optimizer runs, production changes, installations, physical-tolerance changes or completed-cell retries.'])
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
