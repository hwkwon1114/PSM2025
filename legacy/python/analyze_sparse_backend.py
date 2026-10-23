#!/usr/bin/env python3
"""Verify and summarize saved CHOLMOD comparison; render all timed observations."""
import hashlib,json,math,statistics,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image

STYLES={
 'amd_internal':dict(label='Eigen / internal AMD',color='#333333',marker='o'),
 'amd_external':dict(label='Eigen / external AMD',color='#0072B2',marker='s'),
 'cholmod_supernodal':dict(label='CHOLMOD / external AMD',color='#D55E00',marker='D')}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def layout_errors(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();errors=[]
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes=[t.get_window_extent(renderer) for t in texts]
    for t,b in zip(texts,boxes):
        if b.x0<fig.bbox.x0-1 or b.y0<fig.bbox.y0-1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:errors.append('outside figure: '+t.get_text())
    for i,a in enumerate(boxes):
        for j,b in enumerate(boxes[:i]):
            hit=Bbox.intersection(a,b)
            if hit is not None and hit.width>1 and hit.height>1:errors.append('text overlap: '+texts[i].get_text()+' / '+texts[j].get_text())
    for ax in fig.axes:
        for t in ax.texts:
            b=t.get_window_extent(renderer)
            if b.x0<ax.bbox.x0-1 or b.x1>ax.bbox.x1+1 or b.y0<ax.bbox.y0-1 or b.y1>ax.bbox.y1+1:errors.append('annotation outside axes')
    return errors

def layout_test():
    f,a=plt.subplots(figsize=(3,2));a.set_xticks([]);a.set_yticks([]);a.text(.2,.2,'clean')
    assert not layout_errors(f)
    t=a.text(.2,.2,'overlap');assert any('overlap' in e for e in layout_errors(f));t.remove()
    a.text(2,2,'outside');assert any('outside' in e for e in layout_errors(f));plt.close(f)

out=Path(sys.argv[1]).resolve();assert (out/'complete.txt').is_file()
assert not (out/'summary.json').exists() and not (out/'figures').exists()
manifest_counts={}
for name in ['source_sha256.txt','binary.sha256','input_sha256.txt','library_sha256.txt','header_sha256.txt']:
    lines=(out/name).read_text().splitlines()
    for line in lines:
        expected,path=line.split(maxsplit=1);assert sha(path)==expected,path
    manifest_counts[name]=len(lines)
units=[json.loads(line) for line in (out/'unit_checks.jsonl').read_text().splitlines()]
assert len(units)==9 and all(r['passed'] for r in units[:-1]) and units[-1]['tests_passed']==8
profiles=[];all_rows=[]
fields=['seconds','ordering_seconds','permutation_seconds','symbolic_seconds','factor_seconds','solve_seconds']
for file in sorted(out.glob('profile_*.jsonl')):
    rows=[json.loads(line) for line in file.read_text().splitlines()]
    assert [r['mode'] for r in rows]==['amd_internal','amd_external','cholmod_supernodal','cholmod_supernodal','amd_external','amd_internal']
    assert len({r['input'] for r in rows})==1 and len({r['shift'] for r in rows})==1
    assert all(r['valid'] and r['same_shift_and_direction'] and r['checkpoint_checksum_passed'] for r in rows)
    for r in rows:
        assert r['linear_residual']<=1e-8 and r['direction_difference']<=1e-8 and r['descent']<0
        assert all(math.isfinite(r[f]) and r[f]>=0 for f in fields)
        assert sum(r[f] for f in fields[1:])<=r['seconds']
    modes={}
    for mode in STYLES:
        data=[r for r in rows if r['mode']==mode]
        means={f:statistics.mean(r[f] for r in data) for f in fields}
        means['failed_factor_seconds']=statistics.mean(sum(t['seconds'] for t in r['trials'] if not t['pivot_gate_passed']) for r in data)
        means['successful_factor_seconds']=statistics.mean(sum(t['seconds'] for t in r['trials'] if t['pivot_gate_passed']) for r in data)
        modes[mode]=dict(means=means,seconds=[r['seconds'] for r in data],stored_values=[r['factor_stored_values'] for r in data],attempts=[len(r['trials']) for r in data])
    profiles.append(dict(input=rows[0]['input'],base_iteration=rows[0]['base_iteration'],modes=modes,
        current_eigen_over_cholmod=modes['amd_internal']['means']['seconds']/modes['cholmod_supernodal']['means']['seconds'],
        max_linear_residual=max(r['linear_residual'] for r in rows),max_direction_difference=max(r['direction_difference'] for r in rows)))
    all_rows.append(rows)
assert [r['base_iteration'] for r in profiles]==[0,37]
summary=dict(job_id=int(out.name.split('_')[1]),unit_checks=units[-1],manifest_counts=manifest_counts,profiles=profiles,
    script_sha256=sha(__file__),source_profile_hashes={p.name:sha(p) for p in out.glob('profile_*.jsonl')},
    limits=['Two selected indices of one objective/mesh, not independent or held-out problems.',
            'No shell evaluations or optimizer runs. Full-size unshifted near-equilibrium and operational integration untested.',
            'Timed setup-through-verification excludes input parsing, compilation, output serialization and factor teardown.',
            'Both external modes use identical Eigen AMD; CHOLMOD internal ordering/postordering and diagonal bounding disabled.',
            'CHOLMOD numeric slots include rectangular supernode padding; not exact nonzeros or peak memory.',
            'Speedups are same-node frozen kernels, not whole-trajectory speedups; no extrapolated solver speedup measured.'])
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
figdir=out/'figures';figdir.mkdir();stem='sparse_backend_kernel_timing'
with plt.style.context(str(out/'publication.mplstyle')):
    layout_test()
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),sharex=True,sharey=True,constrained_layout=True)
    for index,(ax,rows) in enumerate(zip(axes,all_rows)):
        for row,mode in enumerate(STYLES):
            values=[r['seconds'] for r in rows if r['mode']==mode];style=STYLES[mode];y=2-row
            ax.plot(values,[y-.065,y+.065],linestyle='none',marker=style['marker'],color=style['color'],markersize=4.5)
        ax.set(xlim=(0,6),ylim=(-.4,2.4),xlabel='Kernel time (s)')
        ax.set_xticks([0,2,4,6]);ax.grid(axis='x',color='.9',linewidth=.5)
        ax.set_title(['(a) Initial state','(b) Late Newton state'][index],loc='left')
    axes[0].set_yticks([2,1,0]);axes[0].set_yticklabels([s['label'] for s in STYLES.values()])
    errors=layout_errors(fig)
    if errors:raise AssertionError('; '.join(errors))
    fig.savefig(figdir/(stem+'.pdf'),bbox_inches=None)
    fig.savefig(figdir/(stem+'.png'),dpi=300,bbox_inches=None);plt.close(fig)
with Image.open(figdir/(stem+'.png')) as image:image.convert('L').save(figdir/(stem+'_grayscale.png'))
caption='''Frozen-matrix kernel times. (a) Initial cylinder-y+ model. (b) Saved cylinder-y− Newton model at iteration 37. Each vertically offset marker is one timed repetition; all 12 observations are shown, without smoothing or confidence intervals. These are two preselected states of one objective and mesh, not independent or held-out problems. Current Eigen uses internal AMD; the other two modes use the identical external Eigen-AMD permutation. CHOLMOD uses CPU-only supernodal LLT with internal reordering/postordering disabled. Measurements share one pinned CPU on qnode0021 (Xeon Gold 6230R), with one-thread controls. Timings include ordering, matrix permutation, symbolic analysis, shifted numerical factors, solve and verification; they exclude input parsing, compilation, output serialization and factor teardown. No warm-up repetition is discarded. Both directions of the three-mode order were recorded. This is not a comparison of complete optimization trajectories.
'''
(figdir/(stem+'.caption.md')).write_text(caption)
provenance=dict(source=str(out),styles=STYLES,python=sys.version,matplotlib=matplotlib.__version__,
    script_sha256=sha(__file__),style_sha256=sha(out/'publication.mplstyle'),style_source='scientific-visualization/assets/publication.mplstyle',
    dimensions_inches=[7.2,2.8],png_dpi=300,layout_guard='clean, overlap and overflow fixtures passed; output passed',
    observations=12,omitted=0,aggregation_in_figure='none',uncertainty_intervals='none',
    regeneration='python python/analyze_sparse_backend.py '+str(out)+' (requires fresh output paths)',
    accessibility='Direct row labels and distinct shapes; grayscale generated; CVD simulation not performed',
    visual_review='pending',alt_text='Two panels show all repeated kernel times: Eigen controls around 4.4 and 5.4 seconds; CHOLMOD around 0.71 seconds in both states.')
with (figdir/'provenance.json').open('x') as f:json.dump(provenance,f,indent=2)
print(json.dumps(summary,indent=2))
