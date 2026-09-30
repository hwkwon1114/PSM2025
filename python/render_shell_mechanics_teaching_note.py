#!/usr/bin/env python3
"""Render the linear-algebra-first note; no simulation data or new experiments.

Markdown is the authoritative prose/equation source. One section per PDF page.
Exports vector text/math and an analytic illustrative energy diagram. Checks text
bounds/overlap before committing the PDF; no LaTeX installation required.
"""
import argparse
import hashlib
import json
import platform
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / 'reports' / 'shell_mechanics_teaching_note.md'
OUTPUT=SOURCE.with_suffix('.pdf')

class Page:
    def __init__(self, number, title):
        self.fig=plt.figure(figsize=(8.5,11),dpi=100,facecolor='white')
        self.y=.925
        self.items=[]
        self.fig.text(.085,.973,'SHELL MECHANICS  |  THEORY & IMPLEMENTATION',fontsize=8,color='#526579',va='top')
        self.fig.text(.915,.035,str(number),fontsize=9,color='#526579',ha='right')
        self.fig.lines.append(plt.Line2D([.085,.915],[.951,.951],transform=self.fig.transFigure,color='#9aaaba',lw=.7))
        self.text(title,15.5,bold=True,gap=14,width=58)

    def text(self, content, size=10.1, bold=False, gap=9, width=92, color='#202c38'):
        content=content.replace('**','').replace('`','')
        lines=textwrap.fill(content,width=width,break_long_words=False,break_on_hyphens=False)
        artist=self.fig.text(.09,self.y,lines,fontsize=size,weight='bold' if bold else 'normal',color=color,va='top',linespacing=1.32)
        self.fig.canvas.draw()
        bb=artist.get_window_extent(self.fig.canvas.get_renderer()).transformed(self.fig.transFigure.inverted())
        while bb.x1 > .915 and width > 30:
            width-=1
            artist.set_text(textwrap.fill(content,width=width,break_long_words=False,break_on_hyphens=False))
            self.fig.canvas.draw()
            bb=artist.get_window_extent(self.fig.canvas.get_renderer()).transformed(self.fig.transFigure.inverted())
        self.y=bb.y0-gap/792
        self.items.append(artist)

    def equation(self, content):
        content=content.replace(r'\boldsymbol',r'\mathbf').replace(r'\tfrac12',r'\frac{1}{2}')
        artist=self.fig.text(.5,self.y,'$'+content+'$',ha='center',va='top',fontsize=12,color='#102f4b')
        self.fig.canvas.draw()
        bb=artist.get_window_extent(self.fig.canvas.get_renderer()).transformed(self.fig.transFigure.inverted())
        self.y=bb.y0-16/792
        self.items.append(artist)

    def diagram(self):
        # Exact analytic functions, dimensionless coordinates; not simulation data.
        height=.145
        for i,(name,fun) in enumerate([('Minimum: $x^2$',lambda x:x*x),('Maximum: $-x^2$',lambda x:-x*x),('Neither: $x^3$',lambda x:x**3)]):
            ax=self.fig.add_axes([.12+i*.285,self.y-height-.025,.22,height])
            x=np.linspace(-1,1,201)
            ax.plot(x,fun(x),color='#0072B2',lw=1.7)
            ax.plot([0],[0],'o',color='#D55E00',ms=4)
            ax.axhline(0,color='.7',lw=.6);ax.axvline(0,color='.7',lw=.6)
            ax.set(xlim=(-1,1),ylim=(-1.1,1.1),xticks=[-1,0,1],yticks=[-1,0,1])
            ax.set_title(name,fontsize=9,pad=5)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('Coordinate x',fontsize=8)
            if i==0: ax.set_ylabel('Energy (arbitrary units)',fontsize=8)
            ax.spines[['right','top']].set_visible(False)
        self.y-=height+.092

    def finish(self,pdf):
        self.fig.canvas.draw()
        rend=self.fig.canvas.get_renderer()
        boxes=[]
        for a in self.items:
            b=a.get_window_extent(rend).transformed(self.fig.transFigure.inverted())
            if b.x0<.075 or b.x1>.935 or b.y0<.065 or b.y1>.94:
                raise ValueError('Text outside page bounds: '+a.get_text()[:100]+str(b))
            for old in boxes:
                if b.overlaps(old): raise ValueError('Overlapping body text: '+a.get_text()[:100])
            boxes.append(b)
        pdf.savefig(self.fig)
        plt.close(self.fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    temporary=args.output.with_suffix('.partial.pdf')
    sections=re.split(r'^## ',SOURCE.read_text(),flags=re.M)[1:]
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix','pdf.fonttype':42,'axes.unicode_minus':False})
    with PdfPages(temporary,metadata={'Title':'From permanent strain to shell shape','Author':'PSM2025-zigzag research project','Subject':'Linear-algebra-first teaching note; not a new numerical validation'}) as pdf:
        for number,section in enumerate(sections,1):
            title,body=section.split('\n',1)
            page=Page(number,title)
            for block in re.split(r'\n\s*\n',body.strip()):
                block=block.strip()
                if block.startswith('### '): page.text(block[4:],11.3,bold=True,gap=8,color='#12547a')
                elif block.startswith('$$ '): page.equation(block[3:-3].strip())
                elif block=='::energy_diagram': page.diagram()
                elif block.startswith('> '): page.text(block[2:],10.1,bold=True,gap=10,width=90,color='#12547a')
                elif block.startswith('- '):
                    for item in block.split('\n- '): page.text('• '+(item[2:] if item.startswith('- ') else item),gap=6,width=90)
                elif re.match(r'^\d+\. ',block):
                    for line in block.splitlines(): page.text(line,gap=6,width=90)
                else: page.text(block)
            page.finish(pdf)
    temporary.rename(args.output)
    provenance={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'pages':len(sections),'python':platform.python_version(),'matplotlib':matplotlib.__version__,'figures':'Analytic x^2, -x^2, x^3 illustrations only; no experimental or solver data','layout':'Body text bounds and pairwise overlap checked; PDF visual inspection separate.'}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Wrote',args.output,'pages',len(sections))

if __name__=='__main__':main()
